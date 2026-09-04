"""Draw tree markers onto 8-bit grayscale canvases and write them as PNG files."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # y0, y1, x0, x1 (half-open)


class PackedMask:
    """A boolean mask the size of a canvas, stored as packed bits per row chunk.

    Chunks that were never touched cost nothing, so a sparse forest on a 31501² canvas
    takes a few MB instead of a gigabyte. Used to remember the plain marker masks and the
    forest area while the category canvases are reused for the extent results.
    """

    def __init__(self, height: int, width: int, chunk_rows: int = 1024) -> None:
        if height < 1 or width < 1 or chunk_rows < 1:
            raise ValueError("height, width and chunk_rows must be >= 1")
        self.height = height
        self.width = width
        self.chunk_rows = chunk_rows
        self._chunks: dict[int, np.ndarray] = {}     # chunk start row -> packed bits
        self._open: tuple[int, np.ndarray] | None = None  # chunk being edited by add_box

    @classmethod
    def from_arrays(cls, arrays: Sequence[np.ndarray], chunk_rows: int = 1024) -> "PackedMask":
        """Union of ``arrays`` (non-zero = set), read in row chunks."""
        if not arrays:
            raise ValueError("from_arrays needs at least one array")
        height, width = arrays[0].shape
        packed = cls(height, width, chunk_rows)
        for y0 in range(0, height, chunk_rows):
            union: np.ndarray | None = None
            for array in arrays:
                if array.shape != (height, width):
                    raise ValueError("all arrays must have the same shape")
                rows = array[y0:y0 + chunk_rows] > 0
                union = rows if union is None else union | rows
            if union is not None and union.any():
                packed._chunks[y0] = np.packbits(union)
        return packed

    def _chunk_bounds(self, y0: int) -> tuple[int, int]:
        return y0, min(y0 + self.chunk_rows, self.height)

    def _unpacked(self, y0: int) -> np.ndarray:
        cy0, cy1 = self._chunk_bounds(y0)
        shape = (cy1 - cy0, self.width)
        bits = self._chunks.get(y0)
        if bits is None:
            return np.zeros(shape, dtype=bool)
        return np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool)

    def _flush(self) -> None:
        if self._open is not None:
            y0, rows = self._open
            if rows.any():
                self._chunks[y0] = np.packbits(rows)
            self._open = None

    def add_box(self, box: Box, patch: np.ndarray) -> None:
        """OR a boolean ``patch`` covering ``box`` into the mask (boxes may cross chunk borders)."""
        y0, y1, x0, x1 = box
        if patch.shape != (y1 - y0, x1 - x0):
            raise ValueError("patch shape does not match the box")
        cr = self.chunk_rows
        for cy0 in range((y0 // cr) * cr, y1, cr):
            cy1 = min(cy0 + cr, self.height)
            py0, py1 = max(y0, cy0), min(y1, cy1)
            sub = patch[py0 - y0:py1 - y0]
            if not sub.any():
                continue
            if self._open is None or self._open[0] != cy0:
                self._flush()
                self._open = (cy0, self._unpacked(cy0))
            self._open[1][py0 - cy0:py1 - cy0, x0:x1] |= sub

    @property
    def nbytes(self) -> int:
        self._flush()
        return sum(bits.nbytes for bits in self._chunks.values())

    def count(self) -> int:
        """Number of set pixels."""
        self._flush()
        total = 0
        for y0 in self._chunks:
            total += int(np.count_nonzero(self._unpacked(y0)))
        return total

    def merge_into(self, target: np.ndarray, value: int) -> None:
        """Paint ``value`` into ``target`` wherever this mask is set (existing paint is kept)."""
        self._flush()
        if target.shape != (self.height, self.width):
            raise ValueError("target shape does not match the mask")
        for y0 in self._chunks:
            cy0, cy1 = self._chunk_bounds(y0)
            rows = target[cy0:cy1]
            mask = self._unpacked(y0)
            rows[mask & (rows == 0)] = value


@lru_cache(maxsize=64)
def disk_stencil(radius: int) -> np.ndarray:
    """Boolean ``(2r+1, 2r+1)`` array that is True where ``dx² + dy² <= r²``."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    span = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(span, span, indexing="ij")
    return (dx * dx + dy * dy) <= radius * radius


class MaskCanvas:
    """A single grayscale canvas with clipped filled-circle drawing."""

    def __init__(self, width: int, height: int, value: int = 255) -> None:
        if width < 1 or height < 1:
            raise ValueError("canvas dimensions must be at least 1x1")
        if not 1 <= value <= 255:
            raise ValueError("marker value must be in 1..255")
        self.width = width
        self.height = height
        self.value = value
        # np.zeros uses calloc: pages are only committed once markers touch them.
        self._array = np.zeros((height, width), dtype=np.uint8)
        self._stencils: dict[int, np.ndarray] = {}

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def nbytes(self) -> int:
        return self._array.nbytes

    def _stencil(self, radius: int) -> np.ndarray:
        stencil = self._stencils.get(radius)
        if stencil is None:
            stencil = disk_stencil(radius).astype(np.uint8) * np.uint8(self.value)
            self._stencils[radius] = stencil
        return stencil

    def draw_disk(self, px: int, py: int, radius: int) -> bool:
        """Draw a filled circle; overlapping markers saturate at ``value``.

        Returns False when nothing was drawn (radius 0 or marker fully off-canvas).
        """
        if radius <= 0:
            return False
        x0, y0 = px - radius, py - radius
        x1, y1 = px + radius + 1, py + radius + 1
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x1, self.width), min(y1, self.height)
        if cx1 <= cx0 or cy1 <= cy0:
            return False
        stencil = self._stencil(radius)[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        region = self._array[cy0:cy1, cx0:cx1]
        np.maximum(region, stencil, out=region)
        return True

    def nonzero_pixels(self) -> int:
        return int(np.count_nonzero(self._array))

    def clear(self) -> None:
        """Reset every pixel to 0 while keeping the allocation."""
        self._array.fill(0)

    def merge(self, other: np.ndarray, chunk_rows: int = 1024) -> None:
        """OR another canvas into this one (pixel-wise maximum), in row chunks."""
        if other.shape != self._array.shape:
            raise ValueError(f"cannot merge a {other.shape} canvas into a {self._array.shape} canvas")
        for y0 in range(0, self.height, chunk_rows):
            rows = self._array[y0:y0 + chunk_rows]
            np.maximum(rows, other[y0:y0 + chunk_rows], out=rows)

    def invert(self, chunk_rows: int = 1024) -> tuple[int, int]:
        """Hard boolean invert in place: any non-zero pixel -> 0, every zero pixel -> ``value``.

        Returns ``(pixels_that_were_painted, pixels_painted_now)``.
        """
        painted = 0
        for y0 in range(0, self.height, chunk_rows):
            rows = self._array[y0:y0 + chunk_rows]
            mask = rows > 0
            painted += int(np.count_nonzero(mask))
            rows.fill(self.value)
            rows[mask] = 0
        return painted, self._array.size - painted

    def save(self, path: str | Path, compress_level: int = 6) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Image.fromarray wraps a C-contiguous uint8 array without copying it.
        image = Image.fromarray(self._array)
        image.save(path, format="PNG", compress_level=compress_level)
        return path

    def close(self) -> None:
        self._array = np.zeros((0, 0), dtype=np.uint8)
        self._stencils.clear()


class Renderer:
    """One canvas per tree category; maps size categories to marker radii."""

    def __init__(
        self,
        width: int,
        height: int,
        marker_sizes: Mapping[int, int],
        categories: Iterable[str],
        value: int = 255,
    ) -> None:
        self.marker_sizes = {int(k): int(v) for k, v in marker_sizes.items()}
        if self.marker_sizes.get(0, 0) != 0:
            raise ValueError("marker radius for size 0 must be 0: size-0 trees are never rendered")
        if any(r < 0 for r in self.marker_sizes.values()):
            raise ValueError("marker radii must be >= 0")
        self.width = width
        self.height = height
        self.value = value
        self.canvases: dict[str, MaskCanvas] = {name: MaskCanvas(width, height, value) for name in categories}
        if not self.canvases:
            raise ValueError("renderer needs at least one category")

    @staticmethod
    def estimated_bytes(width: int, height: int, category_count: int) -> int:
        return width * height * category_count

    @property
    def nbytes(self) -> int:
        return sum(c.nbytes for c in self.canvases.values())

    def radius_for(self, size: int) -> int:
        if size == 0:
            return 0
        return self.marker_sizes.get(size, 0)

    def draw(self, category: str, px: int, py: int, size: int) -> bool:
        """Draw a marker for a tree of ``size`` at pixel ``(px, py)``. Size 0 is never drawn."""
        radius = self.radius_for(size)
        if radius <= 0:
            return False
        return self.canvases[category].draw_disk(px, py, radius)

    def clear(self) -> None:
        """Reset all canvases to 0 without reallocating them."""
        for canvas in self.canvases.values():
            canvas.clear()

    def combined_inverse(
        self, extras: Iterable[PackedMask] = (), chunk_rows: int = 1024,
    ) -> tuple[MaskCanvas, int, int]:
        """Collapse every canvas (plus any ``extras``) into the first canvas as the inverted union.

        A pixel painted anywhere becomes 0; a pixel painted nowhere becomes ``value``.
        The category canvases are consumed (save them first). Returns
        ``(canvas, pixels_painted_anywhere, pixels_in_the_inverse)``.
        """
        names = list(self.canvases)
        target = self.canvases[names[0]]
        for name in names[1:]:
            target.merge(self.canvases[name].array, chunk_rows)
        for extra in extras:
            extra.merge_into(target.array, self.value)
        painted, inverse = target.invert(chunk_rows)
        return target, painted, inverse

    def save_all(
        self,
        directory: str | Path,
        compress_level: int = 6,
        name_pattern: str = "{category}.png",
    ) -> dict[str, Path]:
        directory = Path(directory)
        written: dict[str, Path] = {}
        for category, canvas in self.canvases.items():
            target = directory / name_pattern.format(category=category)
            log.info(
                "writing %s (%dx%d, %d marker pixels)",
                target, canvas.width, canvas.height, canvas.nonzero_pixels(),
            )
            written[category] = canvas.save(target, compress_level)
        return written

    def close(self) -> None:
        for canvas in self.canvases.values():
            canvas.close()
        self.canvases.clear()
