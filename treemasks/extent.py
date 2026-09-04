"""Forest extent: the forest floor between the crowns (or the solid forest area), per forest type.

Starting from a canopy (either an existing canopy texture, light = canopy, or this tool's
own rendered tree markers), the step derives the solid *forest area*:

* narrow gaps and notches are closed (``close_radius``) and enclosed holes up to
  ``max_hole`` px across are treated as inside the forest,
* the outer edge is otherwise exactly the canopy's own outline, so it stays as jagged
  as the source,
* every forest pixel belongs to the category whose rendered tree marker is nearest;
  pixels farther than ``max_distance`` from any marker are left alone.

In ``gaps`` mode (the default) the output per category is the forest area minus the
canopy: the ground between the crowns, but only inside the forest. In ``area`` mode it
is the solid forest area itself. The work runs tile by tile with a halo so memory stays
bounded on 31501 x 31501 canvases; only tiles that contain tree markers are processed.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    from .renderer import PackedMask

log = logging.getLogger(__name__)

ORIENTATION_OUTPUT = "output"   # mask is oriented like the final images (after flip_x/flip_y)
ORIENTATION_RENDER = "render"   # mask is oriented like an un-flipped render (min_y on row 0)
ORIENTATIONS = (ORIENTATION_OUTPUT, ORIENTATION_RENDER)

MODE_GAPS = "gaps"   # paint the forest floor between crowns, inside the forest only
MODE_AREA = "area"   # paint the solid forest area
MODES = (MODE_GAPS, MODE_AREA)

Box = tuple[int, int, int, int]  # y0, y1, x0, x1 (half-open)


class ExtentMaskError(ValueError):
    """Raised when the extent step cannot run (bad mask, wrong size, missing scipy...)."""


def _ndimage():
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ExtentMaskError("the forest-extent step needs scipy (pip install scipy)") from exc
    return ndimage


@contextmanager
def _unlimited_image_size() -> Iterator[None]:
    """Lift Pillow's decompression-bomb guard: canopy masks are intentionally huge (31501² px)."""
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


class CanopyMask:
    """A canopy texture with lazy loading, oriented like the output canvases."""

    def __init__(self, path: str | Path, *, threshold: int = 128, flip_x: bool = False, flip_y: bool = False) -> None:
        if not 1 <= threshold <= 255:
            raise ExtentMaskError("canopy threshold must be between 1 and 255")
        self.path = Path(path)
        self.threshold = threshold
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.width = 0
        self.height = 0
        self.mode = ""
        self.canopy_pixels: int | None = None
        self._array: np.ndarray | None = None

    def open(self) -> tuple[int, int]:
        """Read the image header only (size and mode); pixel data stays on disk."""
        if not self.path.is_file():
            raise FileNotFoundError(f"canopy mask not found: {self.path}")
        try:
            with _unlimited_image_size(), Image.open(self.path) as image:
                self.width, self.height = image.size
                self.mode = image.mode
        except Image.UnidentifiedImageError as exc:
            raise ExtentMaskError(f"canopy mask {self.path} is not a readable image: {exc}") from exc
        return self.width, self.height

    def check_size(self, width: int, height: int) -> None:
        if (self.width, self.height) == (0, 0):
            self.open()
        if (self.width, self.height) != (width, height):
            raise ExtentMaskError(
                f"canopy mask {self.path} is {self.width}x{self.height} px but the output canvas is "
                f"{width}x{height} px; the mask must match the output size exactly"
            )

    def load(self) -> np.ndarray:
        """Decode the mask (once) as an 8-bit array, oriented like the output canvases."""
        if self._array is None:
            if (self.width, self.height) == (0, 0):
                self.open()
            with _unlimited_image_size(), Image.open(self.path) as image:
                if image.mode != "L":
                    log.info("converting canopy mask from mode %s to L", image.mode)
                    image = image.convert("L")
                array = np.asarray(image)  # one copy; the decoded PIL buffer is released on exit
            if self.flip_y:
                array = array[::-1]
            if self.flip_x:
                array = array[:, ::-1]
            self._array = array
            self.canopy_pixels = self._count_canopy(array)
            log.info(
                "canopy mask %s loaded (%dx%d, %d canopy pixels at threshold %d)",
                self.path, self.width, self.height, self.canopy_pixels, self.threshold,
            )
        return self._array

    def _count_canopy(self, array: np.ndarray, chunk_rows: int = 1024) -> int:
        total = 0
        for y0 in range(0, array.shape[0], chunk_rows):
            total += int(np.count_nonzero(array[y0:y0 + chunk_rows] >= self.threshold))
        return total

    def close(self) -> None:
        self._array = None


# ----- binary morphology (Euclidean, via distance transforms) --------------------------

def binary_dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """Pixels within ``radius`` of a True pixel. Returns ``mask`` itself when nothing changes."""
    if radius <= 0 or not mask.any():
        return mask
    return _ndimage().distance_transform_edt(~mask) <= radius


def binary_erode(mask: np.ndarray, radius: float) -> np.ndarray:
    """True pixels farther than ``radius`` from any False pixel (outside the array counts as True)."""
    if radius <= 0 or mask.all():
        return mask
    return _ndimage().distance_transform_edt(mask) > radius


def binary_close(mask: np.ndarray, radius: float) -> np.ndarray:
    """Dilate then erode: fills gaps and notches narrower than about ``2 * radius``."""
    if radius <= 0:
        return mask
    return binary_erode(binary_dilate(mask, radius), radius)


def fill_small_holes(mask: np.ndarray, max_hole: int) -> np.ndarray:
    """Fill enclosed False regions whose bounding box is at most ``max_hole`` px on each side.

    Regions touching the array border are never filled (they may be the outside).
    """
    if max_hole <= 0 or mask.all() or not mask.any():
        return mask
    ndimage = _ndimage()
    labels, count = ndimage.label(~mask)
    if count == 0:
        return mask
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    fill = np.ones(count + 1, dtype=bool)
    fill[0] = False
    fill[border] = False
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None or not fill[index]:
            continue
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        if max(height, width) > max_hole:
            fill[index] = False
    if not fill.any():
        return mask
    return mask | fill[labels]


def iter_tiles(height: int, width: int, tile: int, halo: int) -> Iterator[tuple[Box, Box]]:
    """Yield ``(core, window)`` boxes covering the canvas; the window is the core plus ``halo`` on each side."""
    if tile < 1 or halo < 0:
        raise ValueError("tile must be >= 1 and halo >= 0")
    for y0 in range(0, height, tile):
        y1 = min(y0 + tile, height)
        for x0 in range(0, width, tile):
            x1 = min(x0 + tile, width)
            window = (max(0, y0 - halo), min(height, y1 + halo), max(0, x0 - halo), min(width, x1 + halo))
            yield (y0, y1, x0, x1), window


# ----- extent computation ------------------------------------------------------------------

@dataclass
class ExtentStats:
    tiles_total: int = 0
    tiles_processed: int = 0
    forest_pixels: int = 0       # solid forest area pixels inside processed tiles
    unassigned_pixels: int = 0   # forest pixels with no tree marker within max_distance
    painted: dict[str, int] = field(default_factory=dict)


def compute_extent(
    canopy: CanopyMask | None,
    canvases: Mapping[str, np.ndarray],
    *,
    mode: str = MODE_GAPS,
    close_radius: int = 3,
    max_hole: int = 64,
    max_distance: float = 50.0,
    tile: int = 1024,
    value: int = 255,
    area_out: "PackedMask | None" = None,
) -> ExtentStats:
    """Replace each marker canvas, in place, with that category's forest gaps (or solid area).

    ``canopy`` is the canopy texture; when None, the union of all marker canvases is the
    canopy. ``canvases`` maps category -> uint8 marker canvas (all the same shape). Every
    canvas is overwritten with the result for its category, painted with ``value``.
    When ``area_out`` is given, the reachable forest area (all types, before any crown
    subtraction) is OR-ed into it.
    """
    if mode not in MODES:
        raise ExtentMaskError(f"extent mode must be one of {', '.join(MODES)}")
    names = list(canvases)
    if not names:
        raise ExtentMaskError("no category canvases to paint")
    arrays = [canvases[name] for name in names]
    shape = arrays[0].shape
    for name, array in zip(names, arrays):
        if array.shape != shape:
            raise ExtentMaskError(f"canvas {name!r} {array.shape} differs in size from the others {shape}")
    mask = None
    threshold = 1
    if canopy is not None:
        mask = canopy.load()
        threshold = canopy.threshold
        if mask.shape != shape:
            raise ExtentMaskError(f"canvases {shape} and canopy mask {mask.shape} differ in size")

    height, width = shape
    halo = int(max(2 * close_radius, max_hole, math.ceil(max_distance))) + 1
    stats = ExtentStats(painted={name: 0 for name in names})
    results: list[tuple[Box, list[np.ndarray | None]]] = []
    ndimage = _ndimage()

    for core, window in iter_tiles(height, width, tile, halo):
        stats.tiles_total += 1
        wy0, wy1, wx0, wx1 = window
        windows = [array[wy0:wy1, wx0:wx1] for array in arrays]
        present = [bool(w.any()) for w in windows]
        if not any(present):
            continue  # no tree of any type near this tile
        stats.tiles_processed += 1

        if mask is not None:
            crowns = mask[wy0:wy1, wx0:wx1] >= threshold
        else:
            crowns = np.zeros(windows[0].shape, dtype=bool)
            for w, p in zip(windows, present):
                if p:
                    crowns |= w > 0
        forest = binary_close(crowns, close_radius)
        forest = fill_small_holes(forest, max_hole)
        cy0, cy1, cx0, cx1 = core
        core_slice = (slice(cy0 - wy0, cy1 - wy0), slice(cx0 - wx0, cx1 - wx0))
        forest_core = forest[core_slice]
        n_forest = int(np.count_nonzero(forest_core))
        stats.forest_pixels += n_forest
        if not n_forest:
            continue

        # distance from every core pixel to the nearest marker of each category
        distances = np.full((len(names),) + forest_core.shape, np.inf, dtype=np.float64)
        for k, (w, p) in enumerate(zip(windows, present)):
            if not p:
                continue
            markers = w > 0
            distances[k] = 0.0 if markers.all() else ndimage.distance_transform_edt(~markers)[core_slice]
        nearest = distances.min(axis=0)
        owner = distances.argmin(axis=0)
        reachable = forest_core & (nearest <= max_distance)
        stats.unassigned_pixels += n_forest - int(np.count_nonzero(reachable))
        if area_out is not None:
            area_out.add_box(core, reachable)
        if mode == MODE_GAPS:
            reachable &= ~crowns[core_slice]

        packed: list[np.ndarray | None] = [None] * len(names)
        for k, name in enumerate(names):
            painted = reachable & (owner == k)
            count = int(np.count_nonzero(painted))
            if count:
                stats.painted[name] += count
                packed[k] = np.packbits(painted)
        if any(bits is not None for bits in packed):
            results.append((core, packed))

    # Write back only now: the tile loop needed the original markers in neighbouring halos.
    for array in arrays:
        array.fill(0)
    for (cy0, cy1, cx0, cx1), packed in results:
        tile_shape = (cy1 - cy0, cx1 - cx0)
        for k, bits in enumerate(packed):
            if bits is None:
                continue
            painted = np.unpackbits(bits, count=tile_shape[0] * tile_shape[1]).reshape(tile_shape).astype(bool)
            target = arrays[k][cy0:cy1, cx0:cx1]
            target[painted] = value
    log.info(
        "forest extent (%s): %d of %d tiles held trees, %d forest-area pixels, %d unassigned, painted %s",
        mode, stats.tiles_processed, stats.tiles_total, stats.forest_pixels, stats.unassigned_pixels, stats.painted,
    )
    return stats
