"""Build a synthetic canopy texture for demonstrating the forest-extent step.

The real workflow uses a canopy texture exported from the terrain. This script fakes one
for the sample data: it rasterises the two forest polygons of sample.layer, gives them a
wavy, irregular outline, cuts a bite out of the spruce polygon, punches small gaps between
"crowns" into the interior and softens every edge so the result looks like a blurred
canopy texture (light = canopy, dark = no canopy) rather than a hard-edged mask.

Usage:
    python scripts/make_demo_canopy.py OUT.png --width W --height H \
        --world-min-x .. --world-max-x .. --world-min-y .. --world-max-y .. [--flip-x] [--flip-y]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from treemasks.coordinate_mapper import CoordinateMapper  # noqa: E402

# World-space (X, Z) polygons of the two forest generators in sample.layer.
SPRUCE = [(15611.4, 12357.3), (15526.5, 12389.1), (15581.8, 12508.1), (15648.3, 12472.2)]
SORBUS = [(15493.2, 12405.5), (15602.5, 12593.2), (15363.7, 12704.7), (15276.6, 12482.2)]


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Gaussian-filtered white noise scaled to 0..1."""
    noise = ndimage.gaussian_filter(rng.random(shape), sigma)
    return (noise - noise.min()) / max(noise.max() - noise.min(), 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--world-min-x", type=float, required=True)
    ap.add_argument("--world-max-x", type=float, required=True)
    ap.add_argument("--world-min-y", type=float, required=True)
    ap.add_argument("--world-max-y", type=float, required=True)
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mapper = CoordinateMapper(
        args.world_min_x, args.world_max_x, args.world_min_y, args.world_max_y,
        args.width, args.height, flip_y=args.flip_y, flip_x=args.flip_x,
    )
    polygons = [[mapper.to_pixel(x, z) for x, z in poly] for poly in (SPRUCE, SORBUS)]
    if any(p is None for poly in polygons for p in poly):
        raise SystemExit("forest polygons fall outside the world bounds")
    scale = mapper.pixels_per_unit_x  # px per metre

    # Work in a window around the polygons, then paste into the full canvas.
    xs = [p[0] for poly in polygons for p in poly]
    ys = [p[1] for poly in polygons for p in poly]
    pad = int(30 * scale) + 5
    x0, x1 = max(0, min(xs) - pad), min(args.width, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(args.height, max(ys) + pad)
    shape = (y1 - y0, x1 - x0)

    def rasterise(poly):
        image = Image.new("L", (shape[1], shape[0]), 0)
        ImageDraw.Draw(image).polygon([(px - x0, py - y0) for px, py in poly], fill=255)
        return np.array(image) > 0

    spruce = rasterise(polygons[0])
    sorbus = rasterise(polygons[1])
    base = spruce | sorbus
    rng = np.random.default_rng(args.seed)

    # Wavy bite through the spruce polygon so a forest edge appears inside the tree area.
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cx = (polygons[0][0][0] + polygons[0][2][0]) / 2 - x0
    wave = cx + 12 * scale * np.sin(yy / (7.0 * scale)) + 6 * scale * np.sin(yy / (2.3 * scale))
    base &= ~((xx > wave) & (xx < wave + 45 * scale) & spruce)

    # Irregular outline: move the boundary inward by a smoothly varying amount (0..8 m).
    inside = ndimage.distance_transform_edt(base)
    wobble = smooth_noise(rng, shape, sigma=6 * scale) * 8 * scale
    canopy = inside > wobble

    # Gaps between crowns: blotches where a smooth noise field is high, only well inside.
    blotches = smooth_noise(rng, shape, sigma=1.6 * scale) > 0.80
    blotches = ndimage.binary_dilation(blotches, iterations=max(1, int(scale)))
    canopy &= ~(blotches & (inside > 6 * scale))

    # Soft, slightly blurred edges like a real texture.
    soft = ndimage.gaussian_filter(canopy.astype(np.float32), sigma=0.9 * scale)
    window = np.clip(soft * 255.0, 0, 255).astype(np.uint8)

    full = np.zeros((args.height, args.width), dtype=np.uint8)
    full[y0:y1, x0:x1] = window
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full).save(out, compress_level=1)
    print(f"wrote {out} ({args.width}x{args.height}, {int(np.count_nonzero(full >= 128))} canopy pixels)")


if __name__ == "__main__":
    main()
