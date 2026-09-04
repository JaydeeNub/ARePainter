"""Command-line interface for TreeMasks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import ConfigError, apply_overrides, load_config
from .extent import MODES, ORIENTATIONS, ExtentMaskError
from .pipeline import run

DEFAULT_CONFIG = "config.yaml"


def parse_marker_size(text: str) -> tuple[int, int]:
    """Parse ``SIZE=RADIUS`` (e.g. ``2=4``)."""
    size, sep, radius = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected SIZE=RADIUS, got {text!r}")
    try:
        return int(size), int(radius)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integers in SIZE=RADIUS, got {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="treemasks",
        description="Render top-down PNG tree masks from Arma Reforger (Enfusion) .layer files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-i", "--input", action="extend", nargs="+", required=True, metavar="PATH",
        help=".layer file(s) or directory(ies); directories are searched recursively",
    )
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG, metavar="FILE",
        help="YAML or JSON configuration file",
    )
    parser.add_argument("-o", "--output-dir", metavar="DIR", help="override output.directory")

    world = parser.add_argument_group("world bounds (override config)")
    world.add_argument("--world-min-x", type=float, metavar="M")
    world.add_argument("--world-max-x", type=float, metavar="M")
    world.add_argument("--world-min-y", type=float, metavar="M", help="map north axis = file Z")
    world.add_argument("--world-max-y", type=float, metavar="M", help="map north axis = file Z")

    output = parser.add_argument_group("output (override config)")
    output.add_argument("--width", type=int, metavar="PX")
    output.add_argument("--height", type=int, metavar="PX")
    output.add_argument(
        "--flip-x", action=argparse.BooleanOptionalAction, default=None,
        help="mirror the image horizontally so world max_x is the left-most column",
    )
    output.add_argument(
        "--flip-y", action=argparse.BooleanOptionalAction, default=None,
        help="mirror the image vertically so world max_y is the top row",
    )
    output.add_argument(
        "--marker-size", action="append", type=parse_marker_size, metavar="SIZE=RADIUS",
        help="marker radius in pixels for a size category (repeatable, e.g. --marker-size 3=9)",
    )
    output.add_argument("--marker-value", type=int, metavar="0-255", help="gray value of markers")
    output.add_argument("--compress-level", type=int, metavar="0-9", help="PNG compression level")
    output.add_argument("--file-pattern", metavar="GLOB", help="pattern for files inside input directories")

    extent = parser.add_argument_group(
        "forest extent (override config)",
        "Per forest type, paint the forest floor between the crowns (mode gaps) or the solid forest "
        "area (mode area). The forest area is the canopy with small gaps and enclosed holes counted "
        "as inside, bounded by the canopy's own jagged outline; each pixel goes to the nearest tree's "
        "type. The canopy is a texture given with --extent-mask, or else this tool's own rendered "
        "markers. Writes <category>_gaps.png / <category>_area.png next to the plain masks.",
    )
    extent.add_argument("--extent", action="store_true", help="enable the step using the rendered markers as canopy")
    extent.add_argument("--extent-mask", metavar="FILE", help="canopy texture (enables the step); must have the output size")
    extent.add_argument("--extent-mode", choices=MODES, help="gaps: floor between crowns; area: solid forest area")
    extent.add_argument("--no-extent", action="store_true", help="disable the step even if the config enables it")
    extent.add_argument(
        "--extent-threshold", type=int, metavar="1-255",
        help="mask pixels with a value >= threshold count as canopy",
    )
    extent.add_argument(
        "--extent-close-radius", type=int, metavar="PX",
        help="close gaps and notches narrower than about 2*PX before hole filling (0 = off)",
    )
    extent.add_argument(
        "--extent-max-hole", type=int, metavar="PX",
        help="paint enclosed canopy holes up to PX pixels across (0 = off)",
    )
    extent.add_argument(
        "--extent-max-distance", type=float, metavar="PX",
        help="forest pixels farther than PX from any rendered tree stay unpainted",
    )
    extent.add_argument(
        "--extent-orientation", choices=ORIENTATIONS,
        help="'output' if the mask is oriented like the final images, 'render' if like an un-flipped render",
    )

    combined = parser.add_argument_group("combined inverse (override config)")
    combined.add_argument(
        "--combined-inverse", action=argparse.BooleanOptionalAction, default=None,
        help="also write one PNG that is the union of every final category mask, hard-inverted",
    )
    combined.add_argument("--combined-inverse-file", metavar="NAME", help="file name inside the output directory")

    diag = parser.add_argument_group("diagnostics")
    diag.add_argument("--report-unknown", action="store_true", help="list unknown/excluded asset names")
    diag.add_argument("--json-report", metavar="FILE", help="also write the diagnostics as JSON")
    diag.add_argument(
        "--low-memory", action="store_true",
        help="render one category at a time (one canvas in RAM), re-reading the inputs per category",
    )
    diag.add_argument("-v", "--verbose", action="store_true", help="show progress and parser warnings")
    diag.add_argument("-q", "--quiet", action="store_true", help="only print errors and the summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.ERROR if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    log = logging.getLogger("treemasks")

    try:
        config = load_config(args.config)
        config = apply_overrides(
            config,
            {
                "world_min_x": args.world_min_x,
                "world_max_x": args.world_max_x,
                "world_min_y": args.world_min_y,
                "world_max_y": args.world_max_y,
                "width": args.width,
                "height": args.height,
                "flip_x": args.flip_x,
                "flip_y": args.flip_y,
                "marker_sizes": dict(args.marker_size) if args.marker_size else None,
                "marker_value": args.marker_value,
                "output_dir": args.output_dir,
                "compress_level": args.compress_level,
                "file_pattern": args.file_pattern,
                "extent_enabled": True if args.extent else None,
                "extent_mode": args.extent_mode,
                "extent_mask": args.extent_mask,
                "extent_threshold": args.extent_threshold,
                "extent_close_radius": args.extent_close_radius,
                "extent_max_hole": args.extent_max_hole,
                "extent_max_distance": args.extent_max_distance,
                "extent_orientation": args.extent_orientation,
                "extent_disabled": True if args.no_extent else None,
                "combined_inverse": args.combined_inverse,
                "combined_inverse_file": args.combined_inverse_file,
            },
        )
    except ConfigError as exc:
        log.error("%s", exc)
        if not Path(args.config).is_file() and args.config == DEFAULT_CONFIG:
            log.error("no %s in the current directory; pass --config FILE", DEFAULT_CONFIG)
        return 1

    log.info("configuration: %s", config.source)
    log.info(
        "world x %g..%g, y %g..%g -> %dx%d px, flip_x=%s, flip_y=%s, markers=%s",
        config.world.min_x, config.world.max_x, config.world.min_y, config.world.max_y,
        config.output.width, config.output.height, config.coordinate_system.flip_x,
        config.coordinate_system.flip_y, config.rendering.marker_sizes,
    )
    if config.extent.active:
        log.info(
            "forest extent: mode %s, canopy %s, close %d px, holes <= %d px, reach %g px",
            config.extent.mode, config.extent.mask or "rendered markers", config.extent.close_radius,
            config.extent.max_hole, config.extent.max_distance,
        )

    if config.combined_inverse.enabled:
        log.info("combined inverse: %s", config.combined_inverse.filename)

    try:
        result = run(config, args.input, low_memory=args.low_memory)
    except (FileNotFoundError, ExtentMaskError) as exc:
        log.error("%s", exc)
        return 1
    except MemoryError:
        log.error("out of memory allocating the canvases; try --low-memory or a smaller --width/--height")
        return 1

    summary = result.diagnostics.format_summary(
        categories=config.categories,
        sizes=sorted(config.rendering.marker_sizes),
        report_unknown=args.report_unknown,
    )
    print(summary)

    if args.json_report:
        report = {"config": config.to_dict(), "diagnostics": result.diagnostics.to_dict()}
        Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("wrote %s", args.json_report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
