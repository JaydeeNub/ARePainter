"""Orchestrate parser -> classifier -> coordinate mapper -> renderer (-> forest extent)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .classifier import TREE, TreeClassifier
from .config import Config
from .coordinate_mapper import CoordinateMapper
from .diagnostics import Diagnostics, format_bytes, peak_rss_bytes
from .extent import ORIENTATION_RENDER, CanopyMask, compute_extent
from .parser import ParseStats, iter_layer_files, parse_paths
from .renderer import MaskCanvas, PackedMask, Renderer

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    diagnostics: Diagnostics
    outputs: dict[str, Path]
    extent_outputs: dict[str, Path] = field(default_factory=dict)
    combined_inverse: Path | None = None


def build_classifier(config: Config) -> TreeClassifier:
    return TreeClassifier(
        config.trees,
        exclude=config.classifier.exclude,
        size_regex=config.classifier.size_regex,
        entity_classes=config.parser.entity_classes,
        valid_sizes=config.rendering.marker_sizes.keys(),
    )


def build_mapper(config: Config) -> CoordinateMapper:
    return CoordinateMapper(
        min_x=config.world.min_x,
        max_x=config.world.max_x,
        min_y=config.world.min_y,
        max_y=config.world.max_y,
        width=config.output.width,
        height=config.output.height,
        flip_x=config.coordinate_system.flip_x,
        flip_y=config.coordinate_system.flip_y,
    )


def build_canopy(config: Config) -> CanopyMask | None:
    """Open the canopy mask header (if the extent step uses one) and check it matches the canvas."""
    if not config.extent.active or not config.extent.mask:
        return None
    render_oriented = config.extent.orientation == ORIENTATION_RENDER
    canopy = CanopyMask(
        config.extent.mask,
        threshold=config.extent.threshold,
        flip_x=render_oriented and config.coordinate_system.flip_x,
        flip_y=render_oriented and config.coordinate_system.flip_y,
    )
    canopy.open()
    canopy.check_size(config.output.width, config.output.height)
    return canopy


def _stream_pass(
    config: Config,
    files: Sequence[Path],
    classifier: TreeClassifier,
    mapper: CoordinateMapper,
    renderer: Renderer,
    diagnostics: Diagnostics,
    draw_categories: set[str],
    *,
    count: bool,
) -> ParseStats:
    """Stream all files once, drawing trees of ``draw_categories``; gather counters when ``count``."""
    stats = ParseStats()
    x_axis = config.coordinate_system.x_axis
    y_axis = config.coordinate_system.y_axis
    classify = classifier.classify_prefab
    to_pixel = mapper.to_pixel
    draw = renderer.draw
    record = diagnostics.record
    rendered = diagnostics.rendered
    zero_radius = diagnostics.zero_radius
    skipped_size0 = diagnostics.skipped_size0
    out_of_bounds = diagnostics.out_of_bounds

    for entity in parse_paths(files, stats):
        classification = classify(entity.class_name, entity.prefab)
        if count:
            record(classification)
        if classification.status != TREE:
            continue
        category = classification.category
        size = classification.size
        if size == 0:
            if count:
                skipped_size0[category] += 1
            continue
        coords = entity.coords
        pixel = to_pixel(coords[x_axis], coords[y_axis])
        if pixel is None:
            if count:
                out_of_bounds[category] += 1
            continue
        if category in draw_categories:
            if draw(category, pixel[0], pixel[1], size):
                rendered[category] += 1
            else:
                zero_radius[category] += 1
    return stats


def run(config: Config, inputs: Iterable[str | Path], *, low_memory: bool = False) -> RunResult:
    """Render one mask per configured category (plus one painted forest-extent mask each, when enabled)."""
    started = time.perf_counter()
    files = iter_layer_files(inputs, config.parser.file_pattern)
    if not files:
        raise FileNotFoundError("no .layer files found in the given inputs")

    classifier = build_classifier(config)
    mapper = build_mapper(config)
    canopy = build_canopy(config)
    categories = config.categories
    width, height = config.output.width, config.output.height
    extent = config.extent

    diagnostics = Diagnostics(files=[str(f) for f in files], canvas_width=width, canvas_height=height)
    if extent.active:
        diagnostics.extent_enabled = True
        diagnostics.extent_mode = extent.mode
        diagnostics.extent_close_radius = extent.close_radius
        diagnostics.extent_max_hole = extent.max_hole
        diagnostics.extent_max_distance = extent.max_distance
        if canopy is not None:
            diagnostics.extent_mask = str(canopy.path)
            diagnostics.extent_mask_size = (canopy.width, canopy.height)
            diagnostics.extent_threshold = extent.threshold
            diagnostics.extent_orientation = extent.orientation
    outputs: dict[str, Path] = {}
    extent_outputs: dict[str, Path] = {}
    combined = config.combined_inverse
    combined_output: Path | None = None

    one_at_a_time = low_memory and len(categories) > 1
    if one_at_a_time and extent.active:
        log.warning("--low-memory ignored: the forest-extent step needs every category canvas at once")
        one_at_a_time = False
    simultaneous = 1 if one_at_a_time else len(categories)
    if one_at_a_time and combined.enabled:
        simultaneous += 1  # the running union canvas
    diagnostics.canvas_bytes = Renderer.estimated_bytes(width, height, simultaneous)
    log.info(
        "canvas %dx%d, %d categor%s, %s of canvas memory%s%s",
        width, height, len(categories), "y" if len(categories) == 1 else "ies",
        format_bytes(diagnostics.canvas_bytes), " (low-memory mode)" if one_at_a_time else "",
        f", canopy mask adds {format_bytes(width * height)}" if canopy is not None else "",
    )

    if one_at_a_time:
        diagnostics.passes = len(categories)
        union = MaskCanvas(width, height, config.rendering.marker_value) if combined.enabled else None
        for index, category in enumerate(categories):
            log.info("category %d/%d: %s", index + 1, len(categories), category)
            renderer = Renderer(width, height, config.rendering.marker_sizes, [category], config.rendering.marker_value)
            stats = _stream_pass(config, files, classifier, mapper, renderer, diagnostics, {category}, count=index == 0)
            if index == 0:
                _absorb_stats(diagnostics, stats)
            outputs.update(renderer.save_all(config.output.directory, config.output.compress_level))
            if union is not None:
                union.merge(renderer.canvases[category].array)
            renderer.close()
        if union is not None:
            diagnostics.combined_sources = [f"{cat}.png" for cat in categories]
            combined_output = _write_combined_inverse(union, config, diagnostics)
            union.close()
    else:
        diagnostics.passes = 1
        renderer = Renderer(width, height, config.rendering.marker_sizes, categories, config.rendering.marker_value)
        stats = _stream_pass(config, files, classifier, mapper, renderer, diagnostics, set(categories), count=True)
        _absorb_stats(diagnostics, stats)
        outputs.update(renderer.save_all(config.output.directory, config.output.compress_level))

        extras: list[PackedMask] = []
        if extent.active:
            for category, canvas in renderer.canvases.items():
                diagnostics.extent_markers[category] = canvas.nonzero_pixels()
            if canopy is not None:
                canopy.load()
                diagnostics.canopy_pixels = canopy.canopy_pixels
            area_out: PackedMask | None = None
            if combined.enabled:
                # The canvases are about to be replaced by the extent results; keep the plain
                # markers and the forest area as packed bits for the combined inverse.
                extras.append(PackedMask.from_arrays([canvas.array for canvas in renderer.canvases.values()]))
                area_out = PackedMask(height, width)
                extras.append(area_out)
            extent_stats = compute_extent(
                canopy,
                {category: canvas.array for category, canvas in renderer.canvases.items()},
                mode=extent.mode,
                close_radius=extent.close_radius,
                max_hole=extent.max_hole,
                max_distance=extent.max_distance,
                tile=extent.tile,
                value=config.rendering.marker_value,
                area_out=area_out,
            )
            diagnostics.forest_pixels = extent_stats.forest_pixels
            diagnostics.unassigned_pixels = extent_stats.unassigned_pixels
            diagnostics.tiles_processed = extent_stats.tiles_processed
            diagnostics.tiles_total = extent_stats.tiles_total
            diagnostics.extent_painted = dict(extent_stats.painted)
            extent_outputs.update(
                renderer.save_all(config.output.directory, config.output.compress_level, extent.name_pattern)
            )
            if canopy is not None:
                canopy.close()
        if combined.enabled:
            # Every mask is on disk by now, so the category canvases can be consumed in place.
            sources = [f"{cat}.png" for cat in categories]
            if extent.active:
                sources += [extent.name_pattern.format(category=cat) for cat in categories] + ["forest area"]
            diagnostics.combined_sources = sources
            union, painted, inverse = renderer.combined_inverse(extras)
            combined_output = _write_combined_inverse(union, config, diagnostics, painted, inverse)
        renderer.close()

    diagnostics.outputs = {cat: str(path) for cat, path in outputs.items()}
    diagnostics.extent_outputs = {cat: str(path) for cat, path in extent_outputs.items()}
    diagnostics.elapsed_seconds = time.perf_counter() - started
    diagnostics.peak_rss_bytes = peak_rss_bytes()
    return RunResult(
        diagnostics=diagnostics, outputs=outputs, extent_outputs=extent_outputs, combined_inverse=combined_output,
    )


def _write_combined_inverse(
    union: MaskCanvas,
    config: Config,
    diagnostics: Diagnostics,
    painted: int | None = None,
    inverse: int | None = None,
) -> Path:
    """Invert ``union`` (unless already inverted, in which case counts are given) and save it."""
    if painted is None or inverse is None:
        painted, inverse = union.invert()
    target = Path(config.output.directory) / config.combined_inverse.filename
    log.info("writing %s (%d px painted in any category -> 0, %d px -> %d)", target, painted, inverse, union.value)
    path = union.save(target, config.output.compress_level)
    diagnostics.combined_inverse_output = str(path)
    diagnostics.combined_painted_pixels = painted
    diagnostics.combined_inverse_pixels = inverse
    return path


def _absorb_stats(diagnostics: Diagnostics, stats: ParseStats) -> None:
    diagnostics.lines_read = stats.lines
    diagnostics.parser_warning_count = stats.warning_count
    diagnostics.parser_warnings = list(stats.warnings)
    if stats.entities != diagnostics.entities_inspected:
        log.debug("parser saw %d entities, pipeline counted %d", stats.entities, diagnostics.entities_inspected)
