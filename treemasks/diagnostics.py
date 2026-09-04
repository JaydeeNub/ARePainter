"""Run statistics, timing and memory reporting."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .classifier import EXCLUDED, NO_SIZE, OTHER_CLASS, TREE, UNKNOWN, UNKNOWN_SIZE, Classification


def current_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def peak_rss_bytes() -> int | None:
    """Peak resident set size of this process, or None when it cannot be determined."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak:
            return int(peak)
    except ImportError:
        pass
    try:
        import resource  # POSIX only

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage if sys.platform == "darwin" else usage * 1024)
    except (ImportError, AttributeError, OSError):
        pass
    return current_rss_bytes()


def format_bytes(count: int | None) -> str:
    if count is None:
        return "n/a"
    value = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{count} B"


def format_int(value: int) -> str:
    return f"{value:,}"


@dataclass
class Diagnostics:
    """Counters gathered during a run plus timing/memory figures."""

    files: list[str] = field(default_factory=list)
    lines_read: int = 0
    entities_inspected: int = 0
    tree_class_entities: int = 0
    other_class_entities: int = 0
    trees_by_category_size: dict[str, Counter] = field(default_factory=dict)
    rendered: Counter = field(default_factory=Counter)
    skipped_size0: Counter = field(default_factory=Counter)
    out_of_bounds: Counter = field(default_factory=Counter)
    zero_radius: Counter = field(default_factory=Counter)
    unknown_assets: Counter = field(default_factory=Counter)
    excluded_assets: Counter = field(default_factory=Counter)
    no_size_assets: Counter = field(default_factory=Counter)
    unknown_size_assets: Counter = field(default_factory=Counter)
    parser_warnings: list[str] = field(default_factory=list)
    parser_warning_count: int = 0
    outputs: dict[str, str] = field(default_factory=dict)
    canvas_width: int = 0
    canvas_height: int = 0
    canvas_bytes: int = 0
    passes: int = 1
    elapsed_seconds: float = 0.0
    peak_rss_bytes: int | None = None
    # forest extent (forest floor between crowns, or solid area, per forest type)
    extent_enabled: bool = False
    extent_mode: str = ""
    extent_mask: str | None = None        # None: the rendered tree markers are the canopy
    extent_mask_size: tuple[int, int] | None = None
    extent_threshold: int = 0
    extent_close_radius: int = 0
    extent_max_hole: int = 0
    extent_max_distance: float = 0.0
    extent_orientation: str = ""
    canopy_pixels: int | None = None      # raw canopy pixels in the whole mask (texture only)
    forest_pixels: int = 0                # solid forest-area pixels in processed tiles
    unassigned_pixels: int = 0            # forest pixels with no tree marker within max_distance
    tiles_processed: int = 0
    tiles_total: int = 0
    extent_markers: dict[str, int] = field(default_factory=dict)   # plain marker pixels per category
    extent_painted: dict[str, int] = field(default_factory=dict)   # painted pixels per category
    extent_outputs: dict[str, str] = field(default_factory=dict)
    # combined inverse (union of every output mask and the forest area, hard-inverted)
    combined_inverse_output: str | None = None
    combined_sources: list[str] = field(default_factory=list)
    combined_painted_pixels: int = 0     # pixels painted in any source (-> 0)
    combined_inverse_pixels: int = 0     # pixels painted in no source (-> marker value)

    # ----- recording -----------------------------------------------------------------
    def record(self, classification: Classification) -> None:
        self.entities_inspected += 1
        status = classification.status
        if status == OTHER_CLASS:
            self.other_class_entities += 1
            return
        self.tree_class_entities += 1
        if status == TREE:
            counter = self.trees_by_category_size.get(classification.category)
            if counter is None:
                counter = self.trees_by_category_size[classification.category] = Counter()
            counter[classification.size] += 1
        elif status == EXCLUDED:
            self.excluded_assets[classification.asset_name] += 1
        elif status == UNKNOWN:
            self.unknown_assets[classification.asset_name] += 1
        elif status == NO_SIZE:
            self.no_size_assets[classification.asset_name] += 1
        elif status == UNKNOWN_SIZE:
            self.unknown_size_assets[classification.asset_name] += 1

    # ----- totals ----------------------------------------------------------------------
    def trees_in(self, category: str) -> int:
        return sum(self.trees_by_category_size.get(category, Counter()).values())

    @property
    def trees_total(self) -> int:
        return sum(sum(c.values()) for c in self.trees_by_category_size.values())

    @property
    def rendered_total(self) -> int:
        return sum(self.rendered.values())

    @property
    def skipped_size0_total(self) -> int:
        return sum(self.skipped_size0.values())

    @property
    def out_of_bounds_total(self) -> int:
        return sum(self.out_of_bounds.values())

    @property
    def zero_radius_total(self) -> int:
        return sum(self.zero_radius.values())

    @property
    def unknown_total(self) -> int:
        return sum(self.unknown_assets.values())

    @property
    def excluded_total(self) -> int:
        return sum(self.excluded_assets.values())

    @property
    def no_size_total(self) -> int:
        return sum(self.no_size_assets.values())

    @property
    def unknown_size_total(self) -> int:
        return sum(self.unknown_size_assets.values())

    # ----- output ----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "files": list(self.files),
            "lines_read": self.lines_read,
            "entities_inspected": self.entities_inspected,
            "tree_class_entities": self.tree_class_entities,
            "other_class_entities": self.other_class_entities,
            "trees_by_category_size": {
                cat: {str(size): n for size, n in sorted(counter.items())}
                for cat, counter in self.trees_by_category_size.items()
            },
            "rendered": dict(self.rendered),
            "skipped_size0": dict(self.skipped_size0),
            "out_of_bounds": dict(self.out_of_bounds),
            "zero_radius": dict(self.zero_radius),
            "unknown_assets": dict(self.unknown_assets.most_common()),
            "excluded_assets": dict(self.excluded_assets.most_common()),
            "no_size_assets": dict(self.no_size_assets.most_common()),
            "unknown_size_assets": dict(self.unknown_size_assets.most_common()),
            "parser_warning_count": self.parser_warning_count,
            "parser_warnings": list(self.parser_warnings),
            "outputs": dict(self.outputs),
            "canvas": {"width": self.canvas_width, "height": self.canvas_height, "bytes": self.canvas_bytes},
            "passes": self.passes,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "peak_rss_bytes": self.peak_rss_bytes,
            "extent": {
                "enabled": self.extent_enabled,
                "mode": self.extent_mode,
                "mask": self.extent_mask,
                "mask_size": list(self.extent_mask_size) if self.extent_mask_size else None,
                "threshold": self.extent_threshold,
                "close_radius": self.extent_close_radius,
                "max_hole": self.extent_max_hole,
                "max_distance": self.extent_max_distance,
                "orientation": self.extent_orientation,
                "canopy_pixels": self.canopy_pixels,
                "forest_pixels": self.forest_pixels,
                "unassigned_pixels": self.unassigned_pixels,
                "tiles_processed": self.tiles_processed,
                "tiles_total": self.tiles_total,
                "marker_pixels": dict(self.extent_markers),
                "painted_pixels": dict(self.extent_painted),
                "outputs": dict(self.extent_outputs),
            },
            "combined_inverse": {
                "enabled": self.combined_inverse_output is not None,
                "output": self.combined_inverse_output,
                "sources": list(self.combined_sources),
                "painted_union_pixels": self.combined_painted_pixels,
                "inverse_pixels": self.combined_inverse_pixels,
            },
        }

    def format_summary(
        self,
        *,
        categories: Iterable[str] | None = None,
        sizes: Iterable[int] | None = None,
        report_unknown: bool = False,
        top_n: int = 20,
    ) -> str:
        categories = list(categories) if categories is not None else list(self.trees_by_category_size)
        for cat in self.trees_by_category_size:
            if cat not in categories:
                categories.append(cat)
        if sizes is None:
            seen: set[int] = set()
            for counter in self.trees_by_category_size.values():
                seen.update(counter)
            sizes = sorted(seen) if seen else [0, 1, 2, 3]
        sizes = list(sizes)

        lines: list[str] = ["TreeMasks summary", "================="]
        lines.append(f"Files processed      : {format_int(len(self.files))}")
        for path in self.files[:10]:
            lines.append(f"  {path}")
        if len(self.files) > 10:
            lines.append(f"  ... and {len(self.files) - 10} more")
        lines.append(f"Lines read           : {format_int(self.lines_read)}")
        lines.append(f"Entities inspected   : {format_int(self.entities_inspected)}")
        lines.append(f"  tree-class entities: {format_int(self.tree_class_entities)}")
        lines.append(f"  other entities     : {format_int(self.other_class_entities)}")
        lines.append("")

        header = f"{'category':<14}{'trees':>9}" + "".join(f"{'size ' + str(s):>9}" for s in sizes)
        header += f"{'size0 skip':>12}{'out of bnd':>12}{'rendered':>10}"
        lines.append("Trees by category and size category")
        lines.append(header)
        lines.append("-" * len(header))
        for cat in categories:
            counter = self.trees_by_category_size.get(cat, Counter())
            row = f"{cat:<14}{format_int(sum(counter.values())):>9}"
            row += "".join(f"{format_int(counter.get(s, 0)):>9}" for s in sizes)
            row += f"{format_int(self.skipped_size0.get(cat, 0)):>12}"
            row += f"{format_int(self.out_of_bounds.get(cat, 0)):>12}"
            row += f"{format_int(self.rendered.get(cat, 0)):>10}"
            lines.append(row)
        total_row = f"{'TOTAL':<14}{format_int(self.trees_total):>9}"
        total_row += "".join(
            f"{format_int(sum(c.get(s, 0) for c in self.trees_by_category_size.values())):>9}" for s in sizes
        )
        total_row += f"{format_int(self.skipped_size0_total):>12}{format_int(self.out_of_bounds_total):>12}"
        total_row += f"{format_int(self.rendered_total):>10}"
        lines.append(total_row)
        lines.append("")

        lines.append(f"Skipped size-0 trees : {format_int(self.skipped_size0_total)}")
        lines.append(f"Out-of-bounds trees  : {format_int(self.out_of_bounds_total)}")
        if self.zero_radius_total:
            lines.append(f"Zero-radius markers  : {format_int(self.zero_radius_total)}  (size >0 but radius configured as 0)")
        lines.append(
            f"Excluded assets      : {format_int(self.excluded_total)} entities, "
            f"{format_int(len(self.excluded_assets))} distinct assets"
        )
        lines.append(
            f"Unknown tree assets  : {format_int(self.unknown_total)} entities, "
            f"{format_int(len(self.unknown_assets))} distinct assets"
        )
        if self.no_size_total:
            lines.append(
                f"No size in name      : {format_int(self.no_size_total)} entities, "
                f"{format_int(len(self.no_size_assets))} distinct assets"
            )
        if self.unknown_size_total:
            lines.append(
                f"Unknown size digit   : {format_int(self.unknown_size_total)} entities, "
                f"{format_int(len(self.unknown_size_assets))} distinct assets"
            )
        if not report_unknown and (self.unknown_assets or self.excluded_assets or self.no_size_assets):
            lines.append("  (use --report-unknown to list asset names)")
        lines.append(f"Parser warnings      : {format_int(self.parser_warning_count)}")
        for warning in self.parser_warnings[:5]:
            lines.append(f"  {warning}")
        if self.parser_warning_count > 5:
            lines.append(f"  ... {self.parser_warning_count - 5} more (see --verbose)")

        if report_unknown:
            for title, counter in (
                ("Unknown tree assets (not matched by any category)", self.unknown_assets),
                ("Excluded assets (matched an exclude pattern)", self.excluded_assets),
                ("Assets without a size digit", self.no_size_assets),
                ("Assets with a size digit outside the marker table", self.unknown_size_assets),
            ):
                if not counter:
                    continue
                lines.append("")
                lines.append(title)
                for name, count in counter.most_common(top_n):
                    lines.append(f"  {format_int(count):>8}  {name}")
                if len(counter) > top_n:
                    lines.append(f"  ... {len(counter) - top_n} more")

        lines.append("")
        lines.append(f"Outputs ({self.canvas_width}x{self.canvas_height} 8-bit grayscale PNG)")
        for cat, path in self.outputs.items():
            lines.append(f"  {cat:<12} {path}")

        if self.extent_enabled:
            lines.append("")
            what = "forest floor between crowns" if self.extent_mode == "gaps" else "solid forest area"
            lines.append(f"Forest extent ({self.extent_mode}: {what}, per forest type)")
            if self.extent_mask is not None:
                size = f"{self.extent_mask_size[0]}x{self.extent_mask_size[1]}" if self.extent_mask_size else "?"
                lines.append(
                    f"  canopy             : {self.extent_mask} ({size}, threshold >= {self.extent_threshold}, "
                    f"orientation {self.extent_orientation})"
                )
                canopy = format_int(self.canopy_pixels) if self.canopy_pixels is not None else "n/a"
                lines.append(f"  canopy pixels      : {canopy}")
            else:
                lines.append("  canopy             : the rendered tree markers of all categories")
            lines.append(
                f"  forest area        : close {self.extent_close_radius} px, holes up to {self.extent_max_hole} px "
                f"count as inside, reach {self.extent_max_distance:g} px from the nearest tree"
            )
            lines.append(
                f"  forest area px     : {format_int(self.forest_pixels)} in "
                f"{format_int(self.tiles_processed)} of {format_int(self.tiles_total)} tiles with trees"
            )
            lines.append(f"  unassigned         : {format_int(self.unassigned_pixels)} px (no tree within reach)")
            if self.unassigned_pixels:
                lines.append("    raise extent.max_distance if these pixels lie inside your forests")
            ext_header = f"  {'category':<14}{'marker px':>12}{'painted px':>13}{'share':>8}  output"
            lines.append(ext_header)
            lines.append("  " + "-" * (len(ext_header) - 2))
            painted_total = sum(self.extent_painted.values())
            for cat in categories:
                markers = self.extent_markers.get(cat, 0)
                painted = self.extent_painted.get(cat, 0)
                share = f"{100.0 * painted / painted_total:.1f}%" if painted_total else "n/a"
                lines.append(
                    f"  {cat:<14}{format_int(markers):>12}{format_int(painted):>13}{share:>8}  "
                    f"{self.extent_outputs.get(cat, '')}"
                )

        if self.combined_inverse_output is not None:
            lines.append("")
            lines.append(f"Combined inverse     : {self.combined_inverse_output}")
            if self.combined_sources:
                lines.append(f"  union of           : {', '.join(self.combined_sources)}")
            lines.append(
                f"  painted anywhere -> 0 : {format_int(self.combined_painted_pixels)} px;  "
                f"painted nowhere -> marker value : {format_int(self.combined_inverse_pixels)} px"
            )
        lines.append(f"Render passes        : {self.passes}")
        lines.append(f"Canvas memory        : {format_bytes(self.canvas_bytes)}")
        lines.append(f"Peak memory (RSS)    : {format_bytes(self.peak_rss_bytes)}")
        lines.append(f"Execution time       : {self.elapsed_seconds:.2f} s")
        return "\n".join(lines)
