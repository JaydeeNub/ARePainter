"""Load, validate and override TreeMasks configuration (YAML or JSON)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .classifier import DEFAULT_SIZE_REGEX
from .extent import MODE_AREA, MODE_GAPS, MODES, ORIENTATION_OUTPUT, ORIENTATIONS


class ConfigError(ValueError):
    """Raised for missing or invalid configuration values."""


@dataclass(frozen=True)
class WorldBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass(frozen=True)
class OutputSettings:
    width: int
    height: int
    directory: str = "output"
    compress_level: int = 6


@dataclass(frozen=True)
class CoordinateSystem:
    flip_x: bool = False  # mirror horizontally: world max_x lands on column 0
    flip_y: bool = False  # mirror vertically: world max_y lands on row 0
    x_axis: int = 0       # component of "coords X Y Z" used for image x (0 = X, east)
    y_axis: int = 2       # component of "coords X Y Z" used for image y (2 = Z, north)


@dataclass(frozen=True)
class RenderingSettings:
    marker_sizes: dict[int, int]
    marker_value: int = 255


@dataclass(frozen=True)
class ParserSettings:
    file_pattern: str = "*.layer"
    entity_classes: tuple[str, ...] = ("Tree",)


@dataclass(frozen=True)
class ClassifierSettings:
    size_regex: str = DEFAULT_SIZE_REGEX
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtentSettings:
    """Forest-extent step: forest floor between crowns (or solid forest area) per forest type.

    Active when ``enabled`` is set or a canopy ``mask`` is given. Without a mask the tool's
    own rendered tree markers serve as the canopy.
    """

    enabled: bool = False
    mode: str = MODE_GAPS
    mask: str | None = None
    threshold: int = 128
    close_radius: int = 3
    max_hole: int = 64
    max_distance: float = 50.0
    tile: int = 1024
    suffix: str | None = None   # None: "_gaps" or "_area" depending on mode
    orientation: str = ORIENTATION_OUTPUT

    @property
    def active(self) -> bool:
        return self.enabled or bool(self.mask)

    @property
    def effective_suffix(self) -> str:
        if self.suffix:
            return self.suffix
        return "_gaps" if self.mode == MODE_GAPS else "_area"

    @property
    def name_pattern(self) -> str:
        return "{category}" + self.effective_suffix + ".png"


@dataclass(frozen=True)
class CombinedInverseSettings:
    """One extra PNG: the union of every final category mask, hard-inverted."""

    enabled: bool = False
    filename: str = "combined_inverse.png"


@dataclass(frozen=True)
class Config:
    world: WorldBounds
    output: OutputSettings
    rendering: RenderingSettings
    trees: dict[str, tuple[str, ...]]
    coordinate_system: CoordinateSystem = field(default_factory=CoordinateSystem)
    parser: ParserSettings = field(default_factory=ParserSettings)
    classifier: ClassifierSettings = field(default_factory=ClassifierSettings)
    extent: ExtentSettings = field(default_factory=ExtentSettings)
    combined_inverse: CombinedInverseSettings = field(default_factory=CombinedInverseSettings)
    source: str | None = None

    @property
    def categories(self) -> list[str]:
        return list(self.trees)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source", None)
        return data

    def validate(self) -> "Config":
        w = self.world
        if w.max_x <= w.min_x:
            raise ConfigError("world.max_x must be greater than world.min_x")
        if w.max_y <= w.min_y:
            raise ConfigError("world.max_y must be greater than world.min_y")
        if self.output.width < 1 or self.output.height < 1:
            raise ConfigError("output.width and output.height must be at least 1")
        if not 0 <= self.output.compress_level <= 9:
            raise ConfigError("output.compress_level must be between 0 and 9")
        cs = self.coordinate_system
        if cs.x_axis not in (0, 1, 2) or cs.y_axis not in (0, 1, 2) or cs.x_axis == cs.y_axis:
            raise ConfigError("coordinate_system.x_axis and y_axis must be distinct values in 0..2")
        sizes = self.rendering.marker_sizes
        if not sizes:
            raise ConfigError("rendering.marker_sizes must map at least one size category to a radius")
        for size, radius in sizes.items():
            if size < 0 or radius < 0:
                raise ConfigError("rendering.marker_sizes keys and radii must be >= 0")
        if sizes.get(0, 0) != 0:
            raise ConfigError("rendering.marker_sizes[0] must be 0: size-0 trees are never rendered")
        if not 1 <= self.rendering.marker_value <= 255:
            raise ConfigError("rendering.marker_value must be between 1 and 255")
        if not self.trees:
            raise ConfigError("trees must define at least one category with asset patterns")
        for name, patterns in self.trees.items():
            if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ConfigError(f"tree category name {name!r} must be a plain file-name-safe token")
            if not patterns:
                raise ConfigError(f"tree category {name!r} has no asset patterns")
        try:
            regex = re.compile(self.classifier.size_regex)
        except re.error as exc:
            raise ConfigError(f"classifier.size_regex is not a valid regex: {exc}") from exc
        if "size" not in regex.groupindex:
            raise ConfigError("classifier.size_regex must define a named group called 'size'")
        ext = self.extent
        if ext.mode not in MODES:
            raise ConfigError(f"extent.mode must be one of {', '.join(MODES)}")
        if not 1 <= ext.threshold <= 255:
            raise ConfigError("extent.threshold must be between 1 and 255")
        if ext.close_radius < 0:
            raise ConfigError("extent.close_radius must be >= 0")
        if ext.max_hole < 0:
            raise ConfigError("extent.max_hole must be >= 0")
        if ext.max_distance < 0:
            raise ConfigError("extent.max_distance must be >= 0")
        if ext.tile < 64:
            raise ConfigError("extent.tile must be at least 64 pixels")
        if ext.suffix is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", ext.suffix):
            raise ConfigError("extent.suffix must be a file-name-safe token (or null for the default)")
        if ext.orientation not in ORIENTATIONS:
            raise ConfigError(f"extent.orientation must be one of {', '.join(ORIENTATIONS)}")
        name = self.combined_inverse.filename
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ConfigError("combined_inverse.filename must be a plain file name such as combined_inverse.png")
        return self


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ConfigError(f"missing required setting '{where}.{key}'" if where else f"missing required section '{key}'")
    return mapping[key]


def _as_str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"'{where}' must be a list of strings")
    return tuple(value)


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"'{where}' must be an integer")
    try:
        number = float(value)
    except ValueError as exc:
        raise ConfigError(f"'{where}' must be an integer") from exc
    if number != int(number):
        raise ConfigError(f"'{where}' must be an integer")
    return int(number)


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"'{where}' must be a number")
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"'{where}' must be a number") from exc


def _as_optional_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def config_from_mapping(data: Mapping[str, Any], source: str | None = None) -> Config:
    """Build a validated Config from a plain mapping (parsed YAML/JSON)."""
    if not isinstance(data, Mapping):
        raise ConfigError("configuration root must be a mapping")

    world_map = _require(data, "world", "")
    world = WorldBounds(
        min_x=_as_float(_require(world_map, "min_x", "world"), "world.min_x"),
        max_x=_as_float(_require(world_map, "max_x", "world"), "world.max_x"),
        min_y=_as_float(_require(world_map, "min_y", "world"), "world.min_y"),
        max_y=_as_float(_require(world_map, "max_y", "world"), "world.max_y"),
    )

    output_map = _require(data, "output", "")
    output = OutputSettings(
        width=_as_int(_require(output_map, "width", "output"), "output.width"),
        height=_as_int(_require(output_map, "height", "output"), "output.height"),
        directory=str(output_map.get("directory", OutputSettings.directory)),
        compress_level=_as_int(output_map.get("compress_level", OutputSettings.compress_level), "output.compress_level"),
    )

    cs_map = data.get("coordinate_system") or {}
    coordinate_system = CoordinateSystem(
        flip_x=bool(cs_map.get("flip_x", False)),
        flip_y=bool(cs_map.get("flip_y", False)),
        x_axis=_as_int(cs_map.get("x_axis", CoordinateSystem.x_axis), "coordinate_system.x_axis"),
        y_axis=_as_int(cs_map.get("y_axis", CoordinateSystem.y_axis), "coordinate_system.y_axis"),
    )

    rendering_map = _require(data, "rendering", "")
    raw_sizes = _require(rendering_map, "marker_sizes", "rendering")
    if not isinstance(raw_sizes, Mapping):
        raise ConfigError("rendering.marker_sizes must be a mapping of size category -> radius")
    marker_sizes = {
        _as_int(k, "rendering.marker_sizes key"): _as_int(v, f"rendering.marker_sizes[{k}]")
        for k, v in raw_sizes.items()
    }
    rendering = RenderingSettings(
        marker_sizes=marker_sizes,
        marker_value=_as_int(rendering_map.get("marker_value", RenderingSettings.marker_value), "rendering.marker_value"),
    )

    parser_map = data.get("parser") or {}
    parser = ParserSettings(
        file_pattern=str(parser_map.get("file_pattern", ParserSettings.file_pattern)),
        entity_classes=_as_str_tuple(parser_map.get("entity_classes", list(ParserSettings.entity_classes)), "parser.entity_classes"),
    )

    classifier_map = data.get("classifier") or {}
    classifier = ClassifierSettings(
        size_regex=str(classifier_map.get("size_regex", DEFAULT_SIZE_REGEX)),
        exclude=_as_str_tuple(classifier_map.get("exclude"), "classifier.exclude"),
    )

    extent_map = data.get("extent") or {}
    if not isinstance(extent_map, Mapping):
        raise ConfigError("extent must be a mapping")
    raw_suffix = extent_map.get("suffix")
    extent = ExtentSettings(
        enabled=bool(extent_map.get("enabled", False)),
        mode=str(extent_map.get("mode", ExtentSettings.mode)).lower(),
        mask=_as_optional_path(extent_map.get("mask")),
        threshold=_as_int(extent_map.get("threshold", ExtentSettings.threshold), "extent.threshold"),
        close_radius=_as_int(extent_map.get("close_radius", ExtentSettings.close_radius), "extent.close_radius"),
        max_hole=_as_int(extent_map.get("max_hole", ExtentSettings.max_hole), "extent.max_hole"),
        max_distance=_as_float(extent_map.get("max_distance", ExtentSettings.max_distance), "extent.max_distance"),
        tile=_as_int(extent_map.get("tile", ExtentSettings.tile), "extent.tile"),
        suffix=None if raw_suffix is None or str(raw_suffix) == "" else str(raw_suffix),
        orientation=str(extent_map.get("orientation", ExtentSettings.orientation)).lower(),
    )

    combined_map = data.get("combined_inverse") or {}
    if not isinstance(combined_map, Mapping):
        raise ConfigError("combined_inverse must be a mapping")
    combined_inverse = CombinedInverseSettings(
        enabled=bool(combined_map.get("enabled", False)),
        filename=str(combined_map.get("filename", CombinedInverseSettings.filename)),
    )

    trees_map = _require(data, "trees", "")
    if not isinstance(trees_map, Mapping):
        raise ConfigError("trees must be a mapping of category -> list of asset patterns")
    trees = {str(name): _as_str_tuple(patterns, f"trees.{name}") for name, patterns in trees_map.items()}

    return Config(
        world=world,
        output=output,
        rendering=rendering,
        trees=trees,
        coordinate_system=coordinate_system,
        parser=parser,
        classifier=classifier,
        extent=extent,
        combined_inverse=combined_inverse,
        source=source,
    ).validate()


def load_config(path: str | Path) -> Config:
    """Read a YAML (.yaml/.yml) or JSON (.json) configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"configuration file {path} is empty")
    return config_from_mapping(data, source=str(path))


def apply_overrides(config: Config, overrides: Mapping[str, Any]) -> Config:
    """Return a new Config with non-None override values applied (typically from the CLI).

    Recognised keys: world_min_x, world_max_x, world_min_y, world_max_y, width, height,
    flip_x, flip_y, marker_sizes (partial mapping merged over the configured one),
    marker_value, output_dir, compress_level, file_pattern, extent_enabled (True switches the
    step on without a mask), extent_mode, extent_mask, extent_threshold, extent_close_radius,
    extent_max_hole, extent_max_distance, extent_orientation, extent_disabled (True
    switches the extent step off, dropping any configured mask), combined_inverse (bool)
    and combined_inverse_file.
    """
    values = {k: v for k, v in overrides.items() if v is not None}
    if not values:
        return config

    world = config.world
    world_updates = {
        key[len("world_"):]: values[key]
        for key in ("world_min_x", "world_max_x", "world_min_y", "world_max_y")
        if key in values
    }
    if world_updates:
        world = replace(world, **{k: float(v) for k, v in world_updates.items()})

    output = config.output
    output_updates: dict[str, Any] = {}
    if "width" in values:
        output_updates["width"] = int(values["width"])
    if "height" in values:
        output_updates["height"] = int(values["height"])
    if "output_dir" in values:
        output_updates["directory"] = str(values["output_dir"])
    if "compress_level" in values:
        output_updates["compress_level"] = int(values["compress_level"])
    if output_updates:
        output = replace(output, **output_updates)

    coordinate_system = config.coordinate_system
    cs_updates: dict[str, Any] = {}
    if "flip_x" in values:
        cs_updates["flip_x"] = bool(values["flip_x"])
    if "flip_y" in values:
        cs_updates["flip_y"] = bool(values["flip_y"])
    if cs_updates:
        coordinate_system = replace(coordinate_system, **cs_updates)

    rendering = config.rendering
    rendering_updates: dict[str, Any] = {}
    if "marker_sizes" in values:
        merged = dict(rendering.marker_sizes)
        merged.update({int(k): int(v) for k, v in dict(values["marker_sizes"]).items()})
        rendering_updates["marker_sizes"] = merged
    if "marker_value" in values:
        rendering_updates["marker_value"] = int(values["marker_value"])
    if rendering_updates:
        rendering = replace(rendering, **rendering_updates)

    parser = config.parser
    if "file_pattern" in values:
        parser = replace(parser, file_pattern=str(values["file_pattern"]))

    extent = config.extent
    extent_updates: dict[str, Any] = {}
    if values.get("extent_enabled"):
        extent_updates["enabled"] = True
    if "extent_mode" in values:
        extent_updates["mode"] = str(values["extent_mode"]).lower()
    if "extent_mask" in values:
        extent_updates["mask"] = _as_optional_path(values["extent_mask"])
    if "extent_threshold" in values:
        extent_updates["threshold"] = int(values["extent_threshold"])
    if "extent_close_radius" in values:
        extent_updates["close_radius"] = int(values["extent_close_radius"])
    if "extent_max_hole" in values:
        extent_updates["max_hole"] = int(values["extent_max_hole"])
    if "extent_max_distance" in values:
        extent_updates["max_distance"] = float(values["extent_max_distance"])
    if "extent_orientation" in values:
        extent_updates["orientation"] = str(values["extent_orientation"]).lower()
    if values.get("extent_disabled"):
        extent_updates["mask"] = None
        extent_updates["enabled"] = False
    if extent_updates:
        extent = replace(extent, **extent_updates)

    combined = config.combined_inverse
    combined_updates: dict[str, Any] = {}
    if "combined_inverse" in values:
        combined_updates["enabled"] = bool(values["combined_inverse"])
    if "combined_inverse_file" in values:
        combined_updates["filename"] = str(values["combined_inverse_file"])
    if combined_updates:
        combined = replace(combined, **combined_updates)

    return replace(
        config,
        world=world,
        output=output,
        coordinate_system=coordinate_system,
        rendering=rendering,
        parser=parser,
        extent=extent,
        combined_inverse=combined,
    ).validate()
