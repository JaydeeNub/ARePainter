"""Streaming parser for Enfusion (Arma Reforger) ``.layer`` files.

The format is a brace-delimited text tree as written by the World Editor::

    $grp PolylineShapeEntity {                 # group: each anonymous block is an instance
     {
      coords 15611.431 321.007 12357.305      # local position: X east, Y up, Z north
      Points { ... }                           # named property block (ignored)
      {                                        # anonymous block inside an entity = children
       ForestGeneratorEntity : "{GUID}Prefabs/.../JD_Forest_Spruce.et" {
        coords 0 0 0
        {
         $grp Tree : "{GUID}PrefabLibrary/.../t_picea_abies_2sw.et" {
          {
           coords -62.49 -3.921 25.999
           angles 2.488 -36.652 2.561
           scale 0.816
          }
         }
         Tree : "{GUID}.../t_betula_pendula_2sw_aut.et" {
          coords 2.539 -0.852 106.769
         }
        }
       }
      }
     }
    }

``coords`` of a child are relative to its parent entity, so world positions
are obtained by accumulating the ancestor chain (yaw rotation of a parent, if
any, is applied to the child's offset). Parsing is line based and streams the
file: only the chain of open blocks is kept in memory, and entities are
yielded as soon as their closing brace is read.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

log = logging.getLogger(__name__)

# Header line: optional "$grp", optional class name, optional ': "prefab"', optional '"guid"', then "{".
_HEADER_RE = re.compile(
    r'^(?P<grp>\$grp\s+)?'
    r'(?P<name>[A-Za-z_][A-Za-z0-9_]*)?\s*'
    r'(?::\s*"(?P<prefab>[^"]*)")?\s*'
    r'(?:"(?P<guid>[^"]*)")?\s*'
    r'\{$'
)

Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Entity:
    """One placed entity with its resolved world-space position."""

    class_name: str
    prefab: str           # e.g. "{009A68DF0B31B7B7}PrefabLibrary/.../t_picea_abies_2sw.et"
    coords: Vec3          # world position (X east, Y up, Z north)
    source: str           # file the entity came from
    line: int             # line of the entity's opening brace

    @property
    def asset_name(self) -> str:
        return asset_basename(self.prefab)


def asset_basename(prefab: str) -> str:
    """Return the file name part of a prefab reference (``{GUID}path/file.et`` -> ``file.et``)."""
    name = prefab.replace("\\", "/").rsplit("/", 1)[-1]
    if name.startswith("{") and "}" in name:
        name = name[name.index("}") + 1:]
    return name


@dataclass
class ParseStats:
    files: int = 0
    lines: int = 0
    entities: int = 0
    warning_count: int = 0
    warnings: list[str] = field(default_factory=list)
    max_warnings_kept: int = 50

    def warn(self, message: str) -> None:
        self.warning_count += 1
        if len(self.warnings) < self.max_warnings_kept:
            self.warnings.append(message)
        log.debug("parser warning: %s", message)


# Frame kinds on the block stack.
_ROOT, _GROUP, _ENTITY, _CHILDREN, _BLOCK = range(5)


class _EntityFrame:
    """State for an open entity block (either a named entity or a group instance)."""

    __slots__ = ("class_name", "prefab", "parent", "world", "yaw", "has_coords", "children_seen", "line")

    def __init__(self, class_name: str, prefab: str, parent: "_EntityFrame | None", line: int) -> None:
        self.class_name = class_name
        self.prefab = prefab
        self.parent = parent
        # Until "coords" is read the entity sits at its parent's origin.
        self.world: Vec3 = parent.world if parent is not None else (0.0, 0.0, 0.0)
        self.yaw: float = parent.yaw if parent is not None else 0.0
        self.has_coords = False
        self.children_seen = False
        self.line = line

    def set_coords(self, local: Vec3) -> None:
        if self.parent is None:
            self.world = local
        else:
            self.world = transform_local(self.parent.world, self.parent.yaw, local)
        self.has_coords = True

    def set_angles(self, angles: Vec3) -> None:
        # Enfusion "angles" are (pitch, yaw, roll) in degrees; only yaw affects the top-down
        # position of children. Yaw accumulates down the hierarchy.
        base = self.parent.yaw if self.parent is not None else 0.0
        self.yaw = base + angles[1]


def transform_local(origin: Vec3, yaw_degrees: float, local: Vec3) -> Vec3:
    """Convert a child's local offset into world space given the parent's origin and yaw.

    Enfusion is a left-handed, Y-up system, so a positive yaw rotates clockwise when seen
    from above (DirectX ``RotationY`` convention). Pitch and roll are ignored.
    """
    lx, ly, lz = local
    if yaw_degrees:
        rad = math.radians(yaw_degrees)
        c, s = math.cos(rad), math.sin(rad)
        lx, lz = lx * c + lz * s, -lx * s + lz * c
    return (origin[0] + lx, origin[1] + ly, origin[2] + lz)


def _parse_vec3(text: str) -> Vec3 | None:
    parts = text.split()
    if len(parts) < 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


def parse_file(path: str | Path, stats: ParseStats | None = None) -> Iterator[Entity]:
    """Stream every entity in one .layer file, yielding them in closing-brace order."""
    stats = stats if stats is not None else ParseStats()
    source = str(path)
    stack: list[tuple[int, object]] = [(_ROOT, None)]
    lineno = 0
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue

            if line == "}":
                if len(stack) == 1:
                    stats.warn(f"{source}:{lineno}: unexpected closing brace")
                    continue
                kind, payload = stack.pop()
                if kind == _ENTITY:
                    frame: _EntityFrame = payload  # type: ignore[assignment]
                    stats.entities += 1
                    yield Entity(frame.class_name, frame.prefab, frame.world, source, frame.line)
                continue

            if line.endswith("{"):
                match = _HEADER_RE.match(line)
                if match is None:
                    stats.warn(f"{source}:{lineno}: unrecognised block header {line!r}; skipping block")
                    stack.append((_BLOCK, None))
                    continue
                name = match.group("name")
                prefab = match.group("prefab") or ""
                is_group = match.group("grp") is not None
                kind, payload = stack[-1]

                if kind == _BLOCK:
                    stack.append((_BLOCK, None))
                elif kind in (_ROOT, _CHILDREN):
                    parent: _EntityFrame | None = payload  # type: ignore[assignment]
                    if is_group:
                        stack.append((_GROUP, (name or "", prefab, parent)))
                    elif name is None:
                        stack.append((_CHILDREN, parent))  # nested anonymous block: pass through
                    else:
                        stack.append((_ENTITY, _EntityFrame(name, prefab, parent, lineno)))
                elif kind == _GROUP:
                    group_name, group_prefab, parent = payload  # type: ignore[misc]
                    if name is None:
                        stack.append((_ENTITY, _EntityFrame(group_name, group_prefab, parent, lineno)))
                    else:  # unexpected named block inside a group; treat it as an entity
                        stack.append((_ENTITY, _EntityFrame(name, prefab, parent, lineno)))
                elif kind == _ENTITY:
                    frame = payload  # type: ignore[assignment]
                    if name is None and not is_group:
                        frame.children_seen = True
                        stack.append((_CHILDREN, frame))
                    else:  # a named property block such as "Points {"
                        stack.append((_BLOCK, None))
                continue

            if line.endswith("{}"):
                continue  # empty inline block, nothing to track

            kind, payload = stack[-1]
            if kind == _ENTITY:
                frame = payload  # type: ignore[assignment]
                if line.startswith("coords "):
                    vec = _parse_vec3(line[7:])
                    if vec is None:
                        stats.warn(f"{source}:{lineno}: unreadable coords {line!r}")
                    else:
                        if frame.children_seen:
                            stats.warn(
                                f"{source}:{lineno}: coords appear after the children block; "
                                "child positions of this entity may be wrong"
                            )
                        frame.set_coords(vec)
                elif line.startswith("angles "):
                    vec = _parse_vec3(line[7:])
                    if vec is not None:
                        frame.set_angles(vec)

    stats.lines += lineno
    if len(stack) != 1:
        stats.warn(f"{source}: {len(stack) - 1} block(s) still open at end of file")


def iter_layer_files(inputs: Iterable[str | Path], pattern: str = "*.layer") -> list[Path]:
    """Expand files and directories (searched recursively) into an ordered, de-duplicated file list."""
    files: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            candidates = sorted(p for p in path.rglob(pattern) if p.is_file())
        elif path.is_file():
            candidates = [path]
        else:
            raise FileNotFoundError(f"input not found: {path}")
        for candidate in candidates:
            key = candidate.resolve()
            if key not in seen:
                seen.add(key)
                files.append(candidate)
    return files


def parse_paths(paths: Iterable[str | Path], stats: ParseStats | None = None) -> Iterator[Entity]:
    """Stream entities from several files one after another."""
    stats = stats if stats is not None else ParseStats()
    for path in paths:
        log.info("parsing %s", path)
        stats.files += 1
        yield from parse_file(path, stats)
