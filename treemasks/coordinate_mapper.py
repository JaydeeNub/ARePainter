"""Map world-space horizontal coordinates onto output pixel coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinateMapper:
    """Linear world -> pixel mapping.

    ``pixel_x = (world_x - min_x) / (max_x - min_x) * (width - 1)`` (rounded half-up),
    and likewise for y. With ``flip_y`` the row index is inverted so that ``max_y``
    lands on row 0 (top of the image); ``flip_x`` mirrors columns the same way so
    ``max_x`` lands on column 0. Coordinates outside the inclusive world bounds are
    reported as out of bounds (``to_pixel`` returns ``None``).
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: int
    height: int
    flip_y: bool = False
    flip_x: bool = False

    def __post_init__(self) -> None:
        if self.max_x <= self.min_x or self.max_y <= self.min_y:
            raise ValueError("world max bounds must be greater than min bounds")
        if self.width < 1 or self.height < 1:
            raise ValueError("output width and height must be at least 1 pixel")

    @property
    def pixels_per_unit_x(self) -> float:
        return (self.width - 1) / (self.max_x - self.min_x)

    @property
    def pixels_per_unit_y(self) -> float:
        return (self.height - 1) / (self.max_y - self.min_y)

    def in_bounds(self, world_x: float, world_y: float) -> bool:
        return self.min_x <= world_x <= self.max_x and self.min_y <= world_y <= self.max_y

    def to_pixel(self, world_x: float, world_y: float) -> tuple[int, int] | None:
        """Return ``(px, py)`` or ``None`` when the point lies outside the world bounds."""
        if not self.in_bounds(world_x, world_y):
            return None
        px = math.floor((world_x - self.min_x) * self.pixels_per_unit_x + 0.5)
        py = math.floor((world_y - self.min_y) * self.pixels_per_unit_y + 0.5)
        if self.flip_x:
            px = self.width - 1 - px
        if self.flip_y:
            py = self.height - 1 - py
        return px, py
