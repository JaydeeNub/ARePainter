from __future__ import annotations

import random

import pytest

from treemasks.coordinate_mapper import CoordinateMapper


def test_corners_map_to_canvas_extremes():
    mapper = CoordinateMapper(0, 32000, 0, 32000, 31501, 31501)
    assert mapper.to_pixel(0, 0) == (0, 0)
    assert mapper.to_pixel(32000, 32000) == (31500, 31500)
    assert mapper.to_pixel(16000, 16000) == (15750, 15750)


def test_reference_formula_matches():
    mapper = CoordinateMapper(100, 900, -50, 350, 640, 480)
    rng = random.Random(1234)
    for _ in range(500):
        wx = rng.uniform(100, 900)
        wy = rng.uniform(-50, 350)
        expected_x = ((wx - 100) / (900 - 100)) * (640 - 1)
        expected_y = ((wy + 50) / (350 + 50)) * (480 - 1)
        px, py = mapper.to_pixel(wx, wy)
        assert abs(px - expected_x) <= 0.5
        assert abs(py - expected_y) <= 0.5


def test_rounding_is_half_up():
    mapper = CoordinateMapper(0, 10, 0, 10, 11, 11)  # 1 px per unit
    assert mapper.to_pixel(2.5, 3.5) == (3, 4)
    assert mapper.to_pixel(2.49, 3.49) == (2, 3)


def test_flip_y_inverts_rows():
    plain = CoordinateMapper(0, 100, 0, 100, 101, 51)
    flipped = CoordinateMapper(0, 100, 0, 100, 101, 51, flip_y=True)
    assert plain.to_pixel(10, 0) == (10, 0)
    assert flipped.to_pixel(10, 0) == (10, 50)
    assert flipped.to_pixel(10, 100) == (10, 0)
    assert flipped.to_pixel(10, 50) == (10, 25)


def test_flip_x_mirrors_columns():
    plain = CoordinateMapper(0, 100, 0, 100, 51, 101)
    flipped = CoordinateMapper(0, 100, 0, 100, 51, 101, flip_x=True)
    assert plain.to_pixel(0, 10) == (0, 10)
    assert flipped.to_pixel(0, 10) == (50, 10)
    assert flipped.to_pixel(100, 10) == (0, 10)
    assert flipped.to_pixel(50, 10) == (25, 10)


def test_both_flips_rotate_by_180_degrees():
    plain = CoordinateMapper(0, 100, 0, 100, 101, 51)
    both = CoordinateMapper(0, 100, 0, 100, 101, 51, flip_x=True, flip_y=True)
    for wx, wy in ((0, 0), (100, 100), (30, 70), (12.5, 99.9)):
        px, py = plain.to_pixel(wx, wy)
        assert both.to_pixel(wx, wy) == (100 - px, 50 - py)


def test_out_of_bounds_returns_none_and_bounds_are_inclusive():
    mapper = CoordinateMapper(0, 100, 0, 100, 11, 11)
    assert mapper.to_pixel(-0.001, 50) is None
    assert mapper.to_pixel(100.001, 50) is None
    assert mapper.to_pixel(50, -1) is None
    assert mapper.to_pixel(50, 100.5) is None
    assert mapper.to_pixel(0, 100) == (0, 10)
    assert mapper.to_pixel(100, 0) == (10, 0)
    assert mapper.in_bounds(100, 100)
    assert not mapper.in_bounds(100.01, 100)


def test_non_square_scaling():
    mapper = CoordinateMapper(0, 200, 0, 50, 201, 101)
    assert mapper.pixels_per_unit_x == pytest.approx(1.0)
    assert mapper.pixels_per_unit_y == pytest.approx(2.0)
    assert mapper.to_pixel(100, 25) == (100, 50)


def test_offset_world_origin():
    mapper = CoordinateMapper(15250, 15680, 12340, 12720, 431, 381)
    assert mapper.to_pixel(15250, 12340) == (0, 0)
    assert mapper.to_pixel(15548.941, 12383.304) == (299, 43)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        CoordinateMapper(10, 10, 0, 1, 2, 2)
    with pytest.raises(ValueError):
        CoordinateMapper(0, 1, 5, 4, 2, 2)
    with pytest.raises(ValueError):
        CoordinateMapper(0, 1, 0, 1, 0, 2)


def test_single_pixel_canvas():
    mapper = CoordinateMapper(0, 1, 0, 1, 1, 1)
    assert mapper.to_pixel(0.7, 0.2) == (0, 0)
