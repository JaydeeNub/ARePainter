from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from treemasks.renderer import MaskCanvas, Renderer, disk_stencil


@pytest.mark.parametrize("radius, pixels", [(0, 1), (1, 5), (2, 13), (3, 29), (7, 149)])
def test_disk_stencil_pixel_counts(radius, pixels):
    stencil = disk_stencil(radius)
    assert stencil.shape == (2 * radius + 1, 2 * radius + 1)
    assert stencil.dtype == bool
    assert int(stencil.sum()) == pixels
    assert stencil[radius, radius]
    # symmetric in both axes
    assert np.array_equal(stencil, stencil[::-1])
    assert np.array_equal(stencil, stencil[:, ::-1])


def test_disk_stencil_rejects_negative():
    with pytest.raises(ValueError):
        disk_stencil(-1)


def test_draw_disk_center_and_value():
    canvas = MaskCanvas(21, 21)
    assert canvas.draw_disk(10, 10, 3)
    arr = canvas.array
    assert arr.dtype == np.uint8
    assert arr.max() == 255
    assert canvas.nonzero_pixels() == 29
    assert arr[10, 10] == 255 and arr[10, 13] == 255 and arr[10, 14] == 0
    assert arr[7, 10] == 255 and arr[6, 10] == 0


def test_radius_zero_draws_nothing():
    canvas = MaskCanvas(5, 5)
    assert not canvas.draw_disk(2, 2, 0)
    assert canvas.nonzero_pixels() == 0


def test_overlapping_markers_saturate():
    canvas = MaskCanvas(30, 30, value=200)
    canvas.draw_disk(10, 10, 4)
    canvas.draw_disk(12, 10, 4)
    canvas.draw_disk(12, 10, 4)
    arr = canvas.array
    assert arr.max() == 200  # no additive overflow / wrap-around
    expected = np.zeros((30, 30), dtype=bool)
    yy, xx = np.mgrid[0:30, 0:30]
    for cx in (10, 12):
        expected |= (xx - cx) ** 2 + (yy - 10) ** 2 <= 16
    assert np.array_equal(arr > 0, expected)


def test_clipping_at_edges_and_corners():
    canvas = MaskCanvas(10, 8)
    assert canvas.draw_disk(0, 0, 2)          # top-left corner: quadrant only
    assert canvas.draw_disk(9, 7, 3)          # bottom-right corner
    assert canvas.draw_disk(-1, 4, 2)         # centre off-canvas but disk overlaps
    assert not canvas.draw_disk(-5, 4, 2)     # fully outside
    assert not canvas.draw_disk(4, 100, 3)
    arr = canvas.array
    assert arr[0, 0] == 255 and arr[0, 2] == 255 and arr[2, 0] == 255 and arr[1, 1] == 255 and arr[2, 2] == 0
    assert arr[7, 9] == 255 and arr[4, 9] == 255 and arr[7, 6] == 255
    assert arr[4, 0] == 255 and arr[4, 1] == 255 and arr[4, 2] == 0
    assert arr[3, 0] == 255 and arr[2, 1] == 0


def test_canvas_validation():
    with pytest.raises(ValueError):
        MaskCanvas(0, 5)
    with pytest.raises(ValueError):
        MaskCanvas(5, 5, value=0)
    with pytest.raises(ValueError):
        MaskCanvas(5, 5, value=256)


def test_png_round_trip(tmp_path):
    canvas = MaskCanvas(40, 25, value=255)
    canvas.draw_disk(20, 12, 5)
    canvas.draw_disk(2, 2, 1)
    path = canvas.save(tmp_path / "nested" / "mask.png", compress_level=1)
    assert path.is_file()
    with Image.open(path) as image:
        assert image.mode == "L"
        assert image.size == (40, 25)
        loaded = np.array(image)
    assert np.array_equal(loaded, canvas.array)


def test_renderer_size_mapping_and_size_zero_never_rendered(tmp_path):
    renderer = Renderer(50, 50, {0: 0, 1: 2, 2: 4, 3: 7}, ["coniferous", "deciduous"])
    assert renderer.radius_for(0) == 0
    assert renderer.radius_for(3) == 7
    assert renderer.radius_for(9) == 0
    assert not renderer.draw("coniferous", 25, 25, 0)
    assert not renderer.draw("coniferous", 25, 25, 9)
    assert renderer.draw("coniferous", 25, 25, 1)
    assert renderer.draw("deciduous", 25, 25, 3)
    assert renderer.canvases["coniferous"].nonzero_pixels() == 13
    assert renderer.canvases["deciduous"].nonzero_pixels() == 149
    assert renderer.nbytes == 2 * 50 * 50
    assert Renderer.estimated_bytes(31501, 31501, 2) == 2 * 31501 * 31501

    written = renderer.save_all(tmp_path, compress_level=1)
    assert set(written) == {"coniferous", "deciduous"}
    assert (tmp_path / "coniferous.png").is_file() and (tmp_path / "deciduous.png").is_file()
    renderer.close()
    assert renderer.canvases == {}


def test_renderer_rejects_bad_marker_table():
    with pytest.raises(ValueError):
        Renderer(5, 5, {0: 3, 1: 2}, ["c"])
    with pytest.raises(ValueError):
        Renderer(5, 5, {0: 0, 1: -1}, ["c"])
    with pytest.raises(ValueError):
        Renderer(5, 5, {0: 0, 1: 1}, [])


def test_renderer_clear_keeps_allocation():
    renderer = Renderer(41, 41, {0: 0, 1: 2, 2: 0, 3: 7}, ["c"])
    assert renderer.draw("c", 20, 20, 1)
    assert renderer.canvases["c"].nonzero_pixels() == int(disk_stencil(2).sum())
    assert not renderer.draw("c", 20, 20, 0)      # size 0 never drawn
    assert not renderer.draw("c", 20, 20, 2)      # configured radius 0 stays undrawn
    renderer.clear()
    assert renderer.canvases["c"].nonzero_pixels() == 0
    assert renderer.canvases["c"].array.shape == (41, 41)  # allocation kept
    assert renderer.draw("c", 20, 20, 3)
    assert renderer.canvases["c"].nonzero_pixels() == 149


def test_renderer_uses_marker_value():
    renderer = Renderer(9, 9, {0: 0, 1: 1}, ["c"], value=77)
    renderer.draw("c", 4, 4, 1)
    arr = renderer.canvases["c"].array
    assert arr.max() == 77 and arr[4, 4] == 77
