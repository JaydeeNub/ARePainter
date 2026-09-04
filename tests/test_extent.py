from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from treemasks.cli import main
from treemasks.config import ConfigError, apply_overrides, config_from_mapping
from treemasks.extent import (
    CanopyMask,
    ExtentMaskError,
    binary_close,
    binary_dilate,
    binary_erode,
    compute_extent,
    fill_small_holes,
    iter_tiles,
)
from treemasks.pipeline import run
from treemasks.renderer import MaskCanvas

from conftest import base_config_dict
from test_pipeline import load_png

# 0..100 world at 1 px/unit. Spruces left, birches right, one spruce far outside any canopy.
FOREST_LAYER = """\
$grp ForestGeneratorEntity {
 {
  coords 0 0 0
  {
   $grp Tree : "{A}Trees/t_picea_abies_3sw.et" {
    {
     coords 25 0 50
    }
    {
     coords 35 0 50
    }
    {
     coords 5 0 5
    }
   }
   $grp Tree : "{B}Trees/t_betula_pendula_3s_aut.et" {
    {
     coords 65 0 50
    }
    {
     coords 75 0 50
    }
   }
  }
 }
}
"""
XX = np.mgrid[0:101, 0:101][1]


def save_mask(path: Path, array: np.ndarray, mode: str = "L") -> Path:
    image = Image.fromarray(array.astype(np.uint8))
    if mode != "L":
        image = image.convert(mode)
    image.save(path)
    return path


def synthetic_forest(height: int = 101, width: int = 101) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (canopy_with_holes, filled_forest, far_blob) boolean arrays.

    The forest is a block with a saw-tooth (jagged) upper edge and three small holes; the far
    blob is canopy with no trees anywhere near it.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    top = 30 + (xx % 5)
    filled = (xx >= 15) & (xx <= 85) & (yy >= top) & (yy <= 70)
    canopy = filled.copy()
    canopy[49:52, 49:52] = False   # 3x3 hole on the type boundary
    canopy[44:46, 29:31] = False   # 2x2 hole in the spruce half
    canopy[60:65, 70:72] = False   # 5x2 hole in the birch half
    far = (xx >= 90) & (yy >= 90)
    return canopy | far, filled, far


def markers_for_forest() -> dict[str, np.ndarray]:
    """Marker canvases matching FOREST_LAYER at 1 px/unit (radius 3 disks)."""
    con = MaskCanvas(101, 101)
    for x in (25, 35):
        con.draw_disk(x, 50, 3)
    con.draw_disk(5, 5, 3)
    dec = MaskCanvas(101, 101)
    for x in (65, 75):
        dec.draw_disk(x, 50, 3)
    return {"coniferous": con.array, "deciduous": dec.array}


@pytest.fixture
def layer_file(tmp_path):
    path = tmp_path / "forest.layer"
    path.write_text(FOREST_LAYER, encoding="utf-8")
    return path


@pytest.fixture
def canopy_file(tmp_path):
    canopy, _, _ = synthetic_forest()
    return save_mask(tmp_path / "canopy.png", np.where(canopy, 255, 0))


def make_config(tmp_path, mask: Path | None, **extent_overrides):
    data = base_config_dict()
    data["output"]["directory"] = str(tmp_path / "out")
    data["extent"] = {"mask": str(mask) if mask else None, **extent_overrides}
    return config_from_mapping(data)


# ----- CanopyMask -------------------------------------------------------------------

def test_open_reads_header_and_checks_size(tmp_path):
    path = save_mask(tmp_path / "canopy.png", np.zeros((20, 30), np.uint8))
    canopy = CanopyMask(path)
    assert canopy.open() == (30, 20)
    assert canopy.mode == "L"
    canopy.check_size(30, 20)
    with pytest.raises(ExtentMaskError, match="30x20"):
        canopy.check_size(31, 20)
    with pytest.raises(FileNotFoundError):
        CanopyMask(tmp_path / "missing.png").open()
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ExtentMaskError):
        CanopyMask(bad).open()
    with pytest.raises(ExtentMaskError):
        CanopyMask(path, threshold=0)


def test_huge_masks_bypass_pillows_bomb_guard(tmp_path, monkeypatch):
    array = np.zeros((6, 8), np.uint8)
    array[2, 3] = 255
    path = save_mask(tmp_path / "big.png", array)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)  # pretend the mask is far above the limit
    with pytest.raises(Image.DecompressionBombError):
        Image.open(path)
    canopy = CanopyMask(path)
    assert canopy.open() == (8, 6)
    assert Image.MAX_IMAGE_PIXELS == 10  # restored after the header read
    loaded = canopy.load()
    assert loaded[2, 3] == 255 and canopy.canopy_pixels == 1
    assert Image.MAX_IMAGE_PIXELS == 10  # restored after decoding too


def test_load_threshold_and_canopy_pixels(tmp_path):
    array = np.array([[0, 100, 127, 128], [255, 1, 0, 200]], np.uint8)
    path = save_mask(tmp_path / "canopy.png", array)
    canopy = CanopyMask(path, threshold=128)
    loaded = canopy.load()
    assert np.array_equal(loaded, array)
    assert canopy.canopy_pixels == 3
    assert canopy.load() is loaded  # decoded once
    low = CanopyMask(path, threshold=1)
    assert low.canopy_pixels is None  # counted on load only
    low.load()
    assert low.canopy_pixels == 6
    low.close()
    assert low.load() is not loaded


def test_load_converts_rgb_masks(tmp_path):
    array = np.zeros((4, 5), np.uint8)
    array[1, 2] = 255
    path = save_mask(tmp_path / "rgb.png", array, mode="RGB")
    canopy = CanopyMask(path)
    assert canopy.open() == (5, 4) and canopy.mode == "RGB"
    loaded = canopy.load()
    assert loaded.shape == (4, 5) and loaded[1, 2] == 255 and loaded.sum() == 255


def test_load_applies_flips(tmp_path):
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)
    path = save_mask(tmp_path / "m.png", array)
    assert np.array_equal(CanopyMask(path, flip_y=True).load(), array[::-1])
    assert np.array_equal(CanopyMask(path, flip_x=True).load(), array[:, ::-1])
    assert np.array_equal(CanopyMask(path, flip_x=True, flip_y=True).load(), array[::-1, ::-1])


# ----- morphology helpers -----------------------------------------------------------

def test_dilate_erode_and_close():
    mask = np.zeros((15, 15), bool)
    mask[7, 7] = True
    dilated = binary_dilate(mask, 2)
    assert int(dilated.sum()) == 13 and dilated[7, 9] and not dilated[7, 10]
    assert np.array_equal(binary_erode(dilated, 2), mask)
    # empty / full inputs are returned unchanged instead of producing garbage distances
    empty = np.zeros((5, 5), bool)
    assert binary_dilate(empty, 3) is empty
    full = np.ones((5, 5), bool)
    assert binary_erode(full, 3) is full
    assert binary_close(mask, 0) is mask

    # closing bridges a 3 px gap between two bars with radius 2 but not with radius 1
    bars = np.zeros((9, 21), bool)
    bars[2:7, 2:9] = True
    bars[2:7, 12:19] = True
    closed = binary_close(bars, 2)
    assert closed[4, 9:12].all()
    assert not binary_close(bars, 1)[4, 10]
    # closing never removes existing pixels
    assert (closed & bars).sum() == bars.sum()


def test_fill_small_holes_respects_size_and_border():
    mask = np.ones((30, 40), bool)
    mask[5:8, 5:8] = False        # 3x3 enclosed hole -> filled
    mask[10:22, 20:32] = False    # 12x12 enclosed hole -> too big for max_hole 8
    mask[0:3, 30:33] = False      # touches the border -> never filled
    mask[25:27, 10:11] = False    # 2x1 hole -> filled
    filled = fill_small_holes(mask, 8)
    assert filled[5:8, 5:8].all() and filled[25:27, 10:11].all()
    assert not filled[10:22, 20:32].any()
    assert not filled[0:3, 30:33].any()
    assert fill_small_holes(mask, 0) is mask
    assert fill_small_holes(mask, 12).sum() == filled.sum() + 144
    full = np.ones((4, 4), bool)
    assert fill_small_holes(full, 5) is full


def test_iter_tiles_cover_canvas_with_clipped_halos():
    tiles = list(iter_tiles(50, 70, 32, 5))
    cores = [core for core, _ in tiles]
    assert cores == [(0, 32, 0, 32), (0, 32, 32, 64), (0, 32, 64, 70), (32, 50, 0, 32), (32, 50, 32, 64), (32, 50, 64, 70)]
    core, window = tiles[4]
    assert window == (27, 50, 27, 69)
    core, window = tiles[0]
    assert window == (0, 37, 0, 37)
    covered = np.zeros((50, 70), int)
    for y0, y1, x0, x1 in cores:
        covered[y0:y1, x0:x1] += 1
    assert (covered == 1).all()
    with pytest.raises(ValueError):
        list(iter_tiles(5, 5, 0, 1))


# ----- compute_extent ---------------------------------------------------------------

def test_area_mode_paints_filled_forest_split_by_nearest_tree(tmp_path):
    canopy_arr, filled, far = synthetic_forest()
    path = save_mask(tmp_path / "canopy.png", np.where(canopy_arr, 255, 0))
    canvases = markers_for_forest()
    stats = compute_extent(CanopyMask(path), canvases, mode="area", close_radius=0, max_hole=64, max_distance=30)

    con, dec = canvases["coniferous"], canvases["deciduous"]
    expected_con = filled & (XX <= 50)   # ties on x == 50 go to the first category
    expected_dec = filled & (XX > 50)
    assert np.array_equal(con > 0, expected_con)
    assert np.array_equal(dec > 0, expected_dec)
    assert con.max() == 255 and dec.max() == 255
    # holes painted, saw-tooth edge kept exactly, far blob and the lone marker at (5, 5) erased
    assert con[44:46, 29:31].all() and con[49:52, 49:51].all() and dec[60:65, 70:72].all()
    assert not ((con > 0) | (dec > 0))[far].any()
    assert con[5, 5] == 0
    assert not ((con > 0) | (dec > 0))[~filled].any()

    assert stats.tiles_total == 1 and stats.tiles_processed == 1
    assert stats.forest_pixels == int(filled.sum()) + int(far.sum())
    assert stats.unassigned_pixels == int(far.sum())
    assert stats.painted == {"coniferous": int(expected_con.sum()), "deciduous": int(expected_dec.sum())}


def test_gaps_mode_paints_only_holes_inside_the_forest(tmp_path):
    canopy_arr, filled, far = synthetic_forest()
    path = save_mask(tmp_path / "canopy.png", np.where(canopy_arr, 255, 0))
    canvases = markers_for_forest()
    stats = compute_extent(CanopyMask(path), canvases, mode="gaps", close_radius=0, max_hole=64, max_distance=30)
    con, dec = canvases["coniferous"], canvases["deciduous"]
    holes = filled & ~canopy_arr
    assert np.array_equal(con > 0, holes & (XX <= 50))
    assert np.array_equal(dec > 0, holes & (XX > 50))
    assert int((con > 0).sum()) == 4 + 6   # 2x2 hole + left 2 columns of the 3x3 hole
    assert int((dec > 0).sum()) == 10 + 3  # 5x2 hole + right column of the 3x3 hole
    assert not ((con > 0) | (dec > 0))[canopy_arr].any()   # never on a crown
    assert not ((con > 0) | (dec > 0))[~filled].any()      # never outside the forest
    assert stats.painted == {"coniferous": 10, "deciduous": 13}
    assert stats.forest_pixels == int(filled.sum()) + int(far.sum())


def test_markers_are_the_canopy_when_no_mask_is_given():
    canvases = markers_for_forest()
    crowns = (canvases["coniferous"] > 0) | (canvases["deciduous"] > 0)
    # two spruce disks 10 px apart leave a 3 px neck; closing with radius 4 bridges it
    # (radius 3 would pinch the thin neck off again during the erosion half)
    stats = compute_extent(None, canvases, mode="gaps", close_radius=4, max_hole=64, max_distance=30)
    con, dec = canvases["coniferous"], canvases["deciduous"]
    painted = (con > 0) | (dec > 0)
    assert painted.any()
    assert not painted[crowns].any()                      # gaps only, never on a crown
    assert painted[50, 30]                                # the notch between the two spruces
    assert not painted[50, 50]                            # between the two forests: 24 px gap, not closed
    assert not (dec > 0)[:, :50].any() and not (con > 0)[:, 51:].any()
    assert con[5, 5] == 0                                 # the lone spruce has no gaps around it
    assert stats.painted["coniferous"] > 0 and stats.painted["deciduous"] > 0

    # area mode with the same input gives the closed marker blobs
    area = markers_for_forest()
    compute_extent(None, area, mode="area", close_radius=4, max_hole=64, max_distance=30)
    assert (area["coniferous"] > 0)[crowns & (XX <= 50)].all()
    assert (area["coniferous"] > 0)[50, 30]
    assert int(((area["coniferous"] > 0) | (area["deciduous"] > 0)).sum()) == painted.sum() + crowns.sum()


def test_compute_extent_tiling_is_seamless(tmp_path):
    canopy_arr, _, _ = synthetic_forest()
    path = save_mask(tmp_path / "canopy.png", np.where(canopy_arr, 255, 0))
    for mode in ("gaps", "area"):
        single = markers_for_forest()
        tiled = markers_for_forest()
        stats_single = compute_extent(CanopyMask(path), single, mode=mode, close_radius=1, max_hole=6, max_distance=12, tile=1024)
        stats_tiled = compute_extent(CanopyMask(path), tiled, mode=mode, close_radius=1, max_hole=6, max_distance=12, tile=16)
        for category in single:
            assert np.array_equal(single[category], tiled[category]), (mode, category)
        assert stats_tiled.painted == stats_single.painted
        # forest/unassigned counters only cover tiles that hold trees, so they may shrink with
        # small tiles, but forest - unassigned must stay consistent with the area painted
        assert stats_tiled.forest_pixels - stats_tiled.unassigned_pixels == stats_single.forest_pixels - stats_single.unassigned_pixels
        assert stats_tiled.tiles_total == 49 and stats_tiled.tiles_processed < 49
    marker_single = markers_for_forest()
    marker_tiled = markers_for_forest()
    compute_extent(None, marker_single, close_radius=3, max_hole=20, max_distance=20, tile=1024)
    compute_extent(None, marker_tiled, close_radius=3, max_hole=20, max_distance=20, tile=16)
    for category in marker_single:
        assert np.array_equal(marker_single[category], marker_tiled[category])


def test_compute_extent_close_radius_bridges_channels(tmp_path):
    # a canopy split by a 3 px channel: with closing the channel counts as inside (a gap), without it not
    canopy_arr = np.zeros((101, 101), bool)
    canopy_arr[30:70, 15:49] = True
    canopy_arr[30:70, 52:86] = True
    path = save_mask(tmp_path / "canopy.png", np.where(canopy_arr, 255, 0))
    plain = markers_for_forest()
    compute_extent(CanopyMask(path), plain, close_radius=0, max_hole=0, max_distance=40)
    assert not ((plain["coniferous"] > 0) | (plain["deciduous"] > 0)).any()
    closed = markers_for_forest()
    compute_extent(CanopyMask(path), closed, close_radius=2, max_hole=0, max_distance=40)
    channel = ((closed["coniferous"] > 0) | (closed["deciduous"] > 0))
    assert channel[35:65, 49:52].all()
    assert not channel[canopy_arr].any()


def test_compute_extent_validates_inputs(tmp_path):
    path = save_mask(tmp_path / "canopy.png", np.zeros((10, 10), np.uint8))
    with pytest.raises(ExtentMaskError, match="differ"):
        compute_extent(CanopyMask(path), {"c": np.zeros((5, 5), np.uint8)})
    with pytest.raises(ExtentMaskError, match="differ"):
        compute_extent(None, {"c": np.zeros((5, 5), np.uint8), "d": np.zeros((6, 5), np.uint8)})
    with pytest.raises(ExtentMaskError, match="no category"):
        compute_extent(CanopyMask(path), {})
    with pytest.raises(ExtentMaskError, match="mode"):
        compute_extent(None, {"c": np.zeros((5, 5), np.uint8)}, mode="holes")


# ----- pipeline integration ---------------------------------------------------------

def test_pipeline_gaps_outputs_and_diagnostics(tmp_path, layer_file, canopy_file):
    canopy_arr, filled, far = synthetic_forest()
    config = make_config(tmp_path, canopy_file, close_radius=0, max_hole=64, max_distance=30)
    result = run(config, [layer_file])
    diag = result.diagnostics

    assert set(result.outputs) == {"coniferous", "deciduous"}
    assert result.extent_outputs["coniferous"].name == "coniferous_gaps.png"
    plain_con = load_png(result.outputs["coniferous"])
    gaps_con = load_png(result.extent_outputs["coniferous"])
    gaps_dec = load_png(result.extent_outputs["deciduous"])
    holes = filled & ~canopy_arr
    assert np.array_equal(gaps_con > 0, holes & (XX <= 50))
    assert np.array_equal(gaps_dec > 0, holes & (XX > 50))
    # plain masks keep their circles, including the lone spruce outside the canopy
    assert plain_con[5, 5] == 255 and int(np.count_nonzero(plain_con)) == 3 * 29

    assert diag.extent_enabled and diag.extent_mode == "gaps" and diag.extent_mask == str(canopy_file)
    assert (diag.extent_close_radius, diag.extent_max_hole, diag.extent_max_distance) == (0, 64, 30.0)
    assert diag.canopy_pixels == int(canopy_arr.sum())
    assert diag.forest_pixels == int(filled.sum()) + int(far.sum())
    assert diag.unassigned_pixels == int(far.sum())
    assert (diag.tiles_processed, diag.tiles_total) == (1, 1)
    assert diag.extent_markers == {"coniferous": 3 * 29, "deciduous": 2 * 29}
    assert diag.extent_painted == {"coniferous": 10, "deciduous": 13}
    assert diag.passes == 1
    summary = diag.format_summary(categories=["coniferous", "deciduous"], sizes=[0, 1, 2, 3])
    assert "Forest extent (gaps" in summary and "coniferous_gaps.png" in summary
    assert "raise extent.max_distance" in summary  # the far blob is unassigned
    report = diag.to_dict()
    assert report["extent"]["mode"] == "gaps" and report["extent"]["painted_pixels"] == diag.extent_painted
    json.dumps(report)


def test_pipeline_area_mode_and_marker_canopy(tmp_path, layer_file, canopy_file):
    area = run(make_config(tmp_path / "a", canopy_file, mode="area", close_radius=0, max_distance=30), [layer_file])
    _, filled, _ = synthetic_forest()
    assert area.extent_outputs["deciduous"].name == "deciduous_area.png"
    assert np.array_equal(load_png(area.extent_outputs["deciduous"]) > 0, filled & (XX > 50))
    assert "solid forest area" in area.diagnostics.format_summary()

    markers = run(make_config(tmp_path / "m", None, enabled=True, close_radius=4, suffix="_floor"), [layer_file])
    diag = markers.diagnostics
    assert diag.extent_enabled and diag.extent_mask is None and diag.canopy_pixels is None
    assert markers.extent_outputs["coniferous"].name == "coniferous_floor.png"
    gaps = load_png(markers.extent_outputs["coniferous"])
    plain = load_png(markers.outputs["coniferous"])
    assert gaps[50, 30] == 255 and not (gaps[plain > 0]).any()
    assert "rendered tree markers" in diag.format_summary()


def test_low_memory_is_ignored_with_extent(tmp_path, layer_file, canopy_file, caplog):
    normal = run(make_config(tmp_path / "n", canopy_file), [layer_file])
    with caplog.at_level(logging.WARNING, logger="treemasks.pipeline"):
        low = run(make_config(tmp_path / "l", canopy_file), [layer_file], low_memory=True)
    assert "low-memory ignored" in caplog.text
    assert low.diagnostics.passes == 1
    assert low.diagnostics.canvas_bytes == 2 * 101 * 101
    for category in ("coniferous", "deciduous"):
        assert np.array_equal(load_png(normal.extent_outputs[category]), load_png(low.extent_outputs[category]))
    assert low.diagnostics.extent_painted == normal.diagnostics.extent_painted


def test_render_orientation_flips_mask_to_match_output(tmp_path, layer_file):
    canopy_arr, filled, _ = synthetic_forest()
    # mask authored in the un-flipped render orientation; outputs are rendered with both flips
    mask_path = save_mask(tmp_path / "canopy_render.png", np.where(canopy_arr, 255, 0))
    data = base_config_dict()
    data["output"]["directory"] = str(tmp_path / "out")
    data["coordinate_system"] = {"flip_x": True, "flip_y": True}
    data["extent"] = {"mask": str(mask_path), "orientation": "render", "mode": "area", "close_radius": 0, "max_distance": 30}
    result = run(config_from_mapping(data), [layer_file])
    ext_con = load_png(result.extent_outputs["coniferous"])
    ext_dec = load_png(result.extent_outputs["deciduous"])
    flipped_filled = filled[::-1, ::-1]
    assert np.array_equal(ext_con > 0, flipped_filled & (XX >= 50))   # spruces are now on the right
    assert np.array_equal(ext_dec > 0, flipped_filled & (XX < 50))

    # the same mask declared as already output-oriented no longer lines up with the trees:
    # the lone spruce, now at (95, 95), claims the far blob and the saw-tooth edge sits on the wrong side
    data["extent"]["orientation"] = "output"
    result2 = run(config_from_mapping(data), [layer_file])
    ext2_con = load_png(result2.extent_outputs["coniferous"])
    assert ext2_con[95, 95] == 255
    assert not np.array_equal(ext2_con > 0, flipped_filled & (XX >= 50))


def test_wrong_size_or_missing_mask(tmp_path, layer_file):
    small = save_mask(tmp_path / "small.png", np.zeros((50, 50), np.uint8))
    with pytest.raises(ExtentMaskError, match="50x50"):
        run(make_config(tmp_path, small), [layer_file])
    with pytest.raises(FileNotFoundError):
        run(make_config(tmp_path, tmp_path / "nope.png"), [layer_file])
    assert not (tmp_path / "out").exists()  # failed before any canvas was written


def test_extent_disabled_by_default(tmp_path, layer_file):
    result = run(make_config(tmp_path, None), [layer_file])
    assert result.extent_outputs == {}
    assert not result.diagnostics.extent_enabled
    assert "Forest extent" not in result.diagnostics.format_summary()
    assert result.diagnostics.to_dict()["extent"]["enabled"] is False


# ----- configuration ----------------------------------------------------------------

@pytest.mark.parametrize(
    "extent, message",
    [
        ({"mask": "m.png", "threshold": 0}, "threshold"),
        ({"mask": "m.png", "threshold": 256}, "threshold"),
        ({"enabled": True, "mode": "holes"}, "mode"),
        ({"mask": "m.png", "close_radius": -1}, "close_radius"),
        ({"mask": "m.png", "max_hole": -1}, "max_hole"),
        ({"mask": "m.png", "max_distance": -0.5}, "max_distance"),
        ({"mask": "m.png", "tile": 10}, "tile"),
        ({"mask": "m.png", "suffix": "a/b"}, "suffix"),
        ({"mask": "m.png", "orientation": "sideways"}, "orientation"),
        ("not a mapping", "mapping"),
    ],
)
def test_invalid_extent_settings(extent, message):
    data = base_config_dict(extent=extent)
    with pytest.raises(ConfigError, match=message):
        config_from_mapping(data)


def test_extent_settings_and_overrides():
    config = config_from_mapping(base_config_dict())
    ext = config.extent
    assert not ext.active and not ext.enabled and ext.mask is None
    assert ext.mode == "gaps" and ext.name_pattern == "{category}_gaps.png"
    assert (ext.close_radius, ext.max_hole, ext.max_distance, ext.tile) == (3, 64, 50.0, 1024)

    by_flag = apply_overrides(config, {"extent_enabled": True, "extent_mode": "AREA"})
    assert by_flag.extent.active and by_flag.extent.mask is None
    assert by_flag.extent.mode == "area" and by_flag.extent.name_pattern == "{category}_area.png"

    by_mask = apply_overrides(
        config,
        {
            "extent_mask": "canopy.png", "extent_threshold": 200, "extent_close_radius": 4,
            "extent_max_hole": 10, "extent_max_distance": 7.5, "extent_orientation": "RENDER",
        },
    )
    assert by_mask.extent.active and by_mask.extent.mask == "canopy.png"
    assert (by_mask.extent.threshold, by_mask.extent.close_radius, by_mask.extent.max_hole) == (200, 4, 10)
    assert by_mask.extent.max_distance == 7.5 and by_mask.extent.orientation == "render"

    disabled = apply_overrides(by_mask, {"extent_disabled": True})
    assert not disabled.extent.active and disabled.extent.threshold == 200
    blank = apply_overrides(by_mask, {"extent_mask": "   "})
    assert not blank.extent.active

    explicit = config_from_mapping(base_config_dict(extent={"enabled": True, "suffix": "_floor"}))
    assert explicit.extent.active and explicit.extent.name_pattern == "{category}_floor.png"
    empty_suffix = config_from_mapping(base_config_dict(extent={"enabled": True, "suffix": ""}))
    assert empty_suffix.extent.name_pattern == "{category}_gaps.png"
    with pytest.raises(ConfigError):
        apply_overrides(config, {"extent_orientation": "diagonal"})
    with pytest.raises(ConfigError):
        apply_overrides(config, {"extent_mode": "solid"})


def test_cli_extent_flags(tmp_path, layer_file, canopy_file, capsys):
    canopy_arr, filled, _ = synthetic_forest()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(base_config_dict()), encoding="utf-8")
    out_dir = tmp_path / "cli_out"
    report = tmp_path / "report.json"
    code = main(
        [
            "--input", str(layer_file), "--config", str(cfg), "--output-dir", str(out_dir),
            "--extent-mask", str(canopy_file), "--extent-close-radius", "0", "--extent-max-hole", "2",
            "--extent-max-distance", "30", "--json-report", str(report), "--quiet",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Forest extent (gaps" in out and "holes up to 2 px" in out and "reach 30 px" in out
    gaps_con = load_png(out_dir / "coniferous_gaps.png")
    gaps_dec = load_png(out_dir / "deciduous_gaps.png")
    painted = (gaps_con > 0) | (gaps_dec > 0)
    assert painted[44:46, 29:31].all()       # 2x2 hole painted (fits max_hole 2)
    assert not painted[49:52, 49:52].any()   # 3x3 hole left open: outside the forest area
    assert not painted[canopy_arr].any() and not painted[~filled].any()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["config"]["extent"]["max_hole"] == 2
    assert data["diagnostics"]["extent"]["outputs"]["coniferous"].endswith("coniferous_gaps.png")

    # --extent alone uses the markers as canopy; --extent-mode area names the files _area
    assert main(["--input", str(layer_file), "--config", str(cfg), "--output-dir", str(tmp_path / "o1"),
                 "--extent", "--extent-mode", "area", "--extent-close-radius", "4", "--quiet"]) == 0
    area_con = load_png(tmp_path / "o1" / "coniferous_area.png")
    assert area_con[50, 30] == 255 and area_con[5, 5] == 255
    assert "rendered tree markers" in capsys.readouterr().out

    # --no-extent wins over a configured mask; a wrong-size mask is a clean error
    cfg_with_mask = tmp_path / "cfg_mask.json"
    cfg_with_mask.write_text(json.dumps(base_config_dict(extent={"mask": str(canopy_file)})), encoding="utf-8")
    assert main(["--input", str(layer_file), "--config", str(cfg_with_mask), "--output-dir", str(tmp_path / "o2"),
                 "--no-extent", "--quiet"]) == 0
    assert not (tmp_path / "o2" / "coniferous_gaps.png").exists()
    small = save_mask(tmp_path / "small.png", np.zeros((10, 10), np.uint8))
    assert main(["--input", str(layer_file), "--config", str(cfg), "--output-dir", str(tmp_path / "o3"),
                 "--extent-mask", str(small), "--quiet"]) == 1
