from __future__ import annotations

import json

import numpy as np
import pytest

from treemasks.cli import main
from treemasks.config import ConfigError, apply_overrides, config_from_mapping
from treemasks.extent import compute_extent
from treemasks.pipeline import run
from treemasks.renderer import MaskCanvas, PackedMask, Renderer

from conftest import base_config_dict
from test_extent import FOREST_LAYER, XX, markers_for_forest, save_mask, synthetic_forest
from test_pipeline import END_TO_END_LAYER, load_png


@pytest.fixture
def layer_file(tmp_path):
    path = tmp_path / "forest.layer"
    path.write_text(END_TO_END_LAYER, encoding="utf-8")
    return path


def make_config(tmp_path, enabled=True, **sections):
    data = base_config_dict(**sections)
    data["output"]["directory"] = str(tmp_path / "out")
    data["combined_inverse"] = {"enabled": enabled}
    return config_from_mapping(data)


# ----- canvas operations --------------------------------------------------------------

def test_merge_is_pixelwise_or_and_chunked():
    a = MaskCanvas(9, 7, value=200)
    a.draw_disk(2, 2, 1)
    b = np.zeros((7, 9), np.uint8)
    b[5, 6] = 255
    b[2, 2] = 40  # already painted in a: max keeps 200
    a.merge(b, chunk_rows=2)
    assert a.array[5, 6] == 255 and a.array[2, 2] == 200
    assert a.nonzero_pixels() == 5 + 1
    with pytest.raises(ValueError):
        a.merge(np.zeros((3, 3), np.uint8))


def test_invert_is_hard_boolean():
    canvas = MaskCanvas(10, 6, value=255)
    canvas.array[1, 1] = 255
    canvas.array[2, 3] = 7      # any non-zero counts as painted
    canvas.array[5, 9] = 128
    painted, inverse = canvas.invert(chunk_rows=4)
    assert (painted, inverse) == (3, 60 - 3)
    arr = canvas.array
    assert set(np.unique(arr).tolist()) == {0, 255}
    assert arr[1, 1] == 0 and arr[2, 3] == 0 and arr[5, 9] == 0
    assert int(np.count_nonzero(arr)) == 57
    # inverting again restores the boolean pattern with the canvas value
    canvas.invert()
    assert arr[1, 1] == 255 and arr[2, 3] == 255 and int(np.count_nonzero(arr)) == 3


def test_packed_mask_from_arrays_and_merge_into():
    rng = np.random.default_rng(1)
    a = (rng.random((23, 17)) < 0.05).astype(np.uint8) * 255
    b = (rng.random((23, 17)) < 0.05).astype(np.uint8) * 90
    packed = PackedMask.from_arrays([a, b], chunk_rows=5)
    expected = (a > 0) | (b > 0)
    assert packed.count() == int(expected.sum())
    assert 0 < packed.nbytes < expected.size  # packed, and empty chunks are skipped
    target = np.zeros((23, 17), np.uint8)
    target[0, 0] = 7  # existing paint is always kept, whether or not the mask covers it
    expected_target = np.where(expected, 255, 0).astype(np.uint8)
    expected_target[0, 0] = 7
    packed.merge_into(target, 255)
    assert np.array_equal(target, expected_target)
    with pytest.raises(ValueError):
        packed.merge_into(np.zeros((5, 5), np.uint8), 255)
    with pytest.raises(ValueError):
        PackedMask.from_arrays([])
    with pytest.raises(ValueError):
        PackedMask.from_arrays([a, np.zeros((3, 3), np.uint8)])


def test_packed_mask_add_box_across_chunk_borders():
    packed = PackedMask(40, 30, chunk_rows=8)
    reference = np.zeros((40, 30), bool)
    rng = np.random.default_rng(2)
    boxes = [(0, 8, 0, 10), (5, 21, 4, 19), (20, 40, 10, 30), (36, 40, 0, 5), (2, 4, 25, 30)]
    for box in boxes:
        y0, y1, x0, x1 = box
        patch = rng.random((y1 - y0, x1 - x0)) < 0.5
        packed.add_box(box, patch)
        reference[y0:y1, x0:x1] |= patch
    assert packed.count() == int(reference.sum())
    target = np.zeros((40, 30), np.uint8)
    packed.merge_into(target, 200)
    assert np.array_equal(target > 0, reference)
    assert set(np.unique(target).tolist()) <= {0, 200}
    # an all-False patch and a bad shape
    packed.add_box((0, 2, 0, 2), np.zeros((2, 2), bool))
    assert packed.count() == int(reference.sum())
    with pytest.raises(ValueError):
        packed.add_box((0, 2, 0, 2), np.zeros((3, 3), bool))


def test_renderer_combined_inverse_consumes_canvases_and_extras():
    renderer = Renderer(20, 15, {0: 0, 1: 2, 2: 4, 3: 7}, ["a", "b", "c"], value=100)
    renderer.draw("a", 3, 3, 1)
    renderer.draw("b", 10, 7, 2)
    renderer.draw("c", 10, 7, 1)  # fully inside b's marker
    extra = PackedMask(15, 20, chunk_rows=4)
    extra.add_box((12, 15, 16, 20), np.ones((3, 4), bool))
    expected_union = (renderer.canvases["a"].array > 0) | (renderer.canvases["b"].array > 0)
    expected_union[12:15, 16:20] = True
    canvas, painted, inverse = renderer.combined_inverse([extra], chunk_rows=4)
    assert canvas is renderer.canvases["a"]
    assert painted == int(expected_union.sum()) and inverse == 20 * 15 - painted
    assert np.array_equal(canvas.array == 0, expected_union)
    assert set(np.unique(canvas.array).tolist()) == {0, 100}


def test_compute_extent_reports_forest_area():
    canvases = markers_for_forest()
    area = PackedMask(101, 101)
    compute_extent(None, canvases, mode="gaps", close_radius=4, max_hole=64, max_distance=30, tile=16, area_out=area)
    gaps = (canvases["coniferous"] > 0) | (canvases["deciduous"] > 0)
    crowns_union = (markers_for_forest()["coniferous"] > 0) | (markers_for_forest()["deciduous"] > 0)
    dense = np.zeros((101, 101), np.uint8)
    area.merge_into(dense, 255)
    # the reachable forest area is exactly the crowns plus the gaps between them
    assert np.array_equal(dense > 0, gaps | crowns_union)
    assert area.count() == int((gaps | crowns_union).sum())


# ----- pipeline -----------------------------------------------------------------------

def test_pipeline_writes_inverted_union_of_plain_masks(tmp_path, layer_file):
    result = run(make_config(tmp_path), [layer_file])
    assert result.combined_inverse is not None
    assert result.combined_inverse.name == "combined_inverse.png"
    con = load_png(result.outputs["coniferous"])
    dec = load_png(result.outputs["deciduous"])
    combined = load_png(result.combined_inverse)
    assert combined.shape == con.shape
    assert np.array_equal(combined == 0, (con > 0) | (dec > 0))
    assert set(np.unique(combined).tolist()) == {0, 255}
    # the category masks on disk are the un-inverted originals
    assert int(np.count_nonzero(con)) == 2 * 29 and int(np.count_nonzero(dec)) == 13
    diag = result.diagnostics
    assert diag.combined_inverse_output == str(result.combined_inverse)
    assert diag.combined_sources == ["coniferous.png", "deciduous.png"]
    assert diag.combined_painted_pixels == 2 * 29 + 13
    assert diag.combined_inverse_pixels == 101 * 101 - (2 * 29 + 13)
    summary = diag.format_summary()
    assert "Combined inverse" in summary and "combined_inverse.png" in summary
    assert "union of           : coniferous.png, deciduous.png" in summary
    report = diag.to_dict()["combined_inverse"]
    assert report["enabled"] and report["painted_union_pixels"] == 2 * 29 + 13
    assert report["sources"] == ["coniferous.png", "deciduous.png"]
    json.dumps(report)


def test_combined_inverse_uses_marker_value(tmp_path, layer_file):
    config = make_config(tmp_path, rendering={"marker_sizes": {0: 0, 1: 1, 2: 2, 3: 3}, "marker_value": 90})
    result = run(config, [layer_file])
    combined = load_png(result.combined_inverse)
    assert set(np.unique(combined).tolist()) == {0, 90}
    con = load_png(result.outputs["coniferous"])
    assert np.array_equal(combined == 90, (con == 0) & (load_png(result.outputs["deciduous"]) == 0))


def test_combined_inverse_hides_whole_forest_with_marker_canopy(tmp_path):
    layer = tmp_path / "forest.layer"
    layer.write_text(FOREST_LAYER, encoding="utf-8")
    data = base_config_dict()
    data["output"]["directory"] = str(tmp_path / "out")
    data["extent"] = {"enabled": True, "close_radius": 4, "max_distance": 30}
    data["combined_inverse"] = {"enabled": True}
    result = run(config_from_mapping(data), [layer])
    con = load_png(result.outputs["coniferous"])
    dec = load_png(result.outputs["deciduous"])
    gaps_con = load_png(result.extent_outputs["coniferous"])
    gaps_dec = load_png(result.extent_outputs["deciduous"])
    combined = load_png(result.combined_inverse)
    everything = (con > 0) | (dec > 0) | (gaps_con > 0) | (gaps_dec > 0)
    assert np.array_equal(combined == 0, everything)
    # the forest (crowns and the gaps between them) is one black patch: no light pixel left inside
    assert combined[50, 22:39].max() == 0        # across both spruces and the gap between them
    assert combined[50, 62:79].max() == 0        # across both birches
    assert combined[50, 45:55].min() == 255      # the open ground between the two forests stays light
    assert combined[5, 5] == 0                   # the lone spruce circle is a mask too
    assert set(np.unique(combined).tolist()) == {0, 255}
    diag = result.diagnostics
    assert diag.combined_sources == ["coniferous.png", "deciduous.png", "coniferous_gaps.png", "deciduous_gaps.png", "forest area"]
    assert diag.combined_painted_pixels == int(everything.sum())
    assert "forest area" in diag.format_summary()


def test_combined_inverse_hides_texture_forest_area(tmp_path):
    layer = tmp_path / "forest.layer"
    layer.write_text(FOREST_LAYER, encoding="utf-8")
    canopy_arr, filled, far = synthetic_forest()
    mask = save_mask(tmp_path / "canopy.png", np.where(canopy_arr, 255, 0))
    data = base_config_dict()
    data["output"]["directory"] = str(tmp_path / "out")
    data["extent"] = {"mask": str(mask), "close_radius": 0, "max_distance": 30}
    data["combined_inverse"] = {"enabled": True, "filename": "floor_inverse.png"}
    result = run(config_from_mapping(data), [layer])
    assert result.combined_inverse.name == "floor_inverse.png"
    combined = load_png(result.combined_inverse)
    con = load_png(result.outputs["coniferous"])
    dec = load_png(result.outputs["deciduous"])
    # black = the whole forest area (crowns from the texture, gaps, and the circles), even though
    # the texture canopy itself is not an output mask; the far blob had no trees and stays light
    expected_black = filled | (con > 0) | (dec > 0)
    assert np.array_equal(combined == 0, expected_black)
    assert combined[far].min() == 255
    assert combined[5, 5] == 0
    assert result.diagnostics.combined_painted_pixels == int(expected_black.sum())


def test_low_memory_mode_matches(tmp_path, layer_file):
    normal = run(make_config(tmp_path / "n"), [layer_file])
    low = run(make_config(tmp_path / "l"), [layer_file], low_memory=True)
    assert np.array_equal(load_png(normal.combined_inverse), load_png(low.combined_inverse))
    assert low.diagnostics.passes == 2
    assert low.diagnostics.canvas_bytes == 2 * 101 * 101   # one category canvas + the union canvas
    assert low.diagnostics.combined_painted_pixels == normal.diagnostics.combined_painted_pixels
    assert low.diagnostics.combined_sources == normal.diagnostics.combined_sources
    for category in ("coniferous", "deciduous"):
        assert np.array_equal(load_png(normal.outputs[category]), load_png(low.outputs[category]))


def test_single_category(tmp_path, layer_file):
    config = make_config(tmp_path, trees={"coniferous": ["t_picea_abies_*"]})
    result = run(config, [layer_file])
    con = load_png(result.outputs["coniferous"])
    combined = load_png(result.combined_inverse)
    assert np.array_equal(combined == 0, con > 0)


def test_disabled_by_default(tmp_path, layer_file):
    data = base_config_dict()
    data["output"]["directory"] = str(tmp_path / "out")
    result = run(config_from_mapping(data), [layer_file])
    assert result.combined_inverse is None
    assert not (tmp_path / "out" / "combined_inverse.png").exists()
    assert result.diagnostics.combined_inverse_output is None
    assert "Combined inverse" not in result.diagnostics.format_summary()
    assert result.diagnostics.to_dict()["combined_inverse"]["enabled"] is False


# ----- configuration and CLI ----------------------------------------------------------

def test_config_validation_and_overrides():
    config = config_from_mapping(base_config_dict())
    assert not config.combined_inverse.enabled and config.combined_inverse.filename == "combined_inverse.png"
    enabled = apply_overrides(config, {"combined_inverse": True, "combined_inverse_file": "inv.png"})
    assert enabled.combined_inverse.enabled and enabled.combined_inverse.filename == "inv.png"
    disabled = apply_overrides(enabled, {"combined_inverse": False})
    assert not disabled.combined_inverse.enabled and disabled.combined_inverse.filename == "inv.png"
    for bad in ("", "a/b.png", "..", "sub\\x.png"):
        with pytest.raises(ConfigError, match="combined_inverse.filename"):
            config_from_mapping(base_config_dict(combined_inverse={"enabled": True, "filename": bad}))
    with pytest.raises(ConfigError, match="mapping"):
        config_from_mapping(base_config_dict(combined_inverse="yes"))


def test_cli_flags(tmp_path, layer_file, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(base_config_dict()), encoding="utf-8")
    out_dir = tmp_path / "cli_out"
    report = tmp_path / "report.json"
    assert main(["--input", str(layer_file), "--config", str(cfg), "--output-dir", str(out_dir),
                 "--combined-inverse", "--combined-inverse-file", "everything_inverted.png",
                 "--json-report", str(report), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "Combined inverse" in out and "everything_inverted.png" in out
    combined = load_png(out_dir / "everything_inverted.png")
    con = load_png(out_dir / "coniferous.png")
    dec = load_png(out_dir / "deciduous.png")
    assert np.array_equal(combined == 0, (con > 0) | (dec > 0))
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["config"]["combined_inverse"] == {"enabled": True, "filename": "everything_inverted.png"}
    assert data["diagnostics"]["combined_inverse"]["output"].endswith("everything_inverted.png")

    # --no-combined-inverse wins over a config that enables it
    cfg_on = tmp_path / "cfg_on.json"
    cfg_on.write_text(json.dumps(base_config_dict(combined_inverse={"enabled": True})), encoding="utf-8")
    assert main(["--input", str(layer_file), "--config", str(cfg_on), "--output-dir", str(tmp_path / "o2"),
                 "--no-combined-inverse", "--quiet"]) == 0
    assert not (tmp_path / "o2" / "combined_inverse.png").exists()
