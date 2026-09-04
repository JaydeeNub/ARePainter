from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from treemasks.cli import main
from treemasks.config import config_from_mapping
from treemasks.pipeline import run

from conftest import SYNTHETIC_LAYER, base_config_dict

# Layer with known world positions for a 0..100 x 0..100 world rendered at 1 px/unit.
END_TO_END_LAYER = """\
$grp ForestGeneratorEntity {
 {
  coords 10 0 20
  {
   $grp Tree : "{A}Trees/t_picea_abies_3sw.et" {
    {
     coords 10 0 10
    }
    {
     coords 30 0 10
    }
   }
   $grp Tree : "{B}Trees/t_betula_pendula_2s_aut.et" {
    {
     coords 40 0 40
    }
   }
   Tree : "{C}Trees/t_betula_pendula_0_aut.et" {
    coords 5 0 5
   }
   Tree : "{D}Trees/t_sorbus_aucuparia_1w.et" {
    coords 200 0 200
   }
   Tree : "{E}Trees/t_picea_abies_stump_01.et" {
    coords 1 0 1
   }
   Tree : "{F}Bushes/b_corylus_avellana_1_aut.et" {
    coords 2 0 2
   }
   $grp SCR_IndestructibleEnvironmentalEntity : "{G}Rocks/GraniteStone_01_V2.et" {
    {
     coords 3 0 3
    }
   }
  }
 }
}
"""


@pytest.fixture
def layer_file(tmp_path):
    path = tmp_path / "forest.layer"
    path.write_text(END_TO_END_LAYER, encoding="utf-8")
    return path


def make_config(tmp_path, **overrides):
    data = base_config_dict(**overrides)
    data["output"]["directory"] = str(tmp_path / "out")
    return config_from_mapping(data)


def load_png(path) -> np.ndarray:
    with Image.open(path) as image:
        assert image.mode == "L"
        return np.array(image)


def test_end_to_end_positions_and_counters(tmp_path, layer_file):
    config = make_config(tmp_path)
    result = run(config, [layer_file])
    diag = result.diagnostics

    assert set(result.outputs) == {"coniferous", "deciduous"}
    con = load_png(result.outputs["coniferous"])
    dec = load_png(result.outputs["deciduous"])
    assert con.shape == (101, 101) and dec.shape == (101, 101)

    # spruce size 3 -> radius 3 at world (20, 30) and (40, 30): pixel (x=20, y=30) / (x=40, y=30)
    assert con[30, 20] == 255 and con[30, 23] == 255 and con[30, 24] == 0
    assert con[30, 40] == 255 and con[27, 40] == 255 and con[26, 40] == 0
    assert con.sum() // 255 == 2 * 29
    # birch size 2 -> radius 2 at world (50, 60)
    assert dec[60, 50] == 255 and dec[60, 52] == 255 and dec[60, 53] == 0
    assert dec.sum() // 255 == 13
    # size-0 birch at (15, 25) not drawn; stump/bush/rock not drawn
    assert dec[25, 15] == 0 and dec[21, 11] == 0 and con[21, 11] == 0

    assert diag.files == [str(layer_file)]
    assert diag.entities_inspected == 9
    assert diag.tree_class_entities == 7
    assert diag.other_class_entities == 2  # generator + rock
    assert diag.trees_by_category_size["coniferous"] == {3: 2}
    assert diag.trees_by_category_size["deciduous"] == {2: 1, 0: 1, 1: 1}
    assert diag.rendered == {"coniferous": 2, "deciduous": 1}
    assert diag.skipped_size0 == {"deciduous": 1}
    assert diag.out_of_bounds == {"deciduous": 1}
    assert diag.excluded_assets == {"t_picea_abies_stump_01.et": 1}
    assert diag.unknown_assets == {"b_corylus_avellana_1_aut.et": 1}
    assert diag.parser_warning_count == 0
    assert diag.lines_read > 0
    assert diag.elapsed_seconds >= 0
    assert diag.canvas_bytes == 2 * 101 * 101
    assert diag.passes == 1
    assert diag.outputs["coniferous"].endswith("coniferous.png")

    summary = diag.format_summary(categories=config.categories, sizes=[0, 1, 2, 3], report_unknown=True)
    assert "Files processed      : 1" in summary
    assert "b_corylus_avellana_1_aut.et" in summary
    assert "t_picea_abies_stump_01.et" in summary
    assert "Out-of-bounds trees  : 1" in summary
    assert "Skipped size-0 trees : 1" in summary
    report = diag.to_dict()
    assert report["rendered"] == {"coniferous": 2, "deciduous": 1}
    json.dumps(report)  # serialisable


def test_flip_y_moves_rows(tmp_path, layer_file):
    plain = run(make_config(tmp_path), [layer_file])
    flipped_cfg = make_config(tmp_path / "flip", coordinate_system={"flip_y": True})
    flipped = run(flipped_cfg, [layer_file])
    a = load_png(plain.outputs["coniferous"])
    b = load_png(flipped.outputs["coniferous"])
    assert np.array_equal(a[::-1], b)


def test_flip_x_and_y_mirror_both_axes(tmp_path, layer_file):
    plain = run(make_config(tmp_path), [layer_file])
    x_only = run(make_config(tmp_path / "x", coordinate_system={"flip_x": True}), [layer_file])
    both = run(make_config(tmp_path / "xy", coordinate_system={"flip_x": True, "flip_y": True}), [layer_file])
    for category in ("coniferous", "deciduous"):
        a = load_png(plain.outputs[category])
        assert np.array_equal(a[:, ::-1], load_png(x_only.outputs[category]))
        assert np.array_equal(a[::-1, ::-1], load_png(both.outputs[category]))


def test_low_memory_mode_matches_single_pass(tmp_path, layer_file):
    single = run(make_config(tmp_path / "single"), [layer_file])
    low = run(make_config(tmp_path / "low"), [layer_file], low_memory=True)
    for category in ("coniferous", "deciduous"):
        assert np.array_equal(load_png(single.outputs[category]), load_png(low.outputs[category]))
    assert low.diagnostics.passes == 2
    assert low.diagnostics.canvas_bytes == 101 * 101
    assert low.diagnostics.rendered == single.diagnostics.rendered
    assert low.diagnostics.entities_inspected == single.diagnostics.entities_inspected
    assert low.diagnostics.skipped_size0 == single.diagnostics.skipped_size0
    assert low.diagnostics.out_of_bounds == single.diagnostics.out_of_bounds
    assert low.diagnostics.unknown_assets == single.diagnostics.unknown_assets


def test_directory_input_and_multiple_files(tmp_path, layer_file):
    nested = tmp_path / "layers" / "sub"
    nested.mkdir(parents=True)
    (nested / "a.layer").write_text(END_TO_END_LAYER, encoding="utf-8")
    (nested / "b.layer").write_text(SYNTHETIC_LAYER, encoding="utf-8")
    (nested / "ignore.txt").write_text("nope", encoding="utf-8")
    result = run(make_config(tmp_path), [tmp_path / "layers"])
    assert len(result.diagnostics.files) == 2
    assert result.diagnostics.rendered["coniferous"] == 2  # synthetic trees are all out of the 0..100 world
    assert result.diagnostics.out_of_bounds_total >= 1


def test_no_inputs_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        run(make_config(tmp_path), [empty])


def test_cli_end_to_end(tmp_path, layer_file, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(base_config_dict()), encoding="utf-8")
    out_dir = tmp_path / "cli_out"
    report = tmp_path / "report.json"
    code = main(
        [
            "--input", str(layer_file),
            "--config", str(cfg),
            "--output-dir", str(out_dir),
            "--marker-size", "3=1",
            "--report-unknown",
            "--json-report", str(report),
            "--quiet",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "TreeMasks summary" in captured.out
    assert "b_corylus_avellana_1_aut.et" in captured.out
    assert (out_dir / "coniferous.png").is_file()
    con = load_png(out_dir / "coniferous.png")
    assert con.sum() // 255 == 2 * 5  # radius overridden to 1 for size 3
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["config"]["rendering"]["marker_sizes"]["3"] == 1
    assert data["diagnostics"]["rendered"]["coniferous"] == 2


def test_cli_reports_config_errors(tmp_path, capsys):
    code = main(["--input", str(tmp_path), "--config", str(tmp_path / "missing.yaml"), "--quiet"])
    assert code == 1


def test_cli_rejects_bad_marker_size(tmp_path):
    with pytest.raises(SystemExit):
        main(["--input", str(tmp_path), "--marker-size", "three=1"])
