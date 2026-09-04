from __future__ import annotations

import inspect
import math

import pytest

from treemasks.parser import (
    Entity,
    ParseStats,
    asset_basename,
    iter_layer_files,
    parse_file,
    parse_paths,
    transform_local,
)

from conftest import SYNTHETIC_LAYER


@pytest.fixture
def synthetic_file(tmp_path):
    path = tmp_path / "synthetic.layer"
    path.write_text(SYNTHETIC_LAYER, encoding="utf-8")
    return path


def by_asset(entities: list[Entity]) -> dict[str, list[Entity]]:
    grouped: dict[str, list[Entity]] = {}
    for entity in entities:
        grouped.setdefault(entity.asset_name, []).append(entity)
    return grouped


def test_asset_basename_variants():
    assert asset_basename("{009A68DF0B31B7B7}PrefabLibrary/Trees/t_picea_abies_2sw.et") == "t_picea_abies_2sw.et"
    assert asset_basename("{009A68DF0B31B7B7}t_picea_abies_2sw.et") == "t_picea_abies_2sw.et"
    assert asset_basename("Prefabs\\Trees\\t_x_1.et") == "t_x_1.et"
    assert asset_basename("") == ""


def test_parse_file_is_a_generator(synthetic_file):
    result = parse_file(synthetic_file)
    assert inspect.isgenerator(result)


def test_world_coordinates_accumulate_parent_chain(synthetic_file):
    stats = ParseStats()
    entities = list(parse_file(synthetic_file, stats))
    grouped = by_asset(entities)

    # polyline (1000, 50, 2000) + generator (10, 0, 20) + tree (1, -3, 2)
    spruce = grouped["t_picea_abies_2sw.et"]
    assert len(spruce) == 2
    assert spruce[0].coords == pytest.approx((1011.0, 47.0, 2022.0))
    assert spruce[1].coords == pytest.approx((1005.0, 50.0, 2025.0))
    assert spruce[0].class_name == "Tree"
    assert spruce[0].prefab.endswith("Trees/t_picea_abies_2sw.et")

    # single (non-group) tree entity inside the same generator
    birch = grouped["t_betula_pendula_3sw_aut.et"][0]
    assert birch.coords == pytest.approx((1010.5, 49.5, 2019.5))

    # second polyline instance uses its own origin
    small = grouped["t_betula_pendula_0_aut.et"][0]
    assert small.coords == pytest.approx((3001.0, 10.0, 4001.0))

    # rocks are emitted too (classification decides what to keep)
    rock = grouped["GraniteStone_01_V2.et"][0]
    assert rock.class_name == "SCR_IndestructibleEnvironmentalEntity"
    assert rock.coords == pytest.approx((1017.0, 50.0, 2027.0))

    # ShapePoint / ForestGeneratorPointData inside the Points block are not entities
    assert all(e.class_name not in {"ShapePoint", "ForestGeneratorPointData", "Points", "Data"} for e in entities)
    assert stats.warning_count == 0
    assert stats.entities == len(entities)
    assert stats.lines > 0


def test_entity_counts_and_sources(synthetic_file):
    entities = list(parse_file(synthetic_file))
    classes = sorted(e.class_name for e in entities)
    # 2 polyline instances, 2 generators, 2+1+1+1+1 trees, 1 rock
    assert classes.count("PolylineShapeEntity") == 2
    assert classes.count("ForestGeneratorEntity") == 2
    assert classes.count("Tree") == 6
    assert classes.count("SCR_IndestructibleEnvironmentalEntity") == 1
    assert all(e.source == str(synthetic_file) for e in entities)
    assert all(e.line > 0 for e in entities)


def test_rotated_parent_rotates_child_offset(synthetic_file):
    grouped = by_asset(list(parse_file(synthetic_file)))
    parent = grouped["t_sorbus_aucuparia_1w_aut.et"][0]
    child = grouped["t_sorbus_aucuparia_2s.et"][0]
    assert parent.coords == pytest.approx((5.0, 1.0, 6.0))
    # parent yaw 90 degrees: local +X becomes world -Z (left-handed, Y up)
    assert child.coords == pytest.approx((5.0, 1.0, 5.0), abs=1e-9)


@pytest.mark.parametrize(
    "yaw, local, expected",
    [
        (0.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        (90.0, (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        (90.0, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        (180.0, (1.0, 2.0, 3.0), (-1.0, 2.0, -3.0)),
        (-90.0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ],
)
def test_transform_local(yaw, local, expected):
    result = transform_local((10.0, 20.0, 30.0), yaw, local)
    assert result == pytest.approx((10.0 + expected[0], 20.0 + expected[1], 30.0 + expected[2]), abs=1e-9)


def test_transform_local_preserves_length():
    for yaw in (0.0, 33.3, 120.0, 271.5):
        x, _, z = transform_local((0.0, 0.0, 0.0), yaw, (3.0, 0.0, 4.0))
        assert math.hypot(x, z) == pytest.approx(5.0)


def test_unbalanced_and_late_coords_are_reported(tmp_path):
    text = (
        'Tree : "{A}Trees/t_picea_abies_1s.et" {\n'
        " {\n"
        '  Tree : "{B}Trees/t_picea_abies_2s.et" {\n'
        "   coords 1 0 1\n"
        "  }\n"
        " }\n"
        " coords 100 0 100\n"
        "}\n"
        "}\n"
        '$grp Tree : "{C}Trees/t_picea_abies_3s.et" {\n'
        " {\n"
        "  coords 1 2\n"
        " }\n"
    )
    path = tmp_path / "odd.layer"
    path.write_text(text, encoding="utf-8")
    stats = ParseStats()
    entities = list(parse_file(path, stats))
    assert len(entities) == 3
    messages = " ".join(stats.warnings)
    assert "coords appear after the children block" in messages
    assert "unexpected closing brace" in messages
    assert "unreadable coords" in messages
    assert "still open at end of file" in messages
    assert stats.warning_count == 4


def test_empty_inline_block_and_blank_lines(tmp_path):
    path = tmp_path / "inline.layer"
    path.write_text('Tree : "{A}t_x_1s.et" {\n\n Foo {}\n coords 1 2 3\n\n}\n', encoding="utf-8")
    entities = list(parse_file(path))
    assert len(entities) == 1
    assert entities[0].coords == (1.0, 2.0, 3.0)


def test_iter_layer_files_recurses_and_dedupes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "deep").mkdir()
    f1 = tmp_path / "a" / "one.layer"
    f2 = tmp_path / "a" / "deep" / "two.layer"
    other = tmp_path / "a" / "notes.txt"
    for f in (f1, f2, other):
        f.write_text("", encoding="utf-8")
    files = iter_layer_files([tmp_path, f1])
    assert sorted(files) == sorted([f1, f2])  # recursive, file given twice only once
    assert files == sorted(files)  # deterministic order
    assert other not in files
    with pytest.raises(FileNotFoundError):
        iter_layer_files([tmp_path / "missing.layer"])


def test_parse_paths_counts_files(tmp_path):
    p1 = tmp_path / "x.layer"
    p2 = tmp_path / "y.layer"
    p1.write_text('Tree : "{A}t_a_1s.et" {\n coords 1 0 1\n}\n', encoding="utf-8")
    p2.write_text('Tree : "{A}t_a_2s.et" {\n coords 2 0 2\n}\n', encoding="utf-8")
    stats = ParseStats()
    entities = list(parse_paths([p1, p2], stats))
    assert [e.asset_name for e in entities] == ["t_a_1s.et", "t_a_2s.et"]
    assert stats.files == 2
    assert stats.entities == 2
