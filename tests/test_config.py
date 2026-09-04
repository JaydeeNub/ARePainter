from __future__ import annotations

import json

import pytest
import yaml

from treemasks.config import ConfigError, apply_overrides, config_from_mapping, load_config

from conftest import base_config_dict


def test_json_config_with_string_keys(tmp_path):
    data = base_config_dict()
    data["rendering"]["marker_sizes"] = {"0": 0, "1": 3}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    config = load_config(path)
    assert config.rendering.marker_sizes == {0: 0, 1: 3}


def test_yaml_config_defaults(tmp_path):
    data = base_config_dict()
    del data["coordinate_system"]
    del data["classifier"]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = load_config(path)
    assert config.coordinate_system.flip_y is False
    assert config.coordinate_system.flip_x is False
    assert config.classifier.exclude == ()
    assert config.output.compress_level == 6
    assert config.parser.file_pattern == "*.layer"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.pop("world"), "world"),
        (lambda d: d["world"].pop("max_y"), "world.max_y"),
        (lambda d: d["world"].update(max_x=0), "max_x"),
        (lambda d: d["output"].update(width=0), "width"),
        (lambda d: d["output"].update(compress_level=12), "compress_level"),
        (lambda d: d["rendering"]["marker_sizes"].update({0: 2}), "size-0"),
        (lambda d: d["rendering"]["marker_sizes"].update({2: -1}), ">= 0"),
        (lambda d: d["rendering"].update(marker_value=0), "marker_value"),
        (lambda d: d.update(trees={}), "trees"),
        (lambda d: d.update(trees={"bad/name": ["x"]}), "file-name-safe"),
        (lambda d: d.update(trees={"c": []}), "no asset patterns"),
        (lambda d: d.update(classifier={"size_regex": "("}), "valid regex"),
        (lambda d: d.update(classifier={"size_regex": r"_(\d)"}), "named group"),
        (lambda d: d.update(coordinate_system={"x_axis": 0, "y_axis": 0}), "distinct"),
        (lambda d: d["output"].update(width="wide"), "integer"),
    ],
)
def test_invalid_configs_raise(mutate, message):
    data = base_config_dict()
    mutate(data)
    with pytest.raises(ConfigError, match=message):
        config_from_mapping(data)


def test_missing_or_empty_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(empty)
    broken = tmp_path / "broken.yaml"
    broken.write_text("world: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse"):
        load_config(broken)


def test_apply_overrides_merges_and_validates():
    config = config_from_mapping(base_config_dict())
    same = apply_overrides(config, {"width": None, "flip_y": None})
    assert same is config

    updated = apply_overrides(
        config,
        {
            "world_min_x": 10,
            "world_max_x": 20,
            "width": 11,
            "height": 21,
            "flip_x": True,
            "flip_y": True,
            "marker_sizes": {3: 9},
            "marker_value": 128,
            "output_dir": "elsewhere",
            "compress_level": 1,
            "file_pattern": "*.txt",
        },
    )
    assert updated.world.min_x == 10 and updated.world.max_x == 20
    assert updated.world.min_y == 0 and updated.world.max_y == 100  # untouched
    assert (updated.output.width, updated.output.height) == (11, 21)
    assert updated.coordinate_system.flip_x is True
    assert updated.coordinate_system.flip_y is True
    assert updated.rendering.marker_sizes == {0: 0, 1: 1, 2: 2, 3: 9}
    assert updated.rendering.marker_value == 128
    assert updated.output.directory == "elsewhere"
    assert updated.output.compress_level == 1
    assert updated.parser.file_pattern == "*.txt"
    assert config.rendering.marker_sizes == {0: 0, 1: 1, 2: 2, 3: 3}  # original untouched

    with pytest.raises(ConfigError):
        apply_overrides(config, {"world_max_x": -5})
    with pytest.raises(ConfigError):
        apply_overrides(config, {"marker_sizes": {0: 4}})


def test_to_dict_round_trips_through_yaml():
    config = config_from_mapping(base_config_dict())
    dumped = yaml.safe_load(yaml.safe_dump(config.to_dict()))
    rebuilt = config_from_mapping(dumped)
    assert rebuilt.world == config.world
    assert rebuilt.rendering.marker_sizes == config.rendering.marker_sizes
    assert rebuilt.trees == config.trees
