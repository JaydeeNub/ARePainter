from __future__ import annotations

import pytest

from treemasks.classifier import (
    EXCLUDED,
    NO_SIZE,
    OTHER_CLASS,
    TREE,
    UNKNOWN,
    UNKNOWN_SIZE,
    TreeClassifier,
)
from treemasks.parser import Entity

CATEGORIES = {
    "coniferous": ["t_picea_abies_*", "t_piceaabies_*", "t_larix_decidua_*"],
    "deciduous": ["t_betula_pendula_*", "t_sorbus_aucuparia_*", "t_tilia_cordata_*", "t_carpinus_betulus_*"],
}
EXCLUDE = ["*_fallen*", "*_stump_*", "*_stem_*", "*_branch_*"]


@pytest.fixture
def classifier() -> TreeClassifier:
    return TreeClassifier(CATEGORIES, exclude=EXCLUDE)


@pytest.mark.parametrize(
    "name, size",
    [
        ("t_picea_abies_2sw.et", 2),
        ("t_picea_abies_3dw.et", 3),
        ("t_picea_abies_1f.et", 1),
        ("t_picea_abies_0sw.et", 0),
        ("t_betula_pendula_0_aut.et", 0),
        ("t_sorbus_aucuparia_3sw_aut.et", 3),
        ("t_larix_decidua_2fb_aut.et", 2),
        ("t_piceaabies_3f.et", 3),
        ("t_picea_abies_3d_fallen.et", 3),
        ("T_PICEA_ABIES_1SW.ET", 1),
        ("t_picea_abies_stump_03.et", None),
        ("t_picea_abies_stump_01_forest.et", None),
        ("t_betula_pendula_stem_01.et", None),
        ("t_carpinus_betulus_branch_01.et", None),
        ("GraniteStone_01_V2.et", None),
        ("JD_Forest_Spruce_suaut.et", None),
        ("", None),
    ],
)
def test_size_extraction(classifier, name, size):
    assert classifier.extract_size(name) == size


def test_category_matching_is_case_insensitive(classifier):
    assert classifier.match_category("t_picea_abies_2sw.et") == "coniferous"
    assert classifier.match_category("T_Picea_Abies_2SW.et") == "coniferous"
    assert classifier.match_category("t_sorbus_aucuparia_2s.et") == "deciduous"
    assert classifier.match_category("b_corylus_avellana_1_aut.et") is None


def test_first_matching_category_wins():
    clf = TreeClassifier({"first": ["t_*"], "second": ["t_picea_*"]})
    assert clf.classify_name("Tree", "t_picea_abies_1s.et").category == "first"


def test_classification_statuses(classifier):
    tree = classifier.classify_name("Tree", "t_picea_abies_2sw.et")
    assert classifier.classify_name("Tree", "t_picea_abies_2sw.et") is tree  # cached
    assert (tree.status, tree.category, tree.size) == (TREE, "coniferous", 2)
    assert tree.is_tree

    rock = classifier.classify_name("SCR_IndestructibleEnvironmentalEntity", "GraniteStone_01_V2.et")
    assert rock.status == OTHER_CLASS and rock.category is None

    excluded = classifier.classify_name("Tree", "t_picea_abies_3d_fallen.et")
    assert excluded.status == EXCLUDED and excluded.category is None

    stump = classifier.classify_name("Tree", "t_betula_pendula_stump_01.et")
    assert stump.status == EXCLUDED

    unknown = classifier.classify_name("Tree", "b_corylus_avellana_1_aut.et")
    assert unknown.status == UNKNOWN

    no_size = TreeClassifier({"c": ["t_picea_abies_*"]}).classify_name("Tree", "t_picea_abies_stump_03.et")
    assert no_size.status == NO_SIZE and no_size.category == "c"

    too_big = classifier.classify_name("Tree", "t_picea_abies_4s.et")
    assert too_big.status == UNKNOWN_SIZE and too_big.size == 4


def test_entity_class_filter_can_be_disabled():
    clf = TreeClassifier({"rocks": ["granite*"]}, entity_classes=(), size_regex=r"_(?P<size>\d)")
    result = clf.classify_name("SCR_IndestructibleEnvironmentalEntity", "GraniteStone_01_V2.et")
    assert result.status == TREE and result.size == 0

    strict = TreeClassifier({"c": ["t_*"]}, entity_classes=("Tree", "TreeEntity"))
    assert strict.classify_name("treeentity", "t_x_1s.et").status == TREE
    assert strict.classify_name("Bush", "t_x_1s.et").status == OTHER_CLASS


def test_classify_entity_uses_prefab_cache(classifier):
    prefab = "{009A68DF0B31B7B7}PrefabLibrary/CainLibrary/Vegetation/Trees/t_picea_abies_2sw.et"
    a = classifier.classify(Entity("Tree", prefab, (0.0, 0.0, 0.0), "f", 1))
    b = classifier.classify(Entity("Tree", prefab, (1.0, 1.0, 1.0), "f", 9))
    assert a is b
    assert a.asset_name == "t_picea_abies_2sw.et"
    assert a.category == "coniferous"


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        TreeClassifier({})
    with pytest.raises(ValueError):
        TreeClassifier({"c": ["t_*"]}, size_regex=r"_(\d)")


def test_category_names_order():
    clf = TreeClassifier({"b": ["x*"], "a": ["y*"]})
    assert clf.category_names == ["b", "a"]
