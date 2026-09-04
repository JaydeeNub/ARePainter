"""Classify parsed entities into tree categories and size classes by asset name."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .parser import asset_basename

DEFAULT_SIZE_REGEX = r"_(?P<size>\d)[a-z]*(?:_[a-z]+)*\.et$"

# Classification outcomes.
TREE = "tree"                  # category and size known; candidate for rendering
OTHER_CLASS = "other_class"    # entity class is not a tree class (rocks, shapes, generators...)
EXCLUDED = "excluded"          # matched an exclude pattern (stumps, fallen trunks...)
UNKNOWN = "unknown"            # tree-class entity whose asset matches no category
NO_SIZE = "no_size"            # category matched but no size digit could be extracted
UNKNOWN_SIZE = "unknown_size"  # size digit extracted but not present in the marker table


@dataclass(frozen=True, slots=True)
class Classification:
    status: str
    asset_name: str
    category: str | None = None
    size: int | None = None

    @property
    def is_tree(self) -> bool:
        return self.status == TREE


def _compile_globs(patterns: Iterable[str]) -> re.Pattern[str] | None:
    parts = [fnmatch.translate(p.lower()) for p in patterns if p]
    if not parts:
        return None
    return re.compile("|".join(f"(?:{p})" for p in parts))


class TreeClassifier:
    """Map (entity class, asset file name) to a category and size using configurable rules.

    Matching is case-insensitive. Results are cached per prefab because a map
    holds millions of entities but only a few hundred distinct assets.
    """

    def __init__(
        self,
        categories: Mapping[str, Sequence[str]],
        *,
        exclude: Sequence[str] = (),
        size_regex: str = DEFAULT_SIZE_REGEX,
        entity_classes: Sequence[str] = ("Tree",),
        valid_sizes: Iterable[int] = (0, 1, 2, 3),
    ) -> None:
        if not categories:
            raise ValueError("at least one tree category must be configured")
        self._categories: list[tuple[str, re.Pattern[str] | None]] = [
            (name, _compile_globs(patterns)) for name, patterns in categories.items()
        ]
        self._exclude = _compile_globs(exclude)
        self._size_re = re.compile(size_regex, re.IGNORECASE)
        if "size" not in self._size_re.groupindex:
            raise ValueError("size_regex must define a named group called 'size'")
        self._classes = {c.lower() for c in entity_classes}
        self._valid_sizes = set(valid_sizes)
        self._by_name: dict[tuple[str, str], Classification] = {}
        self._by_prefab: dict[tuple[str, str], Classification] = {}

    @property
    def category_names(self) -> list[str]:
        return [name for name, _ in self._categories]

    def extract_size(self, asset_name: str) -> int | None:
        match = self._size_re.search(asset_name.lower())
        return int(match.group("size")) if match else None

    def match_category(self, asset_name: str) -> str | None:
        lowered = asset_name.lower()
        for name, pattern in self._categories:
            if pattern is not None and pattern.match(lowered):
                return name
        return None

    def is_excluded(self, asset_name: str) -> bool:
        return self._exclude is not None and self._exclude.match(asset_name.lower()) is not None

    def classify_name(self, class_name: str, asset_name: str) -> Classification:
        key = (class_name, asset_name)
        cached = self._by_name.get(key)
        if cached is None:
            cached = self._classify_uncached(class_name, asset_name)
            self._by_name[key] = cached
        return cached

    def classify_prefab(self, class_name: str, prefab: str) -> Classification:
        key = (class_name, prefab)
        cached = self._by_prefab.get(key)
        if cached is None:
            cached = self.classify_name(class_name, asset_basename(prefab))
            self._by_prefab[key] = cached
        return cached

    def classify(self, entity) -> Classification:  # entity: parser.Entity
        return self.classify_prefab(entity.class_name, entity.prefab)

    def _classify_uncached(self, class_name: str, asset_name: str) -> Classification:
        if self._classes and class_name.lower() not in self._classes:
            return Classification(OTHER_CLASS, asset_name)
        if self.is_excluded(asset_name):
            return Classification(EXCLUDED, asset_name)
        category = self.match_category(asset_name)
        if category is None:
            return Classification(UNKNOWN, asset_name)
        size = self.extract_size(asset_name)
        if size is None:
            return Classification(NO_SIZE, asset_name, category)
        if size not in self._valid_sizes:
            return Classification(UNKNOWN_SIZE, asset_name, category, size)
        return Classification(TREE, asset_name, category, size)
