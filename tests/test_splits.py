from __future__ import annotations

import json

import pytest

from src.splits import (
    SCHEMA_VERSION,
    balanced_group_folds,
    grouped_cv_splits,
    load_split,
    random_split_indices,
    random_split_subset,
    random_cv_splits,
    validate_split,
    write_split,
)


def test_historical_random_split_is_deterministic():
    first = random_split_indices(101, seed=42)
    second = random_split_indices(101, seed=42)
    assert first == second
    assert [len(part) for part in first] == [80, 10, 11]
    validate_split(dict(zip(("train", "val", "test"), first)), 101)


def test_random_subset_keeps_excluded_rows_out():
    parts = random_split_subset([0, 2, 3, 5, 6, 8, 9], seed=42)
    split = dict(zip(("train", "val", "test"), parts))
    split["excluded"] = [1, 4, 7]
    counts = validate_split(split, 10)
    assert counts["excluded"] == 3
    assert set().union(*(set(part) for part in parts)).isdisjoint({1, 4, 7})


def test_balanced_group_folds_keep_groups_intact():
    labels = ["a"] * 8 + ["b"] * 7 + ["c"] * 6 + ["d"] * 5 + ["e"] * 4
    folds, groups = balanced_group_folds(labels, n_folds=5, seed=7)
    assert sorted(i for fold in folds for i in fold) == list(range(len(labels)))
    assert sorted(group for fold in groups for group in fold) == sorted(set(labels))
    for fold, group_names in zip(folds, groups):
        assert {labels[i] for i in fold} == set(group_names)


def test_grouped_cv_has_disjoint_group_sets():
    labels = [f"g{i // 3}" for i in range(30)]
    for split in grouped_cv_splits(labels, n_folds=5, seed=42):
        validate_split(split, len(labels))
        group_sets = {name: set(split["groups"][name]) for name in ("train", "val", "test")}
        assert group_sets["train"].isdisjoint(group_sets["val"])
        assert group_sets["train"].isdisjoint(group_sets["test"])
        assert group_sets["val"].isdisjoint(group_sets["test"])


def test_random_cv_tests_every_sample_once():
    indices = list(range(103))
    splits = random_cv_splits(indices, n_folds=5, seed=52)
    tested = [index for split in splits for index in split["test"]]
    assert sorted(tested) == indices
    assert len(tested) == len(set(tested))
    for split in splits:
        validate_split(split, len(indices))


def test_split_round_trip_checks_dataset_identity(tmp_path):
    path = tmp_path / "split.json"
    write_split(
        path, "toy", range(5), [5, 6], [7, 8], 10, "abc", "toy", excluded=[9]
    )
    loaded = load_split(path, 10, expected_data_sha256="abc")
    assert loaded["schema_version"] == SCHEMA_VERSION
    with pytest.raises(ValueError, match="SHA256"):
        load_split(path, 10, expected_data_sha256="different")


def test_validate_split_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_split({"train": [0, 1], "val": [1], "test": [2]}, 3)
