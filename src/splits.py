"""Immutable split definitions used by the PRM evaluation protocol."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "prm_split_v1"


def random_split_indices(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Return the historical Python-``random`` split deterministically."""
    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


def random_split_subset(
    indices: Sequence[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split an explicit canonical subset without reintroducing excluded rows."""
    shuffled = [int(i) for i in indices]
    random.Random(seed).shuffle(shuffled)
    n_train = int(train_ratio * len(shuffled))
    n_val = int(val_ratio * len(shuffled))
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def balanced_group_folds(
    labels: Sequence[str], n_folds: int = 5, seed: int = 42
) -> Tuple[List[List[int]], List[List[str]]]:
    """Assign whole groups to folds while approximately balancing sample count."""
    if n_folds < 3:
        raise ValueError("n_folds must be at least 3 to provide train/val/test")
    by_group: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        key = str(label)
        by_group.setdefault(key, []).append(idx)
    if len(by_group) < n_folds:
        raise ValueError(f"only {len(by_group)} groups for {n_folds} folds")

    rng = random.Random(seed)
    groups = list(by_group)
    rng.shuffle(groups)
    groups.sort(key=lambda key: len(by_group[key]), reverse=True)

    fold_indices: List[List[int]] = [[] for _ in range(n_folds)]
    fold_groups: List[List[str]] = [[] for _ in range(n_folds)]
    for group in groups:
        target = min(range(n_folds), key=lambda i: (len(fold_indices[i]), i))
        fold_indices[target].extend(by_group[group])
        fold_groups[target].append(group)
    for indices in fold_indices:
        indices.sort()
    for names in fold_groups:
        names.sort()
    return fold_indices, fold_groups


def grouped_cv_splits(
    labels: Sequence[str], n_folds: int = 5, seed: int = 42
) -> List[Dict[str, Any]]:
    """Create grouped CV splits with the next fold reserved for validation."""
    folds, fold_groups = balanced_group_folds(labels, n_folds=n_folds, seed=seed)
    all_indices = set(range(len(labels)))
    outputs: List[Dict[str, Any]] = []
    for test_fold in range(n_folds):
        val_fold = (test_fold + 1) % n_folds
        test = folds[test_fold]
        val = folds[val_fold]
        train = sorted(all_indices.difference(test).difference(val))
        outputs.append(
            {
                "train": train,
                "val": val,
                "test": test,
                "groups": {
                    "train": sorted(
                        set(labels[i] for i in train)
                    ),
                    "val": fold_groups[val_fold],
                    "test": fold_groups[test_fold],
                },
                "fold": test_fold,
            }
        )
    return outputs


def validate_split(
    split: Mapping[str, Sequence[int]],
    n_samples: int,
    require_full_coverage: bool = True,
) -> Dict[str, int]:
    """Validate bounds, uniqueness, disjointness and optional full coverage."""
    required = ("train", "val", "test")
    missing = [name for name in required if name not in split]
    if missing:
        raise ValueError(f"split is missing keys: {missing}")

    sets = {name: set(int(i) for i in split[name]) for name in required}
    excluded_values = split.get("excluded", [])
    sets["excluded"] = set(int(i) for i in excluded_values)
    for name in (*required, "excluded"):
        source = split[name] if name in required else excluded_values
        if len(sets[name]) != len(source):
            raise ValueError(f"duplicate indices in {name}")
        invalid = [i for i in sets[name] if i < 0 or i >= n_samples]
        if invalid:
            raise ValueError(f"out-of-range indices in {name}: {invalid[:5]}")
    pairs = (
        ("train", "val"), ("train", "test"), ("val", "test"),
        ("train", "excluded"), ("val", "excluded"), ("test", "excluded"),
    )
    for left, right in pairs:
        overlap = sets[left].intersection(sets[right])
        if overlap:
            raise ValueError(f"{left}/{right} overlap: {sorted(overlap)[:5]}")
    covered = set().union(*sets.values())
    if require_full_coverage and len(covered) != n_samples:
        raise ValueError(f"split covers {len(covered)} of {n_samples} samples")
    return {name: len(sets[name]) for name in (*required, "excluded")}


def write_split(
    path: str | Path,
    split_id: str,
    train: Iterable[int],
    val: Iterable[int],
    test: Iterable[int],
    n_samples: int,
    data_sha256: str,
    protocol: str,
    metadata: Mapping[str, Any] | None = None,
    excluded: Iterable[int] | None = None,
) -> Dict[str, Any]:
    """Validate and write one canonical split JSON."""
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split_id": split_id,
        "protocol": protocol,
        "n_samples": int(n_samples),
        "data_sha256": data_sha256,
        "train": sorted(int(i) for i in train),
        "val": sorted(int(i) for i in val),
        "test": sorted(int(i) for i in test),
        "excluded": sorted(int(i) for i in (excluded or [])),
        "metadata": dict(metadata or {}),
    }
    payload["counts"] = validate_split(payload, n_samples)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_split(
    path: str | Path,
    n_samples: int,
    expected_data_sha256: str | None = None,
) -> Dict[str, Any]:
    """Load and validate a canonical split JSON."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported split schema {payload.get('schema_version')!r}"
        )
    if int(payload.get("n_samples", -1)) != n_samples:
        raise ValueError(
            f"split expects {payload.get('n_samples')} samples, found {n_samples}"
        )
    if expected_data_sha256 and payload.get("data_sha256") != expected_data_sha256:
        raise ValueError("dataset SHA256 does not match split definition")
    validate_split(payload, n_samples)
    return payload
