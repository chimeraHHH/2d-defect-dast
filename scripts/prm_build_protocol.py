"""Audit the IMP2D pickle and build the immutable PRM split package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.splits import (
    grouped_cv_splits,
    random_split_indices,
    random_split_subset,
    write_split,
)


G6_HOSTS = {"MoS2", "MoSe2", "MoTe2", "WS2", "WSe2", "WTe2", "MoSSe"}
G45_HOSTS = {
    "NbS2", "NbSe2", "TaS2", "TaSe2", "TiS2", "ZrS2", "ZrSe2",
    "HfS2", "HfSe2",
}
DOPANTS_3D = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
DOPANTS_4D = {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"}


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    with path.open("rb") as handle:
        blob = pickle.load(handle)
    if isinstance(blob, dict) and "data" in blob:
        return list(blob["data"]), blob.get("meta")
    if isinstance(blob, list):
        return blob, None
    raise TypeError(f"unsupported dataset object: {type(blob).__name__}")


def coordinate_fingerprint(sample: Dict[str, Any]) -> str:
    """Hash exact atom order and coordinates after conservative rounding."""
    digest = hashlib.sha256()
    for key, dtype in (("numbers", "<i4"), ("positions", "<f8"), ("cell", "<f8")):
        array = np.asarray(sample[key])
        if np.issubdtype(array.dtype, np.floating):
            array = np.round(array.astype(dtype), decimals=6)
        else:
            array = array.astype(dtype)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def duplicate_summary(values: Iterable[Any]) -> Dict[str, int]:
    counts = Counter(values)
    repeated = [count for count in counts.values() if count > 1]
    return {
        "unique": len(counts),
        "repeated_groups": len(repeated),
        "samples_in_repeated_groups": int(sum(repeated)),
        "max_multiplicity": int(max(counts.values(), default=0)),
    }


def write_sample_table(path: Path, samples: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "sample_index", "id", "unique_id", "host", "dopant", "defecttype",
        "site", "target_eV", "natoms", "spacegroup", "supercell",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample in enumerate(samples):
            meta = sample.get("metadata", {})
            writer.writerow(
                {
                    "sample_index": index,
                    "id": sample.get("id", ""),
                    "unique_id": sample.get("unique_id", ""),
                    "host": meta.get("host", ""),
                    "dopant": meta.get("dopant", ""),
                    "defecttype": meta.get("defecttype", ""),
                    "site": meta.get("site", ""),
                    "target_eV": float(sample["target"]),
                    "natoms": int(meta.get("natoms", len(sample["numbers"]))),
                    "spacegroup": meta.get("spacegroup", ""),
                    "supercell": meta.get("supercell", ""),
                }
            )


def build_splits(
    samples: Sequence[Dict[str, Any]],
    retained_indices: Sequence[int],
    excluded_indices: Sequence[int],
    out_dir: Path,
    data_sha256: str,
) -> List[Dict[str, Any]]:
    split_dir = out_dir / "splits"
    n_samples = len(samples)
    records: List[Dict[str, Any]] = []

    historical = random_split_indices(n_samples, seed=42)
    records.append(
        write_split(
            split_dir / "id_historical_s42.json", "id_historical_s42",
            *historical, n_samples, data_sha256, "historical_random_80_10_10",
            {
                "split_seed": 42,
                "status": "historical_only",
                "warning": "Contains symmetry-equivalent duplicates and was repeatedly inspected.",
            },
        )
    )

    smoke_pool = list(retained_indices[:256])
    smoke_train = smoke_pool[:128]
    smoke_val = smoke_pool[128:192]
    smoke_test = smoke_pool[192:]
    smoke_excluded = sorted(set(range(n_samples)).difference(smoke_pool))
    records.append(
        write_split(
            split_dir / "smoke_protocol.json", "smoke_protocol",
            smoke_train, smoke_val, smoke_test, n_samples, data_sha256,
            "smoke_only",
            {"status": "non_scientific", "purpose": "pipeline schema verification"},
            excluded=smoke_excluded,
        )
    )

    for split_seed in range(42, 47):
        train, val, test = random_split_subset(retained_indices, seed=split_seed)
        split_id = f"id_repeat_s{split_seed}"
        records.append(
            write_split(
                split_dir / f"{split_id}.json", split_id, train, val, test,
                n_samples, data_sha256, "random_80_10_10",
                {
                    "split_seed": split_seed,
                    "status": "development" if split_seed == 42 else "confirmatory_repeat",
                    "warning": "seed 42 was inspected in historical development"
                    if split_seed == 42 else "predefined before PRM reruns",
                    "deduplication": "one representative per rounded-coordinate fingerprint",
                },
                excluded=excluded_indices,
            )
        )

    hosts = [str(s.get("metadata", {}).get("host", "unknown")) for s in samples]
    dopants = [str(s.get("metadata", {}).get("dopant", "unknown")) for s in samples]
    for axis, labels in (("host", hosts), ("dopant", dopants)):
        retained_labels = [labels[i] for i in retained_indices]
        for split in grouped_cv_splits(retained_labels, n_folds=5, seed=42):
            fold = int(split["fold"])
            split_id = f"{axis}_cv5_f{fold}"
            train = [retained_indices[i] for i in split["train"]]
            val = [retained_indices[i] for i in split["val"]]
            test = [retained_indices[i] for i in split["test"]]
            records.append(
                write_split(
                    split_dir / f"{split_id}.json", split_id,
                    train, val, test,
                    n_samples, data_sha256, f"{axis}_grouped_cv5",
                    {
                        "fold": fold, "groups": split["groups"], "assignment_seed": 42,
                        "deduplication": "one representative per rounded-coordinate fingerprint",
                    },
                    excluded=excluded_indices,
                )
            )

    test = [
        i for i in retained_indices for sample in (samples[i],)
        if sample.get("metadata", {}).get("host") in G6_HOSTS
        and sample.get("metadata", {}).get("dopant") in DOPANTS_3D
    ]
    val = [
        i for i in retained_indices for sample in (samples[i],)
        if sample.get("metadata", {}).get("host") in G45_HOSTS
        and sample.get("metadata", {}).get("dopant") in DOPANTS_4D
    ]
    held_out = set(test).union(val)
    train = [i for i in retained_indices if i not in held_out]
    records.append(
        write_split(
            split_dir / "chemistry_block_g6x3d.json",
            "chemistry_block_g6x3d", train, val, test, n_samples, data_sha256,
            "matrix_completion_block",
            {
                "test_host_group": sorted(G6_HOSTS),
                "test_dopant_group": sorted(DOPANTS_3D),
                "validation_host_group": sorted(G45_HOSTS),
                "validation_dopant_group": sorted(DOPANTS_4D),
                "deduplication": "one representative per rounded-coordinate fingerprint",
            },
            excluded=excluded_indices,
        )
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/prm_protocol_v1"))
    args = parser.parse_args()

    data_path = args.data.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sha256 = file_sha256(data_path)
    samples, embedded_meta = load_samples(data_path)
    targets = np.asarray([float(sample["target"]) for sample in samples])
    metadata = [sample.get("metadata", {}) for sample in samples]

    ids = [str(sample.get("id", "")) for sample in samples]
    unique_ids = [str(sample.get("unique_id", "")) for sample in samples]
    semantic_keys = [
        (
            str(meta.get("host", "")), str(meta.get("dopant", "")),
            str(meta.get("defecttype", "")), str(meta.get("site", "")),
        )
        for meta in metadata
    ]
    fingerprints = [coordinate_fingerprint(sample) for sample in samples]
    fingerprint_groups: Dict[str, List[int]] = defaultdict(list)
    for index, fingerprint in enumerate(fingerprints):
        fingerprint_groups[fingerprint].append(index)
    retained_indices = [min(group) for group in fingerprint_groups.values()]
    retained_indices.sort()
    retained_set = set(retained_indices)
    excluded_indices = [i for i in range(len(samples)) if i not in retained_set]
    repeated_groups = [group for group in fingerprint_groups.values() if len(group) > 1]
    duplicate_target_deltas = [
        max(float(samples[i]["target"]) for i in group)
        - min(float(samples[i]["target"]) for i in group)
        for group in repeated_groups
    ]

    semantic_targets: Dict[tuple[str, str, str, str], List[float]] = defaultdict(list)
    for key, target in zip(semantic_keys, targets):
        semantic_targets[key].append(float(target))
    conflicting_semantic_groups = sum(
        1 for values in semantic_targets.values()
        if len(values) > 1 and max(values) - min(values) > 1e-6
    )

    quantile_levels = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    quantiles = np.quantile(targets, quantile_levels)
    audit = {
        "schema_version": "prm_data_audit_v1",
        "dataset": {
            "path_recorded": str(data_path),
            "file_name": data_path.name,
            "size_bytes": data_path.stat().st_size,
            "sha256": sha256,
            "n_samples": len(samples),
            "container_type": "dict[data]" if embedded_meta is not None else "list",
            "embedded_meta": embedded_meta,
        },
        "target_eV": {
            "mean": float(targets.mean()),
            "std": float(targets.std(ddof=1)),
            "min": float(targets.min()),
            "max": float(targets.max()),
            "quantiles": {
                str(level): float(value)
                for level, value in zip(quantile_levels, quantiles)
            },
            "n_negative": int(np.sum(targets < 0)),
            "n_above_7_eV": int(np.sum(targets >= 7)),
            "n_abs_above_20_eV": int(np.sum(np.abs(targets) > 20)),
            "all_finite": bool(np.isfinite(targets).all()),
        },
        "metadata_cardinality": {
            key: len({str(meta.get(key, "")) for meta in metadata})
            for key in ("host", "dopant", "defecttype", "site", "spacegroup", "supercell")
        },
        "metadata_counts": {
            key: dict(sorted(Counter(str(meta.get(key, "")) for meta in metadata).items()))
            for key in ("host", "dopant", "defecttype")
        },
        "duplicates": {
            "id": duplicate_summary(ids),
            "unique_id": duplicate_summary(unique_ids),
            "host_dopant_defecttype_site": duplicate_summary(semantic_keys),
            "rounded_coordinate_fingerprint": duplicate_summary(fingerprints),
            "semantic_groups_with_target_conflicts": conflicting_semantic_groups,
            "canonical_deduplication": {
                "strategy": "retain the lowest sample index per rounded-coordinate fingerprint",
                "n_retained": len(retained_indices),
                "n_excluded": len(excluded_indices),
                "excluded_indices": excluded_indices,
                "max_target_delta_eV_within_duplicate_group": float(
                    max(duplicate_target_deltas, default=0.0)
                ),
            },
            "fingerprint_note": "Atom order, numbers, cell and positions rounded to 1e-6; not invariant to translation or permutation.",
        },
        "formation_energy_provenance": {
            "auditable_from_clean_pickle": False,
            "reason": "The cleaned samples contain the derived target but not pristine, defect and chemical-potential energy components.",
        },
    }

    write_sample_table(out_dir / "samples.csv", samples)
    splits = build_splits(
        samples, retained_indices, excluded_indices, out_dir, sha256
    )
    audit["splits"] = [
        {
            "split_id": split["split_id"],
            "protocol": split["protocol"],
            "counts": split["counts"],
        }
        for split in splits
    ]
    (out_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": "prm_protocol_manifest_v1",
        "data_sha256": sha256,
        "n_samples": len(samples),
        "data_audit": "data_audit.json",
        "sample_table": "samples.csv",
        "splits": [f"splits/{split['split_id']}.json" for split in splits],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
