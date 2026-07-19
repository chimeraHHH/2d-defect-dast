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

from src.defect_identity import dopant_candidate_indices
from src.splits import (
    grouped_cv_splits,
    random_cv_splits,
    random_split_indices,
    random_split_subset,
    random_split_with_calibration,
    write_split,
)


G6_HOSTS = {"MoS2", "MoSe2", "MoTe2", "WS2", "WSe2", "WTe2", "MoSSe"}
G45_HOSTS = {
    "NbS2", "NbSe2", "TaS2", "TaSe2", "TiS2", "ZrS2", "ZrSe2",
    "HfS2", "HfSe2",
}
DOPANTS_3D = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
DOPANTS_4D = {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"}
CANONICALIZATION = (
    "merged duplicate groups; rows with missing raw energy components or "
    "non-unique impurity identity excluded"
)


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


def invariant_distance_fingerprint(
    sample: Dict[str, Any], decimals: int = 4,
) -> str:
    """Hash composition and element-labelled pair distances invariantly."""
    numbers = np.asarray(sample["numbers"], dtype=np.int16)
    distances = np.asarray(sample["dist_matrix"], dtype=float)
    left, right = np.triu_indices(len(numbers), k=1)
    low_z = np.minimum(numbers[left], numbers[right])
    high_z = np.maximum(numbers[left], numbers[right])
    rounded_distance = np.round(distances[left, right], decimals=decimals)
    order = np.lexsort((rounded_distance, high_z, low_z))
    elements, counts = np.unique(numbers, return_counts=True)
    digest = hashlib.sha256()
    digest.update(elements.astype("<i2").tobytes())
    digest.update(counts.astype("<i2").tobytes())
    digest.update(low_z[order].astype("<i2").tobytes())
    digest.update(high_z[order].astype("<i2").tobytes())
    digest.update(rounded_distance[order].astype("<f8").tobytes())
    return digest.hexdigest()


def merged_duplicate_groups(
    fingerprint_sets: Sequence[Sequence[str]],
) -> List[List[int]]:
    """Return connected components induced by multiple duplicate fingerprints."""
    n_samples = len(fingerprint_sets[0])
    if any(len(values) != n_samples for values in fingerprint_sets):
        raise ValueError("fingerprint collections have different lengths")
    parent = list(range(n_samples))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for fingerprints in fingerprint_sets:
        first_by_fingerprint: Dict[str, int] = {}
        for index, fingerprint in enumerate(fingerprints):
            if fingerprint in first_by_fingerprint:
                union(index, first_by_fingerprint[fingerprint])
            else:
                first_by_fingerprint[fingerprint] = index
    components: Dict[int, List[int]] = defaultdict(list)
    for index in range(n_samples):
        components[find(index)].append(index)
    return sorted(components.values(), key=lambda group: group[0])


def duplicate_summary(values: Iterable[Any]) -> Dict[str, int]:
    counts = Counter(values)
    repeated = [count for count in counts.values() if count > 1]
    return {
        "unique": len(counts),
        "repeated_groups": len(repeated),
        "samples_in_repeated_groups": int(sum(repeated)),
        "max_multiplicity": int(max(counts.values(), default=0)),
    }


def write_sample_table(
    path: Path,
    samples: Sequence[Dict[str, Any]],
    retained_indices: set[int],
    exclusion_reasons: Dict[int, Sequence[str]],
) -> None:
    fields = [
        "sample_index", "id", "unique_id", "host", "dopant", "defecttype",
        "site", "target_eV", "natoms", "spacegroup", "supercell",
        "dopant_candidate_count", "canonical_retained", "exclusion_reasons",
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
                    "dopant_candidate_count": len(dopant_candidate_indices(sample)),
                    "canonical_retained": index in retained_indices,
                    "exclusion_reasons": ";".join(exclusion_reasons.get(index, ())),
                }
            )


def audit_defect_identity(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify rows where a relaxed structure cannot label one impurity atom."""
    candidate_counts = [len(dopant_candidate_indices(sample)) for sample in samples]
    missing_indices = [
        index for index, count in enumerate(candidate_counts) if count == 0
    ]
    if missing_indices:
        raise ValueError(
            "dopant element is absent from final structures at sample indices "
            f"{missing_indices[:10]}"
        )
    ambiguous_indices = [
        index for index, count in enumerate(candidate_counts) if count > 1
    ]
    rows = []
    for index in ambiguous_indices:
        sample = samples[index]
        meta = sample.get("metadata", {})
        rows.append(
            {
                "sample_index": index,
                "row_id": int(sample["id"]),
                "host": str(meta.get("host", "")),
                "dopant": str(meta.get("dopant", "")),
                "defecttype": str(meta.get("defecttype", "")),
                "site": str(meta.get("site", "")),
                "n_matching_atoms": candidate_counts[index],
            }
        )
    return {
        "policy": "require exactly one atom matching the dopant element",
        "rationale": (
            "The released relaxed structures contain no persistent atom-identity "
            "tag. Identical same-element nuclei are permutation equivalent, so an "
            "array-order heuristic cannot define a physical impurity node."
        ),
        "n_unique": len(samples) - len(ambiguous_indices),
        "n_missing": len(missing_indices),
        "n_ambiguous": len(ambiguous_indices),
        "ambiguous_indices": ambiguous_indices,
        "ambiguous_row_ids": [row["row_id"] for row in rows],
        "ambiguous_counts": {
            key: dict(sorted(Counter(row[key] for row in rows).items()))
            for key in ("host", "dopant", "defecttype", "site")
        },
        "ambiguous_rows": rows,
    }


def audit_raw_database(
    path: Path, samples: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    from ase.db import connect

    database = connect(str(path))
    valid_ids = []
    valid_rows = {}
    formula_deltas = []
    formula_component_exceptions = []
    target_deltas = []
    metadata_mismatches = 0
    counts = {
        "raw_rows": database.count(),
        "not_converged": 0,
        "missing_or_nonfinite_eform": 0,
        "outside_abs_20_eV": 0,
        "valid_after_filter": 0,
    }
    for row in database.select():
        if not bool(row.get("converged")):
            counts["not_converged"] += 1
            continue
        eform = row.get("eform")
        if eform is None or not np.isfinite(float(eform)):
            counts["missing_or_nonfinite_eform"] += 1
            continue
        if abs(float(eform)) > 20.0:
            counts["outside_abs_20_eV"] += 1
            continue
        counts["valid_after_filter"] += 1
        valid_ids.append(int(row.id))
        valid_rows[int(row.id)] = row
        components = (
            row.get("en2"), row.get("hostenergy"),
            row.get("dopant_chemical_potential"),
        )
        components_finite = all(
            value is not None and np.isfinite(float(value)) for value in components
        )
        # Six raw rows use en2 == 0 as a missing-value sentinel. Four survive
        # the published |eform| <= 20 eV filter and must not be interpreted as
        # physically meaningful total energies.
        if components_finite and float(components[0]) != 0.0:
            calculated = float(components[0]) - float(components[1]) - float(components[2])
            formula_deltas.append(abs(calculated - float(eform)))
        else:
            formula_component_exceptions.append(
                {
                    "row_id": int(row.id),
                    "reason": "en2_zero_sentinel" if components_finite else "nonfinite_component",
                    "en2": None if components[0] is None else float(components[0]),
                    "host": str(row.get("host", "")),
                    "dopant": str(row.get("dopant", "")),
                    "site": str(row.get("site", "")),
                    "defecttype": str(row.get("defecttype", "")),
                    "eform_eV": float(eform),
                }
            )

    sample_ids = [int(sample["id"]) for sample in samples]
    valid_id_set_matches = set(valid_ids) == set(sample_ids)
    if not valid_id_set_matches:
        missing = sorted(set(sample_ids).difference(valid_ids))[:10]
        extra = sorted(set(valid_ids).difference(sample_ids))[:10]
        raise ValueError(
            f"raw database and cleaned data IDs differ; missing={missing}, extra={extra}"
        )
    for sample in samples:
        row = valid_rows[int(sample["id"])]
        target_deltas.append(abs(float(sample["target"]) - float(row.get("eform"))))
        meta = sample.get("metadata", {})
        if any(
            str(meta.get(key, "")) != str(row.get(key, ""))
            for key in ("host", "dopant", "site", "defecttype")
        ):
            metadata_mismatches += 1

    max_formula_delta = float(max(formula_deltas, default=float("nan")))
    max_target_delta = float(max(target_deltas, default=float("nan")))
    if max_formula_delta > 1e-8:
        raise ValueError(f"formation-energy formula mismatch: {max_formula_delta:.6g} eV")
    if max_target_delta > 1e-12 or metadata_mismatches:
        raise ValueError(
            "cleaned data do not exactly reproduce the filtered raw database: "
            f"target_delta={max_target_delta:.6g}, metadata_mismatches={metadata_mismatches}"
        )

    return {
        "path_recorded": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "filter_replay": counts,
        "valid_id_set_matches_cleaned": valid_id_set_matches,
        "n_formula_verified": len(formula_deltas),
        "n_formula_component_exceptions": len(formula_component_exceptions),
        "formula_component_exceptions": formula_component_exceptions,
        "formula": "eform = en2 - hostenergy - dopant_chemical_potential",
        "max_formula_abs_delta_eV": max_formula_delta,
        "mean_formula_abs_delta_eV": float(np.mean(formula_deltas)) if formula_deltas else float("nan"),
        "max_cleaned_target_abs_delta_eV": max_target_delta,
        "metadata_mismatches": metadata_mismatches,
    }


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
                    "canonicalization": CANONICALIZATION,
                },
                excluded=excluded_indices,
            )
        )

    uq_train, uq_val, uq_calibration, uq_test = random_split_with_calibration(
        retained_indices, seed=62,
    )
    records.append(
        write_split(
            split_dir / "uq_calibration_s62.json", "uq_calibration_s62",
            uq_train, uq_val, uq_test, n_samples, data_sha256,
            "random_75_10_5_10_deduplicated_with_heldout_calibration",
            {
                "assignment_seed": 62,
                "selection_data": "validation partition only",
                "calibration_data": "dedicated calibration partition only",
                "test_data": "evaluation only",
                "canonicalization": CANONICALIZATION,
            },
            excluded=excluded_indices,
            calibration=uq_calibration,
        )
    )

    for split in random_cv_splits(retained_indices, n_folds=5, seed=52):
        fold = int(split["fold"])
        split_id = f"id_cv5_f{fold}"
        records.append(
            write_split(
                split_dir / f"{split_id}.json", split_id,
                split["train"], split["val"], split["test"],
                n_samples, data_sha256, "random_cv5_oof",
                {
                    "fold": fold,
                    "assignment_seed": 52,
                    "purpose": "out-of-fold predictions for paper analysis",
                    "canonicalization": CANONICALIZATION,
                },
                excluded=excluded_indices,
            )
        )

    hosts = [str(s.get("metadata", {}).get("host", "unknown")) for s in samples]
    dopants = [str(s.get("metadata", {}).get("dopant", "unknown")) for s in samples]
    pairs = [f"{host}|{dopant}" for host, dopant in zip(hosts, dopants)]
    for axis, labels in (("host", hosts), ("dopant", dopants), ("pair", pairs)):
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
                        "canonicalization": CANONICALIZATION,
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
                "canonicalization": CANONICALIZATION,
            },
            excluded=excluded_indices,
        )
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--raw-db", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/prm_protocol_v2"))
    args = parser.parse_args()

    data_path = args.data.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sha256 = file_sha256(data_path)
    samples, embedded_meta = load_samples(data_path)
    targets = np.asarray([float(sample["target"]) for sample in samples])
    metadata = [sample.get("metadata", {}) for sample in samples]
    raw_database_audit = audit_raw_database(
        args.raw_db.expanduser().resolve(), samples,
    )
    component_exception_ids = {
        int(row["row_id"])
        for row in raw_database_audit["formula_component_exceptions"]
    }
    provenance_excluded_indices = sorted(
        index for index, sample in enumerate(samples)
        if int(sample["id"]) in component_exception_ids
    )
    provenance_excluded_set = set(provenance_excluded_indices)
    defect_identity = audit_defect_identity(samples)
    identity_excluded_indices = list(defect_identity["ambiguous_indices"])
    identity_excluded_set = set(identity_excluded_indices)
    eligibility_excluded_set = provenance_excluded_set.union(identity_excluded_set)

    ids = [str(sample.get("id", "")) for sample in samples]
    unique_ids = [str(sample.get("unique_id", "")) for sample in samples]
    semantic_keys = [
        (
            str(meta.get("host", "")), str(meta.get("dopant", "")),
            str(meta.get("defecttype", "")), str(meta.get("site", "")),
        )
        for meta in metadata
    ]
    coordinate_fingerprints = [coordinate_fingerprint(sample) for sample in samples]
    distance_fingerprints = [
        invariant_distance_fingerprint(sample) for sample in samples
    ]
    all_groups = merged_duplicate_groups(
        [coordinate_fingerprints, distance_fingerprints]
    )
    retained_indices = []
    duplicate_excluded_indices = []
    for group in all_groups:
        eligible = [index for index in group if index not in eligibility_excluded_set]
        if not eligible:
            continue
        retained = min(eligible)
        retained_indices.append(retained)
        duplicate_excluded_indices.extend(
            index for index in eligible if index != retained
        )
    retained_set = set(retained_indices)
    excluded_indices = [i for i in range(len(samples)) if i not in retained_set]
    repeated_groups = [group for group in all_groups if len(group) > 1]
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
    exclusion_reasons: Dict[int, List[str]] = defaultdict(list)
    for index in provenance_excluded_indices:
        exclusion_reasons[index].append("raw_energy_component_missing")
    for index in identity_excluded_indices:
        exclusion_reasons[index].append("non_unique_impurity_identity")
    for index in duplicate_excluded_indices:
        exclusion_reasons[index].append("duplicate_structure")
    if set(exclusion_reasons) != set(excluded_indices):
        raise ValueError("canonical exclusion reasons do not cover every excluded row")

    audit = {
        "schema_version": "prm_data_audit_v2",
        "dataset": {
            "path_recorded": str(data_path),
            "file_name": data_path.name,
            "size_bytes": data_path.stat().st_size,
            "sha256": sha256,
            "n_samples": len(samples),
            "n_modeling_samples": len(retained_indices),
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
            "rounded_coordinate_fingerprint": duplicate_summary(coordinate_fingerprints),
            "invariant_element_pair_distance_fingerprint": duplicate_summary(
                distance_fingerprints
            ),
            "semantic_groups_with_target_conflicts": conflicting_semantic_groups,
            "canonical_deduplication": {
                "strategy": (
                    "retain the lowest sample index in each connected component "
                    "of coordinate and invariant-distance duplicate groups"
                ),
                "n_retained": len(retained_indices),
                "n_excluded": len(excluded_indices),
                "n_duplicate_excluded": len(duplicate_excluded_indices),
                "n_raw_component_excluded": len(provenance_excluded_indices),
                "n_ambiguous_identity_excluded": len(identity_excluded_indices),
                "n_pre_dedup_eligibility_excluded": len(eligibility_excluded_set),
                "n_pre_dedup_eligible": len(samples) - len(eligibility_excluded_set),
                "raw_component_excluded_indices": provenance_excluded_indices,
                "ambiguous_identity_excluded_indices": identity_excluded_indices,
                "excluded_indices": excluded_indices,
                "max_target_delta_eV_within_duplicate_group": float(
                    max(duplicate_target_deltas, default=0.0)
                ),
            },
            "fingerprint_note": (
                "Union of (i) atom-order/cell/Cartesian-coordinate hashes rounded "
                "to 1e-6 and (ii) permutation/translation-invariant, "
                "element-labelled periodic pair-distance spectra rounded to 1e-4 Angstrom."
            ),
        },
        "defect_identity": defect_identity,
        "all_modeling_rows_have_unique_impurity_identity": not any(
            index in identity_excluded_set for index in retained_indices
        ),
        "formation_energy_provenance": {
            "auditable_from_clean_pickle": False,
            "raw_database_audit": raw_database_audit,
            "all_modeling_rows_formula_verified": not any(
                index in provenance_excluded_set for index in retained_indices
            ),
            "status": (
                "cleaned targets verified against the filtered ASE database; "
                "all rows lacking auditable raw energy components are excluded "
                "from modeling splits"
            ),
        },
    }

    write_sample_table(
        out_dir / "samples.csv", samples, retained_set, exclusion_reasons,
    )
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
        "schema_version": "prm_protocol_manifest_v2",
        "supersedes": "artifacts/prm_protocol_v1",
        "supersession_reason": (
            "remove relaxed same-element structures whose impurity atom cannot "
            "be identified without an order-dependent label"
        ),
        "data_sha256": sha256,
        "n_samples": len(samples),
        "n_modeling_samples": len(retained_indices),
        "n_excluded_samples": len(excluded_indices),
        "uq_split_counts": next(
            split["counts"] for split in splits
            if split["split_id"] == "uq_calibration_s62"
        ),
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
