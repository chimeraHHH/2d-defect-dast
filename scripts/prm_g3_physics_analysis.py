#!/usr/bin/env python3
"""Preregistered G3 materials-physics error analysis for IMP2D.

The command is deliberately fail-closed:

* it only runs on the declared research server with CUDA disabled;
* it verifies the immutable dataset and raw-database hashes;
* legacy graph predictions can only produce EXPLORATORY output;
* canonical output requires an accepted G2 manifest enumerating every repaired
  OOF prediction and hash;
* extraction and analysis are separate so the joined descriptor table can be
  audited before outcomes are modeled.

No pandas, statsmodels, patsy, seaborn, or scikit-learn dependency is used.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import pickle
import platform
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.splits import validate_split

EXPECTED_DATA_SHA256 = "1d59cc818d81252d49da525c6d77e2da8549ceb604d54af953d1caa4fb974a9b"
EXPECTED_RAW_DB_SHA256 = "3a71db999b477112da248dcf762c4384e455689953679d58b3d71a91e7148fc4"
EXPECTED_ATOM_FEATURE_SHA256 = "5fa97d7788ef9b1e10be6d874aa9c6c7d55c532d055c187bb9452659983c5bb4"
EXPECTED_HOSTNAME = "WHUServer-L40S"
EXPECTED_N = 10_224
LEGACY_WATERMARK = "EXPLORATORY — LEGACY GRAPH; NOT FOR MANUSCRIPT"
CANONICAL_LABEL = "CANONICAL — REPAIRED G2 OOF"
FEATURE_CUTOFF_A = 5.0
RDF_CUTOFF_A = 6.0
RDF_EDGES_A = np.linspace(0.0, RDF_CUTOFF_A, 13)
HUBER_DELTA = 1.345
PROFILE_GRID_POINTS = 41
ADJUSTED_PROFILE_FEATURES = ("cn_5A", "postrelaxation_min_clearance_A")
CANONICAL_PREDICTION_SEEDS = {
    "pair": {242}, "host": {242, 243, 244}, "dopant": {242, 243, 244},
}

HOST_FAMILIES = {
    "chalcogenide": {
        "Hf2Te6", "HfS2", "HfSe2", "MoS2", "MoSSe", "MoSe2", "MoTe2",
        "NbS2", "NbSe2", "NiSe2", "Pd2Se4", "PtS2", "PtSe2", "Re4S8",
        "Re4Se8", "SnS2", "SnSe2", "TaS2", "TaSe2", "TiS2", "W2Se4",
        "W2Te4", "WS2", "WSe2", "WTe2", "ZrS2", "ZrSe2",
    },
    "carbide_mxene_like": {"Mo2CO2", "Nb2CO2", "Nb4C3", "Ti2CO2", "V2CO2"},
    "elemental_hydrogenated": {"As2", "C2H2", "Ge2", "Ge2H2", "P4", "Si2", "Sn2"},
    "halide_halochalcogenide": {"Bi2I6", "BiITe", "Cr2I6", "PbI2"},
    "oxide": {"TiO2"},
}

IMPURITY_SERIES = {
    "3d": {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"},
    "4d": {"Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"},
    "5d": {"Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"},
    "main_group": {
        "Al", "As", "B", "Ba", "Be", "Bi", "Br", "C", "Ca", "Cl", "Cs",
        "F", "Ga", "Ge", "H", "I", "In", "K", "Li", "Mg", "N", "Na", "O",
        "P", "Pb", "Rb", "S", "Sb", "Se", "Si", "Sn", "Sr", "Te", "Tl", "Xe",
    },
    "f_block_other": {"Lu"},
}

E_ALIGNED_GEOMETRY = (
    "cn_5A", "neighbor_distance_mean_A", "neighbor_distance_max_A",
)
E_MODULE_ALIGNED = E_ALIGNED_GEOMETRY + ("local_abs_electronegativity_contrast",)
MODEL_EDGE_AUDIT_FEATURES = (
    "model_edge_periodic_impurity_self_image_count_5A",
    "host_only_cn_5A",
    "host_only_neighbor_distance_mean_A",
    "host_only_neighbor_distance_max_A",
    "raw_graph_defect_edge_count_delta_5A",
    "raw_graph_defect_edge_topology_match_5A",
    "raw_graph_defect_edge_distance_max_abs_delta_A",
)
INDEPENDENT_GEOMETRY = (
    "smooth_cn_5A", "neighbor_distance_std_A",
    "postrelaxation_min_clearance_A", "depth_boundary_distance",
    "abs_log_extension_factor",
)
CHEMISTRY_FEATURES = (
    "local_abs_electronegativity_contrast",
    "local_signed_covalent_radius_mismatch_A",
    "local_abs_covalent_radius_mismatch_A",
    "local_signed_electronegativity_mismatch",
    "local_signed_valence_mismatch",
    "local_abs_valence_mismatch",
)
PRIMARY_FEATURES = E_ALIGNED_GEOMETRY + INDEPENDENT_GEOMETRY + CHEMISTRY_FEATURES
SENSITIVITY_CONTROLS = (
    "natoms", "cell_area_A2", "host_slab_thickness_A",
    "train_host_support", "train_impurity_support", "abs_target_eV", "abs_conv2",
)
HOST_NOVELTY_FEATURES = (
    "host_mean_covalent_radius_A", "host_std_covalent_radius_A",
    "host_mean_electronegativity", "host_std_electronegativity",
    "host_mean_valence", "host_std_valence", "host_area_per_atom_A2",
    "host_slab_thickness_A", "host_cn_mean_5A", "host_cn_std_5A",
    *(f"host_rdf_{index:02d}" for index in range(12)),
)
IMPURITY_NOVELTY_FEATURES = (
    "impurity_covalent_radius_A", "impurity_electronegativity",
    "impurity_valence", "impurity_period", "impurity_group",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_snapshot() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()
    remote_refs = subprocess.run(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()
    remote_refs = sorted(ref.strip() for ref in remote_refs if ref.strip())
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status,
        "remote_refs": remote_refs,
    }


def require_server_preflight(args: argparse.Namespace) -> dict[str, Any]:
    hostname = socket.gethostname().split(".")[0]
    if hostname != EXPECTED_HOSTNAME:
        raise RuntimeError(
            f"G3 scientific commands may only run on {EXPECTED_HOSTNAME}; got {hostname}"
        )
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] != "":
        raise RuntimeError("G3 requires an explicitly empty CUDA_VISIBLE_DEVICES")
    git = git_snapshot()
    if git["dirty"]:
        raise RuntimeError("G3 requires a clean commit-pinned worktree")
    if not git["remote_refs"]:
        raise RuntimeError("G3 requires HEAD to be reachable from a fetched remote branch")
    runtime = {
        "host_alias": "server:WHUServer-L40S",
        "python_alias": "$PRM_PYTHON",
        "python_executable_sha256": file_sha256(Path(sys.executable)),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "git": git,
    }
    if args.record_resolved_paths:
        runtime["hostname"] = hostname
        runtime["python_executable"] = sys.executable
        runtime["platform"] = platform.platform()
    return runtime


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: {observed} != {expected}")
    return observed


def path_record(path: Path, alias: str, include_resolved: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "alias": alias,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if include_resolved:
        record["supplied_path"] = str(path)
        record["resolved_path"] = str(path.resolve())
    return record


def resolve_contract_path(value: Any) -> Path:
    """Resolve a repo-relative path or an environment alias without publishing it."""
    raw = str(value)
    expanded = os.path.expandvars(raw)
    if not raw or "$" in expanded:
        raise ValueError(f"unresolved contract path alias: {raw}")
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_canonical_seed_entries(
    regime: str, entries: Sequence[Mapping[str, Any]],
) -> None:
    if regime not in CANONICAL_PREDICTION_SEEDS:
        raise ValueError(f"unknown canonical prediction regime: {regime}")
    seeds = [int(entry.get("seed", -1)) for entry in entries]
    expected = CANONICAL_PREDICTION_SEEDS[regime]
    if len(seeds) != len(expected) or len(set(seeds)) != len(seeds) or set(seeds) != expected:
        raise ValueError(f"canonical {regime} seed ensemble changed")


def validate_canonical_initialization_fields(
    acceptance: Mapping[str, Any], corrected_asset_sha256: str,
) -> None:
    asset = acceptance.get("pretrained_asset", {})
    if (
        acceptance.get("initialization_arm") != "corrected_pretrain"
        or asset.get("sha256") != corrected_asset_sha256
        or acceptance.get("atom_feature_table_sha256") != EXPECTED_ATOM_FEATURE_SHA256
    ):
        raise ValueError("canonical G2 initialization lineage fields changed")


def pushed_ancestor(ancestor: str, descendant: str) -> bool:
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT, check=False,
    )
    remote_refs = subprocess.run(
        ["git", "branch", "-r", "--contains", ancestor], cwd=ROOT,
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    return ancestry.returncode == 0 and any(ref.strip() for ref in remote_refs)


def canonical_main_text_claim_eligible(
    evidence_tier: str, p1_gate: bool, p3_gate: bool, p4_gate: bool,
) -> bool:
    return bool(evidence_tier == "canonical" and (p1_gate or p3_gate or p4_gate))


def eligible_adjusted_profile_features(
    eligible_features: Sequence[str],
) -> tuple[str, ...]:
    selected = tuple(
        feature for feature in ADJUSTED_PROFILE_FEATURES
        if feature in set(eligible_features)
    )
    if not selected:
        raise RuntimeError(
            "neither frozen panel-a profile feature passes class-stratified coverage"
        )
    return selected


def graph_samples_from_blob(
    graph_blob: Any, required_graph_builder_version: str | None,
) -> list[Mapping[str, Any]]:
    """Validate the full repaired graph container, including list containers."""
    if isinstance(graph_blob, dict) and "data" in graph_blob:
        graph_samples = graph_blob["data"]
        container_graph_version = graph_blob.get("graph_builder", {}).get("version")
    elif isinstance(graph_blob, list):
        graph_samples = graph_blob
        container_graph_version = None
    else:
        raise TypeError("unsupported graph dataset container")
    if not isinstance(graph_samples, list) or len(graph_samples) != 10_641:
        raise ValueError("prediction graph dataset container must contain 10,641 rows")
    if required_graph_builder_version is not None:
        if (
            container_graph_version is not None
            and container_graph_version != required_graph_builder_version
        ):
            raise ValueError("canonical graph dataset container builder version mismatch")
        if any(
            not isinstance(sample, Mapping)
            or sample.get("graph_builder_version") != required_graph_builder_version
            for sample in graph_samples
        ):
            raise ValueError(
                "not all 10,641 canonical-container rows carry the repaired builder version"
            )
    return graph_samples


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def load_protocol_samples(protocol_dir: Path) -> list[dict[str, Any]]:
    expected_dir = (ROOT / "artifacts/prm_protocol_v2").resolve()
    if protocol_dir.resolve() != expected_dir:
        raise ValueError("formal G3 must use the commit-pinned repository protocol-v2 directory")
    manifest = json.loads((protocol_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != "prm_protocol_manifest_v2":
        raise ValueError("G3 requires protocol v2")
    if manifest.get("data_sha256") != EXPECTED_DATA_SHA256:
        raise ValueError("protocol data identity mismatch")
    with (protocol_dir / "samples.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for row in raw:
        if str(row["canonical_retained"]).lower() != "true":
            continue
        rows.append(
            {
                **row,
                "sample_index": int(row["sample_index"]),
                "id": int(row["id"]),
                "target_eV": float(row["target_eV"]),
                "natoms": int(row["natoms"]),
            }
        )
    if len(rows) != EXPECTED_N:
        raise ValueError(f"canonical population mismatch: {len(rows)} != {EXPECTED_N}")
    if len({row["sample_index"] for row in rows}) != EXPECTED_N:
        raise ValueError("sample indices are not unique")
    validate_frozen_taxonomies(rows)
    return sorted(rows, key=lambda row: row["sample_index"])


def invert_taxonomy(groups: Mapping[str, set[str]], kind: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label, identities in groups.items():
        for identity in identities:
            if identity in mapping:
                raise ValueError(f"duplicate {kind} taxonomy identity: {identity}")
            mapping[identity] = label
    return mapping


HOST_TO_FAMILY = invert_taxonomy(HOST_FAMILIES, "host")
DOPANT_TO_SERIES = invert_taxonomy(IMPURITY_SERIES, "impurity")


def validate_frozen_taxonomies(rows: Sequence[Mapping[str, Any]]) -> None:
    hosts = {str(row["host"]) for row in rows}
    dopants = {str(row["dopant"]) for row in rows}
    if hosts != set(HOST_TO_FAMILY):
        raise ValueError(
            f"host taxonomy mismatch; missing={sorted(hosts-set(HOST_TO_FAMILY))}, "
            f"extra={sorted(set(HOST_TO_FAMILY)-hosts)}"
        )
    if dopants != set(DOPANT_TO_SERIES):
        raise ValueError(
            f"impurity taxonomy mismatch; missing={sorted(dopants-set(DOPANT_TO_SERIES))}, "
            f"extra={sorted(set(DOPANT_TO_SERIES)-dopants)}"
        )


def load_split(protocol_dir: Path, split_id: str) -> dict[str, Any]:
    path = protocol_dir / "splits" / f"{split_id}.json"
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != "prm_split_v1"
        or payload.get("split_id") != split_id
        or int(payload.get("n_samples", -1)) != 10_641
    ):
        raise ValueError(f"split identity mismatch: {path}")
    if payload.get("data_sha256") != EXPECTED_DATA_SHA256:
        raise ValueError(f"split data identity mismatch: {path}")
    observed_counts = validate_split(payload, 10_641)
    if payload.get("counts") != observed_counts or observed_counts.get("excluded") != 417:
        raise ValueError(f"split counts/exclusions mismatch: {path}")
    if sum(observed_counts[name] for name in ("train", "val", "test")) != EXPECTED_N:
        raise ValueError(f"split canonical population mismatch: {path}")
    return payload


def fold_maps(
    protocol_dir: Path, rows_by_index: Mapping[int, Mapping[str, Any]], prefix: str,
) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
    fold_by_index: dict[int, int] = {}
    support: dict[int, tuple[int, int]] = {}
    for fold in range(5):
        split = load_split(protocol_dir, f"{prefix}_cv5_f{fold}")
        if prefix in {"host", "dopant", "pair"}:
            identity = lambda index: (
                str(rows_by_index[int(index)]["host"])
                if prefix == "host" else
                str(rows_by_index[int(index)]["dopant"])
                if prefix == "dopant" else
                f"{rows_by_index[int(index)]['host']}::{rows_by_index[int(index)]['dopant']}"
            )
            identity_sets = {
                partition: {identity(index) for index in split[partition]}
                for partition in ("train", "val", "test")
            }
            if any(
                identity_sets[left] & identity_sets[right]
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
            ):
                raise ValueError(f"{prefix} split identities overlap across partitions")
        host_counts = Counter(str(rows_by_index[int(i)]["host"]) for i in split["train"])
        dopant_counts = Counter(str(rows_by_index[int(i)]["dopant"]) for i in split["train"])
        for index in split["test"]:
            index = int(index)
            if index in fold_by_index:
                raise ValueError(f"duplicate {prefix} test index {index}")
            fold_by_index[index] = fold
            row = rows_by_index[index]
            support[index] = (
                int(host_counts[str(row["host"])]),
                int(dopant_counts[str(row["dopant"])]),
            )
    expected = set(rows_by_index)
    if set(fold_by_index) != expected:
        raise ValueError(f"{prefix} folds do not cover the canonical population")
    return fold_by_index, support


def load_prediction_archive(
    path: Path, expected_split_id: str, expected_indices: Sequence[int],
    target_by_index: Mapping[int, float], expected_hash: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    observed_hash = file_sha256(path)
    if expected_hash is not None and observed_hash != expected_hash:
        raise ValueError(f"prediction hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema_version", "split_id", "split", "indices", "preds", "targets"}
        if set(archive.files) != required:
            raise ValueError(f"prediction schema mismatch: {path}")
        if str(archive["schema_version"].item()) != "prm_predictions_v1":
            raise ValueError(f"prediction version mismatch: {path}")
        if str(archive["split_id"].item()) != expected_split_id:
            raise ValueError(f"prediction split ID mismatch: {path}")
        if str(archive["split"].item()) != "test":
            raise ValueError(f"prediction partition mismatch: {path}")
        indices = np.asarray(archive["indices"], dtype=np.int64)
        preds = np.asarray(archive["preds"], dtype=float)
        targets = np.asarray(archive["targets"], dtype=float)
    order = np.argsort(indices)
    indices, preds, targets = indices[order], preds[order], targets[order]
    if indices.tolist() != sorted(int(i) for i in expected_indices):
        raise ValueError(f"prediction/test split index mismatch: {path}")
    expected_targets = np.asarray([target_by_index[int(i)] for i in indices])
    if not np.allclose(targets, expected_targets, rtol=0.0, atol=1e-5):
        raise ValueError(f"prediction target mismatch: {path}")
    if not np.isfinite(preds).all():
        raise ValueError(f"nonfinite predictions: {path}")
    return indices, preds, observed_hash


def canonical_prediction_specs(
    acceptance_path: Path, protocol_dir: Path,
) -> tuple[dict[str, dict[int, list[dict[str, Any]]]], dict[str, Any]]:
    acceptance = json.loads(acceptance_path.read_text())
    required = {
        "schema_version": "prm_g2_acceptance_v1",
        "milestone": "J-R2-repaired-core-oof",
        "status": "accepted",
        "graph_status": "repaired_exact_mic_invariant_triplets",
        "data_sha256": EXPECTED_DATA_SHA256,
    }
    for key, value in required.items():
        if acceptance.get(key) != value:
            raise ValueError(f"canonical G2 acceptance mismatch for {key}")
    g1 = acceptance.get("g1", {})
    rebuild_spec = g1.get("repaired_dataset_receipt", {})
    final_spec = g1.get("acceptance_receipt", {})
    pretraining_spec = g1.get("corrected_pretraining_receipt", {})
    if not all(
        spec.get("path") and spec.get("sha256")
        for spec in (rebuild_spec, final_spec, pretraining_spec)
    ):
        raise ValueError("canonical G2 acceptance is not bound to all three G1 receipts")
    rebuild_path = resolve_contract_path(rebuild_spec["path"])
    final_path = resolve_contract_path(final_spec["path"])
    pretraining_path = resolve_contract_path(pretraining_spec["path"])
    if file_sha256(rebuild_path) != str(rebuild_spec["sha256"]):
        raise ValueError("G1 repaired-dataset receipt hash mismatch")
    if file_sha256(final_path) != str(final_spec["sha256"]):
        raise ValueError("G1 acceptance receipt hash mismatch")
    if file_sha256(pretraining_path) != str(pretraining_spec["sha256"]):
        raise ValueError("G1 corrected-pretraining receipt hash mismatch")
    rebuild_payload = json.loads(rebuild_path.read_text())
    if (
        rebuild_payload.get("schema_version") != "prm_g1_repaired_dataset_v1"
        or rebuild_payload.get("inputs", {}).get("dataset", {}).get("sha256") != EXPECTED_DATA_SHA256
        or rebuild_payload.get("inputs", {}).get("raw_db", {}).get("sha256") != EXPECTED_RAW_DB_SHA256
        or rebuild_payload.get("container", {}).get("rows") != 10_641
        or rebuild_payload.get("container", {}).get("canonical_rows") != EXPECTED_N
        or rebuild_payload.get("container", {}).get("excluded_rows") != 417
        or rebuild_payload.get("container", {}).get("order_preserved") is not True
        or rebuild_payload.get("container", {}).get("non_graph_fields_preserved") is not True
        or rebuild_payload.get("graph_builder", {}).get("version") != "exact_mic_invariant_triplets_v1"
        or rebuild_payload.get("graph_builder", {}).get("cutoff_A") != FEATURE_CUTOFF_A
        or not rebuild_payload.get("output", {}).get("sha256")
    ):
        raise ValueError("G1 repaired-dataset receipt content violates the frozen contract")
    g1_payload = json.loads(final_path.read_text())
    if (
        g1_payload.get("schema_version") != "prm_g1_acceptance_v1"
        or g1_payload.get("status") != "accepted"
        or g1_payload.get("graph_builder_version") != "exact_mic_invariant_triplets_v1"
        or g1_payload.get("source_data_sha256") != EXPECTED_DATA_SHA256
        or g1_payload.get("repaired_dataset_receipt_sha256") != str(rebuild_spec["sha256"])
        or g1_payload.get("repaired_dataset_sha256") != rebuild_payload["output"]["sha256"]
        or not g1_payload.get("property_tests_passed")
        or not g1_payload.get("pilot_passed")
        or not g1_payload.get("corrected_pretraining_lineage_sha256")
    ):
        raise ValueError("G1 receipt content does not satisfy the accepted repair contract")
    pretraining_payload = json.loads(pretraining_path.read_text())
    corrected_asset_sha256 = str(
        pretraining_payload.get("outputs", {}).get("corrected_asset_sha256", "")
    )
    pretraining_commit = str(pretraining_payload.get("git", {}).get("commit", ""))
    if (
        pretraining_payload.get("schema_version") != "prm_g1c_pretraining_receipt_v1"
        or pretraining_payload.get("git", {}).get("dirty") is not False
        or len(pretraining_commit) != 40
        or pretraining_payload.get("input", {}).get("rows") != 19_902
        or pretraining_payload.get("input", {}).get("graph_builder_version")
        != "exact_mic_invariant_triplets_v1"
        or pretraining_payload.get("input", {}).get("dataset_receipt_sha256")
        != g1_payload.get("corrected_jarvis_dataset_receipt_sha256")
        or len(str(pretraining_payload.get("input", {}).get("dataset_sha256", ""))) != 64
        or len(corrected_asset_sha256) != 64
        or pretraining_payload.get("training_contract", {}).get("seed") != 42
        or pretraining_payload.get("training_contract", {}).get("epochs") != 30
    ):
        raise ValueError("G1 corrected-pretraining receipt content violates the frozen contract")
    repaired = acceptance.get("repaired_dataset", {})
    if (
        repaired.get("source_data_sha256") != EXPECTED_DATA_SHA256
        or repaired.get("graph_dataset_sha256") != rebuild_payload["output"]["sha256"]
        or not repaired.get("path")
    ):
        raise ValueError("canonical G2 acceptance lacks repaired-dataset identity")
    repaired_path = resolve_contract_path(repaired["path"])
    if file_sha256(repaired_path) != repaired["graph_dataset_sha256"]:
        raise ValueError("canonical G2 repaired graph dataset hash mismatch")
    code_commit = str(acceptance.get("code_commit", ""))
    recipe_sha256 = str(acceptance.get("model_recipe_sha256", ""))
    lineage_sha256 = str(acceptance.get("pretrained_asset_lineage_sha256", ""))
    accepted_asset_spec = acceptance.get("pretrained_asset", {})
    accepted_asset_sha256 = str(accepted_asset_spec.get("sha256", ""))
    if (
        len(code_commit) != 40
        or len(recipe_sha256) != 64
        or len(lineage_sha256) != 64
        or len(accepted_asset_sha256) != 64
    ):
        raise ValueError("canonical G2 acceptance lacks code/recipe/pretraining-lineage identity")
    validate_canonical_initialization_fields(acceptance, corrected_asset_sha256)
    accepted_asset_path = resolve_contract_path(accepted_asset_spec.get("path", ""))
    if (
        not accepted_asset_spec.get("path")
        or not accepted_asset_path.is_file()
        or file_sha256(accepted_asset_path) != accepted_asset_sha256
    ):
        raise ValueError("canonical G2 acceptance corrected asset is missing or hash-invalid")
    g1_commit = str(g1_payload.get("code_commit", ""))
    if len(g1_commit) != 40:
        raise ValueError("G1 receipt has no repair commit")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", g1_commit, code_commit],
        cwd=ROOT, check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("the accepted G1 repair commit is not an ancestor of G2")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if not pushed_ancestor(code_commit, current_commit):
        raise ValueError("G2 commit is not a pushed ancestor of the current G3 commit")
    pretraining_ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", pretraining_commit, g1_commit],
        cwd=ROOT, check=False,
    )
    if pretraining_ancestry.returncode != 0:
        raise ValueError("the corrected-pretraining commit is not an ancestor of G1 acceptance")
    if (
        g1_payload["corrected_pretraining_lineage_sha256"] != lineage_sha256
        or str(pretraining_spec["sha256"]) != lineage_sha256
        or accepted_asset_sha256 != corrected_asset_sha256
    ):
        raise ValueError("G1 and G2 corrected-pretraining lineage differ")
    source_root = acceptance.get("prediction_sources")
    if not isinstance(source_root, dict):
        raise ValueError("G2 acceptance has no prediction_sources")
    specs: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for regime in ("pair", "host", "dopant"):
        folds = source_root.get(f"{regime}_cv")
        if not isinstance(folds, dict) or set(folds) != {str(i) for i in range(5)}:
            raise ValueError(f"G2 acceptance {regime} folds are incomplete")
        specs[regime] = {}
        for fold in range(5):
            entries = folds[str(fold)]
            split_id = f"{regime}_cv5_f{fold}"
            split_path = protocol_dir / "splits" / f"{split_id}.json"
            split_payload = load_split(protocol_dir, split_id)
            split_sha256 = file_sha256(split_path)
            if not isinstance(entries, list):
                raise ValueError(f"invalid G2 source list for {regime} fold {fold}")
            validate_canonical_seed_entries(regime, entries)
            specs[regime][fold] = []
            for entry in entries:
                if entry.get("split_sha256") != split_sha256:
                    raise ValueError("G2 acceptance source is not bound to the repo split")
                path = resolve_contract_path(entry["prediction_path"])
                run_manifest_path = resolve_contract_path(entry["run_manifest_path"])
                if file_sha256(run_manifest_path) != str(entry["run_manifest_sha256"]):
                    raise ValueError("G2 run-manifest hash mismatch")
                run_manifest = json.loads(run_manifest_path.read_text())
                entry_seed = int(entry["seed"])
                if (
                    run_manifest.get("schema_version") != "prm_g2_run_manifest_v1"
                    or run_manifest.get("status") != "complete"
                    or run_manifest.get("git", {}).get("dirty") is not False
                    or run_manifest.get("split_id") != split_id
                    or run_manifest.get("split_sha256") != split_sha256
                    or run_manifest.get("split_counts") != split_payload["counts"]
                    or int(run_manifest.get("seed", -1)) != entry_seed
                ):
                    raise ValueError("G2 source run is incomplete or dirty")
                if run_manifest.get("git", {}).get("commit") != code_commit:
                    raise ValueError("G2 source run commit mismatch")
                if run_manifest.get("data", {}).get("data_sha256") != repaired["graph_dataset_sha256"]:
                    raise ValueError("G2 source run repaired-data mismatch")
                if run_manifest.get("model_recipe_sha256") != recipe_sha256:
                    raise ValueError("G2 source run model-recipe mismatch")
                if run_manifest.get("graph_builder_version") != g1_payload["graph_builder_version"]:
                    raise ValueError("G2 source run graph-builder mismatch")
                if run_manifest.get("atom_feature_table_sha256") != EXPECTED_ATOM_FEATURE_SHA256:
                    raise ValueError("G2 source run atom-feature-table mismatch")
                if run_manifest.get("pretrained_asset_lineage_sha256") != lineage_sha256:
                    raise ValueError("G2 source run pretraining-asset lineage mismatch")
                if (
                    run_manifest.get("initialization_arm") != "corrected_pretrain"
                    or run_manifest.get("pretrained_asset_sha256") != corrected_asset_sha256
                    or run_manifest.get("outputs", {}).get("prediction_sha256")
                    != str(entry["prediction_sha256"])
                ):
                    raise ValueError("G2 source run is not bound to the corrected initialization asset")
                specs[regime][fold].append(
                    {"path": path, "sha256": str(entry["prediction_sha256"]), "seed": entry_seed}
                )
    return specs, acceptance


def exploratory_prediction_specs(
    prediction_root: Path,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    seeds = {"pair": (242,), "host": (242, 243, 244), "dopant": (242, 243, 244)}
    specs: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for regime in seeds:
        specs[regime] = {}
        for fold in range(5):
            specs[regime][fold] = [
                {
                    "path": prediction_root / f"{regime}_cv5_f{fold}" / f"seed{seed}" / "test_predictions.npz",
                    "sha256": None,
                    "seed": seed,
                }
                for seed in seeds[regime]
            ]
    return specs


def load_all_oof_predictions(
    specs: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    protocol_dir: Path, target_by_index: Mapping[int, float],
) -> tuple[dict[str, dict[int, float]], list[dict[str, Any]]]:
    all_predictions: dict[str, dict[int, float]] = {}
    sources: list[dict[str, Any]] = []
    for regime in ("pair", "host", "dopant"):
        pooled: dict[int, float] = {}
        for fold in range(5):
            split_id = f"{regime}_cv5_f{fold}"
            split = load_split(protocol_dir, split_id)
            expected_indices = sorted(int(i) for i in split["test"])
            seed_predictions = []
            reference_indices: np.ndarray | None = None
            for spec in specs[regime][fold]:
                path = Path(spec["path"])
                indices, values, observed_hash = load_prediction_archive(
                    path, split_id, expected_indices, target_by_index,
                    expected_hash=spec.get("sha256"),
                )
                if reference_indices is None:
                    reference_indices = indices
                elif not np.array_equal(reference_indices, indices):
                    raise ValueError(f"seed alignment mismatch for {split_id}")
                seed_predictions.append(values)
                sources.append(
                    {
                        "regime": regime,
                        "fold": fold,
                        "seed": int(spec["seed"]),
                        "path_alias": f"prediction_sources/{regime}/fold{fold}/seed{int(spec['seed'])}",
                        "sha256": observed_hash,
                    }
                )
            assert reference_indices is not None
            mean_prediction = np.mean(np.stack(seed_predictions), axis=0)
            for index, value in zip(reference_indices, mean_prediction):
                if int(index) in pooled:
                    raise ValueError(f"duplicate {regime} OOF sample {index}")
                pooled[int(index)] = float(value)
        if set(pooled) != set(target_by_index):
            raise ValueError(f"{regime} OOF does not cover the canonical population")
        all_predictions[regime] = pooled
    return all_predictions, sources


@functools.lru_cache(maxsize=1)
def load_raw_element_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from ase.data import covalent_radii
    from src.features import GROUP, PAULING_EN, get_atom_feature_table

    radius = np.asarray(covalent_radii, dtype=float)
    raw_en = np.asarray(PAULING_EN, dtype=float)
    group = np.asarray(GROUP, dtype=float)
    valence = group.copy()
    valence[group >= 13] = group[group >= 13] - 10
    model_feature_path = ROOT / "data/atom_features_ref.pth"
    require_hash(
        model_feature_path,
        EXPECTED_ATOM_FEATURE_SHA256,
        "reference atom-feature table",
    )
    model_en = np.asarray(
        get_atom_feature_table(model_feature_path)[:, 2].detach().cpu().numpy(),
        dtype=float,
    )
    raw_en[raw_en == 0.0] = np.nan
    radius[~np.isfinite(radius)] = np.nan
    if len(model_en) <= 100 or not np.isfinite(model_en[1:101]).all():
        raise ValueError("reference atom-feature electronegativity column is invalid")
    return radius, raw_en, valence, model_en


def local_structure_descriptors(
    atoms: Any, dopant: str, graph_sample: Mapping[str, Any],
) -> dict[str, float]:
    from ase.data import atomic_numbers
    from ase.neighborlist import neighbor_list

    radius, electronegativity, valence, model_en = load_raw_element_tables()
    numbers = np.asarray(atoms.numbers, dtype=int)
    graph_numbers = np.asarray(graph_sample.get("numbers"), dtype=int)
    if not np.array_equal(graph_numbers, numbers):
        raise ValueError("graph/raw atomic-number order mismatch")
    dopant_z = int(atomic_numbers[dopant])
    candidates = np.flatnonzero(numbers == dopant_z)
    if len(candidates) != 1:
        raise ValueError(f"raw structure has {len(candidates)} atoms matching impurity {dopant}")
    defect = int(candidates[0])
    graph_edge_index = np.asarray(graph_sample.get("edge_index"), dtype=int)
    model_distances = np.asarray(graph_sample.get("edge_dist"), dtype=float)
    if (
        graph_edge_index.ndim != 2
        or graph_edge_index.shape[0] != 2
        or graph_edge_index.shape[1] != len(model_distances)
        or not np.isfinite(model_distances).all()
        or np.any(model_distances < 0)
        or np.any(model_distances > FEATURE_CUTOFF_A + 1e-5)
    ):
        raise ValueError("graph sample edge schema/cutoff mismatch")
    model_i, model_j = graph_edge_index
    model_local = model_i == defect
    model_neighbor_indices = model_j[model_local]
    model_d = model_distances[model_local]
    if len(model_d) == 0:
        raise ValueError("unique impurity has no model-edge neighbors within 5 Angstrom")
    model_neighbor_z = numbers[model_neighbor_indices]
    model_en_contrast = abs(
        float(model_en[dopant_z]) - float(np.mean(model_en[model_neighbor_z]))
    )

    # Physical descriptors are intentionally recomputed from the raw ASE
    # structure.  The graph E features above instead come from the exact graph
    # sample consumed by the corresponding prediction.
    physical_i, physical_j, physical_distances = neighbor_list(
        "ijd", atoms, FEATURE_CUTOFF_A
    )
    physical_i = np.asarray(physical_i, dtype=int)
    physical_j = np.asarray(physical_j, dtype=int)
    physical_distances = np.asarray(physical_distances, dtype=float)
    raw_model_local = physical_i == defect
    host_only_local = raw_model_local & (physical_j != defect)
    neighbor_indices = physical_j[host_only_local]
    d = physical_distances[host_only_local]
    if len(d) == 0:
        raise ValueError("unique impurity has no host-only neighbors within 5 Angstrom")
    neighbor_z = numbers[neighbor_indices]
    r_imp, en_imp, val_imp = radius[dopant_z], electronegativity[dopant_z], valence[dopant_z]
    r_neigh, en_neigh, val_neigh = radius[neighbor_z], electronegativity[neighbor_z], valence[neighbor_z]
    signed_r = r_imp - float(np.mean(r_neigh)) if np.isfinite(r_imp) and np.isfinite(r_neigh).all() else float("nan")
    signed_en = en_imp - float(np.mean(en_neigh)) if np.isfinite(en_imp) and np.isfinite(en_neigh).all() else float("nan")
    signed_val = val_imp - float(np.mean(val_neigh)) if np.isfinite(val_imp) and np.isfinite(val_neigh).all() else float("nan")
    clearance = (
        float(np.min(d - r_imp - r_neigh))
        if np.isfinite(r_imp) and np.isfinite(r_neigh).all() else float("nan")
    )

    host_mask = np.ones(len(numbers), dtype=bool)
    host_mask[defect] = False
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    normal = np.cross(cell[0], cell[1])
    area = float(np.linalg.norm(normal))
    if area <= 1e-12:
        raise ValueError("degenerate in-plane cell")
    normal /= area
    host_projection = positions[host_mask] @ normal
    slab_thickness = float(np.max(host_projection) - np.min(host_projection))

    rdf_i, rdf_j, rdf_distances = neighbor_list("ijd", atoms, RDF_CUTOFF_A)
    rdf_i = np.asarray(rdf_i, dtype=int)
    rdf_j = np.asarray(rdf_j, dtype=int)
    rdf_distances = np.asarray(rdf_distances, dtype=float)
    host_edges = host_mask[rdf_i] & host_mask[rdf_j]
    rdf_counts, _ = np.histogram(rdf_distances[host_edges], bins=RDF_EDGES_A)
    host_count = int(host_mask.sum())
    rdf = rdf_counts.astype(float) / max(host_count, 1)
    host_local = host_mask[physical_i] & host_mask[physical_j]
    host_cn = np.bincount(
        physical_i[host_local], minlength=len(numbers)
    )[host_mask].astype(float)
    host_z = numbers[host_mask]

    graph_order = np.lexsort((model_distances[model_local], model_neighbor_indices))
    raw_neighbor_indices = physical_j[raw_model_local]
    raw_neighbor_distances = physical_distances[raw_model_local]
    raw_order = np.lexsort((raw_neighbor_distances, raw_neighbor_indices))
    edge_topology_match = (
        len(model_neighbor_indices) == len(raw_neighbor_indices)
        and np.array_equal(
            model_neighbor_indices[graph_order], raw_neighbor_indices[raw_order]
        )
    )
    raw_graph_max_delta = (
        float(np.max(np.abs(
            model_d[graph_order] - raw_neighbor_distances[raw_order]
        )))
        if edge_topology_match and len(model_d) else float("nan")
    )

    result = {
        "cn_5A": float(len(model_d)),
        "neighbor_distance_mean_A": float(np.mean(model_d)),
        "neighbor_distance_max_A": float(np.max(model_d)),
        "local_abs_electronegativity_contrast": model_en_contrast,
        "model_edge_periodic_impurity_self_image_count_5A": float(
            np.count_nonzero(model_neighbor_indices == defect)
        ),
        "host_only_cn_5A": float(len(d)),
        "host_only_neighbor_distance_mean_A": float(np.mean(d)),
        "host_only_neighbor_distance_max_A": float(np.max(d)),
        "raw_graph_defect_edge_count_delta_5A": float(
            len(raw_neighbor_indices) - len(model_neighbor_indices)
        ),
        "raw_graph_defect_edge_topology_match_5A": float(edge_topology_match),
        "raw_graph_defect_edge_distance_max_abs_delta_A": raw_graph_max_delta,
        "smooth_cn_5A": float(np.sum(0.5 * (np.cos(np.pi * d / FEATURE_CUTOFF_A) + 1.0))),
        "neighbor_distance_std_A": float(np.std(d, ddof=0)),
        "postrelaxation_min_clearance_A": clearance,
        "local_signed_covalent_radius_mismatch_A": signed_r,
        "local_abs_covalent_radius_mismatch_A": abs(signed_r),
        "local_signed_electronegativity_mismatch": signed_en,
        "local_signed_valence_mismatch": signed_val,
        "local_abs_valence_mismatch": abs(signed_val),
        "cell_area_A2": area,
        "host_slab_thickness_A": slab_thickness,
        "host_atom_count": host_count,
        "host_area_per_atom_A2": area / max(host_count, 1),
        "host_mean_covalent_radius_A": float(np.nanmean(radius[host_z])),
        "host_std_covalent_radius_A": float(np.nanstd(radius[host_z])),
        "host_mean_electronegativity": float(np.nanmean(electronegativity[host_z])),
        "host_std_electronegativity": float(np.nanstd(electronegativity[host_z])),
        "host_mean_valence": float(np.nanmean(valence[host_z])),
        "host_std_valence": float(np.nanstd(valence[host_z])),
        "host_cn_mean_5A": float(np.mean(host_cn)),
        "host_cn_std_5A": float(np.std(host_cn, ddof=0)),
        "impurity_covalent_radius_A": float(radius[dopant_z]),
        "impurity_electronegativity": float(electronegativity[dopant_z]),
        "impurity_valence": float(valence[dopant_z]),
    }
    for index, value in enumerate(rdf):
        result[f"host_rdf_{index:02d}"] = float(value)
    return result


def build_joined_rows(
    raw_db: Path,
    graph_data_path: Path,
    graph_data_sha256: str,
    required_graph_builder_version: str | None,
    rows: Sequence[Mapping[str, Any]],
    pair_fold: Mapping[int, int],
    host_fold: Mapping[int, int],
    dopant_fold: Mapping[int, int],
    pair_support: Mapping[int, tuple[int, int]],
    predictions: Mapping[str, Mapping[int, float]],
    evidence_tier: str,
    evidence_label: str,
    model_status: str,
    report_progress: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Reconstruct the complete joined table from immutable source objects."""
    from ase.data import atomic_numbers
    from ase.db import connect
    from src.features import GROUP, PERIOD

    require_hash(graph_data_path, graph_data_sha256, "prediction graph dataset")
    with graph_data_path.open("rb") as handle:
        graph_blob = pickle.load(handle)
    graph_samples = graph_samples_from_blob(
        graph_blob, required_graph_builder_version,
    )

    database = connect(str(raw_db))
    joined: list[dict[str, Any]] = []
    depth_mismatches = 0
    for position, sample in enumerate(rows, start=1):
        index = int(sample["sample_index"])
        graph_sample = graph_samples[index]
        if (
            int(graph_sample.get("id", -1)) != int(sample["id"])
            or (
                required_graph_builder_version is not None
                and graph_sample.get("graph_builder_version")
                != required_graph_builder_version
            )
        ):
            raise ValueError(f"graph/protocol identity or builder mismatch for sample {index}")
        raw_row = database.get(id=int(sample["id"]))
        for key in ("host", "dopant", "defecttype", "site"):
            if str(raw_row.get(key, "")) != str(sample[key]):
                raise ValueError(f"raw/protocol {key} mismatch for sample {index}")
        if abs(float(raw_row.get("eform")) - float(sample["target_eV"])) > 1e-10:
            raise ValueError(f"raw/protocol target mismatch for sample {index}")
        descriptor = local_structure_descriptors(
            raw_row.toatoms(), str(sample["dopant"]), graph_sample
        )
        depth = finite_float(raw_row.get("depth"))
        extension_factor = finite_float(raw_row.get("extension_factor"))
        conv2 = finite_float(raw_row.get("conv2"))
        en1 = finite_float(raw_row.get("en1"))
        en2 = finite_float(raw_row.get("en2"))
        depth_margin = abs(depth) - 1.0 if math.isfinite(depth) else float("nan")
        depth_boundary_distance = abs(depth_margin) if math.isfinite(depth_margin) else float("nan")
        expected_class = (
            "interstitial" if math.isfinite(depth_margin) and depth_margin < 0 else
            "adsorbate" if math.isfinite(depth_margin) else "unknown"
        )
        depth_consistent = (
            int(expected_class == str(sample["defecttype"]))
            if expected_class != "unknown" else ""
        )
        if depth_consistent == 0:
            depth_mismatches += 1
        xf_log = (
            abs(math.log(extension_factor))
            if math.isfinite(extension_factor) and extension_factor > 0
            else float("nan")
        )
        z = int(atomic_numbers[str(sample["dopant"])])
        pred_pair = float(predictions["pair"][index])
        pred_host = float(predictions["host"][index])
        pred_dopant = float(predictions["dopant"][index])
        target = float(sample["target_eV"])
        joined.append(
            {
                "evidence_tier": evidence_tier,
                "evidence_label": evidence_label,
                "model_status": model_status,
                "sample_index": index,
                "raw_row_id": int(sample["id"]),
                "unique_id": sample["unique_id"],
                "host": sample["host"],
                "host_family": HOST_TO_FAMILY[str(sample["host"])],
                "dopant": sample["dopant"],
                "impurity_series": DOPANT_TO_SERIES[str(sample["dopant"])],
                "defecttype": sample["defecttype"],
                "site": sample["site"],
                "pair_fold": pair_fold[index],
                "host_fold": host_fold[index],
                "dopant_fold": dopant_fold[index],
                "target_eV": target,
                "abs_target_eV": abs(target),
                "pair_prediction_eV": pred_pair,
                "pair_residual_eV": pred_pair - target,
                "pair_absolute_error_eV": abs(pred_pair - target),
                "host_prediction_eV": pred_host,
                "host_absolute_error_eV": abs(pred_host - target),
                "dopant_prediction_eV": pred_dopant,
                "dopant_absolute_error_eV": abs(pred_dopant - target),
                "host_minus_dopant_absolute_error_eV": (
                    abs(pred_host - target) - abs(pred_dopant - target)
                ),
                "natoms": int(sample["natoms"]),
                "spacegroup": sample["spacegroup"],
                "supercell": sample["supercell"],
                "train_host_support": pair_support[index][0],
                "train_impurity_support": pair_support[index][1],
                "depth": depth,
                "depth_class_margin": depth_margin,
                "depth_boundary_distance": depth_boundary_distance,
                "depth_class_consistent": depth_consistent,
                "extension_factor": extension_factor,
                "abs_log_extension_factor": xf_log,
                "xf_pathology": (
                    int(extension_factor > 2.0) if math.isfinite(extension_factor) else ""
                ),
                "conv2": conv2,
                "abs_conv2": abs(conv2) if math.isfinite(conv2) else float("nan"),
                "en1": en1,
                "en2": en2,
                "impurity_period": int(PERIOD[z]),
                "impurity_group": int(GROUP[z]),
                **descriptor,
            }
        )
        if report_progress and position % 500 == 0:
            print(f"reconstructed {position}/{EXPECTED_N}", flush=True)
    return joined, depth_mismatches


def assert_joined_rows_equal(
    observed_rows: Sequence[Mapping[str, str]],
    expected_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed if a CSV row differs from source-reconstructed content."""
    if len(observed_rows) != EXPECTED_N or len(expected_rows) != EXPECTED_N:
        raise ValueError("joined/source population mismatch")
    observed_by_index: dict[int, Mapping[str, str]] = {}
    for row in observed_rows:
        index = int(row.get("sample_index", "-1"))
        if index in observed_by_index:
            raise ValueError(f"duplicate joined sample_index {index}")
        observed_by_index[index] = row
    expected_by_index = {int(row["sample_index"]): row for row in expected_rows}
    if set(observed_by_index) != set(expected_by_index):
        raise ValueError("joined sample_index set differs from the immutable protocol")
    for index, expected in expected_by_index.items():
        observed = observed_by_index[index]
        if set(observed) != set(expected):
            raise ValueError(f"joined columns differ from reconstructed schema at sample {index}")
        for key, expected_value in expected.items():
            observed_value = observed[key]
            if isinstance(expected_value, (int, float, np.integer, np.floating)):
                expected_float = float(expected_value)
                observed_float = finite_float(observed_value)
                if math.isnan(expected_float):
                    matches = math.isnan(observed_float)
                else:
                    matches = math.isclose(
                        observed_float, expected_float, rel_tol=0.0, abs_tol=1e-12
                    )
            else:
                expected_text = "" if expected_value is None else str(expected_value)
                matches = str(observed_value) == expected_text
            if not matches:
                raise ValueError(
                    f"joined source reconstruction mismatch at sample {index}, field {key}"
                )


def extract(args: argparse.Namespace) -> None:
    runtime = require_server_preflight(args)
    data_path, raw_db, protocol_dir = Path(args.data_path), Path(args.raw_db), Path(args.protocol_dir)
    require_hash(data_path, EXPECTED_DATA_SHA256, "cleaned dataset")
    require_hash(raw_db, EXPECTED_RAW_DB_SHA256, "raw IMP2D database")
    rows = load_protocol_samples(protocol_dir)
    rows_by_index = {int(row["sample_index"]): row for row in rows}
    target_by_index = {int(row["sample_index"]): float(row["target_eV"]) for row in rows}
    pair_fold, pair_support = fold_maps(protocol_dir, rows_by_index, "pair")
    host_fold, _ = fold_maps(protocol_dir, rows_by_index, "host")
    dopant_fold, _ = fold_maps(protocol_dir, rows_by_index, "dopant")

    if args.evidence_tier == "canonical":
        if args.model_status != "repaired_g1_g2" or not args.g2_acceptance:
            raise ValueError("canonical G3 requires repaired_g1_g2 and --g2-acceptance")
        specs, acceptance = canonical_prediction_specs(Path(args.g2_acceptance), protocol_dir)
        evidence_label = CANONICAL_LABEL
        acceptance_hash = file_sha256(Path(args.g2_acceptance))
        graph_data_path = resolve_contract_path(
            acceptance["repaired_dataset"]["path"]
        )
        graph_data_sha256 = str(
            acceptance["repaired_dataset"]["graph_dataset_sha256"]
        )
        required_graph_builder_version = "exact_mic_invariant_triplets_v1"
    else:
        if args.model_status != "legacy_order_sensitive_graph":
            raise ValueError("exploratory legacy G3 requires legacy_order_sensitive_graph")
        specs = exploratory_prediction_specs(Path(args.prediction_root))
        acceptance = None
        acceptance_hash = None
        evidence_label = LEGACY_WATERMARK
        graph_data_path = data_path
        graph_data_sha256 = EXPECTED_DATA_SHA256
        required_graph_builder_version = None
    predictions, prediction_sources = load_all_oof_predictions(specs, protocol_dir, target_by_index)

    output_dir = Path(args.output_dir).resolve()
    try:
        output_dir.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("formal G3 output must be outside the repository and paper-facing trees")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    joined, depth_mismatches = build_joined_rows(
        raw_db=raw_db,
        graph_data_path=graph_data_path,
        graph_data_sha256=graph_data_sha256,
        required_graph_builder_version=required_graph_builder_version,
        rows=rows,
        pair_fold=pair_fold,
        host_fold=host_fold,
        dopant_fold=dopant_fold,
        pair_support=pair_support,
        predictions=predictions,
        evidence_tier=args.evidence_tier,
        evidence_label=evidence_label,
        model_status=args.model_status,
    )

    fields = list(joined[0])
    joined_path = output_dir / "joined_descriptors.csv"
    write_csv(joined_path, joined, fields)
    manifest = {
        "schema_version": "prm_g3_extraction_manifest_v1",
        "created_at": utc_now(),
        "evidence_tier": args.evidence_tier,
        "evidence_label": evidence_label,
        "model_status": args.model_status,
        "runtime": runtime,
        "inputs": {
            "dataset": path_record(data_path, "$PRM_DATASET_PATH", args.record_resolved_paths),
            "raw_database": path_record(raw_db, "$PRM_RAW_DB_PATH", args.record_resolved_paths),
            "prediction_graph_dataset": path_record(
                graph_data_path,
                "$PRM_REPAIRED_DATASET_PATH"
                if args.evidence_tier == "canonical" else "$PRM_DATASET_PATH",
                args.record_resolved_paths,
            ),
            "protocol_manifest_sha256": file_sha256(protocol_dir / "manifest.json"),
            "preregistration_sha256": file_sha256(ROOT / "paper_Q1/review/g3_preregistration.json"),
            "g2_acceptance_sha256": acceptance_hash,
            "prediction_sources": prediction_sources,
        },
        "population": {
            "n": len(joined),
            "depth_class_mismatches": depth_mismatches,
            "taxonomies_exact": True,
            "rows_with_periodic_impurity_self_images_5A": sum(
                float(row["model_edge_periodic_impurity_self_image_count_5A"]) > 0
                for row in joined
            ),
            "max_periodic_impurity_self_images_5A": int(max(
                float(row["model_edge_periodic_impurity_self_image_count_5A"])
                for row in joined
            )),
            "raw_graph_defect_edge_topology_mismatch_rows_5A": sum(
                float(row["raw_graph_defect_edge_topology_match_5A"]) == 0
                for row in joined
            ),
            "raw_graph_defect_edge_count_mismatch_rows_5A": sum(
                float(row["raw_graph_defect_edge_count_delta_5A"]) != 0
                for row in joined
            ),
            "raw_graph_defect_edge_distance_max_abs_delta_A": max(
                (
                    float(row["raw_graph_defect_edge_distance_max_abs_delta_A"])
                    for row in joined
                    if math.isfinite(float(row["raw_graph_defect_edge_distance_max_abs_delta_A"]))
                ),
                default=None,
            ),
        },
        "descriptor_contract": {
            "cutoff_A": FEATURE_CUTOFF_A,
            "rdf_edges_A": RDF_EDGES_A.tolist(),
            "e_module_neighbor_semantics": (
                "exact directed graph edges, including nonzero periodic impurity self-images"
            ),
            "physical_neighbor_semantics": (
                "host-only subset used for smooth coordination, clearance, and chemistry"
            ),
            "atom_feature_table_sha256": EXPECTED_ATOM_FEATURE_SHA256,
            "source": "paper_Q1/review/g3_preregistration.json",
        },
        "outputs": {
            "joined_descriptors": {
                "path": "joined_descriptors.csv",
                "sha256": file_sha256(joined_path),
                "n_rows": len(joined),
            }
        },
        "canonical_acceptance_embedded": acceptance if args.record_resolved_paths else None,
    }
    (output_dir / "extraction_manifest.json").write_text(strict_json(manifest))
    (output_dir / "status.json").write_text(
        strict_json(
            {
                "schema_version": "prm_g3_run_status_v1",
                "state": "extracted_not_analyzed",
                "evidence_tier": args.evidence_tier,
                "evidence_label": evidence_label,
                "canonical_main_text_eligible": False,
                "reason": "analysis gate has not run" if args.evidence_tier == "canonical" else LEGACY_WATERMARK,
            }
        )
    )
    print(f"wrote {joined_path}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        raise ValueError("weighted quantile has no finite positive-weight rows")
    order = np.argsort(values[mask], kind="mergesort")
    sorted_values = values[mask][order]
    sorted_weights = weights[mask][order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= sorted_weights.sum()
    return float(np.interp(quantile, cumulative, sorted_values))


def group_codes(values: Sequence[Any]) -> np.ndarray:
    labels = {value: index for index, value in enumerate(sorted(set(values), key=str))}
    return np.asarray([labels[value] for value in values], dtype=np.int64)


def alternating_projection(
    matrix: np.ndarray, groups: Sequence[np.ndarray], weights: np.ndarray,
    tolerance: float = 1e-10, max_iterations: int = 200,
) -> np.ndarray:
    """Weighted within-transform for crossed fixed effects."""
    weights = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(weights) or weights.sum() <= 0:
        raise ValueError("invalid alternating-projection inputs")
    residual = matrix.astype(float, copy=True)
    residual -= np.average(residual, axis=0, weights=weights)
    for _ in range(max_iterations):
        previous = residual.copy()
        for codes in groups:
            n_groups = int(codes.max()) + 1
            denominator = np.bincount(codes, weights=weights, minlength=n_groups)
            for column in range(residual.shape[1]):
                numerator = np.bincount(
                    codes, weights=weights * residual[:, column], minlength=n_groups,
                )
                means = np.divide(
                    numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0,
                )
                residual[:, column] -= means[codes]
        if np.max(np.abs(residual - previous)) < tolerance:
            return residual
    raise RuntimeError("crossed fixed-effect projection did not converge")


def fit_huber_fixed_effects(
    y: np.ndarray, design: np.ndarray, groups: Sequence[np.ndarray],
    base_weights: np.ndarray | None = None, max_iterations: int = 100,
) -> dict[str, Any]:
    """Huber IRLS with fixed-effect projection repeated inside each iteration."""
    n = len(y)
    base = np.ones(n, dtype=float) if base_weights is None else np.asarray(base_weights, dtype=float)
    if y.shape != (n,) or design.shape[0] != n or np.any(base < 0) or base.sum() <= 0:
        raise ValueError("invalid robust fixed-effect inputs")
    robust = np.ones(n, dtype=float)
    beta = np.zeros(design.shape[1], dtype=float)
    for iteration in range(max_iterations):
        total = base * robust
        projected = alternating_projection(np.column_stack([y, design]), groups, total)
        y_within, x_within = projected[:, 0], projected[:, 1:]
        sqrt_w = np.sqrt(total)
        beta_new = np.linalg.lstsq(
            x_within * sqrt_w[:, None], y_within * sqrt_w, rcond=1e-10,
        )[0]
        residual = y_within - x_within @ beta_new
        center = weighted_quantile(residual, 0.5, total)
        mad = weighted_quantile(np.abs(residual - center), 0.5, total)
        scale = max(mad / 0.6744897501960817, 1e-8)
        standardized = np.abs(residual) / scale
        robust_new = np.minimum(1.0, HUBER_DELTA / np.maximum(standardized, 1e-12))
        robust_new[base == 0] = 1.0
        if np.max(np.abs(beta_new - beta)) < 1e-9 and np.max(np.abs(robust_new - robust)) < 1e-7:
            beta, robust = beta_new, robust_new
            break
        beta, robust = beta_new, robust_new
    else:
        raise RuntimeError("Huber fixed-effect IRLS did not converge")
    total = base * robust
    projected = alternating_projection(np.column_stack([y, design]), groups, total)
    y_within, x_within = projected[:, 0], projected[:, 1:]
    residual = y_within - x_within @ beta
    mad = weighted_quantile(np.abs(residual - weighted_quantile(residual, 0.5, total)), 0.5, total)
    scale = max(mad / 0.6744897501960817, 1e-8)
    return {
        "beta": beta,
        "scale": scale,
        "robust_weights": robust,
        "base_weights": base,
        "x_within": x_within,
        "residual_within": residual,
        "y_mean": float(np.average(y, weights=total)),
        "x_mean": np.average(design, axis=0, weights=total),
        "iterations": iteration + 1,
    }


def build_design(
    numeric: Mapping[str, np.ndarray], defecttype: np.ndarray,
    feature_names: Sequence[str], scaling: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, dict[str, float]]]:
    is_interstitial = (defecttype == "interstitial").astype(float)
    learned: dict[str, dict[str, float]] = {}
    scaled_columns = []
    for feature in feature_names:
        values = np.asarray(numeric[feature], dtype=float)
        if scaling is None:
            median = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            iqr = float(q75 - q25)
            if iqr <= 1e-12:
                raise ValueError(f"zero-IQR analysis feature: {feature}")
            learned[feature] = {"median": median, "iqr": iqr}
        else:
            learned[feature] = dict(scaling[feature])
        scaled_columns.append((values - learned[feature]["median"]) / learned[feature]["iqr"])
    scaled = np.column_stack(scaled_columns)
    design = np.column_stack([is_interstitial, scaled, scaled * is_interstitial[:, None]])
    names = (
        ["class_interstitial"]
        + [f"main:{feature}" for feature in feature_names]
        + [f"interaction_interstitial:{feature}" for feature in feature_names]
    )
    return design, names, learned


def profile_design(
    feature_names: Sequence[str], scaling: Mapping[str, Mapping[str, float]],
    class_medians: Mapping[str, Mapping[str, float]], defecttype: str,
    focal_feature: str, focal_value: float,
) -> np.ndarray:
    interstitial = float(defecttype == "interstitial")
    raw = dict(class_medians[defecttype])
    raw[focal_feature] = focal_value
    scaled = np.asarray(
        [(raw[name] - scaling[name]["median"]) / scaling[name]["iqr"] for name in feature_names],
        dtype=float,
    )
    return np.concatenate([[interstitial], scaled, scaled * interstitial])


def backtransformed_contrast(
    fit: Mapping[str, Any], low_profile: np.ndarray, high_profile: np.ndarray,
) -> tuple[float, float, float, float]:
    low_error = backtransformed_error(fit, low_profile)
    high_error = backtransformed_error(fit, high_profile)
    contrast = high_error - low_error
    percent = 100.0 * contrast / max(low_error, 0.05)
    return low_error, high_error, contrast, percent


def backtransformed_error(
    fit: Mapping[str, Any], profile: np.ndarray,
) -> float:
    beta = np.asarray(fit["beta"])
    x_mean = np.asarray(fit["x_mean"])
    y_mean = float(fit["y_mean"])
    eta = y_mean + float(beta @ (np.asarray(profile, dtype=float) - x_mean))
    return max(math.exp(eta) - 0.05, 0.0)


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.clip(0.0, 1.0).tolist()


def coverage_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output = []
    for feature in (
        PRIMARY_FEATURES
        + SENSITIVITY_CONTROLS
        + MODEL_EDGE_AUDIT_FEATURES
        + ("depth", "extension_factor", "conv2", "en1", "en2")
    ):
        record: dict[str, Any] = {"feature": feature}
        coverages = {}
        for label in ("overall", "adsorbate", "interstitial"):
            subset = rows if label == "overall" else [row for row in rows if row["defecttype"] == label]
            finite = sum(math.isfinite(finite_float(row.get(feature))) for row in subset)
            coverages[label] = finite / len(subset)
            record[f"coverage_{label}"] = coverages[label]
        minimum = min(coverages.values())
        record["coverage_gate"] = (
            "main_text_eligible" if minimum >= 0.95 else
            "supplement_only" if minimum >= 0.90 else "excluded"
        )
        output.append(record)
    return output


def aggregate_identity_vectors(
    rows: Sequence[Mapping[str, Any]], identity_key: str,
    feature_names: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    identities = sorted({str(row[identity_key]) for row in rows})
    vectors = []
    for identity in identities:
        members = [row for row in rows if str(row[identity_key]) == identity]
        vector = []
        for feature in feature_names:
            values = np.asarray(
                [finite_float(row.get(feature)) for row in members], dtype=float,
            )
            finite = values[np.isfinite(values)]
            vector.append(float(finite.mean()) if len(finite) else float("nan"))
        vectors.append(vector)
    return identities, np.asarray(vectors, dtype=float)


def fit_identity_novelty(
    train_ids: Sequence[str], train_vectors: np.ndarray,
    test_ids: Sequence[str], test_vectors: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Train-only robust scaling, PCA, and five-identity-neighbor novelty."""
    train = np.asarray(train_vectors, dtype=float)
    test = np.asarray(test_vectors, dtype=float)
    if (
        train.ndim != 2 or test.ndim != 2
        or train.shape[1] != len(feature_names)
        or test.shape[1] != len(feature_names)
        or len(train_ids) != len(train) or len(test_ids) != len(test)
        or set(train_ids) & set(test_ids)
    ):
        raise ValueError("invalid or overlapping identity-level novelty partitions")
    if len(train_ids) < 5:
        raise ValueError("identity novelty requires at least five unique training identities")

    medians = np.empty(train.shape[1], dtype=float)
    train_missing = ~np.isfinite(train)
    test_missing = ~np.isfinite(test)
    for column in range(train.shape[1]):
        finite = train[np.isfinite(train[:, column]), column]
        if len(finite) == 0:
            raise ValueError(f"all training identities miss novelty feature {feature_names[column]}")
        medians[column] = float(np.median(finite))
    train_imputed = np.where(train_missing, medians[None, :], train)
    test_imputed = np.where(test_missing, medians[None, :], test)
    augmented_train = np.column_stack([train_imputed, train_missing.astype(float)])
    augmented_test = np.column_stack([test_imputed, test_missing.astype(float)])
    augmented_names = list(feature_names) + [
        f"{name}__missing" for name in feature_names
    ]

    center = np.median(augmented_train, axis=0)
    q25, q75 = np.quantile(augmented_train, [0.25, 0.75], axis=0)
    iqr = q75 - q25
    retained = iqr > 1e-12
    if not retained.any():
        raise ValueError("all identity novelty columns have zero training IQR")
    scaled_train = (
        augmented_train[:, retained] - center[retained]
    ) / iqr[retained]
    scaled_test = (
        augmented_test[:, retained] - center[retained]
    ) / iqr[retained]
    pca_center = scaled_train.mean(axis=0)
    centered_train = scaled_train - pca_center
    _, singular, components_t = np.linalg.svd(centered_train, full_matrices=False)
    if len(singular) == 0 or singular[0] <= 0:
        raise ValueError("identity novelty PCA has zero rank")
    rank_tolerance = singular[0] * max(centered_train.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > rank_tolerance))
    if rank < 1:
        raise ValueError("identity novelty PCA has zero numerical rank")
    explained = singular[:rank] ** 2
    cumulative = np.cumsum(explained) / explained.sum()
    n_components = min(int(np.searchsorted(cumulative, 0.95) + 1), 10, rank)
    components = components_t[:n_components].T
    train_scores = centered_train @ components
    test_scores = (scaled_test - pca_center) @ components
    pairwise = np.linalg.norm(
        test_scores[:, None, :] - train_scores[None, :, :], axis=2,
    )
    nearest = np.partition(pairwise, kth=4, axis=1)[:, :5]
    novelty = nearest.mean(axis=1)
    if not np.isfinite(novelty).all():
        raise ValueError("identity novelty contains nonfinite distances")
    mapping = {identity: float(value) for identity, value in zip(test_ids, novelty)}
    audit = {
        "n_train_identities": len(train_ids),
        "n_test_identities": len(test_ids),
        "base_features": list(feature_names),
        "retained_augmented_features": [
            name for name, keep in zip(augmented_names, retained) if keep
        ],
        "training_medians": {
            name: float(value) for name, value in zip(feature_names, medians)
        },
        "pca_rank": rank,
        "pca_components": n_components,
        "pca_explained_variance": float(cumulative[n_components - 1]),
        "knn_k": 5,
        "metric": "Euclidean distance in train-only robust-scaled PCA space",
    }
    return mapping, audit


def oof_identity_novelty(
    rows: Sequence[Mapping[str, Any]], protocol_dir: Path, split_prefix: str,
    identity_key: str, feature_names: Sequence[str],
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    rows_by_index = {int(row["sample_index"]): row for row in rows}
    novelty_by_index: dict[int, float] = {}
    fold_audits = []
    for fold in range(5):
        split_id = f"{split_prefix}_cv5_f{fold}"
        split = load_split(protocol_dir, split_id)
        train_rows = [rows_by_index[int(index)] for index in split["train"]]
        test_rows = [rows_by_index[int(index)] for index in split["test"]]
        train_ids, train_vectors = aggregate_identity_vectors(
            train_rows, identity_key, feature_names,
        )
        test_ids, test_vectors = aggregate_identity_vectors(
            test_rows, identity_key, feature_names,
        )
        identity_novelty, audit = fit_identity_novelty(
            train_ids, train_vectors, test_ids, test_vectors, feature_names,
        )
        for row in test_rows:
            index = int(row["sample_index"])
            if index in novelty_by_index:
                raise ValueError(f"duplicate {split_prefix} novelty row {index}")
            novelty_by_index[index] = identity_novelty[str(row[identity_key])]
        fold_audits.append({"split_id": split_id, **audit})
    if set(novelty_by_index) != set(rows_by_index):
        raise ValueError(f"{split_prefix} novelty does not cover the canonical population")
    return novelty_by_index, fold_audits


def identity_robust_scale(
    rows: Sequence[Mapping[str, Any]], identity_key: str,
    values_by_index: Mapping[int, float],
) -> tuple[dict[int, float], dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[identity_key])].append(
            float(values_by_index[int(row["sample_index"])])
        )
    identity_values = []
    for identity, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        if np.max(array) - np.min(array) > 1e-10:
            raise ValueError(f"OOF novelty is not identity-constant for {identity}")
        identity_values.append(float(array.mean()))
    median = float(np.median(identity_values))
    q25, q75 = np.quantile(identity_values, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr <= 1e-12:
        raise ValueError("OOF identity novelty has zero IQR")
    scaled = {
        int(row["sample_index"]): (
            float(values_by_index[int(row["sample_index"])]) - median
        ) / iqr
        for row in rows
    }
    return scaled, {"identity_median": median, "identity_iqr": iqr}


def p3_gap_fit(
    error_gap: np.ndarray, novelty_gap: np.ndarray, defecttype: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    design = np.column_stack([
        novelty_gap,
        (defecttype == "interstitial").astype(float),
    ])
    fit = fit_huber_fixed_effects(
        np.asarray(error_gap, dtype=float), design, [], base_weights=weights,
    )
    effective = np.ones(len(error_gap), dtype=float) if weights is None else weights
    mean_gap = float(np.average(error_gap, weights=effective))
    return mean_gap, float(fit["beta"][0])


def bootstrap_two_sided_p(samples: Sequence[float]) -> float:
    values = np.asarray(samples, dtype=float)
    nonpositive = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    nonnegative = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    return min(1.0, 2.0 * min(nonpositive, nonnegative))


def p3_positive_claim_gate(
    evidence_tier: str, mean_ci: Sequence[float], slope_ci: Sequence[float],
    fold_means: Sequence[float], fold_slopes: Sequence[float],
) -> tuple[int, int, int]:
    positive_mean_folds = sum(float(value) > 0 for value in fold_means)
    positive_slope_folds = sum(float(value) > 0 for value in fold_slopes)
    gate = int(
        evidence_tier == "canonical"
        and float(mean_ci[0]) > 0 and float(slope_ci[0]) > 0
        and positive_mean_folds >= 4 and positive_slope_folds >= 4
    )
    return gate, positive_mean_folds, positive_slope_folds


def run_p3_novelty_analysis(
    rows: Sequence[Mapping[str, Any]], protocol_dir: Path,
    draws: int, seed: int, evidence_tier: str, evidence_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    host_novelty, host_audits = oof_identity_novelty(
        rows, protocol_dir, "host", "host", HOST_NOVELTY_FEATURES,
    )
    impurity_novelty, impurity_audits = oof_identity_novelty(
        rows, protocol_dir, "dopant", "dopant", IMPURITY_NOVELTY_FEATURES,
    )
    host_scaled, host_scale = identity_robust_scale(rows, "host", host_novelty)
    impurity_scaled, impurity_scale = identity_robust_scale(
        rows, "dopant", impurity_novelty,
    )
    novelty_rows = []
    for row in rows:
        index = int(row["sample_index"])
        novelty_rows.append({
            "evidence_tier": evidence_tier,
            "evidence_label": evidence_label,
            "sample_index": index,
            "raw_row_id": int(row["raw_row_id"]),
            "host": row["host"],
            "dopant": row["dopant"],
            "defecttype": row["defecttype"],
            "common_pair_fold_stratum": int(row["pair_fold"]),
            "host_oof_fold": int(row["host_fold"]),
            "impurity_oof_fold": int(row["dopant_fold"]),
            "host_identity_novelty": host_novelty[index],
            "impurity_identity_novelty": impurity_novelty[index],
            "host_identity_novelty_robust_z": host_scaled[index],
            "impurity_identity_novelty_robust_z": impurity_scaled[index],
            "novelty_gap": host_scaled[index] - impurity_scaled[index],
            "host_minus_impurity_absolute_error_eV": finite_float(
                row["host_minus_dopant_absolute_error_eV"]
            ),
        })
    error_gap = np.asarray([
        row["host_minus_impurity_absolute_error_eV"] for row in novelty_rows
    ], dtype=float)
    novelty_gap = np.asarray([row["novelty_gap"] for row in novelty_rows], dtype=float)
    defecttype = np.asarray([row["defecttype"] for row in novelty_rows])
    if not np.isfinite(error_gap).all() or not np.isfinite(novelty_gap).all():
        raise ValueError("P3 paired gaps contain nonfinite values")
    point_mean, point_slope = p3_gap_fit(error_gap, novelty_gap, defecttype)

    pair_labels = [f"{row['host']}::{row['dopant']}" for row in novelty_rows]
    unique_pairs = sorted(set(pair_labels))
    pair_lookup = {pair: index for index, pair in enumerate(unique_pairs)}
    row_pair = np.asarray([pair_lookup[pair] for pair in pair_labels], dtype=int)
    rng = np.random.default_rng(seed)
    mean_draws, slope_draws = [], []
    for _ in range(draws):
        selected = rng.integers(0, len(unique_pairs), size=len(unique_pairs))
        multiplicity = np.bincount(
            selected, minlength=len(unique_pairs),
        ).astype(float)
        mean_gap, slope = p3_gap_fit(
            error_gap, novelty_gap, defecttype, multiplicity[row_pair],
        )
        mean_draws.append(mean_gap)
        slope_draws.append(slope)
    mean_ci = np.quantile(mean_draws, [0.025, 0.975])
    slope_ci = np.quantile(slope_draws, [0.025, 0.975])

    fold_rows = []
    for fold in range(5):
        mask = np.asarray([
            int(row["common_pair_fold_stratum"]) == fold for row in novelty_rows
        ])
        fold_mean, fold_slope = p3_gap_fit(
            error_gap[mask], novelty_gap[mask], defecttype[mask],
        )
        fold_rows.append({
            "pair_fold_stratum": fold,
            "n": int(mask.sum()),
            "mean_error_gap_eV": fold_mean,
            "novelty_gap_slope_eV_per_robust_unit": fold_slope,
        })
    claim_gate, stable_mean, stable_slope = p3_positive_claim_gate(
        evidence_tier, mean_ci, slope_ci,
        [row["mean_error_gap_eV"] for row in fold_rows],
        [row["novelty_gap_slope_eV_per_robust_unit"] for row in fold_rows],
    )
    analysis = {
        "schema_version": "prm_g3_novelty_analysis_v1",
        "evidence_tier": evidence_tier,
        "evidence_label": evidence_label,
        "model": (
            "Huber regression of paired host-minus-impurity absolute-error gap "
            "on robust-scaled host-minus-impurity novelty gap plus class"
        ),
        "cluster": "host_impurity_pair",
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "mean_error_gap_eV": point_mean,
        "mean_error_gap_ci_low_eV": float(mean_ci[0]),
        "mean_error_gap_ci_high_eV": float(mean_ci[1]),
        "mean_error_gap_bootstrap_p": bootstrap_two_sided_p(mean_draws),
        "novelty_gap_slope_eV_per_robust_unit": point_slope,
        "novelty_gap_slope_ci_low": float(slope_ci[0]),
        "novelty_gap_slope_ci_high": float(slope_ci[1]),
        "novelty_gap_slope_bootstrap_p": bootstrap_two_sided_p(slope_draws),
        "positive_pair_strata_mean_gap": stable_mean,
        "positive_pair_strata_novelty_slope": stable_slope,
        "fold_stability": fold_rows,
        "host_oof_identity_scale": host_scale,
        "impurity_oof_identity_scale": impurity_scale,
        "host_fold_audits": host_audits,
        "impurity_fold_audits": impurity_audits,
        "claim_gate": claim_gate,
        "claim_language": (
            "consistent with structural-motif shift" if claim_gate
            else "P3 gate not met; cause remains unisolated"
        ),
    }
    return novelty_rows, analysis


def target_decile(value: float, cutpoints: np.ndarray) -> int:
    return int(np.searchsorted(np.asarray(cutpoints, dtype=float), value, side="right"))


def run_p4_case_selection(
    rows: Sequence[Mapping[str, Any]], evidence_tier: str,
    evidence_label: str, p1_population_gate: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Deterministic no-fallback high/low and pathology case ledger."""
    targets = np.asarray([finite_float(row["target_eV"]) for row in rows], dtype=float)
    if len(rows) != EXPECTED_N or not np.isfinite(targets).all():
        raise ValueError("P4 requires the full finite canonical target population")
    cutpoints = np.quantile(targets, np.arange(0.1, 1.0, 0.1))

    finite_conv_rows = [
        row for row in rows if math.isfinite(finite_float(row.get("abs_conv2")))
    ]
    n_conv_pathology = int(math.ceil(0.01 * EXPECTED_N))
    if len(finite_conv_rows) < n_conv_pathology:
        raise ValueError("fewer finite conv2 rows than the frozen full-population top-1% count")
    conv_ranked = sorted(
        finite_conv_rows,
        key=lambda row: (
            -finite_float(row["abs_conv2"]), int(row["sample_index"]),
        ),
    )
    conv_pathology_indices = {
        int(row["sample_index"]) for row in conv_ranked[:n_conv_pathology]
    }
    conv_rank = {
        int(row["sample_index"]): rank
        for rank, row in enumerate(conv_ranked[:n_conv_pathology], start=1)
    }
    xf_ranked = sorted(
        [
            row for row in rows
            if math.isfinite(finite_float(row.get("extension_factor")))
            and finite_float(row["extension_factor"]) > 2.0
        ],
        key=lambda row: (
            -finite_float(row["pair_absolute_error_eV"]),
            int(row["sample_index"]),
        ),
    )
    xf_rank = {
        int(row["sample_index"]): rank
        for rank, row in enumerate(xf_ranked, start=1)
    }
    unresolved_ranked = sorted(
        [
            row for row in rows
            if not math.isfinite(finite_float(row.get("extension_factor")))
            or not math.isfinite(finite_float(row.get("abs_conv2")))
        ],
        key=lambda row: (
            -finite_float(row["pair_absolute_error_eV"]),
            int(row["sample_index"]),
        ),
    )

    def nonpathological(row: Mapping[str, Any]) -> bool:
        index = int(row["sample_index"])
        xf = finite_float(row.get("extension_factor"))
        conv = finite_float(row.get("abs_conv2"))
        return (
            math.isfinite(xf) and xf <= 2.0
            and math.isfinite(conv) and index not in conv_pathology_indices
        )

    high_pool = sorted(
        [
            row for row in rows
            if str(row["defecttype"]) == "interstitial" and nonpathological(row)
        ],
        key=lambda row: (
            -finite_float(row["pair_absolute_error_eV"]),
            int(row["sample_index"]),
        ),
    )
    high_rank = {
        int(row["sample_index"]): rank
        for rank, row in enumerate(high_pool, start=1)
    }
    selected_high = []
    used_high_hosts: set[str] = set()
    used_high_dopants: set[str] = set()
    for row in high_pool:
        host, dopant = str(row["host"]), str(row["dopant"])
        if host in used_high_hosts or dopant in used_high_dopants:
            continue
        selected_high.append(row)
        used_high_hosts.add(host)
        used_high_dopants.add(dopant)
        if len(selected_high) == 4:
            break

    adsorbates = [row for row in rows if str(row["defecttype"]) == "adsorbate"]
    adsorbate_q25 = float(np.quantile(
        [finite_float(row["pair_absolute_error_eV"]) for row in adsorbates], 0.25,
    ))
    low_pool = [
        row for row in adsorbates
        if finite_float(row["pair_absolute_error_eV"]) <= adsorbate_q25
        and nonpathological(row)
    ]
    low_ranked = sorted(
        low_pool,
        key=lambda row: (
            finite_float(row["pair_absolute_error_eV"]), int(row["sample_index"]),
        ),
    )
    low_rank = {
        int(row["sample_index"]): rank
        for rank, row in enumerate(low_ranked, start=1)
    }
    selected_controls: list[tuple[int, Mapping[str, Any]]] = []
    used_control_hosts: set[str] = set()
    used_control_dopants: set[str] = set()
    for match_id, high in enumerate(selected_high, start=1):
        high_decile = target_decile(finite_float(high["target_eV"]), cutpoints)
        candidates = [
            row for row in low_pool
            if str(row["host_family"]) == str(high["host_family"])
            and target_decile(finite_float(row["target_eV"]), cutpoints) == high_decile
            and str(row["impurity_series"]) == str(high["impurity_series"])
            and str(row["host"]) not in used_control_hosts
            and str(row["dopant"]) not in used_control_dopants
        ]
        candidates.sort(key=lambda row: (
            abs(int(row["natoms"]) - int(high["natoms"])),
            abs(finite_float(row["target_eV"]) - finite_float(high["target_eV"])),
            int(row["sample_index"]),
        ))
        if not candidates:
            continue
        control = candidates[0]
        selected_controls.append((match_id, control))
        used_control_hosts.add(str(control["host"]))
        used_control_dopants.add(str(control["dopant"]))

    pool_rows = []
    for row in rows:
        index = int(row["sample_index"])
        xf = finite_float(row.get("extension_factor"))
        conv = finite_float(row.get("abs_conv2"))
        pool_rows.append({
            "evidence_tier": evidence_tier,
            "evidence_label": evidence_label,
            "sample_index": index,
            "raw_row_id": int(row["raw_row_id"]),
            "host": row["host"],
            "host_family": row["host_family"],
            "dopant": row["dopant"],
            "impurity_series": row["impurity_series"],
            "defecttype": row["defecttype"],
            "site": row["site"],
            "natoms": int(row["natoms"]),
            "target_eV": finite_float(row["target_eV"]),
            "target_decile": target_decile(finite_float(row["target_eV"]), cutpoints),
            "pair_absolute_error_eV": finite_float(row["pair_absolute_error_eV"]),
            "extension_factor": xf,
            "abs_conv2": conv,
            "nonpathological_eligible": int(nonpathological(row)),
            "high_error_interstitial_pool_rank": high_rank.get(index, ""),
            "bottom_quartile_adsorbate_pool_rank": low_rank.get(index, ""),
            "xf_gt_2_pathology_rank": xf_rank.get(index, ""),
            "top_1pct_abs_conv2_pathology_rank": conv_rank.get(index, ""),
            "provenance_missing": int(not math.isfinite(xf) or not math.isfinite(conv)),
        })

    selection_rows = []

    def selected_record(
        row: Mapping[str, Any], role: str, rank: int, match_id: int | str,
    ) -> dict[str, Any]:
        return {
            "evidence_tier": evidence_tier,
            "evidence_label": evidence_label,
            "role": role,
            "role_rank": rank,
            "match_id": match_id,
            "sample_index": int(row["sample_index"]),
            "raw_row_id": int(row["raw_row_id"]),
            "host": row["host"],
            "host_family": row["host_family"],
            "dopant": row["dopant"],
            "impurity_series": row["impurity_series"],
            "defecttype": row["defecttype"],
            "site": row["site"],
            "natoms": int(row["natoms"]),
            "target_eV": finite_float(row["target_eV"]),
            "target_decile": target_decile(finite_float(row["target_eV"]), cutpoints),
            "pair_absolute_error_eV": finite_float(row["pair_absolute_error_eV"]),
            "extension_factor": finite_float(row.get("extension_factor")),
            "abs_conv2": finite_float(row.get("abs_conv2")),
            "cn_5A": finite_float(row.get("cn_5A")),
            "postrelaxation_min_clearance_A": finite_float(
                row.get("postrelaxation_min_clearance_A")
            ),
        }

    for match_id, row in enumerate(selected_high, start=1):
        selection_rows.append(selected_record(
            row, "high_error_nonpathological_interstitial", match_id, match_id,
        ))
    for match_id, row in selected_controls:
        selection_rows.append(selected_record(
            row, "matched_bottom_quartile_adsorbate", match_id, match_id,
        ))
    for rank, row in enumerate(xf_ranked[:4], start=1):
        selection_rows.append(selected_record(row, "xf_gt_2_pathology", rank, ""))
    for rank, row in enumerate(conv_ranked[:4], start=1):
        selection_rows.append(selected_record(
            row, "top_1pct_abs_conv2_pathology", rank, "",
        ))
    for rank, row in enumerate(unresolved_ranked[:4], start=1):
        selection_rows.append(selected_record(
            row, "missing_pathology_provenance", rank, "",
        ))

    complete = len(selected_high) == 4 and len(selected_controls) == 4
    high_families = len({str(row["host_family"]) for row in selected_high})
    main_text_gate = int(
        evidence_tier == "canonical" and p1_population_gate
        and complete and high_families >= 3
    )
    summary = {
        "schema_version": "prm_g3_case_selection_summary_v1",
        "evidence_tier": evidence_tier,
        "evidence_label": evidence_label,
        "target_decile_cutpoints_eV": [float(value) for value in cutpoints],
        "target_decile_tie_rule": "np.searchsorted(side='right')",
        "conv2_pathology_rule": (
            "exactly ceil(0.01*10224)=103 finite rows ranked by descending abs_conv2, "
            "then sample_index"
        ),
        "n_finite_conv2": len(finite_conv_rows),
        "n_conv2_pathology": n_conv_pathology,
        "adsorbate_bottom_quartile_error_threshold_eV": adsorbate_q25,
        "n_high_error_selected": len(selected_high),
        "n_matched_controls_selected": len(selected_controls),
        "high_error_host_families": high_families,
        "case_panel_complete": complete,
        "matching_fallback_used": False,
        "p1_population_effect_gate": bool(p1_population_gate),
        "main_text_gate": main_text_gate,
        "claim_boundary": (
            "deterministic case panel eligible" if main_text_gate
            else "case panel is exploratory, incomplete, or lacks a passing population association"
        ),
    }
    return pool_rows, selection_rows, summary


def bootstrap_family_rows(
    rows: Sequence[Mapping[str, Any]], group_key: str, identity_key: str,
    draws: int, seed: int, evidence_tier: str, evidence_label: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output = []
    for defecttype in ("overall", "adsorbate", "interstitial"):
        typed = rows if defecttype == "overall" else [row for row in rows if row["defecttype"] == defecttype]
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in typed:
            groups[str(row[group_key])].append(row)
        for group, members in sorted(groups.items()):
            identities: dict[str, np.ndarray] = {}
            for identity in sorted({str(row[identity_key]) for row in members}):
                identities[identity] = np.asarray(
                    [float(row["pair_absolute_error_eV"]) for row in members if str(row[identity_key]) == identity]
                )
            sample_values = np.concatenate(list(identities.values()))
            macro = float(np.mean([values.mean() for values in identities.values()]))
            if len(identities) >= 2:
                keys = list(identities)
                sample_draws, macro_draws = [], []
                for _ in range(draws):
                    selected = rng.integers(0, len(keys), size=len(keys))
                    values = [identities[keys[index]] for index in selected]
                    sample_draws.append(float(np.concatenate(values).mean()))
                    macro_draws.append(float(np.mean([value.mean() for value in values])))
                sample_ci = np.quantile(sample_draws, [0.025, 0.975])
                macro_ci = np.quantile(macro_draws, [0.025, 0.975])
            else:
                sample_ci = macro_ci = (float("nan"), float("nan"))
            output.append(
                {
                    "evidence_tier": evidence_tier,
                    "evidence_label": evidence_label,
                    "axis": group_key,
                    "group": group,
                    "defecttype": defecttype,
                    "n_samples": len(members),
                    "n_identities": len(identities),
                    "sample_weighted_mae_eV": float(sample_values.mean()),
                    "sample_weighted_ci_low_eV": float(sample_ci[0]),
                    "sample_weighted_ci_high_eV": float(sample_ci[1]),
                    "identity_macro_mae_eV": macro,
                    "identity_macro_ci_low_eV": float(macro_ci[0]),
                    "identity_macro_ci_high_eV": float(macro_ci[1]),
                    "inferentially_eligible": int(len(identities) >= 5 and len(members) >= 100),
                }
            )
    return output


def analyze(args: argparse.Namespace) -> None:
    runtime = require_server_preflight(args)
    if args.bootstrap_draws != 2000 or args.bootstrap_seed != 20260810:
        raise ValueError(
            "formal G3 requires exactly 2000 bootstrap draws and seed 20260810"
        )
    run_dir = Path(args.output_dir).resolve()
    try:
        run_dir.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("formal G3 output must remain outside the repository")
    analysis_outputs = (
        "coverage.csv", "effect_estimates.csv", "block_tests.csv",
        "family_mae.csv", "physical_figure_data.csv", "analysis_started.json",
        "novelty_rows.csv", "novelty_analysis.json", "case_eligible_pool.csv",
        "case_selection.csv", "case_selection_summary.json",
        "analysis_manifest.json",
    )
    existing_analysis_outputs = [
        name for name in analysis_outputs if (run_dir / name).exists()
    ]
    if existing_analysis_outputs:
        raise FileExistsError(
            "refusing to overwrite prior/partial analysis outputs: "
            + ", ".join(existing_analysis_outputs)
        )
    extraction_manifest_path = run_dir / "extraction_manifest.json"
    joined_path = run_dir / "joined_descriptors.csv"
    if not extraction_manifest_path.is_file() or not joined_path.is_file():
        raise FileNotFoundError("analyze requires a completed extraction in --output-dir")
    extraction = json.loads(extraction_manifest_path.read_text())
    if extraction.get("schema_version") != "prm_g3_extraction_manifest_v1":
        raise ValueError("unsupported G3 extraction manifest")
    current_prereg_hash = file_sha256(ROOT / "paper_Q1/review/g3_preregistration.json")
    if extraction.get("inputs", {}).get("preregistration_sha256") != current_prereg_hash:
        raise ValueError("extraction is not bound to the current preregistration")
    protocol_dir = Path(args.protocol_dir)
    protocol_rows = load_protocol_samples(protocol_dir)
    protocol_by_index = {
        int(row["sample_index"]): row for row in protocol_rows
    }
    current_protocol_hash = file_sha256(protocol_dir / "manifest.json")
    if extraction.get("inputs", {}).get("protocol_manifest_sha256") != current_protocol_hash:
        raise ValueError("extraction is not bound to the commit-pinned protocol")
    if (
        extraction.get("inputs", {}).get("dataset", {}).get("sha256")
        != EXPECTED_DATA_SHA256
        or extraction.get("inputs", {}).get("raw_database", {}).get("sha256")
        != EXPECTED_RAW_DB_SHA256
    ):
        raise ValueError("extraction manifest source hashes violate the frozen contract")
    if extraction["outputs"]["joined_descriptors"]["sha256"] != file_sha256(joined_path):
        raise ValueError("joined descriptor table changed after extraction")
    if extraction["runtime"]["git"]["commit"] != runtime["git"]["commit"]:
        raise ValueError("extraction and analysis commits differ")
    rows_raw = read_csv_rows(joined_path)
    if len(rows_raw) != EXPECTED_N:
        raise ValueError("joined descriptor population mismatch")
    joined_indices = [int(row.get("sample_index", -1)) for row in rows_raw]
    if len(set(joined_indices)) != EXPECTED_N or set(joined_indices) != set(protocol_by_index):
        raise ValueError("joined sample indices do not uniquely match the frozen protocol")
    for row in rows_raw:
        protocol_row = protocol_by_index[int(row["sample_index"])]
        if (
            int(row.get("raw_row_id", -1)) != int(protocol_row["id"])
            or row.get("unique_id") != protocol_row["unique_id"]
            or row.get("host") != protocol_row["host"]
            or row.get("dopant") != protocol_row["dopant"]
            or row.get("defecttype") != protocol_row["defecttype"]
            or row.get("site") != protocol_row["site"]
            or not math.isclose(
                finite_float(row.get("target_eV")), float(protocol_row["target_eV"]),
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise ValueError("joined identity/target differs from the frozen protocol")
    evidence_tier = str(extraction["evidence_tier"])
    evidence_label = str(extraction["evidence_label"])
    model_status = str(extraction["model_status"])
    expected_triples = {
        "exploratory": (LEGACY_WATERMARK, "legacy_order_sensitive_graph"),
        "canonical": (CANONICAL_LABEL, "repaired_g1_g2"),
    }
    if evidence_tier not in expected_triples or (evidence_label, model_status) != expected_triples[evidence_tier]:
        raise ValueError("invalid extraction tier/label/model trust triple")
    if args.evidence_tier != evidence_tier or args.model_status != model_status:
        raise ValueError("analyze CLI tier/model must exactly match extraction")
    for row in rows_raw:
        if (
            row.get("evidence_tier") != evidence_tier
            or row.get("evidence_label") != evidence_label
            or row.get("model_status") != model_status
        ):
            raise ValueError("joined row violates the extraction trust triple")
    if evidence_tier == "canonical":
        if not args.g2_acceptance or not args.data_path or not args.raw_db:
            raise ValueError(
                "canonical analyze requires --g2-acceptance, --data-path, and --raw-db again"
            )
        acceptance_path = Path(args.g2_acceptance)
        if file_sha256(acceptance_path) != extraction["inputs"].get("g2_acceptance_sha256"):
            raise ValueError("canonical analyze G2 acceptance hash differs from extraction")
        data_path = Path(args.data_path)
        raw_db = Path(args.raw_db)
        require_hash(data_path, EXPECTED_DATA_SHA256, "canonical recheck cleaned dataset")
        require_hash(raw_db, EXPECTED_RAW_DB_SHA256, "canonical recheck raw IMP2D database")
        target_by_index = {
            int(row["sample_index"]): float(row["target_eV"])
            for row in protocol_rows
        }
        pair_fold, pair_support = fold_maps(
            protocol_dir, protocol_by_index, "pair"
        )
        host_fold, _ = fold_maps(protocol_dir, protocol_by_index, "host")
        dopant_fold, _ = fold_maps(protocol_dir, protocol_by_index, "dopant")
        canonical_specs, canonical_acceptance = canonical_prediction_specs(
            acceptance_path, protocol_dir
        )
        if (
            extraction.get("inputs", {}).get("prediction_graph_dataset", {}).get("sha256")
            != canonical_acceptance["repaired_dataset"]["graph_dataset_sha256"]
        ):
            raise ValueError("canonical extraction graph dataset differs from G2 acceptance")
        canonical_predictions, canonical_sources = load_all_oof_predictions(
            canonical_specs, protocol_dir, target_by_index
        )
        if canonical_sources != extraction["inputs"].get("prediction_sources"):
            raise ValueError("canonical prediction-source ledger differs from extraction")
        reconstructed_rows, reconstructed_depth_mismatches = build_joined_rows(
            raw_db=raw_db,
            graph_data_path=resolve_contract_path(
                canonical_acceptance["repaired_dataset"]["path"]
            ),
            graph_data_sha256=str(
                canonical_acceptance["repaired_dataset"]["graph_dataset_sha256"]
            ),
            required_graph_builder_version="exact_mic_invariant_triplets_v1",
            rows=protocol_rows,
            pair_fold=pair_fold,
            host_fold=host_fold,
            dopant_fold=dopant_fold,
            pair_support=pair_support,
            predictions=canonical_predictions,
            evidence_tier=evidence_tier,
            evidence_label=evidence_label,
            model_status=model_status,
        )
        assert_joined_rows_equal(rows_raw, reconstructed_rows)
        if reconstructed_depth_mismatches != extraction.get("population", {}).get(
            "depth_class_mismatches"
        ):
            raise ValueError("canonical depth-class audit differs from extraction")
    analysis_started_path = run_dir / "analysis_started.json"
    analysis_started_path.write_text(strict_json({
        "schema_version": "prm_g3_analysis_started_v1",
        "created_at": utc_now(),
        "evidence_tier": evidence_tier,
        "evidence_label": evidence_label,
        "git_commit": runtime["git"]["commit"],
        "extraction_manifest_sha256": file_sha256(extraction_manifest_path),
        "bootstrap_draws": 2000,
        "bootstrap_seed": 20260810,
        "write_once": True,
    }))
    coverage = coverage_rows(rows_raw)
    coverage_by_feature = {row["feature"]: row for row in coverage}
    eligible_features = [
        feature for feature in PRIMARY_FEATURES
        if coverage_by_feature[feature]["coverage_gate"] == "main_text_eligible"
    ]
    if not set(E_MODULE_ALIGNED).issubset(eligible_features):
        raise RuntimeError("the four E-module-aligned features lack class-stratified coverage")
    profile_features = eligible_adjusted_profile_features(eligible_features)

    complete = []
    for row in rows_raw:
        values = {name: finite_float(row.get(name)) for name in eligible_features}
        required = [finite_float(row.get("pair_absolute_error_eV")), *values.values()]
        if all(math.isfinite(value) for value in required):
            complete.append({**row, **values})
    if len(complete) < int(0.90 * EXPECTED_N):
        raise RuntimeError("complete-case primary analysis coverage is below 90 percent")

    y = np.log(np.asarray([finite_float(row["pair_absolute_error_eV"]) for row in complete]) + 0.05)
    defecttype = np.asarray([row["defecttype"] for row in complete])
    numeric = {name: np.asarray([float(row[name]) for row in complete]) for name in eligible_features}
    design, design_names, scaling = build_design(numeric, defecttype, eligible_features)
    groups = [
        group_codes([row["host"] for row in complete]),
        group_codes([row["dopant"] for row in complete]),
        group_codes([row["pair_fold"] for row in complete]),
    ]
    pair_labels = [f"{row['host']}::{row['dopant']}" for row in complete]
    fit = fit_huber_fixed_effects(y, design, groups)
    class_medians = {
        class_name: {
            feature: float(np.median(numeric[feature][defecttype == class_name]))
            for feature in eligible_features
        }
        for class_name in ("adsorbate", "interstitial")
    }
    quantiles = {
        class_name: {
            feature: tuple(float(value) for value in np.quantile(
                numeric[feature][defecttype == class_name], [0.10, 0.90]
            ))
            for feature in eligible_features
        }
        for class_name in ("adsorbate", "interstitial")
    }

    point_effects: dict[tuple[str, str], dict[str, float]] = {}
    for class_name in ("adsorbate", "interstitial"):
        for feature in eligible_features:
            low_value, high_value = quantiles[class_name][feature]
            low_profile = profile_design(
                eligible_features, scaling, class_medians, class_name, feature, low_value,
            )
            high_profile = profile_design(
                eligible_features, scaling, class_medians, class_name, feature, high_value,
            )
            low, high, contrast, percent = backtransformed_contrast(fit, low_profile, high_profile)
            point_effects[(class_name, feature)] = {
                "q10": low_value, "q90": high_value,
                "predicted_low_error_eV": low, "predicted_high_error_eV": high,
                "contrast_eV": contrast, "percent_contrast": percent,
            }

    profile_design_grid: dict[tuple[str, str], list[tuple[float, np.ndarray]]] = {}
    profile_point: dict[tuple[str, str], list[float]] = {}
    for class_name in ("adsorbate", "interstitial"):
        for feature in profile_features:
            q10, q90 = quantiles[class_name][feature]
            values = np.linspace(q10, q90, PROFILE_GRID_POINTS)
            designs = [
                profile_design(
                    eligible_features, scaling, class_medians,
                    class_name, feature, float(value),
                )
                for value in values
            ]
            key = (class_name, feature)
            profile_design_grid[key] = list(zip(values.tolist(), designs))
            profile_point[key] = [
                backtransformed_error(fit, profile) for profile in designs
            ]

    draws = int(args.bootstrap_draws)
    if draws != 2000 or int(args.bootstrap_seed) != 20260810:
        raise ValueError("the preregistered bootstrap count/seed changed during analysis")
    unique_pairs = sorted(set(pair_labels))
    pair_lookup = {pair: index for index, pair in enumerate(unique_pairs)}
    row_pair = np.asarray([pair_lookup[pair] for pair in pair_labels], dtype=int)
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap = {key: [] for key in point_effects}
    profile_bootstrap: dict[tuple[str, str], list[list[float]]] = {
        key: [] for key in profile_design_grid
    }
    bootstrap_beta: list[np.ndarray] = []
    for draw in range(draws):
        selected = rng.integers(0, len(unique_pairs), size=len(unique_pairs))
        multiplicity = np.bincount(selected, minlength=len(unique_pairs)).astype(float)
        weights = multiplicity[row_pair]
        draw_fit = fit_huber_fixed_effects(y, design, groups, base_weights=weights)
        bootstrap_beta.append(np.asarray(draw_fit["beta"], dtype=float))
        for key in bootstrap:
            class_name, feature = key
            low_value, high_value = quantiles[class_name][feature]
            low_profile = profile_design(
                eligible_features, scaling, class_medians, class_name, feature, low_value,
            )
            high_profile = profile_design(
                eligible_features, scaling, class_medians, class_name, feature, high_value,
            )
            bootstrap[key].append(backtransformed_contrast(draw_fit, low_profile, high_profile)[2])
        for key, grid in profile_design_grid.items():
            profile_bootstrap[key].append([
                backtransformed_error(draw_fit, profile) for _, profile in grid
            ])
        if (draw + 1) % 100 == 0:
            print(f"pair bootstrap {draw+1}/{draws}", flush=True)

    fold_signs: dict[tuple[str, str], list[int]] = {key: [] for key in point_effects}
    for fold in range(5):
        mask = np.asarray([int(row["pair_fold"]) == fold for row in complete])
        fold_groups = [group[mask] for group in groups[:2]]
        fold_fit = fit_huber_fixed_effects(y[mask], design[mask], fold_groups)
        for key in fold_signs:
            class_name, feature = key
            q10, q90 = quantiles[class_name][feature]
            low_profile = profile_design(eligible_features, scaling, class_medians, class_name, feature, q10)
            high_profile = profile_design(eligible_features, scaling, class_medians, class_name, feature, q90)
            contrast = backtransformed_contrast(fold_fit, low_profile, high_profile)[2]
            fold_signs[key].append(int(np.sign(contrast)))

    xf_mask = np.asarray([
        row.get("xf_pathology", "") == "0" for row in complete
    ])
    if xf_mask.sum() < int(0.80 * len(complete)):
        raise RuntimeError("XF<=2 sensitivity retains less than 80 percent of primary rows")
    xf_fit = fit_huber_fixed_effects(
        y[xf_mask], design[xf_mask], [group[xf_mask] for group in groups],
    )
    sensitivity_sign: dict[tuple[str, str], int] = {}
    for key in point_effects:
        class_name, feature = key
        q10, q90 = quantiles[class_name][feature]
        low_profile = profile_design(eligible_features, scaling, class_medians, class_name, feature, q10)
        high_profile = profile_design(eligible_features, scaling, class_medians, class_name, feature, q90)
        sensitivity_sign[key] = int(np.sign(backtransformed_contrast(xf_fit, low_profile, high_profile)[2]))

    eligible_controls = [
        feature for feature in SENSITIVITY_CONTROLS
        if coverage_by_feature[feature]["coverage_gate"] == "main_text_eligible"
    ]
    control_mask = np.asarray([
        all(math.isfinite(finite_float(row.get(feature))) for feature in eligible_controls)
        for row in complete
    ])
    if control_mask.sum() < int(0.90 * len(complete)):
        raise RuntimeError("nuisance-control sensitivity retains less than 90 percent of primary rows")
    control_columns, retained_controls = [], []
    for feature in eligible_controls:
        values = np.asarray([finite_float(row.get(feature)) for row in complete])[control_mask]
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        if q75 - q25 <= 1e-12:
            continue
        retained_controls.append(feature)
        control_columns.append((values - median) / float(q75 - q25))
    if not retained_controls:
        raise RuntimeError("no nondegenerate nuisance control survives the frozen coverage gate")
    sensitivity_design = np.column_stack(
        [design[control_mask], np.column_stack(control_columns)]
    )
    sensitivity_groups = [group[control_mask] for group in groups] + [
        group_codes([row["spacegroup"] for row, keep in zip(complete, control_mask) if keep]),
        group_codes([row["supercell"] for row, keep in zip(complete, control_mask) if keep]),
    ]
    nuisance_fit = fit_huber_fixed_effects(
        y[control_mask], sensitivity_design, sensitivity_groups,
    )
    nuisance_sign: dict[tuple[str, str], int] = {}
    control_reference = np.zeros(len(retained_controls), dtype=float)
    for key in point_effects:
        class_name, feature = key
        q10, q90 = quantiles[class_name][feature]
        low_profile = np.concatenate([
            profile_design(eligible_features, scaling, class_medians, class_name, feature, q10),
            control_reference,
        ])
        high_profile = np.concatenate([
            profile_design(eligible_features, scaling, class_medians, class_name, feature, q90),
            control_reference,
        ])
        nuisance_sign[key] = int(np.sign(
            backtransformed_contrast(nuisance_fit, low_profile, high_profile)[2]
        ))

    effects = []
    raw_pvalues = []
    for key, point in point_effects.items():
        samples = np.asarray(bootstrap[key], dtype=float)
        low_ci, high_ci = np.quantile(samples, [0.025, 0.975])
        nonpositive = (np.count_nonzero(samples <= 0) + 1) / (len(samples) + 1)
        nonnegative = (np.count_nonzero(samples >= 0) + 1) / (len(samples) + 1)
        pvalue = min(1.0, 2.0 * min(nonpositive, nonnegative))
        raw_pvalues.append(pvalue)
        direction = int(np.sign(point["contrast_eV"]))
        stable_folds = sum(sign == direction for sign in fold_signs[key])
        effects.append(
            {
                "evidence_tier": evidence_tier,
                "evidence_label": evidence_label,
                "feature": key[1],
                "block": "geometry" if key[1] in E_ALIGNED_GEOMETRY + INDEPENDENT_GEOMETRY else "chemistry",
                "defecttype": key[0],
                **point,
                "ci_low_eV": float(low_ci),
                "ci_high_eV": float(high_ci),
                "bootstrap_p": pvalue,
                "fold_signs": "|".join(str(value) for value in fold_signs[key]),
                "same_direction_folds": stable_folds,
                "xf_le_2_same_direction": int(sensitivity_sign[key] == direction),
                "nuisance_control_same_direction": int(nuisance_sign[key] == direction),
            }
        )
    adjusted = benjamini_hochberg(raw_pvalues)
    for row, qvalue in zip(effects, adjusted):
        row["bh_q"] = qvalue
        magnitude = abs(float(row["contrast_eV"])) >= 0.10 or abs(float(row["percent_contrast"])) >= 20.0
        interval = float(row["ci_low_eV"]) * float(row["ci_high_eV"]) > 0
        row["paper_effect_gate"] = int(
            magnitude and interval and qvalue <= 0.05
            and int(row["same_direction_folds"]) >= 4
            and int(row["xf_le_2_same_direction"]) == 1
            and int(row["nuisance_control_same_direction"]) == 1
            and evidence_tier == "canonical"
        )

    adjusted_profile_rows: list[dict[str, Any]] = []
    for key, grid in profile_design_grid.items():
        class_name, feature = key
        bootstrap_matrix = np.asarray(profile_bootstrap[key], dtype=float)
        if bootstrap_matrix.shape != (draws, PROFILE_GRID_POINTS):
            raise RuntimeError("panel-a adjusted-profile bootstrap shape mismatch")
        lower, upper = np.quantile(bootstrap_matrix, [0.025, 0.975], axis=0)
        q10, q90 = quantiles[class_name][feature]
        for grid_index, ((feature_value, _), point, low_ci, high_ci) in enumerate(
            zip(grid, profile_point[key], lower, upper)
        ):
            adjusted_profile_rows.append({
                "panel": "a",
                "record_type": "adjusted_profile",
                "evidence_tier": evidence_tier,
                "evidence_label": evidence_label,
                "defecttype": class_name,
                "feature": feature,
                "grid_index": grid_index,
                "grid_points": PROFILE_GRID_POINTS,
                "feature_value": float(feature_value),
                "q10": float(q10),
                "q90": float(q90),
                "predicted_abs_error_eV": float(point),
                "pointwise_ci_low_eV": float(low_ci),
                "pointwise_ci_high_eV": float(high_ci),
                "interval_scope": "pointwise_95pct_pair_cluster_bootstrap",
            })

    coefficient_covariance = np.cov(np.stack(bootstrap_beta), rowvar=False, ddof=1)
    block_rows = []
    beta = np.asarray(fit["beta"])
    for block, features in (
        ("geometry", E_ALIGNED_GEOMETRY + INDEPENDENT_GEOMETRY),
        ("chemistry", CHEMISTRY_FEATURES),
    ):
        indices = [
            index for index, name in enumerate(design_names)
            if any(name.endswith(f":{feature}") for feature in features if feature in eligible_features)
        ]
        sub_beta = beta[indices]
        sub_cov = coefficient_covariance[np.ix_(indices, indices)]
        rank = int(np.linalg.matrix_rank(sub_cov, tol=1e-10))
        if rank < 1:
            raise RuntimeError(f"bootstrap covariance for {block} block has zero rank")
        wald = float(sub_beta @ np.linalg.pinv(sub_cov, rcond=1e-10) @ sub_beta)
        block_rows.append(
            {
                "evidence_tier": evidence_tier,
                "evidence_label": evidence_label,
                "block": block,
                "degrees_of_freedom": rank,
                "wald_chi2": wald,
                "p_value": float(chi2.sf(wald, rank)),
                "covariance": "host-impurity-pair bootstrap coefficient covariance",
            }
        )

    numeric_rows = [
        {**row, "pair_absolute_error_eV": finite_float(row["pair_absolute_error_eV"])}
        for row in rows_raw
    ]
    family_rows = bootstrap_family_rows(
        numeric_rows, "host_family", "host", draws, args.bootstrap_seed + 101,
        evidence_tier, evidence_label,
    ) + bootstrap_family_rows(
        numeric_rows, "impurity_series", "dopant", draws, args.bootstrap_seed + 202,
        evidence_tier, evidence_label,
    )
    novelty_rows, novelty_analysis = run_p3_novelty_analysis(
        rows_raw, protocol_dir, draws, int(args.bootstrap_seed),
        evidence_tier, evidence_label,
    )
    any_effect = any(int(row["paper_effect_gate"]) == 1 for row in effects)
    case_pool, case_selection, case_summary = run_p4_case_selection(
        rows_raw, evidence_tier, evidence_label, any_effect,
    )

    write_csv(run_dir / "coverage.csv", coverage, list(coverage[0]))
    write_csv(run_dir / "effect_estimates.csv", effects, list(effects[0]))
    write_csv(run_dir / "block_tests.csv", block_rows, list(block_rows[0]))
    write_csv(run_dir / "family_mae.csv", family_rows, list(family_rows[0]))
    figure_rows = []
    for row in rows_raw:
        common = {
            "evidence_tier": evidence_tier,
            "evidence_label": evidence_label,
            "sample_index": int(row["sample_index"]),
            "raw_row_id": int(row["raw_row_id"]),
            "defecttype": row["defecttype"],
            "pair_absolute_error_eV": finite_float(row["pair_absolute_error_eV"]),
        }
        for feature in ("cn_5A", "postrelaxation_min_clearance_A"):
            figure_rows.append({
                "panel": "a",
                "record_type": "sample_observation",
                "feature": feature,
                "feature_value": finite_float(row.get(feature)),
                **common,
            })
        figure_rows.append({
            "panel": "c",
            "record_type": "sample_observation",
            "feature": "extension_factor",
            "feature_value": finite_float(row.get("extension_factor")),
            "extension_factor": finite_float(row.get("extension_factor")),
            "abs_log_extension_factor": finite_float(
                row.get("abs_log_extension_factor")
            ),
            "xf_pathology": row.get("xf_pathology", ""),
            "reference_x": 2.0,
            "reference_label": "XF = 2",
            **common,
        })
    for row in effects:
        panel = (
            "c" if row["feature"] == "abs_log_extension_factor"
            else "a" if row["feature"] in E_ALIGNED_GEOMETRY + INDEPENDENT_GEOMETRY
            else "b"
        )
        figure_rows.append(
            {
                "panel": panel,
                "record_type": "adjusted_effect", **row,
            }
        )
    figure_rows.extend(adjusted_profile_rows)
    for row in family_rows:
        figure_rows.append({"panel": "d", "record_type": "family_mae", **row})
    figure_fields = sorted({key for row in figure_rows for key in row})
    write_csv(run_dir / "physical_figure_data.csv", figure_rows, figure_fields)
    write_csv(run_dir / "novelty_rows.csv", novelty_rows, list(novelty_rows[0]))
    (run_dir / "novelty_analysis.json").write_text(strict_json(novelty_analysis))
    write_csv(run_dir / "case_eligible_pool.csv", case_pool, list(case_pool[0]))
    write_csv(
        run_dir / "case_selection.csv", case_selection, list(case_selection[0]),
    )
    (run_dir / "case_selection_summary.json").write_text(strict_json(case_summary))

    outputs = {}
    for name in (
        "coverage.csv", "effect_estimates.csv", "block_tests.csv",
        "family_mae.csv", "physical_figure_data.csv", "analysis_started.json",
        "novelty_rows.csv", "novelty_analysis.json", "case_eligible_pool.csv",
        "case_selection.csv", "case_selection_summary.json",
    ):
        outputs[name] = {"sha256": file_sha256(run_dir / name)}
    manifest = {
        "schema_version": "prm_g3_analysis_manifest_v1",
        "created_at": utc_now(),
        "evidence_tier": evidence_tier,
        "evidence_label": evidence_label,
        "runtime": runtime,
        "extraction_manifest_sha256": file_sha256(extraction_manifest_path),
        "joined_descriptors_sha256": file_sha256(joined_path),
        "analysis": {
            "n_complete": len(complete),
            "eligible_features": eligible_features,
            "excluded_or_sm_features": [feature for feature in PRIMARY_FEATURES if feature not in eligible_features],
            "estimator": "Huber IRLS with within-iteration crossed fixed-effect projection",
            "fixed_effects": ["host", "dopant", "pair_fold"],
            "cluster": "host_dopant_pair",
            "bootstrap_draws": draws,
            "bootstrap_seed": int(args.bootstrap_seed),
            "bh_fdr": 0.05,
            "nuisance_controls": retained_controls,
            "nuisance_fixed_effects": ["spacegroup", "supercell"],
            "adjusted_profile_features": list(profile_features),
            "adjusted_profile_grid_points": PROFILE_GRID_POINTS,
            "any_canonical_effect_gate": any_effect,
            "p3_claim_gate": bool(novelty_analysis["claim_gate"]),
            "p4_case_panel_complete": bool(case_summary["case_panel_complete"]),
            "p4_main_text_gate": bool(case_summary["main_text_gate"]),
        },
        "implemented_prespecified_modules": [
            "P0 provenance and coverage",
            "P1 robust local geometry and chemistry",
            "P2 family decomposition",
            "P3 identity-level host-versus-impurity novelty",
            "P4 deterministic structural-case ledger",
        ],
        "pending_prespecified_modules": [],
        "outputs": outputs,
    }
    (run_dir / "analysis_manifest.json").write_text(strict_json(manifest))
    status = {
        "schema_version": "prm_g3_run_status_v1",
        "state": "exploratory_analysis_complete" if evidence_tier == "exploratory" else "canonical_analysis_complete",
        "evidence_tier": evidence_tier,
        "evidence_label": evidence_label,
        "canonical_provenance_verified": evidence_tier == "canonical",
        "canonical_analysis_complete": evidence_tier == "canonical",
        "canonical_main_text_eligible": canonical_main_text_claim_eligible(
            evidence_tier,
            any_effect,
            bool(novelty_analysis["claim_gate"]),
            bool(case_summary["main_text_gate"]),
        ),
        "claim_specific_gates": {
            "any_p1_effect": any_effect,
            "p3_structural_motif_shift": bool(novelty_analysis["claim_gate"]),
            "p4_case_panel": bool(case_summary["main_text_gate"]),
        },
        "reason": (
            LEGACY_WATERMARK if evidence_tier == "exploratory"
            else "canonical analysis complete; manuscript claims remain subject to per-output gates"
        ),
    }
    (run_dir / "status.json").write_text(strict_json(status))
    print(f"analysis complete in {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "analyze", "all"))
    parser.add_argument("--evidence-tier", choices=("exploratory", "canonical"), default="exploratory")
    parser.add_argument(
        "--model-status", choices=("legacy_order_sensitive_graph", "repaired_g1_g2"),
        default="legacy_order_sensitive_graph",
    )
    parser.add_argument("--data-path", default=os.environ.get("PRM_DATASET_PATH", ""))
    parser.add_argument("--raw-db", default=os.environ.get("PRM_RAW_DB_PATH", ""))
    parser.add_argument("--protocol-dir", default=str(ROOT / "artifacts/prm_protocol_v2"))
    parser.add_argument(
        "--prediction-root",
        default=str(ROOT / "artifacts/prm_results/comparison/runs/selected/g111/transfer"),
    )
    parser.add_argument("--g2-acceptance")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    parser.add_argument("--record-resolved-paths", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"analyze", "all"} and (
        args.bootstrap_draws != 2000 or args.bootstrap_seed != 20260810
    ):
        raise ValueError(
            "analyze/all requires the frozen --bootstrap-draws 2000 "
            "and --bootstrap-seed 20260810"
        )
    if args.command in {"extract", "all"}:
        if not args.data_path or not args.raw_db:
            raise ValueError("extract/all requires --data-path and --raw-db or their environment aliases")
        extract(args)
    if args.command in {"analyze", "all"}:
        analyze(args)


if __name__ == "__main__":
    main()
