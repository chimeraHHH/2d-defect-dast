"""Revalidate live G1 artifacts before issuing the canonical acceptance receipt.

This collector is the final G1 trust boundary.  Upstream JSON is treated as an
index, never as proof: the collector reopens the source and repaired datasets,
G1A siblings, corrected pretraining asset, and every pilot output.  It runs
only from a clean pushed server commit with one explicitly assigned, allowed
GPU UUID.  The resulting receipt contains aliases and hashes, not private
filesystem paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.db import connect
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prm_g1_diagnose_geometry import (  # noqa: E402
    CT_UAE_SHA256,
    PREDICTION_ROUNDTRIP_ATOL_EV,
    EXCLUDED_GPU_UUID,
    LEGACY_FIRST32,
    LEGACY_TRAINING_COMMIT,
    LEGACY_PRETRAINED_SHA256,
    N_PERMUTATIONS,
    PERMUTATION_NAMES,
    RAW_DB_SHA256,
    SOURCE_DATA_SHA256,
    DiagnosticDataset,
    cell_obliquity,
    decision_delta,
    decision_maps,
    infer,
    gpu_identity,
    git_snapshot,
    load_archived_predictions,
    quantile_summary,
    sha256_file,
)
from scripts.prm_g1_rebuild_dataset import (  # noqa: E402
    EDGE_DIST_ATOL,
    graph_edge_records,
)
from scripts.prm_g1_freeze_pilots import (  # noqa: E402
    BASE_CONFIG_REPOSITORY_PATH,
    BASE_CONFIG_SHA256,
    DERIVED_SPLIT_REPOSITORY_PATH,
    EXPECTED_OUTPUT_DIRS,
    FREEZE_MANIFEST_REPOSITORY_PATH,
    GENERATED_CONFIG_REPOSITORY_DIR,
    JARVIS_SOURCE_SHA256,
    SOURCE_SPLIT_REPOSITORY_PATH,
    SOURCE_SPLIT_SHA256,
)
from scripts.prm_g1_rebuild_dataset import (  # noqa: E402
    EXPECTED_CANONICAL_ROWS,
    EXPECTED_CONTAINER_ROWS,
    EXPECTED_EXCLUDED_ROWS,
    PROTOCOL_SAMPLES_SHA256,
)
from scripts.prm_g1c_pretrain import EXPECTED_ROWS, TRAINING_CONFIG  # noqa: E402
from src.graph import (  # noqa: E402
    GRAPH_BUILDER_VERSION,
    MAX_TRIPLETS_PER_CENTRE,
    _pbc_distance_matrix,
    build_graph,
)
from src.dataset import CrystalGraphDataset, make_splits  # noqa: E402
from src.models.baseline import CrystalTransformer  # noqa: E402
from src.models.crystal_v2 import (  # noqa: E402
    ENV_ZERO_NEIGHBOR_CORRECTED,
    ENV_ZERO_NEIGHBOR_LEGACY,
    CrystalTransformerV2,
)
from src.prm_assets import load_pretrained_initialization  # noqa: E402
from src.prm_provenance import (  # noqa: E402
    config_sha256,
    validate_training_completion,
)
from src.splits import validate_split  # noqa: E402


GRAPH_FIELDS = {
    "numbers", "positions", "cell", "edge_index", "edge_dist",
    "edge_offset", "triplet_index", "angles", "dist_matrix",
}
EXPECTED_HOST = "WHUServer-L40S"
EXPECTED_ARMS = ("legacy_init", "no_pretrain", "corrected_pretrain")
EXPECTED_CLAIMS = {
    "legacy_init": "legacy-initialization-sensitivity only",
    "no_pretrain": "correctness-clean IMP2D graph pilot; no transfer claim",
    "corrected_pretrain": "end-to-end corrected lineage compatibility pilot",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
GRAPH_FLOAT_FIELDS = {
    "positions", "cell", "edge_dist", "edge_offset", "angles", "dist_matrix",
}
GRAPH_INTEGER_FIELDS = {"numbers", "edge_index", "triplet_index"}
LEGACY_CONFIG_SHA256 = (
    "04cdb90d48f8e55e0c15b92064ca1af4dc5d64dcc3db012c059e3ea58b642f8e",
    "4dbf11fb894a9d8eebeae2998fc036814b9ecaeca445a99c350f7d30acdd65df",
    "fe22352d83e3799b9fd0b16c192a276fdaec707a3713731864abec8c7bdba5a9",
    "16733028c59dde962287424279420cac8aa70ca2f05af6a91da52fb1ae67c120",
    "3177c6aaf7b831123193dd780755206e6fc5bd087563cb53d68e9888e4d9d732",
)
LEGACY_SPLIT_SHA256 = (
    "2a27fd4f3e862d2225048b6255ef3c5d9d2e7b88f326e9cd9bd3da5a0223895a",
    "162e736c1582b5e428b36bd53a6e3aa0b9d5106a5d1c2d1c04a20159d10b6c80",
    "9a6fa6c294a87736e8f125f9c1665029a67f358082773a4f2c136e184f340a12",
    "321d402f5b6a26bb39d4aab69710db6cb72fbfed3e6b50023c88e43ddb33c6fe",
    "878238053281de89abbd08d6cd91798429d6ad55c2d2e06cb52edc3afe48cbe3",
)
ARTIFACT_LEDGER_REPOSITORY_PATH = Path(
    "artifacts/prm_g1/g1_artifact_hash_ledger.json"
)
ARTIFACT_LEDGER_REVIEW_STATE = "reviewed_and_frozen"
ARTIFACT_LEDGER_TRUST_BOUNDARY = (
    "hashes only; no scientific artifact was deserialized; acceptance "
    "requires this exact ledger blob in a later clean pushed commit"
)
LEDGER_RECORD_FIELDS = {"alias", "sha256", "size_bytes"}


def strict_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_json(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        raise ValueError(f"{path.name} schema mismatch")
    return payload


def resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def resolve_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {resolved}")
    return resolved


def require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return observed


def _contains_private_path(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        return any(
            _contains_private_path(key) or _contains_private_path(value)
            for key, value in payload.items()
        )
    if isinstance(payload, (list, tuple)):
        return any(_contains_private_path(value) for value in payload)
    if not isinstance(payload, str):
        return False
    return (
        payload.startswith("/")
        or "/Users/" in payload
        or "/home/" in payload
        or "\\Users\\" in payload
    )


def _validate_ledger_record(
    record: Any, *, alias: str, label: str,
) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) or set(record) != LEDGER_RECORD_FIELDS:
        raise ValueError(f"{label} ledger record fields are not exact")
    if record.get("alias") != alias:
        raise ValueError(f"{label} ledger alias mismatch")
    if SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None:
        raise ValueError(f"{label} ledger SHA256 is invalid")
    size = record.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"{label} ledger size is invalid")
    return record


def load_artifact_ledger(
    path: Path, current: str,
) -> tuple[dict[str, Any], str, str]:
    """Open only a committed ledger blob, before scientific deserialization."""
    canonical = (ROOT / ARTIFACT_LEDGER_REPOSITORY_PATH).resolve()
    if path != canonical:
        raise ValueError("formal G1 artifact ledger path is repository-fixed")
    repository_path = ARTIFACT_LEDGER_REPOSITORY_PATH.as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", repository_path], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    if tracked.returncode != 0:
        raise ValueError("formal G1 artifact ledger is not Git tracked")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{repository_path}"], cwd=ROOT,
        capture_output=True, check=False,
    )
    working_bytes = path.read_bytes()
    if committed.returncode != 0 or working_bytes != committed.stdout:
        raise ValueError("working artifact ledger differs from its HEAD blob")
    ledger_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", repository_path],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    if ledger_commit != current:
        raise ValueError("collector must run from the pushed ledger review commit")
    payload = json.loads(working_bytes)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "created_at", "host_profile", "git", "review_state",
        "artifacts", "trust_boundary",
    }:
        raise ValueError("artifact ledger top-level fields are not exact")
    if (
        payload.get("schema_version") != "prm_g1_artifact_hash_ledger_v1"
        or payload.get("host_profile") != EXPECTED_HOST
        or payload.get("review_state") != ARTIFACT_LEDGER_REVIEW_STATE
        or payload.get("trust_boundary") != ARTIFACT_LEDGER_TRUST_BOUNDARY
        or _contains_private_path(payload)
    ):
        raise ValueError("artifact ledger policy fields are invalid")
    try:
        created_at = datetime.fromisoformat(str(payload["created_at"]))
    except ValueError as exc:
        raise ValueError("artifact ledger timestamp is invalid") from exc
    if created_at.tzinfo is None:
        raise ValueError("artifact ledger timestamp must be timezone-aware")
    source_commit = require_git(payload.get("git"), current, "artifact ledger source")
    if source_commit == current:
        raise ValueError("artifact ledger must be reviewed in a later commit")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "repaired_imp2d", "g1a", "corrected_jarvis",
        "corrected_pretraining", "pilots",
    }:
        raise ValueError("artifact ledger groups are not exact")
    repaired = artifacts["repaired_imp2d"]
    if not isinstance(repaired, Mapping) or set(repaired) != {"dataset", "receipt"}:
        raise ValueError("repaired IMP2D ledger fields are not exact")
    _validate_ledger_record(
        repaired["dataset"], alias="PRM_REPAIRED_DATASET_PATH",
        label="repaired IMP2D dataset",
    )
    _validate_ledger_record(
        repaired["receipt"], alias="IMP2D_REBUILD_RECEIPT",
        label="repaired IMP2D receipt",
    )

    g1a = artifacts["g1a"]
    if not isinstance(g1a, Mapping) or set(g1a) != {
        "summary", "predictions", "sample_diagnostics", "legacy_folds",
    }:
        raise ValueError("G1A ledger fields are not exact")
    _validate_ledger_record(g1a["summary"], alias="G1A_SUMMARY", label="G1A summary")
    _validate_ledger_record(
        g1a["predictions"], alias="G1A_PREDICTIONS", label="G1A predictions"
    )
    _validate_ledger_record(
        g1a["sample_diagnostics"], alias="G1A_SAMPLE_DIAGNOSTICS",
        label="G1A diagnostics",
    )
    folds = g1a["legacy_folds"]
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("G1A ledger must contain five legacy folds")
    for fold, record in enumerate(folds):
        if not isinstance(record, Mapping) or set(record) != {
            "fold", "split_id", "manifest", "checkpoint", "test_predictions",
        }:
            raise ValueError("legacy-fold ledger fields are not exact")
        if record["fold"] != fold or record["split_id"] != f"pair_cv5_f{fold}":
            raise ValueError("legacy-fold ledger order/identity mismatch")
        _validate_ledger_record(
            record["manifest"], alias=f"LEGACY_F{fold}_MANIFEST",
            label=f"legacy fold {fold} manifest",
        )
        _validate_ledger_record(
            record["checkpoint"], alias=f"LEGACY_F{fold}_CHECKPOINT",
            label=f"legacy fold {fold} checkpoint",
        )
        _validate_ledger_record(
            record["test_predictions"],
            alias=f"LEGACY_F{fold}_TEST_PREDICTIONS",
            label=f"legacy fold {fold} predictions",
        )

    jarvis = artifacts["corrected_jarvis"]
    if not isinstance(jarvis, Mapping) or set(jarvis) != {"dataset", "receipt"}:
        raise ValueError("corrected JARVIS ledger fields are not exact")
    _validate_ledger_record(
        jarvis["dataset"], alias="PRM_CORRECTED_JARVIS_DATA_PATH",
        label="corrected JARVIS dataset",
    )
    _validate_ledger_record(
        jarvis["receipt"], alias="JARVIS_REBUILD_RECEIPT",
        label="corrected JARVIS receipt",
    )

    pretraining = artifacts["corrected_pretraining"]
    pretraining_aliases = {
        "receipt": "G1C_RECEIPT",
        "corrected_asset": "G1C_CORRECTED_ASSET",
        "source_export": "G1C_SOURCE_EXPORT",
        "best_checkpoint": "G1C_BEST_CHECKPOINT",
        "metrics": "G1C_METRICS",
    }
    if not isinstance(pretraining, Mapping) or set(pretraining) != set(
        pretraining_aliases
    ):
        raise ValueError("corrected-pretraining ledger fields are not exact")
    for key, alias in pretraining_aliases.items():
        _validate_ledger_record(
            pretraining[key], alias=alias, label=f"corrected pretraining {key}"
        )

    pilots = artifacts["pilots"]
    pilot_aliases = {
        "receipt": "RECEIPT",
        "run_manifest": "MANIFEST",
        "metrics": "METRICS",
        "checkpoint": "CHECKPOINT",
        "split_indices": "SPLIT_INDICES",
        "validation_predictions": "VALIDATION_PREDICTIONS",
        "test_predictions": "TEST_PREDICTIONS",
    }
    if not isinstance(pilots, Mapping) or tuple(pilots) != EXPECTED_ARMS:
        raise ValueError("pilot ledger arms are not exact and ordered")
    for arm in EXPECTED_ARMS:
        records = pilots[arm]
        if not isinstance(records, Mapping) or set(records) != set(pilot_aliases):
            raise ValueError(f"{arm} pilot ledger fields are not exact")
        for key, suffix in pilot_aliases.items():
            _validate_ledger_record(
                records[key], alias=f"{arm.upper()}_{suffix}",
                label=f"{arm} pilot {key}",
            )
    return payload, source_commit, hashlib.sha256(working_bytes).hexdigest()


def require_ledger_file(
    record: Mapping[str, Any], path: Path, *, alias: str, label: str,
) -> None:
    _validate_ledger_record(record, alias=alias, label=label)
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"{label} size differs from reviewed artifact ledger")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} hash differs from reviewed artifact ledger")


def bind_artifact_ledger_files(
    ledger: Mapping[str, Any], files: Mapping[str, Path], dirs: Mapping[str, Path],
) -> None:
    """Hash every ledger-bound CLI artifact before any pickle/torch.load call."""
    artifacts = ledger["artifacts"]
    repaired = artifacts["repaired_imp2d"]
    require_ledger_file(
        repaired["dataset"], files["repaired_data"],
        alias="PRM_REPAIRED_DATASET_PATH", label="repaired IMP2D dataset",
    )
    require_ledger_file(
        repaired["receipt"], files["repaired_receipt"],
        alias="IMP2D_REBUILD_RECEIPT", label="repaired IMP2D receipt",
    )
    g1a = artifacts["g1a"]
    for key, file_key, alias in (
        ("summary", "g1a_summary", "G1A_SUMMARY"),
        ("predictions", "g1a_predictions", "G1A_PREDICTIONS"),
        ("sample_diagnostics", "g1a_diagnostics", "G1A_SAMPLE_DIAGNOSTICS"),
    ):
        require_ledger_file(
            g1a[key], files[file_key], alias=alias, label=f"G1A {key}"
        )
    for fold, record in enumerate(g1a["legacy_folds"]):
        run_dir = (
            dirs["legacy_run_root"] / "selected/g111/transfer"
            / f"pair_cv5_f{fold}" / "seed242"
        )
        for key, name, alias_suffix in (
            ("manifest", "run_manifest.json", "MANIFEST"),
            ("checkpoint", "best.pt", "CHECKPOINT"),
            ("test_predictions", "test_predictions.npz", "TEST_PREDICTIONS"),
        ):
            path = resolve_file(run_dir / name, f"legacy fold {fold} {key}")
            require_ledger_file(
                record[key], path, alias=f"LEGACY_F{fold}_{alias_suffix}",
                label=f"legacy fold {fold} {key}",
            )
    jarvis = artifacts["corrected_jarvis"]
    require_ledger_file(
        jarvis["dataset"], files["g1c_dataset"],
        alias="PRM_CORRECTED_JARVIS_DATA_PATH", label="corrected JARVIS dataset",
    )
    require_ledger_file(
        jarvis["receipt"], files["g1c_dataset_receipt"],
        alias="JARVIS_REBUILD_RECEIPT", label="corrected JARVIS receipt",
    )
    pretrain_dir = files["pretraining_receipt"].parent
    pretraining_paths = {
        "receipt": files["pretraining_receipt"],
        "corrected_asset": files["corrected_asset"],
        "source_export": resolve_file(
            pretrain_dir / "pretrained_embed.pt", "pretraining source export"
        ),
        "best_checkpoint": resolve_file(
            pretrain_dir / "best.pt", "pretraining best checkpoint"
        ),
        "metrics": resolve_file(pretrain_dir / "metrics.json", "pretraining metrics"),
    }
    if files["corrected_asset"] != pretrain_dir / "pretrained_embed_corrected.pt":
        raise ValueError("corrected pretrained asset is not the receipt sibling")
    for key, path in pretraining_paths.items():
        require_ledger_file(
            artifacts["corrected_pretraining"][key], path,
            alias=f"G1C_{key.upper()}", label=f"corrected pretraining {key}",
        )
    pilot_receipt_keys = {
        "legacy_init": "legacy_pilot_receipt",
        "no_pretrain": "no_pretrain_pilot_receipt",
        "corrected_pretrain": "corrected_pilot_receipt",
    }
    pilot_files = {
        "receipt": "g1_pilot_receipt.json",
        "run_manifest": "run_manifest.json",
        "metrics": "metrics.json",
        "checkpoint": "best.pt",
        "split_indices": "split_indices.npz",
        "validation_predictions": "val_predictions.npz",
        "test_predictions": "test_predictions.npz",
    }
    for arm in EXPECTED_ARMS:
        run_dir = dirs[arm]
        suffix = Path(EXPECTED_OUTPUT_DIRS[arm]).parts
        if tuple(run_dir.parts[-len(suffix):]) != suffix:
            raise ValueError(f"{arm} pilot directory is not canonical")
        expected_receipt = run_dir / pilot_files["receipt"]
        if files[pilot_receipt_keys[arm]] != expected_receipt:
            raise ValueError(f"{arm} pilot receipt is not its run sibling")
        for key, filename in pilot_files.items():
            path = resolve_file(run_dir / filename, f"{arm} pilot {key}")
            require_ledger_file(
                artifacts["pilots"][arm][key], path,
                alias=f"{arm.upper()}_{key.upper() if key != 'run_manifest' else 'MANIFEST'}",
                label=f"{arm} pilot {key}",
            )


def require_remote_ancestor(commit: str, current: str, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", str(commit)) is None:
        raise ValueError(f"{label} producer commit is invalid")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=ROOT, capture_output=True, check=False,
    )
    remote_refs = subprocess.run(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT,
        text=True, capture_output=True, check=False,
    ).stdout.splitlines()
    if ancestry.returncode != 0 or not any(ref.strip() for ref in remote_refs):
        raise ValueError(f"{label} producer commit is not a pushed ancestor")


def require_git(snapshot: Any, current: str, label: str) -> str:
    if not isinstance(snapshot, Mapping) or snapshot.get("dirty") is not False:
        raise ValueError(f"{label} receipt was not produced from a clean commit")
    commit = str(snapshot.get("commit", ""))
    require_remote_ancestor(commit, current, label)
    return commit


def gpu_inventory() -> dict[str, str]:
    result = subprocess.run(
        [
            "/usr/bin/nvidia-smi", "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=True,
    )
    inventory: dict[str, str] = {}
    for line in result.stdout.splitlines():
        uuid, name = (value.strip() for value in line.split(",", maxsplit=1))
        if uuid in inventory:
            raise ValueError("nvidia-smi returned a duplicate GPU UUID")
        inventory[uuid] = name
    if EXCLUDED_GPU_UUID not in inventory:
        raise ValueError("the frozen excluded GPU UUID is absent from inventory")
    return inventory


def require_execution_identity(
    payload: Mapping[str, Any], inventory: Mapping[str, str], label: str,
) -> dict[str, str]:
    if payload.get("host_profile") != EXPECTED_HOST:
        raise ValueError(f"{label} host profile mismatch")
    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping):
        environment = payload.get("environment", {})
        gpu = {
            "uuid": environment.get("gpu_uuid"),
            "name": environment.get("gpu_name"),
        }
    uuid, name = str(gpu.get("uuid", "")), str(gpu.get("name", ""))
    if (
        GPU_UUID_RE.fullmatch(uuid) is None
        or uuid == EXCLUDED_GPU_UUID
        or inventory.get(uuid) != name
    ):
        raise ValueError(f"{label} GPU identity is not an allowed live server GPU")
    return {"uuid": uuid, "name": name}


def exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
        except (TypeError, ValueError):
            return False
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            exact_equal(a, b) for a, b in zip(left, right)
        )
    try:
        if math.isnan(float(left)) and math.isnan(float(right)):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def require_close_payload(
    observed: Any, expected: Any, label: str, *, atol: float = 1.0e-12,
) -> None:
    """Compare a recomputed scientific payload without trusting its JSON copy."""
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValueError(f"{label} mapping fields differ from recomputation")
        for key in expected:
            require_close_payload(
                observed[key], expected[key], f"{label}.{key}", atol=atol
            )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(observed, (list, tuple)) or len(observed) != len(expected):
            raise ValueError(f"{label} sequence differs from recomputation")
        for index, (left, right) in enumerate(zip(observed, expected)):
            require_close_payload(left, right, f"{label}[{index}]", atol=atol)
        return
    if isinstance(expected, (int, float, np.integer, np.floating)) and not isinstance(
        expected, bool
    ):
        try:
            left, right = float(observed), float(expected)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} is not numeric") from exc
        if not math.isfinite(left) or not math.isclose(
            left, right, rel_tol=0.0, abs_tol=atol
        ):
            raise ValueError(f"{label} differs from recomputation")
        return
    if observed != expected:
        raise ValueError(f"{label} differs from recomputation")


def require_finite_tensor_mapping(
    payload: Any, expected: Mapping[str, torch.Tensor], label: str,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != set(expected):
        raise ValueError(f"{label} state-dict keys differ from the live model")
    for key, reference in expected.items():
        value = payload.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != tuple(reference.shape)
            or value.dtype != reference.dtype
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{label} tensor {key!r} is invalid")


def require_exact_record(
    observed: Any, expected: Any, label: str,
) -> None:
    """Reject record-level path/config overrides, even if hashes look valid."""
    if not exact_equal(observed, expected):
        raise ValueError(f"{label} differs from deterministic reconstruction")


def require_corrected_model_contract(
    payload: Mapping[str, Any], label: str,
) -> None:
    if payload.get("env_zero_neighbor_mode") != ENV_ZERO_NEIGHBOR_CORRECTED:
        raise ValueError(f"{label} uses the wrong zero-neighbour mode")
    if payload.get("model_source_sha256") != sha256_file(
        ROOT / "src/models/crystal_v2.py"
    ):
        raise ValueError(f"{label} uses the wrong model source")


def require_protocol_prediction_identity(
    indices: np.ndarray,
    targets: np.ndarray,
    protocol_rows: Sequence[Mapping[str, Any]],
    canonical_indices: Sequence[int],
    label: str,
) -> None:
    expected_indices = np.asarray(sorted(int(index) for index in canonical_indices))
    if not np.array_equal(indices, expected_indices):
        raise ValueError(f"{label} canonical membership/order mismatch")
    expected_targets = np.asarray(
        [float(protocol_rows[int(index)]["target_eV"]) for index in indices],
        dtype=np.float32,
    )
    if not np.array_equal(targets.astype(np.float32), expected_targets):
        raise ValueError(f"{label} immutable target mismatch")


def require_prediction_roundtrip(
    live_indices: np.ndarray,
    live_predictions: np.ndarray,
    live_targets: np.ndarray,
    archived_indices: np.ndarray,
    archived_predictions: np.ndarray,
    archived_targets: np.ndarray,
    label: str,
    *,
    atol: float = PREDICTION_ROUNDTRIP_ATOL_EV,
) -> float:
    live_order = np.argsort(live_indices)
    archived_order = np.argsort(archived_indices)
    if not np.array_equal(
        live_indices[live_order], archived_indices[archived_order]
    ):
        raise ValueError(f"{label} inference indices mismatch")
    if not np.allclose(
        live_targets[live_order], archived_targets[archived_order],
        rtol=0.0, atol=1.0e-6,
    ):
        raise ValueError(f"{label} inference targets mismatch")
    deltas = np.abs(
        live_predictions[live_order] - archived_predictions[archived_order]
    )
    maximum = float(np.max(deltas, initial=0.0))
    if maximum > atol:
        raise ValueError(f"{label} checkpoint/prediction roundtrip mismatch")
    return maximum


def require_exact_graph(
    observed: Mapping[str, Any], recomputed: Mapping[str, np.ndarray], label: str,
) -> None:
    for field in sorted(GRAPH_INTEGER_FIELDS | GRAPH_FLOAT_FIELDS):
        if field not in observed or field not in recomputed:
            raise ValueError(f"{label} lacks recomputed graph field {field}")
        left = np.asarray(observed[field])
        right = np.asarray(recomputed[field])
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not np.array_equal(left, right, equal_nan=True)
        ):
            raise ValueError(f"{label} field {field} differs from current build_graph")


def load_protocol_samples(path: Path) -> tuple[list[dict[str, Any]], set[int]]:
    expected_path = (ROOT / "artifacts/prm_protocol_v2/samples.csv").resolve()
    if path != expected_path or sha256_file(path) != PROTOCOL_SAMPLES_SHA256:
        raise ValueError("formal G1 acceptance requires the frozen protocol sample table")
    with path.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != EXPECTED_CONTAINER_ROWS:
        raise ValueError("protocol table row count differs from the frozen container")
    rows: list[dict[str, Any]] = []
    canonical: set[int] = set()
    for expected_index, row in enumerate(raw):
        index = int(row["sample_index"])
        if index != expected_index:
            raise ValueError("protocol sample indices are not the full ordered container")
        converted = {
            **row,
            "sample_index": index,
            "id": int(row["id"]),
            "target_eV": float(row["target_eV"]),
            "natoms": int(row["natoms"]),
        }
        if not math.isfinite(converted["target_eV"]):
            raise ValueError("protocol table contains a nonfinite target")
        if str(row.get("canonical_retained", "")).lower() == "true":
            canonical.add(index)
        rows.append(converted)
    if len(canonical) != EXPECTED_CANONICAL_ROWS:
        raise ValueError("protocol canonical membership count mismatch")
    return rows, canonical


def container_samples(blob: Any) -> tuple[list[Mapping[str, Any]], Mapping[str, Any] | None]:
    if isinstance(blob, list):
        return blob, None
    if isinstance(blob, Mapping) and isinstance(blob.get("data"), list):
        return blob["data"], blob
    raise TypeError("unsupported scientific dataset container")


def non_graph_payload(sample: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    ignored = set(GRAPH_FIELDS) | {"graph_builder_version"}
    if "pbc" not in source:
        ignored.add("pbc")
    return {key: value for key, value in sample.items() if key not in ignored}


def validate_graph_sample(
    sample: Mapping[str, Any], *, expected_pbc: Sequence[bool] | None,
    require_index_id: int | None,
) -> tuple[int, int]:
    if sample.get("graph_builder_version") != GRAPH_BUILDER_VERSION:
        raise ValueError("actual repaired row has the wrong graph-builder version")
    numbers = np.asarray(sample.get("numbers"), dtype=np.int64)
    positions = np.asarray(sample.get("positions"), dtype=float)
    cell = np.asarray(sample.get("cell"), dtype=float)
    if (
        numbers.ndim != 1 or len(numbers) < 1
        or positions.shape != (len(numbers), 3)
        or cell.shape != (3, 3)
        or not np.isfinite(positions).all()
        or not np.isfinite(cell).all()
    ):
        raise ValueError("actual repaired row has invalid atom geometry")
    if require_index_id is not None and int(
        sample.get("id", require_index_id)
    ) != require_index_id:
        raise ValueError("actual corrected JARVIS row order/id changed")
    pbc = np.asarray(sample.get("pbc"), dtype=bool)
    if pbc.shape != (3,) or (
        expected_pbc is not None
        and not np.array_equal(pbc, np.asarray(expected_pbc, dtype=bool))
    ):
        raise ValueError("actual repaired row has invalid PBC flags")
    edge_index = np.asarray(sample.get("edge_index"), dtype=np.int64)
    edge_dist = np.asarray(sample.get("edge_dist"), dtype=float)
    edge_offset = np.asarray(sample.get("edge_offset"), dtype=float)
    triplets = np.asarray(sample.get("triplet_index"), dtype=np.int64)
    angles = np.asarray(sample.get("angles"), dtype=float)
    distances = np.asarray(sample.get("dist_matrix"), dtype=float)
    if (
        edge_index.ndim != 2 or edge_index.shape[0] != 2
        or edge_dist.shape != (edge_index.shape[1],)
        or edge_offset.shape != (edge_index.shape[1], 3)
        or triplets.ndim != 2 or triplets.shape[1] != 3
        or angles.shape != (len(triplets),)
        or distances.shape != (len(numbers), len(numbers))
        or not np.isfinite(edge_dist).all()
        or not np.isfinite(edge_offset).all()
        or not np.isfinite(angles).all()
        or not np.isfinite(distances).all()
    ):
        raise ValueError("actual repaired row graph schema is invalid")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(numbers)):
        raise ValueError("actual repaired edge index is out of range")
    if triplets.size and (triplets.min() < 0 or triplets.max() >= len(numbers)):
        raise ValueError("actual repaired triplet index is out of range")
    if (
        (
            edge_dist.size
            and (edge_dist.min() <= 0.0 or edge_dist.max() > 5.0 + 1.0e-6)
        )
        or not np.array_equal(edge_offset, np.rint(edge_offset))
        or (
            angles.size
            and (angles.min() < -1.0e-6 or angles.max() > math.pi + 1.0e-6)
        )
        or (distances.size and distances.min() < 0.0)
        # Exact-MIC distances are symmetric in exact arithmetic; float32
        # storage quantizes d[i,j] and d[j,i] independently, and one ULP at
        # the >20 A distances of long-c-axis JARVIS bulk cells is 1.9e-6,
        # above a purely absolute 1e-6 bound.  rtol=1e-6 is ~8 float32 ULPs.
        or not np.allclose(distances, distances.T, atol=1.0e-6, rtol=1.0e-6)
    ):
        raise ValueError("actual repaired graph violates cutoff/offset/metric bounds")
    centres = triplets[:, 1] if len(triplets) else np.empty(0, dtype=np.int64)
    counts = np.bincount(centres, minlength=len(numbers))
    if len(counts) and int(counts.max()) > MAX_TRIPLETS_PER_CENTRE:
        raise ValueError("actual repaired row exceeds the invariant triplet cap")
    if len(centres) and np.any(np.diff(centres) < 0):
        raise ValueError("actual repaired triplet centres are not canonical-order grouped")
    for centre in np.unique(centres):
        values = angles[centres == centre]
        if len(values) > 1 and np.any(np.diff(values) < 0.0):
            raise ValueError("actual repaired triplet angles are not invariant-order sorted")
    edge_pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    if any(
        (int(centre), int(left)) not in edge_pairs
        or (int(centre), int(right)) not in edge_pairs
        for left, centre, right in triplets.tolist()
    ):
        raise ValueError("actual repaired triplet centre column is inconsistent with edges")
    if not np.allclose(np.diag(distances), 0.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("actual repaired dense distance diagonal is not zero")
    return int(edge_index.shape[1]), int(len(triplets))


def verify_rebuilt_container(
    source_path: Path, repaired_path: Path, *, expected_rows: int,
    expected_pbc: Sequence[bool] | None, require_index_ids: bool,
    raw_db_path: Path | None = None,
    protocol_rows: Sequence[Mapping[str, Any]] | None = None,
    compare_source_edges: bool = False,
) -> dict[str, Any]:
    with source_path.open("rb") as handle:
        source_blob = pickle.load(handle)
    with repaired_path.open("rb") as handle:
        repaired_blob = pickle.load(handle)
    source_rows, source_wrapper = container_samples(source_blob)
    repaired_rows, repaired_wrapper = container_samples(repaired_blob)
    if len(source_rows) != expected_rows or len(repaired_rows) != expected_rows:
        raise ValueError("source/repaired scientific container row count mismatch")
    if (source_wrapper is None) != (repaired_wrapper is None):
        raise ValueError("source/repaired scientific container type changed")
    if source_wrapper is not None and repaired_wrapper is not None:
        source_top = {key: value for key, value in source_wrapper.items() if key != "data"}
        repaired_top = {
            key: value for key, value in repaired_wrapper.items()
            if key not in {"data", "graph_builder"}
        }
        if not exact_equal(source_top, repaired_top):
            raise ValueError("repaired container changed source-level non-graph metadata")
        if repaired_wrapper.get("graph_builder", {}).get("version") != GRAPH_BUILDER_VERSION:
            raise ValueError("repaired container wrapper lacks the graph-builder version")
    total_edges = total_triplets = 0
    max_edge_dist_delta = 0.0
    source_total_edges = source_total_triplets = 0
    capped_centres = 0
    changed_mic_samples = 0
    max_legacy_to_exact_mic_delta = 0.0
    targets: list[float] = []
    ids: list[int] = []
    pbc_counts: dict[str, int] = defaultdict(int)
    raw_db = connect(str(raw_db_path)) if raw_db_path is not None else None
    for index, (source, repaired) in enumerate(zip(source_rows, repaired_rows)):
        if not isinstance(source, Mapping) or not isinstance(repaired, Mapping):
            raise TypeError("scientific dataset row is not a mapping")
        if require_index_ids and int(source.get("id", index)) != index:
            raise ValueError(f"source JARVIS row identity changed at index {index}")
        if not exact_equal(
            non_graph_payload(source, source), non_graph_payload(repaired, source)
        ):
            raise ValueError(f"repaired row {index} changed a non-graph field")
        numbers = np.asarray(source.get("numbers"), dtype=np.int64)
        positions = np.asarray(source.get("positions"), dtype=np.float64)
        cell = np.asarray(source.get("cell"), dtype=np.float64)
        if (
            numbers.ndim != 1
            or positions.shape != (len(numbers), 3)
            or cell.shape != (3, 3)
        ):
            raise ValueError(f"source row {index} geometry schema is invalid")
        if raw_db is not None:
            row_id = int(source.get("id", -1))
            raw_row = raw_db.get(id=row_id)
            raw_atoms = raw_row.toatoms()
            if (
                int(raw_row.id) != row_id
                or not np.array_equal(numbers, raw_atoms.get_atomic_numbers())
                or not np.allclose(
                    positions, raw_atoms.get_positions(), rtol=0.0, atol=2.0e-5
                )
                or not np.allclose(
                    cell, np.asarray(raw_atoms.get_cell()), rtol=0.0, atol=2.0e-5
                )
            ):
                raise ValueError(f"IMP2D row {index} differs from the frozen raw DB")
            row_pbc = np.asarray(raw_atoms.get_pbc(), dtype=bool)
        else:
            row_pbc = np.asarray(expected_pbc, dtype=bool)
        if row_pbc.shape != (3,):
            raise ValueError(f"row {index} has invalid authoritative PBC")
        pbc_counts["".join("1" if flag else "0" for flag in row_pbc)] += 1
        recomputed = build_graph(
            Atoms(
                numbers=numbers,
                positions=positions,
                cell=cell,
                pbc=row_pbc,
            ),
            cutoff=5.0,
        )
        require_exact_graph(repaired, recomputed, f"repaired row {index}")
        if not np.array_equal(
            np.asarray(repaired.get("pbc"), dtype=bool), row_pbc
        ):
            raise ValueError(f"repaired row {index} PBC differs from authority")
        edges, triplets = validate_graph_sample(
            repaired,
            expected_pbc=row_pbc,
            require_index_id=index if require_index_ids else None,
        )
        total_edges += edges
        total_triplets += triplets
        source_total_edges += int(np.asarray(source["edge_index"]).shape[1])
        source_total_triplets += int(len(np.asarray(source["triplet_index"])))
        degrees = np.bincount(
            np.asarray(repaired["edge_index"], dtype=np.int64)[0],
            minlength=len(numbers),
        )
        capped_centres += int(
            np.sum(degrees * np.maximum(degrees - 1, 0) > MAX_TRIPLETS_PER_CENTRE)
        )
        if compare_source_edges:
            # The IMP2D repair must preserve the local edge multiset; only
            # the enumeration order and float32 storage noise may change.
            # JARVIS callers keep this off because the legacy source edges
            # are wrong by design and are replaced, not preserved.
            source_edges, source_dists = graph_edge_records(source)
            repaired_edges, repaired_dists = graph_edge_records(repaired)
            if source_edges.shape != repaired_edges.shape or not np.array_equal(
                source_edges, repaired_edges
            ):
                raise ValueError(
                    f"row {index} edge topology differs between source and repaired"
                )
            edge_delta = float(
                np.max(np.abs(source_dists - repaired_dists), initial=0.0)
            )
            if edge_delta > EDGE_DIST_ATOL:
                raise ValueError(
                    f"row {index} edge distances differ by {edge_delta} A"
                )
            max_edge_dist_delta = max(max_edge_dist_delta, edge_delta)
        mic_delta = float(np.max(np.abs(
            np.asarray(source["dist_matrix"], dtype=np.float64)
            - np.asarray(repaired["dist_matrix"], dtype=np.float64)
        )))
        changed_mic_samples += int(mic_delta > 1.0e-6)
        max_legacy_to_exact_mic_delta = max(
            max_legacy_to_exact_mic_delta, mic_delta
        )
        target = float(repaired.get("target", float("nan")))
        if not math.isfinite(target):
            raise ValueError("repaired row target is nonfinite")
        targets.append(target)
        ids.append(int(repaired.get("id", -1)))
        if protocol_rows is not None:
            protocol = protocol_rows[index]
            if (
                int(protocol["sample_index"]) != index
                or int(protocol["id"]) != int(source.get("id", -1))
                or int(protocol["natoms"]) != len(numbers)
                or not math.isclose(
                    float(protocol["target_eV"]), target,
                    rel_tol=0.0, abs_tol=1.0e-12,
                )
            ):
                raise ValueError(f"IMP2D row {index} differs from protocol identity/target")
    return {
        "rows": expected_rows,
        "total_edges": total_edges,
        "total_triplets": total_triplets,
        "source_total_edges": source_total_edges,
        "source_total_triplets": source_total_triplets,
        "capped_centres": capped_centres,
        "changed_mic_samples": changed_mic_samples,
        "max_legacy_to_exact_mic_delta_A": max_legacy_to_exact_mic_delta,
        "max_edge_dist_delta_A": (
            max_edge_dist_delta if compare_source_edges else None
        ),
        "targets": np.asarray(targets, dtype=float),
        "ids": np.asarray(ids, dtype=np.int64),
        "pbc_patterns": dict(sorted(pbc_counts.items())),
        "current_build_graph_recomputed_for_all_rows": True,
    }


def verify_repaired_imp2d(
    receipt_path: Path, source_path: Path, repaired_path: Path,
    raw_db_path: Path, protocol_path: Path,
    protocol_rows: Sequence[Mapping[str, Any]], current: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = load_json(receipt_path, "prm_g1_repaired_dataset_v1")
    require_git(receipt.get("git"), current, "IMP2D rebuild")
    if (
        receipt.get("host_profile") != EXPECTED_HOST
        or receipt.get("inputs", {}).get("dataset") != {
            "alias": "PRM_DATASET_PATH", "sha256": SOURCE_DATA_SHA256
        }
        or receipt.get("inputs", {}).get("raw_db") != {
            "alias": "PRM_RAW_DB_PATH", "sha256": RAW_DB_SHA256
        }
        or receipt.get("inputs", {}).get("protocol_samples") != {
            "repository_path": "artifacts/prm_protocol_v2/samples.csv",
            "sha256": PROTOCOL_SAMPLES_SHA256,
        }
        or receipt.get("inputs", {}).get("dataset", {}).get("sha256")
        != SOURCE_DATA_SHA256
        or receipt.get("inputs", {}).get("raw_db", {}).get("sha256")
        != RAW_DB_SHA256
        or receipt.get("inputs", {}).get("protocol_samples", {}).get("sha256")
        != PROTOCOL_SAMPLES_SHA256
        or receipt.get("output", {}).get("sha256") != sha256_file(repaired_path)
        or receipt.get("output", {}).get("alias")
        != "PRM_REPAIRED_DATASET_PATH"
        or int(receipt.get("output", {}).get("size_bytes", -1))
        != repaired_path.stat().st_size
        or receipt.get("container", {}) != {
            "rows": EXPECTED_CONTAINER_ROWS,
            "canonical_rows": EXPECTED_CANONICAL_ROWS,
            "excluded_rows": EXPECTED_EXCLUDED_ROWS,
            "order_preserved": True,
            "non_graph_fields_preserved": True,
        }
        or receipt.get("graph_builder", {}).get("version")
        != GRAPH_BUILDER_VERSION
        or receipt.get("graph_builder", {}).get("source_sha256")
        != sha256_file(ROOT / "src/graph.py")
        or float(receipt.get("graph_builder", {}).get("cutoff_A", -1.0)) != 5.0
        or receipt.get("graph_builder", {}).get("exact_mic_backend")
        != "ase.geometry.find_mic"
        or receipt.get("graph_builder", {}).get("pbc_source")
        != "hash-verified raw IMP2D row after structure identity check"
        or float(receipt.get("graph_builder", {}).get(
            "structure_identity_atol", -1.0
        )) != 2.0e-5
        or int(receipt.get("graph_builder", {}).get(
            "triplet_cap_per_centre", -1
        )) != MAX_TRIPLETS_PER_CENTRE
        or receipt.get("graph_builder", {}).get("triplet_selection") != (
            "complete ordered-angle multiset; all retained for n<=32; "
            "midpoint order statistics floor((2q+1)n/(2*32)) for n>32"
        )
    ):
        raise ValueError("IMP2D rebuild receipt/live artifact contract failed")
    require_sha(source_path, SOURCE_DATA_SHA256, "frozen IMP2D source")
    require_sha(raw_db_path, RAW_DB_SHA256, "frozen IMP2D raw DB")
    require_sha(protocol_path, PROTOCOL_SAMPLES_SHA256, "frozen protocol samples")
    live = verify_rebuilt_container(
        source_path, repaired_path, expected_rows=EXPECTED_CONTAINER_ROWS,
        expected_pbc=None, require_index_ids=False,
        raw_db_path=raw_db_path, protocol_rows=protocol_rows,
        compare_source_edges=True,
    )
    expected_audit = {
        "pbc_patterns": live["pbc_patterns"],
        "total_edges": live["total_edges"],
        "total_triplets": live["total_triplets"],
        "capped_centres": live["capped_centres"],
        "samples_with_legacy_mic_delta_gt_1e-6": live["changed_mic_samples"],
        "max_legacy_to_exact_mic_delta_A": live[
            "max_legacy_to_exact_mic_delta_A"
        ],
        "max_edge_dist_delta_A": live["max_edge_dist_delta_A"],
        "edge_dist_atol_A": EDGE_DIST_ATOL,
    }
    require_close_payload(
        receipt.get("audit"), expected_audit, "IMP2D rebuild audit", atol=1.0e-12
    )
    return receipt, live


def verify_g1c_dataset(
    receipt_path: Path, source_path: Path, repaired_path: Path,
    current: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = load_json(receipt_path, "prm_g1c_jarvis_dataset_v1")
    require_git(receipt.get("git"), current, "G1C dataset rebuild")
    if (
        receipt.get("host_profile") != EXPECTED_HOST
        or receipt.get("input") != {
            "alias": "PRM_JARVIS_DATA_PATH", "sha256": JARVIS_SOURCE_SHA256
        }
        or receipt.get("input", {}).get("sha256") != JARVIS_SOURCE_SHA256
        or receipt.get("output", {}).get("sha256") != sha256_file(repaired_path)
        or receipt.get("output", {}).get("alias")
        != "PRM_CORRECTED_JARVIS_DATA_PATH"
        or int(receipt.get("output", {}).get("size_bytes", -1))
        != repaired_path.stat().st_size
        or receipt.get("container", {}) != {
            "rows": EXPECTED_ROWS,
            "order_preserved": True,
            "targets_and_non_graph_fields_preserved": True,
        }
        or receipt.get("graph_builder", {}).get("version")
        != GRAPH_BUILDER_VERSION
        or receipt.get("graph_builder", {}).get("source_sha256")
        != sha256_file(ROOT / "src/graph.py")
        or float(receipt.get("graph_builder", {}).get("cutoff_A", -1.0)) != 5.0
        or receipt.get("graph_builder", {}).get("pbc") != [True, True, True]
        or receipt.get("graph_builder", {}).get("exact_mic_backend")
        != "ase.geometry.find_mic"
        or int(receipt.get("graph_builder", {}).get(
            "triplet_cap_per_centre", -1
        )) != MAX_TRIPLETS_PER_CENTRE
        or int(receipt.get("graph_builder", {}).get(
            "triplet_centre_column", -1
        )) != 1
    ):
        raise ValueError("G1C dataset receipt/live artifact contract failed")
    require_sha(source_path, JARVIS_SOURCE_SHA256, "frozen JARVIS source")
    live = verify_rebuilt_container(
        source_path, repaired_path, expected_rows=EXPECTED_ROWS,
        expected_pbc=(True, True, True), require_index_ids=True,
    )
    require_close_payload(
        receipt.get("audit"),
        {
            "old_edges": live["source_total_edges"],
            "new_edges": live["total_edges"],
            "old_triplets": live["source_total_triplets"],
            "new_triplets": live["total_triplets"],
            "capped_centres": live["capped_centres"],
        },
        "G1C JARVIS rebuild audit",
    )
    return receipt, live


def verify_property_receipt(
    path: Path, current: str, inventory: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    receipt = load_json(path, "prm_g1_property_tests_v1")
    producer = require_git(receipt.get("git"), current, "G1 property tests")
    if producer != current:
        raise ValueError("G1 property suite must run on the collector commit")
    gpu = require_execution_identity(receipt, inventory, "G1 property tests")
    output = str(receipt.get("stdout_tail", ""))
    expected_sources = {
        "graph_correctness": {
            "repository_path": "tests/test_g1_graph_correctness.py",
            "sha256": sha256_file(ROOT / "tests/test_g1_graph_correctness.py"),
            "tests": 10,
        },
        "acceptance_contract": {
            "repository_path": "tests/test_prm_g1_acceptance_contract.py",
            "sha256": sha256_file(
                ROOT / "tests/test_prm_g1_acceptance_contract.py"
            ),
            "tests": 5,
        },
    }
    if (
        receipt.get("passed") is not True
        or int(receipt.get("return_code", -1)) != 0
        or receipt.get("graph_builder_version") != GRAPH_BUILDER_VERSION
        or receipt.get("graph_builder_source_sha256")
        != sha256_file(ROOT / "src/graph.py")
        or receipt.get("runner_source") != {
            "repository_path": "scripts/prm_g1_run_property_tests.py",
            "sha256": sha256_file(ROOT / "scripts/prm_g1_run_property_tests.py"),
        }
        or receipt.get("env_zero_neighbor_contract") != {
            "corrected": ENV_ZERO_NEIGHBOR_CORRECTED,
            "legacy_g1a_compatibility": ENV_ZERO_NEIGHBOR_LEGACY,
            "model_source_sha256": sha256_file(
                ROOT / "src/models/crystal_v2.py"
            ),
        }
        or receipt.get("test_sources") != expected_sources
        or int(receipt.get("test_count", -1)) != 15
        or "Ran 15 tests" not in output
        or "OK" not in output
        or any(token in output.lower() for token in ("skipped", "failed", "error"))
    ):
        raise ValueError("G1 property-test receipt/current source validation failed")
    result = subprocess.run(
        [
            sys.executable, "-m", "unittest", "-v",
            "tests.test_g1_graph_correctness",
            "tests.test_prm_g1_acceptance_contract",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    live_output = result.stdout + result.stderr
    if (
        result.returncode != 0
        or "Ran 15 tests" not in live_output
        or "OK" not in live_output
        or any(
            token in live_output.lower()
            for token in ("skipped", "failed", "error")
        )
    ):
        raise ValueError("live G1 graph/acceptance property suites failed")
    return receipt, gpu, {
        "tests": 15,
        "return_code": result.returncode,
        "output_sha256": hashlib.sha256(live_output.encode()).hexdigest(),
        "current_commit_reexecuted": True,
    }


def verify_g1a(
    summary_path: Path, predictions_path: Path, diagnostics_path: Path,
    legacy_run_root: Path, source_data_path: Path, repaired_data_path: Path,
    raw_db_path: Path,
    ct_uae_path: Path, legacy_pretrained_path: Path,
    protocol_rows: Sequence[Mapping[str, Any]], canonical_indices: set[int],
    current: str, inventory: Mapping[str, str], device: torch.device,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    if not (
        summary_path.parent == predictions_path.parent == diagnostics_path.parent
        and summary_path.name == "summary.json"
        and predictions_path.name == "predictions.npz"
        and diagnostics_path.name == "sample_diagnostics.csv"
    ):
        raise ValueError("G1A summary/prediction/diagnostic files are not siblings")
    summary = load_json(summary_path, "prm_g1a_geometry_diagnostic_v1")
    require_git(summary.get("git"), current, "G1A diagnostic")
    gpu = require_execution_identity(summary, inventory, "G1A diagnostic")
    # The identity arm imposes the archived (i, j, shift) enumeration because
    # the dataset was built with a pre-3.28 ASE; the shifted-sample count is a
    # measured quantity, so it is bounds-checked rather than compared to a
    # constant.  The remaining scope fields stay an exact frozen contract.
    scope = dict(summary.get("scope") or {})
    identity_enumeration = scope.pop("identity_enumeration", None)
    if (
        not isinstance(identity_enumeration, Mapping)
        or identity_enumeration.get("imposed_from_archived_edge_sequence")
        is not True
        or not isinstance(
            identity_enumeration.get("natural_order_differs_samples"), int
        )
        or not (
            0
            <= identity_enumeration["natural_order_differs_samples"]
            <= EXPECTED_CONTAINER_ROWS
        )
        or identity_enumeration.get("reason") != (
            "the dataset was built with a pre-3.28 ASE whose neighbour "
            "enumeration orders the identical edge set differently from "
            "the current ASE; the archived (i, j, shift) sequence is "
            "imposed for the identity arm so the order-dependent legacy "
            "triplet selection reproduces the archived model inputs "
            "bit-for-bit, and permuted variants keep the natural "
            "current-ASE enumeration as one more arbitrary ordering"
        )
        or set(identity_enumeration) != {
            "imposed_from_archived_edge_sequence",
            "natural_order_differs_samples",
            "reason",
        }
    ):
        raise ValueError("G1A identity-enumeration scope contract failed")
    if (
        scope != {
            "legacy_graph": LEGACY_FIRST32,
            "legacy_env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_LEGACY,
            "legacy_inference_batch_size": 64,
            "permutations": list(PERMUTATION_NAMES),
            "exact_mic_intervention": (
                "dense radial matrix only; legacy local graph retained"
            ),
            "interpretation": (
                "diagnostic sensitivity only; not a formal invariance waiver"
            ),
        }
        or
        summary.get("implementation", {}).get("diagnostic_sha256")
        != sha256_file(ROOT / "scripts/prm_g1_diagnose_geometry.py")
        or summary.get("implementation", {}).get("graph_sha256")
        != sha256_file(ROOT / "src/graph.py")
        or summary.get("implementation", {}).get("model_sha256")
        != sha256_file(ROOT / "src/models/crystal_v2.py")
        or summary.get("inputs", {}).get("dataset", {}).get("sha256")
        != SOURCE_DATA_SHA256
        or summary.get("inputs", {}).get("raw_db", {}).get("sha256")
        != RAW_DB_SHA256
        or summary.get("inputs", {}).get("ct_uae", {}).get("sha256")
        != CT_UAE_SHA256
        or summary.get("coverage") != {
            "canonical_rows": EXPECTED_CANONICAL_ROWS, "folds": 5
        }
        or summary.get("outputs", {}).get("predictions_sha256")
        != sha256_file(predictions_path)
        or summary.get("outputs", {}).get("sample_diagnostics_sha256")
        != sha256_file(diagnostics_path)
    ):
        raise ValueError("G1A summary/live sibling contract failed")
    with np.load(predictions_path, allow_pickle=False) as archive:
        required = {
            "schema_version", "indices", "targets", "baseline",
            "identity_rebuild", "exact_mic", "permutations",
            "permutation_names",
        }
        if set(archive.files) != required:
            raise ValueError("G1A prediction sibling schema mismatch")
        indices = np.asarray(archive["indices"], dtype=np.int64)
        targets = np.asarray(archive["targets"], dtype=float)
        baseline = np.asarray(archive["baseline"], dtype=float)
        identity = np.asarray(archive["identity_rebuild"], dtype=float)
        exact_mic = np.asarray(archive["exact_mic"], dtype=float)
        arrays = [targets, baseline, identity, exact_mic]
        permutations = np.asarray(archive["permutations"], dtype=float)
        names = tuple(str(value) for value in archive["permutation_names"].tolist())
        if (
            str(archive["schema_version"].item()) != "prm_g1a_predictions_v1"
            or indices.shape != (10_224,)
            or len(np.unique(indices)) != EXPECTED_CANONICAL_ROWS
            or any(array.shape != indices.shape or not np.isfinite(array).all()
                   for array in arrays)
            or permutations.shape != (10_224, N_PERMUTATIONS)
            or not np.isfinite(permutations).all()
            or names != PERMUTATION_NAMES
        ):
            raise ValueError("G1A prediction sibling coverage/content mismatch")
    require_protocol_prediction_identity(
        indices, targets, protocol_rows, sorted(canonical_indices),
        "G1A predictions",
    )

    base = CrystalGraphDataset(source_data_path)
    if len(base) != EXPECTED_CONTAINER_ROWS:
        raise ValueError("G1A live source dataset row count changed")
    with source_data_path.open("rb") as handle:
        source_rows, _ = container_samples(pickle.load(handle))
    with repaired_data_path.open("rb") as handle:
        repaired_rows, _ = container_samples(pickle.load(handle))
    raw_db = connect(str(raw_db_path))
    pbc_by_index: dict[int, np.ndarray] = {}
    raw_geometry_by_index: dict[int, dict[str, np.ndarray]] = {}
    exact_distances: dict[int, np.ndarray] = {}
    for index, source in enumerate(source_rows):
        raw_row = raw_db.get(id=int(source["id"]))
        raw_atoms = raw_row.toatoms()
        raw_numbers = raw_atoms.get_atomic_numbers().astype(np.int64)
        raw_positions = raw_atoms.get_positions().astype(np.float64)
        raw_cell = np.asarray(raw_atoms.get_cell(), dtype=np.float64)
        raw_pbc = np.asarray(raw_atoms.get_pbc(), dtype=bool)
        if (
            int(raw_row.id) != int(source["id"])
            or not np.array_equal(
                np.asarray(source["numbers"], dtype=np.int64), raw_numbers
            )
            or not np.allclose(
                np.asarray(source["positions"], dtype=np.float64), raw_positions,
                rtol=0.0, atol=2.0e-5,
            )
            or not np.allclose(
                np.asarray(source["cell"], dtype=np.float64), raw_cell,
                rtol=0.0, atol=2.0e-5,
            )
            or not np.array_equal(
                np.asarray(repaired_rows[index]["pbc"], dtype=bool), raw_pbc
            )
        ):
            raise ValueError(f"G1A raw DB identity/PBC mismatch at row {index}")
        pbc_by_index[index] = raw_pbc
        raw_geometry_by_index[index] = {
            "numbers": raw_numbers,
            "positions": raw_positions,
            "cell": raw_cell,
        }
        exact_distances[index] = _pbc_distance_matrix(
            np.asarray(source["positions"]),
            np.asarray(source["cell"]),
            raw_pbc,
        ).astype(np.float32)
        if not np.array_equal(
            exact_distances[index],
            np.asarray(repaired_rows[index]["dist_matrix"], dtype=np.float32),
        ):
            raise ValueError(
                f"G1A exact-distance replay differs from repaired row {index}"
            )
    archived_identity_full = np.full(EXPECTED_CONTAINER_ROWS, np.nan)
    archived_exact_full = np.full(EXPECTED_CONTAINER_ROWS, np.nan)
    archived_permutations_full = np.full(
        (EXPECTED_CONTAINER_ROWS, N_PERMUTATIONS), np.nan
    )
    archived_targets_full = np.full(EXPECTED_CONTAINER_ROWS, np.nan)
    archived_identity_full[indices] = identity
    archived_exact_full[indices] = exact_mic
    archived_permutations_full[indices] = permutations
    archived_targets_full[indices] = targets
    live_baseline = np.full(EXPECTED_CONTAINER_ROWS, np.nan, dtype=float)
    fold_coverage: set[int] = set()
    checkpoint_receipts: list[dict[str, Any]] = []
    roundtrip_deltas: list[float] = []
    intervention_roundtrip_maxima = {
        "identity_rebuild": 0.0,
        "exact_mic": 0.0,
        **{name: 0.0 for name in PERMUTATION_NAMES},
    }
    checkpoints = summary.get("inputs", {}).get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 5:
        raise ValueError("G1A must bind exactly five legacy checkpoints")
    for fold in range(5):
        split_id = f"pair_cv5_f{fold}"
        config_path = ROOT / (
            f"configs/prm/promoted/g111/transfer/{split_id}_seed242.yaml"
        )
        split_path = ROOT / f"artifacts/prm_protocol_v2/splits/{split_id}.json"
        require_sha(config_path, LEGACY_CONFIG_SHA256[fold], f"legacy config {fold}")
        require_sha(split_path, LEGACY_SPLIT_SHA256[fold], f"legacy split {fold}")
        expected_config = yaml.safe_load(config_path.read_text())
        split = json.loads(split_path.read_text())
        validate_split(split, EXPECTED_CONTAINER_ROWS)
        test_indices = [int(index) for index in split["test"]]
        if (
            split.get("schema_version") != "prm_split_v1"
            or split.get("split_id") != split_id
            or split.get("data_sha256") != SOURCE_DATA_SHA256
            or fold_coverage.intersection(test_indices)
        ):
            raise ValueError(f"legacy split contract failed for {split_id}")
        fold_coverage.update(test_indices)

        run_dir = legacy_run_root / "selected/g111/transfer" / split_id / "seed242"
        manifest_path = resolve_file(run_dir / "run_manifest.json", "legacy manifest")
        checkpoint_path = resolve_file(run_dir / "best.pt", "legacy checkpoint")
        archived_path = resolve_file(
            run_dir / "test_predictions.npz", "legacy predictions"
        )
        manifest = json.loads(manifest_path.read_text())
        validate_training_completion(manifest, manifest_path, verify_outputs=True)
        if (
            manifest.get("schema_version") != "prm_run_manifest_v1"
            or manifest.get("git") != {
                "commit": LEGACY_TRAINING_COMMIT,
                "dirty": False,
                "status_porcelain": [],
            }
            or manifest.get("config") != expected_config
            or manifest.get("config_sha256") != config_sha256(expected_config)
            or manifest.get("data", {}).get("data_sha256") != SOURCE_DATA_SHA256
            or int(manifest.get("data", {}).get("size_bytes", -1))
            != source_data_path.stat().st_size
            or manifest.get("split", {}).get("split_id") != split_id
            or manifest.get("split", {}).get("sha256") != LEGACY_SPLIT_SHA256[fold]
            or manifest.get("split", {}).get("counts") != {
                key: len(split[key]) for key in ("train", "val", "test")
            }
            or int(manifest.get("seed", -1)) != 242
        ):
            raise ValueError(f"legacy manifest contract failed for {split_id}")
        require_remote_ancestor(LEGACY_TRAINING_COMMIT, current, "legacy training")
        expected_assets = {
            "ct_uae": CT_UAE_SHA256,
            "pretrained_embed": LEGACY_PRETRAINED_SHA256,
        }
        assets = manifest.get("assets")
        if not isinstance(assets, Mapping) or set(assets) != set(expected_assets):
            raise ValueError(f"legacy asset inventory failed for {split_id}")
        for asset_name, expected_sha in expected_assets.items():
            asset_record = assets[asset_name]
            asset_path = resolve_file(
                Path(str(asset_record.get("path", ""))),
                f"legacy {asset_name}",
            )
            if (
                asset_record.get("sha256") != expected_sha
                or asset_record.get("expected_sha256") != expected_sha
                or int(asset_record.get("size_bytes", -1)) != asset_path.stat().st_size
                or sha256_file(asset_path) != expected_sha
            ):
                raise ValueError(f"legacy asset binding failed for {split_id}")
        if (
            sha256_file(ct_uae_path) != CT_UAE_SHA256
            or sha256_file(legacy_pretrained_path) != LEGACY_PRETRAINED_SHA256
        ):
            raise ValueError("G1A command assets differ from frozen hashes")
        expected_runtime = deepcopy(expected_config)
        expected_runtime["model_kwargs"]["ct_uae_path"] = assets["ct_uae"]["path"]
        expected_runtime["pretrained_embed"] = assets["pretrained_embed"]["path"]
        require_exact_record(
            manifest.get("runtime_config"), expected_runtime,
            f"legacy runtime config {split_id}",
        )
        if manifest.get("runtime_config_sha256") != config_sha256(expected_runtime):
            raise ValueError(f"legacy runtime config hash failed for {split_id}")

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
            "model", "normalizer", "config", "epoch", "best_val_mae"
        }:
            raise ValueError(f"legacy checkpoint schema failed for {split_id}")
        if (
            checkpoint.get("config") != expected_runtime
            or not (1 <= int(checkpoint.get("epoch", -1)) <= 150)
            or not math.isclose(
                float(checkpoint.get("best_val_mae", float("nan"))),
                float(manifest["metrics"]["best_val_mae"]),
                rel_tol=1.0e-7, abs_tol=1.0e-8,
            )
        ):
            raise ValueError(f"legacy checkpoint metadata failed for {split_id}")
        normalizer = checkpoint.get("normalizer")
        if (
            not isinstance(normalizer, Mapping)
            or set(normalizer) not in ({"mean", "std"}, {"mean", "std", "transform"})
            or not math.isfinite(float(normalizer.get("mean", float("nan"))))
            or not math.isfinite(float(normalizer.get("std", float("nan"))))
            or float(normalizer["std"]) <= 0.0
            or normalizer.get("transform", "none") not in {"none", "log"}
        ):
            raise ValueError(f"legacy normalizer failed for {split_id}")
        kwargs = deepcopy(expected_config["model_kwargs"])
        kwargs["ct_uae_path"] = str(ct_uae_path)
        kwargs["env_zero_neighbor_mode"] = ENV_ZERO_NEIGHBOR_LEGACY
        model = CrystalTransformerV2(**kwargs).to(device)
        require_finite_tensor_mapping(
            checkpoint.get("model"), model.state_dict(),
            f"legacy checkpoint state {split_id}",
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        live_i, live_p, live_t = infer(
            model, Subset(base, test_indices), normalizer, device, 64
        )
        archived_i, archived_p, archived_t = load_archived_predictions(
            archived_path, split_id
        )
        maximum = require_prediction_roundtrip(
            live_i, live_p, live_t,
            archived_i, archived_p, archived_t,
            f"legacy {split_id}", atol=PREDICTION_ROUNDTRIP_ATOL_EV,
        )
        archived_protocol_targets = np.asarray(
            [float(protocol_rows[int(index)]["target_eV"]) for index in archived_i],
            dtype=np.float32,
        )
        if (
            set(archived_i.tolist()) != set(test_indices)
            or not np.array_equal(
                archived_t.astype(np.float32), archived_protocol_targets
            )
        ):
            raise ValueError(f"legacy archived target/split failed for {split_id}")
        live_baseline[archived_i] = archived_p
        live_order = np.argsort(live_i)
        archived_order = np.argsort(archived_i)
        roundtrip_deltas.extend(
            np.abs(
                live_p[live_order] - archived_p[archived_order]
            ).tolist()
        )
        expected_i = np.asarray(test_indices, dtype=np.int64)
        expected_t = archived_targets_full[expected_i]
        identity_live = infer(
            model,
            DiagnosticDataset(
                base, test_indices, pbc_by_index, raw_geometry_by_index,
                permutation_variant=-1,
            ),
            normalizer, device, 64,
        )
        intervention_roundtrip_maxima["identity_rebuild"] = max(
            intervention_roundtrip_maxima["identity_rebuild"],
            require_prediction_roundtrip(
                *identity_live,
                expected_i, archived_identity_full[expected_i], expected_t,
                f"G1A identity {split_id}", atol=PREDICTION_ROUNDTRIP_ATOL_EV,
            ),
        )
        exact_live = infer(
            model,
            DiagnosticDataset(
                base, test_indices, pbc_by_index, raw_geometry_by_index,
                exact_distances=exact_distances,
            ),
            normalizer, device, 64,
        )
        intervention_roundtrip_maxima["exact_mic"] = max(
            intervention_roundtrip_maxima["exact_mic"],
            require_prediction_roundtrip(
                *exact_live,
                expected_i, archived_exact_full[expected_i], expected_t,
                f"G1A exact MIC {split_id}", atol=PREDICTION_ROUNDTRIP_ATOL_EV,
            ),
        )
        for variant, name in enumerate(PERMUTATION_NAMES):
            permutation_live = infer(
                model,
                DiagnosticDataset(
                    base, test_indices, pbc_by_index, raw_geometry_by_index,
                    permutation_variant=variant,
                ),
                normalizer, device, 64,
            )
            intervention_roundtrip_maxima[name] = max(
                intervention_roundtrip_maxima[name],
                require_prediction_roundtrip(
                    *permutation_live,
                    expected_i,
                    archived_permutations_full[expected_i, variant],
                    expected_t,
                    f"G1A permutation {name} {split_id}", atol=PREDICTION_ROUNDTRIP_ATOL_EV,
                ),
            )
        record = {
            "fold": fold,
            "split_id": split_id,
            "seed": 242,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "manifest_sha256": sha256_file(manifest_path),
            "archived_predictions_sha256": sha256_file(archived_path),
            "training_commit": LEGACY_TRAINING_COMMIT,
        }
        require_exact_record(checkpoints[fold], record, f"G1A checkpoint record {fold}")
        checkpoint_receipts.append(record)
        if maximum > PREDICTION_ROUNDTRIP_ATOL_EV:
            raise ValueError("legacy checkpoint roundtrip exceeded hard maximum")
        del model, checkpoint
        torch.cuda.empty_cache()
    if fold_coverage != canonical_indices:
        raise ValueError("legacy pair-fold tests do not equal canonical membership")
    if not np.allclose(
        live_baseline[indices], baseline, rtol=0.0,
        atol=PREDICTION_ROUNDTRIP_ATOL_EV,
    ):
        raise ValueError("G1A baseline differs from live checkpoint inference")

    prediction_range = np.ptp(
        np.column_stack([baseline, permutations]), axis=1
    )
    baseline_mae = float(np.mean(np.abs(baseline - targets)))
    permutation_mae = np.mean(np.abs(permutations - targets[:, None]), axis=0)
    sample_rows = {
        int(row["sample_index"]): dict(row) for row in protocol_rows
    }
    reference_decisions = decision_maps(indices, baseline, sample_rows)
    permutation_decisions = [
        decision_delta(
            reference_decisions,
            decision_maps(indices, permutations[:, variant], sample_rows),
        )
        for variant in range(N_PERMUTATIONS)
    ]
    exact_decisions = decision_delta(
        reference_decisions, decision_maps(indices, exact_mic, sample_rows)
    )
    range_stats = quantile_summary(prediction_range)
    max_mae_change = float(np.max(np.abs(permutation_mae - baseline_mae)))
    max_class_flips = max(
        row["incorporation_class_flips"] for row in permutation_decisions
    )
    max_site_flips = max(
        max(row["global_site_flips"], row["within_class_site_flips"])
        for row in permutation_decisions
    )
    empirical_pass = bool(
        range_stats["p99"] < 0.01
        and range_stats["max"] < 0.05
        and max_mae_change < 0.01
        and max_class_flips == 0
        and max_site_flips == 0
    )

    ranges = dict(zip(indices.tolist(), prediction_range.tolist()))
    exact_deltas = dict(
        zip(indices.tolist(), np.abs(exact_mic - baseline).tolist())
    )
    recomputed_diagnostics: list[dict[str, Any]] = []
    for index, (source, repaired) in enumerate(zip(source_rows, repaired_rows)):
        delta = np.abs(
            np.asarray(source["dist_matrix"], dtype=float)
            - np.asarray(repaired["dist_matrix"], dtype=float)
        )
        degrees = np.bincount(
            np.asarray(source["edge_index"], dtype=np.int64)[0],
            minlength=len(source["numbers"]),
        )
        obliquity, obliquity_bin = cell_obliquity(
            np.asarray(source["cell"]), np.asarray(repaired["pbc"], dtype=bool)
        )
        recomputed_diagnostics.append({
            "sample_index": index,
            "defecttype": str(source.get("metadata", {}).get("defecttype", "")),
            "obliquity": obliquity,
            "obliquity_bin": obliquity_bin,
            "legacy_cap_active": bool(
                np.any(degrees * np.maximum(degrees - 1, 0) > 32)
            ),
            "mic_max_delta_A": float(np.max(delta)),
            "mic_mean_delta_A": float(np.mean(delta)),
            "mic_pair_count_gt_1e-6": int(np.sum(delta > 1.0e-6)),
            "permutation_prediction_range_eV": ranges.get(index, ""),
            "exact_mic_prediction_delta_eV": exact_deltas.get(index, ""),
        })
    with diagnostics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_CONTAINER_ROWS:
        raise ValueError("G1A sample diagnostics row count mismatch")
    for expected, observed in zip(recomputed_diagnostics, rows):
        if int(observed.get("sample_index", -1)) != expected["sample_index"]:
            raise ValueError("G1A diagnostic order changed")
        for key in ("defecttype", "obliquity_bin"):
            if observed.get(key) != expected[key]:
                raise ValueError(f"G1A diagnostic {key} mismatch")
        if str(observed.get("legacy_cap_active", "")).lower() != str(
            expected["legacy_cap_active"]
        ).lower():
            raise ValueError("G1A diagnostic legacy-cap flag mismatch")
        for key in (
            "obliquity", "mic_max_delta_A", "mic_mean_delta_A",
            "mic_pair_count_gt_1e-6", "permutation_prediction_range_eV",
            "exact_mic_prediction_delta_eV",
        ):
            if expected[key] == "":
                if observed.get(key, "") != "":
                    raise ValueError(f"G1A noncanonical diagnostic {key} is not blank")
            elif not math.isclose(
                float(observed.get(key, float("nan"))), float(expected[key]),
                rel_tol=0.0, abs_tol=1.0e-12,
            ):
                raise ValueError(f"G1A diagnostic {key} mismatch")

    canonical_mic_rows = [
        recomputed_diagnostics[index] for index in sorted(canonical_indices)
    ]
    grouped: dict[str, dict[str, list[float]]] = {
        "obliquity_bin": defaultdict(list),
        "legacy_cap_active": defaultdict(list),
        "defecttype": defaultdict(list),
    }
    for row in canonical_mic_rows:
        for axis in grouped:
            grouped[axis][str(row[axis])].append(float(row["mic_max_delta_A"]))
    mic_strata = {
        axis: {
            label: {"n": len(values), **quantile_summary(np.asarray(values))}
            for label, values in values_by_label.items()
        }
        for axis, values_by_label in grouped.items()
    }
    expected_roundtrip = {
        "stored_graph_eV": quantile_summary(np.asarray(roundtrip_deltas)),
        "raw_identity_rebuild_eV": quantile_summary(np.abs(identity - baseline)),
        "hard_max_eV": PREDICTION_ROUNDTRIP_ATOL_EV,
    }
    expected_legacy = {
        "prediction_range_eV": range_stats,
        "baseline_mae_eV": baseline_mae,
        "permutation_mae_eV": permutation_mae.tolist(),
        "max_absolute_mae_change_eV": max_mae_change,
        "decision_deltas": permutation_decisions,
        "thresholds": {
            "prediction_range_p99_eV_lt": 0.01,
            "prediction_range_max_eV_lt": 0.05,
            "absolute_mae_change_eV_lt": 0.01,
            "class_and_site_flips_eq": 0,
        },
        "empirical_stability_pass": empirical_pass,
    }
    expected_mic = {
        "prediction_absolute_delta_eV": quantile_summary(
            np.abs(exact_mic - baseline)
        ),
        "mae_change_eV": float(
            np.mean(np.abs(exact_mic - targets)) - baseline_mae
        ),
        "decision_delta": exact_decisions,
        "distance_sample_summary_A": quantile_summary(np.asarray(
            [row["mic_max_delta_A"] for row in canonical_mic_rows]
        )),
        "distance_strata": mic_strata,
        "distance_scope": "10,224 protocol-canonical rows",
        "full_container_rows_audited": EXPECTED_CONTAINER_ROWS,
    }
    summary_roundtrip = summary.get("checkpoint_roundtrip")
    if not isinstance(summary_roundtrip, Mapping) or set(summary_roundtrip) != set(
        expected_roundtrip
    ):
        raise ValueError(
            "G1A checkpoint roundtrip mapping fields differ from recomputation"
        )
    # stored_graph_eV summarizes live-inference noise: the collector's own
    # re-inference is a fresh nondeterministic CUDA draw, so its quantiles
    # can only agree with the diagnostic's record to the measured noise
    # scale.  Both maxima are separately hard-gated below.  The identity
    # block is recomputed from the archived arrays and stays deterministic.
    require_close_payload(
        summary_roundtrip.get("stored_graph_eV"),
        expected_roundtrip["stored_graph_eV"],
        "G1A checkpoint roundtrip.stored_graph_eV",
        atol=PREDICTION_ROUNDTRIP_ATOL_EV,
    )
    require_close_payload(
        summary_roundtrip.get("raw_identity_rebuild_eV"),
        expected_roundtrip["raw_identity_rebuild_eV"],
        "G1A checkpoint roundtrip.raw_identity_rebuild_eV", atol=1.0e-10,
    )
    require_close_payload(
        summary_roundtrip.get("hard_max_eV"),
        expected_roundtrip["hard_max_eV"],
        "G1A checkpoint roundtrip.hard_max_eV", atol=1.0e-10,
    )
    require_close_payload(
        summary.get("legacy_under_permutations"), expected_legacy,
        "G1A legacy diagnostic", atol=1.0e-10,
    )
    require_close_payload(
        summary.get("old_to_exact_mic"), expected_mic,
        "G1A exact-MIC diagnostic", atol=1.0e-10,
    )
    # ``empirical_pass`` reports the LEGACY model's stability under the 16
    # permutations.  The measured outcome is False (range p99 0.46 eV, max
    # 8.8 eV, incorporation-class and site flips), which is a diagnostic
    # finding that motivates the repair rather than a defect of it; it is
    # recorded verbatim as ``g1a_empirical_stability_pass`` in the acceptance
    # payload and in the summary/live payload comparison above.  Acceptance
    # gates only on pipeline fidelity: both roundtrip maxima must sit under
    # the measured-noise bound.
    if (
        expected_roundtrip["stored_graph_eV"]["max"] > PREDICTION_ROUNDTRIP_ATOL_EV
        or expected_roundtrip["raw_identity_rebuild_eV"]["max"]
        > PREDICTION_ROUNDTRIP_ATOL_EV
    ):
        raise ValueError("G1A recomputed roundtrip gate failed")
    return summary, gpu, {
        "canonical_rows": EXPECTED_CANONICAL_ROWS,
        "folds_strict_loaded_and_reinferred": 5,
        "identity_exact_and_16_permutations_reinferred": True,
        "raw_db_reopened_for_geometry_and_pbc": True,
        "intervention_roundtrip_max_abs_delta_eV": dict(
            sorted(intervention_roundtrip_maxima.items())
        ),
        "checkpoint_roundtrip_max_eV": expected_roundtrip[
            "stored_graph_eV"
        ]["max"],
        "summary_and_csv_recomputed": True,
    }


def verify_pretraining(
    receipt_path: Path, corrected_dataset_path: Path,
    corrected_dataset_receipt_path: Path, corrected_asset_path: Path,
    ct_uae_path: Path, current: str, inventory: Mapping[str, str],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    receipt = load_json(receipt_path, "prm_g1c_pretraining_receipt_v1")
    producer = require_git(receipt.get("git"), current, "G1C pretraining")
    gpu = require_execution_identity(receipt, inventory, "G1C pretraining")
    parent = receipt_path.parent
    source_export = parent / "pretrained_embed.pt"
    checkpoint = parent / "best.pt"
    metrics_path = parent / "metrics.json"
    for path in (source_export, checkpoint, metrics_path, corrected_asset_path):
        if not path.is_file():
            raise FileNotFoundError(f"G1C output missing: {path.name}")
    if corrected_asset_path != (parent / "pretrained_embed_corrected.pt").resolve():
        raise ValueError("corrected pretraining asset is not the receipt sibling")
    inputs = receipt.get("input", {})
    outputs = receipt.get("outputs", {})
    if (
        inputs.get("dataset_sha256") != sha256_file(corrected_dataset_path)
        or inputs.get("dataset_receipt_sha256")
        != sha256_file(corrected_dataset_receipt_path)
        or inputs.get("frozen_source_dataset_sha256") != JARVIS_SOURCE_SHA256
        or inputs.get("graph_builder_version") != GRAPH_BUILDER_VERSION
        or inputs.get("rows") != EXPECTED_ROWS
        or inputs.get("ct_uae_sha256_for_dart_compatibility") != CT_UAE_SHA256
        or inputs.get("downstream_env_zero_neighbor_mode")
        != ENV_ZERO_NEIGHBOR_CORRECTED
        or inputs.get("downstream_model_source_sha256")
        != sha256_file(ROOT / "src/models/crystal_v2.py")
        or receipt.get("training_contract") != TRAINING_CONFIG
        or receipt.get("training_contract_sha256") != config_sha256(TRAINING_CONFIG)
        or receipt.get("split_contract") != {
            "algorithm": "python_random_shuffle_v1",
            "seed": 42,
            "train_rows": 15_921,
            "validation_rows": 1_990,
            "test_rows": 1_991,
        }
        or outputs.get("corrected_asset_sha256") != sha256_file(corrected_asset_path)
        or outputs.get("source_export_sha256") != sha256_file(source_export)
        or outputs.get("best_checkpoint_sha256") != sha256_file(checkpoint)
        or outputs.get("metrics_sha256") != sha256_file(metrics_path)
        or outputs.get("corrected_asset_alias")
        != "PRM_CORRECTED_PRETRAINED_PATH"
    ):
        raise ValueError("G1C pretraining receipt/live artifact contract failed")
    metrics = json.loads(metrics_path.read_text())
    for key in ("best_val_mae", "test_mae", "test_rmse"):
        if (
            not math.isfinite(float(metrics.get(key, float("nan"))))
            or not math.isclose(
                float(metrics[key]), float(receipt.get("metrics", {}).get(key, float("nan"))),
                rel_tol=0.0, abs_tol=1.0e-12,
            )
        ):
            raise ValueError("G1C metric receipt/live file mismatch")
    history = metrics.get("history")
    if (
        not isinstance(history, list)
        or len(history) != 30
        or [int(row.get("epoch", -1)) for row in history] != list(range(1, 31))
        or any(
            set(row) != {
                "epoch", "train_mae", "val_mae", "val_rmse", "lr", "best"
            }
            or not isinstance(row.get("best"), bool)
            or any(
                not math.isfinite(float(row.get(key, float("nan"))))
                for key in ("train_mae", "val_mae", "val_rmse", "lr")
            )
            for row in history
        )
    ):
        raise ValueError("G1C source-task history is incomplete/nonfinite")
    best_epoch = int(min(history, key=lambda row: row["val_mae"])["epoch"])
    best_so_far = float("inf")
    for row in history:
        expected_best = float(row["val_mae"]) < best_so_far
        if row["best"] is not expected_best:
            raise ValueError("G1C source-task history best flags are inconsistent")
        if expected_best:
            best_so_far = float(row["val_mae"])
    if (
        not math.isclose(
            float(metrics["best_val_mae"]),
            min(float(row["val_mae"]) for row in history),
            rel_tol=0.0, abs_tol=1.0e-12,
        )
        or receipt.get("training_audit") != {
            "epochs_completed": 30,
            "best_epoch": best_epoch,
            "finite_history": True,
        }
    ):
        raise ValueError("G1C source-task best/history receipt mismatch")

    reference_model = CrystalTransformer(**TRAINING_CONFIG["model_kwargs"])
    expected_config = dict(TRAINING_CONFIG)
    expected_config["data_path"] = str(corrected_dataset_path)
    expected_config["output_dir"] = str(parent)
    expected_n_params = sum(
        parameter.numel() for parameter in reference_model.parameters()
        if parameter.requires_grad
    )
    if int(metrics.get("n_params", -1)) != expected_n_params:
        raise ValueError("G1C source-task parameter count differs from live model")
    best = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(best, Mapping) or set(best) != {
        "model", "embed_weight", "embed_bias", "local_layers",
        "normalizer", "config", "epoch",
    }:
        raise ValueError("actual corrected source-task checkpoint schema is invalid")
    require_finite_tensor_mapping(
        best["model"], reference_model.state_dict(),
        "corrected source-task checkpoint",
    )
    reference_model.load_state_dict(best["model"], strict=True)
    if (
        best.get("config") != expected_config
        or int(best.get("epoch", -1)) != best_epoch
        or not isinstance(best.get("normalizer"), Mapping)
        or set(best["normalizer"]) != {"mean", "std"}
        or not math.isfinite(float(best["normalizer"].get("mean", float("nan"))))
        or not math.isfinite(float(best["normalizer"].get("std", float("nan"))))
        or float(best["normalizer"]["std"]) <= 0.0
    ):
        raise ValueError("corrected source-task checkpoint metadata is invalid")
    corrected_dataset = CrystalGraphDataset(corrected_dataset_path)
    if corrected_dataset.meta is not None:
        raise ValueError("G1C source task unexpectedly uses embedded split metadata")
    train_set, validation_set, test_set = make_splits(
        corrected_dataset, train_ratio=0.8, val_ratio=0.1, seed=42
    )
    if (
        len(train_set) != 15_921
        or len(validation_set) != 1_990
        or len(test_set) != 1_991
    ):
        raise ValueError("G1C deterministic source-task split counts changed")
    training_targets = torch.tensor(
        [
            corrected_dataset.data[int(index)]["target"]
            for index in train_set.indices
        ],
        dtype=torch.float32,
    )
    expected_normalizer = {
        "mean": float(training_targets.mean().item()),
        "std": float(training_targets.std().item()) + 1.0e-6,
    }
    require_close_payload(
        best["normalizer"], expected_normalizer,
        "G1C deterministic training normalizer", atol=1.0e-12,
    )
    reference_model = reference_model.to(device)
    live_val = infer(
        reference_model, validation_set, best["normalizer"], device, 64
    )
    live_test = infer(
        reference_model, test_set, best["normalizer"], device, 64
    )
    live_val_mae = float(np.mean(np.abs(live_val[1] - live_val[2])))
    live_test_mae = float(np.mean(np.abs(live_test[1] - live_test[2])))
    live_test_rmse = float(np.sqrt(np.mean((live_test[1] - live_test[2]) ** 2)))
    if (
        not np.array_equal(
            np.sort(live_val[0]), np.sort(np.asarray(validation_set.indices))
        )
        or not np.array_equal(
            np.sort(live_test[0]), np.sort(np.asarray(test_set.indices))
        )
        or not math.isclose(
            live_val_mae, float(metrics["best_val_mae"]),
            rel_tol=1.0e-7, abs_tol=1.0e-5,
        )
        or not math.isclose(
            live_test_mae, float(metrics["test_mae"]),
            rel_tol=1.0e-7, abs_tol=1.0e-5,
        )
        or not math.isclose(
            live_test_rmse, float(metrics["test_rmse"]),
            rel_tol=1.0e-7, abs_tol=1.0e-5,
        )
    ):
        raise ValueError("G1C live checkpoint val/test metrics mismatch")
    expected_local = {
        key.removeprefix("local_layers."): value
        for key, value in best["model"].items()
        if key.startswith("local_layers.")
    }
    if (
        not torch.equal(best["embed_weight"], best["model"]["embed.weight"])
        or not torch.equal(best["embed_bias"], best["model"]["embed.bias"])
        or not exact_equal(best["local_layers"], expected_local)
    ):
        raise ValueError("source-task exported tensors differ from best checkpoint")

    source_asset = torch.load(source_export, map_location="cpu", weights_only=True)
    expected_source_keys = {
        "embed_weight", "embed_bias", "local_layers",
        "source_dataset", "source_test_mae",
    }
    if (
        not isinstance(source_asset, Mapping)
        or set(source_asset) != expected_source_keys
        or source_asset.get("source_dataset") != "dft_3d_lite"
        or not math.isclose(
            float(source_asset.get("source_test_mae", float("nan"))),
            float(metrics["test_mae"]), rel_tol=0.0, abs_tol=1.0e-12,
        )
        or not torch.equal(source_asset["embed_weight"], best["embed_weight"])
        or not torch.equal(source_asset["embed_bias"], best["embed_bias"])
        or not exact_equal(source_asset["local_layers"], best["local_layers"])
    ):
        raise ValueError("source export is not derived from the best checkpoint")
    asset = torch.load(corrected_asset_path, map_location="cpu", weights_only=True)
    expected_asset_keys = expected_source_keys | {
        "schema_version", "source_dataset_sha256",
        "source_dataset_receipt_sha256", "graph_builder_version",
        "training_commit", "training_seed", "training_epochs",
        "downstream_env_zero_neighbor_mode", "downstream_model_source_sha256",
    }
    if (
        not isinstance(asset, Mapping)
        or set(asset) != expected_asset_keys
        or asset.get("schema_version") != "prm_g1c_pretrained_initialization_v1"
        or asset.get("source_dataset")
        != "jarvis_dft_3d_lite_corrected_g1c"
        or asset.get("source_dataset_sha256") != sha256_file(corrected_dataset_path)
        or asset.get("source_dataset_receipt_sha256")
        != sha256_file(corrected_dataset_receipt_path)
        or asset.get("graph_builder_version") != GRAPH_BUILDER_VERSION
        or asset.get("training_commit") != producer
        or asset.get("training_seed") != 42
        or asset.get("training_epochs") != 30
        or asset.get("downstream_env_zero_neighbor_mode")
        != ENV_ZERO_NEIGHBOR_CORRECTED
        or asset.get("downstream_model_source_sha256")
        != sha256_file(ROOT / "src/models/crystal_v2.py")
        or not torch.equal(asset["embed_weight"], source_asset["embed_weight"])
        or not torch.equal(asset["embed_bias"], source_asset["embed_bias"])
        or not exact_equal(asset["local_layers"], source_asset["local_layers"])
        or not math.isclose(
            float(asset.get("source_test_mae", float("nan"))),
            float(source_asset["source_test_mae"]), rel_tol=0.0, abs_tol=0.0,
        )
    ):
        raise ValueError("actual corrected pretraining asset lineage is invalid")
    compatibility_model = CrystalTransformerV2(
        atom_fea_len=9, hidden_dim=128, n_local_layers=3, n_global_layers=2,
        num_heads=4, rcut_local=5.0, dmax_global=12.0,
        defect_embedding=True, dropout=0.1, ct_uae_path=str(ct_uae_path),
        use_gated_pooling=True, use_env_enrichment=True,
        use_prenorm_local=True,
        env_zero_neighbor_mode=ENV_ZERO_NEIGHBOR_CORRECTED,
    )
    compatibility = load_pretrained_initialization(
        compatibility_model, corrected_asset_path
    )
    compatibility.pop("checkpoint_path", None)
    compatibility["env_zero_neighbor_mode"] = ENV_ZERO_NEIGHBOR_CORRECTED
    compatibility["model_source_sha256"] = sha256_file(
        ROOT / "src/models/crystal_v2.py"
    )
    require_exact_record(
        receipt.get("dart_g111_compatibility"), compatibility,
        "G1C live DART compatibility",
    )
    if (
        not torch.equal(
            compatibility_model.embed.weight[:, :9].detach().cpu(),
            asset["embed_weight"],
        )
        or not torch.equal(
            compatibility_model.embed.bias.detach().cpu(), asset["embed_bias"]
        )
    ):
        raise ValueError("G1C live compatibility load copied the wrong tensors")
    for key, tensor in asset["local_layers"].items():
        if not torch.equal(
            compatibility_model.local_layers.state_dict()[key].detach().cpu(),
            tensor,
        ):
            raise ValueError("G1C live compatibility loaded the wrong local tensor")
    del reference_model, compatibility_model, corrected_dataset
    torch.cuda.empty_cache()
    return receipt, gpu, {
        "epochs_revalidated": 30,
        "best_checkpoint_state_strict_loaded": True,
        "best_history_flags_recomputed": True,
        "validation_and_test_reinferred": True,
        "validation_mae_eV": live_val_mae,
        "test_mae_eV": live_test_mae,
        "test_rmse_eV": live_test_rmse,
        "source_export_matches_best_checkpoint": True,
        "dart_compatibility_reexecuted": True,
    }


def verify_prediction_archive(
    path: Path, *, split_id: str, partition: str,
    expected_indices: Sequence[int], targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema_version", "split_id", "split", "indices", "preds", "targets"}
        if set(archive.files) != required:
            raise ValueError(f"{partition} prediction archive schema mismatch")
        indices = np.asarray(archive["indices"], dtype=np.int64)
        predictions = np.asarray(archive["preds"], dtype=float)
        observed_targets = np.asarray(archive["targets"], dtype=float)
        expected = np.asarray(expected_indices, dtype=np.int64)
        if (
            str(archive["schema_version"].item()) != "prm_predictions_v1"
            or str(archive["split_id"].item()) != split_id
            or str(archive["split"].item()) != partition
            or len(np.unique(indices)) != len(indices)
            or set(indices.tolist()) != set(expected.tolist())
            or predictions.shape != indices.shape
            or observed_targets.shape != indices.shape
            or not np.isfinite(predictions).all()
            or not np.isfinite(observed_targets).all()
            or not np.allclose(
                observed_targets, targets[indices], rtol=0.0, atol=1.0e-6
            )
        ):
            raise ValueError(f"{partition} prediction archive content mismatch")
        return indices, predictions, observed_targets


def verify_freeze(
    repaired_receipt_path: Path, pretraining_receipt_path: Path,
    current: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    freeze_path = (ROOT / FREEZE_MANIFEST_REPOSITORY_PATH).resolve()
    split_path = (ROOT / DERIVED_SPLIT_REPOSITORY_PATH).resolve()
    freeze = load_json(freeze_path, "prm_g1_pilot_freeze_v1")
    require_remote_ancestor(str(freeze.get("git_commit", "")), current, "pilot freeze")
    source_split_path = (ROOT / SOURCE_SPLIT_REPOSITORY_PATH).resolve()
    base_config_path = (ROOT / BASE_CONFIG_REPOSITORY_PATH).resolve()
    require_sha(source_split_path, SOURCE_SPLIT_SHA256, "pilot source split")
    require_sha(base_config_path, BASE_CONFIG_SHA256, "pilot base config")
    repaired = load_json(repaired_receipt_path, "prm_g1_repaired_dataset_v1")
    corrected = load_json(
        pretraining_receipt_path, "prm_g1c_pretraining_receipt_v1"
    )
    source_split = json.loads(source_split_path.read_text())
    split = json.loads(split_path.read_text())
    validate_split(source_split, EXPECTED_CONTAINER_ROWS)
    validate_split(split, EXPECTED_CONTAINER_ROWS)
    if (
        source_split.get("schema_version") != "prm_split_v1"
        or source_split.get("split_id") != "pair_cv5_f0"
        or source_split.get("data_sha256") != SOURCE_DATA_SHA256
        or int(source_split.get("n_samples", -1)) != EXPECTED_CONTAINER_ROWS
        or source_split.get("counts") != {
            "excluded": 417, "test": 2045, "train": 6134, "val": 2045
        }
    ):
        raise ValueError("pilot source split differs from frozen contract")
    expected_split = deepcopy(source_split)
    expected_split["split_id"] = "g1_pair_cv5_f0_repaired_v1"
    expected_split["data_sha256"] = repaired["output"]["sha256"]
    expected_split.setdefault("metadata", {}).update({
        "g1_graph_builder_version": GRAPH_BUILDER_VERSION,
        "membership_source_split": "pair_cv5_f0",
        "membership_source_sha256": SOURCE_SPLIT_SHA256,
        "purpose": "bounded G1 numerical-stability pilot; not paper evidence",
    })
    require_exact_record(split, expected_split, "derived G1 split")

    base = yaml.safe_load(base_config_path.read_text())
    if not isinstance(base, dict):
        raise ValueError("pilot base config is not a mapping")
    base.update({
        "data_path": "data/processed/cleaned_dataset_g1_repaired_v1.pkl",
        "data_sha256": repaired["output"]["sha256"],
        "split_path": str(DERIVED_SPLIT_REPOSITORY_PATH),
        "epochs": 30,
        "seed": 242,
    })
    base.setdefault("model_kwargs", {})[
        "env_zero_neighbor_mode"
    ] = ENV_ZERO_NEIGHBOR_CORRECTED
    base["g1_pilot_contract"] = {
        "schema_version": "prm_g1_pilot_config_v1",
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "graph_builder_source_sha256": sha256_file(ROOT / "src/graph.py"),
        "env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_CORRECTED,
        "model_source_sha256": sha256_file(ROOT / "src/models/crystal_v2.py"),
        "repaired_dataset_receipt_sha256": sha256_file(repaired_receipt_path),
        "repaired_dataset_builder_commit": repaired["git"]["commit"],
        "source_split_sha256": SOURCE_SPLIT_SHA256,
        "base_config_sha256": BASE_CONFIG_SHA256,
        "derived_split_sha256": sha256_file(split_path),
        "catastrophic_validation_mae_eV_lt": 3.0,
        "test_partition_role": (
            "collected_by legacy trainer but forbidden for pilot decisions"
        ),
    }
    expected_configs: dict[str, dict[str, Any]] = {}
    legacy = deepcopy(base)
    legacy["output_dir"] = EXPECTED_OUTPUT_DIRS["legacy_init"]
    legacy["g1_pilot_contract"].update({
        "arm": "legacy_init",
        "claim_boundary": EXPECTED_CLAIMS["legacy_init"],
    })
    legacy["asset_sha256"]["pretrained_embed"] = LEGACY_PRETRAINED_SHA256
    expected_configs["legacy_init"] = legacy
    no_pretrain = deepcopy(base)
    no_pretrain["output_dir"] = EXPECTED_OUTPUT_DIRS["no_pretrain"]
    no_pretrain.pop("pretrained_embed", None)
    no_pretrain["asset_sha256"].pop("pretrained_embed", None)
    no_pretrain["g1_pilot_contract"].update({
        "arm": "no_pretrain",
        "claim_boundary": EXPECTED_CLAIMS["no_pretrain"],
    })
    expected_configs["no_pretrain"] = no_pretrain
    corrected_config = deepcopy(base)
    corrected_config["output_dir"] = EXPECTED_OUTPUT_DIRS["corrected_pretrain"]
    corrected_config["pretrained_embed"] = (
        "results/g1c/pretrained_embed_corrected.pt"
    )
    corrected_config["asset_sha256"]["pretrained_embed"] = corrected[
        "outputs"
    ]["corrected_asset_sha256"]
    corrected_config["g1_pilot_contract"].update({
        "arm": "corrected_pretrain",
        "claim_boundary": EXPECTED_CLAIMS["corrected_pretrain"],
        "corrected_pretraining_receipt_sha256": sha256_file(
            pretraining_receipt_path
        ),
        "corrected_pretraining_commit": corrected["git"]["commit"],
    })
    expected_configs["corrected_pretrain"] = corrected_config

    records = freeze.get("configs")
    if (
        not isinstance(records, list)
        or len(records) != 3
        or [record.get("arm") for record in records] != list(EXPECTED_ARMS)
    ):
        raise ValueError("pilot freeze must contain three ordered unique arms")
    config_records: dict[str, dict[str, Any]] = {}
    expected_records: list[dict[str, Any]] = []
    for arm in EXPECTED_ARMS:
        record = records[len(expected_records)]
        path = (ROOT / GENERATED_CONFIG_REPOSITORY_DIR / (
            f"{arm}_pair_cv5_f0_seed242.yaml"
        )).resolve()
        expected_path = ROOT / GENERATED_CONFIG_REPOSITORY_DIR / (
            f"{arm}_pair_cv5_f0_seed242.yaml"
        )
        if path != expected_path.resolve() or not path.is_file():
            raise ValueError("pilot frozen config path mismatch")
        config = yaml.safe_load(path.read_text())
        require_exact_record(config, expected_configs[arm], f"pilot config {arm}")
        expected_record = {
            "arm": arm,
            "repository_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "config_sha256": config_sha256(expected_configs[arm]),
        }
        require_exact_record(record, expected_record, f"pilot config record {arm}")
        expected_records.append(expected_record)
        config_records[arm] = {
            "path": path,
            "config": expected_configs[arm],
            "record": expected_record,
        }
    expected_freeze = {
        "schema_version": "prm_g1_pilot_freeze_v1",
        "git_commit": freeze["git_commit"],
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "graph_builder_source_sha256": sha256_file(ROOT / "src/graph.py"),
        "env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_CORRECTED,
        "model_source_sha256": sha256_file(ROOT / "src/models/crystal_v2.py"),
        "repaired_dataset_sha256": repaired["output"]["sha256"],
        "repaired_dataset_receipt_sha256": sha256_file(repaired_receipt_path),
        "source_split_repository_path": str(SOURCE_SPLIT_REPOSITORY_PATH),
        "source_split_sha256": SOURCE_SPLIT_SHA256,
        "base_config_repository_path": str(BASE_CONFIG_REPOSITORY_PATH),
        "base_config_sha256": BASE_CONFIG_SHA256,
        "derived_split_repository_path": str(DERIVED_SPLIT_REPOSITORY_PATH),
        "derived_split_sha256": sha256_file(split_path),
        "corrected_pretraining_receipt_sha256": sha256_file(
            pretraining_receipt_path
        ),
        "configs": expected_records,
    }
    require_exact_record(freeze, expected_freeze, "pilot freeze manifest")
    return freeze, split, config_records


def verify_pilot(
    arm: str, receipt_path: Path, run_dir: Path,
    config_record: Mapping[str, Any], freeze: Mapping[str, Any],
    split: Mapping[str, Any], targets: np.ndarray,
    dataset: CrystalGraphDataset, repaired_data_path: Path,
    ct_uae_path: Path, legacy_pretrained_path: Path,
    corrected_asset_path: Path,
    repaired_receipt_path: Path, pretraining_receipt_path: Path,
    pretraining_asset_sha: str, current: str,
    inventory: Mapping[str, str], device: torch.device,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    canonical_receipt = (run_dir / "g1_pilot_receipt.json").resolve()
    if receipt_path != canonical_receipt:
        raise ValueError(f"{arm} receipt is not the actual pilot sibling")
    receipt = load_json(receipt_path, "prm_g1_pilot_receipt_v1")
    producer = require_git(receipt.get("git"), current, f"{arm} pilot")
    if producer != current:
        raise ValueError(f"{arm} pilot must run on the collector commit")
    gpu = require_execution_identity(receipt, inventory, f"{arm} pilot")
    config = config_record["config"]
    config_path = config_record["path"]
    pilot = config.get("g1_pilot_contract", {})
    if (
        receipt.get("arm") != arm
        or receipt.get("claim_boundary") != EXPECTED_CLAIMS[arm]
        or pilot.get("arm") != arm
        or pilot.get("claim_boundary") != EXPECTED_CLAIMS[arm]
        or config.get("epochs") != 30
        or config.get("seed") != 242
        or config.get("output_dir") != EXPECTED_OUTPUT_DIRS[arm]
        or float(pilot.get("catastrophic_validation_mae_eV_lt", -1.0)) != 3.0
        or receipt.get("inputs", {}).get("config_sha256")
        != sha256_file(config_path)
        or receipt.get("inputs", {}).get("controlled_config_sha256")
        != config_sha256(config)
        or receipt.get("inputs", {}).get("dataset_receipt_sha256")
        != sha256_file(repaired_receipt_path)
        or receipt.get("inputs", {}).get("dataset_sha256")
        != sha256_file(repaired_data_path)
        or receipt.get("inputs", {}).get("split_sha256")
        != freeze.get("derived_split_sha256")
        or receipt.get("inputs", {}).get("source_split_sha256")
        != SOURCE_SPLIT_SHA256
        or receipt.get("inputs", {}).get("freeze_manifest_sha256")
        != sha256_file(ROOT / FREEZE_MANIFEST_REPOSITORY_PATH)
        or receipt.get("inputs", {}).get("ct_uae_sha256") != CT_UAE_SHA256
        or receipt.get("inputs", {}).get("env_zero_neighbor_mode")
        != ENV_ZERO_NEIGHBOR_CORRECTED
        or receipt.get("inputs", {}).get("model_source_sha256")
        != sha256_file(ROOT / "src/models/crystal_v2.py")
        or config.get("model_kwargs", {}).get("env_zero_neighbor_mode")
        != ENV_ZERO_NEIGHBOR_CORRECTED
    ):
        raise ValueError(f"{arm} pilot receipt/config/freeze contract failed")
    require_corrected_model_contract(pilot, f"{arm} pilot config")
    expected_pretrained: str | None
    if arm == "legacy_init":
        expected_pretrained = LEGACY_PRETRAINED_SHA256
        if receipt.get("inputs", {}).get("pretraining_receipt_sha256") is not None:
            raise ValueError("legacy pilot unexpectedly binds corrected pretraining")
    elif arm == "no_pretrain":
        expected_pretrained = None
        if (
            receipt.get("inputs", {}).get("pretrained_sha256") is not None
            or receipt.get("inputs", {}).get("pretraining_receipt_sha256") is not None
        ):
            raise ValueError("no-pretrain pilot contains a pretraining lineage")
    else:
        expected_pretrained = pretraining_asset_sha
        if receipt.get("inputs", {}).get("pretraining_receipt_sha256") != sha256_file(
            pretraining_receipt_path
        ):
            raise ValueError("corrected pilot pretraining receipt hash mismatch")
    if receipt.get("inputs", {}).get("pretrained_sha256") != expected_pretrained:
        raise ValueError(f"{arm} pilot initialization asset mismatch")
    expected_pretrained_path: Path | None
    if arm == "legacy_init":
        expected_pretrained_path = legacy_pretrained_path
    elif arm == "corrected_pretrain":
        expected_pretrained_path = corrected_asset_path
    else:
        expected_pretrained_path = None
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best.pt"
    split_indices_path = run_dir / "split_indices.npz"
    val_path = run_dir / "val_predictions.npz"
    test_path = run_dir / "test_predictions.npz"
    validate_training_completion(
        json.loads(manifest_path.read_text()), manifest_path, verify_outputs=True
    )
    manifest = json.loads(manifest_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    history = metrics.get("history")
    acceptance = receipt.get("acceptance", {})
    if (
        manifest.get("schema_version") != "prm_run_manifest_v1"
        or require_git(manifest.get("git"), current, f"{arm} run manifest")
        != current
        or
        manifest.get("git", {}).get("commit") != producer
        or manifest.get("git", {}).get("dirty") is not False
        or manifest.get("config") != config
        or manifest.get("config_sha256") != config_sha256(config)
        or manifest.get("data", {}).get("path") != str(repaired_data_path)
        or manifest.get("data", {}).get("data_sha256")
        != sha256_file(repaired_data_path)
        or int(manifest.get("data", {}).get("size_bytes", -1))
        != repaired_data_path.stat().st_size
        or manifest.get("split", {}).get("split_id")
        != "g1_pair_cv5_f0_repaired_v1"
        or manifest.get("split", {}).get("sha256") != freeze.get("derived_split_sha256")
        or manifest.get("split", {}).get("path")
        != str((ROOT / DERIVED_SPLIT_REPOSITORY_PATH).resolve())
        or manifest.get("split", {}).get("counts") != {
            key: len(split[key]) for key in ("train", "val", "test")
        }
        or manifest.get("seed") != 242
        or manifest.get("environment", {}).get("hostname") != EXPECTED_HOST
        or manifest.get("environment", {}).get("device") != "cuda"
        or manifest.get("environment", {}).get("cuda_available") is not True
        or manifest.get("environment", {}).get("device_name") != gpu["name"]
        or manifest.get("execution") != {
            "max_steps": 0,
            "resume_requested": False,
            "label_noise_stream": "model_seed_and_epoch_v1",
        }
        or not isinstance(history, list) or len(history) != 30
        or acceptance.get("epochs_completed") != 30
        or acceptance.get("finite_history") is not True
        or acceptance.get("checkpoint_readable") is not True
        or acceptance.get("passed") is not True
        or float(acceptance.get("catastrophic_threshold_eV_lt", -1.0)) != 3.0
        or not math.isfinite(float(acceptance.get("best_validation_mae_eV", float("nan"))))
        or float(acceptance["best_validation_mae_eV"]) >= 3.0
        or not math.isclose(
            float(acceptance["best_validation_mae_eV"]),
            float(metrics.get("best_val_mae", float("nan"))),
            rel_tol=0.0, abs_tol=1.0e-12,
        )
    ):
        raise ValueError(f"{arm} actual training manifest/history gate failed")
    command = manifest.get("command")
    if (
        not isinstance(command, list)
        or len(command) < 7
        or command[-6:] != [
            "-m", "src.train_enhanced", "--config",
            str(config_path), "--device", "cuda",
        ]
    ):
        raise ValueError(f"{arm} trainer command is not the frozen full run")
    if not all(
        set(row) == {
            "epoch", "train_mae", "val_mae", "val_rmse", "lr",
            "time_sec", "best",
        }
        and isinstance(row.get("best"), bool)
        and all(
            math.isfinite(float(row[key]))
            for key in ("train_mae", "val_mae", "val_rmse", "lr", "time_sec")
        )
        for row in history
    ):
        raise ValueError(f"{arm} pilot history contains a nonfinite value")
    if [int(row["epoch"]) for row in history] != list(range(1, 31)):
        raise ValueError(f"{arm} pilot epoch sequence is incomplete")
    best_so_far = float("inf")
    best_epoch = -1
    for row in history:
        expected_best = float(row["val_mae"]) < best_so_far
        if row["best"] is not expected_best:
            raise ValueError(f"{arm} pilot history best flags are inconsistent")
        if expected_best:
            best_so_far = float(row["val_mae"])
            best_epoch = int(row["epoch"])
    expected_acceptance = {
        "epochs_completed": 30,
        "best_validation_mae_eV": float(metrics["best_val_mae"]),
        "catastrophic_threshold_eV_lt": 3.0,
        "finite_history": True,
        "checkpoint_readable": True,
        "passed": float(metrics["best_val_mae"]) < 3.0,
    }
    require_close_payload(
        acceptance, expected_acceptance, f"{arm} receipt acceptance"
    )
    expected_runtime = deepcopy(config)
    expected_runtime.setdefault("model_kwargs", {})[
        "ct_uae_path"
    ] = str(ct_uae_path)
    if expected_pretrained_path is None:
        expected_runtime.pop("pretrained_embed", None)
    else:
        expected_runtime["pretrained_embed"] = str(expected_pretrained_path)
    require_exact_record(
        manifest.get("runtime_config"), expected_runtime,
        f"{arm} runtime config",
    )
    if manifest.get("runtime_config_sha256") != config_sha256(expected_runtime):
        raise ValueError(f"{arm} runtime config hash mismatch")
    expected_asset_paths: dict[str, Path] = {"ct_uae": ct_uae_path}
    if expected_pretrained_path is not None:
        expected_asset_paths["pretrained_embed"] = expected_pretrained_path
    expected_assets = {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "expected_sha256": sha256_file(path),
        }
        for name, path in expected_asset_paths.items()
    }
    require_exact_record(
        manifest.get("assets"), expected_assets, f"{arm} live assets"
    )

    model_kwargs = deepcopy(expected_runtime["model_kwargs"])
    initialization_model = CrystalTransformerV2(**model_kwargs)
    if expected_pretrained_path is None:
        if "pretraining" in manifest:
            raise ValueError("no-pretrain arm contains a pretraining report")
    else:
        live_pretraining = load_pretrained_initialization(
            initialization_model, expected_pretrained_path
        )
        require_exact_record(
            manifest.get("pretraining"), live_pretraining,
            f"{arm} live pretraining report",
        )
    output_hashes = {
        "run_manifest_sha256": sha256_file(manifest_path),
        "metrics_sha256": sha256_file(metrics_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_indices_sha256": sha256_file(split_indices_path),
        "validation_predictions_sha256": sha256_file(val_path),
        "test_predictions_sha256": sha256_file(test_path),
    }
    if receipt.get("outputs") != output_hashes:
        raise ValueError(f"{arm} pilot live output hashes differ from receipt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "model", "normalizer", "config", "epoch", "best_val_mae"
    }:
        raise ValueError(f"{arm} pilot checkpoint schema is invalid")
    model = CrystalTransformerV2(**model_kwargs).to(device)
    require_finite_tensor_mapping(
        checkpoint.get("model"), model.state_dict(), f"{arm} checkpoint state"
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    normalizer = checkpoint.get("normalizer")
    if (
        checkpoint.get("config") != expected_runtime
        or int(checkpoint.get("epoch", -1)) != best_epoch
        or not math.isclose(
            float(checkpoint.get("best_val_mae", float("nan"))),
            float(metrics["best_val_mae"]), rel_tol=1.0e-7, abs_tol=1.0e-8,
        )
        or not isinstance(normalizer, Mapping)
        or set(normalizer) != {"mean", "std", "transform"}
        or normalizer.get("transform") not in {"none", "log"}
        or not math.isfinite(float(normalizer.get("mean", float("nan"))))
        or not math.isfinite(float(normalizer.get("std", float("nan"))))
        or float(normalizer["std"]) <= 0.0
    ):
        raise ValueError(f"{arm} checkpoint metadata/normalizer is invalid")
    training_targets = torch.tensor(
        [dataset.data[int(index)]["target"] for index in split["train"]],
        dtype=torch.float32,
    )
    if normalizer["transform"] == "log":
        training_targets = torch.sign(training_targets) * torch.log1p(
            torch.abs(training_targets)
        )
    require_close_payload(
        normalizer,
        {
            "mean": float(training_targets.mean().item()),
            "std": float(training_targets.std().item()) + 1.0e-6,
            "transform": normalizer["transform"],
        },
        f"{arm} deterministic training normalizer",
        atol=1.0e-12,
    )
    with np.load(split_indices_path, allow_pickle=False) as archive:
        if (
            set(archive.files) != {"schema_version", "train", "val", "test", "split_id"}
            or str(archive["schema_version"].item()) != "prm_split_indices_v1"
            or str(archive["split_id"].item()) != "g1_pair_cv5_f0_repaired_v1"
            or any(
                not np.array_equal(
                    np.asarray(archive[partition], dtype=np.int64),
                    np.asarray(split[partition], dtype=np.int64),
                )
                for partition in ("train", "val", "test")
            )
        ):
            raise ValueError(f"{arm} pilot actual split membership mismatch")
    archived_val = verify_prediction_archive(
        val_path, split_id="g1_pair_cv5_f0_repaired_v1", partition="val",
        expected_indices=split["val"], targets=targets,
    )
    archived_test = verify_prediction_archive(
        test_path, split_id="g1_pair_cv5_f0_repaired_v1", partition="test",
        expected_indices=split["test"], targets=targets,
    )
    live_val = infer(
        model, Subset(dataset, [int(index) for index in split["val"]]),
        normalizer, device, 64,
    )
    live_test = infer(
        model, Subset(dataset, [int(index) for index in split["test"]]),
        normalizer, device, 64,
    )
    val_max = require_prediction_roundtrip(
        *live_val, *archived_val, f"{arm} validation", atol=PREDICTION_ROUNDTRIP_ATOL_EV
    )
    test_max = require_prediction_roundtrip(
        *live_test, *archived_test, f"{arm} test", atol=PREDICTION_ROUNDTRIP_ATOL_EV
    )
    live_val_mae = float(np.mean(np.abs(
        np.asarray(live_val[1], dtype=float)
        - np.asarray(live_val[2], dtype=float)
    )))
    if (
        not math.isclose(
            live_val_mae, float(metrics["validation"]["mae"]),
            rel_tol=1.0e-7, abs_tol=1.0e-5,
        )
        or not math.isclose(
            live_val_mae, float(metrics["best_val_mae"]),
            rel_tol=1.0e-7, abs_tol=1.0e-5,
        )
        or not math.isclose(
            live_val_mae,
            float(receipt["acceptance"]["best_validation_mae_eV"]),
            rel_tol=1.0e-7, abs_tol=1.0e-5,
        )
    ):
        raise ValueError(f"{arm} live validation MAE gate mismatch")
    if receipt.get("test_metric_policy") != (
        "trainer output exists but is excluded from all G1 decisions and claims"
    ):
        raise ValueError(f"{arm} pilot test-use boundary changed")
    del model, initialization_model, checkpoint
    torch.cuda.empty_cache()
    return receipt, gpu, {
        "checkpoint_strict_loaded": True,
        "best_history_flags_and_checkpoint_epoch_recomputed": True,
        "live_validation_mae_eV": live_val_mae,
        "validation_reinference_max_abs_delta_eV": val_max,
        "test_reinference_max_abs_delta_eV": test_max,
        "pretraining_compatibility_reexecuted": expected_pretrained_path is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--raw-db", type=Path, required=True)
    parser.add_argument("--protocol-samples", type=Path, required=True)
    parser.add_argument("--repaired-data", type=Path, required=True)
    parser.add_argument("--repaired-dataset-receipt", type=Path, required=True)
    parser.add_argument("--property-tests-receipt", type=Path, required=True)
    parser.add_argument("--g1a-summary", type=Path, required=True)
    parser.add_argument("--g1a-predictions", type=Path, required=True)
    parser.add_argument("--g1a-sample-diagnostics", type=Path, required=True)
    parser.add_argument("--legacy-run-root", type=Path, required=True)
    parser.add_argument("--jarvis-source-data", type=Path, required=True)
    parser.add_argument("--g1c-dataset", type=Path, required=True)
    parser.add_argument("--g1c-dataset-receipt", type=Path, required=True)
    parser.add_argument("--g1c-pretraining-receipt", type=Path, required=True)
    parser.add_argument("--g1c-corrected-asset", type=Path, required=True)
    parser.add_argument("--ct-uae", type=Path, required=True)
    parser.add_argument("--legacy-pretrained", type=Path, required=True)
    parser.add_argument("--legacy-pilot-receipt", type=Path, required=True)
    parser.add_argument("--legacy-pilot-dir", type=Path, required=True)
    parser.add_argument("--no-pretrain-pilot-receipt", type=Path, required=True)
    parser.add_argument("--no-pretrain-pilot-dir", type=Path, required=True)
    parser.add_argument("--corrected-pilot-receipt", type=Path, required=True)
    parser.add_argument("--corrected-pilot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    current_git = git_snapshot()
    runtime_gpu = gpu_identity()
    current = current_git["commit"]
    inventory = gpu_inventory()
    if (
        GPU_UUID_RE.fullmatch(runtime_gpu["uuid"]) is None
        or runtime_gpu["uuid"] == EXCLUDED_GPU_UUID
        or inventory.get(runtime_gpu["uuid"]) != runtime_gpu["name"]
    ):
        raise ValueError("collector GPU identity differs from live inventory")
    device = torch.device("cuda")

    files = {
        "source_data": resolve_file(args.source_data, "source IMP2D data"),
        "raw_db": resolve_file(args.raw_db, "raw IMP2D database"),
        "protocol_samples": resolve_file(
            args.protocol_samples, "protocol samples"
        ),
        "repaired_data": resolve_file(args.repaired_data, "repaired IMP2D data"),
        "repaired_receipt": resolve_file(
            args.repaired_dataset_receipt, "repaired IMP2D receipt"
        ),
        "property_receipt": resolve_file(args.property_tests_receipt, "property receipt"),
        "g1a_summary": resolve_file(args.g1a_summary, "G1A summary"),
        "g1a_predictions": resolve_file(args.g1a_predictions, "G1A predictions"),
        "g1a_diagnostics": resolve_file(
            args.g1a_sample_diagnostics, "G1A sample diagnostics"
        ),
        "jarvis_source": resolve_file(args.jarvis_source_data, "JARVIS source"),
        "g1c_dataset": resolve_file(args.g1c_dataset, "corrected JARVIS data"),
        "g1c_dataset_receipt": resolve_file(
            args.g1c_dataset_receipt, "corrected JARVIS receipt"
        ),
        "pretraining_receipt": resolve_file(
            args.g1c_pretraining_receipt, "G1C pretraining receipt"
        ),
        "corrected_asset": resolve_file(args.g1c_corrected_asset, "corrected asset"),
        "ct_uae": resolve_file(args.ct_uae, "ct-UAE asset"),
        "legacy_pretrained": resolve_file(
            args.legacy_pretrained, "legacy pretraining asset"
        ),
        "legacy_pilot_receipt": resolve_file(
            args.legacy_pilot_receipt, "legacy pilot receipt"
        ),
        "no_pretrain_pilot_receipt": resolve_file(
            args.no_pretrain_pilot_receipt, "no-pretrain pilot receipt"
        ),
        "corrected_pilot_receipt": resolve_file(
            args.corrected_pilot_receipt, "corrected pilot receipt"
        ),
    }
    dirs = {
        "legacy_run_root": resolve_dir(args.legacy_run_root, "legacy run root"),
        "legacy_init": resolve_dir(args.legacy_pilot_dir, "legacy pilot directory"),
        "no_pretrain": resolve_dir(
            args.no_pretrain_pilot_dir, "no-pretrain pilot directory"
        ),
        "corrected_pretrain": resolve_dir(
            args.corrected_pilot_dir, "corrected pilot directory"
        ),
    }
    output = args.output.expanduser().resolve()
    all_paths = list(files.values()) + list(dirs.values()) + [output]
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("all acceptance inputs and output must resolve distinctly")
    if output.exists():
        raise FileExistsError("versioned G1 acceptance output already exists")
    if output.parent != (ROOT / "artifacts/prm_g1").resolve():
        raise ValueError("formal G1 acceptance must be written to artifacts/prm_g1")
    require_sha(files["ct_uae"], CT_UAE_SHA256, "ct-UAE")
    require_sha(
        files["legacy_pretrained"], LEGACY_PRETRAINED_SHA256,
        "legacy pretraining asset",
    )
    protocol_rows, canonical_indices = load_protocol_samples(
        files["protocol_samples"]
    )

    pilot_roots = []
    for arm in EXPECTED_ARMS:
        suffix = Path(EXPECTED_OUTPUT_DIRS[arm]).parts
        if tuple(dirs[arm].parts[-len(suffix):]) != suffix:
            raise ValueError(f"{arm} pilot directory is not the frozen output path")
        root = dirs[arm]
        for _ in suffix:
            root = root.parent
        pilot_roots.append(root)
    if len(set(pilot_roots)) != 1:
        raise ValueError("all three pilots must share one external result root")

    repaired_receipt, imp2d_live = verify_repaired_imp2d(
        files["repaired_receipt"], files["source_data"], files["repaired_data"],
        files["raw_db"], files["protocol_samples"], protocol_rows, current,
    )
    property_receipt, property_gpu, property_live = verify_property_receipt(
        files["property_receipt"], current, inventory,
    )
    g1a, g1a_gpu, g1a_live = verify_g1a(
        files["g1a_summary"], files["g1a_predictions"], files["g1a_diagnostics"],
        dirs["legacy_run_root"], files["source_data"], files["repaired_data"],
        files["raw_db"],
        files["ct_uae"], files["legacy_pretrained"], protocol_rows,
        canonical_indices, current, inventory, device,
    )
    g1c_dataset_receipt, jarvis_live = verify_g1c_dataset(
        files["g1c_dataset_receipt"], files["jarvis_source"],
        files["g1c_dataset"], current,
    )
    pretraining, pretraining_gpu, pretraining_live = verify_pretraining(
        files["pretraining_receipt"], files["g1c_dataset"],
        files["g1c_dataset_receipt"], files["corrected_asset"],
        files["ct_uae"], current, inventory, device,
    )
    freeze, split, config_records = verify_freeze(
        files["repaired_receipt"], files["pretraining_receipt"], current,
    )
    if (
        freeze.get("repaired_dataset_sha256") != sha256_file(files["repaired_data"])
        or split.get("data_sha256") != sha256_file(files["repaired_data"])
    ):
        raise ValueError("freeze/split do not bind the actual repaired IMP2D file")
    dataset = CrystalGraphDataset(files["repaired_data"])
    dataset_targets = np.asarray(
        [float(sample["target"]) for sample in dataset.data], dtype=np.float32
    )
    if (
        len(dataset) != EXPECTED_CONTAINER_ROWS
        or not np.array_equal(
            dataset_targets, imp2d_live["targets"].astype(np.float32)
        )
    ):
        raise ValueError("live pilot dataset targets differ from rebuilt audit")

    pilot_receipt_paths = {
        "legacy_init": files["legacy_pilot_receipt"],
        "no_pretrain": files["no_pretrain_pilot_receipt"],
        "corrected_pretrain": files["corrected_pilot_receipt"],
    }
    pilots: dict[str, dict[str, Any]] = {}
    pilot_gpus: dict[str, dict[str, str]] = {}
    pilot_live: dict[str, dict[str, Any]] = {}
    for arm in EXPECTED_ARMS:
        pilots[arm], pilot_gpus[arm], pilot_live[arm] = verify_pilot(
            arm, pilot_receipt_paths[arm], dirs[arm], config_records[arm],
            freeze, split, imp2d_live["targets"], dataset,
            files["repaired_data"], files["ct_uae"],
            files["legacy_pretrained"], files["corrected_asset"],
            files["repaired_receipt"], files["pretraining_receipt"],
            pretraining["outputs"]["corrected_asset_sha256"],
            current, inventory, device,
        )
    pilot_commits = {pilots[arm]["git"]["commit"] for arm in EXPECTED_ARMS}
    if pilot_commits != {current}:
        raise ValueError("all three pilot arms must run from the collector commit")
    execution_gpus = [property_gpu, g1a_gpu, pretraining_gpu, *pilot_gpus.values()]
    expected_runtime_gpu = {
        "uuid": runtime_gpu["uuid"], "name": runtime_gpu["name"]
    }
    if any(gpu != expected_runtime_gpu for gpu in execution_gpus):
        raise ValueError("all G1 GPU evidence must use the collector-assigned UUID")
    checkpoint_hashes = {
        pilots[arm]["outputs"]["checkpoint_sha256"] for arm in EXPECTED_ARMS
    }
    if len(checkpoint_hashes) != 3:
        raise ValueError("pilot arms do not have three distinct checkpoints")

    producer_commits = {
        "imp2d_rebuild": repaired_receipt["git"]["commit"],
        "property_tests": property_receipt["git"]["commit"],
        "g1a": g1a["git"]["commit"],
        "jarvis_rebuild": g1c_dataset_receipt["git"]["commit"],
        "corrected_pretraining": pretraining["git"]["commit"],
        "pilot_freeze": freeze["git_commit"],
        "pilots": current,
    }
    payload = {
        "schema_version": "prm_g1_acceptance_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted",
        "host_profile": EXPECTED_HOST,
        "collector_gpu": runtime_gpu,
        "code_commit": current,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "graph_builder_source_sha256": sha256_file(ROOT / "src/graph.py"),
        "model_source_sha256": sha256_file(ROOT / "src/models/crystal_v2.py"),
        "env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_CORRECTED,
        "env_zero_neighbor_contract": {
            "corrected": ENV_ZERO_NEIGHBOR_CORRECTED,
            "legacy_g1a_compatibility": ENV_ZERO_NEIGHBOR_LEGACY,
        },
        "source_data_sha256": SOURCE_DATA_SHA256,
        "raw_db_sha256": RAW_DB_SHA256,
        "protocol_samples_sha256": PROTOCOL_SAMPLES_SHA256,
        "repaired_dataset_receipt_sha256": sha256_file(files["repaired_receipt"]),
        "repaired_dataset_sha256": sha256_file(files["repaired_data"]),
        "repaired_dataset_live_audit": {
            "rows": imp2d_live["rows"],
            "total_edges": imp2d_live["total_edges"],
            "total_triplets": imp2d_live["total_triplets"],
            "non_graph_fields_and_order_exact": True,
            "raw_db_identity_and_pbc_reopened": True,
            "protocol_identity_targets_and_canonical_membership_exact": True,
            "current_build_graph_all_fields_recomputed": True,
        },
        "property_tests_passed": True,
        "property_tests_receipt_sha256": sha256_file(files["property_receipt"]),
        "property_tests_live_audit": property_live,
        "g1a_summary_sha256": sha256_file(files["g1a_summary"]),
        "g1a_predictions_sha256": sha256_file(files["g1a_predictions"]),
        "g1a_sample_diagnostics_sha256": sha256_file(files["g1a_diagnostics"]),
        "g1a_empirical_stability_pass": bool(
            g1a["legacy_under_permutations"]["empirical_stability_pass"]
        ),
        "g1a_live_audit": g1a_live,
        "corrected_jarvis_dataset_receipt_sha256": sha256_file(
            files["g1c_dataset_receipt"]
        ),
        "corrected_jarvis_dataset_sha256": sha256_file(files["g1c_dataset"]),
        "corrected_jarvis_live_audit": {
            "rows": jarvis_live["rows"],
            "total_edges": jarvis_live["total_edges"],
            "total_triplets": jarvis_live["total_triplets"],
            "id_order_targets_non_graph_exact": True,
            "pbc_all_true": True,
            "current_build_graph_all_fields_recomputed": True,
        },
        "corrected_pretraining_lineage_sha256": sha256_file(
            files["pretraining_receipt"]
        ),
        "corrected_pretrained_asset_sha256": sha256_file(files["corrected_asset"]),
        "corrected_pretraining_live_audit": pretraining_live,
        "pilot_passed": True,
        "pilot_freeze_sha256": sha256_file(ROOT / FREEZE_MANIFEST_REPOSITORY_PATH),
        "pilot_split_sha256": sha256_file(ROOT / DERIVED_SPLIT_REPOSITORY_PATH),
        "pilot_receipts": {
            arm: sha256_file(pilot_receipt_paths[arm]) for arm in EXPECTED_ARMS
        },
        "pilot_live_checkpoint_reinference": pilot_live,
        "execution_gpu_bindings": {
            "property_tests": property_gpu,
            "g1a": g1a_gpu,
            "corrected_pretraining": pretraining_gpu,
            "pilots": pilot_gpus,
        },
        "producer_commits": producer_commits,
        "live_artifact_binding": {
            "ct_uae_sha256": CT_UAE_SHA256,
            "legacy_pretrained_sha256": LEGACY_PRETRAINED_SHA256,
            "jarvis_source_sha256": JARVIS_SOURCE_SHA256,
            "protocol_samples_sha256": PROTOCOL_SAMPLES_SHA256,
            "raw_db_sha256": RAW_DB_SHA256,
            "all_upstream_files_reopened": True,
            "all_scientific_outputs_recomputed_or_checkpoint_reinferred": True,
        },
        "scope_boundary": "G1 accepted; G2 remains unauthorized",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output.parent, prefix=output.name + ".",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(strict_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(strict_json({"status": "accepted", "sha256": sha256_file(output)}))


if __name__ == "__main__":
    main()
