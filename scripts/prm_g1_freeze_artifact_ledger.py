"""Freeze all external G1 evidence hashes into a reviewable Git milestone.

This command never deserializes a scientific artifact.  It records byte hashes
and sizes only, while the repository is clean and pushed.  The resulting
ledger must then be reviewed, committed, and pushed before the final collector
is allowed to deserialize repaired dataset pickles or admit checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_HOST = "WHUServer-L40S"
LEDGER_REPOSITORY_PATH = Path(
    "artifacts/prm_g1/g1_artifact_hash_ledger.json"
)
EXPECTED_ARMS = ("legacy_init", "no_pretrain", "corrected_pretrain")
EXPECTED_OUTPUT_DIRS = {
    "legacy_init": "g1/pilots/legacy_init/pair_cv5_f0/seed242",
    "no_pretrain": "g1/pilots/no_pretrain/pair_cv5_f0/seed242",
    "corrected_pretrain": "g1/pilots/corrected_pretrain/pair_cv5_f0/seed242",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def clean_pushed_git() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    remote_refs = subprocess.run(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()
    remote_refs = sorted(ref.strip() for ref in remote_refs if ref.strip())
    if not commit or status or not remote_refs:
        raise SystemExit(
            "artifact-ledger freeze requires a clean commit on a fetched remote ref"
        )
    return {"commit": commit, "dirty": False, "remote_refs": remote_refs}


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


def record(path: Path, alias: str) -> dict[str, Any]:
    return {
        "alias": alias,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repaired-data", type=Path, required=True)
    parser.add_argument("--repaired-dataset-receipt", type=Path, required=True)
    parser.add_argument("--g1a-summary", type=Path, required=True)
    parser.add_argument("--g1a-predictions", type=Path, required=True)
    parser.add_argument("--g1a-sample-diagnostics", type=Path, required=True)
    parser.add_argument("--legacy-run-root", type=Path, required=True)
    parser.add_argument("--g1c-dataset", type=Path, required=True)
    parser.add_argument("--g1c-dataset-receipt", type=Path, required=True)
    parser.add_argument("--g1c-pretraining-receipt", type=Path, required=True)
    parser.add_argument("--g1c-corrected-asset", type=Path, required=True)
    parser.add_argument("--legacy-pilot-receipt", type=Path, required=True)
    parser.add_argument("--legacy-pilot-dir", type=Path, required=True)
    parser.add_argument("--no-pretrain-pilot-receipt", type=Path, required=True)
    parser.add_argument("--no-pretrain-pilot-dir", type=Path, required=True)
    parser.add_argument("--corrected-pilot-receipt", type=Path, required=True)
    parser.add_argument("--corrected-pilot-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / LEDGER_REPOSITORY_PATH
    )
    args = parser.parse_args()

    if socket.gethostname() != EXPECTED_HOST:
        raise SystemExit("formal G1 artifact ledger requires WHUServer-L40S")
    git = clean_pushed_git()
    output = args.output.expanduser().resolve()
    if output != (ROOT / LEDGER_REPOSITORY_PATH).resolve():
        raise SystemExit("formal G1 artifact ledger path is repository-fixed")
    if output.exists():
        raise FileExistsError("versioned G1 artifact ledger already exists")

    files = {
        "repaired_data": resolve_file(args.repaired_data, "repaired IMP2D"),
        "repaired_receipt": resolve_file(
            args.repaired_dataset_receipt, "repaired IMP2D receipt"
        ),
        "g1a_summary": resolve_file(args.g1a_summary, "G1A summary"),
        "g1a_predictions": resolve_file(args.g1a_predictions, "G1A predictions"),
        "g1a_diagnostics": resolve_file(
            args.g1a_sample_diagnostics, "G1A diagnostics"
        ),
        "g1c_dataset": resolve_file(args.g1c_dataset, "corrected JARVIS"),
        "g1c_dataset_receipt": resolve_file(
            args.g1c_dataset_receipt, "corrected JARVIS receipt"
        ),
        "pretraining_receipt": resolve_file(
            args.g1c_pretraining_receipt, "corrected pretraining receipt"
        ),
        "corrected_asset": resolve_file(
            args.g1c_corrected_asset, "corrected pretrained asset"
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
        "legacy_init": resolve_dir(args.legacy_pilot_dir, "legacy pilot"),
        "no_pretrain": resolve_dir(args.no_pretrain_pilot_dir, "no-pretrain pilot"),
        "corrected_pretrain": resolve_dir(
            args.corrected_pilot_dir, "corrected pilot"
        ),
    }
    if len(set(files.values()) | set(dirs.values()) | {output}) != (
        len(files) + len(dirs) + 1
    ):
        raise ValueError("ledger inputs/output must resolve distinctly")
    if not (
        files["g1a_summary"].parent
        == files["g1a_predictions"].parent
        == files["g1a_diagnostics"].parent
    ):
        raise ValueError("G1A ledger artifacts are not siblings")
    pretrain_parent = files["pretraining_receipt"].parent
    if files["corrected_asset"] != pretrain_parent / "pretrained_embed_corrected.pt":
        raise ValueError("corrected asset is not the pretraining receipt sibling")
    pretrain_siblings = {
        "receipt": files["pretraining_receipt"],
        "corrected_asset": files["corrected_asset"],
        "source_export": resolve_file(
            pretrain_parent / "pretrained_embed.pt", "source export"
        ),
        "best_checkpoint": resolve_file(pretrain_parent / "best.pt", "source best"),
        "metrics": resolve_file(pretrain_parent / "metrics.json", "source metrics"),
    }

    legacy_folds = []
    for fold in range(5):
        split_id = f"pair_cv5_f{fold}"
        run_dir = (
            dirs["legacy_run_root"] / "selected/g111/transfer"
            / split_id / "seed242"
        )
        legacy_folds.append({
            "fold": fold,
            "split_id": split_id,
            "manifest": record(
                resolve_file(run_dir / "run_manifest.json", "legacy manifest"),
                f"LEGACY_F{fold}_MANIFEST",
            ),
            "checkpoint": record(
                resolve_file(run_dir / "best.pt", "legacy checkpoint"),
                f"LEGACY_F{fold}_CHECKPOINT",
            ),
            "test_predictions": record(
                resolve_file(run_dir / "test_predictions.npz", "legacy predictions"),
                f"LEGACY_F{fold}_TEST_PREDICTIONS",
            ),
        })

    pilot_receipts = {
        "legacy_init": files["legacy_pilot_receipt"],
        "no_pretrain": files["no_pretrain_pilot_receipt"],
        "corrected_pretrain": files["corrected_pilot_receipt"],
    }
    pilots: dict[str, Any] = {}
    for arm in EXPECTED_ARMS:
        run_dir = dirs[arm]
        suffix = Path(EXPECTED_OUTPUT_DIRS[arm]).parts
        if tuple(run_dir.parts[-len(suffix):]) != suffix:
            raise ValueError(f"{arm} pilot directory is not canonical")
        receipt_path = pilot_receipts[arm]
        if receipt_path != run_dir / "g1_pilot_receipt.json":
            raise ValueError(f"{arm} receipt is not its run sibling")
        receipt_payload = json.loads(receipt_path.read_text())
        if (
            receipt_payload.get("schema_version") != "prm_g1_pilot_receipt_v1"
            or receipt_payload.get("arm") != arm
            or receipt_payload.get("git", {}).get("commit") != git["commit"]
            or receipt_payload.get("git", {}).get("dirty") is not False
        ):
            raise ValueError(f"{arm} pilot was not produced by ledger source commit")
        pilots[arm] = {
            "receipt": record(receipt_path, f"{arm.upper()}_RECEIPT"),
            "run_manifest": record(
                resolve_file(run_dir / "run_manifest.json", "pilot manifest"),
                f"{arm.upper()}_MANIFEST",
            ),
            "metrics": record(
                resolve_file(run_dir / "metrics.json", "pilot metrics"),
                f"{arm.upper()}_METRICS",
            ),
            "checkpoint": record(
                resolve_file(run_dir / "best.pt", "pilot checkpoint"),
                f"{arm.upper()}_CHECKPOINT",
            ),
            "split_indices": record(
                resolve_file(run_dir / "split_indices.npz", "pilot split"),
                f"{arm.upper()}_SPLIT_INDICES",
            ),
            "validation_predictions": record(
                resolve_file(run_dir / "val_predictions.npz", "pilot val"),
                f"{arm.upper()}_VALIDATION_PREDICTIONS",
            ),
            "test_predictions": record(
                resolve_file(run_dir / "test_predictions.npz", "pilot test"),
                f"{arm.upper()}_TEST_PREDICTIONS",
            ),
        }

    payload = {
        "schema_version": "prm_g1_artifact_hash_ledger_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_profile": EXPECTED_HOST,
        "git": git,
        "review_state": "pending_human_review",
        "artifacts": {
            "repaired_imp2d": {
                "dataset": record(files["repaired_data"], "PRM_REPAIRED_DATASET_PATH"),
                "receipt": record(files["repaired_receipt"], "IMP2D_REBUILD_RECEIPT"),
            },
            "g1a": {
                "summary": record(files["g1a_summary"], "G1A_SUMMARY"),
                "predictions": record(files["g1a_predictions"], "G1A_PREDICTIONS"),
                "sample_diagnostics": record(
                    files["g1a_diagnostics"], "G1A_SAMPLE_DIAGNOSTICS"
                ),
                "legacy_folds": legacy_folds,
            },
            "corrected_jarvis": {
                "dataset": record(files["g1c_dataset"], "PRM_CORRECTED_JARVIS_DATA_PATH"),
                "receipt": record(
                    files["g1c_dataset_receipt"], "JARVIS_REBUILD_RECEIPT"
                ),
            },
            "corrected_pretraining": {
                key: record(path, f"G1C_{key.upper()}")
                for key, path in pretrain_siblings.items()
            },
            "pilots": pilots,
        },
        "trust_boundary": (
            "hashes only; no scientific artifact was deserialized; acceptance "
            "requires this exact ledger blob in a later clean pushed commit"
        ),
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
    print(strict_json({
        "status": "candidate_ledger_written",
        "sha256": sha256_file(output),
        "next": "review, commit, and push the ledger before acceptance",
    }))


if __name__ == "__main__":
    main()
