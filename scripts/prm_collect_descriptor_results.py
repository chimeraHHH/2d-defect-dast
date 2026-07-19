"""Freeze completed descriptor baselines before neural runs are available."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from scripts.prm_collect_results import (
    aggregate_fold_rows,
    file_sha256,
    load_descriptor_runs,
    select_descriptor_families,
    summarize_folds,
    write_csv,
)


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_PAPER_SPLITS = {
    "id_repeat": 5,
    "id_cv": 5,
    "pair_cv": 5,
    "host_cv": 5,
    "dopant_cv": 5,
    "chemistry_block": 1,
}


def git_snapshot() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    return {
        "commit": commit or None, "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def validate_descriptor_manifest(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any],
    protocol_manifest_sha256: str,
) -> None:
    if manifest.get("schema_version") != "prm_descriptor_manifest_v2":
        raise ValueError("unsupported descriptor manifest schema")
    if manifest.get("status") != "complete":
        raise ValueError("descriptor baseline batch is incomplete")
    coverage = manifest.get("formal_split_coverage", {})
    if coverage != {"complete": 27, "total": 27, "all_complete": True}:
        raise ValueError(f"formal descriptor split coverage is incomplete: {coverage}")
    if manifest.get("selection_data") != "validation only":
        raise ValueError("descriptor hyperparameters were not selected on validation data")
    if manifest.get("data_sha256") != protocol.get("data_sha256"):
        raise ValueError("descriptor dataset hash does not match the frozen protocol")
    if manifest.get("data_file_sha256") != protocol.get("data_sha256"):
        raise ValueError("descriptor input file hash was not independently verified")
    if manifest.get("protocol_manifest_sha256") != protocol_manifest_sha256:
        raise ValueError("descriptor protocol manifest hash mismatch")
    if (
        not manifest.get("git", {}).get("commit")
        or manifest.get("git", {}).get("dirty")
    ):
        raise ValueError("final descriptor batch Git provenance is inadmissible")
    encoding = manifest.get("metric_encoding", {})
    normalizer_git = encoding.get("normalizer_git", {})
    if (
        encoding.get("schema_version") != "prm_nullable_correlations_v1"
        or not normalizer_git.get("commit")
        or normalizer_git.get("dirty")
    ):
        raise ValueError("descriptor nullable-metric normalization is inadmissible")
    artifacts = manifest.get("split_artifacts", {})
    if set(artifacts) != set(manifest.get("splits", [])) or len(artifacts) != 27:
        raise ValueError("descriptor split artifact hashes are incomplete")


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def archive_descriptor_artifacts(
    descriptor_root: Path,
    out_dir: Path,
    descriptor_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    archive_root = out_dir / "runs"
    staging_root = out_dir / ".runs.tmp"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    source_manifest = descriptor_root / "manifest.json"
    shutil.copyfile(source_manifest, staging_root / "manifest.json")
    archived_runs = []
    for split_id in sorted(descriptor_manifest["splits"]):
        source_dir = descriptor_root / split_id
        target_dir = staging_root / split_id
        target_dir.mkdir()
        record = descriptor_manifest["split_artifacts"][split_id]
        archived = {"split_id": split_id}
        for name, hash_key in (
            ("metrics.json", "metrics_sha256"),
            ("predictions.npz", "predictions_sha256"),
        ):
            source = source_dir / name
            expected_sha = record[hash_key]
            if file_sha256(source) != expected_sha:
                raise ValueError(f"descriptor {name} hash mismatch: {split_id}")
            target = target_dir / name
            shutil.copyfile(source, target)
            if file_sha256(target) != expected_sha:
                raise OSError(f"archived descriptor {name} hash mismatch: {split_id}")
            key = "metrics" if name == "metrics.json" else "predictions"
            archived[f"{key}_path"] = repository_path(
                archive_root / split_id / name
            )
            archived[f"{key}_sha256"] = expected_sha
        archived_runs.append(archived)

    if archive_root.exists():
        shutil.rmtree(archive_root)
    staging_root.replace(archive_root)
    return {
        "manifest": {
            "path": repository_path(archive_root / "manifest.json"),
            "sha256": file_sha256(archive_root / "manifest.json"),
        },
        "runs": archived_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v2",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts/prm_results/descriptors",
    )
    args = parser.parse_args()

    descriptor_root = args.descriptor_root.resolve()
    protocol_dir = args.protocol_dir.resolve()
    protocol_path = protocol_dir / "manifest.json"
    descriptor_manifest_path = descriptor_root / "manifest.json"
    protocol = json.loads(protocol_path.read_text())
    descriptor_manifest = json.loads(descriptor_manifest_path.read_text())
    validate_descriptor_manifest(
        descriptor_manifest, protocol, file_sha256(protocol_path),
    )
    for split_id, record in descriptor_manifest["split_artifacts"].items():
        split_dir = descriptor_root / split_id
        if record.get("metrics_sha256") != file_sha256(split_dir / "metrics.json"):
            raise ValueError(f"descriptor metrics hash mismatch: {split_id}")
        if record.get("predictions_sha256") != file_sha256(split_dir / "predictions.npz"):
            raise ValueError(f"descriptor predictions hash mismatch: {split_id}")

    rows = load_descriptor_runs(descriptor_root, protocol_dir)
    observed = {
        regime: len({row["split_id"] for row in rows if row["regime"] == regime})
        for regime in EXPECTED_PAPER_SPLITS
    }
    if observed != EXPECTED_PAPER_SPLITS:
        raise ValueError(f"paper descriptor split counts differ: {observed}")
    selection = select_descriptor_families(rows)
    fold_rows = aggregate_fold_rows(rows)
    summary_rows = summarize_folds(fold_rows)
    selected_rows = [
        row for row in summary_rows
        if row["model"] == "descriptor:mean"
        or row["model"] == f"descriptor:{selection[row['regime']]['selected_family']}"
    ]
    collector_git = git_snapshot()
    if not collector_git["commit"] or collector_git["dirty"]:
        raise ValueError("descriptor collection requires a clean Git commit")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "run_metrics.csv", rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "selected_summary.csv", selected_rows)
    (out_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    archive = archive_descriptor_artifacts(
        descriptor_root, out_dir, descriptor_manifest,
    )
    sources = [
        record for record in archive["runs"]
        if record["split_id"] != "uq_calibration_s62"
    ]
    bundle = {
        "schema_version": "prm_descriptor_bundle_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector_git": collector_git,
        "data_sha256": protocol["data_sha256"],
        "protocol_manifest": {
            "path": repository_path(protocol_path), "sha256": file_sha256(protocol_path),
        },
        "descriptor_manifest": {
            **archive["manifest"],
            "training_git": descriptor_manifest["git"],
            "metric_encoding": descriptor_manifest["metric_encoding"],
        },
        "selection_data": "validation only",
        "expected_split_counts": EXPECTED_PAPER_SPLITS,
        "n_metric_rows": len(rows),
        "n_sources": len(sources),
        "n_archived_runs": len(archive["runs"]),
        "selection": selection,
        "sources": sources,
        "archived_runs": archive["runs"],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(selection, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
