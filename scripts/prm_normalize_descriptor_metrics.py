"""Normalize undefined descriptor correlations without retraining models."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.prm_metrics import regression_metrics
from src.prm_provenance import load_protocol_targets, validate_protocol_targets

METRIC_ENCODING_SCHEMA = "prm_nullable_correlations_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def strict_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def _validate_source_manifest(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any], protocol_path: Path,
) -> None:
    if manifest.get("schema_version") != "prm_descriptor_manifest_v2":
        raise ValueError("descriptor normalization requires a v2 source manifest")
    if manifest.get("status") != "complete":
        raise ValueError("descriptor normalization requires a complete source batch")
    if manifest.get("data_sha256") != protocol.get("data_sha256"):
        raise ValueError("descriptor normalization dataset hash mismatch")
    if manifest.get("protocol_manifest_sha256") != file_sha256(protocol_path):
        raise ValueError("descriptor normalization protocol hash mismatch")
    splits = manifest.get("splits")
    artifacts = manifest.get("split_artifacts")
    if not isinstance(splits, list) or not isinstance(artifacts, Mapping):
        raise ValueError("descriptor normalization source coverage is incomplete")
    if not splits or set(splits) != set(artifacts):
        raise ValueError("descriptor normalization source hashes are incomplete")


def normalize_descriptor_root(
    source_root: Path, output_root: Path, protocol_dir: Path,
    *, normalizer_git: Mapping[str, Any],
) -> Dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    protocol_dir = protocol_dir.resolve()
    if source_root == output_root:
        raise ValueError("descriptor normalization output must differ from its source")
    if output_root.exists():
        raise FileExistsError(
            f"descriptor normalization output already exists: {output_root}"
        )
    if not normalizer_git.get("commit") or normalizer_git.get("dirty"):
        raise ValueError("descriptor normalization requires a clean Git commit")

    protocol_path = protocol_dir / "manifest.json"
    source_manifest_path = source_root / "manifest.json"
    protocol = json.loads(protocol_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    _validate_source_manifest(source_manifest, protocol, protocol_path)
    protocol_targets = load_protocol_targets(protocol_dir / "samples.csv")

    staging = output_root.with_name(output_root.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    artifact_hashes: Dict[str, Dict[str, str]] = {}
    try:
        for split_id in sorted(source_manifest["splits"]):
            source_dir = source_root / split_id
            source_metrics = source_dir / "metrics.json"
            source_predictions = source_dir / "predictions.npz"
            source_record = source_manifest["split_artifacts"][split_id]
            if source_record.get("metrics_sha256") != file_sha256(source_metrics):
                raise ValueError(f"source descriptor metrics hash mismatch: {split_id}")
            if source_record.get("predictions_sha256") != file_sha256(
                source_predictions
            ):
                raise ValueError(f"source descriptor predictions hash mismatch: {split_id}")

            split_path = protocol_dir / "splits" / f"{split_id}.json"
            split = json.loads(split_path.read_text())
            metrics = json.loads(source_metrics.read_text())
            if metrics.get("schema_version") != "prm_descriptor_results_v1":
                raise ValueError(f"unsupported descriptor metrics schema: {split_id}")
            if metrics.get("split_id") != split_id:
                raise ValueError(f"descriptor metrics split mismatch: {split_id}")
            if metrics.get("split_sha256") != file_sha256(split_path):
                raise ValueError(f"descriptor metrics split hash mismatch: {split_id}")
            with np.load(source_predictions, allow_pickle=False) as archive:
                expected_fields = {
                    "schema_version", "split_id", "model_names",
                    "val_indices", "val_targets", "val_predictions",
                    "test_indices", "test_targets", "test_predictions",
                }
                if set(archive.files) != expected_fields:
                    raise ValueError(f"descriptor prediction fields are incomplete: {split_id}")
                if (
                    str(archive["schema_version"].item())
                    != "prm_descriptor_predictions_v1"
                ):
                    raise ValueError(f"unsupported descriptor prediction schema: {split_id}")
                if str(archive["split_id"].item()) != split_id:
                    raise ValueError(f"descriptor prediction split mismatch: {split_id}")
                model_names = [str(name) for name in archive["model_names"].tolist()]
                arrays = {
                    name: np.asarray(archive[name])
                    for name in archive.files
                    if name not in {"schema_version", "split_id", "model_names"}
                }
            if model_names != sorted(set(model_names)):
                raise ValueError(f"descriptor model names are invalid: {split_id}")
            if set(metrics.get("results", {})) != set(model_names):
                raise ValueError(f"descriptor metrics/model names differ: {split_id}")

            for partition, prefix in (("validation", "val"), ("test", "test")):
                indices = arrays[f"{prefix}_indices"]
                raw_targets = arrays[f"{prefix}_targets"]
                predictions = np.asarray(arrays[f"{prefix}_predictions"], dtype=float)
                expected_indices = np.asarray(
                    split["val" if prefix == "val" else "test"]
                )
                if not np.array_equal(indices, expected_indices):
                    raise ValueError(f"descriptor {partition} indices differ: {split_id}")
                canonical_targets = validate_protocol_targets(
                    indices, raw_targets, protocol_targets,
                    context=f"descriptor {split_id} {partition}",
                )
                if predictions.shape != (len(model_names), len(indices)):
                    raise ValueError(
                        f"descriptor {partition} predictions misalign: {split_id}"
                    )
                if not np.isfinite(predictions).all():
                    raise ValueError(
                        f"descriptor {partition} predictions are non-finite: {split_id}"
                    )
                for index, family in enumerate(model_names):
                    metrics["results"][family][partition] = regression_metrics(
                        canonical_targets, predictions[index]
                    )

            metrics["metric_encoding"] = METRIC_ENCODING_SCHEMA
            target_dir = staging / split_id
            target_dir.mkdir()
            target_metrics = target_dir / "metrics.json"
            target_predictions = target_dir / "predictions.npz"
            target_metrics.write_text(strict_json(metrics) + "\n")
            shutil.copyfile(source_predictions, target_predictions)
            artifact_hashes[split_id] = {
                "metrics_sha256": file_sha256(target_metrics),
                "predictions_sha256": file_sha256(target_predictions),
            }

        normalized_manifest = dict(source_manifest)
        normalized_manifest["split_artifacts"] = artifact_hashes
        normalized_manifest["metric_encoding"] = {
            "schema_version": METRIC_ENCODING_SCHEMA,
            "undefined_correlations": "JSON null when either comparison vector is constant",
            "source_manifest_sha256": file_sha256(source_manifest_path),
            "normalizer_git": dict(normalizer_git),
            "normalized_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "manifest.json").write_text(strict_json(normalized_manifest) + "\n")
        staging.replace(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return normalized_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v2",
    )
    args = parser.parse_args()
    snapshot = git_snapshot()
    manifest = normalize_descriptor_root(
        args.source_root, args.out_dir, args.protocol_dir,
        normalizer_git=snapshot,
    )
    print(strict_json(manifest["metric_encoding"]))


if __name__ == "__main__":
    main()
