"""Fail-closed provenance checks shared by PRM schedulers and collectors."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import yaml

from src.prm_metrics import regression_metrics


DART_ASSET_KEYS = ("ct_uae", "pretrained_embed")
_EDGE_GATE_PATTERN = re.compile(r"^\d+\.edge_gate\.")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_OUTPUT_NAMES = {
    "metrics": "metrics.json",
    "checkpoint": "best.pt",
    "split_indices": "split_indices.npz",
    "validation_predictions": "val_predictions.npz",
    "test_predictions": "test_predictions.npz",
    "calibration_predictions": "calibration_predictions.npz",
}


@dataclass(frozen=True)
class ExpectedConfig:
    path: Path
    config: Dict[str, Any]
    sha256: str


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol_targets(path: Path) -> Dict[int, float]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_index", "target_eV"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"protocol sample table lacks target fields: {path}")
        targets: Dict[int, float] = {}
        for row in reader:
            index = int(row["sample_index"])
            target = float(row["target_eV"])
            if index in targets:
                raise ValueError(f"duplicate protocol sample index {index}: {path}")
            if not math.isfinite(target):
                raise ValueError(f"non-finite protocol target at sample {index}: {path}")
            targets[index] = target
    if not targets:
        raise ValueError(f"protocol sample table is empty: {path}")
    return targets


def validate_protocol_targets(
    indices: Sequence[int] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    protocol_targets: Mapping[int, float],
    *,
    context: str,
) -> None:
    raw_indices = np.asarray(indices)
    values = np.asarray(targets, dtype=float)
    if (
        raw_indices.ndim != 1
        or not np.issubdtype(raw_indices.dtype, np.integer)
        or values.ndim != 1
        or len(raw_indices) != len(values)
        or len(np.unique(raw_indices)) != len(raw_indices)
    ):
        raise ValueError(f"{context} target vectors are not uniquely aligned")
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains non-finite targets")
    normalized_indices = raw_indices.astype(np.int64, copy=False)
    missing = [int(index) for index in normalized_indices if int(index) not in protocol_targets]
    if missing:
        raise ValueError(f"{context} contains unknown protocol indices: {missing[:10]}")
    expected = np.asarray(
        [protocol_targets[int(index)] for index in normalized_indices], dtype=float
    )
    matches = np.isclose(values, expected, rtol=0.0, atol=1e-10)
    if not np.all(matches):
        position = int(np.flatnonzero(~matches)[0])
        index = int(normalized_indices[position])
        raise ValueError(
            f"{context} target mismatch at sample {index}: "
            f"observed={values[position]:.16g}, protocol={expected[position]:.16g}"
        )


def load_expected_configs(paths: Iterable[Path]) -> Dict[str, ExpectedConfig]:
    expected: Dict[str, ExpectedConfig] = {}
    for path in paths:
        resolved = path.resolve()
        config = yaml.safe_load(resolved.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"configuration is not a mapping: {resolved}")
        output_dir = str(config.get("output_dir", ""))
        if not output_dir:
            raise ValueError(f"configuration has no output_dir: {resolved}")
        if output_dir in expected:
            raise ValueError(
                f"duplicate expected output_dir {output_dir!r}: "
                f"{expected[output_dir].path} and {resolved}"
            )
        expected[output_dir] = ExpectedConfig(
            path=resolved,
            config=config,
            sha256=config_sha256(config),
        )
    return expected


def validate_manifest_config(
    manifest: Mapping[str, Any],
    expected_configs: Mapping[str, ExpectedConfig],
    manifest_path: Path,
) -> ExpectedConfig:
    embedded = manifest.get("config")
    if not isinstance(embedded, dict):
        raise ValueError(f"run manifest has no embedded configuration: {manifest_path}")
    output_dir = str(embedded.get("output_dir", ""))
    expected = expected_configs.get(output_dir)
    if expected is None:
        raise ValueError(
            f"run output {output_dir!r} has no controlled configuration: {manifest_path}"
        )
    embedded_hash = config_sha256(embedded)
    recorded_hash = manifest.get("config_sha256")
    if recorded_hash != embedded_hash:
        raise ValueError(f"run manifest config/hash mismatch: {manifest_path}")
    if embedded_hash != expected.sha256:
        raise ValueError(
            f"stale run configuration at {manifest_path}; expected {expected.path}"
        )
    return expected


def validate_training_completion(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    verify_outputs: bool = True,
) -> None:
    """Require a full epoch history before a neural run can be admitted."""
    if manifest.get("status") != "complete":
        raise ValueError(f"training run is not complete: {manifest_path}")
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError(f"training execution contract is missing: {manifest_path}")
    if int(execution.get("max_steps", -1)) != 0:
        raise ValueError(f"truncated training run is inadmissible: {manifest_path}")

    config = manifest.get("config")
    metrics = manifest.get("metrics")
    if not isinstance(config, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError(f"training config or metrics are missing: {manifest_path}")
    if float(config.get("label_noise_std", 0.0)) > 0 and (
        execution.get("label_noise_stream") != "model_seed_and_epoch_v1"
    ):
        raise ValueError(f"label-noise stream is not reproducible: {manifest_path}")
    epochs = int(config.get("epochs", 0))
    history = metrics.get("history")
    if epochs <= 0 or not isinstance(history, list) or len(history) != epochs:
        raise ValueError(
            f"training history does not cover all {epochs} epochs: {manifest_path}"
        )
    observed_epochs = [int(row.get("epoch", -1)) for row in history]
    if observed_epochs != list(range(1, epochs + 1)):
        raise ValueError(f"training epoch sequence is incomplete: {manifest_path}")
    history_values = [row.get("val_mae") for row in history]
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in history_values
    ):
        raise ValueError(f"training history contains invalid validation MAE: {manifest_path}")

    best_val = metrics.get("best_val_mae")
    if not isinstance(best_val, (int, float)) or not math.isfinite(best_val):
        raise ValueError(f"best validation MAE is invalid: {manifest_path}")
    if not math.isclose(best_val, min(history_values), rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError(f"best validation MAE disagrees with history: {manifest_path}")
    required_metrics = ("mae", "rmse", "bias", "spearman", "r2")
    for partition in ("validation", "test"):
        values = metrics.get(partition)
        if not isinstance(values, Mapping):
            raise ValueError(f"{partition} metrics are missing: {manifest_path}")
        for key in required_metrics:
            value = values.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(
                    f"invalid {partition} {key} metric in {manifest_path}"
                )
    if not math.isclose(
        float(metrics["validation"]["mae"]), float(best_val),
        rel_tol=1e-7, abs_tol=1e-8,
    ):
        raise ValueError(f"final validation MAE is not the selected checkpoint: {manifest_path}")
    if int(metrics.get("n_params", 0)) <= 0:
        raise ValueError(f"model parameter count is invalid: {manifest_path}")
    split = manifest.get("split", {})
    if metrics.get("split_id") != split.get("split_id"):
        raise ValueError(f"metric/split identity mismatch: {manifest_path}")
    if manifest.get("seed") != config.get("seed"):
        raise ValueError(f"manifest/config seed mismatch: {manifest_path}")
    if verify_outputs:
        validate_output_artifacts(manifest, manifest_path)


def validate_output_artifacts(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Verify every paper-facing output against its recorded digest."""
    outputs = manifest.get("outputs")
    hashes = manifest.get("output_sha256")
    if not isinstance(outputs, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError(f"output artifact contract is missing: {manifest_path}")
    required = {
        "metrics", "checkpoint", "split_indices",
        "validation_predictions", "test_predictions",
    }
    split_counts = manifest.get("split", {}).get("counts", {})
    if int(split_counts.get("calibration", 0)) > 0:
        required.add("calibration_predictions")
    if set(outputs) != required or set(hashes) != required:
        raise ValueError(f"output artifact set is incomplete: {manifest_path}")

    resolved: Dict[str, Path] = {}
    for key in sorted(required):
        relative = outputs.get(key)
        if relative != _OUTPUT_NAMES[key]:
            raise ValueError(f"noncanonical {key} output path: {manifest_path}")
        expected_hash = hashes.get(key)
        if not isinstance(expected_hash, str) or _SHA256_PATTERN.fullmatch(expected_hash) is None:
            raise ValueError(f"invalid {key} output hash: {manifest_path}")
        path = manifest_path.parent / relative
        if not path.is_file():
            raise ValueError(f"missing {key} output artifact: {path}")
        if file_sha256(path) != expected_hash:
            raise ValueError(f"{key} output hash mismatch: {path}")
        resolved[key] = path

    try:
        recorded_metrics = json.loads(resolved["metrics"].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable metrics artifact: {resolved['metrics']}") from exc
    if recorded_metrics != manifest.get("metrics"):
        raise ValueError(f"metrics file/manifest mismatch: {manifest_path}")
    validate_prediction_artifacts(manifest, manifest_path, resolved)


def validate_prediction_artifacts(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    resolved_outputs: Mapping[str, Path],
) -> None:
    """Recompute partition metrics from the frozen prediction artifacts."""
    split_id = manifest.get("split", {}).get("split_id")
    split_counts = manifest.get("split", {}).get("counts", {})
    partition_names = ["train", "val", "test"]
    if "calibration_predictions" in resolved_outputs:
        partition_names.append("calibration")

    with np.load(resolved_outputs["split_indices"], allow_pickle=False) as archive:
        expected_keys = {"schema_version", "split_id", *partition_names}
        if set(archive.files) != expected_keys:
            raise ValueError(f"split-index partitions are incomplete: {manifest_path}")
        if str(archive["schema_version"].item()) != "prm_split_indices_v1":
            raise ValueError(f"unsupported split-index schema: {manifest_path}")
        if str(archive["split_id"].item()) != split_id:
            raise ValueError(f"split-index identity mismatch: {manifest_path}")
        partitions = {}
        for name in partition_names:
            raw = np.asarray(archive[name])
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"invalid {name} split indices: {manifest_path}")
            indices = raw.astype(np.int64, copy=False)
            if np.any(indices < 0) or len(np.unique(indices)) != len(indices):
                raise ValueError(f"non-unique {name} split indices: {manifest_path}")
            if len(indices) != int(split_counts.get(name, -1)):
                raise ValueError(f"{name} split count mismatch: {manifest_path}")
            partitions[name] = indices
    for left_index, left_name in enumerate(partition_names):
        for right_name in partition_names[left_index + 1:]:
            if np.intersect1d(partitions[left_name], partitions[right_name]).size:
                raise ValueError(
                    f"overlapping {left_name}/{right_name} split indices: {manifest_path}"
                )

    prediction_contracts = [
        ("validation", "validation_predictions", "val"),
        ("test", "test_predictions", "test"),
    ]
    if "calibration_predictions" in resolved_outputs:
        prediction_contracts.append(
            ("calibration", "calibration_predictions", "calibration")
        )
    metric_keys = ("mae", "rmse", "bias", "pearson", "spearman", "r2")
    for metric_partition, output_key, split_name in prediction_contracts:
        path = resolved_outputs[output_key]
        with np.load(path, allow_pickle=False) as archive:
            expected_keys = {
                "schema_version", "split_id", "split", "indices", "preds", "targets",
            }
            if set(archive.files) != expected_keys:
                raise ValueError(f"prediction artifact fields are incomplete: {path}")
            if str(archive["schema_version"].item()) != "prm_predictions_v1":
                raise ValueError(f"unsupported prediction schema: {path}")
            if str(archive["split_id"].item()) != split_id:
                raise ValueError(f"prediction split identity mismatch: {path}")
            if str(archive["split"].item()) != split_name:
                raise ValueError(f"prediction partition mismatch: {path}")
            raw_indices = np.asarray(archive["indices"])
            predictions = np.asarray(archive["preds"], dtype=float)
            targets = np.asarray(archive["targets"], dtype=float)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError(f"invalid prediction indices: {path}")
        indices = raw_indices.astype(np.int64, copy=False)
        if (
            predictions.ndim != 1
            or targets.ndim != 1
            or len(indices) != len(predictions)
            or len(indices) != len(targets)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError(f"unaligned prediction vectors: {path}")
        if not np.array_equal(np.sort(indices), np.sort(partitions[split_name])):
            raise ValueError(f"prediction indices do not match frozen split: {path}")
        if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
            raise ValueError(f"non-finite prediction values: {path}")
        recomputed = regression_metrics(targets, predictions)
        recorded = manifest.get("metrics", {}).get(metric_partition, {})
        if int(recorded.get("n", -1)) != recomputed["n"]:
            raise ValueError(f"recorded prediction count mismatch: {path}")
        for key in metric_keys:
            if not math.isclose(
                float(recorded.get(key, float("nan"))),
                float(recomputed[key]),
                rel_tol=1e-7,
                abs_tol=1e-8,
            ):
                raise ValueError(f"recorded {metric_partition} {key} mismatch: {path}")


def validate_dart_assets(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Reject DART runs without the exact assets and audited partial load."""
    config = manifest.get("config")
    if not isinstance(config, Mapping) or not config.get("asset_integrity_required"):
        raise ValueError(f"DART run has no required asset contract: {manifest_path}")
    expected = config.get("asset_sha256")
    if not isinstance(expected, Mapping) or set(expected) != set(DART_ASSET_KEYS):
        raise ValueError(f"DART asset hash contract is incomplete: {manifest_path}")
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != set(DART_ASSET_KEYS):
        raise ValueError(f"DART run asset records are incomplete: {manifest_path}")
    for key in DART_ASSET_KEYS:
        record = assets[key]
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid {key} asset record: {manifest_path}")
        if record.get("sha256") != expected[key]:
            raise ValueError(f"{key} asset hash mismatch: {manifest_path}")
        if record.get("expected_sha256") != expected[key]:
            raise ValueError(f"{key} expected hash was not enforced: {manifest_path}")
        if int(record.get("size_bytes", 0)) <= 0:
            raise ValueError(f"empty {key} asset: {manifest_path}")

    report = manifest.get("pretraining")
    if not isinstance(report, Mapping):
        raise ValueError(f"DART run has no pretraining report: {manifest_path}")
    if report.get("schema_version") != "prm_pretraining_report_v1":
        raise ValueError(f"unsupported DART pretraining report: {manifest_path}")
    if report.get("checkpoint_sha256") != expected["pretrained_embed"]:
        raise ValueError(f"pretraining checkpoint/report mismatch: {manifest_path}")
    if not report.get("source_dataset"):
        raise ValueError(f"pretraining source dataset is missing: {manifest_path}")
    source_test_mae = report.get("source_test_mae_eV")
    if not isinstance(source_test_mae, (int, float)) or not math.isfinite(source_test_mae):
        raise ValueError(f"pretraining source metric is invalid: {manifest_path}")
    if report.get("all_model_parameters_trainable") is not True:
        raise ValueError(f"pretrained DART parameters were not all trainable: {manifest_path}")

    model_kwargs = config.get("model_kwargs", {})
    hidden_dim = int(model_kwargs.get("hidden_dim", 0))
    atom_fea_len = int(model_kwargs.get("atom_fea_len", 0))
    projection = report.get("input_projection", {})
    checkpoint_shape = projection.get("checkpoint_weight_shape")
    model_shape = projection.get("model_weight_shape")
    if checkpoint_shape != [hidden_dim, atom_fea_len]:
        raise ValueError(f"pretraining input slice is incompatible: {manifest_path}")
    if (
        not isinstance(model_shape, list)
        or len(model_shape) != 2
        or model_shape[0] != hidden_dim
        or model_shape[1] < atom_fea_len
    ):
        raise ValueError(f"invalid DART input projection report: {manifest_path}")
    if projection.get("copied_rows") != [0, hidden_dim]:
        raise ValueError(f"incomplete pretrained input rows: {manifest_path}")
    if projection.get("copied_columns") != [0, atom_fea_len]:
        raise ValueError(f"incomplete pretrained input columns: {manifest_path}")
    if projection.get("bias_copied") is not True:
        raise ValueError(f"pretrained input bias was not copied: {manifest_path}")
    if projection.get("seeded_model_columns") != model_shape[1] - atom_fea_len:
        raise ValueError(f"seeded input-column count is inconsistent: {manifest_path}")

    local = report.get("local_layers", {})
    loaded_keys = local.get("loaded_keys")
    seeded_keys = local.get("seeded_model_keys")
    if not isinstance(loaded_keys, list) or not isinstance(seeded_keys, list):
        raise ValueError(f"invalid local-layer pretraining report: {manifest_path}")
    if len(set(loaded_keys)) != len(loaded_keys) or len(set(seeded_keys)) != len(seeded_keys):
        raise ValueError(f"duplicate local-layer keys in report: {manifest_path}")
    if local.get("loaded_tensor_count") != len(loaded_keys):
        raise ValueError(f"loaded local-layer count is inconsistent: {manifest_path}")
    if local.get("checkpoint_tensor_count") != len(loaded_keys):
        raise ValueError(f"not all checkpoint local tensors were loaded: {manifest_path}")
    if local.get("model_tensor_count") != len(loaded_keys) + len(seeded_keys):
        raise ValueError(f"model local-layer count is inconsistent: {manifest_path}")
    if local.get("unexpected_checkpoint_keys") or local.get("shape_mismatches"):
        raise ValueError(f"incompatible checkpoint tensors were accepted: {manifest_path}")
    if any(_EDGE_GATE_PATTERN.match(key) is None for key in seeded_keys):
        raise ValueError(f"non-gate local tensors remained seeded: {manifest_path}")
    expected_seeded = (
        4 * int(model_kwargs.get("n_local_layers", 0))
        if model_kwargs.get("use_prenorm_local") else 0
    )
    if len(seeded_keys) != expected_seeded:
        raise ValueError(f"unexpected number of seeded edge-gate tensors: {manifest_path}")
