from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.prm_metrics import regression_metrics
from src.prm_provenance import (
    config_sha256,
    file_sha256,
    load_expected_configs,
    validate_manifest_config,
    validate_training_completion,
)


def _write_config(path, output_dir, epochs=150):
    path.write_text(
        f"output_dir: {output_dir}\n"
        "split_path: artifacts/split.json\n"
        "seed: 142\n"
        f"epochs: {epochs}\n"
    )


def test_manifest_configuration_must_match_controlled_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, "factorial/g000/split42_seed142")
    expected = load_expected_configs([config_path])
    config = expected["factorial/g000/split42_seed142"].config
    manifest = {"config": config, "config_sha256": config_sha256(config)}

    record = validate_manifest_config(manifest, expected, tmp_path / "run_manifest.json")
    assert record.path == config_path.resolve()

    stale = {**config, "epochs": 151}
    stale_manifest = {"config": stale, "config_sha256": config_sha256(stale)}
    with pytest.raises(ValueError, match="stale run configuration"):
        validate_manifest_config(stale_manifest, expected, tmp_path / "run_manifest.json")


def test_manifest_rejects_an_internally_false_hash(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, "factorial/g000/split42_seed142")
    expected = load_expected_configs([config_path])
    config = expected["factorial/g000/split42_seed142"].config
    manifest = {"config": config, "config_sha256": "0" * 64}

    with pytest.raises(ValueError, match="config/hash mismatch"):
        validate_manifest_config(manifest, expected, tmp_path / "run_manifest.json")


def test_expected_configs_require_unique_output_directories(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    _write_config(first, "same/output")
    _write_config(second, "same/output")
    with pytest.raises(ValueError, match="duplicate expected output_dir"):
        load_expected_configs([first, second])


def _complete_manifest():
    config = {"epochs": 2, "seed": 142}
    return {
        "status": "complete",
        "execution": {"max_steps": 0, "resume_requested": False},
        "config": config,
        "seed": 142,
        "split": {"split_id": "id_repeat_s42"},
        "metrics": {
            "split_id": "id_repeat_s42",
            "n_params": 100,
            "best_val_mae": 0.4,
            "history": [
                {"epoch": 1, "val_mae": 0.5},
                {"epoch": 2, "val_mae": 0.4},
            ],
            "validation": {
                "mae": 0.4, "rmse": 0.5, "bias": 0.0,
                "spearman": 0.7, "r2": 0.6,
            },
            "test": {
                "mae": 0.45, "rmse": 0.55, "bias": 0.01,
                "spearman": 0.65, "r2": 0.5,
            },
        },
    }


def test_complete_training_requires_every_configured_epoch():
    manifest = _complete_manifest()
    validate_training_completion(
        manifest, Path("run_manifest.json"), verify_outputs=False
    )

    manifest["metrics"]["history"].pop()
    with pytest.raises(ValueError, match="does not cover all"):
        validate_training_completion(
            manifest, Path("run_manifest.json"), verify_outputs=False
        )


def test_truncated_training_cannot_be_admitted():
    manifest = _complete_manifest()
    manifest["execution"]["max_steps"] = 1
    with pytest.raises(ValueError, match="truncated"):
        validate_training_completion(
            manifest, Path("run_manifest.json"), verify_outputs=False
        )


def test_noisy_training_requires_the_independent_rng_stream():
    manifest = _complete_manifest()
    manifest["config"]["label_noise_std"] = 0.03
    with pytest.raises(ValueError, match="label-noise stream"):
        validate_training_completion(
            manifest, Path("run_manifest.json"), verify_outputs=False
        )

    manifest["execution"]["label_noise_stream"] = "model_seed_and_epoch_v1"
    validate_training_completion(
        manifest, Path("run_manifest.json"), verify_outputs=False
    )


def test_complete_training_verifies_every_output_digest(tmp_path):
    manifest = _complete_manifest()
    split_id = manifest["split"]["split_id"]
    split_indices = {
        "train": np.asarray([0, 1], dtype=np.int64),
        "val": np.asarray([2, 3], dtype=np.int64),
        "test": np.asarray([4, 5], dtype=np.int64),
    }
    validation_targets = np.asarray([0.0, 1.0])
    validation_predictions = np.asarray([0.2, 0.8])
    test_targets = np.asarray([0.0, 2.0])
    test_predictions = np.asarray([0.5, 1.5])
    manifest["split"]["counts"] = {key: len(value) for key, value in split_indices.items()}
    manifest["metrics"]["validation"] = regression_metrics(
        validation_targets, validation_predictions
    )
    manifest["metrics"]["test"] = regression_metrics(test_targets, test_predictions)
    manifest["metrics"]["best_val_mae"] = manifest["metrics"]["validation"]["mae"]
    manifest["metrics"]["history"][-1]["val_mae"] = manifest["metrics"]["best_val_mae"]
    outputs = {
        "metrics": "metrics.json",
        "checkpoint": "best.pt",
        "split_indices": "split_indices.npz",
        "validation_predictions": "val_predictions.npz",
        "test_predictions": "test_predictions.npz",
    }
    (tmp_path / "metrics.json").write_text(json.dumps(manifest["metrics"], sort_keys=True))
    (tmp_path / "best.pt").write_bytes(b"checkpoint-content")
    np.savez_compressed(
        tmp_path / "split_indices.npz",
        schema_version=np.asarray("prm_split_indices_v1"),
        split_id=np.asarray(split_id),
        **split_indices,
    )
    for name, indices, targets, predictions, split_name in (
        ("val_predictions.npz", split_indices["val"], validation_targets, validation_predictions, "val"),
        ("test_predictions.npz", split_indices["test"], test_targets, test_predictions, "test"),
    ):
        np.savez_compressed(
            tmp_path / name,
            schema_version=np.asarray("prm_predictions_v1"),
            split_id=np.asarray(split_id),
            split=np.asarray(split_name),
            indices=indices,
            preds=predictions,
            targets=targets,
        )
    manifest["outputs"] = outputs
    manifest["output_sha256"] = {
        key: file_sha256(tmp_path / name) for key, name in outputs.items()
    }
    manifest_path = tmp_path / "run_manifest.json"

    validate_training_completion(manifest, manifest_path)

    (tmp_path / "test_predictions.npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_training_completion(manifest, manifest_path)

    np.savez_compressed(
        tmp_path / "test_predictions.npz",
        schema_version=np.asarray("prm_predictions_v1"),
        split_id=np.asarray(split_id),
        split=np.asarray("test"),
        indices=split_indices["test"],
        preds=np.asarray([0.0, 2.0]),
        targets=test_targets,
    )
    manifest["output_sha256"]["test_predictions"] = file_sha256(
        tmp_path / "test_predictions.npz"
    )
    with pytest.raises(ValueError, match="recorded test mae mismatch"):
        validate_training_completion(manifest, manifest_path)
