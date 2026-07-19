from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    manifest["split"]["counts"] = {"train": 8, "val": 1, "test": 1}
    outputs = {
        "metrics": "metrics.json",
        "checkpoint": "best.pt",
        "split_indices": "split_indices.npz",
        "validation_predictions": "val_predictions.npz",
        "test_predictions": "test_predictions.npz",
    }
    for key, name in outputs.items():
        path = tmp_path / name
        if key == "metrics":
            path.write_text(json.dumps(manifest["metrics"], sort_keys=True))
        else:
            path.write_bytes(f"{key}-content".encode())
    manifest["outputs"] = outputs
    manifest["output_sha256"] = {
        key: file_sha256(tmp_path / name) for key, name in outputs.items()
    }
    manifest_path = tmp_path / "run_manifest.json"

    validate_training_completion(manifest, manifest_path)

    (tmp_path / "test_predictions.npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_training_completion(manifest, manifest_path)
