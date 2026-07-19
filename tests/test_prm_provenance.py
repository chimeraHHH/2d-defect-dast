from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.prm_metrics import regression_metrics
from src.prm_provenance import (
    archive_training_artifacts,
    config_sha256,
    file_sha256,
    load_expected_configs,
    load_protocol_targets,
    load_verified_factorial_selection,
    validate_descriptor_evidence_bundle,
    validate_manifest_config,
    validate_protocol_targets,
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


def test_prediction_targets_must_match_protocol_sample_table(tmp_path):
    sample_table = tmp_path / "samples.csv"
    sample_table.write_text("sample_index,target_eV\n0,1.5\n1,-0.25\n")
    targets = load_protocol_targets(sample_table)

    validate_protocol_targets(
        np.asarray([1, 0]), np.asarray([-0.25, 1.5]), targets,
        context="test predictions",
    )

    with pytest.raises(ValueError, match="target mismatch at sample 1"):
        validate_protocol_targets(
            np.asarray([1, 0]), np.asarray([-0.2, 1.5]), targets,
            context="test predictions",
        )


def test_prediction_targets_accept_only_exact_float32_protocol_rounding(tmp_path):
    sample_table = tmp_path / "samples.csv"
    sample_table.write_text(
        "sample_index,target_eV\n0,12.3456789012345\n1,-3.210987654321\n"
    )
    targets = load_protocol_targets(sample_table)
    rounded = np.asarray([targets[0], targets[1]], dtype=np.float32)

    validate_protocol_targets(
        np.asarray([0, 1]), rounded, targets, context="neural predictions"
    )

    changed = rounded.copy()
    changed[0] = np.nextafter(changed[0], np.float32(np.inf))
    with pytest.raises(ValueError, match="target mismatch at sample 0"):
        validate_protocol_targets(
            np.asarray([0, 1]), changed, targets, context="neural predictions"
        )


def test_prediction_targets_reject_low_precision_storage(tmp_path):
    sample_table = tmp_path / "samples.csv"
    sample_table.write_text("sample_index,target_eV\n0,1.5\n")
    targets = load_protocol_targets(sample_table)

    with pytest.raises(ValueError, match="unsupported target dtype float16"):
        validate_protocol_targets(
            np.asarray([0]), np.asarray([1.5], dtype=np.float16), targets,
            context="neural predictions",
        )


def _write_verified_factorial_selection(tmp_path):
    selection_core = {
        "rule": "minimum mean validation MAE",
        "selection_data": "validation only",
        "selected_variant": "g101",
        "validation_mae": {"mean": 0.4},
        "locked_test": {"mae": {"mean": 0.5}},
    }
    collector_git = {
        "commit": "collector-commit", "dirty": False, "status_porcelain": []
    }
    archive_records = []
    archive_names = {
        "manifest": "run_manifest.json",
        "metrics": "metrics.json",
        "split_indices": "split_indices.npz",
        "validation_predictions": "val_predictions.npz",
        "test_predictions": "test_predictions.npz",
    }
    for index in range(40):
        run_dir = tmp_path / "runs" / f"run{index:02d}"
        run_dir.mkdir(parents=True)
        artifacts = {}
        for key, name in archive_names.items():
            path = run_dir / name
            path.write_text(f"{key}-{index}")
            artifacts[key] = {
                "path": str(path),
                "archive_relative_path": str(path.relative_to(tmp_path)),
                "sha256": file_sha256(path),
            }
        archive_records.append(
            {
                "output_dir": f"factorial/g{index % 8:03b}/run{index:02d}",
                "artifacts": artifacts,
                "omitted_outputs": {
                    "checkpoint": {
                        "sha256": "a" * 64,
                        "reason": "checkpoint retained outside Git",
                    }
                },
            }
        )
    bundle = {
        "schema_version": "prm_factorial_bundle_v1",
        "collector_git": collector_git,
        "data_sha256": "data-sha",
        "n_runs": 40,
        "n_archived_runs": 40,
        "archived_runs": archive_records,
        "repeats": list(range(42, 47)),
        "selection": selection_core,
        "sources": [
            {"git": {"commit": "training-commit", "dirty": False}}
            for _ in range(40)
        ],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True))
    selection = {
        "schema_version": "prm_factorial_selection_v1",
        **selection_core,
        "factorial_bundle": "bundle.json",
        "factorial_bundle_sha256": file_sha256(bundle_path),
        "data_sha256": "data-sha",
        "n_runs": 40,
        "training_commits": ["training-commit"],
        "collector_git": collector_git,
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection, sort_keys=True))
    return selection_path, bundle_path, selection, bundle


def test_factorial_selection_is_bound_to_complete_clean_bundle(tmp_path):
    selection_path, _, expected_selection, expected_bundle = (
        _write_verified_factorial_selection(tmp_path)
    )

    selection, bundle = load_verified_factorial_selection(selection_path)

    assert selection == expected_selection
    assert bundle == expected_bundle


def test_factorial_selection_rejects_bundle_mutation(tmp_path):
    selection_path, bundle_path, _, bundle = _write_verified_factorial_selection(
        tmp_path
    )
    bundle["selection"]["selected_variant"] = "g111"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True))

    with pytest.raises(ValueError, match="bundle hash mismatch"):
        load_verified_factorial_selection(selection_path)


def test_factorial_selection_rejects_dirty_training_source(tmp_path):
    selection_path, bundle_path, selection, bundle = (
        _write_verified_factorial_selection(tmp_path)
    )
    bundle["sources"][0]["git"]["dirty"] = True
    bundle_path.write_text(json.dumps(bundle, sort_keys=True))
    selection["factorial_bundle_sha256"] = file_sha256(bundle_path)
    selection_path.write_text(json.dumps(selection, sort_keys=True))

    with pytest.raises(ValueError, match="inadmissible training sources"):
        load_verified_factorial_selection(selection_path)


def test_factorial_selection_rejects_archived_artifact_mutation(tmp_path):
    selection_path, _, _, _ = _write_verified_factorial_selection(tmp_path)
    (tmp_path / "runs" / "run00" / "metrics.json").write_text("mutated")

    with pytest.raises(ValueError, match="archived artifact hash mismatch"):
        load_verified_factorial_selection(selection_path)


def test_training_evidence_archive_copies_numerical_outputs_not_checkpoint(tmp_path):
    run_dir = tmp_path / "source" / "factorial" / "g000" / "run00"
    run_dir.mkdir(parents=True)
    output_names = {
        "metrics": "metrics.json",
        "checkpoint": "best.pt",
        "split_indices": "split_indices.npz",
        "validation_predictions": "val_predictions.npz",
        "test_predictions": "test_predictions.npz",
    }
    for key, name in output_names.items():
        (run_dir / name).write_bytes(f"{key}-content".encode())
    output_hashes = {
        key: file_sha256(run_dir / name) for key, name in output_names.items()
    }
    manifest = {
        "schema_version": "prm_run_manifest_v1",
        "status": "complete",
        "config": {"output_dir": "factorial/g000/run00"},
        "config_sha256": "config-sha",
        "git": {"commit": "training-commit", "dirty": False},
        "outputs": output_names,
        "output_sha256": output_hashes,
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    records = archive_training_artifacts(
        [manifest_path],
        tmp_path / "evidence" / "runs",
        repository_root=tmp_path,
        strip_output_prefix="factorial",
    )

    archived_dir = tmp_path / "evidence" / "runs" / "g000" / "run00"
    assert (archived_dir / "run_manifest.json").is_file()
    assert (archived_dir / "metrics.json").is_file()
    assert (archived_dir / "test_predictions.npz").is_file()
    assert not (archived_dir / "best.pt").exists()
    assert records[0]["omitted_outputs"]["checkpoint"]["sha256"] == (
        output_hashes["checkpoint"]
    )

    (run_dir / "metrics.json").write_text("mutated")
    with pytest.raises(ValueError, match="metrics source hash mismatch"):
        archive_training_artifacts(
            [manifest_path],
            tmp_path / "invalid" / "runs",
            repository_root=tmp_path,
            strip_output_prefix="factorial",
        )


def test_repository_descriptor_evidence_matches_frozen_protocol():
    root = Path(__file__).resolve().parent.parent

    bundle = validate_descriptor_evidence_bundle(
        root / "artifacts/prm_results/descriptors/manifest.json",
        root / "artifacts/prm_protocol_v2",
        repository_root=root,
    )

    assert bundle["n_archived_runs"] == 27
    assert bundle["n_sources"] == 26


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
