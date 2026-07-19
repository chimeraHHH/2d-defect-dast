import json

import numpy as np

from scripts.prm_descriptor_baselines import (
    file_sha256,
    mean_baseline,
    prior_split_provenance,
    result_is_complete,
)
from src.prm_metrics import regression_metrics


def test_result_is_complete_requires_matching_schemas(tmp_path):
    output = tmp_path / "baselines" / "descriptors" / "id_cv5_f0"
    output.mkdir(parents=True)
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "split_id": "id_cv5_f0",
                "train": [0, 1],
                "val": [2, 3],
                "test": [4, 5],
            }
        )
    )
    val_targets = np.asarray([0.0, 1.0])
    val_predictions = np.asarray([0.2, 0.8])
    test_targets = np.asarray([0.0, 2.0])
    test_predictions = np.asarray([0.5, 1.5])
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "prm_descriptor_results_v1",
                "split_id": "id_cv5_f0",
                "split_sha256": file_sha256(split_path),
                "selection_data": "validation only",
                "results": {
                    "mean": {
                        "selected_by": "fixed non-tuned baseline",
                        "selected_candidate": "training-target mean",
                        "candidates": [],
                        "validation": regression_metrics(
                            val_targets, val_predictions
                        ),
                        "test": regression_metrics(test_targets, test_predictions),
                    }
                },
            }
        )
    )
    np.savez_compressed(
        output / "predictions.npz",
        schema_version=np.asarray("prm_descriptor_predictions_v1"),
        split_id=np.asarray("id_cv5_f0"),
        model_names=np.asarray(["mean"]),
        val_indices=np.asarray([2, 3], dtype=np.int64),
        val_targets=val_targets,
        val_predictions=val_predictions[None, :],
        test_indices=np.asarray([4, 5], dtype=np.int64),
        test_targets=test_targets,
        test_predictions=test_predictions[None, :],
    )
    assert result_is_complete(tmp_path, "id_cv5_f0", split_path)
    assert not result_is_complete(tmp_path, "id_cv5_f0")

    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["test_indices"] = np.asarray([3, 5], dtype=np.int64)
    np.savez_compressed(output / "predictions.npz", **arrays)
    assert not result_is_complete(tmp_path, "id_cv5_f0", split_path)
    assert not result_is_complete(tmp_path, "id_cv5_f1", split_path)


def test_prior_manifest_is_promoted_to_per_split_provenance():
    manifest = {
        "git": {"commit": "abc", "dirty": False},
        "started_at": "start",
        "completed_at": "end",
        "splits": ["a", "b"],
    }
    provenance = prior_split_provenance(manifest)
    assert set(provenance) == {"a", "b"}
    assert provenance["a"]["git"]["commit"] == "abc"


def test_mean_baseline_uses_training_targets_only():
    samples = [
        {"metadata": {"host": "h1", "dopant": "d1"}},
        {"metadata": {"host": "h2", "dopant": "d2"}},
        {"metadata": {"host": "h3", "dopant": "d3"}},
        {"metadata": {"host": "h4", "dopant": "d4"}},
    ]
    targets = np.asarray([1.0, 3.0, 100.0, -100.0])
    indices = {
        "train": np.asarray([0, 1]),
        "val": np.asarray([2]),
        "test": np.asarray([3]),
    }
    result, val_pred, test_pred = mean_baseline(samples, targets, indices)
    assert result["training_target_mean"] == 2.0
    assert val_pred.tolist() == [2.0]
    assert test_pred.tolist() == [2.0]
