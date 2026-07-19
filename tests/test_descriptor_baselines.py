import json

import numpy as np

from scripts.prm_descriptor_baselines import (
    mean_baseline,
    prior_split_provenance,
    result_is_complete,
)


def test_result_is_complete_requires_matching_schemas(tmp_path):
    output = tmp_path / "baselines" / "descriptors" / "id_cv5_f0"
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "prm_descriptor_results_v1",
                "split_id": "id_cv5_f0",
            }
        )
    )
    np.savez_compressed(
        output / "predictions.npz",
        schema_version=np.asarray("prm_descriptor_predictions_v1"),
        split_id=np.asarray("id_cv5_f0"),
    )
    assert result_is_complete(tmp_path, "id_cv5_f0")
    split_path = tmp_path / "split.json"
    split_path.write_text("{}")
    assert not result_is_complete(tmp_path, "id_cv5_f0", split_path)
    assert not result_is_complete(tmp_path, "id_cv5_f1")


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
