import json

import numpy as np

from scripts.prm_descriptor_baselines import prior_split_provenance, result_is_complete


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
