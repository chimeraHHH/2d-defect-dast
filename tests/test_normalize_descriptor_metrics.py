import json

import numpy as np

from scripts.prm_normalize_descriptor_metrics import (
    file_sha256,
    normalize_descriptor_root,
)


def test_descriptor_normalizer_encodes_constant_correlations_as_json_null(tmp_path):
    protocol_dir = tmp_path / "protocol"
    splits_dir = protocol_dir / "splits"
    splits_dir.mkdir(parents=True)
    protocol_path = protocol_dir / "manifest.json"
    protocol_path.write_text(json.dumps({"data_sha256": "data"}) + "\n")
    (protocol_dir / "samples.csv").write_text(
        "sample_index,target_eV\n0,0.0\n1,1.0\n2,2.0\n3,3.0\n"
    )
    split_id = "id_cv5_f0"
    split_path = splits_dir / f"{split_id}.json"
    split_path.write_text(
        json.dumps(
            {
                "split_id": split_id,
                "train": [], "val": [0, 1], "test": [2, 3],
            }
        ) + "\n"
    )

    source = tmp_path / "source"
    split_root = source / split_id
    split_root.mkdir(parents=True)
    metrics_path = split_root / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": "prm_descriptor_results_v1",
                "split_id": split_id,
                "split_sha256": file_sha256(split_path),
                "results": {
                    "mean": {
                        "validation": {
                            "pearson": float("nan"), "spearman": float("nan")
                        },
                        "test": {"pearson": float("nan"), "spearman": float("nan")},
                    }
                },
            },
            indent=2,
        ) + "\n"
    )
    predictions_path = split_root / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        schema_version=np.asarray("prm_descriptor_predictions_v1"),
        split_id=np.asarray(split_id),
        model_names=np.asarray(["mean"]),
        val_indices=np.asarray([0, 1]),
        val_targets=np.asarray([0.0, 1.0]),
        val_predictions=np.asarray([[0.5, 0.5]]),
        test_indices=np.asarray([2, 3]),
        test_targets=np.asarray([2.0, 3.0]),
        test_predictions=np.asarray([[0.5, 0.5]]),
    )
    source_manifest = {
        "schema_version": "prm_descriptor_manifest_v2",
        "status": "complete",
        "data_sha256": "data",
        "protocol_manifest_sha256": file_sha256(protocol_path),
        "splits": [split_id],
        "split_artifacts": {
            split_id: {
                "metrics_sha256": file_sha256(metrics_path),
                "predictions_sha256": file_sha256(predictions_path),
            }
        },
    }
    (source / "manifest.json").write_text(json.dumps(source_manifest) + "\n")

    output = tmp_path / "normalized"
    manifest = normalize_descriptor_root(
        source,
        output,
        protocol_dir,
        normalizer_git={"commit": "normalizer", "dirty": False},
    )

    normalized_text = (output / split_id / "metrics.json").read_text()
    normalized = json.loads(normalized_text)
    assert "NaN" not in normalized_text
    assert normalized["results"]["mean"]["validation"]["pearson"] is None
    assert normalized["results"]["mean"]["validation"]["spearman"] is None
    assert normalized["results"]["mean"]["test"]["pearson"] is None
    assert normalized["results"]["mean"]["test"]["spearman"] is None
    assert manifest["metric_encoding"]["normalizer_git"]["commit"] == "normalizer"
    assert manifest["split_artifacts"][split_id]["metrics_sha256"] == file_sha256(
        output / split_id / "metrics.json"
    )
