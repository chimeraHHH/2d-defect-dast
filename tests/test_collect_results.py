import json

import numpy as np
import pytest

from scripts.prm_collect_results import (
    file_sha256,
    paired_sample_comparison,
    regression_metrics,
    regime_for_split,
    select_descriptor_families,
    validate_descriptor_root,
    write_csv,
)


def test_regime_names_are_derived_from_frozen_split_ids():
    assert regime_for_split("id_cv5_f2") == "id_cv"
    assert regime_for_split("pair_cv5_f4") == "pair_cv"
    assert regime_for_split("host_cv5_f0") == "host_cv"
    assert regime_for_split("dopant_cv5_f3") == "dopant_cv"
    assert regime_for_split("chemistry_block_g6x3d") == "chemistry_block"


def test_descriptor_family_selection_ignores_test_metrics():
    rows = [
        {
            "model": "descriptor:a", "family": "a", "regime": "id_cv",
            "validation_mae": 0.4, "test_mae": 10.0,
        },
        {
            "model": "descriptor:b", "family": "b", "regime": "id_cv",
            "validation_mae": 0.5, "test_mae": 0.1,
        },
    ]
    selected = select_descriptor_families(rows)
    assert selected["id_cv"]["selected_family"] == "a"
    assert selected["id_cv"]["selection_data"] == "validation only"


def test_paired_bootstrap_difference_is_positive_for_worse_comparator():
    targets = np.zeros(100)
    dart = np.full(100, 0.1)
    comparator = np.full(100, 0.5)
    result = paired_sample_comparison(targets, dart, comparator, seed=1, draws=500)
    assert result["mae_difference_comparator_minus_dart_eV"] > 0.0
    assert result["ci_low_eV"] > 0.0
    assert result["n_resampling_units"] == 100


def test_paired_bootstrap_resamples_declared_clusters():
    targets = np.zeros(4)
    dart = np.asarray([0.0, 0.0, 0.0, 1.0])
    comparator = np.asarray([1.0, 1.0, 1.0, 0.0])

    result = paired_sample_comparison(
        targets, dart, comparator, seed=1, draws=500,
        groups=["host-a", "host-a", "host-a", "host-b"],
    )

    assert result["mae_difference_comparator_minus_dart_eV"] == 0.5
    assert result["n"] == 4
    assert result["n_resampling_units"] == 2


def test_pooled_metrics_include_group_and_low_energy_diagnostics():
    targets = np.arange(-5.0, 5.0)
    predictions = targets.copy()
    predictions[0] += 1.0
    hosts = ["A"] * 5 + ["B"] * 5
    dopants = ["X", "Y"] * 5

    metrics = regression_metrics(targets, predictions, hosts, dopants)

    assert metrics["spearman"] < 1.0
    assert metrics["favorable_n"] == 6
    assert metrics["low_energy_k"] == 1
    assert metrics["low_energy_mae"] == 1.0
    assert metrics["low_energy_recall"] == 1.0
    assert metrics["host_macro_mae"] == 0.1
    assert metrics["dopant_macro_mae"] == 0.1


def test_descriptor_root_requires_observed_data_and_artifact_hashes(tmp_path):
    protocol_dir = tmp_path / "protocol"
    split_dir = protocol_dir / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "id_cv5_f0.json").write_text("{}")
    protocol_path = protocol_dir / "manifest.json"
    protocol_path.write_text(json.dumps({"data_sha256": "data-sha"}))

    descriptor_root = tmp_path / "descriptors"
    result_dir = descriptor_root / "id_cv5_f0"
    result_dir.mkdir(parents=True)
    metrics_path = result_dir / "metrics.json"
    predictions_path = result_dir / "predictions.npz"
    metrics_path.write_text("{}")
    predictions_path.write_bytes(b"predictions")
    manifest = {
        "schema_version": "prm_descriptor_manifest_v2",
        "status": "complete",
        "selection_data": "validation only",
        "data_sha256": "data-sha",
        "data_file_sha256": "data-sha",
        "protocol_manifest_sha256": file_sha256(protocol_path),
        "git": {"dirty": False},
        "splits": ["id_cv5_f0"],
        "split_artifacts": {
            "id_cv5_f0": {
                "metrics_sha256": file_sha256(metrics_path),
                "predictions_sha256": file_sha256(predictions_path),
            }
        },
    }
    manifest_path = descriptor_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    validate_descriptor_root(descriptor_root, protocol_dir)

    manifest["data_file_sha256"] = "copied-not-observed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="independently verified"):
        validate_descriptor_root(descriptor_root, protocol_dir)


def test_write_csv_uses_lf_line_endings(tmp_path):
    output = tmp_path / "metrics.csv"

    write_csv(output, [{"split": "id_cv5_f0", "mae": 0.5}])

    assert output.read_bytes() == b"split,mae\nid_cv5_f0,0.5\n"
