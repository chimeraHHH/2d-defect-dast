import numpy as np
import pytest

from scripts.prm_materials_analysis import (
    analyse_preferences,
    attach_screening_references,
    bootstrap_mean,
    cluster_bootstrap_mean,
    load_prediction_array,
    summarize_error_heterogeneity,
    uniform_top_k_hit_probability,
)


def row(index, host, dopant, defecttype, site, target, prediction):
    return {
        "sample_index": index,
        "host": host,
        "dopant": dopant,
        "defecttype": defecttype,
        "site": site,
        "target_eV": target,
        "prediction_eV": prediction,
    }


def test_preference_and_screening_regret_are_computed_by_host_dopant_pair():
    samples = [
        row(0, "H1", "C", "adsorbate", "ads0", 0.0, 1.0),
        row(1, "H1", "C", "adsorbate", "ads1", 1.0, 0.0),
        row(2, "H1", "C", "interstitial", "int0", 2.0, 3.0),
        row(3, "H1", "C", "interstitial", "int1", 3.0, 2.0),
        row(4, "H2", "O", "adsorbate", "ads0", 2.0, 0.0),
        row(5, "H2", "O", "interstitial", "int0", 0.0, 1.0),
    ]
    pairs, sites = analyse_preferences(samples)
    first = next(item for item in pairs if item["host"] == "H1")
    second = next(item for item in pairs if item["host"] == "H2")
    assert first["preference_correct"] == 1
    assert first["global_site_correct"] == 0
    assert first["global_screening_regret_eV"] == 1.0
    assert second["preference_correct"] == 0
    adsorbate = next(
        item for item in sites
        if item["host"] == "H1" and item["defecttype"] == "adsorbate"
    )
    assert adsorbate["exact_site_correct"] == 0
    assert adsorbate["screening_regret_eV"] == 1.0
    assert adsorbate["top2_eligible"] == 0
    assert adsorbate["true_best_in_predicted_top2"] is None
    assert all(item["host"] != "H2" for item in sites)


def test_site_selection_accepts_tied_true_minimum_and_requires_three_for_top2():
    samples = [
        row(0, "H1", "C", "adsorbate", "ads0", 0.0, 2.0),
        row(1, "H1", "C", "adsorbate", "ads1", 5e-9, 0.0),
        row(2, "H1", "C", "adsorbate", "ads2", 1.0, 1.0),
    ]

    _, sites = analyse_preferences(samples)

    assert len(sites) == 1
    assert sites[0]["n_tied_true_best_sites"] == 2
    assert sites[0]["exact_site_correct"] == 1
    assert sites[0]["top2_eligible"] == 1
    assert sites[0]["true_best_in_predicted_top2"] == 1


def test_tied_defect_type_minima_are_ineligible_for_binary_preference():
    samples = [
        row(0, "H1", "C", "adsorbate", "ads0", 0.0, 1.0),
        row(1, "H1", "C", "interstitial", "int0", 5e-9, 0.0),
    ]

    pairs, _ = analyse_preferences(samples)

    assert len(pairs) == 1
    assert pairs[0]["true_preference"] == "tie"
    assert pairs[0]["preference_eligible"] == 0
    assert pairs[0]["preference_correct"] is None


def test_screening_references_use_majority_class_and_exact_uniform_chance():
    samples = [
        row(0, "H1", "C", "adsorbate", "a0", 0.0, 0.0),
        row(1, "H1", "C", "adsorbate", "a1", 1.0, 1.0),
        row(2, "H1", "C", "interstitial", "i0", 2.0, 2.0),
        row(3, "H1", "C", "interstitial", "i1", 3.0, 3.0),
        row(4, "H2", "N", "adsorbate", "a0", 2.0, 2.0),
        row(5, "H2", "N", "adsorbate", "a1", 3.0, 3.0),
        row(6, "H2", "N", "interstitial", "i0", 0.0, 0.0),
        row(7, "H2", "N", "interstitial", "i1", 1.0, 1.0),
        row(8, "H3", "O", "adsorbate", "a0", 0.0, 0.0),
        row(9, "H3", "O", "adsorbate", "a1", 1.0, 1.0),
        row(10, "H3", "O", "interstitial", "i0", 2.0, 2.0),
        row(11, "H3", "O", "interstitial", "i1", 3.0, 3.0),
    ]

    pairs, sites = analyse_preferences(samples)
    metadata = attach_screening_references(pairs, sites)

    assert metadata == {
        "majority_class": "adsorbate",
        "class_counts": {"adsorbate": 2, "interstitial": 1},
    }
    assert [item["majority_preference_correct"] for item in pairs] == [1, 0, 1]
    assert all(item["global_uniform_exact_expectation"] == 0.25 for item in pairs)
    assert all(item["uniform_exact_expectation"] == 0.5 for item in sites)


def test_uniform_top_k_reference_accounts_for_tied_minima():
    assert uniform_top_k_hit_probability(4, 1, 2) == 0.5
    assert uniform_top_k_hit_probability(4, 2, 2) == pytest.approx(5 / 6)


def test_materials_bootstrap_is_chunked_and_requires_multiple_units():
    result = bootstrap_mean([0.0, 1.0, 2.0, 3.0], seed=1, draws=513)

    assert result["mean"] == 1.5
    assert result["n"] == 4
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]

    with pytest.raises(ValueError, match="at least two decision units"):
        bootstrap_mean([1.0], seed=1, draws=10)


def test_site_bootstrap_keeps_rows_from_each_pair_in_one_cluster():
    rows = [
        {"host": "H1", "dopant": "C", "score": 0.0},
        {"host": "H1", "dopant": "C", "score": 0.0},
        {"host": "H2", "dopant": "N", "score": 1.0},
    ]

    result = cluster_bootstrap_mean(rows, "score", seed=7, draws=5000)

    assert result["mean"] == pytest.approx(1.0 / 3.0)
    assert result["n"] == 3
    assert result["n_clusters"] == 2
    assert result["resampling_unit"] == "host_dopant_pair"
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 1.0


def test_materials_prediction_loader_preserves_target_storage_precision(tmp_path):
    path = tmp_path / "test_predictions.npz"
    np.savez(
        path,
        schema_version=np.asarray("prm_predictions_v1"),
        split=np.asarray("test"),
        split_id=np.asarray("pair_cv5_f0"),
        indices=np.asarray([1]),
        preds=np.asarray([0.5], dtype=np.float32),
        targets=np.asarray([1.234567890123], dtype=np.float32),
    )

    indices, predictions, targets = load_prediction_array(path)

    assert indices.dtype == np.int64
    assert predictions.dtype == np.float64
    assert targets.dtype == np.float32


def test_materials_prediction_loader_rejects_noninteger_indices(tmp_path):
    path = tmp_path / "test_predictions.npz"
    np.savez(
        path,
        schema_version=np.asarray("prm_predictions_v1"),
        split=np.asarray("test"),
        split_id=np.asarray("pair_cv5_f0"),
        indices=np.asarray([1.0]),
        preds=np.asarray([0.5]),
        targets=np.asarray([1.0]),
    )

    with pytest.raises(ValueError, match="indices are not integers"):
        load_prediction_array(path)


def test_materials_prediction_loader_rejects_partition_or_split_mismatch(tmp_path):
    path = tmp_path / "test_predictions.npz"
    np.savez(
        path,
        schema_version=np.asarray("prm_predictions_v1"),
        split=np.asarray("validation"),
        split_id=np.asarray("pair_cv5_f0"),
        indices=np.asarray([1]),
        preds=np.asarray([0.5]),
        targets=np.asarray([1.0]),
    )

    with pytest.raises(ValueError, match="partition mismatch"):
        load_prediction_array(path, expected_split_id="pair_cv5_f0")

    np.savez(
        path,
        schema_version=np.asarray("prm_predictions_v1"),
        split=np.asarray("test"),
        split_id=np.asarray("pair_cv5_f1"),
        indices=np.asarray([1]),
        preds=np.asarray([0.5]),
        targets=np.asarray([1.0]),
    )
    with pytest.raises(ValueError, match="split ID mismatch"):
        load_prediction_array(path, expected_split_id="pair_cv5_f0")


def test_error_heterogeneity_summarizes_groups_without_selection_claims():
    rows = [
        {"axis": "host", "group": "H1", "n": 10, "mae_eV": 0.8, "rmse_eV": 1.0, "bias_eV": 0.1},
        {"axis": "host", "group": "H2", "n": 20, "mae_eV": 0.4, "rmse_eV": 0.6, "bias_eV": -0.1},
        {"axis": "dopant", "group": "C", "n": 8, "mae_eV": 0.9, "rmse_eV": 1.1, "bias_eV": 0.2},
        {"axis": "dopant", "group": "O", "n": 18, "mae_eV": 0.3, "rmse_eV": 0.5, "bias_eV": -0.2},
        {"axis": "defecttype", "group": "adsorbate", "n": 20, "mae_eV": 0.5, "rmse_eV": 0.7, "bias_eV": 0.0},
        {"axis": "defecttype", "group": "interstitial", "n": 16, "mae_eV": 0.6, "rmse_eV": 0.8, "bias_eV": 0.1},
        {"axis": "site", "group": "ads0", "n": 12, "mae_eV": 0.4, "rmse_eV": 0.6, "bias_eV": 0.0},
        {"axis": "site", "group": "int0", "n": 10, "mae_eV": 0.7, "rmse_eV": 0.9, "bias_eV": 0.2},
    ]

    summary = summarize_error_heterogeneity(rows)

    assert summary["host"]["sample_count_mae_spearman"] == pytest.approx(-1.0)
    assert summary["dopant"]["sample_count_mae_spearman"] == pytest.approx(-1.0)
    assert summary["host"]["worst_groups"][0]["group"] == "H1"
    assert "not used for model selection" in summary["basis"]
    assert [row["group"] for row in summary["defecttype"]] == [
        "adsorbate", "interstitial",
    ]
