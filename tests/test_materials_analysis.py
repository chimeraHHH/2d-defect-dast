import numpy as np
import pytest

from scripts.prm_materials_analysis import (
    analyse_preferences,
    bootstrap_mean,
    load_prediction_array,
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


def test_materials_bootstrap_is_chunked_and_requires_multiple_units():
    result = bootstrap_mean([0.0, 1.0, 2.0, 3.0], seed=1, draws=513)

    assert result["mean"] == 1.5
    assert result["n"] == 4
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]

    with pytest.raises(ValueError, match="at least two decision units"):
        bootstrap_mean([1.0], seed=1, draws=10)


def test_materials_prediction_loader_preserves_target_storage_precision(tmp_path):
    path = tmp_path / "test_predictions.npz"
    np.savez(
        path,
        schema_version=np.asarray("prm_predictions_v1"),
        indices=np.asarray([1]),
        preds=np.asarray([0.5], dtype=np.float32),
        targets=np.asarray([1.234567890123], dtype=np.float32),
    )

    indices, predictions, targets = load_prediction_array(path)

    assert indices.dtype == np.int64
    assert predictions.dtype == np.float64
    assert targets.dtype == np.float32
