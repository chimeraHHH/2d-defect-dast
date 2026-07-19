import numpy as np

from scripts.prm_collect_results import (
    paired_sample_comparison,
    regression_metrics,
    regime_for_split,
    select_descriptor_families,
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
