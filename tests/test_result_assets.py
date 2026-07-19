import pytest

from scripts.prm_make_result_assets import (
    comparison_status,
    effect_status,
    finite_format,
    latex_escape,
    render_macros,
    validate_contract,
)


def test_directional_statuses_respect_interval_signs():
    assert effect_status(-0.2, -0.1) == "improves"
    assert effect_status(0.1, 0.2) == "worsens"
    assert effect_status(-0.1, 0.2) == "inconclusive"
    assert comparison_status(0.1, 0.2) == "dart_better"
    assert comparison_status(-0.2, -0.1) == "comparator_better"
    assert comparison_status(-0.1, 0.2) == "inconclusive"


def test_latex_and_number_formatting_are_stable():
    assert latex_escape("host_pair%") == r"host\_pair\%"
    assert finite_format(-0.0001, digits=3, signed=True) == "+0.000"
    assert finite_format(float("nan")) == "--"


def minimal_claims():
    benchmark = {
        "descriptor_model": "descriptor:lightgbm",
        "models": {
            "dart": {"mae": 0.5, "low_energy_mae": 0.7, "low_energy_recall": 0.6},
            "schnet": {"mae": 0.6},
            "descriptor:lightgbm": {"mae": 0.8},
        },
    }
    return {
        "selected_variant": "g101",
        "factorial": {
            "validation_mae": {"mean": 0.4, "ci_low": 0.3, "ci_high": 0.5},
            "locked_test": {"mae": {"mean": 0.45, "ci_low": 0.35, "ci_high": 0.55}},
        },
        "benchmarks": {regime: benchmark for regime in (
            "id_cv", "pair_cv", "host_cv", "dopant_cv", "chemistry_block"
        )},
        "uq": {
            "point_prediction": {"mae": 0.4}, "gaussian_nll": 1.0,
            "mean_gaussian_crps_eV": 0.3,
            "uncertainty_absolute_error_spearman": 0.5,
            "selective_prediction": {"aurc_eV": 0.2, "excess_aurc_eV": 0.1},
            "intervals": {
                key: {"observed_test_coverage": float(key), "mean_test_width_eV": 1.0}
                for key in ("0.50", "0.80", "0.90", "0.95")
            },
        },
        "materials": {
            "defect_type_preference": {
                "accuracy": {"mean": 0.8}, "margin_mae_eV": {"mean": 0.2},
                "margin_spearman": 0.7, "global_screening_regret_eV": {"mean": 0.1},
                "global_exact_site_accuracy": {"mean": 0.4},
            },
            "within_defect_type_site_selection": {
                "exact_accuracy": {"mean": 0.5}, "top2_accuracy": {"mean": 0.7},
                "screening_regret_eV": {"mean": 0.08},
            },
        },
    }


def test_macro_rendering_uses_machine_values_without_placeholders():
    macros = render_macros(minimal_claims())
    assert r"\newcommand{\PRMSelectedVariant}{\texttt{g101}}" in macros
    assert r"\newcommand{\PRMIdCvDARTMAE}{0.500}" in macros
    assert "TODO" not in macros


def contract_inputs():
    regimes = ("id_cv", "pair_cv", "host_cv", "dopant_cv", "chemistry_block")
    clean = {"commit": "abc", "dirty": False}
    protocol = {"data_sha256": "data"}
    factorial = {
        "schema_version": "prm_factorial_bundle_v1", "collector_git": clean,
        "data_sha256": "data", "n_runs": 40,
        "selection": {
            "selected_variant": "g111", "selection_data": "validation only",
        },
    }
    descriptor_selection = {
        regime: {"selected_family": "lightgbm"} for regime in regimes
    }
    comparison = {
        "schema_version": "prm_comparison_bundle_v1", "collector_git": clean,
        "data_sha256": "data",
        "selection": {"selected_variant": "g111"},
        "descriptor_selection": descriptor_selection,
    }
    uq = {
        "schema_version": "prm_uq_results_v1", "collector_git": clean,
        "data_sha256": "data", "n_members": 5,
        "selection": {"selected_variant": "g111"},
        "calibration_contract": {"dedicated_calibration_partition": 528},
        "test": {"n": 1058},
    }
    materials = {
        "schema_version": "prm_materials_analysis_v1", "collector_git": clean,
        "data_sha256": "data",
        "selection": {"selected_variant": "g111"},
        "sample_oof": {"n": 10572},
    }
    pooled = []
    paired = []
    for regime in regimes:
        for model in ("dart", "schnet", "descriptor:lightgbm", "descriptor:mean"):
            pooled.append({"regime": regime, "model": model})
        paired.extend(
            [
                {"regime": regime, "comparator": "schnet"},
                {"regime": regime, "comparator": "descriptor:lightgbm"},
            ]
        )
    return protocol, factorial, comparison, uq, materials, pooled, paired


def test_complete_result_contract_accepts_only_aligned_evidence():
    inputs = contract_inputs()
    assert validate_contract(*inputs) == "g111"

    broken = list(inputs)
    broken[3] = dict(broken[3], data_sha256="stale")
    with pytest.raises(ValueError, match="data hash"):
        validate_contract(*broken)
