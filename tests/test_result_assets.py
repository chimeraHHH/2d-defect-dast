import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import scripts.prm_make_result_assets as result_assets
from scripts.prm_make_result_assets import (
    READY_MARKER_CONTENT,
    comparison_status,
    effect_status,
    factorial_effect_claim,
    finite_format,
    invalidate_ready_marker,
    latex_escape,
    render_applicability_table,
    render_benchmark_table,
    render_error_heterogeneity_narrative,
    render_factorial_narrative,
    render_factorial_table,
    render_macros,
    render_schnet_readout_narrative,
    render_schnet_readout_table,
    render_screening_table,
    render_transfer_narrative,
    render_uq_table,
    require_recorded_output_hashes,
    strict_json,
    text_sha256,
    validate_contract,
    validate_training_archive,
    write_manifest_then_ready_marker,
    write_ready_marker,
)


def test_directional_statuses_respect_interval_signs():
    assert effect_status(-0.2, -0.1) == "improves"
    assert effect_status(0.1, 0.2) == "worsens"
    assert effect_status(-0.1, 0.2) == "inconclusive"
    assert comparison_status(0.1, 0.2) == "dart_better"
    assert comparison_status(-0.2, -0.1) == "comparator_better"
    assert comparison_status(-0.1, 0.2) == "inconclusive"


def test_factorial_effect_claim_counts_direction_and_rejects_bad_repeats():
    row = {
        "mean": -0.2,
        "ci_low": -0.3,
        "ci_high": -0.1,
        "n_paired_repeats": 5,
        "repeat_effects": [-0.1, -0.2, 0.1, -0.3, -0.4],
    }

    claim = factorial_effect_claim(row, "validation", "G")

    assert claim["status"] == "improves"
    assert claim["direction_agreeing_repeats"] == 4
    with pytest.raises(ValueError, match="repeat effects are incomplete"):
        factorial_effect_claim(
            {**row, "repeat_effects": [-0.1, float("nan"), -0.3, -0.4, -0.5]},
            "validation",
            "G",
        )


def test_latex_and_number_formatting_are_stable():
    assert latex_escape("host_pair%") == r"host\_pair\%"
    assert finite_format(-0.0001, digits=3, signed=True) == "+0.000"
    assert finite_format(None) == "--"
    assert finite_format(float("nan")) == "--"
    with pytest.raises(ValueError, match="Out of range float values"):
        strict_json({"paper_metric": float("nan")})


def test_recorded_output_hashes_are_recomputed_before_asset_generation(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("value\n1\n")
    second.write_text("value\n2\n")
    payload = {
        "output_sha256": {
            "first": hashlib.sha256(first.read_bytes()).hexdigest(),
            "second": hashlib.sha256(second.read_bytes()).hexdigest(),
        }
    }
    paths = {"first_path": first, "second_path": second}
    expected = {"first": "first_path", "second": "second_path"}
    require_recorded_output_hashes(payload, "fixture", paths, expected)

    second.write_text("value\n3\n")
    with pytest.raises(ValueError, match="output hash mismatch for second"):
        require_recorded_output_hashes(payload, "fixture", paths, expected)


def test_training_archive_recomputes_every_artifact_hash(tmp_path):
    bundle_path = tmp_path / "results" / "bundle.json"
    archive_dir = bundle_path.parent / "runs" / "model"
    archive_dir.mkdir(parents=True)
    artifacts = {}
    for name in ("manifest", "metrics"):
        path = archive_dir / f"{name}.json"
        path.write_text(f"{name}\n")
        artifacts[name] = {
            "archive_relative_path": str(path.relative_to(bundle_path.parent)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    payload = {
        "n_archived_runs": 1,
        "archived_runs": [
            {
                "output_dir": "model",
                "config_sha256": "a" * 64,
                "git": {"commit": "training", "dirty": False},
                "artifacts": artifacts,
                "omitted_outputs": {"checkpoint": {"sha256": "b" * 64}},
            }
        ],
    }

    validate_training_archive(
        payload,
        bundle_path,
        expected_count=1,
        expected_artifacts={"manifest", "metrics"},
    )

    (archive_dir / "metrics.json").write_text("changed\n")
    with pytest.raises(ValueError, match="archived metrics hash mismatch"):
        validate_training_archive(
            payload,
            bundle_path,
            expected_count=1,
            expected_artifacts={"manifest", "metrics"},
        )


def minimal_claims():
    metrics = {
        "mae": 0.5,
        "host_macro_mae": 0.6,
        "dopant_macro_mae": 0.7,
        "favorable_mae": 0.8,
        "low_energy_mae": 0.9,
        "low_energy_recall": 0.6,
    }
    benchmark = {
        "descriptor_model": "descriptor:lightgbm",
        "models": {
            "dart": metrics,
            "schnet": {**metrics, "mae": 0.6},
            "descriptor:lightgbm": {**metrics, "mae": 0.8},
        },
        "paired_comparisons": {
            comparator: {
                "delta_comparator_minus_dart_eV": delta,
                "ci_low_eV": delta - 0.1,
                "ci_high_eV": delta + 0.1,
                "status": comparison_status(delta - 0.1, delta + 0.1),
            }
            for comparator, delta in (
                ("schnet", 0.1),
                ("descriptor:lightgbm", 0.3),
            )
        },
    }
    return {
        "selected_variant": "g101",
        "factorial": {
            "validation_mae": {"mean": 0.4, "ci_low": 0.3, "ci_high": 0.5},
            "locked_test": {"mae": {"mean": 0.45, "ci_low": 0.35, "ci_high": 0.55}},
            "effects": {
                split: {
                    term: {
                        "mean_eV": values[0],
                        "ci_low_eV": values[1],
                        "ci_high_eV": values[2],
                        "status": effect_status(values[1], values[2]),
                        "n_paired_repeats": 5,
                        "direction_agreeing_repeats": 5,
                    }
                    for term, values in (
                        ("G", (-0.2, -0.3, -0.1)),
                        ("E", (0.0, -0.1, 0.1)),
                        ("P", (0.2, 0.1, 0.3)),
                        ("G:E", (0.0, -0.1, 0.1)),
                        ("G:P", (0.0, -0.1, 0.1)),
                        ("E:P", (0.0, -0.1, 0.1)),
                        ("G:E:P", (0.0, -0.1, 0.1)),
                    )
                }
                for split in ("validation", "test")
            },
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
                key: {
                    "observed_test_coverage": float(key),
                    "mean_test_width_eV": 1.0,
                    "conformal_quantile": 1.5,
                }
                for key in ("0.50", "0.80", "0.90", "0.95")
            },
        },
        "materials": {
            "defect_type_preference": {
                "accuracy": summary(0.8),
                "margin_mae_eV": summary(0.2),
                "margin_spearman": 0.7,
                "global_screening_regret_eV": summary(0.1),
                "global_exact_site_accuracy": summary(0.4),
            },
            "within_defect_type_site_selection": {
                "exact_accuracy": summary(0.5),
                "top2_accuracy": summary(0.7),
                "screening_regret_eV": summary(0.08),
            },
            "error_heterogeneity": {
                "basis": "descriptive",
                "host": {
                    "sample_count_mae_spearman": -0.4,
                    "worst_groups": [
                        {"group": "H_1", "n": 12, "mae_eV": 0.9, "bias_eV": 0.1}
                    ],
                },
                "dopant": {
                    "sample_count_mae_spearman": -0.3,
                    "worst_groups": [
                        {"group": "X", "n": 14, "mae_eV": 0.8, "bias_eV": -0.1}
                    ],
                },
            },
        },
        "schnet_readout_sensitivity": {
            "analysis_role": "post_hoc_exploratory_robustness",
            "intervention": {
                "model_kwargs.readout": {"from": "add", "to": "mean"}
            },
            "add": {"n": 100, "mae": 1.0},
            "mean": {"n": 100, "mae": 0.8},
            "paired_host_cluster_bootstrap": {
                "mae_difference_mean_minus_add_eV": -0.2,
                "ci_low_eV": -0.3,
                "ci_high_eV": -0.1,
                "n": 100,
            },
            "fold_directional_consistency": {
                "mean_better_folds": 4,
                "n_folds": 5,
            },
            "sample_natoms_vs_absolute_error_spearman": {
                "add": 0.4,
                "mean": 0.2,
            },
            "host_median_natoms_vs_mae_spearman": {
                "add": 0.3,
                "mean": 0.1,
            },
            "fold_metrics": [
                {
                    "split_id": f"host_cv5_f{fold}",
                    "n": 20,
                    "add_mae_eV": 1.0,
                    "mean_mae_eV": 0.8,
                    "mean_minus_add_mae_eV": -0.2,
                    "mean_better": True,
                }
                for fold in range(5)
            ],
        },
    }


def summary(mean):
    return {"mean": mean, "ci_low": mean - 0.05, "ci_high": mean + 0.05, "n": 10}


def test_macro_rendering_uses_machine_values_without_placeholders():
    macros = render_macros(minimal_claims())
    assert r"\newcommand{\PRMSelectedVariant}{\texttt{g101}}" in macros
    assert r"\newcommand{\PRMIdCvDARTMAE}{0.500}" in macros
    assert r"\newcommand{\PRMSchNetMeanHostMAE}{0.800}" in macros
    assert (
        r"\newcommand{\PRMSchNetMeanSampleSizeErrorSpearman}{0.200}"
        in macros
    )
    assert "TODO" not in macros


def test_result_narratives_preserve_direction_and_inconclusive_status():
    claims = minimal_claims()

    factorial = render_factorial_narrative(claims)
    transfer = render_transfer_narrative(claims)
    readout = render_schnet_readout_narrative(claims)
    heterogeneity = render_error_heterogeneity_narrative(claims)

    assert "G$: $\\Delta=-0.200$" in factorial
    assert "(supported reduction)" in factorial
    assert "(supported increase)" in factorial
    assert "No prespecified interaction contrast" in factorial
    assert "$G$ 5/5" in factorial
    assert "descriptive uncertainty summaries" in factorial
    assert "SchNet $\\Delta=+0.100$" in transfer
    assert "(inconclusive)" in transfer
    assert "(supports lower DART error)" in transfer
    assert "from 1.000 to 0.800" in readout
    assert "(supports lower error for mean pooling)" in readout
    assert "4/5 folds" in readout
    assert "H\\_1" in heterogeneity
    assert "descriptive associations" in heterogeneity


def test_generated_table_body_rows_have_latex_terminators():
    claims = minimal_claims()
    factorial = {
        "selection": {"selected_variant": "g101"},
        "variant_summary": [
            {
                "variant": "g101",
                "validation_mae": summary(0.4),
                "test_mae": summary(0.45),
            }
        ],
    }
    tables = (
        render_factorial_table(factorial),
        render_benchmark_table(claims),
        render_applicability_table(claims),
        render_uq_table({"test": claims["uq"]}),
        render_screening_table(claims["materials"]),
        render_schnet_readout_table(claims),
    )
    for table in tables:
        body = table.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0].strip()
        rows = body.splitlines()
        assert rows
        assert all(row.endswith(r"\\") for row in rows)


def test_applicability_table_does_not_equate_nonpositive_targets_with_stability():
    table = render_applicability_table(minimal_claims())
    assert "Nonpositive-target MAE" in table
    assert "Stable MAE" not in table
    assert "no eligible nonpositive test" in table


def test_applicability_table_renders_empty_nonpositive_stratum_as_dash():
    claims = minimal_claims()
    chemistry = deepcopy(claims["benchmarks"]["chemistry_block"])
    chemistry["models"]["dart"]["favorable_mae"] = None
    claims["benchmarks"] = {
        **claims["benchmarks"], "chemistry_block": chemistry,
    }

    table = render_applicability_table(claims)
    chemistry_row = next(
        line for line in table.splitlines() if line.startswith("Chemistry block")
    )

    assert " & -- & " in chemistry_row


def test_ready_marker_is_invalidated_until_all_assets_succeed(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    stale = generated / "results_ready.tex"
    stale.write_text("stale\n")

    ready = invalidate_ready_marker(generated)
    assert ready == stale
    assert not ready.exists()

    write_ready_marker(ready)
    assert ready.read_text().endswith(r"\def\PRMResultAssetsReady{1}" + "\n")


def test_manifest_precedes_ready_marker_and_partial_marker_is_removed(
    tmp_path, monkeypatch,
):
    generated = tmp_path / "paper/generated"
    generated.mkdir(parents=True)
    ready = generated / "results_ready.tex"
    manifest_path = tmp_path / "results/paper/result_assets.json"
    marker_key = result_assets.repository_path(ready)
    manifest = {
        "schema_version": "test",
        "outputs": {marker_key: text_sha256(READY_MARKER_CONTENT)},
    }
    observed = {}

    def fail_after_partial_marker(path):
        observed["manifest"] = json.loads(manifest_path.read_text())
        path.write_text("partial\n")
        raise OSError("injected marker failure")

    monkeypatch.setattr(result_assets, "write_ready_marker", fail_after_partial_marker)
    with pytest.raises(OSError, match="injected marker failure"):
        write_manifest_then_ready_marker(manifest, manifest_path, ready)

    assert observed["manifest"] == manifest
    assert not ready.exists()


def test_main_invalidates_stale_marker_before_loading_inputs(
    tmp_path, monkeypatch,
):
    paper_dir = tmp_path / "paper"
    generated = paper_dir / "generated"
    generated.mkdir(parents=True)
    stale = generated / "results_ready.tex"
    stale.write_text(READY_MARKER_CONTENT)
    observed = {}

    def clean_snapshot():
        observed["marker_present_when_snapshotted"] = stale.exists()
        return {"commit": "clean", "dirty": False, "status_porcelain": []}

    monkeypatch.setattr(result_assets, "git_snapshot", clean_snapshot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prm_make_result_assets.py",
            "--result-root", str(tmp_path / "missing-results"),
            "--protocol-dir", str(tmp_path / "missing-protocol"),
            "--paper-dir", str(paper_dir),
        ],
    )

    with pytest.raises(FileNotFoundError):
        result_assets.main()

    assert observed["marker_present_when_snapshotted"] is True
    assert not stale.exists()


def test_manuscript_uses_only_macros_emitted_by_result_generator():
    definitions = set(
        re.findall(r"\\newcommand\{\\(PRM[A-Za-z]+)\}", render_macros(minimal_claims()))
    )
    paper_root = Path(__file__).resolve().parents[1] / "paper_Q1/sections"
    manuscript = "\n".join(
        (paper_root / name).read_text()
        for name in ("results.tex", "discussion.tex", "conclusion.tex")
    )
    uses = set(re.findall(r"\\(PRM[A-Za-z]+)", manuscript))
    assert uses <= definitions


def contract_inputs():
    regimes = ("id_cv", "pair_cv", "host_cv", "dopant_cv", "chemistry_block")
    clean = {"commit": "abc", "dirty": False}
    protocol = {
        "data_sha256": "data", "n_modeling_samples": 10224,
        "uq_split_counts": {"calibration": 511, "test": 1023},
    }
    factorial = {
        "schema_version": "prm_factorial_bundle_v1", "collector_git": clean,
        "data_sha256": "data", "n_runs": 40,
        "n_archived_runs": 40, "archived_runs": [{}] * 40,
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
        "n_archived_runs": 91, "archived_runs": [{}] * 91,
        "configuration_coverage": {
            model: {
                "n_expected": 48,
                "n_observed": 48,
                "complete": True,
                "missing_output_dirs": [],
                "expected_config_sha256": [
                    f"{index:064x}" for index in range(48)
                ],
            }
            for model in ("dart", "schnet")
        },
    }
    uq = {
        "schema_version": "prm_uq_results_v1", "collector_git": clean,
        "data_sha256": "data", "n_members": 5,
        "n_archived_runs": 5, "archived_runs": [{}] * 5,
        "selection": {"selected_variant": "g111"},
        "calibration_contract": {"dedicated_calibration_partition": 511},
        "test": {"n": 1023},
        "member_sources": [
            {
                "seed": seed,
                "config_sha256": f"{seed:064x}",
                "split_id": "uq_calibration_s62",
            }
            for seed in range(5)
        ],
    }
    materials = {
        "schema_version": "prm_materials_analysis_v1", "collector_git": clean,
        "data_sha256": "data",
        "selection": {"selected_variant": "g111"},
        "sample_oof": {"n": 10224},
        "sources": [
            {
                "config_sha256": f"{fold + 100:064x}",
                "split_id": f"pair_cv5_f{fold}",
            }
            for fold in range(5)
        ],
    }
    schnet_readout = {
        "schema_version": "prm_schnet_readout_sensitivity_bundle_v1",
        "collector_git": clean,
        "data_sha256": "data",
        "n_runs": {"add_reference": 15, "mean_sensitivity": 15},
        "n_archived_runs": 15,
        "archived_runs": [{}] * 15,
    }
    schnet_readout_summary = {
        "analysis_role": "post_hoc_exploratory_robustness",
        "intervention": {
            "model_kwargs.readout": {"from": "add", "to": "mean"}
        },
        "fold_directional_consistency": {
            "mean_better_folds": 3,
            "n_folds": 5,
        },
        "paired_host_cluster_bootstrap": {"n": 10224},
    }
    schnet_readout_folds = [
        {
            "split_id": f"host_cv5_f{fold}",
            "n": 2044 if fold == 4 else 2045,
        }
        for fold in range(5)
    ]
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
    return (
        protocol, factorial, comparison, uq, materials, schnet_readout,
        schnet_readout_summary, pooled, paired, schnet_readout_folds,
    )


def test_complete_result_contract_accepts_only_aligned_evidence():
    inputs = contract_inputs()
    assert validate_contract(*inputs) == "g111"

    broken = list(inputs)
    broken[3] = dict(broken[3], data_sha256="stale")
    with pytest.raises(ValueError, match="data hash"):
        validate_contract(*broken)

    broken = list(inputs)
    comparison = dict(broken[2])
    coverage = {
        model: dict(record)
        for model, record in comparison["configuration_coverage"].items()
    }
    coverage["schnet"]["complete"] = False
    comparison["configuration_coverage"] = coverage
    broken[2] = comparison
    with pytest.raises(ValueError, match="incomplete schnet configuration"):
        validate_contract(*broken)
