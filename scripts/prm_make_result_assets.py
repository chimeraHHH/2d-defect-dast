"""Generate paper figures, tables, macros, and claims from canonical bundles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
REGIME_ORDER = ("id_cv", "pair_cv", "host_cv", "dopant_cv", "chemistry_block")
REGIME_LABELS = {
    "id_cv": "Random OOF",
    "pair_cv": "Pair OOF",
    "host_cv": "Host OOF",
    "dopant_cv": "Impurity OOF",
    "chemistry_block": "Chemistry block",
}
REGIME_MACROS = {
    "id_cv": "IdCv",
    "pair_cv": "PairCv",
    "host_cv": "HostCv",
    "dopant_cv": "DopantCv",
    "chemistry_block": "ChemistryBlock",
}
TERM_ORDER = ("G", "E", "P", "G:E", "G:P", "E:P", "G:E:P")
COLORS = {
    "dart": "#2A6F97",
    "schnet": "#C15B38",
    "descriptor": "#3B7D5A",
    "mean": "#777777",
    "validation": "#3B7EA1",
    "test": "#B64B4B",
    "oracle": "#222222",
    "grid": "#D9D9D9",
    "ink": "#222222",
    "muted": "#666666",
}
RECORDED_OUTPUTS = {
    "comparison": {
        "run_metrics": "comparison_run_metrics",
        "fold_metrics": "comparison_fold_metrics",
        "summary": "comparison_summary",
        "pooled_metrics": "pooled",
        "paired_comparisons": "paired",
        "descriptor_selection": "comparison_descriptor_selection",
    },
    "uq": {
        "predictions": "uq_predictions",
        "interval_calibration": "uq_intervals",
        "risk_coverage": "uq_risk",
    },
    "materials": {
        "sample_predictions": "sample_predictions",
        "pair_preferences": "pair_preferences",
        "site_selection": "site_selection",
        "group_errors": "group_errors",
    },
}
READY_MARKER_CONTENT = (
    "% Auto-generated after all canonical paper assets succeeded; do not edit.\n"
    "\\def\\PRMResultAssetsReady{1}\n"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def strict_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def git_snapshot() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    return {
        "commit": commit or None, "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require_schema(payload: Mapping[str, Any], expected: str, name: str) -> None:
    if payload.get("schema_version") != expected:
        raise ValueError(
            f"{name} schema is {payload.get('schema_version')!r}, expected {expected!r}"
        )


def require_clean_collector(payload: Mapping[str, Any], name: str) -> None:
    snapshot = payload.get("collector_git", {})
    if not snapshot.get("commit"):
        raise ValueError(f"{name} has no collector commit")
    if snapshot.get("dirty"):
        raise ValueError(f"{name} was collected from a dirty worktree")


def require_recorded_output_hashes(
    payload: Mapping[str, Any], name: str, paths: Mapping[str, Path],
    expected: Mapping[str, str],
) -> None:
    recorded = payload.get("output_sha256")
    if not isinstance(recorded, Mapping) or set(recorded) != set(expected):
        raise ValueError(f"{name} output hash contract is incomplete")
    for output_name, path_key in expected.items():
        observed = file_sha256(paths[path_key])
        if recorded[output_name] != observed:
            raise ValueError(
                f"{name} output hash mismatch for {output_name}: "
                f"recorded={recorded[output_name]}, observed={observed}"
            )


def validate_contract(
    protocol: Mapping[str, Any], factorial: Mapping[str, Any],
    comparison: Mapping[str, Any], uq: Mapping[str, Any],
    materials: Mapping[str, Any], pooled_rows: Sequence[Mapping[str, str]],
    paired_rows: Sequence[Mapping[str, str]],
) -> str:
    require_schema(factorial, "prm_factorial_bundle_v1", "factorial bundle")
    require_schema(comparison, "prm_comparison_bundle_v1", "comparison bundle")
    require_schema(uq, "prm_uq_results_v1", "UQ bundle")
    require_schema(materials, "prm_materials_analysis_v1", "materials bundle")
    for name, payload in (
        ("factorial bundle", factorial), ("comparison bundle", comparison),
        ("UQ bundle", uq), ("materials bundle", materials),
    ):
        require_clean_collector(payload, name)

    data_sha256 = str(protocol["data_sha256"])
    observed_hashes = {
        str(factorial.get("data_sha256")), str(comparison.get("data_sha256")),
        str(uq.get("data_sha256")), str(materials.get("data_sha256")),
    }
    if observed_hashes != {data_sha256}:
        raise ValueError(f"result bundles do not share the protocol data hash: {observed_hashes}")

    factorial_selection = factorial.get("selection", {})
    if factorial_selection.get("selection_data") != "validation only":
        raise ValueError("factorial architecture was not selected on validation data only")
    selected_variant = str(factorial_selection.get("selected_variant"))
    variants = {
        selected_variant,
        str(comparison.get("selection", {}).get("selected_variant")),
        str(uq.get("selection", {}).get("selected_variant")),
        str(materials.get("selection", {}).get("selected_variant")),
    }
    if len(variants) != 1 or not selected_variant.startswith("g"):
        raise ValueError(f"selected architecture differs across bundles: {variants}")
    if int(factorial.get("n_runs", -1)) != 40:
        raise ValueError("factorial bundle must contain all 40 paired runs")
    if int(uq.get("n_members", -1)) != 5:
        raise ValueError("UQ bundle must contain five ensemble members")
    expected_uq = protocol.get("uq_split_counts", {})
    expected_calibration = int(expected_uq.get("calibration", -1))
    expected_test = int(expected_uq.get("test", -1))
    expected_modeling = int(protocol.get("n_modeling_samples", -1))
    if int(uq.get("calibration_contract", {}).get("dedicated_calibration_partition", -1)) != expected_calibration:
        raise ValueError("UQ bundle does not use the frozen calibration partition")
    if int(uq.get("test", {}).get("n", -1)) != expected_test:
        raise ValueError("UQ test set does not match the frozen split")
    if int(materials.get("sample_oof", {}).get("n", -1)) != expected_modeling:
        raise ValueError("materials OOF predictions do not cover the canonical set")

    pooled_keys = {(row["regime"], row["model"]) for row in pooled_rows}
    descriptor_selection = comparison.get("descriptor_selection", {})
    for regime in REGIME_ORDER:
        descriptor = f"descriptor:{descriptor_selection[regime]['selected_family']}"
        for model in ("dart", "schnet", descriptor, "descriptor:mean"):
            if (regime, model) not in pooled_keys:
                raise ValueError(f"missing pooled metrics for {regime}/{model}")
        comparators = {
            row["comparator"] for row in paired_rows if row["regime"] == regime
        }
        if "schnet" not in comparators or descriptor not in comparators:
            raise ValueError(f"missing paired comparisons for {regime}")
    return selected_variant


def nullable_float(row: Mapping[str, Any], key: str) -> float | None:
    value = row[key]
    return None if value in (None, "") else float(value)


def find_row(
    rows: Sequence[Mapping[str, Any]], **conditions: str,
) -> Mapping[str, Any]:
    matches = [
        row for row in rows
        if all(str(row.get(key)) == str(value) for key, value in conditions.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {conditions}, found {len(matches)}")
    return matches[0]


def effect_status(ci_low: float, ci_high: float) -> str:
    if ci_high < 0.0:
        return "improves"
    if ci_low > 0.0:
        return "worsens"
    return "inconclusive"


def comparison_status(ci_low: float, ci_high: float) -> str:
    """Interpret comparator-minus-DART absolute-error confidence intervals."""
    if ci_low > 0.0:
        return "dart_better"
    if ci_high < 0.0:
        return "comparator_better"
    return "inconclusive"


def model_label(model: str) -> str:
    labels = {
        "dart": "DART", "schnet": "SchNet", "descriptor:mean": "Mean",
        "descriptor:lightgbm": "LightGBM",
        "descriptor:hist_gradient_boosting": "Histogram GB",
        "descriptor:random_forest": "Random forest",
        "descriptor:ridge": "Ridge",
    }
    return labels.get(model, model.replace("descriptor:", ""))


def finite_format(value: float, digits: int = 3, signed: bool = False) -> str:
    if not math.isfinite(float(value)):
        return "--"
    threshold = 0.5 * 10 ** (-digits)
    numeric = 0.0 if abs(float(value)) < threshold else float(value)
    pattern = f"{{:{'+' if signed else ''}.{digits}f}}"
    return pattern.format(numeric)


def percent_format(value: float, digits: int = 1) -> str:
    return finite_format(100.0 * float(value), digits)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def bootstrap_text(summary: Mapping[str, Any], digits: int = 3) -> str:
    return (
        f"{finite_format(float(summary['mean']), digits)} "
        f"[{finite_format(float(summary['ci_low']), digits)}, "
        f"{finite_format(float(summary['ci_high']), digits)}]"
    )


def input_paths(result_root: Path, protocol_dir: Path) -> Dict[str, Path]:
    return {
        "protocol": protocol_dir / "manifest.json",
        "factorial": result_root / "factorial/bundle.json",
        "comparison": result_root / "comparison/manifest.json",
        "comparison_run_metrics": result_root / "comparison/run_metrics.csv",
        "comparison_fold_metrics": result_root / "comparison/fold_metrics.csv",
        "comparison_summary": result_root / "comparison/summary.csv",
        "pooled": result_root / "comparison/pooled_metrics.csv",
        "paired": result_root / "comparison/paired_comparisons.csv",
        "comparison_descriptor_selection": (
            result_root / "comparison/descriptor_selection.json"
        ),
        "uq": result_root / "uq/metrics.json",
        "uq_intervals": result_root / "uq/interval_calibration.csv",
        "uq_risk": result_root / "uq/risk_coverage.csv",
        "uq_predictions": result_root / "uq/predictions.npz",
        "materials": result_root / "materials/summary.json",
        "sample_predictions": result_root / "materials/sample_predictions.csv",
        "pair_preferences": result_root / "materials/pair_preferences.csv",
        "site_selection": result_root / "materials/site_selection.csv",
        "group_errors": result_root / "materials/group_errors.csv",
    }


def load_inputs(paths: Mapping[str, Path]) -> Dict[str, Any]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("canonical paper inputs are incomplete:\n" + "\n".join(missing))
    inputs = {
        "protocol": json.loads(paths["protocol"].read_text()),
        "factorial": json.loads(paths["factorial"].read_text()),
        "comparison": json.loads(paths["comparison"].read_text()),
        "pooled": read_csv(paths["pooled"]),
        "paired": read_csv(paths["paired"]),
        "uq": json.loads(paths["uq"].read_text()),
        "uq_intervals": read_csv(paths["uq_intervals"]),
        "uq_risk": read_csv(paths["uq_risk"]),
        "materials": json.loads(paths["materials"].read_text()),
        "pair_preferences": read_csv(paths["pair_preferences"]),
        "site_selection": read_csv(paths["site_selection"]),
        "group_errors": read_csv(paths["group_errors"]),
    }
    for name in ("protocol", "factorial", "comparison", "uq", "materials"):
        strict_json(inputs[name])
    for name, expected in RECORDED_OUTPUTS.items():
        require_recorded_output_hashes(inputs[name], name, paths, expected)
    return inputs


def build_claims(inputs: Mapping[str, Any], selected_variant: str) -> Dict[str, Any]:
    factorial = inputs["factorial"]
    comparison = inputs["comparison"]
    pooled = inputs["pooled"]
    paired = inputs["paired"]
    uq = inputs["uq"]
    materials = inputs["materials"]

    effects: Dict[str, Any] = {}
    for split in ("validation", "test"):
        effects[split] = {}
        for term in TERM_ORDER:
            row = find_row(factorial["effects"], split=split, metric="mae", term=term)
            effects[split][term] = {
                "mean_eV": float(row["mean"]),
                "ci_low_eV": float(row["ci_low"]),
                "ci_high_eV": float(row["ci_high"]),
                "status": effect_status(float(row["ci_low"]), float(row["ci_high"])),
            }

    benchmarks: Dict[str, Any] = {}
    for regime in REGIME_ORDER:
        descriptor_model = (
            "descriptor:" + comparison["descriptor_selection"][regime]["selected_family"]
        )
        models = {}
        for model in ("dart", "schnet", descriptor_model, "descriptor:mean"):
            row = find_row(pooled, regime=regime, model=model)
            models[model] = {
                key: nullable_float(row, key)
                for key in (
                    "mae", "rmse", "bias", "spearman", "r2",
                    "host_macro_mae", "dopant_macro_mae", "favorable_mae",
                    "low_energy_mae", "low_energy_recall",
                )
            }
            models[model]["n"] = int(row["n"])
        comparisons = {}
        for comparator in ("schnet", descriptor_model):
            row = find_row(paired, regime=regime, comparator=comparator)
            low = float(row["ci_low_eV"])
            high = float(row["ci_high_eV"])
            comparisons[comparator] = {
                "delta_comparator_minus_dart_eV": float(
                    row["mae_difference_comparator_minus_dart_eV"]
                ),
                "ci_low_eV": low, "ci_high_eV": high,
                "status": comparison_status(low, high),
                "n": int(row["n"]),
            }
        benchmarks[regime] = {
            "descriptor_model": descriptor_model,
            "models": models,
            "paired_comparisons": comparisons,
        }

    return {
        "schema_version": "prm_paper_claims_v1",
        "selected_variant": selected_variant,
        "selection_data": "validation only",
        "factorial": {
            "validation_mae": factorial["selection"]["validation_mae"],
            "locked_test": factorial["selection"]["locked_test"],
            "effects": effects,
        },
        "benchmarks": benchmarks,
        "uq": uq["test"],
        "materials": {
            "sample_oof": materials["sample_oof"],
            "defect_type_preference": materials["defect_type_preference"],
            "within_defect_type_site_selection": materials["within_defect_type_site_selection"],
            "interpretation_boundary": materials["interpretation_boundary"],
        },
    }


def macro_line(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def render_macros(claims: Mapping[str, Any]) -> str:
    lines = [
        "% Auto-generated by scripts/prm_make_result_assets.py; do not edit.",
        macro_line("PRMSelectedVariant", rf"\texttt{{{claims['selected_variant']}}}"),
    ]
    factorial = claims["factorial"]
    validation = factorial["validation_mae"]
    locked_mae = factorial["locked_test"]["mae"]
    lines.extend(
        [
            macro_line("PRMSelectedValidationMAE", finite_format(validation["mean"])),
            macro_line("PRMSelectedValidationMAELow", finite_format(validation["ci_low"])),
            macro_line("PRMSelectedValidationMAEHigh", finite_format(validation["ci_high"])),
            macro_line("PRMSelectedTestMAE", finite_format(locked_mae["mean"])),
            macro_line("PRMSelectedTestMAELow", finite_format(locked_mae["ci_low"])),
            macro_line("PRMSelectedTestMAEHigh", finite_format(locked_mae["ci_high"])),
        ]
    )
    for regime in REGIME_ORDER:
        prefix = "PRM" + REGIME_MACROS[regime]
        benchmark = claims["benchmarks"][regime]
        descriptor = benchmark["descriptor_model"]
        lines.extend(
            [
                macro_line(prefix + "DARTMAE", finite_format(benchmark["models"]["dart"]["mae"])),
                macro_line(prefix + "SchNetMAE", finite_format(benchmark["models"]["schnet"]["mae"])),
                macro_line(prefix + "DescriptorMAE", finite_format(benchmark["models"][descriptor]["mae"])),
                macro_line(prefix + "DescriptorName", latex_escape(model_label(descriptor))),
                macro_line(prefix + "DARTLowEnergyMAE", finite_format(benchmark["models"]["dart"]["low_energy_mae"])),
                macro_line(prefix + "DARTLowEnergyRecall", percent_format(benchmark["models"]["dart"]["low_energy_recall"])),
            ]
        )

    uq = claims["uq"]
    lines.extend(
        [
            macro_line("PRMUQTestMAE", finite_format(uq["point_prediction"]["mae"])),
            macro_line("PRMUQNLL", finite_format(uq["gaussian_nll"])),
            macro_line("PRMUQCRPS", finite_format(uq["mean_gaussian_crps_eV"])),
            macro_line("PRMUQErrorSpearman", finite_format(uq["uncertainty_absolute_error_spearman"])),
            macro_line("PRMUQAURC", finite_format(uq["selective_prediction"]["aurc_eV"])),
            macro_line("PRMUQExcessAURC", finite_format(uq["selective_prediction"]["excess_aurc_eV"])),
        ]
    )
    for key, suffix in (("0.50", "Fifty"), ("0.80", "Eighty"), ("0.90", "Ninety"), ("0.95", "NinetyFive")):
        interval = uq["intervals"][key]
        lines.extend(
            [
                macro_line("PRMUQ" + suffix + "Coverage", percent_format(interval["observed_test_coverage"])),
                macro_line("PRMUQ" + suffix + "Width", finite_format(interval["mean_test_width_eV"])),
            ]
        )

    preference = claims["materials"]["defect_type_preference"]
    site = claims["materials"]["within_defect_type_site_selection"]
    lines.extend(
        [
            macro_line("PRMPreferenceAccuracy", percent_format(preference["accuracy"]["mean"])),
            macro_line("PRMPreferenceMarginMAE", finite_format(preference["margin_mae_eV"]["mean"])),
            macro_line("PRMPreferenceMarginSpearman", finite_format(preference["margin_spearman"])),
            macro_line("PRMGlobalRegret", finite_format(preference["global_screening_regret_eV"]["mean"])),
            macro_line("PRMGlobalExactAccuracy", percent_format(preference["global_exact_site_accuracy"]["mean"])),
            macro_line("PRMWithinTypeExactAccuracy", percent_format(site["exact_accuracy"]["mean"])),
            macro_line("PRMWithinTypeTopTwoAccuracy", percent_format(site["top2_accuracy"]["mean"])),
            macro_line("PRMWithinTypeRegret", finite_format(site["screening_regret_eV"]["mean"])),
        ]
    )
    return "\n".join(lines) + "\n"


def render_factorial_table(factorial: Mapping[str, Any]) -> str:
    selected = factorial["selection"]["selected_variant"]
    rows = []
    for item in sorted(factorial["variant_summary"], key=lambda row: row["variant"]):
        variant = item["variant"]
        bits = list(variant[1:])
        label = rf"\textbf{{\texttt{{{variant}}}}}" if variant == selected else rf"\texttt{{{variant}}}"
        validation = bootstrap_text(item["validation_mae"])
        test = bootstrap_text(item["test_mae"])
        if variant == selected:
            validation = rf"\textbf{{{validation}}}"
            test = rf"\textbf{{{test}}}"
        rows.append(f"{label} & {' & '.join(bits)} & {validation} & {test} \\\\")
    body = "\n".join(rows)
    return rf"""% Auto-generated; do not edit.
\begin{{table*}}[t]
\caption{{Paired $2^3$ architecture factorial. Values are mean MAE with
95\% bootstrap intervals over five paired repeats, in eV. G denotes gated
readout, E defect-environment enrichment, and P the composite pre-norm
distance-gated local block. Bold marks the validation-selected variant; test
MAE was not used in its selection.}}
\label{{tab:factorial}}
\centering
\begin{{tabular}}{{lccc cc}}
\toprule
Variant & G & E & P & Validation MAE & Locked test MAE \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def delta_cell(comparison: Mapping[str, Any]) -> str:
    return (
        f"{finite_format(comparison['delta_comparator_minus_dart_eV'], signed=True)} "
        f"[{finite_format(comparison['ci_low_eV'], signed=True)}, "
        f"{finite_format(comparison['ci_high_eV'], signed=True)}]"
    )


def render_benchmark_table(claims: Mapping[str, Any]) -> str:
    rows = []
    descriptor_names = set()
    for regime in REGIME_ORDER:
        benchmark = claims["benchmarks"][regime]
        descriptor = benchmark["descriptor_model"]
        descriptor_names.add(model_label(descriptor))
        models = benchmark["models"]
        rows.append(
            f"{REGIME_LABELS[regime]} & {finite_format(models['dart']['mae'])} & "
            f"{finite_format(models['schnet']['mae'])} & {finite_format(models[descriptor]['mae'])} & "
            f"{delta_cell(benchmark['paired_comparisons']['schnet'])} & "
            f"{delta_cell(benchmark['paired_comparisons'][descriptor])} \\\\")
    descriptor_note = ", ".join(sorted(descriptor_names))
    body = "\n".join(rows)
    return rf"""% Auto-generated; do not edit.
\begin{{table*}}[t]
\caption{{Pooled out-of-fold or single-block test MAE (eV). The descriptor
column uses the family selected separately within each regime by validation
MAE ({latex_escape(descriptor_note)} in the resulting selections). $\Delta$
is comparator minus DART absolute error with a paired 95\% bootstrap interval
using the regime-matched resampling unit; positive values favor DART.}}
\label{{tab:benchmark}}
\centering
\begin{{tabular}}{{lccc cc}}
\toprule
Regime & DART & SchNet & Descriptor & $\Delta_{{\rm SchNet}}$ & $\Delta_{{\rm desc.}}$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def render_applicability_table(claims: Mapping[str, Any]) -> str:
    rows = []
    for regime in REGIME_ORDER:
        dart = claims["benchmarks"][regime]["models"]["dart"]
        rows.append(
            f"{REGIME_LABELS[regime]} & {finite_format(dart['host_macro_mae'])} & "
            f"{finite_format(dart['dopant_macro_mae'])} & {finite_format(dart['favorable_mae'])} & "
            f"{finite_format(dart['low_energy_mae'])} & {percent_format(dart['low_energy_recall'])} \\\\")
    body = "\n".join(rows)
    return rf"""% Auto-generated; do not edit.
\begin{{table*}}[t]
\caption{{Applicability-domain diagnostics for DART. Macro MAEs weight each
host or impurity equally. Stable MAE uses $E_\mathrm{{f}}\leq0$; low-decile
MAE and recall use the true and predicted lowest 10\% of each pooled test
regime. Energy errors are in eV and recall is in percent.}}
\label{{tab:applicability}}
\centering
\begin{{tabular}}{{lccccc}}
\toprule
Regime & Host macro & Impurity macro & Stable MAE & Low-decile MAE & Low-decile recall \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def render_uq_table(uq: Mapping[str, Any]) -> str:
    rows = []
    for nominal in ("0.50", "0.80", "0.90", "0.95"):
        interval = uq["test"]["intervals"][nominal]
        rows.append(
            f"{percent_format(float(nominal), 0)} & "
            f"{percent_format(interval['observed_test_coverage'])} & "
            f"{finite_format(interval['mean_test_width_eV'])} & "
            f"{finite_format(interval['conformal_quantile'])} \\\\")
    body = "\n".join(rows)
    return rf"""% Auto-generated; do not edit.
\begin{{table}}[t]
\caption{{Held-out split-conformal interval performance on the frozen test
structures. Coverage is in percent and mean width is in eV.}}
\label{{tab:uq}}
\centering
\begin{{tabular}}{{cccc}}
\toprule
Nominal & Observed & Mean width & Quantile \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def render_screening_table(materials: Mapping[str, Any]) -> str:
    preference = materials["defect_type_preference"]
    site = materials["within_defect_type_site_selection"]
    entries = (
        (r"Incorporation-class preference accuracy (\%)", preference["accuracy"], True),
        (r"Global exact-site accuracy (\%)", preference["global_exact_site_accuracy"], True),
        ("Global screening regret (eV)", preference["global_screening_regret_eV"], False),
        (r"Within-class exact-site accuracy (\%)", site["exact_accuracy"], True),
        (r"Within-class top-2 accuracy (\%)", site["top2_accuracy"], True),
        ("Within-class screening regret (eV)", site["screening_regret_eV"], False),
    )
    rows = []
    for label, summary, as_percent in entries:
        if as_percent:
            value = percent_format(summary["mean"])
            low = percent_format(summary["ci_low"])
            high = percent_format(summary["ci_high"])
        else:
            value = finite_format(summary["mean"])
            low = finite_format(summary["ci_low"])
            high = finite_format(summary["ci_high"])
        rows.append(f"{label} & {value} & [{low}, {high}] & {int(summary['n'])} \\\\")
    body = "\n".join(rows)
    return rf"""% Auto-generated; do not edit.
\begin{{table*}}[t]
\caption{{Out-of-fold screening metrics from host--impurity-pair folds.
Intervals are 95\% nonparametric cluster-bootstrap intervals over
host--impurity pairs; multiple within-class decisions from one pair remain
together in every resample.}}
\label{{tab:screening}}
\centering
\begin{{tabular}}{{lccc}}
\toprule
Metric & Mean & 95\% interval & $n$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavuserif",
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12, 1.04, label, transform=axis.transAxes, ha="left", va="bottom",
        fontsize=8.5, fontweight="bold", color=COLORS["ink"],
    )


def errorbar_components(summary: Mapping[str, Any]) -> tuple[float, list[list[float]]]:
    mean = float(summary["mean"])
    return mean, [[mean - float(summary["ci_low"])], [float(summary["ci_high"]) - mean]]


def save_figure(figure: plt.Figure, output_pdf: Path) -> list[Path]:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")
    figure.savefig(output_pdf)
    figure.savefig(output_png)
    plt.close(figure)
    return [output_pdf, output_png]


def invalidate_ready_marker(generated_dir: Path) -> Path:
    ready_path = generated_dir / "results_ready.tex"
    ready_path.unlink(missing_ok=True)
    return ready_path


def write_ready_marker(ready_path: Path) -> None:
    ready_path.write_text(READY_MARKER_CONTENT)


def write_manifest_then_ready_marker(
    manifest: Mapping[str, Any], manifest_path: Path, ready_path: Path,
) -> None:
    marker_key = repository_path(ready_path)
    expected_hash = text_sha256(READY_MARKER_CONTENT)
    if manifest.get("outputs", {}).get(marker_key) != expected_hash:
        raise ValueError("result-assets manifest has an invalid ready-marker hash")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(strict_json(manifest) + "\n")
    try:
        write_ready_marker(ready_path)
        if file_sha256(ready_path) != expected_hash:
            raise ValueError("ready-marker hash mismatch after writing")
    except Exception:
        ready_path.unlink(missing_ok=True)
        raise


def plot_factorial(factorial: Mapping[str, Any], output_pdf: Path) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.75), gridspec_kw={"width_ratios": [1.05, 0.95]})
    variants = sorted(factorial["variant_summary"], key=lambda row: row["variant"])
    x = np.arange(len(variants))
    selected = factorial["selection"]["selected_variant"]
    selected_index = [row["variant"] for row in variants].index(selected)
    axes[0].axvspan(selected_index - 0.45, selected_index + 0.45, color="#E7F0F4", zorder=0)
    for offset, split, color, marker, label in (
        (-0.12, "validation_mae", COLORS["validation"], "o", "Validation"),
        (0.12, "test_mae", COLORS["test"], "s", "Locked test"),
    ):
        means = np.asarray([row[split]["mean"] for row in variants], dtype=float)
        lows = np.asarray([row[split]["ci_low"] for row in variants], dtype=float)
        highs = np.asarray([row[split]["ci_high"] for row in variants], dtype=float)
        axes[0].errorbar(
            x + offset, means, yerr=np.vstack([means - lows, highs - means]),
            fmt=marker, ms=4.0, lw=0.9, capsize=2.0, color=color, label=label, zorder=2,
        )
    axes[0].set_xticks(x, [row["variant"] for row in variants], rotation=45, ha="right")
    axes[0].set_ylabel("MAE (eV)")
    axes[0].set_title("Paired factorial variants", loc="left", pad=5)
    axes[0].legend(frameon=False, loc="best")
    axes[0].grid(axis="y", color=COLORS["grid"], lw=0.45)
    axes[0].spines[["top", "right"]].set_visible(False)
    panel_label(axes[0], "(a)")

    y = np.arange(len(TERM_ORDER))
    for offset, split, color, marker, label in (
        (-0.10, "validation", COLORS["validation"], "o", "Validation"),
        (0.10, "test", COLORS["test"], "s", "Locked test"),
    ):
        rows = [
            find_row(factorial["effects"], split=split, metric="mae", term=term)
            for term in TERM_ORDER
        ]
        means = np.asarray([row["mean"] for row in rows], dtype=float)
        lows = np.asarray([row["ci_low"] for row in rows], dtype=float)
        highs = np.asarray([row["ci_high"] for row in rows], dtype=float)
        axes[1].errorbar(
            means, y + offset, xerr=np.vstack([means - lows, highs - means]),
            fmt=marker, ms=4.0, lw=0.9, capsize=2.0, color=color, label=label,
        )
    axes[1].axvline(0.0, color=COLORS["ink"], lw=0.7)
    axes[1].set_yticks(y, TERM_ORDER)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"Factorial effect on MAE (eV)")
    axes[1].set_title("Orthogonal effects", loc="left", pad=5)
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(axis="x", color=COLORS["grid"], lw=0.45)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", length=0)
    panel_label(axes[1], "(b)")
    figure.tight_layout(w_pad=2.0)
    return save_figure(figure, output_pdf)


def plot_transfer(
    comparison: Mapping[str, Any], pooled_rows: Sequence[Mapping[str, str]],
    paired_rows: Sequence[Mapping[str, str]], output_pdf: Path,
) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), gridspec_kw={"width_ratios": [1.05, 0.95]})
    x = np.arange(len(REGIME_ORDER))
    model_specs = (
        ("dart", -0.24, COLORS["dart"], "o", "DART"),
        ("schnet", -0.08, COLORS["schnet"], "s", "SchNet"),
        ("descriptor", 0.08, COLORS["descriptor"], "D", "Selected descriptor"),
        ("descriptor:mean", 0.24, COLORS["mean"], "^", "Mean predictor"),
    )
    for model, offset, color, marker, label in model_specs:
        values = []
        for regime in REGIME_ORDER:
            actual = model
            if model == "descriptor":
                actual = "descriptor:" + comparison["descriptor_selection"][regime]["selected_family"]
            row = find_row(pooled_rows, regime=regime, model=actual)
            values.append(float(row["mae"]))
        axes[0].plot(x + offset, values, marker=marker, ls="none", ms=4.3, color=color, label=label)
    axes[0].set_xticks(x, [REGIME_LABELS[regime] for regime in REGIME_ORDER], rotation=25, ha="right")
    axes[0].set_ylabel("Pooled test MAE (eV)")
    axes[0].set_title("Interpolation and chemical transfer", loc="left", pad=5)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0].grid(axis="y", color=COLORS["grid"], lw=0.45)
    axes[0].spines[["top", "right"]].set_visible(False)
    panel_label(axes[0], "(a)")

    y = np.arange(len(REGIME_ORDER))
    for offset, model_kind, color, marker, label in (
        (-0.10, "schnet", COLORS["schnet"], "s", "SchNet - DART"),
        (0.10, "descriptor", COLORS["descriptor"], "D", "Descriptor - DART"),
    ):
        rows = []
        for regime in REGIME_ORDER:
            comparator = model_kind
            if model_kind == "descriptor":
                comparator = "descriptor:" + comparison["descriptor_selection"][regime]["selected_family"]
            rows.append(find_row(paired_rows, regime=regime, comparator=comparator))
        means = np.asarray([row["mae_difference_comparator_minus_dart_eV"] for row in rows], dtype=float)
        lows = np.asarray([row["ci_low_eV"] for row in rows], dtype=float)
        highs = np.asarray([row["ci_high_eV"] for row in rows], dtype=float)
        axes[1].errorbar(
            means, y + offset, xerr=np.vstack([means - lows, highs - means]),
            fmt=marker, ms=4.0, lw=0.9, capsize=2.0, color=color, label=label,
        )
    axes[1].axvline(0.0, color=COLORS["ink"], lw=0.7)
    axes[1].set_yticks(y, [REGIME_LABELS[regime] for regime in REGIME_ORDER])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"Paired $\Delta$MAE (comparator - DART, eV)")
    axes[1].set_title("Paired regime-matched contrasts", loc="left", pad=5)
    axes[1].legend(frameon=False, loc="best")
    axes[1].grid(axis="x", color=COLORS["grid"], lw=0.45)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", length=0)
    panel_label(axes[1], "(b)")
    figure.tight_layout(w_pad=2.0)
    return save_figure(figure, output_pdf)


def plot_uq(
    uq: Mapping[str, Any], interval_rows: Sequence[Mapping[str, str]],
    risk_rows: Sequence[Mapping[str, str]], predictions_path: Path,
    output_pdf: Path,
) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.25))
    nominal = np.asarray([float(row["nominal_coverage"]) for row in interval_rows])
    observed = np.asarray([float(row["observed_test_coverage"]) for row in interval_rows])
    widths = np.asarray([float(row["mean_test_width_eV"]) for row in interval_rows])
    axes[0].plot([0.45, 1.0], [0.45, 1.0], ls="--", lw=0.7, color=COLORS["muted"])
    axes[0].plot(nominal, observed, "o-", color=COLORS["validation"], ms=4.0)
    for x_value, y_value, width in zip(nominal, observed, widths):
        axes[0].annotate(f"{width:.2f} eV", (x_value, y_value), xytext=(3, -8), textcoords="offset points", fontsize=5.2)
    axes[0].set_xlim(0.45, 1.0)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Observed coverage")
    axes[0].set_title("Conformal calibration", loc="left", pad=5)
    axes[0].spines[["top", "right"]].set_visible(False)
    panel_label(axes[0], "(a)")

    coverage = np.asarray([float(row["coverage"]) for row in risk_rows])
    risk = np.asarray([float(row["mae_risk_eV"]) for row in risk_rows])
    oracle = np.asarray([float(row["oracle_mae_risk_eV"]) for row in risk_rows])
    axes[1].plot(coverage, risk, color=COLORS["dart"], label="Calibrated uncertainty")
    axes[1].plot(coverage, oracle, color=COLORS["oracle"], ls="--", label="Oracle ordering")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Retained coverage")
    axes[1].set_ylabel("MAE risk (eV)")
    axes[1].set_title("Selective prediction", loc="left", pad=5)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].spines[["top", "right"]].set_visible(False)
    panel_label(axes[1], "(b)")

    with np.load(predictions_path, allow_pickle=False) as archive:
        targets = np.asarray(archive["test_targets"], dtype=float)
        mean = np.asarray(archive["test_mean"], dtype=float)
        sigma = np.asarray(archive["test_sigma"], dtype=float)
    absolute_error = np.abs(mean - targets)
    image = axes[2].hexbin(
        sigma, absolute_error, gridsize=28, mincnt=1, bins="log",
        cmap="Greys", linewidths=0.0,
    )
    colorbar = figure.colorbar(image, ax=axes[2], fraction=0.05, pad=0.03)
    colorbar.set_label("log count", fontsize=6.2)
    colorbar.ax.tick_params(labelsize=5.5)
    rho = float(uq["test"]["uncertainty_absolute_error_spearman"])
    axes[2].text(0.04, 0.94, rf"$\rho_s={rho:.2f}$", transform=axes[2].transAxes, va="top")
    axes[2].set_xlabel(r"Calibrated $\sigma$ (eV)")
    axes[2].set_ylabel("Absolute error (eV)")
    axes[2].set_title("Error discrimination", loc="left", pad=5)
    axes[2].spines[["top", "right"]].set_visible(False)
    panel_label(axes[2], "(c)")
    figure.tight_layout(w_pad=1.7)
    return save_figure(figure, output_pdf)


def empirical_cdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    array = np.sort(np.asarray(values, dtype=float))
    return array, np.arange(1, len(array) + 1, dtype=float) / len(array)


def plot_screening(
    materials: Mapping[str, Any], pair_rows: Sequence[Mapping[str, str]],
    site_rows: Sequence[Mapping[str, str]], output_pdf: Path,
) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.35))
    true_margin = np.asarray([float(row["true_margin_eV"]) for row in pair_rows])
    predicted_margin = np.asarray([float(row["predicted_margin_eV"]) for row in pair_rows])
    limit = max(float(np.max(np.abs(true_margin))), float(np.max(np.abs(predicted_margin))))
    axes[0].scatter(true_margin, predicted_margin, s=5, alpha=0.18, color=COLORS["dart"], edgecolors="none")
    axes[0].plot([-limit, limit], [-limit, limit], ls="--", lw=0.7, color=COLORS["muted"])
    rho = float(materials["defect_type_preference"]["margin_spearman"])
    axes[0].text(0.04, 0.94, rf"$\rho_s={rho:.2f}$", transform=axes[0].transAxes, va="top")
    axes[0].set_xlabel("Reference preference margin (eV)")
    axes[0].set_ylabel("Predicted preference margin (eV)")
    axes[0].set_title("Incorporation-class preference", loc="left", pad=5)
    axes[0].spines[["top", "right"]].set_visible(False)
    panel_label(axes[0], "(a)")

    global_regret = [float(row["global_screening_regret_eV"]) for row in pair_rows]
    site_regret = [float(row["screening_regret_eV"]) for row in site_rows]
    for values, color, label in (
        (global_regret, COLORS["test"], "Global site"),
        (site_regret, COLORS["descriptor"], "Within class"),
    ):
        x, y = empirical_cdf(values)
        axes[1].step(x, y, where="post", color=color, label=label)
    axes[1].set_xlabel("Screening regret (eV)")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("Decision regret", loc="left", pad=5)
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].spines[["top", "right"]].set_visible(False)
    panel_label(axes[1], "(b)")

    preference = materials["defect_type_preference"]
    site = materials["within_defect_type_site_selection"]
    entries = (
        ("Class", preference["accuracy"]),
        ("Global site", preference["global_exact_site_accuracy"]),
        ("Within class", site["exact_accuracy"]),
        ("Top-2", site["top2_accuracy"]),
    )
    y = np.arange(len(entries))
    means = np.asarray([entry[1]["mean"] for entry in entries], dtype=float)
    lows = np.asarray([entry[1]["ci_low"] for entry in entries], dtype=float)
    highs = np.asarray([entry[1]["ci_high"] for entry in entries], dtype=float)
    axes[2].errorbar(
        means, y, xerr=np.vstack([means - lows, highs - means]),
        fmt="o", ms=4, lw=0.9, capsize=2, color=COLORS["dart"],
    )
    axes[2].set_yticks(y, [entry[0] for entry in entries])
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Accuracy")
    axes[2].set_title("Screening accuracy", loc="left", pad=5)
    axes[2].grid(axis="x", color=COLORS["grid"], lw=0.45)
    axes[2].spines[["top", "right", "left"]].set_visible(False)
    axes[2].tick_params(axis="y", length=0)
    panel_label(axes[2], "(c)")
    figure.tight_layout(w_pad=1.8)
    return save_figure(figure, output_pdf)


def write_outputs(
    inputs: Mapping[str, Any], paths: Mapping[str, Path], claims: Mapping[str, Any],
    paper_dir: Path, result_root: Path, collector_git: Mapping[str, Any],
) -> Dict[str, Any]:
    generated_dir = paper_dir / "generated"
    figure_dir = paper_dir / "figures"
    generated_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    ready_path = invalidate_ready_marker(generated_dir)

    output_paths = []
    claims_path = generated_dir / "claims.json"
    claims_path.write_text(strict_json(claims) + "\n")
    output_paths.append(claims_path)
    text_outputs = {
        "results_macros.tex": render_macros(claims),
        "tab_factorial.tex": render_factorial_table(inputs["factorial"]),
        "tab_benchmark.tex": render_benchmark_table(claims),
        "tab_applicability.tex": render_applicability_table(claims),
        "tab_uq.tex": render_uq_table(inputs["uq"]),
        "tab_screening.tex": render_screening_table(inputs["materials"]),
    }
    for name, content in text_outputs.items():
        path = generated_dir / name
        path.write_text(content)
        output_paths.append(path)

    output_paths.extend(plot_factorial(inputs["factorial"], figure_dir / "fig_factorial.pdf"))
    output_paths.extend(
        plot_transfer(
            inputs["comparison"], inputs["pooled"], inputs["paired"],
            figure_dir / "fig_transfer.pdf",
        )
    )
    output_paths.extend(
        plot_uq(
            inputs["uq"], inputs["uq_intervals"], inputs["uq_risk"],
            paths["uq_predictions"], figure_dir / "fig_uq.pdf",
        )
    )
    output_paths.extend(
        plot_screening(
            inputs["materials"], inputs["pair_preferences"], inputs["site_selection"],
            figure_dir / "fig_screening.pdf",
        )
    )
    output_hashes = {
        repository_path(path): file_sha256(path) for path in output_paths
    }
    output_hashes[repository_path(ready_path)] = text_sha256(READY_MARKER_CONTENT)

    manifest = {
        "schema_version": "prm_paper_assets_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector_git": collector_git,
        "data_sha256": inputs["protocol"]["data_sha256"],
        "selected_variant": claims["selected_variant"],
        "selection_data": "validation only",
        "inputs": {
            name: {"path": repository_path(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "outputs": output_hashes,
    }
    manifest_path = result_root / "paper/result_assets.json"
    write_manifest_then_ready_marker(manifest, manifest_path, ready_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", type=Path, default=ROOT / "artifacts/prm_results",
    )
    parser.add_argument(
        "--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v2",
    )
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper_Q1")
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    protocol_dir = args.protocol_dir.resolve()
    paper_dir = args.paper_dir.resolve()
    paths = input_paths(result_root, protocol_dir)
    inputs = load_inputs(paths)
    selected = validate_contract(
        inputs["protocol"], inputs["factorial"], inputs["comparison"],
        inputs["uq"], inputs["materials"], inputs["pooled"], inputs["paired"],
    )
    claims = build_claims(inputs, selected)
    strict_json(claims)
    collector_git = git_snapshot()
    if collector_git["dirty"]:
        raise ValueError("paper assets must be generated from a clean worktree")
    manifest = write_outputs(
        inputs, paths, claims, paper_dir, result_root, collector_git,
    )
    print(strict_json(manifest))


if __name__ == "__main__":
    main()
