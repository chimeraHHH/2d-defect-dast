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
DESCRIPTOR_SELECTED_MODEL = "descriptor:validation_selected"
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
TERM_LABELS = {
    "G": "G", "E": "E", "P": "P",
    "G:E": "G\u00d7E", "G:P": "G\u00d7P", "E:P": "E\u00d7P", "G:E:P": "G\u00d7E\u00d7P",
}
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
    "schnet_readout": {
        "summary": "schnet_readout_summary",
        "run_metrics": "schnet_readout_run_metrics",
        "fold_metrics": "schnet_readout_fold_metrics",
        "host_metrics": "schnet_readout_host_metrics",
        "predictions": "schnet_readout_predictions",
    },
}
READY_MARKER_CONTENT = (
    "% Auto-generated after all canonical paper assets succeeded; do not edit.\n"
    "\\def\\PRMResultAssetsReady{1}\n"
)


def variant_display_label(variant: str) -> str:
    if not isinstance(variant, str) or not variant.startswith("g") or len(variant) != 4:
        raise ValueError(f"invalid factorial variant {variant!r}")
    bits = variant[1:]
    if any(bit not in {"0", "1"} for bit in bits):
        raise ValueError(f"invalid factorial variant {variant!r}")
    enabled = [name for name, bit in zip(("G", "E", "P"), bits) if bit == "1"]
    return "+".join(enabled) if enabled else "Base"


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


def validate_training_archive(
    payload: Mapping[str, Any],
    payload_path: Path,
    *,
    expected_count: int,
    expected_artifacts: set[str],
) -> None:
    records = payload.get("archived_runs")
    if (
        int(payload.get("n_archived_runs", -1)) != expected_count
        or not isinstance(records, list)
        or len(records) != expected_count
    ):
        raise ValueError(
            f"training archive count mismatch for {payload_path}: "
            f"expected {expected_count}"
        )

    base = payload_path.resolve().parent
    output_dirs = []
    config_hashes = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid training archive record in {payload_path}")
        output_dir = record.get("output_dir")
        config_sha256 = record.get("config_sha256")
        git = record.get("git")
        artifacts = record.get("artifacts")
        omitted = record.get("omitted_outputs")
        checkpoint = (
            omitted.get("checkpoint") if isinstance(omitted, Mapping) else None
        )
        if (
            not isinstance(output_dir, str)
            or not output_dir
            or Path(output_dir).is_absolute()
            or ".." in Path(output_dir).parts
            or not isinstance(config_sha256, str)
            or len(config_sha256) != 64
            or any(character not in "0123456789abcdef" for character in config_sha256)
            or not isinstance(git, Mapping)
            or not git.get("commit")
            or git.get("dirty") is not False
            or not isinstance(artifacts, Mapping)
            or set(artifacts) != expected_artifacts
            or not isinstance(checkpoint, Mapping)
            or not isinstance(checkpoint.get("sha256"), str)
            or len(checkpoint["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in checkpoint["sha256"]
            )
        ):
            raise ValueError(f"incomplete training archive record in {payload_path}")
        output_dirs.append(output_dir)
        config_hashes.append(config_sha256)

        for name, artifact in artifacts.items():
            if not isinstance(artifact, Mapping):
                raise ValueError(
                    f"invalid archived {name} record in {payload_path}"
                )
            relative = Path(str(artifact.get("archive_relative_path", "")))
            expected_sha256 = artifact.get("sha256")
            path = (base / relative).resolve()
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or not path.is_relative_to(base)
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_sha256
                )
                or not path.is_file()
                or file_sha256(path) != expected_sha256
            ):
                raise ValueError(
                    f"archived {name} hash mismatch for {path}"
                )
    if (
        len(set(output_dirs)) != expected_count
        or len(set(config_hashes)) != expected_count
    ):
        raise ValueError(f"training archive records are not unique in {payload_path}")


def validate_contract(
    protocol: Mapping[str, Any], factorial: Mapping[str, Any],
    comparison: Mapping[str, Any], uq: Mapping[str, Any],
    materials: Mapping[str, Any], schnet_readout: Mapping[str, Any],
    schnet_readout_summary: Mapping[str, Any],
    pooled_rows: Sequence[Mapping[str, str]],
    paired_rows: Sequence[Mapping[str, str]],
    schnet_readout_fold_rows: Sequence[Mapping[str, str]],
) -> str:
    require_schema(factorial, "prm_factorial_bundle_v1", "factorial bundle")
    require_schema(comparison, "prm_comparison_bundle_v1", "comparison bundle")
    require_schema(uq, "prm_uq_results_v1", "UQ bundle")
    require_schema(materials, "prm_materials_analysis_v1", "materials bundle")
    require_schema(
        schnet_readout,
        "prm_schnet_readout_sensitivity_bundle_v1",
        "SchNet readout sensitivity bundle",
    )
    for name, payload in (
        ("factorial bundle", factorial), ("comparison bundle", comparison),
        ("UQ bundle", uq), ("materials bundle", materials),
        ("SchNet readout sensitivity bundle", schnet_readout),
    ):
        require_clean_collector(payload, name)

    data_sha256 = str(protocol["data_sha256"])
    observed_hashes = {
        str(factorial.get("data_sha256")), str(comparison.get("data_sha256")),
        str(uq.get("data_sha256")), str(materials.get("data_sha256")),
        str(schnet_readout.get("data_sha256")),
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
    archive_counts = (
        ("factorial", factorial, 40),
        ("comparison", comparison, 91),
        ("UQ", uq, 5),
    )
    for name, payload, expected_count in archive_counts:
        if (
            int(payload.get("n_archived_runs", -1)) != expected_count
            or not isinstance(payload.get("archived_runs"), list)
            or len(payload["archived_runs"]) != expected_count
        ):
            raise ValueError(
                f"{name} bundle lacks all {expected_count} archived runs"
            )
    configuration_coverage = comparison.get("configuration_coverage")
    if not isinstance(configuration_coverage, Mapping):
        raise ValueError("comparison bundle lacks controlled configuration coverage")
    for model in ("dart", "schnet"):
        coverage = configuration_coverage.get(model)
        if not isinstance(coverage, Mapping):
            raise ValueError(f"comparison bundle lacks {model} configuration coverage")
        hashes = coverage.get("expected_config_sha256")
        if (
            int(coverage.get("n_expected", -1)) != 48
            or int(coverage.get("n_observed", -1)) != 48
            or coverage.get("complete") is not True
            or coverage.get("missing_output_dirs") != []
            or not isinstance(hashes, list)
            or len(hashes) != 48
            or len(set(hashes)) != 48
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
        ):
            raise ValueError(
                f"comparison bundle has incomplete {model} configuration coverage"
            )
    if int(uq.get("n_members", -1)) != 5:
        raise ValueError("UQ bundle must contain five ensemble members")
    if (
        schnet_readout.get("n_runs")
        != {"add_reference": 15, "mean_sensitivity": 15}
        or int(schnet_readout.get("n_archived_runs", -1)) != 15
        or not isinstance(schnet_readout.get("archived_runs"), list)
        or len(schnet_readout["archived_runs"]) != 15
    ):
        raise ValueError("SchNet readout sensitivity lacks all 15 paired runs")
    if (
        schnet_readout_summary.get("analysis_role")
        != "post_hoc_exploratory_robustness"
        or schnet_readout_summary.get("intervention")
        != {"model_kwargs.readout": {"from": "add", "to": "mean"}}
    ):
        raise ValueError("SchNet readout sensitivity role or intervention is invalid")
    fold_ids = {
        str(row.get("split_id")) for row in schnet_readout_fold_rows
    }
    directional = schnet_readout_summary.get("fold_directional_consistency", {})
    if (
        len(schnet_readout_fold_rows) != 5
        or fold_ids != {f"host_cv5_f{fold}" for fold in range(5)}
        or int(directional.get("n_folds", -1)) != 5
        or int(directional.get("mean_better_folds", -1)) not in range(6)
    ):
        raise ValueError("SchNet readout sensitivity fold coverage is incomplete")
    uq_sources = uq.get("member_sources")
    if (
        not isinstance(uq_sources, list)
        or len(uq_sources) != 5
        or any(not isinstance(source, Mapping) for source in uq_sources)
        or len({source.get("seed") for source in uq_sources}) != 5
        or len({source.get("config_sha256") for source in uq_sources}) != 5
        or {source.get("split_id") for source in uq_sources}
        != {"uq_calibration_s62"}
    ):
        raise ValueError("UQ bundle member evidence is incomplete")
    materials_sources = materials.get("sources")
    if (
        not isinstance(materials_sources, list)
        or len(materials_sources) != 5
        or any(not isinstance(source, Mapping) for source in materials_sources)
        or len({source.get("config_sha256") for source in materials_sources}) != 5
        or {source.get("split_id") for source in materials_sources}
        != {f"pair_cv5_f{fold}" for fold in range(5)}
    ):
        raise ValueError("materials bundle pair-OOF evidence is incomplete")
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
    if (
        sum(int(row.get("n", -1)) for row in schnet_readout_fold_rows)
        != expected_modeling
        or int(
            schnet_readout_summary.get(
                "paired_host_cluster_bootstrap", {}
            ).get("n", -1)
        )
        != expected_modeling
    ):
        raise ValueError(
            "SchNet readout sensitivity does not cover the canonical set"
        )

    pooled_keys = {(row["regime"], row["model"]) for row in pooled_rows}
    descriptor_selection = comparison.get("descriptor_selection", {})
    for regime in REGIME_ORDER:
        selection = descriptor_selection[regime]
        expected_descriptor_splits = (
            {"chemistry_block_g6x3d"}
            if regime == "chemistry_block"
            else {f"{regime}5_f{fold}" for fold in range(5)}
        )
        if (
            selection.get("selection_unit") != "split"
            or selection.get("selected_model") != DESCRIPTOR_SELECTED_MODEL
            or set(selection.get("split_selections", {}))
            != expected_descriptor_splits
            or sum(
                int(count)
                for count in selection.get("family_counts", {}).values()
            )
            != len(expected_descriptor_splits)
        ):
            raise ValueError(
                f"descriptor family selection is not foldwise for {regime}"
            )
        descriptor = DESCRIPTOR_SELECTED_MODEL
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


def factorial_effect_claim(
    row: Mapping[str, Any], split: str, term: str,
) -> Dict[str, Any]:
    repeat_effects = np.asarray(row["repeat_effects"], dtype=float)
    n_paired_repeats = int(row["n_paired_repeats"])
    mean = float(row["mean"])
    if (
        repeat_effects.ndim != 1
        or len(repeat_effects) != n_paired_repeats
        or n_paired_repeats < 2
        or not np.isfinite(repeat_effects).all()
        or not math.isfinite(mean)
    ):
        raise ValueError(
            f"factorial repeat effects are incomplete for {split} {term}"
        )
    ci_low = float(row["ci_low"])
    ci_high = float(row["ci_high"])
    return {
        "mean_eV": mean,
        "ci_low_eV": ci_low,
        "ci_high_eV": ci_high,
        "status": effect_status(ci_low, ci_high),
        "n_paired_repeats": n_paired_repeats,
        "direction_agreeing_repeats": int(
            np.sum(np.sign(repeat_effects) == np.sign(mean))
        ),
    }


def model_label(model: str) -> str:
    labels = {
        "dart": "DART", "schnet": "SchNet-add", "descriptor:mean": "Mean",
        "descriptor:lightgbm": "LightGBM",
        "descriptor:hist_gradient_boosting": "Histogram GB",
        "descriptor:random_forest": "Random forest",
        "descriptor:ridge": "Ridge",
        DESCRIPTOR_SELECTED_MODEL: "Validation-selected descriptor",
    }
    return labels.get(model, model.replace("descriptor:", ""))


def finite_format(
    value: float | None, digits: int = 3, signed: bool = False,
) -> str:
    if value is None or not math.isfinite(float(value)):
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
        "schnet_readout": (
            result_root / "sensitivity/schnet_readout/manifest.json"
        ),
        "schnet_readout_summary": (
            result_root / "sensitivity/schnet_readout/summary.json"
        ),
        "schnet_readout_run_metrics": (
            result_root / "sensitivity/schnet_readout/run_metrics.csv"
        ),
        "schnet_readout_fold_metrics": (
            result_root / "sensitivity/schnet_readout/fold_metrics.csv"
        ),
        "schnet_readout_host_metrics": (
            result_root / "sensitivity/schnet_readout/host_metrics.csv"
        ),
        "schnet_readout_predictions": (
            result_root / "sensitivity/schnet_readout/predictions.npz"
        ),
        "schnet_readout_config_manifest": (
            ROOT / "configs/prm/sensitivity/schnet_mean_host/manifest.json"
        ),
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
        "schnet_readout": json.loads(paths["schnet_readout"].read_text()),
        "schnet_readout_summary": json.loads(
            paths["schnet_readout_summary"].read_text()
        ),
        "schnet_readout_fold_metrics": read_csv(
            paths["schnet_readout_fold_metrics"]
        ),
        "schnet_readout_config_manifest": json.loads(
            paths["schnet_readout_config_manifest"].read_text()
        ),
    }
    for name in (
        "protocol", "factorial", "comparison", "uq", "materials",
        "schnet_readout", "schnet_readout_summary",
        "schnet_readout_config_manifest",
    ):
        strict_json(inputs[name])
    for name, expected in RECORDED_OUTPUTS.items():
        require_recorded_output_hashes(inputs[name], name, paths, expected)
    standard_artifacts = {
        "manifest", "metrics", "split_indices", "validation_predictions",
        "test_predictions",
    }
    validate_training_archive(
        inputs["factorial"],
        paths["factorial"],
        expected_count=40,
        expected_artifacts=standard_artifacts,
    )
    validate_training_archive(
        inputs["comparison"],
        paths["comparison"],
        expected_count=91,
        expected_artifacts=standard_artifacts,
    )
    validate_training_archive(
        inputs["uq"],
        paths["uq"],
        expected_count=5,
        expected_artifacts=standard_artifacts | {"calibration_predictions"},
    )
    validate_training_archive(
        inputs["schnet_readout"],
        paths["schnet_readout"],
        expected_count=15,
        expected_artifacts=standard_artifacts,
    )
    if (
        inputs["schnet_readout"].get("config_manifest", {}).get("sha256")
        != file_sha256(paths["schnet_readout_config_manifest"])
    ):
        raise ValueError("SchNet readout configuration manifest hash mismatch")
    if (
        inputs["schnet_readout"].get("parent_comparison_manifest", {}).get("sha256")
        != file_sha256(paths["comparison"])
    ):
        raise ValueError("SchNet readout parent comparison manifest hash mismatch")
    return inputs


def build_claims(inputs: Mapping[str, Any], selected_variant: str) -> Dict[str, Any]:
    factorial = inputs["factorial"]
    comparison = inputs["comparison"]
    pooled = inputs["pooled"]
    paired = inputs["paired"]
    uq = inputs["uq"]
    materials = inputs["materials"]
    schnet_readout = inputs["schnet_readout_summary"]

    effects: Dict[str, Any] = {}
    for split in ("validation", "test"):
        effects[split] = {}
        for term in TERM_ORDER:
            row = find_row(factorial["effects"], split=split, metric="mae", term=term)
            effects[split][term] = factorial_effect_claim(row, split, term)

    benchmarks: Dict[str, Any] = {}
    for regime in REGIME_ORDER:
        descriptor_selection = comparison["descriptor_selection"][regime]
        descriptor_model = str(descriptor_selection["selected_model"])
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
            "descriptor_family_counts": descriptor_selection["family_counts"],
            "models": models,
            "paired_comparisons": comparisons,
        }

    return {
        "schema_version": "prm_paper_claims_v2",
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
            "error_heterogeneity": materials["error_heterogeneity"],
            "interpretation_boundary": materials["interpretation_boundary"],
        },
        "schnet_readout_sensitivity": {
            "analysis_role": schnet_readout["analysis_role"],
            "intervention": schnet_readout["intervention"],
            "add": schnet_readout["add"],
            "mean": schnet_readout["mean"],
            "paired_host_cluster_bootstrap": (
                schnet_readout["paired_host_cluster_bootstrap"]
            ),
            "fold_directional_consistency": (
                schnet_readout["fold_directional_consistency"]
            ),
            "sample_natoms_vs_absolute_error_spearman": (
                schnet_readout["sample_natoms_vs_absolute_error_spearman"]
            ),
            "host_median_natoms_vs_mae_spearman": (
                schnet_readout["host_median_natoms_vs_mae_spearman"]
            ),
            "fold_metrics": inputs["schnet_readout_fold_metrics"],
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
            macro_line(
                "PRMPreferenceReferenceAccuracy",
                percent_format(
                    preference["accuracy_reference"]["accuracy"]["mean"]
                ),
            ),
            macro_line(
                "PRMPreferenceGain",
                percent_format(
                    preference["accuracy_reference"][
                        "model_minus_reference"
                    ]["mean"]
                ),
            ),
            macro_line(
                "PRMPreferenceGainLow",
                percent_format(
                    preference["accuracy_reference"][
                        "model_minus_reference"
                    ]["ci_low"]
                ),
            ),
            macro_line(
                "PRMPreferenceGainHigh",
                percent_format(
                    preference["accuracy_reference"][
                        "model_minus_reference"
                    ]["ci_high"]
                ),
            ),
            macro_line(
                "PRMGlobalReferenceAccuracy",
                percent_format(
                    preference["global_exact_site_reference"]["accuracy"][
                        "mean"
                    ]
                ),
            ),
            macro_line(
                "PRMGlobalAccuracyGain",
                percent_format(
                    preference["global_exact_site_reference"][
                        "model_minus_reference"
                    ]["mean"]
                ),
            ),
            macro_line(
                "PRMGlobalAccuracyGainLow",
                percent_format(
                    preference["global_exact_site_reference"][
                        "model_minus_reference"
                    ]["ci_low"]
                ),
            ),
            macro_line(
                "PRMGlobalAccuracyGainHigh",
                percent_format(
                    preference["global_exact_site_reference"][
                        "model_minus_reference"
                    ]["ci_high"]
                ),
            ),
            macro_line(
                "PRMWithinTypeReferenceExactAccuracy",
                percent_format(
                    site["exact_accuracy_reference"]["accuracy"]["mean"]
                ),
            ),
            macro_line(
                "PRMWithinTypeExactAccuracyGain",
                percent_format(
                    site["exact_accuracy_reference"][
                        "model_minus_reference"
                    ]["mean"]
                ),
            ),
            macro_line(
                "PRMWithinTypeExactAccuracyGainLow",
                percent_format(
                    site["exact_accuracy_reference"][
                        "model_minus_reference"
                    ]["ci_low"]
                ),
            ),
            macro_line(
                "PRMWithinTypeExactAccuracyGainHigh",
                percent_format(
                    site["exact_accuracy_reference"][
                        "model_minus_reference"
                    ]["ci_high"]
                ),
            ),
            macro_line(
                "PRMWithinTypeReferenceTopTwoAccuracy",
                percent_format(
                    site["top2_accuracy_reference"]["accuracy"]["mean"]
                ),
            ),
            macro_line(
                "PRMWithinTypeTopTwoAccuracyGain",
                percent_format(
                    site["top2_accuracy_reference"][
                        "model_minus_reference"
                    ]["mean"]
                ),
            ),
            macro_line(
                "PRMWithinTypeTopTwoAccuracyGainLow",
                percent_format(
                    site["top2_accuracy_reference"][
                        "model_minus_reference"
                    ]["ci_low"]
                ),
            ),
            macro_line(
                "PRMWithinTypeTopTwoAccuracyGainHigh",
                percent_format(
                    site["top2_accuracy_reference"][
                        "model_minus_reference"
                    ]["ci_high"]
                ),
            ),
        ]
    )
    sensitivity = claims["schnet_readout_sensitivity"]
    paired_readout = sensitivity["paired_host_cluster_bootstrap"]
    lines.extend(
        [
            macro_line(
                "PRMSchNetAddHostMAE",
                finite_format(sensitivity["add"]["mae"]),
            ),
            macro_line(
                "PRMSchNetMeanHostMAE",
                finite_format(sensitivity["mean"]["mae"]),
            ),
            macro_line(
                "PRMSchNetMeanMinusAddHostMAE",
                finite_format(
                    paired_readout["mae_difference_mean_minus_add_eV"],
                    signed=True,
                ),
            ),
            macro_line(
                "PRMSchNetMeanMinusAddHostMAELow",
                finite_format(paired_readout["ci_low_eV"], signed=True),
            ),
            macro_line(
                "PRMSchNetMeanMinusAddHostMAEHigh",
                finite_format(paired_readout["ci_high_eV"], signed=True),
            ),
            macro_line(
                "PRMSchNetMeanBetterFolds",
                str(
                    sensitivity["fold_directional_consistency"][
                        "mean_better_folds"
                    ]
                ),
            ),
            macro_line(
                "PRMSchNetAddSampleSizeErrorSpearman",
                finite_format(
                    sensitivity[
                        "sample_natoms_vs_absolute_error_spearman"
                    ]["add"]
                ),
            ),
            macro_line(
                "PRMSchNetMeanSampleSizeErrorSpearman",
                finite_format(
                    sensitivity[
                        "sample_natoms_vs_absolute_error_spearman"
                    ]["mean"]
                ),
            ),
            macro_line(
                "PRMSchNetAddHostSizeErrorSpearman",
                finite_format(
                    sensitivity[
                        "host_median_natoms_vs_mae_spearman"
                    ]["add"]
                ),
            ),
            macro_line(
                "PRMSchNetMeanHostSizeErrorSpearman",
                finite_format(
                    sensitivity[
                        "host_median_natoms_vs_mae_spearman"
                    ]["mean"]
                ),
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def render_factorial_narrative(claims: Mapping[str, Any]) -> str:
    effects = claims["factorial"]["effects"]["validation"]
    status_text = {
        "improves": "supported reduction",
        "worsens": "supported increase",
        "inconclusive": "inconclusive",
    }

    def entry(term: str) -> str:
        effect = effects[term]
        digits = 4 if (
            effect["status"] != "inconclusive"
            and min(abs(effect["ci_low_eV"]), abs(effect["ci_high_eV"])) < 0.0005
        ) else 3
        return (
            rf"${term}$: $\Delta={finite_format(effect['mean_eV'], digits, signed=True)}$ "
            rf"[{finite_format(effect['ci_low_eV'], digits, signed=True)}, "
            rf"{finite_format(effect['ci_high_eV'], digits, signed=True)}]~eV "
            rf"({status_text[effect['status']]})"
        )

    main_effects = "; ".join(entry(term) for term in ("G", "E", "P"))
    directional_interactions = [
        term for term in TERM_ORDER[3:]
        if effects[term]["status"] != "inconclusive"
    ]
    if directional_interactions:
        interaction_text = (
            "Directional validation intervals were also observed for "
            + "; ".join(entry(term) for term in directional_interactions)
            + "."
        )
        if len(directional_interactions) < len(TERM_ORDER[3:]):
            interaction_text += (
                " All other prespecified interaction intervals included zero."
            )
    else:
        interaction_text = (
            "No prespecified interaction contrast had a 95\\% validation "
            "interval excluding zero."
        )
    directional_terms = [
        term for term in TERM_ORDER
        if effects[term]["status"] != "inconclusive"
    ]
    if directional_terms:
        consistency_text = (
            " The repeat-level contrast agreed with the pooled point-estimate "
            "direction in "
            + ", ".join(
                rf"${term}$ "
                f"{effects[term]['direction_agreeing_repeats']}/"
                f"{effects[term]['n_paired_repeats']}"
                for term in directional_terms
            )
            + " paired repeats."
        )
    else:
        consistency_text = ""
    return (
        "% Auto-generated from canonical factorial claims; do not edit.\n"
        "On validation MAE, the enabled-minus-disabled main-effect contrasts "
        f"were {main_effects}. {interaction_text}{consistency_text} "
        "With five paired repeats, the percentile intervals and sign counts "
        "are descriptive uncertainty summaries rather than large-sample "
        "hypothesis tests.\n"
    )


def render_transfer_narrative(claims: Mapping[str, Any]) -> str:
    status_text = {
        "dart_better": "supports lower DART error",
        "comparator_better": "supports lower comparator error",
        "inconclusive": "inconclusive",
    }
    sentences = []
    for regime in REGIME_ORDER:
        benchmark = claims["benchmarks"][regime]
        descriptor = benchmark["descriptor_model"]
        entries = []
        for comparator, label in (
            ("schnet", model_label("schnet")),
            (descriptor, model_label(descriptor)),
        ):
            comparison = benchmark["paired_comparisons"][comparator]
            entries.append(
                rf"{latex_escape(label)} $\Delta="
                rf"{finite_format(comparison['delta_comparator_minus_dart_eV'], signed=True)}$ "
                rf"[{finite_format(comparison['ci_low_eV'], signed=True)}, "
                rf"{finite_format(comparison['ci_high_eV'], signed=True)}]~eV "
                rf"({status_text[comparison['status']]})"
            )
        sentences.append(
            f"For {REGIME_LABELS[regime]}, the paired comparator-minus-DART "
            "absolute-error differences were " + " and ".join(entries) + "."
        )
    return (
        "% Auto-generated from canonical paired comparisons; do not edit.\n"
        + " ".join(sentences)
        + "\n"
    )


def render_schnet_readout_narrative(claims: Mapping[str, Any]) -> str:
    sensitivity = claims["schnet_readout_sensitivity"]
    paired = sensitivity["paired_host_cluster_bootstrap"]
    status = effect_status(float(paired["ci_low_eV"]), float(paired["ci_high_eV"]))
    status_text = {
        "improves": "supports lower error for mean pooling",
        "worsens": "supports higher error for mean pooling",
        "inconclusive": "is inconclusive",
    }
    directional = sensitivity["fold_directional_consistency"]
    return (
        "% Auto-generated from the controlled SchNet readout sensitivity; "
        "do not edit.\n"
        "In a post-hoc host-CV sensitivity that changed only SchNet's graph "
        "readout, replacing atomwise addition by mean pooling changed pooled "
        rf"MAE from {finite_format(sensitivity['add']['mae'])} to "
        rf"{finite_format(sensitivity['mean']['mae'])}~eV. "
        "The paired host-cluster mean-minus-add absolute-error difference was "
        rf"$\Delta={finite_format(paired['mae_difference_mean_minus_add_eV'], signed=True)}$ "
        rf"[{finite_format(paired['ci_low_eV'], signed=True)}, "
        rf"{finite_format(paired['ci_high_eV'], signed=True)}]~eV "
        f"({status_text[status]}), and mean pooling had lower fold MAE in "
        f"{int(directional['mean_better_folds'])}/"
        f"{int(directional['n_folds'])} folds. "
        "This bounded intervention tests sensitivity to graph readout; it does "
        "not replace the prespecified additive-SchNet comparator or decompose "
        "the remaining end-to-end DART--SchNet difference.\n"
    )


def render_error_heterogeneity_narrative(claims: Mapping[str, Any]) -> str:
    heterogeneity = claims["materials"]["error_heterogeneity"]
    host = heterogeneity["host"]
    dopant = heterogeneity["dopant"]
    worst_host = host["worst_groups"][0]
    worst_dopant = dopant["worst_groups"][0]
    return (
        "% Auto-generated from pair-held-out group errors; do not edit.\n"
        "Across host and impurity groups, the Spearman associations between "
        "group sample count and group MAE were "
        rf"{finite_format(host['sample_count_mae_spearman'])} and "
        rf"{finite_format(dopant['sample_count_mae_spearman'])}, respectively. "
        f"The largest group MAEs occurred for host "
        rf"{latex_escape(worst_host['group'])} "
        rf"({finite_format(worst_host['mae_eV'])}~eV; "
        rf"$n={int(worst_host['n'])}$) and impurity "
        rf"{latex_escape(worst_dopant['group'])} "
        rf"({finite_format(worst_dopant['mae_eV'])}~eV; "
        rf"$n={int(worst_dopant['n'])}$). "
        "Because group size covaries with chemistry, these are descriptive "
        "associations rather than evidence that sample count alone causes the "
        "observed errors.\n"
    )


def render_schnet_readout_table(claims: Mapping[str, Any]) -> str:
    sensitivity = claims["schnet_readout_sensitivity"]
    rows = []
    for row in sorted(
        sensitivity["fold_metrics"], key=lambda item: item["split_id"]
    ):
        fold = int(str(row["split_id"]).rsplit("f", 1)[1]) + 1
        rows.append(
            f"Fold {fold} & {int(row['n'])} & "
            f"{finite_format(float(row['add_mae_eV']))} & "
            f"{finite_format(float(row['mean_mae_eV']))} & "
            f"{finite_format(float(row['mean_minus_add_mae_eV']), signed=True)} "
            r"\\"
        )
    paired = sensitivity["paired_host_cluster_bootstrap"]
    rows.append(
        r"\textbf{Pooled OOF} & "
        f"{int(paired['n'])} & "
        rf"\textbf{{{finite_format(sensitivity['add']['mae'])}}} & "
        rf"\textbf{{{finite_format(sensitivity['mean']['mae'])}}} & "
        rf"\textbf{{{finite_format(paired['mae_difference_mean_minus_add_eV'], signed=True)}}} "
        r"\\"
    )
    body = "\n".join(rows)
    interval = (
        f"[{finite_format(paired['ci_low_eV'], signed=True)}, "
        f"{finite_format(paired['ci_high_eV'], signed=True)}]"
    )
    return rf"""% Auto-generated; do not edit.
\begin{{table}}[htbp]
\caption{{Post-hoc SchNet graph-readout sensitivity under the fixed
host-held-out protocol. Each fold prediction averages seeds 342--344. The
difference is mean-readout MAE minus additive-readout MAE, in eV. The pooled
host-cluster bootstrap 95\% interval for this difference is {interval}~eV.
The prespecified additive-readout result remains the main comparator.}}
\label{{tab:schnet-readout}}
\centering
\begin{{tabular}}{{lrrrr}}
\toprule
Partition & $n$ & Add MAE & Mean MAE & Mean $-$ add \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


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
    descriptor_families = set()
    for regime in REGIME_ORDER:
        benchmark = claims["benchmarks"][regime]
        descriptor = benchmark["descriptor_model"]
        descriptor_families.update(benchmark["descriptor_family_counts"])
        models = benchmark["models"]
        rows.append(
            f"{REGIME_LABELS[regime]} & {finite_format(models['dart']['mae'])} & "
            f"{finite_format(models['schnet']['mae'])} & {finite_format(models[descriptor]['mae'])} & "
            f"{delta_cell(benchmark['paired_comparisons']['schnet'])} & "
            f"{delta_cell(benchmark['paired_comparisons'][descriptor])} \\\\")
    descriptor_note = ", ".join(
        model_label(f"descriptor:{family}")
        for family in sorted(descriptor_families)
    )
    body = "\n".join(rows)
    return rf"""% Auto-generated; do not edit.
\begin{{table*}}[t]
\caption{{Pooled out-of-fold or single-block test MAE (eV). SchNet-add denotes
the prespecified periodic SchNet recipe with atomwise-additive readout. The descriptor
column selects a learned family independently within every split by that
split's validation MAE ({latex_escape(descriptor_note)} were selected across
the resulting folds). $\Delta$
is comparator minus DART absolute error with a paired 95\% bootstrap interval
using the regime-matched resampling unit; positive values favor DART.}}
\label{{tab:benchmark}}
\centering
\begin{{tabular}}{{lccc cc}}
\toprule
Regime & DART & SchNet-add & Descriptor & $\Delta_{{\rm SchNet-add}}$ & $\Delta_{{\rm desc.}}$ \\
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
host or impurity equally. Nonpositive-target MAE uses
$E_\mathrm{{f}}\leq0$; low-decile MAE and recall use the true and predicted
lowest 10\% of each pooled test regime. Energy errors are in eV and recall is
in percent. A dash denotes a regime with no eligible nonpositive test
targets.}}
\label{{tab:applicability}}
\centering
\begin{{tabular}}{{lccccc}}
\toprule
Regime & Host macro & Impurity macro & Nonpositive-target MAE & Low-decile MAE & Low-decile recall \\
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
\caption{{Internal split-conformal interval performance on the dedicated test
partition. These structures were excluded from ensemble fitting, checkpoint
selection, and calibration, but not from the earlier architecture-development
corpus. Coverage is in percent and mean width is in eV.}}
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
        (
            r"Incorporation-class preference accuracy (\%)",
            preference["accuracy"],
            preference["accuracy_reference"],
            True,
        ),
        (
            r"Global exact-site accuracy (\%)",
            preference["global_exact_site_accuracy"],
            preference["global_exact_site_reference"],
            True,
        ),
        (
            "Global screening regret (eV)",
            preference["global_screening_regret_eV"],
            None,
            False,
        ),
        (
            r"Within-class exact-site accuracy (\%)",
            site["exact_accuracy"],
            site["exact_accuracy_reference"],
            True,
        ),
        (
            r"Within-class top-2 accuracy (\%)",
            site["top2_accuracy"],
            site["top2_accuracy_reference"],
            True,
        ),
        (
            "Within-class screening regret (eV)",
            site["screening_regret_eV"],
            None,
            False,
        ),
    )
    rows = []
    for label, summary, reference, as_percent in entries:
        if as_percent:
            value = percent_format(summary["mean"])
            low = percent_format(summary["ci_low"])
            high = percent_format(summary["ci_high"])
            reference_value = percent_format(reference["accuracy"]["mean"])
            gain = reference["model_minus_reference"]
            gain_text = (
                f"{percent_format(gain['mean'])} "
                f"[{percent_format(gain['ci_low'])}, "
                f"{percent_format(gain['ci_high'])}]"
            )
        else:
            value = finite_format(summary["mean"])
            low = finite_format(summary["ci_low"])
            high = finite_format(summary["ci_high"])
            reference_value = "--"
            gain_text = "--"
        rows.append(
            f"{label} & {value} [{low}, {high}] & {reference_value} & "
            f"{gain_text} & {int(summary['n'])} \\\\"
        )
    body = "\n".join(rows)
    return rf"""% Auto-generated; do not edit.
\begin{{table*}}[t]
\caption{{Out-of-fold screening metrics from host--impurity-pair folds.
Intervals are 95\% nonparametric cluster-bootstrap intervals over
host--impurity pairs; multiple within-class decisions from one pair remain
together in every resample. The incorporation-class reference always predicts
the empirical majority class; site references select uniformly over the
observed candidate set, with ties retained. Accuracy gains are model minus
reference in percentage points.}}
\label{{tab:screening}}
\centering
\begin{{tabular}}{{lcccc}}
\toprule
Metric & Model [95\% interval] & Reference & Gain [95\% interval] & $n$ \\
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
    axes[0].set_xticks(
        x,
        [variant_display_label(str(row["variant"])) for row in variants],
        rotation=38,
        ha="right",
    )
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
    axes[1].set_yticks(y, [TERM_LABELS[term] for term in TERM_ORDER])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"Factorial effect on MAE (eV)")
    axes[1].set_title("Orthogonal effects", loc="left", pad=5)
    axes[1].legend(frameon=False, loc="upper right")
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
    figure = plt.figure(figsize=(7.05, 2.72))
    outer_grid = figure.add_gridspec(
        1, 2, width_ratios=(1.02, 1.68), wspace=0.31,
        left=0.105, right=0.985, bottom=0.235, top=0.84,
    )
    delta_grid = outer_grid[0, 1].subgridspec(
        1, 2, width_ratios=(1.20, 0.43), wspace=0.08,
    )
    axis_profile = figure.add_subplot(outer_grid[0, 0])
    axis_delta = figure.add_subplot(delta_grid[0, 0])
    axis_delta_far = figure.add_subplot(delta_grid[0, 1], sharey=axis_delta)

    y = np.arange(len(REGIME_ORDER))
    dart_values = np.asarray([
        float(find_row(pooled_rows, regime=regime, model="dart")["mae"])
        for regime in REGIME_ORDER
    ])
    axis_profile.hlines(
        y, 0.0, dart_values, color=COLORS["grid"], lw=1.0, zorder=1,
    )
    axis_profile.scatter(
        dart_values, y, s=24, marker="o", color=COLORS["dart"],
        edgecolors="white", linewidths=0.5, zorder=2,
    )
    for y_value, mae in zip(y, dart_values):
        label_left = mae < dart_values[0] - 0.01
        axis_profile.text(
            mae - 0.025 if label_left else mae + 0.025,
            y_value,
            f"{mae:.3f}",
            ha="right" if label_left else "left",
            va="center", fontsize=5.5, color=COLORS["ink"],
        )
    axis_profile.axvline(
        dart_values[0], color=COLORS["dart"], lw=0.7, ls=(0, (3, 2)), zorder=0,
    )
    axis_profile.set_yticks(y, [REGIME_LABELS[regime] for regime in REGIME_ORDER])
    axis_profile.invert_yaxis()
    axis_profile.set_xlim(0.0, 1.16)
    axis_profile.set_xlabel("DART pooled test MAE (eV)")
    axis_profile.set_title("DART error across transfer regimes", loc="left", pad=5)
    axis_profile.grid(axis="x", color=COLORS["grid"], lw=0.45, zorder=0)
    axis_profile.spines[["top", "right", "left"]].set_visible(False)
    axis_profile.tick_params(axis="y", length=0)
    panel_label(axis_profile, "(a)")

    contrast_specs = (
        (-0.10, "schnet", COLORS["schnet"], "s", "SchNet-add - DART"),
        (0.10, "descriptor", COLORS["descriptor"], "D", "Descriptor - DART"),
    )
    host_schnet = None
    for offset, model_kind, color, marker, label in contrast_specs:
        rows = []
        for regime in REGIME_ORDER:
            comparator = (
                DESCRIPTOR_SELECTED_MODEL if model_kind == "descriptor" else model_kind
            )
            rows.append(find_row(paired_rows, regime=regime, comparator=comparator))
        means = np.asarray(
            [row["mae_difference_comparator_minus_dart_eV"] for row in rows],
            dtype=float,
        )
        lows = np.asarray([row["ci_low_eV"] for row in rows], dtype=float)
        highs = np.asarray([row["ci_high_eV"] for row in rows], dtype=float)
        local = means < 2.0
        axis_delta.errorbar(
            means[local], (y + offset)[local],
            xerr=np.vstack([means[local] - lows[local], highs[local] - means[local]]),
            fmt=marker, ms=4.2, lw=0.9, capsize=2.0, color=color, label=label,
            zorder=2,
        )
        far = ~local
        if np.any(far):
            axis_delta_far.errorbar(
                means[far], (y + offset)[far],
                xerr=np.vstack([means[far] - lows[far], highs[far] - means[far]]),
                fmt=marker, ms=4.2, lw=0.9, capsize=2.0, color=color, zorder=2,
            )
            far_index = int(np.flatnonzero(far)[0])
            host_schnet = (means[far_index], lows[far_index], highs[far_index], y[far_index] + offset)

    axis_delta.axvline(0.0, color=COLORS["ink"], lw=0.7)
    axis_delta.set_xlim(-0.05, 1.18)
    axis_delta_far.set_xlim(2.75, 9.65)
    axis_delta.set_xticks([0.0, 0.4, 0.8, 1.2])
    axis_delta_far.set_xticks([3, 6, 9])
    axis_delta.set_yticks(y, [REGIME_LABELS[regime] for regime in REGIME_ORDER])
    axis_delta.invert_yaxis()
    axis_delta.set_title("Paired comparator-minus-DART contrasts", loc="left", pad=5)
    axis_delta.legend(frameon=False, loc="lower right")
    for axis in (axis_delta, axis_delta_far):
        axis.grid(axis="x", color=COLORS["grid"], lw=0.45, zorder=0)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axis_delta_far.tick_params(axis="y", left=False, labelleft=False)
    axis_delta_far.spines["left"].set_visible(False)
    if host_schnet is not None:
        mean, low, high, y_value = host_schnet
        axis_delta_far.text(
            mean, y_value - 0.24, f"{mean:.2f} [{low:.2f}, {high:.2f}]",
            ha="center", va="bottom", fontsize=5.0, color=COLORS["schnet"],
        )

    break_size = 0.035
    axis_delta.plot(
        (1 - break_size, 1 + break_size), (-break_size, +break_size),
        transform=axis_delta.transAxes, color=COLORS["ink"], clip_on=False, lw=0.8,
    )
    axis_delta_far.plot(
        (-break_size, +break_size), (-break_size, +break_size),
        transform=axis_delta_far.transAxes, color=COLORS["ink"], clip_on=False, lw=0.8,
    )
    panel_label(axis_delta, "(b)")
    figure.text(
        0.745, 0.055, r"Paired $\Delta$MAE (comparator - DART, eV)",
        ha="center", va="center", fontsize=7.5,
    )
    return save_figure(figure, output_pdf)


def plot_error_heterogeneity(
    materials: Mapping[str, Any], group_rows: Sequence[Mapping[str, str]],
    output_pdf: Path,
) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.75))
    heterogeneity = materials["error_heterogeneity"]
    for panel, axis_name, marker, color, label in (
        (0, "host", "o", COLORS["dart"], "Host"),
        (1, "dopant", "s", COLORS["descriptor"], "Impurity"),
    ):
        rows = [row for row in group_rows if row["axis"] == axis_name]
        counts = np.asarray([int(row["n"]) for row in rows], dtype=float)
        errors = np.asarray([float(row["mae_eV"]) for row in rows], dtype=float)
        axes[panel].scatter(
            counts, errors, marker=marker, s=16, alpha=0.72, color=color,
            edgecolors="white", linewidths=0.3,
        )
        for row in sorted(rows, key=lambda item: -float(item["mae_eV"]))[:3]:
            axes[panel].annotate(
                str(row["group"]),
                (int(row["n"]), float(row["mae_eV"])),
                xytext=(3, 2), textcoords="offset points", fontsize=5.2,
            )
        rho = float(heterogeneity[axis_name]["sample_count_mae_spearman"])
        axes[panel].text(
            0.04, 0.95, rf"$\rho_s={rho:.2f}$",
            transform=axes[panel].transAxes, va="top",
        )
        axes[panel].set_xscale("log")
        axes[panel].set_xlabel(f"{label} sample count")
        axes[panel].set_ylabel("Group MAE (eV)")
        axes[panel].set_title(f"{label}-level error", loc="left", pad=5)
        axes[panel].grid(color=COLORS["grid"], lw=0.4)
        axes[panel].spines[["top", "right"]].set_visible(False)
        panel_label(axes[panel], f"({chr(ord('a') + panel)})")

    categories = [
        row for row in group_rows if row["axis"] in {"defecttype", "site"}
    ]
    categories.sort(
        key=lambda row: (
            0 if row["axis"] == "defecttype" else 1,
            str(row["group"]),
        )
    )
    positions = np.arange(len(categories))
    values = np.asarray([float(row["mae_eV"]) for row in categories])
    category_colors = [
        COLORS["test"] if row["axis"] == "defecttype" else COLORS["muted"]
        for row in categories
    ]
    axes[2].scatter(values, positions, s=18, c=category_colors, zorder=2)
    axes[2].hlines(
        positions, 0.0, values, color=COLORS["grid"], lw=0.7, zorder=1,
    )
    axes[2].set_yticks(
        positions,
        [
            ("class: " if row["axis"] == "defecttype" else "site: ")
            + str(row["group"])
            for row in categories
        ],
    )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("MAE (eV)")
    axes[2].set_title("Incorporation and site labels", loc="left", pad=5)
    axes[2].grid(axis="x", color=COLORS["grid"], lw=0.4)
    axes[2].spines[["top", "right", "left"]].set_visible(False)
    axes[2].tick_params(axis="y", length=0, labelsize=5.2)
    panel_label(axes[2], "(c)")
    figure.tight_layout(w_pad=1.6)
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
    displayed_error = np.clip(absolute_error, 1e-3, None)
    image = axes[2].hexbin(
        sigma, displayed_error, gridsize=28, mincnt=1, bins="log",
        xscale="linear", yscale="log", cmap="Greys", linewidths=0.0,
    )
    colorbar = figure.colorbar(image, ax=axes[2], fraction=0.05, pad=0.03)
    colorbar.set_label("log count", fontsize=6.2)
    colorbar.ax.tick_params(labelsize=5.5)
    rho = float(uq["test"]["uncertainty_absolute_error_spearman"])
    axes[2].text(0.04, 0.94, rf"$\rho_s={rho:.2f}$", transform=axes[2].transAxes, va="top")
    axes[2].set_xlabel(r"Calibrated $\sigma$ (eV)")
    axes[2].set_ylabel("Absolute error (eV; log scale)")
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
    for values, color, linestyle, label in (
        (global_regret, COLORS["test"], "-", "Global site"),
        (site_regret, COLORS["descriptor"], "--", "Within class"),
    ):
        x, y = empirical_cdf(values)
        axes[1].step(
            x, y, where="post", color=color, linestyle=linestyle,
            linewidth=1.1, label=label,
        )
    axes[1].set_xlabel("Screening regret (eV)")
    axes[1].set_xscale("symlog", linthresh=0.05, linscale=0.8)
    axes[1].set_xticks([0.0, 0.1, 1.0, 10.0], ["0", "0.1", "1", "10"])
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
        "results_factorial_narrative.tex": render_factorial_narrative(claims),
        "results_transfer_narrative.tex": render_transfer_narrative(claims),
        "results_schnet_readout_narrative.tex": (
            render_schnet_readout_narrative(claims)
        ),
        "results_error_heterogeneity_narrative.tex": (
            render_error_heterogeneity_narrative(claims)
        ),
        "tab_factorial.tex": render_factorial_table(inputs["factorial"]),
        "tab_benchmark.tex": render_benchmark_table(claims),
        "tab_applicability.tex": render_applicability_table(claims),
        "tab_uq.tex": render_uq_table(inputs["uq"]),
        "tab_screening.tex": render_screening_table(inputs["materials"]),
        "tab_schnet_readout.tex": render_schnet_readout_table(claims),
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
        plot_error_heterogeneity(
            inputs["materials"], inputs["group_errors"],
            figure_dir / "fig_error_heterogeneity.pdf",
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
        "schema_version": "prm_paper_assets_v2",
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
    collector_git = git_snapshot()
    if collector_git["dirty"]:
        raise ValueError("paper assets must be generated from a clean worktree")
    invalidate_ready_marker(paper_dir / "generated")
    paths = input_paths(result_root, protocol_dir)
    inputs = load_inputs(paths)
    selected = validate_contract(
        inputs["protocol"], inputs["factorial"], inputs["comparison"],
        inputs["uq"], inputs["materials"], inputs["schnet_readout"],
        inputs["schnet_readout_summary"], inputs["pooled"], inputs["paired"],
        inputs["schnet_readout_fold_metrics"],
    )
    claims = build_claims(inputs, selected)
    strict_json(claims)
    manifest = write_outputs(
        inputs, paths, claims, paper_dir, result_root, collector_git,
    )
    print(strict_json(manifest))


if __name__ == "__main__":
    main()
