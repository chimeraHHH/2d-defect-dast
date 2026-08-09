"""Independently verify every paper-facing numerical result and asset hash."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent
DESCRIPTOR_SELECTED_MODEL = "descriptor:validation_selected"
REGIMES = ("id_cv", "pair_cv", "host_cv", "dopant_cv", "chemistry_block")
REGIME_PREFIX = {
    "id_cv": "IdCv",
    "pair_cv": "PairCv",
    "host_cv": "HostCv",
    "dopant_cv": "DopantCv",
    "chemistry_block": "ChemistryBlock",
}
REGIME_LABEL = {
    "id_cv": "Random OOF",
    "pair_cv": "Pair OOF",
    "host_cv": "Host OOF",
    "dopant_cv": "Impurity OOF",
    "chemistry_block": "Chemistry block",
}
DISPLAY_FAMILY = {
    "hist_gradient_boosting": "Histogram GB",
    "lightgbm": "LightGBM",
    "mean": "Mean",
    "random_forest": "Random forest",
    "ridge": "Ridge",
    DESCRIPTOR_SELECTED_MODEL: "Validation-selected descriptor",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_json_numbers(value: Any) -> Any:
    """Remove platform-specific floating-point tails from audit output."""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("audit report contains a nonfinite number")
        canonical = float(f"{number:.15g}")
        return 0.0 if canonical == 0.0 else canonical
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {
            key: canonicalize_json_numbers(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [canonicalize_json_numbers(item) for item in value]
    return value


def finite_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    value = float(spearmanr(np.asarray(left), np.asarray(right)).statistic)
    if not np.isfinite(value):
        raise ValueError("Spearman correlation is not finite")
    return value


class Audit:
    def __init__(self) -> None:
        self.n_checks = 0
        self.failures: list[dict[str, Any]] = []

    def exact(self, name: str, observed: Any, expected: Any) -> None:
        self.n_checks += 1
        if observed != expected:
            self.failures.append(
                {"check": name, "observed": observed, "expected": expected}
            )

    def close(
        self,
        name: str,
        observed: float,
        expected: float,
        *,
        atol: float = 1.0e-10,
    ) -> None:
        self.n_checks += 1
        if not math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=atol
        ):
            self.failures.append(
                {
                    "check": name,
                    "observed": float(observed),
                    "expected": float(expected),
                    "atol": atol,
                }
            )

    def contains(self, name: str, text: str, fragment: str) -> None:
        self.n_checks += 1
        if fragment not in text:
            self.failures.append(
                {"check": name, "missing_fragment": fragment}
            )


def regime_for_split(split_id: str) -> str:
    for regime in ("id_cv", "pair_cv", "host_cv", "dopant_cv"):
        if split_id.startswith(f"{regime}5_f"):
            return regime
    if split_id == "chemistry_block_g6x3d":
        return "chemistry_block"
    if split_id.startswith("id_repeat_s"):
        return "id_repeat"
    if split_id == "uq_calibration_s62":
        return "uq"
    if split_id == "id_historical_s42":
        return "historical"
    if split_id == "smoke_protocol":
        return "smoke"
    raise ValueError(f"unrecognized split ID: {split_id}")


def load_prediction_file(
    path: Path, *, descriptor_family: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if descriptor_family is None:
            indices = np.asarray(archive["indices"], dtype=np.int64)
            targets = np.asarray(archive["targets"], dtype=float)
            predictions = np.asarray(archive["preds"], dtype=float)
        else:
            names = [str(name) for name in archive["model_names"].tolist()]
            position = names.index(descriptor_family)
            indices = np.asarray(archive["test_indices"], dtype=np.int64)
            targets = np.asarray(archive["test_targets"], dtype=float)
            predictions = np.asarray(
                archive["test_predictions"][position], dtype=float
            )
    order = np.argsort(indices)
    return indices[order], targets[order], predictions[order]


def average_prediction_files(
    paths: Sequence[Path], *, descriptor_family: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("prediction group is empty")
    members = [
        load_prediction_file(path, descriptor_family=descriptor_family)
        for path in sorted(paths)
    ]
    reference_indices, reference_targets = members[0][:2]
    for indices, targets, _ in members[1:]:
        if not np.array_equal(indices, reference_indices) or not np.allclose(
            targets, reference_targets, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError(f"unaligned prediction members under {paths[0].parent}")
    return (
        reference_indices,
        reference_targets,
        np.mean([member[2] for member in members], axis=0),
    )


def concatenate_folds(
    members: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    materialized = list(members)
    indices = np.concatenate([item[0] for item in materialized])
    targets = np.concatenate([item[1] for item in materialized])
    predictions = np.concatenate([item[2] for item in materialized])
    if len(np.unique(indices)) != len(indices):
        raise ValueError("pooled prediction indices are duplicated")
    order = np.argsort(indices)
    return indices[order], targets[order], predictions[order]


def load_samples(path: Path) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in read_csv(path):
        retained = row["canonical_retained"].strip().lower() == "true"
        output[int(row["sample_index"])] = {
            "target_eV": float(row["target_eV"]),
            "host": row["host"],
            "dopant": row["dopant"],
            "natoms": int(row["natoms"]),
            "canonical_retained": retained,
        }
    return output


def load_comparison_predictions(
    root: Path,
    descriptor_selection: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    comparison_root = root / "artifacts/prm_results/comparison/runs"
    grouped: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for manifest_path in sorted(comparison_root.glob("**/run_manifest.json")):
        relative = manifest_path.relative_to(comparison_root).as_posix()
        if relative.startswith("baselines/schnet/"):
            model = "schnet"
        elif relative.startswith("selected/g111/transfer/"):
            model = "dart"
        else:
            continue
        split_id = str(read_json(manifest_path)["split"]["split_id"])
        regime = regime_for_split(split_id)
        if regime in REGIMES:
            grouped[(model, regime, split_id)].append(
                manifest_path.parent / "test_predictions.npz"
            )

    pooled: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model in ("dart", "schnet"):
        for regime in REGIMES:
            split_ids = sorted(
                key[2]
                for key in grouped
                if key[0] == model and key[1] == regime
            )
            pooled[(regime, model)] = concatenate_folds(
                average_prediction_files(grouped[(model, regime, split_id)])
                for split_id in split_ids
            )

    descriptor_root = root / "artifacts/prm_results/descriptors/runs"
    for regime in REGIMES:
        split_paths = [
            prediction_path
            for prediction_path in sorted(
                descriptor_root.glob("*/predictions.npz")
            )
            if regime_for_split(prediction_path.parent.name) == regime
        ]
        split_selections = descriptor_selection[regime]["split_selections"]
        pooled[(regime, DESCRIPTOR_SELECTED_MODEL)] = concatenate_folds(
            [
                load_prediction_file(
                    path,
                    descriptor_family=str(
                        split_selections[path.parent.name]["selected_family"]
                    ),
                )
                for path in split_paths
            ]
        )
        pooled[(regime, "descriptor:mean")] = concatenate_folds(
            [
                load_prediction_file(path, descriptor_family="mean")
                for path in split_paths
            ]
        )
    return pooled


def group_mae(
    targets: np.ndarray, predictions: np.ndarray, labels: Sequence[str]
) -> tuple[float, float, int]:
    label_array = np.asarray(labels)
    values = [
        float(np.mean(np.abs(predictions[label_array == label] - targets[label_array == label])))
        for label in sorted(set(labels))
    ]
    return float(np.mean(values)), float(np.max(values)), len(values)


def comparison_metrics(
    indices: np.ndarray,
    predictions: np.ndarray,
    samples: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    targets = np.asarray([samples[int(index)]["target_eV"] for index in indices])
    hosts = [str(samples[int(index)]["host"]) for index in indices]
    dopants = [str(samples[int(index)]["dopant"]) for index in indices]
    residual = predictions - targets
    target_ss = np.sum((targets - targets.mean()) ** 2)
    output: dict[str, Any] = {
        "n": len(indices),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "spearman": (
            None
            if len(np.unique(predictions)) < 2
            else finite_spearman(targets, predictions)
        ),
        "r2": float(1.0 - np.sum(residual**2) / target_ss),
    }
    favorable = targets <= 0.0
    output["favorable_n"] = int(favorable.sum())
    output["favorable_mae"] = (
        float(np.mean(np.abs(residual[favorable]))) if favorable.any() else None
    )
    k = max(1, int(round(len(targets) * 0.1)))
    true_low = np.argsort(targets)[:k]
    predicted_low = np.argsort(predictions)[:k]
    output.update(
        {
            "low_energy_k": k,
            "low_energy_mae": float(np.mean(np.abs(residual[true_low]))),
            "low_energy_recall": float(
                len(set(true_low).intersection(predicted_low)) / k
            ),
            "predicted_low_energy_mean_target_eV": float(
                np.mean(targets[predicted_low])
            ),
        }
    )
    host_macro, host_worst, n_hosts = group_mae(
        targets, predictions, hosts
    )
    dopant_macro, dopant_worst, n_dopants = group_mae(
        targets, predictions, dopants
    )
    output.update(
        {
            "host_macro_mae": host_macro,
            "host_worst_group_mae": host_worst,
            "n_hosts": n_hosts,
            "dopant_macro_mae": dopant_macro,
            "dopant_worst_group_mae": dopant_worst,
            "n_dopants": n_dopants,
        }
    )
    return output


def paired_bootstrap(
    targets: np.ndarray,
    reference: np.ndarray,
    comparator: np.ndarray,
    *,
    seed: int,
    draws: int,
    groups: Sequence[str] | None,
) -> dict[str, float]:
    differences = np.abs(comparator - targets) - np.abs(reference - targets)
    if groups is None:
        totals = differences
        counts = np.ones(len(differences), dtype=np.int64)
    else:
        _, inverse = np.unique(np.asarray(groups), return_inverse=True)
        totals = np.bincount(inverse, weights=differences)
        counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        draw = rng.integers(0, len(totals), size=(size, len(totals)))
        chunks.append(totals[draw].sum(axis=1) / counts[draw].sum(axis=1))
    values = np.concatenate(chunks)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "mean": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_units": len(totals),
    }


def bootstrap_mean(
    values: Sequence[float], *, samples: int, seed: int
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def cluster_bootstrap_rows(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    seed: int,
    draws: int = 20_000,
) -> dict[str, float | int]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["host"]), str(row["dopant"]))].append(
            float(row[value_key])
        )
    totals = np.asarray([sum(values) for values in grouped.values()])
    counts = np.asarray([len(values) for values in grouped.values()])
    values = np.asarray([value for values in grouped.values() for value in values])
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        draw = rng.integers(0, len(grouped), size=(size, len(grouped)))
        chunks.append(totals[draw].sum(axis=1) / counts[draw].sum(axis=1))
    means = np.concatenate(chunks)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": len(values),
        "n_clusters": len(grouped),
    }


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "1.0", "true"}


def parse_macros(path: Path) -> dict[str, str]:
    pattern = re.compile(
        r"^\\newcommand\{\\(?P<name>PRM[A-Za-z]+)\}\{(?P<value>.*)\}$"
    )
    output = {}
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            output[match.group("name")] = match.group("value")
    return output


def signed(value: float) -> str:
    return f"{value:+.3f}"


def audit_asset_hashes(root: Path, audit: Audit) -> dict[str, int]:
    manifest = read_json(root / "artifacts/prm_results/paper/result_assets.json")
    for name, record in manifest["inputs"].items():
        path = root / record["path"]
        audit.exact(f"asset input exists: {name}", path.exists(), True)
        if path.exists():
            audit.exact(
                f"asset input hash: {name}",
                file_sha256(path),
                record["sha256"],
            )
    for relative, expected in manifest["outputs"].items():
        path = root / relative
        audit.exact(f"asset output exists: {relative}", path.exists(), True)
        if path.exists():
            audit.exact(
                f"asset output hash: {relative}", file_sha256(path), expected
            )
    audit.exact("paper asset schema", manifest["schema_version"], "prm_paper_assets_v2")
    audit.exact("paper output count", len(manifest["outputs"]), 23)
    return {
        "inputs": len(manifest["inputs"]),
        "outputs": len(manifest["outputs"]),
    }


def audit_comparison(
    root: Path,
    audit: Audit,
    samples: Mapping[int, Mapping[str, Any]],
    descriptor_selection: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, float]],
]:
    predictions = load_comparison_predictions(root, descriptor_selection)
    direct: dict[tuple[str, str], dict[str, Any]] = {}
    for (regime, model), (indices, stored_targets, values) in predictions.items():
        canonical_targets = np.asarray(
            [samples[int(index)]["target_eV"] for index in indices]
        )
        audit.exact(
            f"canonical sample retention: {regime}/{model}",
            all(samples[int(index)]["canonical_retained"] for index in indices),
            True,
        )
        audit.exact(
            f"canonical index uniqueness: {regime}/{model}",
            len(np.unique(indices)),
            len(indices),
        )
        audit.exact(
            f"stored target alignment: {regime}/{model}",
            bool(
                np.allclose(
                    stored_targets, canonical_targets, rtol=0.0, atol=1.0e-6
                )
            ),
            True,
        )
        direct[(regime, model)] = comparison_metrics(indices, values, samples)

    pooled_rows = read_csv(
        root / "artifacts/prm_results/comparison/pooled_metrics.csv"
    )
    integer_fields = {
        "n", "favorable_n", "low_energy_k", "n_hosts", "n_dopants"
    }
    for row in pooled_rows:
        key = (row["regime"], row["model"])
        expected = direct[key]
        for field, value in expected.items():
            if (
                key[1] == "descriptor:mean"
                and field
                in {"low_energy_recall", "predicted_low_energy_mean_target_eV"}
            ):
                # The foldwise mean predictor has tied scores. Its arbitrary
                # within-tie ranking is not paper-facing and can differ across
                # NumPy sorting implementations.
                continue
            observed_raw = row[field]
            name = f"pooled {key[0]}/{key[1]}/{field}"
            if value is None:
                audit.exact(name, observed_raw, "")
            elif field in integer_fields:
                audit.exact(name, int(observed_raw), int(value))
            else:
                audit.close(name, float(observed_raw), float(value))

    paired_rows = read_csv(
        root / "artifacts/prm_results/comparison/paired_comparisons.csv"
    )
    paired_direct: dict[tuple[str, str], dict[str, float]] = {}
    for row in paired_rows:
        regime = row["regime"]
        comparator = row["comparator"]
        descriptor = DESCRIPTOR_SELECTED_MODEL
        comparator_order = ["schnet", descriptor, "descriptor:mean"]
        comparator_index = comparator_order.index(comparator)
        regime_index = REGIMES.index(regime)
        indices, _, dart_prediction = predictions[(regime, "dart")]
        comparator_indices, _, comparator_prediction = predictions[
            (regime, comparator)
        ]
        audit.exact(
            f"paired index alignment: {regime}/{comparator}",
            bool(np.array_equal(indices, comparator_indices)),
            True,
        )
        targets = np.asarray(
            [samples[int(index)]["target_eV"] for index in indices]
        )
        hosts = [str(samples[int(index)]["host"]) for index in indices]
        dopants = [str(samples[int(index)]["dopant"]) for index in indices]
        if regime == "host_cv":
            groups = hosts
        elif regime == "dopant_cv":
            groups = dopants
        elif regime in ("pair_cv", "chemistry_block"):
            groups = [
                f"{host}::{dopant}"
                for host, dopant in zip(hosts, dopants)
            ]
        else:
            groups = None
        result = paired_bootstrap(
            targets,
            dart_prediction,
            comparator_prediction,
            seed=20265000 + 10 * regime_index + comparator_index,
            draws=10_000,
            groups=groups,
        )
        paired_direct[(regime, comparator)] = result
        audit.close(
            f"paired delta: {regime}/{comparator}",
            float(row["mae_difference_comparator_minus_dart_eV"]),
            result["mean"],
        )
        audit.close(
            f"paired low: {regime}/{comparator}",
            float(row["ci_low_eV"]),
            result["ci_low"],
        )
        audit.close(
            f"paired high: {regime}/{comparator}",
            float(row["ci_high_eV"]),
            result["ci_high"],
        )
        audit.exact(
            f"paired units: {regime}/{comparator}",
            int(row["n_resampling_units"]),
            result["n_units"],
        )
    return predictions, direct, paired_direct


def audit_factorial(
    root: Path, audit: Audit
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    rows = []
    run_csv = {
        (row["variant"], int(row["repeat"])): row
        for row in read_csv(root / "artifacts/prm_results/factorial/runs.csv")
    }
    for run_dir in sorted(
        (root / "artifacts/prm_results/factorial/runs").glob("g*/split*_seed*")
    ):
        variant = run_dir.parent.name
        manifest = read_json(run_dir / "run_manifest.json")
        split_id = str(manifest["split"]["split_id"])
        repeat = int(split_id.rsplit("s", 1)[1])
        record: dict[str, Any] = {
            "variant": variant,
            "repeat": repeat,
            "bits": tuple(int(bit) for bit in variant[1:]),
        }
        for partition, file_name in (
            ("validation", "val_predictions.npz"),
            ("test", "test_predictions.npz"),
        ):
            with np.load(run_dir / file_name, allow_pickle=False) as archive:
                target = np.asarray(archive["targets"], dtype=float)
                prediction = np.asarray(archive["preds"], dtype=float)
            record[f"{partition}_mae"] = float(
                np.mean(np.abs(prediction - target))
            )
            audit.close(
                f"factorial run MAE: {variant}/{repeat}/{partition}",
                float(run_csv[(variant, repeat)][f"{partition}_mae"]),
                record[f"{partition}_mae"],
                atol=2.0e-7,
            )
        rows.append(record)
    audit.exact("factorial run count", len(rows), 40)

    bundle = read_json(root / "artifacts/prm_results/factorial/bundle.json")
    samples = int(bundle["bootstrap"]["samples"])
    variant_stats: dict[str, dict[str, dict[str, float]]] = {}
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        variant_stats[variant] = {
            "validation": bootstrap_mean(
                [row["validation_mae"] for row in subset],
                samples=samples,
                seed=20260719 + int(variant[1:], 2),
            ),
            "test": bootstrap_mean(
                [row["test_mae"] for row in subset],
                samples=samples,
                seed=20260819 + int(variant[1:], 2),
            ),
        }
    for item in bundle["variant_summary"]:
        variant = item["variant"]
        for partition in ("validation", "test"):
            for field in ("mean", "std", "ci_low", "ci_high"):
                audit.close(
                    f"factorial summary: {variant}/{partition}/{field}",
                    item[f"{partition}_mae"][field],
                    variant_stats[variant][partition][field],
                    atol=2.0e-7,
                )

    selected = min(
        variant_stats,
        key=lambda variant: (
            variant_stats[variant]["validation"]["mean"],
            sum(int(bit) for bit in variant[1:]),
            variant,
        ),
    )
    audit.exact(
        "factorial validation-only selection",
        bundle["selection"]["selected_variant"],
        selected,
    )

    terms = {
        "G": (0,),
        "E": (1,),
        "P": (2,),
        "G:E": (0, 1),
        "G:P": (0, 2),
        "E:P": (1, 2),
        "G:E:P": (0, 1, 2),
    }
    effects: dict[str, Any] = {}
    for partition in ("validation", "test"):
        effects[partition] = {}
        for term, positions in terms.items():
            repeats = []
            for repeat in range(42, 47):
                repeat_rows = [row for row in rows if row["repeat"] == repeat]
                positive = []
                negative = []
                for row in repeat_rows:
                    sign_value = int(
                        np.prod(
                            [
                                1 if row["bits"][position] else -1
                                for position in positions
                            ]
                        )
                    )
                    (positive if sign_value > 0 else negative).append(
                        row[f"{partition}_mae"]
                    )
                repeats.append(float(np.mean(positive) - np.mean(negative)))
            effect = bootstrap_mean(
                repeats,
                samples=samples,
                seed=(
                    20261000
                    + 100 * len(positions)
                    + sum(positions)
                    + (0 if partition == "validation" else 10)
                ),
            )
            effect["repeat_effects"] = repeats
            effects[partition][term] = effect
            stored = next(
                item
                for item in bundle["effects"]
                if item["split"] == partition
                and item["metric"] == "mae"
                and item["term"] == term
            )
            for field in ("mean", "std", "ci_low", "ci_high"):
                audit.close(
                    f"factorial effect: {partition}/{term}/{field}",
                    stored[field],
                    effect[field],
                    atol=2.0e-7,
                )
            for index, value in enumerate(repeats):
                audit.close(
                    f"factorial repeat effect: {partition}/{term}/{index}",
                    stored["repeat_effects"][index],
                    value,
                    atol=2.0e-7,
                )

    selected_test = [
        row["test_mae"] for row in rows if row["variant"] == selected
    ]
    locked_test = bootstrap_mean(
        selected_test, samples=samples, seed=20262000
    )
    for field in ("mean", "std", "ci_low", "ci_high"):
        audit.close(
            f"factorial locked test: {field}",
            bundle["selection"]["locked_test"]["mae"][field],
            locked_test[field],
            atol=2.0e-7,
        )
    return bundle, variant_stats, {
        "selected": selected,
        "locked_test": locked_test,
        "effects": effects,
    }


def audit_uq(root: Path, audit: Audit) -> dict[str, Any]:
    metrics = read_json(root / "artifacts/prm_results/uq/metrics.json")
    with np.load(
        root / "artifacts/prm_results/uq/predictions.npz", allow_pickle=False
    ) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    target = arrays["test_targets"].astype(float)
    prediction = arrays["test_mean"].astype(float)
    raw_std = arrays["test_raw_std"].astype(float)
    sigma = arrays["test_sigma"].astype(float)
    calibration = metrics["calibration_contract"]["variance_calibration"]
    reconstructed_sigma = np.sqrt(
        (float(calibration["scale"]) * raw_std) ** 2
        + float(calibration["floor_eV"]) ** 2
    )
    audit.exact(
        "UQ calibrated sigma reconstruction",
        bool(np.allclose(sigma, reconstructed_sigma, rtol=0.0, atol=1.0e-12)),
        True,
    )
    error = np.abs(prediction - target)
    z = (target - prediction) / sigma
    gaussian_nll = float(
        np.mean(np.log(sigma) + 0.5 * math.log(2.0 * math.pi) + 0.5 * z**2)
    )
    phi = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    crps = sigma * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * phi
        - 1.0 / math.sqrt(math.pi)
    )
    order = np.argsort(sigma, kind="stable")
    risk = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    oracle = np.cumsum(np.sort(error)) / np.arange(1, len(error) + 1)
    direct: dict[str, Any] = {
        "n": len(target),
        "mae": float(error.mean()),
        "gaussian_nll": gaussian_nll,
        "mean_gaussian_crps_eV": float(crps.mean()),
        "uncertainty_absolute_error_spearman": finite_spearman(sigma, error),
        "aurc_eV": float(risk.mean()),
        "oracle_aurc_eV": float(oracle.mean()),
        "excess_aurc_eV": float(risk.mean() - oracle.mean()),
        "intervals": {},
    }
    stored_test = metrics["test"]
    audit.exact("UQ test count", stored_test["n"], direct["n"])
    audit.close(
        "UQ MAE", stored_test["point_prediction"]["mae"], direct["mae"]
    )
    for field in (
        "gaussian_nll",
        "mean_gaussian_crps_eV",
        "uncertainty_absolute_error_spearman",
    ):
        audit.close(f"UQ {field}", stored_test[field], direct[field])
    for field in ("aurc_eV", "oracle_aurc_eV", "excess_aurc_eV"):
        audit.close(
            f"UQ {field}",
            stored_test["selective_prediction"][field],
            direct[field],
        )

    calibration_target = arrays["calibration_targets"].astype(float)
    calibration_prediction = arrays["calibration_mean"].astype(float)
    calibration_sigma = arrays["calibration_sigma"].astype(float)
    conformal_rows = arrays["conformal_rows"].astype(int)
    scores = np.sort(
        np.abs(
            calibration_target[conformal_rows]
            - calibration_prediction[conformal_rows]
        )
        / calibration_sigma[conformal_rows]
    )
    for coverage in (0.50, 0.80, 0.90, 0.95):
        key = f"{coverage:.2f}"
        rank = min(len(scores), int(math.ceil((len(scores) + 1) * coverage)))
        quantile = float(scores[rank - 1])
        observed = float(np.mean(error <= quantile * sigma))
        width = float(np.mean(2.0 * quantile * sigma))
        direct["intervals"][key] = {
            "conformal_quantile": quantile,
            "observed_test_coverage": observed,
            "mean_test_width_eV": width,
        }
        for field in (
            "conformal_quantile",
            "observed_test_coverage",
            "mean_test_width_eV",
        ):
            audit.close(
                f"UQ interval {key}/{field}",
                stored_test["intervals"][key][field],
                direct["intervals"][key][field],
            )
    return direct


def numeric_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        converted: dict[str, Any] = dict(row)
        for key, value in row.items():
            if key in {"host", "dopant", "defecttype", "true_preference",
                       "predicted_preference", "true_global_best",
                       "predicted_global_best", "true_best_site",
                       "predicted_best_site", "axis", "group"}:
                continue
            if value == "":
                converted[key] = None
            else:
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
        output.append(converted)
    return output


def audit_materials(root: Path, audit: Audit) -> dict[str, Any]:
    summary = read_json(root / "artifacts/prm_results/materials/summary.json")
    pair_rows = numeric_rows(
        read_csv(root / "artifacts/prm_results/materials/pair_preferences.csv")
    )
    site_rows = numeric_rows(
        read_csv(root / "artifacts/prm_results/materials/site_selection.csv")
    )
    preference_rows = [
        row for row in pair_rows if bool(int(row["preference_eligible"]))
    ]
    top2_rows = [row for row in site_rows if bool(int(row["top2_eligible"]))]
    direct = {
        "accuracy": cluster_bootstrap_rows(
            preference_rows, "preference_correct", seed=20263001
        ),
        "margin_mae_eV": cluster_bootstrap_rows(
            pair_rows, "margin_absolute_error_eV", seed=20263002
        ),
        "global_screening_regret_eV": cluster_bootstrap_rows(
            pair_rows, "global_screening_regret_eV", seed=20263003
        ),
        "global_exact_site_accuracy": cluster_bootstrap_rows(
            pair_rows, "global_site_correct", seed=20263004
        ),
        "exact_accuracy": cluster_bootstrap_rows(
            site_rows, "exact_site_correct", seed=20263005
        ),
        "top2_accuracy": cluster_bootstrap_rows(
            top2_rows, "true_best_in_predicted_top2", seed=20263006
        ),
        "screening_regret_eV": cluster_bootstrap_rows(
            site_rows, "screening_regret_eV", seed=20263007
        ),
        "preference_reference_accuracy": cluster_bootstrap_rows(
            preference_rows, "majority_preference_correct", seed=20263008
        ),
        "preference_gain": cluster_bootstrap_rows(
            preference_rows, "preference_gain_over_majority", seed=20263009
        ),
        "global_reference_accuracy": cluster_bootstrap_rows(
            pair_rows, "global_uniform_exact_expectation", seed=20263010
        ),
        "global_gain": cluster_bootstrap_rows(
            pair_rows, "global_exact_gain_over_uniform", seed=20263011
        ),
        "exact_reference_accuracy": cluster_bootstrap_rows(
            site_rows, "uniform_exact_expectation", seed=20263012
        ),
        "exact_gain": cluster_bootstrap_rows(
            site_rows, "exact_gain_over_uniform", seed=20263013
        ),
        "top2_reference_accuracy": cluster_bootstrap_rows(
            top2_rows, "uniform_top2_expectation", seed=20263014
        ),
        "top2_gain": cluster_bootstrap_rows(
            top2_rows, "top2_gain_over_uniform", seed=20263015
        ),
        "margin_spearman": finite_spearman(
            [row["true_margin_eV"] for row in pair_rows],
            [row["predicted_margin_eV"] for row in pair_rows],
        ),
    }
    preference = summary["defect_type_preference"]
    within = summary["within_defect_type_site_selection"]
    stored_map = {
        "accuracy": preference["accuracy"],
        "margin_mae_eV": preference["margin_mae_eV"],
        "global_screening_regret_eV": preference[
            "global_screening_regret_eV"
        ],
        "global_exact_site_accuracy": preference[
            "global_exact_site_accuracy"
        ],
        "exact_accuracy": within["exact_accuracy"],
        "top2_accuracy": within["top2_accuracy"],
        "screening_regret_eV": within["screening_regret_eV"],
        "preference_reference_accuracy": preference["accuracy_reference"][
            "accuracy"
        ],
        "preference_gain": preference["accuracy_reference"][
            "model_minus_reference"
        ],
        "global_reference_accuracy": preference[
            "global_exact_site_reference"
        ]["accuracy"],
        "global_gain": preference["global_exact_site_reference"][
            "model_minus_reference"
        ],
        "exact_reference_accuracy": within["exact_accuracy_reference"][
            "accuracy"
        ],
        "exact_gain": within["exact_accuracy_reference"][
            "model_minus_reference"
        ],
        "top2_reference_accuracy": within["top2_accuracy_reference"][
            "accuracy"
        ],
        "top2_gain": within["top2_accuracy_reference"][
            "model_minus_reference"
        ],
    }
    for metric, result in direct.items():
        if metric == "margin_spearman":
            audit.close(
                "materials margin Spearman",
                preference["margin_spearman"],
                result,
            )
            continue
        for field in ("mean", "std", "ci_low", "ci_high", "n", "n_clusters"):
            if field in ("n", "n_clusters"):
                audit.exact(
                    f"materials {metric}/{field}",
                    stored_map[metric][field],
                    result[field],
                )
            else:
                audit.close(
                    f"materials {metric}/{field}",
                    stored_map[metric][field],
                    result[field],
                )

    class_counts: dict[str, int] = defaultdict(int)
    for row in preference_rows:
        class_counts[str(row["true_preference"])] += 1
    majority_class = min(
        class_counts, key=lambda label: (-class_counts[label], label)
    )
    reference = preference["accuracy_reference"]
    audit.exact(
        "materials preference majority class",
        reference["majority_class"],
        majority_class,
    )
    audit.exact(
        "materials preference class counts",
        reference["class_counts"],
        dict(sorted(class_counts.items())),
    )

    group_rows = numeric_rows(
        read_csv(root / "artifacts/prm_results/materials/group_errors.csv")
    )
    heterogeneity = {}
    for axis in ("host", "dopant"):
        members = [row for row in group_rows if row["axis"] == axis]
        rho = finite_spearman(
            [row["n"] for row in members], [row["mae_eV"] for row in members]
        )
        worst = sorted(
            members, key=lambda row: (-row["mae_eV"], str(row["group"]))
        )[0]
        heterogeneity[axis] = {"spearman": rho, "worst": worst}
        audit.close(
            f"materials {axis} count-error Spearman",
            summary["error_heterogeneity"][axis][
                "sample_count_mae_spearman"
            ],
            rho,
        )
        audit.exact(
            f"materials {axis} worst group",
            summary["error_heterogeneity"][axis]["worst_groups"][0]["group"],
            worst["group"],
        )
    direct["heterogeneity"] = heterogeneity
    return direct


def audit_sensitivity(
    root: Path,
    audit: Audit,
    samples: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    result_root = root / "artifacts/prm_results/sensitivity/schnet_readout"
    summary = read_json(result_root / "summary.json")
    with np.load(result_root / "predictions.npz", allow_pickle=False) as archive:
        indices = np.asarray(archive["indices"], dtype=np.int64)
        target = np.asarray(archive["targets"], dtype=float)
        add = np.asarray(archive["add_predictions"], dtype=float)
        mean = np.asarray(archive["mean_predictions"], dtype=float)
    hosts = [str(samples[int(index)]["host"]) for index in indices]
    natoms = np.asarray([samples[int(index)]["natoms"] for index in indices])
    add_error = np.abs(add - target)
    mean_error = np.abs(mean - target)
    paired = paired_bootstrap(
        target,
        add,
        mean,
        seed=20260729,
        draws=20_000,
        groups=hosts,
    )
    direct: dict[str, Any] = {
        "add_mae": float(add_error.mean()),
        "mean_mae": float(mean_error.mean()),
        "difference": float((mean_error - add_error).mean()),
        "ci_low": paired["ci_low"],
        "ci_high": paired["ci_high"],
        "sample_size_spearman": {
            "add": finite_spearman(natoms, add_error),
            "mean": finite_spearman(natoms, mean_error),
        },
    }
    audit.close("sensitivity add MAE", summary["add"]["mae"], direct["add_mae"])
    audit.close(
        "sensitivity mean MAE", summary["mean"]["mae"], direct["mean_mae"]
    )
    paired_stored = summary["paired_host_cluster_bootstrap"]
    audit.close(
        "sensitivity paired difference",
        paired_stored["mae_difference_mean_minus_add_eV"],
        direct["difference"],
    )
    audit.close(
        "sensitivity paired low", paired_stored["ci_low_eV"], direct["ci_low"]
    )
    audit.close(
        "sensitivity paired high",
        paired_stored["ci_high_eV"],
        direct["ci_high"],
    )
    for readout in ("add", "mean"):
        audit.close(
            f"sensitivity sample size correlation/{readout}",
            summary["sample_natoms_vs_absolute_error_spearman"][readout],
            direct["sample_size_spearman"][readout],
        )

    labels = np.asarray(hosts)
    host_rows = []
    for host in sorted(set(hosts)):
        mask = labels == host
        host_rows.append(
            {
                "host": host,
                "n": int(mask.sum()),
                "median_natoms": float(np.median(natoms[mask])),
                "add_mae_eV": float(add_error[mask].mean()),
                "mean_mae_eV": float(mean_error[mask].mean()),
            }
        )
    direct["host_size_spearman"] = {
        "add": finite_spearman(
            [row["median_natoms"] for row in host_rows],
            [row["add_mae_eV"] for row in host_rows],
        ),
        "mean": finite_spearman(
            [row["median_natoms"] for row in host_rows],
            [row["mean_mae_eV"] for row in host_rows],
        ),
    }
    for readout in ("add", "mean"):
        audit.close(
            f"sensitivity host size correlation/{readout}",
            summary["host_median_natoms_vs_mae_spearman"][readout],
            direct["host_size_spearman"][readout],
        )

    stored_host_rows = {
        row["host"]: row
        for row in numeric_rows(read_csv(result_root / "host_metrics.csv"))
    }
    for row in host_rows:
        stored = stored_host_rows[row["host"]]
        for field in ("n", "median_natoms", "add_mae_eV", "mean_mae_eV"):
            if field == "n":
                audit.exact(
                    f"sensitivity host row {row['host']}/{field}",
                    int(stored[field]),
                    row[field],
                )
            else:
                audit.close(
                    f"sensitivity host row {row['host']}/{field}",
                    stored[field],
                    row[field],
                )

    index_position = {int(index): position for position, index in enumerate(indices)}
    fold_direct = []
    for fold in range(5):
        split_id = f"host_cv5_f{fold}"
        split = read_json(
            root / f"artifacts/prm_protocol_v2/splits/{split_id}.json"
        )
        positions = np.asarray(
            [index_position[int(index)] for index in split["test"]], dtype=int
        )
        add_mae = float(add_error[positions].mean())
        mean_mae = float(mean_error[positions].mean())
        fold_direct.append(
            {
                "split_id": split_id,
                "n": len(positions),
                "add_mae_eV": add_mae,
                "mean_mae_eV": mean_mae,
                "mean_minus_add_mae_eV": mean_mae - add_mae,
            }
        )
    stored_folds = {
        row["split_id"]: row
        for row in numeric_rows(read_csv(result_root / "fold_metrics.csv"))
    }
    for row in fold_direct:
        stored = stored_folds[row["split_id"]]
        audit.exact(
            f"sensitivity fold {row['split_id']}/n",
            int(stored["n"]),
            row["n"],
        )
        for field in (
            "add_mae_eV", "mean_mae_eV", "mean_minus_add_mae_eV"
        ):
            audit.close(
                f"sensitivity fold {row['split_id']}/{field}",
                stored[field],
                row[field],
            )
    direct["folds"] = fold_direct
    direct["mean_better_folds"] = sum(
        row["mean_mae_eV"] < row["add_mae_eV"] for row in fold_direct
    )
    audit.exact(
        "sensitivity directional folds",
        summary["fold_directional_consistency"]["mean_better_folds"],
        direct["mean_better_folds"],
    )
    return direct


def build_expected_macros(
    descriptor_selection: Mapping[str, Mapping[str, Any]],
    comparison: Mapping[tuple[str, str], Mapping[str, Any]],
    factorial: Mapping[str, Any],
    uq: Mapping[str, Any],
    materials: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> dict[str, str]:
    expected = {
        "PRMSelectedVariant": rf"\texttt{{{factorial['selected']}}}",
        "PRMSelectedValidationMAE": (
            f"{factorial['validation']['mean']:.3f}"
        ),
        "PRMSelectedValidationMAELow": (
            f"{factorial['validation']['ci_low']:.3f}"
        ),
        "PRMSelectedValidationMAEHigh": (
            f"{factorial['validation']['ci_high']:.3f}"
        ),
        "PRMSelectedTestMAE": f"{factorial['locked_test']['mean']:.3f}",
        "PRMSelectedTestMAELow": (
            f"{factorial['locked_test']['ci_low']:.3f}"
        ),
        "PRMSelectedTestMAEHigh": (
            f"{factorial['locked_test']['ci_high']:.3f}"
        ),
    }
    for regime in REGIMES:
        prefix = REGIME_PREFIX[regime]
        expected.update(
            {
                f"PRM{prefix}DARTMAE": (
                    f"{comparison[(regime, 'dart')]['mae']:.3f}"
                ),
                f"PRM{prefix}SchNetMAE": (
                    f"{comparison[(regime, 'schnet')]['mae']:.3f}"
                ),
                f"PRM{prefix}DescriptorMAE": (
                    f"{comparison[(regime, DESCRIPTOR_SELECTED_MODEL)]['mae']:.3f}"
                ),
                f"PRM{prefix}DescriptorName": DISPLAY_FAMILY[
                    DESCRIPTOR_SELECTED_MODEL
                ],
                f"PRM{prefix}DARTLowEnergyMAE": (
                    f"{comparison[(regime, 'dart')]['low_energy_mae']:.3f}"
                ),
                f"PRM{prefix}DARTLowEnergyRecall": (
                    f"{100.0 * comparison[(regime, 'dart')]['low_energy_recall']:.1f}"
                ),
            }
        )
    expected.update(
        {
            "PRMUQTestMAE": f"{uq['mae']:.3f}",
            "PRMUQNLL": f"{uq['gaussian_nll']:.3f}",
            "PRMUQCRPS": f"{uq['mean_gaussian_crps_eV']:.3f}",
            "PRMUQErrorSpearman": (
                f"{uq['uncertainty_absolute_error_spearman']:.3f}"
            ),
            "PRMUQAURC": f"{uq['aurc_eV']:.3f}",
            "PRMUQExcessAURC": f"{uq['excess_aurc_eV']:.3f}",
        }
    )
    coverage_names = {
        "0.50": "Fifty",
        "0.80": "Eighty",
        "0.90": "Ninety",
        "0.95": "NinetyFive",
    }
    for key, name in coverage_names.items():
        expected[f"PRMUQ{name}Coverage"] = (
            f"{100.0 * uq['intervals'][key]['observed_test_coverage']:.1f}"
        )
        expected[f"PRMUQ{name}Width"] = (
            f"{uq['intervals'][key]['mean_test_width_eV']:.3f}"
        )
    expected.update(
        {
            "PRMPreferenceAccuracy": (
                f"{100.0 * materials['accuracy']['mean']:.1f}"
            ),
            "PRMPreferenceMarginMAE": (
                f"{materials['margin_mae_eV']['mean']:.3f}"
            ),
            "PRMPreferenceMarginSpearman": (
                f"{materials['margin_spearman']:.3f}"
            ),
            "PRMGlobalRegret": (
                f"{materials['global_screening_regret_eV']['mean']:.3f}"
            ),
            "PRMGlobalExactAccuracy": (
                f"{100.0 * materials['global_exact_site_accuracy']['mean']:.1f}"
            ),
            "PRMWithinTypeExactAccuracy": (
                f"{100.0 * materials['exact_accuracy']['mean']:.1f}"
            ),
            "PRMWithinTypeTopTwoAccuracy": (
                f"{100.0 * materials['top2_accuracy']['mean']:.1f}"
            ),
            "PRMWithinTypeRegret": (
                f"{materials['screening_regret_eV']['mean']:.3f}"
            ),
            "PRMPreferenceReferenceAccuracy": (
                f"{100.0 * materials['preference_reference_accuracy']['mean']:.1f}"
            ),
            "PRMPreferenceGain": (
                f"{100.0 * materials['preference_gain']['mean']:.1f}"
            ),
            "PRMPreferenceGainLow": (
                f"{100.0 * materials['preference_gain']['ci_low']:.1f}"
            ),
            "PRMPreferenceGainHigh": (
                f"{100.0 * materials['preference_gain']['ci_high']:.1f}"
            ),
            "PRMGlobalReferenceAccuracy": (
                f"{100.0 * materials['global_reference_accuracy']['mean']:.1f}"
            ),
            "PRMGlobalAccuracyGain": (
                f"{100.0 * materials['global_gain']['mean']:.1f}"
            ),
            "PRMGlobalAccuracyGainLow": (
                f"{100.0 * materials['global_gain']['ci_low']:.1f}"
            ),
            "PRMGlobalAccuracyGainHigh": (
                f"{100.0 * materials['global_gain']['ci_high']:.1f}"
            ),
            "PRMWithinTypeReferenceExactAccuracy": (
                f"{100.0 * materials['exact_reference_accuracy']['mean']:.1f}"
            ),
            "PRMWithinTypeExactAccuracyGain": (
                f"{100.0 * materials['exact_gain']['mean']:.1f}"
            ),
            "PRMWithinTypeExactAccuracyGainLow": (
                f"{100.0 * materials['exact_gain']['ci_low']:.1f}"
            ),
            "PRMWithinTypeExactAccuracyGainHigh": (
                f"{100.0 * materials['exact_gain']['ci_high']:.1f}"
            ),
            "PRMWithinTypeReferenceTopTwoAccuracy": (
                f"{100.0 * materials['top2_reference_accuracy']['mean']:.1f}"
            ),
            "PRMWithinTypeTopTwoAccuracyGain": (
                f"{100.0 * materials['top2_gain']['mean']:.1f}"
            ),
            "PRMWithinTypeTopTwoAccuracyGainLow": (
                f"{100.0 * materials['top2_gain']['ci_low']:.1f}"
            ),
            "PRMWithinTypeTopTwoAccuracyGainHigh": (
                f"{100.0 * materials['top2_gain']['ci_high']:.1f}"
            ),
            "PRMSchNetAddHostMAE": f"{sensitivity['add_mae']:.3f}",
            "PRMSchNetMeanHostMAE": f"{sensitivity['mean_mae']:.3f}",
            "PRMSchNetMeanMinusAddHostMAE": (
                f"{sensitivity['difference']:.3f}"
            ),
            "PRMSchNetMeanMinusAddHostMAELow": (
                f"{sensitivity['ci_low']:.3f}"
            ),
            "PRMSchNetMeanMinusAddHostMAEHigh": (
                f"{sensitivity['ci_high']:.3f}"
            ),
            "PRMSchNetMeanBetterFolds": (
                f"{sensitivity['mean_better_folds']}"
            ),
            "PRMSchNetAddSampleSizeErrorSpearman": (
                f"{sensitivity['sample_size_spearman']['add']:.3f}"
            ),
            "PRMSchNetMeanSampleSizeErrorSpearman": (
                f"{sensitivity['sample_size_spearman']['mean']:.3f}"
            ),
            "PRMSchNetAddHostSizeErrorSpearman": (
                f"{sensitivity['host_size_spearman']['add']:.3f}"
            ),
            "PRMSchNetMeanHostSizeErrorSpearman": (
                f"{sensitivity['host_size_spearman']['mean']:.3f}"
            ),
        }
    )
    return expected


def audit_macros_and_tables(
    root: Path,
    audit: Audit,
    expected_macros: Mapping[str, str],
    descriptor_selection: Mapping[str, Mapping[str, Any]],
    comparison: Mapping[tuple[str, str], Mapping[str, Any]],
    paired: Mapping[tuple[str, str], Mapping[str, float]],
    variant_stats: Mapping[str, Mapping[str, Mapping[str, float]]],
    factorial_direct: Mapping[str, Any],
    uq: Mapping[str, Any],
    materials: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> dict[str, int]:
    macros = parse_macros(root / "paper_Q1/generated/results_macros.tex")
    audit.exact("macro name set", sorted(macros), sorted(expected_macros))
    for name, expected in expected_macros.items():
        audit.exact(f"macro value: {name}", macros.get(name), expected)

    manuscript_files = [
        root / "paper_Q1/main.tex",
        root / "paper_Q1/supplement.tex",
        *sorted((root / "paper_Q1/sections").glob("*.tex")),
    ]
    manuscript = "\n".join(path.read_text() for path in manuscript_files)
    used = set(re.findall(r"\\(PRM[A-Za-z]+)", manuscript))
    audit.exact("manuscript macro resolution", sorted(used - set(macros)), [])
    result_text = "\n".join(
        (root / f"paper_Q1/sections/{name}.tex").read_text()
        for name in ("results", "discussion", "conclusion")
    )
    audit.exact(
        "hard-coded decimal result claims",
        re.findall(r"(?<![A-Za-z])[-+]?\d+\.\d+", result_text),
        [],
    )

    table_fragments = 0
    benchmark = (
        root / "paper_Q1/generated/tab_benchmark.tex"
    ).read_text()
    applicability = (
        root / "paper_Q1/generated/tab_applicability.tex"
    ).read_text()
    for regime in REGIMES:
        dart = comparison[(regime, "dart")]
        schnet = comparison[(regime, "schnet")]
        descriptor = comparison[(regime, DESCRIPTOR_SELECTED_MODEL)]
        schnet_pair = paired[(regime, "schnet")]
        descriptor_pair = paired[(regime, DESCRIPTOR_SELECTED_MODEL)]
        row = (
            f"{REGIME_LABEL[regime]} & {dart['mae']:.3f} & "
            f"{schnet['mae']:.3f} & {descriptor['mae']:.3f} & "
            f"{signed(schnet_pair['mean'])} "
            f"[{signed(schnet_pair['ci_low'])}, {signed(schnet_pair['ci_high'])}] & "
            f"{signed(descriptor_pair['mean'])} "
            f"[{signed(descriptor_pair['ci_low'])}, "
            f"{signed(descriptor_pair['ci_high'])}]"
        )
        audit.contains(f"benchmark row: {regime}", benchmark, row)
        table_fragments += 1
        favorable = (
            "--" if dart["favorable_mae"] is None
            else f"{dart['favorable_mae']:.3f}"
        )
        row = (
            f"{REGIME_LABEL[regime]} & {dart['host_macro_mae']:.3f} & "
            f"{dart['dopant_macro_mae']:.3f} & {favorable} & "
            f"{dart['low_energy_mae']:.3f} & "
            f"{100.0 * dart['low_energy_recall']:.1f}"
        )
        audit.contains(f"applicability row: {regime}", applicability, row)
        table_fragments += 1

    factorial_table = (
        root / "paper_Q1/generated/tab_factorial.tex"
    ).read_text()
    selected = factorial_direct["selected"]
    for variant, partitions in variant_stats.items():
        bits = " & ".join(variant[1:])
        variant_cell = rf"\texttt{{{variant}}}"
        validation = (
            f"{partitions['validation']['mean']:.3f} "
            f"[{partitions['validation']['ci_low']:.3f}, "
            f"{partitions['validation']['ci_high']:.3f}]"
        )
        test = (
            f"{partitions['test']['mean']:.3f} "
            f"[{partitions['test']['ci_low']:.3f}, "
            f"{partitions['test']['ci_high']:.3f}]"
        )
        if variant == selected:
            variant_cell = rf"\textbf{{{variant_cell}}}"
            validation = rf"\textbf{{{validation}}}"
            test = rf"\textbf{{{test}}}"
        fragment = f"{variant_cell} & {bits} & {validation} & {test}"
        audit.contains(f"factorial table row: {variant}", factorial_table, fragment)
        table_fragments += 1

    uq_table = (root / "paper_Q1/generated/tab_uq.tex").read_text()
    for key in ("0.50", "0.80", "0.90", "0.95"):
        interval = uq["intervals"][key]
        fragment = (
            f"{int(round(100.0 * float(key)))} & "
            f"{100.0 * interval['observed_test_coverage']:.1f} & "
            f"{interval['mean_test_width_eV']:.3f} & "
            f"{interval['conformal_quantile']:.3f}"
        )
        audit.contains(f"UQ table row: {key}", uq_table, fragment)
        table_fragments += 1

    screening_table = (
        root / "paper_Q1/generated/tab_screening.tex"
    ).read_text()
    screening_rows = [
        (
            "Incorporation-class preference accuracy (\\%)",
            materials["accuracy"],
            materials["preference_reference_accuracy"],
            materials["preference_gain"],
            True,
        ),
        (
            "Global exact-site accuracy (\\%)",
            materials["global_exact_site_accuracy"],
            materials["global_reference_accuracy"],
            materials["global_gain"],
            True,
        ),
        (
            "Global screening regret (eV)",
            materials["global_screening_regret_eV"],
            None,
            None,
            False,
        ),
        (
            "Within-class exact-site accuracy (\\%)",
            materials["exact_accuracy"],
            materials["exact_reference_accuracy"],
            materials["exact_gain"],
            True,
        ),
        (
            "Within-class top-2 accuracy (\\%)",
            materials["top2_accuracy"],
            materials["top2_reference_accuracy"],
            materials["top2_gain"],
            True,
        ),
        (
            "Within-class screening regret (eV)",
            materials["screening_regret_eV"],
            None,
            None,
            False,
        ),
    ]
    for label, result, reference, gain, percent in screening_rows:
        scale = 100.0 if percent else 1.0
        decimals = 1 if percent else 3
        reference_text = (
            f"{scale * reference['mean']:.{decimals}f}"
            if reference is not None else "--"
        )
        gain_text = (
            f"{scale * gain['mean']:.{decimals}f} "
            f"[{scale * gain['ci_low']:.{decimals}f}, "
            f"{scale * gain['ci_high']:.{decimals}f}]"
            if gain is not None else "--"
        )
        fragment = (
            f"{label} & {scale * result['mean']:.{decimals}f} "
            f"[{scale * result['ci_low']:.{decimals}f}, "
            f"{scale * result['ci_high']:.{decimals}f}] & "
            f"{reference_text} & {gain_text} & {result['n']}"
        )
        audit.contains(f"screening table row: {label}", screening_table, fragment)
        table_fragments += 1

    sensitivity_table = (
        root / "paper_Q1/generated/tab_schnet_readout.tex"
    ).read_text()
    for fold_index, row in enumerate(sensitivity["folds"], start=1):
        fragment = (
            f"Fold {fold_index} & {row['n']} & {row['add_mae_eV']:.3f} & "
            f"{row['mean_mae_eV']:.3f} & "
            f"{row['mean_minus_add_mae_eV']:.3f}"
        )
        audit.contains(
            f"sensitivity table row: {fold_index}", sensitivity_table, fragment
        )
        table_fragments += 1
    pooled_fragment = (
        f"10224 & \\textbf{{{sensitivity['add_mae']:.3f}}} & "
        f"\\textbf{{{sensitivity['mean_mae']:.3f}}} & "
        f"\\textbf{{{sensitivity['difference']:.3f}}}"
    )
    audit.contains("sensitivity pooled table row", sensitivity_table, pooled_fragment)
    table_fragments += 1

    factorial_narrative = (
        root / "paper_Q1/generated/results_factorial_narrative.tex"
    ).read_text()
    for term in ("G", "E", "P", "G:P", "E:P"):
        effect = factorial_direct["effects"]["validation"][term]
        digits = 4 if (
            effect["ci_low"] * effect["ci_high"] > 0.0
            and min(abs(effect["ci_low"]), abs(effect["ci_high"])) < 0.0005
        ) else 3
        fragment = (
            f"$\\Delta={effect['mean']:+.{digits}f}$ "
            f"[{effect['ci_low']:+.{digits}f}, {effect['ci_high']:+.{digits}f}]"
        )
        audit.contains(
            f"factorial narrative effect: {term}", factorial_narrative, fragment
        )
    heterogeneity_text = (
        root
        / "paper_Q1/generated/results_error_heterogeneity_narrative.tex"
    ).read_text()
    host = materials["heterogeneity"]["host"]
    dopant = materials["heterogeneity"]["dopant"]
    audit.contains(
        "heterogeneity correlations",
        heterogeneity_text,
        f"{host['spearman']:.3f} and {dopant['spearman']:.3f}",
    )
    for axis, record in (("host", host), ("impurity", dopant)):
        worst = record["worst"]
        audit.contains(
            f"heterogeneity worst {axis}",
            heterogeneity_text,
            f"{axis} {worst['group']} ({worst['mae_eV']:.3f}~eV; "
            f"$n={int(worst['n'])}$)",
        )
    return {
        "defined_macros": len(macros),
        "used_macros": len(used),
        "table_rows_checked": table_fragments,
    }


def audit_method_claims(
    root: Path, audit: Audit
) -> dict[str, Any]:
    protocol = read_json(root / "artifacts/prm_protocol_v2/manifest.json")
    data_audit = read_json(root / "artifacts/prm_protocol_v2/data_audit.json")
    overlap = read_json(
        root / "artifacts/prm_results/operations/pretraining_overlap_3a98513.json"
    )
    raw = data_audit["formation_energy_provenance"]["raw_database_audit"]
    filters = raw["filter_replay"]
    duplicate = data_audit["duplicates"]["canonical_deduplication"]
    chemistry_split = next(
        item for item in data_audit["splits"]
        if item["split_id"] == "chemistry_block_g6x3d"
    )
    dart_manifest = next(
        (root / "artifacts/prm_results/comparison/runs/selected/g111/transfer")
        .glob("**/run_manifest.json")
    )
    schnet_manifest = next(
        (root / "artifacts/prm_results/comparison/runs/baselines/schnet")
        .glob("**/run_manifest.json")
    )
    facts = {
        "raw_rows": filters["raw_rows"],
        "unconverged": filters["not_converged"],
        "missing_or_nonfinite": filters["missing_or_nonfinite_eform"],
        "outside_abs_20_eV": filters["outside_abs_20_eV"],
        "source_filtered_rows": protocol["n_samples"],
        "raw_component_exclusions": duplicate["n_raw_component_excluded"],
        "ambiguous_identity_exclusions": duplicate[
            "n_ambiguous_identity_excluded"
        ],
        "duplicate_exclusions": duplicate["n_duplicate_excluded"],
        "modeling_rows": protocol["n_modeling_samples"],
        "chemistry_split": chemistry_split["counts"],
        "uq_split": protocol["uq_split_counts"],
        "pretraining_records": overlap["pretraining"]["n_records"],
        "overlap_hosts": overlap["overlap"]["host_reduced_formula"][
            "n_imp2d_hosts_present"
        ],
        "overlap_reduced_formulas": overlap["overlap"][
            "host_reduced_formula"
        ]["n_imp2d_reduced_formulas_present"],
        "overlap_records": overlap["overlap"]["host_reduced_formula"][
            "n_pretraining_records"
        ],
        "overlap_impurities": overlap["overlap"]["impurity_element"][
            "n_imp2d_impurities_present"
        ],
        "dart_parameters": read_json(dart_manifest)["metrics"]["n_params"],
        "schnet_parameters": read_json(schnet_manifest)["metrics"]["n_params"],
    }
    methods = " ".join(
        (root / "paper_Q1/sections/methods.tex").read_text().split()
    )
    expected_fragments = [
        f"{facts['raw_rows']:,} rows",
        f"{facts['unconverged']} unconverged",
        f"{facts['missing_or_nonfinite']} rows with a missing",
        f"{facts['outside_abs_20_eV']} rows with",
        f"leaving {facts['source_filtered_rows']:,} structures",
        f"In {facts['ambiguous_identity_exclusions']} rows",
        f"contain {facts['duplicate_exclusions']} redundant rows",
        f"contains {facts['modeling_rows']:,} structures",
        (
            f"contains {facts['chemistry_split']['train']:,} training, "
            f"{facts['chemistry_split']['val']:,} validation, and "
            f"{facts['chemistry_split']['test']:,} test structures"
        ),
        f"retained {facts['pretraining_records']:,} pristine",
        (
            f"{facts['overlap_hosts']} of the 44 IMP2D host labels "
            f"({facts['overlap_reduced_formulas']} of 42 distinct"
        ),
        f"among {facts['overlap_records']} JARVIS source records",
        f"all {facts['overlap_impurities']} IMP2D impurity elements",
        f"{facts['dart_parameters']:,} trainable parameters",
        f"with {facts['schnet_parameters']:,} for SchNet",
    ]
    for index, fragment in enumerate(expected_fragments):
        audit.contains(f"method claim {index + 1}", methods, fragment)
    abstract = " ".join((root / "paper_Q1/main.tex").read_text().split())
    audit.contains(
        "abstract modeling set",
        abstract,
        f"{facts['modeling_rows']:,} provenance-audited neutral IMP2D",
    )
    supplement = " ".join(
        (root / "paper_Q1/supplement.tex").read_text().split()
    )
    audit.contains("supplement bootstrap count", supplement, "20,000 bootstrap")
    audit.contains("supplement host count", supplement, "the 44 host materials")
    return facts


def compact_comparison(
    metrics: Mapping[tuple[str, str], Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        regime: {
            model: metrics[(regime, model)]["mae"]
            for model in sorted(
                key[1] for key in metrics if key[0] == regime
            )
        }
        for regime in REGIMES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/prm_results/paper/numeric_audit.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output

    audit = Audit()
    hash_counts = audit_asset_hashes(root, audit)
    samples = load_samples(root / "artifacts/prm_protocol_v2/samples.csv")
    descriptor_selection = read_json(
        root / "artifacts/prm_results/comparison/descriptor_selection.json"
    )
    _, comparison, paired = audit_comparison(
        root, audit, samples, descriptor_selection
    )
    _, variant_stats, factorial_direct = audit_factorial(root, audit)
    selected = factorial_direct["selected"]
    factorial_for_macros = {
        "selected": selected,
        "validation": variant_stats[selected]["validation"],
        "locked_test": factorial_direct["locked_test"],
    }
    uq = audit_uq(root, audit)
    materials = audit_materials(root, audit)
    sensitivity = audit_sensitivity(root, audit, samples)
    expected_macros = build_expected_macros(
        descriptor_selection,
        comparison,
        factorial_for_macros,
        uq,
        materials,
        sensitivity,
    )
    latex_counts = audit_macros_and_tables(
        root,
        audit,
        expected_macros,
        descriptor_selection,
        comparison,
        paired,
        variant_stats,
        factorial_direct,
        uq,
        materials,
        sensitivity,
    )
    method_facts = audit_method_claims(root, audit)

    report = {
        "schema_version": "prm_paper_numeric_audit_v1",
        "status": "pass" if not audit.failures else "fail",
        "checks": {
            "n_checks": audit.n_checks,
            "n_failures": len(audit.failures),
            "failures": audit.failures,
        },
        "asset_hashes": hash_counts,
        "latex_contract": latex_counts,
        "scope_notes": [
            (
                "Neural target arrays are stored as float32; alignment with "
                "the float64 protocol table is checked at 1e-6 eV."
            ),
            (
                "Low-energy ranking diagnostics for the tied descriptor mean "
                "predictor are excluded because they are implementation-order "
                "dependent and are not used in the manuscript."
            ),
            (
                "Report floats are serialized to 15 significant digits to "
                "remove platform-specific libm tails without affecting any "
                "audit tolerance or paper-facing value."
            ),
        ],
        "direct_recomputations": {
            "comparison_mae_eV": compact_comparison(comparison),
            "factorial": {
                "selected_variant": selected,
                "validation_mae_eV": variant_stats[selected]["validation"],
                "locked_test_mae_eV": factorial_direct["locked_test"],
            },
            "uq": uq,
            "materials": materials,
            "schnet_readout": sensitivity,
            "method_facts": method_facts,
        },
    }
    report = canonicalize_json_numbers(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_checks": audit.n_checks,
                "n_failures": len(audit.failures),
                "output": str(output),
            },
            indent=2,
        )
    )
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
