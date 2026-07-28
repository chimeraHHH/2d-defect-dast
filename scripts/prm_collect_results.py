"""Collect DART, SchNet and descriptor results under the frozen PRM splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.prm_metrics import finite_spearman, low_energy_metrics, macro_group_mae
from src.prm_provenance import (
    ExpectedConfig,
    archive_training_artifacts,
    load_verified_factorial_selection,
    load_protocol_targets,
    load_expected_configs,
    validate_dart_assets,
    validate_descriptor_evidence_bundle,
    validate_manifest_config,
    validate_protocol_targets,
    validate_training_completion,
)

SCALAR_METRICS = ("mae", "rmse", "bias", "spearman", "r2")
EXPECTED_RUNS = {
    "dart": {
        "id_repeat": 5, "id_cv": 5, "pair_cv": 5,
        "host_cv": 15, "dopant_cv": 15, "chemistry_block": 3,
    },
    "schnet": {
        "id_repeat": 5, "id_cv": 5, "pair_cv": 5,
        "host_cv": 15, "dopant_cv": 15, "chemistry_block": 3,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_snapshot() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    return {"commit": commit or None, "dirty": bool(status), "status_porcelain": status.splitlines()}


def regime_for_split(split_id: str) -> str:
    if split_id.startswith("id_repeat_s"):
        return "id_repeat"
    if split_id.startswith("id_cv5_f"):
        return "id_cv"
    if split_id.startswith("pair_cv5_f"):
        return "pair_cv"
    if split_id.startswith("host_cv5_f"):
        return "host_cv"
    if split_id.startswith("dopant_cv5_f"):
        return "dopant_cv"
    if split_id == "chemistry_block_g6x3d":
        return "chemistry_block"
    raise ValueError(f"unrecognized result split: {split_id}")


def expected_split_hash(protocol_dir: Path, split_id: str) -> str:
    return file_sha256(protocol_dir / "splits" / f"{split_id}.json")


def validate_descriptor_root(descriptor_root: Path, protocol_dir: Path) -> None:
    manifest_path = descriptor_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    protocol_path = protocol_dir / "manifest.json"
    protocol = json.loads(protocol_path.read_text())
    if manifest.get("schema_version") != "prm_descriptor_manifest_v2":
        raise ValueError("descriptor results lack independently verified data provenance")
    if manifest.get("status") != "complete":
        raise ValueError("descriptor baseline batch is incomplete")
    if manifest.get("selection_data") != "validation only":
        raise ValueError("descriptor baseline selection used non-validation data")
    if manifest.get("data_sha256") != protocol.get("data_sha256"):
        raise ValueError("descriptor dataset/protocol hash mismatch")
    if manifest.get("data_file_sha256") != protocol.get("data_sha256"):
        raise ValueError("descriptor input file hash was not independently verified")
    if manifest.get("protocol_manifest_sha256") != file_sha256(protocol_path):
        raise ValueError("descriptor protocol manifest hash mismatch")
    if (
        not manifest.get("git", {}).get("commit")
        or manifest.get("git", {}).get("dirty")
    ):
        raise ValueError("descriptor baseline Git provenance is inadmissible")
    encoding = manifest.get("metric_encoding", {})
    normalizer_git = encoding.get("normalizer_git", {})
    if (
        encoding.get("schema_version") != "prm_nullable_correlations_v1"
        or not normalizer_git.get("commit")
        or normalizer_git.get("dirty")
    ):
        raise ValueError("descriptor nullable-metric normalization is inadmissible")

    expected_splits = {
        path.stem for path in (protocol_dir / "splits").glob("*.json")
        if path.stem not in {"id_historical_s42", "smoke_protocol"}
    }
    if set(manifest.get("splits", [])) != expected_splits:
        raise ValueError("descriptor split coverage differs from the frozen protocol")
    artifacts = manifest.get("split_artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_splits:
        raise ValueError("descriptor split artifact hashes are incomplete")
    for split_id, record in artifacts.items():
        metrics_path = descriptor_root / split_id / "metrics.json"
        predictions_path = descriptor_root / split_id / "predictions.npz"
        if record.get("metrics_sha256") != file_sha256(metrics_path):
            raise ValueError(f"descriptor metrics hash mismatch: {split_id}")
        if record.get("predictions_sha256") != file_sha256(predictions_path):
            raise ValueError(f"descriptor predictions hash mismatch: {split_id}")


def load_neural_runs(
    manifest_paths: Sequence[Path], model: str, protocol_dir: Path,
    expected_data_sha256: str,
    expected_configs: Mapping[str, ExpectedConfig],
) -> List[Dict[str, Any]]:
    rows = []
    for path in manifest_paths:
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete":
            continue
        if manifest.get("schema_version") != "prm_run_manifest_v1":
            raise ValueError(f"unsupported run manifest: {path}")
        if manifest.get("git", {}).get("dirty"):
            raise ValueError(f"dirty run is inadmissible: {path}")
        expected_config = validate_manifest_config(manifest, expected_configs, path)
        validate_training_completion(manifest, path)
        if model == "dart":
            validate_dart_assets(manifest, path)
        if manifest["data"]["data_sha256"] != expected_data_sha256:
            raise ValueError(f"dataset mismatch: {path}")
        split_id = manifest["split"]["split_id"]
        if manifest["split"].get("sha256") != expected_split_hash(protocol_dir, split_id):
            raise ValueError(f"split mismatch: {path}")
        metrics = manifest["metrics"]
        row: Dict[str, Any] = {
            "model": model, "family": model, "regime": regime_for_split(split_id),
            "split_id": split_id, "seed": int(manifest["seed"]),
            "manifest_path": str(path), "manifest_sha256": file_sha256(path),
            "prediction_path": str(path.parent / "test_predictions.npz"),
            "git_commit": manifest["git"]["commit"],
            "config_sha256": manifest["config_sha256"],
            "expected_config_path": str(expected_config.path),
            "controlled_output_dir": str(expected_config.config["output_dir"]),
        }
        for partition in ("validation", "test"):
            for metric in SCALAR_METRICS:
                row[f"{partition}_{metric}"] = float(metrics[partition][metric])
        rows.append(row)
    return rows


def load_descriptor_runs(
    descriptor_root: Path, protocol_dir: Path,
) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(descriptor_root.glob("*/metrics.json")):
        payload = json.loads(path.read_text())
        split_id = payload["split_id"]
        try:
            regime = regime_for_split(split_id)
        except ValueError:
            continue
        if payload.get("split_sha256") != expected_split_hash(protocol_dir, split_id):
            raise ValueError(f"stale descriptor split: {path}")
        for family, result in sorted(payload["results"].items()):
            row: Dict[str, Any] = {
                "model": f"descriptor:{family}", "family": family,
                "regime": regime, "split_id": split_id, "seed": 42,
                "manifest_path": str(path), "manifest_sha256": file_sha256(path),
                "prediction_path": str(path.parent / "predictions.npz"),
                "git_commit": None,
            }
            for partition in ("validation", "test"):
                for metric in SCALAR_METRICS:
                    value = result[partition][metric]
                    row[f"{partition}_{metric}"] = (
                        None if value is None else float(value)
                    )
            rows.append(row)
    return rows


def comparison_archive_manifests(
    rows: Sequence[Mapping[str, Any]], result_root: Path,
) -> List[Path]:
    """Return transfer/SchNet manifests; factorial evidence is archived once."""
    factorial_root = (result_root / "factorial").resolve()
    paths = []
    for row in rows:
        if row.get("model") not in {"dart", "schnet"}:
            continue
        path = Path(str(row["manifest_path"])).resolve()
        if path.is_relative_to(factorial_root):
            continue
        paths.append(path)
    unique = sorted(set(paths))
    if len(unique) != len(paths):
        raise ValueError("comparison archive contains duplicate neural manifests")
    return unique


def validate_expected_config_coverage(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    expected_configs: Mapping[str, ExpectedConfig],
    result_root: Path,
    *,
    require_all: bool = True,
) -> Dict[str, Any]:
    model_rows = [row for row in rows if row.get("model") == model]
    observed = Counter(str(row.get("controlled_output_dir", "")) for row in model_rows)
    expected = set(expected_configs)
    unknown = sorted(set(observed) - expected)
    duplicates = sorted(
        output_dir for output_dir, count in observed.items() if count != 1
    )
    if unknown:
        raise ValueError(f"{model} runs include uncontrolled outputs: {unknown}")
    if duplicates:
        raise ValueError(
            f"{model} controlled configurations are not represented once: {duplicates}"
        )

    resolved_root = result_root.resolve()
    for row in model_rows:
        output_dir = str(row["controlled_output_dir"])
        expected_manifest = (resolved_root / output_dir / "run_manifest.json").resolve()
        observed_manifest = Path(str(row["manifest_path"])).resolve()
        if not expected_manifest.is_relative_to(resolved_root):
            raise ValueError(
                f"{model} controlled output escapes the result root: {output_dir}"
            )
        if observed_manifest != expected_manifest:
            raise ValueError(
                f"{model} manifest is outside its controlled output directory: "
                f"expected {expected_manifest}, observed {observed_manifest}"
            )

    missing = sorted(expected - set(observed))
    if require_all and missing:
        raise ValueError(
            f"{model} campaign lacks {len(missing)} controlled configurations: {missing}"
        )
    return {
        "n_expected": len(expected),
        "n_observed": len(observed),
        "complete": not missing,
        "missing_output_dirs": missing,
        "expected_config_sha256": sorted(
            expected_config.sha256 for expected_config in expected_configs.values()
        ),
    }


def validate_neural_campaign_commits(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any],
    *, require_all: bool = True,
) -> Dict[str, str]:
    campaigns = {
        "dart_factorial": {
            str(row.get("git_commit")) for row in rows
            if row.get("model") == "dart" and row.get("regime") == "id_repeat"
        },
        "dart_transfer": {
            str(row.get("git_commit")) for row in rows
            if row.get("model") == "dart" and row.get("regime") != "id_repeat"
        },
        "schnet": {
            str(row.get("git_commit")) for row in rows
            if row.get("model") == "schnet"
        },
    }
    observed = {}
    for campaign, commits in campaigns.items():
        if not commits and not require_all:
            continue
        if len(commits) != 1 or "None" in commits or "" in commits:
            raise ValueError(
                f"{campaign} results do not share one recorded code commit: "
                f"{sorted(commits)}"
            )
        observed[campaign] = next(iter(commits))
    expected_factorial = selection.get("training_commits")
    if (
        "dart_factorial" in observed
        and expected_factorial != [observed["dart_factorial"]]
    ):
        raise ValueError("selected DART repeats do not match the bound factorial commit")
    return observed


def select_descriptor_families(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = {}
    regimes = sorted({row["regime"] for row in rows if str(row["model"]).startswith("descriptor:")})
    for regime in regimes:
        candidates = []
        families = sorted({
            str(row["family"]) for row in rows
            if row["regime"] == regime and str(row["model"]).startswith("descriptor:")
        })
        for family in families:
            values = [
                float(row["validation_mae"]) for row in rows
                if row["regime"] == regime and row["family"] == family
                and str(row["model"]).startswith("descriptor:")
            ]
            if not values or not np.isfinite(values).all():
                raise ValueError(f"invalid descriptor validation MAE for {regime}/{family}")
            candidates.append(
                {"family": family, "mean_validation_mae_eV": float(np.mean(values)), "n": len(values)}
            )
        candidates.sort(key=lambda item: (item["mean_validation_mae_eV"], item["family"]))
        selected[regime] = {
            "selection_data": "validation only", "selected_family": candidates[0]["family"],
            "candidates": candidates,
        }
    return selected


def aggregate_fold_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["regime"]), str(row["split_id"]))].append(row)
    outputs = []
    for (model, regime, split_id), members in sorted(groups.items()):
        output: Dict[str, Any] = {
            "model": model, "regime": regime, "split_id": split_id,
            "n_seeds": len(members),
        }
        for partition in ("validation", "test"):
            for metric in SCALAR_METRICS:
                values = [member[f"{partition}_{metric}"] for member in members]
                output[f"{partition}_{metric}"] = (
                    None if any(value is None for value in values)
                    else float(np.mean(values))
                )
        outputs.append(output)
    return outputs


def bootstrap_ci(
    values: Sequence[float | None], seed: int, draws: int = 50_000,
) -> Dict[str, Any]:
    if not values:
        raise ValueError("fold summary requires at least one value")
    if any(value is None for value in values):
        n_defined = sum(value is not None for value in values)
        return {
            "mean": None, "std": None, "ci_low": None, "ci_high": None,
            "n": len(values), "n_defined": n_defined,
            "interval_status": "undefined_or_incomplete_metric",
        }
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("fold summary contains non-finite values")
    if len(array) == 1:
        return {
            "mean": float(array[0]), "std": 0.0, "ci_low": None,
            "ci_high": None, "n": 1, "n_defined": 1,
            "interval_status": "not_estimable_single_fold",
        }
    if draws < 1:
        raise ValueError("fold bootstrap requires at least one draw")
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(array.mean()), "std": float(array.std(ddof=1)),
        "ci_low": float(low), "ci_high": float(high), "n": len(array),
        "n_defined": len(array), "interval_status": "estimated",
    }


def summarize_folds(fold_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        groups[(str(row["model"]), str(row["regime"]))].append(row)
    outputs = []
    for index, ((model, regime), members) in enumerate(sorted(groups.items())):
        output: Dict[str, Any] = {"model": model, "regime": regime, "n_folds": len(members)}
        for partition in ("validation", "test"):
            for metric in SCALAR_METRICS:
                values = [member[f"{partition}_{metric}"] for member in members]
                stats = bootstrap_ci(values, seed=20264000 + index * 20 + len(output))
                for key, value in stats.items():
                    output[f"{partition}_{metric}_{key}"] = value
        outputs.append(output)
    return outputs


def load_prediction_array(
    path: Path, model: str, expected_split_id: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if model.startswith("descriptor:"):
            expected_fields = {
                "schema_version", "split_id", "model_names",
                "val_indices", "val_targets", "val_predictions",
                "test_indices", "test_targets", "test_predictions",
            }
            if set(archive.files) != expected_fields:
                raise ValueError(f"descriptor prediction fields are incomplete: {path}")
            if str(archive["schema_version"].item()) != "prm_descriptor_predictions_v1":
                raise ValueError(f"unsupported descriptor prediction schema: {path}")
            family = model.split(":", 1)[1]
            names = [str(name) for name in archive["model_names"].tolist()]
            if family not in names:
                raise ValueError(f"descriptor family {family} is absent: {path}")
            position = names.index(family)
            raw_indices = np.asarray(archive["test_indices"])
            targets = np.asarray(archive["test_targets"])
            predictions = np.asarray(archive["test_predictions"][position], dtype=float)
        else:
            expected_fields = {
                "schema_version", "split_id", "split", "indices", "preds", "targets",
            }
            if set(archive.files) != expected_fields:
                raise ValueError(f"neural prediction fields are incomplete: {path}")
            if str(archive["schema_version"].item()) != "prm_predictions_v1":
                raise ValueError(f"unsupported neural prediction schema: {path}")
            if str(archive["split"].item()) != "test":
                raise ValueError(f"neural prediction partition mismatch: {path}")
            raw_indices = np.asarray(archive["indices"])
            targets = np.asarray(archive["targets"])
            predictions = np.asarray(archive["preds"], dtype=float)
        split_id = str(archive["split_id"].item())
    if expected_split_id is not None and split_id != expected_split_id:
        raise ValueError(f"prediction split mismatch for {path}")
    if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError(f"prediction indices are invalid: {path}")
    indices = raw_indices.astype(np.int64, copy=False)
    if (
        targets.ndim != 1
        or predictions.ndim != 1
        or len(indices) == 0
        or len(indices) != len(targets)
        or len(indices) != len(predictions)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError(f"prediction vectors are not uniquely aligned: {path}")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError(f"prediction vectors contain non-finite values: {path}")
    order = np.argsort(indices)
    return indices[order], targets[order], predictions[order]


def seed_averaged_prediction(
    rows: Sequence[Mapping[str, Any]], model: str, split_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    members = [row for row in rows if row["model"] == model and row["split_id"] == split_id]
    if not members:
        raise ValueError(f"no predictions for {model} on {split_id}")
    reference_indices = None
    reference_targets = None
    predictions = []
    for member in members:
        indices, targets, values = load_prediction_array(
            Path(member["prediction_path"]), model, expected_split_id=split_id,
        )
        if reference_indices is None:
            reference_indices, reference_targets = indices, targets
        elif not np.array_equal(indices, reference_indices) or not np.allclose(
            targets, reference_targets, rtol=0.0, atol=1e-10,
        ):
            raise ValueError(f"seed predictions do not align for {model} on {split_id}")
        predictions.append(values)
    return reference_indices, reference_targets, np.mean(predictions, axis=0)


def pooled_predictions(
    rows: Sequence[Mapping[str, Any]], model: str, regime: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_ids = sorted({
        str(row["split_id"]) for row in rows
        if row["model"] == model and row["regime"] == regime
    })
    all_indices, all_targets, all_predictions = [], [], []
    for split_id in split_ids:
        indices, targets, predictions = seed_averaged_prediction(rows, model, split_id)
        all_indices.append(indices)
        all_targets.append(targets)
        all_predictions.append(predictions)
    indices = np.concatenate(all_indices)
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{regime} is not an out-of-fold or single-holdout prediction set")
    order = np.argsort(indices)
    return (
        indices[order], np.concatenate(all_targets)[order],
        np.concatenate(all_predictions)[order],
    )


def read_sample_metadata(path: Path) -> Dict[int, Dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["sample_index"]): {
            "host": str(row["host"]), "dopant": str(row["dopant"]),
            "target_eV": float(row["target_eV"]),
        }
        for row in rows
    }


def regression_metrics(
    targets: np.ndarray, predictions: np.ndarray,
    hosts: Sequence[str] | None = None, dopants: Sequence[str] | None = None,
) -> Dict[str, Any]:
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if (
        targets.shape != predictions.shape
        or targets.ndim != 1
        or len(targets) < 2
        or not np.isfinite(targets).all()
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("pooled metrics require finite aligned vectors")
    residual = predictions - targets
    denominator = np.sum((targets - np.mean(targets)) ** 2)
    if denominator <= 0.0:
        raise ValueError("pooled metrics require non-constant targets")
    spearman = (
        None if len(np.unique(predictions)) < 2
        else finite_spearman(targets, predictions, context="pooled Spearman correlation")
    )
    output = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "bias": float(np.mean(residual)),
        "spearman": spearman,
        "r2": float(1.0 - np.sum(residual ** 2) / denominator),
    }
    favorable = np.asarray(targets) <= 0.0
    output["favorable_n"] = int(np.sum(favorable))
    if not np.any(favorable):
        raise ValueError("pooled evaluation has no non-positive formation energies")
    output["favorable_mae"] = float(np.mean(np.abs(residual[favorable])))
    low = low_energy_metrics(targets, predictions, fraction=0.1)
    low_order = np.argsort(np.asarray(targets), kind="stable")[: int(low["k"])]
    output.update(
        {
            "low_energy_k": int(low["k"]),
            "low_energy_mae": float(np.mean(np.abs(residual[low_order]))),
            "low_energy_recall": float(low["top_k_recall"]),
            "predicted_low_energy_mean_target_eV": float(
                low["mean_true_energy_in_predicted_top_k_eV"]
            ),
        }
    )
    if hosts is not None:
        host_metrics = macro_group_mae(targets, predictions, hosts)
        output["host_macro_mae"] = float(host_metrics["macro_mae"])
        output["host_worst_group_mae"] = float(host_metrics["worst_group_mae"])
        output["n_hosts"] = int(host_metrics["n_groups"])
    if dopants is not None:
        dopant_metrics = macro_group_mae(targets, predictions, dopants)
        output["dopant_macro_mae"] = float(dopant_metrics["macro_mae"])
        output["dopant_worst_group_mae"] = float(dopant_metrics["worst_group_mae"])
        output["n_dopants"] = int(dopant_metrics["n_groups"])
    return output


def paired_sample_comparison(
    targets: np.ndarray, reference: np.ndarray, comparator: np.ndarray,
    seed: int, draws: int = 10_000,
    groups: Sequence[str] | None = None,
) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=float)
    reference = np.asarray(reference, dtype=float)
    comparator = np.asarray(comparator, dtype=float)
    if (
        targets.shape != reference.shape
        or targets.shape != comparator.shape
        or targets.ndim != 1
        or len(targets) == 0
        or not np.isfinite(targets).all()
        or not np.isfinite(reference).all()
        or not np.isfinite(comparator).all()
        or draws < 1
    ):
        raise ValueError("paired bootstrap requires finite aligned vectors and draws")
    differences = np.abs(comparator - targets) - np.abs(reference - targets)
    if groups is None:
        cluster_sums = differences
        cluster_counts = np.ones(len(differences), dtype=np.int64)
    else:
        labels = np.asarray(groups)
        if len(labels) != len(differences):
            raise ValueError("bootstrap group labels do not align with predictions")
        _, inverse = np.unique(labels, return_inverse=True)
        cluster_sums = np.bincount(inverse, weights=differences)
        cluster_counts = np.bincount(inverse)
    n_units = len(cluster_sums)
    if n_units < 2:
        raise ValueError("paired bootstrap requires at least two resampling units")

    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, draws, 256):
        size = min(256, draws - start)
        indices = rng.integers(0, n_units, size=(size, n_units))
        chunks.append(
            cluster_sums[indices].sum(axis=1)
            / cluster_counts[indices].sum(axis=1)
        )
    means = np.concatenate(chunks)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mae_difference_comparator_minus_dart_eV": float(differences.mean()),
        "ci_low_eV": float(low), "ci_high_eV": float(high),
        "n": len(differences), "n_resampling_units": n_units,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path,
        default=ROOT / "artifacts/prm_results/factorial/selection.json",
    )
    parser.add_argument("--factorial-bundle", type=Path, default=None)
    parser.add_argument(
        "--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v2",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "artifacts/prm_results/comparison",
    )
    parser.add_argument(
        "--descriptor-evidence", type=Path,
        default=ROOT / "artifacts/prm_results/descriptors/manifest.json",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    protocol_dir = args.protocol_dir.resolve()
    protocol = json.loads((protocol_dir / "manifest.json").read_text())
    sample_metadata = read_sample_metadata(protocol_dir / "samples.csv")
    protocol_targets = load_protocol_targets(protocol_dir / "samples.csv")
    selection, factorial_bundle = load_verified_factorial_selection(
        args.selection, args.factorial_bundle
    )
    if selection["data_sha256"] != protocol["data_sha256"]:
        raise ValueError("factorial selection does not match the frozen protocol dataset")
    variant = selection["selected_variant"]
    dart_paths = sorted((result_root / "factorial" / variant).glob("split*_seed*/run_manifest.json"))
    dart_paths += sorted(
        (result_root / "selected" / variant / "transfer").glob("*/seed*/run_manifest.json")
    )
    schnet_paths = sorted((result_root / "baselines" / "schnet").glob("*/seed*/run_manifest.json"))
    dart_config_paths = sorted(
        (ROOT / "configs/prm/generated/factorial").glob(f"{variant}_*.yaml")
    )
    dart_config_paths += sorted(
        (ROOT / "configs/prm/promoted" / variant / "transfer").glob("*.yaml")
    )
    dart_expected_configs = load_expected_configs(dart_config_paths)
    schnet_expected_configs = load_expected_configs(
        sorted((ROOT / "configs/prm/generated/schnet").glob("*.yaml"))
    )
    dart_rows = load_neural_runs(
        dart_paths, "dart", protocol_dir, protocol["data_sha256"],
        dart_expected_configs,
    )
    schnet_rows = load_neural_runs(
        schnet_paths, "schnet", protocol_dir, protocol["data_sha256"],
        schnet_expected_configs,
    )
    configuration_coverage = {
        "dart": validate_expected_config_coverage(
            dart_rows,
            "dart",
            dart_expected_configs,
            result_root,
            require_all=not args.allow_incomplete,
        ),
        "schnet": validate_expected_config_coverage(
            schnet_rows,
            "schnet",
            schnet_expected_configs,
            result_root,
            require_all=not args.allow_incomplete,
        ),
    }
    rows = dart_rows + schnet_rows
    descriptor_root = result_root / "baselines" / "descriptors"
    validate_descriptor_root(descriptor_root, protocol_dir)
    descriptor_evidence = validate_descriptor_evidence_bundle(
        args.descriptor_evidence,
        protocol_dir,
        repository_root=ROOT,
    )
    if file_sha256(descriptor_root / "manifest.json") != descriptor_evidence[
        "descriptor_manifest"
    ]["sha256"]:
        raise ValueError("comparison descriptor source differs from GitHub evidence")
    rows += load_descriptor_runs(descriptor_root, protocol_dir)

    if not args.allow_incomplete:
        for model, regimes in EXPECTED_RUNS.items():
            for regime, expected in regimes.items():
                observed = sum(row["model"] == model and row["regime"] == regime for row in rows)
                if observed != expected:
                    raise ValueError(
                        f"expected {expected} {model}/{regime} runs, found {observed}"
                    )

    campaign_commits = validate_neural_campaign_commits(
        rows, selection, require_all=not args.allow_incomplete
    )

    descriptor_selection = select_descriptor_families(rows)
    retained_rows = [
        row for row in rows
        if not str(row["model"]).startswith("descriptor:")
        or row["family"] == descriptor_selection[row["regime"]]["selected_family"]
        or row["family"] == "mean"
    ]
    fold_rows = aggregate_fold_rows(retained_rows)
    summary_rows = summarize_folds(fold_rows)

    pooled_rows = []
    comparison_rows = []
    pooled_regimes = ("id_cv", "pair_cv", "host_cv", "dopant_cv", "chemistry_block")
    for regime_index, regime in enumerate(pooled_regimes):
        models = ["dart", "schnet"]
        selected_descriptor = f"descriptor:{descriptor_selection[regime]['selected_family']}"
        models.extend([selected_descriptor, "descriptor:mean"])
        models = list(dict.fromkeys(models))
        pooled = {}
        for model in models:
            try:
                indices, targets, predictions = pooled_predictions(retained_rows, model, regime)
            except ValueError:
                if args.allow_incomplete:
                    continue
                raise
            canonical_targets = validate_protocol_targets(
                indices, targets, protocol_targets,
                context=f"{regime}/{model} pooled predictions",
            )
            pooled[model] = (indices, canonical_targets, predictions)
            hosts = [sample_metadata[int(index)]["host"] for index in indices]
            dopants = [sample_metadata[int(index)]["dopant"] for index in indices]
            pooled_rows.append(
                {
                    "model": model, "regime": regime, "n": len(indices),
                    **regression_metrics(
                        canonical_targets, predictions, hosts, dopants
                    ),
                }
            )
        if "dart" not in pooled:
            continue
        reference_indices, reference_targets, reference_predictions = pooled["dart"]
        reference_hosts = [
            sample_metadata[int(index)]["host"] for index in reference_indices
        ]
        reference_dopants = [
            sample_metadata[int(index)]["dopant"] for index in reference_indices
        ]
        if regime == "host_cv":
            bootstrap_groups = reference_hosts
            bootstrap_unit = "host"
        elif regime == "dopant_cv":
            bootstrap_groups = reference_dopants
            bootstrap_unit = "dopant"
        elif regime in ("pair_cv", "chemistry_block"):
            bootstrap_groups = [
                f"{host}::{dopant}"
                for host, dopant in zip(reference_hosts, reference_dopants)
            ]
            bootstrap_unit = "host_dopant_pair"
        else:
            bootstrap_groups = None
            bootstrap_unit = "sample"
        for comparator_index, comparator in enumerate(models[1:]):
            if comparator not in pooled:
                continue
            indices, targets, predictions = pooled[comparator]
            if not np.array_equal(indices, reference_indices) or not np.allclose(
                targets, reference_targets, rtol=0.0, atol=1e-10,
            ):
                raise ValueError(f"pooled predictions do not align for {regime}/{comparator}")
            comparison_rows.append(
                {
                    "regime": regime, "reference": "dart", "comparator": comparator,
                    "bootstrap_unit": bootstrap_unit,
                    **paired_sample_comparison(
                        targets, reference_predictions, predictions,
                        seed=20265000 + 10 * regime_index + comparator_index,
                        groups=bootstrap_groups,
                    ),
                }
            )

    collector_git = git_snapshot()
    if not collector_git["commit"] or collector_git["dirty"]:
        raise ValueError("comparison collection requires a clean Git commit")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    archived_runs = []
    if not args.allow_incomplete:
        archive_manifests = comparison_archive_manifests(rows, result_root)
        if len(archive_manifests) != 91:
            raise ValueError(
                "expected 43 selected-DART transfer and 48 SchNet runs for archival, "
                f"found {len(archive_manifests)}"
            )
        archived_runs = archive_training_artifacts(
            archive_manifests,
            out_dir / "runs",
            repository_root=ROOT,
        )
    write_csv(out_dir / "run_metrics.csv", retained_rows)
    write_csv(out_dir / "fold_metrics.csv", fold_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "pooled_metrics.csv", pooled_rows)
    write_csv(out_dir / "paired_comparisons.csv", comparison_rows)
    descriptor_selection_path = out_dir / "descriptor_selection.json"
    descriptor_selection_path.write_text(
        json.dumps(
            descriptor_selection, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
    )
    output_sha256 = {
        name: file_sha256(out_dir / filename)
        for name, filename in {
            "run_metrics": "run_metrics.csv",
            "fold_metrics": "fold_metrics.csv",
            "summary": "summary.csv",
            "pooled_metrics": "pooled_metrics.csv",
            "paired_comparisons": "paired_comparisons.csv",
            "descriptor_selection": "descriptor_selection.json",
        }.items()
    }
    manifest = {
        "schema_version": "prm_comparison_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collector_git": collector_git,
        "selection": {
            "path": str(args.selection.resolve()), "sha256": file_sha256(args.selection),
            "selected_variant": variant, "selection_data": selection["selection_data"],
            "factorial_bundle_sha256": selection["factorial_bundle_sha256"],
            "factorial_training_commit": selection["training_commits"][0],
            "factorial_collector_git": factorial_bundle["collector_git"],
        },
        "data_sha256": protocol["data_sha256"],
        "descriptor_evidence": {
            "path": str(args.descriptor_evidence.resolve()),
            "sha256": file_sha256(args.descriptor_evidence),
            "collector_git": descriptor_evidence["collector_git"],
            "training_git": descriptor_evidence["descriptor_manifest"]["training_git"],
        },
        "n_run_rows": len(retained_rows), "n_fold_rows": len(fold_rows),
        "neural_campaign_commits": campaign_commits,
        "configuration_coverage": configuration_coverage,
        "archive_policy": {
            "scope": "43 selected-DART transfer runs and 48 SchNet runs; "
                     "selected factorial runs are bound through the factorial bundle",
            "included": [
                "run_manifest.json", "metrics.json", "split_indices.npz",
                "val_predictions.npz", "test_predictions.npz",
            ],
            "checkpoint": "SHA-256 recorded; binary retained outside Git",
        },
        "n_archived_runs": len(archived_runs),
        "archived_runs": archived_runs,
        "descriptor_selection": descriptor_selection,
        "output_sha256": output_sha256,
        "aggregation": {
            "fold_metrics": "mean over model seeds within each split",
            "pooled_metrics": "mean prediction over model seeds, then concatenate disjoint test folds",
            "paired_ci": (
                "paired nonparametric bootstrap over aligned samples for random CV; "
                "over held-out hosts, dopants, or host-dopant pairs for grouped regimes"
            ),
        },
    }
    encoded_manifest = json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False
    )
    (out_dir / "manifest.json").write_text(encoded_manifest + "\n")
    print(encoded_manifest)


if __name__ == "__main__":
    main()
