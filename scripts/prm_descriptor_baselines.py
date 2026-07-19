"""Validation-selected descriptor baselines under every PRM split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from ase.data import atomic_masses, covalent_radii
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prm_metrics import low_energy_metrics, macro_group_mae, regression_metrics
from src.defect_identity import unique_defect_index
from src.splits import load_split


ELECTRONEGATIVITY = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
    9: 3.98, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19,
    16: 2.58, 17: 3.16, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54,
    23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91,
    29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55,
    35: 2.96, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60,
    42: 2.16, 43: 1.90, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93,
    48: 1.69, 49: 1.78, 50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66,
    55: 0.79, 56: 0.89, 57: 1.10, 72: 1.30, 73: 1.50, 74: 2.36,
    75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00,
    81: 1.62, 82: 2.33, 83: 2.02,
}
SITE_NAMES = tuple([f"ads{i}" for i in range(6)] + [f"int{i}" for i in range(6)])
DEFECT_TYPES = ("adsorbate", "interstitial")


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


def defect_index(sample: Dict[str, Any]) -> int:
    return unique_defect_index(sample)


def property_values(numbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = numbers.astype(int)
    mass = np.asarray([atomic_masses[value] for value in z], dtype=float)
    radius = np.asarray([covalent_radii[value] for value in z], dtype=float)
    en = np.asarray([ELECTRONEGATIVITY.get(int(value), 2.0) for value in z], dtype=float)
    return z.astype(float), mass, radius, en


def summarize(values: np.ndarray) -> List[float]:
    return [
        float(values.mean()), float(values.std()), float(values.min()),
        float(values.max()), float(np.median(values)),
    ]


def featurize(sample: Dict[str, Any]) -> np.ndarray:
    numbers = np.asarray(sample["numbers"], dtype=int)
    cell = np.asarray(sample["cell"], dtype=float)
    distances = np.asarray(sample["dist_matrix"], dtype=float)
    meta = sample.get("metadata", {})
    index = defect_index(sample)
    host_numbers = np.delete(numbers, index)
    all_props = property_values(numbers)
    host_props = property_values(host_numbers)
    dopant_props = [props[index] for props in all_props]

    lengths = np.linalg.norm(cell, axis=1)
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    volume = abs(np.linalg.det(cell))
    mass_total = float(np.sum(all_props[1]))
    local = np.sort(distances[index][distances[index] > 1e-8])
    nearest = np.pad(local[:6], (0, max(0, 6 - len(local))), constant_values=20.0)

    features: List[float] = []
    for values in all_props:
        features.extend(summarize(values))
    for values in host_props:
        features.extend(summarize(values))
    features.extend(float(value) for value in dopant_props)
    features.extend(
        [
            float(dopant_props[i] - host_props[i].mean())
            for i in range(len(dopant_props))
        ]
    )
    features.extend(
        [
            float(len(numbers)), float(len(set(numbers.tolist()))),
            *[float(value) for value in lengths], float(area), float(volume),
            mass_total / max(volume, 1e-8),
            *[float(value) for value in nearest],
            float(np.sum(local < 3.0)), float(np.sum(local < 5.0)),
            float(local.mean()), float(local.std()),
        ]
    )
    features.extend(float(meta.get("site") == name) for name in SITE_NAMES)
    features.extend(float(meta.get("defecttype") == name) for name in DEFECT_TYPES)
    return np.asarray(features, dtype=np.float32)


def candidate_models(seed: int, n_jobs: int) -> Dict[str, List[Tuple[str, Any]]]:
    candidates: Dict[str, List[Tuple[str, Any]]] = {
        "ridge": [
            (f"alpha={alpha}", make_pipeline(StandardScaler(), Ridge(alpha=alpha)))
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0)
        ],
        "random_forest": [
            (
                f"max_features={max_features},min_samples_leaf={leaf}",
                RandomForestRegressor(
                    n_estimators=400, max_features=max_features,
                    min_samples_leaf=leaf, n_jobs=n_jobs, random_state=seed,
                ),
            )
            for max_features in (0.5, 1.0) for leaf in (1, 2)
        ],
        "hist_gradient_boosting": [
            (
                f"max_leaf_nodes={leaves},l2={l2}",
                HistGradientBoostingRegressor(
                    max_iter=400, learning_rate=0.05, max_leaf_nodes=leaves,
                    l2_regularization=l2, early_stopping=False, random_state=seed,
                ),
            )
            for leaves in (31, 63) for l2 in (0.0, 1.0)
        ],
    }
    try:
        from lightgbm import LGBMRegressor

        candidates["lightgbm"] = [
            (
                f"num_leaves={leaves},min_child_samples={child}",
                LGBMRegressor(
                    n_estimators=600, learning_rate=0.05, num_leaves=leaves,
                    min_child_samples=child, reg_lambda=1.0, subsample=0.9,
                    colsample_bytree=0.9, n_jobs=n_jobs, random_state=seed,
                    verbosity=-1,
                ),
            )
            for leaves in (31, 63) for child in (20, 50)
        ]
    except ImportError:
        pass
    return candidates


def prediction_summary(
    samples: Sequence[Dict[str, Any]], targets: np.ndarray,
    indices: Dict[str, np.ndarray], val_pred: np.ndarray, test_pred: np.ndarray,
) -> Dict[str, Any]:
    test_meta = [samples[i].get("metadata", {}) for i in indices["test"]]
    return {
        "validation": regression_metrics(targets[indices["val"]], val_pred),
        "test": regression_metrics(targets[indices["test"]], test_pred),
        "test_host_macro": macro_group_mae(
            targets[indices["test"]], test_pred,
            [meta.get("host", "") for meta in test_meta],
        ),
        "test_dopant_macro": macro_group_mae(
            targets[indices["test"]], test_pred,
            [meta.get("dopant", "") for meta in test_meta],
        ),
        "test_low_energy": low_energy_metrics(targets[indices["test"]], test_pred),
    }


def mean_baseline(
    samples: Sequence[Dict[str, Any]], targets: np.ndarray,
    indices: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    train_mean = float(targets[indices["train"]].mean())
    val_pred = np.full(len(indices["val"]), train_mean, dtype=float)
    test_pred = np.full(len(indices["test"]), train_mean, dtype=float)
    result = {
        "selected_by": "fixed non-tuned baseline",
        "selected_candidate": "training-target mean",
        "training_target_mean": train_mean,
        "candidates": [],
        **prediction_summary(samples, targets, indices, val_pred, test_pred),
    }
    return result, val_pred, test_pred


def evaluate_split(
    split_path: Path,
    samples: Sequence[Dict[str, Any]],
    features: np.ndarray,
    targets: np.ndarray,
    result_root: Path,
    seed: int,
    n_jobs: int,
) -> Dict[str, Any]:
    split = load_split(split_path, len(samples))
    split_id = split["split_id"]
    output_dir = result_root / "baselines" / "descriptors" / split_id
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = {name: np.asarray(split[name], dtype=int) for name in ("train", "val", "test")}
    results: Dict[str, Any] = {}
    val_predictions: Dict[str, np.ndarray] = {}
    test_predictions: Dict[str, np.ndarray] = {}
    mean_result, mean_val, mean_test = mean_baseline(samples, targets, indices)
    results["mean"] = mean_result
    val_predictions["mean"] = mean_val
    test_predictions["mean"] = mean_test

    for family, candidates in candidate_models(seed, n_jobs).items():
        candidate_rows = []
        selected = None
        selected_val_mae = float("inf")
        selected_val_pred = None
        for candidate_name, model in candidates:
            started = time.time()
            model.fit(features[indices["train"]], targets[indices["train"]])
            val_pred = model.predict(features[indices["val"]])
            val_metrics = regression_metrics(targets[indices["val"]], val_pred)
            candidate_rows.append(
                {
                    "candidate": candidate_name,
                    "validation": val_metrics,
                    "fit_seconds": time.time() - started,
                }
            )
            if val_metrics["mae"] < selected_val_mae:
                selected_val_mae = val_metrics["mae"]
                selected = (candidate_name, model)
                selected_val_pred = val_pred
        assert selected is not None and selected_val_pred is not None
        candidate_name, model = selected
        test_pred = model.predict(features[indices["test"]])
        val_predictions[family] = selected_val_pred
        test_predictions[family] = test_pred
        results[family] = {
            "selected_by": "minimum validation MAE",
            "selected_candidate": candidate_name,
            "candidates": candidate_rows,
            **prediction_summary(
                samples, targets, indices, selected_val_pred, test_pred,
            ),
        }

    model_names = sorted(results)
    np.savez_compressed(
        output_dir / "predictions.npz",
        schema_version=np.asarray("prm_descriptor_predictions_v1"),
        split_id=np.asarray(split_id),
        model_names=np.asarray(model_names),
        val_indices=indices["val"],
        val_targets=targets[indices["val"]],
        val_predictions=np.stack([val_predictions[name] for name in model_names]),
        test_indices=indices["test"],
        test_targets=targets[indices["test"]],
        test_predictions=np.stack([test_predictions[name] for name in model_names]),
    )
    payload = {
        "schema_version": "prm_descriptor_results_v1",
        "split_id": split_id,
        "split_path": str(split_path),
        "split_sha256": file_sha256(split_path),
        "selection_data": "validation only",
        "results": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _metric_blocks_match(
    recorded: Mapping[str, Any], recomputed: Mapping[str, Any],
) -> bool:
    for key in ("n", "mae", "rmse", "bias", "pearson", "spearman", "r2"):
        if key not in recorded or key not in recomputed:
            return False
        if key == "n":
            if int(recorded[key]) != int(recomputed[key]):
                return False
            continue
        left, right = float(recorded[key]), float(recomputed[key])
        if math.isnan(left) and math.isnan(right):
            continue
        if not math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-8):
            return False
    return True


def validate_descriptor_result(
    result_root: Path, split_id: str, split_path: Path,
) -> None:
    output_dir = result_root / "baselines" / "descriptors" / split_id
    metrics = json.loads((output_dir / "metrics.json").read_text())
    split = json.loads(split_path.read_text())
    if metrics.get("schema_version") != "prm_descriptor_results_v1":
        raise ValueError(f"unsupported descriptor metrics schema: {split_id}")
    if metrics.get("split_id") != split_id or split.get("split_id") != split_id:
        raise ValueError(f"descriptor split identity mismatch: {split_id}")
    if metrics.get("split_sha256") != file_sha256(split_path):
        raise ValueError(f"descriptor split hash mismatch: {split_id}")
    if metrics.get("selection_data") != "validation only":
        raise ValueError(f"descriptor selection used non-validation data: {split_id}")

    with np.load(output_dir / "predictions.npz", allow_pickle=False) as archive:
        expected_fields = {
            "schema_version", "split_id", "model_names",
            "val_indices", "val_targets", "val_predictions",
            "test_indices", "test_targets", "test_predictions",
        }
        if set(archive.files) != expected_fields:
            raise ValueError(f"descriptor prediction fields are incomplete: {split_id}")
        if str(archive["schema_version"].item()) != "prm_descriptor_predictions_v1":
            raise ValueError(f"unsupported descriptor prediction schema: {split_id}")
        if str(archive["split_id"].item()) != split_id:
            raise ValueError(f"descriptor prediction split mismatch: {split_id}")
        model_names = [str(name) for name in archive["model_names"].tolist()]
        val_indices = np.asarray(archive["val_indices"])
        val_targets = np.asarray(archive["val_targets"], dtype=float)
        val_predictions = np.asarray(archive["val_predictions"], dtype=float)
        test_indices = np.asarray(archive["test_indices"])
        test_targets = np.asarray(archive["test_targets"], dtype=float)
        test_predictions = np.asarray(archive["test_predictions"], dtype=float)

    if model_names != sorted(set(model_names)):
        raise ValueError(f"descriptor model names are not unique and sorted: {split_id}")
    if set(metrics.get("results", {})) != set(model_names):
        raise ValueError(f"descriptor metrics/model names differ: {split_id}")
    expected_val = np.asarray(split["val"], dtype=np.int64)
    expected_test = np.asarray(split["test"], dtype=np.int64)
    for name, observed, expected in (
        ("validation", val_indices, expected_val),
        ("test", test_indices, expected_test),
    ):
        if observed.ndim != 1 or not np.issubdtype(observed.dtype, np.integer):
            raise ValueError(f"invalid descriptor {name} indices: {split_id}")
        if not np.array_equal(observed.astype(np.int64, copy=False), expected):
            raise ValueError(f"descriptor {name} indices differ from split: {split_id}")
    if val_targets.shape != (len(expected_val),) or test_targets.shape != (len(expected_test),):
        raise ValueError(f"descriptor target vectors are misaligned: {split_id}")
    if val_predictions.shape != (len(model_names), len(expected_val)):
        raise ValueError(f"descriptor validation prediction shape mismatch: {split_id}")
    if test_predictions.shape != (len(model_names), len(expected_test)):
        raise ValueError(f"descriptor test prediction shape mismatch: {split_id}")
    if not all(
        np.isfinite(array).all()
        for array in (val_targets, test_targets, val_predictions, test_predictions)
    ):
        raise ValueError(f"descriptor predictions contain non-finite values: {split_id}")

    for index, family in enumerate(model_names):
        result = metrics["results"][family]
        validation = regression_metrics(val_targets, val_predictions[index])
        test = regression_metrics(test_targets, test_predictions[index])
        if not _metric_blocks_match(result.get("validation", {}), validation):
            raise ValueError(f"descriptor validation metrics mismatch: {split_id}/{family}")
        if not _metric_blocks_match(result.get("test", {}), test):
            raise ValueError(f"descriptor test metrics mismatch: {split_id}/{family}")
        candidates = result.get("candidates", [])
        if family == "mean":
            if candidates or result.get("selected_by") != "fixed non-tuned baseline":
                raise ValueError(f"mean descriptor baseline contract mismatch: {split_id}")
            continue
        if result.get("selected_by") != "minimum validation MAE" or not candidates:
            raise ValueError(f"descriptor selection contract mismatch: {split_id}/{family}")
        candidate_names = [str(row.get("candidate")) for row in candidates]
        if len(candidate_names) != len(set(candidate_names)):
            raise ValueError(f"duplicate descriptor candidates: {split_id}/{family}")
        selected = min(
            candidates,
            key=lambda row: float(row.get("validation", {}).get("mae", float("inf"))),
        )
        if result.get("selected_candidate") != selected.get("candidate"):
            raise ValueError(f"descriptor candidate selection mismatch: {split_id}/{family}")


def result_is_complete(
    result_root: Path, split_id: str, split_path: Path | None = None,
) -> bool:
    if split_path is None:
        return False
    try:
        validate_descriptor_result(result_root, split_id, split_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def ensure_mean_baseline(
    split_path: Path, samples: Sequence[Dict[str, Any]], targets: np.ndarray,
    result_root: Path,
) -> bool:
    split = load_split(split_path, len(samples))
    split_id = split["split_id"]
    output_dir = result_root / "baselines" / "descriptors" / split_id
    if not result_is_complete(result_root, split_id, split_path):
        return False
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.npz"
    payload = json.loads(metrics_path.read_text())
    with np.load(predictions_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    model_names = [str(name) for name in arrays["model_names"].tolist()]
    if "mean" in payload.get("results", {}) and "mean" in model_names:
        return False

    indices = {name: np.asarray(split[name], dtype=int) for name in ("train", "val", "test")}
    result, val_pred, test_pred = mean_baseline(samples, targets, indices)
    val_by_model = {
        name: arrays["val_predictions"][index]
        for index, name in enumerate(model_names)
    }
    test_by_model = {
        name: arrays["test_predictions"][index]
        for index, name in enumerate(model_names)
    }
    val_by_model["mean"] = val_pred
    test_by_model["mean"] = test_pred
    payload.setdefault("results", {})["mean"] = result
    sorted_names = sorted(val_by_model)
    arrays["model_names"] = np.asarray(sorted_names)
    arrays["val_predictions"] = np.stack([val_by_model[name] for name in sorted_names])
    arrays["test_predictions"] = np.stack([test_by_model[name] for name in sorted_names])

    temporary_predictions = predictions_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_predictions, **arrays)
    temporary_predictions.replace(predictions_path)
    temporary_metrics = metrics_path.with_suffix(".tmp.json")
    temporary_metrics.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_metrics.replace(metrics_path)
    return True


def prior_split_provenance(manifest: Dict[str, Any]) -> Dict[str, Any]:
    provenance = dict(manifest.get("split_provenance", {}))
    if provenance or not manifest:
        return provenance
    batch = {
        "git": manifest.get("git"),
        "started_at": manifest.get("started_at"),
        "completed_at": manifest.get("completed_at"),
    }
    return {split_id: batch for split_id in manifest.get("splits", [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v2")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--split-glob", action="append", dest="split_globs",
        help="Repeat to select multiple split patterns; defaults to all splits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started = time.time()
    result_root = args.result_root.resolve()
    output_dir = result_root / "baselines" / "descriptors"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    try:
        previous_manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        previous_manifest = {}

    protocol_manifest_path = args.protocol_dir / "manifest.json"
    protocol_manifest = json.loads(protocol_manifest_path.read_text())
    expected_data_sha256 = protocol_manifest["data_sha256"]
    observed_data_sha256 = file_sha256(args.data)
    if observed_data_sha256 != expected_data_sha256:
        raise ValueError(
            "descriptor input dataset does not match the frozen protocol: "
            f"expected {expected_data_sha256}, observed {observed_data_sha256}"
        )
    with args.data.open("rb") as handle:
        blob = pickle.load(handle)
    samples = blob["data"] if isinstance(blob, dict) and "data" in blob else blob
    targets = np.asarray([float(sample["target"]) for sample in samples], dtype=float)

    audit_path = args.protocol_dir / protocol_manifest["data_audit"]
    audit = json.loads(audit_path.read_text())
    excluded_indices = np.asarray(
        audit["duplicates"]["canonical_deduplication"]["excluded_indices"],
        dtype=np.int64,
    )
    excluded_set = set(excluded_indices.tolist())
    retained_indices = np.asarray(
        [index for index in range(len(samples)) if index not in excluded_set],
        dtype=np.int64,
    )
    print(
        f"Featurizing {len(retained_indices)} canonical rows "
        f"({len(excluded_indices)} excluded) ...",
        flush=True,
    )
    retained_features = np.stack([featurize(samples[index]) for index in retained_indices])
    features = np.zeros(
        (len(samples), retained_features.shape[1]), dtype=retained_features.dtype,
    )
    features[retained_indices] = retained_features
    feature_digest = hashlib.sha256()
    feature_digest.update(retained_indices.tobytes())
    feature_digest.update(retained_features.tobytes())
    feature_hash = feature_digest.hexdigest()

    data_sha256 = observed_data_sha256
    formal_paths = [
        path for path in sorted((args.protocol_dir / "splits").glob("*.json"))
        if path.stem not in ("id_historical_s42", "smoke_protocol")
    ]
    for path in formal_paths:
        split = json.loads(path.read_text())
        if set(split.get("excluded", ())) != excluded_set:
            raise ValueError(f"formal split exclusions differ from protocol audit: {path.stem}")
    selected = {
        path.resolve()
        for pattern in (args.split_globs or ["*.json"])
        for path in (args.protocol_dir / "splits").glob(pattern)
    }
    split_paths = [path for path in formal_paths if path.resolve() in selected]
    if not split_paths:
        raise SystemExit("no formal protocol splits matched --split-glob")

    reuse_compatible = (
        previous_manifest.get("data_sha256") == data_sha256
        and previous_manifest.get("feature_matrix_sha256") == feature_hash
    )
    evaluated = []
    reused = []
    upgraded = []
    for path in split_paths:
        complete = result_is_complete(result_root, path.stem, path)
        if complete and reuse_compatible and not args.force:
            print(f"[{path.stem}] validating reusable result", flush=True)
            if ensure_mean_baseline(path, samples, targets, result_root):
                upgraded.append(path.stem)
            reused.append(path.stem)
            continue
        print(f"[{path.stem}] fitting descriptor candidates", flush=True)
        evaluate_split(
            path, samples, features, targets, result_root, args.seed, args.n_jobs,
        )
        print(f"[{path.stem}] complete", flush=True)
        evaluated.append(path.stem)

    git = git_snapshot()
    completed_at = datetime.now(timezone.utc).isoformat()
    batch = {
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "completed_at": completed_at,
        "git": git,
        "requested_splits": [path.stem for path in split_paths],
        "evaluated_splits": evaluated,
        "reused_splits": reused,
        "upgraded_splits": upgraded,
        "wall_seconds": time.time() - started,
    }
    batches = list(previous_manifest.get("batches", []))
    if previous_manifest and not batches:
        batches.append(
            {
                "started_at": previous_manifest.get("started_at"),
                "completed_at": previous_manifest.get("completed_at"),
                "git": previous_manifest.get("git"),
                "requested_splits": previous_manifest.get("splits", []),
                "evaluated_splits": previous_manifest.get("splits", []),
                "reused_splits": [],
                "upgraded_splits": [],
                "wall_seconds": previous_manifest.get("wall_seconds"),
            }
        )
    batches.append(batch)
    split_provenance = prior_split_provenance(previous_manifest)
    for split_id in evaluated + upgraded:
        split_provenance[split_id] = {
            "git": git, "started_at": batch["started_at"],
            "completed_at": completed_at,
        }
    complete_splits = [
        path.stem for path in formal_paths
        if result_is_complete(result_root, path.stem, path)
    ]
    split_artifacts = {
        split_id: {
            "metrics_sha256": file_sha256(
                output_dir / split_id / "metrics.json"
            ),
            "predictions_sha256": file_sha256(
                output_dir / split_id / "predictions.npz"
            ),
        }
        for split_id in complete_splits
    }
    requested_complete = all(
        result_is_complete(result_root, path.stem, path) for path in split_paths
    )
    manifest = {
        "schema_version": "prm_descriptor_manifest_v2",
        "status": "complete" if requested_complete else "incomplete",
        "started_at": batch["started_at"],
        "completed_at": completed_at,
        "git": git,
        "environment": {
            "hostname": socket.gethostname(), "python": sys.version,
            "platform": platform.platform(),
        },
        "data_path": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "data_file_sha256": observed_data_sha256,
        "protocol_manifest_sha256": file_sha256(args.protocol_dir / "manifest.json"),
        "feature_matrix_sha256": feature_hash,
        "feature_matrix_scope": "canonical retained rows only, hashed with sample indices",
        "n_featurized_samples": len(retained_indices),
        "n_samples": len(samples),
        "n_features": int(features.shape[1]),
        "selection_data": "validation only",
        "splits": complete_splits,
        "requested_splits": batch["requested_splits"],
        "evaluated_splits": evaluated,
        "reused_splits": reused,
        "upgraded_splits": upgraded,
        "formal_split_coverage": {
            "complete": len(complete_splits),
            "total": len(formal_paths),
            "all_complete": len(complete_splits) == len(formal_paths),
        },
        "split_provenance": split_provenance,
        "split_artifacts": split_artifacts,
        "batches": batches,
        "wall_seconds": batch["wall_seconds"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
