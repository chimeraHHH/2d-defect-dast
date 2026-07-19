"""Validation-selected descriptor baselines under every PRM split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prm_metrics import low_energy_metrics, macro_group_mae, regression_metrics
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
SITE_NAMES = tuple([f"ads{i}" for i in range(7)] + [f"int{i}" for i in range(5)])
DEFECT_TYPES = ("adsorbate", "interstitial")


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
    numbers = np.asarray(sample["numbers"], dtype=int)
    dopant = sample.get("metadata", {}).get("dopant", "")
    z = atomic_numbers.get(str(dopant))
    matches = np.flatnonzero(numbers == z) if z is not None else np.array([], dtype=int)
    return int(matches[-1]) if len(matches) else len(numbers) - 1


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
        test_meta = [samples[i].get("metadata", {}) for i in indices["test"]]
        results[family] = {
            "selected_by": "minimum validation MAE",
            "selected_candidate": candidate_name,
            "candidates": candidate_rows,
            "validation": regression_metrics(targets[indices["val"]], selected_val_pred),
            "test": regression_metrics(targets[indices["test"]], test_pred),
            "test_host_macro": macro_group_mae(
                targets[indices["test"]], test_pred, [meta.get("host", "") for meta in test_meta]
            ),
            "test_dopant_macro": macro_group_mae(
                targets[indices["test"]], test_pred, [meta.get("dopant", "") for meta in test_meta]
            ),
            "test_low_energy": low_energy_metrics(targets[indices["test"]], test_pred),
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
        "selection_data": "validation only",
        "results": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v1")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--split-glob", default="*.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    with args.data.open("rb") as handle:
        blob = pickle.load(handle)
    samples = blob["data"] if isinstance(blob, dict) and "data" in blob else blob
    targets = np.asarray([float(sample["target"]) for sample in samples], dtype=float)
    features = np.stack([featurize(sample) for sample in samples])
    feature_hash = hashlib.sha256(features.tobytes()).hexdigest()

    split_paths = sorted((args.protocol_dir / "splits").glob(args.split_glob))
    split_paths = [
        path for path in split_paths
        if path.stem not in ("id_historical_s42", "smoke_protocol")
    ]
    outputs = [
        evaluate_split(
            path, samples, features, targets, args.result_root.resolve(),
            args.seed, args.n_jobs,
        )
        for path in split_paths
    ]
    manifest = {
        "schema_version": "prm_descriptor_manifest_v1",
        "status": "complete",
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "git": git_snapshot(),
        "environment": {
            "hostname": socket.gethostname(), "python": sys.version,
            "platform": platform.platform(),
        },
        "data_path": str(args.data.resolve()),
        "data_sha256": json.loads((args.protocol_dir / "manifest.json").read_text())["data_sha256"],
        "feature_matrix_sha256": feature_hash,
        "n_samples": len(samples),
        "n_features": int(features.shape[1]),
        "selection_data": "validation only",
        "splits": [output["split_id"] for output in outputs],
        "wall_seconds": time.time() - started,
    }
    output_dir = args.result_root.resolve() / "baselines" / "descriptors"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
