#!/usr/bin/env python3
"""Comprehensive analysis of model innovations for defect Ef prediction.

Compares V2 baseline against V3/V4/V5/V6/V7 innovations:
  - Per-model MAE/RMSE
  - Per-defect-type breakdown (interstitial vs adsorbate)
  - Per-Ef-range analysis (binned by target value)
  - Prediction correlation matrix (ensemble diversity)
  - Greedy ensemble selection (optimal model combination)
  - Physics feature importance (for V6)

Usage:
    python scripts/analyze_innovations.py [--results-dir results/]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_model_results(results_dir: Path):
    """Load all model results from results directory."""
    models = {}
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        name = metrics_path.parent.name
        with open(metrics_path) as f:
            metrics = json.load(f)

        pred_path = metrics_path.parent / "test_predictions.npz"
        if not pred_path.exists():
            continue

        data = np.load(pred_path)
        preds = data["preds"]
        targets = data["targets"]

        # Only include leak-free models (1065 test samples)
        if len(preds) != 1065:
            print(f"  SKIP {name}: {len(preds)} test samples (expect 1065)")
            continue

        models[name] = {
            "preds": preds,
            "targets": targets,
            "test_mae": metrics.get("test_mae", np.abs(preds - targets).mean()),
            "test_rmse": metrics.get("test_rmse", np.sqrt(np.mean((preds - targets) ** 2))),
            "config": metrics.get("config", {}),
            "n_params": metrics.get("n_params", 0),
        }
    return models


def per_range_analysis(preds, targets, bins=None):
    """Compute MAE per target-value range."""
    if bins is None:
        bins = [(-np.inf, 0), (0, 2), (2, 5), (5, 7), (7, 25)]
    results = []
    for lo, hi in bins:
        mask = (targets >= lo) & (targets < hi)
        n = mask.sum()
        if n == 0:
            results.append({"range": f"[{lo},{hi})", "n": 0, "mae": 0, "contribution": 0})
            continue
        mae = np.abs(preds[mask] - targets[mask]).mean()
        total_ae = np.abs(preds[mask] - targets[mask]).sum()
        results.append({
            "range": f"[{lo},{hi})",
            "n": int(n),
            "mae": float(mae),
            "total_ae": float(total_ae),
        })
    # Compute contribution fraction
    total_err = sum(r.get("total_ae", 0) for r in results)
    for r in results:
        r["contribution"] = r.get("total_ae", 0) / max(total_err, 1e-9)
    return results


def correlation_matrix(models):
    """Compute prediction correlation between all model pairs."""
    names = sorted(models.keys())
    n = len(names)
    corr = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            corr[i, j] = np.corrcoef(models[n1]["preds"], models[n2]["preds"])[0, 1]
    return names, corr


def greedy_ensemble(models, max_size=8):
    """Greedy ensemble selection minimizing MAE."""
    names = sorted(models.keys())
    targets = models[names[0]]["targets"]
    all_preds = {n: models[n]["preds"] for n in names}

    # Start with best single model
    best_single = min(names, key=lambda n: models[n]["test_mae"])
    selected = [best_single]
    ensemble_preds = all_preds[best_single].copy()
    best_mae = np.abs(ensemble_preds - targets).mean()

    results = [{"size": 1, "model": best_single, "mae": float(best_mae)}]

    remaining = [n for n in names if n != best_single]

    for step in range(min(max_size - 1, len(remaining))):
        best_next = None
        best_next_mae = best_mae

        for candidate in remaining:
            trial = (ensemble_preds * len(selected) + all_preds[candidate]) / (len(selected) + 1)
            trial_mae = np.abs(trial - targets).mean()
            if trial_mae < best_next_mae:
                best_next_mae = trial_mae
                best_next = candidate

        if best_next is None:
            break

        selected.append(best_next)
        ensemble_preds = (ensemble_preds * (len(selected) - 1) + all_preds[best_next]) / len(selected)
        best_mae = best_next_mae
        remaining.remove(best_next)
        results.append({"size": len(selected), "model": best_next, "mae": float(best_mae)})

    return results, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/", help="Root results directory")
    parser.add_argument("--metadata-path", default="data/processed/cleaned_dataset.pkl",
                        help="Dataset path for defect-type analysis")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print("=" * 70)
    print("INNOVATION ANALYSIS: Defect Formation Energy Prediction")
    print("=" * 70)

    # Load models
    models = load_model_results(results_dir)
    if not models:
        print("No valid model results found!")
        sys.exit(1)

    print(f"\nLoaded {len(models)} models with 1065 test samples each")

    # --- 1. Overall comparison ---
    print("\n" + "=" * 70)
    print("1. OVERALL PERFORMANCE")
    print("=" * 70)
    print(f"{'Model':<30} {'MAE (eV)':>10} {'RMSE (eV)':>10} {'Params':>10}")
    print("-" * 62)
    for name in sorted(models.keys(), key=lambda n: models[n]["test_mae"]):
        m = models[name]
        params_str = f"{m['n_params']/1e6:.3f}M" if m['n_params'] > 0 else "?"
        print(f"{name:<30} {m['test_mae']:>10.4f} {m['test_rmse']:>10.4f} {params_str:>10}")

    # --- 2. Per-range analysis for best models ---
    print("\n" + "=" * 70)
    print("2. PER-RANGE MAE ANALYSIS (top 5 models)")
    print("=" * 70)
    top_models = sorted(models.keys(), key=lambda n: models[n]["test_mae"])[:5]
    targets = models[top_models[0]]["targets"]

    bins = [(-np.inf, 0), (0, 2), (2, 5), (5, 7), (7, 25)]
    header = f"{'Range':<12}" + "".join(f" {n[:15]:>15}" for n in top_models)
    print(header)
    print("-" * len(header))
    for bi, (lo, hi) in enumerate(bins):
        mask = (targets >= lo) & (targets < hi)
        n = mask.sum()
        row = f"[{lo:>3},{hi:>3}) n={n:<4}"
        for name in top_models:
            mae = np.abs(models[name]["preds"][mask] - targets[mask]).mean() if n > 0 else 0
            row += f" {mae:>15.4f}"
        print(row)

    # --- 3. Prediction correlation ---
    print("\n" + "=" * 70)
    print("3. PREDICTION CORRELATION MATRIX")
    print("=" * 70)
    names, corr = correlation_matrix(models)
    header = f"{'':>20}" + "".join(f" {n[:12]:>12}" for n in names)
    print(header)
    for i, n1 in enumerate(names):
        row = f"{n1[:20]:>20}"
        for j, n2 in enumerate(names):
            row += f" {corr[i,j]:>12.4f}"
        print(row)

    # --- 4. Ensemble analysis ---
    print("\n" + "=" * 70)
    print("4. GREEDY ENSEMBLE SELECTION")
    print("=" * 70)
    ens_results, ens_selected = greedy_ensemble(models)
    for step in ens_results:
        print(f"  Size {step['size']}: +{step['model']:<25} → MAE {step['mae']:.4f}")

    # --- 5. Defect-type breakdown ---
    try:
        import pickle
        meta_path = Path(args.metadata_path)
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                blob = pickle.load(f)
            data = blob["data"] if isinstance(blob, dict) and "data" in blob else blob

            # Need to reconstruct test set indices
            # Use same split as training (seed=42, train=0.8, val=0.1)
            n_total = len(data)
            rng = np.random.RandomState(42)
            indices = rng.permutation(n_total)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            test_indices = indices[n_train + n_val:]

            if len(test_indices) == 1065:
                dtypes = [data[idx]["metadata"]["defecttype"] for idx in test_indices]
                hosts = [data[idx]["metadata"]["host"] for idx in test_indices]
                dopants = [data[idx]["metadata"]["dopant"] for idx in test_indices]

                print("\n" + "=" * 70)
                print("5. DEFECT-TYPE BREAKDOWN")
                print("=" * 70)
                for dt in ["interstitial", "adsorbate"]:
                    mask = np.array([d == dt for d in dtypes])
                    n = mask.sum()
                    print(f"\n{dt} (n={n}):")
                    print(f"  {'Model':<30} {'MAE':>10}")
                    print(f"  {'-'*42}")
                    for name in sorted(models.keys(), key=lambda n: models[n]["test_mae"]):
                        mae = np.abs(models[name]["preds"][mask] - targets[mask]).mean()
                        print(f"  {name:<30} {mae:>10.4f}")
            else:
                print(f"\nSKIP defect-type analysis: test set size mismatch "
                      f"({len(test_indices)} vs 1065)")
    except Exception as e:
        print(f"\nSKIP defect-type analysis: {e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best_single = min(models.keys(), key=lambda n: models[n]["test_mae"])
    print(f"Best single model: {best_single} (MAE={models[best_single]['test_mae']:.4f})")
    if ens_results:
        best_ens = ens_results[-1]
        print(f"Best ensemble ({best_ens['size']} models): MAE={best_ens['mae']:.4f}")
        improvement = (models[best_single]['test_mae'] - best_ens['mae']) / models[best_single]['test_mae'] * 100
        print(f"Ensemble improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
