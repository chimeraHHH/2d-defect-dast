#!/usr/bin/env python3
"""Evaluate a new model's test predictions and compare with existing models.

Usage:
    python scripts/eval_new_model.py results/v2_logtarget/test_predictions.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_predictions(path):
    data = np.load(path)
    return data["preds"], data["targets"]


def analyze_model(preds, targets, name="Model"):
    errors = preds - targets
    abs_errors = np.abs(errors)
    mae = abs_errors.mean()
    rmse = np.sqrt((errors**2).mean())

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Overall: MAE={mae:.4f} eV, RMSE={rmse:.4f} eV")
    print(f"  Median AE: {np.median(abs_errors):.4f} eV")
    print(f"  Max AE: {abs_errors.max():.4f} eV (target={targets[abs_errors.argmax()]:.2f})")

    # Error by target range
    print(f"\n  {'Range':>16s}  {'N':>5s}  {'MAE':>7s}  {'RMSE':>7s}  {'Med':>7s}")
    print(f"  {'-'*48}")
    ranges = [(-15, -3), (-3, 0), (0, 3), (3, 7), (7, 25)]
    for lo, hi in ranges:
        mask = (targets >= lo) & (targets < hi)
        n = mask.sum()
        if n > 0:
            m = abs_errors[mask].mean()
            r = np.sqrt((errors[mask]**2).mean())
            med = np.median(abs_errors[mask])
            print(f"  [{lo:6.1f}, {hi:5.1f})  {n:5d}  {m:7.4f}  {r:7.4f}  {med:7.4f}")

    # Top/bottom error contribution
    sorted_ae = np.sort(abs_errors)[::-1]
    n10 = max(1, len(sorted_ae) // 10)
    top10_pct = sorted_ae[:n10].sum() / sorted_ae.sum() * 100
    print(f"\n  Top 10% worst → {top10_pct:.1f}% of total error")

    return mae, rmse, preds


def compare_with_existing(new_preds, targets, new_name):
    """Compare new model with existing best models."""
    ref_models = {
        "v2_gated_s43": "results/v2_gated_pooling_s43/test_predictions.npz",
        "v2_gated_s42": "results/v2_gated_pooling/test_predictions.npz",
        "v2_enhanced":  "results/v2_enhanced_env/test_predictions.npz",
        "multi_src_v4": "results/multi_source_v4_s42/test_predictions.npz",
        "v2_long250":   "results/v2_long250_s43/test_predictions.npz",
        "v2_distill":   "results/v2_distill/test_predictions.npz",
    }

    print(f"\n{'='*60}")
    print(f"  Prediction Correlation with Existing Models")
    print(f"{'='*60}")

    all_preds = {"_NEW_": new_preds}
    for name, path in ref_models.items():
        p = ROOT / path
        if p.exists():
            data = np.load(p)
            all_preds[name] = data["preds"]
            r = np.corrcoef(new_preds, data["preds"])[0, 1]
            ref_mae = np.mean(np.abs(data["preds"] - targets))
            print(f"  {new_name} vs {name:20s}: r={r:.4f} (ref MAE={ref_mae:.4f})")

    # Greedy ensemble: add new model to existing best ensemble
    print(f"\n{'='*60}")
    print(f"  Ensemble Analysis (greedy selection)")
    print(f"{'='*60}")

    eligible = {k: v for k, v in all_preds.items() if k != "_NEW_"}
    eligible["_NEW_"] = new_preds

    # Start with best single model
    best_single = min(eligible.keys(),
                      key=lambda k: np.mean(np.abs(eligible[k] - targets)))
    print(f"  Best single: {best_single} (MAE={np.mean(np.abs(eligible[best_single] - targets)):.4f})")

    # Greedy add
    selected = [best_single]
    for step in range(min(7, len(eligible) - 1)):
        best_add = None
        best_mae = float("inf")
        current_preds = np.mean([eligible[s] for s in selected], axis=0)
        for k in eligible:
            if k in selected:
                continue
            candidate = (current_preds * len(selected) + eligible[k]) / (len(selected) + 1)
            mae = np.mean(np.abs(candidate - targets))
            if mae < best_mae:
                best_mae = mae
                best_add = k
        if best_add is None:
            break
        selected.append(best_add)
        ens_pred = np.mean([eligible[s] for s in selected], axis=0)
        ens_mae = np.mean(np.abs(ens_pred - targets))
        marker = " ← NEW" if best_add == "_NEW_" else ""
        print(f"  +{best_add:20s} → {len(selected)}-ens MAE={ens_mae:.4f}{marker}")

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", help="Path to test_predictions.npz")
    parser.add_argument("--name", default=None, help="Model name for display")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    name = args.name or npz_path.parent.name

    preds, targets = load_predictions(npz_path)
    mae, rmse, preds = analyze_model(preds, targets, name)
    compare_with_existing(preds, targets, name)


if __name__ == "__main__":
    main()
