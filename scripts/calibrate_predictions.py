#!/usr/bin/env python3
"""Post-hoc calibration: fit slope/bias correction on val set, apply to test.

Approach: isotonic regression or piecewise-linear calibration on validation
predictions, then apply the calibration to test predictions.

This corrects systematic biases (e.g., underprediction of extreme Ef values)
without retraining the model.

Usage:
    python scripts/calibrate_predictions.py \
        --results-dir results/v2_gated_pooling_s43
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_val_predictions(results_dir, data_path):
    """Load model and re-run on val set to get val predictions."""
    import torch
    from torch.utils.data import DataLoader
    from src.dataset import CrystalGraphDataset, collate_fn, make_splits
    from src.train_enhanced import Normalizer, move_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(results_dir) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt.get("config", {})
    model_type = config.get("model", "CrystalTransformer")

    # Build model
    if "model" in ckpt and isinstance(ckpt["model"], dict) and "config" in ckpt:
        state_dict = ckpt["model"]
        norm_data = ckpt.get("normalizer", {})
        mean = norm_data.get("mean", 0.0)
        std = norm_data.get("std", 1.0)
        transform = norm_data.get("transform", "none")
    else:
        state_dict = ckpt.get("swa_model_state") or ckpt.get("model_state")
        mean = ckpt.get("normalizer_mean", 0.0)
        std = ckpt.get("normalizer_std", 1.0)
        transform = "none"

    is_v2 = model_type in ("v2", "CrystalTransformerV2")
    if is_v2:
        from src.models.crystal_v2 import CrystalTransformerV2
        model = CrystalTransformerV2(**config.get("model_kwargs", {}))
    else:
        from src.models.baseline import CrystalTransformer
        model = CrystalTransformer(**config.get("model_kwargs", {}))

    model.load_state_dict(state_dict)
    model.to(device).eval()

    normalizer = Normalizer(torch.tensor([mean]), transform=transform)
    normalizer.mean = mean
    normalizer.std = std

    # Load dataset and get val split
    asph_path = config.get("asph_features_path")
    if asph_path:
        asph_path = ROOT / asph_path
    dataset = CrystalGraphDataset(ROOT / data_path, asph_features_path=asph_path)
    _, val_set, _ = make_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42)

    val_loader = DataLoader(val_set, batch_size=128, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch(batch, device)
            preds_norm = model(batch)
            preds = normalizer.denorm(preds_norm)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["target"].cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def piecewise_linear_calibrate(val_preds, val_targets, test_preds, n_bins=10):
    """Fit piecewise-linear calibration on val, apply to test."""
    # Sort by prediction
    sorted_idx = np.argsort(val_preds)
    sorted_preds = val_preds[sorted_idx]
    sorted_targets = val_targets[sorted_idx]

    # Create bins
    bin_edges = np.percentile(sorted_preds, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Compute bin-wise correction (average bias)
    corrections = []
    bin_centers = []
    for i in range(n_bins):
        mask = (val_preds >= bin_edges[i]) & (val_preds < bin_edges[i + 1])
        if mask.sum() > 0:
            avg_pred = val_preds[mask].mean()
            avg_target = val_targets[mask].mean()
            corrections.append(avg_target - avg_pred)
            bin_centers.append(avg_pred)

    if len(bin_centers) < 2:
        return test_preds

    # Interpolate corrections for test predictions
    corrections = np.array(corrections)
    bin_centers = np.array(bin_centers)
    test_corrections = np.interp(test_preds, bin_centers, corrections)

    return test_preds + test_corrections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", nargs="+", required=True,
                        help="Result directories with best.pt and test_predictions.npz")
    parser.add_argument("--data-path", default="data/processed/cleaned_dataset.pkl")
    parser.add_argument("--method", choices=["isotonic", "piecewise", "both"],
                        default="both")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    for results_dir in args.results_dir:
        results_dir = ROOT / results_dir
        name = results_dir.name
        print(f"\n{'='*60}")
        print(f"Calibrating: {name}")
        print(f"{'='*60}")

        # Load test predictions
        test_npz = np.load(results_dir / "test_predictions.npz")
        test_preds = test_npz["preds"]
        test_targets = test_npz["targets"]
        uncalib_mae = np.mean(np.abs(test_preds - test_targets))
        print(f"  Uncalibrated test MAE: {uncalib_mae:.4f}")

        # Get val predictions (requires running model)
        try:
            val_preds, val_targets = load_val_predictions(results_dir, args.data_path)
            print(f"  Val set: {len(val_preds)} samples")
            val_mae = np.mean(np.abs(val_preds - val_targets))
            print(f"  Val MAE: {val_mae:.4f}")
        except Exception as e:
            print(f"  Error loading val predictions: {e}")
            continue

        if args.method in ("isotonic", "both"):
            # Isotonic regression
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(val_preds, val_targets)
            cal_test = ir.predict(test_preds)
            cal_mae = np.mean(np.abs(cal_test - test_targets))
            improvement = (uncalib_mae - cal_mae) / uncalib_mae * 100
            print(f"  Isotonic calibrated MAE: {cal_mae:.4f} ({improvement:+.1f}%)")

        if args.method in ("piecewise", "both"):
            # Piecewise linear
            for nb in [5, 10, 20]:
                cal_test = piecewise_linear_calibrate(
                    val_preds, val_targets, test_preds, n_bins=nb
                )
                cal_mae = np.mean(np.abs(cal_test - test_targets))
                improvement = (uncalib_mae - cal_mae) / uncalib_mae * 100
                print(f"  Piecewise-{nb} calibrated MAE: {cal_mae:.4f} ({improvement:+.1f}%)")


if __name__ == "__main__":
    main()
