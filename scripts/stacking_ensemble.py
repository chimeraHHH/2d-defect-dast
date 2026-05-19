#!/usr/bin/env python3
"""Stacking ensemble: learn optimal model combination weights.

Instead of equal-weight averaging, train a Ridge regression on validation
set predictions to find the best linear combination. This can assign higher
weight to more accurate models and exploit complementary error patterns.

Usage:
    python scripts/stacking_ensemble.py \
        --checkpoints results/v2_gated_pooling/best.pt \
                      results/v2_gated_pooling_s43/best.pt \
                      ...
        --data-path data/processed/cleaned_dataset.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_test_predictions(results_dirs):
    """Load test predictions from npz files."""
    models = {}
    targets = None
    for d in results_dirs:
        p = ROOT / d / "test_predictions.npz"
        if not p.exists():
            print(f"  SKIP {d}: no test_predictions.npz")
            continue
        data = np.load(p)
        preds = data["preds"]
        t = data["targets"]
        if targets is None:
            targets = t
        elif len(t) != len(targets):
            print(f"  SKIP {d}: {len(t)} vs {len(targets)} samples")
            continue
        name = Path(d).name
        models[name] = preds
    return models, targets


def stacking_ensemble(val_preds_dict, val_targets, test_preds_dict, alpha_range=None):
    """Learn stacking weights on val set, apply to test set.

    Uses Ridge regression with cross-validation for regularization.
    """
    names = sorted(val_preds_dict.keys())
    X_val = np.column_stack([val_preds_dict[n] for n in names])
    y_val = val_targets

    X_test = np.column_stack([test_preds_dict[n] for n in names])

    if alpha_range is None:
        alpha_range = np.logspace(-4, 4, 50)

    # Cross-validated Ridge
    ridge_cv = RidgeCV(alphas=alpha_range, cv=5, scoring="neg_mean_absolute_error")
    ridge_cv.fit(X_val, y_val)

    # In-sample fit quality
    val_pred_stacked = ridge_cv.predict(X_val)
    val_mae = np.mean(np.abs(val_pred_stacked - y_val))

    # Test prediction
    test_pred_stacked = ridge_cv.predict(X_test)

    # Weights (normalized)
    coefs = ridge_cv.coef_
    intercept = ridge_cv.intercept_

    return {
        "test_preds": test_pred_stacked,
        "val_mae": val_mae,
        "alpha": float(ridge_cv.alpha_),
        "weights": {name: float(c) for name, c in zip(names, coefs)},
        "intercept": float(intercept),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dirs", nargs="+", required=True,
                        help="Result directories with test_predictions.npz")
    args = parser.parse_args()

    # Load test predictions
    models, targets = load_test_predictions(args.results_dirs)
    print(f"Loaded {len(models)} models, {len(targets)} test samples")

    # Individual MAEs
    print("\nIndividual MAEs:")
    for name in sorted(models.keys()):
        mae = np.mean(np.abs(models[name] - targets))
        print(f"  {name:40s}: {mae:.4f}")

    # Simple average ensemble
    all_preds = np.array([models[n] for n in sorted(models.keys())])
    avg_pred = all_preds.mean(axis=0)
    avg_mae = np.mean(np.abs(avg_pred - targets))
    print(f"\nEqual-weight ensemble: MAE={avg_mae:.4f}")

    # Since we can't easily get val predictions from checkpoints here,
    # use leave-one-out on test set as a proxy (for analysis only)
    # In practice, you'd use val set predictions

    # Cross-validated stacking on test set (for analysis)
    print("\nStacking (5-fold cross-validated on test set):")
    names = sorted(models.keys())
    X = np.column_stack([models[n] for n in names])

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        cv_preds = cross_val_predict(ridge, X, targets, cv=5)
        cv_mae = np.mean(np.abs(cv_preds - targets))

        # Also fit on full data to see weights
        ridge.fit(X, targets)

        print(f"  alpha={alpha:6.2f}: CV MAE={cv_mae:.4f}", end="")
        # Show top 3 weights
        coefs = ridge.coef_
        top_idx = np.argsort(np.abs(coefs))[::-1][:3]
        for i in top_idx:
            print(f"  {names[i][:20]}={coefs[i]:.3f}", end="")
        print()

    # Optimal RidgeCV
    alphas = np.logspace(-4, 4, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_absolute_error")
    cv_preds = cross_val_predict(ridge_cv, X, targets, cv=5)
    cv_mae = np.mean(np.abs(cv_preds - targets))

    ridge_cv.fit(X, targets)
    print(f"\n  Optimal (alpha={ridge_cv.alpha_:.4f}): CV MAE={cv_mae:.4f}")

    print("\n  Learned weights:")
    for name, coef in sorted(zip(names, ridge_cv.coef_), key=lambda x: -abs(x[1])):
        print(f"    {name:40s}: {coef:+.4f}")
    print(f"    {'intercept':40s}: {ridge_cv.intercept_:+.4f}")


if __name__ == "__main__":
    main()
