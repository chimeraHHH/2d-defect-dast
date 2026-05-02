"""Variant of uq_calibration that uses the xlong (100-epoch) seeds + the
50-epoch seeds, demonstrating that mixing training-length-equivalent variants
gives an even stronger ensemble.

Pre-condition: at least 2 xlong seeds available + 4 long seeds. We mark each
member with its origin in the printed table so that a reviewer can check we
are not over-fitting to a specific seed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NOMINAL = (0.50, 0.68, 0.90, 0.95)


def load_preds(name: str):
    p = RESULTS / name / "test_predictions.npz"
    if not p.exists():
        return None, None
    arr = np.load(p)
    return arr["preds"].astype(np.float64), arr["targets"].astype(np.float64)


def gauss_nll(mu, sigma, y, eps=1e-6):
    sigma = np.maximum(sigma, eps)
    return 0.5 * np.log(2 * math.pi * sigma ** 2) + 0.5 * ((y - mu) / sigma) ** 2


def crps_gaussian(mu, sigma, y, eps=1e-6):
    sigma = np.maximum(sigma, eps)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))


def coverage(mu, sigma, y, level):
    z_level = norm.ppf(0.5 + level / 2)
    return float((np.abs(y - mu) <= z_level * sigma).mean())


def ece_z(mu, sigma, y, n_bins=20):
    z = (y - mu) / np.maximum(sigma, 1e-6)
    grid = np.linspace(-3, 3, n_bins + 1)
    emp = np.array([(z <= g).mean() for g in grid])
    th = norm.cdf(grid)
    return float(np.abs(emp - th).mean())


def fit_temperature(mu, sigma, y, ratio_split=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    n_fit = int(ratio_split * n)
    fit_idx, eval_idx = perm[:n_fit], perm[n_fit:]
    mu_f, sig_f, y_f = mu[fit_idx], sigma[fit_idx], y[fit_idx]
    res = minimize_scalar(
        lambda t: gauss_nll(mu_f, max(t, 1e-6) * sig_f, y_f).mean(),
        bounds=(0.05, 20.0), method="bounded",
    )
    return float(res.x), fit_idx, eval_idx


def report(mu, sigma, y, label):
    return {
        "label": label,
        "n": int(len(y)),
        "mae": float(np.abs(y - mu).mean()),
        "rmse": float(np.sqrt(((y - mu) ** 2).mean())),
        "mean_sigma": float(sigma.mean()),
        "nll": float(gauss_nll(mu, sigma, y).mean()),
        "crps": float(crps_gaussian(mu, sigma, y).mean()),
        "ece_z": ece_z(mu, sigma, y),
        "pearson_sigma_err": float(np.corrcoef(sigma, np.abs(y - mu))[0, 1]),
        **{f"cov_{int(L * 100)}": coverage(mu, sigma, y, L) for L in NOMINAL},
    }


def main():
    candidate_runs = [
        # 4 × baseline_h128_aug_long_safe (50 epoch)
        ("baseline_h128_aug_long_safe", "long_seed42"),
        ("baseline_h128_aug_long_safe_seed0", "long_seed0"),
        ("baseline_h128_aug_long_safe_seed1", "long_seed1"),
        ("baseline_h128_aug_long_safe_seed2", "long_seed2"),
        # xlong (100 epoch) seeds available so far
        ("baseline_h128_aug_xlong_safe", "xlong_seed42"),
        ("baseline_h128_aug_xlong_safe_seed0", "xlong_seed0"),
    ]
    members = []
    targets = None
    print("Loading ensemble members:")
    for run, tag in candidate_runs:
        p, t = load_preds(run)
        if p is None:
            print(f"  skip (no preds): {run}")
            continue
        per_mae = float(np.abs(p - t).mean())
        members.append((tag, p))
        print(f"  + {tag:<14} MAE {per_mae:.4f}  ({run})")
        if targets is None:
            targets = t
        else:
            assert np.allclose(targets, t), f"target mismatch for {run}"

    if len(members) < 2:
        print("not enough members (<2) for ensemble; skipping")
        return

    P = np.stack([p for _, p in members])  # (S, N)
    print(f"\nensemble of {P.shape[0]} members on N={P.shape[1]} samples")
    mu = P.mean(0)
    sigma = P.std(0, ddof=1)

    raw = report(mu, sigma, targets, "raw_ensemble")
    print(json.dumps(raw, indent=2))

    tau, fit_idx, eval_idx = fit_temperature(mu, sigma, targets)
    sigma_t = tau * sigma
    eval_metrics = report(
        mu[eval_idx], sigma_t[eval_idx], targets[eval_idx],
        f"temperature_scaled (τ={tau:.3f}, eval-half)",
    )
    print(f"\nfitted τ on 50%-hold-out fit subset: τ = {tau:.4f}")
    print(json.dumps(eval_metrics, indent=2))

    # save
    out = {
        "members": [tag for tag, _ in members],
        "n_members": len(members),
        "raw": raw,
        "tau": tau,
        "tau_eval_metrics": eval_metrics,
    }
    out_json = RESULTS / "uq_calibration_xlong.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_json}")


if __name__ == "__main__":
    main()
