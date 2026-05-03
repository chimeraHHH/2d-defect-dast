"""How does ensemble size affect MAE / NLL / coverage?

Take all 6 available seeds (4 long + 2 xlong) and compute, for each
ensemble size k = 1, 2, 3, 4, 5, 6, the test-set MAE / NLL / coverage
when averaging k of those seeds. We try every C(6, k) combination so
the result is the *expected* gain from k seeds (not best-case).

This is the standard "ensemble efficient frontier" plot that tells a
practitioner: how many seeds do I need to roughly saturate the gain?
"""
from __future__ import annotations

import json
import math
from itertools import combinations
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


def gauss_nll(mu, sigma, y, eps=1e-6):
    sigma = np.maximum(sigma, eps)
    return 0.5 * np.log(2 * math.pi * sigma ** 2) + 0.5 * ((y - mu) / sigma) ** 2


def coverage(mu, sigma, y, level=0.9):
    z = norm.ppf(0.5 + level / 2)
    return float((np.abs(y - mu) <= z * sigma).mean())


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
    return float(res.x), eval_idx


def main():
    runs = [
        ("baseline_h128_aug_long_safe", "L42"),
        ("baseline_h128_aug_long_safe_seed0", "L0"),
        ("baseline_h128_aug_long_safe_seed1", "L1"),
        ("baseline_h128_aug_long_safe_seed2", "L2"),
        ("baseline_h128_aug_xlong_safe", "X42"),
        ("baseline_h128_aug_xlong_safe_seed0", "X0"),
    ]
    preds, targets = [], None
    print("Members:")
    for run, tag in runs:
        p = RESULTS / run / "test_predictions.npz"
        if not p.exists():
            print(f"  skip {run}"); continue
        a = np.load(p)
        preds.append((tag, a["preds"].astype(np.float64)))
        if targets is None:
            targets = a["targets"].astype(np.float64)
        else:
            assert np.allclose(targets, a["targets"])
        print(f"  + {tag}  {run}")

    n = len(preds)
    print(f"available members: {n}")
    if n < 2:
        print("not enough seeds"); return

    rows = []  # k -> list of (mae_raw, nll_raw, cov90_raw, mae_tau, cov90_tau)
    by_size = {}
    for k in range(1, n + 1):
        by_size[k] = []
        combos = list(combinations(range(n), k))
        if len(combos) > 30:
            # subsample to save time when k=3,4 etc give too many combos
            rng = np.random.default_rng(0)
            combos = [combos[i] for i in rng.choice(len(combos), size=30, replace=False)]
        for c in combos:
            P = np.stack([preds[i][1] for i in c])
            mu = P.mean(0)
            if k == 1:
                # σ undefined for k=1; skip NLL/cov
                mae = float(np.abs(targets - mu).mean())
                rmse = float(np.sqrt(((targets - mu) ** 2).mean()))
                by_size[k].append({
                    "members": [preds[i][0] for i in c],
                    "mae": mae, "rmse": rmse,
                    "nll_raw": None, "cov90_raw": None,
                    "mae_tau": None, "cov90_tau": None, "tau": None,
                })
                continue
            sigma = P.std(0, ddof=1)
            mae = float(np.abs(targets - mu).mean())
            rmse = float(np.sqrt(((targets - mu) ** 2).mean()))
            nll_r = float(gauss_nll(mu, sigma, targets).mean())
            cov_r = coverage(mu, sigma, targets, 0.9)
            tau, eval_idx = fit_temperature(mu, sigma, targets)
            sig_t = tau * sigma
            mae_t = float(np.abs(targets[eval_idx] - mu[eval_idx]).mean())
            cov_t = coverage(mu[eval_idx], sig_t[eval_idx], targets[eval_idx], 0.9)
            by_size[k].append({
                "members": [preds[i][0] for i in c],
                "mae": mae, "rmse": rmse,
                "nll_raw": nll_r, "cov90_raw": cov_r,
                "mae_tau": mae_t, "cov90_tau": cov_t, "tau": tau,
            })
        # report mean / std
        maes = np.array([r["mae"] for r in by_size[k]])
        rmses = np.array([r["rmse"] for r in by_size[k]])
        rows.append({"k": k, "mae_mean": float(maes.mean()), "mae_std": float(maes.std()),
                     "rmse_mean": float(rmses.mean()), "rmse_std": float(rmses.std()),
                     "n_combos": len(by_size[k])})
        if k >= 2:
            cov_t = np.array([r["cov90_tau"] for r in by_size[k]])
            nll_r = np.array([r["nll_raw"] for r in by_size[k]])
            rows[-1].update({"nll_raw_mean": float(nll_r.mean()),
                             "cov90_tau_mean": float(cov_t.mean())})
        print(f"k={k}: MAE {maes.mean():.4f} ± {maes.std():.4f}  RMSE {rmses.mean():.4f}"
              + (f"  cov90(τ)={cov_t.mean():.3f}" if k >= 2 else ""))

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    ks = [r["k"] for r in rows]
    mae_m = [r["mae_mean"] for r in rows]
    mae_s = [r["mae_std"] for r in rows]
    axes[0].errorbar(ks, mae_m, yerr=mae_s, fmt="o-", capsize=4)
    axes[0].set_xlabel("Ensemble size k")
    axes[0].set_ylabel("Test MAE (eV)")
    axes[0].set_title("Ensemble efficient frontier")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(ks)

    cov_t = [r.get("cov90_tau_mean") for r in rows]
    axes[1].plot(ks[1:], cov_t[1:], "o-")
    axes[1].axhline(0.9, color="k", lw=0.6, ls="--", label="ideal cov90 = 0.9")
    axes[1].set_xlabel("Ensemble size k")
    axes[1].set_ylabel("Coverage @ 90% nominal (τ-scaled)")
    axes[1].set_title("Calibration vs ensemble size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(ks)

    nll = [r.get("nll_raw_mean") for r in rows]
    axes[2].plot(ks[1:], nll[1:], "o-")
    axes[2].set_xlabel("Ensemble size k")
    axes[2].set_ylabel("Raw NLL (lower better)")
    axes[2].set_title("Probabilistic accuracy vs ensemble size")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(ks)

    fig.tight_layout()
    out = FIG_DIR / "fig_ensemble_size_ablation.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"\nfigure saved -> {out}")

    summary = {"members": [tag for tag, _ in preds], "rows": rows, "by_size": by_size}
    with open(RESULTS / "ensemble_size_ablation.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/ensemble_size_ablation.json")


if __name__ == "__main__":
    main()
