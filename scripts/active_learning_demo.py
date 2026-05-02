"""Active-learning oracle demo: σ-thresholded vs random selection.

Sorting test samples by ensemble σ and comparing the MAE of the top-k%
"most confident" subset to a random k% subset shows directly whether
the σ signal can be used to cherry-pick high-quality predictions.

In a real active-learning loop the inverse is used: the *bottom* k%
(highest σ) are sent for DFT verification while the rest are auto-accepted.
We report both:

  - "auto-accept top-k% confident" → low MAE expected (good)
  - "send bottom-k% to DFT" → fraction of total error captured

Output:
  - results/active_learning_curve.json
  - paper/figures/fig_active_learning.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    runs = [
        "baseline_h128_aug_long_safe",
        "baseline_h128_aug_long_safe_seed0",
        "baseline_h128_aug_long_safe_seed1",
        "baseline_h128_aug_long_safe_seed2",
        "baseline_h128_aug_xlong_safe",
        "baseline_h128_aug_xlong_safe_seed0",
    ]
    P, targets = [], None
    for r in runs:
        f = RESULTS / r / "test_predictions.npz"
        if not f.exists():
            continue
        a = np.load(f)
        P.append(a["preds"].astype(np.float64))
        if targets is None:
            targets = a["targets"].astype(np.float64)
    P = np.stack(P)
    mu = P.mean(0)
    sigma = P.std(0, ddof=1)
    err = np.abs(targets - mu)

    n = len(targets)
    fractions = np.linspace(0.05, 1.0, 20)
    sigma_order = np.argsort(sigma)  # ascending → low σ first

    rng = np.random.default_rng(0)

    rows = []
    for f in fractions:
        k = max(1, int(f * n))
        # confident subset: lowest-σ k samples
        conf_idx = sigma_order[:k]
        conf_mae = float(err[conf_idx].mean())
        # uncertain subset: highest-σ k samples
        unc_idx = sigma_order[-k:]
        unc_mae = float(err[unc_idx].mean())
        # random subset: average over 50 random samples of size k
        rand_maes = [float(err[rng.choice(n, k, replace=False)].mean()) for _ in range(50)]
        rand_mae_mean = float(np.mean(rand_maes))
        rand_mae_std = float(np.std(rand_maes))
        rows.append({
            "fraction": float(f), "k": int(k),
            "conf_subset_mae": conf_mae,
            "uncertain_subset_mae": unc_mae,
            "random_subset_mae_mean": rand_mae_mean,
            "random_subset_mae_std": rand_mae_std,
        })

    # how much of total absolute error is in the top-K% uncertain?
    cum_err = np.cumsum(err[sigma_order[::-1]])
    cum_err_frac = cum_err / err.sum()
    fraction_grid = np.arange(1, n + 1) / n

    print("Active-learning demo results:")
    print(f"  Top-10% confident (low σ):    MAE = {rows[1]['conf_subset_mae']:.4f}")
    print(f"  Top-10% uncertain (high σ):    MAE = {rows[1]['uncertain_subset_mae']:.4f}")
    print(f"  Random 10%:                    MAE = {rows[1]['random_subset_mae_mean']:.4f}")
    print()
    # find x at which 50% of total |err| is captured
    target_p = 0.5
    idx50 = int(np.searchsorted(cum_err_frac, target_p))
    print(f"  Sending top-{idx50/n*100:.1f}% most-uncertain samples to DFT captures 50% of all |err|")
    target_p = 0.8
    idx80 = int(np.searchsorted(cum_err_frac, target_p))
    print(f"  Sending top-{idx80/n*100:.1f}% most-uncertain samples to DFT captures 80% of all |err|")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fracs = [r["fraction"] for r in rows]
    conf = [r["conf_subset_mae"] for r in rows]
    unc = [r["uncertain_subset_mae"] for r in rows]
    rnd_m = [r["random_subset_mae_mean"] for r in rows]
    rnd_s = [r["random_subset_mae_std"] for r in rows]

    ax = axes[0]
    ax.plot(fracs, conf, "o-", label="lowest-σ subset (most confident)")
    ax.plot(fracs, unc, "s-", color="red", label="highest-σ subset (most uncertain)")
    ax.errorbar(fracs, rnd_m, yerr=rnd_s, fmt="^--", color="gray", label="random subset (50 trials)", capsize=3, alpha=0.7)
    ax.set_xlabel("Subset fraction")
    ax.set_ylabel("Subset MAE (eV)")
    ax.set_title("Active-learning oracle: σ-sorted subsets")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(fraction_grid, cum_err_frac)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.axhline(0.8, color="gray", lw=0.5, ls=":")
    ax.axvline(idx50 / n, color="gray", lw=0.5, ls=":")
    ax.axvline(idx80 / n, color="gray", lw=0.5, ls=":")
    ax.text(idx50 / n + 0.01, 0.45, f"50% err @ top-{idx50/n*100:.1f}%", fontsize=8)
    ax.text(idx80 / n + 0.01, 0.75, f"80% err @ top-{idx80/n*100:.1f}%", fontsize=8)
    ax.set_xlabel("Fraction of test set sent to DFT (sorted by σ desc)")
    ax.set_ylabel("Cumulative |error| captured")
    ax.set_title("Active-learning efficiency")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "fig_active_learning.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"figure saved -> {out}")

    summary = {
        "n_test": int(n),
        "rows": rows,
        "top_uncertain_to_capture_50pct_err": float(idx50 / n),
        "top_uncertain_to_capture_80pct_err": float(idx80 / n),
    }
    out_json = RESULTS / "active_learning_curve.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()
