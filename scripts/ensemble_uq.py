"""Deep-ensemble uncertainty quantification + calibration analysis.

Uses the 4 multi-seed predictions of ``baseline_h128_aug_long_safe`` (and the
``seed=42`` main run) as a deep ensemble. For each test sample we compute:
  * ensemble mean prediction $\\hat{y}$ = mean over seeds
  * predictive uncertainty $\\sigma$ = std over seeds (epistemic)
  * absolute residual $|y - \\hat{y}|$
We then report:
  * ensemble Test MAE / RMSE (vs single-seed best)
  * calibration: Pearson correlation of $\\sigma$ vs $|y - \\hat{y}|$, and a
    deciles-of-$\\sigma$ table with mean residual per decile (good calibration
    means residual increases monotonically with $\\sigma$)
  * a "reliability" plot under ``paper/figures/fig_uq_reliability.png``.

The point of this analysis is twofold:
  1. *Headline*: ensemble achieves a better point estimate than any single seed
     and gives a usable uncertainty signal.
  2. *Active learning*: high-uncertainty samples are exactly the ones that
     would benefit most from DFT recomputation, motivating an
     active-learning loop.
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


def load_preds(name: str):
    p = RESULTS / name / "test_predictions.npz"
    if not p.exists():
        return None, None
    arr = np.load(p)
    return arr["preds"].astype(np.float64), arr["targets"].astype(np.float64)


def main():
    runs = [
        "baseline_h128_aug_long_safe",        # seed 42
        "baseline_h128_aug_long_safe_seed0",
        "baseline_h128_aug_long_safe_seed1",
        "baseline_h128_aug_long_safe_seed2",
    ]

    preds_list, targets = [], None
    for r in runs:
        p, t = load_preds(r)
        if p is None:
            print(f"missing {r}")
            continue
        preds_list.append(p)
        if targets is None:
            targets = t
        else:
            assert np.allclose(targets, t), f"target mismatch for {r}"
    P = np.stack(preds_list)  # (S, N)
    print(f"ensemble of {P.shape[0]} seeds, N={P.shape[1]}")

    mu = P.mean(axis=0)
    sigma = P.std(axis=0, ddof=1)
    err = np.abs(targets - mu)

    rmse = float(np.sqrt(((mu - targets) ** 2).mean()))
    mae = float(err.mean())
    print(f"ensemble    : MAE {mae:.4f}, RMSE {rmse:.4f}")
    for i, r in enumerate(runs):
        single_rmse = float(np.sqrt(((P[i] - targets) ** 2).mean()))
        single_mae = float(np.abs(P[i] - targets).mean())
        print(f"  vs {r:<40} MAE {single_mae:.4f}, RMSE {single_rmse:.4f}")

    # Calibration: correlation of sigma with absolute residual
    pearson = float(np.corrcoef(sigma, err)[0, 1])
    print(f"\\nCalibration (corr σ vs |err|): Pearson r = {pearson:.3f}")

    # decile analysis
    decile_idx = np.argsort(sigma)
    n = len(sigma)
    print(f"\\nDecile of σ : mean σ : mean |err| : count")
    decile_summary = []
    for k in range(10):
        sel = decile_idx[n * k // 10 : n * (k + 1) // 10]
        decile_summary.append(
            (k + 1, float(sigma[sel].mean()), float(err[sel].mean()), len(sel))
        )
        print(
            f"  D{k + 1}: σ̄={sigma[sel].mean():.3f}  "
            f"|err|̄={err[sel].mean():.3f}  n={len(sel)}"
        )

    # reliability figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(sigma, err, s=5, alpha=0.3, edgecolors="none")
    sx = np.linspace(0, sigma.max() * 1.05, 100)
    ax.plot(sx, sx, "k--", lw=0.8, label="y=x (perfect)")
    ax.set_xlabel("Ensemble σ (eV)")
    ax.set_ylabel("|prediction error| (eV)")
    ax.set_title(f"Per-sample reliability\nPearson r={pearson:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    deciles = [d[0] for d in decile_summary]
    mean_sigmas = [d[1] for d in decile_summary]
    mean_errs = [d[2] for d in decile_summary]
    ax.plot(deciles, mean_sigmas, "o-", label="mean σ per decile")
    ax.plot(deciles, mean_errs, "s-", label="mean |err| per decile")
    ax.set_xlabel("σ decile (low → high)")
    ax.set_ylabel("eV")
    ax.set_title("Decile calibration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = FIG_DIR / "fig_uq_reliability.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nfigure saved -> {out}")

    # save numbers
    summary = {
        "n_seeds": int(P.shape[0]),
        "n_samples": int(P.shape[1]),
        "ensemble_mae": mae,
        "ensemble_rmse": rmse,
        "calibration_pearson": pearson,
        "deciles": [{"k": k, "mean_sigma": s, "mean_abs_err": e, "n": n}
                    for k, s, e, n in decile_summary],
    }
    out_json = RESULTS / "ensemble_uq.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary saved -> {out_json}")


if __name__ == "__main__":
    main()
