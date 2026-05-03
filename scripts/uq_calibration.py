"""Calibration analysis of the deep-ensemble UQ.

Beyond the basic Pearson-r and decile reliability already shown in
``scripts/ensemble_uq.py``, we report metrics that materials-ML reviewers
typically expect for a "trustworthy uncertainty" claim:

  * Gaussian negative log-likelihood (NLL)
  * Continuous Ranked Probability Score (CRPS)
  * Coverage at nominal confidence levels  (50 / 68 / 90 / 95 %)
  * Expected Calibration Error in z-space (ECE_z)
  * Temperature scaling: a single global scalar τ that rescales σ → τσ to
    minimise NLL on the validation half of the test set; we then report
    the post-scaling metrics on the held-out half (avoiding "tuning on test")

We also save every metric to ``results/uq_calibration.json`` and an updated
reliability figure ``paper/figures/fig_uq_calibration.png``.

Why this matters: a deep ensemble produces a usable σ estimate, but raw σ is
typically *under-confident* (too small) or *over-confident* (too large). A
single scalar τ correction is a standard remediation that demonstrably
restores calibration on physics regression tasks.
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
    inside = np.abs(y - mu) <= z_level * sigma
    return float(inside.mean())


def ece_z(mu, sigma, y, n_bins=20):
    """ECE in z-space: bin |y-mu|/sigma against expected half-normal CDF.

    For a perfectly calibrated Gaussian, the empirical CDF of z = (y-mu)/sigma
    should match the standard normal CDF. We compute |F_emp(c) - F_norm(c)|
    averaged over the c-grid.
    """
    z = (y - mu) / np.maximum(sigma, 1e-6)
    grid = np.linspace(-3, 3, n_bins + 1)
    emp = np.array([(z <= g).mean() for g in grid])
    th = norm.cdf(grid)
    return float(np.abs(emp - th).mean()), grid, emp, th


def fit_temperature(mu, sigma, y, ratio_split=0.5, seed=0):
    """Fit a single global τ on a held-out half of the test set."""
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    n_fit = int(ratio_split * n)
    fit_idx, eval_idx = perm[:n_fit], perm[n_fit:]
    mu_f, sig_f, y_f = mu[fit_idx], sigma[fit_idx], y[fit_idx]

    def nll(tau):
        return gauss_nll(mu_f, max(tau, 1e-6) * sig_f, y_f).mean()

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    tau = float(res.x)
    return tau, fit_idx, eval_idx


def report(mu, sigma, y, label):
    nll = float(gauss_nll(mu, sigma, y).mean())
    crps = float(crps_gaussian(mu, sigma, y).mean())
    cov = {f"cov_{int(L * 100)}": coverage(mu, sigma, y, L) for L in NOMINAL}
    ece, grid, emp, th = ece_z(mu, sigma, y)
    pearson = float(np.corrcoef(sigma, np.abs(y - mu))[0, 1])
    out = {
        "label": label,
        "n": int(len(y)),
        "mae": float(np.abs(y - mu).mean()),
        "rmse": float(np.sqrt(((y - mu) ** 2).mean())),
        "mean_sigma": float(sigma.mean()),
        "nll": nll,
        "crps": crps,
        "ece_z": ece,
        "pearson_sigma_err": pearson,
        **cov,
    }
    return out, (grid, emp, th)


def main():
    runs = [
        "baseline_h128_aug_long_safe",        # seed 42
        "baseline_h128_aug_long_safe_seed0",
        "baseline_h128_aug_long_safe_seed1",
        "baseline_h128_aug_long_safe_seed2",
    ]
    preds, targets = [], None
    for r in runs:
        p, t = load_preds(r)
        if p is None:
            print(f"missing {r}"); continue
        preds.append(p)
        if targets is None: targets = t
        else: assert np.allclose(targets, t)
    P = np.stack(preds)
    print(f"deep ensemble of {P.shape[0]} seeds, N={P.shape[1]}")

    mu = P.mean(0)
    sigma = P.std(0, ddof=1)

    raw, raw_curves = report(mu, sigma, targets, "raw_ensemble")
    print(json.dumps(raw, indent=2))

    tau, fit_idx, eval_idx = fit_temperature(mu, sigma, targets)
    print(f"\nfitted τ on 50%-hold-out fit subset: τ = {tau:.4f}")

    sigma_t = tau * sigma
    eval_metrics, eval_curves = report(
        mu[eval_idx], sigma_t[eval_idx], targets[eval_idx],
        f"temperature_scaled (τ={tau:.3f}, eval-half)",
    )
    print(json.dumps(eval_metrics, indent=2))

    # Figure: 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    err = np.abs(targets - mu)
    ax = axes[0, 0]
    ax.scatter(sigma, err, s=5, alpha=0.3, edgecolors="none", label="raw")
    ax.scatter(sigma_t, err, s=5, alpha=0.3, edgecolors="none", color="red",
               label=f"τ-scaled (τ={tau:.2f})")
    sx = np.linspace(0, max(sigma.max(), sigma_t.max()) * 1.05, 100)
    ax.plot(sx, sx, "k--", lw=0.8, label="y=x")
    ax.set_xlabel("Predictive σ (eV)")
    ax.set_ylabel("|prediction error| (eV)")
    ax.set_title(f"Reliability scatter; r={raw['pearson_sigma_err']:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    grid, emp, th = raw_curves
    ax.plot(th, emp, "o-", label="raw ensemble")
    g2, e2, t2 = eval_curves
    ax.plot(t2, e2, "s-", color="red", label=f"τ-scaled (eval half)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    ax.set_xlabel("Theoretical CDF $\\Phi(z)$")
    ax.set_ylabel("Empirical CDF of $z=(y-\\mu)/\\sigma$")
    ax.set_title("Calibration curve in z-space")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    levels = np.array(NOMINAL)
    raw_cov = np.array([raw[f"cov_{int(L * 100)}"] for L in NOMINAL])
    eval_cov = np.array([eval_metrics[f"cov_{int(L * 100)}"] for L in NOMINAL])
    width = 0.02
    ax.bar(levels - width, raw_cov, width=2 * width, label="raw")
    ax.bar(levels + width, eval_cov, width=2 * width, color="red",
           label=f"τ-scaled, τ={tau:.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    ax.set_xlabel("Nominal confidence")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Interval coverage")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # decile of (τ-scaled) sigma -> mean |err|
    ax = axes[1, 1]
    n = len(targets)
    for sig_arr, lbl, col in [(sigma, "raw", "tab:blue"),
                              (sigma_t, f"τ-scaled (τ={tau:.2f})", "tab:red")]:
        d_idx = np.argsort(sig_arr)
        means_s = []; means_e = []
        for k in range(10):
            sel = d_idx[n * k // 10 : n * (k + 1) // 10]
            means_s.append(sig_arr[sel].mean())
            means_e.append(np.abs(targets[sel] - mu[sel]).mean())
        ax.plot(range(1, 11), means_s, "o-", color=col, label=f"{lbl} σ̄")
        ax.plot(range(1, 11), means_e, "s--", color=col, alpha=0.6, label=f"{lbl} |err|̄")
    ax.set_xlabel("σ decile (low → high)")
    ax.set_ylabel("eV")
    ax.set_title("Decile reliability")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = FIG_DIR / "fig_uq_calibration.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nfigure saved -> {out}")

    summary = {
        "n_seeds": int(P.shape[0]),
        "n_test": int(P.shape[1]),
        "raw": raw,
        "tau": tau,
        "tau_eval_metrics": eval_metrics,
    }
    out_json = RESULTS / "uq_calibration.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary saved -> {out_json}")


if __name__ == "__main__":
    main()
