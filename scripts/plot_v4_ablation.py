"""Generate paper-quality figures for v4 ablation and results.

Produces:
  1. Progressive improvement bar chart
  2. Ensemble size vs MAE curve
  3. UQ reliability diagram
  4. Individual model MAE distribution
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_progressive_improvement():
    """Bar chart showing contribution of each optimization step."""
    steps = [
        ("Baseline\n(MSE, 100ep)", 0.513),
        ("+ct-UAE\n(128-dim)", 0.504),
        ("+MAE loss", 0.452),
        ("+Warmup\n(LR 5e-4)", 0.411),
        ("+150ep\n+SWA", 0.407),
    ]
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(steps)))
    bars = ax.bar(range(len(steps)), values, color=colors, edgecolor="black", linewidth=0.5)

    # Add delta annotations
    for i in range(1, len(steps)):
        delta = values[i] - values[i-1]
        ax.annotate(f"{delta:+.3f}",
                    xy=(i, values[i] + 0.005),
                    ha="center", va="bottom", fontsize=8, color="red")

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("Progressive Improvement: Single Model")
    ax.axhline(0.540, color="gray", linestyle="--", linewidth=1, label="ALIGNN (0.540)")
    ax.axhline(0.407, color="green", linestyle=":", linewidth=1, label="Best single (0.407)")
    ax.set_ylim(0.35, 0.58)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(FIG_DIR / "fig_v4_progressive.png")
    plt.close(fig)
    print(f"  Saved fig_v4_progressive.png")


def plot_ensemble_curve():
    """Ensemble size vs MAE showing saturation — SS only vs SS+MS combined."""
    data = np.load(RESULTS / "ensemble_online.npz", allow_pickle=True)
    individual = data["individual_preds"]
    targets = data["targets"]
    names = list(data["model_names"])

    # Load UQ results for SS-only best-k values
    with open(RESULTS / "uq_v4_ensemble.json") as f:
        uq = json.load(f)

    ks_ss = [int(k) for k in uq["best_per_k"].keys()]
    maes_ss = [uq["best_per_k"][str(k)]["mae"] for k in ks_ss]

    # Load combined results (SS+MS)
    with open(RESULTS / "ensemble_combined.json") as f:
        combined = json.load(f)

    ks_comb = [int(k) for k in combined["greedy_selection"].keys()]
    maes_comb = [combined["greedy_selection"][str(k)]["mae"] for k in ks_comb]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.8))
    ax.plot(ks_ss, maes_ss, "o-", color="steelblue", markersize=5, linewidth=1.5,
            label="Single-source only (26 models)")
    ax.plot(ks_comb, maes_comb, "s-", color="darkred", markersize=5, linewidth=2,
            label="SS + Multi-source (28 models)")

    # Mark best combined
    best_comb_idx = np.argmin(maes_comb)
    ax.scatter([ks_comb[best_comb_idx]], [maes_comb[best_comb_idx]], s=120,
               color="red", zorder=5, marker="*",
               label=f"Best combined: k={ks_comb[best_comb_idx]}, MAE={maes_comb[best_comb_idx]:.4f}")

    # Mark best SS
    best_ss_idx = np.argmin(maes_ss)
    ax.scatter([ks_ss[best_ss_idx]], [maes_ss[best_ss_idx]], s=80,
               color="blue", zorder=5,
               label=f"Best SS: k={ks_ss[best_ss_idx]}, MAE={maes_ss[best_ss_idx]:.4f}")

    # Reference lines
    ax.axhline(0.540, color="orange", linestyle=":", linewidth=1, label="ALIGNN (0.540)")
    ax.axhline(0.407, color="green", linestyle=":", linewidth=1, label="Best single (0.407)")

    ax.set_xlabel("Ensemble size (k)")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("Ensemble Saturation: Single-Source vs Combined")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(alpha=0.3)
    ax.set_xlim(1.5, 10.5)
    ax.set_ylim(0.35, 0.42)

    fig.savefig(FIG_DIR / "fig_v4_ensemble_curve.png")
    plt.close(fig)
    print(f"  Saved fig_v4_ensemble_curve.png")


def plot_reliability_diagram():
    """UQ reliability diagram: expected vs observed confidence."""
    data = np.load(RESULTS / "ensemble_online.npz", allow_pickle=True)
    individual = data["individual_preds"]
    targets = data["targets"]
    names = list(data["model_names"])

    # Best-6 ensemble
    best6 = ["uae_mae_warmup_s45", "uae_mae_warmup_s46", "deep_s42",
             "150ep_s42", "150ep_s43", "150ep_s45"]
    idx = [list(names).index(m) for m in best6]
    P = individual[idx]
    mu = P.mean(axis=0)
    sigma = P.std(axis=0, ddof=1)

    # Calibrate tau on half
    N = len(targets)
    rng = np.random.default_rng(42)
    cal_idx = rng.choice(N, N // 2, replace=False)
    eval_idx = np.setdiff1d(np.arange(N), cal_idx)

    from scipy.optimize import minimize_scalar
    def obj(tau):
        s = sigma[cal_idx] * tau
        return (0.5*np.log(2*np.pi*s**2) + 0.5*((targets[cal_idx]-mu[cal_idx])/s)**2).mean()
    tau = minimize_scalar(obj, bounds=(0.1, 10), method="bounded").x

    # Reliability on eval set
    sigma_cal = sigma[eval_idx] * tau
    z = (targets[eval_idx] - mu[eval_idx]) / sigma_cal

    # Compute expected vs observed coverage
    confidence_levels = np.linspace(0.05, 0.99, 20)
    observed = []
    for cl in confidence_levels:
        z_crit = norm.ppf((1 + cl) / 2)
        obs = (np.abs(z) <= z_crit).mean()
        observed.append(obs)
    observed = np.array(observed)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(confidence_levels, observed, "o-", color="darkblue", markersize=4,
            linewidth=2, label=f"Best-6 (τ={tau:.2f})")

    # Also plot raw (no tau)
    z_raw = (targets[eval_idx] - mu[eval_idx]) / sigma[eval_idx]
    observed_raw = [(np.abs(z_raw) <= norm.ppf((1 + cl) / 2)).mean() for cl in confidence_levels]
    ax.plot(confidence_levels, observed_raw, "s--", color="orange", markersize=3,
            linewidth=1.5, alpha=0.7, label="Raw (no τ)")

    ax.set_xlabel("Expected confidence level")
    ax.set_ylabel("Observed coverage")
    ax.set_title("UQ Reliability Diagram")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    fig.savefig(FIG_DIR / "fig_v4_reliability.png")
    plt.close(fig)
    print(f"  Saved fig_v4_reliability.png")


def plot_model_distribution():
    """Distribution of individual model MAEs showing diversity."""
    with open(RESULTS / "uq_v4_ensemble.json") as f:
        uq = json.load(f)

    maes = uq["individual_maes"]

    # Categorize by recipe
    categories = {
        "MSE baseline": [n for n in maes if n.startswith("100ep_")],
        "UAE+MSE": [n for n in maes if n.startswith("uae_s")],
        "UAE+Huber": [n for n in maes if n.startswith("uae_huber")],
        "UAE+MAE": [n for n in maes if n.startswith("uae_mae_s")],
        "UAE+MAE+warmup": [n for n in maes if n.startswith("uae_mae_warmup")],
        "Deep": [n for n in maes if n.startswith("deep_s") or n.startswith("deep_huber")],
        "No-UAE": [n for n in maes if n.startswith("no_uae")],
        "150ep": [n for n in maes if n.startswith("150ep_")],
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    positions = []
    labels = []
    colors_map = plt.cm.Set2(np.linspace(0, 1, len(categories)))

    pos = 0
    for i, (cat, models) in enumerate(categories.items()):
        vals = [maes[m] for m in models]
        for v in vals:
            ax.scatter(v, pos, color=colors_map[i], s=80, edgecolors="black", linewidth=0.5, zorder=3)
            pos += 0.3
        positions.append(pos - 0.3 * len(vals) / 2)
        labels.append(f"{cat}\n(n={len(models)})")
        pos += 0.5

    ax.axvline(0.407, color="green", linestyle="--", linewidth=1.5, label="Best single (0.407)")
    ax.axvline(0.368, color="red", linestyle="--", linewidth=1.5, label="Best-5 ens (0.368)")
    ax.axvline(0.540, color="gray", linestyle=":", linewidth=1, label="ALIGNN (0.540)")

    ax.set_xlabel("Test MAE (eV)")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Individual Model Performance by Recipe")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0.38, 0.56)

    fig.savefig(FIG_DIR / "fig_v4_model_distribution.png")
    plt.close(fig)
    print(f"  Saved fig_v4_model_distribution.png")


if __name__ == "__main__":
    print("Generating v4 paper figures...")
    plot_progressive_improvement()
    plot_ensemble_curve()
    plot_reliability_diagram()
    plot_model_distribution()
    print("\nDone!")
