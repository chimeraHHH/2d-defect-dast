"""C16: High-throughput screening (HTS) workflow demonstration.

Builds on the existing 4-seed deep ensemble (§5.9 + temperature scaling)
to demonstrate a complete materials-discovery deployment workflow:

  1. Predict mean Ef and σ for each candidate (here: held-out test set
     as proxy for an unscreened pool).
  2. Apply temperature τ from §5.9 to get calibrated σ.
  3. Filter by physical / chemical constraints:
       - low Ef (thermodynamically favoured)
       - low σ (model is confident)
       - per-host top-k for chemical diversity
  4. Output a "synthesis recommendation" table and verify against
     ground-truth Ef (since we have it for the test set).

This converts §5.9's "calibrated UQ" into actionable predictions that
match how the ML model would actually be used in a materials lab.

Outputs
-------
- results/hts_demo.json         (recommendation table + verification stats)
- paper/figures/fig_hts_demo.png (parity + recommendation overlay)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import split_indices  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 4-seed members of the leak-free ensemble
SEED_DIRS = [
    "baseline_h128_aug_long_safe",        # seed=42
    "baseline_h128_aug_long_safe_seed0",
    "baseline_h128_aug_long_safe_seed1",
    "baseline_h128_aug_long_safe_seed2",
]

# Filtering thresholds for the recommendation
EF_MAX_EV = 1.0           # only flag low-energy candidates (≤ 1 eV)
SIGMA_MAX_EV = 0.5        # only flag confident predictions
PER_HOST_TOP = 3          # diversity: at most 3 per host element


def load_ensemble():
    """Stack 4-seed predictions and return mu, sigma, targets."""
    preds = []
    for d in SEED_DIRS:
        p = RESULTS / d / "test_predictions.npz"
        arr = np.load(p)
        preds.append(arr["preds"])
        targets = arr["targets"]
    preds = np.stack(preds, axis=0)
    return preds.mean(0), preds.std(0), targets


def main():
    print("Loading dataset + 4-seed ensemble...")
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    _, _, test_idx = split_indices(len(data), 0.8, 0.1, 42)
    print(f"  {len(test_idx)} test samples (held-out screening pool)")

    mu, sigma, targets = load_ensemble()
    print(f"  ensemble mu: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"  ensemble sigma: [{sigma.min():.4f}, {sigma.max():.4f}]")

    # apply temperature scaling from uq_calibration.json
    cal = json.load(open(RESULTS / "uq_calibration.json"))
    tau = cal["tau"]
    sigma_cal = sigma * tau
    print(f"  temperature τ = {tau:.3f}  →  calibrated σ ∈ [{sigma_cal.min():.3f}, {sigma_cal.max():.3f}]")

    # build candidate table
    candidates = []
    for i, idx in enumerate(test_idx):
        m = data[idx]["metadata"]
        candidates.append({
            "test_idx": int(i),
            "host": m["host"],
            "dopant": m["dopant"],
            "site": m.get("site", ""),
            "defecttype": m.get("defecttype", ""),
            "natoms": int(m.get("natoms", 0)),
            "mu_pred": float(mu[i]),
            "sigma_cal": float(sigma_cal[i]),
            "ef_true": float(targets[i]),
        })

    # ── Filter ────────────────────────────────────────────────────────
    print(f"\nFiltering pool (Ef≤{EF_MAX_EV}, σ≤{SIGMA_MAX_EV})...")
    pre_filter = len(candidates)
    pool = [c for c in candidates
            if c["mu_pred"] <= EF_MAX_EV and c["sigma_cal"] <= SIGMA_MAX_EV]
    print(f"  passed Ef + σ filter: {len(pool)} / {pre_filter}")

    # diversify: at most PER_HOST_TOP per host
    by_host = {}
    for c in sorted(pool, key=lambda x: x["mu_pred"]):
        h = c["host"]
        by_host.setdefault(h, []).append(c)
    diversified = []
    for h, lst in by_host.items():
        diversified.extend(lst[:PER_HOST_TOP])
    diversified.sort(key=lambda x: x["mu_pred"])
    print(f"  after per-host diversification (top-{PER_HOST_TOP}): {len(diversified)}")

    # take top-15 overall
    recommendations = diversified[:15]
    print(f"\n=== Top-15 Recommended Defects for Synthesis ===")
    print(f"{'rank':>4}  {'host':<8} {'dopant':<6} {'type':<14}"
          f" {'Ef±σ (predicted)':<22}  {'Ef true (DFT)':<14}  {'|err|':>6}")
    for r, c in enumerate(recommendations, 1):
        ef_str = f"{c['mu_pred']:+.3f} ± {c['sigma_cal']:.3f}"
        print(f"{r:>4}  {c['host']:<8} {c['dopant']:<6} "
              f"{c['defecttype']:<14} {ef_str:<22}  "
              f"{c['ef_true']:>+8.3f}      "
              f"{abs(c['mu_pred']-c['ef_true']):>5.3f}")

    # ── Verification stats ────────────────────────────────────────────
    rec_mu = np.array([c["mu_pred"] for c in recommendations])
    rec_sigma = np.array([c["sigma_cal"] for c in recommendations])
    rec_true = np.array([c["ef_true"] for c in recommendations])
    rec_err = np.abs(rec_mu - rec_true)
    rec_z = (rec_true - rec_mu) / np.maximum(rec_sigma, 1e-3)

    in_2sigma = float((np.abs(rec_z) <= 2).mean())
    rec_mae = float(rec_err.mean())
    rec_max_err = float(rec_err.max())

    # how many of recommendations are actually <= EF_MAX_EV (true positives)?
    actually_low_energy = float((rec_true <= EF_MAX_EV).mean())

    print(f"\n=== Verification (n={len(recommendations)} recommendations) ===")
    print(f"  Recommendation MAE       : {rec_mae:.4f} eV")
    print(f"  Worst-case error         : {rec_max_err:.4f} eV")
    print(f"  Within 2σ confidence     : {in_2sigma*100:.1f}%")
    print(f"  Actually low-Ef (≤1 eV)  : {actually_low_energy*100:.1f}%")

    # baseline: random 15 from pool
    rng = np.random.default_rng(42)
    rand_idx = rng.choice(len(candidates), 15, replace=False)
    rand_mu = np.array([candidates[i]["mu_pred"] for i in rand_idx])
    rand_true = np.array([candidates[i]["ef_true"] for i in rand_idx])
    rand_low_ef = float((rand_true <= EF_MAX_EV).mean())
    print(f"\nReference: 15 random samples from same pool")
    print(f"  Random low-Ef rate       : {rand_low_ef*100:.1f}%")
    print(f"  Improvement              : {(actually_low_energy/max(rand_low_ef,1e-6) - 1)*100:+.1f}%")

    # save results
    out = {
        "thresholds": {
            "ef_max_eV": EF_MAX_EV,
            "sigma_max_eV": SIGMA_MAX_EV,
            "per_host_top": PER_HOST_TOP,
        },
        "tau_calibration": tau,
        "pool_size": len(candidates),
        "after_ef_sigma_filter": len(pool),
        "after_diversification": len(diversified),
        "recommendations": recommendations,
        "verification": {
            "n_recommendations": len(recommendations),
            "rec_mae_eV": rec_mae,
            "max_err_eV": rec_max_err,
            "frac_within_2sigma": in_2sigma,
            "frac_actually_low_ef": actually_low_energy,
            "random_baseline_low_ef": rand_low_ef,
        },
    }
    out_json = RESULTS / "hts_demo.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_json}")

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: parity plot of all test set, recommendations highlighted
    ax = axes[0]
    ax.scatter(targets, mu, s=8, alpha=0.25, c="#888", label=f"All test (n={len(targets)})")
    rec_idx = [c["test_idx"] for c in recommendations]
    ax.errorbar(targets[rec_idx], mu[rec_idx],
                yerr=sigma_cal[rec_idx],
                fmt="s", color="#1f77b4", ecolor="#1f77b4",
                ms=8, capsize=3, lw=1.2,
                label=f"Top-15 recommendations")
    lo, hi = min(targets.min(), mu.min())-0.2, max(targets.max(), mu.max())+0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Ground-truth Ef (eV)", fontsize=11)
    ax.set_ylabel("Predicted Ef (eV)", fontsize=11)
    ax.set_title("Screening pool + recommendations (parity)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel B: cumulative low-Ef rate vs ranking
    ax = axes[1]
    ranked_by_pred = sorted(candidates, key=lambda c: c["mu_pred"])
    cum_low_pred = np.cumsum(
        [1.0 if c["ef_true"] <= EF_MAX_EV else 0.0 for c in ranked_by_pred]
    ) / np.arange(1, len(ranked_by_pred) + 1)
    ranked_by_random = list(candidates)
    rng2 = np.random.default_rng(0)
    rng2.shuffle(ranked_by_random)
    cum_low_rand = np.cumsum(
        [1.0 if c["ef_true"] <= EF_MAX_EV else 0.0 for c in ranked_by_random]
    ) / np.arange(1, len(ranked_by_random) + 1)
    n = np.arange(1, len(ranked_by_pred) + 1)
    ax.plot(n, cum_low_pred * 100, "-", color="#1f77b4", lw=2,
            label="Ranked by predicted Ef (ascending)")
    ax.plot(n, cum_low_rand * 100, "--", color="#d62728", lw=1.5,
            label="Random baseline")
    ax.axvline(15, color="gray", ls=":", lw=1)
    ax.text(16, 80, "top-15", fontsize=9, color="gray")
    ax.set_xlabel("Number of candidates inspected", fontsize=11)
    ax.set_ylabel(f"Hit rate (%, true Ef ≤ {EF_MAX_EV} eV)", fontsize=11)
    ax.set_title("Cumulative low-Ef hit rate vs ranking strategy", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(ranked_by_pred))

    fig.tight_layout()
    out_fig = FIG_DIR / "fig_hts_demo.png"
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"figure saved -> {out_fig}")


if __name__ == "__main__":
    main()
