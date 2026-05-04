"""C17 deliverable: ranked DFT priority queue (CSV) + analysis.

Even without DFT access, this is a production-ready output showing
which (host, dopant, defect_type) combinations should be computed
NEXT to maximize information gain about the model.

Ranks the 287 generated candidates by:
  combined_score = (sigma_cal_normalized × adversarial_weight)
                 - (penalty_for_extreme_mu)
                 + (chemistry_diversity_bonus)

Outputs
-------
- results/c17_dft_priority_queue.csv (ranked list)
- paper/figures/fig_c17_priority_queue.png  (σ vs μ scatter)
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ADVERSARIAL_W = 0.6
EXTREME_PENALTY = 0.2     # penalty if |μ| > 5 eV (likely unphysical)
DIVERSITY_W = 0.2

PRED_PATH = RESULTS / "candidates_c17_predictions.json"
META_PATH = RESULTS / "candidates_c17_meta.json"


def main():
    preds = json.load(open(PRED_PATH))["predictions"]
    print(f"Loaded {len(preds)} candidate predictions")

    sigma = np.array([p["sigma_cal"] for p in preds])
    mu = np.array([p["mu"] for p in preds])
    hosts = [p["host"] for p in preds]
    dopants = [p["dopant"] for p in preds]
    defect_types = [p["defect_type"] for p in preds]

    # normalise σ to [0, 1]
    sigma_n = (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-9)

    # extreme-μ penalty
    extreme_mask = np.abs(mu) > 5.0
    mu_penalty = np.where(extreme_mask, 1.0, 0.0)

    # diversity bonus: rare host = +1, rare dopant = +1
    host_counts = Counter(hosts)
    dop_counts = Counter(dopants)
    div = np.array([
        1.0 / np.sqrt(host_counts[h]) + 1.0 / np.sqrt(dop_counts[d])
        for h, d in zip(hosts, dopants)
    ])
    div_n = (div - div.min()) / (div.max() - div.min() + 1e-9)

    # combined score
    score = ADVERSARIAL_W * sigma_n - EXTREME_PENALTY * mu_penalty + DIVERSITY_W * div_n

    # rank
    order = np.argsort(-score)

    # write CSV
    csv_path = RESULTS / "c17_dft_priority_queue.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "host", "dopant", "defect_type", "natoms",
                    "predicted_Ef_eV", "calibrated_sigma_eV",
                    "adversarial_score", "extreme", "score"])
        for rank, idx in enumerate(order, 1):
            p = preds[idx]
            w.writerow([rank, p["host"], p["dopant"], p["defect_type"],
                        p["natoms"], f"{mu[idx]:.4f}", f"{sigma[idx]:.4f}",
                        f"{sigma_n[idx]:.4f}",
                        "Y" if extreme_mask[idx] else "N",
                        f"{score[idx]:.4f}"])
    print(f"saved -> {csv_path}")

    # print top 20
    print(f"\n=== Top-20 DFT priority queue ===")
    print(f"{'rank':>4}  {'host':<8} {'dopant':<6} {'type':<14} "
          f"{'pred_Ef':>9} {'σ_cal':>9} {'score':>7}")
    for r, idx in enumerate(order[:20], 1):
        p = preds[idx]
        print(f"{r:>4}  {p['host']:<8} {p['dopant']:<6} "
              f"{p['defect_type']:<14} "
              f"{mu[idx]:>+9.3f} {sigma[idx]:>9.3f} {score[idx]:>7.3f}")

    # figure: σ vs μ
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(mu, sigma, c=score, cmap="plasma",
                     s=20, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="priority score")
    # highlight top-15
    top15 = order[:15]
    ax.scatter(mu[top15], sigma[top15], facecolors="none", edgecolors="red",
                s=140, linewidths=2, label="Top-15 priority")
    ax.set_xlabel("Predicted Ef (eV)")
    ax.set_ylabel("Calibrated σ (eV)")
    ax.set_title(f"C17 DFT priority queue ({len(preds)} candidates)")
    ax.axvline(5, color="gray", ls=":", alpha=0.5)
    ax.axvline(-5, color="gray", ls=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_fig = FIG_DIR / "fig_c17_priority_queue.png"
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"figure saved -> {out_fig}")


if __name__ == "__main__":
    main()
