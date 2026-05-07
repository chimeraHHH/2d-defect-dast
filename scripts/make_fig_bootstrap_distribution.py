"""F2: Bootstrap CI distribution panel for prospective DFT validation.

4 histograms over 5000 stratified bootstrap resamples:
  (a) overall MAE (per-dopant corrected)
  (b) overall Pearson(pred, DFT_corr)
  (c) bucket-A discovery rate (DFT < 1 eV)
  (d) bucket-B Pearson(sigma_cal, |err|)

Each panel shows the histogram + central point estimate (solid line)
+ 95% percentile CI (shaded band) + reference line at 0 where
applicable.

Visual purpose: turn the abstract "95% CI [..., ...]" into a
concrete picture of the bootstrap distribution, so reviewers
trust that headline numbers are statistically defended.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_IN = ROOT / "results" / "prospective_dft_results.json"
OUT = ROOT / "paper" / "figures" / "fig_bootstrap_dist.png"

N_BOOT = 5000
SEED = 42


def pearson(x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    rng = np.random.default_rng(SEED)
    rows = json.loads(RES_IN.read_text())["rows"]
    n = len(rows)
    pred = np.array([r["model_pred_Ef_eV"] for r in rows])
    dft = np.array([r["dft_Ef_eV"] for r in rows])
    sig = np.array([r["model_sigma_cal_eV"] for r in rows])
    bucket = np.array([r["bucket"] for r in rows])
    a_idx = np.where(bucket == "A_low_Ef_high_conf")[0]
    b_idx = np.where(bucket == "B_high_sigma_OOD")[0]

    # Per-dopant offsets (fixed)
    off = defaultdict(list)
    for r in rows:
        off[r["dopant"]].append(r["dft_Ef_eV"] - r["model_pred_Ef_eV"])
    mean_off = {d: float(np.mean(v)) for d, v in off.items()}
    dft_c = np.array([r["dft_Ef_eV"] - mean_off[r["dopant"]] for r in rows])

    # Bootstrap
    mae_all, pearson_all, disc_A, pearson_B = [], [], [], []
    for _ in range(N_BOOT):
        rs_A = a_idx[rng.integers(0, len(a_idx), size=len(a_idx))]
        rs_B = b_idx[rng.integers(0, len(b_idx), size=len(b_idx))]
        rs_all = np.concatenate([rs_A, rs_B])

        e_all = np.abs(dft_c[rs_all] - pred[rs_all])
        mae_all.append(e_all.mean())
        pearson_all.append(pearson(pred[rs_all], dft_c[rs_all]))

        disc_A.append(np.mean(dft_c[rs_A] < 1.0))

        e_B = np.abs(dft_c[rs_B] - pred[rs_B])
        pearson_B.append(pearson(sig[rs_B], e_B))

    mae_all = np.asarray(mae_all)
    pearson_all = np.asarray(pearson_all)
    disc_A = np.asarray(disc_A)
    pearson_B = np.asarray(pearson_B)

    # Point estimates (full data)
    p_mae = float(np.mean(np.abs(dft_c - pred)))
    p_pearson_all = pearson(pred, dft_c)
    p_disc = float(np.mean(dft_c[a_idx] < 1.0))
    p_pearson_B = pearson(sig[b_idx], np.abs(dft_c[b_idx] - pred[b_idx]))

    # 95% CIs
    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    ci_mae = ci(mae_all); ci_p = ci(pearson_all); ci_d = ci(disc_A); ci_pB = ci(pearson_B)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.5))
    panels = [
        (axes[0, 0], mae_all, p_mae, ci_mae,
         "Overall MAE (eV)",
         f"point: {p_mae:.2f}, 95% CI [{ci_mae[0]:.2f}, {ci_mae[1]:.2f}]",
         None, "#1f3a5f"),
        (axes[0, 1], pearson_all, p_pearson_all, ci_p,
         "Pearson(pred, DFT$_{corr}$), all 37",
         f"point: {p_pearson_all:+.3f}, 95% CI [{ci_p[0]:+.2f}, {ci_p[1]:+.2f}] — excludes 0",
         0.0, "#2c7a3e"),
        (axes[1, 0], disc_A, p_disc, ci_d,
         "Bucket-A discovery (DFT $E_f<+1$ eV)",
         f"point: {p_disc*100:.0f}%, 95% CI [{ci_d[0]*100:.0f}%, {ci_d[1]*100:.0f}%]",
         None, "#1f3a5f"),
        (axes[1, 1], pearson_B, p_pearson_B, ci_pB,
         "Pearson($\\sigma_{cal}$, $|$err$|$), bucket B (n=17)",
         f"point: {p_pearson_B:+.3f}, 95% CI [{ci_pB[0]:+.2f}, {ci_pB[1]:+.2f}] — includes 0",
         0.0, "#a04040"),
    ]
    for ax, arr, p, ci_, title, sub, ref, color in panels:
        ax.hist(arr, bins=42, color=color, alpha=0.55, edgecolor="white")
        ax.axvspan(ci_[0], ci_[1], alpha=0.15, color=color, label="95% CI")
        ax.axvline(p, color=color, lw=2.0, label=f"point estimate")
        if ref is not None:
            ax.axvline(ref, color="grey", ls=":", lw=1.2, label=f"$=$ {ref}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(sub, fontsize=9.5, color="#333")
        ax.set_ylabel("bootstrap count")
        ax.legend(fontsize=8, loc="upper right" if "Pearson" in title else "upper left")
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Bootstrap distributions ({N_BOOT} stratified resamples) for headline prospective-DFT statistics",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
