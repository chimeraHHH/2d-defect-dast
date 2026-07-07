"""Figures for the blind cross-code OOD benchmark section.

Outputs
-------
- paper/figures/fig_blind_ood_parity.png  (parity plot, per-host colours,
  sigma_cal error bars, raw + per-host-offset-corrected panels)
- paper/figures/fig_blind_ood_kshot.png   (k-shot offset calibration curves
  + UQ coverage bar chart)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

pred_d = json.load(open(ROOT / "results" / "blind_ood_predictions.json"))
ana = json.load(open(ROOT / "results" / "blind_ood_analysis.json"))

rows = [r for r in pred_d["predictions"] if r["usable_flag"] == "yes"]
r81 = next(r for r in pred_d["predictions"] if r["sample_id"] == 81)

HOSTS = ("graphene", "VS2", "CrS2")
COLORS = {"graphene": "#444444", "VS2": "#1f77b4", "CrS2": "#d62728"}
LABELS = {"graphene": "graphene", "VS2": r"VS$_2$", "CrS2": r"CrS$_2$"}

pred = np.array([r["E_pred_eV"] for r in rows])
dft = np.array([r["E_dft_eV"] for r in rows])
sig = np.array([r["sigma_cal_eV"] for r in rows])
host = np.array([r["host"] for r in rows])
err = pred - dft

# ---------------------------------------------------------------- parity
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharey=False)

for ax, corrected in zip(axes, (False, True)):
    for h in HOSTS:
        m = host == h
        y = pred[m] - (err[m].mean() if corrected else 0.0)
        ax.errorbar(dft[m], y, yerr=sig[m], fmt="o", ms=5, lw=0.8,
                    capsize=2, alpha=0.85, color=COLORS[h],
                    label=LABELS[h])
    # sample 81 (needs_review) on raw panel only
    if not corrected:
        ax.scatter([r81["E_dft_eV"]], [r81["E_pred_eV"]], marker="x",
                   s=70, color="purple", zorder=5,
                   label="#81 (flagged)")
    lims = (-4.5, 8.5)
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r"$E_f^{\mathrm{DFT}}$ (GPAW) [eV]")
    ax.grid(alpha=0.25)

axes[0].set_ylabel(r"$E_f^{\mathrm{pred}}$ [eV]")
g = ana["global_raw"]
axes[0].set_title(
    f"raw: MAE {g['mae']:.2f} eV, bias {g['bias']:+.2f} eV", fontsize=10)
p = ana["pooled_offset_loo"]
axes[1].set_title(
    f"per-host offset removed: MAE {p['mae']:.2f} eV (LOO)", fontsize=10)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Blind cross-code OOD benchmark (38 GPAW samples, "
             "4-seed ensemble)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out1 = FIGS / "fig_blind_ood_parity.png"
fig.savefig(out1, dpi=220)
print(f"saved -> {out1}")

# --------------------------------------------------- k-shot + coverage
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))

ks = [1, 2, 3, 5]
for h in HOSTS:
    blk = ana["per_host"][h]
    means = [blk["kshot_offset"][f"k={k}"]["mae_mean"] for k in ks]
    stds = [blk["kshot_offset"][f"k={k}"]["mae_std"] for k in ks]
    ax1.errorbar(ks, means, yerr=stds, marker="o", ms=4, capsize=3,
                 color=COLORS[h], label=LABELS[h])
    ax1.axhline(blk["mae"], color=COLORS[h], ls=":", lw=1, alpha=0.6)
ax1.set_xlabel("k (DFT calculations used for offset)")
ax1.set_ylabel("test MAE [eV]")
ax1.set_xticks(ks)
ax1.set_title("k-shot per-host offset calibration\n"
              "(dotted: raw uncorrected MAE)", fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.25)

# coverage bars
cov_raw = ana["uq"]["coverage"]
cov_deb = ana["uq"]["coverage_after_loo_debias"]
nominal = [0.68, 0.90, 0.95]
labels = ["68%", "90%", "95%"]
x = np.arange(3)
w = 0.28
ax2.bar(x - w, nominal, w, color="#bbbbbb", label="nominal")
ax2.bar(x, [cov_raw["nominal_68"], cov_raw["nominal_90"],
            cov_raw["nominal_95"]], w, color="#1f77b4",
        label="empirical (raw)")
ax2.bar(x + w, [cov_deb["nominal_68"], cov_deb["nominal_90"],
                cov_deb["nominal_95"]], w, color="#2ca02c",
        label="empirical (offset-corrected)")
ax2.set_xticks(x, labels)
ax2.set_ylim(0, 1.12)
ax2.set_ylabel("coverage")
ax2.set_title(
    f"UQ coverage under blind OOD\n"
    f"(mean $\\sigma_{{cal}}$ {ana['uq']['mean_sigma_cal']:.2f} eV = "
    f"{ana['uq']['sigma_amplification']:.1f}$\\times$ ID level)",
    fontsize=10)
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(alpha=0.25, axis="y")

fig.tight_layout()
out2 = FIGS / "fig_blind_ood_kshot.png"
fig.savefig(out2, dpi=220)
print(f"saved -> {out2}")
