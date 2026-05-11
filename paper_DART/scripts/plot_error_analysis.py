#!/usr/bin/env python3
"""
Error analysis subplots for the DART paper (Fig. 2):
(a) By defect type — bar chart
(b) By Ef magnitude — bar chart with sample counts
(c) Predicted vs True scatter plot
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

root = Path(__file__).resolve().parent.parent
with open(root / "paper_error_analysis.json") as f:
    data = json.load(f)

# ── (a) By defect type ────────────────────────────────────────────────
bdt = data["by_defect_type"]

# ── (b) By Ef range ───────────────────────────────────────────────────
ber = data["by_ef_range"]
# Order the ranges properly
range_order = ["[-20,-2)", "[-2,0)", "[0,2)", "[2,5)", "[5,10)", "[10,20)"]
range_labels = ["<−2", "[−2,0)", "[0,2)", "[2,5)", "[5,10)", "≥10"]

# ── (c) Scatter: predicted vs true ────────────────────────────────────
ps = data["per_sample"]
y_true = np.array([float(s["eform_true"]) for s in ps])
y_pred = np.array([float(s["eform_pred"]) for s in ps])

# ── Create figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# Panel (a): Defect type bar chart
ax = axes[0]
types = ["adsorbate", "interstitial"]
labels = ["Adsorbate", "Interstitial"]
maes = [bdt[t]["mae"] for t in types]
ns = [bdt[t]["n"] for t in types]
colors = ["#4C78A8", "#E45756"]  # Vega blue/red
bars = ax.bar(labels, maes, color=colors, width=0.5, edgecolor='white', linewidth=1.2)
for bar, n, mae in zip(bars, ns, maes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{mae:.3f}\n(n={n})', ha='center', va='bottom', fontsize=9)
ax.set_ylabel("MAE (eV)")
ax.set_ylim(0, max(maes) * 1.35)
ax.set_title("(a) By defect type", fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel (b): Ef range bar chart
ax = axes[1]
range_maes = [ber[r]["mae"] for r in range_order]
range_ns = [ber[r]["n"] for r in range_order]
cmap_b = plt.cm.YlOrRd
color_norm = plt.Normalize(vmin=0, vmax=max(range_maes))
bar_colors = [cmap_b(color_norm(m)) for m in range_maes]
bars = ax.bar(range_labels, range_maes, color=bar_colors, width=0.65,
              edgecolor='white', linewidth=1.0)
for bar, n in zip(bars, range_ns):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f'n={n}', ha='center', va='bottom', fontsize=7, color='#555')
ax.set_xlabel("$E_f$ range (eV)")
ax.set_ylabel("MAE (eV)")
ax.set_title("(b) By $E_f$ magnitude", fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel (c): Scatter plot
ax = axes[2]
# Color by absolute error
abs_err = np.abs(y_true - y_pred)
sc = ax.scatter(y_true, y_pred, c=abs_err, cmap='RdYlGn_r', s=8, alpha=0.6,
                vmin=0, vmax=2.0, edgecolors='none', rasterized=True)
# Diagonal line
lims = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
ax.plot(lims, lims, 'k--', lw=1.0, alpha=0.5, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("DFT $E_f$ (eV)")
ax.set_ylabel("Predicted $E_f$ (eV)")
ax.set_title("(c) Predicted vs. DFT", fontweight='bold')
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Colorbar for scatter
cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("|Error| (eV)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# MAE annotation
mae_all = np.mean(abs_err)
rmse_all = np.sqrt(np.mean(abs_err**2))
r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
ax.text(0.05, 0.92, f'MAE = {mae_all:.3f} eV\nRMSE = {rmse_all:.3f} eV\n$R^2$ = {r2:.3f}',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()

out = root / "figures"
fig.savefig(out / "error_analysis.pdf", dpi=300, bbox_inches='tight')
fig.savefig(out / "error_analysis.png", dpi=300, bbox_inches='tight')
print(f"Saved error_analysis to {out}")
plt.close()
