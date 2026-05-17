"""
Generate P1 Screening Heatmap Visualization.
Creates a 44x65 heatmap of predicted formation energies.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "results/p1_screening"
FIGURES_DIR = "results/p1_screening/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load prediction matrix
print("Loading screening data...")
with open(os.path.join(OUTPUT_DIR, "summary.json")) as f:
    summary = json.load(f)

hosts = summary["hosts"]
dopants = summary["dopants"]

# Read CSV into matrix
pred_matrix = np.full((len(hosts), len(dopants)), np.nan)
with open(os.path.join(OUTPUT_DIR, "screening_pred_ef.csv")) as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header
    for i, row in enumerate(reader):
        for j, val in enumerate(row[1:]):
            if val:
                pred_matrix[i, j] = float(val)

unc_matrix = np.full((len(hosts), len(dopants)), np.nan)
with open(os.path.join(OUTPUT_DIR, "screening_uncertainty.csv")) as f:
    reader = csv.reader(f)
    next(reader)
    for i, row in enumerate(reader):
        for j, val in enumerate(row[1:]):
            if val:
                unc_matrix[i, j] = float(val)

# === Figure 1: Full Screening Heatmap ===
print("Generating full screening heatmap...")
fig, ax = plt.subplots(figsize=(22, 14))

# Use diverging colormap centered at 0
vmin, vmax = np.nanpercentile(pred_matrix, [2, 98])
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

im = ax.imshow(pred_matrix, aspect="auto", cmap="RdBu_r", norm=norm,
               interpolation="nearest")
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Predicted Formation Energy (eV)", fontsize=12)

ax.set_xticks(range(len(dopants)))
ax.set_xticklabels(dopants, rotation=90, fontsize=6)
ax.set_yticks(range(len(hosts)))
ax.set_yticklabels(hosts, fontsize=7)
ax.set_xlabel("Dopant Element", fontsize=12)
ax.set_ylabel("Host Material", fontsize=12)
ax.set_title("DART Predicted Formation Energy Map (44 Hosts x 65 Dopants)", fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "screening_heatmap_full.pdf"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "screening_heatmap_full.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {FIGURES_DIR}/screening_heatmap_full.pdf")

# === Figure 2: Uncertainty Heatmap ===
print("Generating uncertainty heatmap...")
fig, ax = plt.subplots(figsize=(22, 14))

im = ax.imshow(unc_matrix, aspect="auto", cmap="YlOrRd",
               interpolation="nearest", vmin=0, vmax=np.nanpercentile(unc_matrix, 95))
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Ensemble Uncertainty (eV)", fontsize=12)

ax.set_xticks(range(len(dopants)))
ax.set_xticklabels(dopants, rotation=90, fontsize=6)
ax.set_yticks(range(len(hosts)))
ax.set_yticklabels(hosts, fontsize=7)
ax.set_xlabel("Dopant Element", fontsize=12)
ax.set_ylabel("Host Material", fontsize=12)
ax.set_title("DART Prediction Uncertainty Map (Ensemble Std)", fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "uncertainty_heatmap.pdf"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "uncertainty_heatmap.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {FIGURES_DIR}/uncertainty_heatmap.pdf")

# === Figure 3: Host-averaged profile ===
print("Generating host-averaged profile...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean Ef per host (averaged across dopants)
host_mean_ef = np.nanmean(pred_matrix, axis=1)
host_std_ef = np.nanstd(pred_matrix, axis=1)
sorted_idx = np.argsort(host_mean_ef)

ax = axes[0]
ax.barh(range(len(hosts)), host_mean_ef[sorted_idx], xerr=host_std_ef[sorted_idx],
        color="steelblue", alpha=0.7, ecolor="gray", capsize=2)
ax.set_yticks(range(len(hosts)))
ax.set_yticklabels([hosts[i] for i in sorted_idx], fontsize=6)
ax.set_xlabel("Mean Predicted Ef (eV)")
ax.set_title("Host Materials (sorted by mean Ef)")
ax.axvline(0, color="red", linestyle="--", alpha=0.5)

# Mean Ef per dopant (averaged across hosts)
dopant_mean_ef = np.nanmean(pred_matrix, axis=0)
dopant_std_ef = np.nanstd(pred_matrix, axis=0)
sorted_idx_d = np.argsort(dopant_mean_ef)

ax = axes[1]
ax.barh(range(len(dopants)), dopant_mean_ef[sorted_idx_d],
        xerr=dopant_std_ef[sorted_idx_d],
        color="darkorange", alpha=0.7, ecolor="gray", capsize=1)
ax.set_yticks(range(len(dopants)))
ax.set_yticklabels([dopants[i] for i in sorted_idx_d], fontsize=5)
ax.set_xlabel("Mean Predicted Ef (eV)")
ax.set_title("Dopant Elements (sorted by mean Ef)")
ax.axvline(0, color="red", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "host_dopant_profiles.pdf"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "host_dopant_profiles.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {FIGURES_DIR}/host_dopant_profiles.pdf")

# === Figure 4: Top candidates (low Ef + low uncertainty) ===
print("Generating candidate ranking...")
fig, ax = plt.subplots(figsize=(10, 8))

# Flatten matrices and filter non-nan
valid_mask = ~np.isnan(pred_matrix) & ~np.isnan(unc_matrix)
flat_pred = pred_matrix[valid_mask]
flat_unc = unc_matrix[valid_mask]

scatter = ax.scatter(flat_pred, flat_unc, c=flat_pred, cmap="RdBu_r",
                    alpha=0.5, s=10, edgecolors="none")
ax.set_xlabel("Predicted Formation Energy (eV)", fontsize=12)
ax.set_ylabel("Ensemble Uncertainty (eV)", fontsize=12)
ax.set_title("Screening Landscape: Ef vs Uncertainty", fontsize=14)
plt.colorbar(scatter, ax=ax, label="Ef (eV)")

# Mark the "sweet spot" (low Ef + low uncertainty)
ax.axhline(0.1, color="green", linestyle="--", alpha=0.5, label="Low uncertainty threshold")
ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="Ef = 0")
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ef_vs_uncertainty.pdf"), dpi=150, bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "ef_vs_uncertainty.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: {FIGURES_DIR}/ef_vs_uncertainty.pdf")

print("\nAll figures generated successfully!")
print(f"Output directory: {FIGURES_DIR}/")
