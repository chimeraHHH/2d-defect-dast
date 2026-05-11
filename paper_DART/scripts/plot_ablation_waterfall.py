#!/usr/bin/env python3
"""
Training pipeline ablation waterfall chart for the DART paper.
Shows cumulative improvement from each training recipe component.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
})

# ── Data from Table 2 ─────────────────────────────────────────────────
steps = [
    "MSE loss\nplateau LR",
    "→ MAE (L1) loss",
    "→ + cosine\nwarmup",
    "→ + 150 ep\n+ SWA",
]
maes = [0.516, 0.474, 0.426, 0.415]
stds = [0.006, 0.018, 0.011, 0.008]
deltas = [0, -0.042, -0.048, -0.011]  # improvement from previous

# Colors: first bar is baseline, rest are improvement steps
colors = ['#95a5a6']  # grey for baseline
colors += ['#27ae60' if d < 0 else '#e74c3c' for d in deltas[1:]]

# ── Plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(steps))
bars = ax.bar(x, maes, color=colors, width=0.55, edgecolor='white',
              linewidth=1.5, zorder=3)

# Error bars
ax.errorbar(x, maes, yerr=stds, fmt='none', ecolor='#333333',
            elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)

# Value labels
for i, (bar, mae, std) in enumerate(zip(bars, maes, stds)):
    # MAE value on top
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.008,
            f'{mae:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Delta annotation (except first)
    if i > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.02,
                f'{deltas[i]:+.3f}', ha='center', va='top', fontsize=9,
                color='white', fontweight='bold')

# ALIGNN reference line
ax.axhline(y=0.540, color='#e74c3c', linestyle='--', linewidth=1.2,
           alpha=0.7, zorder=1)
ax.text(len(steps) - 0.5, 0.543, 'ALIGNN (0.540)', ha='right', va='bottom',
        fontsize=9, color='#e74c3c', fontstyle='italic')

# Ensemble reference line
ax.axhline(y=0.359, color='#2980b9', linestyle=':', linewidth=1.2,
           alpha=0.7, zorder=1)
ax.text(len(steps) - 0.5, 0.362, 'Ensemble (0.359)', ha='right', va='bottom',
        fontsize=9, color='#2980b9', fontstyle='italic')

ax.set_xticks(x)
ax.set_xticklabels(steps, fontsize=9)
ax.set_ylabel("Test MAE (eV)", fontsize=11)
ax.set_title("Cumulative training pipeline ablation", fontweight='bold')
ax.set_ylim(0.30, 0.60)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, zorder=0)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "figures"
fig.savefig(out / "ablation_waterfall.pdf", dpi=300, bbox_inches='tight')
fig.savefig(out / "ablation_waterfall.png", dpi=300, bbox_inches='tight')
print(f"Saved ablation_waterfall to {out}")
plt.close()
