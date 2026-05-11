#!/usr/bin/env python3
"""
LOHO (Leave-One-Host-Out) Tier 1 bar chart for the DART paper.
Shows per-host MAE for G6-TMDs, with Mo vs W grouping.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
})

# ── Data from Table 3 ─────────────────────────────────────────────────
hosts = ['MoSe₂', 'MoTe₂', 'MoSSe', 'MoS₂', 'WTe₂', 'WSe₂', 'WS₂']
dart_mae = [0.377, 0.480, 0.484, 0.522, 0.506, 0.653, 0.759]
naive_mae = [2.874, 1.910, 2.019, 3.507, 2.005, 2.883, 3.253]
is_mo = [True, True, True, True, False, False, False]

# Sort by DART MAE
idx = np.argsort(dart_mae)
hosts = [hosts[i] for i in idx]
dart_mae = [dart_mae[i] for i in idx]
naive_mae = [naive_mae[i] for i in idx]
is_mo = [is_mo[i] for i in idx]

# ── Plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

x = np.arange(len(hosts))
width = 0.35

colors_dart = ['#4C78A8' if m else '#E45756' for m in is_mo]
colors_naive = ['#89B8D8' if m else '#F2A09F' for m in is_mo]

bars1 = ax.bar(x - width/2, dart_mae, width, color=colors_dart,
               edgecolor='white', linewidth=1.0, label='DART')
bars2 = ax.bar(x + width/2, naive_mae, width, color=colors_naive,
               edgecolor='white', linewidth=1.0, label='Naive', alpha=0.6)

# Value labels on DART bars
for bar, mae in zip(bars1, dart_mae):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f'{mae:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Tier 0 reference line
ax.axhline(y=0.407, color='#2CA02C', linestyle='--', linewidth=1.2,
           alpha=0.8, zorder=0)
ax.text(len(hosts) - 0.5, 0.407 + 0.06, 'Tier 0 (ID): 0.407',
        ha='right', va='bottom', fontsize=8, color='#2CA02C', fontstyle='italic')

# Mean OOD line
mean_ood = np.mean(dart_mae)
ax.axhline(y=mean_ood, color='#9467BD', linestyle=':', linewidth=1.2,
           alpha=0.8, zorder=0)
ax.text(len(hosts) - 0.5, mean_ood + 0.06, f'Tier 1 mean: {mean_ood:.3f}',
        ha='right', va='bottom', fontsize=8, color='#9467BD', fontstyle='italic')

ax.set_xticks(x)
ax.set_xticklabels(hosts, fontsize=10)
ax.set_ylabel("MAE (eV)", fontsize=11)
ax.set_title("Leave-One-Host-Out (LOHO) evaluation on G6-TMDs", fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(naive_mae) * 1.15)

# Legend
legend_elements = [
    Patch(facecolor='#4C78A8', label='DART (Mo-based)'),
    Patch(facecolor='#E45756', label='DART (W-based)'),
    Patch(facecolor='#89B8D8', alpha=0.6, label='Naive (Mo-based)'),
    Patch(facecolor='#F2A09F', alpha=0.6, label='Naive (W-based)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "figures"
fig.savefig(out / "loho_bar.pdf", dpi=300, bbox_inches='tight')
fig.savefig(out / "loho_bar.png", dpi=300, bbox_inches='tight')
print(f"Saved loho_bar to {out}")
plt.close()
