#!/usr/bin/env python3
"""
Per-dopant MAE mapped onto the periodic table.
Generates Fig. X for the DART paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from collections import Counter
from pathlib import Path

# ── Periodic table layout (row, col) for each element ──────────────────
# Standard 18-column layout
PT_LAYOUT = {
    # Row 1
    'H':  (0, 0),  'He': (0, 17),
    # Row 2
    'Li': (1, 0),  'Be': (1, 1),
    'B':  (1, 12), 'C':  (1, 13), 'N':  (1, 14), 'O':  (1, 15),
    'F':  (1, 16), 'Ne': (1, 17),
    # Row 3
    'Na': (2, 0),  'Mg': (2, 1),
    'Al': (2, 12), 'Si': (2, 13), 'P':  (2, 14), 'S':  (2, 15),
    'Cl': (2, 16), 'Ar': (2, 17),
    # Row 4
    'K':  (3, 0),  'Ca': (3, 1),
    'Sc': (3, 2),  'Ti': (3, 3),  'V':  (3, 4),  'Cr': (3, 5),
    'Mn': (3, 6),  'Fe': (3, 7),  'Co': (3, 8),  'Ni': (3, 9),
    'Cu': (3, 10), 'Zn': (3, 11), 'Ga': (3, 12), 'Ge': (3, 13),
    'As': (3, 14), 'Se': (3, 15), 'Br': (3, 16), 'Kr': (3, 17),
    # Row 5
    'Rb': (4, 0),  'Sr': (4, 1),
    'Y':  (4, 2),  'Zr': (4, 3),  'Nb': (4, 4),  'Mo': (4, 5),
    'Tc': (4, 6),  'Ru': (4, 7),  'Rh': (4, 8),  'Pd': (4, 9),
    'Ag': (4, 10), 'Cd': (4, 11), 'In': (4, 12), 'Sn': (4, 13),
    'Sb': (4, 14), 'Te': (4, 15), 'I':  (4, 16), 'Xe': (4, 17),
    # Row 6
    'Cs': (5, 0),  'Ba': (5, 1),
    'La': (5, 2),  'Hf': (5, 3),  'Ta': (5, 4),  'W':  (5, 5),
    'Re': (5, 6),  'Os': (5, 7),  'Ir': (5, 8),  'Pt': (5, 9),
    'Au': (5, 10), 'Hg': (5, 11), 'Tl': (5, 12), 'Pb': (5, 13),
    'Bi': (5, 14), 'Po': (5, 15), 'At': (5, 16), 'Rn': (5, 17),
    # Row 7
    'Fr': (6, 0),  'Ra': (6, 1),
    'Ac': (6, 2),
    # Lanthanides (row 8, offset)
    'Ce': (8, 3),  'Pr': (8, 4),  'Nd': (8, 5),  'Pm': (8, 6),
    'Sm': (8, 7),  'Eu': (8, 8),  'Gd': (8, 9),  'Tb': (8, 10),
    'Dy': (8, 11), 'Ho': (8, 12), 'Er': (8, 13), 'Tm': (8, 14),
    'Yb': (8, 15), 'Lu': (8, 16),
}

# ── Load data ──────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent
with open(root / "paper_error_analysis.json") as f:
    data = json.load(f)

dopant_mae = data["by_dopant"]  # {element: float}
dopant_counts = Counter(s["dopant"] for s in data["per_sample"])

# ── Build arrays ───────────────────────────────────────────────────────
elements = sorted(dopant_mae.keys())
mae_vals = [dopant_mae[el] for el in elements]

print(f"Dopants: {len(elements)}, MAE range: [{min(mae_vals):.3f}, {max(mae_vals):.3f}]")

# ── Color mapping ──────────────────────────────────────────────────────
# Use a perceptually uniform diverging colormap
# Low MAE = blue (good), High MAE = red (bad)
vmin, vmax = 0.0, 1.4  # cap for visual clarity (Ta=1.34 is max)
cmap = plt.cm.RdYlBu_r  # reversed: blue=low, red=high
# Actually we want low=good(blue), high=bad(red) → RdYlBu reversed is red-low, blue-high
# Let's use RdYlGn: green=low(good), red=high(bad)
cmap = plt.cm.RdYlGn_r  # green=low MAE, red=high MAE ... actually _r reverses
# RdYlGn: Red(low) → Yellow(mid) → Green(high). _r reverses to Green(low) → Red(high)
# We want: Green(low MAE, good) → Red(high MAE, bad) → that IS RdYlGn_r!
# Wait: RdYlGn goes Red → Yellow → Green. RdYlGn_r goes Green → Yellow → Red.
# So norm(0) = Green (good), norm(1) = Red (bad). Perfect.
cmap = plt.cm.RdYlGn_r
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# ── Plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5.5))

cell_w, cell_h = 1.0, 1.0
pad = 0.06

for el in elements:
    if el not in PT_LAYOUT:
        print(f"  Warning: {el} not in PT_LAYOUT, skipping")
        continue
    row, col = PT_LAYOUT[el]
    mae = dopant_mae[el]
    n = dopant_counts.get(el, 0)

    # Cell position (y is inverted so row 0 is at top)
    x = col * (cell_w + pad)
    y = -row * (cell_h + pad)

    # Color
    color = cmap(norm(min(mae, vmax)))

    # Draw cell
    rect = FancyBboxPatch(
        (x, y), cell_w, cell_h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor='#444444', linewidth=0.8
    )
    ax.add_patch(rect)

    # Text: element symbol (center)
    # Choose text color based on background brightness
    rgb = mcolors.to_rgb(color)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    text_color = 'white' if luminance < 0.45 else 'black'

    ax.text(x + cell_w / 2, y + cell_h * 0.62, el,
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=text_color, family='sans-serif')

    # MAE value below symbol
    ax.text(x + cell_w / 2, y + cell_h * 0.28, f'{mae:.2f}',
            ha='center', va='center', fontsize=7,
            color=text_color, family='sans-serif')

    # Sample count (top-right corner, small)
    ax.text(x + cell_w * 0.88, y + cell_h * 0.88, f'n={n}',
            ha='right', va='center', fontsize=4.5,
            color=text_color, alpha=0.7, family='sans-serif')

# Draw empty cells for context (elements NOT in our dataset)
context_elements = set(PT_LAYOUT.keys()) - set(elements)
for el in context_elements:
    row, col = PT_LAYOUT[el]
    x = col * (cell_w + pad)
    y = -row * (cell_h + pad)
    rect = FancyBboxPatch(
        (x, y), cell_w, cell_h,
        boxstyle="round,pad=0.02",
        facecolor='#f0f0f0', edgecolor='#cccccc', linewidth=0.5
    )
    ax.add_patch(rect)
    ax.text(x + cell_w / 2, y + cell_h / 2, el,
            ha='center', va='center', fontsize=8,
            color='#bbbbbb', family='sans-serif')

# ── Colorbar ───────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.25, 0.02, 0.50, 0.03])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('MAE (eV)', fontsize=11)
cbar.ax.tick_params(labelsize=9)

# ── Annotations ────────────────────────────────────────────────────────
# Highlight key findings
# Best: Bi (0.059), Fe (0.137), Pb (0.157)
# Worst: Ta (1.339), Mn (1.151), Sc (0.768)

ax.set_xlim(-0.5, 18 * (cell_w + pad))
ax.set_ylim(-9.5 * (cell_h + pad), cell_h + 0.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Per-dopant prediction error (MAE) on IMP2D test set',
             fontsize=13, fontweight='bold', pad=10)

plt.tight_layout(rect=[0, 0.06, 1, 1])

# Save
out_dir = root / "figures"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "periodic_table_mae.pdf", dpi=300, bbox_inches='tight')
fig.savefig(out_dir / "periodic_table_mae.png", dpi=300, bbox_inches='tight')
print(f"Saved to {out_dir / 'periodic_table_mae.pdf'}")
plt.close()
