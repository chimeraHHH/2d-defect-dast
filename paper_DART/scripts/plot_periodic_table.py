#!/usr/bin/env python3
"""
Per-dopant MAE mapped onto the periodic table (compact version for paper).
Only rows 1-6 shown; Lu placed in the La position footnote area.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from collections import Counter
from pathlib import Path

# ── Periodic table layout (row, col) — rows 0-5 only ────────────────
PT_LAYOUT = {
    'H':  (0, 0),  'He': (0, 17),
    'Li': (1, 0),  'Be': (1, 1),
    'B':  (1, 12), 'C':  (1, 13), 'N':  (1, 14), 'O':  (1, 15),
    'F':  (1, 16), 'Ne': (1, 17),
    'Na': (2, 0),  'Mg': (2, 1),
    'Al': (2, 12), 'Si': (2, 13), 'P':  (2, 14), 'S':  (2, 15),
    'Cl': (2, 16), 'Ar': (2, 17),
    'K':  (3, 0),  'Ca': (3, 1),
    'Sc': (3, 2),  'Ti': (3, 3),  'V':  (3, 4),  'Cr': (3, 5),
    'Mn': (3, 6),  'Fe': (3, 7),  'Co': (3, 8),  'Ni': (3, 9),
    'Cu': (3, 10), 'Zn': (3, 11), 'Ga': (3, 12), 'Ge': (3, 13),
    'As': (3, 14), 'Se': (3, 15), 'Br': (3, 16), 'Kr': (3, 17),
    'Rb': (4, 0),  'Sr': (4, 1),
    'Y':  (4, 2),  'Zr': (4, 3),  'Nb': (4, 4),  'Mo': (4, 5),
    'Tc': (4, 6),  'Ru': (4, 7),  'Rh': (4, 8),  'Pd': (4, 9),
    'Ag': (4, 10), 'Cd': (4, 11), 'In': (4, 12), 'Sn': (4, 13),
    'Sb': (4, 14), 'Te': (4, 15), 'I':  (4, 16), 'Xe': (4, 17),
    'Cs': (5, 0),  'Ba': (5, 1),
    'La': (5, 2),  'Hf': (5, 3),  'Ta': (5, 4),  'W':  (5, 5),
    'Re': (5, 6),  'Os': (5, 7),  'Ir': (5, 8),  'Pt': (5, 9),
    'Au': (5, 10), 'Hg': (5, 11), 'Tl': (5, 12), 'Pb': (5, 13),
    'Bi': (5, 14), 'Po': (5, 15), 'At': (5, 16), 'Rn': (5, 17),
    # Lu: place as a small annotation beside La
    'Lu': (5, 2.55),  # offset to the right of La — handled specially
}

# ── Load data ──────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent
with open(root / "paper_error_analysis.json") as f:
    data = json.load(f)

dopant_mae = data["by_dopant"]
dopant_counts = Counter(s["dopant"] for s in data["per_sample"])

elements = sorted(dopant_mae.keys())
mae_vals = [dopant_mae[el] for el in elements]
print(f"Dopants: {len(elements)}, MAE range: [{min(mae_vals):.3f}, {max(mae_vals):.3f}]")

# ── Color mapping ──────────────────────────────────────────────────────
vmin, vmax = 0.0, 1.4
cmap = plt.cm.RdYlGn_r
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# ── Plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12.5, 4.2))

cell_w, cell_h = 1.0, 1.0
pad = 0.05

for el in elements:
    if el not in PT_LAYOUT:
        print(f"  Warning: {el} not in PT_LAYOUT, skipping")
        continue
    if el == 'Lu':
        continue  # handle separately

    row, col = PT_LAYOUT[el]
    mae = dopant_mae[el]
    n = dopant_counts.get(el, 0)

    x = col * (cell_w + pad)
    y = -row * (cell_h + pad)
    color = cmap(norm(min(mae, vmax)))

    rect = FancyBboxPatch(
        (x, y), cell_w, cell_h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor='#444444', linewidth=0.7
    )
    ax.add_patch(rect)

    rgb = mcolors.to_rgb(color)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    text_color = 'white' if luminance < 0.45 else 'black'

    ax.text(x + cell_w / 2, y + cell_h * 0.62, el,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=text_color, family='sans-serif')
    ax.text(x + cell_w / 2, y + cell_h * 0.25, f'{mae:.2f}',
            ha='center', va='center', fontsize=6.5,
            color=text_color, family='sans-serif')
    ax.text(x + cell_w * 0.88, y + cell_h * 0.90, f'{n}',
            ha='right', va='center', fontsize=4,
            color=text_color, alpha=0.6, family='sans-serif')

# ── Lu: small annotation box below La ─────────────────────────────────
if 'Lu' in dopant_mae:
    mae_lu = dopant_mae['Lu']
    n_lu = dopant_counts.get('Lu', 0)
    # Place it as a footnote-style box in the lower-left
    lx, ly = 0.3, -6.5 * (cell_h + pad)
    ax.text(lx, ly + 0.2, f'* Lu: MAE = {mae_lu:.2f} eV (n={n_lu})',
            fontsize=7, color='#555', family='sans-serif', style='italic')

# ── Empty context cells ───────────────────────────────────────────────
context_elements = set(PT_LAYOUT.keys()) - set(elements) - {'Lu'}
for el in context_elements:
    row, col = PT_LAYOUT[el]
    if not isinstance(col, int):
        continue
    x = col * (cell_w + pad)
    y = -row * (cell_h + pad)
    rect = FancyBboxPatch(
        (x, y), cell_w, cell_h,
        boxstyle="round,pad=0.02",
        facecolor='#f5f5f5', edgecolor='#d0d0d0', linewidth=0.4
    )
    ax.add_patch(rect)
    ax.text(x + cell_w / 2, y + cell_h / 2, el,
            ha='center', va='center', fontsize=7,
            color='#c0c0c0', family='sans-serif')

# ── Colorbar ───────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.30, 0.04, 0.42, 0.035])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('MAE (eV)', fontsize=10)
cbar.ax.tick_params(labelsize=8)

ax.set_xlim(-0.5, 18 * (cell_w + pad) + 0.2)
ax.set_ylim(-6.8 * (cell_h + pad), cell_h + 0.5)
ax.set_aspect('equal')
ax.axis('off')

plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.10)

out_dir = root / "figures"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "periodic_table_mae.pdf", dpi=300, bbox_inches='tight')
fig.savefig(out_dir / "periodic_table_mae.png", dpi=300, bbox_inches='tight')
print(f"Saved to {out_dir / 'periodic_table_mae.pdf'}")
plt.close()
