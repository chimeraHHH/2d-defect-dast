#!/usr/bin/env python3
"""
DART architecture diagram for the paper.
Clean horizontal flow diagram using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

fig, ax = plt.subplots(figsize=(14, 4.5))

# ── Colors ─────────────────────────────────────────────────────────────
c_input = '#E8F5E9'
c_defect = '#FFCDD2'
c_embed = '#C8E6C9'
c_local = '#BBDEFB'
c_global = '#FFE0B2'
c_read = '#E1BEE7'
c_text = '#37474F'
c_arrow = '#546E7A'

def draw_block(ax, x, y, w, h, color, label, sublabel='', sublabel2=''):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor='#78909C', linewidth=1.5
    )
    ax.add_patch(rect)
    cx, cy = x + w / 2, y + h / 2
    if sublabel2:
        ax.text(cx, cy + 0.15, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=c_text, family='sans-serif')
        ax.text(cx, cy - 0.08, sublabel, ha='center', va='center',
                fontsize=7.5, color='#616161', family='sans-serif')
        ax.text(cx, cy - 0.25, sublabel2, ha='center', va='center',
                fontsize=7.5, color='#616161', family='sans-serif')
    elif sublabel:
        ax.text(cx, cy + 0.08, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=c_text, family='sans-serif')
        ax.text(cx, cy - 0.15, sublabel, ha='center', va='center',
                fontsize=7.5, color='#616161', family='sans-serif')
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=c_text, family='sans-serif')
    return cx, cy

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=2.0,
                                connectionstyle='arc3,rad=0'))

# ── Layout parameters ──────────────────────────────────────────────────
bw = 2.0   # block width
bh = 0.9   # block height
gap = 0.6  # horizontal gap
y0 = 1.0   # center row y

# ── Block positions (left to right) ───────────────────────────────────
# 1. Input (supercell)
x1 = 0
cx1, cy1 = draw_block(ax, x1, y0, bw, bh, c_input,
                       'Defect Supercell', 'PBC graph G')

# 2. Node features
x_feat = x1
y_feat = y0 - 1.2
draw_block(ax, x_feat - 0.15, y_feat, bw + 0.3, bh * 0.75, c_input,
           'Node: 9-dim x_i', 'Edge: RBF(d_ij)')

# 3. Defect embedding
x_def = x1
y_def = y_feat - 0.95
draw_block(ax, x_def, y_def, bw, bh * 0.7, c_defect,
           'Defect flag', 'host=0, defect=1')

# 4. Embedding layer
x2 = x1 + bw + gap
cx2, cy2 = draw_block(ax, x2, y0, bw, bh, c_embed,
                       'Linear Embed', 'h = Wx + e_def')

# 5. Local layers
x3 = x2 + bw + gap
cx3, cy3 = draw_block(ax, x3, y0, bw + 0.3, bh, c_local,
                       'Local Layers x3', 'SchNet + bond angles', 'r_cut = 5 A')

# 6. Global attention
x4 = x3 + bw + 0.3 + gap
cx4, cy4 = draw_block(ax, x4, y0, bw + 0.3, bh, c_global,
                       'Global Attn x2', 'dist-biased self-attn', 'd_max = 12 A')

# 7. Readout
x5 = x4 + bw + 0.3 + gap
cx5, cy5 = draw_block(ax, x5, y0, bw, bh, c_read,
                       'Readout', 'mean pool + MLP')

# 8. Output
x_out = x5 + bw + gap * 0.6
ax.text(x_out + 0.4, y0 + bh / 2, r'$\hat{E}_f$', ha='center', va='center',
        fontsize=18, fontweight='bold', color='#7B1FA2', family='serif')

# ── Arrows ─────────────────────────────────────────────────────────────
# Input → Embed
draw_arrow(ax, x1 + bw, y0 + bh / 2, x2, y0 + bh / 2)

# Features → Embed (curved up)
ax.annotate('', xy=(x2 + bw / 2, y0), xytext=(x_feat + bw / 2 + 0.15, y_feat + bh * 0.75),
            arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.5,
                            connectionstyle='arc3,rad=-0.2'))

# Defect → Embed (curved up)
ax.annotate('', xy=(x2 + bw * 0.3, y0), xytext=(x_def + bw / 2, y_def + bh * 0.7),
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5,
                            connectionstyle='arc3,rad=-0.3'))

# Embed → Local
draw_arrow(ax, x2 + bw, y0 + bh / 2, x3, y0 + bh / 2)

# Local → Global
draw_arrow(ax, x3 + bw + 0.3, y0 + bh / 2, x4, y0 + bh / 2)

# Global → Readout
draw_arrow(ax, x4 + bw + 0.3, y0 + bh / 2, x5, y0 + bh / 2)

# Readout → E_f
draw_arrow(ax, x5 + bw, y0 + bh / 2, x_out + 0.1, y0 + bh / 2)

# ── Stage labels (top) ────────────────────────────────────────────────
label_y = y0 + bh + 0.2
ax.text(x1 + bw / 2, label_y, 'Input', ha='center', fontsize=9,
        fontweight='bold', color='#2E7D32', family='sans-serif')
ax.text(x2 + bw / 2, label_y, 'Embed', ha='center', fontsize=9,
        fontweight='bold', color='#2E7D32', family='sans-serif')
ax.text(x3 + (bw + 0.3) / 2, label_y, 'Near-field', ha='center', fontsize=9,
        fontweight='bold', color='#1565C0', family='sans-serif')
ax.text(x4 + (bw + 0.3) / 2, label_y, 'Far-field', ha='center', fontsize=9,
        fontweight='bold', color='#E65100', family='sans-serif')
ax.text(x5 + bw / 2, label_y, 'Output', ha='center', fontsize=9,
        fontweight='bold', color='#7B1FA2', family='sans-serif')

# ── Bottom annotations ────────────────────────────────────────────────
ann_y = y0 - 0.2
ax.text(x3 + (bw + 0.3) / 2, ann_y, 'bond lengths, angles\ncoordination geometry',
        ha='center', fontsize=7.5, color='#1565C0', family='sans-serif',
        style='italic')
ax.text(x4 + (bw + 0.3) / 2, ann_y, 'defect strain field\nlong-range ~1/r² coupling',
        ha='center', fontsize=7.5, color='#E65100', family='sans-serif',
        style='italic')

# ── Parameter box ─────────────────────────────────────────────────────
ax.text(x5 + bw / 2, ann_y - 0.2, '~0.75M params\n$d_{hidden}$=128',
        ha='center', fontsize=7.5, color='#455A64', family='sans-serif',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#B0BEC5'))

# ── Clean up ──────────────────────────────────────────────────────────
ax.set_xlim(-0.5, x_out + 1.5)
ax.set_ylim(y_def - 0.3, label_y + 0.3)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "figures"
fig.savefig(out / "architecture.pdf", dpi=300, bbox_inches='tight')
fig.savefig(out / "architecture.png", dpi=300, bbox_inches='tight')
print(f"Saved architecture diagram to {out}")
plt.close()
