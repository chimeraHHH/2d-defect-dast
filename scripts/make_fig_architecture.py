"""F0: PeriDefT architecture schematic.

Two-row figure:
  Row A (top): data flow — graph input → encoder (local + global) →
               multi-source readout heads → calibrated mu, sigma
  Row B (bottom): PFA detail — minimum-image fractional displacement
                  + Fourier basis cos/sin → additive attention bias

Goal: every reader who skips the methods section still understands
PeriDefT's architecture from this one figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures" / "fig_peridelft_arch.png"


def box(ax, x, y, w, h, label, sub=None, color="#cdd9e8", edge="#1f3a5f", fs=10):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        linewidth=1.3, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    cy = y + h * (0.62 if sub else 0.5)
    ax.text(x + w / 2, cy, label,
            ha="center", va="center", fontsize=fs, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub,
                ha="center", va="center", fontsize=fs - 1.5, color="#333")


def arrow(ax, x0, y0, x1, y1, color="#1f3a5f", lw=1.4):
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=lw, color=color),
    )


def main():
    fig = plt.figure(figsize=(13.5, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.18)
    axA = fig.add_subplot(gs[0]); axA.set_xlim(0, 14); axA.set_ylim(0, 4); axA.axis("off")
    axB = fig.add_subplot(gs[1]); axB.set_xlim(0, 14); axB.set_ylim(0, 3.4); axB.axis("off")

    # =========== Row A: data flow ===========
    axA.text(7, 3.7, "PeriDefT architecture",
             ha="center", fontsize=15, fontweight="bold")
    axA.text(7, 3.4, "graph input → hybrid encoder → multi-source readout → calibrated $E_f$ + $\\sigma_{cal}$",
             ha="center", fontsize=10, color="#444", fontstyle="italic")

    # 1. Input graph
    box(axA, 0.2, 1.1, 1.6, 1.5,
        "Defect\nsupercell",
        "atom features (9d)\nedges + dist + angles\n28 atoms",
        color="#e8e0cd")

    # 2. Encoder: 2 columns (local + global) → merged
    box(axA, 2.3, 2.0, 2.0, 0.7,
        "Local SchNet",
        "continuous filter\n$r \\leq 5$ Å",
        color="#d9e8cd", fs=10)
    box(axA, 2.3, 1.0, 2.0, 0.7,
        "Global self-attn",
        "scaled dot-product",
        color="#d9e8cd", fs=10)
    # PFA bias annotation
    box(axA, 2.3, 0.15, 2.0, 0.65,
        "+ PFA bias",
        "(see panel B)",
        color="#cdd9e8", fs=9.5)
    arrow(axA, 4.3, 0.5, 4.7, 1.3)

    # 3. Merge / Repeat × N
    box(axA, 4.7, 1.1, 1.5, 1.5,
        "Merge\n+ FFN × 4",
        "h = 128\n0.75 M params",
        color="#cdd9e8")

    # 4. Multi-source readout heads (4 stacked)
    src_names = ["IMP2D 8.5k", "JARVIS-2D 70", "JARVIS-3D 381", "JARVIS DFT-3D 17.9k"]
    src_colors = ["#cdd9e8", "#d9cde8", "#e8cdd9", "#d9e8cd"]
    for k, (nm, col) in enumerate(zip(src_names, src_colors)):
        y = 2.4 - k * 0.55
        box(axA, 6.7, y - 0.22, 2.4, 0.45, nm, color=col, fs=9.5)
        arrow(axA, 6.2, 1.85, 6.65, y)
    axA.text(7.9, 0.05, "per-source linear head", ha="center", fontsize=9, color="#444", fontstyle="italic")

    # 5. Ensemble + calibration
    box(axA, 9.6, 1.5, 2.0, 1.2,
        "6-seed ensemble",
        "+ temperature\nscaling",
        color="#e8e0cd")

    # 6. Output
    box(axA, 12.0, 1.5, 1.7, 1.2,
        "$\\mu$, $\\sigma_{cal}$",
        "calibrated\n90\\% cov 93.4\\%",
        color="#cdd9e8")

    # arrows main row
    arrow(axA, 1.85, 1.85, 2.3, 2.35)
    arrow(axA, 1.85, 1.85, 2.3, 1.35)
    arrow(axA, 9.15, 1.85, 9.55, 2.1)
    arrow(axA, 11.6, 2.1, 11.95, 2.1)

    # =========== Row B: PFA detail ===========
    axB.text(7, 3.0, "Periodic Fourier Bias (PFA)",
             ha="center", fontsize=13, fontweight="bold")
    axB.text(7, 2.7, "exact translation invariance via $2\\pi$-periodicity of cos/sin on fractional displacement",
             ha="center", fontsize=10, color="#444", fontstyle="italic")

    # left: schematic of fractional displacement
    box(axB, 0.5, 1.0, 2.4, 1.2,
        "Pair $(i,j)$",
        "$\\mathbf{f}_{ij}$ = min-image\nfractional displacement\nin $[-0.5, 0.5)^3$",
        color="#e8e0cd", fs=10)

    # middle: Fourier basis equation
    axB.text(4.2, 1.6,
             r"$b_{ij} = \sum_k w_k \cos(2\pi \mathbf{n}_k \cdot \mathbf{f}_{ij}) + v_k \sin(2\pi \mathbf{n}_k \cdot \mathbf{f}_{ij})$",
             ha="left", fontsize=11.5, color="#1f3a5f")
    axB.text(4.2, 1.1,
             r"$\mathbf{n}_k \in \mathbb{Z}^3$, $|\mathbf{n}_k|_\infty \leq 3$ (truncated reciprocal lattice)",
             ha="left", fontsize=9.5, color="#444", fontstyle="italic")
    axB.text(4.2, 0.7,
             r"~2k learnable parameters; ablation = single boolean flag",
             ha="left", fontsize=9.5, color="#444", fontstyle="italic")

    # right: invariance illustration
    # mini sin/cos plot
    inset = fig.add_axes([0.74, 0.10, 0.22, 0.20])
    x = np.linspace(-0.5, 0.5, 200)
    inset.plot(x, np.cos(2 * np.pi * x), label=r"$\cos(2\pi f)$", c="#1f3a5f", lw=1.4)
    inset.plot(x, np.sin(2 * np.pi * x), label=r"$\sin(2\pi f)$", c="#a04040", lw=1.4)
    inset.axvline(-0.5, ls=":", c="grey", lw=0.6)
    inset.axvline( 0.5, ls=":", c="grey", lw=0.6)
    inset.set_xlabel(r"fractional displacement $f$", fontsize=8.5)
    inset.set_xlim(-0.5, 0.5); inset.set_ylim(-1.2, 1.2)
    inset.tick_params(axis="both", labelsize=7.5)
    inset.legend(fontsize=7, loc="upper right", frameon=False)
    inset.spines["top"].set_visible(False); inset.spines["right"].set_visible(False)

    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
