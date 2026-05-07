"""F1: Prospective DFT validation workflow figure.

Produces paper/figures/fig_prospective_workflow.png — a 5-stage
diagram showing the data flow:

  287 candidates  →  60 selected  →  125 SCFs  →  37 valid  →  analysis

Each stage shows quantities and the bucket A/B split.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures" / "fig_prospective_workflow.png"


def box(ax, x, y, w, h, label, sub=None, color="#cdd9e8", edge="#1f3a5f"):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        linewidth=1.4, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.62, label,
            ha="center", va="center", fontsize=11, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub,
                ha="center", va="center", fontsize=9, color="#333")


def arrow(ax, x0, y0, x1, y1, label=None, color="#1f3a5f"):
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.3, color=color),
    )
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.06, label,
                ha="center", va="bottom", fontsize=9,
                color="#1f3a5f", fontstyle="italic")


def main():
    fig, ax = plt.subplots(figsize=(13.8, 4.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.5); ax.axis("off")

    # Stage 1: candidate pool
    box(ax, 0.2, 1.5, 2.0, 1.4,
        "287 candidates",
        "v1.2 generated\nOOD pool",
        color="#e8e0cd")

    # Stage 2: selection
    box(ax, 2.8, 2.6, 2.0, 1.0,
        "Bucket A: 30",
        "low-$E_f$ confident\n(rank by $\\mu$)",
        color="#cdd9e8")
    box(ax, 2.8, 0.8, 2.0, 1.0,
        "Bucket B: 30",
        "high-$\\sigma$ OOD\n(rank by $\\sigma_{cal}$)",
        color="#e8cdd9")

    # Stage 3: QE inputs
    box(ax, 5.6, 1.5, 2.2, 1.4,
        "125 QE inputs",
        "60 defects\n+27 pristine\n+38 atomic $\\mu$",
        color="#d9e8cd")

    # Stage 4: DFT runs
    box(ax, 8.4, 1.5, 2.2, 1.4,
        "QE 7.3.1 / GPU",
        "$\\sim$11 h on RTX 5090\nNVHPC 25.5 + cc=120",
        color="#cdd9e8")

    # Stage 5: results
    box(ax, 11.2, 1.5, 2.6, 1.4,
        "37 valid Ef",
        "22 La/Cs PSL fail\n+1 SCF divergence\nN=20A + 17B",
        color="#e8e0cd")

    # arrows
    arrow(ax, 2.3, 2.6, 2.7, 3.0, label=None)
    arrow(ax, 2.3, 1.8, 2.7, 1.4, label=None)
    arrow(ax, 4.85, 3.0, 5.55, 2.4, label=None)
    arrow(ax, 4.85, 1.4, 5.55, 1.95, label=None)
    arrow(ax, 7.85, 2.2, 8.4, 2.2, label=None)
    arrow(ax, 10.65, 2.2, 11.2, 2.2, label=None)

    # subtitle row at top
    ax.text(7, 4.2,
            "Prospective DFT validation pipeline",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(7, 3.85,
            "model recommends → strategically split → first-principles validate",
            ha="center", fontsize=10, color="#444", fontstyle="italic")

    # bottom annotations
    ax.text(7, 0.25,
            r"Bucket-A 70% DFT-confirmed at $E_f<+1$ eV  ·  "
            r"Bucket-B Pearson($\sigma_{cal}$, $|\mathrm{err}|$) $=-0.29$  ·  "
            r"deployment ceiling 5.5$\times$ in-distribution MAE",
            ha="center", fontsize=9, color="#1f3a5f", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
