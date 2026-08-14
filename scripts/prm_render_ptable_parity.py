"""Render the impurity-element periodic-table MAE map and the parity plot.

Pure aggregation and plotting over the repaired pair-held-out out-of-fold
predictions in ``artifacts/prm_results/materials_repaired/sample_predictions.csv``
(canonical retained rows only).  No training or inference happens here.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#0072B2", "#D55E00"

LAYOUT_ROWS = [
    "H  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  He",
    "Li Be .  .  .  .  .  .  .  .  .  .  B  C  N  O  F  Ne",
    "Na Mg .  .  .  .  .  .  .  .  .  .  Al Si P  S  Cl Ar",
    "K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr",
    "Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe",
    "Cs Ba *  Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn",
]
LANTHANIDES = "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()


def element_positions() -> dict[str, tuple[float, float]]:
    positions = {}
    for row_index, row in enumerate(LAYOUT_ROWS):
        for column_index, symbol in enumerate(row.split()):
            if symbol not in (".", "*"):
                positions[symbol] = (column_index, -row_index)
    for k, symbol in enumerate(LANTHANIDES):
        positions[symbol] = (2.5 + k, -7.0)
    return positions


def style() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def load_predictions() -> list[dict]:
    rows = list(csv.DictReader(open(
        ROOT / "artifacts/prm_results/materials_repaired/sample_predictions.csv")))
    retained = [r for r in rows if r["canonical_retained"] == "True"]
    assert len(retained) == 10224, len(retained)
    return retained


def render_ptable(rows: list[dict], output: Path) -> None:
    errors = defaultdict(list)
    for row in rows:
        errors[row["dopant"]].append(float(row["absolute_error_eV"]))
    mae = {element: float(np.mean(v)) for element, v in errors.items()}
    positions = element_positions()
    unknown = set(mae) - set(positions)
    assert not unknown, unknown

    figure, ax = plt.subplots(figsize=(7.05, 3.6))
    values = np.asarray(list(mae.values()))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=np.quantile(values, 0.95))
    colormap = matplotlib.colormaps["viridis"]
    for element, (x, y) in positions.items():
        if element in mae:
            color = colormap(norm(mae[element]))
            luminance = matplotlib.colors.rgb_to_hsv(color[:3])[2]
            text_color = "white" if luminance < 0.72 else "black"
            ax.add_patch(Rectangle((x, y), 0.94, 0.94, facecolor=color,
                                   edgecolor="0.4", linewidth=0.3))
            ax.text(x + 0.47, y + 0.60, element, ha="center", va="center",
                    fontsize=6.6, fontweight="bold", color=text_color)
            ax.text(x + 0.47, y + 0.24, str(len(errors[element])), ha="center",
                    va="center", fontsize=5.2, color=text_color)
        else:
            ax.add_patch(Rectangle((x, y), 0.94, 0.94, facecolor="0.94",
                                   edgecolor="0.75", linewidth=0.3))
            ax.text(x + 0.47, y + 0.45, element, ha="center", va="center",
                    fontsize=6.0, color="0.55")
    ax.set_xlim(-0.3, 18.4)
    ax.set_ylim(-7.6, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    bar = figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax,
        fraction=0.032, pad=0.01, extend="max")
    bar.set_label("pair-held-out MAE (eV)", fontsize=7.5)
    bar.ax.tick_params(labelsize=6.5)
    figure.tight_layout()
    figure.savefig(output, dpi=400)
    print(f"wrote {output}")


def render_parity(rows: list[dict], output: Path) -> None:
    figure, ax = plt.subplots(figsize=(3.4, 3.2))
    limits = (-6.5, 21)
    maes = {}
    for defect_type, color, marker in (
        ("adsorbate", BLUE, "o"), ("interstitial", ORANGE, "^"),
    ):
        subset = [r for r in rows if r["defecttype"] == defect_type]
        target = np.asarray([float(r["target_eV"]) for r in subset])
        prediction = np.asarray([float(r["prediction_eV"]) for r in subset])
        maes[defect_type] = np.mean(np.abs(target - prediction))
        ax.scatter(target, prediction, s=4, marker=marker, color=color,
                   alpha=0.3, linewidths=0, rasterized=True,
                   label=f"{defect_type} (MAE {maes[defect_type]:.3f} eV)")
    pooled = np.mean([float(r["absolute_error_eV"]) for r in rows])
    ax.plot(limits, limits, color="0.5", linewidth=0.6)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("reference $E_f$ (eV)")
    ax.set_ylabel("predicted $E_f$ (eV)")
    legend = ax.legend(frameon=False, loc="upper left", handletextpad=0.2,
                       markerscale=2.2,
                       title=f"pooled MAE {pooled:.3f} eV", title_fontsize=7)
    legend._legend_box.align = "left"
    for handle in legend.legend_handles:
        handle.set_alpha(0.9)
    figure.tight_layout()
    figure.savefig(output, dpi=400)
    print(f"wrote {output}")


def main() -> None:
    style()
    rows = load_predictions()
    figdir = ROOT / "paper_Q1/figures"
    render_ptable(rows, figdir / "fig_impurity_ptable.pdf")
    render_parity(rows, figdir / "fig_parity_pair.pdf")


if __name__ == "__main__":
    main()
