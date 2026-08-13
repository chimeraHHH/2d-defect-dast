"""Render the PRM v19 prediction-and-transfer main figure.

This is a pure visualization over the archived repaired pair-held-out
predictions and transfer metrics.  It performs no training or inference.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"

REGIMES = ("id", "pair", "chemistry_block", "dopant", "host")
REGIME_LABELS = (
    "random",
    "pair\nheld out",
    "chemistry\nblock",
    "impurity\nheld out",
    "host\nheld out",
)
MODEL_STYLES = {
    "dart": (BLUE, "o", "-", "DART"),
    "schnet": (ORANGE, "s", "--", "SchNet-add"),
    "descriptor": (GREEN, "^", ":", "selected descriptor"),
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })


def load_rows(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    retained = [row for row in rows if row["canonical_retained"] == "True"]
    if len(retained) != 10_224:
        raise ValueError(f"expected 10,224 retained rows, found {len(retained)}")
    return retained


def render(output: Path) -> None:
    rows = load_rows(
        ROOT / "artifacts/prm_results/materials_repaired/sample_predictions.csv"
    )
    metrics = json.loads(
        (ROOT / "artifacts/prm_results/repaired_transfer_metrics.json").read_text()
    )

    configure_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(7.05, 2.85), gridspec_kw={"width_ratios": (1.04, 1.32)}
    )

    # (a) Repaired pair-held-out parity, class-resolved.
    axis = axes[0]
    for defect_type, color, marker, label in (
        ("adsorbate", BLUE, "o", "adsorbate"),
        ("interstitial", ORANGE, "^", "interstitial"),
    ):
        subset = [row for row in rows if row["defecttype"] == defect_type]
        target = np.asarray([float(row["target_eV"]) for row in subset])
        prediction = np.asarray([float(row["prediction_eV"]) for row in subset])
        mae = float(np.mean(np.abs(target - prediction)))
        axis.scatter(
            target,
            prediction,
            s=4.2,
            marker=marker,
            color=color,
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            label=f"{label} (MAE {mae:.3f} eV)",
        )
    lower, upper = -6.5, 21.0
    axis.plot([lower, upper], [lower, upper], color="0.45", linewidth=0.7)
    pooled = float(np.mean([float(row["absolute_error_eV"]) for row in rows]))
    axis.text(
        0.04,
        0.96,
        f"pooled MAE {pooled:.3f} eV",
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel(r"reference $E_f$ (eV)")
    axis.set_ylabel(r"predicted $E_f$ (eV)")
    axis.legend(frameon=False, loc="lower right", markerscale=2.1)
    axis.set_title("Pair-held-out predictions", loc="left", pad=5)
    axis.text(-0.19, 1.04, "(a)", transform=axis.transAxes,
              fontweight="bold", fontsize=9)

    # (b) Error hierarchy across increasingly difficult transfer regimes.
    axis = axes[1]
    x = np.arange(len(REGIMES))
    for model, (color, marker, linestyle, label) in MODEL_STYLES.items():
        values: list[float] = []
        for regime in REGIMES:
            entry = metrics[model][regime]
            key = "test_mae_mean" if model == "descriptor" else "pooled_mean_of_folds"
            values.append(float(entry[key]))
        axis.plot(
            x,
            values,
            linestyle=linestyle,
            marker=marker,
            color=color,
            linewidth=1.15,
            markersize=4.6,
            label=label,
        )
        if model != "descriptor":
            for position, regime in enumerate(REGIMES):
                fold_values = metrics[model][regime].get("fold_maes", [])
                axis.scatter(
                    np.full(len(fold_values), position + 0.07),
                    fold_values,
                    s=7,
                    marker=marker,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.55,
                    alpha=0.7,
                )
    axis.set_yscale("log")
    axis.set_xticks(x, REGIME_LABELS)
    axis.set_ylabel("test MAE (eV)")
    axis.set_title("Chemical-transfer hierarchy", loc="left", pad=5)
    axis.grid(axis="y", color="0.88", linewidth=0.45)
    axis.legend(frameon=False, loc="upper left", handlelength=2.0)
    axis.text(-0.16, 1.04, "(b)", transform=axis.transAxes,
              fontweight="bold", fontsize=9)

    figure.tight_layout(w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400)
    figure.savefig(output.with_suffix(".svg"), dpi=400)
    plt.close(figure)


if __name__ == "__main__":
    render(ROOT / "paper_Q1/figures/fig_prediction_transfer_v19.pdf")
