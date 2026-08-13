"""Render the transfer-comparison and reranking figures for PRM.

Pure aggregation and plotting over existing archives: the transfer figure
reads ``artifacts/prm_results/repaired_transfer_metrics.json`` and the
reranking figure reads the repaired materials collector's CSV/JSON outputs.
No training or inference happens here.  Identity is triple-encoded (a
CVD-validated Okabe--Ito triple, marker shape, line style) so both figures
stay legible in grayscale.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

MODEL_STYLE = {
    "dart": ("#0072B2", "o", "-", "DART (repaired)"),
    "schnet": ("#D55E00", "s", "--", "SchNet (additive)"),
    "descriptor": ("#009E73", "^", ":", "selected descriptor"),
}
REGIME_ORDER = ["id", "pair", "chemistry_block", "dopant", "host"]
REGIME_LABEL = {
    "id": "random", "pair": "pair\nheld out", "chemistry_block": "chemistry\nblock",
    "dopant": "impurity\nheld out", "host": "host\nheld out",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def render_transfer(metrics_path: Path, output: Path) -> None:
    data = json.loads(metrics_path.read_text())
    figure, ax = plt.subplots(figsize=(3.4, 2.7))
    x = np.arange(len(REGIME_ORDER))
    for model, (color, marker, linestyle, label) in MODEL_STYLE.items():
        values, fold_scatter = [], []
        for k, regime in enumerate(REGIME_ORDER):
            entry = data[model].get(regime)
            if model == "descriptor":
                values.append(entry["test_mae_mean"])
            else:
                values.append(entry["pooled_mean_of_folds"])
                for fold_value in entry["fold_maes"]:
                    fold_scatter.append((k, fold_value))
        ax.plot(x, values, linestyle, marker=marker, color=color,
                markersize=4.5, linewidth=1.1, label=label)
        if fold_scatter:
            xs, ys = zip(*fold_scatter)
            ax.scatter(np.asarray(xs) + 0.08, ys, s=5, marker=marker,
                       facecolors="none", edgecolors=color, linewidths=0.6,
                       alpha=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL[r] for r in REGIME_ORDER])
    ax.set_ylabel("test MAE (eV)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)
    figure.tight_layout()
    figure.savefig(output, dpi=400)
    print(f"wrote {output}")


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def render_reranking(materials_dir: Path, output: Path) -> None:
    pairs = list(csv.DictReader(open(materials_dir / "pair_preferences.csv")))
    sites = list(csv.DictReader(open(materials_dir / "site_selection.csv")))
    summary = json.loads((materials_dir / "summary.json").read_text())
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.45))

    # (a) predicted vs reference class margins
    eligible = [r for r in pairs if r["preference_eligible"] == "1"]
    true_margin = np.asarray([float(r["true_margin_eV"]) for r in eligible])
    pred_margin = np.asarray([float(r["predicted_margin_eV"]) for r in eligible])
    correct = np.asarray([r["preference_correct"] == "1" for r in eligible], dtype=bool)
    ax = axes[0]
    for mask, color, label in (
        (correct, "#0072B2", "class correct"),
        (~correct, "#D55E00", "class flipped"),
    ):
        ax.scatter(true_margin[mask], pred_margin[mask], s=4,
                   color=color, alpha=0.35, linewidths=0, label=label,
                   rasterized=True)
    limit = 4.0
    ax.plot([-limit, limit], [-limit, limit], color="0.5", linewidth=0.6)
    ax.axhline(0, color="0.8", linewidth=0.5)
    ax.axvline(0, color="0.8", linewidth=0.5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("reference class margin (eV)")
    ax.set_ylabel("predicted class margin (eV)")
    legend = ax.legend(frameon=False, loc="upper left", handletextpad=0.2,
                       markerscale=2.5)
    for handle in legend.legend_handles:
        handle.set_alpha(0.9)

    # (b) reranking regret ECDFs
    ax = axes[1]
    global_regret = np.asarray([
        float(r["global_screening_regret_eV"]) for r in pairs
        if r["global_screening_regret_eV"] not in ("", "nan")
    ])
    within_regret = np.asarray([
        float(r["screening_regret_eV"]) for r in sites
        if r["screening_regret_eV"] not in ("", "nan")
    ])
    for values, color, linestyle, label in (
        (global_regret, "#0072B2", "-", "global"),
        (within_regret, "#D55E00", "--", "within class"),
    ):
        xs, ys = ecdf(values)
        ax.step(xs, ys, linestyle, color=color, linewidth=1.2, label=label,
                where="post")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("reranking regret (eV)")
    ax.set_ylabel("cumulative fraction")
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)

    # (c) decision accuracies with cluster-bootstrap intervals
    ax = axes[2]
    pref = summary["defect_type_preference"]
    within = summary["within_defect_type_site_selection"]
    rows = [
        ("class\npreference", pref["accuracy"],
         pref["accuracy_reference"]["accuracy"]["mean"]),
        ("global\nexact site", pref["global_exact_site_accuracy"],
         pref["global_exact_site_reference"]["accuracy"]["mean"]),
        ("within-class\nexact site", within["exact_accuracy"],
         within["exact_accuracy_reference"]["accuracy"]["mean"]),
        ("within-class\ntop two", within["top2_accuracy"],
         within["top2_accuracy_reference"]["accuracy"]["mean"]),
    ]
    y = np.arange(len(rows))[::-1]
    for (label, block, reference), y_pos in zip(rows, y):
        mean = 100 * block["mean"]
        ax.errorbar(mean, y_pos,
                    xerr=[[mean - 100 * block["ci_low"]],
                          [100 * block["ci_high"] - mean]],
                    fmt="o", color="#0072B2", markersize=4,
                    elinewidth=1.0, capsize=1.8, capthick=1.0)
        ax.plot(100 * reference, y_pos, marker="d", color="0.35",
                markersize=4.5, linestyle="none")
    ax.plot([], [], "o", color="#0072B2", markersize=4, label="model")
    ax.plot([], [], "d", color="0.35", markersize=4.5, linestyle="none",
            label="reference")
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in rows])
    ax.set_xlabel("decision accuracy (\\%)" if False else "decision accuracy (%)")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.3)

    for label, ax in zip(("(a)", "(b)", "(c)"), axes):
        ax.text(-0.24, 1.04, label, transform=ax.transAxes,
                fontweight="bold", fontsize=9, va="bottom")
    figure.tight_layout(w_pad=1.4)
    figure.savefig(output, dpi=400)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default=str(
        ROOT / "artifacts/prm_results/repaired_transfer_metrics.json"))
    parser.add_argument("--materials", default=str(
        ROOT / "artifacts/prm_results/materials_repaired"))
    parser.add_argument("--figdir", default=str(ROOT / "paper_Q1/figures"))
    args = parser.parse_args()
    style()
    figdir = Path(args.figdir)
    render_transfer(Path(args.metrics), figdir / "fig_transfer_repaired.pdf")
    render_reranking(Path(args.materials), figdir / "fig_reranking_repaired.pdf")


if __name__ == "__main__":
    main()
