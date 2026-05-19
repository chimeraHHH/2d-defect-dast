#!/usr/bin/env python3
"""Generate paper-quality figures for innovation ablation study.

Reads metrics.json and test_predictions.npz from each experiment to produce:
  1. Training curves (val MAE vs epoch) comparing innovations
  2. Per-range MAE heatmap across innovations
  3. Prediction vs true scatter with per-range colouring
  4. Innovation contribution bar chart (delta MAE vs V2 baseline)
  5. Ensemble size vs MAE curve

Usage:
    python scripts/make_innovation_figures.py [--results-dir results/] [--out paper/figures/]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Try importing matplotlib — fall back to Agg backend on headless servers
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available; skipping figures")
    sys.exit(0)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Registry of innovations with display properties
MODELS = {
    # V2 baselines
    "v2_gated_pooling_s43": {"label": "V2 (baseline)", "color": "#888888", "ls": "-", "zorder": 1},
    "v2_gated_pooling": {"label": "V2 (s42)", "color": "#aaaaaa", "ls": "--", "zorder": 0},
    # Individual innovations
    "v3_deftype": {"label": "+ Defect-type cond.", "color": "#e74c3c", "ls": "-", "zorder": 2},
    "v4_moe": {"label": "+ MoE readout", "color": "#3498db", "ls": "-", "zorder": 2},
    "v6_physics": {"label": "+ Physics features", "color": "#2ecc71", "ls": "-", "zorder": 2},
    "v9_contrast": {"label": "+ D-H contrast", "color": "#9b59b6", "ls": "-", "zorder": 2},
    "v2_ema": {"label": "+ EMA", "color": "#f39c12", "ls": "-", "zorder": 2},
    "v11_lds": {"label": "+ LDS", "color": "#1abc9c", "ls": "-", "zorder": 2},
    "v13_rnc": {"label": "+ RnC + LDS", "color": "#e67e22", "ls": "-", "zorder": 2},
    # Combined
    "v8_ema_physics": {"label": "EMA+Phys+Def", "color": "#c0392b", "ls": "-.", "zorder": 3},
    "v10_best_combo": {"label": "V10 (all best)", "color": "#2c3e50", "ls": "-", "zorder": 4},
    "v12_lds_physics": {"label": "V12 LDS+Phys", "color": "#16a085", "ls": "-.", "zorder": 3},
}


def load_metrics(results_dir: Path, name: str, min_epochs: int = 20):
    """Load training history if available and sufficiently trained."""
    mpath = results_dir / name / "metrics.json"
    if not mpath.exists():
        return None
    with open(mpath) as f:
        m = json.load(f)
    history = m.get("history", [])
    if len(history) < min_epochs:
        return None
    return m


def load_predictions(results_dir: Path, name: str, min_epochs: int = 20):
    """Load test predictions if model sufficiently trained."""
    npz = results_dir / name / "test_predictions.npz"
    if not npz.exists():
        return None, None
    # Check training epoch count to filter smoke tests
    mpath = results_dir / name / "metrics.json"
    if mpath.exists():
        with open(mpath) as f:
            m = json.load(f)
        if len(m.get("history", [])) < min_epochs:
            return None, None
    data = np.load(npz)
    return data["preds"], data["targets"]


# ----------------------------------------------------------------
# Figure 1: Training curves
# ----------------------------------------------------------------
def fig_training_curves(all_metrics, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, m in sorted(all_metrics.items(), key=lambda x: MODELS.get(x[0], {}).get("zorder", 5)):
        info = MODELS.get(name, {"label": name, "color": "#333333", "ls": "-", "zorder": 5})
        history = m["history"]
        epochs = [h["epoch"] for h in history]
        val_mae = [h["val_mae"] for h in history]
        ax.plot(epochs, val_mae, label=info["label"], color=info["color"],
                linestyle=info["ls"], linewidth=1.5, alpha=0.85,
                zorder=info.get("zorder", 5))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (eV)")
    ax.set_title("Innovation Ablation: Training Convergence")
    ax.set_ylim(0.3, 1.2)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "training_curves_innovations.pdf")
    fig.savefig(out_dir / "training_curves_innovations.png")
    plt.close(fig)
    print("  Saved: training_curves_innovations.pdf/png")


# ----------------------------------------------------------------
# Figure 2: Per-range MAE heatmap
# ----------------------------------------------------------------
def fig_per_range_heatmap(all_preds, targets, out_dir):
    bins = [(0, 2, "[0,2)"), (2, 5, "[2,5)"), (5, 7, "[5,7)"), (7, 25, "[7,25)")]
    model_names = sorted(all_preds.keys(),
                          key=lambda n: np.abs(all_preds[n] - targets).mean())
    labels = [MODELS.get(n, {}).get("label", n) for n in model_names]

    data = np.zeros((len(model_names), len(bins)))
    for i, name in enumerate(model_names):
        preds = all_preds[name]
        for j, (lo, hi, _) in enumerate(bins):
            mask = (targets >= lo) & (targets < hi)
            if mask.sum() > 0:
                data[i, j] = float(np.abs(preds[mask] - targets[mask]).mean())

    fig, ax = plt.subplots(figsize=(6, max(4, len(model_names) * 0.45)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([b[2] for b in bins])
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target Range (eV)")
    ax.set_title("Per-Range MAE (eV)")

    # Annotate cells
    for i in range(len(model_names)):
        for j in range(len(bins)):
            val = data[i, j]
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, "%.3f" % val, ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="MAE (eV)", shrink=0.8)
    fig.savefig(out_dir / "per_range_heatmap.pdf")
    fig.savefig(out_dir / "per_range_heatmap.png")
    plt.close(fig)
    print("  Saved: per_range_heatmap.pdf/png")


# ----------------------------------------------------------------
# Figure 3: Innovation contribution bar chart
# ----------------------------------------------------------------
def fig_innovation_bars(all_preds, targets, baseline_mae, out_dir):
    deltas = {}
    for name, preds in all_preds.items():
        if name.startswith("v2_gated_pooling"):
            continue  # skip baselines
        mae = float(np.abs(preds - targets).mean())
        deltas[name] = mae - baseline_mae

    # Sort by delta (negative = improvement)
    sorted_names = sorted(deltas, key=lambda n: deltas[n])
    labels = [MODELS.get(n, {}).get("label", n) for n in sorted_names]
    vals = [deltas[n] for n in sorted_names]
    colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_names) * 0.4)))
    ax.barh(range(len(sorted_names)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta MAE vs V2 baseline (eV)")
    ax.set_title("Innovation Contribution (negative = improvement)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(out_dir / "innovation_delta_bars.pdf")
    fig.savefig(out_dir / "innovation_delta_bars.png")
    plt.close(fig)
    print("  Saved: innovation_delta_bars.pdf/png")


# ----------------------------------------------------------------
# Figure 4: Pred vs True scatter
# ----------------------------------------------------------------
def fig_pred_vs_true(preds, targets, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Color by range
    colors = np.zeros(len(targets), dtype=int)
    colors[(targets >= 0) & (targets < 2)] = 0
    colors[(targets >= 2) & (targets < 5)] = 1
    colors[(targets >= 5) & (targets < 7)] = 2
    colors[targets >= 7] = 3
    cmap_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    range_labels = ["[0,2)", "[2,5)", "[5,7)", "[7,25)"]

    for i, (c, label) in enumerate(zip(cmap_colors, range_labels)):
        mask = colors == i
        ax.scatter(targets[mask], preds[mask], c=c, s=8, alpha=0.6, label=label)

    mn, mx = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8, alpha=0.5, label="y=x")
    ax.set_xlabel("True Ef (eV)")
    ax.set_ylabel("Predicted Ef (eV)")
    mae = float(np.abs(preds - targets).mean())
    label = MODELS.get(model_name, {}).get("label", model_name)
    ax.set_title("%s | MAE=%.3f eV" % (label, mae))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.savefig(out_dir / ("pred_vs_true_%s.pdf" % model_name))
    fig.savefig(out_dir / ("pred_vs_true_%s.png" % model_name))
    plt.close(fig)
    print("  Saved: pred_vs_true_%s.pdf/png" % model_name)


# ----------------------------------------------------------------
# Figure 5: Ensemble size vs MAE
# ----------------------------------------------------------------
def fig_ensemble_curve(all_preds, targets, out_dir):
    model_names = sorted(all_preds.keys(),
                          key=lambda n: np.abs(all_preds[n] - targets).mean())
    if len(model_names) < 2:
        return

    # Greedy ensemble construction
    selected = [model_names[0]]
    ensemble_pred = all_preds[model_names[0]].copy()
    maes = [float(np.abs(ensemble_pred - targets).mean())]
    remaining = [n for n in model_names if n != model_names[0]]

    for _ in range(min(14, len(remaining))):
        best_next = None
        best_mae = maes[-1]
        for cand in remaining:
            trial = (ensemble_pred * len(selected) + all_preds[cand]) / (len(selected) + 1)
            trial_mae = float(np.abs(trial - targets).mean())
            if trial_mae < best_mae:
                best_mae = trial_mae
                best_next = cand
        if best_next is None:
            break
        selected.append(best_next)
        ensemble_pred = (ensemble_pred * (len(selected) - 1) + all_preds[best_next]) / len(selected)
        maes.append(best_mae)
        remaining.remove(best_next)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(maes) + 1), maes, "o-", color="#2c3e50", linewidth=2, markersize=6)
    for i, (mae, name) in enumerate(zip(maes, selected)):
        label = MODELS.get(name, {}).get("label", name)
        ax.annotate(label, (i + 1, mae), textcoords="offset points",
                    xytext=(8, 5 if i % 2 == 0 else -12), fontsize=7, alpha=0.8)
    ax.set_xlabel("Ensemble Size")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("Greedy Ensemble Selection")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "ensemble_curve.pdf")
    fig.savefig(out_dir / "ensemble_curve.png")
    plt.close(fig)
    print("  Saved: ensemble_curve.pdf/png")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--out", default="paper/figures/innovations/")
    parser.add_argument("--data-path", default="data/processed/cleaned_dataset.pkl")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all available metrics
    all_metrics = {}
    all_preds = {}
    targets = None
    for name in MODELS:
        m = load_metrics(results_dir, name)
        if m is not None:
            all_metrics[name] = m
        p, t = load_predictions(results_dir, name)
        if p is not None:
            all_preds[name] = p
            if targets is None:
                targets = t

    print("Loaded %d models with metrics, %d with predictions" %
          (len(all_metrics), len(all_preds)))

    if not all_metrics and not all_preds:
        print("No data found! Check results directory.")
        sys.exit(1)

    # Find baseline MAE
    baseline_mae = None
    for bname in ["v2_gated_pooling_s43", "v2_gated_pooling"]:
        if bname in all_preds:
            baseline_mae = float(np.abs(all_preds[bname] - targets).mean())
            break

    # Generate figures
    print("\nGenerating figures...")
    if all_metrics:
        fig_training_curves(all_metrics, out_dir)

    if all_preds and targets is not None:
        fig_per_range_heatmap(all_preds, targets, out_dir)

        if baseline_mae is not None:
            fig_innovation_bars(all_preds, targets, baseline_mae, out_dir)

        # Pred vs true for best model and baseline
        best_name = min(all_preds.keys(), key=lambda n: np.abs(all_preds[n] - targets).mean())
        fig_pred_vs_true(all_preds[best_name], targets, best_name, out_dir)
        if "v2_gated_pooling_s43" in all_preds:
            fig_pred_vs_true(all_preds["v2_gated_pooling_s43"], targets,
                             "v2_gated_pooling_s43", out_dir)

        fig_ensemble_curve(all_preds, targets, out_dir)

    print("\nDone! Figures saved to %s" % out_dir)


if __name__ == "__main__":
    main()
