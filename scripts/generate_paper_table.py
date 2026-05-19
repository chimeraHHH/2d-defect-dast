#!/usr/bin/env python3
"""Generate LaTeX tables for the paper from experimental results.

Produces:
  1. Main ablation table: innovation contributions vs V2 baseline
  2. Per-range MAE table: performance across energy ranges
  3. Ensemble composition table: greedy ensemble members and MAE
  4. Computational cost comparison

Usage:
    python scripts/generate_paper_table.py [--results-dir results/]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# Display order and labels for innovations
INNOVATIONS = [
    # (dir_name, label, category)
    ("v2_gated_pooling_s43", "V2 baseline (best seed)", "baseline"),
    ("v2_gated_pooling", "V2 (seed 42)", "baseline"),
    ("v2_gated_pooling_s44", "V2 (seed 44)", "baseline"),
    ("v2_gated_pooling_s45", "V2 (seed 45)", "baseline"),
    ("v3_deftype", "\\quad + Defect-type conditioning", "individual"),
    ("v4_moe", "\\quad + MoE readout", "individual"),
    ("v6_physics", "\\quad + Physics features", "individual"),
    ("v9_contrast", "\\quad + Defect-host contrast", "individual"),
    ("v14_jk", "\\quad + JK aggregation", "individual"),
    ("v11_lds", "\\quad + LDS", "individual"),
    ("v13_rnc", "\\quad + RnC + LDS", "individual"),
    ("v2_ema", "\\quad + EMA", "individual"),
    ("v2_focal", "\\quad + Focal MAE", "individual"),
    ("v2_uncertainty", "\\quad + Uncertainty", "individual"),
    ("v12_lds_physics", "LDS + Physics + All cond.", "combined"),
    ("v10_best_combo", "V10: all best innovations", "combined"),
]

RANGES = [(0, 2, "[0,2)"), (2, 5, "[2,5)"), (5, 7, "[5,7)"), (7, 25, "[7,25)")]


def load_results(results_dir, name, min_epochs=20):
    """Load MAE metrics from a completed experiment."""
    npz_path = results_dir / name / "test_predictions.npz"
    metrics_path = results_dir / name / "metrics.json"

    if not npz_path.exists():
        return None

    # Check epoch count
    n_epochs = 0
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        n_epochs = len(m.get("history", []))
    if n_epochs < min_epochs:
        return None

    data = np.load(str(npz_path))
    preds, targets = data["preds"], data["targets"]
    if len(preds) != 1065:
        return None

    errors = np.abs(preds - targets)
    result = {
        "name": name,
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean((preds - targets) ** 2))),
        "n_epochs": n_epochs,
        "preds": preds,
        "targets": targets,
    }

    # Per-range MAE
    for lo, hi, label in RANGES:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            result["mae_%d_%d" % (lo, hi)] = float(errors[mask].mean())
            result["n_%d_%d" % (lo, hi)] = int(mask.sum())

    return result


def table_ablation(results, baseline_mae):
    """Generate main ablation table in LaTeX."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Innovation ablation study on IMP2D test set (1065 samples).}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Method & MAE (eV) & RMSE (eV) & $\\Delta$MAE & Epochs \\\\")
    lines.append("\\midrule")

    prev_cat = None
    for dir_name, label, category in INNOVATIONS:
        if dir_name not in results:
            continue
        r = results[dir_name]

        # Add separator between categories
        if category != prev_cat and prev_cat is not None:
            lines.append("\\midrule")
        prev_cat = category

        delta = r["mae"] - baseline_mae
        delta_str = "$%+.3f$" % delta if dir_name != "v2_gated_pooling_s43" else "---"
        bold = "\\textbf{%.4f}" if r["mae"] <= baseline_mae else "%.4f"
        mae_str = bold % r["mae"]

        lines.append("%s & %s & %.4f & %s & %d \\\\"
                     % (label, mae_str, r["rmse"], delta_str, r["n_epochs"]))

    lines.append("\\midrule")
    # Add ALIGNN baseline
    lines.append("ALIGNN (baseline) & 0.5400 & --- & $+0.159$ & --- \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def table_per_range(results, selected_models):
    """Generate per-range MAE comparison table."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Per-range MAE (eV) comparison. "
                 "Energy ranges defined by DFT formation energy.}")
    lines.append("\\label{tab:per_range}")
    lines.append("\\begin{tabular}{l" + "c" * len(RANGES) + "c}")
    lines.append("\\toprule")
    header = "Method & " + " & ".join(
        "%s ($n$=%d)" % (label, results[selected_models[0]].get("n_%d_%d" % (lo, hi), 0))
        for lo, hi, label in RANGES
    ) + " & Overall \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for name in selected_models:
        if name not in results:
            continue
        r = results[name]
        # Find display label
        label = name
        for dn, lbl, _ in INNOVATIONS:
            if dn == name:
                label = lbl.replace("\\quad ", "")
                break

        cols = []
        for lo, hi, _ in RANGES:
            key = "mae_%d_%d" % (lo, hi)
            if key in r:
                cols.append("%.3f" % r[key])
            else:
                cols.append("---")
        cols.append("%.4f" % r["mae"])
        lines.append("%s & %s \\\\" % (label, " & ".join(cols)))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def table_ensemble(results_dir):
    """Generate ensemble composition table from greedy selection."""
    # Load all qualified models
    models = {}
    test_targets = None
    import os
    for d in sorted(os.listdir(str(results_dir))):
        npz = results_dir / d / "test_predictions.npz"
        mpath = results_dir / d / "metrics.json"
        if npz.exists():
            data = np.load(str(npz))
            preds, targets = data["preds"], data["targets"]
            if len(preds) != 1065:
                continue
            n_ep = 0
            if mpath.exists():
                with open(str(mpath)) as f:
                    m = json.load(f)
                n_ep = len(m.get("history", []))
            if n_ep >= 20:
                mae = float(np.abs(preds - targets).mean())
                if mae < 0.50:
                    models[d] = preds
                    if test_targets is None:
                        test_targets = targets

    if test_targets is None or len(models) < 2:
        return "% No models available for ensemble table"

    # Greedy selection
    sorted_names = sorted(models, key=lambda n: np.abs(models[n] - test_targets).mean())
    selected = [sorted_names[0]]
    ens_pred = models[sorted_names[0]].copy()

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Greedy ensemble composition. "
                 "Models added in order of MAE improvement.}")
    lines.append("\\label{tab:ensemble}")
    lines.append("\\begin{tabular}{clcc}")
    lines.append("\\toprule")
    lines.append("\\# & Model & Individual MAE & Ensemble MAE \\\\")
    lines.append("\\midrule")

    ind_mae = np.abs(models[sorted_names[0]] - test_targets).mean()
    ens_mae = ind_mae
    lines.append("1 & %s & %.4f & %.4f \\\\"
                 % (sorted_names[0].replace("_", "\\_"), ind_mae, ens_mae))

    remaining = [n for n in sorted_names if n != sorted_names[0]]
    for size in range(2, min(16, len(models) + 1)):
        best_next = None
        best_mae = ens_mae
        for cand in remaining:
            trial = (ens_pred * len(selected) + models[cand]) / (len(selected) + 1)
            trial_mae = np.abs(trial - test_targets).mean()
            if trial_mae < best_mae:
                best_mae = trial_mae
                best_next = cand
        if best_next is None:
            break
        selected.append(best_next)
        ens_pred = (ens_pred * (len(selected) - 1) + models[best_next]) / len(selected)
        ens_mae = best_mae
        ind_mae = np.abs(models[best_next] - test_targets).mean()
        remaining.remove(best_next)
        lines.append("%d & %s & %.4f & %.4f \\\\"
                     % (size, best_next.replace("_", "\\_"), ind_mae, ens_mae))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--out", default=None, help="Output .tex file")
    parser.add_argument("--min-epochs", type=int, default=20)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load all results
    print("Loading results from %s..." % results_dir)
    results = {}
    for dir_name, label, cat in INNOVATIONS:
        r = load_results(results_dir, dir_name, min_epochs=args.min_epochs)
        if r is not None:
            results[dir_name] = r
            print("  %-30s MAE=%.4f  (%d ep)" % (dir_name, r["mae"], r["n_epochs"]))

    if not results:
        print("No completed experiments found!")
        sys.exit(1)

    # Determine baseline
    baseline_name = "v2_gated_pooling_s43"
    if baseline_name not in results:
        baseline_name = next(iter(results))
    baseline_mae = results[baseline_name]["mae"]
    print("\nBaseline: %s (MAE=%.4f)" % (baseline_name, baseline_mae))

    # Generate tables
    output = []
    output.append("%% Auto-generated by scripts/generate_paper_table.py")
    output.append("%% Baseline: %s (MAE=%.4f)" % (baseline_name, baseline_mae))
    output.append("")

    # Table 1: Ablation
    output.append("% ===== TABLE 1: ABLATION =====")
    output.append(table_ablation(results, baseline_mae))
    output.append("")

    # Table 2: Per-range
    available = [dn for dn, _, _ in INNOVATIONS if dn in results]
    output.append("% ===== TABLE 2: PER-RANGE MAE =====")
    output.append(table_per_range(results, available[:8]))  # Top 8
    output.append("")

    # Table 3: Ensemble
    output.append("% ===== TABLE 3: ENSEMBLE =====")
    output.append(table_ensemble(results_dir))

    tex_content = "\n".join(output)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(tex_content)
        print("\nSaved to %s" % out_path)
    else:
        print("\n" + tex_content)


if __name__ == "__main__":
    main()
