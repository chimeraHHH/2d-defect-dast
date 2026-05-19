#!/usr/bin/env python3
"""Generate ablation table for paper: contribution of each innovation.

Reads test_predictions.npz from each model and computes:
  1. Overall MAE/RMSE
  2. Per-range MAE (especially [7,25) eV bottleneck)
  3. Per-defect-type MAE (interstitial vs adsorbate)
  4. Ensemble potential (greedy ensemble with diversity)
  5. Innovation contribution: delta-MAE vs V2 baseline

Output: LaTeX table + CSV for paper.

Usage:
    python scripts/generate_ablation_table.py [--results-dir results/]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# Innovation registry: maps result directory names to their innovations
# ------------------------------------------------------------------
INNOVATIONS = {
    # V2 ablation
    "v2_ablate_pooling_only": {"label": "V2 (pooling only)", "innovations": ["pooling"]},
    "v2_ablate_env_only": {"label": "V2 (env only)", "innovations": ["env"]},
    "v2_ablate_prenorm_only": {"label": "V2 (prenorm only)", "innovations": ["prenorm"]},
    # V2 full (baseline for innovations)
    "v2_gated_pooling": {"label": "V2 (full, s42)", "innovations": ["pooling", "env", "prenorm"]},
    "v2_gated_pooling_s43": {"label": "V2 (full, s43)", "innovations": ["pooling", "env", "prenorm"]},
    "v2_gated_pooling_s44": {"label": "V2 (full, s44)", "innovations": ["pooling", "env", "prenorm"]},
    "v2_gated_pooling_s45": {"label": "V2 (full, s45)", "innovations": ["pooling", "env", "prenorm"]},
    # Individual innovations (added on top of V2 full)
    "v3_deftype": {"label": "+ Defect-type cond. (V3)", "innovations": ["deftype"]},
    "v4_moe": {"label": "+ MoE readout (V4)", "innovations": ["moe", "deftype"]},
    "v6_physics": {"label": "+ Physics features (V6)", "innovations": ["physics"]},
    "v9_contrast": {"label": "+ D-H contrast (V9)", "innovations": ["contrast"]},
    "v2_ema": {"label": "+ EMA", "innovations": ["ema"]},
    "v2_asph": {"label": "+ ASPH topo (V5)", "innovations": ["asph"]},
    "v2_focal": {"label": "+ Focal MAE", "innovations": ["focal"]},
    "v2_uncertainty": {"label": "+ Uncertainty", "innovations": ["uncertainty"]},
    "v11_lds": {"label": "+ LDS (V11)", "innovations": ["lds"]},
    "v12_lds_physics": {"label": "LDS+Phys+Cond (V12)", "innovations": ["lds", "physics", "deftype", "contrast"]},
    # Combined
    "v8_ema_physics": {"label": "EMA + Physics + Deftype", "innovations": ["ema", "physics", "deftype"]},
    "v10_best_combo": {"label": "V10 (all best)", "innovations": ["ema", "physics", "deftype", "contrast", "jk"]},
    "v7_all": {"label": "V7 (all + MoE + ASPH)", "innovations": ["deftype", "moe", "physics", "asph"]},
    # Enhanced env (V2 + stronger env enrichment)
    "v2_enhanced_env": {"label": "V2 + enhanced env", "innovations": ["pooling", "env_v2", "prenorm"]},
    # Distillation seeds
    "v2_distill": {"label": "V2 distill s42", "innovations": ["distill"]},
    "v2_distill_s43": {"label": "V2 distill s43", "innovations": ["distill"]},
}


def load_model(results_dir, name, min_epochs=20):
    """Load a model's predictions if available and sufficiently trained."""
    npz_path = results_dir / name / "test_predictions.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    preds, targets = data["preds"], data["targets"]
    if len(preds) != 1065:
        return None
    # Also load metrics for n_params and epoch count
    metrics_path = results_dir / name / "metrics.json"
    n_params = 0
    n_epochs = 0
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
            n_params = m.get("n_params", 0)
            n_epochs = len(m.get("history", []))
    # Skip smoke tests and under-trained models
    if n_epochs < min_epochs:
        return None
    return {"preds": preds, "targets": targets, "n_params": n_params,
            "n_epochs": n_epochs}


def per_range_mae(preds, targets):
    """Compute MAE per target-value range."""
    bins = [(-np.inf, 0), (0, 2), (2, 5), (5, 7), (7, 25)]
    results = {}
    for lo, hi in bins:
        mask = (targets >= lo) & (targets < hi)
        n = mask.sum()
        if n > 0:
            results["[%s,%s)" % (str(lo), str(hi))] = {
                "n": int(n),
                "mae": float(np.abs(preds[mask] - targets[mask]).mean()),
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--data-path", default="data/processed/cleaned_dataset.pkl")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load all available models
    models = {}
    for name in INNOVATIONS:
        m = load_model(results_dir, name)
        if m is not None:
            models[name] = m

    if not models:
        print("No models found with test predictions!")
        sys.exit(1)

    targets = list(models.values())[0]["targets"]

    # Load defect types for per-type analysis
    defect_types = None
    try:
        data_path = Path(args.data_path)
        if data_path.exists():
            with open(data_path, "rb") as f:
                blob = pickle.load(f)
            data = blob["data"] if isinstance(blob, dict) and "data" in blob else blob
            n_total = len(data)
            rng = np.random.RandomState(42)
            indices = rng.permutation(n_total)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)
            test_indices = indices[n_train + n_val:]
            if len(test_indices) == 1065:
                defect_types = np.array([data[idx]["metadata"]["defecttype"]
                                         for idx in test_indices])
    except Exception as e:
        print("Warning: could not load defect types: %s" % e)

    # Find V2 baseline MAE for computing deltas
    baseline_mae = None
    for name in ["v2_gated_pooling_s43", "v2_gated_pooling"]:
        if name in models:
            baseline_mae = np.abs(models[name]["preds"] - targets).mean()
            baseline_name = name
            break

    # Print ablation table
    print("=" * 100)
    print("ABLATION TABLE: Innovation Contribution to Defect Ef Prediction")
    print("=" * 100)
    print()

    header = "%-30s %8s %8s %8s %8s %8s %8s" % (
        "Model", "MAE", "RMSE", "Delta",
        "MAE[7+]", "MAE_int", "Params"
    )
    print(header)
    print("-" * 100)

    # Order: ablations first, then innovations, then combined
    order = [
        # V2 ablations
        "v2_ablate_pooling_only", "v2_ablate_env_only", "v2_ablate_prenorm_only",
        # V2 full seeds
        "v2_gated_pooling", "v2_gated_pooling_s43", "v2_gated_pooling_s44", "v2_gated_pooling_s45",
        # Individual innovations
        "v3_deftype", "v4_moe", "v6_physics", "v9_contrast",
        "v2_ema", "v2_asph", "v2_focal", "v2_uncertainty",
        "v11_lds", "v12_lds_physics",
        # Combined
        "v8_ema_physics", "v10_best_combo", "v7_all",
        # Distillation
        "v2_distill", "v2_distill_s43",
    ]

    for name in order:
        if name not in models:
            continue
        m = models[name]
        preds = m["preds"]
        mae = np.abs(preds - targets).mean()
        rmse = np.sqrt(np.mean((preds - targets) ** 2))

        # Delta vs baseline
        delta_str = ""
        if baseline_mae is not None:
            delta = mae - baseline_mae
            delta_str = "%+.4f" % delta

        # [7,25) MAE
        high_mask = targets >= 7
        mae_high = np.abs(preds[high_mask] - targets[high_mask]).mean() if high_mask.any() else 0

        # Interstitial MAE
        mae_int = ""
        if defect_types is not None:
            int_mask = defect_types == "interstitial"
            if int_mask.any():
                mae_int = "%.4f" % np.abs(preds[int_mask] - targets[int_mask]).mean()

        # Params
        params_str = "%.3fM" % (m["n_params"] / 1e6) if m["n_params"] > 0 else "?"

        label = INNOVATIONS.get(name, {}).get("label", name)
        print("%-30s %8.4f %8.4f %8s %8.4f %8s %8s" % (
            label, mae, rmse, delta_str, mae_high, mae_int, params_str
        ))

    # Ensemble analysis
    print()
    print("=" * 100)
    print("ENSEMBLE ANALYSIS")
    print("=" * 100)

    # Greedy ensemble selection
    model_names = sorted(models.keys(), key=lambda n: np.abs(models[n]["preds"] - targets).mean())
    best_name = model_names[0]
    selected = [best_name]
    ensemble_preds = models[best_name]["preds"].copy()
    best_mae = np.abs(ensemble_preds - targets).mean()
    print("Size 1: %s → MAE %.4f" % (best_name, best_mae))

    remaining = [n for n in model_names if n != best_name]
    for step in range(min(9, len(remaining))):
        best_next = None
        best_next_mae = best_mae
        for candidate in remaining:
            trial = (ensemble_preds * len(selected) + models[candidate]["preds"]) / (len(selected) + 1)
            trial_mae = np.abs(trial - targets).mean()
            if trial_mae < best_next_mae:
                best_next_mae = trial_mae
                best_next = candidate
        if best_next is None:
            break
        selected.append(best_next)
        ensemble_preds = (ensemble_preds * (len(selected) - 1) + models[best_next]["preds"]) / len(selected)
        best_mae = best_next_mae
        remaining.remove(best_next)
        label = INNOVATIONS.get(best_next, {}).get("label", best_next)
        print("Size %d: + %s → MAE %.4f" % (len(selected), label, best_mae))

    # Prediction diversity (correlation matrix)
    print()
    print("Prediction correlation (lower = more diverse):")
    innovation_models = [n for n in model_names if n in INNOVATIONS and
                         any(i in INNOVATIONS[n].get("innovations", [])
                             for i in ["deftype", "moe", "physics", "contrast", "asph", "focal", "uncertainty"])]
    for i, n1 in enumerate(innovation_models[:8]):
        for j, n2 in enumerate(innovation_models[:8]):
            if j > i:
                r = np.corrcoef(models[n1]["preds"], models[n2]["preds"])[0, 1]
                if r < 0.995:
                    print("  %s vs %s: r=%.4f (diverse!)" % (
                        INNOVATIONS[n1]["label"][:20], INNOVATIONS[n2]["label"][:20], r))


if __name__ == "__main__":
    main()
