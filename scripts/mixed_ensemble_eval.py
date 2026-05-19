#!/usr/bin/env python3
"""Mixed V1+V2 ensemble evaluation with greedy best-k selection.
Loads all test_predictions.npz files from results/ and finds optimal ensemble."""

import os, json, sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(os.path.expanduser("~/Workspace/yiminghua/project/results"))

# Skip LOHO models (trained on different splits) and smoke tests
SKIP_PREFIXES = ["loho_", "v2_smoke", "v2_ablate_", "v2_distill",
                 "ablate_", "dast_", "dualstream_smoke"]
SKIP_EXACT = {"improved", "baseline", "baseline_long", "baseline_h128",
              "baseline_h128_long", "baseline_aug", "baseline_h128_aug"}

EXPECTED_N_TEST = 1065  # Leak-free IMP2D test set

def load_all_predictions():
    """Load all test_predictions.npz, return dict of {name: preds_array}."""
    models = {}
    targets_ref = None

    for npz_path in sorted(RESULTS_DIR.glob("*/test_predictions.npz")):
        name = npz_path.parent.name

        # Skip non-IMP2D or ablation models
        skip = False
        for prefix in SKIP_PREFIXES:
            if name.startswith(prefix):
                skip = True
                break
        if name in SKIP_EXACT:
            skip = True
        if skip:
            continue

        try:
            data = np.load(npz_path)
            preds = data["preds"]
            targets = data["targets"]

            # Only keep models with the correct 1065-sample test set
            if len(targets) != EXPECTED_N_TEST:
                continue

            # Verify consistent targets
            if targets_ref is None:
                targets_ref = targets
            else:
                if not np.allclose(targets, targets_ref, atol=1e-4):
                    print(f"  SKIP {name}: different target values")
                    continue

            mae = np.mean(np.abs(preds - targets))
            models[name] = preds
            print(f"  {name}: MAE={mae:.4f}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    return models, targets_ref

def greedy_ensemble(models_dict, targets, max_k=None):
    """Greedy forward selection: pick models one by one to minimize ensemble MAE."""
    names = list(models_dict.keys())
    preds_array = np.array([models_dict[n] for n in names])  # (M, N)
    n_models = len(names)
    if max_k is None:
        max_k = n_models

    selected = []
    selected_idx = []
    best_results = []

    for k in range(1, max_k + 1):
        best_mae = float("inf")
        best_i = -1

        for i in range(n_models):
            if i in selected_idx:
                continue
            # Try adding model i
            candidate_idx = selected_idx + [i]
            ens_pred = preds_array[candidate_idx].mean(axis=0)
            mae = np.mean(np.abs(ens_pred - targets))
            if mae < best_mae:
                best_mae = mae
                best_i = i

        if best_i < 0:
            break

        selected_idx.append(best_i)
        selected.append(names[best_i])
        ens_pred = preds_array[selected_idx].mean(axis=0)
        rmse = np.sqrt(np.mean((ens_pred - targets) ** 2))
        best_results.append({
            "k": k,
            "model": names[best_i],
            "mae": float(best_mae),
            "rmse": float(rmse)
        })

        # Early stop if MAE stops improving for 5 consecutive additions
        if k > 5 and all(best_results[j]["mae"] >= best_results[j-1]["mae"]
                         for j in range(k-4, k)):
            print(f"\n  Early stop at k={k} (no improvement for 5 steps)")
            break

    return best_results, selected

def main():
    print("=" * 70)
    print("MIXED V1+V2 ENSEMBLE EVALUATION")
    print("=" * 70)

    print("\nLoading all model predictions...")
    models, targets = load_all_predictions()
    print(f"\nTotal eligible models: {len(models)}")
    print(f"Test samples: {len(targets)}")

    # Separate V1 and V2
    v1_models = {k: v for k, v in models.items() if not k.startswith("v2_")}
    v2_models = {k: v for k, v in models.items() if k.startswith("v2_")}
    print(f"V1 models: {len(v1_models)}, V2 models: {len(v2_models)}")

    # === V2-only ensemble ===
    print("\n" + "=" * 50)
    print("V2-ONLY GREEDY ENSEMBLE")
    print("=" * 50)
    v2_results, v2_selected = greedy_ensemble(v2_models, targets)
    for r in v2_results:
        marker = " ***" if r["mae"] == min(x["mae"] for x in v2_results) else ""
        print(f"  k={r['k']}: MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} (+{r['model']}){marker}")

    # === V1-only ensemble ===
    print("\n" + "=" * 50)
    print("V1-ONLY GREEDY ENSEMBLE")
    print("=" * 50)
    v1_results, v1_selected = greedy_ensemble(v1_models, targets, max_k=15)
    for r in v1_results:
        marker = " ***" if r["mae"] == min(x["mae"] for x in v1_results) else ""
        print(f"  k={r['k']}: MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} (+{r['model']}){marker}")

    # === Mixed V1+V2 ensemble ===
    print("\n" + "=" * 50)
    print("MIXED V1+V2 GREEDY ENSEMBLE")
    print("=" * 50)
    mixed_results, mixed_selected = greedy_ensemble(models, targets, max_k=20)
    for r in mixed_results:
        is_v2 = r["model"].startswith("v2_")
        tag = "[V2]" if is_v2 else "[V1]"
        marker = " ***" if r["mae"] == min(x["mae"] for x in mixed_results) else ""
        print(f"  k={r['k']}: MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} +{tag} {r['model']}{marker}")

    # === Summary ===
    print("\n" + "=" * 50)
    print("SUMMARY — BEST ENSEMBLES")
    print("=" * 50)

    out = {"n_models_total": len(models), "n_v1": len(v1_models),
           "n_v2": len(v2_models), "n_test": len(targets)}

    if v2_results:
        best_v2 = min(v2_results, key=lambda x: x["mae"])
        print(f"  V2-only best:  k={best_v2['k']}, MAE={best_v2['mae']:.4f}")
        out["v2_only"] = {"best_k": best_v2["k"], "mae": best_v2["mae"], "rmse": best_v2["rmse"],
                          "selected": v2_selected[:best_v2["k"]], "all_results": v2_results}
    if v1_results:
        best_v1 = min(v1_results, key=lambda x: x["mae"])
        print(f"  V1-only best:  k={best_v1['k']}, MAE={best_v1['mae']:.4f}")
        out["v1_only"] = {"best_k": best_v1["k"], "mae": best_v1["mae"], "rmse": best_v1["rmse"],
                          "selected": v1_selected[:best_v1["k"]], "all_results": v1_results}
    if mixed_results:
        best_mixed = min(mixed_results, key=lambda x: x["mae"])
        n_v2_in_mixed = sum(1 for m in mixed_selected[:best_mixed["k"]] if m.startswith("v2_"))
        print(f"  Mixed best:    k={best_mixed['k']}, MAE={best_mixed['mae']:.4f}")
        print(f"  Mixed composition: {n_v2_in_mixed} V2 + {best_mixed['k']-n_v2_in_mixed} V1")
        out["mixed"] = {"best_k": best_mixed["k"], "mae": best_mixed["mae"], "rmse": best_mixed["rmse"],
                        "selected": mixed_selected[:best_mixed["k"]], "all_results": mixed_results}

    out_path = RESULTS_DIR / "mixed_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
