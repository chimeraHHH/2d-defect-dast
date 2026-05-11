"""Combined ensemble evaluation: single-source v4 + multi-source v4 models.

After multi-source v4 training completes, this script evaluates whether
adding multi-source models to the existing 26-model ensemble improves
performance. Multi-source models see 4x more data but have split attention,
so they may provide complementary predictions.

Usage:
  python scripts/ensemble_combined.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.models import CrystalTransformer

RESULTS = ROOT / "results"


def load_single_source_preds():
    """Load pre-computed predictions from ensemble_online.npz."""
    data = np.load(RESULTS / "ensemble_online.npz", allow_pickle=True)
    return {
        "preds_dict": {n: data["individual_preds"][i] for i, n in enumerate(data["model_names"])},
        "targets": data["targets"],
    }


def load_multi_source_preds():
    """Load predictions from multi-source v4 runs (test_predictions.npz)."""
    preds = {}
    for tag in ["multi_source_v4_s42", "multi_source_v4_s43", "multi_source_v4_deep_s42"]:
        npz_path = RESULTS / tag / "test_predictions.npz"
        if npz_path.exists():
            d = np.load(npz_path)
            preds[f"ms4_{tag.split('_')[-1]}"] = d["preds"]
            # Verify targets match
            print(f"  loaded {tag}: {len(d['preds'])} predictions")
    return preds


def greedy_forward_select(all_preds, targets, max_k=10):
    """Greedy forward selection of ensemble members."""
    names = list(all_preds.keys())
    selected = []
    results = []

    for k in range(1, min(max_k + 1, len(names) + 1)):
        best_mae, best_name = float("inf"), None
        for name in names:
            if name in selected:
                continue
            trial = selected + [name]
            P = np.stack([all_preds[n] for n in trial])
            mu = P.mean(axis=0)
            mae = np.abs(mu - targets).mean()
            if mae < best_mae:
                best_mae = mae
                best_name = name
        selected.append(best_name)
        results.append({"k": k, "mae": float(best_mae), "added": best_name, "members": list(selected)})
        print(f"  k={k}: MAE {best_mae:.4f} (added {best_name})")

    return results


def main():
    print("Loading single-source predictions...")
    ss_data = load_single_source_preds()
    targets = ss_data["targets"]
    all_preds = dict(ss_data["preds_dict"])
    print(f"  {len(all_preds)} single-source models, {len(targets)} test samples")

    print("\nLoading multi-source v4 predictions...")
    ms_preds = load_multi_source_preds()
    if not ms_preds:
        print("  No multi-source predictions found yet. Run this after training completes.")
        return

    # Verify targets alignment
    for name, p in ms_preds.items():
        if len(p) != len(targets):
            print(f"  WARNING: {name} has {len(p)} preds vs {len(targets)} targets!")
            continue
        all_preds[name] = p

    print(f"\n  Total models: {len(all_preds)} ({len(ss_data['preds_dict'])} SS + {len(ms_preds)} MS)")

    # Individual MAEs
    print("\n  Multi-source individual MAEs:")
    for name in ms_preds:
        if name in all_preds:
            mae = np.abs(all_preds[name] - targets).mean()
            print(f"    {name}: {mae:.4f}")

    # Greedy forward selection on combined pool
    print("\n=== Greedy Forward Selection (combined pool) ===")
    results = greedy_forward_select(all_preds, targets, max_k=10)

    # Also compare: best-5 SS only vs best with MS
    print("\n=== Comparison ===")
    ss_only = {k: v for k, v in all_preds.items() if not k.startswith("ms4_")}
    names_ss = list(ss_only.keys())
    # Best-5 from single-source
    best5_ss = ["uae_mae_warmup_s46", "deep_s42", "150ep_s42", "150ep_s43", "150ep_s45"]
    P5 = np.stack([all_preds[n] for n in best5_ss])
    mae5_ss = np.abs(P5.mean(axis=0) - targets).mean()
    print(f"  Best-5 (SS only): MAE {mae5_ss:.4f}")

    # Best-5 from combined
    best5_combined = results[4]["members"] if len(results) >= 5 else results[-1]["members"]
    P5c = np.stack([all_preds[n] for n in best5_combined])
    mae5_c = np.abs(P5c.mean(axis=0) - targets).mean()
    print(f"  Best-5 (combined): MAE {mae5_c:.4f} <- {best5_combined}")
    print(f"  Improvement: {(mae5_ss - mae5_c)/mae5_ss*100:.2f}%")

    # Save results
    output = {
        "n_single_source": len(ss_data["preds_dict"]),
        "n_multi_source": len(ms_preds),
        "greedy_selection": results,
        "comparison": {
            "best5_ss": {"mae": float(mae5_ss), "models": best5_ss},
            "best5_combined": {"mae": float(mae5_c), "models": best5_combined},
        },
    }
    out_path = RESULTS / "ensemble_combined.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
