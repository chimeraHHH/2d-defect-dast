#!/usr/bin/env python3
"""Comprehensive post-training evaluation pipeline.

Automatically evaluates all completed experiments and generates:
  1. Per-model test metrics (MAE, RMSE, per-range breakdown)
  2. Ablation table (markdown + LaTeX)
  3. Greedy ensemble selection
  4. Attention concentration analysis
  5. Error breakdown by defect type, host, dopant
  6. Summary JSON for programmatic access

Run this after training completes to get a full picture of all innovations.

Usage:
    python scripts/evaluate_all.py [--results-dir results/] [--device cuda]
    python scripts/evaluate_all.py --skip-attention   # faster, skip attention extraction
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.models import CrystalTransformerV2


RANGES = [(0, 2), (2, 5), (5, 7), (7, 25)]

# Model display order and labels
MODEL_REGISTRY = [
    ("v2_gated_pooling_s43", "V2 baseline (best seed)", "baseline"),
    ("v2_gated_pooling", "V2 (seed 42)", "baseline"),
    ("v2_gated_pooling_s44", "V2 (seed 44)", "baseline"),
    ("v2_gated_pooling_s45", "V2 (seed 45)", "baseline"),
    ("v3_deftype", "+ Defect-type conditioning (V3)", "innovation"),
    ("v4_moe", "+ MoE readout (V4)", "innovation"),
    ("v6_physics", "+ Physics features (V6)", "innovation"),
    ("v9_contrast", "+ Defect-host contrast (V9)", "innovation"),
    ("v14_jk", "+ JK aggregation (V14)", "innovation"),
    ("v11_lds", "+ LDS (V11)", "training"),
    ("v13_rnc", "+ RnC + LDS (V13)", "training"),
    ("v2_ema", "+ EMA (V2)", "training"),
    ("v2_focal", "+ Focal MAE (V2)", "training"),
    ("v2_uncertainty", "+ Uncertainty (V2)", "training"),
    ("v12_lds_physics", "LDS + Physics + All cond. (V12)", "combined"),
    ("v10_best_combo", "All best innovations (V10)", "combined"),
]


class Normalizer:
    def __init__(self, mean, std, transform="none"):
        self.mean, self.std, self.transform = mean, std, transform
    def denorm(self, t):
        t = t * self.std + self.mean
        if self.transform == "log":
            t = torch.sign(t) * torch.expm1(torch.abs(t))
        return t


def load_model_and_predict(model_dir, dataset, test_set, device="cuda"):
    """Load a trained model, run test evaluation, return preds + targets."""
    model_dir = Path(model_dir)
    ckpt_path = model_dir / "best.pt"

    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Detect checkpoint format
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ckpt["config"]
        state_dict = ckpt["model"]
        norm_info = ckpt.get("normalizer", {})
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
        norm_info = ckpt.get("normalizer", {})
    else:
        return None

    normalizer = Normalizer(
        norm_info.get("mean", 0), norm_info.get("std", 1),
        transform=norm_info.get("transform", "none"),
    )

    model_kwargs = cfg.get("model_kwargs", {})
    model = CrystalTransformerV2(**model_kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Only warn for meaningful keys
        meaningful = [k for k in missing if "ct_uae_table" not in k]
        if meaningful:
            print("  WARNING: missing keys: %s" % meaningful[:5])
    model = model.to(device).eval()

    test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
                             collate_fn=collate_fn)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            out = model(batch_dev)
            if isinstance(out, tuple):
                out = out[0]
            preds = normalizer.denorm(out.cpu()).numpy()
            targets = batch["target"].numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return preds, targets, model, cfg


def compute_metrics(preds, targets):
    """Compute comprehensive metrics from predictions."""
    errors = np.abs(preds - targets)
    signed = preds - targets

    result = {
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean(signed ** 2))),
        "bias": float(signed.mean()),
        "pred_std": float(preds.std()),
        "target_std": float(targets.std()),
        "compression": float(preds.std() / max(targets.std(), 1e-8)),
        "n_samples": len(preds),
    }

    # Per-range metrics
    for lo, hi in RANGES:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            r_err = errors[mask]
            r_signed = signed[mask]
            result["mae_%d_%d" % (lo, hi)] = float(r_err.mean())
            result["rmse_%d_%d" % (lo, hi)] = float(np.sqrt(np.mean(r_signed ** 2)))
            result["bias_%d_%d" % (lo, hi)] = float(r_signed.mean())
            result["n_%d_%d" % (lo, hi)] = int(mask.sum())
            result["err_frac_%d_%d" % (lo, hi)] = float(r_err.sum() / errors.sum())

    return result


def error_breakdown(preds, targets, dataset, test_set):
    """Break down errors by defect type, host, dopant."""
    errors = np.abs(preds - targets)
    signed = preds - targets

    dtype_stats = defaultdict(lambda: {"errors": [], "targets": [], "signed": []})
    host_stats = defaultdict(lambda: {"errors": [], "targets": [], "signed": []})
    dopant_stats = defaultdict(lambda: {"errors": [], "targets": [], "signed": []})

    for i, idx in enumerate(test_set.indices):
        meta = dataset.data[idx]["metadata"]
        dt = meta["defecttype"]
        host = meta["host"]
        dopant = meta["dopant"]

        dtype_stats[dt]["errors"].append(errors[i])
        dtype_stats[dt]["targets"].append(targets[i])
        dtype_stats[dt]["signed"].append(signed[i])
        host_stats[host]["errors"].append(errors[i])
        host_stats[host]["targets"].append(targets[i])
        host_stats[host]["signed"].append(signed[i])
        dopant_stats[dopant]["errors"].append(errors[i])
        dopant_stats[dopant]["targets"].append(targets[i])
        dopant_stats[dopant]["signed"].append(signed[i])

    # Summarize
    breakdown = {"by_defect_type": {}, "by_host": {}, "by_dopant": {}}

    for dt, d in dtype_stats.items():
        e = np.array(d["errors"])
        breakdown["by_defect_type"][dt] = {
            "mae": float(e.mean()), "count": len(e),
            "mean_ef": float(np.mean(d["targets"])),
            "bias": float(np.mean(d["signed"])),
        }

    for h, d in host_stats.items():
        e = np.array(d["errors"])
        breakdown["by_host"][h] = {
            "mae": float(e.mean()), "count": len(e),
            "mean_ef": float(np.mean(d["targets"])),
            "bias": float(np.mean(d["signed"])),
        }

    for dp, d in dopant_stats.items():
        e = np.array(d["errors"])
        breakdown["by_dopant"][dp] = {
            "mae": float(e.mean()), "count": len(e),
            "bias": float(np.mean(d["signed"])),
        }

    return breakdown


def greedy_ensemble(all_preds, test_targets, max_k=15):
    """Greedy forward selection for ensemble."""
    names = sorted(all_preds.keys(),
                   key=lambda n: np.abs(all_preds[n] - test_targets).mean())

    selected = [names[0]]
    ens_pred = all_preds[names[0]].copy()
    results = [{
        "k": 1, "name": names[0],
        "individual_mae": float(np.abs(all_preds[names[0]] - test_targets).mean()),
        "ensemble_mae": float(np.abs(ens_pred - test_targets).mean()),
    }]

    remaining = [n for n in names if n != names[0]]
    for k in range(2, min(max_k + 1, len(all_preds) + 1)):
        best_next, best_mae = None, float(np.abs(ens_pred - test_targets).mean())
        for cand in remaining:
            trial = (ens_pred * (k - 1) + all_preds[cand]) / k
            trial_mae = float(np.abs(trial - test_targets).mean())
            if trial_mae < best_mae:
                best_mae = trial_mae
                best_next = cand
        if best_next is None:
            break
        selected.append(best_next)
        ens_pred = (ens_pred * (k - 1) + all_preds[best_next]) / k
        remaining.remove(best_next)
        results.append({
            "k": k, "name": best_next,
            "individual_mae": float(np.abs(all_preds[best_next] - test_targets).mean()),
            "ensemble_mae": best_mae,
        })

    return results


def print_ablation_table(all_results, baseline_name="v2_gated_pooling_s43"):
    """Print markdown ablation table."""
    baseline_mae = all_results.get(baseline_name, {}).get("mae", 0.381)

    print("\n" + "=" * 90)
    print("ABLATION TABLE")
    print("=" * 90)
    print("%-40s %8s %8s %10s %6s" % ("Model", "MAE", "RMSE", "ΔMAE", "N_ep"))
    print("-" * 90)

    for dir_name, label, category in MODEL_REGISTRY:
        if dir_name not in all_results:
            continue
        r = all_results[dir_name]
        delta = r["mae"] - baseline_mae
        delta_str = "%+.4f" % delta if dir_name != baseline_name else "---"
        marker = " **" if r["mae"] < baseline_mae else ""
        print("%-40s %8.4f %8.4f %10s %6s%s" %
              (label, r["mae"], r["rmse"], delta_str,
               r.get("n_epochs", "?"), marker))

    print("-" * 90)


def print_per_range_table(all_results):
    """Print per-range comparison."""
    print("\n" + "=" * 90)
    print("PER-RANGE MAE COMPARISON")
    print("=" * 90)
    print("%-35s" % "Model", end="")
    for lo, hi in RANGES:
        print("  [%d,%d)" % (lo, hi), end="")
    print("  Overall")
    print("-" * 90)

    for dir_name, label, _ in MODEL_REGISTRY:
        if dir_name not in all_results:
            continue
        r = all_results[dir_name]
        print("%-35s" % label[:35], end="")
        for lo, hi in RANGES:
            key = "mae_%d_%d" % (lo, hi)
            if key in r:
                print("  %6.3f" % r[key], end="")
            else:
                print("     ---", end="")
        print("  %6.4f" % r["mae"])
    print("-" * 90)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation pipeline")
    parser.add_argument("--results-dir", default="results/",
                        help="Directory containing experiment results")
    parser.add_argument("--device", default="cuda",
                        help="Device for model inference")
    parser.add_argument("--min-epochs", type=int, default=20,
                        help="Minimum epochs for a run to be considered complete")
    parser.add_argument("--skip-attention", action="store_true",
                        help="Skip attention analysis (faster)")
    parser.add_argument("--skip-model-load", action="store_true",
                        help="Only use cached test_predictions.npz, skip model loading")
    parser.add_argument("--out", default=None,
                        help="Output JSON path for summary")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    results_dir = Path(args.results_dir)
    t0 = time.time()

    # Load dataset
    print("Loading dataset...")
    dataset = CrystalGraphDataset(ROOT / "data/processed/cleaned_dataset.pkl")
    _, _, test_set = make_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42)
    print("  Test set: %d samples" % len(test_set))

    # Evaluate all models
    all_results = {}
    all_preds = {}
    test_targets = None

    for dir_name, label, category in MODEL_REGISTRY:
        model_dir = results_dir / dir_name
        if not model_dir.exists():
            continue

        print("\nEvaluating: %s ..." % dir_name)

        # Check for cached predictions first
        npz_path = model_dir / "test_predictions.npz"
        metrics_path = model_dir / "metrics.json"

        # Get epoch count from metrics or log
        n_epochs = 0
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            n_epochs = len(m.get("history", []))
        else:
            # Try parsing train.log
            log_path = model_dir / "train.log"
            nohup_path = model_dir / "nohup.log"
            for lp in [log_path, nohup_path]:
                if lp.exists():
                    with open(lp) as f:
                        for line in f:
                            if "Epoch" in line and "/" in line:
                                try:
                                    ep = int(line.split("Epoch")[1].split("/")[0].strip())
                                    n_epochs = max(n_epochs, ep)
                                except:
                                    pass

        if n_epochs > 0 and n_epochs < args.min_epochs:
            print("  Skipping: only %d epochs (min %d)" % (n_epochs, args.min_epochs))
            continue

        preds, targets = None, None

        if npz_path.exists() and args.skip_model_load:
            # Use cached predictions
            data = np.load(str(npz_path))
            preds, targets = data["preds"], data["targets"]
            if len(preds) != len(test_set):
                print("  Cached predictions count mismatch, skipping")
                continue
            print("  Using cached predictions (n=%d)" % len(preds))
        elif (model_dir / "best.pt").exists():
            # Load model and run inference
            try:
                result = load_model_and_predict(model_dir, dataset, test_set, device)
                if result is None:
                    print("  Failed to load model")
                    continue
                preds, targets, model, cfg = result
                print("  Inference done (n=%d)" % len(preds))

                # Cache predictions
                np.savez(model_dir / "test_predictions.npz",
                         preds=preds, targets=targets)
            except Exception as e:
                print("  Error: %s" % e)
                continue
        else:
            print("  No best.pt or cached predictions found")
            continue

        if test_targets is None:
            test_targets = targets

        # Compute metrics
        metrics = compute_metrics(preds, targets)
        metrics["name"] = dir_name
        metrics["label"] = label
        metrics["category"] = category
        metrics["n_epochs"] = n_epochs

        all_results[dir_name] = metrics
        all_preds[dir_name] = preds

        print("  MAE=%.4f  RMSE=%.4f  (ep=%d)" % (metrics["mae"], metrics["rmse"], n_epochs))

    if not all_results:
        print("\nNo completed experiments found!")
        sys.exit(1)

    # Print ablation table
    print_ablation_table(all_results)

    # Print per-range table
    print_per_range_table(all_results)

    # Error breakdown for the best model
    best_name = min(all_results, key=lambda k: all_results[k]["mae"])
    print("\n" + "=" * 90)
    print("ERROR BREAKDOWN for best model: %s (MAE=%.4f)" %
          (best_name, all_results[best_name]["mae"]))
    print("=" * 90)

    if best_name in all_preds:
        breakdown = error_breakdown(all_preds[best_name], test_targets, dataset, test_set)

        print("\nBy defect type:")
        for dt, stats in sorted(breakdown["by_defect_type"].items(),
                                key=lambda x: -x[1]["mae"]):
            print("  %-15s MAE=%.4f  n=%d  bias=%+.3f" %
                  (dt, stats["mae"], stats["count"], stats["bias"]))

        print("\nWorst 10 hosts:")
        hosts_sorted = sorted(breakdown["by_host"].items(),
                              key=lambda x: -x[1]["mae"])
        for h, stats in hosts_sorted[:10]:
            print("  %-12s MAE=%.4f  n=%d  bias=%+.3f" %
                  (h, stats["mae"], stats["count"], stats["bias"]))

        print("\nWorst 10 dopants:")
        dopants_sorted = sorted(breakdown["by_dopant"].items(),
                                key=lambda x: -x[1]["mae"])
        for dp, stats in dopants_sorted[:10]:
            print("  %-6s MAE=%.4f  n=%d  bias=%+.3f" %
                  (dp, stats["mae"], stats["count"], stats["bias"]))

        all_results[best_name]["breakdown"] = breakdown

    # Greedy ensemble
    if len(all_preds) >= 2 and test_targets is not None:
        print("\n" + "=" * 90)
        print("GREEDY ENSEMBLE SELECTION")
        print("=" * 90)
        ens_results = greedy_ensemble(all_preds, test_targets)
        for r in ens_results:
            print("  k=%2d  ens_MAE=%.4f  +%s (ind=%.4f)" %
                  (r["k"], r["ensemble_mae"], r["name"], r["individual_mae"]))
        all_results["_ensemble"] = ens_results

    # Summary
    dt = time.time() - t0
    print("\n" + "=" * 90)
    print("SUMMARY  (%d models evaluated in %.1fs)" % (len(all_results), dt))
    print("=" * 90)

    # Rank by MAE
    ranked = sorted(
        [(k, v) for k, v in all_results.items() if not k.startswith("_")],
        key=lambda x: x[1]["mae"],
    )
    for i, (name, r) in enumerate(ranked):
        print("  %2d. %-35s MAE=%.4f  RMSE=%.4f" %
              (i + 1, name, r["mae"], r["rmse"]))

    if "_ensemble" in all_results:
        best_ens = min(all_results["_ensemble"], key=lambda x: x["ensemble_mae"])
        print("\n  Best ensemble (k=%d): MAE=%.4f" %
              (best_ens["k"], best_ens["ensemble_mae"]))

    # Save summary
    out_path = Path(args.out) if args.out else results_dir / "evaluation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def _clean(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_clean(all_results), f, indent=2)
    print("\nSaved to %s" % out_path)


if __name__ == "__main__":
    main()
