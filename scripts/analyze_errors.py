#!/usr/bin/env python3
"""Analyze test set error patterns for the best single V2 model."""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def main():
    # Load dataset
    with open(ROOT / "data/processed/cleaned_dataset.pkl", "rb") as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and "data" in blob:
        data = blob["data"]
        meta = blob.get("meta", {})
    else:
        data = blob
        meta = {}

    # Get test indices
    n = len(data)
    if "n_train" in meta:
        n_tr = meta["n_train"]
        n_va = meta["n_val"]
        test_idx = list(range(n_tr + n_va, n_tr + n_va + meta["n_test"]))
    else:
        import random
        rng = random.Random(42)
        idx = list(range(n))
        rng.shuffle(idx)
        n_tr = int(0.8 * n)
        n_va = int(0.1 * n)
        test_idx = idx[n_tr + n_va:]

    # Load best model predictions
    preds_data = np.load(ROOT / "results/v2_gated_pooling_s43/test_predictions.npz")
    preds = preds_data["preds"]
    targets = preds_data["targets"]

    # Analyze errors by host and dopant
    host_errors = defaultdict(list)
    dopant_errors = defaultdict(list)
    target_bins = defaultdict(list)

    for i, tidx in enumerate(test_idx[:len(preds)]):
        sample = data[tidx]
        md = sample.get("metadata", {})
        host = md.get("host", "unknown")
        dopant = md.get("dopant", "unknown")
        err = abs(float(preds[i] - targets[i]))

        host_errors[host].append(err)
        dopant_errors[dopant].append(err)

        t = abs(float(targets[i]))
        if t < 1:
            target_bins["<1 eV"].append(err)
        elif t < 3:
            target_bins["1-3 eV"].append(err)
        elif t < 6:
            target_bins["3-6 eV"].append(err)
        else:
            target_bins[">6 eV"].append(err)

    print("=== Error by Host Material (top 15 by count) ===")
    hosts_sorted = sorted(host_errors.items(), key=lambda x: -len(x[1]))
    for host, errs in hosts_sorted[:15]:
        errs = np.array(errs)
        print(f"  {host:20s}: n={len(errs):4d}  MAE={errs.mean():.3f}  max={errs.max():.3f}")

    print("\n=== Hardest Hosts (MAE > 0.5, n >= 5) ===")
    hard_hosts = [(h, e) for h, e in host_errors.items() if len(e) >= 5 and np.mean(e) > 0.5]
    hard_hosts.sort(key=lambda x: -np.mean(x[1]))
    for host, errs in hard_hosts[:10]:
        errs = np.array(errs)
        print(f"  {host:20s}: n={len(errs):4d}  MAE={errs.mean():.3f}  max={errs.max():.3f}")

    print("\n=== Error by Dopant (top 15 by count) ===")
    dopants_sorted = sorted(dopant_errors.items(), key=lambda x: -len(x[1]))
    for dop, errs in dopants_sorted[:15]:
        errs = np.array(errs)
        print(f"  {dop:10s}: n={len(errs):4d}  MAE={errs.mean():.3f}  max={errs.max():.3f}")

    print("\n=== Error by Target Magnitude ===")
    for bn in ["<1 eV", "1-3 eV", "3-6 eV", ">6 eV"]:
        errs = target_bins.get(bn, [])
        if errs:
            errs = np.array(errs)
            print(f"  {bn:10s}: n={len(errs):4d}  MAE={errs.mean():.3f}  median={np.median(errs):.3f}  max={errs.max():.3f}")

    print("\n=== Worst 10 Predictions ===")
    errors = np.abs(preds - targets)
    worst_idx = np.argsort(-errors)[:10]
    for rank, wi in enumerate(worst_idx):
        tidx = test_idx[wi]
        md = data[tidx].get("metadata", {})
        host = md.get("host", "?")
        dopant = md.get("dopant", "?")
        print(f"  #{rank+1}: error={errors[wi]:.3f}  pred={preds[wi]:.3f}  true={targets[wi]:.3f}  "
              f"host={host} dopant={dopant}")

    print(f"\n=== Overall ===")
    print(f"  Total test: {len(preds)}")
    print(f"  MAE: {errors.mean():.4f}")
    print(f"  RMSE: {np.sqrt((errors**2).mean()):.4f}")
    print(f"  Median AE: {np.median(errors):.4f}")
    print(f"  % within 0.1 eV: {(errors < 0.1).mean()*100:.1f}%")
    print(f"  % within 0.3 eV: {(errors < 0.3).mean()*100:.1f}%")
    print(f"  % within 0.5 eV: {(errors < 0.5).mean()*100:.1f}%")
    print(f"  % within 1.0 eV: {(errors < 1.0).mean()*100:.1f}%")

if __name__ == "__main__":
    main()
