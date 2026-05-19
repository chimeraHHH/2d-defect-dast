#!/usr/bin/env python3
"""Detailed error analysis by host, dopant, and defect type.

Produces a comprehensive breakdown of model errors for paper discussion:
  - Per defect type (interstitial vs adsorbate)
  - Per host material (worst/best)
  - Per dopant element (worst/best)
  - Worst individual predictions
  - Error by target energy range
  - Bias analysis (systematic over/under-prediction)

Usage:
    python scripts/error_analysis_detailed.py results/v2_gated_pooling_s43
    python scripts/error_analysis_detailed.py results/v3_deftype results/v6_physics --compare
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, make_splits


def analyse(results_dir, min_epochs=20):
    """Load predictions and metadata for error analysis."""
    results_dir = Path(results_dir)
    npz = results_dir / "test_predictions.npz"
    metrics_path = results_dir / "metrics.json"

    # Load predictions
    if not npz.exists():
        print("  No test_predictions.npz found")
        return None

    # Check epoch count
    n_epochs = 0
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        n_epochs = len(m.get("history", []))

    data = np.load(str(npz))
    preds, targets = data["preds"], data["targets"]
    errors = np.abs(preds - targets)
    signed = preds - targets

    # Load dataset for metadata
    dataset = CrystalGraphDataset(ROOT / "data/processed/cleaned_dataset.pkl")
    _, _, test_set = make_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42)

    if len(preds) != len(test_set):
        print("  Prediction count mismatch: %d vs %d" % (len(preds), len(test_set)))
        return None

    # Collect metadata
    dtype_data = defaultdict(lambda: {"errors": [], "targets": [], "signed": []})
    host_data = defaultdict(lambda: {"errors": [], "targets": [], "signed": []})
    dopant_data = defaultdict(lambda: {"errors": [], "targets": [], "signed": []})
    samples = []

    for i, idx in enumerate(test_set.indices):
        meta = dataset.data[idx]["metadata"]
        dt = meta["defecttype"]
        host = meta["host"]
        dopant = meta["dopant"]

        dtype_data[dt]["errors"].append(errors[i])
        dtype_data[dt]["targets"].append(targets[i])
        dtype_data[dt]["signed"].append(signed[i])

        host_data[host]["errors"].append(errors[i])
        host_data[host]["targets"].append(targets[i])
        host_data[host]["signed"].append(signed[i])

        dopant_data[dopant]["errors"].append(errors[i])
        dopant_data[dopant]["targets"].append(targets[i])
        dopant_data[dopant]["signed"].append(signed[i])

        samples.append({
            "host": host, "defecttype": dt, "dopant": dopant,
            "target": targets[i], "pred": preds[i], "error": errors[i],
            "signed": signed[i],
        })

    # ── Report ────────────────────────────────────────────────────
    mae = errors.mean()
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS: %s  (MAE=%.4f, n=%d, ep=%s)"
          % (results_dir.name, mae, len(errors), n_epochs or "?"))
    print("=" * 80)

    # Per defect type
    print("\n--- Per Defect Type ---")
    print("  %-15s %8s %6s %8s %8s %8s" % ("Type", "MAE", "Count", "%Total", "MeanEf", "Bias"))
    for dt in sorted(dtype_data.keys()):
        d = dtype_data[dt]
        e = np.array(d["errors"])
        t = np.array(d["targets"])
        s = np.array(d["signed"])
        pct = len(e) / len(errors) * 100
        print("  %-15s %8.4f %6d %7.1f%% %8.3f %+8.3f" %
              (dt, e.mean(), len(e), pct, t.mean(), s.mean()))

    # Per host (worst 15)
    print("\n--- Worst 15 Hosts ---")
    print("  %-12s %8s %6s %8s %8s %8s" % ("Host", "MAE", "Count", "MeanEf", "Bias", "ErrFrac"))
    host_stats = []
    for h in host_data:
        d = host_data[h]
        e = np.array(d["errors"])
        host_stats.append((h, e.mean(), len(e), np.mean(d["targets"]),
                          np.mean(d["signed"]), e.sum() / errors.sum() * 100))
    host_stats.sort(key=lambda x: -x[1])
    for h, hmae, n, mt, bias, efrac in host_stats[:15]:
        print("  %-12s %8.4f %6d %8.2f %+8.3f %7.1f%%" %
              (h, hmae, n, mt, bias, efrac))

    # Per dopant (worst 10)
    print("\n--- Worst 10 Dopants ---")
    print("  %-6s %8s %6s %8s" % ("Dopant", "MAE", "Count", "Bias"))
    dopant_stats = []
    for dp in dopant_data:
        d = dopant_data[dp]
        e = np.array(d["errors"])
        dopant_stats.append((dp, e.mean(), len(e), np.mean(d["signed"])))
    dopant_stats.sort(key=lambda x: -x[1])
    for dp, dmae, n, bias in dopant_stats[:10]:
        print("  %-6s %8.4f %6d %+8.3f" % (dp, dmae, n, bias))

    # Worst individual predictions
    print("\n--- Worst 15 Individual Predictions ---")
    samples.sort(key=lambda x: -x["error"])
    for s in samples[:15]:
        print("  %-10s %-12s %-4s  Ef=%7.2f  pred=%7.2f  err=%6.2f" %
              (s["host"], s["defecttype"], s["dopant"],
               s["target"], s["pred"], s["error"]))

    # Per-range error distribution
    print("\n--- Per-Range Statistics ---")
    print("  %-8s %8s %8s %8s %8s %6s %7s" %
          ("Range", "MAE", "RMSE", "Bias", "PredStd", "Count", "%Err"))
    ranges = [(0, 2), (2, 5), (5, 7), (7, 25)]
    for lo, hi in ranges:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            r_err = errors[mask]
            r_signed = signed[mask]
            r_preds = preds[mask]
            efrac = r_err.sum() / errors.sum() * 100
            print("  [%d,%d)   %8.4f %8.4f %+8.4f %8.4f %6d %6.1f%%" %
                  (lo, hi, r_err.mean(),
                   np.sqrt(np.mean(r_signed ** 2)),
                   r_signed.mean(),
                   r_preds.std(),
                   mask.sum(), efrac))

    return {
        "name": results_dir.name,
        "mae": mae,
        "dtype_data": dtype_data,
        "host_stats": host_stats,
        "dopant_stats": dopant_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--min-epochs", type=int, default=20)
    args = parser.parse_args()

    results = []
    for d in args.dirs:
        r = analyse(d, min_epochs=args.min_epochs)
        if r:
            results.append(r)

    if args.compare and len(results) > 1:
        print("\n" + "=" * 80)
        print("COMPARISON: Per-host MAE improvement")
        print("=" * 80)
        base = results[0]
        for r in results[1:]:
            print("\n%s vs %s:" % (r["name"], base["name"]))
            base_hosts = {h[0]: h[1] for h in base["host_stats"]}
            r_hosts = {h[0]: h[1] for h in r["host_stats"]}
            common = set(base_hosts.keys()) & set(r_hosts.keys())
            diffs = [(h, r_hosts[h] - base_hosts[h], base_hosts[h], r_hosts[h])
                     for h in common]
            diffs.sort(key=lambda x: x[1])
            print("  Top 5 improved hosts:")
            for h, delta, b_mae, r_mae in diffs[:5]:
                print("    %-12s: %+.4f  (%.4f -> %.4f)" % (h, delta, b_mae, r_mae))
            print("  Top 5 degraded hosts:")
            for h, delta, b_mae, r_mae in diffs[-5:]:
                print("    %-12s: %+.4f  (%.4f -> %.4f)" % (h, delta, b_mae, r_mae))


if __name__ == "__main__":
    main()
