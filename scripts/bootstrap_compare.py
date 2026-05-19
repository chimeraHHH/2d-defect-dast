#!/usr/bin/env python3
"""Bootstrap significance testing for model comparison.

Computes bootstrap confidence intervals for MAE differences between models.
Essential for paper claims: "V3 significantly outperforms V2" requires p-values.

Methods:
  1. Paired bootstrap: resample (pred_A[i] - pred_B[i]) to test if ΔMAE != 0
  2. Per-range bootstrap: test significance within each energy range
  3. Effect size: Cohen's d for practical significance

Usage:
    python scripts/bootstrap_compare.py results/v2_gated_pooling_s43 results/v3_deftype
    python scripts/bootstrap_compare.py results/ --all  # compare all pairs
"""
import argparse
import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RANGES = [(0, 2, "[0,2)"), (2, 5, "[2,5)"), (5, 7, "[5,7)"), (7, 25, "[7,25)")]


def load_predictions(path):
    """Load test predictions from npz file."""
    path = Path(path)
    npz = path / "test_predictions.npz" if path.is_dir() else path
    if not npz.exists():
        return None, None
    data = np.load(str(npz))
    return data["preds"], data["targets"]


def paired_bootstrap(errors_a, errors_b, n_boot=10000, seed=42):
    """Paired bootstrap test for H0: MAE_A = MAE_B.

    Returns:
        delta: MAE_A - MAE_B (negative means A is better)
        ci_lo, ci_hi: 95% confidence interval for delta
        p_value: two-sided p-value
    """
    rng = np.random.RandomState(seed)
    n = len(errors_a)
    delta_obs = errors_a.mean() - errors_b.mean()

    # Bootstrap distribution of ΔMAE
    deltas = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        deltas[b] = errors_a[idx].mean() - errors_b[idx].mean()

    ci_lo = np.percentile(deltas, 2.5)
    ci_hi = np.percentile(deltas, 97.5)

    # Two-sided p-value: fraction of bootstrap samples crossing zero
    if delta_obs > 0:
        p_val = 2 * np.mean(deltas <= 0)
    else:
        p_val = 2 * np.mean(deltas >= 0)
    p_val = min(p_val, 1.0)

    return {
        "delta": float(delta_obs),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(p_val),
        "significant_005": p_val < 0.05,
        "significant_001": p_val < 0.01,
    }


def cohens_d(errors_a, errors_b):
    """Cohen's d effect size for paired differences."""
    diff = errors_a - errors_b
    d = diff.mean() / max(diff.std(), 1e-8)
    # Interpretation: 0.2=small, 0.5=medium, 0.8=large
    if abs(d) < 0.2:
        interp = "negligible"
    elif abs(d) < 0.5:
        interp = "small"
    elif abs(d) < 0.8:
        interp = "medium"
    else:
        interp = "large"
    return float(d), interp


def compare_pair(name_a, preds_a, name_b, preds_b, targets, n_boot=10000):
    """Full comparison between two models."""
    errors_a = np.abs(preds_a - targets)
    errors_b = np.abs(preds_b - targets)

    result = {
        "model_a": name_a,
        "model_b": name_b,
        "mae_a": float(errors_a.mean()),
        "mae_b": float(errors_b.mean()),
    }

    # Overall bootstrap
    boot = paired_bootstrap(errors_a, errors_b, n_boot=n_boot)
    result["overall"] = boot

    d, interp = cohens_d(errors_a, errors_b)
    result["overall"]["cohens_d"] = d
    result["overall"]["effect_size"] = interp

    # Per-range bootstrap
    result["per_range"] = {}
    for lo, hi, label in RANGES:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() >= 10:
            r_boot = paired_bootstrap(errors_a[mask], errors_b[mask], n_boot=n_boot)
            r_d, r_interp = cohens_d(errors_a[mask], errors_b[mask])
            r_boot["cohens_d"] = r_d
            r_boot["effect_size"] = r_interp
            r_boot["n_samples"] = int(mask.sum())
            result["per_range"][label] = r_boot

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", help="Model result directories to compare")
    parser.add_argument("--all", action="store_true",
                        help="Compare all pairs in the results directory")
    parser.add_argument("--baseline", default="v2_gated_pooling_s43",
                        help="Baseline model for --all comparisons")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Number of bootstrap samples")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.all:
        # Compare all models against baseline
        results_dir = Path(args.dirs[0]) if args.dirs else Path("results/")
        baseline_preds, baseline_targets = load_predictions(results_dir / args.baseline)
        if baseline_preds is None:
            print("Baseline %s not found!" % args.baseline)
            sys.exit(1)

        print("Baseline: %s (MAE=%.4f)" %
              (args.baseline, np.abs(baseline_preds - baseline_targets).mean()))
        print("Bootstrap samples: %d\n" % args.n_boot)

        comparisons = []
        for d in sorted(results_dir.iterdir()):
            if not d.is_dir() or d.name == args.baseline:
                continue
            preds, targets = load_predictions(d)
            if preds is None or len(preds) != len(baseline_preds):
                continue

            comp = compare_pair(
                args.baseline, baseline_preds,
                d.name, preds, baseline_targets,
                n_boot=args.n_boot,
            )
            comparisons.append(comp)

            sig = "***" if comp["overall"]["significant_001"] else (
                "**" if comp["overall"]["significant_005"] else "ns"
            )
            print("%-30s MAE=%.4f  ΔMAE=%+.4f  95%%CI=[%+.4f, %+.4f]  p=%.4f %s  d=%.2f(%s)" % (
                d.name,
                comp["mae_b"],
                comp["overall"]["delta"],
                comp["overall"]["ci_lo"],
                comp["overall"]["ci_hi"],
                comp["overall"]["p_value"],
                sig,
                comp["overall"]["cohens_d"],
                comp["overall"]["effect_size"],
            ))

        if args.out:
            with open(args.out, "w") as f:
                json.dump(comparisons, f, indent=2)
            print("\nSaved to %s" % args.out)

    elif len(args.dirs) >= 2:
        # Compare specific pairs
        preds_a, targets_a = load_predictions(args.dirs[0])
        preds_b, targets_b = load_predictions(args.dirs[1])
        if preds_a is None or preds_b is None:
            print("Could not load predictions!")
            sys.exit(1)
        if len(preds_a) != len(preds_b):
            print("Prediction count mismatch!")
            sys.exit(1)

        name_a = Path(args.dirs[0]).name
        name_b = Path(args.dirs[1]).name

        comp = compare_pair(name_a, preds_a, name_b, preds_b, targets_a,
                            n_boot=args.n_boot)

        print("=" * 70)
        print("BOOTSTRAP COMPARISON: %s vs %s" % (name_a, name_b))
        print("=" * 70)
        print("  %s MAE: %.4f" % (name_a, comp["mae_a"]))
        print("  %s MAE: %.4f" % (name_b, comp["mae_b"]))
        print("")
        print("  ΔMAE (A−B): %+.4f" % comp["overall"]["delta"])
        print("  95%% CI:     [%+.4f, %+.4f]" %
              (comp["overall"]["ci_lo"], comp["overall"]["ci_hi"]))
        print("  p-value:    %.4f  %s" % (
            comp["overall"]["p_value"],
            "***" if comp["overall"]["significant_001"] else (
                "**" if comp["overall"]["significant_005"] else "not significant"
            ),
        ))
        print("  Cohen's d:  %.3f (%s)" %
              (comp["overall"]["cohens_d"], comp["overall"]["effect_size"]))

        print("\nPer-range:")
        for label, r in comp["per_range"].items():
            sig = "***" if r["significant_001"] else (
                "**" if r["significant_005"] else "ns"
            )
            print("  %s: ΔMAE=%+.4f  CI=[%+.4f,%+.4f]  p=%.3f %s  n=%d" % (
                label, r["delta"], r["ci_lo"], r["ci_hi"],
                r["p_value"], sig, r["n_samples"],
            ))

        if args.out:
            with open(args.out, "w") as f:
                json.dump(comp, f, indent=2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
