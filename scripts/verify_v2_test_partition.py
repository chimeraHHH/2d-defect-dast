"""Verify that the multi-source seed=42 IMP2D test fold is identical to
the canonical leak-free 1065-sample test fold used by single-source
baselines.

Motivation (paper Limitation 1): single-source baselines are evaluated on
the ordered leak-free 1065-sample fold, while multi-source runs partition
IMP2D with split_indices(N, 0.8, 0.1, seed). If the seed=42 partition is
the *same* 1065 samples, then the v2 seed=42 result (0.4929 eV) is already
an apples-to-apples comparison with the v1 single-source leak-free result
(0.516 eV) on an identical test partition -- resolving the core of
Limitation 1 without any retraining.

Method: compare the `targets` arrays saved in test_predictions.npz by
(a) the canonical single-source baseline run(s) and (b) multi-source runs
that used split_indices(seed=42). Element-wise, order-sensitive equality
of the 1065 DFT labels is an (astronomically strong) fingerprint of the
partition identity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "weights&results" / "project" / "results"

CANONICAL = "baseline_h128_aug_long_safe"       # v1 leak-free baseline (seed42)
CANDIDATES = [
    "baseline_h128_aug_long_safe_seed0",         # different *training* seed,
    "baseline_h128_aug_long_safe_seed1",         # same ordered test fold
    "baseline_h128_aug_long_safe_seed2",
    "multi_source_train_v2_aug",                 # v2 multi-source + aug (seed42)
    "multi_source_v4_s42",                       # v4 multi-source (seed42)
    "multi_source_v4_s43",                       # v4 multi-source (seed43) - control
    "multi_source_v4_deep_s42",
]


def load_targets(run: str):
    p = RES / run / "test_predictions.npz"
    if not p.exists():
        return None, None
    z = np.load(p)
    return z["targets"], z["preds"]


def main():
    t_ref, _ = load_targets(CANONICAL)
    assert t_ref is not None, f"missing canonical run {CANONICAL}"
    print(f"canonical fold: {CANONICAL}  n={len(t_ref)}")

    report = {"canonical": CANONICAL, "n_canonical": int(len(t_ref)),
              "comparisons": []}
    for run in CANDIDATES:
        t, p = load_targets(run)
        if t is None:
            print(f"  {run:<38} MISSING npz")
            report["comparisons"].append({"run": run, "status": "missing"})
            continue
        same_n = len(t) == len(t_ref)
        exact = bool(same_n and np.allclose(t, t_ref, atol=1e-6))
        same_set = bool(same_n and np.allclose(np.sort(t), np.sort(t_ref),
                                               atol=1e-6))
        mae = float(np.abs(p - t).mean())
        print(f"  {run:<38} n={len(t):>4}  exact-order={exact!s:<5} "
              f"same-set={same_set!s:<5} MAE={mae:.4f}")
        report["comparisons"].append({
            "run": run, "n": int(len(t)), "identical_order": exact,
            "identical_set": same_set, "mae_eV": round(mae, 4),
        })

    out = ROOT / "results" / "v2_test_partition_verification.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
