"""Collect and summarize OOD experiment results.

Usage:
  python scripts/ood_collect_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "ood"


def collect_p0():
    """Collect leave-one-G6-host-out results."""
    g6_hosts = ['MoS2', 'MoSe2', 'MoTe2', 'WS2', 'WSe2', 'WTe2', 'MoSSe']
    results = {}

    for host in g6_hosts:
        metrics_path = RESULTS / f"loho_{host}_s42" / "metrics.json"
        if not metrics_path.exists():
            print(f"  [P0] {host}: NOT FOUND")
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        results[host] = m
        print(f"  [P0] {host}: OOD MAE = {m['test_mae']:.4f} eV "
              f"(naive = {m['naive_baseline_mae']:.4f}, "
              f"val = {m['best_val_mae']:.4f})")

    if results:
        maes = [r["test_mae"] for r in results.values()]
        naive = [r["naive_baseline_mae"] for r in results.values()]
        print(f"\n  P0 Summary (n={len(results)} folds):")
        print(f"    Mean OOD MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f} eV")
        print(f"    Mean naive:   {np.mean(naive):.4f} ± {np.std(naive):.4f} eV")
        print(f"    Improvement over naive: {(1 - np.mean(maes)/np.mean(naive))*100:.1f}%")
        print(f"    Range: [{min(maes):.4f}, {max(maes):.4f}]")

        # Best/worst hosts
        sorted_hosts = sorted(results.items(), key=lambda x: x[1]["test_mae"])
        print(f"    Easiest: {sorted_hosts[0][0]} ({sorted_hosts[0][1]['test_mae']:.4f})")
        print(f"    Hardest: {sorted_hosts[-1][0]} ({sorted_hosts[-1][1]['test_mae']:.4f})")

    return results


def collect_p1():
    """Collect block-out results."""
    metrics_path = RESULTS / "block_g6x3d_s42" / "metrics.json"
    if not metrics_path.exists():
        print(f"  [P1] block_g6x3d: NOT FOUND")
        return None
    with open(metrics_path) as f:
        m = json.load(f)
    print(f"  [P1] G6×3d block: OOD MAE = {m['test_mae']:.4f} eV "
          f"(naive = {m['naive_baseline_mae']:.4f})")

    # Per-host breakdown
    if "per_host_mae" in m:
        print(f"    Per-host breakdown:")
        for host, mae in sorted(m["per_host_mae"].items(), key=lambda x: x[1]):
            print(f"      {host:<12}: {mae:.4f}")

    return m


def make_summary_table(p0_results, p1_result):
    """Generate the graduated evaluation table."""
    print(f"\n{'='*65}")
    print("GRADUATED GENERALIZATION ASSESSMENT")
    print(f"{'='*65}")

    rows = [
        ("ID (random split)", 0.362, "Same distribution (28-model ensemble)"),
    ]

    if p0_results:
        maes = [r["test_mae"] for r in p0_results.values()]
        rows.append((
            f"Intra-family OOD (P0, {len(p0_results)}-fold)",
            np.mean(maes),
            f"Leave-one-G6-host-out [±{np.std(maes):.3f}]"
        ))

    if p1_result:
        rows.append((
            "Compositional OOD (P1)",
            p1_result["test_mae"],
            "G6×3d block-out (matrix completion)"
        ))

    rows.append(("Full OOD (prospective DFT)", 2.66, "Novel materials (v3)"))

    print(f"\n{'Scenario':<35} {'MAE (eV)':<12} {'Description'}")
    print(f"{'-'*35} {'-'*12} {'-'*40}")
    for name, mae, desc in rows:
        print(f"{name:<35} {mae:<12.3f} {desc}")

    # Comparison with ALIGNN
    print(f"\n  vs ALIGNN (0.540 eV):")
    for name, mae, _ in rows:
        ratio = mae / 0.540
        print(f"    {name:<35}: {ratio:.2f}× ALIGNN")

    # Save structured summary
    summary = {
        "graduated_evaluation": [
            {"tier": i, "scenario": name, "mae_eV": float(mae), "description": desc}
            for i, (name, mae, desc) in enumerate(rows)
        ],
        "p0_results": {h: {"test_mae": r["test_mae"], "val_mae": r["best_val_mae"],
                           "naive_mae": r["naive_baseline_mae"]}
                       for h, r in (p0_results or {}).items()},
        "p1_result": {"test_mae": p1_result["test_mae"],
                      "per_host_mae": p1_result.get("per_host_mae", {})}
                     if p1_result else None,
    }

    out_path = RESULTS / "ood_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary -> {out_path}")
    return summary


def main():
    print("Collecting OOD experiment results...\n")

    print("=== P0: Leave-One-G6-Host-Out ===")
    p0 = collect_p0()

    print("\n=== P1: G6×3d Block-Out ===")
    p1 = collect_p1()

    if p0 or p1:
        make_summary_table(p0, p1)


if __name__ == "__main__":
    main()
