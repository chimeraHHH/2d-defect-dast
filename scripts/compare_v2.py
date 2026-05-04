"""Compare v2 architecture variants against the v1 baseline.

Reads ``results/<run>/metrics.json`` for a configurable list of runs and
prints a Markdown table sorted by test MAE.  Used to settle Phase 1 →
Phase 2 routing decisions.

Usage:
    python -m scripts.compare_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RUNS = [
    ("baseline_h128_aug_long_safe", "v1 baseline (h128, 50ep)"),
    ("v2_pfa_h128_aug_long_safe", "v2 full (PFA + multi-scale + defect)"),
    ("v2_pfa_only", "v2 PFA only"),
    ("v2_ablate_no_pfa", "v2 − PFA"),
    ("v2_ablate_no_long_range", "v2 − long-range (single RBF)"),
    ("v2_ablate_no_defect_bias", "v2 − defect bias"),
]


def read_run(name: str) -> dict | None:
    path = ROOT / "results" / name / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", help="Optional space-separated run names")
    args = parser.parse_args()
    runs = (
        [(r, r) for r in args.runs] if args.runs else DEFAULT_RUNS
    )

    table = []
    for run, label in runs:
        m = read_run(run)
        if m is None:
            table.append((label, run, None, None, None, None))
            continue
        epochs = m.get("history", [])
        last_lr = epochs[-1]["lr"] if epochs else None
        epochs_run = len(epochs)
        params = m.get("n_params", 0) / 1e6
        test_mae = m.get("test_mae")
        test_rmse = m.get("test_rmse")
        table.append((label, run, params, epochs_run, test_mae, test_rmse))

    table.sort(key=lambda r: (r[4] is None, r[4] if r[4] is not None else 1e9))
    print()
    print(f"| {'Variant':<46} | {'run':<32} | {'M':>5} | {'epochs':>6} | {'MAE':>6} | {'RMSE':>6} |")
    print(f"|{'-'*48}|{'-'*34}|{'-'*7}|{'-'*8}|{'-'*8}|{'-'*8}|")
    for label, run, params, epochs_run, mae, rmse in table:
        mae_s = f"{mae:.4f}" if mae is not None else "  -- "
        rmse_s = f"{rmse:.4f}" if rmse is not None else "  -- "
        params_s = f"{params:.3f}" if params is not None else "  -- "
        epochs_s = f"{epochs_run}" if epochs_run is not None else "  -- "
        print(f"| {label:<46} | {run:<32} | {params_s:>5} | {epochs_s:>6} | {mae_s:>6} | {rmse_s:>6} |")
    print()

    # Highlight delta vs baseline
    base = next((r for r in table if r[1] == "baseline_h128_aug_long_safe"), None)
    if base and base[4] is not None:
        print(f"Baseline reference: MAE {base[4]:.4f} eV ({base[2]:.3f} M params)\n")
        for label, run, params, epochs_run, mae, rmse in table:
            if run == "baseline_h128_aug_long_safe" or mae is None:
                continue
            delta_pct = (mae - base[4]) / base[4] * 100
            sign = "+" if delta_pct > 0 else ""
            print(f"  {label:<46}  Δ MAE {sign}{delta_pct:+.1f}%")


if __name__ == "__main__":
    main()
