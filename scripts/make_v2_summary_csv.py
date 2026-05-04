"""Single-source-of-truth CSV summary of v2 results so far.

Writes results/v2_summary.csv with one row per run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SINGLE_RUNS = [
    ("baseline_h128_aug_long_safe", "v1 baseline (leak-free aug, seed=42)"),
    ("v2_pfa_h128_aug_long_safe", "v2 full (PFA + LR + DB)"),
    ("v2_pfa_only", "v2 PFA only"),
    ("v2_ablate_no_long_range", "v2 - long-range"),
    ("v2_ablate_no_pfa", "v2 - PFA"),
    ("v2_ablate_no_defect_bias", "v2 - defect bias"),
]

MULTI_RUNS = [
    ("multi_source_train_v2.json", 42, "v2 multi-source seed=42 (headline)"),
    ("multi_source_train_v2_seed0.json", 0, "v2 multi-source seed=0"),
    ("multi_source_train_v2_seed1.json", 1, "v2 multi-source seed=1"),
    ("multi_source_train_v2_seed2.json", 2, "v2 multi-source seed=2"),
]


def main() -> None:
    rows = []

    for run, label in SINGLE_RUNS:
        f = RESULTS / run / "metrics.json"
        if not f.exists():
            continue
        m = json.load(open(f))
        cfg = m.get("config", {})
        rows.append({
            "category": "single-source",
            "run": run,
            "label": label,
            "params_M": round(m["n_params"] / 1e6, 4),
            "epochs": cfg.get("epochs"),
            "seed": cfg.get("seed"),
            "test_mae_eV": round(m["test_mae"], 4),
            "test_rmse_eV": round(m["test_rmse"], 4),
            "best_val_mae_eV": round(m["best_val_mae"], 4),
            "test_set": "leak_free_v1 ordered (1065 samples)",
        })

    # multi-source seeds
    multi_maes = []
    for run, seed, label in MULTI_RUNS:
        f = RESULTS / run
        if not f.exists():
            continue
        m = json.load(open(f))
        rows.append({
            "category": "multi-source",
            "run": run.replace(".json", ""),
            "label": label,
            "params_M": round(m["n_params"] / 1e6, 4),
            "epochs": len(m.get("history", [])),
            "seed": seed,
            "test_mae_eV": round(m["test_mae_imp2d_eV"], 4),
            "test_rmse_eV": round(m["test_rmse_imp2d_eV"], 4),
            "best_val_mae_eV": round(m["best_val_mae_imp2d_eV"], 4),
            "test_set": "split_indices(seed=N) (1064 samples)",
        })
        multi_maes.append(m["test_mae_imp2d_eV"])

    if multi_maes:
        import numpy as np
        mean = float(np.mean(multi_maes))
        std = float(np.std(multi_maes, ddof=1))
        rows.append({
            "category": "multi-source",
            "run": "v2_multi_source_4seed_summary",
            "label": "v2 multi-source 4-seed mean +/- std",
            "params_M": "0.8199",
            "epochs": "60",
            "seed": "{42,0,1,2}",
            "test_mae_eV": f"{mean:.4f} +/- {std:.4f}",
            "test_rmse_eV": "-",
            "best_val_mae_eV": "-",
            "test_set": "varies per seed (split-dependent variance)",
        })

    # External references (not from metrics.json)
    rows.append({
        "category": "literature",
        "run": "ALIGNN_team_repro",
        "label": "ALIGNN (team reproduction, leak-free 1065)",
        "params_M": "4.030",
        "epochs": "-",
        "seed": "-",
        "test_mae_eV": "0.5400",
        "test_rmse_eV": "1.1670",
        "best_val_mae_eV": "-",
        "test_set": "leak_free_v1 ordered (1065 samples)",
    })
    rows.append({
        "category": "v1 multi-source",
        "run": "multi_source_train_v1",
        "label": "v1 multi-source (CrystalTransformer + 4 DBs, seed=42)",
        "params_M": "0.815",
        "epochs": "60",
        "seed": "42",
        "test_mae_eV": "0.5546",
        "test_rmse_eV": "-",
        "best_val_mae_eV": "0.5443",
        "test_set": "split_indices(seed=42) (1064 samples)",
    })

    out = RESULTS / "v2_summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    # also print a markdown-style preview
    print()
    print(" | ".join(rows[0].keys()))
    for r in rows:
        print(" | ".join(str(v) for v in r.values()))


if __name__ == "__main__":
    main()
