"""Aggregate every metric across the project into a single CSV + markdown.

Combines:
  - Per-run test_mae / test_rmse from each results/<run>/metrics.json
  - 4-seed mean / std for the long_safe / xlong_safe families
  - Deep-ensemble metrics (raw + τ-scaled) from results/uq_calibration*.json
  - Error-decomposition rows from results/error_decomposition.json
  - LOHO per-host rows from results/loho_summary.json (when available)

Output:
  - results/all_metrics.csv
  - results/all_metrics.md
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _read_metrics(name: str):
    p = RESULTS / name / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _read_json(p: Path):
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    rows = []  # list of dicts

    # ---- per-run baseline metrics ----
    runs = sorted(d.name for d in RESULTS.iterdir() if d.is_dir() and (RESULTS / d / "metrics.json").exists())
    for run in runs:
        m = _read_metrics(run)
        cfg = m["config"]
        rows.append({
            "category": "per-run",
            "run": run,
            "model": cfg.get("model", "?"),
            "data": Path(cfg.get("data_path", "")).name,
            "params_M": round(m["n_params"] / 1e6, 3),
            "epochs": cfg.get("epochs"),
            "seed": cfg.get("seed"),
            "test_mae": round(m["test_mae"], 4),
            "test_rmse": round(m["test_rmse"], 4),
        })

    # ---- multi-seed aggregates ----
    families = defaultdict(list)
    for r in rows:
        if r["run"].startswith("baseline_h128_aug_long_safe"):
            families["baseline_h128_aug_long_safe"].append(r)
        if r["run"].startswith("baseline_h128_aug_xlong_safe"):
            families["baseline_h128_aug_xlong_safe"].append(r)
    for fam, items in families.items():
        if len(items) >= 2:
            import numpy as np
            maes = np.array([i["test_mae"] for i in items])
            rmses = np.array([i["test_rmse"] for i in items])
            rows.append({
                "category": "multi-seed mean ± std",
                "run": f"{fam}  ({len(items)} seeds)",
                "model": items[0]["model"],
                "data": items[0]["data"],
                "params_M": items[0]["params_M"],
                "epochs": items[0]["epochs"],
                "seed": "mean",
                "test_mae": f"{maes.mean():.4f} ± {maes.std():.4f}",
                "test_rmse": f"{rmses.mean():.4f} ± {rmses.std():.4f}",
            })

    # ---- deep ensemble UQ ----
    for fname, label in [("uq_calibration.json", "deep ensemble (4 long seeds)"),
                          ("uq_calibration_xlong.json", "deep ensemble (4 long + 2 xlong)")]:
        u = _read_json(RESULTS / fname)
        if u is None:
            continue
        raw = u["raw"]
        rows.append({
            "category": "ensemble (raw)",
            "run": label,
            "model": "ensemble",
            "data": "leak_free_v1",
            "params_M": "n×0.747",
            "epochs": "—",
            "seed": "—",
            "test_mae": round(raw["mae"], 4),
            "test_rmse": round(raw["rmse"], 4),
        })
        ev = u.get("tau_eval_metrics")
        if ev is not None:
            rows.append({
                "category": f"ensemble (τ={u['tau']:.2f}, eval-half)",
                "run": label,
                "model": "ensemble + τ",
                "data": "leak_free_v1",
                "params_M": "n×0.747",
                "epochs": "—",
                "seed": "—",
                "test_mae": round(ev["mae"], 4),
                "test_rmse": round(ev["rmse"], 4),
            })

    # ---- LOHO summary ----
    loho_summary = _read_json(RESULTS / "loho_summary.json")
    if loho_summary and loho_summary.get("hosts"):
        for h in loho_summary["hosts"]:
            rows.append({
                "category": "LOHO",
                "run": f"loho_{h['host']}",
                "model": "baseline h128",
                "data": f"loho_{h['host']}.pkl",
                "params_M": 0.747,
                "epochs": 50,
                "seed": 42,
                "test_mae": round(h["loho_mae"], 4),
                "test_rmse": round(h["loho_rmse"], 4),
            })
    else:
        # If summary not yet built, scan loho_* directly
        for d in sorted(RESULTS.glob("loho_*")):
            mp = d / "metrics.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            rows.append({
                "category": "LOHO (raw)",
                "run": d.name,
                "model": "baseline h128",
                "data": f"{d.name}.pkl",
                "params_M": round(m["n_params"] / 1e6, 3),
                "epochs": m["config"].get("epochs"),
                "seed": m["config"].get("seed"),
                "test_mae": round(m["test_mae"], 4),
                "test_rmse": round(m["test_rmse"], 4),
            })

    # ---- MC-Dropout ----
    mc = _read_json(RESULTS / "mc_dropout_vs_ensemble.json")
    if mc:
        for entry in mc.get("methods", []):
            raw = entry["raw"]
            rows.append({
                "category": "UQ method comparison",
                "run": entry["method"],
                "model": "various",
                "data": "leak_free_v1",
                "params_M": "—",
                "epochs": "—",
                "seed": "—",
                "test_mae": round(raw["mae"], 4),
                "test_rmse": round(raw["rmse"], 4),
            })

    # ---- Feature importance summary row ----
    fi = _read_json(RESULTS / "feature_importance.json")
    if fi:
        # add a "ablation feature x" row per feature
        for r in fi["rows"]:
            rows.append({
                "category": "Feature ablation (permutation)",
                "run": f"perm-{r['feature_name']}",
                "model": "baseline h128",
                "data": "leak_free_v1",
                "params_M": 0.747,
                "epochs": "—",
                "seed": "—",
                "test_mae": round(fi["baseline_mae"] + r["mean_delta_mae"], 4),
                "test_rmse": "—",
            })

    # ---- write CSV ----
    csv_path = RESULTS / "all_metrics.csv"
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # ---- write markdown ----
    md_path = RESULTS / "all_metrics.md"
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    md = ["# All metrics — automated aggregation", ""]
    for cat, items in by_cat.items():
        md.append(f"## {cat}")
        md.append("")
        md.append("| run | model | data | params (M) | epochs | seed | Test MAE | Test RMSE |")
        md.append("|---|---|---|---|---|---|---|---|")
        items_sorted = sorted(items, key=lambda r: str(r["test_mae"]))
        for r in items_sorted:
            md.append(f"| {r['run']} | {r['model']} | {r['data']} | {r['params_M']} | {r['epochs']} | {r['seed']} | {r['test_mae']} | {r['test_rmse']} |")
        md.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
