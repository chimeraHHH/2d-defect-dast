"""Error decomposition by physical category, with figures and JSON.

Loads a run's ``test_predictions.npz`` and joins predictions to per-sample
metadata via the cleaned dataset (the canonical 1065 cleaned-test split is
identical between augmented_dataset_safe and cleaned_dataset for the same
seed=42 split).

Decomposition axes:
  1. Host material (e.g. MoS2, WS2, ...)
  2. Dopant element (e.g. Mo, Cl, B, ...)
  3. Defect type (interstitial / adsorption)
  4. Supercell size buckets (≤25, 26-50, 51-75, >75)
  5. Dopant-period bucket: main-group / 3d / 4d / 5d / 4f
  6. |target Ef| magnitude buckets

Outputs:
  - paper/figures/fig_error_by_category.png   (4-panel bar chart)
  - results/error_decomposition.json          (every category's MAE / std / n)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, make_splits  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Period of an element by atomic number
def _period(Z: int) -> str:
    if Z is None or Z <= 0:
        return "?"
    if Z <= 2:
        return "1"
    if Z <= 10:
        return "2"
    if Z <= 18:
        return "3"
    if Z <= 36:
        return "4"
    if Z <= 54:
        return "5"
    if Z <= 86:
        return "6"
    return "7"


def _block(Z: int) -> str:
    """Rough s/p/d/f block classification."""
    if Z is None or Z <= 0:
        return "?"
    if Z in {1, 2}: return "s"
    main_group = {3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18,
                  19, 20, 31, 32, 33, 34, 35, 36,
                  37, 38, 49, 50, 51, 52, 53, 54,
                  55, 56, 81, 82, 83, 84, 85, 86}
    if Z in main_group: return "main"
    if 21 <= Z <= 30: return "3d"
    if 39 <= Z <= 48: return "4d"
    if 57 <= Z <= 71: return "4f"
    if 72 <= Z <= 80: return "5d"
    if 89 <= Z <= 103: return "5f"
    return "?"


def _show(rows, top=5, max_label=20):
    rows = sorted(rows, key=lambda r: r["mae"])
    out = {"best": [], "worst": []}
    print(f"  best:")
    for r in rows[:top]:
        print(f"    {r['key'][:max_label]:<{max_label}} MAE {r['mae']:.3f} ± {r['std']:.3f}  (n={r['n']})")
        out["best"].append(r)
    print(f"  worst:")
    for r in rows[-top:][::-1]:
        print(f"    {r['key'][:max_label]:<{max_label}} MAE {r['mae']:.3f} ± {r['std']:.3f}  (n={r['n']})")
        out["worst"].append(r)
    return out


def _stats_per_key(d):
    rows = []
    for k, errs in d.items():
        rows.append({
            "key": str(k),
            "mae": float(np.mean(errs)),
            "std": float(np.std(errs)),
            "n": int(len(errs)),
        })
    return rows


def main(run_name: str = "baseline_h128_aug_long_safe"):
    npz = np.load(ROOT / f"results/{run_name}/test_predictions.npz")
    preds, targets = npz["preds"], npz["targets"]

    cfg_path = ROOT / "configs" / f"{run_name}.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    safe_path = ROOT / cfg["data_path"]
    cleaned_path = ROOT / "data/processed/cleaned_dataset.pkl"
    ds_path = safe_path if safe_path.exists() else cleaned_path
    ds = CrystalGraphDataset(ds_path)
    _, _, test_set = make_splits(
        ds,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 42),
    )
    test_ids = list(test_set.indices)

    err = preds - targets
    abs_err = np.abs(err)
    print(f"=== Error decomposition for {run_name} ===")
    print(f"  Test size: {len(targets)}")
    print(f"  Overall MAE: {abs_err.mean():.4f} eV  RMSE: {np.sqrt((err**2).mean()):.4f} eV")

    by_host = defaultdict(list)
    by_dopant = defaultdict(list)
    by_dtype = defaultdict(list)
    by_natoms = defaultdict(list)
    by_block = defaultdict(list)
    by_ef_mag = defaultdict(list)

    for i, idx in enumerate(test_ids):
        meta = ds.data[idx]["metadata"]
        host = meta.get("host", "?") or "?"
        dopant = meta.get("dopant", "?") or "?"
        dtype = meta.get("defecttype", "?") or "?"
        natoms = ds.data[idx]["numbers"].size

        by_host[host].append(abs_err[i])
        by_dopant[dopant].append(abs_err[i])
        by_dtype[dtype].append(abs_err[i])

        bucket = (
            "≤25" if natoms <= 25
            else "26-50" if natoms <= 50
            else "51-75" if natoms <= 75
            else ">75"
        )
        by_natoms[bucket].append(abs_err[i])

        # period block of the dopant
        try:
            from ase.data import atomic_numbers as _AZ
            zd = _AZ.get(dopant, None)
            blk = _block(zd) if zd is not None else "?"
        except Exception:
            blk = "?"
        by_block[blk].append(abs_err[i])

        ef = float(targets[i])
        ef_buc = (
            "|Ef|<1" if abs(ef) < 1
            else "1-3" if abs(ef) < 3
            else "3-6" if abs(ef) < 6
            else "≥6"
        )
        by_ef_mag[ef_buc].append(abs_err[i])

    summary = {
        "run": run_name,
        "test_size": int(len(targets)),
        "overall_mae": float(abs_err.mean()),
        "overall_rmse": float(np.sqrt((err**2).mean())),
    }

    print("\n-- by defect type --")
    summary["by_defect_type"] = _show(_stats_per_key(by_dtype), top=10)
    print("\n-- by supercell size --")
    rows = _stats_per_key(by_natoms)
    rows = sorted(rows, key=lambda r: ["≤25", "26-50", "51-75", ">75"].index(r["key"]))
    for r in rows:
        print(f"  {r['key']:<8} MAE {r['mae']:.3f} ± {r['std']:.3f}  (n={r['n']})")
    summary["by_supercell_size"] = rows
    print("\n-- by host material (top/bottom 5, n>=20) --")
    summary["by_host"] = _show(_stats_per_key(
        {k: v for k, v in by_host.items() if len(v) >= 20}), top=5)
    print("\n-- by dopant element (top/bottom 5, n>=15) --")
    summary["by_dopant"] = _show(_stats_per_key(
        {k: v for k, v in by_dopant.items() if len(v) >= 15}), top=5)
    print("\n-- by dopant period block --")
    rows = _stats_per_key(by_block)
    for r in sorted(rows, key=lambda r: r["mae"]):
        print(f"  {r['key']:<6} MAE {r['mae']:.3f} ± {r['std']:.3f}  (n={r['n']})")
    summary["by_dopant_block"] = rows
    print("\n-- by |Ef| magnitude --")
    rows = _stats_per_key(by_ef_mag)
    rows = sorted(rows, key=lambda r: ["|Ef|<1", "1-3", "3-6", "≥6"].index(r["key"]))
    for r in rows:
        print(f"  {r['key']:<8} MAE {r['mae']:.3f} ± {r['std']:.3f}  (n={r['n']})")
    summary["by_ef_magnitude"] = rows

    # ------------- figure -------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    rows = sorted(_stats_per_key(by_dtype), key=lambda r: r["mae"])
    keys = [r["key"] for r in rows]; maes = [r["mae"] for r in rows]; stds = [r["std"] for r in rows]
    ns = [r["n"] for r in rows]
    bars = ax.bar(range(len(keys)), maes, yerr=stds, capsize=4, color="tab:blue")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(keys, ns)])
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("By defect type")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    rows = sorted(_stats_per_key(by_natoms),
                  key=lambda r: ["≤25", "26-50", "51-75", ">75"].index(r["key"]))
    keys = [r["key"] for r in rows]; maes = [r["mae"] for r in rows]; stds = [r["std"] for r in rows]
    ns = [r["n"] for r in rows]
    ax.bar(range(len(keys)), maes, yerr=stds, capsize=4, color="tab:orange")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(keys, ns)])
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("By supercell size")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    rows = sorted(_stats_per_key(by_block), key=lambda r: r["mae"])
    keys = [r["key"] for r in rows]; maes = [r["mae"] for r in rows]; stds = [r["std"] for r in rows]
    ns = [r["n"] for r in rows]
    ax.bar(range(len(keys)), maes, yerr=stds, capsize=4, color="tab:green")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(keys, ns)])
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("By dopant period block")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    rows = sorted(_stats_per_key(by_ef_mag),
                  key=lambda r: ["|Ef|<1", "1-3", "3-6", "≥6"].index(r["key"]))
    keys = [r["key"] for r in rows]; maes = [r["mae"] for r in rows]; stds = [r["std"] for r in rows]
    ns = [r["n"] for r in rows]
    ax.bar(range(len(keys)), maes, yerr=stds, capsize=4, color="tab:red")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(keys, ns)])
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("By |Ef| magnitude")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Error decomposition for {run_name}\n"
                 f"Overall MAE = {summary['overall_mae']:.3f} eV   "
                 f"RMSE = {summary['overall_rmse']:.3f} eV")
    fig.tight_layout()
    out = FIG_DIR / "fig_error_by_category.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"\nfigure saved -> {out}")

    out_json = ROOT / "results" / "error_decomposition.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary saved -> {out_json}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline_h128_aug_long_safe"
    main(name)
