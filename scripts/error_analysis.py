"""Error analysis: break down test errors by host material, dopant, defect type.

Loads the best run's predictions and joins them back to the cleaned dataset
to find which physical categories drive prediction error.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, make_splits  # noqa: E402


def main(run_name: str = "baseline_h128_aug_long") -> None:
    npz = np.load(ROOT / f"results/{run_name}/test_predictions.npz")
    preds, targets = npz["preds"], npz["targets"]

    # Re-derive the test split index list to recover sample metadata.
    # The aug_long config trained on augmented dataset, but the test split
    # there is also drawn from the augmented set; for error analysis we still
    # join via formula → host/dopant/defect type via the original cleaned set
    # so categories are stable.
    cfg_path = ROOT / "configs" / f"{run_name}.yaml"
    import yaml

    cfg = yaml.safe_load(open(cfg_path))
    ds_path = ROOT / cfg["data_path"]
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

    print(f"=== Error analysis for {run_name} ===")
    print(f"Test size: {len(targets)}")
    print(f"Overall MAE: {abs_err.mean():.4f} eV, RMSE: {np.sqrt((err**2).mean()):.4f} eV")
    print()

    by_host: dict = defaultdict(list)
    by_dopant: dict = defaultdict(list)
    by_dtype: dict = defaultdict(list)
    by_natoms: dict = defaultdict(list)

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

    def _show(title: str, d: dict, top: int = 10):
        print(f"-- {title} --")
        rows = [(k, np.mean(v), np.std(v), len(v)) for k, v in d.items()]
        # ascending by mean
        rows.sort(key=lambda r: r[1])
        for k, mean, std, n in rows[:top]:
            print(f"  best  {k:<14} MAE {mean:.3f} ± {std:.3f}  (n={n})")
        for k, mean, std, n in rows[-top:][::-1]:
            print(f"  worst {k:<14} MAE {mean:.3f} ± {std:.3f}  (n={n})")
        print()

    _show("By defect type", by_dtype)
    _show("By supercell size bucket", by_natoms)
    _show("By host material (top/bottom 5 with n≥30)",
          {k: v for k, v in by_host.items() if len(v) >= 30}, top=5)
    _show("By dopant element (top/bottom 5 with n≥30)",
          {k: v for k, v in by_dopant.items() if len(v) >= 30}, top=5)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline_h128_aug_long"
    main(name)
