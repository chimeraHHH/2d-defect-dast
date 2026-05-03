"""Permutation-importance for the 9 element-wise input features.

For each of the 9 feature dimensions of the lookup table
(group, period, electronegativity, covalent r, vdW r, valence e-, IE, EA, mass),
we shuffle that single column **across elements** (Z=1..100), then re-predict
the entire 1065 cleaned-test set. The MAE increase relative to the unshuffled
baseline tells us how strongly the model relies on each feature.

Why this is meaningful: each per-atom feature vector x_i is a row of a
Z-indexed lookup table; shuffling one column maps every atom of element Z
to a different (but valid) value of that descriptor, breaking that feature's
information while leaving the rest intact. We repeat the shuffle 10 times to
average over noise.

Output:
  - results/feature_importance.json
  - paper/figures/fig_feature_importance.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "group",
    "period",
    "electronegativity",
    "covalent_radius",
    "vdW_radius",
    "valence_electrons",
    "ionisation_energy",
    "electron_affinity",
    "atomic_mass",
]


def evaluate_set(model, dataset_subset, perm_table, normalizer_mean, normalizer_std,
                 batch_size=64):
    """Evaluate over a Subset, but replace each sample's x by perm_table[Z]."""
    from torch.utils.data import DataLoader

    # we need to swap the dataset's atom_features lookup temporarily
    ds = dataset_subset.dataset
    orig_table = ds.atom_features
    ds.atom_features = perm_table
    try:
        loader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        preds_all, tgts_all = [], []
        for batch in loader:
            with torch.no_grad():
                p = model(batch) * normalizer_std + normalizer_mean
            preds_all.append(p.cpu().numpy())
            tgts_all.append(batch["target"].cpu().numpy())
        preds = np.concatenate(preds_all)
        tgts = np.concatenate(tgts_all)
        return float(np.abs(preds - tgts).mean()), float(np.sqrt(((preds - tgts) ** 2).mean()))
    finally:
        ds.atom_features = orig_table


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/baseline_h128_aug_long_safe.yaml"))
    cleaned_path = ROOT / "data/processed/cleaned_dataset.pkl"
    safe_path = ROOT / cfg["data_path"]
    ds = CrystalGraphDataset(safe_path if safe_path.exists() else cleaned_path)
    _, _, test_set = make_splits(
        ds, cfg.get("train_ratio", 0.8), cfg.get("val_ratio", 0.1), cfg.get("seed", 42),
    )

    model = CrystalTransformer(**cfg["model_kwargs"])
    ckpt = torch.load(ROOT / "results/baseline_h128_aug_long_safe/best.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    nmean, nstd = ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"]

    import sys as _sys, time as _time
    base_table = ds.atom_features.clone()  # (Z+1, 9)
    t0 = _time.time()
    base_mae, base_rmse = evaluate_set(model, test_set, base_table, nmean, nstd)
    print(f"Baseline MAE = {base_mae:.4f}  (one eval took {_time.time()-t0:.1f}s)", flush=True)

    # for each feature dimension, shuffle column across elements, average over reps
    n_reps = 5
    rng = np.random.default_rng(0)
    rows = []
    for d in range(9):
        deltas = []
        td0 = _time.time()
        for r in range(n_reps):
            t = base_table.clone()
            # we shuffle the values in column d across rows 1..100 (Z=0 is padding)
            perm = rng.permutation(100) + 1
            t[1:101, d] = base_table[perm, d]
            mae, _ = evaluate_set(model, test_set, t, nmean, nstd)
            deltas.append(mae - base_mae)
        deltas = np.array(deltas)
        rows.append({
            "feature_idx": d,
            "feature_name": FEATURE_NAMES[d],
            "mean_delta_mae": float(deltas.mean()),
            "std_delta_mae": float(deltas.std()),
            "n_reps": n_reps,
        })
        print(f"  d={d:2d} {FEATURE_NAMES[d]:<22} ΔMAE = {deltas.mean():+.4f} ± {deltas.std():.4f}  "
              f"({_time.time()-td0:.1f}s)", flush=True)

    # figure
    rows_sorted = sorted(rows, key=lambda r: r["mean_delta_mae"], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["feature_name"] for r in rows_sorted]
    means = [r["mean_delta_mae"] for r in rows_sorted]
    stds = [r["std_delta_mae"] for r in rows_sorted]
    ax.barh(range(len(names)), means, xerr=stds, capsize=4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("ΔMAE (eV) when feature is permuted")
    ax.set_title(f"Permutation feature importance\nbaseline_h128_aug_long_safe (base MAE={base_mae:.3f})")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIG_DIR / "fig_feature_importance.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"saved {out}")

    summary = {
        "baseline_mae": base_mae,
        "baseline_rmse": base_rmse,
        "n_reps": n_reps,
        "rows": rows_sorted,
    }
    with open(ROOT / "results/feature_importance.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/feature_importance.json")


if __name__ == "__main__":
    main()
