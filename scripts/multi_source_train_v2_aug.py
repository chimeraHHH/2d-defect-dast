"""Multi-source training v2 + leak-free 3x aug on every source.

Builds on `multi_source_train_v2.py` with three key changes:
  1. Each source loads from its `aug_<source>_safe.pkl` produced by
     `build_full_aug_multi.py`, with explicit ordered (train | val | test)
     boundaries from meta — leak-free.
  2. The IMP2D test set is the SAME 1065 samples as in
     `cleaned_dataset.pkl[split_indices(seed=42)[2]]` so headline
     numbers are apples-to-apples with previous multi-source results.
  3. The DFT-3D source weight is REDUCED from 0.3 to 0.1, since the
     LOHO ID/OOD trade-off in v2_loho_multi suggested DFT-3D was
     pulling the shared backbone toward 3D bulk-pristine chemistry.
     The intent here is to keep DFT-3D as a large auxiliary corpus
     for backbone regularisation without letting it dominate the
     gradient direction.

Output: results/multi_source_train_v2_aug.json + best.pt + test_predictions.npz
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import collate_fn  # noqa: E402
import scripts.multi_source_train_v2 as base  # noqa: E402

AUG_PATHS = {
    "IMP2D": ROOT / "data" / "processed" / "aug_imp2d_safe.pkl",
    "JARVIS-2D": ROOT / "data" / "processed" / "aug_jarvis_2d_safe.pkl",
    "JARVIS-3D": ROOT / "data" / "processed" / "aug_jarvis_3d_safe.pkl",
    "DFT-3D": ROOT / "data" / "processed" / "aug_dft_3d_safe.pkl",
}
SOURCES = list(AUG_PATHS.keys())
EPOCHS = 50
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
SEED_DEFAULT = 42
# Reduced DFT-3D weight: 0.3 (v2) → 0.10 to mitigate the bulk-pristine
# bias surfaced in the LOHO multi-source experiment.
SOURCE_WEIGHTS = {"IMP2D": 1.0, "JARVIS-2D": 0.5, "JARVIS-3D": 0.5, "DFT-3D": 0.1}


def load_source_split(name: str, path: Path):
    """Return (train_aug, val, test) lists of dicts using the ordered meta."""
    with open(path, "rb") as f:
        blob = pickle.load(f)
    data = blob["data"]
    meta = blob["meta"]
    n_tr, n_va, n_te = meta["n_train"], meta["n_val"], meta["n_test"]
    train_aug = data[:n_tr]
    val = data[n_tr:n_tr + n_va]
    test = data[n_tr + n_va:n_tr + n_va + n_te]
    return train_aug, val, test, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or (
        "multi_source_train_v2_aug"
        if args.seed == SEED_DEFAULT
        else f"multi_source_train_v2_aug_seed{args.seed}"
    )

    t_start = time.time()
    device = base.get_device()
    print(f"Device: {device}  seed={args.seed}  tag={tag}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples_per_source, val_samples, test_samples = {}, {}, {}
    metas = {}
    for name, path in AUG_PATHS.items():
        if not path.exists():
            print(f"  WARN: {name} not found at {path}, skipping")
            continue
        train_aug, val, test, meta = load_source_split(name, path)
        samples_per_source[name] = train_aug
        if name == "IMP2D":
            val_samples[name] = val
            test_samples[name] = test
        metas[name] = meta
        print(f"  loaded {name}: train_aug={len(train_aug)} val={len(val)} test={len(test)}")

    mean_per_source, std_per_source = {}, {}
    for name, samples in samples_per_source.items():
        targets = np.array([s["target"] for s in samples], dtype=np.float64)
        mean_per_source[name] = float(targets.mean())
        std_per_source[name] = float(targets.std() + 1e-6)
        print(f"    {name}: n={len(samples)} mean={mean_per_source[name]:.4f} std={std_per_source[name]:.4f}")

    train_set = base.MultiSourceDataset(samples_per_source, mean_per_source, std_per_source)
    val_set = base.MultiSourceDataset(val_samples, mean_per_source, std_per_source)
    test_set = base.MultiSourceDataset(test_samples, mean_per_source, std_per_source)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=base.collate_fn_multi)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=base.collate_fn_multi)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=base.collate_fn_multi)

    print(f"\nDataset sizes: train {len(train_set)} val {len(val_set)} test {len(test_set)}")

    # Patch source weights on `base` so train_one + ds use the new values.
    base.SOURCE_WEIGHTS = SOURCE_WEIGHTS
    print(f"  source weights: {SOURCE_WEIGHTS}")

    model = base.MultiHeadPeriodicTransformer(
        n_sources=len(SOURCES),
        atom_fea_len=9, hidden_dim=128, n_local_layers=3, n_global_layers=2,
        num_heads=4, rcut_local=5.0, dmax_global=12.0,
        defect_embedding=True, dropout=0.1,
        use_pfa=True, use_long_range=False, use_defect_bias=False,
    )
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"\nModel params: {n_params/1e6:.4f}M")

    print(f"\nTraining {tag} for up to {args.epochs} epochs (seed={args.seed})...")
    model, history, best_val = base.train_one(model, train_loader, val_loader, device, epochs=args.epochs)

    test_mae, test_rmse, n_test = base.evaluate_imp2d(model, test_loader, device)
    print(f"\nFinal IMP2D test MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  (N={n_test})")

    out = {
        "config": {"seed": args.seed, "epochs": args.epochs, "lr": LR,
                   "batch_size": BATCH_SIZE, "patience": PATIENCE,
                   "source_weights": SOURCE_WEIGHTS,
                   "backbone": "PeriodicCrystalTransformer (PFA-only) + multi-head, ALL-source 3x aug"},
        "metas": {k: {kk: vv for kk, vv in v.items() if kk != "build_time_min"} for k, v in metas.items()},
        "n_params": n_params,
        "best_val_mae_imp2d_eV": best_val,
        "test_mae_imp2d_eV": test_mae,
        "test_rmse_imp2d_eV": test_rmse,
        "v1_baseline_test_mae_eV": 0.516,
        "v1_multi_source_test_mae_eV": 0.555,
        "v2_multi_source_test_mae_eV": 0.4929,
        "delta_pct_vs_v2_multi_source": (test_mae - 0.4929) / 0.4929 * 100,
        "history": history,
        "wall_min": (time.time() - t_start) / 60.0,
    }
    out_path = base.RESULTS / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    ckpt_dir = base.RESULTS / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": out["config"],
                "src_means": mean_per_source, "src_stds": std_per_source},
               ckpt_dir / "best.pt")
    # save IMP2D-head test predictions for ensemble UQ
    preds_arr, targets_arr = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred_norm = model(batch)
            pred = pred_norm * batch["src_std"] + batch["src_mean"]
            mask = batch["source_id"] == 0
            if not mask.any():
                continue
            preds_arr.append(pred[mask].cpu().numpy())
            targets_arr.append(batch["target"][mask].cpu().numpy())
    np.savez(ckpt_dir / "test_predictions.npz",
             preds=np.concatenate(preds_arr) if preds_arr else np.array([]),
             targets=np.concatenate(targets_arr) if targets_arr else np.array([]))
    print(f"\nWritten {out_path}; total {out['wall_min']:.1f} min")


if __name__ == "__main__":
    main()
