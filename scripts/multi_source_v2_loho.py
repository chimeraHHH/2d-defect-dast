"""Leave-one-host-out training with v2-PFA multi-source backbone.

For each host h ∈ {MoS2, Cr2I6, C2H2, TaSe2, MoSSe}:
  - load `loho_<h>.pkl` which encodes a leak-free split where train = all
    non-host IMP2D × 3 aug, val = 10% random of train, test = all host
    samples (untouched);
  - join with JARVIS-2D / 3D / DFT-3D as auxiliary sources;
  - train a 4-head ``MultiHeadPeriodicTransformer`` (PFA-only) for 50 epochs;
  - evaluate IMP2D head on the held-out host test samples.

Output:
  - ``results/v2_loho_multi_<h>.json`` with test_mae / rmse / history
  - ``results/v2_loho_multi_<h>/best.pt`` and ``test_predictions.npz``

Comparing the resulting 5-host degradation profile to the v1 single-source
LOHO baseline (results/loho_summary.json) directly answers whether v2 + multi-
source helps OOD generalisation.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits  # noqa: E402
import scripts.multi_source_train_v2 as base  # noqa: E402

OTHER_DATA_PATHS = {
    "JARVIS-2D":  ROOT / "data" / "processed" / "jarvis_2d.pkl",
    "JARVIS-3D":  ROOT / "data" / "processed" / "jarvis_3d.pkl",
    "DFT-3D":     ROOT / "data" / "processed" / "dft_3d_lite.pkl",
}
SOURCES = ["IMP2D", *OTHER_DATA_PATHS.keys()]
EPOCHS = 50
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
SEED = 42
SOURCE_WEIGHTS = {"IMP2D": 1.0, "JARVIS-2D": 0.5, "JARVIS-3D": 0.5, "DFT-3D": 0.3}


def evaluate_loho(model, loader, device):
    model.eval()
    err, sq, n = 0.0, 0.0, 0
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred_norm = model(batch)
            pred = pred_norm * batch["src_std"] + batch["src_mean"]
            tgt = batch["target"]
            mask = batch["source_id"] == 0
            if not mask.any():
                continue
            d = (pred[mask] - tgt[mask]).abs()
            err += d.sum().item()
            sq += (d ** 2).sum().item()
            n += int(mask.sum().item())
            preds.append(pred[mask].cpu().numpy())
            targets.append(tgt[mask].cpu().numpy())
    mae = err / max(n, 1)
    rmse = (sq / max(n, 1)) ** 0.5
    preds_arr = np.concatenate(preds) if preds else np.array([])
    targets_arr = np.concatenate(targets) if targets else np.array([])
    return mae, rmse, n, preds_arr, targets_arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True, choices=["MoS2", "Cr2I6", "C2H2", "TaSe2", "MoSSe"])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()

    t_start = time.time()
    device = base.get_device()
    print(f"Device: {device}  host={args.host}  seed={SEED}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load LOHO IMP2D dataset (leak-free ordered split)
    loho_path = ROOT / "data" / "processed" / f"loho_{args.host}.pkl"
    with open(loho_path, "rb") as f:
        loho_blob = pickle.load(f)
    loho_data = loho_blob["data"] if isinstance(loho_blob, dict) else loho_blob
    meta = loho_blob.get("meta", {}) if isinstance(loho_blob, dict) else {}
    n_tr = meta["n_train"]; n_va = meta["n_val"]; n_te = meta["n_test"]
    imp2d_train = loho_data[:n_tr]
    imp2d_val = loho_data[n_tr:n_tr + n_va]
    imp2d_test = loho_data[n_tr + n_va:n_tr + n_va + n_te]
    print(f"  loho IMP2D split: train {len(imp2d_train)} val {len(imp2d_val)} test {len(imp2d_test)} (host={args.host})")

    # Load other sources (full)
    other = {}
    for name, path in OTHER_DATA_PATHS.items():
        if not path.exists():
            print(f"  WARN: {name} not found at {path}, skipping")
            continue
        with open(path, "rb") as f:
            other[name] = pickle.load(f)
        print(f"  loaded {name}: {len(other[name])} samples")

    samples_per_source = {"IMP2D": imp2d_train, **other}
    val_samples = {"IMP2D": imp2d_val}
    test_samples = {"IMP2D": imp2d_test}

    mean_per_source, std_per_source = {}, {}
    for name, samples in samples_per_source.items():
        targets = np.array([s["target"] for s in samples], dtype=np.float64)
        mean_per_source[name] = float(targets.mean())
        std_per_source[name] = float(targets.std() + 1e-6)
        print(f"  {name}: n={len(samples)} mean={mean_per_source[name]:.4f} std={std_per_source[name]:.4f}")

    train_set = base.MultiSourceDataset(samples_per_source, mean_per_source, std_per_source)
    val_set = base.MultiSourceDataset(val_samples, mean_per_source, std_per_source)
    test_set = base.MultiSourceDataset(test_samples, mean_per_source, std_per_source)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=base.collate_fn_multi)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=base.collate_fn_multi)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=base.collate_fn_multi)

    print(f"\nDataset sizes: train {len(train_set)}  val {len(val_set)}  test {len(test_set)}")

    model = base.MultiHeadPeriodicTransformer(
        n_sources=len(SOURCES),
        atom_fea_len=9, hidden_dim=128, n_local_layers=3, n_global_layers=2,
        num_heads=4, rcut_local=5.0, dmax_global=12.0,
        defect_embedding=True, dropout=0.1,
        use_pfa=True, use_long_range=False, use_defect_bias=False,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {n_params/1e6:.4f}M ({n_params:,})")

    # Override SEED constant on base for any reused trainer state
    base.SEED = SEED
    print(f"\nTraining v2-PFA multi-source LOHO (host={args.host}) for up to {args.epochs} epochs...")
    model, history, best_val = base.train_one(model, train_loader, val_loader, device, epochs=args.epochs)

    test_mae, test_rmse, n_test, preds_arr, targets_arr = evaluate_loho(model, test_loader, device)
    print(f"\nLOHO {args.host} test MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  (N={n_test})")

    # v1 LOHO single-source baseline numbers (from results/loho_summary.json)
    V1_LOHO_BASELINE = {
        "MoS2":  {"test_mae": 0.5163, "deg_factor": 1.000},
        "Cr2I6": {"test_mae": 0.8401, "deg_factor": 1.628},
        "C2H2":  {"test_mae": 2.3987, "deg_factor": 4.648},
        "TaSe2": {"test_mae": 0.6364, "deg_factor": 1.233},
        "MoSSe": {"test_mae": 0.4479, "deg_factor": 0.868},
    }
    v1_mae = V1_LOHO_BASELINE[args.host]["test_mae"]
    v1_deg = V1_LOHO_BASELINE[args.host]["deg_factor"]

    out = {
        "host": args.host,
        "config": {"epochs": args.epochs, "lr": LR, "batch_size": BATCH_SIZE, "seed": SEED,
                   "source_weights": SOURCE_WEIGHTS,
                   "backbone": "PeriodicCrystalTransformer (PFA-only) + multi-head"},
        "n_params": n_params,
        "n_test": n_test,
        "best_val_mae_imp2d_eV": best_val,
        "test_mae_imp2d_eV": test_mae,
        "test_rmse_imp2d_eV": test_rmse,
        "v1_loho_singlesource_test_mae_eV": v1_mae,
        "v1_loho_singlesource_degradation_factor": v1_deg,
        "delta_pct_vs_v1_loho_singlesource": (test_mae - v1_mae) / v1_mae * 100,
        "history": history,
        "wall_min": (time.time() - t_start) / 60.0,
    }
    out_path = base.RESULTS / f"v2_loho_multi_{args.host}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    ckpt_dir = base.RESULTS / f"v2_loho_multi_{args.host}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": out["config"],
                "src_means": mean_per_source, "src_stds": std_per_source,
                "host": args.host}, ckpt_dir / "best.pt")
    np.savez(ckpt_dir / "test_predictions.npz", preds=preds_arr, targets=targets_arr)
    print(f"\nWritten {out_path}")
    print(f"Wall time: {out['wall_min']:.1f} min")


if __name__ == "__main__":
    main()
