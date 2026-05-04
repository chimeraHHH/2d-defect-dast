"""Run multi_source_train_v2 with a configurable seed and output dir.

Used by the 3-seed verification queue after the headline v2 multi-source result.
Imports the main training routine from ``multi_source_train_v2`` and just
overrides SEED + output paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.multi_source_train_v2 as base  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tag", type=str, default=None,
                   help="suffix for run dir; default seed{N}")
    args = p.parse_args()

    tag = args.tag or f"seed{args.seed}"
    base.SEED = args.seed
    # Override output paths inside base.main by monkey-patching at module level
    import torch, numpy as np, time, pickle
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t_start = time.time()
    device = base.get_device()
    print(f"Device: {device}  seed={args.seed}  tag={tag}")

    sources_data = {}
    for name, path in base.DATA_PATHS.items():
        if not path.exists():
            print(f"  WARN: {name} not found at {path}, skipping")
            continue
        with open(path, "rb") as f:
            sources_data[name] = pickle.load(f)
        print(f"  loaded {name}: {len(sources_data[name])} samples")

    imp2d = sources_data["IMP2D"]
    train_idx, val_idx, test_idx = base.split_indices(len(imp2d), 0.8, 0.1, args.seed)
    imp2d_train = [imp2d[i] for i in train_idx]
    imp2d_val = [imp2d[i] for i in val_idx]
    imp2d_test = [imp2d[i] for i in test_idx]
    print(f"\nIMP2D split (seed={args.seed}): train {len(imp2d_train)} val {len(imp2d_val)} test {len(imp2d_test)}")

    other_sources = {n: d for n, d in sources_data.items() if n != "IMP2D"}
    samples_per_source = {"IMP2D": imp2d_train, **other_sources}
    val_samples = {"IMP2D": imp2d_val}
    test_samples = {"IMP2D": imp2d_test}

    mean_per_source = {}
    std_per_source = {}
    for name, samples in samples_per_source.items():
        targets = np.array([s["target"] for s in samples], dtype=np.float64)
        mean_per_source[name] = float(targets.mean())
        std_per_source[name] = float(targets.std() + 1e-6)
        print(f"  {name}: n={len(samples)} mean={mean_per_source[name]:.4f} std={std_per_source[name]:.4f}")

    train_set = base.MultiSourceDataset(samples_per_source, mean_per_source, std_per_source)
    val_set = base.MultiSourceDataset(val_samples, mean_per_source, std_per_source)
    test_set = base.MultiSourceDataset(test_samples, mean_per_source, std_per_source)

    train_loader = DataLoader(train_set, batch_size=base.BATCH_SIZE, shuffle=True, collate_fn=base.collate_fn_multi)
    val_loader = DataLoader(val_set, batch_size=base.BATCH_SIZE, shuffle=False, collate_fn=base.collate_fn_multi)
    test_loader = DataLoader(test_set, batch_size=base.BATCH_SIZE, shuffle=False, collate_fn=base.collate_fn_multi)

    model = base.MultiHeadPeriodicTransformer(
        n_sources=len(base.SOURCES),
        atom_fea_len=9, hidden_dim=128, n_local_layers=3, n_global_layers=2,
        num_heads=4, rcut_local=5.0, dmax_global=12.0,
        defect_embedding=True, dropout=0.1,
        use_pfa=True, use_long_range=False, use_defect_bias=False,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {n_params/1e6:.4f}M")

    print(f"\nTraining for up to {base.EPOCHS} epochs (seed={args.seed})...")
    model, history, best_val = base.train_one(model, train_loader, val_loader, device, epochs=base.EPOCHS)

    test_mae, test_rmse, n_test = base.evaluate_imp2d(model, test_loader, device)
    print(f"\nFinal IMP2D test MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  (N={n_test})")

    out = {
        "config": {"seed": args.seed, "epochs": base.EPOCHS, "lr": base.LR,
                   "batch_size": base.BATCH_SIZE,
                   "source_weights": base.SOURCE_WEIGHTS,
                   "backbone": "PeriodicCrystalTransformer (PFA-only)"},
        "n_params": n_params,
        "best_val_mae_imp2d_eV": best_val,
        "test_mae_imp2d_eV": test_mae,
        "test_rmse_imp2d_eV": test_rmse,
        "v1_baseline_test_mae_eV": 0.516,
        "v1_multi_source_test_mae_eV": 0.555,
        "delta_pct_vs_v1_baseline": (test_mae - 0.516) / 0.516 * 100,
        "delta_pct_vs_v1_multisource": (test_mae - 0.555) / 0.555 * 100,
        "history": history,
        "wall_min": (time.time() - t_start) / 60.0,
    }
    out_path = base.RESULTS / f"multi_source_train_v2_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    ckpt_dir = base.RESULTS / f"multi_source_v2_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": out["config"],
                "sources": base.SOURCES, "src_means": mean_per_source,
                "src_stds": std_per_source}, ckpt_dir / "best.pt")
    # save test predictions for ensemble UQ later
    import torch as _torch
    preds_arr, targets_arr = [], []
    model.eval()
    with _torch.no_grad():
        for batch in test_loader:
            batch = {k: (v.to(device) if _torch.is_tensor(v) else v) for k, v in batch.items()}
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
    print(f"\nWritten {out_path}")
    print(f"Wall time: {out['wall_min']:.1f} min")


if __name__ == "__main__":
    main()
