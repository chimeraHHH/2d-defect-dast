#!/usr/bin/env python3
"""Test-Time Augmentation (TTA) evaluation for CrystalTransformerV2.

Applies the same geometric augmentations used during training (rotation,
perturbation, strain) at inference time, runs the model N times per sample,
and averages predictions. This improves accuracy for free (no retraining).

Usage:
    python scripts/tta_ensemble_eval.py \
        --checkpoints results/v2_gated_pooling/best.pt \
                      results/v2_gated_pooling_s43/best.pt \
        --n-aug 16 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.augment_online import OnlineAugTransform
from src.train_enhanced import Normalizer, move_batch

# ── Detect V1 vs V2 from config ──────────────────────────────────────────
def load_model_and_normalizer(ckpt_path, device):
    """Load a model + normalizer from a checkpoint."""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Determine model type from config or state_dict keys
    config = ckpt.get("config", {})
    model_type = config.get("model", "CrystalTransformer")

    # Detect checkpoint format (V2 vs V1)
    if "model" in ckpt and isinstance(ckpt["model"], dict) and "config" in ckpt:
        # V2 checkpoint format: {"model": state_dict, "config": ..., "normalizer": ...}
        state_dict = ckpt["model"]
        norm_data = ckpt.get("normalizer", {})
        mean = norm_data.get("mean", 0.0)
        std = norm_data.get("std", 1.0)
    else:
        # V1 checkpoint format
        state_dict = ckpt.get("swa_model_state") or ckpt.get("model_state")
        mean = ckpt.get("normalizer_mean", 0.0)
        std = ckpt.get("normalizer_std", 1.0)

    # Build model based on model type
    is_v2 = model_type in ("v2", "CrystalTransformerV2")
    if is_v2:
        from src.models.crystal_v2 import CrystalTransformerV2
        model = CrystalTransformerV2(**config.get("model_kwargs", {}))
    else:
        from src.models.baseline import CrystalTransformer
        model = CrystalTransformer(**config.get("model_kwargs", {}))

    model.load_state_dict(state_dict)
    model.to(device).eval()

    normalizer = Normalizer(torch.tensor([mean]))
    normalizer.mean = mean
    normalizer.std = std

    return model, normalizer, model_type


def augment_batch(batch, transform, device):
    """Apply augmentation to each sample in a collated batch, then re-collate.

    Since augmentation works on individual samples (need per-sample edges),
    we uncollate -> augment -> re-collate.
    """
    # Uncollate: extract individual samples from the padded batch
    num_atoms_list = batch["num_atoms_list"]
    B = len(num_atoms_list)
    samples = []

    for i in range(B):
        n = num_atoms_list[i]
        sample = {
            "x": batch["x"][i, :n].cpu(),
            "defect_mask": batch["defect_mask"][i, :n].cpu(),
            "dist_matrix": batch["dist_matrix"][i, :n, :n].cpu(),
            "positions": batch["positions"][i, :n].cpu(),
            "cell": batch["cell"][i].cpu(),
            "target": batch["target"][i].cpu(),
            "num_atoms": n,
            "edge_index": batch["edge_index_list"][i].cpu(),
            "edge_dist": batch["edge_dist_list"][i].cpu(),
            "triplet_index": batch["triplet_index_list"][i].cpu(),
            "angles": batch["angles_list"][i].cpu(),
            "atomic_numbers": batch["atomic_numbers"][i, :n].cpu(),
        }
        if batch.get("edge_offset_list"):
            sample["edge_offset"] = batch["edge_offset_list"][i].cpu()

        # Apply augmentation
        sample = transform(sample)
        samples.append(sample)

    # Re-collate
    aug_batch = collate_fn(samples)
    return move_batch(aug_batch, device)


@torch.no_grad()
def evaluate_tta(models_and_norms, loader, device, n_aug=16,
                 sigma_range=(0.01, 0.05), strain_range=2.0):
    """Run TTA evaluation: for each batch, apply n_aug augmentations and average."""

    transform = OnlineAugTransform(
        sigma_range=sigma_range,
        strain_range=strain_range,
        rotate_prob=1.0,
        perturb_prob=0.8,
        strain_prob=0.3,
    )

    all_preds_per_model = {i: [] for i in range(len(models_and_norms))}
    all_preds_tta = {i: [] for i in range(len(models_and_norms))}
    all_targets = []

    for batch_idx, batch in enumerate(loader):
        batch = move_batch(batch, device)
        target = batch["target"]
        all_targets.append(target.cpu())
        B = target.shape[0]

        # Clean (no augmentation) predictions
        for m_idx, (model, normalizer, _) in enumerate(models_and_norms):
            preds_norm = model(batch)
            preds = normalizer.denorm(preds_norm)
            all_preds_per_model[m_idx].append(preds.cpu())

        # TTA predictions: accumulate across augmentations
        tta_accum = {i: torch.zeros(B, device="cpu") for i in range(len(models_and_norms))}

        for aug_i in range(n_aug):
            aug_batch = augment_batch(batch, transform, device)
            for m_idx, (model, normalizer, _) in enumerate(models_and_norms):
                preds_norm = model(aug_batch)
                preds = normalizer.denorm(preds_norm)
                tta_accum[m_idx] += preds.cpu()

        for m_idx in range(len(models_and_norms)):
            tta_avg = tta_accum[m_idx] / n_aug
            all_preds_tta[m_idx].append(tta_avg)

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}")

    # Concatenate
    targets = torch.cat(all_targets).numpy()
    results = {}

    clean_preds_all = {}
    tta_preds_all = {}

    for m_idx, (_, _, model_type) in enumerate(models_and_norms):
        clean_preds = torch.cat(all_preds_per_model[m_idx]).numpy()
        tta_preds = torch.cat(all_preds_tta[m_idx]).numpy()
        clean_preds_all[m_idx] = clean_preds
        tta_preds_all[m_idx] = tta_preds

    return clean_preds_all, tta_preds_all, targets


def compute_metrics(preds, targets):
    """Compute MAE and RMSE."""
    err = preds - targets
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    return mae, rmse


def greedy_ensemble(preds_dict, targets, max_k=None):
    """Greedy forward selection for best ensemble."""
    names = sorted(preds_dict.keys())
    preds_array = np.array([preds_dict[n] for n in names])
    n_models = len(names)
    if max_k is None:
        max_k = n_models

    selected_idx = []
    results = []
    for k in range(1, max_k + 1):
        best_mae = float("inf")
        best_i = -1
        for i in range(n_models):
            if i in selected_idx:
                continue
            candidate = selected_idx + [i]
            ens_pred = preds_array[candidate].mean(axis=0)
            mae = np.mean(np.abs(ens_pred - targets))
            if mae < best_mae:
                best_mae = mae
                best_i = i
        if best_i < 0:
            break
        selected_idx.append(best_i)
        ens_pred = preds_array[selected_idx].mean(axis=0)
        rmse = np.sqrt(np.mean((ens_pred - targets) ** 2))
        results.append({"k": k, "name": names[best_i], "mae": float(best_mae), "rmse": float(rmse)})
    return results


def main():
    parser = argparse.ArgumentParser(description="TTA evaluation")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to model checkpoint files")
    parser.add_argument("--n-aug", type=int, default=16,
                        help="Number of augmentation runs per sample")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-path", default="data/processed/cleaned_dataset.pkl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CrystalGraphDataset(ROOT / args.data_path)
    _, _, test_set = make_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"Test set: {len(test_set)} samples, {len(test_loader)} batches")
    print(f"TTA augmentations: {args.n_aug}")

    # Load models
    models_and_norms = []
    for ckpt_path in args.checkpoints:
        model, normalizer, model_type = load_model_and_normalizer(ckpt_path, device)
        models_and_norms.append((model, normalizer, model_type))
        print(f"Loaded {Path(ckpt_path).parent.name} ({model_type})")

    # Run TTA evaluation
    t0 = time.time()
    clean_preds, tta_preds, targets = evaluate_tta(
        models_and_norms, test_loader, device, n_aug=args.n_aug
    )
    elapsed = time.time() - t0
    print(f"\nInference time: {elapsed:.1f}s")

    # === Individual results ===
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 60)
    ckpt_names = [Path(p).parent.name for p in args.checkpoints]

    for m_idx, name in enumerate(ckpt_names):
        clean_mae, clean_rmse = compute_metrics(clean_preds[m_idx], targets)
        tta_mae, tta_rmse = compute_metrics(tta_preds[m_idx], targets)
        improvement = (clean_mae - tta_mae) / clean_mae * 100
        print(f"  {name}:")
        print(f"    Clean:  MAE={clean_mae:.4f}  RMSE={clean_rmse:.4f}")
        print(f"    TTA-{args.n_aug}: MAE={tta_mae:.4f}  RMSE={tta_rmse:.4f}  ({improvement:+.1f}%)")

    # === Clean ensemble ===
    n_models = len(ckpt_names)
    if n_models > 1:
        print("\n" + "=" * 60)
        print("ENSEMBLE RESULTS")
        print("=" * 60)

        # All-model clean ensemble
        clean_ens = np.mean([clean_preds[i] for i in range(n_models)], axis=0)
        cm, cr = compute_metrics(clean_ens, targets)
        print(f"  Clean {n_models}-ensemble: MAE={cm:.4f} RMSE={cr:.4f}")

        # All-model TTA ensemble
        tta_ens = np.mean([tta_preds[i] for i in range(n_models)], axis=0)
        tm, tr = compute_metrics(tta_ens, targets)
        improvement = (cm - tm) / cm * 100
        print(f"  TTA-{args.n_aug} {n_models}-ensemble: MAE={tm:.4f} RMSE={tr:.4f}  ({improvement:+.1f}%)")

        # Mixed clean+TTA (2*n_models effective models)
        mixed_preds = {}
        for i, name in enumerate(ckpt_names):
            mixed_preds[f"clean_{name}"] = clean_preds[i]
            mixed_preds[f"tta_{name}"] = tta_preds[i]

        print(f"\n  Greedy ensemble (clean + TTA = {2*n_models} virtual models):")
        greedy_results = greedy_ensemble(mixed_preds, targets)
        for r in greedy_results[:10]:
            marker = " ***" if r["mae"] == min(x["mae"] for x in greedy_results) else ""
            print(f"    k={r['k']}: MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} (+{r['name']}){marker}")

    # Save results
    out = {
        "n_aug": args.n_aug,
        "n_models": n_models,
        "individual": {},
        "n_test": len(targets),
    }
    for m_idx, name in enumerate(ckpt_names):
        cm, cr = compute_metrics(clean_preds[m_idx], targets)
        tm, tr = compute_metrics(tta_preds[m_idx], targets)
        out["individual"][name] = {
            "clean_mae": float(cm), "clean_rmse": float(cr),
            "tta_mae": float(tm), "tta_rmse": float(tr),
        }
    if n_models > 1:
        out["ensemble_clean"] = {"mae": float(cm), "rmse": float(cr)}
        out["ensemble_tta"] = {"mae": float(tm), "rmse": float(tr)}

    out_path = args.output or str(ROOT / "results" / "tta_ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save predictions
    np.savez(
        str(Path(out_path).with_suffix(".npz")),
        targets=targets,
        **{f"clean_{name}": clean_preds[i] for i, name in enumerate(ckpt_names)},
        **{f"tta_{name}": tta_preds[i] for i, name in enumerate(ckpt_names)},
    )


if __name__ == "__main__":
    main()
