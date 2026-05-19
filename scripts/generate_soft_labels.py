#!/usr/bin/env python3
"""Generate ensemble soft labels for knowledge distillation.

Runs the top-K models from greedy ensemble selection on the FULL dataset
(train+val+test) and saves averaged predictions per sample. These soft
labels are then used as auxiliary targets during distillation training.

Usage:
    python scripts/generate_soft_labels.py \
        --checkpoints results/v2_gated_pooling_s43/best.pt \
                      results/v2_enhanced_env_s43/best.pt \
                      results/v2_gated_pooling_s44/best.pt \
        --output data/processed/soft_labels.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn
from src.train_enhanced import Normalizer, move_batch


def load_model_and_normalizer(ckpt_path, device):
    """Load a model + normalizer from a checkpoint."""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_type = config.get("model", "CrystalTransformer")

    if "model" in ckpt and isinstance(ckpt["model"], dict) and "config" in ckpt:
        state_dict = ckpt["model"]
        norm_data = ckpt.get("normalizer", {})
        mean = norm_data.get("mean", 0.0)
        std = norm_data.get("std", 1.0)
    else:
        state_dict = ckpt.get("swa_model_state") or ckpt.get("model_state")
        mean = ckpt.get("normalizer_mean", 0.0)
        std = ckpt.get("normalizer_std", 1.0)

    is_v2 = model_type in ("v2", "CrystalTransformerV2")
    if is_v2:
        from src.models.crystal_v2 import CrystalTransformerV2
        model = CrystalTransformerV2(**config.get("model_kwargs", {}))
    else:
        from src.models.baseline import CrystalTransformer
        model = CrystalTransformer(**config.get("model_kwargs", {}))

    model.load_state_dict(state_dict)
    model.to(device).eval()

    transform = norm_data.get("transform", "none") if isinstance(norm_data, dict) else "none"
    normalizer = Normalizer(torch.tensor([mean]), transform=transform)
    normalizer.mean = mean
    normalizer.std = std
    return model, normalizer


@torch.no_grad()
def generate_predictions(models_and_norms, loader, device):
    """Run all models on the full dataset, return per-sample ensemble predictions."""
    all_preds = {i: [] for i in range(len(models_and_norms))}
    all_targets = []

    for batch_idx, batch in enumerate(loader):
        batch = move_batch(batch, device)
        target = batch["target"]
        all_targets.append(target.cpu().numpy())

        for m_idx, (model, normalizer) in enumerate(models_and_norms):
            preds_norm = model(batch)
            preds = normalizer.denorm(preds_norm)
            all_preds[m_idx].append(preds.cpu().numpy())

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}")

    targets = np.concatenate(all_targets)
    per_model = {i: np.concatenate(all_preds[i]) for i in range(len(models_and_norms))}
    # Ensemble average
    ensemble_preds = np.mean([per_model[i] for i in range(len(models_and_norms))], axis=0)
    return ensemble_preds, per_model, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-path", default="data/processed/cleaned_dataset.pkl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="data/processed/soft_labels.pkl")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load FULL dataset (no splitting — we need predictions for every sample)
    dataset = CrystalGraphDataset(ROOT / args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches")

    # Load models
    models_and_norms = []
    for ckpt_path in args.checkpoints:
        model, normalizer = load_model_and_normalizer(ckpt_path, device)
        models_and_norms.append((model, normalizer))
        print(f"Loaded: {Path(ckpt_path).parent.name}")

    # Generate predictions
    t0 = time.time()
    ensemble_preds, per_model_preds, targets = generate_predictions(
        models_and_norms, loader, device
    )
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s")

    # Statistics
    mae_ensemble = np.mean(np.abs(ensemble_preds - targets))
    print(f"Ensemble MAE on full dataset: {mae_ensemble:.4f}")
    for i in range(len(models_and_norms)):
        mae_i = np.mean(np.abs(per_model_preds[i] - targets))
        print(f"  Model {i}: MAE={mae_i:.4f}")

    # Save soft labels — per-sample ensemble predictions
    output = {
        "soft_labels": ensemble_preds.astype(np.float32),  # (N,) array
        "n_samples": len(dataset),
        "n_models": len(models_and_norms),
        "checkpoints": [str(p) for p in args.checkpoints],
        "mae_ensemble": float(mae_ensemble),
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved soft labels to {out_path}")


if __name__ == "__main__":
    main()
