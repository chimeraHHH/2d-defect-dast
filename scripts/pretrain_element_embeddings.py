"""Pretrain element embeddings + local interaction layers on DFT-3D bulk crystals.

DFT-3D contains 19,902 bulk crystal structures with formation energies,
covering 89 elements — much broader chemical space than IMP2D's 65 dopants.
We train a lightweight CrystalTransformer on this dataset, then extract:
  - embed.weight, embed.bias  (9 → hidden_dim linear projection)
  - local_layers state_dict   (SchNet message-passing weights)

These are saved as a checkpoint that train_enhanced.py can load via the
`pretrained_embed` config key, giving the IMP2D model a better starting
point for element representation.

Usage:
  python -m scripts.pretrain_element_embeddings --config configs/pretrain_dft3d.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.models.baseline import CrystalTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Normalizer:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.mean = float(tensor.mean().item())
        self.std = float(tensor.std().item()) + 1e-6

    def norm(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.mean) / self.std

    def denorm(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.std + self.mean


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def evaluate(model, loader, normalizer, device):
    model.eval()
    abs_err, sq_err, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            target = batch["target"]
            preds = normalizer.denorm(model(batch))
            err = preds - target
            abs_err += err.abs().sum().item()
            sq_err += err.pow(2).sum().item()
            n += target.numel()
    return {"mae": abs_err / max(n, 1), "rmse": math.sqrt(sq_err / max(n, 1))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 42))

    dataset = CrystalGraphDataset(ROOT / cfg["data_path"])
    train_set, val_set, test_set = make_splits(
        dataset,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 42),
    )

    train_loader = DataLoader(train_set, batch_size=cfg.get("batch_size", 64),
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg.get("batch_size", 64),
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=cfg.get("batch_size", 64),
                             shuffle=False, collate_fn=collate_fn)

    targets = torch.tensor(
        [dataset.data[i]["target"] for i in train_set.indices], dtype=torch.float32
    )
    normalizer = Normalizer(targets)

    model = CrystalTransformer(**cfg.get("model_kwargs", {})).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("optimizer", {}).get("lr", 3e-4),
        weight_decay=cfg.get("optimizer", {}).get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()
    epochs = cfg.get("epochs", 30)
    grad_clip = cfg.get("grad_clip", 5.0)
    best_val_mae = float("inf")
    ckpt_path = out_dir / "best.pt"

    print(f"Pretraining on DFT-3D: {len(train_set)}/{len(val_set)}/{len(test_set)} "
          f"| {n_params/1e6:.3f}M params | device={device}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_abs, n_seen = 0.0, 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            target_norm = normalizer.norm(batch["target"])
            preds_norm = model(batch)
            loss = criterion(preds_norm, target_norm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            with torch.no_grad():
                preds = normalizer.denorm(preds_norm)
                train_abs += (preds - batch["target"]).abs().sum().item()
            n_seen += batch["target"].numel()

        train_mae = train_abs / max(n_seen, 1)
        val_m = evaluate(model, val_loader, normalizer, device)
        scheduler.step(val_m["mae"])

        improved = val_m["mae"] < best_val_mae
        if improved:
            best_val_mae = val_m["mae"]
            torch.save({
                "model": model.state_dict(),
                "embed_weight": model.embed.weight.data.clone(),
                "embed_bias": model.embed.bias.data.clone(),
                "local_layers": model.local_layers.state_dict(),
                "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
                "config": cfg,
                "epoch": epoch,
            }, ckpt_path)

        dt = time.time() - t0
        print(f"Ep {epoch:02d}/{epochs} | train {train_mae:.4f} | "
              f"val {val_m['mae']:.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"{dt:.1f}s {'*' if improved else ''}")

    # Final test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_m = evaluate(model, test_loader, normalizer, device)
    print(f"\n[Final] Test MAE {test_m['mae']:.4f} | RMSE {test_m['rmse']:.4f}")

    # Save lightweight embed-only checkpoint for downstream use
    embed_ckpt_path = out_dir / "pretrained_embed.pt"
    torch.save({
        "embed_weight": ckpt["embed_weight"],
        "embed_bias": ckpt["embed_bias"],
        "local_layers": ckpt["local_layers"],
        "source_dataset": "dft_3d_lite",
        "source_test_mae": test_m["mae"],
    }, embed_ckpt_path)
    print(f"Saved pretrained embeddings to {embed_ckpt_path}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test_mae": test_m["mae"], "test_rmse": test_m["rmse"],
                    "best_val_mae": best_val_mae, "n_params": n_params}, f, indent=2)


if __name__ == "__main__":
    main()
