"""Training loop shared by baseline and improved models."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.models import CrystalTransformer, DefectAwareTransformer


MODEL_REGISTRY = {
    "baseline": CrystalTransformer,
    "improved": DefectAwareTransformer,
}


class Normalizer:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.mean = float(tensor.mean().item())
        self.std = float(tensor.std().item()) + 1e-6

    def norm(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.mean) / self.std

    def denorm(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.std + self.mean

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def evaluate(model, loader, normalizer, device):
    model.eval()
    abs_err, sq_err, n = 0.0, 0.0, 0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            target = batch["target"]
            preds_norm = model(batch)
            preds = normalizer.denorm(preds_norm)
            err = preds - target
            abs_err += err.abs().sum().item()
            sq_err += err.pow(2).sum().item()
            n += target.numel()
            preds_all.append(preds.cpu())
            targets_all.append(target.cpu())
    mae = abs_err / max(n, 1)
    rmse = math.sqrt(sq_err / max(n, 1))
    return {
        "mae": mae,
        "rmse": rmse,
        "preds": torch.cat(preds_all).numpy() if preds_all else np.array([]),
        "targets": torch.cat(targets_all).numpy() if targets_all else np.array([]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=0, help="optional debug cap")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    metrics_path = out_dir / "metrics.json"
    ckpt_path = out_dir / "best.pt"

    # MPS softmax + masked_fill currently produces sporadic NaNs after training,
    # so the default device is CPU on Apple Silicon. Override with --device mps
    # if a future torch release fixes the issue.
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    set_seed(cfg.get("seed", 42))

    # Data
    dataset_path = ROOT / cfg["data_path"]
    dataset = CrystalGraphDataset(dataset_path)
    train_set, val_set, test_set = make_splits(
        dataset,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 42),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.get("batch_size", 16),
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.get("batch_size", 16), shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.get("batch_size", 16), shuffle=False, collate_fn=collate_fn
    )

    # Normalizer
    targets = torch.tensor([dataset.data[i]["target"] for i in train_set.indices], dtype=torch.float32)
    normalizer = Normalizer(targets)

    # Model
    model_cls = MODEL_REGISTRY[cfg["model"]]
    model_kwargs = cfg.get("model_kwargs", {})
    model = model_cls(**model_kwargs).to(device)

    # Optim
    optim_kwargs = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_kwargs.get("lr", 1e-3),
        weight_decay=optim_kwargs.get("weight_decay", 1e-4),
    )
    sched_kwargs = cfg.get("scheduler", {"factor": 0.5, "patience": 5})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", **sched_kwargs
    )

    loss_name = cfg.get("loss", "huber")
    if loss_name == "huber":
        criterion = nn.HuberLoss(delta=cfg.get("huber_delta", 1.0))
    elif loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError(loss_name)

    epochs = cfg.get("epochs", 30)
    grad_clip = cfg.get("grad_clip", 1.0)

    history = []
    best_val_mae = float("inf")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(log_path, "w") as logf:
        msg = (
            f"Config: {json.dumps(cfg, ensure_ascii=False)}\n"
            f"Device: {device}\n"
            f"Model: {model_cls.__name__} | params={n_params/1e6:.3f}M\n"
            f"Train/Val/Test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}\n"
            f"Target stats: mean={normalizer.mean:.4f} std={normalizer.std:.4f}\n"
        )
        print(msg)
        logf.write(msg)
        logf.flush()

        global_step = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            train_loss, train_abs, n_seen = 0.0, 0.0, 0
            for it, batch in enumerate(train_loader, start=1):
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
                    abs_err = (preds - batch["target"]).abs().sum().item()
                bs = int(batch["target"].numel())
                train_loss += loss.item() * bs
                train_abs += abs_err
                n_seen += bs
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    break

            train_mae = train_abs / max(n_seen, 1)

            val_metrics = evaluate(model, val_loader, normalizer, device)
            scheduler.step(val_metrics["mae"])

            improved = val_metrics["mae"] < best_val_mae
            if improved:
                best_val_mae = val_metrics["mae"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "normalizer": normalizer.state_dict(),
                        "config": cfg,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )

            dt = time.time() - t0
            row = {
                "epoch": epoch,
                "train_mae": train_mae,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "lr": optimizer.param_groups[0]["lr"],
                "time_sec": dt,
                "best": improved,
            }
            history.append(row)
            line = (
                f"Epoch {epoch:02d}/{epochs} | train MAE {train_mae:.4f} | "
                f"val MAE {val_metrics['mae']:.4f} RMSE {val_metrics['rmse']:.4f} | "
                f"lr {row['lr']:.2e} | {dt:.1f}s {'*' if improved else ''}"
            )
            print(line)
            logf.write(line + "\n")
            logf.flush()
            if args.max_steps and global_step >= args.max_steps:
                break

        # final test
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, test_loader, normalizer, device)
        final_line = (
            f"\n[Final] Test MAE {test_metrics['mae']:.4f} | RMSE {test_metrics['rmse']:.4f}\n"
        )
        print(final_line)
        logf.write(final_line)

    summary = {
        "config": cfg,
        "n_params": n_params,
        "history": history,
        "best_val_mae": best_val_mae,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    # save predictions for plotting
    np.savez(
        out_dir / "test_predictions.npz",
        preds=test_metrics["preds"],
        targets=test_metrics["targets"],
    )


if __name__ == "__main__":
    main()
