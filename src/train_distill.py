"""Knowledge distillation: 4-model V1 ensemble teacher → V2 student.

The ensemble of 4 trained CrystalTransformer (V1) models acts as a
"teacher" that provides soft regression targets.  The V2 student is
trained on a weighted combination of:
  1. Task loss: L1(pred, target_normalized)       — ground truth
  2. Distill loss: L1(pred, teacher_avg_pred)      — soft targets
  3. Hidden loss: MSE(student_hidden, teacher_hidden) — representation matching

Usage:
  python -m src.train_distill --config configs/v2_distill.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.sampler import HostBalancedSampler
from src.augment_online import OnlineAugTransform, OnlineAugDataset
from src.models import CrystalTransformer, CrystalTransformerV2
from src.train_enhanced import (
    Normalizer, set_seed, move_batch, apply_stochastic_depth, evaluate,
)


class EnsembleTeacher(nn.Module):
    """Frozen ensemble of V1 models that produces soft targets."""

    def __init__(self, checkpoints: List[str], device: torch.device,
                 model_kwargs: dict) -> None:
        super().__init__()
        self.models = nn.ModuleList()
        self.normalizers = []
        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = CrystalTransformer(**model_kwargs)
            # Handle SWA wrapped state dicts
            state_dict = ckpt.get("swa_model_state", ckpt.get("model_state"))
            if state_dict is None:
                state_dict = ckpt
            # Remove 'module.' prefix from SWA
            cleaned = {}
            for k, v in state_dict.items():
                cleaned[k.replace("module.", "")] = v
            model.load_state_dict(cleaned, strict=False)
            model.to(device)
            model.eval()
            self.models.append(model)
            # Load normalizer
            if "normalizer_mean" in ckpt:
                norm = Normalizer(torch.zeros(1))
                norm.mean = ckpt["normalizer_mean"]
                norm.std = ckpt["normalizer_std"]
            else:
                norm = None
            self.normalizers.append(norm)

        for p in self.parameters():
            p.requires_grad_(False)
        print(f"Loaded {len(self.models)} teacher models")

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor]):
        """Returns (avg_pred_normalized, avg_hidden).

        avg_pred_normalized is in the *student's* normalized space.
        We return raw (denormalized) predictions and let the caller re-normalize.
        """
        preds_raw = []
        for i, model in enumerate(self.models):
            pred_norm = model(batch)
            if self.normalizers[i] is not None:
                pred_raw = self.normalizers[i].denorm(pred_norm)
            else:
                pred_raw = pred_norm
            preds_raw.append(pred_raw)
        # Average in real (eV) space
        avg_raw = torch.stack(preds_raw).mean(dim=0)
        return avg_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["output_dir"] = cfg["output_dir"] + f"_s{args.seed}"

    split_seed = cfg.get("split_seed", 42)
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    metrics_path = out_dir / "metrics.json"
    ckpt_path = out_dir / "best.pt"

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = CrystalGraphDataset(ROOT / cfg["data_path"])
    train_set, val_set, test_set = make_splits(
        dataset,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=split_seed,
    )
    set_seed(cfg.get("seed", 42))

    # Online augmentation
    use_online_aug = cfg.get("online_aug", False)
    aug_transform = None
    if use_online_aug:
        aug_cfg = cfg.get("online_aug_cfg", {})
        aug_transform = OnlineAugTransform(
            sigma_range=tuple(aug_cfg.get("sigma_range", [0.01, 0.05])),
            strain_range=aug_cfg.get("strain_range", 2.0),
            rotate_prob=aug_cfg.get("rotate_prob", 1.0),
            perturb_prob=aug_cfg.get("perturb_prob", 0.8),
            strain_prob=aug_cfg.get("strain_prob", 0.3),
        )

    # Host-balanced sampling
    use_balanced = cfg.get("host_balanced", False)
    if use_balanced:
        sampler = HostBalancedSampler(
            dataset, subset_indices=train_set.indices,
            samples_per_host=cfg.get("samples_per_host", 200),
            seed=cfg.get("seed", 42),
        )
        loader_ds = OnlineAugDataset(dataset, transform=aug_transform) if use_online_aug else dataset
        train_loader = DataLoader(
            loader_ds, batch_size=cfg.get("batch_size", 64),
            sampler=sampler, collate_fn=collate_fn,
        )
    else:
        loader_ds = OnlineAugDataset(train_set, transform=aug_transform) if use_online_aug else train_set
        train_loader = DataLoader(
            loader_ds, batch_size=cfg.get("batch_size", 64),
            shuffle=True, collate_fn=collate_fn,
        )
    val_loader = DataLoader(val_set, batch_size=cfg.get("batch_size", 64),
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=cfg.get("batch_size", 64),
                             shuffle=False, collate_fn=collate_fn)

    # Student normalizer
    targets = torch.tensor(
        [dataset.data[i]["target"] for i in train_set.indices], dtype=torch.float32
    )
    normalizer = Normalizer(targets)

    # ---- Teacher ensemble ----
    teacher_ckpts = cfg["teacher_checkpoints"]
    teacher_kwargs = cfg.get("teacher_model_kwargs", cfg.get("model_kwargs", {}))
    # Remove V2-specific keys for teacher (V1)
    teacher_kw_clean = {k: v for k, v in teacher_kwargs.items()
                        if k not in ("use_gated_pooling", "use_env_enrichment",
                                     "use_prenorm_local")}
    teacher = EnsembleTeacher(
        [str(ROOT / p) for p in teacher_ckpts],
        device, teacher_kw_clean,
    )

    # ---- Student V2 ----
    student_kwargs = cfg.get("model_kwargs", {})
    student = CrystalTransformerV2(**student_kwargs).to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)

    # Stochastic depth
    drop_path_rate = cfg.get("drop_path_rate", 0.0)
    if drop_path_rate > 0:
        apply_stochastic_depth(student, drop_path_rate)

    # ---- Optimizer ----
    optim_kwargs = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=optim_kwargs.get("lr", 5e-4),
        weight_decay=optim_kwargs.get("weight_decay", 1e-4),
    )
    epochs = cfg.get("epochs", 150)
    warmup_epochs = cfg.get("warmup_epochs", 10)
    sched_kwargs = cfg.get("scheduler", {"eta_min": 1e-6})
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs,
        eta_min=sched_kwargs.get("eta_min", 1e-6),
    )
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = main_scheduler

    # Loss weights
    task_weight = cfg.get("task_weight", 1.0)
    distill_weight = cfg.get("distill_weight", 0.5)
    label_noise_std = cfg.get("label_noise_std", 0.03)
    grad_clip = cfg.get("grad_clip", 5.0)

    criterion = nn.L1Loss()

    # SWA
    use_swa = cfg.get("use_swa", False)
    swa_start_epoch = cfg.get("swa_start_epoch", max(1, epochs - 30))
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = AveragedModel(student)
        swa_lr = cfg.get("swa_lr", 1e-4)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=5)

    history = []
    best_val_mae = float("inf")

    with open(log_path, "w") as logf:
        msg = (
            f"Config: {json.dumps(cfg, ensure_ascii=False)}\n"
            f"Device: {device}\n"
            f"Student: CrystalTransformerV2 | params={n_params / 1e6:.3f}M\n"
            f"Teacher: {len(teacher.models)}-model ensemble\n"
            f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}\n"
            f"Weights: task={task_weight} distill={distill_weight}\n"
            f"Target stats: mean={normalizer.mean:.4f} std={normalizer.std:.4f}\n"
        )
        print(msg)
        logf.write(msg)
        logf.flush()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            student.train()
            train_loss, train_abs, n_seen = 0.0, 0.0, 0
            distill_loss_sum = 0.0

            for batch in train_loader:
                batch = move_batch(batch, device)
                target = batch["target"]

                # Label noise
                if label_noise_std > 0:
                    noise = torch.randn_like(target) * label_noise_std
                    target_noisy = target + noise
                else:
                    target_noisy = target
                target_norm = normalizer.norm(target_noisy)

                # Teacher soft targets (in real eV space)
                teacher_pred_raw = teacher(batch)
                teacher_pred_norm = normalizer.norm(teacher_pred_raw)

                # Student forward
                student_pred_norm = student(batch)

                # Losses
                task_loss = criterion(student_pred_norm, target_norm)
                distill_loss = criterion(student_pred_norm, teacher_pred_norm)

                total_loss = (task_weight * task_loss
                              + distill_weight * distill_loss)

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                optimizer.step()

                with torch.no_grad():
                    preds = normalizer.denorm(student_pred_norm)
                    abs_err = (preds - batch["target"]).abs().sum().item()
                bs = int(target.numel())
                train_loss += task_loss.item() * bs
                train_abs += abs_err
                distill_loss_sum += distill_loss.item() * bs
                n_seen += bs

            train_mae = train_abs / max(n_seen, 1)
            avg_distill = distill_loss_sum / max(n_seen, 1)

            # SWA
            in_swa = use_swa and epoch >= swa_start_epoch
            if in_swa:
                swa_model.update_parameters(student)
                swa_scheduler.step()
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
                val_metrics = evaluate(student, val_loader, normalizer, device, swa_model)
            else:
                val_metrics = evaluate(student, val_loader, normalizer, device)
                scheduler.step()

            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch, "train_mae": train_mae,
                "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"],
                "distill_loss": avg_distill,
                "lr": lr_now, "time": dt,
            }
            history.append(record)

            line = (
                f"Ep {epoch:3d}/{epochs} | "
                f"train_mae {train_mae:.4f} | "
                f"val_mae {val_metrics['mae']:.4f} | "
                f"distill {avg_distill:.4f} | "
                f"lr {lr_now:.2e} | "
                f"{'SWA ' if in_swa else ''}"
                f"{dt:.1f}s"
            )
            print(line)
            logf.write(line + "\n")
            logf.flush()

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                save_dict = {
                    "model_state": student.state_dict(),
                    "normalizer_mean": normalizer.mean,
                    "normalizer_std": normalizer.std,
                    "epoch": epoch,
                    "val_mae": val_metrics["mae"],
                    "config": cfg,
                }
                if swa_model is not None:
                    save_dict["swa_model_state"] = swa_model.state_dict()
                torch.save(save_dict, ckpt_path)
                print(f"  >> New best val MAE: {best_val_mae:.4f} (saved)")
                logf.write(f"  >> New best val MAE: {best_val_mae:.4f}\n")

        # Final test
        best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(best_ckpt["model_state"])
        if swa_model is not None and "swa_model_state" in best_ckpt:
            swa_model.load_state_dict(best_ckpt["swa_model_state"])
        test_metrics = evaluate(
            student, test_loader, normalizer, device,
            swa_model if use_swa else None,
        )
        summary = {
            "best_val_mae": best_val_mae,
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "n_params": n_params,
            "best_epoch": best_ckpt["epoch"],
            "history": history,
        }
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)

        final_msg = (
            f"\nDone. Best val MAE: {best_val_mae:.4f} | "
            f"Test MAE: {test_metrics['mae']:.4f} | "
            f"Test RMSE: {test_metrics['rmse']:.4f}\n"
        )
        print(final_msg)
        logf.write(final_msg)


if __name__ == "__main__":
    main()
