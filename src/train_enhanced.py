"""Enhanced training loop with P0+P1 improvements.

P0-2: Host-balanced sampling
P1-1: Auxiliary defect-classification head
P1-3: SWA, Stochastic Depth (DropPath), Label Smoothing (noise)
Online augmentation: random rotation + perturbation + strain per epoch
Adversarial training: FGSM on input features for robust representations

Usage:
  python -m src.train_enhanced --config configs/enhanced_online_adv.yaml
"""
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
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.sampler import HostBalancedSampler
from src.augment_online import OnlineAugTransform, OnlineAugDataset, adversarial_perturbation
from src.models import (
    CrystalTransformer,
    DefectAwareTransformer,
    PeriodicCrystalTransformer,
    DualStreamPeriodicTransformer,
    compute_invariance_loss,
)

MODEL_REGISTRY = {
    "baseline": CrystalTransformer,
    "improved": DefectAwareTransformer,
    "periodic": PeriodicCrystalTransformer,
    "dualstream": DualStreamPeriodicTransformer,
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


def move_batch(batch, device):
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


# ---- Mixup in feature space (same-host constraint relaxed to batch-level) ----
def mixup_batch(batch, alpha=0.2):
    """Feature-space Mixup: interpolate embeddings and targets within a batch."""
    if alpha <= 0:
        return batch, None
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # keep lambda >= 0.5 for stability
    bs = batch["x"].size(0)
    perm = torch.randperm(bs)
    mixed_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
            if k == "target":
                mixed_batch[k] = lam * v + (1.0 - lam) * v[perm]
            elif k in ("x", "positions"):
                mixed_batch[k] = lam * v + (1.0 - lam) * v[perm]
            elif k == "dist_matrix":
                mixed_batch[k] = lam * v + (1.0 - lam) * v[perm]
            else:
                mixed_batch[k] = v
        else:
            mixed_batch[k] = v
    return mixed_batch, lam


# ---- Auxiliary defect classification head ----
class DefectClassifierHead(nn.Module):
    """Per-atom binary classifier: is this atom the defect site?"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h: (B, N, C), mask: (B, N) bool. Returns logits (B, N)."""
        return self.head(h).squeeze(-1) * mask.float()


# ---- Stochastic Depth wrapper ----
class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep)
        return x * random_tensor / keep


def apply_stochastic_depth(model, drop_rate: float = 0.1):
    """Wrap residual connections in GeometricTransformerBlock with DropPath."""
    n_blocks = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in ("GeometricTransformerBlock", "PeriodicGeometricBlock"):
            n_blocks += 1
    if n_blocks == 0:
        return
    # linearly increasing drop rate
    idx = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in ("GeometricTransformerBlock", "PeriodicGeometricBlock"):
            rate = drop_rate * (idx + 1) / n_blocks
            module._droppath = DropPath(rate)
            idx += 1


def evaluate(model, loader, normalizer, device, swa_model=None):
    eval_model = swa_model if swa_model is not None else model
    eval_model.eval()
    abs_err, sq_err, n = 0.0, 0.0, 0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            target = batch["target"]
            preds_norm = eval_model(batch)
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
        "mae": mae, "rmse": rmse,
        "preds": torch.cat(preds_all).numpy() if preds_all else np.array([]),
        "targets": torch.cat(targets_all).numpy() if targets_all else np.array([]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
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

    # Data split uses fixed split_seed (default 42) for reproducibility across ensemble members.
    # Model init uses cfg["seed"] which may differ per run.
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

    # P0-2: Host-balanced sampling
    use_balanced = cfg.get("host_balanced", False)
    if use_balanced:
        # Sampler always indexes into the full dataset by original indices
        sampler = HostBalancedSampler(
            dataset,
            subset_indices=train_set.indices,
            samples_per_host=cfg.get("samples_per_host", 50),
            seed=cfg.get("seed", 42),
        )
        # Wrap full dataset with online aug (sampler restricts to train indices)
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

    # Normalizer
    targets = torch.tensor(
        [dataset.data[i]["target"] for i in train_set.indices], dtype=torch.float32
    )
    normalizer = Normalizer(targets)

    # ---- Model ----
    model_cls = MODEL_REGISTRY[cfg["model"]]
    model = model_cls(**cfg.get("model_kwargs", {})).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # P1-3: Stochastic Depth
    drop_path_rate = cfg.get("drop_path_rate", 0.0)
    if drop_path_rate > 0:
        apply_stochastic_depth(model, drop_path_rate)

    # P1-1: Auxiliary defect classifier
    use_aux_defect = cfg.get("aux_defect_weight", 0.0) > 0
    aux_defect_head = None
    if use_aux_defect:
        hidden_dim = cfg.get("model_kwargs", {}).get("hidden_dim", 128)
        aux_defect_head = DefectClassifierHead(hidden_dim).to(device)
        n_params += sum(p.numel() for p in aux_defect_head.parameters())

    # P1-2: Load pretrained element embeddings + local layers
    pretrained_embed_path = cfg.get("pretrained_embed", None)
    if pretrained_embed_path:
        embed_ckpt = torch.load(ROOT / pretrained_embed_path, map_location=device, weights_only=False)
        loaded = []
        if "embed_weight" in embed_ckpt and hasattr(model, "embed"):
            with torch.no_grad():
                pre_w = embed_ckpt["embed_weight"]
                cur_w = model.embed.weight
                if pre_w.shape == cur_w.shape:
                    cur_w.copy_(pre_w)
                else:
                    # UAE widens input dim: copy pretrained cols into first slice
                    n_pre = pre_w.shape[1]
                    cur_w[:, :n_pre].copy_(pre_w)
                if "embed_bias" in embed_ckpt:
                    model.embed.bias.copy_(embed_ckpt["embed_bias"])
            loaded.append("embed")
        if "local_layers" in embed_ckpt and hasattr(model, "local_layers"):
            model.local_layers.load_state_dict(embed_ckpt["local_layers"], strict=False)
            loaded.append("local_layers")
        print(f"Loaded pretrained {'+'.join(loaded)} from {pretrained_embed_path}")

    # ---- Optimizer ----
    all_params = list(model.parameters())
    if aux_defect_head is not None:
        all_params += list(aux_defect_head.parameters())
    optim_kwargs = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        all_params,
        lr=optim_kwargs.get("lr", 3e-4),
        weight_decay=optim_kwargs.get("weight_decay", 1e-4),
    )
    sched_type = cfg.get("scheduler_type", "plateau")
    sched_kwargs = cfg.get("scheduler", {"factor": 0.5, "patience": 5})
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("epochs", 50),
            eta_min=sched_kwargs.get("eta_min", 1e-6),
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", **sched_kwargs
        )

    loss_name = cfg.get("loss", "mse")
    if loss_name == "huber":
        criterion = nn.HuberLoss(delta=cfg.get("huber_delta", 1.0))
    elif loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError(loss_name)

    epochs = cfg.get("epochs", 50)
    grad_clip = cfg.get("grad_clip", 5.0)

    # P0-2: Mixup (disabled for graph data — see feedback)
    mixup_alpha = cfg.get("mixup_alpha", 0.0)

    # Adversarial training
    adv_eps = cfg.get("adv_eps", 0.0)
    adv_weight = cfg.get("adv_weight", 0.0)
    use_adv = adv_eps > 0 and adv_weight > 0

    # P1-3: Label smoothing (Gaussian noise on targets)
    label_noise_std = cfg.get("label_noise_std", 0.0)

    # P1-3: SWA
    use_swa = cfg.get("use_swa", False)
    swa_start_epoch = cfg.get("swa_start_epoch", max(1, epochs - 10))
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = AveragedModel(model)
        swa_lr = cfg.get("swa_lr", 1e-4)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=5)

    # Aux loss weights
    aux_defect_w = cfg.get("aux_defect_weight", 0.0)

    history = []
    best_val_mae = float("inf")

    with open(log_path, "w") as logf:
        msg = (
            f"Config: {json.dumps(cfg, ensure_ascii=False)}\n"
            f"Device: {device}\n"
            f"Model: {model_cls.__name__} | params={n_params / 1e6:.3f}M\n"
            f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}\n"
            f"Enhancements: balanced={use_balanced} online_aug={use_online_aug} "
            f"adv={use_adv}(eps={adv_eps},w={adv_weight}) "
            f"droppath={drop_path_rate} label_noise={label_noise_std} "
            f"swa={use_swa}(ep{swa_start_epoch}) aux_defect={aux_defect_w}\n"
            f"Target stats: mean={normalizer.mean:.4f} std={normalizer.std:.4f}\n"
        )
        print(msg)
        logf.write(msg)
        logf.flush()

        global_step = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            if aux_defect_head is not None:
                aux_defect_head.train()
            train_loss, train_abs, n_seen = 0.0, 0.0, 0
            aux_defect_loss_sum = 0.0

            for batch in train_loader:
                batch = move_batch(batch, device)

                # P0-2: Mixup
                if mixup_alpha > 0 and random.random() < 0.5:
                    batch, lam = mixup_batch(batch, alpha=mixup_alpha)
                else:
                    lam = None

                target = batch["target"]

                # P1-3: Label noise
                if label_noise_std > 0:
                    noise = torch.randn_like(target) * label_noise_std
                    target_noisy = target + noise
                else:
                    target_noisy = target

                target_norm = normalizer.norm(target_noisy)

                # Adversarial training path
                adv_loss_val = 0.0
                if use_adv:
                    preds_norm, task_loss, adv_loss = adversarial_perturbation(
                        model, batch, criterion, target_norm, eps=adv_eps,
                    )
                    total_loss = task_loss + adv_weight * adv_loss
                    adv_loss_val = adv_loss.item()
                else:
                    preds_norm = model(batch)
                    task_loss = criterion(preds_norm, target_norm)
                    total_loss = task_loss

                # P1-1: Aux defect classification
                aux_loss = torch.tensor(0.0, device=device)
                if aux_defect_head is not None and aux_defect_w > 0:
                    defect_target = batch["defect_mask"].float()
                    mask = batch["atom_mask"]
                    with torch.no_grad():
                        _x_aux = batch["x"]
                        if getattr(model, "ct_uae_table", None) is not None:
                            z = batch.get("atomic_numbers")
                            if z is not None:
                                z_c = z.clamp(0, model.ct_uae_table.shape[0] - 1)
                                _x_aux = torch.cat([_x_aux, model.ct_uae_table[z_c]], dim=-1)
                        h_embed = model.embed(_x_aux)
                    logits = aux_defect_head(h_embed, mask)
                    aux_loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, defect_target, weight=mask.float(),
                        reduction="sum"
                    ) / mask.float().sum().clamp(min=1.0)
                    total_loss = total_loss + aux_defect_w * aux_loss

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                optimizer.step()

                with torch.no_grad():
                    preds = normalizer.denorm(preds_norm)
                    abs_err = (preds - batch["target"]).abs().sum().item()
                bs = int(batch["target"].numel())
                train_loss += task_loss.item() * bs
                train_abs += abs_err
                aux_defect_loss_sum += aux_loss.item() * bs
                n_seen += bs
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    break

            train_mae = train_abs / max(n_seen, 1)

            # P1-3: SWA update
            in_swa = use_swa and epoch >= swa_start_epoch
            if in_swa:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
                val_metrics = evaluate(model, val_loader, normalizer, device, swa_model)
            else:
                val_metrics = evaluate(model, val_loader, normalizer, device)
                if sched_type == "cosine":
                    scheduler.step()
                else:
                    scheduler.step(val_metrics["mae"])

            improved = val_metrics["mae"] < best_val_mae
            if improved:
                best_val_mae = val_metrics["mae"]
                save_model = swa_model.module if in_swa else model
                save_dict = {
                    "model": save_model.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                }
                if aux_defect_head is not None:
                    save_dict["aux_defect_head"] = aux_defect_head.state_dict()
                torch.save(save_dict, ckpt_path)

            dt = time.time() - t0
            aux_str = f"aux_def {aux_defect_loss_sum / max(n_seen, 1):.4f} | " if aux_defect_w > 0 else ""
            swa_str = " [SWA]" if in_swa else ""
            row = {
                "epoch": epoch, "train_mae": train_mae,
                "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"],
                "lr": optimizer.param_groups[0]["lr"], "time_sec": dt,
                "best": improved,
            }
            history.append(row)
            line = (
                f"Epoch {epoch:02d}/{epochs} | train MAE {train_mae:.4f} | "
                f"{aux_str}"
                f"val MAE {val_metrics['mae']:.4f} RMSE {val_metrics['rmse']:.4f} | "
                f"lr {row['lr']:.2e} | {dt:.1f}s {'*' if improved else ''}{swa_str}"
            )
            print(line)
            logf.write(line + "\n")
            logf.flush()
            if args.max_steps and global_step >= args.max_steps:
                break

        # Final test
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, test_loader, normalizer, device)
        final_line = f"\n[Final] Test MAE {test_metrics['mae']:.4f} | RMSE {test_metrics['rmse']:.4f}\n"
        print(final_line)
        logf.write(final_line)

    summary = {
        "config": cfg, "n_params": n_params, "history": history,
        "best_val_mae": best_val_mae,
        "test_mae": test_metrics["mae"], "test_rmse": test_metrics["rmse"],
    }
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(out_dir / "test_predictions.npz",
             preds=test_metrics["preds"], targets=test_metrics["targets"])


if __name__ == "__main__":
    main()
