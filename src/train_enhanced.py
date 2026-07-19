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
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

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
from src.splits import load_split
from src.sampler import HostBalancedSampler
from src.prm_assets import load_pretrained_initialization, verify_training_assets
from src.prm_provenance import config_sha256
from src.augment_online import OnlineAugTransform, OnlineAugDataset, adversarial_perturbation
from src.models import (
    CrystalTransformer,
    DefectAwareTransformer,
    PeriodicCrystalTransformer,
    DualStreamPeriodicTransformer,
    compute_invariance_loss,
    CrystalTransformerV2,
)

MODEL_REGISTRY = {
    "baseline": CrystalTransformer,
    "improved": DefectAwareTransformer,
    "periodic": PeriodicCrystalTransformer,
    "dualstream": DualStreamPeriodicTransformer,
    "v2": CrystalTransformerV2,
}


def resolve_path(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def resolve_runtime_assets(
    controlled_config: Dict[str, Any],
    ct_uae_override: str | None = None,
    pretrained_override: str | None = None,
) -> Dict[str, Any]:
    runtime = deepcopy(controlled_config)
    ct_uae_path = runtime.get("model_kwargs", {}).get("ct_uae_path")
    if ct_uae_override:
        runtime.setdefault("model_kwargs", {})["ct_uae_path"] = ct_uae_override
    elif ct_uae_path:
        runtime["model_kwargs"]["ct_uae_path"] = str(resolve_path(ct_uae_path))
    if pretrained_override:
        runtime["pretrained_embed"] = pretrained_override
    elif runtime.get("pretrained_embed"):
        runtime["pretrained_embed"] = str(resolve_path(runtime["pretrained_embed"]))
    return runtime


def git_snapshot() -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_snapshot(device: torch.device) -> Dict[str, Any]:
    cuda_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": cuda_name,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class Normalizer:
    def __init__(self, tensor: torch.Tensor, transform: str = "none") -> None:
        self.transform = transform
        if transform == "log":
            tensor = torch.sign(tensor) * torch.log1p(torch.abs(tensor))
        self.mean = float(tensor.mean().item())
        self.std = float(tensor.std().item()) + 1e-6

    def _fwd(self, t: torch.Tensor) -> torch.Tensor:
        if self.transform == "log":
            return torch.sign(t) * torch.log1p(torch.abs(t))
        return t

    def _inv(self, t: torch.Tensor) -> torch.Tensor:
        if self.transform == "log":
            return torch.sign(t) * torch.expm1(torch.abs(t))
        return t

    def norm(self, t: torch.Tensor) -> torch.Tensor:
        return (self._fwd(t) - self.mean) / self.std

    def denorm(self, t: torch.Tensor) -> torch.Tensor:
        return self._inv(t * self.std + self.mean)

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std, "transform": self.transform}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def move_batch(batch, device):
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


# ---- Label Distribution Smoothing (Yang et al., ICML 2021) ----
# "Delving into Deep Imbalanced Regression": reweight samples by
# inverse smoothed label density so that rare energy ranges (esp.
# the [7,25) eV bottleneck) get proportionally more gradient signal.
def _gaussian_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing (no scipy dependency)."""
    width = int(4 * sigma + 0.5)
    x = np.arange(-width, width + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def compute_lds_weights(
    targets: torch.Tensor,
    n_bins: int = 100,
    sigma: float = 2.0,
    reweight: str = "sqrt_inv",
    clip_max: float = 10.0,
):
    """Build a target-value → weight lookup for LDS reweighting.

    Args:
        targets: 1-D tensor of training targets (original scale).
        n_bins: histogram resolution.
        sigma: Gaussian kernel width for smoothing.
        reweight: "inv" for 1/density, "sqrt_inv" for 1/sqrt(density).
        clip_max: cap on per-sample weight to avoid outlier dominance.

    Returns:
        weight_fn(target_batch) → weight tensor, mean ≈ 1.
    """
    t_np = targets.numpy().astype(np.float64)
    t_min, t_max = float(t_np.min()) - 0.01, float(t_np.max()) + 0.01
    counts, bin_edges = np.histogram(t_np, bins=n_bins,
                                      range=(t_min, t_max))
    smoothed = _gaussian_smooth_1d(counts.astype(np.float64), sigma)
    smoothed = np.clip(smoothed, 1.0, None)

    if reweight == "inv":
        raw_w = 1.0 / smoothed
    elif reweight == "sqrt_inv":
        raw_w = 1.0 / np.sqrt(smoothed)
    else:
        raise ValueError(f"Unknown LDS reweight mode: {reweight}")

    raw_w = np.clip(raw_w, None, clip_max * raw_w.mean())
    raw_w = raw_w / raw_w.mean()  # normalise bins → mean = 1

    bin_weights_np = raw_w
    bin_width = (t_max - t_min) / n_bins

    # Sample-level normalisation: ensure E[w_i] ≈ 1 over training set,
    # so LDS does not change overall gradient scale (only rebalances).
    sample_idx = ((t_np - t_min) / bin_width).astype(int)
    sample_idx = np.clip(sample_idx, 0, n_bins - 1)
    sample_weights = bin_weights_np[sample_idx]
    sample_mean = float(sample_weights.mean())
    bin_weights_np = bin_weights_np / max(sample_mean, 1e-8)

    bin_weights = torch.tensor(bin_weights_np, dtype=torch.float32)

    def weight_fn(target_values: torch.Tensor) -> torch.Tensor:
        idx = ((target_values.float() - t_min) / bin_width).long()
        idx = idx.clamp(0, n_bins - 1)
        return bin_weights[idx.cpu()].to(target_values.device)

    return weight_fn


# ---- Rank-N-Contrast for regression (Zha et al., NeurIPS 2023) ----
# "Supervised Contrastive Learning for Pre-trained Language Model
# Fine-tuning" and "Rank-N-Contrast: Learning Continuous Representations
# for Regression" — adapted for graph-level regression.
# The loss encourages the feature space to preserve target ordering:
# samples with similar Ef should cluster, dissimilar ones should separate.
def rnc_loss(
    features: torch.Tensor,        # (B, C) — L2-normalised graph reps
    targets: torch.Tensor,         # (B,)   — regression targets
    temperature: float = 0.1,
    label_diff_sigma: float = 1.0, # bandwidth for target similarity kernel
) -> torch.Tensor:
    """Rank-N-Contrast loss for continuous regression.

    For each anchor i, positive weight w_ij = exp(-|y_i - y_j|²/σ²).
    L_i = -log( Σ_j w_ij·exp(z_i·z_j/τ) / Σ_k exp(z_i·z_k/τ) )

    This pushes apart samples with very different targets and pulls
    together those with similar targets — a continuous generalisation
    of SupCon (Khosla et al., NeurIPS 2020).
    """
    B = features.size(0)
    if B < 4:
        return torch.tensor(0.0, device=features.device)

    # Cosine similarity matrix
    sim = torch.mm(features, features.t()) / temperature  # (B, B)

    # Target similarity kernel (soft positives)
    target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)  # (B, B)
    weights = torch.exp(-target_diff.pow(2) / (2 * label_diff_sigma ** 2))

    # Mask out self-similarities
    mask_self = torch.eye(B, device=features.device).bool()
    weights = weights.masked_fill(mask_self, 0.0)

    # Numerical stability: subtract max from sim
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim)
    exp_sim = exp_sim.masked_fill(mask_self, 0.0)

    # Weighted positive similarities
    pos = (weights * exp_sim).sum(dim=1)  # (B,)
    neg = exp_sim.sum(dim=1)              # (B,)

    loss = -torch.log(pos / neg.clamp(min=1e-8) + 1e-8).mean()
    return loss


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
class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy: θ_ema = decay · θ_ema + (1 − decay) · θ_model.
    The EMA model typically generalises better than the raw checkpoint,
    giving 0.5-2 % relative MAE reduction on held-out data.

    References:
        Polyak & Juditsky, SIAM J. Control Optim. 1992.
        Tarvainen & Valpola, NeurIPS 2017 ("Mean Teacher").
        Used in Uni-Mol, GemNet-OC, DeiT and most SoTA molecular models.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup: dict = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].lerp_(p.data, 1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Swap model params with EMA shadow (call before eval)."""
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module) -> None:
        """Restore original params (call after eval)."""
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


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
    preds_all, targets_all, indices_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            target = batch["target"]
            model_out = eval_model(batch)
            # Handle uncertainty output: (pred, log_var) tuple
            preds_norm = model_out[0] if isinstance(model_out, tuple) else model_out
            preds = normalizer.denorm(preds_norm)
            err = preds - target
            abs_err += err.abs().sum().item()
            sq_err += err.pow(2).sum().item()
            n += target.numel()
            preds_all.append(preds.cpu())
            targets_all.append(target.cpu())
            indices_all.append(batch["sample_index"].detach().cpu())
    mae = abs_err / max(n, 1)
    rmse = math.sqrt(sq_err / max(n, 1))
    preds_np = torch.cat(preds_all).numpy() if preds_all else np.array([])
    targets_np = torch.cat(targets_all).numpy() if targets_all else np.array([])
    indices_np = torch.cat(indices_all).numpy() if indices_all else np.array([], dtype=int)
    bias = float(np.mean(preds_np - targets_np)) if len(preds_np) else float("nan")
    if len(preds_np) > 1 and np.std(preds_np) > 0 and np.std(targets_np) > 0:
        pearson = float(np.corrcoef(preds_np, targets_np)[0, 1])
        pred_rank = np.argsort(np.argsort(preds_np, kind="stable"), kind="stable")
        target_rank = np.argsort(np.argsort(targets_np, kind="stable"), kind="stable")
        spearman = float(np.corrcoef(pred_rank, target_rank)[0, 1])
    else:
        pearson = float("nan")
        spearman = float("nan")
    target_ss = float(np.sum((targets_np - np.mean(targets_np)) ** 2)) if len(targets_np) else 0.0
    residual_ss = float(np.sum((preds_np - targets_np) ** 2)) if len(preds_np) else 0.0
    r2 = 1.0 - residual_ss / target_ss if target_ss > 0 else float("nan")
    return {
        "mae": mae, "rmse": rmse,
        "bias": bias, "pearson": pearson, "spearman": spearman, "r2": r2,
        "preds": preds_np, "targets": targets_np, "indices": indices_np,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest.pt checkpoint")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["output_dir"] = cfg["output_dir"] + f"_s{args.seed}"

    if args.data_path is not None:
        cfg["data_path"] = args.data_path
    if args.split_path is not None:
        cfg["split_path"] = args.split_path
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    # Preserve the controlled experiment config before resolving machine-local
    # asset paths. Collectors compare this copy with the versioned YAML.
    controlled_cfg = deepcopy(cfg)
    cfg = resolve_runtime_assets(
        controlled_cfg,
        ct_uae_override=os.environ.get("PRM_CT_UAE_PATH"),
        pretrained_override=os.environ.get("PRM_PRETRAINED_EMBED"),
    )
    asset_records = verify_training_assets(cfg)

    split_seed = cfg.get("split_seed", 42)

    # ── Limit CPU threads to avoid overloading shared servers ────────
    # Without this, each process spawns ~200 OMP/MKL threads on a 256-core
    # machine, causing severe contention when multiple runs share the node.
    n_workers = cfg.get("num_workers", 4)
    cpu_threads = cfg.get("cpu_threads", 8)
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    torch.set_num_threads(cpu_threads)

    configured_output = Path(cfg["output_dir"]).expanduser()
    results_root = os.environ.get("PRM_RESULTS_ROOT")
    if configured_output.is_absolute():
        out_dir = configured_output
    elif results_root:
        out_dir = Path(results_root).expanduser() / configured_output
    else:
        out_dir = ROOT / configured_output
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
    asph_path = cfg.get("asph_features_path")
    if asph_path:
        asph_path = ROOT / asph_path
    soft_labels_path = cfg.get("soft_labels_path")
    if soft_labels_path:
        soft_labels_path = ROOT / soft_labels_path
    data_value = os.environ.get("PRM_DATA_PATH", cfg["data_path"])
    data_path = resolve_path(data_value)
    dataset = CrystalGraphDataset(data_path,
                                   asph_features_path=asph_path,
                                   soft_labels_path=soft_labels_path)
    split_payload = None
    if cfg.get("split_path"):
        split_path = resolve_path(cfg["split_path"])
        split_payload = load_split(
            split_path,
            len(dataset),
            expected_data_sha256=cfg.get("data_sha256"),
        )
        train_set = Subset(dataset, split_payload["train"])
        val_set = Subset(dataset, split_payload["val"])
        test_set = Subset(dataset, split_payload["test"])
        calibration_set = (
            Subset(dataset, split_payload["calibration"])
            if "calibration" in split_payload else None
        )
        split_id = split_payload["split_id"]
    else:
        split_path = None
        train_set, val_set, test_set = make_splits(
            dataset,
            train_ratio=cfg.get("train_ratio", 0.8),
            val_ratio=cfg.get("val_ratio", 0.1),
            seed=split_seed,
        )
        calibration_set = None
        split_id = f"legacy_random_s{split_seed}"

    split_counts = {
        "train": len(train_set), "val": len(val_set), "test": len(test_set),
    }
    if calibration_set is not None:
        split_counts["calibration"] = len(calibration_set)

    run_manifest = {
        "schema_version": "prm_run_manifest_v1",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "git": git_snapshot(),
        "config": controlled_cfg,
        "config_sha256": config_sha256(controlled_cfg),
        "runtime_config": cfg,
        "runtime_config_sha256": config_sha256(cfg),
        "execution": {
            "max_steps": int(args.max_steps),
            "resume_requested": bool(args.resume),
        },
        "assets": asset_records,
        "data": {
            "path": str(data_path),
            "size_bytes": data_path.stat().st_size,
            "data_sha256": split_payload.get("data_sha256") if split_payload else cfg.get("data_sha256"),
        },
        "split": {
            "split_id": split_id,
            "path": str(split_path) if split_path else None,
            "sha256": file_sha256(split_path) if split_path else None,
            "counts": split_counts,
        },
        "seed": cfg.get("seed", 42),
        "environment": environment_snapshot(device),
    }
    write_json(out_dir / "run_manifest.json", run_manifest)
    split_arrays = {
        "train": np.asarray(train_set.indices, dtype=np.int64),
        "val": np.asarray(val_set.indices, dtype=np.int64),
        "test": np.asarray(test_set.indices, dtype=np.int64),
        "split_id": np.asarray(split_id),
    }
    if calibration_set is not None:
        split_arrays["calibration"] = np.asarray(
            calibration_set.indices, dtype=np.int64,
        )
    np.savez(out_dir / "split_indices.npz", **split_arrays)

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
    sampler = None
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
            num_workers=n_workers, pin_memory=True,
            persistent_workers=(n_workers > 0),
        )
    else:
        loader_ds = OnlineAugDataset(train_set, transform=aug_transform) if use_online_aug else train_set
        train_loader = DataLoader(
            loader_ds, batch_size=cfg.get("batch_size", 64),
            shuffle=True, collate_fn=collate_fn,
            num_workers=n_workers, pin_memory=True,
            persistent_workers=(n_workers > 0),
        )
    val_loader = DataLoader(val_set, batch_size=cfg.get("batch_size", 64),
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=n_workers, pin_memory=True,
                            persistent_workers=(n_workers > 0))
    test_loader = DataLoader(test_set, batch_size=cfg.get("batch_size", 64),
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=n_workers, pin_memory=True,
                             persistent_workers=(n_workers > 0))
    calibration_loader = (
        DataLoader(
            calibration_set, batch_size=cfg.get("batch_size", 64),
            shuffle=False, collate_fn=collate_fn, num_workers=n_workers,
            pin_memory=True, persistent_workers=(n_workers > 0),
        )
        if calibration_set is not None else None
    )

    # Normalizer
    targets = torch.tensor(
        [dataset.data[i]["target"] for i in train_set.indices], dtype=torch.float32
    )
    target_transform = cfg.get("target_transform", "none")
    normalizer = Normalizer(targets, transform=target_transform)

    # LDS reweighting (Yang et al., ICML 2021)
    use_lds = cfg.get("use_lds", False)
    lds_weight_fn = None
    if use_lds:
        lds_weight_fn = compute_lds_weights(
            targets,
            n_bins=cfg.get("lds_bins", 100),
            sigma=cfg.get("lds_sigma", 2.0),
            reweight=cfg.get("lds_reweight", "sqrt_inv"),
            clip_max=cfg.get("lds_clip", 10.0),
        )
        # Print distribution info
        weights_all = lds_weight_fn(targets)
        print("LDS weights: min=%.2f max=%.2f mean=%.2f std=%.2f"
              % (weights_all.min(), weights_all.max(), weights_all.mean(), weights_all.std()))

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

    # Load the exact shared input/local initialization and record what matched.
    pretrained_embed_path = cfg.get("pretrained_embed", None)
    if pretrained_embed_path:
        pretraining_report = load_pretrained_initialization(
            model, resolve_path(pretrained_embed_path)
        )
        expected_checkpoint_hash = asset_records.get("pretrained_embed", {}).get("sha256")
        if pretraining_report["checkpoint_sha256"] != expected_checkpoint_hash:
            raise RuntimeError("pretraining report does not match the verified checkpoint")
        run_manifest["pretraining"] = pretraining_report
        write_json(out_dir / "run_manifest.json", run_manifest)
        local_report = pretraining_report["local_layers"]
        print(
            "Loaded pretrained input slice and "
            f"{local_report['loaded_tensor_count']} local tensors from "
            f"{pretrained_embed_path}; "
            f"{len(local_report['seeded_model_keys'])} model tensors remain seeded"
        )

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
    warmup_epochs = cfg.get("warmup_epochs", 0)
    if sched_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.get("epochs", 50) - warmup_epochs,
            eta_min=sched_kwargs.get("eta_min", 1e-6),
        )
    else:
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", **sched_kwargs
        )
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs])
    else:
        scheduler = main_scheduler

    loss_name = cfg.get("loss", "mse")
    if loss_name == "huber":
        criterion = nn.HuberLoss(delta=cfg.get("huber_delta", 1.0))
    elif loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "mae":
        criterion = nn.L1Loss()
    elif loss_name == "focal_mae":
        # Focal MAE: dynamically upweight hard samples based on error magnitude.
        # L = mean( w_i * |e_i| ) where w_i = (|e_i|/mean(|e|))^gamma
        # gamma=0 → standard MAE; gamma=1 → approximately MSE; 0.5 is moderate.
        focal_gamma = cfg.get("focal_gamma", 0.5)
        def _focal_mae(pred, target):
            errors = torch.abs(pred - target)
            with torch.no_grad():
                weights = (errors / errors.mean().clamp(min=1e-6)) ** focal_gamma
                weights = weights / weights.mean()  # normalise so mean weight = 1
            return (weights * errors).mean()
        criterion = _focal_mae
    else:
        raise ValueError(loss_name)

    # Heteroscedastic uncertainty training (Kendall & Gal, NeurIPS 2017)
    # When model predicts (Ef, log_variance), loss becomes:
    #   L = |y - ŷ| * exp(-s) + s   where s = log(σ²)
    # This naturally learns per-sample difficulty and downweights outliers.
    use_heteroscedastic = cfg.get("model_kwargs", {}).get("predict_uncertainty", False)

    # Rank-N-Contrast auxiliary loss (Zha et al., NeurIPS 2023)
    rnc_weight = cfg.get("rnc_weight", 0.0)
    rnc_temp = cfg.get("rnc_temperature", 0.1)
    rnc_sigma = cfg.get("rnc_label_sigma", 1.0)
    use_rnc = rnc_weight > 0

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

    # Knowledge distillation
    distill_alpha = cfg.get("distill_alpha", 1.0)  # 1.0 = no distillation
    use_distill = distill_alpha < 1.0 and soft_labels_path is not None

    # EMA (Exponential Moving Average) — Polyak 1992 / Mean Teacher (NeurIPS 2017)
    use_ema = cfg.get("use_ema", False)
    ema_decay = cfg.get("ema_decay", 0.999)
    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=ema_decay)

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
    start_epoch = 1
    global_step = 0

    # ── Resume from checkpoint ──────────────────────────────────────────
    if args.resume:
        latest_path = out_dir / "latest.pt"
        if latest_path.exists():
            print(f"Loading resume checkpoint from {latest_path} ...")
            resume_ckpt = torch.load(latest_path, map_location=device,
                                     weights_only=False)
            if config_sha256(resume_ckpt.get("config", {})) != config_sha256(cfg):
                raise ValueError("resume checkpoint configuration does not match this run")
            model.load_state_dict(resume_ckpt["model"])
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            best_val_mae = resume_ckpt.get("best_val_mae", float("inf"))
            history = resume_ckpt.get("history", [])
            start_epoch = resume_ckpt["epoch"] + 1
            global_step = int(resume_ckpt.get("global_step", 0))
            if ema is not None and "ema_shadow" in resume_ckpt:
                ema.shadow = resume_ckpt["ema_shadow"]
                ema.backup = resume_ckpt["ema_backup"]
            if aux_defect_head is not None and "aux_defect_head" in resume_ckpt:
                aux_defect_head.load_state_dict(resume_ckpt["aux_defect_head"])
            if use_swa and "swa_model" in resume_ckpt:
                swa_model.load_state_dict(resume_ckpt["swa_model"])
            if "rng_state" not in resume_ckpt:
                raise ValueError("resume checkpoint lacks reproducible RNG state")
            restore_rng_state(resume_ckpt["rng_state"])
            print(f"▶ Resumed from epoch {resume_ckpt['epoch']} "
                  f"(best_val_mae={best_val_mae:.4f}, "
                  f"remaining={epochs - resume_ckpt['epoch']} epochs)")
        else:
            print("WARNING: --resume specified but no latest.pt found, "
                  "starting fresh")

    with open(log_path, "a" if start_epoch > 1 else "w") as logf:
        if start_epoch > 1:
            msg = f"\n{'='*60}\nResumed from epoch {start_epoch - 1}, continuing...\n{'='*60}\n"
        else:
            msg = (
                f"Config: {json.dumps(cfg, ensure_ascii=False)}\n"
                f"Device: {device}\n"
                f"Model: {model_cls.__name__} | params={n_params / 1e6:.3f}M\n"
                f"Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}\n"
                f"Enhancements: balanced={use_balanced} online_aug={use_online_aug} "
                f"adv={use_adv}(eps={adv_eps},w={adv_weight}) "
                f"droppath={drop_path_rate} label_noise={label_noise_std} "
                f"ema={use_ema}(decay={ema_decay}) lds={use_lds} rnc={use_rnc} "
                f"swa={use_swa}(ep{swa_start_epoch}) aux_defect={aux_defect_w} "
                f"distill={use_distill}(alpha={distill_alpha})\n"
                f"Target stats: mean={normalizer.mean:.4f} std={normalizer.std:.4f}\n"
            )
        print(msg)
        logf.write(msg)
        logf.flush()

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            model.train()
            if sampler is not None:
                sampler.set_epoch(epoch - 1)
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
                    # Forward: request hidden features when RnC is active
                    model_out = model(batch, return_hidden=use_rnc)
                    hidden_for_rnc = None
                    if use_rnc and isinstance(model_out, tuple):
                        # return_hidden=True → (pred, pooled_features)
                        preds_norm, hidden_for_rnc = model_out
                    elif use_heteroscedastic and isinstance(model_out, tuple):
                        preds_norm, log_var = model_out
                    else:
                        preds_norm = model_out if not isinstance(model_out, tuple) else model_out[0]

                    # --- Task loss ---
                    if use_heteroscedastic and hidden_for_rnc is None:
                        log_var = log_var.clamp(-6, 6)
                        base_err = torch.abs(preds_norm - target_norm)
                        task_loss = (base_err * torch.exp(-log_var) + log_var).mean()
                    elif lds_weight_fn is not None:
                        with torch.no_grad():
                            lds_w = lds_weight_fn(target)
                        per_sample = torch.abs(preds_norm - target_norm)
                        task_loss = (lds_w * per_sample).mean()
                    else:
                        task_loss = criterion(preds_norm, target_norm)

                    total_loss = task_loss

                    # --- RnC contrastive regularisation ---
                    if use_rnc and hidden_for_rnc is not None:
                        feat_norm = torch.nn.functional.normalize(
                            hidden_for_rnc, dim=-1)
                        rnc_l = rnc_loss(feat_norm, target,
                                         temperature=rnc_temp,
                                         label_diff_sigma=rnc_sigma)
                        total_loss = total_loss + rnc_weight * rnc_l

                # Knowledge distillation loss
                if use_distill and "soft_label" in batch:
                    soft_target_norm = normalizer.norm(batch["soft_label"])
                    distill_loss = criterion(preds_norm, soft_target_norm)
                    total_loss = distill_alpha * task_loss + (1.0 - distill_alpha) * distill_loss

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

                # MoE balance loss (encourages uniform expert utilisation)
                moe_balance_w = cfg.get("model_kwargs", {}).get("moe_balance_weight", 0.0)
                if moe_balance_w > 0 and hasattr(model, "readout") and hasattr(model.readout, "balance_loss"):
                    total_loss = total_loss + moe_balance_w * model.readout.balance_loss

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                optimizer.step()

                # EMA update after each optimizer step
                if ema is not None:
                    ema.update(model)

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

            # Apply EMA shadow for evaluation (when not in SWA phase)
            eval_with_ema = ema is not None and not in_swa
            if eval_with_ema:
                ema.apply_shadow(model)

            if in_swa:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
                val_metrics = evaluate(model, val_loader, normalizer, device, swa_model)
            else:
                val_metrics = evaluate(model, val_loader, normalizer, device)
                if sched_type == "cosine" or warmup_epochs > 0:
                    scheduler.step()
                else:
                    scheduler.step(val_metrics["mae"])

            improved = val_metrics["mae"] < best_val_mae
            if improved:
                best_val_mae = val_metrics["mae"]
                # When EMA is active, model currently holds EMA weights → save those
                save_model = swa_model.module if in_swa else model
                save_dict = {
                    "model": save_model.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "best_val_mae": best_val_mae,
                }
                if aux_defect_head is not None:
                    save_dict["aux_defect_head"] = aux_defect_head.state_dict()
                torch.save(save_dict, ckpt_path)

            # Restore original weights after EMA evaluation
            if eval_with_ema:
                ema.restore(model)

            dt = time.time() - t0
            aux_str = f"aux_def {aux_defect_loss_sum / max(n_seen, 1):.4f} | " if aux_defect_w > 0 else ""
            ema_str = " [EMA]" if eval_with_ema else ""
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
                f"lr {row['lr']:.2e} | {dt:.1f}s {'*' if improved else ''}{ema_str}{swa_str}"
            )
            print(line)
            logf.write(line + "\n")
            logf.flush()

            # Incremental metrics save (every epoch) for monitoring
            _partial = {
                "config": cfg, "n_params": n_params, "history": history,
                "best_val_mae": best_val_mae,
            }
            with open(metrics_path, "w") as _mf:
                json.dump(_partial, _mf, indent=2)

            # Save resume checkpoint (full training state for --resume)
            _resume_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_mae": best_val_mae,
                "history": history,
                "global_step": global_step,
                "normalizer": normalizer.state_dict(),
                "config": cfg,
                "rng_state": capture_rng_state(),
            }
            if ema is not None:
                _resume_state["ema_shadow"] = ema.shadow
                _resume_state["ema_backup"] = ema.backup
            if aux_defect_head is not None:
                _resume_state["aux_defect_head"] = aux_defect_head.state_dict()
            if use_swa and swa_model is not None:
                _resume_state["swa_model"] = swa_model.state_dict()
            torch.save(_resume_state, out_dir / "latest.pt")

            if args.max_steps and global_step >= args.max_steps:
                break

        # Final test
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        test_metrics = evaluate(model, test_loader, normalizer, device)
        val_final = evaluate(model, val_loader, normalizer, device)
        calibration_metrics = (
            evaluate(model, calibration_loader, normalizer, device)
            if calibration_loader is not None else None
        )
        final_line = f"\n[Final] Test MAE {test_metrics['mae']:.4f} | RMSE {test_metrics['rmse']:.4f}\n"
        print(final_line)
        logf.write(final_line)

    summary = {
        "config": cfg, "n_params": n_params, "history": history,
        "best_val_mae": best_val_mae,
        "split_id": split_id,
        "validation": {
            key: val_final[key] for key in ("mae", "rmse", "bias", "pearson", "spearman", "r2")
        },
        "test": {
            key: test_metrics[key] for key in ("mae", "rmse", "bias", "pearson", "spearman", "r2")
        },
        "test_mae": test_metrics["mae"], "test_rmse": test_metrics["rmse"],
    }
    if calibration_metrics is not None:
        summary["calibration"] = {
            key: calibration_metrics[key]
            for key in ("mae", "rmse", "bias", "pearson", "spearman", "r2")
        }
    # Save model internals for interpretability analysis
    if hasattr(model, 'jk_weights'):
        import torch.nn.functional as _F
        jk_w = _F.softmax(model.jk_weights.detach().cpu(), dim=0).numpy()
        summary["jk_weights"] = jk_w.tolist()
        print(f"JK weights: {jk_w}")
    if hasattr(model, 'defect_type_embed') and model.defect_type_embed is not None:
        dt_norms = model.defect_type_embed.weight.detach().cpu().norm(dim=1).numpy()
        summary["defect_type_embed_norms"] = dt_norms.tolist()
        print(f"Defect-type embed norms: {dt_norms}")
    if hasattr(model, 'readout') and hasattr(model.readout, 'balance_loss'):
        summary["moe_balance_loss"] = float(model.readout.balance_loss)

    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(out_dir / "test_predictions.npz",
             schema_version=np.asarray("prm_predictions_v1"),
             split_id=np.asarray(split_id), split=np.asarray("test"),
             indices=test_metrics["indices"],
             preds=test_metrics["preds"], targets=test_metrics["targets"])
    # Save validation predictions for post-hoc calibration
    np.savez(out_dir / "val_predictions.npz",
             schema_version=np.asarray("prm_predictions_v1"),
             split_id=np.asarray(split_id), split=np.asarray("val"),
             indices=val_final["indices"],
             preds=val_final["preds"], targets=val_final["targets"])
    if calibration_metrics is not None:
        np.savez(
            out_dir / "calibration_predictions.npz",
            schema_version=np.asarray("prm_predictions_v1"),
            split_id=np.asarray(split_id), split=np.asarray("calibration"),
            indices=calibration_metrics["indices"],
            preds=calibration_metrics["preds"],
            targets=calibration_metrics["targets"],
        )
    output_paths = {
        "metrics": str(metrics_path),
        "checkpoint": str(ckpt_path),
        "validation_predictions": str(out_dir / "val_predictions.npz"),
        "test_predictions": str(out_dir / "test_predictions.npz"),
    }
    if calibration_metrics is not None:
        output_paths["calibration_predictions"] = str(
            out_dir / "calibration_predictions.npz"
        )
    run_manifest.update(
        {
            "status": "truncated" if args.max_steps else "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "outputs": output_paths,
            "metrics": summary,
        }
    )
    write_json(out_dir / "run_manifest.json", run_manifest)


if __name__ == "__main__":
    main()
