"""Train the periodic SchNet comparator under the frozen PRM protocol."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn
from src.models.schnet_pbc import PBCSchNet
from src.prm_metrics import regression_metrics
from src.prm_provenance import config_sha256
from src.augment_online import OnlineAugDataset, OnlineAugTransform
from src.sampler import HostBalancedSampler
from src.splits import load_split
from src.train_enhanced import (
    Normalizer,
    capture_rng_state,
    file_sha256,
    move_batch,
    make_label_noise_generator,
    resolve_path,
    restore_rng_state,
    set_seed,
)


def git_snapshot() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    return {"commit": commit or None, "dirty": bool(status), "status_porcelain": status.splitlines()}


def evaluate(model, loader, normalizer, device) -> Dict[str, Any]:
    model.eval()
    predictions, targets, indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            prediction = normalizer.denorm(model(batch))
            predictions.append(prediction.cpu().numpy())
            targets.append(batch["target"].cpu().numpy())
            indices.append(batch["sample_index"].cpu().numpy())
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    return {
        **regression_metrics(target, pred),
        "predictions": pred,
        "targets": target,
        "indices": np.concatenate(indices),
    }


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup_epochs: int,
    eta_min: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the shared linear-warmup/cosine schedule used by neural models."""
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must satisfy 0 <= warmup_epochs < epochs")
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=eta_min
    )
    if warmup_epochs == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.split_path:
        cfg["split_path"] = args.split_path
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    result_root = Path(os.environ.get("PRM_RESULTS_ROOT", ROOT)).expanduser()
    output_path = Path(cfg["output_dir"])
    output_dir = output_path if output_path.is_absolute() else result_root / output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = resolve_path(os.environ.get("PRM_DATA_PATH", cfg["data_path"]))
    split_path = resolve_path(cfg["split_path"])

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg.get("cpu_threads", 8)))

    dataset = CrystalGraphDataset(data_path)
    split = load_split(
        split_path, len(dataset), expected_data_sha256=cfg.get("data_sha256")
    )
    sets = {
        name: Subset(dataset, split[name]) for name in ("train", "val", "test")
    }
    np.savez_compressed(
        output_dir / "split_indices.npz",
        schema_version=np.asarray("prm_split_indices_v1"),
        split_id=np.asarray(split["split_id"]),
        **{
            name: np.asarray(values.indices, dtype=np.int64)
            for name, values in sets.items()
        },
    )
    n_workers = int(cfg.get("num_workers", 4))
    train_sampler = None
    augmentation = None
    if cfg.get("online_aug", False):
        aug_cfg = cfg.get("online_aug_cfg", {})
        augmentation = OnlineAugTransform(
            sigma_range=tuple(aug_cfg.get("sigma_range", [0.01, 0.05])),
            strain_range=float(aug_cfg.get("strain_range", 2.0)),
            rotate_prob=float(aug_cfg.get("rotate_prob", 1.0)),
            perturb_prob=float(aug_cfg.get("perturb_prob", 0.8)),
            strain_prob=float(aug_cfg.get("strain_prob", 0.3)),
        )
    if cfg.get("host_balanced", False):
        train_dataset = OnlineAugDataset(dataset, transform=augmentation)
        train_sampler = HostBalancedSampler(
            dataset,
            subset_indices=sets["train"].indices,
            samples_per_host=int(cfg.get("samples_per_host", 200)),
            seed=int(cfg["seed"]),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=int(cfg.get("batch_size", 32)),
            sampler=train_sampler, collate_fn=collate_fn, num_workers=n_workers,
            pin_memory=device.type == "cuda", persistent_workers=n_workers > 0,
        )
    else:
        train_dataset = OnlineAugDataset(sets["train"], transform=augmentation)
        train_loader = DataLoader(
            train_dataset, batch_size=int(cfg.get("batch_size", 32)),
            shuffle=True, collate_fn=collate_fn, num_workers=n_workers,
            pin_memory=device.type == "cuda", persistent_workers=n_workers > 0,
        )
    loaders = {
        "train": train_loader,
        **{
            name: DataLoader(
                sets[name], batch_size=int(cfg.get("batch_size", 32)),
                shuffle=False, collate_fn=collate_fn, num_workers=n_workers,
                pin_memory=device.type == "cuda", persistent_workers=n_workers > 0,
            )
            for name in ("val", "test")
        },
    }
    train_targets = torch.as_tensor(
        [dataset.data[i]["target"] for i in sets["train"].indices], dtype=torch.float32
    )
    normalizer = Normalizer(train_targets)
    model = PBCSchNet(**cfg.get("model_kwargs", {})).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 150))
    scheduler = build_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=int(cfg.get("warmup_epochs", 0)),
        eta_min=float(cfg.get("eta_min", 1e-6)),
    )

    manifest = {
        "schema_version": "prm_run_manifest_v1",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "git": git_snapshot(),
        "config": cfg,
        "config_sha256": config_sha256(cfg),
        "execution": {
            "max_steps": int(args.max_steps),
            "resume_requested": bool(args.resume),
            "label_noise_stream": "model_seed_and_epoch_v1",
        },
        "data": {
            "path": str(data_path), "size_bytes": data_path.stat().st_size,
            "data_sha256": split["data_sha256"],
        },
        "split": {
            "split_id": split["split_id"], "path": str(split_path),
            "sha256": file_sha256(split_path), "counts": split["counts"],
        },
        "seed": cfg["seed"],
        "environment": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version, "torch": torch.__version__,
            "torch_geometric": __import__("torch_geometric").__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    start_epoch = 1
    history = []
    best_val = float("inf")
    global_steps = 0
    checkpoint_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if config_sha256(checkpoint.get("config", {})) != config_sha256(cfg):
            raise ValueError("resume checkpoint configuration does not match this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_val = checkpoint["best_val_mae"]
        start_epoch = checkpoint["epoch"] + 1
        global_steps = int(checkpoint.get("global_steps", 0))
        if "rng_state" not in checkpoint:
            raise ValueError("resume checkpoint lacks reproducible RNG state")
        restore_rng_state(checkpoint["rng_state"])

    with (output_dir / "train.log").open("a" if start_epoch > 1 else "w") as log:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()
            model.train()
            label_noise_generator = make_label_noise_generator(
                device, seed=int(cfg["seed"]), epoch=epoch
            )
            if train_sampler is not None:
                train_sampler.set_epoch(epoch - 1)
            train_error = 0.0
            n_seen = 0
            for batch in loaders["train"]:
                batch = move_batch(batch, device)
                label_noise = float(cfg.get("label_noise_std", 0.0))
                noisy_target = batch["target"]
                if label_noise > 0:
                    noise = torch.randn(
                        noisy_target.shape,
                        dtype=noisy_target.dtype,
                        device=noisy_target.device,
                        generator=label_noise_generator,
                    )
                    noisy_target = noisy_target + noise * label_noise
                target_norm = normalizer.norm(noisy_target)
                prediction_norm = model(batch)
                loss = torch.mean(torch.abs(prediction_norm - target_norm))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg.get("grad_clip", 5.0))
                )
                optimizer.step()
                with torch.no_grad():
                    prediction = normalizer.denorm(prediction_norm)
                    train_error += torch.sum(torch.abs(prediction - batch["target"])).item()
                    n_seen += len(prediction)
                global_steps += 1
                if args.max_steps and global_steps >= args.max_steps:
                    break
            scheduler.step()
            validation = evaluate(model, loaders["val"], normalizer, device)
            improved = validation["mae"] < best_val
            if improved:
                best_val = validation["mae"]
                torch.save(
                    {"model": model.state_dict(), "normalizer": normalizer.state_dict(), "config": cfg},
                    checkpoint_path,
                )
            row = {
                "epoch": epoch, "train_mae": train_error / max(n_seen, 1),
                "val_mae": validation["mae"], "val_rmse": validation["rmse"],
                "best": improved, "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - epoch_start,
            }
            history.append(row)
            line = (
                f"Epoch {epoch:03d}/{epochs} train={row['train_mae']:.4f} "
                f"val={row['val_mae']:.4f} best={best_val:.4f} sec={row['seconds']:.1f}"
            )
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            torch.save(
                {
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "history": history, "best_val_mae": best_val,
                    "global_steps": global_steps,
                    "rng_state": capture_rng_state(),
                    "config": cfg,
                },
                latest_path,
            )
            if args.max_steps and global_steps >= args.max_steps:
                break

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    validation = evaluate(model, loaders["val"], normalizer, device)
    test = evaluate(model, loaders["test"], normalizer, device)
    scalar_keys = ("n", "mae", "rmse", "bias", "pearson", "spearman", "r2")
    metrics = {
        "schema_version": "prm_schnet_results_v1",
        "split_id": split["split_id"], "n_params": n_params,
        "best_val_mae": best_val, "history": history,
        "validation": {key: validation[key] for key in scalar_keys},
        "test": {key: test[key] for key in scalar_keys},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    for name, values in (("val", validation), ("test", test)):
        np.savez_compressed(
            output_dir / f"{name}_predictions.npz",
            schema_version=np.asarray("prm_predictions_v1"),
            split_id=np.asarray(split["split_id"]), split=np.asarray(name),
            indices=values["indices"], preds=values["predictions"], targets=values["targets"],
        )
    manifest.update(
        {
            "status": "truncated" if args.max_steps else "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "outputs": {
                "metrics": "metrics.json",
                "checkpoint": "best.pt",
                "split_indices": "split_indices.npz",
                "validation_predictions": "val_predictions.npz",
                "test_predictions": "test_predictions.npz",
            },
        }
    )
    manifest["output_sha256"] = {
        key: file_sha256(output_dir / relative_path)
        for key, relative_path in manifest["outputs"].items()
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
