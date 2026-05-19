#!/usr/bin/env python3
"""Evaluate a best.pt checkpoint without waiting for training to finish.

Loads the saved best checkpoint, runs inference on the test set, and prints
per-range MAE + overall metrics. Optionally saves test_predictions.npz.

Handles two checkpoint formats:
  1. New format: dict with keys {model, config, normalizer, epoch, ...}
  2. Legacy format: raw state_dict or {model_state_dict: ...}

Usage:
    python scripts/eval_checkpoint.py results/v3_deftype
    python scripts/eval_checkpoint.py results/v4_moe --save
    python scripts/eval_checkpoint.py results/v3_deftype results/v6_physics  # compare
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.models import CrystalTransformerV2


class Normalizer:
    def __init__(self, mean, std, transform="none"):
        self.mean, self.std, self.transform = mean, std, transform

    def norm(self, t):
        if self.transform == "log":
            t = torch.sign(t) * torch.log1p(torch.abs(t))
        return (t - self.mean) / self.std

    def denorm(self, t):
        t = t * self.std + self.mean
        if self.transform == "log":
            t = torch.sign(t) * torch.expm1(torch.abs(t))
        return t


def evaluate(model_dir: str, save: bool = False, device: str = "cuda"):
    model_dir = Path(model_dir)
    ckpt_path = model_dir / "best.pt"
    metrics_path = model_dir / "metrics.json"

    if not ckpt_path.exists():
        print("  %s: no best.pt found" % model_dir.name)
        return None

    # ── Load checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Determine checkpoint format
    has_embedded_config = isinstance(ckpt, dict) and "config" in ckpt

    if has_embedded_config:
        # New format: config + normalizer embedded in checkpoint
        cfg = ckpt["config"]
        norm_info = ckpt.get("normalizer", {})
        state_dict = ckpt["model"]  # OrderedDict
        ckpt_epoch = ckpt.get("epoch", "?")
        best_val_mae = ckpt.get("best_val_mae", "?")
    else:
        # Legacy format: need external config
        cfg = None
        state_dict = None
        ckpt_epoch = "?"
        best_val_mae = "?"

        if metrics_path.exists():
            with open(metrics_path) as f:
                meta = json.load(f)
            cfg = meta.get("config", {})
            ckpt_epoch = len(meta.get("history", []))
            best_val_mae = meta.get("best_val_mae", "?")
        else:
            yaml_candidates = list(Path("configs").glob("*%s*" % model_dir.name))
            if yaml_candidates:
                import yaml
                with open(yaml_candidates[0]) as f:
                    cfg = yaml.safe_load(f)
            else:
                print("  %s: no config found (no metrics.json, no yaml)" % model_dir.name)
                return None

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt  # raw state_dict

    # Also check metrics.json for epoch count if available
    n_epochs_done = ckpt_epoch
    if metrics_path.exists() and n_epochs_done == "?":
        try:
            with open(metrics_path) as f:
                meta = json.load(f)
            n_epochs_done = len(meta.get("history", []))
            if best_val_mae == "?":
                best_val_mae = meta.get("best_val_mae", "?")
        except Exception:
            pass

    # ── Build normalizer ─────────────────────────────────────────────
    if has_embedded_config and isinstance(norm_info, dict) and "mean" in norm_info:
        normalizer = Normalizer(
            norm_info["mean"], norm_info["std"],
            transform=norm_info.get("transform", "none"),
        )
    else:
        # Recompute from training data
        data_path = cfg.get("data_path", "data/processed/cleaned_dataset.pkl")
        dataset_tmp = CrystalGraphDataset(ROOT / data_path)
        train_indices = make_splits(
            dataset_tmp, train_ratio=0.8, val_ratio=0.1, seed=42,
        )[0].indices
        targets_train = torch.tensor(
            [dataset_tmp.data[i]["target"] for i in train_indices], dtype=torch.float32
        )
        normalizer = Normalizer(
            float(targets_train.mean()), float(targets_train.std()) + 1e-6,
            transform=cfg.get("target_transform", "none"),
        )

    # ── Load dataset and test split ──────────────────────────────────
    data_path = cfg.get("data_path", "data/processed/cleaned_dataset.pkl")
    dataset = CrystalGraphDataset(ROOT / data_path)
    _, _, test_set = make_splits(
        dataset,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=42,
    )
    test_loader = DataLoader(
        test_set, batch_size=128, shuffle=False, collate_fn=collate_fn,
    )

    # ── Build model with correct kwargs ──────────────────────────────
    model_kwargs = cfg.get("model_kwargs", {})
    model = CrystalTransformerV2(**model_kwargs)

    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Filter out expected missing keys (aux heads etc.)
        real_missing = [k for k in missing if "aux_" not in k]
        if real_missing:
            print("  WARNING: %d missing keys: %s" % (len(real_missing), real_missing[:5]))
    if unexpected:
        print("  WARNING: %d unexpected keys: %s" % (len(unexpected), unexpected[:5]))

    model = model.to(device).eval()

    # ── Inference ────────────────────────────────────────────────────
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            preds = normalizer.denorm(out).cpu().numpy()
            targets = batch["target"].cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # ── Metrics ──────────────────────────────────────────────────────
    mae = float(np.abs(preds - targets).mean())
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    ranges = [(0, 2, "[0,2)"), (2, 5, "[2,5)"), (5, 7, "[5,7)"), (7, 25, "[7,25)")]
    range_str = []
    for lo, hi, label in ranges:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            r_mae = float(np.abs(preds[mask] - targets[mask]).mean())
            range_str.append("%s:%.3f(%d)" % (label, r_mae, mask.sum()))

    val_str = "%.4f" % best_val_mae if isinstance(best_val_mae, (int, float)) else str(best_val_mae)
    print("  %-25s ep=%-4s  val=%-8s  test MAE=%.4f RMSE=%.4f  %s"
          % (model_dir.name, n_epochs_done, val_str, mae, rmse,
             "  ".join(range_str)))

    if save:
        out_path = model_dir / "test_predictions.npz"
        np.savez(out_path, preds=preds, targets=targets)
        print("    -> saved %s" % out_path)

    return {"name": model_dir.name, "mae": mae, "rmse": rmse, "preds": preds,
            "targets": targets, "n_epochs": n_epochs_done}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Model result directories")
    parser.add_argument("--save", action="store_true",
                        help="Save test_predictions.npz")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("Evaluating %d checkpoint(s)..." % len(args.dirs))
    results = []
    for d in args.dirs:
        r = evaluate(d, save=args.save, device=args.device)
        if r:
            results.append(r)

    if len(results) > 1:
        print("\nRanking:")
        for i, r in enumerate(sorted(results, key=lambda x: x["mae"])):
            print("  %d. %s: MAE=%.4f" % (i + 1, r["name"], r["mae"]))


if __name__ == "__main__":
    main()
