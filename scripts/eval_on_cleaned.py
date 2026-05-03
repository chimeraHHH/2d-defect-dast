"""Re-evaluate aug-trained checkpoints on the *cleaned* (non-augmented) test
split, for a fair head-to-head with no-aug baselines and ALIGNN.

The augmented dataset shuffles original/rotated/perturbed samples across
train/val/test; this means a model trained on it has seen near-duplicates of
its test samples (data leakage). To remove that leakage we evaluate on the
test split of cleaned_dataset.pkl (1065 originals, the same split used by the
no-aug baseline at seed=42).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits  # noqa: E402
from src.models import CrystalTransformer, DefectAwareTransformer  # noqa: E402
from src.train import Normalizer, evaluate, move_batch  # noqa: E402

REGISTRY = {"baseline": CrystalTransformer, "improved": DefectAwareTransformer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run name (e.g. baseline_aug_long)")
    parser.add_argument("--clean-data", default="data/processed/cleaned_dataset.pkl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = ROOT / "results" / args.run
    cfg_path = ROOT / "configs" / f"{args.run}.yaml"
    if not cfg_path.exists():
        # fall back to inferring config from metrics.json
        meta = json.load(open(run_dir / "metrics.json"))
        cfg = meta["config"]
    else:
        cfg = yaml.safe_load(open(cfg_path))

    device = torch.device(args.device)

    # Build the cleaned dataset and use seed=42 80/10/10 split (same as no-aug
    # baseline). For runs that trained at non-42 seeds, we still evaluate on
    # the same canonical test set so all numbers refer to the same held-out
    # samples.
    ds = CrystalGraphDataset(ROOT / args.clean_data)
    train_set, _, test_set = make_splits(
        ds, train_ratio=0.8, val_ratio=0.1, seed=42
    )

    # Normalizer must come from the AUG run's training set, otherwise denorm
    # is wrong. We pull mean/std from the saved checkpoint.
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    if "normalizer" in ckpt:
        norm = Normalizer(torch.zeros(2))
        norm.mean = float(ckpt["normalizer"]["mean"])
        norm.std = float(ckpt["normalizer"]["std"])
    else:
        # fallback: compute from cleaned train indices
        targets = torch.tensor([ds.data[i]["target"] for i in train_set.indices], dtype=torch.float32)
        norm = Normalizer(targets)

    cls = REGISTRY[cfg["model"]]
    model = cls(**cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["model"])

    loader = DataLoader(test_set, batch_size=cfg.get("batch_size", 64), shuffle=False, collate_fn=collate_fn)

    metrics = evaluate(model, loader, norm, device)
    out = {
        "run": args.run,
        "test_set": "cleaned 1065 originals (seed=42)",
        "test_mae": metrics["mae"],
        "test_rmse": metrics["rmse"],
        "n_samples": int(metrics["preds"].size),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    np.savez(run_dir / "test_predictions_cleaned.npz", preds=metrics["preds"], targets=metrics["targets"])

    # also write to metrics_cleaned.json
    out_path = run_dir / "metrics_cleaned.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
