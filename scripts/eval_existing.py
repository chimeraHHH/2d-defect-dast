"""Evaluate an existing checkpoint and produce metrics.json + predictions npz.

Useful when training was interrupted before the final test pass: this script
loads the best.pt under ``results/<run>/`` and runs the same final-test logic
that ``src/train.py`` performs at epoch end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits  # noqa: E402
from src.models import CrystalTransformer, DefectAwareTransformer  # noqa: E402
from src.train import Normalizer, evaluate, move_batch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


REGISTRY = {"baseline": CrystalTransformer, "improved": DefectAwareTransformer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ds = CrystalGraphDataset(ROOT / cfg["data_path"])
    train_set, val_set, test_set = make_splits(
        ds,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 42),
    )
    loader = DataLoader(
        test_set,
        batch_size=cfg.get("batch_size", 16),
        shuffle=False,
        collate_fn=collate_fn,
    )
    targets = torch.tensor(
        [ds.data[i]["target"] for i in train_set.indices], dtype=torch.float32
    )
    normalizer = Normalizer(targets)

    cls = REGISTRY[cfg["model"]]
    model = cls(**cfg.get("model_kwargs", {})).to(device)
    ck = torch.load(ROOT / args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if "normalizer" in ck:
        normalizer.mean = ck["normalizer"]["mean"]
        normalizer.std = ck["normalizer"]["std"]

    test_metrics = evaluate(model, loader, normalizer, device)
    summary = {
        "config": cfg,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "history": [],  # not available
        "best_val_mae": ck.get("val_mae", float("nan")),
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "note": "metrics produced post-hoc by eval_existing.py from an interrupted run",
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(
        out_dir / "test_predictions.npz",
        preds=test_metrics["preds"],
        targets=test_metrics["targets"],
    )
    print(
        f"Wrote {metrics_path} | test MAE {test_metrics['mae']:.4f} | "
        f"RMSE {test_metrics['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
