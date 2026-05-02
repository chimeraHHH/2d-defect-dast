"""Visualise the global attention map of the trained DAST model.

Loads the best-of-run checkpoint, picks a random validation sample, and
saves a heatmap of the attention weights from the last star-sparse layer
to ``paper/figures/fig_attention_map.png``.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits  # noqa: E402
from src.models import DefectAwareTransformer  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/improved.yaml")
    parser.add_argument("--ckpt", default="results/improved/best.pt")
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    with open(ROOT / args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ds = CrystalGraphDataset(ROOT / cfg["data_path"])
    _, val, _ = make_splits(ds, train_ratio=cfg["train_ratio"], val_ratio=cfg["val_ratio"], seed=cfg["seed"])

    sample = val[args.sample_idx]
    batch = collate_fn([sample])
    batch = {
        k: (v if not isinstance(v, torch.Tensor) else v) for k, v in batch.items()
    }

    model = DefectAwareTransformer(**cfg["model_kwargs"]).to("cpu")
    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    captured = {}

    def hook(module, args, kwargs, output):
        h = kwargs.get("h") if "h" in kwargs else (args[0] if args else None)
        mask_pair = kwargs.get("mask_pair") if "mask_pair" in kwargs else (args[1] if len(args) > 1 else None)
        dist_pair = kwargs.get("dist_pair") if "dist_pair" in kwargs else (args[2] if len(args) > 2 else None)
        defect_pair = kwargs.get("defect_pair") if "defect_pair" in kwargs else (args[3] if len(args) > 3 else None)
        if h is None or mask_pair is None:
            return
        b, n, c = h.shape
        head, d = module.num_heads, module.head_dim
        qkv = module.qkv(h).reshape(b, n, 3, head, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / d**0.5
        rbf = module.dist_rbf(dist_pair)
        bias = module.dist_bias(rbf).permute(0, 3, 1, 2)
        scores = scores + bias
        scores = scores + defect_pair.unsqueeze(1) * module.defect_bias.view(1, head, 1, 1)
        scores = scores.masked_fill(~mask_pair.unsqueeze(1), -1e9)
        attn = F.softmax(scores, dim=-1).detach()
        captured["attn"] = attn.cpu().numpy()

    last_attn = model.global_layers[-1].attn
    handle = last_attn.register_forward_hook(hook, with_kwargs=True)

    with torch.no_grad():
        _ = model(batch)
    handle.remove()

    attn = captured.get("attn")
    if attn is None:
        print("Failed to capture attention", file=sys.stderr)
        return

    n = sample["num_atoms"]
    head_avg = attn[0].mean(axis=0)[: n + 1, : n + 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(head_avg, aspect="auto", cmap="viridis")
    ax.set_xlabel("Key (atom index, last column = virtual)")
    ax.set_ylabel("Query (atom index)")
    ax.set_title("DAST star-sparse attention (head-averaged)")
    fig.colorbar(im, ax=ax, label="weight")
    fig.tight_layout()
    out = FIG_DIR / "fig_attention_map.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
