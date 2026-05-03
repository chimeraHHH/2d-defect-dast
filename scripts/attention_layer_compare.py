"""How does attention pattern evolve from layer 1 to layer 2?

The CrystalTransformer has 2 GeometricTransformerBlock layers in the global
path. We extract head-averaged attention from both, compare on the same
sample, and aggregate the entropy and defect-attention statistics across
200 test samples.

Hypothesis (informed by the §5.10 finding): layer 1 likely does broad
"set-up" attention (higher entropy, less defect-focused), while layer 2
sharpens and concentrates on the defect (lower entropy, higher defect-attn).

Output:
  - paper/figures/fig_attention_layer_compare.png
  - results/attention_layer_compare.json
"""
from __future__ import annotations

import json
import math
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
from src.models import CrystalTransformer  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def attn_for_block(block, x, dist_matrix, mask):
    b, n, c = x.shape
    h, d = block.num_heads, block.head_dim
    x_norm = block.norm1(x)
    qkv = block.qkv(x_norm).reshape(b, n, 3, h, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    rbf = block.dist_rbf(dist_matrix)
    bias = block.bias_mlp(rbf).permute(0, 3, 1, 2)
    scores = scores + bias
    scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
    attn = F.softmax(scores, dim=-1)
    return torch.nan_to_num(attn, nan=0.0)


def run_to_global(model, batch):
    x = batch["x"]; mask = batch["atom_mask"]
    dist_matrix = batch["dist_matrix"]
    defect_mask = batch.get("defect_mask")
    h = model.embed(x)
    if model.defect_embedding is not None and defect_mask is not None:
        h = h + model.defect_embedding(defect_mask)
    b, n_max, c = h.shape
    nl = batch["num_atoms_list"]
    fi = []
    for i, n_i in enumerate(nl):
        fi.append(torch.arange(n_i, dtype=torch.long) + i * n_max)
    fi = torch.cat(fi)
    flat_h = h.reshape(b * n_max, c).index_select(0, fi)
    edge_index, edge_dist, triplet_index, angles = model._flatten_edges(
        nl, batch["edge_index_list"], batch["edge_dist_list"],
        batch["triplet_index_list"], batch["angles_list"],
        device=torch.device("cpu"),
    )
    erbf = model.edge_rbf(edge_dist)
    for layer in model.local_layers:
        flat_h = layer(flat_h, edge_index, erbf, triplet_index, angles)
    h_local = torch.zeros(b * n_max, c).index_copy_(0, fi, flat_h).reshape(b, n_max, c)
    return h_local, dist_matrix, mask


def both_layers(model, sample):
    h, dist_matrix, mask = run_to_global(model, collate_fn([sample]))
    h_in = h
    layer_attns = []
    for layer in model.global_layers:
        layer_attns.append(attn_for_block(layer, h_in, dist_matrix, mask).detach())
        h_in = layer(h_in, dist_matrix, mask)
    return layer_attns  # list of (1, H, N, N)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/baseline_h128_aug_long_safe.yaml"))
    cleaned = ROOT / "data/processed/cleaned_dataset.pkl"
    safe = ROOT / cfg["data_path"]
    ds = CrystalGraphDataset(safe if safe.exists() else cleaned)
    _, _, test_set = make_splits(
        ds, cfg.get("train_ratio", 0.8), cfg.get("val_ratio", 0.1), cfg.get("seed", 42),
    )
    model = CrystalTransformer(**cfg["model_kwargs"])
    ckpt = torch.load(ROOT / "results/baseline_h128_aug_long_safe/best.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- pick one sample & plot per-head heatmaps for layer 1 and layer 2 ----
    pick = None
    for i in range(len(test_set)):
        s = test_set[i]
        if s["defect_mask"].sum().item() == 1 and 28 <= s["num_atoms"] <= 50:
            pick = i; break
    if pick is None:
        pick = 0
    sample = test_set[pick]
    n = sample["num_atoms"]
    d_idx = int(sample["defect_mask"].argmax().item())
    print(f"sample idx={pick}, n={n}, defect_idx={d_idx}")

    layer_attns = both_layers(model, sample)
    L = len(layer_attns)
    H = layer_attns[0].shape[1]
    fig, axes = plt.subplots(L, H, figsize=(15, 4.2 * L))

    # plot per-head per-layer
    for layer_i in range(L):
        for head_i in range(H):
            ax = axes[layer_i, head_i]
            a = layer_attns[layer_i][0, head_i, :n, :n].cpu().numpy()
            im = ax.imshow(a, cmap="viridis", vmin=0)
            ax.axhline(d_idx, color="red", lw=0.6, alpha=0.7)
            ax.axvline(d_idx, color="red", lw=0.6, alpha=0.7)
            ax.set_title(f"Layer {layer_i + 1}, head {head_i + 1}")
            if layer_i == L - 1:
                ax.set_xlabel("Key")
            if head_i == 0:
                ax.set_ylabel("Query")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Per-head attention by layer (sample idx={pick})")
    fig.tight_layout()
    out = FIG_DIR / "fig_attention_layer_compare.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"saved {out}")

    # ---- aggregate stats across 200 samples ----
    n_agg = min(200, len(test_set))
    ent_acc = [np.zeros(H) for _ in range(L)]
    counts = [np.zeros(H) for _ in range(L)]
    inc_def = [np.zeros(H) for _ in range(L)]
    inc_def_n = [np.zeros(H) for _ in range(L)]
    for i in range(n_agg):
        s = test_set[i]
        if s["defect_mask"].sum().item() != 1:
            continue
        d_i = int(s["defect_mask"].argmax().item())
        n_i = s["num_atoms"]
        with torch.no_grad():
            la = both_layers(model, s)
        for li, attn in enumerate(la):
            arr = attn[0, :, :n_i, :n_i].cpu().numpy()  # (H, N, N)
            for h_i in range(H):
                ent = -(arr[h_i] * np.log(arr[h_i] + 1e-9)).sum(-1).mean()
                ent_acc[li][h_i] += float(ent)
                counts[li][h_i] += 1
                inc_def[li][h_i] += float(arr[h_i, :, d_i].mean())
                inc_def_n[li][h_i] += 1
    layer_stats = []
    for li in range(L):
        layer_stats.append({
            "layer": li + 1,
            "head_entropy_nats": (ent_acc[li] / np.maximum(counts[li], 1)).tolist(),
            "head_incoming_attn_to_defect_mean": (inc_def[li] / np.maximum(inc_def_n[li], 1)).tolist(),
        })

    print(json.dumps(layer_stats, indent=2))
    with open(ROOT / "results/attention_layer_compare.json", "w") as f:
        json.dump({"layers": layer_stats, "n_aggregate": n_agg}, f, indent=2)
    print("saved results/attention_layer_compare.json")


if __name__ == "__main__":
    main()
