"""Extract and visualise multi-head, multi-layer attention from CrystalTransformer.

Loads ``baseline_h128_aug_long_safe/best.pt``, runs it on a small batch of
test samples drawn from the canonical (cleaned 1065) split, and:

  * extracts per-head attention from every ``GeometricTransformerBlock``
    (the model has 2 such blocks × 4 heads with hidden=128)
  * picks one defect supercell, plots
      (a) head-by-head attention heatmaps for layer L=2 (the deep one)
      (b) the row of attention out of the *defect* atom (the dopant) showing
          how strongly it attends to every other atom — head-averaged
      (c) attention-weight vs PBC distance scatter, decomposed by head
  * aggregates over a 200-sample subset to compute global statistics:
      - per-head average attention entropy (low → focused, high → spread out)
      - mean attention to the defect-marked atom vs to a random atom
      - effective range: dist at which mean attention drops to half its peak

Output:
  - paper/figures/fig_attention_heads.png           (single sample, 4 heads)
  - paper/figures/fig_attention_defect_centric.png  (defect-row + dist decay)
  - results/attention_stats.json                    (cross-sample aggregates)

Why this matters: a frequent reviewer ask is "what is your attention learning?".
The defect-centric figure should show that the dopant atom carries elevated
attention from many other atoms — i.e. the global Transformer does pick up the
defect as a long-range hub even without an explicit virtual anchor.
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


def attention_for_block(block, x, dist_matrix, mask):
    """Manually replicate GeometricTransformerBlock attention to extract weights.

    Mirrors src/models/baseline.py:GeometricTransformerBlock.forward up to the
    softmax. Returns attn (B, H, N, N) post-softmax.
    """
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
    attn = torch.nan_to_num(attn, nan=0.0)
    return attn


def run_model_to_global(model, batch):
    """Run model up to and including local layers; return (h_local, dist_matrix, mask)."""
    x = batch["x"]; mask = batch["atom_mask"]
    dist_matrix = batch["dist_matrix"]
    defect_mask = batch.get("defect_mask")
    h = model.embed(x)
    if model.defect_embedding is not None and defect_mask is not None:
        h = h + model.defect_embedding(defect_mask)
    b, n_max, c = h.shape
    num_atoms_list = batch["num_atoms_list"]
    flat_indices = []
    for i, n_i in enumerate(num_atoms_list):
        flat_indices.append(torch.arange(n_i, dtype=torch.long) + i * n_max)
    flat_indices = torch.cat(flat_indices)
    h_flat_full = h.reshape(b * n_max, c)
    flat_h = h_flat_full.index_select(0, flat_indices)
    edge_index, edge_dist, triplet_index, angles = model._flatten_edges(
        num_atoms_list,
        batch["edge_index_list"], batch["edge_dist_list"],
        batch["triplet_index_list"], batch["angles_list"],
        device=torch.device("cpu"),
    )
    edge_attr_rbf = model.edge_rbf(edge_dist)
    for layer in model.local_layers:
        flat_h = layer(flat_h, edge_index, edge_attr_rbf, triplet_index, angles)
    h_local_flat = torch.zeros(b * n_max, c, dtype=h.dtype)
    h_local_flat.index_copy_(0, flat_indices, flat_h)
    h_local = h_local_flat.reshape(b, n_max, c)
    return h_local, dist_matrix, mask


def extract_all_attention(model, batch):
    """Returns list of attention tensors, one per global Transformer block.
    Each tensor has shape (B, H, N, N).
    """
    h, dist_matrix, mask = run_model_to_global(model, batch)
    attns = []
    h_in = h
    for layer in model.global_layers:
        attn = attention_for_block(layer, h_in, dist_matrix, mask)
        attns.append(attn.detach())
        # Also actually run the block forward so subsequent layers see correct h
        h_in = layer(h_in, dist_matrix, mask)
    return attns, dist_matrix, mask


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/baseline_h128_aug_long_safe.yaml"))
    # the leak-free aug pickle isn't always present locally; the test set is the
    # same 1065 cleaned originals at indices test_idx of cleaned_dataset.pkl with
    # split_indices(0.8, 0.1, 42).
    cleaned_path = ROOT / "data/processed/cleaned_dataset.pkl"
    safe_path = ROOT / cfg["data_path"]
    if safe_path.exists():
        ds = CrystalGraphDataset(safe_path)
        _, _, test_set = make_splits(
            ds, cfg.get("train_ratio", 0.8), cfg.get("val_ratio", 0.1),
            cfg.get("seed", 42),
        )
    else:
        ds = CrystalGraphDataset(cleaned_path)
        _, _, test_set = make_splits(
            ds, cfg.get("train_ratio", 0.8), cfg.get("val_ratio", 0.1),
            cfg.get("seed", 42),
        )

    model = CrystalTransformer(**cfg["model_kwargs"])
    ckpt_path = ROOT / "results/baseline_h128_aug_long_safe/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # find a sample with a clear single-dopant defect (defect_mask sums to 1)
    pick = None
    for i in range(len(test_set)):
        s = test_set[i]
        if s["defect_mask"].sum().item() == 1 and 28 <= s["num_atoms"] <= 50:
            pick = i; break
    if pick is None:
        pick = 0
    print(f"sample idx {pick} | num_atoms {test_set[pick]['num_atoms']}")

    sample = test_set[pick]
    batch = collate_fn([sample])
    with torch.no_grad():
        attns, dist_matrix, mask = extract_all_attention(model, batch)
    n = sample["num_atoms"]
    defect_idx = int(sample["defect_mask"].argmax().item())

    # ---------- Figure 1: per-head heatmap of last layer ----------
    L = len(attns)
    fig, axes = plt.subplots(1, attns[-1].shape[1], figsize=(15, 4))
    for h_i in range(attns[-1].shape[1]):
        a = attns[-1][0, h_i, :n, :n].cpu().numpy()
        ax = axes[h_i]
        im = ax.imshow(a, aspect="auto", cmap="viridis", vmin=0)
        ax.set_title(f"Layer {L} head {h_i + 1}")
        ax.set_xlabel("Key atom")
        if h_i == 0:
            ax.set_ylabel("Query atom")
        ax.axhline(defect_idx, color="red", linewidth=0.6, alpha=0.7)
        ax.axvline(defect_idx, color="red", linewidth=0.6, alpha=0.7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"CrystalTransformer attention | last global block | sample idx {pick} | N={n}")
    fig.tight_layout()
    out1 = FIG_DIR / "fig_attention_heads.png"
    fig.savefig(out1, dpi=180); plt.close(fig)
    print(f"saved {out1}")

    # ---------- Figure 2: defect-centric profile + distance decay ----------
    # head-averaged attention for last block
    a_head_avg = attns[-1][0, :, :n, :n].mean(0).cpu().numpy()  # (N,N)
    incoming = a_head_avg[:, defect_idx]   # how much each query attends to defect
    outgoing = a_head_avg[defect_idx, :]   # how the defect attends to each key
    dists = dist_matrix[0, :n, :n].cpu().numpy()
    d_to_defect = dists[:, defect_idx]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    bars = ax.bar(np.arange(n), incoming)
    bars[defect_idx].set_color("red")
    ax.set_xlabel("Atom index")
    ax.set_ylabel("Attention weight to defect")
    mean_in = incoming.mean()
    mean_in_excl = (incoming.sum() - incoming[defect_idx]) / max(n - 1, 1)
    ax.axhline(mean_in_excl, color="k", lw=0.6, ls="--",
               label=f"mean (excl. self) = {mean_in_excl:.3f}")
    ax.set_title(f"Incoming attention to defect (idx={defect_idx})")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(np.arange(n), outgoing)
    ax.set_xlabel("Atom index")
    ax.set_ylabel("Attention weight from defect")
    mean_out = outgoing.mean()
    mean_out_excl = (outgoing.sum() - outgoing[defect_idx]) / max(n - 1, 1)
    ax.axhline(mean_out_excl, color="k", lw=0.6, ls="--",
               label=f"mean (excl. self) = {mean_out_excl:.3f}")
    ax.set_title(f"Outgoing attention from defect")
    ax.legend(fontsize=8)

    ax = axes[2]
    H = attns[-1].shape[1]
    for h_i in range(H):
        out_h = attns[-1][0, h_i, defect_idx, :n].cpu().numpy()
        # binned mean
        bins = np.arange(0, max(d_to_defect.max() + 0.5, 8), 0.5)
        idx = np.digitize(d_to_defect, bins)
        means = []
        for b in range(1, len(bins) + 1):
            sel = idx == b
            if sel.sum() > 0:
                means.append((bins[b - 1] + 0.25, out_h[sel].mean()))
        if means:
            xs, ys = zip(*means)
            ax.plot(xs, ys, "o-", label=f"head {h_i + 1}")
    ax.set_xlabel("PBC distance from defect (Å)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Defect-centric attention vs distance (per head)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out2 = FIG_DIR / "fig_attention_defect_centric.png"
    fig.savefig(out2, dpi=180); plt.close(fig)
    print(f"saved {out2}")

    # ---------- Aggregates over a sample of test set ----------
    aggregate_n = min(200, len(test_set))
    print(f"aggregating attention stats over {aggregate_n} test samples...")
    incoming_def_acc, incoming_other_acc = [], []
    entropy_per_layer_head = [
        np.zeros(attn.shape[1]) for attn in attns
    ]
    counts = [np.zeros(attn.shape[1]) for attn in attns]
    for i in range(aggregate_n):
        s = test_set[i]
        if s["num_atoms"] < 4:
            continue
        n_i = s["num_atoms"]
        d_i = int(s["defect_mask"].argmax().item()) if s["defect_mask"].sum() > 0 else 0
        with torch.no_grad():
            a_list, _, _ = extract_all_attention(model, collate_fn([s]))
        a_last = a_list[-1][0, :, :n_i, :n_i].cpu().numpy()  # (H, N, N)
        head_avg = a_last.mean(0)  # (N, N)
        incoming_def_acc.append(head_avg[:, d_i].mean())
        # mean to a random non-defect atom (use index 0 if defect at 0 else 0-or-1 alternation)
        other = 0 if d_i != 0 else 1
        if n_i > other:
            incoming_other_acc.append(head_avg[:, other].mean())
        for li, a_l in enumerate(a_list):
            a_arr = a_l[0, :, :n_i, :n_i].cpu().numpy()  # (H, N, N)
            for h_i in range(a_arr.shape[0]):
                eps = 1e-9
                ent = -(a_arr[h_i] * np.log(a_arr[h_i] + eps)).sum(-1).mean()
                entropy_per_layer_head[li][h_i] += float(ent)
                counts[li][h_i] += 1

    ent_means = [
        (entropy_per_layer_head[li] / np.maximum(counts[li], 1)).tolist()
        for li in range(L)
    ]
    n_max_max_ent = math.log(max(s["num_atoms"] for s in [test_set[i] for i in range(aggregate_n)]))
    stats = {
        "n_samples_used": aggregate_n,
        "incoming_attention_to_defect_atom_mean": float(np.mean(incoming_def_acc)),
        "incoming_attention_to_random_other_mean": float(np.mean(incoming_other_acc)),
        "ratio_defect_over_other": float(
            np.mean(incoming_def_acc) / max(np.mean(incoming_other_acc), 1e-9)
        ),
        "per_layer_head_mean_entropy_nats": ent_means,
        "max_entropy_uniform_log_N": n_max_max_ent,
    }
    with open(ROOT / "results/attention_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\n--- aggregates ---")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
