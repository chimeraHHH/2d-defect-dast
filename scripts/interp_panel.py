"""Build a single multi-panel figure that combines attention + occlusion
across multiple defect samples to demonstrate the findings are robust to
the specific test sample chosen.

Layout (3 rows × 4 columns):

  row | sample type           | col 0           col 1            col 2          col 3
  ----+----------------------+----------------+----------------+--------------+-----------
   1  | adsorbate, MoS2-like | atomic xy +    | head-avg attn  | per-head     | |Δ| vs
   2  | interstitial, large  | Δ_i colorbar   | row defect-row | dist decay   | distance
   3  | adsorbate, exotic    |                |                |              |

Each row picks a different test sample. We rely on the metadata for the
defect-type and host info to choose contrasting examples.

Saves: paper/figures/fig_interp_panel.png + results/interp_panel_meta.json
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
    h_ff = h.reshape(b * n_max, c)
    flat_h = h_ff.index_select(0, fi)
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


def predict_with_mask(model, batch, normalizer_mean, normalizer_std):
    with torch.no_grad():
        return model(batch).item() * normalizer_std + normalizer_mean


def occlusion(model, sample, normalizer_mean, normalizer_std):
    n = sample["num_atoms"]
    full = predict_with_mask(model, collate_fn([sample]), normalizer_mean, normalizer_std)
    delta = np.zeros(n)
    for i in range(n):
        b = collate_fn([sample])
        b["atom_mask"][0, i] = False
        delta[i] = full - predict_with_mask(model, b, normalizer_mean, normalizer_std)
    return full, delta


def get_attention(model, sample):
    h, dist_matrix, mask = run_to_global(model, collate_fn([sample]))
    h_in = h
    last_attn = None
    for layer in model.global_layers:
        last_attn = attention_for_block(layer, h_in, dist_matrix, mask).detach()
        h_in = layer(h_in, dist_matrix, mask)
    return last_attn[0].cpu().numpy()  # (H, N, N)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/baseline_h128_aug_long_safe.yaml"))
    cleaned_path = ROOT / "data/processed/cleaned_dataset.pkl"
    safe_path = ROOT / cfg["data_path"]
    ds = CrystalGraphDataset(safe_path if safe_path.exists() else cleaned_path)
    _, _, test_set = make_splits(
        ds, cfg.get("train_ratio", 0.8), cfg.get("val_ratio", 0.1), cfg.get("seed", 42),
    )

    model = CrystalTransformer(**cfg["model_kwargs"])
    ckpt = torch.load(ROOT / "results/baseline_h128_aug_long_safe/best.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    nmean, nstd = ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"]

    # ---- pick 3 contrasting samples ----
    picks = []
    bucket_used = {"adsorbate_TMD": False, "interstitial_large": False, "adsorbate_other": False}
    for i in range(len(test_set)):
        s = test_set[i]
        if s["defect_mask"].sum().item() != 1:
            continue
        meta = ds.data[test_set.indices[i]]["metadata"]
        host = meta.get("host", "?") or "?"
        dt = meta.get("defecttype", "?") or "?"
        n = s["num_atoms"]
        if not bucket_used["adsorbate_TMD"] and dt == "adsorbate" and host in {"MoS2", "WS2", "MoSe2", "WSe2", "MoTe2"} and 28 <= n <= 50:
            picks.append((i, "adsorbate, " + host)); bucket_used["adsorbate_TMD"] = True
        elif not bucket_used["interstitial_large"] and dt == "interstitial" and 50 <= n <= 80:
            picks.append((i, "interstitial, " + host)); bucket_used["interstitial_large"] = True
        elif not bucket_used["adsorbate_other"] and dt == "adsorbate" and host not in {"MoS2", "WS2", "MoSe2", "WSe2", "MoTe2"} and 28 <= n <= 50:
            picks.append((i, "adsorbate, " + host)); bucket_used["adsorbate_other"] = True
        if all(bucket_used.values()):
            break

    if not picks:
        print("could not pick samples; falling back"); picks = [(0, "fallback"),]

    print(f"chose {len(picks)} samples: {picks}")

    fig, axes = plt.subplots(len(picks), 4, figsize=(17, 4.2 * len(picks)))
    if len(picks) == 1:
        axes = np.array([axes])

    summary = []
    for row, (idx, lbl) in enumerate(picks):
        sample = test_set[idx]
        n = sample["num_atoms"]
        d_idx = int(sample["defect_mask"].argmax().item())
        meta = ds.data[test_set.indices[idx]]["metadata"]

        # occlusion
        full_pred, delta = occlusion(model, sample, nmean, nstd)
        target = sample["target"].item()

        # attention (last layer)
        attn = get_attention(model, sample)  # (H, N, N)
        head_avg = attn[:, :n, :n].mean(0)

        pos = sample["positions"].numpy()
        d_arr = np.linalg.norm(pos - pos[d_idx], axis=1)
        nums = ds.data[test_set.indices[idx]]["numbers"]

        # col 0 — Δ_i in xy
        ax = axes[row, 0]
        cmap = plt.get_cmap("RdBu_r")
        norm = plt.Normalize(vmin=-np.abs(delta).max(), vmax=np.abs(delta).max())
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=delta, cmap=cmap, norm=norm, s=160, edgecolors="k")
        ax.scatter(pos[d_idx, 0], pos[d_idx, 1], facecolors="none", edgecolors="lime", s=380, lw=2, label="defect")
        for k, (x, y) in enumerate(pos[:, :2]):
            ax.text(x, y, str(nums[k]), fontsize=4.5, ha="center", va="center")
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
        ax.set_title(f"{lbl}\nE_f^DFT={target:.2f} eV  E_f^pred={full_pred:.2f} eV")
        if row == 0:
            ax.legend(fontsize=8)
        fig.colorbar(sc, ax=ax, label="Δᵢ (eV)", fraction=0.046)

        # col 1 — head-averaged attention map
        ax = axes[row, 1]
        im = ax.imshow(head_avg, cmap="viridis", vmin=0)
        ax.axhline(d_idx, color="red", lw=0.6, alpha=0.7)
        ax.axvline(d_idx, color="red", lw=0.6, alpha=0.7)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        ax.set_title("Head-avg attn (last layer)")
        fig.colorbar(im, ax=ax, fraction=0.046)

        # col 2 — defect column (incoming attention)
        ax = axes[row, 2]
        incoming = head_avg[:, d_idx]
        bars = ax.bar(np.arange(n), incoming)
        bars[d_idx].set_color("red")
        avg_other = (incoming.sum() - incoming[d_idx]) / max(n - 1, 1)
        ax.axhline(avg_other, color="k", ls="--", lw=0.6, label=f"mean ex-self={avg_other:.3f}")
        ax.set_xlabel("Atom index")
        ax.set_ylabel("Attn → defect")
        ax.set_title("Incoming attention to defect")
        ax.legend(fontsize=7)

        # col 3 — |Δᵢ| vs distance, log-scale
        ax = axes[row, 3]
        adlt = np.abs(delta)
        ax.scatter(d_arr, adlt, color="tab:blue")
        ax.scatter(d_arr[d_idx], adlt[d_idx], color="red", s=80, label="defect")
        ax.set_xlabel("Distance from defect (Å)")
        ax.set_ylabel("|Δᵢ| (eV)")
        ax.set_yscale("log")
        ax.set_title("Energy contribution vs distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        summary.append({
            "row": row,
            "label": lbl,
            "host": meta.get("host"),
            "dopant": meta.get("dopant"),
            "defecttype": meta.get("defecttype"),
            "natoms": int(n),
            "defect_idx": d_idx,
            "target_ef": float(target),
            "predicted_ef": float(full_pred),
            "incoming_attn_to_defect": float(head_avg[:, d_idx].mean()),
            "incoming_attn_to_random_other": float(np.delete(head_avg[:, 0 if d_idx != 0 else 1], []).mean()),
            "delta_at_defect": float(delta[d_idx]),
            "fraction_attribution_at_defect": float(np.abs(delta[d_idx]) / max(np.abs(delta).sum(), 1e-9)),
        })

    fig.suptitle("Interpretability across defect types: attention + occlusion (one row per sample)",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_interp_panel.png"
    fig.savefig(out, dpi=170); plt.close(fig)
    print(f"saved {out}")

    out_json = ROOT / "results/interp_panel_meta.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n--- chosen samples ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
