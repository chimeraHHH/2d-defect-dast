"""Sanity check: does the defect-as-hub interpretability finding hold for
the LOHO MoS2 model? This model has NEVER seen MoS2 during training but is
asked to predict on 308 MoS2 samples.

If we still find:
  - high attention to defect (vs random other)
  - high occlusion-attribution at defect
on these 308 OOD samples, the interpretability story generalises beyond
training-distribution chemistry.

We aggregate over a subsample of the LOHO MoS2 test set and report the same
two key statistics from §5.10.

Output: results/loho_interp_check.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

RESULTS = ROOT / "results"


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
    return torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)


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


def predict(model, batch, nmean, nstd):
    with torch.no_grad():
        return model(batch).item() * nstd + nmean


def occlusion_per_atom(model, sample, nmean, nstd):
    n = sample["num_atoms"]
    full = predict(model, collate_fn([sample]), nmean, nstd)
    delta = np.zeros(n)
    for i in range(n):
        b = collate_fn([sample])
        b["atom_mask"][0, i] = False
        delta[i] = full - predict(model, b, nmean, nstd)
    return full, delta


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="MoS2", help="LOHO host name")
    args = ap.parse_args()
    host = args.host
    print(f"=== LOHO interp check for host={host} ===")

    cfg_path = ROOT / f"configs/loho_{host}.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    safe = ROOT / cfg["data_path"]
    cleaned = ROOT / "data/processed/cleaned_dataset.pkl"

    if not safe.exists():
        print(f"LOHO {host} pickle not local; using cleaned + {host} filter")
        ds_full = CrystalGraphDataset(cleaned)
        host_idx = [i for i, s in enumerate(ds_full.data)
                    if (s["metadata"].get("host") or "?") == host]
        print(f"  found {len(host_idx)} {host} samples")
        from torch.utils.data import Subset
        test_set = Subset(ds_full, host_idx)
        ds = ds_full
    else:
        ds = CrystalGraphDataset(safe)
        _, _, test_set = make_splits(ds, cfg.get("train_ratio", 0.8),
                                      cfg.get("val_ratio", 0.1), cfg.get("seed", 42))

    model = CrystalTransformer(**cfg["model_kwargs"])
    ckpt_path = ROOT / f"results/loho_{host}/best.pt"
    if not ckpt_path.exists():
        print(f"!! missing {ckpt_path}; can't run")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    nmean, nstd = ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"]

    # ---- aggregate ----
    n_agg = min(80, len(test_set))
    print(f"aggregating LOHO MoS2 model interpretability over {n_agg} OOD samples...")
    inc_def, inc_other = [], []
    delta_def, delta_other = [], []
    fraction_at_def = []
    for i in range(n_agg):
        s = test_set[i]
        if s["defect_mask"].sum().item() != 1:
            continue
        d_i = int(s["defect_mask"].argmax().item())
        n_i = s["num_atoms"]
        # attention
        h, dist_matrix, mask = run_to_global(model, collate_fn([s]))
        h_in = h
        last_attn = None
        for layer in model.global_layers:
            last_attn = attn_for_block(layer, h_in, dist_matrix, mask).detach()
            h_in = layer(h_in, dist_matrix, mask)
        head_avg = last_attn[0, :, :n_i, :n_i].mean(0).cpu().numpy()
        inc_def.append(float(head_avg[:, d_i].mean()))
        other = 0 if d_i != 0 else 1
        if n_i > other:
            inc_other.append(float(head_avg[:, other].mean()))
        # occlusion
        _, delta = occlusion_per_atom(model, s, nmean, nstd)
        adlt = np.abs(delta)
        delta_def.append(float(adlt[d_i]))
        other_arr = adlt.copy(); other_arr[d_i] = np.nan
        delta_other.append(float(np.nanmean(other_arr)))
        fraction_at_def.append(float(adlt[d_i] / max(adlt.sum(), 1e-9)))

    summary = {
        "n_samples_used": len(inc_def),
        "model": f"loho_{host} (trained without any {host} sample)",
        "test_set": f"{host} hosts (OOD samples; up to 80 sampled)",
        "incoming_attn_to_defect_mean": float(np.mean(inc_def)),
        "incoming_attn_to_random_other_mean": float(np.mean(inc_other)),
        "ratio_defect_over_other_attn": float(np.mean(inc_def) / max(np.mean(inc_other), 1e-9)),
        "occlusion_abs_delta_at_defect_mean": float(np.mean(delta_def)),
        "occlusion_abs_delta_at_other_mean": float(np.mean(delta_other)),
        "ratio_defect_over_other_delta": float(np.mean(delta_def) / max(np.mean(delta_other), 1e-9)),
        "fraction_attribution_at_defect_atom_mean": float(np.mean(fraction_at_def)),
        "fraction_attribution_at_defect_atom_std": float(np.std(fraction_at_def)),
    }
    print(json.dumps(summary, indent=2))
    out_path = RESULTS / f"loho_interp_check_{host}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
