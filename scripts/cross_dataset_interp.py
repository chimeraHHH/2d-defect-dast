"""Cross-dataset interpretability: does the defect-as-hub attention pattern
hold when applied to JARVIS vacancy structures?

For vacancy defects, the "defect site" is the nearest neighbor to the missing
atom. The question: does the model still pay disproportionate attention to
this site, even though it was trained on impurity (addition) defects?

Tests:
  1. Attention analysis on JARVIS-2D samples
  2. Occlusion attribution on JARVIS-2D samples
  3. Comparison with IMP2D in-distribution statistics

Output: results/cross_dataset_interp.json + figures
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn
from src.models import CrystalTransformer

RESULTS = ROOT / "results"
FIGURES = ROOT / "paper/figures"


def load_model(run_dir="baseline_h128_aug_long_safe"):
    ckpt_path = RESULTS / run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_kwargs = cfg.get("model_kwargs", {
        "atom_fea_len": 9, "hidden_dim": 128, "n_local_layers": 3,
        "n_global_layers": 2, "num_heads": 4,
    })
    model = CrystalTransformer(**model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"]


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


def analyse_dataset(model, dataset, nmean, nstd, n_max=60, label=""):
    """Run attention + occlusion analysis on dataset samples."""
    inc_def, inc_other = [], []
    delta_def, delta_other = [], []
    fraction_at_def = []
    n_used = 0

    for i in range(min(n_max, len(dataset))):
        s = dataset[i]
        if s["defect_mask"].sum().item() != 1:
            continue
        d_i = int(s["defect_mask"].argmax().item())
        n_i = s["num_atoms"]

        # Attention analysis
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

        # Occlusion analysis
        _, delta = occlusion_per_atom(model, s, nmean, nstd)
        adlt = np.abs(delta)
        delta_def.append(float(adlt[d_i]))
        other_arr = adlt.copy(); other_arr[d_i] = np.nan
        delta_other.append(float(np.nanmean(other_arr)))
        fraction_at_def.append(float(adlt[d_i] / max(adlt.sum(), 1e-9)))
        n_used += 1

        if (n_used) % 10 == 0:
            print(f"  [{label}] processed {n_used} samples...", flush=True)

    if not inc_def:
        return None

    return {
        "n_samples": n_used,
        "attn_to_defect_mean": float(np.mean(inc_def)),
        "attn_to_other_mean": float(np.mean(inc_other)),
        "attn_ratio": float(np.mean(inc_def) / max(np.mean(inc_other), 1e-9)),
        "occlusion_defect_mean": float(np.mean(delta_def)),
        "occlusion_other_mean": float(np.mean(delta_other)),
        "occlusion_ratio": float(np.mean(delta_def) / max(np.mean(delta_other), 1e-9)),
        "fraction_at_defect_mean": float(np.mean(fraction_at_def)),
        "fraction_at_defect_std": float(np.std(fraction_at_def)),
    }


def main():
    print("=" * 60)
    print("Cross-Dataset Interpretability Analysis")
    print("=" * 60)

    model, nmean, nstd = load_model()

    # Load datasets
    ds_2d = CrystalGraphDataset(ROOT / "data/processed/jarvis_2d.pkl")
    print(f"JARVIS-2D: {len(ds_2d)} samples")

    # Also load IMP2D test for reference
    from src.dataset import split_indices
    from torch.utils.data import Subset
    ds_imp2d = CrystalGraphDataset(ROOT / "data/processed/cleaned_dataset.pkl")
    _, _, test_idx = split_indices(len(ds_imp2d), 0.8, 0.1, 42)
    imp2d_test = Subset(ds_imp2d, test_idx)

    results = {}

    # IMP2D reference (subsample for speed)
    print("\n--- IMP2D test (reference, 50 samples) ---")
    res_imp = analyse_dataset(model, imp2d_test, nmean, nstd, n_max=50, label="IMP2D")
    if res_imp:
        results["IMP2D_test"] = res_imp
        print(f"  Attn ratio: {res_imp['attn_ratio']:.1f}×")
        print(f"  Occlusion fraction: {res_imp['fraction_at_defect_mean']:.1%}")

    # JARVIS-2D
    print("\n--- JARVIS-2D (vacancy, 60 samples) ---")
    res_j2d = analyse_dataset(model, ds_2d, nmean, nstd, n_max=60, label="JARVIS-2D")
    if res_j2d:
        results["JARVIS_2D"] = res_j2d
        print(f"  Attn ratio: {res_j2d['attn_ratio']:.1f}×")
        print(f"  Occlusion fraction: {res_j2d['fraction_at_defect_mean']:.1%}")

    # ---- Figures ----
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Attention ratio comparison
    ax = axes[0]
    datasets_plot = []
    ratios_attn = []
    ratios_occ = []
    colors = []
    if "IMP2D_test" in results:
        datasets_plot.append("IMP2D\n(in-dist)")
        ratios_attn.append(results["IMP2D_test"]["attn_ratio"])
        ratios_occ.append(results["IMP2D_test"]["fraction_at_defect_mean"] * 100)
        colors.append("steelblue")
    if "JARVIS_2D" in results:
        datasets_plot.append("JARVIS-2D\n(vacancy, OOD)")
        ratios_attn.append(results["JARVIS_2D"]["attn_ratio"])
        ratios_occ.append(results["JARVIS_2D"]["fraction_at_defect_mean"] * 100)
        colors.append("coral")

    bars = ax.bar(datasets_plot, ratios_attn, color=colors, edgecolor="black")
    for bar, val in zip(bars, ratios_attn):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}×", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attention Ratio (defect / other)")
    ax.set_title("Defect-as-Hub Attention Pattern")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Occlusion fraction comparison
    ax = axes[1]
    bars = ax.bar(datasets_plot, ratios_occ, color=colors, edgecolor="black")
    for bar, val in zip(bars, ratios_occ):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attribution at Defect Site (%)")
    ax.set_title("Occlusion Attribution Localization")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig_path = FIGURES / "fig_cross_dataset_interp.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {fig_path}")

    # Save
    out_path = RESULTS / "cross_dataset_interp.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("INTERPRETABILITY TRANSFER SUMMARY")
    print(f"{'='*60}")
    for name in ["IMP2D_test", "JARVIS_2D"]:
        if name in results:
            r = results[name]
            print(f"  {name}:")
            print(f"    Attention ratio:       {r['attn_ratio']:.1f}×")
            print(f"    Occlusion defect/other: {r['occlusion_ratio']:.1f}×")
            print(f"    Attribution at defect:  {r['fraction_at_defect_mean']:.1%} ± {r['fraction_at_defect_std']:.1%}")


if __name__ == "__main__":
    main()
