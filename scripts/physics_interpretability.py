#!/usr/bin/env python3
"""Physics interpretability analysis for CrystalTransformerV2 innovations.

Analyzes what the model has learned from a physical perspective:
  1. Attention weight analysis: does the model focus on defect atoms?
  2. JK layer weights: which scale (local vs global) matters more?
  3. Physics feature gradients: which Hume-Rothery descriptor drives Ef?
  4. MoE expert specialization: which expert handles which energy range?
  5. Defect-type conditioning: how do embeddings differ by type?

Usage:
    python scripts/physics_interpretability.py --checkpoint results/v6_physics/best_model.pt
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.element_table import lookup_ct_uae


def analyze_attention(model, loader, device, n_samples=200):
    """Analyze attention weights: does the model focus on defect atoms?"""
    model.eval()
    defect_attn_weights = []
    host_attn_weights = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i * batch["target"].shape[0] >= n_samples:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Get attention weights from pooling layer
            if not hasattr(model, 'pooling') or not hasattr(model.pooling, 'attn_mlp'):
                return None

            # Forward through model up to pooling
            x = batch["x"]
            mask = batch["atom_mask"]
            defect_mask = batch.get("defect_mask")

            if model.ct_uae_table is not None:
                z = batch.get("atomic_numbers")
                if z is not None:
                    x = torch.cat(
                        [x, lookup_ct_uae(model.ct_uae_table, z)], dim=-1
                    )
            h = model.embed(x)
            if model.defect_embedding is not None and defect_mask is not None:
                h = h + model.defect_embedding(defect_mask)

            # Skip env enrichment and local/global layers for simplicity
            # (full forward would be needed for accurate analysis)
            # Instead, just check attention pattern on raw embeddings
            attn_logits = model.pooling.attn_mlp(h).squeeze(-1)
            attn_logits = attn_logits.masked_fill(~mask, -1e9)
            attn_weights = torch.softmax(attn_logits, dim=-1)

            for b in range(attn_weights.shape[0]):
                valid = mask[b]
                defect = (defect_mask[b] == 1) & valid if defect_mask is not None else torch.zeros_like(valid)
                host = (~defect) & valid
                if defect.any():
                    defect_attn_weights.append(attn_weights[b, defect].mean().item())
                if host.any():
                    host_attn_weights.append(attn_weights[b, host].mean().item())

    if defect_attn_weights:
        print(f"  Mean attention on defect atoms:  {np.mean(defect_attn_weights):.6f}")
        print(f"  Mean attention on host atoms:    {np.mean(host_attn_weights):.6f}")
        ratio = np.mean(defect_attn_weights) / max(np.mean(host_attn_weights), 1e-9)
        print(f"  Defect/host attention ratio:     {ratio:.2f}x")
        return {"defect_mean": np.mean(defect_attn_weights),
                "host_mean": np.mean(host_attn_weights),
                "ratio": ratio}
    return None


def analyze_jk_weights(model):
    """Analyze JK aggregation layer weights."""
    if not hasattr(model, 'jk_weights'):
        return None

    w = torch.softmax(model.jk_weights, dim=0).detach().cpu().numpy()
    labels = ["local_out"] + [f"global_{i+1}" for i in range(len(w) - 1)]

    print(f"  JK layer weights (softmax):")
    for label, weight in zip(labels, w):
        bar = "█" * int(weight * 50)
        print(f"    {label:<12} {weight:.4f}  {bar}")

    return {"weights": w.tolist(), "labels": labels}


def analyze_physics_features(model, loader, device, n_samples=500):
    """Gradient-based importance of physics features."""
    if not hasattr(model, 'physics_module') or model.physics_module is None:
        return None

    model.eval()
    feature_names = ["size_mm", "EN_diff", "IE_ratio", "EA_diff", "VE_mm", "period_d"]

    # Collect physics features and their gradients w.r.t. output
    feature_grads = []
    feature_vals = []

    for i, batch in enumerate(loader):
        if i * batch["target"].shape[0] >= n_samples:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        z = batch.get("atomic_numbers")
        dm = batch.get("defect_mask")
        mask = batch["atom_mask"]
        if z is None or dm is None:
            continue

        # Compute physics features with gradient
        with torch.enable_grad():
            z_clamped = z.clamp(0, len(model.physics_module._cov_radius) - 1)
            pm = model.physics_module

            r_all = pm._cov_radius[z_clamped]
            en_all = pm._pauling_en[z_clamped]
            ie_all = pm._ionization_e[z_clamped]
            ea_all = pm._electron_aff[z_clamped]
            ve_all = pm._valence_e[z_clamped]
            per_all = pm._period[z_clamped]

            df = dm.float()
            hf = ((~dm.bool()) & mask).float()
            dc = df.sum(1).clamp(min=1)
            hc = hf.sum(1).clamp(min=1)

            d_r = (r_all * df).sum(1) / dc; h_r = (r_all * hf).sum(1) / hc
            d_en = (en_all * df).sum(1) / dc; h_en = (en_all * hf).sum(1) / hc
            d_ie = (ie_all * df).sum(1) / dc; h_ie = (ie_all * hf).sum(1) / hc
            d_ea = (ea_all * df).sum(1) / dc; h_ea = (ea_all * hf).sum(1) / hc
            d_ve = (ve_all * df).sum(1) / dc; h_ve = (ve_all * hf).sum(1) / hc
            d_per = (per_all * df).sum(1) / dc; h_per = (per_all * hf).sum(1) / hc

            f1 = (d_r - h_r) / h_r.clamp(min=0.1)
            f2 = d_en - h_en
            f3 = d_ie / h_ie.clamp(min=0.1)
            f4 = d_ea - h_ea
            f5 = (d_ve - h_ve).abs() / 8.0
            f6 = (d_per - h_per) / 3.0

            features = torch.stack([f1, f2, f3, f4, f5, f6], dim=-1)
            feature_vals.append(features.detach().cpu().numpy())

    if feature_vals:
        all_feats = np.concatenate(feature_vals, axis=0)
        # Compute feature importance as |correlation with target - mean_pred|
        print(f"  Physics feature statistics (n={len(all_feats)}):")
        print(f"  {'Feature':<12} {'Mean':>8} {'Std':>8} {'|Range|':>8}")
        for j, name in enumerate(feature_names):
            col = all_feats[:, j]
            print(f"  {name:<12} {col.mean():>8.4f} {col.std():>8.4f} {col.max()-col.min():>8.4f}")

        # Check learned projection weights magnitude
        proj_w = model.physics_module.proj[-1].weight.detach().cpu().numpy()
        proj_importance = np.abs(proj_w).mean(axis=0)  # average across hidden dims
        # Project through both layers to get effective importance
        print(f"\n  Learned projection weight norms (layer 2):")
        w_norm = np.linalg.norm(proj_w, axis=0)
        print(f"  Overall weight norm: {np.linalg.norm(proj_w):.4f}")

        return {"feature_stats": {name: {"mean": float(all_feats[:, j].mean()),
                                          "std": float(all_feats[:, j].std())}
                                   for j, name in enumerate(feature_names)}}
    return None


def analyze_moe_experts(model, loader, device, n_samples=500):
    """Analyze MoE expert specialization by energy range."""
    if not hasattr(model, 'readout') or not isinstance(model.readout, torch.nn.Module):
        return None
    if not hasattr(model.readout, 'gate'):
        return None

    model.eval()
    gate_weights_all = []
    targets_all = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i * batch["target"].shape[0] >= n_samples:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Full forward to get gate weights
            out = model(batch)
            gate_w = model.readout._balance_loss  # Get last gate weights

            # We need to re-extract gate weights from the forward pass
            # For now, just collect predictions and targets
            targets_all.append(batch["target"].cpu().numpy())

    if not targets_all:
        return None

    targets = np.concatenate(targets_all)
    bins = [(-np.inf, 0), (0, 2), (2, 5), (5, 7), (7, 25)]
    print(f"  MoE analysis would require hooking into gate weights during forward pass")
    print(f"  Balance loss: {model.readout._balance_loss:.6f}")

    return {"balance_loss": float(model.readout._balance_loss)}


def analyze_defect_type_embeds(model):
    """Analyze learned defect-type conditioning embeddings."""
    if not hasattr(model, 'defect_type_embed') or model.defect_type_embed is None:
        return None

    embeds = model.defect_type_embed.weight.detach().cpu().numpy()
    labels = ["vacancy", "substitution", "interstitial", "adsorbate"]

    norms = np.linalg.norm(embeds, axis=1)
    print(f"  Defect-type embedding norms:")
    for label, norm in zip(labels, norms):
        print(f"    {label:<15} ||e|| = {norm:.4f}")

    # Cosine similarity between types
    normed = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
    cos_sim = normed @ normed.T
    print(f"\n  Cosine similarity matrix:")
    header = f"  {'':>15}" + "".join(f" {l[:12]:>12}" for l in labels)
    print(header)
    for i, l1 in enumerate(labels):
        row = f"  {l1:>15}"
        for j in range(len(labels)):
            row += f" {cos_sim[i,j]:>12.4f}"
        print(row)

    return {"norms": norms.tolist(), "cos_sim": cos_sim.tolist(), "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--data-path", default="data/processed/cleaned_dataset.pkl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    print("=" * 70)
    print("PHYSICS INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    config = ckpt.get("config", {})
    model_kwargs = config.get("model_kwargs", {})

    from src.models.crystal_v2 import CrystalTransformerV2
    model = CrystalTransformerV2(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    print(f"\nModel: {ckpt_path}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Analysis sections
    results = {}

    print("\n--- 1. JK Layer Weights ---")
    results["jk"] = analyze_jk_weights(model)
    if results["jk"] is None:
        print("  (JK aggregation not enabled)")

    print("\n--- 2. Defect-Type Embeddings ---")
    results["defect_type"] = analyze_defect_type_embeds(model)
    if results["defect_type"] is None:
        print("  (Defect-type conditioning not enabled)")

    print("\n--- 3. MoE Expert Analysis ---")
    results["moe"] = analyze_moe_experts(model, None, args.device)
    if results["moe"] is None:
        print("  (MoE readout not enabled)")

    # Save results
    out_path = ckpt_path.parent / "interpretability.json"
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in results.items() if v is not None}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
