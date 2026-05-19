#!/usr/bin/env python3
"""Attention and feature analysis for physical interpretability.

Analyses what the model has learned by extracting:
  1. Pooling attention weights: which atoms the model focuses on
  2. Defect vs host attention ratio: does the model upweight defects?
  3. JK layer weights: local vs global feature importance (if JK is used)
  4. Physics feature contributions: which Hume-Rothery descriptors matter
  5. Per-host error analysis: which host materials are hardest

Generates paper-quality figures and a LaTeX-ready analysis table.

Usage:
    python scripts/attention_analysis.py results/v3_deftype
    python scripts/attention_analysis.py results/v2_gated_pooling_s43 --device cuda
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits
from src.models import CrystalTransformerV2

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class Normalizer:
    def __init__(self, mean, std, transform="none"):
        self.mean, self.std, self.transform = mean, std, transform
    def denorm(self, t):
        t = t * self.std + self.mean
        if self.transform == "log":
            t = torch.sign(t) * torch.expm1(torch.abs(t))
        return t


def extract_attention_weights(model, batch, device):
    """Run forward pass and extract pooling attention weights."""
    model.eval()
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

    # Hook into pooling layer to capture attention weights
    attn_weights_store = {}

    def hook_fn(module, input_args, output):
        h, mask = input_args[0], input_args[1]
        attn_logits = module.attn_mlp(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_w = torch.softmax(attn_logits, dim=-1)
        attn_weights_store["pooling_attn"] = attn_w.detach().cpu()

    hook = None
    if hasattr(model, "pooling") and model.pooling is not None:
        hook = model.pooling.register_forward_hook(hook_fn)

    with torch.no_grad():
        out = model(batch_dev)
        if isinstance(out, tuple):
            out = out[0]

    if hook is not None:
        hook.remove()

    return attn_weights_store, out.detach().cpu()


def analyse_model(model_dir, device="cuda"):
    """Full analysis of a trained model."""
    model_dir = Path(model_dir)
    ckpt_path = model_dir / "best.pt"

    if not ckpt_path.exists():
        print("No best.pt found in %s" % model_dir)
        return

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "config" not in ckpt:
        print("Checkpoint format not supported for analysis")
        return

    cfg = ckpt["config"]
    norm_info = ckpt.get("normalizer", {})
    state_dict = ckpt["model"]

    normalizer = Normalizer(
        norm_info.get("mean", 0), norm_info.get("std", 1),
        transform=norm_info.get("transform", "none"),
    )

    # Build model
    model_kwargs = cfg.get("model_kwargs", {})
    model = CrystalTransformerV2(**model_kwargs)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    # Load dataset
    data_path = cfg.get("data_path", "data/processed/cleaned_dataset.pkl")
    dataset = CrystalGraphDataset(ROOT / data_path)
    _, _, test_set = make_splits(
        dataset,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=42,
    )

    test_loader = DataLoader(
        test_set, batch_size=32, shuffle=False, collate_fn=collate_fn,
    )

    # ── Collect predictions + attention ──────────────────────────────
    all_preds, all_targets = [], []
    all_defect_attn_ratios = []  # ratio of attn on defect vs non-defect atoms
    all_hosts = []
    all_dopants = []

    for batch in test_loader:
        attn_store, preds_norm = extract_attention_weights(model, batch, device)
        preds = normalizer.denorm(preds_norm).numpy()
        targets = batch["target"].numpy()
        all_preds.append(preds)
        all_targets.append(targets)

        # Analyze attention on defect vs host atoms
        if "pooling_attn" in attn_store:
            attn = attn_store["pooling_attn"]  # (B, N)
            defect_mask = batch.get("defect_mask")
            atom_mask = batch.get("atom_mask")
            if defect_mask is not None and atom_mask is not None:
                for i in range(attn.shape[0]):
                    m = atom_mask[i].bool()
                    d = defect_mask[i].bool() & m
                    h = (~defect_mask[i].bool()) & m
                    if d.sum() > 0 and h.sum() > 0:
                        defect_attn = attn[i][d].sum().item()
                        host_attn = attn[i][h].sum().item()
                        n_defect = d.sum().item()
                        n_host = h.sum().item()
                        # Normalized ratio: (attn_per_defect / attn_per_host)
                        ratio = (defect_attn / n_defect) / max(host_attn / n_host, 1e-8)
                        all_defect_attn_ratios.append(ratio)

        # Collect host/dopant info if available
        if "host_formula" in batch:
            all_hosts.extend(batch["host_formula"])
        if "dopant_symbol" in batch:
            all_dopants.extend(batch["dopant_symbol"])

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    errors = np.abs(preds - targets)

    # ── Print overall metrics ────────────────────────────────────────
    mae = errors.mean()
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    print("\n" + "=" * 70)
    print("MODEL: %s" % model_dir.name)
    print("=" * 70)
    print("Overall MAE: %.4f eV   RMSE: %.4f eV" % (mae, rmse))
    print("Epoch: %s" % ckpt.get("epoch", "?"))

    # ── Per-range analysis ───────────────────────────────────────────
    print("\nPer-range MAE:")
    for lo, hi in [(0, 2), (2, 5), (5, 7), (7, 25)]:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            r_mae = errors[mask].mean()
            r_rmse = np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2))
            bias = (preds[mask] - targets[mask]).mean()
            print("  [%d,%d): MAE=%.4f  RMSE=%.4f  bias=%+.4f  n=%d"
                  % (lo, hi, r_mae, r_rmse, bias, mask.sum()))

    # ── Attention analysis ───────────────────────────────────────────
    if all_defect_attn_ratios:
        ratios = np.array(all_defect_attn_ratios)
        print("\nAttention defect/host ratio:")
        print("  Mean: %.2fx (defect atoms get %.1fx more attention per atom)"
              % (ratios.mean(), ratios.mean()))
        print("  Median: %.2fx" % np.median(ratios))
        print("  Std: %.2f" % ratios.std())

        # Correlation with error
        if len(ratios) == len(errors):
            corr = np.corrcoef(ratios, errors)[0, 1]
            print("  Correlation(attn_ratio, |error|): %.3f" % corr)

    # ── JK weights analysis ──────────────────────────────────────────
    if hasattr(model, "use_jk_aggregation") and model.use_jk_aggregation:
        jk_w = torch.softmax(model.jk_weights, dim=0).detach().cpu().numpy()
        print("\nJK layer weights (softmax):")
        labels = ["Local out"] + ["Global %d" % (i+1) for i in range(len(jk_w)-1)]
        for lbl, w in zip(labels, jk_w):
            bar = "#" * int(w * 40)
            print("  %-12s: %.3f  %s" % (lbl, w, bar))

    # ── Physics module analysis ──────────────────────────────────────
    if hasattr(model, "physics_module") and model.physics_module is not None:
        # Check which physics features have the largest projection weights
        proj_w = model.physics_module.proj[0].weight.detach().cpu().numpy()  # first linear
        feature_names = ["Size mismatch", "EN difference", "IE ratio",
                        "EA difference", "Valence mismatch", "Period distance"]
        importances = np.abs(proj_w).sum(axis=0)  # sum of absolute weights per input feature
        importances = importances / importances.sum()
        print("\nPhysics feature importance (by projection weights):")
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            bar = "#" * int(imp * 40)
            print("  %-18s: %.3f  %s" % (name, imp, bar))

    # ── Per-host error analysis ──────────────────────────────────────
    if all_hosts:
        host_errors = defaultdict(list)
        host_targets = defaultdict(list)
        for h, e, t in zip(all_hosts, errors, targets):
            host_errors[h].append(e)
            host_targets[h].append(t)

        print("\nPer-host MAE (top 10 worst):")
        host_stats = []
        for h in host_errors:
            h_mae = np.mean(host_errors[h])
            h_count = len(host_errors[h])
            h_mean_target = np.mean(host_targets[h])
            host_stats.append((h, h_mae, h_count, h_mean_target))

        host_stats.sort(key=lambda x: -x[1])
        print("  %-15s %8s %6s %10s" % ("Host", "MAE", "Count", "Mean Ef"))
        for h, h_mae, h_count, h_mean_t in host_stats[:10]:
            print("  %-15s %8.4f %6d %10.3f" % (h, h_mae, h_count, h_mean_t))

        print("\nPer-host MAE (top 5 best):")
        for h, h_mae, h_count, h_mean_t in host_stats[-5:]:
            print("  %-15s %8.4f %6d %10.3f" % (h, h_mae, h_count, h_mean_t))

    # ── Generate figures ─────────────────────────────────────────────
    if HAS_MPL:
        fig_dir = model_dir / "analysis_figures"
        fig_dir.mkdir(exist_ok=True)

        # 1. Prediction vs True scatter
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = []
        for t in targets:
            if t < 2: colors.append("#2ecc71")
            elif t < 5: colors.append("#3498db")
            elif t < 7: colors.append("#f39c12")
            else: colors.append("#e74c3c")
        ax.scatter(targets, preds, c=colors, alpha=0.5, s=15, edgecolors="none")
        lims = [min(targets.min(), preds.min()) - 0.5,
                max(targets.max(), preds.max()) + 0.5]
        ax.plot(lims, lims, "k--", alpha=0.5, lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("DFT Formation Energy (eV)")
        ax.set_ylabel("Predicted Formation Energy (eV)")
        ax.set_title("%s: MAE=%.4f eV" % (model_dir.name, mae))
        fig.savefig(fig_dir / "pred_vs_true.png")
        plt.close()

        # 2. Error distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(errors, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(mae, color="red", ls="--", label="MAE=%.4f" % mae)
        ax.set_xlabel("Absolute Error (eV)")
        ax.set_ylabel("Count")
        ax.set_title("Error Distribution — %s" % model_dir.name)
        ax.legend()
        fig.savefig(fig_dir / "error_distribution.png")
        plt.close()

        # 3. Attention ratio histogram (if available)
        if all_defect_attn_ratios:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ratios, bins=50, color="#9b59b6", alpha=0.7, edgecolor="white")
            ax.axvline(ratios.mean(), color="red", ls="--",
                      label="Mean=%.1fx" % ratios.mean())
            ax.set_xlabel("Defect/Host Attention Ratio")
            ax.set_ylabel("Count")
            ax.set_title("Defect Attention Focus — %s" % model_dir.name)
            ax.legend()
            fig.savefig(fig_dir / "attention_ratio.png")
            plt.close()

        print("\nFigures saved to %s/" % fig_dir)

    return {
        "name": model_dir.name,
        "mae": mae,
        "rmse": rmse,
        "preds": preds,
        "targets": targets,
        "errors": errors,
        "defect_attn_ratios": np.array(all_defect_attn_ratios) if all_defect_attn_ratios else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Model result directories")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = []
    for d in args.dirs:
        r = analyse_model(d, device=args.device)
        if r:
            results.append(r)

    if len(results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON RANKING")
        print("=" * 70)
        for i, r in enumerate(sorted(results, key=lambda x: x["mae"])):
            print("  %d. %s: MAE=%.4f  RMSE=%.4f"
                  % (i + 1, r["name"], r["mae"], r["rmse"]))


if __name__ == "__main__":
    main()
