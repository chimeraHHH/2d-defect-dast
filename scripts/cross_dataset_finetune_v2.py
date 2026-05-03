"""Improved cross-dataset fine-tuning with better hyperparameters.

Improvements over v1:
  1. Warmup + cosine annealing LR schedule
  2. Lower LR for backbone, higher for readout head
  3. Proper train/val/test normalizer computed from JARVIS data
  4. Longer training with early stopping
  5. Multiple random seeds for stability

Output: results/cross_dataset_finetune_v2.json
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn
from src.models import CrystalTransformer

RESULTS = ROOT / "results"
FIGURES = ROOT / "paper/figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained(run_dir="baseline_h128_aug_long_safe"):
    ckpt_path = RESULTS / run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_kwargs = cfg.get("model_kwargs", {
        "atom_fea_len": 9, "hidden_dim": 128, "n_local_layers": 3,
        "n_global_layers": 2, "num_heads": 4,
    })
    model = CrystalTransformer(**model_kwargs)
    model.load_state_dict(ckpt["model"])
    return model, model_kwargs


def compute_normalizer(dataset, indices):
    targets = [dataset.data[i]["target"] for i in indices]
    return float(np.mean(targets)), float(np.std(targets))


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, nmean, nstd, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        pred = model(batch)
        target_norm = (batch["target"] - nmean) / nstd
        loss = nn.functional.huber_loss(pred.squeeze(), target_norm, delta=1.0)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(batch["target"])
        n += len(batch["target"])
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, nmean, nstd, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        pred = model(batch)
        pred_real = pred.cpu().numpy() * nstd + nmean
        preds.extend(pred_real.flatten().tolist())
        targets.extend(batch["target"].cpu().numpy().tolist())
    preds, targets = np.array(preds), np.array(targets)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets)**2)))
    return mae, rmse, preds, targets


def finetune_run(dataset, train_idx, val_idx, test_idx,
                 pretrained=True, epochs=60, lr_backbone=2e-5, lr_head=5e-4,
                 seed=42, label=""):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    nmean, nstd = compute_normalizer(dataset, train_idx)

    if pretrained:
        model, _ = load_pretrained()
    else:
        model = CrystalTransformer(atom_fea_len=9, hidden_dim=128,
                                    n_local_layers=3, n_global_layers=2, num_heads=4)
    model.to(DEVICE)

    # Differential learning rates
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "readout" in name or "fc_out" in name or "output" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if pretrained:
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32,
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32,
                             shuffle=False, collate_fn=collate_fn, num_workers=0)

    total_steps = epochs * len(train_loader)
    warmup_steps = min(total_steps // 10, 50)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_mae": [], "test_mae": []}

    t0 = time.time()
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, scheduler, nmean, nstd, DEVICE)
        val_mae, _, _, _ = evaluate(model, val_loader, nmean, nstd, DEVICE)
        test_mae, _, _, _ = evaluate(model, test_loader, nmean, nstd, DEVICE)

        history["train_loss"].append(loss)
        history["val_mae"].append(val_mae)
        history["test_mae"].append(test_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{label}] ep {epoch+1:3d}: loss={loss:.4f}, "
                  f"val={val_mae:.3f}, test={test_mae:.3f}", flush=True)

        if patience_counter >= 15:
            print(f"  [{label}] early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    final_mae, final_rmse, preds, targets = evaluate(model, test_loader, nmean, nstd, DEVICE)
    dt = time.time() - t0

    return {
        "test_MAE": final_mae,
        "test_RMSE": final_rmse,
        "best_val_MAE": best_val_mae,
        "n_train": len(train_idx),
        "epochs_run": len(history["train_loss"]),
        "time_s": dt,
        "history": history,
    }


def main():
    print("=" * 60)
    print("Cross-Dataset Fine-Tune v2 (improved)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    ds_2d = CrystalGraphDataset(ROOT / "data/processed/jarvis_2d.pkl")
    ds_3d = CrystalGraphDataset(ROOT / "data/processed/jarvis_3d.pkl")
    print(f"JARVIS-2D: {len(ds_2d)}, JARVIS-3D: {len(ds_3d)}")

    results = {}

    # ---- JARVIS-2D few-shot with 3 seeds ----
    print(f"\n{'='*60}")
    print("Exp 1: JARVIS-2D few-shot (3 seeds)")

    rng = random.Random(42)
    idx_2d = list(range(len(ds_2d)))
    rng.shuffle(idx_2d)
    test_idx_2d = idx_2d[-20:]
    val_idx_2d = idx_2d[-30:-20]
    train_pool_2d = idx_2d[:-30]

    few_shot_results = {}
    for k in [10, 20, 30]:
        if k > len(train_pool_2d):
            break
        train_k = train_pool_2d[:k]
        seed_results_ft = []
        seed_results_sc = []

        for seed in [42, 123, 456]:
            print(f"\n  --- k={k}, seed={seed} ---")
            res_ft = finetune_run(ds_2d, train_k, val_idx_2d, test_idx_2d,
                                  pretrained=True, epochs=60,
                                  lr_backbone=2e-5, lr_head=5e-4,
                                  seed=seed, label=f"FT-k{k}-s{seed}")
            seed_results_ft.append(res_ft["test_MAE"])

            res_sc = finetune_run(ds_2d, train_k, val_idx_2d, test_idx_2d,
                                  pretrained=False, epochs=60,
                                  seed=seed, label=f"SC-k{k}-s{seed}")
            seed_results_sc.append(res_sc["test_MAE"])

        ft_mean = float(np.mean(seed_results_ft))
        ft_std = float(np.std(seed_results_ft))
        sc_mean = float(np.mean(seed_results_sc))
        sc_std = float(np.std(seed_results_sc))
        improvement = (sc_mean - ft_mean) / sc_mean * 100

        few_shot_results[f"k{k}"] = {
            "ft_MAE_mean": ft_mean, "ft_MAE_std": ft_std,
            "sc_MAE_mean": sc_mean, "sc_MAE_std": sc_std,
            "improvement_pct": improvement,
            "ft_seeds": seed_results_ft,
            "sc_seeds": seed_results_sc,
        }
        print(f"  k={k}: FT={ft_mean:.3f}±{ft_std:.3f}, SC={sc_mean:.3f}±{sc_std:.3f}, "
              f"imp={improvement:+.1f}%")

    results["jarvis_2d_few_shot_v2"] = few_shot_results

    # ---- JARVIS-3D full with 3 seeds ----
    print(f"\n{'='*60}")
    print("Exp 2: JARVIS-3D full (3 seeds)")

    rng3d = random.Random(42)
    idx_3d = list(range(len(ds_3d)))
    rng3d.shuffle(idx_3d)
    n3d = len(idx_3d)
    n_train_3d = int(0.8 * n3d)
    n_val_3d = int(0.1 * n3d)
    train_idx_3d = idx_3d[:n_train_3d]
    val_idx_3d = idx_3d[n_train_3d:n_train_3d + n_val_3d]
    test_idx_3d = idx_3d[n_train_3d + n_val_3d:]

    seed_ft_3d = []
    seed_sc_3d = []
    for seed in [42, 123, 456]:
        print(f"\n  --- 3D full, seed={seed} ---")
        res_ft = finetune_run(ds_3d, train_idx_3d, val_idx_3d, test_idx_3d,
                              pretrained=True, epochs=80,
                              lr_backbone=2e-5, lr_head=5e-4,
                              seed=seed, label=f"FT-3D-s{seed}")
        seed_ft_3d.append(res_ft["test_MAE"])

        res_sc = finetune_run(ds_3d, train_idx_3d, val_idx_3d, test_idx_3d,
                              pretrained=False, epochs=80,
                              seed=seed, label=f"SC-3D-s{seed}")
        seed_sc_3d.append(res_sc["test_MAE"])

    ft3d_mean = float(np.mean(seed_ft_3d))
    ft3d_std = float(np.std(seed_ft_3d))
    sc3d_mean = float(np.mean(seed_sc_3d))
    sc3d_std = float(np.std(seed_sc_3d))
    results["jarvis_3d_full_v2"] = {
        "ft_MAE_mean": ft3d_mean, "ft_MAE_std": ft3d_std,
        "sc_MAE_mean": sc3d_mean, "sc_MAE_std": sc3d_std,
        "improvement_pct": (sc3d_mean - ft3d_mean) / sc3d_mean * 100,
        "ft_seeds": seed_ft_3d, "sc_seeds": seed_sc_3d,
    }
    print(f"\n  3D full: FT={ft3d_mean:.3f}±{ft3d_std:.3f}, "
          f"SC={sc3d_mean:.3f}±{sc3d_std:.3f}")

    # ---- Save ----
    out_path = RESULTS / "cross_dataset_finetune_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved -> {out_path}")

    # ---- Summary figure ----
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Few-shot
    ax = axes[0]
    ks = sorted([int(k[1:]) for k in few_shot_results.keys()])
    ft_means = [few_shot_results[f"k{k}"]["ft_MAE_mean"] for k in ks]
    ft_stds = [few_shot_results[f"k{k}"]["ft_MAE_std"] for k in ks]
    sc_means = [few_shot_results[f"k{k}"]["sc_MAE_mean"] for k in ks]
    sc_stds = [few_shot_results[f"k{k}"]["sc_MAE_std"] for k in ks]
    x = np.arange(len(ks))
    w = 0.35
    ax.bar(x - w/2, sc_means, w, yerr=sc_stds, label="From scratch",
           color="lightcoral", edgecolor="black", capsize=3)
    ax.bar(x + w/2, ft_means, w, yerr=ft_stds, label="Fine-tuned (IMP2D)",
           color="steelblue", edgecolor="black", capsize=3)
    ax.set_xlabel("Training samples (k)")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("Few-Shot Transfer: JARVIS-2D (3 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 3D full
    ax = axes[1]
    labels = ["From scratch", "Fine-tuned\n(IMP2D)"]
    means = [sc3d_mean, ft3d_mean]
    stds = [sc3d_std, ft3d_std]
    colors = ["lightcoral", "steelblue"]
    bars = ax.bar(labels, means, yerr=stds, color=colors,
                  edgecolor="black", capsize=5)
    for bar, val, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("JARVIS-3D Full (3 seeds)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig_cross_dataset_fewshot_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_cross_dataset_fewshot_v2.png")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY (v2, with error bars)")
    print(f"{'='*60}")
    for k in ks:
        r = few_shot_results[f"k{k}"]
        print(f"  k={k:3d}: FT={r['ft_MAE_mean']:.3f}±{r['ft_MAE_std']:.3f}, "
              f"SC={r['sc_MAE_mean']:.3f}±{r['sc_MAE_std']:.3f}, "
              f"imp={r['improvement_pct']:+.1f}%")
    print(f"  3D:    FT={ft3d_mean:.3f}±{ft3d_std:.3f}, "
          f"SC={sc3d_mean:.3f}±{sc3d_std:.3f}")


if __name__ == "__main__":
    main()
