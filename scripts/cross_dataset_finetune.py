"""Fine-tune IMP2D-trained model on JARVIS vacancy data to measure transfer
learning efficiency.

Experiments:
  1. Few-shot fine-tune on JARVIS-2D (k=10,20,30,50 samples)
  2. Full fine-tune on JARVIS-3D (80/10/10 split, 30 epochs)
  3. Learning curves: how fast does the gap close?

The fine-tuning uses a lower learning rate (1e-4 vs 3e-3 for from-scratch)
and initialises from the IMP2D checkpoint.

Output: results/cross_dataset_finetune.json + learning curve figures
"""
from __future__ import annotations

import json
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
    return model, ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"]


def train_epoch(model, loader, optimizer, nmean, nstd, device):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        pred = model(batch)
        target_norm = (batch["target"] - nmean) / nstd
        loss = nn.functional.mse_loss(pred.squeeze(), target_norm)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
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


def finetune_experiment(model_init_fn, dataset, train_idx, val_idx, test_idx,
                        nmean, nstd, epochs=30, lr=1e-4, label=""):
    model = model_init_fn()
    model.to(DEVICE)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if val_idx else None
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False,
                            collate_fn=collate_fn, num_workers=0) if val_set else None
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6)

    history = {"train_loss": [], "val_mae": [], "test_mae": []}
    best_val_mae = float("inf")
    best_state = None

    t0 = time.time()
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, nmean, nstd, DEVICE)
        history["train_loss"].append(loss)

        if val_loader:
            val_mae, _, _, _ = evaluate(model, val_loader, nmean, nstd, DEVICE)
            history["val_mae"].append(val_mae)
            scheduler.step(val_mae)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            scheduler.step(loss)
            history["val_mae"].append(None)

        test_mae, test_rmse, _, _ = evaluate(model, test_loader, nmean, nstd, DEVICE)
        history["test_mae"].append(test_mae)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_str = f"val_MAE={val_mae:.3f}" if val_loader else "no_val"
            print(f"  [{label}] ep {epoch+1:3d}: loss={loss:.4f}, {val_str}, test_MAE={test_mae:.3f}")

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    final_mae, final_rmse, preds, targets = evaluate(model, test_loader, nmean, nstd, DEVICE)
    dt = time.time() - t0

    return {
        "test_MAE": final_mae,
        "test_RMSE": final_rmse,
        "best_val_MAE": best_val_mae if val_loader else None,
        "n_train": len(train_idx),
        "n_val": len(val_idx) if val_idx else 0,
        "n_test": len(test_idx),
        "epochs": epochs,
        "lr": lr,
        "time_s": dt,
        "history": history,
    }


def main():
    print("=" * 70)
    print("Cross-Dataset Fine-Tune Transfer Learning")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load datasets
    ds_2d = CrystalGraphDataset(ROOT / "data/processed/jarvis_2d.pkl")
    ds_3d = CrystalGraphDataset(ROOT / "data/processed/jarvis_3d.pkl")
    print(f"JARVIS-2D: {len(ds_2d)} samples")
    print(f"JARVIS-3D: {len(ds_3d)} samples")

    # Get pretrained model and normaliser
    _, nmean_imp, nstd_imp = load_pretrained()
    print(f"IMP2D normalizer: mean={nmean_imp:.4f}, std={nstd_imp:.4f}")

    results = {}

    # ================================================================
    # Exp 1: Few-shot fine-tune on JARVIS-2D
    # ================================================================
    print(f"\n{'='*60}")
    print("Experiment 1: Few-shot fine-tune on JARVIS-2D")
    print("Split: k train / 10 val / rest test")

    rng = random.Random(42)
    idx_2d = list(range(len(ds_2d)))
    rng.shuffle(idx_2d)

    # Fixed val/test: last 20 samples
    test_idx_2d = idx_2d[-20:]
    val_idx_2d = idx_2d[-30:-20]
    train_pool_2d = idx_2d[:-30]

    few_shot_results = {}
    for k in [5, 10, 20, 30]:
        if k > len(train_pool_2d):
            break
        train_k = train_pool_2d[:k]
        print(f"\n  --- k={k} ---")

        # Fine-tuned from IMP2D
        def make_ft():
            m, _, _ = load_pretrained()
            return m
        res_ft = finetune_experiment(
            make_ft, ds_2d, train_k, val_idx_2d, test_idx_2d,
            nmean_imp, nstd_imp, epochs=40, lr=5e-5, label=f"FT-k{k}")

        # From scratch (random init)
        def make_scratch():
            return CrystalTransformer(
                atom_fea_len=9, hidden_dim=128, n_local_layers=3,
                n_global_layers=2, num_heads=4)
        res_scratch = finetune_experiment(
            make_scratch, ds_2d, train_k, val_idx_2d, test_idx_2d,
            nmean_imp, nstd_imp, epochs=40, lr=1e-3, label=f"Scratch-k{k}")

        few_shot_results[f"k{k}"] = {
            "fine_tuned": res_ft,
            "from_scratch": res_scratch,
            "improvement_pct": (res_scratch["test_MAE"] - res_ft["test_MAE"]) /
                               res_scratch["test_MAE"] * 100,
        }
        print(f"    FT MAE={res_ft['test_MAE']:.3f}, Scratch MAE={res_scratch['test_MAE']:.3f}, "
              f"improvement={few_shot_results[f'k{k}']['improvement_pct']:.1f}%")

    results["jarvis_2d_few_shot"] = few_shot_results

    # ================================================================
    # Exp 2: Full fine-tune on JARVIS-3D (80/10/10)
    # ================================================================
    print(f"\n{'='*60}")
    print("Experiment 2: Full fine-tune on JARVIS-3D (80/10/10)")

    rng3d = random.Random(42)
    idx_3d = list(range(len(ds_3d)))
    rng3d.shuffle(idx_3d)
    n3d = len(idx_3d)
    n_train_3d = int(0.8 * n3d)
    n_val_3d = int(0.1 * n3d)
    train_idx_3d = idx_3d[:n_train_3d]
    val_idx_3d = idx_3d[n_train_3d:n_train_3d + n_val_3d]
    test_idx_3d = idx_3d[n_train_3d + n_val_3d:]
    print(f"  Split: {n_train_3d} train / {n_val_3d} val / {len(test_idx_3d)} test")

    # Fine-tuned
    def make_ft_3d():
        m, _, _ = load_pretrained()
        return m
    res_ft_3d = finetune_experiment(
        make_ft_3d, ds_3d, train_idx_3d, val_idx_3d, test_idx_3d,
        nmean_imp, nstd_imp, epochs=50, lr=5e-5, label="FT-3D")

    # From scratch
    def make_scratch_3d():
        return CrystalTransformer(
            atom_fea_len=9, hidden_dim=128, n_local_layers=3,
            n_global_layers=2, num_heads=4)
    res_scratch_3d = finetune_experiment(
        make_scratch_3d, ds_3d, train_idx_3d, val_idx_3d, test_idx_3d,
        nmean_imp, nstd_imp, epochs=50, lr=1e-3, label="Scratch-3D")

    results["jarvis_3d_full"] = {
        "fine_tuned": res_ft_3d,
        "from_scratch": res_scratch_3d,
        "improvement_pct": (res_scratch_3d["test_MAE"] - res_ft_3d["test_MAE"]) /
                           res_scratch_3d["test_MAE"] * 100,
    }
    print(f"\n  FT MAE={res_ft_3d['test_MAE']:.3f}, Scratch MAE={res_scratch_3d['test_MAE']:.3f}, "
          f"improvement={results['jarvis_3d_full']['improvement_pct']:.1f}%")

    # ================================================================
    # Exp 3: Data efficiency curve on JARVIS-3D
    # ================================================================
    print(f"\n{'='*60}")
    print("Experiment 3: Data efficiency curve on JARVIS-3D")

    efficiency_results = {}
    for frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        n_use = max(5, int(frac * n_train_3d))
        train_sub = train_idx_3d[:n_use]
        print(f"\n  --- frac={frac:.0%} ({n_use} samples) ---")

        def make_ft_sub():
            m, _, _ = load_pretrained()
            return m
        res = finetune_experiment(
            make_ft_sub, ds_3d, train_sub, val_idx_3d, test_idx_3d,
            nmean_imp, nstd_imp, epochs=30, lr=5e-5, label=f"FT-{frac:.0%}")

        efficiency_results[f"frac_{frac}"] = {
            "n_train": n_use,
            "test_MAE": res["test_MAE"],
            "test_RMSE": res["test_RMSE"],
        }
        print(f"    MAE={res['test_MAE']:.3f}")

    results["jarvis_3d_efficiency"] = efficiency_results

    # ================================================================
    # Generate figures
    # ================================================================
    print(f"\n{'='*60}")
    print("Generating figures...")
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Fig 1: Few-shot comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted([int(k[1:]) for k in few_shot_results.keys()])
    ft_maes = [few_shot_results[f"k{k}"]["fine_tuned"]["test_MAE"] for k in ks]
    sc_maes = [few_shot_results[f"k{k}"]["from_scratch"]["test_MAE"] for k in ks]
    x = np.arange(len(ks))
    w = 0.35
    bars1 = ax.bar(x - w/2, sc_maes, w, label="From scratch", color="lightcoral", edgecolor="black")
    bars2 = ax.bar(x + w/2, ft_maes, w, label="Fine-tuned (IMP2D)", color="steelblue", edgecolor="black")
    ax.set_xlabel("Training samples (k)")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("Few-Shot Transfer: JARVIS-2D Vacancies")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars2, ft_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", fontsize=9)
    fig.savefig(FIGURES / "fig_cross_dataset_fewshot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_cross_dataset_fewshot.png")

    # Fig 2: Data efficiency curve (JARVIS-3D)
    fig, ax = plt.subplots(figsize=(8, 5))
    fracs = sorted([float(k.split("_")[1]) for k in efficiency_results.keys()])
    n_trains = [efficiency_results[f"frac_{f}"]["n_train"] for f in fracs]
    eff_maes = [efficiency_results[f"frac_{f}"]["test_MAE"] for f in fracs]
    ax.plot(n_trains, eff_maes, "o-", color="steelblue", linewidth=2, markersize=8,
            label="Fine-tuned (IMP2D → JARVIS-3D)")
    ax.axhline(res_scratch_3d["test_MAE"], ls="--", c="lightcoral",
               label=f"From scratch (100%): {res_scratch_3d['test_MAE']:.3f}")
    ax.set_xlabel("Number of JARVIS-3D training samples")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("Data Efficiency: IMP2D Pre-training on JARVIS-3D")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for xi, yi in zip(n_trains, eff_maes):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    fig.savefig(FIGURES / "fig_cross_dataset_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_cross_dataset_efficiency.png")

    # Fig 3: Learning curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # JARVIS-3D learning curves
    ax = axes[0]
    if "jarvis_3d_full" in results:
        h_ft = results["jarvis_3d_full"]["fine_tuned"]["history"]["test_mae"]
        h_sc = results["jarvis_3d_full"]["from_scratch"]["history"]["test_mae"]
        ax.plot(range(1, len(h_ft)+1), h_ft, label="Fine-tuned", color="steelblue", linewidth=2)
        ax.plot(range(1, len(h_sc)+1), h_sc, label="From scratch", color="lightcoral", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test MAE (eV)")
        ax.set_title("JARVIS-3D: Fine-tune vs From-scratch")
        ax.legend()

    # JARVIS-2D few-shot (k=20)
    ax = axes[1]
    best_k = str(max(ks))
    if f"k{best_k}" in few_shot_results:
        h_ft = few_shot_results[f"k{best_k}"]["fine_tuned"]["history"]["test_mae"]
        h_sc = few_shot_results[f"k{best_k}"]["from_scratch"]["history"]["test_mae"]
        ax.plot(range(1, len(h_ft)+1), h_ft, label="Fine-tuned", color="steelblue", linewidth=2)
        ax.plot(range(1, len(h_sc)+1), h_sc, label="From scratch", color="lightcoral", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test MAE (eV)")
        ax.set_title(f"JARVIS-2D (k={best_k}): Fine-tune vs From-scratch")
        ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES / "fig_cross_dataset_learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_cross_dataset_learning_curves.png")

    # Save results (strip history for JSON brevity)
    results_clean = json.loads(json.dumps(results, default=str))
    out_path = RESULTS / "cross_dataset_finetune.json"
    with open(out_path, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nResults saved -> {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("\nFew-shot JARVIS-2D:")
    for k in ks:
        ft = few_shot_results[f"k{k}"]["fine_tuned"]["test_MAE"]
        sc = few_shot_results[f"k{k}"]["from_scratch"]["test_MAE"]
        imp = few_shot_results[f"k{k}"]["improvement_pct"]
        print(f"  k={k:3d}: FT={ft:.3f}, Scratch={sc:.3f}, improvement={imp:+.1f}%")

    print(f"\nFull JARVIS-3D:")
    print(f"  FT:      MAE={res_ft_3d['test_MAE']:.3f}")
    print(f"  Scratch: MAE={res_scratch_3d['test_MAE']:.3f}")
    print(f"  Improvement: {results['jarvis_3d_full']['improvement_pct']:.1f}%")


if __name__ == "__main__":
    main()
