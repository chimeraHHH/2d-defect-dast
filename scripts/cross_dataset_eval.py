"""Cross-dataset transfer validation: evaluate IMP2D-trained models on JARVIS
vacancy data (zero-shot) and measure transfer gaps.

Three levels of transfer:
  1. IMP2D → JARVIS-2D: same dimensionality, different defect type + DFT code
  2. IMP2D → JARVIS-3D: different dimensionality + defect type + DFT code
  3. Fine-tuned: small JARVIS subset → rest of JARVIS

Output: results/cross_dataset_eval.json + bar chart figure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn
from src.models import CrystalTransformer

RESULTS = ROOT / "results"
FIGURES = ROOT / "paper/figures"


def load_model(run_dir: str, device="cpu"):
    ckpt_path = RESULTS / run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_kwargs = cfg.get("model_kwargs", {
        "node_dim": 9,
        "hidden_dim": 128,
        "n_local_layers": 3,
        "n_global_layers": 2,
        "num_heads": 4,
        "num_rbf": 64,
        "cutoff": 5.0,
    })
    model = CrystalTransformer(**model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval()
    nmean = ckpt["normalizer"]["mean"]
    nstd = ckpt["normalizer"]["std"]
    return model, nmean, nstd


def evaluate_on_dataset(model, dataset, nmean, nstd, batch_size=32):
    """Run model on every sample, return (predictions, targets, metadata)."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(batch)
            pred_real = pred.cpu().numpy() * nstd + nmean
            all_preds.extend(pred_real.tolist())
            all_targets.extend((batch["target"].cpu().numpy() * 1.0).tolist())

    # targets from dataset are already in real eV (not normalised)
    # but model expects normalised targets during training
    # Actually, targets in our pipeline are stored as raw eV in the pickle,
    # and normalisation happens in the training loop. The DataLoader returns
    # raw eV values. The model output is normalised, so we denormalise it.
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    return preds, targets


def evaluate_ensemble(run_dirs, dataset, batch_size=32):
    """Run multiple models and return ensemble mean + std."""
    all_model_preds = []
    for run_dir in run_dirs:
        model, nmean, nstd = load_model(run_dir)
        preds, targets = evaluate_on_dataset(model, dataset, nmean, nstd, batch_size)
        all_model_preds.append(preds)

    all_model_preds = np.array(all_model_preds)
    ens_mean = all_model_preds.mean(axis=0)
    ens_std = all_model_preds.std(axis=0)
    return ens_mean, ens_std, targets


def compute_metrics(preds, targets):
    err = preds - targets
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    r2 = float(1 - np.sum(err**2) / np.sum((targets - targets.mean())**2))
    median_ae = float(np.median(np.abs(err)))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MedianAE": median_ae, "N": len(targets)}


def main():
    print("=" * 70)
    print("Cross-Dataset Transfer Validation")
    print("IMP2D-trained model → JARVIS vacancy data")
    print("=" * 70)

    # Load external datasets
    jarvis_2d_path = ROOT / "data/processed/jarvis_2d.pkl"
    jarvis_3d_path = ROOT / "data/processed/jarvis_3d.pkl"

    if not jarvis_2d_path.exists():
        print("Run scripts/prepare_jarvis.py first!")
        return

    ds_2d = CrystalGraphDataset(jarvis_2d_path)
    print(f"JARVIS-2D: {len(ds_2d)} samples loaded")

    ds_3d = CrystalGraphDataset(jarvis_3d_path)
    print(f"JARVIS-3D: {len(ds_3d)} samples loaded")

    # Available model runs for ensemble
    single_run = "baseline_h128_aug_long_safe"
    ensemble_runs = [
        "baseline_h128_aug_long_safe",
    ]
    # Check for seed variants
    for seed in [0, 1, 2]:
        seed_dir = f"baseline_h128_aug_long_safe_seed{seed}"
        if (RESULTS / seed_dir / "best.pt").exists():
            ensemble_runs.append(seed_dir)
    # Check for xlong variants
    for suffix in ["", "_seed0", "_seed1", "_seed2"]:
        xlong_dir = f"baseline_h128_aug_xlong_safe{suffix}"
        if (RESULTS / xlong_dir / "best.pt").exists():
            ensemble_runs.append(xlong_dir)

    print(f"\nAvailable ensemble members: {len(ensemble_runs)}")
    for r in ensemble_runs:
        print(f"  - {r}")

    # Also load IMP2D test set for baseline comparison
    imp2d_path = ROOT / "data/processed/cleaned_dataset.pkl"
    if imp2d_path.exists():
        ds_imp2d = CrystalGraphDataset(imp2d_path)
        from src.dataset import split_indices
        _, _, test_idx = split_indices(len(ds_imp2d), 0.8, 0.1, 42)
        from torch.utils.data import Subset
        imp2d_test = Subset(ds_imp2d, test_idx)
        print(f"IMP2D test set: {len(imp2d_test)} samples (for reference)")
    else:
        imp2d_test = None

    results = {}

    # ---- 1. Single model evaluation ----
    print(f"\n{'='*60}")
    print(f"[1] Single model zero-shot: {single_run}")
    model, nmean, nstd = load_model(single_run)

    if imp2d_test is not None:
        print("  Evaluating on IMP2D test set (reference)...")
        preds_imp, targets_imp = evaluate_on_dataset(model, imp2d_test, nmean, nstd)
        m_imp = compute_metrics(preds_imp, targets_imp)
        print(f"    IMP2D test: MAE={m_imp['MAE']:.3f}, RMSE={m_imp['RMSE']:.3f}, R²={m_imp['R2']:.3f}")
        results["imp2d_test_single"] = m_imp

    print("  Evaluating on JARVIS-2D (zero-shot)...")
    preds_j2d, targets_j2d = evaluate_on_dataset(model, ds_2d, nmean, nstd)
    m_j2d = compute_metrics(preds_j2d, targets_j2d)
    print(f"    JARVIS-2D: MAE={m_j2d['MAE']:.3f}, RMSE={m_j2d['RMSE']:.3f}, R²={m_j2d['R2']:.3f}")
    results["jarvis_2d_single"] = m_j2d

    print("  Evaluating on JARVIS-3D (zero-shot)...")
    preds_j3d, targets_j3d = evaluate_on_dataset(model, ds_3d, nmean, nstd)
    m_j3d = compute_metrics(preds_j3d, targets_j3d)
    print(f"    JARVIS-3D: MAE={m_j3d['MAE']:.3f}, RMSE={m_j3d['RMSE']:.3f}, R²={m_j3d['R2']:.3f}")
    results["jarvis_3d_single"] = m_j3d

    # ---- 2. Ensemble evaluation (if >1 member) ----
    if len(ensemble_runs) > 1:
        print(f"\n{'='*60}")
        print(f"[2] Ensemble ({len(ensemble_runs)} members) zero-shot")

        if imp2d_test is not None:
            ens_preds, ens_std, ens_targets = evaluate_ensemble(ensemble_runs, imp2d_test)
            m_ens_imp = compute_metrics(ens_preds, ens_targets)
            m_ens_imp["mean_sigma"] = float(np.mean(ens_std))
            print(f"    IMP2D test: MAE={m_ens_imp['MAE']:.3f}, σ̄={m_ens_imp['mean_sigma']:.3f}")
            results["imp2d_test_ensemble"] = m_ens_imp

        ens_preds_2d, ens_std_2d, ens_targets_2d = evaluate_ensemble(ensemble_runs, ds_2d)
        m_ens_2d = compute_metrics(ens_preds_2d, ens_targets_2d)
        m_ens_2d["mean_sigma"] = float(np.mean(ens_std_2d))
        print(f"    JARVIS-2D: MAE={m_ens_2d['MAE']:.3f}, σ̄={m_ens_2d['mean_sigma']:.3f}")
        results["jarvis_2d_ensemble"] = m_ens_2d

        ens_preds_3d, ens_std_3d, ens_targets_3d = evaluate_ensemble(ensemble_runs, ds_3d)
        m_ens_3d = compute_metrics(ens_preds_3d, ens_targets_3d)
        m_ens_3d["mean_sigma"] = float(np.mean(ens_std_3d))
        print(f"    JARVIS-3D: MAE={m_ens_3d['MAE']:.3f}, σ̄={m_ens_3d['mean_sigma']:.3f}")
        results["jarvis_3d_ensemble"] = m_ens_3d

    # ---- 3. Per-host breakdown for JARVIS-2D ----
    print(f"\n{'='*60}")
    print("[3] Per-host MAE breakdown (JARVIS-2D, single model)")
    host_results = {}
    for i in range(len(ds_2d)):
        s = ds_2d.data[i]
        host = s["metadata"]["host"]
        if host not in host_results:
            host_results[host] = {"preds": [], "targets": []}
        host_results[host]["preds"].append(preds_j2d[i])
        host_results[host]["targets"].append(targets_j2d[i])

    per_host = {}
    for host, vals in sorted(host_results.items()):
        p, t = np.array(vals["preds"]), np.array(vals["targets"])
        mae = float(np.mean(np.abs(p - t)))
        per_host[host] = {"MAE": mae, "N": len(t),
                          "mean_Ef": float(np.mean(t)),
                          "std_Ef": float(np.std(t))}
    results["jarvis_2d_per_host"] = per_host
    for h, v in sorted(per_host.items(), key=lambda x: x[1]["MAE"]):
        print(f"    {h:12s}: MAE={v['MAE']:.3f} eV (N={v['N']}, Ef={v['mean_Ef']:.2f}±{v['std_Ef']:.2f})")

    # ---- 4. Parity plots ----
    print(f"\n{'='*60}")
    print("[4] Generating figures...")
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # IMP2D parity
    if imp2d_test is not None:
        ax = axes[0]
        ax.scatter(targets_imp, preds_imp, s=8, alpha=0.3, c="steelblue")
        lims = [min(targets_imp.min(), preds_imp.min()) - 0.5,
                max(targets_imp.max(), preds_imp.max()) + 0.5]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("DFT $E_f$ (eV)")
        ax.set_ylabel("Predicted $E_f$ (eV)")
        ax.set_title(f"IMP2D test (MAE={m_imp['MAE']:.3f} eV)")

    # JARVIS-2D parity
    ax = axes[1]
    ax.scatter(targets_j2d, preds_j2d, s=30, alpha=0.6, c="coral",
               edgecolors="darkred", linewidths=0.5)
    lims = [min(targets_j2d.min(), preds_j2d.min()) - 0.5,
            max(targets_j2d.max(), preds_j2d.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DFT $E_f$ (eV)")
    ax.set_ylabel("Predicted $E_f$ (eV)")
    ax.set_title(f"JARVIS-2D zero-shot (MAE={m_j2d['MAE']:.3f} eV)")

    # JARVIS-3D parity
    ax = axes[2]
    ax.scatter(targets_j3d, preds_j3d, s=8, alpha=0.3, c="seagreen")
    lims = [min(targets_j3d.min(), preds_j3d.min()) - 0.5,
            max(targets_j3d.max(), preds_j3d.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DFT $E_f$ (eV)")
    ax.set_ylabel("Predicted $E_f$ (eV)")
    ax.set_title(f"JARVIS-3D zero-shot (MAE={m_j3d['MAE']:.3f} eV)")

    plt.tight_layout()
    fig_path = FIGURES / "fig_cross_dataset_parity.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")

    # ---- 5. Transfer gap bar chart ----
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["IMP2D\n(in-dist)", "JARVIS-2D\n(zero-shot)", "JARVIS-3D\n(zero-shot)"]
    mae_vals = [
        results.get("imp2d_test_single", {}).get("MAE", 0),
        results["jarvis_2d_single"]["MAE"],
        results["jarvis_3d_single"]["MAE"],
    ]
    colors = ["steelblue", "coral", "seagreen"]
    bars = ax.bar(labels, mae_vals, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("MAE (eV)")
    ax.set_title("Cross-Dataset Transfer: IMP2D → JARVIS")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if results.get("imp2d_test_single", {}).get("MAE"):
        ref = results["imp2d_test_single"]["MAE"]
        ratio_2d = m_j2d["MAE"] / ref
        ratio_3d = m_j3d["MAE"] / ref
        ax.text(0.95, 0.95,
                f"Transfer gap:\n2D: {ratio_2d:.2f}×\n3D: {ratio_3d:.2f}×",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3",
                                        facecolor="lightyellow", edgecolor="gray"))

    fig_path2 = FIGURES / "fig_cross_dataset_bars.png"
    fig.savefig(fig_path2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path2}")

    # ---- 6. Error vs Ef magnitude ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ae_2d = np.abs(preds_j2d - targets_j2d)
    ax.scatter(targets_j2d, ae_2d, s=30, alpha=0.6, c="coral", edgecolors="darkred", linewidths=0.5)
    ax.set_xlabel("DFT $E_f$ (eV)")
    ax.set_ylabel("|Error| (eV)")
    ax.set_title("JARVIS-2D: Error vs Formation Energy")
    ax.axhline(m_j2d["MAE"], ls="--", c="gray", label=f"MAE={m_j2d['MAE']:.3f}")
    ax.legend()

    ax = axes[1]
    ae_3d = np.abs(preds_j3d - targets_j3d)
    ax.scatter(targets_j3d, ae_3d, s=8, alpha=0.3, c="seagreen")
    ax.set_xlabel("DFT $E_f$ (eV)")
    ax.set_ylabel("|Error| (eV)")
    ax.set_title("JARVIS-3D: Error vs Formation Energy")
    ax.axhline(m_j3d["MAE"], ls="--", c="gray", label=f"MAE={m_j3d['MAE']:.3f}")
    ax.legend()

    plt.tight_layout()
    fig_path3 = FIGURES / "fig_cross_dataset_error_vs_ef.png"
    fig.savefig(fig_path3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path3}")

    # ---- Save results ----
    out_path = RESULTS / "cross_dataset_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved -> {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    ref_mae = results.get("imp2d_test_single", {}).get("MAE", None)
    if ref_mae:
        print(f"  IMP2D test (reference):  MAE = {ref_mae:.3f} eV")
        print(f"  JARVIS-2D (zero-shot):   MAE = {m_j2d['MAE']:.3f} eV  ({m_j2d['MAE']/ref_mae:.2f}× gap)")
        print(f"  JARVIS-3D (zero-shot):   MAE = {m_j3d['MAE']:.3f} eV  ({m_j3d['MAE']/ref_mae:.2f}× gap)")
    else:
        print(f"  JARVIS-2D (zero-shot):   MAE = {m_j2d['MAE']:.3f} eV")
        print(f"  JARVIS-3D (zero-shot):   MAE = {m_j3d['MAE']:.3f} eV")


if __name__ == "__main__":
    main()
