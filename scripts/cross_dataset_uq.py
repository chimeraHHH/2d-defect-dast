"""Cross-dataset UQ analysis: does ensemble uncertainty correlate with
actual prediction error on out-of-distribution JARVIS data?

Key questions:
  1. Is ensemble σ higher on JARVIS than IMP2D test? (calibration awareness)
  2. Does σ ranking predict error magnitude? (σ-error correlation)
  3. Coverage: do confidence intervals cover true values at nominal rates?

Uses all available ensemble members from IMP2D training.
Output: results/cross_dataset_uq.json + figures
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
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, split_indices
from src.models import CrystalTransformer
from torch.utils.data import DataLoader, Subset

RESULTS = ROOT / "results"
FIGURES = ROOT / "paper/figures"


def load_model(run_dir):
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


def predict_all(model, dataset, nmean, nstd, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    preds = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch).cpu().numpy() * nstd + nmean
            preds.extend(pred.flatten().tolist())
    return np.array(preds)


def get_targets(dataset):
    targets = []
    for i in range(len(dataset)):
        s = dataset[i] if hasattr(dataset, "__getitem__") else dataset.dataset[dataset.indices[i]]
        if isinstance(s, dict) and "target" in s:
            targets.append(float(s["target"]))
        else:
            targets.append(float(s["target"].item()) if isinstance(s["target"], torch.Tensor) else float(s["target"]))
    return np.array(targets)


def ensemble_predict(run_dirs, dataset, batch_size=32):
    all_preds = []
    for rd in run_dirs:
        model, nmean, nstd = load_model(rd)
        preds = predict_all(model, dataset, nmean, nstd, batch_size)
        all_preds.append(preds)
    all_preds = np.array(all_preds)
    return all_preds.mean(axis=0), all_preds.std(axis=0)


def coverage_at_level(mu, sigma, targets, level):
    z = stats.norm.ppf(0.5 + level / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    covered = np.sum((targets >= lower) & (targets <= upper))
    return float(covered / len(targets))


def main():
    print("=" * 60)
    print("Cross-Dataset UQ Analysis")
    print("=" * 60)

    # Find all ensemble members
    ensemble_runs = []
    for name in ["baseline_h128_aug_long_safe",
                  "baseline_h128_aug_long_safe_seed0",
                  "baseline_h128_aug_long_safe_seed1",
                  "baseline_h128_aug_long_safe_seed2",
                  "baseline_h128_aug_xlong_safe",
                  "baseline_h128_aug_xlong_safe_seed0",
                  "baseline_h128_aug_xlong_safe_seed1",
                  "baseline_h128_aug_xlong_safe_seed2"]:
        if (RESULTS / name / "best.pt").exists():
            ensemble_runs.append(name)

    print(f"Ensemble members: {len(ensemble_runs)}")
    if len(ensemble_runs) < 2:
        print("Need ≥2 ensemble members for UQ analysis. Using single model with noise estimation.")
        return

    # Load datasets
    ds_2d = CrystalGraphDataset(ROOT / "data/processed/jarvis_2d.pkl")
    ds_3d = CrystalGraphDataset(ROOT / "data/processed/jarvis_3d.pkl")

    imp2d_path = ROOT / "data/processed/cleaned_dataset.pkl"
    ds_imp2d = CrystalGraphDataset(imp2d_path)
    _, _, test_idx = split_indices(len(ds_imp2d), 0.8, 0.1, 42)
    imp2d_test = Subset(ds_imp2d, test_idx)

    datasets = {
        "IMP2D_test": imp2d_test,
        "JARVIS_2D": ds_2d,
        "JARVIS_3D": ds_3d,
    }

    results = {}
    all_data = {}

    for name, ds in datasets.items():
        print(f"\n--- {name} ({len(ds)} samples) ---")
        mu, sigma = ensemble_predict(ensemble_runs, ds)

        # Get targets
        if isinstance(ds, Subset):
            targets = np.array([ds_imp2d.data[i]["target"] for i in ds.indices])
        else:
            targets = np.array([s["target"] for s in ds.data])

        errors = np.abs(mu - targets)
        mae = float(np.mean(errors))

        # Correlation between σ and |error|
        spearman_r, spearman_p = stats.spearmanr(sigma, errors)

        # Coverage
        coverage_levels = [0.50, 0.68, 0.80, 0.90, 0.95]
        coverages = {f"cov_{int(l*100)}": coverage_at_level(mu, sigma, targets, l)
                     for l in coverage_levels}

        # z-scores
        z_scores = (mu - targets) / np.maximum(sigma, 1e-6)
        ece_z = float(np.abs(np.mean(np.abs(z_scores) < 1.0) - 0.6827))

        result = {
            "N": len(ds),
            "MAE": mae,
            "mean_sigma": float(np.mean(sigma)),
            "median_sigma": float(np.median(sigma)),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "ECE_z": ece_z,
            **coverages,
        }
        results[name] = result
        all_data[name] = {"mu": mu, "sigma": sigma, "targets": targets, "errors": errors}

        print(f"  MAE={mae:.3f}, σ̄={result['mean_sigma']:.3f}")
        print(f"  Spearman(σ, |err|) = {spearman_r:.3f} (p={spearman_p:.4f})")
        print(f"  Coverage: " + ", ".join(f"{int(l*100)}%={coverages[f'cov_{int(l*100)}']:.1%}"
                                          for l in coverage_levels))

    # ---- Figures ----
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Fig 1: σ distribution comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in [("IMP2D_test", "steelblue"), ("JARVIS_2D", "coral"), ("JARVIS_3D", "seagreen")]:
        sigma = all_data[name]["sigma"]
        ax.hist(sigma, bins=30, alpha=0.5, color=color, edgecolor="black",
                label=f"{name} (σ̄={np.mean(sigma):.2f})", density=True)
    ax.set_xlabel("Ensemble σ (eV)")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution: In-dist vs Cross-dataset")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES / "fig_cross_dataset_uq_sigma_dist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: σ vs |error| scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, color) in zip(axes, [("IMP2D_test", "steelblue"),
                                         ("JARVIS_2D", "coral"),
                                         ("JARVIS_3D", "seagreen")]):
        sigma = all_data[name]["sigma"]
        errors = all_data[name]["errors"]
        ax.scatter(sigma, errors, s=20, alpha=0.5, c=color, edgecolors="black", linewidths=0.3)
        ax.set_xlabel("Ensemble σ (eV)")
        ax.set_ylabel("|Prediction Error| (eV)")
        r = results[name]["spearman_r"]
        ax.set_title(f"{name}\nSpearman r={r:.3f}")
        # trend line
        if len(sigma) > 5:
            z = np.polyfit(sigma, errors, 1)
            xs = np.linspace(sigma.min(), sigma.max(), 100)
            ax.plot(xs, np.polyval(z, xs), "k--", lw=1, alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_cross_dataset_uq_sigma_vs_error.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: Coverage comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    levels = [50, 68, 80, 90, 95]
    x = np.arange(len(levels))
    w = 0.25
    for i, (name, color) in enumerate([("IMP2D_test", "steelblue"),
                                        ("JARVIS_2D", "coral"),
                                        ("JARVIS_3D", "seagreen")]):
        covs = [results[name][f"cov_{l}"] * 100 for l in levels]
        ax.bar(x + i*w, covs, w, label=name, color=color, edgecolor="black")
    ax.plot(x + w, levels, "k^--", markersize=8, label="Nominal")
    ax.set_xlabel("Nominal Coverage Level (%)")
    ax.set_ylabel("Actual Coverage (%)")
    ax.set_title("Confidence Interval Coverage: In-dist vs Cross-dataset")
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"{l}%" for l in levels])
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIGURES / "fig_cross_dataset_uq_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures saved to {FIGURES}")

    # Save results
    out_path = RESULTS / "cross_dataset_uq.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("UQ TRANSFER SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'MAE':>6} {'σ̄':>6} {'Spearman':>9} {'Cov90':>7}")
    for name in ["IMP2D_test", "JARVIS_2D", "JARVIS_3D"]:
        r = results[name]
        print(f"{name:<15} {r['MAE']:>6.3f} {r['mean_sigma']:>6.3f} "
              f"{r['spearman_r']:>9.3f} {r['cov_90']:>6.1%}")


if __name__ == "__main__":
    main()
