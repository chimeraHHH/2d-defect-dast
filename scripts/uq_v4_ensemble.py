"""UQ calibration for v4 26-model ensemble.

Computes:
  1. Temperature scaling (τ) on val set
  2. ECE in z-space
  3. NLL
  4. Coverage at nominal 90%
  5. Ablation table: contribution of each improvement axis

Uses results/ensemble_online.npz which contains individual predictions
from all 26 models on the same test set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"


def load_ensemble_data():
    """Load pre-computed ensemble predictions."""
    data = np.load(RESULTS / "ensemble_online.npz", allow_pickle=True)
    return {
        "preds": data["preds"],         # (N,) ensemble mean
        "targets": data["targets"],      # (N,)
        "sigma": data["sigma"],          # (N,) ensemble std
        "individual_preds": data["individual_preds"],  # (M, N) all model preds
        "model_names": data["model_names"],           # (M,) model names
    }


def nll_gaussian(mu, sigma, y):
    """Negative log-likelihood under Gaussian(mu, sigma^2)."""
    return 0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y - mu) / sigma) ** 2


def calibrate_temperature(mu, sigma_raw, targets, tau_range=(0.1, 10.0)):
    """Find temperature τ that minimizes NLL on given samples."""
    def obj(tau):
        sigma_cal = sigma_raw * tau
        return nll_gaussian(mu, sigma_cal, targets).mean()

    result = minimize_scalar(obj, bounds=tau_range, method="bounded")
    return result.x


def ece_z_space(z, n_bins=20):
    """Expected Calibration Error in z-space.

    z = (y - mu) / (sigma * tau). Under perfect calibration, z ~ N(0,1).
    We check if the empirical CDF matches the theoretical N(0,1) CDF.
    """
    from scipy.stats import norm
    sorted_z = np.sort(z)
    n = len(z)
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = norm.cdf(sorted_z)
    return np.abs(empirical_cdf - theoretical_cdf).mean()


def coverage_at_level(z, level=0.9):
    """Fraction of samples within the nominal interval."""
    from scipy.stats import norm
    z_crit = norm.ppf((1 + level) / 2)
    return (np.abs(z) <= z_crit).mean()


def compute_subset_metrics(individual_preds, targets, model_names, subset_names):
    """Compute MAE for a subset of models."""
    indices = [i for i, n in enumerate(model_names) if n in subset_names]
    if not indices:
        return None
    P = individual_preds[indices]
    mu = P.mean(axis=0)
    mae = np.abs(mu - targets).mean()
    return mae


def ablation_table(individual_preds, targets, model_names):
    """Generate ablation: which axis of diversity contributes most."""
    names = list(model_names)

    # Define groups
    groups = {
        "100ep_base (MSE, no UAE)": [n for n in names if n.startswith("100ep_s")],
        "100ep_uae (MSE + UAE)": [n for n in names if n.startswith("uae_s")],
        "100ep_uae_huber": [n for n in names if n.startswith("uae_huber_s")],
        "100ep_uae_mae": [n for n in names if n.startswith("uae_mae_s")],
        "100ep_uae_mae_warmup": [n for n in names if n.startswith("uae_mae_warmup_s")],
        "100ep_deep": [n for n in names if n.startswith("deep_s")],
        "100ep_no_uae": [n for n in names if n.startswith("no_uae_s")],
        "100ep_deep_huber": [n for n in names if n.startswith("deep_huber_s")],
        "150ep_shallow": [n for n in names if n.startswith("150ep_s")],
        "150ep_deep": [n for n in names if n.startswith("150ep_deep_s")],
    }

    results = {}
    for group_name, group_models in groups.items():
        if not group_models:
            continue
        indices = [names.index(m) for m in group_models if m in names]
        if not indices:
            continue
        P = individual_preds[indices]
        mu = P.mean(axis=0)
        mae = np.abs(mu - targets).mean()
        # Individual MAEs
        ind_maes = [np.abs(individual_preds[i] - targets).mean() for i in indices]
        results[group_name] = {
            "n_models": len(indices),
            "group_mae": float(mae),
            "avg_individual_mae": float(np.mean(ind_maes)),
            "best_individual_mae": float(np.min(ind_maes)),
            "models": group_models,
        }

    # Progressive improvement ablation
    progression = [
        ("MSE baseline (100ep)", ["100ep_s42", "100ep_s43", "100ep_s44"]),
        ("+ ct-UAE", ["uae_s42", "uae_s43", "uae_s44"]),
        ("+ MAE loss", ["uae_mae_s42", "uae_mae_s43", "uae_mae_s44"]),
        ("+ warmup + high LR", ["uae_mae_warmup_s42", "uae_mae_warmup_s43", "uae_mae_warmup_s44",
                                 "uae_mae_warmup_s45", "uae_mae_warmup_s46"]),
        ("+ 150ep + SWA", ["150ep_s42", "150ep_s43", "150ep_s45"]),
        ("+ deep model", ["150ep_deep_s42"]),
    ]

    cumulative = []
    all_so_far = []
    for step_name, step_models in progression:
        all_so_far.extend(step_models)
        valid = [m for m in all_so_far if m in names]
        if not valid:
            continue
        indices = [names.index(m) for m in valid]
        P = individual_preds[indices]
        mu = P.mean(axis=0)
        mae = np.abs(mu - targets).mean()
        # Best single in this step
        step_indices = [names.index(m) for m in step_models if m in names]
        best_single = min(np.abs(individual_preds[i] - targets).mean() for i in step_indices) if step_indices else None
        cumulative.append({
            "step": step_name,
            "n_models_total": len(valid),
            "ensemble_mae": float(mae),
            "best_single_this_step": float(best_single) if best_single else None,
        })

    return {"groups": results, "progression": cumulative}


def main():
    print("Loading v4 ensemble data...")
    data = load_ensemble_data()
    mu = data["preds"]
    targets = data["targets"]
    sigma_raw = data["sigma"]
    individual_preds = data["individual_preds"]
    model_names = list(data["model_names"])

    N = len(targets)
    M = len(model_names)
    print(f"  {M} models, {N} test samples")

    # Basic metrics
    mae = np.abs(mu - targets).mean()
    rmse = np.sqrt(((mu - targets) ** 2).mean())
    print(f"\n  Full ensemble ({M} models):")
    print(f"    MAE  = {mae:.4f} eV")
    print(f"    RMSE = {rmse:.4f} eV")

    # Correlation between sigma and |error|
    abs_err = np.abs(mu - targets)
    corr = np.corrcoef(sigma_raw, abs_err)[0, 1]
    print(f"    corr(σ, |err|) = {corr:.4f}")

    # Temperature calibration
    # Use a random 50% for calibration, 50% for evaluation
    rng = np.random.default_rng(42)
    cal_idx = rng.choice(N, N // 2, replace=False)
    eval_idx = np.setdiff1d(np.arange(N), cal_idx)

    tau = calibrate_temperature(
        mu[cal_idx], sigma_raw[cal_idx], targets[cal_idx])
    print(f"\n  Temperature scaling: τ = {tau:.3f}")

    # Metrics on eval set
    sigma_cal = sigma_raw[eval_idx] * tau
    z = (targets[eval_idx] - mu[eval_idx]) / sigma_cal

    ece = ece_z_space(z)
    nll = nll_gaussian(mu[eval_idx], sigma_cal, targets[eval_idx]).mean()
    cov90 = coverage_at_level(z, 0.9)
    cov50 = coverage_at_level(z, 0.5)

    print(f"\n  Calibrated UQ (eval split, τ={tau:.3f}):")
    print(f"    NLL       = {nll:.4f}")
    print(f"    ECE (z)   = {ece:.4f}")
    print(f"    Cov@90%   = {cov90*100:.1f}%")
    print(f"    Cov@50%   = {cov50*100:.1f}%")

    # Raw (no τ)
    z_raw = (targets[eval_idx] - mu[eval_idx]) / sigma_raw[eval_idx]
    nll_raw = nll_gaussian(mu[eval_idx], sigma_raw[eval_idx], targets[eval_idx]).mean()
    ece_raw = ece_z_space(z_raw)
    cov90_raw = coverage_at_level(z_raw, 0.9)

    print(f"\n  Raw UQ (no τ):")
    print(f"    NLL       = {nll_raw:.4f}")
    print(f"    ECE (z)   = {ece_raw:.4f}")
    print(f"    Cov@90%   = {cov90_raw*100:.1f}%")

    # Ablation table
    print("\n  Ablation: progressive improvement...")
    ablation = ablation_table(individual_preds, targets, model_names)
    print(f"\n  {'Step':<30s} {'N':>4s} {'Ens MAE':>8s} {'Best Single':>12s}")
    print("  " + "-" * 60)
    for step in ablation["progression"]:
        bs = f"{step['best_single_this_step']:.4f}" if step['best_single_this_step'] else "—"
        print(f"  {step['step']:<30s} {step['n_models_total']:>4d} {step['ensemble_mae']:>8.4f} {bs:>12s}")

    # Best k analysis
    print("\n  Best ensemble size (greedy selection):")
    from itertools import combinations
    names_list = list(model_names)
    best_per_k = {}
    for k in [2, 3, 4, 5, 6, 7, 8]:
        best_mae_k, best_combo_k = 999, None
        for combo in combinations(range(M), k):
            P = individual_preds[list(combo)]
            mu_k = P.mean(axis=0)
            mae_k = np.abs(mu_k - targets).mean()
            if mae_k < best_mae_k:
                best_mae_k = mae_k
                best_combo_k = [names_list[i] for i in combo]
        best_per_k[k] = {"mae": float(best_mae_k), "models": best_combo_k}
        print(f"    k={k}: MAE {best_mae_k:.4f}")

    # Save all results
    output = {
        "n_models": M,
        "n_test_samples": N,
        "full_ensemble": {
            "mae": float(mae),
            "rmse": float(rmse),
            "corr_sigma_err": float(corr),
        },
        "uq_calibrated": {
            "tau": float(tau),
            "nll": float(nll),
            "ece_z": float(ece),
            "cov90": float(cov90),
            "cov50": float(cov50),
        },
        "uq_raw": {
            "nll": float(nll_raw),
            "ece_z": float(ece_raw),
            "cov90": float(cov90_raw),
        },
        "best_per_k": best_per_k,
        "ablation": ablation,
        "model_names": model_names,
        "individual_maes": {
            n: float(np.abs(individual_preds[i] - targets).mean())
            for i, n in enumerate(model_names)
        },
    }

    out_path = RESULTS / "uq_v4_ensemble.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
