"""MC-Dropout uncertainty quantification, compared to deep ensemble.

The trained ``baseline_h128_aug_long_safe`` model has dropout=0.1 in both the
GeometricTransformerBlock FFN and the readout MLP. By keeping the model in
train mode at inference time, we run K stochastic forward passes per sample
and use the per-sample mean / std as a UQ estimate (Gal & Ghahramani, 2016).

We compare:
  * MC-Dropout 30 forward passes
  * 4-seed deep ensemble (already computed)
  * 6-member mixed ensemble (already computed)

Metrics: MAE, RMSE, NLL, ECE_z, cov90 (raw + τ-scaled).

Output:
  - results/mc_dropout_vs_ensemble.json
  - paper/figures/fig_uq_method_compare.png
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
import yaml
from scipy.optimize import minimize_scalar
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"

NOMINAL = (0.50, 0.68, 0.90, 0.95)


def gauss_nll(mu, sigma, y, eps=1e-6):
    sigma = np.maximum(sigma, eps)
    return 0.5 * np.log(2 * math.pi * sigma ** 2) + 0.5 * ((y - mu) / sigma) ** 2


def coverage(mu, sigma, y, level):
    z = norm.ppf(0.5 + level / 2)
    return float((np.abs(y - mu) <= z * sigma).mean())


def ece_z(mu, sigma, y, n_bins=20):
    z = (y - mu) / np.maximum(sigma, 1e-6)
    grid = np.linspace(-3, 3, n_bins + 1)
    emp = np.array([(z <= g).mean() for g in grid])
    th = norm.cdf(grid)
    return float(np.abs(emp - th).mean())


def fit_temperature(mu, sigma, y, ratio_split=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    n_fit = int(ratio_split * n)
    fit_idx, eval_idx = perm[:n_fit], perm[n_fit:]
    mu_f, sig_f, y_f = mu[fit_idx], sigma[fit_idx], y[fit_idx]
    res = minimize_scalar(
        lambda t: gauss_nll(mu_f, max(t, 1e-6) * sig_f, y_f).mean(),
        bounds=(0.05, 20.0), method="bounded",
    )
    return float(res.x), eval_idx


def report(mu, sigma, y, label):
    return {
        "label": label,
        "n": int(len(y)),
        "mae": float(np.abs(y - mu).mean()),
        "rmse": float(np.sqrt(((y - mu) ** 2).mean())),
        "mean_sigma": float(sigma.mean()),
        "nll": float(gauss_nll(mu, sigma, y).mean()),
        "ece_z": ece_z(mu, sigma, y),
        "pearson_sigma_err": float(np.corrcoef(sigma, np.abs(y - mu))[0, 1]),
        **{f"cov_{int(L * 100)}": coverage(mu, sigma, y, L) for L in NOMINAL},
    }


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
    nmean, nstd = ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"]

    # === MC-Dropout ===
    K = 30
    print(f"running MC-Dropout with K={K} forward passes...")
    from torch.utils.data import DataLoader
    loader = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # we need to keep dropout enabled at inference; setting model.train() is the
    # canonical way (Gal 2016) but we don't want to use BatchNorm running stats
    # — there is no BN in this model, only LayerNorm and Dropout, so .train() is
    # safe for our purpose.
    model.train()
    # disable layernorm grad / running stats — already off for layernorm.
    # for safety set requires_grad=False.
    for p in model.parameters():
        p.requires_grad = False

    all_mc = []
    targets = None
    for k in range(K):
        preds = []
        with torch.no_grad():
            for batch in loader:
                p = model(batch) * nstd + nmean
                preds.append(p.numpy())
        preds = np.concatenate(preds)
        if targets is None:
            tgts = []
            for batch in loader:
                tgts.append(batch["target"].numpy())
            targets = np.concatenate(tgts)
        all_mc.append(preds)
    P_mc = np.stack(all_mc)  # (K, N)
    mu_mc = P_mc.mean(0)
    sigma_mc = P_mc.std(0, ddof=1)
    print(f"MC-Dropout: MAE {np.abs(targets - mu_mc).mean():.4f}, "
          f"mean σ {sigma_mc.mean():.4f}")

    # === Load deep ensemble for comparison ===
    def load_preds(name):
        p = RESULTS / name / "test_predictions.npz"
        if not p.exists():
            return None
        return np.load(p)["preds"].astype(np.float64)

    ens4 = []
    for r in [
        "baseline_h128_aug_long_safe",
        "baseline_h128_aug_long_safe_seed0",
        "baseline_h128_aug_long_safe_seed1",
        "baseline_h128_aug_long_safe_seed2",
    ]:
        p = load_preds(r)
        if p is not None:
            ens4.append(p)
    P_e4 = np.stack(ens4)
    mu_e4, sigma_e4 = P_e4.mean(0), P_e4.std(0, ddof=1)

    ens6 = list(ens4)
    for r in [
        "baseline_h128_aug_xlong_safe",
        "baseline_h128_aug_xlong_safe_seed0",
    ]:
        p = load_preds(r)
        if p is not None:
            ens6.append(p)
    P_e6 = np.stack(ens6)
    mu_e6, sigma_e6 = P_e6.mean(0), P_e6.std(0, ddof=1)

    methods = [
        ("MC-Dropout (K=30)", mu_mc, sigma_mc),
        ("4-seed ensemble", mu_e4, sigma_e4),
        ("6-member mixed ensemble", mu_e6, sigma_e6),
    ]

    summary = []
    for name, mu, sigma in methods:
        raw = report(mu, sigma, targets, f"{name} | raw")
        tau, eval_idx = fit_temperature(mu, sigma, targets)
        sigma_t = tau * sigma
        eva = report(mu[eval_idx], sigma_t[eval_idx], targets[eval_idx],
                     f"{name} | τ={tau:.2f}")
        summary.append({"method": name, "tau": tau, "raw": raw, "tau_eval": eva})
        print(f"\n{name}: raw NLL {raw['nll']:.3f}, cov90 {raw['cov_90']:.3f}; "
              f"τ={tau:.2f} → cov90 {eva['cov_90']:.3f}")

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    labels = [s["method"] for s in summary]

    ax = axes[0]
    raw_mae = [s["raw"]["mae"] for s in summary]
    eva_mae = [s["tau_eval"]["mae"] for s in summary]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, raw_mae, w, label="raw")
    ax.bar(x + w / 2, eva_mae, w, label="τ-scaled (eval-half)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("Point estimate (lower better)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    raw_nll = [s["raw"]["nll"] for s in summary]
    eva_nll = [s["tau_eval"]["nll"] for s in summary]
    ax.bar(x - w / 2, raw_nll, w, label="raw")
    ax.bar(x + w / 2, eva_nll, w, label="τ-scaled")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("NLL")
    ax.set_title("Negative log-likelihood (lower better)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    levels = np.array(NOMINAL)
    for s, marker in zip(summary, ["o", "s", "^"]):
        cov_eva = [s["tau_eval"][f"cov_{int(L * 100)}"] for L in NOMINAL]
        ax.plot(levels, cov_eva, marker + "-", label=f"{s['method']} (τ={s['tau']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6, label="ideal")
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage at nominal levels (τ-scaled)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "fig_uq_method_compare.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"\nfigure saved -> {out}")

    out_json = RESULTS / "mc_dropout_vs_ensemble.json"
    with open(out_json, "w") as f:
        json.dump({"K": K, "methods": summary}, f, indent=2)
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()
