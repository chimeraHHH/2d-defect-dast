"""Does the model's σ track physical-category difficulty?

We have:
  * 6-member ensemble σ per test sample
  * per-sample metadata (host, dopant, defect type, |Ef|)

Question: do high-difficulty categories (large 5d dopants, |Ef|≥6 eV,
exotic hosts) also have systematically higher σ? If yes, the UQ signal
is *meaningfully* aligned with physical complexity rather than just
"random noise".

Output: results/uq_by_category.json + paper/figures/fig_uq_by_category.png
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, make_splits  # noqa: E402

RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"


def _block(Z):
    if Z is None or Z <= 0:
        return "?"
    if Z in {1, 2}: return "s"
    main_group = {3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18,
                  19, 20, 31, 32, 33, 34, 35, 36,
                  37, 38, 49, 50, 51, 52, 53, 54,
                  55, 56, 81, 82, 83, 84, 85, 86}
    if Z in main_group: return "main"
    if 21 <= Z <= 30: return "3d"
    if 39 <= Z <= 48: return "4d"
    if 57 <= Z <= 71: return "4f"
    if 72 <= Z <= 80: return "5d"
    return "?"


def main():
    runs = [
        "baseline_h128_aug_long_safe",
        "baseline_h128_aug_long_safe_seed0",
        "baseline_h128_aug_long_safe_seed1",
        "baseline_h128_aug_long_safe_seed2",
        "baseline_h128_aug_xlong_safe",
        "baseline_h128_aug_xlong_safe_seed0",
    ]
    P, targets = [], None
    for r in runs:
        f = RESULTS / r / "test_predictions.npz"
        if not f.exists(): continue
        a = np.load(f)
        P.append(a["preds"].astype(np.float64))
        if targets is None:
            targets = a["targets"].astype(np.float64)
    P = np.stack(P)
    mu = P.mean(0); sigma = P.std(0, ddof=1); err = np.abs(targets - mu)

    # join to metadata via cleaned_dataset.pkl test split
    cfg = yaml.safe_load(open(ROOT / "configs/baseline_h128_aug_long_safe.yaml"))
    cleaned = ROOT / "data/processed/cleaned_dataset.pkl"
    safe = ROOT / cfg["data_path"]
    ds = CrystalGraphDataset(safe if safe.exists() else cleaned)
    _, _, test_set = make_splits(ds, cfg.get("train_ratio", 0.8),
                                  cfg.get("val_ratio", 0.1), cfg.get("seed", 42))
    test_ids = list(test_set.indices)

    # group σ and err by category
    sigma_by_block = defaultdict(list); err_by_block = defaultdict(list)
    sigma_by_dtype = defaultdict(list); err_by_dtype = defaultdict(list)
    sigma_by_efmag = defaultdict(list); err_by_efmag = defaultdict(list)

    for i, idx in enumerate(test_ids):
        meta = ds.data[idx]["metadata"]
        dopant = meta.get("dopant", "?") or "?"
        dtype = meta.get("defecttype", "?") or "?"
        try:
            from ase.data import atomic_numbers as _AZ
            zd = _AZ.get(dopant, None)
            blk = _block(zd) if zd is not None else "?"
        except Exception:
            blk = "?"
        sigma_by_block[blk].append(sigma[i])
        err_by_block[blk].append(err[i])
        sigma_by_dtype[dtype].append(sigma[i])
        err_by_dtype[dtype].append(err[i])
        ef = float(targets[i])
        b = ("|Ef|<1" if abs(ef) < 1 else "1-3" if abs(ef) < 3
             else "3-6" if abs(ef) < 6 else "≥6")
        sigma_by_efmag[b].append(sigma[i])
        err_by_efmag[b].append(err[i])

    def _agg(d):
        return {k: {
            "n": int(len(v)),
            "mean_sigma": float(np.mean(v)),
            "median_sigma": float(np.median(v)),
        } for k, v in d.items()}

    def _agg_err(d):
        return {k: {
            "n": int(len(v)),
            "mean_abs_err": float(np.mean(v)),
        } for k, v in d.items()}

    summary = {
        "by_dopant_block_sigma": _agg(sigma_by_block),
        "by_dopant_block_err": _agg_err(err_by_block),
        "by_defect_type_sigma": _agg(sigma_by_dtype),
        "by_defect_type_err": _agg_err(err_by_dtype),
        "by_ef_magnitude_sigma": _agg(sigma_by_efmag),
        "by_ef_magnitude_err": _agg_err(err_by_efmag),
    }

    # figure: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    cats = [
        ("by dopant block", sigma_by_block, err_by_block, ["s", "3d", "main", "4f", "4d", "5d"]),
        ("by defect type", sigma_by_dtype, err_by_dtype, ["adsorbate", "interstitial"]),
        ("by |Ef| magnitude", sigma_by_efmag, err_by_efmag, ["|Ef|<1", "1-3", "3-6", "≥6"]),
    ]
    for ax, (title, sd, ed, order) in zip(axes, cats):
        keys = [k for k in order if k in sd]
        sigmas = [np.mean(sd[k]) for k in keys]
        errs = [np.mean(ed[k]) for k in keys]
        ns = [len(sd[k]) for k in keys]
        x = np.arange(len(keys))
        w = 0.35
        ax.bar(x - w / 2, sigmas, w, label="mean σ", color="tab:blue")
        ax.bar(x + w / 2, errs, w, label="mean |err|", color="tab:orange")
        ax.set_xticks(x); ax.set_xticklabels([f"{k}\nn={n}" for k, n in zip(keys, ns)])
        ax.set_ylabel("eV")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Ensemble σ tracks physical-category difficulty (6-member, raw)")
    fig.tight_layout()
    out = FIG_DIR / "fig_uq_by_category.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"saved {out}")

    print("\n--- by dopant block ---")
    for k in ["s", "3d", "main", "4f", "4d", "5d"]:
        if k not in sigma_by_block: continue
        s_mean = np.mean(sigma_by_block[k]); e_mean = np.mean(err_by_block[k])
        print(f"  {k:<6} σ̄={s_mean:.3f}  |err|̄={e_mean:.3f}  ratio σ/|err|={s_mean/max(e_mean, 1e-9):.2f}")

    print("\n--- by defect type ---")
    for k in ["adsorbate", "interstitial"]:
        if k not in sigma_by_dtype: continue
        s_mean = np.mean(sigma_by_dtype[k]); e_mean = np.mean(err_by_dtype[k])
        print(f"  {k:<13} σ̄={s_mean:.3f}  |err|̄={e_mean:.3f}")

    print("\n--- by |Ef| magnitude ---")
    for k in ["|Ef|<1", "1-3", "3-6", "≥6"]:
        if k not in sigma_by_efmag: continue
        s_mean = np.mean(sigma_by_efmag[k]); e_mean = np.mean(err_by_efmag[k])
        print(f"  {k:<8} σ̄={s_mean:.3f}  |err|̄={e_mean:.3f}")

    with open(RESULTS / "uq_by_category.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/uq_by_category.json")


if __name__ == "__main__":
    main()
