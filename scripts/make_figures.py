"""Render the figures used in the report:
  * fig_main: parity scatter for baseline vs improved on the test set
  * fig_curves: validation MAE per epoch for every run in results/
  * fig_error_dist: histogram of test absolute errors for the headline pair
  * fig_attention_pattern: visualisation of the star-sparse mask for one
    representative sample (illustrative).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_run(name: str) -> Dict:
    p = RESULTS / name / "metrics.json"
    if not p.exists():
        return {}
    with open(p, "r") as f:
        data = json.load(f)
    npz_path = RESULTS / name / "test_predictions.npz"
    if npz_path.exists():
        npz = np.load(npz_path)
        data["preds"] = npz["preds"]
        data["targets"] = npz["targets"]
    return data


def fig_parity(runs: Dict[str, Dict], out: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5), sharey=True)
    if len(runs) == 1:
        axes = [axes]
    lo, hi = -12, 22
    for ax, (name, data) in zip(axes, runs.items()):
        if "preds" not in data:
            ax.set_visible(False)
            continue
        ax.scatter(data["targets"], data["preds"], s=8, alpha=0.4, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        mae = float(data.get("test_mae", np.nan))
        rmse = float(data.get("test_rmse", np.nan))
        ax.set_title(f"{name}\nMAE={mae:.3f} eV, RMSE={rmse:.3f} eV")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("DFT formation energy (eV)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Predicted formation energy (eV)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_curves(runs: Dict[str, Dict], out: Path, keys: list = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    items = list(runs.items())
    if keys is not None:
        items = [(k, v) for k, v in items if k in keys]
    for name, data in items:
        history = data.get("history") or []
        if not history:
            continue
        epochs = [row["epoch"] for row in history]
        val = [row["val_mae"] for row in history]
        ax.plot(epochs, val, marker="o", ms=3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (eV)")
    ax.set_title("Validation MAE per epoch")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_error_dist(runs: Dict[str, Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, data in runs.items():
        if "preds" not in data:
            continue
        err = data["preds"] - data["targets"]
        ax.hist(err, bins=80, density=True, alpha=0.5, label=name)
    ax.set_xlabel("Prediction error (eV)")
    ax.set_ylabel("Density")
    ax.set_title("Test-set error distribution")
    ax.set_xlim(-6, 6)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_metric_table(runs: Dict[str, Dict], out_path: Path) -> None:
    rows: List[List] = []
    rows.append(["model", "params (M)", "test MAE (eV)", "test RMSE (eV)"])
    for name, data in runs.items():
        if not data:
            continue
        rows.append(
            [
                name,
                f"{data['n_params']/1e6:.3f}",
                f"{data['test_mae']:.4f}",
                f"{data['test_rmse']:.4f}",
            ]
        )
    out_path.write_text("\n".join("\t".join(map(str, r)) for r in rows))


def main() -> None:
    # Auto-discover every run under results/
    all_runs: Dict[str, Dict] = {}
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        data = _load_run(d.name)
        if data:
            all_runs[d.name] = data
    if not all_runs:
        print("No runs found in results/. Train something first.", file=sys.stderr)
        return

    # Highlight set: 3 representative configurations spanning the result range
    headline_keys = [
        "baseline_h128_aug_xlong_safe",  # best (honest, leak-free)
        "baseline_h128_aug_long_safe",   # standard config (honest)
        "baseline",                       # plain baseline (no aug)
    ]
    headline = {
        f"{k} (test MAE {v['test_mae']:.3f})": v
        for k, v in all_runs.items()
        if k in headline_keys
    }

    fig_parity(headline, FIG_DIR / "fig_parity.png")
    # focused curves for the core narrative; full one for appendix
    fig_curves(
        all_runs,
        FIG_DIR / "fig_curves_core.png",
        keys=[
            "baseline",
            "baseline_long",
            "baseline_aug",
            "baseline_aug_long",
            "baseline_h128_long",
            "baseline_h128_aug_long",
            "improved",
            "dast_dense",
        ],
    )
    fig_curves(all_runs, FIG_DIR / "fig_curves_all.png")
    fig_error_dist(headline, FIG_DIR / "fig_error_dist.png")
    fig_metric_table(all_runs, FIG_DIR / "metrics_table.tsv")
    print("Figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
