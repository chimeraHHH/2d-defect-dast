"""Generate the v2-section figures for the paper.

Produces:
  - paper/figures/fig_v2_ablation.png       (5 single-source v2 variants vs v1 baseline)
  - paper/figures/fig_v2_multi_source.png   (baseline → v1-multi → v2-multi progression with seed scatter)
  - paper/figures/fig_v2_training_curves.png (val MAE per epoch, 4 seeds)
  - paper/figures/fig_v2_summary.png        (one-glance dashboard with all comparisons)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titlesize": 12,
    "figure.dpi": 130,
})


def read_metrics(name: str):
    p = RESULTS / name / "metrics.json"
    if not p.exists():
        return None
    return json.load(open(p))


def read_json(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.load(open(p))


# ── 1. v2 single-source ablation bar ──
def fig_v2_ablation():
    runs = [
        ("baseline_h128_aug_long_safe", "v1 baseline\n(h128, leak-free)", "#888888"),
        ("v2_pfa_h128_aug_long_safe", "v2 full\n(PFA + LR + DB)", "#1f77b4"),
        ("v2_pfa_only", "v2 PFA only\n(LR off, DB off)", "#2ca02c"),
        ("v2_ablate_no_long_range", "v2 − LR\n(PFA + DB)", "#ff7f0e"),
        ("v2_ablate_no_pfa", "v2 − PFA\n(LR + DB)", "#d62728"),
        ("v2_ablate_no_defect_bias", "v2 − DB\n(PFA + LR)", "#9467bd"),
    ]
    labels, values, rmses, colors, params = [], [], [], [], []
    for run, label, c in runs:
        m = read_metrics(run)
        if m is None:
            continue
        labels.append(label)
        values.append(m["test_mae"])
        rmses.append(m["test_rmse"])
        colors.append(c)
        params.append(m["n_params"] / 1e6)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    for x, v, p in zip(xs, values, params):
        ax.text(x, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(x, 0.04, f"{p:.3f}M", ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")
    baseline_val = values[0]
    ax.axhline(baseline_val, color="grey", linestyle="--", alpha=0.6, lw=1, label=f"baseline = {baseline_val:.4f}")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Phase 1 single-source ablation (h=128, 50 epoch, leak-free aug, seed=42)\n"
                 "All five v2 variants land within 2σ of the v1 baseline — no statistical winner.",
                 loc="left")
    ax.set_ylim(0, max(values) * 1.15)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIG / "fig_v2_ablation.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── 2. multi-source progression with 4-seed scatter ──
def fig_v2_multi_source():
    seeds_data = []
    for s in (42, 0, 1, 2):
        f = RESULTS / (f"multi_source_train_v2_seed{s}.json" if s != 42 else "multi_source_train_v2.json")
        if not f.exists():
            continue
        d = json.load(open(f))
        seeds_data.append((s, d["test_mae_imp2d_eV"]))
    seeds_data.sort(key=lambda x: x[0])
    seeds, maes = zip(*seeds_data)
    mean_mae = float(np.mean(maes))
    std_mae = float(np.std(maes, ddof=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                    gridspec_kw={"width_ratios": [3, 1.2]})

    # left: progression bars
    labels = [
        "ALIGNN\n(team repro)",
        "v1 baseline\n(leak-free aug, seed=42)",
        "v1 multi-source\n(no aug, seed=42)",
        "v2 multi-source\n(PFA + 4 DB, seed=42)",
        "v2 multi-source\n(4-seed mean ± std)",
    ]
    vals = [0.540, 0.516, 0.555, 0.4929, mean_mae]
    errs = [0, 0, 0, 0, std_mae]
    cols = ["#777777", "#888888", "#aa6644", "#1f77b4", "#1f77b4"]
    xs = np.arange(len(labels))
    bars = ax1.bar(xs, vals, yerr=errs, color=cols, alpha=0.85,
                   edgecolor="black", linewidth=0.8, capsize=6)
    for x, v in zip(xs, vals):
        ax1.text(x, v + 0.012, f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.axhline(0.516, color="grey", linestyle="--", alpha=0.6, lw=1)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Test MAE (eV)")
    ax1.set_ylim(0, 0.62)
    ax1.set_title("v2 multi-source breaks the data bottleneck\n"
                  "(seed=42 is apples-to-apples vs v1 multi-source)",
                  loc="left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # right: per-seed scatter
    ys = list(maes)
    xs2 = np.zeros(len(ys)) + 1
    ax2.scatter(xs2, ys, s=120, color="#1f77b4", alpha=0.85, edgecolor="black", zorder=3)
    for s, y in zip(seeds, ys):
        ax2.text(1.04, y, f"seed={s}: {y:.4f}", va="center", fontsize=9)
    ax2.errorbar([0.85], [mean_mae], yerr=std_mae, fmt="s", color="black",
                 ms=10, capsize=8, mfc="orange", zorder=2,
                 label=f"mean = {mean_mae:.4f}\nstd = {std_mae:.4f}")
    ax2.set_xlim(0.6, 1.4)
    ax2.set_xticks([])
    ax2.set_ylabel("Test MAE (eV)")
    ax2.set_title("4-seed scatter\n(split varies with seed)", loc="left", fontsize=11)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(min(ys) - 0.04, max(ys) + 0.04)

    plt.tight_layout()
    out = FIG / "fig_v2_multi_source.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── 3. training curves: 4 seeds val MAE ──
def fig_v2_training_curves():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    seeds_files = [
        (42, "multi_source_train_v2.json", "#1f77b4"),
        (0, "multi_source_train_v2_seed0.json", "#2ca02c"),
        (1, "multi_source_train_v2_seed1.json", "#ff7f0e"),
        (2, "multi_source_train_v2_seed2.json", "#d62728"),
    ]
    for s, fname, color in seeds_files:
        p = RESULTS / fname
        if not p.exists():
            continue
        d = json.load(open(p))
        eps = [h["epoch"] for h in d["history"]]
        vals = [h["val_mae_imp2d"] for h in d["history"]]
        ax.plot(eps, vals, label=f"seed={s}  test={d['test_mae_imp2d_eV']:.4f}",
                color=color, lw=1.6, alpha=0.85)
    ax.axhline(0.516, color="grey", linestyle="--", alpha=0.55, label="v1 baseline 0.516 (different test set)")
    ax.axhline(0.555, color="#aa6644", linestyle=":", alpha=0.55, label="v1 multi-source 0.555")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE on IMP2D (eV)")
    ax.set_title("v2 multi-source training: 4 seeds converge below v1 baselines", loc="left")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 3.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIG / "fig_v2_training_curves.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── 4. summary dashboard ──
def fig_v2_summary():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # panel 1: best so far
    ax = axes[0]
    labels = ["v1\nbaseline", "v1\nmulti-source", "v2 single-source\n(full)", "v2 multi-source\n(seed=42)", "v2 multi-source\n(4-seed mean)"]
    vals = [0.516, 0.555, 0.5193, 0.4929, 0.4856]
    errs = [0.016, 0, 0, 0, 0.025]
    colors = ["#888888", "#aa6644", "#1f77b4", "#1f77b4", "#1f77b4"]
    xs = np.arange(len(vals))
    ax.bar(xs, vals, yerr=errs, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.8, capsize=6)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("Headline: v2 multi-source wins")
    ax.set_ylim(0, 0.62)
    ax.axhline(0.516, color="grey", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # panel 2: v2 single-source ablation as % vs baseline
    ax = axes[1]
    runs2 = [
        ("v2_pfa_h128_aug_long_safe", "v2 full"),
        ("v2_pfa_only", "v2 PFA only"),
        ("v2_ablate_no_long_range", "v2 − LR"),
        ("v2_ablate_no_pfa", "v2 − PFA"),
        ("v2_ablate_no_defect_bias", "v2 − DB"),
    ]
    base = read_metrics("baseline_h128_aug_long_safe")["test_mae"]
    deltas, lbl = [], []
    for run, name in runs2:
        m = read_metrics(run)
        if m is None: continue
        deltas.append((m["test_mae"] - base) / base * 100)
        lbl.append(name)
    cs = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    xs = np.arange(len(deltas))
    bars = ax.bar(xs, deltas, color=cs, alpha=0.85, edgecolor="black", linewidth=0.8)
    for x, v in zip(xs, deltas):
        sign = "+" if v >= 0 else ""
        ax.text(x, v + (0.15 if v >= 0 else -0.4), f"{sign}{v:.1f}%",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=0.7)
    ax.fill_between([-0.5, len(xs) - 0.5], -3.1, 3.1, color="grey", alpha=0.15,
                    label=r"$\pm$2$\sigma$ baseline noise (±3.1%)")
    ax.set_xticks(xs)
    ax.set_xticklabels(lbl, fontsize=9, rotation=20)
    ax.set_ylabel("Δ MAE vs baseline (%)")
    ax.set_title("Phase 1: single-source v2 lands inside 2σ noise band")
    ax.set_ylim(-3, 8)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # panel 3: 4-seed scatter
    ax = axes[2]
    seeds_data = []
    for s in (42, 0, 1, 2):
        f = RESULTS / (f"multi_source_train_v2_seed{s}.json" if s != 42 else "multi_source_train_v2.json")
        if f.exists():
            seeds_data.append((s, json.load(open(f))["test_mae_imp2d_eV"]))
    seeds_data.sort(key=lambda x: x[0])
    seeds, maes = zip(*seeds_data)
    xs = np.arange(len(seeds))
    ax.scatter(xs, maes, s=140, color="#1f77b4", alpha=0.85, edgecolor="black", zorder=3)
    for x, s, y in zip(xs, seeds, maes):
        ax.annotate(f"seed={s}\n{y:.4f}", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    mean = float(np.mean(maes))
    std = float(np.std(maes, ddof=1))
    ax.axhline(mean, color="orange", lw=1.5, label=f"mean = {mean:.4f}")
    ax.fill_between([-0.5, len(xs) - 0.5], mean - std, mean + std, color="orange", alpha=0.18,
                    label=f"±1σ = {std:.4f}")
    ax.axhline(0.555, color="#aa6644", linestyle=":", alpha=0.5, label="v1 multi-source 0.555")
    ax.axhline(0.516, color="grey", linestyle="--", alpha=0.5, label="v1 baseline 0.516")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s={s}" for s in seeds])
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("v2 multi-source 4-seed scatter")
    ax.set_ylim(0.42, 0.58)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("v2 (PFA + multi-source) summary  |  Phase 1 + 4-seed verification  |  2026-05-04",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIG / "fig_v2_summary.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_v3_dualstream():
    """4-seed dualstream + aug vs v1 baseline + v2 multi-source (apples-to-apples)."""
    seed_data = []
    for s in (42, 0, 1, 2):
        f = RESULTS / ("dualstream_h128_aug" if s == 42 else f"dualstream_h128_aug_seed{s}") / "metrics.json"
        if not f.exists():
            continue
        m = json.load(open(f))
        seed_data.append((s, m["test_mae"]))
    seed_data.sort(key=lambda x: x[0])
    if not seed_data:
        return
    seeds, maes = zip(*seed_data)
    mean = float(np.mean(maes))
    std = float(np.std(maes, ddof=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                    gridspec_kw={"width_ratios": [3, 1.2]})

    labels = [
        "ALIGNN\n(team repro)",
        "v1 baseline\n4-seed",
        "v2 PFA full\n(seed=42)",
        "v3 dualstream + aug\n4-seed",
        "v2 multi-source\n4-seed (different test)",
    ]
    vals = [0.540, 0.537, 0.519, mean, 0.486]
    errs = [0, 0.014, 0, std, 0.025]
    cols = ["#777777", "#888888", "#1f77b4", "#9467bd", "#2ca02c"]
    xs = np.arange(len(labels))
    bars = ax1.bar(xs, vals, yerr=errs, color=cols, alpha=0.85,
                   edgecolor="black", linewidth=0.8, capsize=6)
    for x, v, e in zip(xs, vals, errs):
        ax1.text(x, v + (e if e > 0 else 0) + 0.012,
                 f"{v:.4f}" + (f"\n±{e:.4f}" if e > 0 else ""),
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.axhline(0.537, color="grey", linestyle="--", alpha=0.55, lw=1)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Test MAE (eV)")
    ax1.set_title("v3 dualstream + leak-free aug: 4-seed verification\n"
                  "Statistically tied with v1 baseline; v2 multi-source remains the lowest reported MAE",
                  loc="left", fontsize=11)
    ax1.set_ylim(0, 0.62)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # right: per-seed scatter
    ys = list(maes)
    ax2.scatter(np.zeros(len(ys)) + 1, ys, s=130, color="#9467bd",
                alpha=0.85, edgecolor="black", zorder=3)
    for s, y in zip(seeds, ys):
        ax2.text(1.04, y, f"seed={s}: {y:.4f}", va="center", fontsize=9)
    ax2.errorbar([0.85], [mean], yerr=std, fmt="s", color="black",
                 ms=10, capsize=8, mfc="orange", zorder=2,
                 label=f"mean = {mean:.4f}\nstd = {std:.4f}")
    ax2.set_xlim(0.6, 1.4)
    ax2.set_xticks([])
    ax2.set_ylabel("Test MAE (eV)")
    ax2.set_title("Per-seed scatter", loc="left", fontsize=11)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(min(ys) - 0.04, max(ys) + 0.04)

    plt.tight_layout()
    out = FIG / "fig_v3_dualstream.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("Generating v2/v3 figures ...")
    fig_v2_ablation()
    fig_v2_multi_source()
    fig_v2_training_curves()
    fig_v2_summary()
    fig_v3_dualstream()
    print("\nDone.")


if __name__ == "__main__":
    main()
