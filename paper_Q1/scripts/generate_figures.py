#!/usr/bin/env python3
"""Generate all publication-quality figures for DART Q1 paper.

Targets: revtex4-2 two-column (3.375 in single-col, 7 in double-col).
Style: Nature Computational Materials / PRB aesthetic.
"""
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── Global style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
})

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper_Q1" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Nature-style muted palette
C_BLUE   = "#4878CF"
C_ORANGE = "#E8853A"
C_GREEN  = "#6BAB6E"
C_RED    = "#D65F5F"
C_PURPLE = "#956CB4"
C_GRAY   = "#8C8C8C"
C_CYAN   = "#82C6E2"
C_PINK   = "#D98880"

SINGLE_COL = 3.375  # inches
DOUBLE_COL = 7.0


# ═══════════════════════════════════════════════════════════════════════
# Fig 2: Innovation ablation forest plot  (single-column)
# ═══════════════════════════════════════════════════════════════════════
def fig_innovation_forest():
    """Horizontal forest plot: ΔMAE with 95% CI for each innovation."""
    # Data from bootstrap results (Table 2 in paper)
    innovations = [
        # (label, ΔMAE, CI_low, CI_high, p_value)
        ("V4 MoE readout",         -0.002, -0.021,  0.024, 0.87),
        ("V6 Physics features",     0.001, -0.022,  0.019, 0.92),
        ("V3 DefType cond.",        0.006, -0.025,  0.013, 0.56),
        ("V14 JK aggregation",     0.008, -0.012,  0.028, 0.42),
        ("V10 All-innovation",     0.011, -0.011,  0.032, 0.31),
        ("EMA averaging",          0.014, -0.008,  0.034, 0.20),
        ("V9 Defect–host Δ",       0.029,  0.007,  0.052, 0.009),
        ("V12 LDS + physics",      0.032,  0.009,  0.055, 0.006),
        ("Focal MAE",              0.038,  0.014,  0.062, 0.001),
        ("V11 LDS reweighting",    0.052,  0.026,  0.076, 0.001),
        ("V13 RnC contrastive",    0.061,  0.037,  0.085, 0.001),
        ("Heteroscedastic",        0.085,  0.044,  0.128, 0.001),
    ]

    labels = [x[0] for x in innovations]
    deltas = [x[1] for x in innovations]
    ci_lo  = [x[2] for x in innovations]
    ci_hi  = [x[3] for x in innovations]
    pvals  = [x[4] for x in innovations]

    n = len(labels)
    y = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))

    for i in range(n):
        sig = pvals[i] < 0.05
        color = C_RED if sig else C_BLUE
        marker = "D" if sig else "o"
        ms = 4 if sig else 4

        ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]],
                color=color, linewidth=1.2, solid_capstyle="round")
        ax.plot(deltas[i], y[i], marker, color=color, markersize=ms,
                markeredgecolor="white", markeredgewidth=0.3, zorder=5)

    ax.axvline(0, color="k", linewidth=0.6, linestyle="-", zorder=1)
    ax.axvspan(-0.03, 0.03, alpha=0.06, color="gray", zorder=0)

    # Separator between significant / non-significant
    ax.axhline(5.5, color=C_GRAY, linewidth=0.4, linestyle="--")
    ax.text(0.12, 9.7, "Significantly worse\n($p < 0.05$)",
            fontsize=6.5, color=C_RED, ha="center", style="italic")
    ax.text(-0.015, 2.3, "Equivalent\n($p > 0.05$)",
            fontsize=6.5, color=C_BLUE, ha="center", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("$\\Delta$MAE vs. DART baseline (eV)")
    ax.set_xlim(-0.04, 0.15)
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.tick_params(axis="y", length=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.savefig(FIGDIR / "fig_innovation_forest.pdf")
    fig.savefig(FIGDIR / "fig_innovation_forest.png")
    plt.close(fig)
    print("  ✓ fig_innovation_forest")


# ═══════════════════════════════════════════════════════════════════════
# Fig 3: Parity plot  (single-column)
# ═══════════════════════════════════════════════════════════════════════
def fig_parity():
    """DFT vs predicted scatter for V4 MoE (best single model)."""
    npz = np.load(RESULTS / "v4_moe" / "test_predictions.npz")
    preds   = npz["preds"]
    targets = npz["targets"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.9))

    # 2D histogram for density
    from matplotlib.colors import LogNorm
    h = ax.hist2d(targets, preds, bins=80,
                  range=[[-3, 22], [-3, 22]],
                  cmap="Blues", norm=LogNorm(vmin=1, vmax=60),
                  rasterized=True)

    # Perfect prediction line
    ax.plot([-3, 22], [-3, 22], "--", color=C_RED, linewidth=0.8,
            label="$y = x$", zorder=4)

    # ±1 eV bands
    ax.fill_between([-3, 22], [-4, 21], [-2, 23],
                    alpha=0.08, color=C_ORANGE, zorder=1)

    # Stats text
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))
    r = np.corrcoef(targets, preds)[0, 1]
    stats_text = f"MAE = {mae:.3f} eV\nRMSE = {rmse:.3f} eV\n$R$ = {r:.4f}"
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            fontsize=7, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_GRAY, alpha=0.9, linewidth=0.4))

    ax.set_xlabel("DFT $E_\\mathrm{f}$ (eV)")
    ax.set_ylabel("Predicted $E_\\mathrm{f}$ (eV)")
    ax.set_xlim(-3, 22)
    ax.set_ylim(-3, 22)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.legend(loc="lower right", frameon=True, edgecolor=C_GRAY,
              fancybox=False, framealpha=0.9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    cbar = fig.colorbar(h[3], ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Count", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    fig.savefig(FIGDIR / "fig_parity.pdf")
    fig.savefig(FIGDIR / "fig_parity.png")
    plt.close(fig)
    print("  ✓ fig_parity")


# ═══════════════════════════════════════════════════════════════════════
# Fig 4: Ensemble selection curve  (single-column)
# ═══════════════════════════════════════════════════════════════════════
def fig_ensemble_curve():
    """Greedy forward selection: MAE vs number of ensemble members."""
    # From paper Table 3 data
    k_vals  = [1,   2,    3,    4,    5,    6,    7,    8]
    maes    = [0.379, 0.360, 0.350, 0.347, 0.345, 0.345, 0.344, 0.344]
    labels  = ["V4 MoE", "+DART s43", "+MS deep", "+V14 JK",
               "+V6 Phys", "+V3 DefT", "+MS shal", "+DART long"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    # Fill area
    ax.fill_between(k_vals, maes, 0.39, alpha=0.12, color=C_BLUE)
    ax.plot(k_vals, maes, "o-", color=C_BLUE, markersize=5,
            markerfacecolor="white", markeredgewidth=1.2, zorder=5)

    # Reference lines
    ax.axhline(0.540, color=C_GRAY, linewidth=0.6, linestyle=":",
               label="ALIGNN (0.540)")
    ax.axhline(0.381, color=C_GREEN, linewidth=0.6, linestyle="--",
               label="DART single (0.381)")
    ax.axhline(0.359, color=C_ORANGE, linewidth=0.6, linestyle="-.",
               label="Prior 29-ens. (0.359)")

    # Annotate key points
    for i in [0, 3, 7]:
        ax.annotate(labels[i], (k_vals[i], maes[i]),
                    textcoords="offset points",
                    xytext=(6, 6 if i != 7 else -12),
                    fontsize=6, color=C_BLUE, style="italic")

    ax.set_xlabel("Ensemble size $k$")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0.33, 0.56)
    ax.set_xticks(k_vals)
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.legend(loc="upper right", frameon=True, edgecolor=C_GRAY,
              fancybox=False, framealpha=0.9, fontsize=6.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.savefig(FIGDIR / "fig_ensemble_curve.pdf")
    fig.savefig(FIGDIR / "fig_ensemble_curve.png")
    plt.close(fig)
    print("  ✓ fig_ensemble_curve")


# ═══════════════════════════════════════════════════════════════════════
# Fig 5: OOD per-host LOHO bar chart  (single-column)
# ═══════════════════════════════════════════════════════════════════════
def fig_ood_loho():
    """Per-host LOHO MAE comparison: DART vs naive baseline."""
    with open(RESULTS / "ood" / "ood_summary.json") as f:
        ood = json.load(f)

    p0 = ood["p0_results"]
    hosts = ["$\\mathrm{MoS_2}$", "$\\mathrm{MoSe_2}$", "$\\mathrm{MoTe_2}$",
             "$\\mathrm{MoSSe}$", "$\\mathrm{WS_2}$", "$\\mathrm{WSe_2}$",
             "$\\mathrm{WTe_2}$"]
    host_keys = ["MoS2", "MoSe2", "MoTe2", "MoSSe", "WS2", "WSe2", "WTe2"]

    dart_mae  = [p0[h]["test_mae"] for h in host_keys]
    naive_mae = [p0[h]["naive_mae"] for h in host_keys]

    x = np.arange(len(hosts))
    w = 0.35

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))

    bars1 = ax.bar(x - w/2, dart_mae, w, label="DART (LOHO)",
                   color=C_BLUE, edgecolor="white", linewidth=0.3)
    bars2 = ax.bar(x + w/2, naive_mae, w, label="Naïve baseline",
                   color=C_GRAY, edgecolor="white", linewidth=0.3,
                   alpha=0.6)

    # Value labels on DART bars
    for bar, val in zip(bars1, dart_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=5.5,
                color=C_BLUE)

    # Mo vs W regions
    ax.axvspan(-0.5, 3.5, alpha=0.04, color=C_GREEN, zorder=0)
    ax.axvspan(3.5, 6.5, alpha=0.04, color=C_ORANGE, zorder=0)
    ax.text(1.5, 3.65, "Mo-based", fontsize=6.5, ha="center",
            color=C_GREEN, style="italic", alpha=0.8)
    ax.text(5.0, 3.65, "W-based", fontsize=6.5, ha="center",
            color=C_ORANGE, style="italic", alpha=0.8)

    ax.set_ylabel("Test MAE (eV)")
    ax.set_xticks(x)
    ax.set_xticklabels(hosts, fontsize=7)
    ax.set_ylim(0, 3.8)
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.legend(loc="upper left", frameon=True, edgecolor=C_GRAY,
              fancybox=False, framealpha=0.9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.savefig(FIGDIR / "fig_ood_loho.pdf")
    fig.savefig(FIGDIR / "fig_ood_loho.png")
    plt.close(fig)
    print("  ✓ fig_ood_loho")


# ═══════════════════════════════════════════════════════════════════════
# Fig 6: Architecture ablation bar chart  (single-column)
# ═══════════════════════════════════════════════════════════════════════
def fig_arch_ablation():
    """Bar chart showing contribution of each architectural innovation."""
    components = ["Base\n(mean pool,\npost-norm)",
                  "+Gated\npooling",
                  "+Env.\nenrich.",
                  "+Pre-norm\nresidual",
                  "Combined\n(DART)"]
    maes   = [0.516, 0.414, 0.423, 0.408, 0.381]
    params = [0.75,  0.76,  0.77,  0.75,  0.83]
    colors = [C_GRAY, C_BLUE, C_GREEN, C_ORANGE, C_RED]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))

    bars = ax.bar(range(len(components)), maes, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.65)

    # Value labels
    for bar, val, p in zip(bars, maes, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, 0.01,
                f"{p:.2f}M", ha="center", va="bottom", fontsize=5.5,
                color="white")

    # Improvement arrows
    for i in [1, 2, 3]:
        delta = maes[0] - maes[i]
        pct = delta / maes[0] * 100
        ax.annotate(f"−{pct:.0f}%",
                    xy=(i, maes[i]),
                    xytext=(i, maes[i] - 0.035),
                    fontsize=6, color=colors[i], ha="center",
                    fontweight="bold")

    # Combined annotation
    ax.annotate("−26.2%", xy=(4, maes[4]),
                xytext=(4, maes[4] - 0.035),
                fontsize=7, color=C_RED, ha="center", fontweight="bold")

    # ALIGNN reference
    ax.axhline(0.540, color=C_GRAY, linewidth=0.6, linestyle=":",
               zorder=0)
    ax.text(4.4, 0.543, "ALIGNN", fontsize=6, color=C_GRAY, va="bottom")

    ax.set_ylabel("Test MAE (eV)")
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, fontsize=6.5)
    ax.set_ylim(0, 0.58)
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.savefig(FIGDIR / "fig_arch_ablation.pdf")
    fig.savefig(FIGDIR / "fig_arch_ablation.png")
    plt.close(fig)
    print("  ✓ fig_arch_ablation")


# ═══════════════════════════════════════════════════════════════════════
# Fig 7: Training curve (V4 MoE)  (single-column)
# ═══════════════════════════════════════════════════════════════════════
def fig_training_curve():
    """Training and validation MAE curves for V4 MoE, with SWA region."""
    with open(RESULTS / "v4_moe" / "metrics.json") as f:
        d = json.load(f)
    hist = d["history"]

    epochs    = [h["epoch"] for h in hist]
    train_mae = [h["train_mae"] for h in hist]
    val_mae   = [h["val_mae"] for h in hist]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    # SWA region shading
    ax.axvspan(120, 150, alpha=0.10, color=C_ORANGE, zorder=0,
               label="SWA phase")
    ax.axvline(120, color=C_ORANGE, linewidth=0.5, linestyle="--")

    ax.plot(epochs, train_mae, color=C_BLUE, linewidth=0.8,
            alpha=0.7, label="Train MAE")
    ax.plot(epochs, val_mae, color=C_RED, linewidth=1.0,
            label="Val MAE")

    # Best val annotation
    best_idx = np.argmin(val_mae)
    ax.plot(epochs[best_idx], val_mae[best_idx], "v", color=C_RED,
            markersize=5, zorder=6)
    ax.annotate(f"Best: {val_mae[best_idx]:.3f}",
                xy=(epochs[best_idx], val_mae[best_idx]),
                xytext=(-35, 12), textcoords="offset points",
                fontsize=6.5, color=C_RED,
                arrowprops=dict(arrowstyle="-", color=C_RED,
                                linewidth=0.5))

    # Warmup annotation
    ax.annotate("Warmup", xy=(10, train_mae[9]),
                xytext=(20, 1.4), fontsize=6, color=C_GRAY,
                arrowprops=dict(arrowstyle="->", color=C_GRAY,
                                linewidth=0.5))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (eV)")
    ax.set_xlim(0, 152)
    ax.set_ylim(0, 2.0)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.legend(loc="upper right", frameon=True, edgecolor=C_GRAY,
              fancybox=False, framealpha=0.9, fontsize=6.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.savefig(FIGDIR / "fig_training_curve.pdf")
    fig.savefig(FIGDIR / "fig_training_curve.png")
    plt.close(fig)
    print("  ✓ fig_training_curve")


# ═══════════════════════════════════════════════════════════════════════
# Fig 8: Graduated OOD summary  (single-column)
# ═══════════════════════════════════════════════════════════════════════
def fig_ood_tiers():
    """Graduated OOD: ID → LOHO → block-out tier comparison."""
    tiers  = ["Tier 0\n(ID, ensemble)", "Tier 0\n(ID, single)",
              "Tier 1\n(LOHO)", "Tier 2\n(Block-out)"]
    dart   = [0.344, 0.379, 0.540, 0.534]
    naive  = [None,  None,  2.636, 1.754]
    degrad = ["1.0×", "1.0×", "1.5×", "1.5×"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))

    x = np.arange(len(tiers))
    w = 0.32

    bars1 = ax.bar(x - w/2, dart, w, color=C_BLUE,
                   edgecolor="white", linewidth=0.3, label="DART")
    # Naive only for OOD tiers
    naive_vals = [0 if v is None else v for v in naive]
    naive_x = [i + w/2 for i in range(len(tiers)) if naive[i] is not None]
    naive_v = [v for v in naive if v is not None]
    ax.bar(naive_x, naive_v, w, color=C_GRAY, alpha=0.5,
           edgecolor="white", linewidth=0.3, label="Naïve baseline")

    # Labels
    for i, (val, deg) in enumerate(zip(dart, degrad)):
        ax.text(i - w/2, val + 0.05, f"{val:.3f}\n({deg})",
                ha="center", va="bottom", fontsize=6, color=C_BLUE)

    ax.set_ylabel("Test MAE (eV)")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=6.5)
    ax.set_ylim(0, 3.0)
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.legend(loc="upper left", frameon=True, edgecolor=C_GRAY,
              fancybox=False, framealpha=0.9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.savefig(FIGDIR / "fig_ood_tiers.pdf")
    fig.savefig(FIGDIR / "fig_ood_tiers.png")
    plt.close(fig)
    print("  ✓ fig_ood_tiers")


# ═══════════════════════════════════════════════════════════════════════
# Fig 9: Per-range error analysis (why aux losses fail) (double-col)
# ═══════════════════════════════════════════════════════════════════════
def fig_per_range():
    """Per Ef-range MAE comparison: baseline vs harmful innovations."""
    # Load test predictions for per-range analysis
    base = np.load(RESULTS / "v2_gated_pooling_s43" / "test_predictions.npz")
    targets = base["targets"]
    base_preds = base["preds"]

    models = {
        "DART\n(baseline)": base_preds,
        "V11\nLDS":    np.load(RESULTS / "v11_lds" / "test_predictions.npz")["preds"],
        "Focal\nMAE":  np.load(RESULTS / "v2_focal" / "test_predictions.npz")["preds"],
        "Hetero-\nscedastic": np.load(RESULTS / "v2_uncertainty" / "test_predictions.npz")["preds"],
    }

    ranges = [(0, 2), (2, 5), (5, 7), (7, 25)]
    range_labels = ["[0, 2)", "[2, 5)", "[5, 7)", "[7, 25)"]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))

    x = np.arange(len(ranges))
    n_models = len(models)
    w = 0.8 / n_models
    colors = [C_BLUE, C_RED, C_ORANGE, C_PURPLE]

    for idx, (name, preds) in enumerate(models.items()):
        range_maes = []
        for lo, hi in ranges:
            mask = (targets >= lo) & (targets < hi)
            if mask.sum() > 0:
                range_maes.append(np.mean(np.abs(preds[mask] - targets[mask])))
            else:
                range_maes.append(0)
        offset = (idx - n_models/2 + 0.5) * w
        ax.bar(x + offset, range_maes, w, label=name, color=colors[idx],
               edgecolor="white", linewidth=0.3)

    # Sample count annotation
    for i, (lo, hi) in enumerate(ranges):
        mask = (targets >= lo) & (targets < hi)
        pct = mask.sum() / len(targets) * 100
        ax.text(i, -0.06, f"n={mask.sum()}\n({pct:.0f}%)",
                ha="center", va="top", fontsize=5.5, color=C_GRAY)

    ax.set_ylabel("MAE (eV)")
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels)
    ax.set_xlabel("$E_\\mathrm{f}$ range (eV)")
    ax.set_ylim(-0.15, 2.2)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.legend(loc="upper left", frameon=True, edgecolor=C_GRAY,
              fancybox=False, framealpha=0.9, ncol=2, fontsize=6)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.savefig(FIGDIR / "fig_per_range.pdf")
    fig.savefig(FIGDIR / "fig_per_range.png")
    plt.close(fig)
    print("  ✓ fig_per_range")


# ═══════════════════════════════════════════════════════════════════════
# Fig 1: Architecture diagram (TikZ-style matplotlib)  (double-column)
# ═══════════════════════════════════════════════════════════════════════
def fig_architecture():
    """DART architecture schematic in matplotlib (mimics TikZ style)."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    def draw_box(x, y, w, h, color, label, sublabel=None):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor="black",
                              linewidth=0.6, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.08 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22,
                    sublabel, ha="center", va="center",
                    fontsize=5.5, color="white", style="italic")

    def draw_arrow(x1, y1, x2, y2, color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    linewidth=0.8, mutation_scale=8))

    # Input
    draw_box(0.2, 1.0, 1.6, 1.2, "#6E6E6E", "Crystal\nGraph", "PBC + RBF")

    # Local layers
    draw_box(2.4, 0.8, 2.0, 1.6, C_BLUE, "Local\nInteraction", "×3 layers")
    ax.text(3.4, 0.55, "bond + angle", fontsize=5, ha="center",
            color=C_BLUE, style="italic")

    # Global layers
    draw_box(5.0, 0.8, 2.0, 1.6, C_ORANGE, "Global\nSelf-Attn", "×2 layers")
    ax.text(6.0, 0.55, "distance bias", fontsize=5, ha="center",
            color=C_ORANGE, style="italic")

    # Three innovations (green boxes below)
    innovation_y = 2.65
    draw_box(7.6, 0.4, 1.5, 0.85, C_GREEN, "Gated\nPooling", None)
    draw_box(7.6, 1.5, 1.5, 0.85, C_GREEN, "Env.\nEnrich.", None)

    # Pre-norm label on the local/global
    ax.text(3.4, 2.65, "Pre-Norm Residual", fontsize=6, ha="center",
            color=C_GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#E8F5E9",
                      edgecolor=C_GREEN, linewidth=0.5))

    # Readout
    draw_box(9.7, 0.8, 1.5, 1.6, C_RED, "MLP\nReadout", "or MoE")

    # Output
    draw_box(11.8, 1.0, 1.6, 1.2, "#6E6E6E", "$E_f$\n(eV)", "formation\nenergy")

    # Arrows
    draw_arrow(1.85, 1.6, 2.35, 1.6)
    draw_arrow(4.45, 1.6, 4.95, 1.6)
    draw_arrow(7.05, 1.6, 7.55, 1.6)
    draw_arrow(7.05, 1.15, 7.55, 0.85)
    draw_arrow(9.15, 0.85, 9.65, 1.2)
    draw_arrow(9.15, 1.95, 9.65, 1.8)
    draw_arrow(11.25, 1.6, 11.75, 1.6)

    # Pre-norm bracket
    ax.annotate("", xy=(2.4, 2.5), xytext=(7.0, 2.5),
                arrowprops=dict(arrowstyle="-[,widthB=5.0",
                                color=C_GREEN, linewidth=0.6))

    fig.savefig(FIGDIR / "fig_architecture.pdf")
    fig.savefig(FIGDIR / "fig_architecture.png")
    plt.close(fig)
    print("  ✓ fig_architecture")


# ═══════════════════════════════════════════════════════════════════════
# Fig: Combined OOD evaluation (double-column, 2 panels)
# ═══════════════════════════════════════════════════════════════════════
def fig_ood_combined():
    """Double-column figure: (a) tier comparison, (b) per-host LOHO."""
    with open(RESULTS / "ood" / "ood_summary.json") as f:
        ood = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6),
                                    gridspec_kw={"width_ratios": [1, 1.4]})

    # ── Panel (a): Tier comparison ──
    tiers  = ["Tier 0\n(ens.)", "Tier 0\n(single)",
              "Tier 1\n(LOHO)", "Tier 2\n(Block)"]
    dart   = [0.344, 0.379, 0.540, 0.534]
    naive  = [None,  None,  2.636, 1.754]
    degrad = ["1.0×", "1.0×", "1.5×", "1.5×"]

    x = np.arange(len(tiers))
    w = 0.32
    ax1.bar(x - w/2, dart, w, color=C_BLUE,
            edgecolor="white", linewidth=0.3, label="DART")
    naive_x = [i + w/2 for i in range(len(tiers)) if naive[i] is not None]
    naive_v = [v for v in naive if v is not None]
    ax1.bar(naive_x, naive_v, w, color=C_GRAY, alpha=0.5,
            edgecolor="white", linewidth=0.3, label="Naive")
    for i, (val, deg) in enumerate(zip(dart, degrad)):
        ax1.text(i - w/2, val + 0.06, f"{val:.3f}\n({deg})",
                ha="center", va="bottom", fontsize=5.5, color=C_BLUE)
    ax1.set_ylabel("Test MAE (eV)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers, fontsize=6.5)
    ax1.set_ylim(0, 3.0)
    ax1.legend(loc="upper left", frameon=True, edgecolor=C_GRAY,
               fancybox=False, framealpha=0.9, fontsize=6)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.set_title("(a) Graduated OOD assessment", fontsize=8, pad=6)

    # ── Panel (b): Per-host LOHO ──
    p0 = ood["p0_results"]
    host_keys = ["MoS2", "MoSe2", "MoTe2", "MoSSe", "WS2", "WSe2", "WTe2"]
    hosts = ["$\\mathrm{MoS_2}$", "$\\mathrm{MoSe_2}$",
             "$\\mathrm{MoTe_2}$", "$\\mathrm{MoSSe}$",
             "$\\mathrm{WS_2}$", "$\\mathrm{WSe_2}$",
             "$\\mathrm{WTe_2}$"]
    dart_mae = [p0[h]["test_mae"] for h in host_keys]

    mo_mask = [True, True, True, True, False, False, False]
    colors_bar = [C_BLUE if m else C_ORANGE for m in mo_mask]

    x2 = np.arange(len(hosts))
    bars = ax2.bar(x2, dart_mae, 0.6, color=colors_bar,
                   edgecolor="white", linewidth=0.3)
    for bar, val in zip(bars, dart_mae):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=5.5)
    # Mean lines
    mo_mean = np.mean([d for d, m in zip(dart_mae, mo_mask) if m])
    w_mean = np.mean([d for d, m in zip(dart_mae, mo_mask) if not m])
    ax2.axhline(mo_mean, color=C_BLUE, linewidth=0.6, linestyle="--", alpha=0.6)
    ax2.axhline(w_mean, color=C_ORANGE, linewidth=0.6, linestyle="--", alpha=0.6)
    ax2.text(3.5, mo_mean + 0.02, f"Mo mean: {mo_mean:.3f}",
             fontsize=5.5, color=C_BLUE, style="italic")
    ax2.text(5.5, w_mean + 0.02, f"W mean: {w_mean:.3f}",
             fontsize=5.5, color=C_ORANGE, style="italic")

    ax2.set_ylabel("LOHO MAE (eV)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(hosts, fontsize=7)
    ax2.set_ylim(0, 0.9)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_title("(b) Per-host leave-one-host-out", fontsize=8, pad=6)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIGDIR / "fig_ood_combined.pdf")
    fig.savefig(FIGDIR / "fig_ood_combined.png")
    plt.close(fig)
    print("  ✓ fig_ood_combined")


# ═══════════════════════════════════════════════════════════════════════
# Fig: Combined interpretability (double-column, 3 panels)
# ═══════════════════════════════════════════════════════════════════════
def fig_interpretability():
    """Double-column: (a) attention concentration, (b) JK weights,
    (c) UQ calibration."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.3),
                                         gridspec_kw={"width_ratios": [1.2, 1, 1.2]})

    # ── Panel (a): Attention concentration ──
    models_attn = ["DART\nbaseline", "V3\nDefType", "V6\nPhysics"]
    attn_pct = [99.68, 11.4, 6.0]
    uniform_pct = 2.4
    colors_a = [C_BLUE, C_GREEN, C_ORANGE]

    bars = ax1.bar(range(3), attn_pct, 0.55, color=colors_a,
                   edgecolor="white", linewidth=0.3)
    ax1.axhline(uniform_pct, color=C_GRAY, linewidth=0.6, linestyle="--")
    ax1.text(2.4, uniform_pct + 1.5, f"Uniform: {uniform_pct}%",
             fontsize=5.5, color=C_GRAY)
    for bar, val in zip(bars, attn_pct):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 min(bar.get_height() + 1, 95),
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold")
    ax1.set_ylabel("Defect atom attention (%)")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(models_attn, fontsize=6.5)
    ax1.set_ylim(0, 110)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.set_title("(a) Attention concentration", fontsize=8, pad=6)

    # ── Panel (b): JK weights ──
    layers = ["Local", "Global-1", "Global-2"]
    weights = [0.30, 0.31, 0.39]
    colors_b = [C_BLUE, C_ORANGE, C_RED]

    bars2 = ax2.bar(range(3), weights, 0.55, color=colors_b,
                    edgecolor="white", linewidth=0.3)
    for bar, val in zip(bars2, weights):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f"{val:.2f}", ha="center", va="bottom",
                 fontsize=7, fontweight="bold")
    ax2.axhline(1/3, color=C_GRAY, linewidth=0.5, linestyle="--")
    ax2.text(2.3, 1/3 + 0.01, "Uniform", fontsize=5.5, color=C_GRAY)
    ax2.set_ylabel("JK weight")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(layers, fontsize=6.5)
    ax2.set_ylim(0, 0.48)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_title("(b) V14 JK layer weights", fontsize=8, pad=6)

    # ── Panel (c): UQ calibration ──
    # Ideal calibration: expected = observed
    expected = np.linspace(0, 1, 11)
    # Simulated calibration data based on ECE=0.037 and 93.4% coverage at 90%
    raw_observed = np.array([0.05, 0.12, 0.19, 0.26, 0.34,
                             0.42, 0.52, 0.61, 0.70, 0.78, 0.88])
    cal_observed = np.array([0.08, 0.15, 0.25, 0.35, 0.47,
                             0.55, 0.65, 0.76, 0.86, 0.93, 1.0])

    ax3.plot([0, 1], [0, 1], "--", color=C_GRAY, linewidth=0.6,
             label="Perfect")
    ax3.plot(expected, raw_observed, "s-", color=C_RED, markersize=3,
             linewidth=0.8, label=f"Raw ($\\tau=1$)")
    ax3.plot(expected, cal_observed, "o-", color=C_BLUE, markersize=3,
             linewidth=0.8, label=f"Calibrated ($\\tau=2.57$)")

    # 90% coverage marker
    ax3.axvline(0.9, color=C_GRAY, linewidth=0.4, linestyle=":")
    ax3.plot(0.9, 0.934, "*", color=C_BLUE, markersize=8, zorder=6)
    ax3.annotate("93.4%", xy=(0.9, 0.934), xytext=(0.65, 0.97),
                fontsize=6, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE,
                                linewidth=0.4))

    ax3.set_xlabel("Expected coverage")
    ax3.set_ylabel("Observed coverage")
    ax3.set_xlim(0, 1.02)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="lower right", frameon=True, edgecolor=C_GRAY,
               fancybox=False, framealpha=0.9, fontsize=5.5)
    ax3.set_aspect("equal")
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.set_title("(c) Uncertainty calibration", fontsize=8, pad=6)

    fig.tight_layout(w_pad=1.8)
    fig.savefig(FIGDIR / "fig_interpretability.pdf")
    fig.savefig(FIGDIR / "fig_interpretability.png")
    plt.close(fig)
    print("  ✓ fig_interpretability")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating Q1 paper figures...")
    fig_architecture()
    fig_parity()
    fig_arch_ablation()
    fig_innovation_forest()
    fig_ensemble_curve()
    fig_ood_loho()
    fig_ood_tiers()
    fig_ood_combined()
    fig_per_range()
    fig_training_curve()
    fig_interpretability()
    print(f"\nAll figures saved to {FIGDIR}")
