#!/usr/bin/env python3
"""Generate all figures for the DART Q1 paper.

Produces publication-quality figures matching PRB / Nature Comput. Mater. style:
- Fig 1: Architecture diagram (TikZ, separate .tex file)
- Fig 2: Innovation ablation forest plot (bootstrap CIs)
- Fig 3: Ensemble greedy selection curve
- Fig 4: OOD graduated evaluation + per-host breakdown
- Fig 5: Physical interpretability panel (attention + JK + UQ calibration)
- Fig 6: Scaling law (data vs params)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Style setup (PRB-compatible) ──────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'lines.linewidth': 1.0,
    'text.usetex': False,  # safer for portability
    'mathtext.fontset': 'dejavuserif',
})

# PRB-style widths
COL_W = 3.375   # single column (inches)
DBL_W = 7.0     # double column
RESULTS_DIR = Path(__file__).parent.parent / 'results'
FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# Color palette (colorblind-safe, Nature-style)
C_BLUE    = '#2166AC'
C_RED     = '#B2182B'
C_GREEN   = '#1B7837'
C_ORANGE  = '#E66101'
C_PURPLE  = '#7B3294'
C_GRAY    = '#636363'
C_LIGHT   = '#D9D9D9'
C_EQUIV   = '#4393C3'   # equivalent to baseline
C_WORSE   = '#D6604D'   # significantly worse
C_BEST    = '#2166AC'   # best / baseline


# ══════════════════════════════════════════════════════════════════════
# Fig 2: Innovation ablation forest plot
# ══════════════════════════════════════════════════════════════════════
def fig2_innovation_forest():
    """Bootstrap CI forest plot for 12 innovations."""
    # Data from bootstrap analysis (Table 2 in paper)
    innovations = [
        # (label, delta_mae, ci_low, ci_high, p_value)
        ('V4 MoE readout',          -0.002, -0.021, +0.024, 0.87),
        ('V6 Physics features',     +0.001, -0.022, +0.019, 0.92),
        ('V3 DefType cond.',        +0.006, -0.025, +0.013, 0.56),
        ('V14 JK aggregation',      +0.008, -0.012, +0.028, 0.42),
        ('V10 All combined',        +0.011, -0.011, +0.032, 0.31),
        ('EMA averaging',           +0.014, -0.008, +0.034, 0.20),
        # --- significance boundary ---
        ('V9 Defect-host $\\Delta$', +0.029, +0.007, +0.052, 0.009),
        ('V12 LDS + physics',       +0.032, +0.009, +0.055, 0.006),
        ('Focal MAE',               +0.038, +0.014, +0.062, 0.001),
        ('V11 LDS reweight',        +0.052, +0.026, +0.076, 0.0005),
        ('V13 RnC + LDS',           +0.061, +0.037, +0.085, 0.0002),
        ('Heteroscedastic',         +0.085, +0.044, +0.128, 0.0001),
    ]

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))

    n = len(innovations)
    y_pos = np.arange(n)[::-1]

    for i, (label, delta, ci_lo, ci_hi, pval) in enumerate(innovations):
        y = y_pos[i]
        is_sig = pval < 0.05
        color = C_WORSE if is_sig else C_EQUIV
        marker = 's' if is_sig else 'o'

        # CI bar
        ax.plot([ci_lo, ci_hi], [y, y], color=color, linewidth=1.5,
                solid_capstyle='round', zorder=2)
        # Point estimate
        ax.scatter(delta, y, color=color, marker=marker, s=25, zorder=3,
                   edgecolors='white', linewidths=0.3)

        # p-value annotation (right-aligned)
        if pval < 0.001:
            pstr = '$p<$0.001'
        else:
            pstr = f'$p$={pval:.2f}'
        ax.text(0.155, y, pstr, va='center', fontsize=5.5, color=C_GRAY,
                ha='right')

    # Significance boundary line
    boundary_y = (y_pos[5] + y_pos[6]) / 2
    ax.axhline(boundary_y, color=C_GRAY, linewidth=0.5, linestyle='--', alpha=0.7)
    ax.text(-0.04, boundary_y + 0.25, '$\\alpha = 0.05$', fontsize=6,
            color=C_GRAY, ha='center', style='italic')

    # Zero line (baseline)
    ax.axvline(0, color='black', linewidth=0.7, linestyle='-', zorder=1)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([inn[0] for inn in innovations])
    ax.set_xlabel('$\\Delta$MAE (eV)  [innovation $-$ baseline]')
    ax.set_xlim(-0.05, 0.155)

    # Shade regions
    ax.axvspan(-0.05, 0, alpha=0.04, color=C_GREEN, zorder=0)
    ax.axvspan(0, 0.155, alpha=0.04, color=C_RED, zorder=0)
    ax.text(-0.025, -0.8, 'Better', fontsize=6, ha='center', color=C_GREEN, style='italic')
    ax.text(0.075, -0.8, 'Worse', fontsize=6, ha='center', color=C_RED, style='italic')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    fig.savefig(FIG_DIR / 'fig2_innovation_forest.pdf')
    fig.savefig(FIG_DIR / 'fig2_innovation_forest.png')
    plt.close(fig)
    print('[OK] Fig 2: Innovation forest plot')


# ══════════════════════════════════════════════════════════════════════
# Fig 3: Ensemble greedy selection curve
# ══════════════════════════════════════════════════════════════════════
def fig3_ensemble_curve():
    """Greedy ensemble selection curve."""
    # Data from Table 3 in paper
    k_vals  = [1,    2,    3,    4,    5,    6,    7,    8]
    mae_vals = [0.379, 0.360, 0.350, 0.347, 0.345, 0.345, 0.344, 0.344]
    labels  = ['V4 MoE', '+ DART s43', '+ MS deep', '+ V14 JK',
               '+ V6 Phys', '+ V3 DefT', '+ MS shal', '+ DART long']

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))

    # Main curve
    ax.plot(k_vals, mae_vals, 'o-', color=C_BLUE, markersize=4,
            linewidth=1.2, markeredgecolor='white', markeredgewidth=0.4, zorder=3)

    # Prior ensemble reference
    ax.axhline(0.359, color=C_GRAY, linewidth=0.7, linestyle='--', zorder=1)
    ax.text(8.15, 0.359, 'Prior 29-model\nensemble (0.359)', fontsize=5.5,
            va='center', color=C_GRAY)

    # ALIGNN reference
    ax.axhline(0.540, color=C_RED, linewidth=0.7, linestyle=':', zorder=1, alpha=0.6)
    ax.text(8.15, 0.540, 'ALIGNN (0.540)', fontsize=5.5, va='center', color=C_RED, alpha=0.8)

    # DART single reference
    ax.axhline(0.381, color=C_ORANGE, linewidth=0.7, linestyle=':', zorder=1, alpha=0.6)
    ax.text(8.15, 0.381, 'DART single (0.381)', fontsize=5.5, va='center', color=C_ORANGE, alpha=0.8)

    # Selective annotations — only key points to avoid clutter
    key_annotations = {
        0: ('V4 MoE', 5, -10, C_GRAY),
        1: ('+ DART s43', 5, 7, C_GRAY),
        2: ('+ MS deep', 5, -10, C_GRAY),
        3: ('+ V14 JK', 5, 7, C_PURPLE),
    }
    for i, (k, mae, lbl) in enumerate(zip(k_vals, mae_vals, labels)):
        if i in key_annotations:
            _, dx, dy, color = key_annotations[i]
            ax.annotate(key_annotations[i][0], (k, mae),
                        textcoords='offset points', xytext=(dx, dy),
                        fontsize=5, color=color, ha='left')

    ax.set_xlabel('Ensemble size $k$')
    ax.set_ylabel('Test MAE (eV)')
    ax.set_xticks(k_vals)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.330, 0.560)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Highlight best
    ax.scatter([8], [0.344], s=60, facecolors='none', edgecolors=C_BLUE,
               linewidths=1.5, zorder=4)
    ax.annotate('0.344 eV ($-$36%)', (8, 0.344), textcoords='offset points',
                xytext=(-50, -12), fontsize=7, fontweight='bold', color=C_BLUE,
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=0.7))

    fig.savefig(FIG_DIR / 'fig3_ensemble_curve.pdf')
    fig.savefig(FIG_DIR / 'fig3_ensemble_curve.png')
    plt.close(fig)
    print('[OK] Fig 3: Ensemble curve')


# ══════════════════════════════════════════════════════════════════════
# Fig 4: OOD graduated evaluation
# ══════════════════════════════════════════════════════════════════════
def fig4_ood_evaluation():
    """Two-panel: (a) tier comparison bars, (b) per-host LOHO breakdown."""
    # Load OOD data
    with open(RESULTS_DIR / 'ood' / 'ood_summary.json') as f:
        ood_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.4),
                                    gridspec_kw={'width_ratios': [1, 1.3]})

    # Panel (a): Tier comparison
    tiers = ['Tier 0\n(ID, ens.)', 'Tier 0\n(ID, single)', 'Tier 1\n(LOHO)', 'Tier 2\n(Block)']
    dart_mae  = [0.344, 0.379, 0.540, 0.534]
    naive_mae = [None, None, 2.636, 1.754]

    x = np.arange(len(tiers))
    w = 0.3

    bars1 = ax1.bar(x - w/2, dart_mae, w, color=C_BLUE, label='DART', zorder=3,
                    edgecolor='white', linewidth=0.5)
    naive_vals = [v if v is not None else 0 for v in naive_mae]
    naive_colors = [C_RED if v is not None else 'none' for v in naive_mae]
    bars2 = ax1.bar(x + w/2, naive_vals, w, color=naive_colors, label='Naive baseline',
                    zorder=3, edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, val in zip(bars1, dart_mae):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.3f}', ha='center', fontsize=6, color=C_BLUE)
    for bar, val in zip(bars2, naive_mae):
        if val is not None:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{val:.3f}', ha='center', fontsize=6, color=C_RED)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers, fontsize=6.5)
    ax1.set_ylabel('MAE (eV)')
    ax1.set_ylim(0, 3.0)
    ax1.legend(fontsize=6, loc='upper left', frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) Graduated OOD assessment', fontsize=8, pad=6)

    # Degradation factor annotations
    ax1.annotate('$1.0\\times$', (0, 0.344), textcoords='offset points',
                xytext=(0, -12), fontsize=5.5, ha='center', color=C_GRAY)
    ax1.annotate('$1.5\\times$', (2, 0.540), textcoords='offset points',
                xytext=(0, -12), fontsize=5.5, ha='center', color=C_GRAY)
    ax1.annotate('$1.5\\times$', (3, 0.534), textcoords='offset points',
                xytext=(0, -12), fontsize=5.5, ha='center', color=C_GRAY)

    # Panel (b): Per-host LOHO breakdown
    p0 = ood_data['p0_results']
    hosts = ['MoS$_2$', 'MoSe$_2$', 'MoTe$_2$', 'WS$_2$', 'WSe$_2$', 'WTe$_2$', 'MoSSe']
    host_keys = ['MoS2', 'MoSe2', 'MoTe2', 'WS2', 'WSe2', 'WTe2', 'MoSSe']
    dart_loho = [p0[k]['test_mae'] for k in host_keys]
    naive_loho = [p0[k]['naive_mae'] for k in host_keys]

    x2 = np.arange(len(hosts))

    # Color Mo-based vs W-based differently
    colors_host = [C_BLUE, C_BLUE, C_BLUE, C_RED, C_RED, C_RED, C_PURPLE]

    bars_h = ax2.bar(x2, dart_loho, 0.6, color=colors_host, zorder=3,
                     edgecolor='white', linewidth=0.5, alpha=0.85)

    # Naive as gray overlay markers
    ax2.scatter(x2, naive_loho, marker='v', s=20, color=C_GRAY, zorder=4,
                label='Naive baseline')

    for i, (d, n) in enumerate(zip(dart_loho, naive_loho)):
        ax2.text(i, d + 0.04, f'{d:.2f}', ha='center', fontsize=5.5,
                color=colors_host[i])

    ax2.set_xticks(x2)
    ax2.set_xticklabels(hosts, fontsize=6.5)
    ax2.set_ylabel('LOHO MAE (eV)')
    ax2.set_ylim(0, 4.0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(b) Per-host leave-one-out', fontsize=8, pad=6)

    # Legend for Mo vs W
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_BLUE,
               markersize=6, label='Mo-based'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_RED,
               markersize=6, label='W-based'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_PURPLE,
               markersize=6, label='Janus'),
        Line2D([0], [0], marker='v', color=C_GRAY, linestyle='None',
               markersize=5, label='Naive'),
    ]
    ax2.legend(handles=legend_elements, fontsize=5.5, loc='upper right',
               frameon=False, ncol=2)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG_DIR / 'fig4_ood_evaluation.pdf')
    fig.savefig(FIG_DIR / 'fig4_ood_evaluation.png')
    plt.close(fig)
    print('[OK] Fig 4: OOD evaluation')


# ══════════════════════════════════════════════════════════════════════
# Fig 5: Physical interpretability panel
# ══════════════════════════════════════════════════════════════════════
def fig5_interpretability():
    """Three-panel: (a) attention concentration, (b) JK weights, (c) UQ calibration."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(DBL_W, 2.2))

    # Panel (a): Attention concentration comparison
    models = ['DART\nbaseline', 'V3\nDefType', 'V6\nPhysics']
    defect_attn = [99.68, 11.4, 6.0]  # % on defect atom
    expected_uniform = 2.4  # ~1/42 atoms

    bars_a = ax1.bar(models, defect_attn, color=[C_BLUE, C_PURPLE, C_GREEN],
                     width=0.55, edgecolor='white', linewidth=0.5, zorder=3)
    ax1.axhline(expected_uniform, color=C_GRAY, linewidth=0.7, linestyle='--', zorder=1)
    ax1.text(2.35, expected_uniform + 1.5, f'Uniform ({expected_uniform}%)',
             fontsize=5.5, color=C_GRAY)

    for bar, val in zip(bars_a, defect_attn):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val}%', ha='center', fontsize=6.5, fontweight='bold')

    ax1.set_ylabel('Defect atom attention (%)')
    ax1.set_ylim(0, 115)
    ax1.set_title('(a) Pooling attention', fontsize=8, pad=6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel (b): JK layer aggregation weights (V14)
    layers = ['Local\n(L1-3)', 'Global-1\n(L4)', 'Global-2\n(L5)']
    jk_weights = [0.30, 0.31, 0.39]

    jk_colors = ['#E6910180', '#2166AC99', '#2166AC']  # with alpha in hex
    bars_b = ax2.bar(layers, jk_weights, color=jk_colors,
                     width=0.55, edgecolor='white', linewidth=0.5, zorder=3)

    for bar, val in zip(bars_b, jk_weights):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.2f}', ha='center', fontsize=7, fontweight='bold')

    ax2.set_ylabel('JK softmax weight')
    ax2.set_ylim(0, 0.50)
    ax2.set_title('(b) V14 layer weights', fontsize=8, pad=6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel (c): UQ reliability diagram
    # Nominal coverage levels and empirical coverages (from UQ data)
    nominal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    # Raw ensemble coverages (before calibration) - approximate from raw data
    raw_empirical = [0.06, 0.13, 0.19, 0.27, 0.40, 0.47, 0.56, 0.64, 0.72, 0.78]
    # After temperature scaling (tau=2.60)
    cal_empirical = [0.14, 0.24, 0.35, 0.46, 0.55, 0.66, 0.75, 0.85, 0.934, 0.946]

    ax3.plot([0, 1], [0, 1], 'k--', linewidth=0.6, zorder=1, label='Ideal')
    ax3.plot(nominal, raw_empirical, 'o-', color=C_RED, markersize=3,
             linewidth=0.8, label=f'Raw ($\\tau$=1)', zorder=2)
    ax3.plot(nominal, cal_empirical, 's-', color=C_BLUE, markersize=3,
             linewidth=0.8, label=f'Calibrated ($\\tau$=2.60)', zorder=3)

    # Shade ideal zone
    ax3.fill_between([0, 1], [0, 1], alpha=0.03, color='gray', zorder=0)

    # Highlight 90% coverage
    ax3.annotate('93.4% @ 90%', (0.9, 0.934), textcoords='offset points',
                xytext=(-40, 5), fontsize=5.5, color=C_BLUE,
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=0.5))

    ax3.set_xlabel('Nominal coverage')
    ax3.set_ylabel('Empirical coverage')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.legend(fontsize=5.5, loc='lower right', frameon=False)
    ax3.set_title('(c) UQ reliability', fontsize=8, pad=6)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG_DIR / 'fig5_interpretability.pdf')
    fig.savefig(FIG_DIR / 'fig5_interpretability.png')
    plt.close(fig)
    print('[OK] Fig 5: Interpretability panel')


# ══════════════════════════════════════════════════════════════════════
# Fig 6: Scaling law analysis
# ══════════════════════════════════════════════════════════════════════
def fig6_scaling_law():
    """Data scaling vs parameter scaling power laws."""
    with open(RESULTS_DIR / 'scaling_law.json') as f:
        scaling_data = json.load(f)

    runs = scaling_data['runs']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.4))

    # Panel (a): Data scaling (fix h=128)
    data_runs = [r for r in runs if r['hidden_dim'] == 128]
    data_runs.sort(key=lambda r: r['n_train'])
    n_trains = [r['n_train'] for r in data_runs]
    mae_data = [r['test_mae'] for r in data_runs]

    ax1.loglog(n_trains, mae_data, 'o-', color=C_BLUE, markersize=5,
               linewidth=1.2, markeredgecolor='white', markeredgewidth=0.4, zorder=3)

    # Power law fit
    log_n = np.log(n_trains)
    log_mae = np.log(mae_data)
    coeffs = np.polyfit(log_n, log_mae, 1)
    alpha = coeffs[0]
    fit_n = np.linspace(min(n_trains), max(n_trains), 50)
    fit_mae = np.exp(np.polyval(coeffs, np.log(fit_n)))
    ax1.loglog(fit_n, fit_mae, '--', color=C_BLUE, linewidth=0.7, alpha=0.6)

    ax1.set_xlabel('Training samples $N$')
    ax1.set_ylabel('Test MAE (eV)')
    # Note: exponent from abbreviated 30-epoch runs; full 150-epoch analysis
    # gives α≈-0.40. Display the fitted value with context.
    ax1.set_title(f'(a) Data scaling ($\\alpha \\approx$ {alpha:.2f})', fontsize=8, pad=6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel (b): Parameter scaling (fix n=8000 or largest)
    param_runs_8k = [r for r in runs if r['n_train'] == 8000]
    param_runs_8k.sort(key=lambda r: r['n_params_M'])
    if not param_runs_8k:
        param_runs_8k = [r for r in runs if r['n_train'] == max(r['n_train'] for r in runs)]
        param_runs_8k.sort(key=lambda r: r['n_params_M'])

    params_M = [r['n_params_M'] for r in param_runs_8k]
    mae_params = [r['test_mae'] for r in param_runs_8k]

    ax2.semilogx(params_M, mae_params, 's-', color=C_RED, markersize=5,
                 linewidth=1.2, markeredgecolor='white', markeredgewidth=0.4, zorder=3)

    # Fit
    if len(params_M) > 1:
        log_p = np.log(params_M)
        log_mae_p = np.log(mae_params)
        coeffs_p = np.polyfit(log_p, log_mae_p, 1)
        beta = coeffs_p[0]
        fit_p = np.linspace(min(params_M), max(params_M), 50)
        fit_mae_p = np.exp(np.polyval(coeffs_p, np.log(fit_p)))
        ax2.semilogx(fit_p, fit_mae_p, '--', color=C_RED, linewidth=0.7, alpha=0.6)
        ax2.set_title(f'(b) Param scaling ($\\beta \\approx$ {beta:.2f})', fontsize=8, pad=6)
    else:
        ax2.set_title('(b) Parameter scaling', fontsize=8, pad=6)

    ax2.set_xlabel('Parameters (M)')
    ax2.set_ylabel('Test MAE (eV)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG_DIR / 'fig6_scaling_law.pdf')
    fig.savefig(FIG_DIR / 'fig6_scaling_law.png')
    plt.close(fig)
    print('[OK] Fig 6: Scaling law')


# ══════════════════════════════════════════════════════════════════════
# Fig 1: Architecture diagram (TikZ)
# ══════════════════════════════════════════════════════════════════════
def fig1_architecture_tikz():
    """Generate TikZ code for the architecture diagram."""
    tikz_code = r"""\documentclass[border=3pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric, fit,
                 decorations.pathreplacing, calc, backgrounds}
\usepackage{amsmath,amssymb}

\definecolor{localblue}{HTML}{4393C3}
\definecolor{globalorange}{HTML}{E66101}
\definecolor{innovgreen}{HTML}{1B7837}
\definecolor{readoutpurple}{HTML}{7B3294}
\definecolor{inputgray}{HTML}{D9D9D9}
\definecolor{bgblue}{HTML}{DEEBF7}
\definecolor{bgorange}{HTML}{FEE6CE}
\definecolor{bggreen}{HTML}{D5E8D4}

\begin{document}
\begin{tikzpicture}[
    node distance=0.5cm and 0.8cm,
    block/.style={draw, rounded corners=2pt, minimum height=0.65cm,
                  minimum width=2.0cm, font=\footnotesize, thick,
                  align=center},
    localblock/.style={block, fill=bgblue, draw=localblue},
    globalblock/.style={block, fill=bgorange, draw=globalorange},
    innovblock/.style={block, fill=bggreen, draw=innovgreen},
    ioblock/.style={block, fill=inputgray!50, draw=gray},
    readblock/.style={block, fill=readoutpurple!15, draw=readoutpurple},
    arrow/.style={-{Stealth[length=4pt]}, thick},
    label/.style={font=\scriptsize, text=gray},
    bracetext/.style={font=\scriptsize, text=gray, align=center},
]

% === Input ===
\node[ioblock, minimum width=3cm] (input) {Crystal graph $\mathcal{G}$\\
  \scriptsize $\mathbf{x}_i$ (9-dim) $+$ $\delta_i$ defect flag};

% === Embedding ===
\node[ioblock, below=0.4cm of input, minimum width=3cm] (embed) {
  Node embedding\\[-2pt]
  \scriptsize $\mathbf{h}_i^{(0)} = \mathbf{W}\mathbf{x}_i + \mathbf{e}_{\mathrm{def}}(\delta_i)$};

% === Local layers ===
\node[localblock, below=0.5cm of embed, minimum width=3cm] (local) {
  Local interaction $\times 3$\\[-2pt]
  \scriptsize Bond + angle convolution};

% === Innovation: Local env enrichment ===
\node[innovblock, right=0.6cm of local, minimum width=2.4cm] (envrich) {
  \textbf{(II)} Env enrichment\\[-2pt]
  \scriptsize CN, $\bar{d}$, $\Delta\chi$};

% === Global layers ===
\node[globalblock, below=0.5cm of local, minimum width=3cm] (global) {
  Global self-attention $\times 2$\\[-2pt]
  \scriptsize Distance-biased $\phi(d_{ij}^{\mathrm{PBC}})$};

% === Innovation: Pre-norm ===
\node[innovblock, right=0.6cm of global, minimum width=2.4cm] (prenorm) {
  \textbf{(III)} Pre-norm residual\\[-2pt]
  \scriptsize LN $\to$ SubLayer $\to$ Add};

% === Innovation: Gated pooling ===
\node[readblock, below=0.5cm of global, minimum width=3cm] (pool) {
  \textbf{(I)} Gated attn pooling\\[-2pt]
  \scriptsize $\sigma(\mathbf{W}_g) \odot$ AttnPool $+$ MaxPool};

% === MLP head ===
\node[readblock, below=0.4cm of pool, minimum width=3cm] (mlp) {
  MLP readout $\to$ $\hat{E}_f$};

% === Arrows ===
\draw[arrow] (input)  -- (embed);
\draw[arrow] (embed)  -- (local);
\draw[arrow] (local)  -- (global);
\draw[arrow] (global) -- (pool);
\draw[arrow] (pool)   -- (mlp);

% Innovation arrows
\draw[arrow, innovgreen, dashed] (envrich) -- (local);
\draw[arrow, innovgreen, dashed] (prenorm) -- (global);

% === Side braces ===
\draw[decorate, decoration={brace, amplitude=5pt, mirror},
      localblue, thick]
  ($(local.south west) + (-0.15, 0)$) --
  ($(local.north west) + (-0.15, 0)$)
  node[midway, left=6pt, bracetext, text=localblue]
  {Short-range\\$r < 5$\,\AA};

\draw[decorate, decoration={brace, amplitude=5pt, mirror},
      globalorange, thick]
  ($(global.south west) + (-0.15, 0)$) --
  ($(global.north west) + (-0.15, 0)$)
  node[midway, left=6pt, bracetext, text=globalorange]
  {Long-range\\$r < 12$\,\AA};

% === Legend ===
\node[below=0.8cm of mlp, font=\scriptsize, text=gray, align=center] (legend) {
  \textcolor{localblue}{$\blacksquare$} Local layers \quad
  \textcolor{globalorange}{$\blacksquare$} Global layers \quad
  \textcolor{innovgreen}{$\blacksquare$} Innovations \quad
  \textcolor{readoutpurple}{$\blacksquare$} Readout
};

\end{tikzpicture}
\end{document}
"""
    with open(FIG_DIR / 'fig1_architecture.tex', 'w') as f:
        f.write(tikz_code)
    print('[OK] Fig 1: Architecture TikZ source')


# ══════════════════════════════════════════════════════════════════════
# Fig 7: Architecture ablation waterfall chart
# ══════════════════════════════════════════════════════════════════════
def fig7_arch_ablation():
    """Waterfall chart showing cumulative improvement from each innovation."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    components = ['Base\n(mean pool,\npost-norm)', '+Gated\nPooling',
                  '+Env\nEnrich', '+Pre-\nNorm', 'Combined\n(DART)']
    mae_values = [0.516, 0.414, 0.423, 0.408, 0.381]
    delta_pct  = [0, -19.9, -18.0, -21.0, -26.2]

    x = np.arange(len(components))
    colors = [C_GRAY, C_BLUE, C_GREEN, C_ORANGE, C_BLUE]

    bars = ax.bar(x, mae_values, width=0.6, color=colors, edgecolor='white',
                  linewidth=0.5, zorder=3, alpha=0.85)

    # Value + delta labels
    for i, (bar, mae, dp) in enumerate(zip(bars, mae_values, delta_pct)):
        y_top = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y_top + 0.008,
                f'{mae:.3f}', ha='center', fontsize=6.5, fontweight='bold')
        if dp != 0:
            ax.text(bar.get_x() + bar.get_width()/2, y_top - 0.025,
                    f'{dp:+.1f}%', ha='center', fontsize=5.5, color='white',
                    fontweight='bold')

    # ALIGNN reference
    ax.axhline(0.540, color=C_RED, linewidth=0.7, linestyle=':', alpha=0.6)
    ax.text(4.35, 0.540, 'ALIGNN\n(0.540)', fontsize=5.5, va='center',
            color=C_RED, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=6.5)
    ax.set_ylabel('Test MAE (eV)')
    ax.set_ylim(0.30, 0.58)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(FIG_DIR / 'fig7_arch_ablation.pdf')
    fig.savefig(FIG_DIR / 'fig7_arch_ablation.png')
    plt.close(fig)
    print('[OK] Fig 7: Architecture ablation waterfall')


# ══════════════════════════════════════════════════════════════════════
# Fig 8: Per-range MAE decomposition (why auxiliary losses fail)
# ══════════════════════════════════════════════════════════════════════
def fig_per_range():
    """Per-range MAE heatmap showing why auxiliary losses fail."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    # Per-range ΔMAE data (innovation − baseline) from per-range analysis
    # Ranges: [0,2), [2,5), [5,7), [7,25)
    # Data proportions: 48.5%, 34.2%, 10.2%, 7.1%
    methods = ['V11 LDS', 'Focal MAE', 'V13 RnC+LDS', 'Heterosced.']
    ranges = ['[0, 2) eV\n48.5%', '[2, 5) eV\n34.2%', '[5, 7) eV\n10.2%', '[7, 25) eV\n7.1%']

    # ΔMAE (positive = worse) per range for each method
    delta_data = np.array([
        [+0.074, +0.020, -0.032, -0.126],   # V11 LDS
        [+0.049, +0.031, -0.015, -0.124],   # Focal MAE
        [+0.082, +0.040, +0.011, -0.098],   # V13 RnC+LDS
        [+0.035, +0.015, +0.088, +0.695],   # Heteroscedastic
    ])

    x = np.arange(len(ranges))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    method_colors = [C_BLUE, C_ORANGE, C_PURPLE, C_RED]

    for i, (method, color) in enumerate(zip(methods, method_colors)):
        vals = delta_data[i]
        bar_colors = [C_GREEN if v < 0 else color for v in vals]
        bars = ax.bar(x + offsets[i], vals, width, color=bar_colors,
                      edgecolor='white', linewidth=0.3, label=method, zorder=3,
                      alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.6, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ranges, fontsize=6.5)
    ax.set_ylabel('$\\Delta$MAE (eV) [innovation $-$ baseline]')
    ax.set_ylim(-0.20, 0.80)

    # Shade regions
    ax.axhspan(-0.20, 0, alpha=0.03, color=C_GREEN, zorder=0)
    ax.axhspan(0, 0.80, alpha=0.03, color=C_RED, zorder=0)
    ax.text(-0.55, -0.10, 'Better', fontsize=5.5, color=C_GREEN, style='italic',
            rotation=90, va='center')
    ax.text(-0.55, 0.25, 'Worse', fontsize=5.5, color=C_RED, style='italic',
            rotation=90, va='center')

    ax.legend(fontsize=5.5, ncol=2, loc='upper left', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(FIG_DIR / 'fig_per_range.pdf')
    fig.savefig(FIG_DIR / 'fig_per_range.png')
    plt.close(fig)
    print('[OK] Fig 8: Per-range decomposition')


# ══════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating DART Q1 paper figures...')
    print(f'Output directory: {FIG_DIR}')
    print()

    fig1_architecture_tikz()
    fig2_innovation_forest()
    fig3_ensemble_curve()
    fig4_ood_evaluation()
    fig5_interpretability()
    fig6_scaling_law()
    fig7_arch_ablation()
    fig_per_range()

    print()
    print(f'All figures saved to {FIG_DIR}/')
    print('PDF + PNG versions for all data figures.')
    print('TikZ source for architecture diagram (compile separately).')
