"""Render the physical error map figures for PRM.

Reads only the canonical G3 figure sidecar
(``artifacts/prm_g3/canonical/physical_figure_data.csv``) and writes the
main-text gated-effects figure (``fig_error_map.pdf``: adjusted profiles and
mismatch contrasts) plus the supplemental macro figure
(``fig_error_map_macro.pdf``: XF diagnostic and family macro errors with
identity counts).  Encoding follows the manuscript's
accessibility contract: the two incorporation classes are separated by hue
(Okabe--Ito blue/vermillion, CVD-validated), line style, and marker shape,
so the figure stays legible in grayscale; gated effects are filled markers,
ungated ones open.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CLASS_COLOR = {"adsorbate": "#0072B2", "interstitial": "#D55E00"}
CLASS_STYLE = {"adsorbate": "-", "interstitial": "--"}
CLASS_MARKER = {"adsorbate": "o", "interstitial": "s"}
OVERALL_COLOR = "#3a3a3a"

FEATURE_SHORT = {
    "local_signed_valence_mismatch": "signed valence",
    "local_abs_valence_mismatch": "|valence|",
    "local_signed_electronegativity_mismatch": "signed electroneg.",
    "local_abs_electronegativity_contrast": "|electroneg.|",
    "e_module_effective_abs_electronegativity_contrast": "|electroneg.| (E module)",
    "local_signed_covalent_radius_mismatch_A": "signed radius",
    "local_abs_covalent_radius_mismatch_A": "|radius|",
}
GROUP_SHORT = {
    "chalcogenide": "chalcogenide",
    "carbide_mxene_like": "carbide/MXene-like",
    "elemental_hydrogenated": "elemental/hydrogenated",
    "halide_halochalcogenide": "halide/halochalc.",
    "oxide": "oxide",
    "3d": "3d", "4d": "4d", "5d": "5d",
    "main_group": "main group",
    "f_block_other": "f-block/other",
}
PROFILE_LABEL = {
    "cn_5A": "coordination number (5 Å)",
    "postrelaxation_min_clearance_A": "min. clearance (Å)",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel_a(axes, data: pd.DataFrame) -> None:
    profiles = data[(data.panel == "a") & (data.record_type == "adjusted_profile")]
    for ax, feature in zip(axes, ("cn_5A", "postrelaxation_min_clearance_A")):
        for class_name in ("adsorbate", "interstitial"):
            sub = profiles[
                (profiles.feature == feature)
                & (profiles.defecttype == class_name)
            ].sort_values("feature_value")
            ax.fill_between(
                sub.feature_value, sub.pointwise_ci_low_eV,
                sub.pointwise_ci_high_eV, color=CLASS_COLOR[class_name],
                alpha=0.16, linewidth=0,
            )
            ax.plot(
                sub.feature_value, sub.predicted_abs_error_eV,
                CLASS_STYLE[class_name], color=CLASS_COLOR[class_name],
                linewidth=1.4, label=class_name,
            )
        ax.set_xlabel(PROFILE_LABEL[feature])
    axes[0].set_ylabel(r"adjusted $|$error$|$ (eV)")
    axes[0].legend(frameon=False, loc="upper left", handlelength=1.6)


def panel_b(ax, data: pd.DataFrame) -> None:
    effects = data[(data.panel == "b") & (data.record_type == "adjusted_effect")]
    order = [
        "local_signed_valence_mismatch", "local_abs_valence_mismatch",
        "local_signed_electronegativity_mismatch",
        "local_abs_electronegativity_contrast",
        "e_module_effective_abs_electronegativity_contrast",
        "local_signed_covalent_radius_mismatch_A",
        "local_abs_covalent_radius_mismatch_A",
    ]
    for row_index, feature in enumerate(order):
        y_base = len(order) - 1 - row_index
        for class_name, offset in (("adsorbate", 0.18), ("interstitial", -0.18)):
            row = effects[
                (effects.feature == feature) & (effects.defecttype == class_name)
            ].iloc[0]
            gated = int(row.paper_effect_gate) == 1
            color = CLASS_COLOR[class_name]
            ax.errorbar(
                row.contrast_eV, y_base + offset,
                xerr=[[row.contrast_eV - row.ci_low_eV],
                      [row.ci_high_eV - row.contrast_eV]],
                fmt=CLASS_MARKER[class_name], markersize=4.2,
                markerfacecolor=color if gated else "white",
                markeredgecolor=color, color=color,
                elinewidth=1.0, capsize=1.6, capthick=1.0,
            )
    ax.axvline(0.0, color="0.55", linewidth=0.7, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FEATURE_SHORT[f] for f in reversed(order)])
    ax.set_xlabel(r"q10$\rightarrow$q90 contrast (eV)")
    ax.set_ylim(-0.6, len(order) - 0.4)


def panel_c(ax, data: pd.DataFrame) -> None:
    obs = data[(data.panel == "c") & (data.record_type == "sample_observation")]
    for class_name in ("adsorbate", "interstitial"):
        sub = obs[obs.defecttype == class_name]
        ax.scatter(
            sub.extension_factor, sub.pair_absolute_error_eV,
            s=2.5, marker=CLASS_MARKER[class_name],
            color=CLASS_COLOR[class_name], alpha=0.18, linewidths=0,
            rasterized=True, label=class_name,
        )
    ax.axvline(2.0, color="0.3", linewidth=0.8, linestyle=":")
    ax.text(2.12, 12.0, "XF = 2", fontsize=7, color="0.25")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("out-of-plane expansion factor XF")
    ax.set_ylabel(r"pair-OOF $|$error$|$ (eV)")
    legend = ax.legend(frameon=False, loc="lower left", handletextpad=0.2,
                       markerscale=3.0)
    for handle in legend.legend_handles:
        handle.set_alpha(0.9)


def panel_d(ax, data: pd.DataFrame) -> None:
    families = data[(data.panel == "d") & (data.record_type == "family_mae")]
    host_order = [
        "oxide", "carbide_mxene_like", "halide_halochalcogenide",
        "elemental_hydrogenated", "chalcogenide",
    ]
    series_order = ["f_block_other", "5d", "4d", "3d", "main_group"]
    groups = [("host_family", g) for g in host_order] + [
        ("impurity_series", g) for g in series_order
    ]
    positions = {}
    y = len(groups) + 0.5
    for axis, group in groups:
        y -= 1.0
        if (axis, group) == ("impurity_series", series_order[0]):
            y -= 0.7
        positions[(axis, group)] = y
    for (axis, group), y_pos in positions.items():
        for defecttype, offset in (
            ("overall", 0.0), ("adsorbate", 0.24), ("interstitial", -0.24),
        ):
            row = families[
                (families.axis == axis) & (families.group == group)
                & (families.defecttype == defecttype)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            if not np.isfinite(row.sample_weighted_mae_eV):
                continue
            color = OVERALL_COLOR if defecttype == "overall" else CLASS_COLOR[defecttype]
            marker = "D" if defecttype == "overall" else CLASS_MARKER[defecttype]
            ax.errorbar(
                row.sample_weighted_mae_eV, y_pos + offset,
                xerr=[[row.sample_weighted_mae_eV - row.sample_weighted_ci_low_eV],
                      [row.sample_weighted_ci_high_eV - row.sample_weighted_mae_eV]],
                fmt=marker, markersize=3.4 if defecttype == "overall" else 3.8,
                color=color, elinewidth=0.9, capsize=1.4, capthick=0.9,
            )
    counts = {}
    for (axis, group) in positions:
        row = families[
            (families.axis == axis) & (families.group == group)
            & (families.defecttype == "overall")
        ]
        if not row.empty:
            row = row.iloc[0]
            counts[(axis, group)] = (int(row.n_identities), int(row.n_samples))
    tick_positions = [positions[key] for key in positions]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([
        f"{GROUP_SHORT[group]} ({counts[(axis, group)][0]}/{counts[(axis, group)][1]})"
        if (axis, group) in counts else GROUP_SHORT[group]
        for axis, group in positions
    ])
    divider_y = (positions[("host_family", host_order[-1])]
                 + positions[("impurity_series", series_order[0])]) / 2
    ax.axhline(divider_y, color="0.8", linewidth=0.6)
    top = max(tick_positions) + 0.8
    ax.text(0.02, top, "host family", fontsize=7, color="0.25", style="italic")
    ax.text(
        0.02, positions[("impurity_series", series_order[0])] + 0.55,
        "impurity series", fontsize=7, color="0.25", style="italic",
    )
    ax.errorbar([], [], fmt="D", color=OVERALL_COLOR, markersize=3.4,
                label="overall")
    ax.set_xlabel("sample-weighted MAE (eV)")
    ax.set_ylim(min(tick_positions) - 0.8, top + 0.5)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-data", default=str(
        ROOT / "artifacts/prm_g3/canonical/physical_figure_data.csv"
    ))
    parser.add_argument("--output", default=str(
        ROOT / "paper_Q1/figures/fig_error_map.pdf"
    ))
    parser.add_argument("--preview", default=None)
    args = parser.parse_args()
    data = pd.read_csv(args.figure_data)
    if set(data.evidence_tier.dropna().unique()) != {"canonical"}:
        raise ValueError("figure data is not uniformly canonical tier")
    style()

    # Main-text figure: gated physics only (profiles + mismatch contrasts).
    figure = plt.figure(figsize=(7.05, 2.9))
    outer = figure.add_gridspec(
        1, 2, wspace=0.30, left=0.10, right=0.985, top=0.93, bottom=0.155,
    )
    slot_a = outer[0, 0].subgridspec(1, 2, wspace=0.08)
    ax_a1 = figure.add_subplot(slot_a[0, 0])
    ax_a2 = figure.add_subplot(slot_a[0, 1], sharey=ax_a1)
    plt.setp(ax_a2.get_yticklabels(), visible=False)
    ax_b = figure.add_subplot(outer[0, 1])
    panel_a((ax_a1, ax_a2), data)
    panel_b(ax_b, data)
    for label, ax, x_off in (("(a)", ax_a1, -0.28), ("(b)", ax_b, -0.42)):
        ax.text(x_off, 1.03, label, transform=ax.transAxes,
                fontweight="bold", fontsize=9, va="bottom")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400)
    if args.preview:
        figure.savefig(args.preview, dpi=180)
    print(f"wrote {output}")

    # Supplemental macro figure: XF diagnostic + family macro errors.
    figure = plt.figure(figsize=(7.05, 3.1))
    outer = figure.add_gridspec(
        1, 2, wspace=0.42, left=0.09, right=0.985, top=0.93, bottom=0.15,
    )
    ax_c = figure.add_subplot(outer[0, 0])
    ax_d = figure.add_subplot(outer[0, 1])
    panel_c(ax_c, data)
    panel_d(ax_d, data)
    for label, ax, x_off in (("(a)", ax_c, -0.13), ("(b)", ax_d, -0.58)):
        ax.text(x_off, 1.03, label, transform=ax.transAxes,
                fontweight="bold", fontsize=9, va="bottom")
    macro_output = output.parent / "fig_error_map_macro.pdf"
    figure.savefig(macro_output, dpi=400)
    print(f"wrote {macro_output}")


if __name__ == "__main__":
    main()
