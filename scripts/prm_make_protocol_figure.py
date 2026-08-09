"""Generate the protocol overview figure from the frozen PRM artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ase.data import atomic_numbers
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent.parent
COLORS = {
    "ink": "#222222",
    "muted": "#666666",
    "grid": "#D9D9D9",
    "train": "#B8B8B8",
    "validation": "#3B7EA1",
    "calibration": "#E5A33D",
    "test": "#B64B4B",
    "adsorbate": "#2A6F97",
    "interstitial": "#C15B38",
    "host_atom": "#6D8EA0",
    "impurity_atom": "#C15B38",
}

SELECTED_CONFIG = ROOT / "configs/prm/promoted/g111/transfer/id_cv5_f0_seed242.yaml"
MODEL_SOURCE = ROOT / "src/models/crystal_v2.py"
GLOBAL_BLOCK_SOURCE = ROOT / "src/models/baseline.py"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def read_samples(path: Path) -> list[Dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            **row,
            "sample_index": int(row["sample_index"]),
            "target_eV": float(row["target_eV"]),
        }
        for row in rows
    ]


def retained_samples(
    samples: Sequence[Mapping[str, Any]], audit: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    excluded = set(
        int(index)
        for index in audit["duplicates"]["canonical_deduplication"]["excluded_indices"]
    )
    retained = [row for row in samples if int(row["sample_index"]) not in excluded]
    expected = int(audit["dataset"]["n_modeling_samples"])
    if len(retained) != expected:
        raise ValueError(f"expected {expected} retained samples, found {len(retained)}")
    return retained


def chemistry_matrix(
    samples: Sequence[Mapping[str, Any]],
) -> Tuple[list[str], list[str], np.ndarray]:
    hosts = sorted({str(row["host"]) for row in samples})
    dopants = sorted(
        {str(row["dopant"]) for row in samples},
        key=lambda symbol: (atomic_numbers.get(symbol, 999), symbol),
    )
    host_position = {name: index for index, name in enumerate(hosts)}
    dopant_position = {name: index for index, name in enumerate(dopants)}
    matrix = np.zeros((len(hosts), len(dopants)), dtype=int)
    for row in samples:
        matrix[host_position[str(row["host"])], dopant_position[str(row["dopant"])]] += 1
    return hosts, dopants, matrix


def split_counts(path: Path) -> Dict[str, int]:
    payload = json.loads(path.read_text())
    return {
        partition: len(payload.get(partition, []))
        for partition in ("train", "val", "calibration", "test")
    }


def mean_split_counts(paths: Iterable[Path]) -> Dict[str, float]:
    rows = [split_counts(path) for path in paths]
    if not rows:
        raise ValueError("no split files supplied")
    return {
        partition: float(np.mean([row[partition] for row in rows]))
        for partition in ("train", "val", "calibration", "test")
    }


def partition_profiles(split_dir: Path) -> list[Dict[str, Any]]:
    profiles = [
        {
            "label": "Paired ID",
            **mean_split_counts(sorted(split_dir.glob("id_repeat_s*.json"))),
        },
        {
            "label": "Random OOF",
            **mean_split_counts(sorted(split_dir.glob("id_cv5_f*.json"))),
        },
        {
            "label": "Host OOF",
            **mean_split_counts(sorted(split_dir.glob("host_cv5_f*.json"))),
        },
        {
            "label": "Impurity OOF",
            **mean_split_counts(sorted(split_dir.glob("dopant_cv5_f*.json"))),
        },
        {
            "label": "Pair OOF",
            **mean_split_counts(sorted(split_dir.glob("pair_cv5_f*.json"))),
        },
        {
            "label": "Chem. block",
            **split_counts(split_dir / "chemistry_block_g6x3d.json"),
        },
        {
            "label": "UQ holdout",
            **split_counts(split_dir / "uq_calibration_s62.json"),
        },
    ]
    for profile in profiles:
        profile["total"] = sum(
            float(profile[name]) for name in ("train", "val", "calibration", "test")
        )
    return profiles


def latex_formula(text: str) -> str:
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", text)
    if not parts or "".join(element + number for element, number in parts) != text:
        return text
    body = "".join(
        element + (rf"$_{{{number}}}$" if number else "")
        for element, number in parts
    )
    return body


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "mathtext.fontset": "dejavuserif",
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.08, 1.04, label, transform=axis.transAxes, ha="left", va="bottom",
        fontsize=8.5, fontweight="bold", color=COLORS["ink"],
    )


def draw_provenance(axis: plt.Axes, audit: Mapping[str, Any]) -> None:
    raw = audit["formation_energy_provenance"]["raw_database_audit"]["filter_replay"]
    canonical = audit["duplicates"]["canonical_deduplication"]
    steps = [
        ("ASE source", int(raw["raw_rows"]), ""),
        (
            "Source filter", int(raw["valid_after_filter"]),
            f"-{raw['not_converged']} unconverged; "
            f"-{raw['missing_or_nonfinite_eform']} missing; "
            f"-{raw['outside_abs_20_eV']} outside range",
        ),
        (
            "Raw-energy audit", int(raw["valid_after_filter"]) - int(canonical["n_raw_component_excluded"]),
            f"-{canonical['n_raw_component_excluded']} zero-sentinel rows",
        ),
        (
            "Impurity-identity audit", int(canonical["n_pre_dedup_eligible"]),
            f"-{canonical['n_ambiguous_identity_excluded']} non-unique atom labels",
        ),
        (
            "Canonical set", int(canonical["n_retained"]),
            f"-{canonical['n_duplicate_excluded']} redundant structures",
        ),
    ]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    y_positions = np.linspace(0.91, 0.09, len(steps))
    for index, ((title, count, note), y) in enumerate(zip(steps, y_positions)):
        face = "#F3F3F3" if index < len(steps) - 1 else "#E7F0F4"
        edge = COLORS["grid"] if index < len(steps) - 1 else COLORS["validation"]
        box = FancyBboxPatch(
            (0.08, y - 0.058), 0.84, 0.116,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=0.7, edgecolor=edge, facecolor=face,
        )
        axis.add_patch(box)
        axis.text(0.12, y + 0.018, title, ha="left", va="center", fontsize=6.2)
        axis.text(
            0.88, y + 0.018, f"{count:,}", ha="right", va="center",
            fontsize=7.0, fontweight="bold", color=COLORS["ink"],
        )
        if note:
            axis.text(
                0.12, y - 0.025, note, ha="left", va="center",
                fontsize=4.8, color=COLORS["muted"], linespacing=1.05,
            )
        if index < len(steps) - 1:
            next_y = y_positions[index + 1]
            axis.annotate(
                "", xy=(0.5, next_y + 0.064), xytext=(0.5, y - 0.064),
                arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": COLORS["muted"]},
            )
    panel_label(axis, "(a)")
    axis.set_title("Data provenance", loc="left", x=0.055, pad=5)


def draw_target_distribution(
    axis: plt.Axes, samples: Sequence[Mapping[str, Any]],
) -> None:
    arrays = {
        defect_type: np.asarray(
            [row["target_eV"] for row in samples if row["defecttype"] == defect_type],
            dtype=float,
        )
        for defect_type in ("adsorbate", "interstitial")
    }
    bins = np.linspace(-20.0, 20.0, 61)
    for defect_type in ("adsorbate", "interstitial"):
        axis.hist(
            arrays[defect_type], bins=bins, density=True, histtype="step",
            linewidth=1.2, color=COLORS[defect_type],
            label=f"{defect_type.capitalize()} (n={len(arrays[defect_type]):,})",
        )
    axis.axvline(0.0, color=COLORS["muted"], lw=0.6, ls="--", zorder=0)
    axis.set_xlim(-20, 20)
    axis.set_xlabel(r"Formation energy $E_{\mathrm{f}}$ (eV)")
    axis.set_ylabel("Density")
    axis.legend(frameon=False, loc="upper left", handlelength=1.5)
    axis.spines[["top", "right"]].set_visible(False)
    panel_label(axis, "(b)")
    axis.set_title("Canonical target distribution", loc="left", x=0.055, pad=5)


def sparse_ticks(values: Sequence[str], maximum: int) -> Tuple[np.ndarray, list[str]]:
    step = max(1, int(np.ceil(len(values) / maximum)))
    positions = np.arange(0, len(values), step)
    return positions, [values[index] for index in positions]


def draw_chemistry_matrix(
    axis: plt.Axes, hosts: Sequence[str], dopants: Sequence[str], matrix: np.ndarray,
) -> None:
    palette = ListedColormap(["#FFFFFF", "#DDE8ED", "#8EB5C5", "#3B7EA1", "#1E4D63"])
    norm = BoundaryNorm([-0.5, 0.5, 2.5, 5.5, 10.5, matrix.max() + 0.5], palette.N)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=palette, norm=norm)
    x_positions, x_labels = sparse_ticks(dopants, 17)
    y_positions, y_labels = sparse_ticks(hosts, 14)
    axis.set_xticks(x_positions, x_labels, rotation=90)
    axis.set_yticks(y_positions, [latex_formula(label) for label in y_labels])
    axis.set_xlabel("Impurity element (atomic-number order)")
    axis.set_ylabel("Host")
    axis.tick_params(length=2, pad=1)
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(COLORS["grid"])
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
    colorbar.set_ticks([0, 1.5, 4, 8, min(13, matrix.max())])
    colorbar.set_ticklabels(["0", "1-2", "3-5", "6-10", ">10"])
    colorbar.ax.tick_params(labelsize=5.5, width=0.4, length=2)
    colorbar.set_label("Structures", fontsize=6.5)
    panel_label(axis, "(c)")
    axis.set_title(
        f"Host-impurity coverage ({len(hosts)} hosts x {len(dopants)} elements)",
        loc="left", pad=5,
    )


def draw_partition_profiles(axis: plt.Axes, profiles: Sequence[Mapping[str, Any]]) -> None:
    labels = [str(profile["label"]) for profile in profiles]
    y = np.arange(len(profiles))
    left = np.zeros(len(profiles), dtype=float)
    names = ("train", "val", "calibration", "test")
    display = {"train": "Train", "val": "Validation", "calibration": "Calibration", "test": "Test"}
    hatches = {"train": "", "val": "///", "calibration": "xx", "test": "..."}
    for name in names:
        fractions = np.asarray(
            [float(profile[name]) / float(profile["total"]) for profile in profiles]
        )
        axis.barh(
            y, fractions, left=left, height=0.66,
            color=COLORS["validation" if name == "val" else name],
            edgecolor=COLORS["ink"], linewidth=0.35, hatch=hatches[name],
            label=display[name],
        )
        left += fractions
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    axis.set_xlabel("Partition fraction (%)")
    axis.legend(
        frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.01),
        ncol=4, columnspacing=0.7, handlelength=1.0,
    )
    axis.grid(axis="x", color=COLORS["grid"], lw=0.45, zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    panel_label(axis, "(d)")
    axis.set_title("Frozen evaluation partitions", loc="left", pad=12, fontsize=7.2)


def architecture_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str = "",
    *,
    facecolor: str = "#F4F4F4",
    edgecolor: str = COLORS["grid"],
    title_color: str = COLORS["ink"],
    fontsize: float = 5.3,
) -> None:
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=0.65, edgecolor=edgecolor, facecolor=facecolor,
    )
    axis.add_patch(box)
    x, y = xy
    axis.text(
        x + width / 2, y + height * (0.64 if body else 0.50), title,
        ha="center", va="center", fontsize=fontsize, fontweight="bold",
        color=title_color, linespacing=1.0,
    )
    if body:
        axis.text(
            x + width / 2, y + height * 0.27, body,
            ha="center", va="center", fontsize=fontsize - 0.4,
            color=COLORS["muted"], linespacing=1.0,
        )


def architecture_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
) -> None:
    axis.annotate(
        "", xy=end, xytext=start,
        arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": color,
                    "shrinkA": 0, "shrinkB": 0},
    )


def prepare_architecture_axis(axis: plt.Axes, panel: str, title: str) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    panel_label(axis, panel)
    axis.set_title(title, loc="left", x=0.02, pad=5, fontsize=7.2)


def draw_periodic_graph_schematic(axis: plt.Axes) -> None:
    prepare_architecture_axis(axis, "(a)", "Periodic defect graph")
    axis.add_patch(Rectangle(
        (0.08, 0.29), 0.84, 0.59, fill=False, lw=0.75,
        edgecolor=COLORS["muted"], linestyle=(0, (3, 2)),
    ))
    rows = []
    for row in range(4):
        y = 0.38 + row * 0.145
        offset = 0.08 if row % 2 else 0.0
        for column in range(5):
            x = 0.18 + column * 0.16 + offset
            if x < 0.90:
                rows.append((x, y))
    centre_index = min(
        range(len(rows)),
        key=lambda index: (rows[index][0] - 0.50) ** 2
        + (rows[index][1] - 0.60) ** 2,
    )
    centre = rows[centre_index]
    for index, first in enumerate(rows):
        for second in rows[index + 1:]:
            distance = float(np.hypot(first[0] - second[0], first[1] - second[1]))
            if distance < 0.185:
                axis.plot(
                    [first[0], second[0]], [first[1], second[1]],
                    color="#B7C5CC", lw=0.55, zorder=1,
                )
    axis.add_patch(Circle(
        centre, 0.245, fill=False, lw=0.9,
        edgecolor=COLORS["impurity_atom"], linestyle=(0, (3, 2)), zorder=0,
    ))
    for index, position in enumerate(rows):
        is_impurity = index == centre_index
        axis.add_patch(Circle(
            position, 0.027 if not is_impurity else 0.038,
            facecolor=(COLORS["impurity_atom"] if is_impurity else COLORS["host_atom"]),
            edgecolor="white", lw=0.5, zorder=3,
        ))
    axis.annotate(
        "unique impurity", xy=centre, xytext=(0.08, 0.92),
        ha="left", va="center", fontsize=5.1, color=COLORS["impurity_atom"],
        arrowprops={"arrowstyle": "->", "lw": 0.6,
                    "color": COLORS["impurity_atom"]},
    )
    axis.text(
        0.50, 0.25, "5 \u00c5 periodic pair/angle graph",
        ha="center", va="center", fontsize=5.2, color=COLORS["ink"],
    )
    axis.text(
        0.50, 0.13, "all-pairs minimum-image distances",
        ha="center", va="center", fontsize=5.0, color=COLORS["muted"],
    )
    axis.text(
        0.50, 0.05, "schematic; not a selected material",
        ha="center", va="center", fontsize=4.5, color=COLORS["muted"],
    )


def draw_atom_encoding(axis: plt.Axes) -> None:
    prepare_architecture_axis(axis, "(b)", "Atom encoding and E")
    architecture_box(
        axis, (0.04, 0.77), 0.42, 0.13, "Fixed ct-UAE", "128 features",
        facecolor="#EDF3F6", edgecolor=COLORS["validation"],
    )
    architecture_box(
        axis, (0.54, 0.77), 0.42, 0.13, "Element table", "9 normalized attributes",
    )
    architecture_arrow(axis, (0.25, 0.75), (0.43, 0.67))
    architecture_arrow(axis, (0.75, 0.75), (0.57, 0.67))
    architecture_box(
        axis, (0.22, 0.56), 0.56, 0.11, "Linear 137 \u2192 128",
        facecolor="#F4F4F4",
    )
    architecture_arrow(axis, (0.50, 0.55), (0.50, 0.48))
    architecture_box(
        axis, (0.08, 0.35), 0.84, 0.13,
        "+ learned impurity-status embedding", "one compositionally unique node",
        facecolor="#F9EEE9", edgecolor=COLORS["impurity_atom"],
    )
    architecture_arrow(axis, (0.50, 0.34), (0.50, 0.27))
    architecture_box(
        axis, (0.04, 0.07), 0.92, 0.20,
        "E \u2014 defect-local enrichment",
        "coordination; mean/max distance;\n"
        "absolute electronegativity contrast;\n"
        "projected and added only at the impurity",
        facecolor="#FFF4DB", edgecolor=COLORS["calibration"], fontsize=4.9,
    )


def draw_interaction_stack(axis: plt.Axes) -> None:
    prepare_architecture_axis(axis, "(c)", "Local and global context")
    architecture_box(
        axis, (0.05, 0.57), 0.90, 0.31,
        "P \u00d7 3 \u2014 composite local block",
        "Pre-LayerNorm; 32-distance-RBF filter;\n"
        "scalar distance gate; 32-angle-RBF triplets;\n"
        "sum aggregation and residual update (5 \u00c5 graph)",
        facecolor="#EAF4EC", edgecolor="#3B7D5A", fontsize=5.2,
    )
    architecture_arrow(axis, (0.50, 0.55), (0.50, 0.47))
    architecture_box(
        axis, (0.05, 0.15), 0.90, 0.32,
        "Geometric Transformer \u00d7 2",
        "4-head all-atom self-attention; learned\n"
        "per-head radial bias from minimum-image distance;\n"
        "32 RBF centres spanning 0\u201312 \u00c5",
        facecolor="#EDF3F6", edgecolor=COLORS["validation"], fontsize=5.2,
    )
    axis.text(
        0.50, 0.06, "12 \u00c5 is the radial grid endpoint, not an attention cutoff",
        ha="center", va="center", fontsize=4.5, color=COLORS["muted"],
    )


def draw_graph_readout(axis: plt.Axes) -> None:
    prepare_architecture_axis(axis, "(d)", "G readout and prediction")
    architecture_box(
        axis, (0.04, 0.77), 0.43, 0.14,
        "Atom-attention", "weighted sum", facecolor="#EDF3F6",
        edgecolor=COLORS["validation"],
    )
    architecture_box(
        axis, (0.53, 0.77), 0.43, 0.14,
        "Channelwise", "maximum", facecolor="#F4F4F4",
    )
    architecture_arrow(axis, (0.25, 0.75), (0.43, 0.66))
    architecture_arrow(axis, (0.75, 0.75), (0.57, 0.66))
    architecture_box(
        axis, (0.16, 0.52), 0.68, 0.14,
        "G \u2014 scalar-gated fusion", "graph-dependent mixing weight",
        facecolor="#E7F0F4", edgecolor=COLORS["validation"],
    )
    architecture_arrow(axis, (0.50, 0.51), (0.50, 0.43))
    architecture_box(
        axis, (0.18, 0.29), 0.64, 0.14,
        "LayerNorm + MLP", "128 \u2192 128 \u2192 1",
    )
    architecture_arrow(axis, (0.50, 0.28), (0.50, 0.20))
    architecture_box(
        axis, (0.24, 0.07), 0.52, 0.13,
        r"Formation energy $\hat E_{\mathrm{f}}$", facecolor="#F9EEE9",
        edgecolor=COLORS["impurity_atom"], title_color=COLORS["impurity_atom"],
    )


def build_architecture_summary(
    selected_config: Path = SELECTED_CONFIG,
    model_source: Path = MODEL_SOURCE,
    global_block_source: Path = GLOBAL_BLOCK_SOURCE,
) -> Dict[str, Any]:
    config = yaml.safe_load(selected_config.read_text())
    kwargs = config["model_kwargs"]
    required_flags = {
        "use_gated_pooling": True,
        "use_env_enrichment": True,
        "use_prenorm_local": True,
    }
    for name, expected in required_flags.items():
        if bool(kwargs.get(name)) is not expected:
            raise ValueError(f"selected architecture requires {name}={expected}")
    architecture = {
        "selected_variant": "g111",
        "selection_data": "validation only",
        "atom_input_dimension": 128 + int(kwargs["atom_fea_len"]),
        "ct_uae_dimension": 128,
        "elemental_attribute_dimension": int(kwargs["atom_fea_len"]),
        "hidden_dimension": int(kwargs["hidden_dim"]),
        "local_layers": int(kwargs["n_local_layers"]),
        "global_layers": int(kwargs["n_global_layers"]),
        "attention_heads": int(kwargs["num_heads"]),
        "local_graph_radius_A": float(kwargs["rcut_local"]),
        "radial_bias_grid_endpoint_A": float(kwargs["dmax_global"]),
        "radial_bias_grid_is_attention_cutoff": False,
        "environment_features": [
            "coordination_count", "mean_neighbor_distance",
            "maximum_neighbor_distance", "absolute_electronegativity_contrast",
        ],
        "modules": {"G": True, "E": True, "P": True},
    }
    return {
        "architecture": architecture,
        "inputs": {
            "selected_config": {
                "path": repository_path(selected_config),
                "sha256": file_sha256(selected_config),
            },
            "model_source": {
                "path": repository_path(model_source),
                "sha256": file_sha256(model_source),
            },
            "global_block_source": {
                "path": repository_path(global_block_source),
                "sha256": file_sha256(global_block_source),
            },
        },
    }


def make_architecture_figure(output_pdf: Path, output_png: Path | None) -> None:
    configure_style()
    figure, axes = plt.subplots(
        1, 4, figsize=(7.05, 3.10),
        gridspec_kw={"width_ratios": [1.02, 1.05, 1.17, 1.00]},
    )
    draw_periodic_graph_schematic(axes[0])
    draw_atom_encoding(axes[1])
    draw_interaction_stack(axes[2])
    draw_graph_readout(axes[3])
    figure.subplots_adjust(
        left=0.035, right=0.992, bottom=0.045, top=0.90, wspace=0.20,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf)
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_png)
    plt.close(figure)


def build_summary(
    protocol_dir: Path,
) -> Tuple[Dict[str, Any], list[Mapping[str, Any]], Tuple[list[str], list[str], np.ndarray]]:
    audit_path = protocol_dir / "data_audit.json"
    sample_path = protocol_dir / "samples.csv"
    manifest_path = protocol_dir / "manifest.json"
    audit = json.loads(audit_path.read_text())
    samples = read_samples(sample_path)
    retained = retained_samples(samples, audit)
    hosts, dopants, matrix = chemistry_matrix(retained)
    profiles = partition_profiles(protocol_dir / "splits")
    canonical = audit["duplicates"]["canonical_deduplication"]
    defect_counts = Counter(str(row["defecttype"]) for row in retained)
    summary = {
        "schema_version": "prm_protocol_figure_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "protocol_manifest": {
                "path": repository_path(manifest_path), "sha256": file_sha256(manifest_path)
            },
            "data_audit": {
                "path": repository_path(audit_path), "sha256": file_sha256(audit_path)
            },
            "sample_table": {
                "path": repository_path(sample_path), "sha256": file_sha256(sample_path)
            },
        },
        "counts": {
            "raw_rows": audit["formation_energy_provenance"]["raw_database_audit"]["filter_replay"]["raw_rows"],
            "source_filtered_rows": audit["dataset"]["n_samples"],
            "raw_component_exclusions": canonical["n_raw_component_excluded"],
            "ambiguous_identity_exclusions": canonical["n_ambiguous_identity_excluded"],
            "duplicate_exclusions": canonical["n_duplicate_excluded"],
            "canonical_rows": canonical["n_retained"],
            "hosts": len(hosts),
            "dopants": len(dopants),
            "host_dopant_cells": int(np.count_nonzero(matrix)),
            "defect_types": dict(sorted(defect_counts.items())),
        },
        "partition_profiles": profiles,
    }
    if int(matrix.sum()) != int(canonical["n_retained"]):
        raise ValueError("chemistry matrix does not sum to the canonical set")
    return summary, retained, (hosts, dopants, matrix)


def make_figure(
    protocol_dir: Path,
    output_pdf: Path,
    output_png: Path | None,
    architecture_output_pdf: Path,
    architecture_output_png: Path | None,
    sidecar: Path,
    selected_config: Path = SELECTED_CONFIG,
    model_source: Path = MODEL_SOURCE,
    global_block_source: Path = GLOBAL_BLOCK_SOURCE,
) -> Dict[str, Any]:
    configure_style()
    summary, samples, chemistry = build_summary(protocol_dir)
    audit = json.loads((protocol_dir / "data_audit.json").read_text())
    profiles = summary["partition_profiles"]

    figure = plt.figure(figsize=(7.05, 5.05))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(0.34, 0.66), height_ratios=(0.60, 0.40),
        left=0.075, right=0.965, bottom=0.105, top=0.955, wspace=0.34, hspace=0.42,
    )
    provenance_axis = figure.add_subplot(grid[0, 0])
    target_axis = figure.add_subplot(grid[1, 0])
    chemistry_axis = figure.add_subplot(grid[0, 1])
    partition_axis = figure.add_subplot(grid[1, 1])

    draw_provenance(provenance_axis, audit)
    draw_target_distribution(target_axis, samples)
    draw_chemistry_matrix(chemistry_axis, *chemistry)
    draw_partition_profiles(partition_axis, profiles)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf)
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_png)
    plt.close(figure)

    architecture_summary = build_architecture_summary(
        selected_config, model_source, global_block_source,
    )
    make_architecture_figure(architecture_output_pdf, architecture_output_png)
    summary["architecture"] = architecture_summary["architecture"]
    summary["inputs"].update(architecture_summary["inputs"])

    summary["outputs"] = {
        "protocol_pdf": {
            "path": repository_path(output_pdf), "sha256": file_sha256(output_pdf)
        },
        "architecture_pdf": {
            "path": repository_path(architecture_output_pdf),
            "sha256": file_sha256(architecture_output_pdf),
        },
    }
    if output_png is not None:
        summary["outputs"]["protocol_png"] = {
            "path": repository_path(output_png), "sha256": file_sha256(output_png)
        }
    if architecture_output_png is not None:
        summary["outputs"]["architecture_png"] = {
            "path": repository_path(architecture_output_png),
            "sha256": file_sha256(architecture_output_png),
        }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v2",
    )
    parser.add_argument(
        "--output-pdf", type=Path,
        default=ROOT / "paper_Q1/figures/fig_protocol_overview.pdf",
    )
    parser.add_argument(
        "--output-png", type=Path,
        default=ROOT / "paper_Q1/figures/fig_protocol_overview.png",
    )
    parser.add_argument(
        "--architecture-output-pdf", type=Path,
        default=ROOT / "paper_Q1/figures/fig_dart_architecture.pdf",
    )
    parser.add_argument(
        "--architecture-output-png", type=Path,
        default=ROOT / "paper_Q1/figures/fig_dart_architecture.png",
    )
    parser.add_argument(
        "--selected-config", type=Path, default=SELECTED_CONFIG,
    )
    parser.add_argument(
        "--model-source", type=Path, default=MODEL_SOURCE,
    )
    parser.add_argument(
        "--global-block-source", type=Path, default=GLOBAL_BLOCK_SOURCE,
    )
    parser.add_argument(
        "--sidecar", type=Path,
        default=ROOT / "artifacts/prm_results/paper/protocol_figure.json",
    )
    args = parser.parse_args()
    summary = make_figure(
        args.protocol_dir.resolve(), args.output_pdf.resolve(),
        args.output_png.resolve() if args.output_png else None,
        args.architecture_output_pdf.resolve(),
        (
            args.architecture_output_png.resolve()
            if args.architecture_output_png else None
        ),
        args.sidecar.resolve(),
        args.selected_config.resolve(), args.model_source.resolve(),
        args.global_block_source.resolve(),
    )
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
