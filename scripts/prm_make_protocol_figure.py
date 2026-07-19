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
from ase.data import atomic_numbers
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch


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
}


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
    for name in names:
        fractions = np.asarray(
            [float(profile[name]) / float(profile["total"]) for profile in profiles]
        )
        axis.barh(
            y, fractions, left=left, height=0.66, color=COLORS["validation" if name == "val" else name],
            edgecolor="white", linewidth=0.35, label=display[name],
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
        "schema_version": "prm_protocol_figure_v2",
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
    protocol_dir: Path, output_pdf: Path, output_png: Path | None, sidecar: Path,
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

    summary["outputs"] = {
        "pdf": {"path": repository_path(output_pdf), "sha256": file_sha256(output_pdf)},
    }
    if output_png is not None:
        summary["outputs"]["png"] = {
            "path": repository_path(output_png), "sha256": file_sha256(output_png)
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
        "--sidecar", type=Path,
        default=ROOT / "artifacts/prm_results/paper/protocol_figure.json",
    )
    args = parser.parse_args()
    summary = make_figure(
        args.protocol_dir.resolve(), args.output_pdf.resolve(),
        args.output_png.resolve() if args.output_png else None, args.sidecar.resolve(),
    )
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
