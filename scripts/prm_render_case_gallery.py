"""Render the structure case gallery for the Supplemental Material.

Pure visualization over existing artifacts: atomic structures come from
``artifacts/prm_assets/case_structures.json`` (extracted verbatim from the
IMP2D source database rows referenced by the frozen case-selection output
``artifacts/prm_g3/canonical/case_selection.csv``), and each panel is
annotated with that row's pair-held-out absolute error.  No training,
inference, or geometry modification happens here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

SYMBOL = {1: "H", 3: "Li", 6: "C", 8: "O", 16: "S", 24: "Cr", 32: "Ge",
          34: "Se", 40: "Zr", 42: "Mo", 50: "Sn", 52: "Te", 74: "W",
          75: "Re", 77: "Ir", 83: "Bi"}
# Jmol/CPK colors and covalent radii (Angstrom) for the elements involved.
COLOR = {"H": "#E8E8E8", "Li": "#CC80FF", "C": "#909090", "O": "#FF0D0D",
         "S": "#E6C200", "Cr": "#8A99C7", "Ge": "#668F8F", "Se": "#FFA100",
         "Zr": "#94E0E0", "Mo": "#54B5B5", "Sn": "#668080", "Te": "#D47A00",
         "W": "#2194D6", "Re": "#267DAB", "Ir": "#175487", "Bi": "#9E4FB5"}
RADIUS = {"H": 0.31, "Li": 1.28, "C": 0.76, "O": 0.66, "S": 1.05,
          "Cr": 1.39, "Ge": 1.20, "Se": 1.20, "Zr": 1.75, "Mo": 1.54,
          "Sn": 1.39, "Te": 1.38, "W": 1.62, "Re": 1.51, "Ir": 1.41,
          "Bi": 1.48}

# Columns follow the frozen case-selection match_id (1--4): top row the
# four highest-error nonpathological interstitials, bottom row the matched
# bottom-quartile adsorbate controls (match 1 has no eligible control).
PANELS_TOP = ["17500", "842", "8763", "1607"]
PANELS_BOTTOM = [None, "2861", "16403", "437"]
ROW_TITLES = ["Highest-error non-pathological interstitials (pair-held-out)",
              "Matched bottom-quartile adsorbate controls (pair-held-out)"]


def load_cases() -> dict[str, dict]:
    cases = {}
    for row in csv.DictReader(open(ROOT / "artifacts/prm_g3/canonical/case_selection.csv")):
        cases[row["raw_row_id"]] = row
    return cases


def replicate(structure: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    cell = np.asarray(structure["cell"])
    positions = np.asarray(structure["positions"])
    symbols = [SYMBOL[z] for z in structure["Z"]]
    impurity_index = next(i for i, s in enumerate(symbols) if symbols.count(s) == 1)
    # Wrap in-plane fractional coordinates and center the impurity laterally.
    inv = np.linalg.inv(cell)
    frac = positions @ inv
    frac[:, :2] -= np.floor(frac[:, :2])
    shift = np.array([0.5, 0.5]) - frac[impurity_index, :2]
    frac[:, :2] = (frac[:, :2] + shift) % 1.0
    tiles, tile_symbols, impurity_flags = [], [], []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            block = frac.copy()
            block[:, 0] += i
            block[:, 1] += j
            tiles.append(block @ cell)
            tile_symbols.extend(symbols)
            impurity_flags.extend(
                [(i == 0 and j == 0 and k == impurity_index)
                 for k in range(len(symbols))])
    coordinates = np.vstack(tiles)
    # Keep a window of one cell width around the centered impurity.
    center = (np.array([0.5, 0.5, 0.0]) @ cell).ravel()
    span = 0.72 * max(np.linalg.norm(cell[0]), np.linalg.norm(cell[1]))
    keep = (np.abs(coordinates[:, 0] - center[0]) < span) \
        & (np.abs(coordinates[:, 1] - center[1]) < span)
    return (coordinates[keep],
            [s for s, k in zip(tile_symbols, keep) if k],
            np.asarray(impurity_flags)[keep])


def draw_panel(ax, structure: dict, meta: dict) -> None:
    coordinates, symbols, impurity = replicate(structure)
    # Side view: x horizontal, z vertical, depth along y (painter's order).
    order = np.argsort(coordinates[:, 1])[::-1]
    radii = np.asarray([RADIUS[s] for s in symbols])
    cut = 1.15 * (radii[:, None] + radii[None, :])
    delta = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    bonded = (delta < cut) & (delta > 0.1)
    for a in range(len(symbols)):
        for b in range(a + 1, len(symbols)):
            if bonded[a, b]:
                ax.plot(coordinates[[a, b], 0], coordinates[[a, b], 2],
                        color="0.75", linewidth=0.5, zorder=1)
    for index in order:
        s = symbols[index]
        ax.add_patch(Circle(
            (coordinates[index, 0], coordinates[index, 2]), 0.42 * RADIUS[s] + 0.18,
            facecolor=COLOR[s], zorder=2 + coordinates[index, 1] * 1e-3,
            edgecolor="black" if impurity[index] else "0.35",
            linewidth=1.1 if impurity[index] else 0.3))
    imp_xy = coordinates[impurity][0]
    ax.annotate(symbols[int(np.flatnonzero(impurity)[0])],
                (imp_xy[0], imp_xy[2]), textcoords="offset points",
                xytext=(7, 7), fontsize=7.5, fontweight="bold")
    ax.set_xlim(coordinates[:, 0].min() - 2.2, coordinates[:, 0].max() + 2.2)
    ax.set_ylim(coordinates[:, 2].min() - 2.2, coordinates[:, 2].max() + 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    host = meta["host"]
    digits = "".join(c if not c.isdigit() else f"$_{c}$" for c in host)
    ax.set_title(
        f"{meta['dopant']}@{digits} {meta['site']}\n"
        f"$|\\Delta|$ = {float(meta['pair_absolute_error_eV']):.2f} eV,"
        f" $E_f$ = {float(meta['target_eV']):.1f} eV",
        fontsize=7.5, pad=2)


def main() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})
    structures = json.loads((ROOT / "artifacts/prm_assets/case_structures.json").read_text())
    cases = load_cases()
    figure, axes = plt.subplots(2, 4, figsize=(7.05, 4.4))
    for raw_id, ax in zip(PANELS_TOP + PANELS_BOTTOM, axes.flat):
        if raw_id is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "no eligible\nmatched control\n(match 1)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7.5, color="0.45")
            continue
        meta = cases[raw_id]
        structure = structures[raw_id]
        assert meta["host"] == structure["kvp"]["host"]
        assert meta["dopant"] == structure["kvp"]["dopant"]
        draw_panel(ax, structure, meta)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02,
                           hspace=0.75, wspace=0.08)
    for y, title in zip((0.965, 0.475), ROW_TITLES):
        figure.text(0.5, y, title, ha="center", fontsize=8.5)
    output = ROOT / "paper_Q1/figures/fig_case_gallery.pdf"
    figure.savefig(output, dpi=400)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
