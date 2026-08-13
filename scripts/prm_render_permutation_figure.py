"""Render the permutation-repair figure for PRM.

Pure plotting over hash-verified G1A artifacts: the per-structure legacy
permutation spread comes from ``artifacts/prm_g1/g1a/sample_diagnostics.csv``
(sha256 pinned in g1_acceptance.json) and the per-permutation pooled MAEs
from ``artifacts/prm_g1/g1a/summary.json``.  The repaired bound is the
largest intervention round-trip delta recorded in g1_acceptance.json.  No
training or inference happens here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#0072B2", "#D55E00"


def style() -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def main() -> None:
    style()
    rows = list(csv.DictReader(open(ROOT / "artifacts/prm_g1/g1a/sample_diagnostics.csv")))
    summary = json.loads((ROOT / "artifacts/prm_g1/g1a/summary.json").read_text())["legacy_under_permutations"]
    acceptance = json.loads((ROOT / "artifacts/prm_g1/g1_acceptance.json").read_text())
    repaired_bound = max(
        acceptance["g1a_live_audit"]["intervention_roundtrip_max_abs_delta_eV"].values())

    spread = {"interstitial": [], "adsorbate": []}
    for row in rows:
        value = row["permutation_prediction_range_eV"]
        if value not in ("", "nan"):
            spread[row["defecttype"]].append(float(value))
    pooled = np.asarray(spread["interstitial"] + spread["adsorbate"])
    assert np.isclose(pooled.max(), summary["prediction_range_eV"]["max"])

    figure, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(3.4, 3.0), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.12})

    # (a) ECDF of the legacy per-structure permutation spread, by class.
    for key, color, linestyle in (
        ("adsorbate", BLUE, "-"), ("interstitial", ORANGE, "--"),
    ):
        values = np.sort(np.asarray(spread[key]))
        ax_top.step(values, np.arange(1, len(values) + 1) / len(values),
                    linestyle, color=color, linewidth=1.1,
                    label=f"legacy, {key}", where="post")
    ax_top.axvline(repaired_bound, color="0.2", linewidth=0.9)
    ax_top.text(repaired_bound * 1.6, 0.06,
                "repaired:\nall structures\n$\\leq$ round-trip\nnoise",
                fontsize=6.5, color="0.2", va="bottom")
    ax_top.set_ylabel("cumulative fraction")
    ax_top.set_ylim(0, 1.02)
    ax_top.legend(frameon=False, loc="upper left", handlelength=1.8)

    # (b) Scales: aggregate metric versus single-structure sensitivity.
    entries = [
        ("repaired: max round-trip delta", repaired_bound, "0.2", "o"),
        ("legacy: pooled-MAE change (max over 16 permutations)",
         summary["max_absolute_mae_change_eV"], BLUE, "s"),
        ("legacy: per-structure spread, p99",
         summary["prediction_range_eV"]["p99"], ORANGE, "^"),
        ("legacy: per-structure spread, max",
         summary["prediction_range_eV"]["max"], ORANGE, "D"),
    ]
    for k, (label, value, color, marker) in enumerate(entries):
        y = len(entries) - 1 - k
        ax_bot.plot([1.1e-6, value], [y, y], color="0.85", linewidth=0.8, zorder=1)
        ax_bot.plot(value, y, marker, color=color, markersize=4.5, zorder=2)
        ax_bot.text(1.3e-6, y + 0.22, label, fontsize=6.2, va="bottom",
                    color="0.15")
    ax_bot.set_yticks([])
    ax_bot.set_ylim(-0.5, len(entries) - 0.1)
    ax_bot.set_xscale("log")
    ax_bot.set_xlim(1e-6, 40)
    ax_bot.set_xlabel("prediction change under atom relabeling (eV)")
    ax_bot.spines["left"].set_visible(False)

    for label, ax in zip(("(a)", "(b)"), (ax_top, ax_bot)):
        ax.text(-0.16, 1.02, label, transform=ax.transAxes,
                fontweight="bold", fontsize=9, va="bottom")
    figure.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.14)
    output = ROOT / "paper_Q1/figures/fig_permutation_repair.pdf"
    figure.savefig(output, dpi=400)
    print(f"wrote {output}")
    print("repaired bound", repaired_bound,
          "legacy p99", summary["prediction_range_eV"]["p99"],
          "max", summary["prediction_range_eV"]["max"],
          "pooled-MAE max change", summary["max_absolute_mae_change_eV"])


if __name__ == "__main__":
    main()
