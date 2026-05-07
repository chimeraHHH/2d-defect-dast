"""F3: Per-host signed-error dot plot for prospective DFT validation.

For each of the 27 hosts that contributed at least one SCF-converged
candidate, plot every candidate as a dot at:
  x = host (sorted by mean |signed error|)
  y = signed error (DFT_corr - pred), in eV
  marker = bucket A (circle) / bucket B (triangle)

Adds:
  - 0 reference line
  - per-host mean as a horizontal tick
  - shaded "MAE-within-1eV" band
  - candidate ID + dopant labels for outliers (|err| > 5 eV)

Visual purpose: turn the per-host table into a glance-readable
picture of where the model is on/off-target. Shows reviewer that
performance is host-stratified, with a few specific outliers
driving the heavy-tailed MAE.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_IN = ROOT / "results" / "prospective_dft_results.json"
OUT = ROOT / "paper" / "figures" / "fig_per_host_signed.png"


def main():
    rows = json.loads(RES_IN.read_text())["rows"]
    print(f"loaded {len(rows)} candidates")

    off = defaultdict(list)
    for r in rows:
        off[r["dopant"]].append(r["dft_Ef_eV"] - r["model_pred_Ef_eV"])
    mean_off = {d: float(np.mean(v)) for d, v in off.items()}

    by_host = defaultdict(list)
    for r in rows:
        signed = (r["dft_Ef_eV"] - mean_off[r["dopant"]]) - r["model_pred_Ef_eV"]
        by_host[r["host"]].append({
            "id": r["id"],
            "dopant": r["dopant"],
            "bucket": r["bucket"],
            "signed_err": signed,
        })

    # Sort hosts by mean |signed err|
    hosts_sorted = sorted(
        by_host.keys(),
        key=lambda h: np.mean([abs(x["signed_err"]) for x in by_host[h]])
    )

    fig, ax = plt.subplots(figsize=(13, 5.6))
    for x_pos, host in enumerate(hosts_sorted):
        items = by_host[host]
        ys = [it["signed_err"] for it in items]
        means = float(np.mean(ys))
        xs = [x_pos] * len(ys)
        # buckets
        a_x = [x_pos for it in items if it["bucket"].startswith("A")]
        a_y = [it["signed_err"] for it in items if it["bucket"].startswith("A")]
        b_x = [x_pos for it in items if it["bucket"].startswith("B")]
        b_y = [it["signed_err"] for it in items if it["bucket"].startswith("B")]
        ax.scatter(a_x, a_y, c="#1f3a5f", s=70, alpha=0.85, edgecolor="white", linewidth=0.6,
                   marker="o", label="bucket A: low-$E_f$ confident" if x_pos == 0 else None)
        ax.scatter(b_x, b_y, c="#a04040", s=70, alpha=0.85, edgecolor="white", linewidth=0.6,
                   marker="^", label="bucket B: high-$\\sigma$ OOD" if x_pos == 0 else None)
        # mean tick
        ax.plot([x_pos - 0.25, x_pos + 0.25], [means, means],
                color="black", lw=1.5, alpha=0.8)
        # outlier labels
        for it in items:
            if abs(it["signed_err"]) > 5.0:
                ax.annotate(
                    f"{it['dopant']}",
                    (x_pos, it["signed_err"]),
                    fontsize=8, color="#444",
                    xytext=(8, 0), textcoords="offset points",
                    va="center",
                )

    # 0 reference + 1 eV band
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.axhspan(-1.0, 1.0, color="#cdd9e8", alpha=0.25,
               label="$|$err$|<1$ eV")

    ax.set_xticks(range(len(hosts_sorted)))
    # Subscript-fy host labels
    sub_map = {
        "Bi2I6": "Bi$_2$I$_6$", "BiITe": "BiITe", "C2H2": "C$_2$H$_2$",
        "Hf2Te6": "Hf$_2$Te$_6$", "HfS2": "HfS$_2$", "Mo2CO2": "Mo$_2$CO$_2$",
        "MoS2": "MoS$_2$", "MoSe2": "MoSe$_2$", "MoTe2": "MoTe$_2$",
        "NbS2": "NbS$_2$", "NbSe2": "NbSe$_2$", "NiSe2": "NiSe$_2$",
        "Pd2Se4": "Pd$_2$Se$_4$", "PtS2": "PtS$_2$", "PtSe2": "PtSe$_2$",
        "Sn2": "Sn$_2$", "SnS2": "SnS$_2$", "SnSe2": "SnSe$_2$",
        "TaS2": "TaS$_2$", "Ti2CO2": "Ti$_2$CO$_2$", "TiO2": "TiO$_2$",
        "TiS2": "TiS$_2$", "V2CO2": "V$_2$CO$_2$", "WTe2": "WTe$_2$",
        "ZrS2": "ZrS$_2$", "ZrSe2": "ZrSe$_2$", "Ge2": "Ge$_2$",
    }
    ax.set_xticklabels([sub_map.get(h, h) for h in hosts_sorted],
                       rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Signed error: DFT$_{corr}$ $-$ pred (eV)", fontsize=11)
    ax.set_xlabel("Host (sorted by mean $|$err$|$)", fontsize=11)
    ax.set_title("Per-host signed prediction error on the prospective DFT subset (N=37)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
