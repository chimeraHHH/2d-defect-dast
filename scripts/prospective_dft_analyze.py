"""Prospective DFT validation analysis: parity plot + sigma calibration.

Reads results/prospective_dft_results.json (produced by
prospective_dft_collect.py) and emits:
  - paper/figures/fig_prospective_dft.png (3-panel summary)
  - results/prospective_dft_summary.json   (raw + per-dopant-corrected stats)

The model was trained on IMP2D where mu_dopant uses bulk-elemental
reference, but our DFT uses isolated-atom mu. The difference
mu_atomic - mu_bulk is a per-dopant constant (the cohesive energy of
the elemental phase). We compute and remove it via per-dopant offset
so within-dopant Pearson is comparable to the model's training target.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "prospective_dft_results.json"
OUT_FIG = ROOT / "paper" / "figures" / "fig_prospective_dft.png"
OUT_SUM = ROOT / "results" / "prospective_dft_summary.json"


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    blob = json.loads(RES.read_text())
    rows = blob["rows"]
    if not rows:
        print("no rows")
        return
    print(f"N = {len(rows)} candidates with full DFT references")

    # Per-dopant offset correction (atomic mu -> bulk-equivalent reference).
    dop_off = defaultdict(list)
    for r in rows:
        dop_off[r["dopant"]].append(r["dft_Ef_eV"] - r["model_pred_Ef_eV"])
    mean_off = {d: float(np.mean(v)) for d, v in dop_off.items()}

    ef_pred = np.array([r["model_pred_Ef_eV"] for r in rows])
    ef_dft_raw = np.array([r["dft_Ef_eV"] for r in rows])
    ef_dft_corr = np.array([r["dft_Ef_eV"] - mean_off[r["dopant"]] for r in rows])
    sig = np.array([r["model_sigma_cal_eV"] for r in rows])
    err_raw = np.abs(ef_dft_raw - ef_pred)
    err_corr = np.abs(ef_dft_corr - ef_pred)
    bucket = np.array([r["bucket"] for r in rows])
    mA = bucket == "A_low_Ef_high_conf"
    mB = bucket == "B_high_sigma_OOD"

    def stats(pred, dft, sig, err, mask):
        if mask.sum() < 2:
            return None
        return {
            "n": int(mask.sum()),
            "mae_eV": float(err[mask].mean()),
            "median_abs_err_eV": float(np.median(err[mask])),
            "rmse_eV": float(np.sqrt((err[mask] ** 2).mean())),
            "pearson_pred_vs_dft": float(np.corrcoef(pred[mask], dft[mask])[0, 1]),
            "spearman_pred_vs_dft": spearman(pred[mask], dft[mask]),
            "pearson_sigma_vs_abs_err": float(np.corrcoef(sig[mask], err[mask])[0, 1]),
            "spearman_sigma_vs_abs_err": spearman(sig[mask], err[mask]),
        }

    summary = {
        "n_total": len(rows),
        "raw": {
            "all": stats(ef_pred, ef_dft_raw, sig, err_raw, np.ones(len(rows), bool)),
            "bucketA": stats(ef_pred, ef_dft_raw, sig, err_raw, mA),
            "bucketB": stats(ef_pred, ef_dft_raw, sig, err_raw, mB),
        },
        "per_dopant_corrected": {
            "all": stats(ef_pred, ef_dft_corr, sig, err_corr, np.ones(len(rows), bool)),
            "bucketA": stats(ef_pred, ef_dft_corr, sig, err_corr, mA),
            "bucketB": stats(ef_pred, ef_dft_corr, sig, err_corr, mB),
            "discovery_A": {
                "frac_dft_below_1eV": float(np.mean(ef_dft_corr[mA] < 1.0)),
                "frac_dft_below_0eV": float(np.mean(ef_dft_corr[mA] < 0.0)),
            },
            "median_abs_err_A_eV": float(np.median(err_corr[mA])),
            "median_abs_err_B_eV": float(np.median(err_corr[mB])),
        },
        "per_dopant_offsets": mean_off,
    }
    OUT_SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["per_dopant_corrected"], indent=2))

    # Figure: 3 panels with corrected values
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel 1: parity (corrected) by bucket
    ax = axes[0]
    ax.scatter(ef_pred[mA], ef_dft_corr[mA], c="C0", s=64, edgecolor="k",
               label=f"A: low-Ef confident (n={mA.sum()})")
    ax.scatter(ef_pred[mB], ef_dft_corr[mB], c="C3", s=64, edgecolor="k",
               marker="^", label=f"B: high-σ OOD (n={mB.sum()})")
    lo = min(ef_pred.min(), ef_dft_corr.min()) - 1
    hi = max(ef_pred.max(), ef_dft_corr.max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y=x")
    ax.axhline(1.0, c="gray", lw=0.5, alpha=0.5)
    ax.axhline(0.0, c="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("Model predicted $E_f$ (eV)")
    ax.set_ylabel("DFT $E_f^{\\rm corr}$ (eV)")
    pe = summary["per_dopant_corrected"]["all"]["pearson_pred_vs_dft"]
    pe_A = summary["per_dopant_corrected"]["bucketA"]["pearson_pred_vs_dft"]
    ax.set_title(f"Parity, per-dopant corrected\nPearson all={pe:+.2f}  A={pe_A:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: sigma vs |err|
    ax = axes[1]
    ax.scatter(sig[mA], err_corr[mA], c="C0", s=64, edgecolor="k",
               label=f"A (n={mA.sum()})")
    ax.scatter(sig[mB], err_corr[mB], c="C3", s=64, edgecolor="k",
               marker="^", label=f"B (n={mB.sum()})")
    ax.set_xlabel("Calibrated $\\sigma$ (eV)")
    ax.set_ylabel("$|E_f^{\\rm DFT,corr} - E_f^{\\rm pred}|$ (eV)")
    pe_sig = summary["per_dopant_corrected"]["all"]["pearson_sigma_vs_abs_err"]
    sp_sig = summary["per_dopant_corrected"]["all"]["spearman_sigma_vs_abs_err"]
    pe_sig_B = summary["per_dopant_corrected"]["bucketB"]["pearson_sigma_vs_abs_err"]
    ax.set_title(
        f"σ-calibration\nall: Pearson={pe_sig:+.2f}  Spearman={sp_sig:+.2f}\nB only: Pearson={pe_sig_B:+.2f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: discovery rate bar
    ax = axes[2]
    f1 = summary["per_dopant_corrected"]["discovery_A"]["frac_dft_below_1eV"]
    f0 = summary["per_dopant_corrected"]["discovery_A"]["frac_dft_below_0eV"]
    bars = ax.bar(["A: $E_f^{\\rm DFT}<+1$ eV", "A: $E_f^{\\rm DFT}<0$ eV\n(exothermic)"],
                   [100 * f1, 100 * f0], color=["C0", "C2"], edgecolor="k")
    for b, v in zip(bars, [100 * f1, 100 * f0]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of bucket A candidates")
    ax.set_title(f"Discovery rate (Bucket A, n={int(mA.sum())})")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Prospective DFT validation, $N$={len(rows)}: "
        f"60 candidates → 37 SCF-converged (22 La/Cs PSL fails + 1 SCF divergence excluded)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    print(f"\nsaved {OUT_FIG}")


if __name__ == "__main__":
    main()
