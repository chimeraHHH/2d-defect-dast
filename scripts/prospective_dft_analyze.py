"""Make parity plot + sigma calibration plot from
results/prospective_dft_results.json.

Outputs:
- paper/figures/fig_prospective_dft.png
- summary stats printed to stdout (and stored in results/prospective_dft_summary.json)
"""
from __future__ import annotations

import json
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
        print("no rows; nothing to plot")
        return

    ef_dft = np.array([r["dft_Ef_eV"] for r in rows])
    ef_pred = np.array([r["model_pred_Ef_eV"] for r in rows])
    sig = np.array([r["model_sigma_cal_eV"] for r in rows])
    err = np.array([r["abs_error_eV"] for r in rows])
    bucket = np.array([r["bucket"] for r in rows])
    mA = bucket == "A_low_Ef_high_conf"
    mB = bucket == "B_high_sigma_OOD"

    summary = {
        "n_total": int(len(rows)),
        "n_A": int(mA.sum()),
        "n_B": int(mB.sum()),
        "mae_eV": float(err.mean()),
        "rmse_eV": float(np.sqrt((err ** 2).mean())),
        "median_abs_err_eV": float(np.median(err)),
        "pearson_pred_vs_dft": (
            float(np.corrcoef(ef_pred, ef_dft)[0, 1]) if len(rows) > 1 else None
        ),
        "spearman_pred_vs_dft": spearman(ef_pred, ef_dft) if len(rows) > 1 else None,
        "pearson_sigma_vs_abs_err": (
            float(np.corrcoef(sig, err)[0, 1]) if len(rows) > 1 else None
        ),
        "spearman_sigma_vs_abs_err": spearman(sig, err) if len(rows) > 1 else None,
        "bucketA": {
            "n": int(mA.sum()),
            "mae_eV": float(err[mA].mean()) if mA.sum() else None,
            "frac_predicted_low_Ef_below_0": float(np.mean(ef_pred[mA] < 0)) if mA.sum() else None,
            "frac_dft_confirmed_below_0": float(np.mean(ef_dft[mA] < 0)) if mA.sum() else None,
            "pearson_pred_vs_dft": (
                float(np.corrcoef(ef_pred[mA], ef_dft[mA])[0, 1]) if mA.sum() > 1 else None
            ),
        },
        "bucketB": {
            "n": int(mB.sum()),
            "mae_eV": float(err[mB].mean()) if mB.sum() else None,
            "pearson_sigma_vs_abs_err": (
                float(np.corrcoef(sig[mB], err[mB])[0, 1]) if mB.sum() > 1 else None
            ),
            "spearman_sigma_vs_abs_err": (
                spearman(sig[mB], err[mB]) if mB.sum() > 1 else None
            ),
        },
    }
    OUT_SUM.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- Panel 1: parity by bucket ---
    ax = axes[0]
    ax.scatter(ef_pred[mA], ef_dft[mA], c="C0", s=58, edgecolor="k", label="A: low-Ef confident")
    ax.scatter(ef_pred[mB], ef_dft[mB], c="C3", s=58, edgecolor="k", marker="^", label="B: high-σ OOD")
    lo = min(ef_pred.min(), ef_dft.min()) - 0.5
    hi = max(ef_pred.max(), ef_dft.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xlabel("Model predicted $E_f$ (eV)")
    ax.set_ylabel("DFT $E_f$ (eV)")
    pearson_all = summary["pearson_pred_vs_dft"]
    pearson_A = summary["bucketA"]["pearson_pred_vs_dft"]
    title = f"Parity (Pearson all={pearson_all:+.2f}"
    if pearson_A is not None:
        title += f", A={pearson_A:+.2f}"
    title += ")"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # --- Panel 2: σ vs |err| ---
    ax = axes[1]
    ax.scatter(sig[mA], err[mA], c="C0", s=58, edgecolor="k", label="A")
    ax.scatter(sig[mB], err[mB], c="C3", s=58, edgecolor="k", marker="^", label="B")
    ax.set_xlabel("Calibrated $\\sigma$ (eV)")
    ax.set_ylabel("$|E_f^{\\rm DFT} - E_f^{\\rm pred}|$ (eV)")
    sp = summary["spearman_sigma_vs_abs_err"]
    pe = summary["pearson_sigma_vs_abs_err"]
    ax.set_title(
        f"σ-calibration (Pearson={pe:+.2f}, Spearman={sp:+.2f})"
        if pe is not None else "σ-calibration"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # --- Panel 3: bucket-wise MAE comparison ---
    ax = axes[2]
    if mA.sum():
        ax.bar(["A: low-Ef\nconfident"], [err[mA].mean()], yerr=[err[mA].std() / np.sqrt(mA.sum())],
               color="C0", edgecolor="k", capsize=8)
    if mB.sum():
        ax.bar(["B: high-σ\nOOD"], [err[mB].mean()], yerr=[err[mB].std() / np.sqrt(mB.sum())],
               color="C3", edgecolor="k", capsize=8)
    ax.set_ylabel("MAE (eV)")
    ax.set_title("Per-bucket MAE")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Prospective DFT validation: 60 candidates (30A + 30B)  •  "
        f"overall MAE={summary['mae_eV']:.2f} eV",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    print(f"\nsaved {OUT_FIG}")


if __name__ == "__main__":
    main()
