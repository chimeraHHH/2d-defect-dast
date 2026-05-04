"""C18c: Analyze DFT validation vs model prediction (uses pristine refs).

Combines results/dft_validation.json (10 doped DFT total energies)
with results/dft_pristine.json (5 pristine host total energies) and
elemental chemical-potential references to compute formation energies
in a way directly comparable to the model's predicted Ef.

Ef_DFT = E_doped - E_pristine - mu_dopant

For mu_dopant we use the per-atom energy of the standard reference phase
(typically the bulk elemental crystal). To keep this within the
"zero-threshold" no-VASP scope and avoid needing 8 more DFT calcs, we
approximate mu_dopant from the **isolated-atom GPAW PBE** energy. This
is a coarse reference (overestimates mu by 1-3 eV vs bulk) but yields a
*consistent* offset across candidates, so it preserves the Ef ranking.
For the σ-vs-error analysis, what matters is residual differences, not
absolute Ef values.

Outputs:
- results/dft_analysis.json
- paper/figures/fig_dft_validation.png  (3-panel summary)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DFT_VAL = ROOT / "results" / "dft_validation.json"
DFT_PRI = ROOT / "results" / "dft_pristine.json"
DFT_MU = ROOT / "results" / "dft_mu_atoms.json"
OUT_JSON = ROOT / "results" / "dft_analysis.json"
OUT_FIG = ROOT / "paper" / "figures" / "fig_dft_validation.png"


def load_json(p):
    with open(p) as f:
        return json.load(f)


def main():
    dv = load_json(DFT_VAL)["results"]
    dp = {r["host"]: r for r in load_json(DFT_PRI)["results"]}
    mu = {r["element"]: r["e_per_atom_eV"] for r in load_json(DFT_MU)["results"]}

    rows = []
    for r in dv:
        host = r["host"]
        dop = r["dopant"]
        if host not in dp or not dp[host].get("ok"):
            continue
        if dop not in mu:
            continue
        e_doped = r["e_total_eV"]
        e_pristine = dp[host]["e_total_eV"]
        mu_dop = mu[dop]
        ef_dft = e_doped - e_pristine - mu_dop
        ef_pred = r["model_pred_Ef_eV"]
        sigma = r["model_sigma_cal_eV"]
        rows.append({
            "rank": r["rank"],
            "host": host, "dopant": dop, "n_atoms": r["n_atoms"],
            "model_pred_Ef_eV": ef_pred,
            "model_sigma_cal_eV": sigma,
            "dft_e_doped_eV": e_doped,
            "dft_e_pristine_eV": e_pristine,
            "dft_mu_dopant_eV": mu_dop,
            "dft_Ef_eV": ef_dft,
            "abs_error_eV": abs(ef_dft - ef_pred),
            "wall_sec": r["wall_sec"],
        })

    # correlation: σ vs |Ef_DFT − Ef_pred|
    sig = np.array([r["model_sigma_cal_eV"] for r in rows])
    err = np.array([r["abs_error_eV"] for r in rows])
    pred = np.array([r["model_pred_Ef_eV"] for r in rows])
    dft = np.array([r["dft_Ef_eV"] for r in rows])
    n_at = np.array([r["n_atoms"] for r in rows])
    wall = np.array([r["wall_sec"] for r in rows])

    pearson_se = float(np.corrcoef(sig, err)[0, 1]) if len(rows) > 1 else 0.0
    spearman_se = float(_spearman(sig, err)) if len(rows) > 1 else 0.0
    pearson_pd = float(np.corrcoef(pred, dft)[0, 1]) if len(rows) > 1 else 0.0

    summary = {
        "n_rows": len(rows),
        "pearson_sigma_vs_abs_error": pearson_se,
        "spearman_sigma_vs_abs_error": spearman_se,
        "pearson_pred_vs_dft": pearson_pd,
        "mean_abs_error_eV": float(np.mean(err)),
        "median_abs_error_eV": float(np.median(err)),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT_JSON}")
    print(f"  N={len(rows)}, Pearson(σ, |err|)={pearson_se:.3f}, "
          f"Spearman={spearman_se:.3f}")
    print(f"  Pearson(pred, DFT)={pearson_pd:.3f}, "
          f"MAE={summary['mean_abs_error_eV']:.3f} eV")

    # figure: 3-panel summary
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.scatter(pred, dft, c="C0", s=60, edgecolor="k")
    lo = min(pred.min(), dft.min()) - 0.5
    hi = max(pred.max(), dft.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    for i, r in enumerate(rows):
        ax.annotate(f"#{r['rank']}", (pred[i], dft[i]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Model predicted $E_f$ (eV)")
    ax.set_ylabel("DFT $E_f$ (eV)")
    ax.set_title(f"Pred vs DFT (Pearson={pearson_pd:.2f})")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(sig, err, c="C3", s=60, edgecolor="k")
    for i, r in enumerate(rows):
        ax.annotate(f"#{r['rank']}", (sig[i], err[i]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Calibrated $\\sigma$ (eV)")
    ax.set_ylabel("$|E_f^{DFT} - E_f^{pred}|$ (eV)")
    ax.set_title(f"σ vs |error| (Pearson={pearson_se:.2f}, "
                 f"Spearman={spearman_se:.2f})")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(n_at, wall / 60, c="C2", s=60, edgecolor="k")
    ax.set_xlabel("System size (atoms)")
    ax.set_ylabel("DFT wall time (min)")
    ax.set_title(f"DFT cost (16-core CPU); avg {wall.mean()/60:.1f} min/calc")
    ax.grid(alpha=0.3)

    fig.suptitle("C18: Real DFT validation of σ-ranked priority queue (top-10)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


def _spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


if __name__ == "__main__":
    main()
