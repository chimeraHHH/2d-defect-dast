"""Aggregate LOHO results into a paper-ready table + figure.

For each run ``results/loho_<host>/`` we have:
  * ``metrics.json`` — config, history, test_mae, test_rmse
  * ``test_predictions.npz`` — per-sample preds vs targets

We combine these with the in-distribution reference number
``baseline_h128_aug_long_safe`` to compute, per host:
  * LOHO Test MAE / RMSE / R²
  * In-distribution reference Test MAE on the random-split test (constant
    across hosts)
  * Degradation factor =  LOHO MAE / in-distribution MAE
  * Number of test samples per host

Outputs:
  - results/loho_summary.json
  - paper/figures/fig_loho_bars.png

Then we update paper/main.md §5.8 with the table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HOSTS = ["MoS2", "Cr2I6", "C2H2", "TaSe2", "MoSSe"]
ID_REFERENCE_RUN = "baseline_h128_aug_long_safe"


def _r2(preds, targets):
    ss_res = float(((targets - preds) ** 2).sum())
    ss_tot = float(((targets - targets.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-9)


def main():
    summary = {"in_distribution_reference": ID_REFERENCE_RUN, "hosts": []}

    # Reference numbers
    ref_metrics = json.loads((RESULTS / ID_REFERENCE_RUN / "metrics.json").read_text())
    ref_mae = ref_metrics["test_mae"]
    ref_rmse = ref_metrics["test_rmse"]
    summary["in_distribution_test_mae"] = ref_mae
    summary["in_distribution_test_rmse"] = ref_rmse

    rows = []
    for h in HOSTS:
        run = f"loho_{h}"
        m_path = RESULTS / run / "metrics.json"
        npz_path = RESULTS / run / "test_predictions.npz"
        if not m_path.exists():
            print(f"!! missing {m_path}; skipping host {h}")
            continue
        m = json.loads(m_path.read_text())
        if not npz_path.exists():
            preds = targets = None
            r2 = None
        else:
            arr = np.load(npz_path)
            preds = arr["preds"].astype(np.float64)
            targets = arr["targets"].astype(np.float64)
            r2 = _r2(preds, targets)
        n_test = m["config"].get("n_test")  # may not be present
        rows.append({
            "host": h,
            "n_test": n_test if n_test is not None else (len(targets) if targets is not None else None),
            "loho_mae": float(m["test_mae"]),
            "loho_rmse": float(m["test_rmse"]),
            "loho_r2": r2,
            "id_reference_mae": ref_mae,
            "id_reference_rmse": ref_rmse,
            "degradation_factor_mae": float(m["test_mae"] / ref_mae),
            "degradation_factor_rmse": float(m["test_rmse"] / ref_rmse),
        })
    summary["hosts"] = rows

    print("=== LOHO summary ===")
    print(f"In-distribution reference (random split test, same model):  "
          f"MAE {ref_mae:.3f}  RMSE {ref_rmse:.3f}")
    print()
    fmt = "{:<10} {:>6} {:>10} {:>10} {:>8} {:>10}"
    print(fmt.format("host", "n_test", "MAE_loho", "RMSE_loho", "R²", "MAE/ref"))
    for r in rows:
        print(fmt.format(
            r["host"],
            str(r["n_test"]) if r["n_test"] is not None else "-",
            f"{r['loho_mae']:.3f}",
            f"{r['loho_rmse']:.3f}",
            f"{r['loho_r2']:.3f}" if r["loho_r2"] is not None else "-",
            f"{r['degradation_factor_mae']:.2f}×",
        ))
    if rows:
        avg_mae = float(np.mean([r["loho_mae"] for r in rows]))
        avg_deg = float(np.mean([r["degradation_factor_mae"] for r in rows]))
        summary["avg_loho_mae"] = avg_mae
        summary["avg_degradation_factor"] = avg_deg
        print(f"\nAverage: MAE {avg_mae:.3f}  degradation {avg_deg:.2f}×")

    # ------- figure -------
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        labels = [r["host"] for r in rows]
        loho_maes = [r["loho_mae"] for r in rows]
        x = np.arange(len(labels))
        ax.bar(x, loho_maes, color="tab:red", label="LOHO MAE")
        ax.axhline(ref_mae, color="k", lw=1, ls="--", label=f"In-dist ref = {ref_mae:.3f} eV")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Test MAE (eV)")
        ax.set_title("Leave-One-Host-Out: per-host MAE")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1]
        deg = [r["degradation_factor_mae"] for r in rows]
        ax.bar(x, deg, color="tab:orange")
        ax.axhline(1.0, color="k", lw=1, ls="--", label="no degradation")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("LOHO MAE / In-dist MAE")
        ax.set_title("OOD degradation factor")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        out = FIG_DIR / "fig_loho_bars.png"
        fig.savefig(out, dpi=180); plt.close(fig)
        print(f"\nfigure saved -> {out}")

    out_json = RESULTS / "loho_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary saved -> {out_json}")


if __name__ == "__main__":
    main()
