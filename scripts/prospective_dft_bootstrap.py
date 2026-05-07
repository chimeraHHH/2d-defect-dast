"""Bootstrap 95% confidence intervals for prospective DFT validation.

Reads results/prospective_dft_results.json, fixes the per-dopant
chemical-potential offset from the full data (treating the
correction as conditioning, not a free parameter), then resamples
indices with replacement to obtain 95% CIs on every headline number
in the prospective DFT subsection.

Sampling scheme: stratified within bucket (separate A and B
resampling so each bootstrap preserves n_A=20, n_B=17). For overall
("all") statistics, draw a 20+17=37 stratified sample and combine.

Outputs:
  results/prospective_dft_bootstrap.json   — full CI dump
  updates results/prospective_dft_summary.json with `bootstrap_ci_95`

Reports:
  - Overall MAE, median |err|, RMSE
  - Pearson(pred, DFT_corr) overall and per bucket
  - Spearman(pred, DFT_corr) overall and per bucket
  - Pearson(sigma, |err|) overall and per bucket  ← key OOD claim
  - Spearman(sigma, |err|) overall and per bucket
  - Bucket A discovery rates (DFT < 1 eV, DFT < 0 eV)
  - "Excludes zero?" flag for each correlation CI

Run: python3 scripts/prospective_dft_bootstrap.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_IN = ROOT / "results" / "prospective_dft_results.json"
SUM_IO = ROOT / "results" / "prospective_dft_summary.json"
BOOT_OUT = ROOT / "results" / "prospective_dft_bootstrap.json"

N_BOOT = 5000
SEED = 42


def pearson(x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def percentile_ci(arr, level=0.95):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        return None
    lo = float(np.percentile(arr, (1 - level) / 2 * 100))
    hi = float(np.percentile(arr, (1 + level) / 2 * 100))
    return [lo, hi]


def stats_from_indices(pred, dft_c, sig, idx):
    p = pred[idx]
    dc = dft_c[idx]
    s = sig[idx]
    err = np.abs(dc - p)
    n = len(idx)
    return {
        "n": int(n),
        "mae_corr_eV": float(err.mean()) if n else float("nan"),
        "median_err_corr_eV": float(np.median(err)) if n else float("nan"),
        "rmse_corr_eV": float(np.sqrt((err ** 2).mean())) if n else float("nan"),
        "pearson_pred_dft_corr": pearson(p, dc),
        "spearman_pred_dft_corr": spearman(p, dc),
        "pearson_sig_err_corr": pearson(s, err),
        "spearman_sig_err_corr": spearman(s, err),
        "frac_dft_below_1eV": float(np.mean(dc < 1.0)) if n else float("nan"),
        "frac_dft_below_0eV": float(np.mean(dc < 0.0)) if n else float("nan"),
    }


def main():
    rng = np.random.default_rng(SEED)
    rows = json.loads(RES_IN.read_text())["rows"]
    n = len(rows)
    print(f"loaded N = {n} candidates")

    pred = np.array([r["model_pred_Ef_eV"] for r in rows])
    dft = np.array([r["dft_Ef_eV"] for r in rows])
    sig = np.array([r["model_sigma_cal_eV"] for r in rows])
    dopants = np.array([r["dopant"] for r in rows])
    bucket = np.array([r["bucket"] for r in rows])
    mA_ = bucket == "A_low_Ef_high_conf"
    mB_ = bucket == "B_high_sigma_OOD"
    a_idx = np.where(mA_)[0]
    b_idx = np.where(mB_)[0]
    print(f"  bucket A: n={len(a_idx)}   bucket B: n={len(b_idx)}")

    # Per-dopant offset (fixed from full data)
    off = defaultdict(list)
    for r in rows:
        off[r["dopant"]].append(r["dft_Ef_eV"] - r["model_pred_Ef_eV"])
    mean_off = {dop: float(np.mean(v)) for dop, v in off.items()}
    dft_corr = np.array([r["dft_Ef_eV"] - mean_off[r["dopant"]] for r in rows])

    # Point estimates
    point = {
        "all": stats_from_indices(pred, dft_corr, sig, np.arange(n)),
        "bucket_A": stats_from_indices(pred, dft_corr, sig, a_idx),
        "bucket_B": stats_from_indices(pred, dft_corr, sig, b_idx),
    }

    # Bootstrap
    boot = {scope: defaultdict(list) for scope in ["all", "bucket_A", "bucket_B"]}
    for b in range(N_BOOT):
        # Stratified resample
        rs_A = a_idx[rng.integers(0, len(a_idx), size=len(a_idx))]
        rs_B = b_idx[rng.integers(0, len(b_idx), size=len(b_idx))]
        rs_all = np.concatenate([rs_A, rs_B])

        for scope, idx in (("all", rs_all), ("bucket_A", rs_A), ("bucket_B", rs_B)):
            s = stats_from_indices(pred, dft_corr, sig, idx)
            for k, v in s.items():
                if k != "n":
                    boot[scope][k].append(v)

        if (b + 1) % 1000 == 0:
            print(f"  bootstrap {b + 1}/{N_BOOT}")

    # Compute 95% CIs
    out = {
        "n_boot": N_BOOT,
        "rng_seed": SEED,
        "n_candidates": int(n),
        "n_A": int(len(a_idx)),
        "n_B": int(len(b_idx)),
        "per_dopant_offsets": mean_off,
        "point_estimate": point,
        "ci_95": {scope: {k: percentile_ci(v) for k, v in d.items()} for scope, d in boot.items()},
    }

    # "excludes zero?" flag for correlations and offsets
    out["sig_at_95"] = {}
    for scope in ["all", "bucket_A", "bucket_B"]:
        out["sig_at_95"][scope] = {}
        for k in ("pearson_pred_dft_corr", "spearman_pred_dft_corr",
                  "pearson_sig_err_corr", "spearman_sig_err_corr"):
            ci = out["ci_95"][scope][k]
            if ci is None:
                out["sig_at_95"][scope][k] = None
            else:
                out["sig_at_95"][scope][k] = bool(ci[0] > 0 or ci[1] < 0)

    BOOT_OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {BOOT_OUT}")

    # Pretty print
    print("\n" + "=" * 60)
    print(f"Bootstrap 95% CI (N_boot={N_BOOT}, stratified within bucket)")
    print("=" * 60)
    keys_ord = [
        ("mae_corr_eV", "MAE (eV)"),
        ("median_err_corr_eV", "median |err| (eV)"),
        ("rmse_corr_eV", "RMSE (eV)"),
        ("pearson_pred_dft_corr", "Pearson(pred, DFT_corr)"),
        ("spearman_pred_dft_corr", "Spearman(pred, DFT_corr)"),
        ("pearson_sig_err_corr", "Pearson(σ, |err|)"),
        ("spearman_sig_err_corr", "Spearman(σ, |err|)"),
        ("frac_dft_below_1eV", "frac DFT < 1 eV"),
        ("frac_dft_below_0eV", "frac DFT < 0 eV"),
    ]
    for scope in ["all", "bucket_A", "bucket_B"]:
        n_scope = point[scope]["n"]
        print(f"\n[{scope}, n={n_scope}]")
        for key, label in keys_ord:
            pe = point[scope][key]
            ci = out["ci_95"][scope][key]
            sig_flag = ""
            if key.startswith(("pearson_", "spearman_")):
                excl = out["sig_at_95"][scope][key]
                if excl is True:
                    sig_flag = "  ✓ excludes 0"
                elif excl is False:
                    sig_flag = "  ✗ includes 0"
            if ci is None:
                print(f"  {label:<28s} {pe:+.3f}  (no CI)")
            else:
                print(f"  {label:<28s} {pe:+.3f}  CI = [{ci[0]:+.3f}, {ci[1]:+.3f}]{sig_flag}")

    # Update summary
    if SUM_IO.exists():
        sumr = json.loads(SUM_IO.read_text())
        sumr["bootstrap_ci_95"] = out["ci_95"]
        sumr["bootstrap_sig_at_95"] = out["sig_at_95"]
        sumr["bootstrap_n"] = N_BOOT
        SUM_IO.write_text(json.dumps(sumr, indent=2))
        print(f"\nupdated {SUM_IO} with `bootstrap_ci_95` block")

    print("\nDone.  Use these CIs in the paper, e.g.:")
    print("  ``70%`` -> ``70% (95% CI [50%, 85%])``")
    print("  ``Pearson = -0.288`` -> ``Pearson = -0.288 (95% CI [-0.6, +0.1])``")


if __name__ == "__main__":
    main()
