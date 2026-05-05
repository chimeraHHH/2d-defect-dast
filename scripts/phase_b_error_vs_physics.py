"""Phase B1 — explain residual error of v1 baseline GNN with physics features.

Loads:
  results/baseline_h128_aug_long_safe/test_predictions.npz   (preds, targets)
  results/phase_a_descriptors.npz                             (per-atom)
  + per-sample summary built on the fly via phase_a_lightgbm_physics

For each test sample we compute |error| = |GNN_pred - target|, then ask:
  * how does |error| correlate with each physics feature?
  * does a LightGBM regressor on physics features predict |error| well
    (i.e. is the GNN error a structured function of measurable physics)?
  * which sample categories have the largest residuals (defect type,
    dopant block, host family, host_natoms quantile, dopant electronegativity
    bin)?

This converts the abstract claim "the GNN is data-bottlenecked" into a
specific, falsifiable failure-mode map.

Output: results/phase_b_error_vs_physics.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402
from scipy.stats import spearmanr, pearsonr  # noqa: E402

from scripts.phase_a_lightgbm_physics import per_sample_physics_features  # noqa: E402
from scripts.phase_a_descriptors import build_pair_reference_table  # noqa: E402


def main():
    src = ROOT / "data" / "processed" / "cleaned_dataset_with_pristine.pkl"
    with open(src, "rb") as f:
        blob = pickle.load(f)
    data = blob["data"]

    # Same split_indices(seed=42) test fold
    import random
    rng = random.Random(42)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_train = int(0.8 * len(data))
    n_val = int(0.1 * len(data))
    test_idx = indices[n_train + n_val:]

    preds_npz = ROOT / "results" / "baseline_h128_aug_long_safe" / "test_predictions.npz"
    arr = np.load(preds_npz)
    preds = arr["preds"]
    targets = arr["targets"]
    if preds.shape[0] != len(test_idx):
        raise RuntimeError(
            f"baseline preds shape {preds.shape[0]} != test_idx {len(test_idx)}"
        )
    abs_err = np.abs(preds - targets)
    print(f"baseline test MAE = {abs_err.mean():.4f} eV  N={len(test_idx)}")

    # Build per-sample physics features for the test fold
    print("building bond-pair reference from train fold...")
    train_idx = indices[:n_train]
    ref_table = build_pair_reference_table(
        [data[i] for i in train_idx], max_samples=2000
    )

    t0 = time.time()
    feats = []
    for k, idx in enumerate(test_idx):
        s = data[idx]
        feats.append(per_sample_physics_features(s, ref_table=ref_table))
        if (k + 1) % 200 == 0:
            print(f"  built {k+1}/{len(test_idx)} feats  ({time.time()-t0:.0f}s)")
    feature_names = list(feats[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in feats])
    print(f"X shape {X.shape}")

    # Spearman + Pearson against |error|
    correlations = []
    for k, name in enumerate(feature_names):
        col = X[:, k]
        # mask out non-finite
        mask = np.isfinite(col) & np.isfinite(abs_err)
        if mask.sum() < 50:
            continue
        sp = spearmanr(col[mask], abs_err[mask]).statistic
        pe = pearsonr(col[mask], abs_err[mask]).statistic
        correlations.append({"feature": name, "spearman": float(sp), "pearson": float(pe)})
    correlations.sort(key=lambda x: -abs(x["spearman"]))

    print("\nTop |Spearman| with |error|:")
    for r in correlations[:10]:
        print(f"  {r['feature']:<35s}  ρ={r['spearman']:+.3f}  Pearson={r['pearson']:+.3f}")

    # LightGBM on physics features → predict |error|
    n = len(test_idx)
    n_tr = int(0.7 * n)
    rng2 = np.random.default_rng(0)
    perm = rng2.permutation(n)
    tr_i, te_i = perm[:n_tr], perm[n_tr:]
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        min_data_in_leaf=20, lambda_l2=0.5, verbose=-1, random_state=42,
    )
    model.fit(X[tr_i], abs_err[tr_i])
    pred_err = model.predict(X[te_i])
    err_r2 = float(1.0 - np.sum((pred_err - abs_err[te_i]) ** 2)
                   / np.sum((abs_err[te_i] - abs_err[te_i].mean()) ** 2))
    print(f"\nLightGBM-on-physics → predict |error|:  test R² = {err_r2:.3f}")
    print("  (high R² means the GNN's residual error is a structured "
          "function of measurable physics)")

    # Failure-mode breakdown by defect_type, dopant_block, host_natoms, host
    def _bin_stats(values, name):
        out = {}
        unique = np.unique(values)
        for v in unique:
            mask = values == v
            if mask.sum() < 10:
                continue
            out[str(v)] = {
                "n": int(mask.sum()),
                "mean_abs_err": float(abs_err[mask].mean()),
                "p90_abs_err": float(np.quantile(abs_err[mask], 0.9)),
            }
        return {"by": name, "groups": out}

    defect_types = np.array([f["defect_type_int"] for f in feats])
    dopant_blocks = np.array([f["dopant_block"] for f in feats])
    n_atoms_arr = np.array([f["n_atoms"] for f in feats])
    delta_chi_arr = np.array([f["delta_chi"] for f in feats])
    delta_rcov_arr = np.array([f["delta_rcov"] for f in feats])

    # Bin n_atoms into 4 quantile bins
    n_atoms_q = np.searchsorted(
        np.quantile(n_atoms_arr, [0.25, 0.5, 0.75]), n_atoms_arr
    )
    delta_chi_q = np.searchsorted(
        np.quantile(delta_chi_arr, [0.25, 0.5, 0.75]), delta_chi_arr
    )
    delta_rcov_q = np.searchsorted(
        np.quantile(delta_rcov_arr, [0.25, 0.5, 0.75]), delta_rcov_arr
    )

    breakdowns = [
        _bin_stats(defect_types, "defect_type_int (0=ads, 1=int)"),
        _bin_stats(dopant_blocks, "dopant_block (0=s,1=p,2=d,3=f)"),
        _bin_stats(n_atoms_q, "n_atoms quartile (0=small)"),
        _bin_stats(delta_chi_q, "delta_chi quartile (0=close, 3=far)"),
        _bin_stats(delta_rcov_q, "delta_rcov quartile (0=close, 3=far)"),
    ]

    out = {
        "n_test": int(n),
        "baseline_test_mae": float(abs_err.mean()),
        "spearman_correlations_with_abs_error": correlations,
        "lightgbm_predicting_abs_error_test_r2": err_r2,
        "failure_modes_by_category": breakdowns,
    }
    out_path = ROOT / "results" / "phase_b_error_vs_physics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    print()
    print("=== Failure modes ===")
    for bd in breakdowns:
        print(f"\n{bd['by']}:")
        for k, v in bd["groups"].items():
            print(f"  {k:<6s}  n={v['n']:>4d}  MAE={v['mean_abs_err']:.4f}  P90={v['p90_abs_err']:.4f}")


if __name__ == "__main__":
    main()
