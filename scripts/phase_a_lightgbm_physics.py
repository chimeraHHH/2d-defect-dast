"""Phase A3 — LightGBM physics-only baseline.

Fits a LightGBM regressor to the formation energy using ONLY the
hand-crafted physical descriptors from phase_a_descriptors.npz +
some per-sample geometric summary statistics. No GNN, no learned
embeddings.

The MAE achieved by this model gives a quantitative
**physical lower bound** on what a model can learn from local
geometry + chemistry features. Compared against:
  * v1 baseline (CrystalTransformer, single seed=42): 0.5161 eV
  * v1 baseline 4-seed: 0.537 ± 0.014 eV
  * v2 multi-source 4-seed: 0.486 ± 0.025 eV
the LightGBM number says how much of the MAE is "explainable by
local geometry alone".

Output: results/phase_a_lightgbm_physics.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

from scripts.phase_a_descriptors import (  # noqa: E402
    PAULING_CHI, COVALENT_R, VALENCE_E, block_of, descriptors_per_sample,
    build_pair_reference_table,
)


def per_sample_physics_features(s: dict, ref_table: Dict = None) -> Dict[str, float]:
    """Aggregate the per-atom physical descriptors into a graph-level
    feature vector (one row per sample for LightGBM)."""
    d = descriptors_per_sample(s, ref_table=ref_table)
    bs_max = d["bond_strain_max"]
    bs_mean = d["bond_strain_mean"]
    angd = d["angle_distortion_max"]
    coc = d["coord_change"]
    dist = d["distance_to_defect"]
    n = d["n_atoms"]

    near = dist < 3.0
    mid = (dist >= 3.0) & (dist < 5.0)
    far = (dist >= 5.0) & (dist < 7.0)
    very_far = dist >= 7.0

    def _safe_mean(a):
        return float(np.nanmean(a)) if a.size else 0.0

    feat = {
        # geometry summaries
        "n_atoms": float(n),
        "bond_strain_max_global": float(np.nanmax(bs_max)) if bs_max.size else 0.0,
        "bond_strain_mean_global": float(np.nanmean(bs_mean)) if bs_mean.size else 0.0,
        "angle_distortion_max_global": float(np.nanmax(angd)) if angd.size else 0.0,
        "coord_change_at_defect": float(coc[d["defect_atom_index"]])
            if 0 <= d["defect_atom_index"] < n else 0.0,
        "coord_change_max_other": float(np.nanmax(np.delete(coc, d["defect_atom_index"])))
            if n > 1 else 0.0,
        "bond_strain_mean_near": _safe_mean(bs_mean[near]),
        "bond_strain_mean_mid": _safe_mean(bs_mean[mid]),
        "bond_strain_mean_far": _safe_mean(bs_mean[far]),
        "bond_strain_mean_very_far": _safe_mean(bs_mean[very_far]),
        "n_strained": float((bs_max > 0.02).sum()),
        "frac_strained": float((bs_max > 0.02).mean()),
        # chemistry features
        "delta_chi": d["delta_chi"],
        "delta_rcov": d["delta_rcov"],
        "delta_valence": d["delta_valence"],
        "dopant_block": float(d["dopant_block"]),
        "z_dopant": float(d["z_dopant"]),
        "defect_type_int": float(d["defect_type_int"]),
        "host_natoms": float(d["host_natoms"]),
        # raw chemistry for reference
        "chi_dopant": float(PAULING_CHI.get(int(d["z_dopant"]), 0.0)),
        "rcov_dopant": float(COVALENT_R.get(int(d["z_dopant"]), 0.0)),
        "valence_dopant": float(VALENCE_E.get(int(d["z_dopant"]), 0.0)),
    }
    return feat


def build_split(data: List[dict], seed: int = 42) -> Dict[str, List[int]]:
    """Same split_indices(seed=42) used by v1 baseline."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_train = int(0.8 * len(data))
    n_val = int(0.1 * len(data))
    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }


def main():
    import pickle
    src = ROOT / "data" / "processed" / "cleaned_dataset_with_pristine.pkl"
    with open(src, "rb") as f:
        blob = pickle.load(f)
    data = blob["data"]
    splits = build_split(data, seed=42)
    print(f"split: {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")

    print("building bond-pair reference table from train fold...")
    ref_table = build_pair_reference_table(
        [data[i] for i in splits["train"]], max_samples=2000
    )

    t0 = time.time()
    feats_train, feats_val, feats_test = [], [], []
    y_train, y_val, y_test = [], [], []
    for split_name, fbuf, ybuf in [("train", feats_train, y_train),
                                     ("val", feats_val, y_val),
                                     ("test", feats_test, y_test)]:
        for i, idx in enumerate(splits[split_name]):
            s = data[idx]
            f = per_sample_physics_features(s, ref_table=ref_table)
            fbuf.append(f)
            ybuf.append(float(s["target"]))
        print(f"  built {split_name} features ({len(fbuf)} samples)  {time.time()-t0:.0f}s")

    feature_names = list(feats_train[0].keys())
    X_train = np.array([[f[k] for k in feature_names] for f in feats_train])
    X_val = np.array([[f[k] for k in feature_names] for f in feats_val])
    X_test = np.array([[f[k] for k in feature_names] for f in feats_test])
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    print(f"X_train shape {X_train.shape}, X_test shape {X_test.shape}")

    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=63,
        min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=5, lambda_l2=0.5, verbose=-1, random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    pred_test = model.predict(X_test)
    test_mae = float(np.mean(np.abs(pred_test - y_test)))
    test_rmse = float(np.sqrt(np.mean((pred_test - y_test) ** 2)))
    pred_val = model.predict(X_val)
    val_mae = float(np.mean(np.abs(pred_val - y_val)))

    # baseline reference: predict mean
    mean_pred_mae = float(np.mean(np.abs(np.mean(y_train) - y_test)))

    feature_importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: -x[1]
    )

    print()
    print(f"  LightGBM physics-only test MAE  : {test_mae:.4f} eV")
    print(f"  LightGBM physics-only test RMSE : {test_rmse:.4f} eV")
    print(f"  mean-predictor baseline          : {mean_pred_mae:.4f} eV")
    print(f"  v1 GNN baseline (seed=42)        : 0.5161 eV")
    print(f"  v2 multi-source 4-seed mean      : 0.486 eV")
    print()
    print(f"  GNN over-LightGBM gain           : {(test_mae - 0.5161):.4f} eV "
          f"({(test_mae - 0.5161)/0.5161*100:+.1f}%)")
    print(f"  LightGBM closes "
          f"{(mean_pred_mae - test_mae) / (mean_pred_mae - 0.5161) * 100:.1f}%"
          f" of the gap from mean-predictor to GNN baseline")

    print()
    print("top 10 feature importances:")
    for name, imp in feature_importance[:10]:
        print(f"  {name:<35s}  {imp}")

    out = {
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "test_mae_eV": test_mae,
        "test_rmse_eV": test_rmse,
        "val_mae_eV": val_mae,
        "mean_predictor_mae_eV": mean_pred_mae,
        "v1_baseline_test_mae": 0.5161,
        "v2_multi_source_4seed_mean": 0.486,
        "feature_importances": [(n, int(i)) for n, i in feature_importance],
        "feature_names": feature_names,
    }
    out_path = ROOT / "results" / "phase_a_lightgbm_physics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
