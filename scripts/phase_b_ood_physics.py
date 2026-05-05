"""Phase B2 — LOHO host distribution shift in physics-feature space.

For each of the five LOHO hosts (MoS2, Cr2I6, C2H2, TaSe2, MoSSe) we
compare the distribution of physics descriptors against the
training distribution (the other 39 hosts), to quantitatively answer:

  * Which hosts have physics features outside the training distribution?
  * Does the OOD-extreme host (C2H2, 4.65x degradation) actually have
    extreme physics, or is the failure orthogonal to obvious physics?
  * Which features (chemistry vs strain vs coord change) drive the
    distribution shift?

For each (host, feature) we report:
  * train mean / std (over non-host samples)
  * holdout mean / std (over host samples)
  * standardised mean shift (Cohen's d):
      d = (mu_holdout - mu_train) / sqrt((sigma_train^2 + sigma_holdout^2)/2)
  * KS statistic (probability the two are drawn from the same distribution)

Output: results/phase_b_ood_physics.json

This makes the LOHO failure quantitatively grounded in physics descriptor
distribution shift, not just a model-side observation.
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

from scipy.stats import ks_2samp  # noqa: E402

from scripts.phase_a_lightgbm_physics import per_sample_physics_features  # noqa: E402
from scripts.phase_a_descriptors import build_pair_reference_table  # noqa: E402

LOHO_HOSTS = ["MoS2", "Cr2I6", "C2H2", "TaSe2", "MoSSe"]

# v1 single-source LOHO baseline numbers (from results/loho_summary.json)
V1_LOHO_DEGRADATION = {
    "MoS2": 1.000,
    "Cr2I6": 1.628,
    "C2H2": 4.648,
    "TaSe2": 1.233,
    "MoSSe": 0.868,
}


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Standardised mean difference; ((mu_y - mu_x) / pooled std)."""
    if x.size < 2 or y.size < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = np.sqrt((sx * sx + sy * sy) / 2.0 + 1e-12)
    return float((my - mx) / pooled)


def main():
    src = ROOT / "data" / "processed" / "cleaned_dataset_with_pristine.pkl"
    with open(src, "rb") as f:
        blob = pickle.load(f)
    data = blob["data"]
    print(f"loaded {len(data)} samples")

    # Build reference table from a random subset (avoid host bias)
    print("building bond-pair reference table from random subset...")
    ref_table = build_pair_reference_table(data, max_samples=2000)

    print("computing per-sample physics features for all 10641 samples ...")
    t0 = time.time()
    feats = []
    hosts = []
    for k, s in enumerate(data):
        feats.append(per_sample_physics_features(s, ref_table=ref_table))
        hosts.append(s.get("metadata", {}).get("host", "?"))
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(data)}  ({time.time()-t0:.0f}s)")
    feature_names = list(feats[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in feats])
    hosts = np.array(hosts)
    print(f"X shape {X.shape}  feature_names = {len(feature_names)}")

    out_per_host = {}
    print("\nLOHO physics distribution shift:")
    for host in LOHO_HOSTS:
        mask = hosts == host
        n_held = int(mask.sum())
        if n_held < 10:
            print(f"  {host}: only {n_held} samples; skipping")
            continue
        n_train = int((~mask).sum())
        # per-feature analysis
        per_feature = []
        for k, name in enumerate(feature_names):
            x_train = X[~mask, k]
            x_held = X[mask, k]
            # filter non-finite
            x_train = x_train[np.isfinite(x_train)]
            x_held = x_held[np.isfinite(x_held)]
            if x_train.size < 10 or x_held.size < 10:
                continue
            try:
                ks = float(ks_2samp(x_train, x_held).statistic)
            except Exception:
                ks = 0.0
            d = cohen_d(x_train, x_held)
            per_feature.append({
                "feature": name,
                "train_mean": float(np.mean(x_train)),
                "train_std": float(np.std(x_train)),
                "holdout_mean": float(np.mean(x_held)),
                "holdout_std": float(np.std(x_held)),
                "cohen_d": d,
                "ks_stat": ks,
            })
        # Aggregate: max |Cohen's d| across features
        per_feature.sort(key=lambda x: -abs(x["cohen_d"]))
        max_abs_d = abs(per_feature[0]["cohen_d"])
        sum_abs_d = float(np.sum([abs(p["cohen_d"]) for p in per_feature]))
        out_per_host[host] = {
            "n_holdout": n_held,
            "n_train": n_train,
            "v1_loho_degradation_factor": V1_LOHO_DEGRADATION.get(host, None),
            "max_abs_cohen_d": max_abs_d,
            "sum_abs_cohen_d": sum_abs_d,
            "top_5_shifted_features": per_feature[:5],
            "feature_full_table": per_feature,
        }
        print(f"\n{host}  v1 LOHO {V1_LOHO_DEGRADATION.get(host)}× degradation")
        print(f"  n_train_other={n_train}  n_holdout={n_held}")
        print(f"  max |Cohen's d| = {max_abs_d:.3f}, sum = {sum_abs_d:.2f}")
        print(f"  top-5 shifted features:")
        for p in per_feature[:5]:
            print(
                f"    {p['feature']:<30s}"
                f"  d={p['cohen_d']:+.2f}  KS={p['ks_stat']:.2f}  "
                f"train µ={p['train_mean']:.3f} σ={p['train_std']:.3f}"
                f"  holdout µ={p['holdout_mean']:.3f} σ={p['holdout_std']:.3f}"
            )

    # Correlate physics-shift score with v1 LOHO degradation factor
    rows = []
    for host, info in out_per_host.items():
        if info["v1_loho_degradation_factor"] is not None:
            rows.append((info["v1_loho_degradation_factor"],
                         info["sum_abs_cohen_d"], info["max_abs_cohen_d"]))
    if len(rows) >= 3:
        deg = np.array([r[0] for r in rows])
        sum_d = np.array([r[1] for r in rows])
        max_d = np.array([r[2] for r in rows])
        from scipy.stats import pearsonr, spearmanr
        sp_sum = spearmanr(deg, sum_d).statistic
        pe_sum = pearsonr(deg, sum_d).statistic
        sp_max = spearmanr(deg, max_d).statistic
        pe_max = pearsonr(deg, max_d).statistic
        print()
        print(f"Across {len(rows)} hosts: degradation_factor vs physics-shift score")
        print(f"  sum_abs_cohen_d:  Spearman ρ = {sp_sum:+.2f},  Pearson = {pe_sum:+.2f}")
        print(f"  max_abs_cohen_d:  Spearman ρ = {sp_max:+.2f},  Pearson = {pe_max:+.2f}")
    else:
        sp_sum = pe_sum = sp_max = pe_max = None

    out_per_host["__cross_host_correlation__"] = {
        "spearman_degradation_vs_sum_abs_d": sp_sum,
        "pearson_degradation_vs_sum_abs_d": pe_sum,
        "spearman_degradation_vs_max_abs_d": sp_max,
        "pearson_degradation_vs_max_abs_d": pe_max,
        "n_hosts_compared": len(rows) if rows else 0,
    }
    out_path = ROOT / "results" / "phase_b_ood_physics.json"
    with open(out_path, "w") as f:
        json.dump(out_per_host, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
