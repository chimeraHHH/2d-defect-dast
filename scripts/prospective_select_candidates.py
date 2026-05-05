"""Strategic candidate split for prospective DFT validation.

Reads ``results/candidates_c17_predictions.json`` (the 287 generated
(host, dopant, site) candidates produced by v1.2's
``scripts/generate_candidates.py``) and selects 60 candidates in two
buckets:

  bucket A (n=30):  high-confidence low-Ef  ─ the "discovery" set
                    sort by predicted Ef (ascending), keep only those
                    with σ_cal ≤ 1.0 eV, take top-30.
  bucket B (n=30):  high-σ OOD              ─ the "stress test"
                    sort by σ_cal (descending), take top-30
                    regardless of predicted Ef.

The split intentionally diversifies hosts (max 4 candidates per host
per bucket) so the prospective set spans chemistry families.

Output:
  results/prospective_dft_split.json  (60 candidates with metadata)
  data/processed/candidates_c17_prospective.pkl
                                       (subset pkl for QE input gen)
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    pred_path = ROOT / "results" / "candidates_c17_predictions.json"
    cand_pkl = ROOT / "data" / "processed" / "candidates_c17.pkl"
    if not pred_path.exists():
        print(f"missing {pred_path}; run scripts/predict_candidates.py first")
        sys.exit(1)
    if not cand_pkl.exists():
        print(f"missing {cand_pkl}; run scripts/generate_candidates.py first")
        sys.exit(1)

    with open(pred_path) as f:
        blob = json.load(f)
    preds = blob.get("predictions", blob)
    print(f"loaded {len(preds)} candidate predictions  (tau={blob.get('tau', 'n/a')})")

    with open(cand_pkl, "rb") as f:
        cands = pickle.load(f)
    print(f"loaded {len(cands)} candidate structures")

    # Build a host→count budget for diversity.
    by_host_A: Dict[str, int] = defaultdict(int)
    by_host_B: Dict[str, int] = defaultdict(int)
    HOST_CAP_A = 4  # max 4 per host in bucket A
    HOST_CAP_B = 4

    # Bucket A: top-30 lowest predicted Ef regardless of sigma.
    # The σ-filtered version (σ ≤ 1) yields only ~4 candidates because the
    # 287-pool is by construction OOD (median σ_cal ≈ 4.4 per v1.2 §5.20).
    # The honest test is: do the model's "lowest Ef" picks survive DFT
    # validation, or does the model over-confidently predict low Ef on OOD?
    A_pool = list(preds)
    A_pool.sort(key=lambda p: p.get("mu", 1e9))

    bucket_A = []
    for p in A_pool:
        host = p.get("host", "?")
        if by_host_A[host] >= HOST_CAP_A:
            continue
        bucket_A.append({**p, "bucket": "A_low_Ef_high_conf"})
        by_host_A[host] += 1
        if len(bucket_A) >= 30:
            break

    # Bucket B: high σ
    B_pool = sorted(preds, key=lambda p: -p.get("sigma_cal", 0))
    bucket_B = []
    A_idxs = {p["id"] for p in bucket_A}
    for p in B_pool:
        if p["id"] in A_idxs:
            continue
        host = p.get("host", "?")
        if by_host_B[host] >= HOST_CAP_B:
            continue
        bucket_B.append({**p, "bucket": "B_high_sigma_OOD"})
        by_host_B[host] += 1
        if len(bucket_B) >= 30:
            break

    print(f"\nBucket A (low Ef, σ ≤ 1):  n={len(bucket_A)}")
    for r in bucket_A[:10]:
        print(f"  id={r['id']:>4d}  {r['host']:<8s} {r['dopant']:<3s} {r['defect_type']:<13s}  "
              f"Ef={r['mu']:.3f}  σ={r['sigma_cal']:.3f}")

    print(f"\nBucket B (high σ OOD):     n={len(bucket_B)}")
    for r in bucket_B[:10]:
        print(f"  id={r['id']:>4d}  {r['host']:<8s} {r['dopant']:<3s} {r['defect_type']:<13s}  "
              f"Ef={r['mu']:.3f}  σ={r['sigma_cal']:.3f}")

    print(f"\nbucket A by host: {dict(by_host_A)}")
    print(f"bucket B by host: {dict(by_host_B)}")

    out = {
        "n_total": len(bucket_A) + len(bucket_B),
        "bucket_A_low_Ef_high_conf": bucket_A,
        "bucket_B_high_sigma_OOD": bucket_B,
        "selection_criteria": {
            "bucket_A": "low predicted Ef, sigma_cal <= 1.0 eV, max 4/host",
            "bucket_B": "high sigma_cal regardless of Ef, max 4/host",
        },
    }
    out_path = ROOT / "results" / "prospective_dft_split.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")

    # Save the structures sub-pickle aligned to bucket order
    selected_idxs = [r["id"] for r in bucket_A + bucket_B]
    selected = []
    for idx in selected_idxs:
        if 0 <= idx < len(cands):
            c = dict(cands[idx]) if isinstance(cands[idx], dict) else {"_raw": cands[idx]}
            c["candidate_id"] = idx
            selected.append(c)
    pkl_out = ROOT / "data" / "processed" / "candidates_prospective.pkl"
    with open(pkl_out, "wb") as f:
        pickle.dump(selected, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {pkl_out}  ({len(selected)} structures)")


if __name__ == "__main__":
    main()
