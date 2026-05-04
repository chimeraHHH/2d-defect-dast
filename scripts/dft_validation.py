"""C18: Real DFT validation of C17 priority queue (GPAW PW PBE).

Closes the active learning loop with REAL quantum-mechanical computation.
For each top-N candidate from the σ-ranked priority queue, compute a
single-point GPAW PBE energy and compare with the model's prediction.

Key questions this answers
--------------------------
1. Does the model's σ correlate with actual prediction error on truly OOD
   chemistry? (i.e., is σ a real OOD detector, not just self-consistency?)
2. What is the cost of one DFT calc on the rented 5090 server (CPU mode)?
3. Can we use the DFT-computed energies to retrain and improve MAE on
   IMP2D test set?

Outputs
-------
- results/dft_validation.json     (energies + predictions per candidate)
- paper/figures/fig_dft_validation.png
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

from ase import Atoms  # noqa: E402

CANDIDATES_PATH = ROOT / "data" / "processed" / "candidates_c17.pkl"
PRIORITY_PATH = ROOT / "results" / "c17_dft_priority_queue.csv"
PRED_PATH = ROOT / "results" / "candidates_c17_predictions.json"
RESULTS = ROOT / "results"
LOG_DIR = ROOT / "logs" / "dft"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 10            # how many priority candidates to validate
PW_CUTOFF = 300       # eV; balanced accuracy/speed
KPTS = (2, 2, 1)      # 2D supercells need only k-mesh in plane
XC = "PBE"


def sample_to_atoms(s):
    """Internal sample dict → ASE Atoms with vacuum padding."""
    Z = s["numbers"]
    pos = s["positions"]
    cell = s["cell"].copy()
    # IMP2D supercells already have vacuum in z; ensure cell[2,2] >= 12 A
    if cell[2, 2] < 12:
        cell[2, 2] = max(15.0, cell[2, 2] + 10.0)
    return Atoms(numbers=Z, positions=pos, cell=cell, pbc=True)


def run_one_dft(s, idx, log_path):
    """Run single-point GPAW PW PBE on one candidate."""
    from gpaw import GPAW, PW

    atoms = sample_to_atoms(s)
    n = len(atoms)

    # log file per calc
    txt = str(log_path / f"calc_{idx:04d}.log")
    atoms.calc = GPAW(
        mode=PW(PW_CUTOFF),
        kpts=KPTS,
        xc=XC,
        txt=txt,
        symmetry={"point_group": False},  # avoid relaxation of symmetry
        # accelerate convergence on these single-points:
        convergence={"energy": 1e-3, "density": 1e-3, "eigenstates": 1e-4},
        maxiter=80,
    )
    t0 = time.time()
    try:
        e_total = float(atoms.get_potential_energy())
        wall = time.time() - t0
        return {
            "ok": True,
            "e_total_eV": e_total,
            "n_atoms": int(n),
            "wall_sec": float(wall),
            "e_per_atom_eV": float(e_total / n),
        }
    except Exception as exc:
        wall = time.time() - t0
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "n_atoms": int(n), "wall_sec": float(wall)}


def main():
    t_start = time.time()

    print(f"Loading candidates: {CANDIDATES_PATH}")
    with open(CANDIDATES_PATH, "rb") as f:
        candidates = pickle.load(f)
    print(f"  {len(candidates)} total")

    # rank by priority CSV
    import csv
    priority_ids = []
    with open(PRIORITY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            priority_ids.append(int(row["rank"]) - 1)  # rank is 1-indexed
            if len(priority_ids) >= 100:  # only need top 100 indices
                break

    # load predictions to attach μ, σ
    pred_data = json.load(open(PRED_PATH))
    preds_by_id = {p["id"]: p for p in pred_data["predictions"]}

    # but priority CSV ranks are different from sample ids — we need to map
    # by reading the priority queue with its original sample_id column
    priority_rows = []
    with open(PRIORITY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            priority_rows.append(row)
    # we have host, dopant, defect_type, predicted_Ef, calibrated_sigma
    # find the matching candidate index by host+dopant+defect_type
    rank2idx = {}
    for r_idx, row in enumerate(priority_rows[:TOP_N]):
        h = row["host"]; d = row["dopant"]; t = row["defect_type"]
        for c_idx, c in enumerate(candidates):
            m = c["metadata"]
            if (m["host"] == h and m["dopant"] == d and
                    m.get("defecttype") == t):
                rank2idx[r_idx] = c_idx
                break

    print(f"\n=== Running real DFT on top-{TOP_N} priority candidates ===")
    print(f"  GPAW PW({PW_CUTOFF} eV) {XC}, k-pts={KPTS}\n")

    results = []
    for r_idx in range(TOP_N):
        if r_idx not in rank2idx:
            print(f"  rank {r_idx+1}: SKIP (not found in candidates)")
            continue
        c_idx = rank2idx[r_idx]
        s = candidates[c_idx]
        m = s["metadata"]
        prow = priority_rows[r_idx]
        n = len(s["numbers"])

        print(f"\nrank {r_idx+1}: {m['host']}:{m['dopant']} {m['defecttype']}  "
              f"({n} atoms)")
        print(f"  predicted Ef = {prow['predicted_Ef_eV']} eV, "
              f"σ_cal = {prow['calibrated_sigma_eV']} eV")
        print(f"  starting DFT...", flush=True)

        ret = run_one_dft(s, c_idx, LOG_DIR)
        print(f"  -> {ret}")

        results.append({
            "rank": r_idx + 1,
            "candidate_idx": c_idx,
            "host": m["host"],
            "dopant": m["dopant"],
            "defect_type": m["defecttype"],
            "n_atoms": n,
            "model_pred_Ef_eV": float(prow["predicted_Ef_eV"]),
            "model_sigma_cal_eV": float(prow["calibrated_sigma_eV"]),
            "dft_pw_cutoff_eV": PW_CUTOFF,
            "dft_kpts": list(KPTS),
            "dft_xc": XC,
            **ret,
        })

        # incremental save
        out = {
            "config": {"top_n": TOP_N, "pw_cutoff_eV": PW_CUTOFF,
                       "kpts": list(KPTS), "xc": XC},
            "results": results,
            "wall_time_min": (time.time() - t_start) / 60,
        }
        with open(RESULTS / "dft_validation.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved partial to {RESULTS / 'dft_validation.json'}",
              flush=True)

    print(f"\n=== Done. Total wall time: {(time.time()-t_start)/60:.1f} min ===")
    print(f"  successful: {sum(1 for r in results if r.get('ok'))}/{len(results)}")


if __name__ == "__main__":
    main()
