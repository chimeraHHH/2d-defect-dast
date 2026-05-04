"""C17 Stage 0: Validate MACE-MP-0 as pseudo-DFT oracle on IMP2D.

If MACE-MP-0 (a foundation model trained on Materials Project trajectories)
can give formation-energy estimates correlating well with our IMP2D ground
truth, we can use it to pseudo-label thousands of new candidate defects
without running real DFT.

Decision rule
-------------
- MAE(pseudo, true) < 0.7 eV  AND  Pearson r > 0.85
    → use as pseudo-DFT oracle (Stage 3 = real foundation labels)
- else
    → fallback to self-distillation (Stage 3 = ensemble-mean labels)

Outputs
-------
- results/mace_mp_validation.json
- paper/figures/fig_mace_mp_validation.png  (parity)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_VALIDATE = 100   # sample size for the validation
SEED = 42


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_to_atoms(s):
    """Convert internal sample dict to ASE Atoms."""
    return Atoms(
        numbers=s["numbers"],
        positions=s["positions"],
        cell=s["cell"],
        pbc=True,
    )


def main():
    device = get_device()
    print(f"Device: {device}")

    # load MACE-MP-0
    print("Loading MACE-MP-0 (foundation model)...")
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", dispersion=False,
                   default_dtype="float32", device=str(device))
    print("  ready.")

    # load IMP2D + sample
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"IMP2D: {len(data)} samples")

    rng = np.random.default_rng(SEED)
    val_idx = rng.choice(len(data), size=N_VALIDATE, replace=False)

    # for each sample, compute MACE-MP-0 total energy on the defect cell
    # we need a reference for "formation" but a quick proxy is:
    #   E_per_atom_pseudo = E_total_MACE / N_atoms
    #   E_form_pseudo ≈ a * E_per_atom + b
    # we'll fit (a, b) via linear regression on the validation set
    per_atom_e_mace = []
    per_atom_e_true = []
    natoms_list = []
    targets = []
    metadata = []

    t0 = time.time()
    for i, idx in enumerate(val_idx):
        s = data[idx]
        atoms = sample_to_atoms(s)
        atoms.calc = calc
        try:
            e_total = float(atoms.get_potential_energy())
        except Exception as e:
            print(f"  skip {i}: {e}")
            continue
        n = len(s["numbers"])
        per_atom_e_mace.append(e_total / n)
        per_atom_e_true.append(s["target"] / n)
        natoms_list.append(n)
        targets.append(s["target"])
        metadata.append({
            "host": s["metadata"]["host"],
            "dopant": s["metadata"]["dopant"],
            "natoms": n,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_VALIDATE}  ({time.time()-t0:.0f}s)", flush=True)

    per_atom_e_mace = np.array(per_atom_e_mace)
    per_atom_e_true = np.array(per_atom_e_true)
    targets = np.array(targets)
    natoms = np.array(natoms_list)

    # raw correlation
    pearson_per_atom = float(np.corrcoef(per_atom_e_mace, per_atom_e_true)[0, 1])

    # try the simplest pseudo-label: linear calibration
    A = np.column_stack([per_atom_e_mace, np.ones_like(per_atom_e_mace)])
    coefs, *_ = np.linalg.lstsq(A, per_atom_e_true, rcond=None)
    a, b = float(coefs[0]), float(coefs[1])
    pred_per_atom = a * per_atom_e_mace + b
    pred_total = pred_per_atom * natoms

    mae = float(np.abs(pred_total - targets).mean())
    rmse = float(np.sqrt(((pred_total - targets) ** 2).mean()))
    pearson_total = float(np.corrcoef(pred_total, targets)[0, 1])

    # also try a per-host calibration to see how heterogeneous it is
    host_arr = np.array([m["host"] for m in metadata])
    per_host_results = {}
    for h in np.unique(host_arr):
        mask = host_arr == h
        if mask.sum() < 3:
            continue
        per_host_results[h] = {
            "n": int(mask.sum()),
            "pearson_per_atom": float(np.corrcoef(
                per_atom_e_mace[mask], per_atom_e_true[mask])[0, 1]),
            "true_mean_eV": float(targets[mask].mean()),
        }

    print(f"\n=== MACE-MP-0 validation on {len(per_atom_e_mace)} IMP2D samples ===")
    print(f"  Per-atom Pearson r = {pearson_per_atom:.3f}")
    print(f"  After linear calibration (a={a:.3f}, b={b:.3f}):")
    print(f"    MAE  = {mae:.4f} eV")
    print(f"    RMSE = {rmse:.4f} eV")
    print(f"    Pearson r (total) = {pearson_total:.3f}")

    decision = (mae < 0.7) and (pearson_total > 0.85)
    print(f"\nDecision rule: MAE<0.7 & r>0.85 → {decision}")
    print(f"  → {'USE MACE-MP-0 as pseudo-DFT oracle' if decision else 'FALLBACK to self-distillation'}")

    out = {
        "n_validate": int(len(per_atom_e_mace)),
        "pearson_per_atom": pearson_per_atom,
        "calibration_a": a,
        "calibration_b": b,
        "mae_total_eV": mae,
        "rmse_total_eV": rmse,
        "pearson_total": pearson_total,
        "decision_use_pseudo_dft": decision,
        "per_host": per_host_results,
        "wall_time_min": (time.time() - t0) / 60,
    }
    out_json = RESULTS / "mace_mp_validation.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_json}")

    # parity figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(per_atom_e_true, per_atom_e_mace, s=14, alpha=0.6)
    axes[0].set_xlabel("True per-atom Ef (eV/atom)")
    axes[0].set_ylabel("MACE-MP-0 per-atom E (eV/atom)")
    axes[0].set_title(f"Per-atom raw (r={pearson_per_atom:.2f})")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(targets, pred_total, s=14, alpha=0.6)
    lo, hi = min(targets.min(), pred_total.min())-0.5, max(targets.max(), pred_total.max())+0.5
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
    axes[1].set_xlabel("True Ef per cell (eV)")
    axes[1].set_ylabel("MACE-MP-0 calibrated pseudo Ef (eV)")
    axes[1].set_title(f"Calibrated (MAE={mae:.3f}, r={pearson_total:.2f})")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_fig = FIG_DIR / "fig_mace_mp_validation.png"
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"figure saved -> {out_fig}")


if __name__ == "__main__":
    main()
