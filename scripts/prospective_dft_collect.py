"""Parse QE pw.x outputs from the prospective 60-candidate DFT validation.

Inputs (per QE output file in <DFT_DIR>):
  cand{NNN}.out        — defect supercell SCF
  pristine_<host>.out  — host pristine
  atom_<El>.out        — isolated-atom mu reference

For each of the 60 candidates we compute (interstitial only):
    Ef_DFT = E_defect - E_pristine_host - mu_dopant
where mu_dopant is the per-atom energy of the isolated-atom QE calc
(consistent with the original C18c convention; for parity-plot purposes
this fixes a constant offset per dopant).

Output: results/prospective_dft_results.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Default location after `scp -r` from remote
DFT_DIR_DEFAULT = ROOT / "results" / "qe_outputs"
RY = 13.6056980659  # 1 Ry in eV (QE convention)


def parse_total_energy_ry(path: Path) -> float | None:
    """Return final '!    total energy' value in Ry, or None if not converged."""
    if not path.exists():
        return None
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return None
    if "JOB DONE" not in text:
        return None
    # "!    total energy              =    -2867.38086481 Ry"
    matches = re.findall(r"^\s*!\s*total energy\s*=\s*(-?\d+\.\d+)\s*Ry", text, re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])


def parse_n_atoms(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return None
    m = re.search(r"number of atoms/cell\s*=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def parse_wall_seconds(path: Path) -> float | None:
    """Parse 'PWSCF        :   18m 5.34s CPU  23m17.05s WALL'."""
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    m = re.search(
        r"PWSCF\s*:.*?(\d+(?:\.\d+)?)s\s*WALL", text, re.DOTALL
    )
    if not m:
        m2 = re.search(r"(\d+)m\s*([\d.]+)s\s*WALL", text)
        if not m2:
            return None
        return float(m2.group(1)) * 60 + float(m2.group(2))
    return float(m.group(1))


def main():
    dft_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DFT_DIR_DEFAULT
    if not dft_dir.exists():
        print(f"DFT outputs dir not found: {dft_dir}")
        sys.exit(1)
    print(f"reading QE outputs from {dft_dir}")

    split_path = ROOT / "results" / "prospective_dft_split.json"
    split = json.loads(split_path.read_text())
    cands = split["bucket_A_low_Ef_high_conf"] + split["bucket_B_high_sigma_OOD"]
    print(f"60 candidates loaded ({len(split['bucket_A_low_Ef_high_conf'])}A + "
          f"{len(split['bucket_B_high_sigma_OOD'])}B)")

    # Parse atom mu's
    mu_atom = {}  # element -> E_total_eV (per single atom)
    for f in sorted(dft_dir.glob("atom_*.out")):
        el = f.stem.replace("atom_", "")
        e_ry = parse_total_energy_ry(f)
        if e_ry is not None:
            mu_atom[el] = e_ry * RY
    print(f"mu_atom: {len(mu_atom)} elements parsed: {sorted(mu_atom.keys())}")

    # Parse pristine
    pristine = {}  # host -> E_total_eV
    for f in sorted(dft_dir.glob("pristine_*.out")):
        host = f.stem.replace("pristine_", "")
        e_ry = parse_total_energy_ry(f)
        if e_ry is not None:
            pristine[host] = e_ry * RY
    print(f"pristine: {len(pristine)} hosts parsed: {sorted(pristine.keys())}")

    # Parse candidates
    rows = []
    n_done = n_skipped = 0
    for c in cands:
        cid = c["id"]
        # The QE input filenames are cand{ID:03d}_<bucket>_<host>_<dopant>.in
        # so the .out files use the same prefix.
        matches = sorted(dft_dir.glob(f"cand{cid:03d}_*.out"))
        if not matches:
            n_skipped += 1
            continue
        out_path = matches[0]
        e_def_ry = parse_total_energy_ry(out_path)
        if e_def_ry is None:
            n_skipped += 1
            continue
        n_done += 1
        e_def = e_def_ry * RY
        host = c["host"]
        dop = c["dopant"]
        if host not in pristine or dop not in mu_atom:
            print(f"  cand{cid:03d} {host}/{dop}: missing reference (host={host in pristine} "
                  f"dop={dop in mu_atom})")
            continue
        ef_dft = e_def - pristine[host] - mu_atom[dop]
        wall = parse_wall_seconds(out_path)
        rows.append({
            "id": cid,
            "bucket": c["bucket"],
            "host": host,
            "dopant": dop,
            "defect_type": c["defect_type"],
            "n_atoms": c["natoms"],
            "model_pred_Ef_eV": float(c["mu"]),
            "model_sigma_cal_eV": float(c["sigma_cal"]),
            "model_sigma_raw_eV": float(c["sigma_raw"]),
            "dft_e_defect_eV": e_def,
            "dft_e_pristine_eV": pristine[host],
            "dft_mu_dopant_eV": mu_atom[dop],
            "dft_Ef_eV": ef_dft,
            "abs_error_eV": abs(ef_dft - float(c["mu"])),
            "wall_sec": wall,
        })

    print(f"\nparsed {n_done} of 60 candidates ({n_skipped} skipped: not converged)")
    print(f"with full reference: {len(rows)}")

    # Stats
    if rows:
        ef_dft = np.array([r["dft_Ef_eV"] for r in rows])
        ef_pred = np.array([r["model_pred_Ef_eV"] for r in rows])
        sig = np.array([r["model_sigma_cal_eV"] for r in rows])
        err = np.array([r["abs_error_eV"] for r in rows])
        bucket = np.array([r["bucket"] for r in rows])
        mA = bucket == "A_low_Ef_high_conf"
        mB = bucket == "B_high_sigma_OOD"
        print(f"\n=== overall ===")
        print(f"  N = {len(rows)}")
        print(f"  MAE  = {err.mean():.3f} eV")
        print(f"  RMSE = {np.sqrt((err**2).mean()):.3f} eV")
        if len(rows) > 1:
            print(f"  Pearson(pred, DFT) = {np.corrcoef(ef_pred, ef_dft)[0,1]:.3f}")
            print(f"  Pearson(σ, |err|)  = {np.corrcoef(sig, err)[0,1]:.3f}")
        if mA.sum() > 1:
            print(f"\n=== bucket A (low Ef, high conf) ===")
            print(f"  N = {mA.sum()}")
            print(f"  MAE = {err[mA].mean():.3f} eV")
            print(f"  Pearson(pred, DFT) = {np.corrcoef(ef_pred[mA], ef_dft[mA])[0,1]:.3f}")
        if mB.sum() > 1:
            print(f"\n=== bucket B (high σ OOD) ===")
            print(f"  N = {mB.sum()}")
            print(f"  MAE = {err[mB].mean():.3f} eV")
            print(f"  Pearson(σ, |err|) = {np.corrcoef(sig[mB], err[mB])[0,1]:.3f}")

    out = {
        "n_candidates": len(cands),
        "n_dft_done": n_done,
        "n_with_refs": len(rows),
        "rows": rows,
    }
    out_path = ROOT / "results" / "prospective_dft_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
