"""Generate Quantum ESPRESSO pw.x input files for the prospective DFT
candidate set + the chemical-potential references they require.

Outputs into /root/2d-defect-dast/data/qe_inputs/ (on the remote host) when
deployed. Locally writes to data/qe_inputs/ for inspection.

Two modes:
  --kind candidates    : generate one pw.x .in per candidate from
                         candidates_prospective.pkl
  --kind mu_atoms      : generate one pw.x .in per isolated atom
                         (chemistry potentials Δμ_dopant)
  --kind mu_bulk       : generate one pw.x .in per bulk reservoir
                         (per-element bulk crystal)
  --kind pristine      : generate pristine host supercells for
                         E_host(supercell) reference

Pseudopotentials: SSSP efficiency 1.3.0 (PBE) — must be pre-downloaded
to the same directory referenced by ``pseudo_dir`` in &CONTROL.

Conversion notes:
  v1.2 GPAW used PW(300 eV) = 22.05 Ry. We use ecutwfc=50 Ry
  (~680 eV), ecutrho=400 Ry to match SSSP precision recommendation;
  this gives ~few-meV converged energies and is fast on RTX 5090
  via GPU offload.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# SSSP efficiency 1.3.0 PBE pseudopotential filenames keyed by element.
# Source: https://www.materialscloud.org/discover/sssp/table/efficiency
# Subset covering elements appearing in our 60 candidates + their hosts.
SSSP_PBE_EFFICIENCY = {
    "H":  "H_ONCV_PBE-1.0.upf",
    "He": "He_ONCV_PBE-1.0.upf",
    "Li": "li_pbe_v1.4.uspp.F.UPF",
    "Be": "Be_ONCV_PBE-1.0.upf",
    "B":  "b_pbe_v1.4.uspp.F.UPF",
    "C":  "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N":  "N.pbe-n-radius_5.UPF",
    "O":  "O.pbe-n-kjpaw_psl.0.1.UPF",
    "F":  "f_pbe_v1.4.uspp.F.UPF",
    "Na": "na_pbe_v1.5.uspp.F.UPF",
    "Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "P":  "P.pbe-n-rrkjus_psl.1.0.0.UPF",
    "S":  "s_pbe_v1.4.uspp.F.UPF",
    "Cl": "cl_pbe_v1.4.uspp.F.UPF",
    "K":  "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ca": "Ca_pbe_v1.uspp.F.UPF",
    "Sc": "Sc_ONCV_PBE-1.0.upf",
    "Ti": "ti_pbe_v1.4.uspp.F.UPF",
    "V":  "v_pbe_v1.4.uspp.F.UPF",
    "Cr": "cr_pbe_v1.5.uspp.F.UPF",
    "Mn": "mn_pbe_v1.5.uspp.F.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
    "Co": "Co_pbe_v1.2.uspp.F.UPF",
    "Ni": "ni_pbe_v1.4.uspp.F.UPF",
    "Cu": "Cu_pbe_v1.2.uspp.F.UPF",
    "Zn": "Zn_pbe_v1.uspp.F.UPF",
    "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Ge": "ge_pbe_v1.4.uspp.F.UPF",
    "As": "As.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Se": "Se_pbe_v1.uspp.F.UPF",
    "Br": "br_pbe_v1.4.uspp.F.UPF",
    "Rb": "Rb_ONCV_PBE-1.0.upf",
    "Sr": "Sr_pbe_v1.uspp.F.UPF",
    "Y":  "Y_pbe_v1.uspp.F.UPF",
    "Zr": "Zr_pbe_v1.uspp.F.UPF",
    "Nb": "Nb.pbe-spn-kjpaw_psl.0.3.0.UPF",
    "Mo": "Mo_ONCV_PBE-1.0.upf",
    "Tc": "Tc_ONCV_PBE-1.0.upf",
    "Ru": "Ru_ONCV_PBE-1.0.upf",
    "Rh": "Rh_ONCV_PBE-1.0.upf",
    "Pd": "Pd_ONCV_PBE-1.0.upf",
    "Ag": "Ag_ONCV_PBE-1.0.upf",
    "Cd": "Cd.pbe-dn-rrkjus_psl.0.3.1.UPF",
    "In": "In.pbe-dn-rrkjus_psl.0.2.2.UPF",
    "Sn": "Sn_pbe_v1.uspp.F.UPF",
    "Sb": "sb_pbe_v1.4.uspp.F.UPF",
    "Te": "Te_pbe_v1.uspp.F.UPF",
    "I":  "I.pbe-n-kjpaw_psl.0.2.UPF",
    "Cs": "Cs_pbe_v1.uspp.F.UPF",
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "La": "La.pbe-spfn-kjpaw_psl.1.0.0.UPF",
    "Ce": "Ce.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Pr": "Pr.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Nd": "Nd.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Sm": "Sm.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Eu": "Eu.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Gd": "Gd.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Tb": "Tb.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Dy": "Dy.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Ho": "Ho.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Er": "Er.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Tm": "Tm.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Yb": "Yb.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Lu": "Lu.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Hf": "Hf-sp.oncvpsp.upf",
    "Ta": "Ta_pbe_v1.uspp.F.UPF",
    "W":  "W_pbe_v1.2.uspp.F.UPF",
    "Re": "Re_pbe_v1.2.uspp.F.UPF",
    "Os": "Os_pbe_v1.2.uspp.F.UPF",
    "Ir": "Ir_pbe_v1.2.uspp.F.UPF",
    "Pt": "pt_pbe_v1.4.uspp.F.UPF",
    "Au": "Au_ONCV_PBE-1.0.upf",
    "Hg": "Hg_ONCV_PBE-1.0.upf",
    "Tl": "Tl_pbe_v1.2.uspp.F.UPF",
    "Pb": "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF",
    "Bi": "bi_pbe_v1.4.uspp.F.UPF",
}

# QE input template
QE_INPUT_TEMPLATE = """\
&CONTROL
    calculation = '{calc}'
    prefix      = '{prefix}'
    pseudo_dir  = '{pseudo_dir}'
    outdir      = '{outdir}'
    verbosity   = 'low'
    tprnfor     = .true.
    tstress     = .true.
    nstep       = 60
    forc_conv_thr = 1.0d-3
    etot_conv_thr = 1.0d-5
/
&SYSTEM
    ibrav   = 0
    nat     = {nat}
    ntyp    = {ntyp}
    ecutwfc = 50.0
    ecutrho = 400.0
    occupations = 'smearing'
    smearing    = 'mp'
    degauss     = 0.01
{spin_block}
/
&ELECTRONS
    conv_thr        = 1.0d-7
    mixing_beta     = 0.4
    electron_maxstep = 200
    diagonalization = 'david'
/
&IONS
    ion_dynamics = 'bfgs'
/

CELL_PARAMETERS angstrom
{cell_str}

ATOMIC_SPECIES
{species_str}

ATOMIC_POSITIONS angstrom
{positions_str}

K_POINTS automatic
{kpts_str}
"""


def build_input(
    atoms: Atoms,
    prefix: str,
    pseudo_dir: str,
    outdir: str,
    kpts: Tuple[int, int, int] = (2, 2, 1),
    spin: bool = False,
    calc: str = "scf",
) -> str:
    species = sorted(set(atoms.get_chemical_symbols()))
    species_lines = []
    missing = []
    for s in species:
        z = atomic_numbers[s]
        from ase.data import atomic_masses
        m = atomic_masses[z]
        upf = SSSP_PBE_EFFICIENCY.get(s)
        if upf is None:
            missing.append(s)
            upf = f"MISSING_{s}.UPF"
        species_lines.append(f"  {s:<3s} {m:>9.4f}  {upf}")
    species_str = "\n".join(species_lines)
    if missing:
        print(f"  WARNING: missing pseudo for {missing}", file=sys.stderr)

    cell = atoms.get_cell()
    cell_str = "\n".join(
        f"  {cell[i, 0]:>14.8f} {cell[i, 1]:>14.8f} {cell[i, 2]:>14.8f}"
        for i in range(3)
    )
    positions = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    positions_str = "\n".join(
        f"  {syms[j]:<3s} {positions[j, 0]:>14.8f} "
        f"{positions[j, 1]:>14.8f} {positions[j, 2]:>14.8f}"
        for j in range(len(atoms))
    )
    kpts_str = f"  {kpts[0]} {kpts[1]} {kpts[2]} 0 0 0"
    spin_block = ""
    if spin:
        spin_block = "    nspin = 2\n    starting_magnetization(1) = 0.5"

    return QE_INPUT_TEMPLATE.format(
        calc=calc,
        prefix=prefix,
        pseudo_dir=pseudo_dir,
        outdir=outdir,
        nat=len(atoms),
        ntyp=len(species),
        spin_block=spin_block,
        cell_str=cell_str,
        species_str=species_str,
        positions_str=positions_str,
        kpts_str=kpts_str,
    )


def candidate_to_atoms(c: dict) -> Atoms:
    """Build an ASE Atoms object from a candidate dict.
    Candidates were built by scripts/generate_candidates.py and
    should have numbers, positions, cell.
    """
    return Atoms(
        numbers=c["numbers"],
        positions=c["positions"],
        cell=c["cell"],
        pbc=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["candidates", "mu_atoms", "mu_bulk", "pristine"],
                   default="candidates")
    p.add_argument("--out-dir", default="data/qe_inputs/candidates")
    p.add_argument("--pseudo-dir", default="/root/qe_pseudos")
    p.add_argument("--outdir", default="/root/qe_scratch")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "candidates":
        cand_pkl = ROOT / "data" / "processed" / "candidates_prospective.pkl"
        with open(cand_pkl, "rb") as f:
            cands = pickle.load(f)
        split = json.load(open(ROOT / "results" / "prospective_dft_split.json"))
        # build a mapping from candidate_id to bucket label
        bucket_map = {}
        for r in split["bucket_A_low_Ef_high_conf"]:
            bucket_map[r["id"]] = ("A", r["host"], r["dopant"], r["mu"], r["sigma_cal"])
        for r in split["bucket_B_high_sigma_OOD"]:
            bucket_map[r["id"]] = ("B", r["host"], r["dopant"], r["mu"], r["sigma_cal"])

        manifest = []
        for c in cands:
            cid = c.get("candidate_id", -1)
            atoms = candidate_to_atoms(c)
            bucket, host, dopant, pred_mu, sigma = bucket_map.get(
                cid, ("?", c.get("metadata", {}).get("host", "?"),
                      c.get("metadata", {}).get("dopant", "?"), 0.0, 0.0)
            )
            prefix = f"cand{cid:03d}_{bucket}_{host}_{dopant}"
            inp = build_input(
                atoms, prefix=prefix,
                pseudo_dir=args.pseudo_dir, outdir=args.outdir,
                kpts=(2, 2, 1), spin=False, calc="scf",
            )
            inp_path = out_dir / f"{prefix}.in"
            with open(inp_path, "w") as f:
                f.write(inp)
            manifest.append({
                "candidate_id": cid,
                "bucket": bucket,
                "host": host,
                "dopant": dopant,
                "n_atoms": len(atoms),
                "model_pred_Ef_eV": pred_mu,
                "model_sigma_cal_eV": sigma,
                "input_file": inp_path.name,
                "prefix": prefix,
            })
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote {len(manifest)} QE inputs to {out_dir}")

    elif args.kind == "mu_atoms":
        # Isolated atom in 15 Å cubic vacuum cell, gamma-point k-mesh.
        elements_needed = set()
        for c in pickle.load(open(ROOT / "data" / "processed" / "candidates_prospective.pkl", "rb")):
            for s in {chemical_symbols[int(z)] for z in c["numbers"]}:
                elements_needed.add(s)
        print(f"elements appearing in 60 candidates: {sorted(elements_needed)}")
        for s in sorted(elements_needed):
            atoms = Atoms([s], positions=[[7.5, 7.5, 7.5]], cell=[15, 15, 15], pbc=True)
            prefix = f"atom_{s}"
            inp = build_input(atoms, prefix=prefix,
                              pseudo_dir=args.pseudo_dir, outdir=args.outdir,
                              kpts=(1, 1, 1), spin=True, calc="scf")
            with open(out_dir / f"{prefix}.in", "w") as f:
                f.write(inp)
        print(f"wrote {len(elements_needed)} atomic-mu inputs to {out_dir}")

    elif args.kind == "pristine":
        # For each unique host appearing in the 60 candidates, build the
        # pristine supercell from the candidate's cell + numbers minus
        # the dopant atom (defect_atom_index).
        cand_pkl = ROOT / "data" / "processed" / "candidates_prospective.pkl"
        cands = pickle.load(open(cand_pkl, "rb"))
        seen_hosts = set()
        for c in cands:
            cid = c.get("candidate_id", -1)
            host = c.get("metadata", {}).get("host", "?")
            if host in seen_hosts:
                continue
            seen_hosts.add(host)
            didx = int(c.get("defect_atom_index", -1))
            if didx < 0:
                # heuristic: last atom
                didx = len(c["numbers"]) - 1
            keep = [i for i in range(len(c["numbers"])) if i != didx]
            atoms = Atoms(
                numbers=[c["numbers"][i] for i in keep],
                positions=[c["positions"][i] for i in keep],
                cell=c["cell"], pbc=True,
            )
            prefix = f"pristine_{host}"
            inp = build_input(atoms, prefix=prefix,
                              pseudo_dir=args.pseudo_dir, outdir=args.outdir,
                              kpts=(2, 2, 1), spin=False, calc="scf")
            with open(out_dir / f"{prefix}.in", "w") as f:
                f.write(inp)
        print(f"wrote {len(seen_hosts)} pristine-host inputs to {out_dir}")


if __name__ == "__main__":
    main()
