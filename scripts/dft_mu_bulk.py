"""C18e: Bulk-elemental GPAW PBE energies for the 8 dopant elements.

Replaces the coarse isolated-atom mu values with the standard bulk
elemental reference used by most defect-formation databases (incl.
IMP2D). For each dopant element, build the standard bulk crystal via
ase.build.bulk() and compute single-point GPAW PW PBE.

Outputs:
- results/dft_mu_bulk.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ase.build import bulk  # noqa: E402

DFT_VAL = ROOT / "results" / "dft_validation.json"
RESULTS = ROOT / "results"
LOG_DIR = ROOT / "logs" / "dft_mu_bulk"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Standard bulk crystal structures.
# - Pb / Ir / Rh: fcc (cubic close-packed)
# - In: body-centred tetragonal (ase bulk: 'bct' wants a, c)
# - Bi: rhombohedral A7 (alpha-Bi)
# - La: dhcp at RT, but use fcc (a~5.30) as the MP-defaults reference
# - Ru: hcp (a, c/a~1.58)
# - S: complex S8 ring (alpha-S, 16 atoms/cell) — too expensive for our
#   scope here; we keep the isolated-atom mu (-1.006 eV) for S and note
#   the asymmetry in the analysis.
BULK_SPEC = {
    "Pb": ("fcc",          dict(a=4.95)),
    "Bi": ("rhombohedral", dict(a=4.75, alpha=57.35)),
    "In": ("bct",          dict(a=3.25, c=4.95)),
    "Ir": ("fcc",          dict(a=3.84)),
    "La": ("fcc",          dict(a=5.30)),
    "Rh": ("fcc",          dict(a=3.80)),
    "Ru": ("hcp",          dict(a=2.71, covera=1.58)),
    # "S": skipped — see note above
}


def make_bulk(element):
    spec = BULK_SPEC.get(element)
    if spec is None:
        return None
    crys, kw = spec
    if crys in ("fcc", "bcc"):
        return bulk(element, crys, cubic=True, **kw)
    return bulk(element, crys, **kw)


def run_bulk(element, atoms):
    from gpaw import GPAW, PW
    n = len(atoms)
    atoms.calc = GPAW(
        mode=PW(300),
        kpts=(6, 6, 6),  # converges total E to <50 meV/atom for our needs
        xc="PBE",
        txt=str(LOG_DIR / f"bulk_{element}.log"),
        symmetry="off",  # full symmetry breaks for some cell types
        convergence={"energy": 1e-3, "density": 1e-3, "eigenstates": 1e-4},
        occupations={"name": "fermi-dirac", "width": 0.05},
        maxiter=200,
    )
    t0 = time.time()
    try:
        e = float(atoms.get_potential_energy())
        return {"ok": True, "element": element, "n_atoms": n,
                "e_total_eV": e, "e_per_atom_eV": e / n,
                "wall_sec": time.time() - t0}
    except Exception as exc:
        return {"ok": False, "element": element, "n_atoms": n,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_sec": time.time() - t0}


def main():
    with open(DFT_VAL) as f:
        dv = json.load(f)["results"]
    elements = sorted({r["dopant"] for r in dv})
    print(f"Elements: {elements}\n")

    # resume: load existing successful results so we don't recompute
    out_path = RESULTS / "dft_mu_bulk.json"
    existing = {}
    if out_path.exists():
        prev = json.loads(out_path.read_text()).get("results", [])
        for r in prev:
            if r.get("ok"):
                existing[r["element"]] = r
        print(f"Resume: {len(existing)} elements already done: "
              f"{list(existing)}\n")

    results = list(existing.values())
    t_start = time.time()
    for el in elements:
        if el in existing:
            print(f"  {el}: skip (already in JSON)")
            continue
        atoms = make_bulk(el)
        if atoms is None:
            print(f"\n  {el}: SKIP (no bulk spec — falls back to atomic mu)")
            results.append({"ok": False, "element": el, "skipped": True,
                            "note": "no bulk spec; analyzer falls back to atomic mu"})
            (RESULTS / "dft_mu_bulk.json").write_text(json.dumps(
                {"results": results,
                 "wall_min": (time.time() - t_start) / 60,
                 "bulk_spec": {k: list(v) for k, v in BULK_SPEC.items()}},
                indent=2))
            continue
        print(f"\n  bulk {el} ({len(atoms)} atoms)...", flush=True)
        r = run_bulk(el, atoms)
        print(f"    -> {r}", flush=True)
        results.append(r)
        out = {"results": results, "wall_min": (time.time() - t_start) / 60,
               "bulk_spec": {k: list(v) for k, v in BULK_SPEC.items()}}
        (RESULTS / "dft_mu_bulk.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== Done. {(time.time()-t_start)/60:.1f} min ===")
    print(f"  successful: {sum(1 for r in results if r.get('ok'))}/{len(results)}")


if __name__ == "__main__":
    main()
