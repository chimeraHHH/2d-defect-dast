"""Read imp2d.db, filter, build graphs, and dump pickle.

Usage:
    python scripts/prepare_dataset.py [--db PATH] [--out PATH] [--cutoff 5.0]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from ase.db import connect

# project import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import build_graph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "data/raw/imp2d.db"))
    parser.add_argument(
        "--out",
        default=str(ROOT / "data/processed/cleaned_dataset.pkl"),
    )
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max", type=int, default=0, help="optional cap for debug")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {db_path}")
    db = connect(str(db_path))
    total = db.count()
    print(f"Total rows: {total}")

    samples = []
    skipped = {
        "not_converged": 0,
        "missing_eform": 0,
        "outlier": 0,
        "graph_error": 0,
    }

    t0 = time.time()
    for n_seen, row in enumerate(db.select(), start=1):
        if not row.get("converged"):
            skipped["not_converged"] += 1
            continue
        eform = row.get("eform")
        if eform is None or np.isnan(eform):
            skipped["missing_eform"] += 1
            continue
        if abs(eform) > 20.0:
            skipped["outlier"] += 1
            continue

        atoms = row.toatoms()
        try:
            g = build_graph(atoms, cutoff=args.cutoff)
        except Exception as exc:  # pragma: no cover - defensive
            skipped["graph_error"] += 1
            print(f"  graph error on row {row.id}: {exc}", file=sys.stderr)
            continue

        sample = {
            "id": int(row.id),
            "unique_id": str(row.unique_id),
            "numbers": g["numbers"],
            "positions": g["positions"],
            "cell": g["cell"],
            "edge_index": g["edge_index"],
            "edge_dist": g["edge_dist"],
            "edge_offset": g["edge_offset"],
            "triplet_index": g["triplet_index"],
            "angles": g["angles"],
            "dist_matrix": g["dist_matrix"],
            "target": float(eform),
            "metadata": {
                "host": str(row.get("host", "")),
                "dopant": str(row.get("dopant", "")),
                "site": str(row.get("site", "")),
                "defecttype": str(row.get("defecttype", "")),
                "natoms": int(row.natoms),
                "spacegroup": str(row.get("host_spacegroup", "")),
                "supercell": str(row.get("supercell", "")),
            },
        }
        samples.append(sample)
        if args.max and len(samples) >= args.max:
            break
        if n_seen % 1000 == 0:
            dt = time.time() - t0
            print(
                f"  scanned {n_seen}/{total}, kept {len(samples)}, "
                f"speed {n_seen / max(dt, 1e-6):.1f} rows/s"
            )

    dt = time.time() - t0
    print(f"\nFinished in {dt:.1f}s. Kept {len(samples)} samples; skipped {skipped}")

    with open(out_path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
