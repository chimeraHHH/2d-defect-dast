"""Rebuild the frozen IMP2D graph fields for G1 without changing row identity.

The source pickle remains the authoritative 10,641-row container and order.
The raw ASE database is used only to recover and verify each row's PBC flags.
No filtering, canonical compaction, target rewrite, or in-place overwrite is
permitted.  The output is a new versioned pickle plus a hash-bound receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.db import connect

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import (  # noqa: E402
    GRAPH_BUILDER_VERSION,
    MAX_TRIPLETS_PER_CENTRE,
    build_graph,
)


SOURCE_DATA_SHA256 = "1d59cc818d81252d49da525c6d77e2da8549ceb604d54af953d1caa4fb974a9b"
RAW_DB_SHA256 = "3a71db999b477112da248dcf762c4384e455689953679d58b3d71a91e7148fc4"
EXPECTED_CONTAINER_ROWS = 10_641
EXPECTED_CANONICAL_ROWS = 10_224
EXPECTED_EXCLUDED_ROWS = 417
PROTOCOL_SAMPLES_SHA256 = (
    "03e8bb9fc68f22aecba336dbcb40a35eceab51e7b6497d4af7bd69d6042f1168"
)
GRAPH_FIELDS = (
    "numbers", "positions", "cell", "edge_index", "edge_dist", "edge_offset",
    "triplet_index", "angles", "dist_matrix",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_snapshot() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    remote_refs = subprocess.run(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.splitlines()
    remote_refs = sorted(ref.strip() for ref in remote_refs if ref.strip())
    if not commit or status or not remote_refs:
        raise SystemExit(
            "G1 rebuild requires a clean commit reachable from a fetched remote ref"
        )
    return {"commit": commit, "dirty": False, "remote_refs": remote_refs}


def strict_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


EDGE_DIST_ATOL = 1.0e-5


def graph_edge_records(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return canonically ordered edge topology and float64 edge distances.

    Edges are lexsorted by (source, target, distance) so that stored and
    recomputed edge sets can be compared positionally.  Distances are kept as
    float64 and compared with :data:`EDGE_DIST_ATOL` rather than rounded
    equality: the stored ``edge_dist`` is float32, so a decimal-rounding
    comparison flips on values that sit within one float32 quantum of a
    rounding boundary even when the physical difference is below 1e-6 A.
    """
    edges = np.asarray(sample["edge_index"], dtype=np.int64).reshape(2, -1)
    dists = np.asarray(sample["edge_dist"], dtype=np.float64)
    order = np.lexsort((dists, edges[1], edges[0]))
    return edges[:, order], dists[order]


def verify_raw_structure(
    sample: dict[str, Any], raw_atoms: Atoms, *, atol: float,
) -> None:
    np.testing.assert_array_equal(
        np.asarray(sample["numbers"], dtype=np.int64),
        raw_atoms.get_atomic_numbers().astype(np.int64),
    )
    np.testing.assert_allclose(
        np.asarray(sample["positions"], dtype=np.float64),
        raw_atoms.get_positions(), rtol=0.0, atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(sample["cell"], dtype=np.float64),
        np.asarray(raw_atoms.get_cell()), rtol=0.0, atol=atol,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--raw-db", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--protocol-samples", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", default=SOURCE_DATA_SHA256)
    parser.add_argument("--expected-raw-db-sha256", default=RAW_DB_SHA256)
    parser.add_argument("--structure-atol", type=float, default=2.0e-5)
    parser.add_argument("--cutoff", type=float, default=5.0)
    args = parser.parse_args()

    git = git_snapshot()
    if socket.gethostname() != "WHUServer-L40S":
        raise SystemExit("formal G1 IMP2D rebuild requires WHUServer-L40S")
    source = args.source_data.expanduser().resolve()
    raw_db = args.raw_db.expanduser().resolve()
    output = args.output_data.expanduser().resolve()
    receipt = args.output_receipt.expanduser().resolve()
    protocol_samples = args.protocol_samples.expanduser().resolve()
    resolved_paths = (source, raw_db, protocol_samples, output, receipt)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise SystemExit("all G1 input/output paths must resolve to distinct files")
    for path in (source, raw_db, protocol_samples):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output == source:
        raise SystemExit("refusing to overwrite the frozen source dataset")
    if output.exists() or receipt.exists():
        raise FileExistsError("versioned G1 output or receipt already exists")
    expected_protocol = (
        ROOT / "artifacts/prm_protocol_v2/samples.csv"
    ).resolve()
    if protocol_samples != expected_protocol:
        raise SystemExit("formal G1 rebuild requires the canonical protocol table")
    if (
        args.expected_source_sha256 != SOURCE_DATA_SHA256
        or args.expected_raw_db_sha256 != RAW_DB_SHA256
        or sha256_file(protocol_samples) != PROTOCOL_SAMPLES_SHA256
    ):
        raise SystemExit("formal G1 frozen source/protocol hash contract mismatch")
    if sha256_file(source) != args.expected_source_sha256:
        raise SystemExit("frozen source dataset SHA256 mismatch")
    if sha256_file(raw_db) != args.expected_raw_db_sha256:
        raise SystemExit("raw IMP2D database SHA256 mismatch")
    if args.cutoff != 5.0 or args.structure_atol != 2.0e-5:
        raise SystemExit(
            "formal G1 rebuild requires cutoff=5.0 and structure_atol=2e-5"
        )

    with source.open("rb") as handle:
        source_blob = pickle.load(handle)
    if isinstance(source_blob, dict) and "data" in source_blob:
        samples = source_blob["data"]
    elif isinstance(source_blob, list):
        samples = source_blob
    else:
        raise TypeError("unsupported source dataset container")
    if len(samples) != EXPECTED_CONTAINER_ROWS:
        raise ValueError(
            f"expected {EXPECTED_CONTAINER_ROWS} source rows, found {len(samples)}"
        )

    # The protocol remains the immutable membership map.  It must retain the
    # full 10,641-index container with exactly 10,224 canonical rows.
    import csv
    with protocol_samples.open(newline="") as handle:
        protocol_rows = list(csv.DictReader(handle))
    retained = sum(
        str(row.get("canonical_retained", "")).lower() == "true"
        for row in protocol_rows
    )
    if (
        len(protocol_rows) != EXPECTED_CONTAINER_ROWS
        or retained != EXPECTED_CANONICAL_ROWS
        or len(protocol_rows) - retained != EXPECTED_EXCLUDED_ROWS
    ):
        raise ValueError("protocol canonical/excluded membership contract mismatch")

    db = connect(str(raw_db))
    rebuilt_samples = []
    pbc_patterns: Counter[str] = Counter()
    total_edges = 0
    total_triplets = 0
    capped_centres = 0
    changed_mic_samples = 0
    max_mic_delta = 0.0
    max_edge_dist_delta = 0.0
    t0 = time.time()

    for index, source_sample in enumerate(samples):
        sample = dict(source_sample)
        row_id = int(sample["id"])
        raw_row = db.get(id=row_id)
        if int(raw_row.id) != row_id:
            raise ValueError(f"raw row ID mismatch at source index {index}")
        raw_atoms = raw_row.toatoms()
        verify_raw_structure(sample, raw_atoms, atol=args.structure_atol)
        pbc = np.asarray(raw_atoms.get_pbc(), dtype=bool)
        pbc_patterns["".join("1" if value else "0" for value in pbc)] += 1

        atoms = Atoms(
            numbers=np.asarray(sample["numbers"], dtype=np.int64),
            positions=np.asarray(sample["positions"], dtype=np.float64),
            cell=np.asarray(sample["cell"], dtype=np.float64),
            pbc=pbc,
        )
        graph = build_graph(atoms, cutoff=args.cutoff)
        old_edges, old_dists = graph_edge_records(sample)
        new_edges, new_dists = graph_edge_records(graph)
        if old_edges.shape != new_edges.shape or not np.array_equal(
            old_edges, new_edges
        ):
            raise ValueError(
                f"edge topology changed while recovering PBC at index {index}"
            )
        edge_delta = float(
            np.max(np.abs(old_dists - new_dists), initial=0.0)
        )
        if edge_delta > EDGE_DIST_ATOL:
            raise ValueError(
                f"edge distances changed by {edge_delta} (> {EDGE_DIST_ATOL} A) "
                f"while recovering PBC at index {index}"
            )
        max_edge_dist_delta = max(max_edge_dist_delta, edge_delta)
        old_dist = np.asarray(sample["dist_matrix"], dtype=np.float64)
        delta = float(np.max(np.abs(old_dist - graph["dist_matrix"])))
        max_mic_delta = max(max_mic_delta, delta)
        changed_mic_samples += int(delta > 1.0e-6)

        for field in GRAPH_FIELDS:
            sample[field] = graph[field]
        sample["pbc"] = pbc
        sample["graph_builder_version"] = GRAPH_BUILDER_VERSION
        rebuilt_samples.append(sample)

        total_edges += int(graph["edge_index"].shape[1])
        total_triplets += int(len(graph["angles"]))
        degrees = np.bincount(
            graph["edge_index"][0], minlength=len(graph["numbers"])
        )
        capped_centres += int(np.sum(degrees * (degrees - 1) > MAX_TRIPLETS_PER_CENTRE))
        if (index + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(
                f"rebuilt {index + 1}/{len(samples)} rows "
                f"({(index + 1) / max(elapsed, 1e-9):.1f} rows/s)",
                flush=True,
            )

    if isinstance(source_blob, dict):
        output_blob = dict(source_blob)
        output_blob["data"] = rebuilt_samples
        output_blob["graph_builder"] = {
            "version": GRAPH_BUILDER_VERSION,
            "source_data_sha256": args.expected_source_sha256,
        }
    else:
        output_blob = rebuilt_samples

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=output.name + ".", suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(output_blob, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    output_sha256 = sha256_file(output)
    payload = {
        "schema_version": "prm_g1_repaired_dataset_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_profile": "WHUServer-L40S",
        "git": git,
        "inputs": {
            "dataset": {"alias": "PRM_DATASET_PATH", "sha256": args.expected_source_sha256},
            "raw_db": {"alias": "PRM_RAW_DB_PATH", "sha256": args.expected_raw_db_sha256},
            "protocol_samples": {
                "repository_path": "artifacts/prm_protocol_v2/samples.csv",
                "sha256": sha256_file(protocol_samples),
            },
        },
        "output": {
            "alias": "PRM_REPAIRED_DATASET_PATH",
            "sha256": output_sha256,
            "size_bytes": output.stat().st_size,
        },
        "container": {
            "rows": len(rebuilt_samples),
            "canonical_rows": retained,
            "excluded_rows": len(protocol_rows) - retained,
            "order_preserved": True,
            "non_graph_fields_preserved": True,
        },
        "graph_builder": {
            "version": GRAPH_BUILDER_VERSION,
            "source_repository_path": "src/graph.py",
            "source_sha256": sha256_file(ROOT / "src/graph.py"),
            "cutoff_A": args.cutoff,
            "exact_mic_backend": "ase.geometry.find_mic",
            "pbc_source": "hash-verified raw IMP2D row after structure identity check",
            "structure_identity_atol": args.structure_atol,
            "triplet_cap_per_centre": MAX_TRIPLETS_PER_CENTRE,
            "triplet_selection": (
                "complete ordered-angle multiset; all retained for n<=32; "
                "midpoint order statistics floor((2q+1)n/(2*32)) for n>32"
            ),
        },
        "audit": {
            "pbc_patterns": dict(sorted(pbc_patterns.items())),
            "total_edges": total_edges,
            "total_triplets": total_triplets,
            "capped_centres": capped_centres,
            "samples_with_legacy_mic_delta_gt_1e-6": changed_mic_samples,
            "max_legacy_to_exact_mic_delta_A": max_mic_delta,
            "max_edge_dist_delta_A": max_edge_dist_delta,
            "edge_dist_atol_A": EDGE_DIST_ATOL,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "elapsed_seconds": time.time() - t0,
    }
    receipt.write_text(strict_json(payload))
    print(strict_json({"output_sha256": output_sha256, "receipt": receipt.name}))


if __name__ == "__main__":
    main()
