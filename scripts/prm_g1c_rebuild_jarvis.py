"""Rebuild graph fields in the frozen 19,902-row JARVIS source-task pickle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import (  # noqa: E402
    GRAPH_BUILDER_VERSION,
    MAX_TRIPLETS_PER_CENTRE,
    build_graph,
)


SOURCE_SHA256 = "41284603e6eeb6920fe132975570e657e69601a414257bb48d2196d7389369fa"
EXPECTED_ROWS = 19_902
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


def clean_commit() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    remote_refs = subprocess.run(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.splitlines()
    remote_refs = sorted(ref.strip() for ref in remote_refs if ref.strip())
    if not commit or dirty or not remote_refs:
        raise SystemExit(
            "G1C JARVIS rebuild requires a clean commit reachable from a fetched remote ref"
        )
    return {"commit": commit, "dirty": False, "remote_refs": remote_refs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", default=SOURCE_SHA256)
    parser.add_argument("--cutoff", type=float, default=5.0)
    args = parser.parse_args()

    git = clean_commit()
    if socket.gethostname() != "WHUServer-L40S":
        raise SystemExit("formal G1C JARVIS rebuild requires WHUServer-L40S")
    source = args.source_data.expanduser().resolve()
    output = args.output_data.expanduser().resolve()
    receipt = args.output_receipt.expanduser().resolve()
    if len({source, output, receipt}) != 3:
        raise SystemExit("G1C input/output paths must resolve to distinct files")
    if args.expected_source_sha256 != SOURCE_SHA256:
        raise SystemExit("formal G1C source hash is frozen and cannot be overridden")
    if not source.is_file() or sha256_file(source) != SOURCE_SHA256:
        raise SystemExit("frozen JARVIS dataset SHA256 mismatch")
    if output.exists() or receipt.exists():
        raise FileExistsError("versioned G1C output or receipt already exists")
    if args.cutoff != 5.0:
        raise SystemExit("formal G1C rebuild requires cutoff=5.0")

    with source.open("rb") as handle:
        blob = pickle.load(handle)
    if not isinstance(blob, list) or len(blob) != EXPECTED_ROWS:
        raise ValueError("frozen JARVIS container must be a 19,902-row list")

    rebuilt = []
    old_edges = old_triplets = new_edges = new_triplets = capped_centres = 0
    t0 = time.time()
    for index, original in enumerate(blob):
        sample = dict(original)
        numbers = np.asarray(sample["numbers"], dtype=np.int64)
        positions = np.asarray(sample["positions"], dtype=np.float64)
        cell = np.asarray(sample["cell"], dtype=np.float64)
        if (
            int(sample.get("id", index)) != index
            or len(numbers) < 2
            or positions.shape != (len(numbers), 3)
            or cell.shape != (3, 3)
            or not np.isfinite(float(sample["target"]))
        ):
            raise ValueError(f"JARVIS source identity/schema mismatch at index {index}")
        graph = build_graph(
            Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True),
            cutoff=args.cutoff,
        )
        old_edges += int(np.asarray(sample["edge_index"]).shape[1])
        old_triplets += int(len(sample["angles"]))
        new_edges += int(graph["edge_index"].shape[1])
        new_triplets += int(len(graph["angles"]))
        degrees = np.bincount(graph["edge_index"][0], minlength=len(numbers))
        capped_centres += int(np.sum(degrees * (degrees - 1) > MAX_TRIPLETS_PER_CENTRE))
        for field in GRAPH_FIELDS:
            sample[field] = graph[field]
        sample["pbc"] = np.asarray([True, True, True], dtype=bool)
        sample["graph_builder_version"] = GRAPH_BUILDER_VERSION
        rebuilt.append(sample)
        if (index + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(
                f"rebuilt {index + 1}/{len(blob)} JARVIS rows "
                f"({(index + 1) / max(elapsed, 1e-9):.1f} rows/s)",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=output.name + ".", suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            pickle.dump(rebuilt, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    payload: dict[str, Any] = {
        "schema_version": "prm_g1c_jarvis_dataset_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_profile": "WHUServer-L40S",
        "git": git,
        "input": {"alias": "PRM_JARVIS_DATA_PATH", "sha256": args.expected_source_sha256},
        "output": {
            "alias": "PRM_CORRECTED_JARVIS_DATA_PATH",
            "sha256": sha256_file(output), "size_bytes": output.stat().st_size,
        },
        "container": {
            "rows": len(rebuilt), "order_preserved": True,
            "targets_and_non_graph_fields_preserved": True,
        },
        "graph_builder": {
            "version": GRAPH_BUILDER_VERSION,
            "source_repository_path": "src/graph.py",
            "source_sha256": sha256_file(ROOT / "src/graph.py"),
            "cutoff_A": args.cutoff,
            "pbc": [True, True, True],
            "exact_mic_backend": "ase.geometry.find_mic",
            "triplet_cap_per_centre": MAX_TRIPLETS_PER_CENTRE,
            "triplet_centre_column": 1,
        },
        "audit": {
            "old_edges": old_edges, "new_edges": new_edges,
            "old_triplets": old_triplets, "new_triplets": new_triplets,
            "capped_centres": capped_centres,
        },
        "elapsed_seconds": time.time() - t0,
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output_sha256": payload["output"]["sha256"],
        "receipt_sha256": sha256_file(receipt),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
