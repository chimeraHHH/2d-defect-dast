"""G1A legacy-checkpoint permutation and exact-MIC diagnostic.

This command deliberately embeds the frozen legacy graph path (first 32
ordered neighbour pairs plus component-wise fractional wrapping).  It never
uses the repaired :func:`src.graph.build_graph` for the old-checkpoint
permutation test.  A second intervention changes only the dense radial matrix
to exact MIC while leaving the frozen legacy local graph untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import re
import socket
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from ase.neighborlist import neighbor_list
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prm_materials_analysis import analyse_preferences  # noqa: E402
from src.dataset import (  # noqa: E402
    DEFECT_TYPE_MAP,
    CrystalGraphDataset,
    collate_fn,
)
from src.graph import _pbc_distance_matrix  # noqa: E402
from src.models.crystal_v2 import (  # noqa: E402
    ENV_ZERO_NEIGHBOR_LEGACY,
    CrystalTransformerV2,
)


SOURCE_DATA_SHA256 = "1d59cc818d81252d49da525c6d77e2da8549ceb604d54af953d1caa4fb974a9b"
RAW_DB_SHA256 = "3a71db999b477112da248dcf762c4384e455689953679d58b3d71a91e7148fc4"
CT_UAE_SHA256 = "ac77b2720b7bb8a3b290d6bfbae482857962d55daa7fc3486f2c7f44c41c60dc"
LEGACY_PRETRAINED_SHA256 = "5dd085b1393acee4db83422241102595a5d011c18a880a44c09947b29274a3bc"
LEGACY_TRAINING_COMMIT = "6c401374baa9ac22b5fa4353d467366ac7cc222f"
EXCLUDED_GPU_UUID = "GPU-33963073-e04e-1698-5d02-1a16d098c931"
LEGACY_FIRST32 = "legacy_first32_component_wrap_v1"
N_PERMUTATIONS = 16
PERMUTATION_NAMES = tuple(
    [f"random_{index:02d}" for index in range(8)]
    + [
        "reverse", "atomic_number_asc", "atomic_number_desc",
        "defect_distance_asc", "defect_distance_desc", "cartesian_lex_asc",
        "cartesian_lex_desc", "even_then_odd",
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


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
            "G1A requires a clean commit reachable from a fetched remote ref"
        )
    return {"commit": commit, "dirty": False, "remote_refs": remote_refs}


def gpu_identity() -> dict[str, str]:
    observed_host = socket.gethostname()
    if observed_host != "WHUServer-L40S":
        raise SystemExit(f"G1A host mismatch: {observed_host!r}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"GPU-[0-9a-fA-F-]+", visible) is None:
        raise SystemExit("G1 requires one full GPU UUID in CUDA_VISIBLE_DEVICES")
    if visible == EXCLUDED_GPU_UUID:
        raise SystemExit("physical GPU 2 UUID is excluded from G1")
    result = subprocess.run(
        [
            "/usr/bin/nvidia-smi", "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=True,
    )
    matches = []
    for line in result.stdout.splitlines():
        index, uuid, name = (value.strip() for value in line.split(",", maxsplit=2))
        if uuid == visible:
            matches.append({"uuid": uuid, "name": name})
    if len(matches) != 1:
        raise SystemExit("assigned GPU UUID was not uniquely resolved by nvidia-smi")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("CUDA UUID isolation did not expose exactly one PyTorch device")
    if torch.cuda.get_device_name(0) != matches[0]["name"]:
        raise SystemExit("PyTorch CUDA device does not match the assigned GPU UUID")
    return {"host_profile": observed_host, **matches[0]}


def legacy_componentwise_distances(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    try:
        inverse = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return np.linalg.norm(diff, axis=-1)
    fractional = diff @ inverse
    fractional -= np.round(fractional)
    return np.linalg.norm(fractional @ cell, axis=-1).astype(np.float32)


def legacy_build_graph(atoms: Atoms, cutoff: float = 5.0) -> dict[str, np.ndarray]:
    """Frozen reproduction of pre-G1 ``src.graph.build_graph``."""
    i, j, d, displacements, shifts = neighbor_list("ijdDS", atoms, cutoff=cutoff)
    triplets: list[tuple[int, int, int]] = []
    angles: list[float] = []
    if len(i):
        order = np.argsort(i, kind="stable")
        sorted_i, sorted_j = i[order], j[order]
        sorted_d = displacements[order]
        centres, starts = np.unique(sorted_i, return_index=True)
        stops = np.append(starts[1:], len(sorted_i))
        for centre, start, stop in zip(centres, starts, stops):
            degree = int(stop - start)
            if degree < 2:
                continue
            local_j = sorted_j[start:stop]
            vectors = sorted_d[start:stop]
            norms = np.linalg.norm(vectors, axis=1) + 1.0e-12
            count = 0
            for left in range(degree):
                for right in range(degree):
                    if left == right:
                        continue
                    cosine = float(
                        np.dot(vectors[left], vectors[right])
                        / (norms[left] * norms[right])
                    )
                    triplets.append(
                        (int(local_j[left]), int(centre), int(local_j[right]))
                    )
                    angles.append(math.acos(max(-1.0, min(1.0, cosine))))
                    count += 1
                    if count >= 32:
                        break
                if count >= 32:
                    break
    return {
        "numbers": atoms.get_atomic_numbers().astype(np.int64),
        "positions": atoms.get_positions().astype(np.float32),
        "cell": np.asarray(atoms.get_cell()).astype(np.float32),
        "edge_index": np.vstack([i, j]).astype(np.int64),
        "edge_dist": d.astype(np.float32),
        "edge_offset": shifts.astype(np.float32),
        "triplet_index": (
            np.asarray(triplets, dtype=np.int64)
            if triplets else np.empty((0, 3), dtype=np.int64)
        ),
        "angles": (
            np.asarray(angles, dtype=np.float32)
            if angles else np.empty((0,), dtype=np.float32)
        ),
        "dist_matrix": legacy_componentwise_distances(
            atoms.get_positions().astype(np.float32),
            np.asarray(atoms.get_cell()).astype(np.float32),
        ),
    }


def assert_legacy_identity_graph(
    archived: dict[str, Any], rebuilt: dict[str, np.ndarray], index: int,
) -> None:
    for field in ("numbers", "edge_index", "triplet_index"):
        if not np.array_equal(np.asarray(archived[field]), rebuilt[field]):
            raise ValueError(f"legacy identity {field} mismatch at sample {index}")
    for field in (
        "positions", "cell", "edge_dist", "edge_offset", "angles", "dist_matrix",
    ):
        np.testing.assert_allclose(
            np.asarray(archived[field]), rebuilt[field], rtol=0.0, atol=1.0e-6,
            err_msg=f"legacy identity {field} mismatch at sample {index}",
        )


def permutation_for(sample: dict[str, Any], variant: int) -> np.ndarray:
    n_atoms = len(sample["numbers"])
    if variant < 8:
        seed = (20260810 + 1_000_003 * int(sample["id"]) + 97_409 * variant) % (2**63 - 1)
        return np.random.default_rng(seed).permutation(n_atoms)
    numbers = np.asarray(sample["numbers"])
    positions = np.asarray(sample["positions"])
    defect = np.flatnonzero(np.asarray(sample["defect_mask"], dtype=bool))
    defect_index = int(defect[0]) if len(defect) else 0
    distances = np.linalg.norm(positions - positions[defect_index], axis=1)
    index = variant - 8
    if index == 0:
        return np.arange(n_atoms - 1, -1, -1)
    if index == 1:
        return np.lexsort((np.arange(n_atoms), numbers))
    if index == 2:
        return np.lexsort((np.arange(n_atoms), -numbers))
    if index == 3:
        return np.lexsort((np.arange(n_atoms), distances))
    if index == 4:
        return np.lexsort((np.arange(n_atoms), -distances))
    lex = np.lexsort((np.arange(n_atoms), positions[:, 2], positions[:, 1], positions[:, 0]))
    if index == 5:
        return lex
    if index == 6:
        return lex[::-1]
    return np.concatenate([np.arange(0, n_atoms, 2), np.arange(1, n_atoms, 2)])


def custom_item(
    base: CrystalGraphDataset,
    index: int,
    graph: dict[str, np.ndarray],
    defect_mask: np.ndarray,
) -> dict[str, Any]:
    numbers = torch.from_numpy(graph["numbers"]).long()
    metadata = base.data[index].get("metadata", {})
    return {
        "sample_index": torch.tensor(index, dtype=torch.long),
        "x": base.atom_features[numbers],
        "atomic_numbers": numbers,
        "defect_mask": torch.from_numpy(defect_mask.astype(np.int64)),
        "defect_type": torch.tensor(
            DEFECT_TYPE_MAP.get(str(metadata.get("defecttype", "vacancy")), 0),
            dtype=torch.long,
        ),
        "edge_index": torch.from_numpy(graph["edge_index"]),
        "edge_dist": torch.from_numpy(graph["edge_dist"]),
        "edge_offset": torch.from_numpy(graph["edge_offset"]).float(),
        "triplet_index": torch.from_numpy(graph["triplet_index"]),
        "angles": torch.from_numpy(graph["angles"]),
        "dist_matrix": torch.from_numpy(graph["dist_matrix"]),
        "positions": torch.from_numpy(graph["positions"]),
        "cell": torch.from_numpy(graph["cell"]),
        "target": torch.tensor(float(base.data[index]["target"]), dtype=torch.float32),
        "num_atoms": len(numbers),
    }


class DiagnosticDataset(Dataset):
    def __init__(
        self,
        base: CrystalGraphDataset,
        indices: Sequence[int],
        pbc_by_index: dict[int, np.ndarray],
        raw_geometry_by_index: dict[int, dict[str, np.ndarray]],
        *,
        permutation_variant: int | None = None,
        exact_distances: dict[int, np.ndarray] | None = None,
    ) -> None:
        self.base = base
        self.indices = [int(index) for index in indices]
        self.pbc_by_index = pbc_by_index
        self.raw_geometry_by_index = raw_geometry_by_index
        self.permutation_variant = permutation_variant
        self.exact_distances = exact_distances

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        sample = self.base.data[index]
        defect_mask = np.asarray(sample["defect_mask"], dtype=np.int64)
        if self.permutation_variant is not None:
            raw = self.raw_geometry_by_index[index]
            permutation = (
                np.arange(len(raw["numbers"]))
                if self.permutation_variant < 0
                else permutation_for({**sample, **raw}, self.permutation_variant)
            )
            atoms = Atoms(
                numbers=raw["numbers"][permutation],
                positions=raw["positions"][permutation],
                cell=raw["cell"],
                pbc=self.pbc_by_index[index],
            )
            graph = legacy_build_graph(atoms)
            defect_mask = defect_mask[permutation]
        else:
            graph = {key: sample[key] for key in (
                "numbers", "positions", "cell", "edge_index", "edge_dist",
                "edge_offset", "triplet_index", "angles", "dist_matrix",
            )}
            if self.exact_distances is not None:
                graph = dict(graph)
                graph["dist_matrix"] = self.exact_distances[index]
        return custom_item(self.base, index, graph, defect_mask)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def denormalize(values: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    output = values * float(state["std"]) + float(state["mean"])
    if state.get("transform", "none") == "log":
        output = torch.sign(output) * torch.expm1(torch.abs(output))
    return output


def infer(
    model: torch.nn.Module,
    dataset: Dataset,
    normalizer: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    indices, predictions, targets = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch)
            output = output[0] if isinstance(output, tuple) else output
            indices.append(batch["sample_index"].detach().cpu().numpy())
            predictions.append(denormalize(output, normalizer).detach().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())
    return np.concatenate(indices), np.concatenate(predictions), np.concatenate(targets)


def load_archived_predictions(
    path: Path, expected_split_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "schema_version", "split_id", "split", "indices", "preds", "targets",
        }
        if set(archive.files) != expected:
            raise ValueError(f"legacy prediction schema fields mismatch: {path.name}")
        if str(archive["schema_version"].item()) != "prm_predictions_v1":
            raise ValueError("legacy prediction schema version mismatch")
        if str(archive["split_id"].item()) != expected_split_id:
            raise ValueError("legacy prediction split ID mismatch")
        if str(archive["split"].item()) != "test":
            raise ValueError("legacy prediction partition mismatch")
        indices = np.asarray(archive["indices"])
        predictions = np.asarray(archive["preds"], dtype=float)
        targets = np.asarray(archive["targets"], dtype=float)
        if (
            not np.issubdtype(indices.dtype, np.integer)
            or indices.ndim != 1
            or predictions.shape != indices.shape
            or targets.shape != indices.shape
            or len(np.unique(indices)) != len(indices)
            or not np.isfinite(predictions).all()
            or not np.isfinite(targets).all()
        ):
            raise ValueError("legacy predictions are not finite, aligned, and unique")
        return (
            indices.astype(np.int64, copy=False), predictions, targets,
        )


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def decision_maps(
    indices: np.ndarray,
    predictions: np.ndarray,
    sample_rows: dict[int, dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    rows = []
    for index, prediction in zip(indices, predictions):
        row = dict(sample_rows[int(index)])
        row["prediction_eV"] = float(prediction)
        rows.append(row)
    pair_rows, site_rows = analyse_preferences(rows)
    return (
        {(str(row["host"]), str(row["dopant"])): row for row in pair_rows},
        {
            (str(row["host"]), str(row["dopant"]), str(row["defecttype"])): row
            for row in site_rows
        },
    )


def decision_delta(
    reference: tuple[dict, dict], candidate: tuple[dict, dict]
) -> dict[str, Any]:
    ref_pairs, ref_sites = reference
    out_pairs, out_sites = candidate
    if ref_pairs.keys() != out_pairs.keys() or ref_sites.keys() != out_sites.keys():
        raise ValueError("screening decision coverage changed")
    pair_regret = np.asarray([
        float(out_pairs[key]["global_screening_regret_eV"])
        - float(ref_pairs[key]["global_screening_regret_eV"])
        for key in ref_pairs
    ])
    site_regret = np.asarray([
        float(out_sites[key]["screening_regret_eV"])
        - float(ref_sites[key]["screening_regret_eV"])
        for key in ref_sites
    ])
    return {
        "incorporation_class_flips": sum(
            ref_pairs[key]["predicted_preference"] != out_pairs[key]["predicted_preference"]
            for key in ref_pairs
        ),
        "global_site_flips": sum(
            ref_pairs[key]["predicted_global_best"] != out_pairs[key]["predicted_global_best"]
            for key in ref_pairs
        ),
        "within_class_site_flips": sum(
            ref_sites[key]["predicted_best_site"] != out_sites[key]["predicted_best_site"]
            for key in ref_sites
        ),
        "global_regret_change_eV": quantile_summary(np.abs(pair_regret)),
        "within_class_regret_change_eV": quantile_summary(np.abs(site_regret)),
    }


def cell_obliquity(cell: np.ndarray, pbc: np.ndarray) -> tuple[float, str]:
    periodic = np.asarray(cell)[np.asarray(pbc, dtype=bool)]
    values = []
    for left in range(len(periodic)):
        for right in range(left + 1, len(periodic)):
            denominator = np.linalg.norm(periodic[left]) * np.linalg.norm(periodic[right])
            if denominator > 0:
                values.append(abs(float(np.dot(periodic[left], periodic[right]) / denominator)))
    value = max(values, default=0.0)
    label = "near_orthogonal" if value < 0.05 else "moderate" if value < 0.25 else "oblique"
    return value, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-db", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--ct-uae", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    git = git_snapshot()
    runtime_gpu = gpu_identity()
    if args.batch_size != 64:
        raise SystemExit("formal G1A requires frozen batch_size=64")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("formal G1A inference requires an assigned CUDA GPU")
    paths = [
        args.data.expanduser().resolve(), args.raw_db.expanduser().resolve(),
        args.run_root.expanduser().resolve(), args.protocol_dir.expanduser().resolve(),
        args.ct_uae.expanduser().resolve(), args.out_dir.expanduser().resolve(),
    ]
    data_path, raw_db_path, run_root, protocol_dir, ct_uae_path, out_dir = paths
    if out_dir.exists():
        raise FileExistsError("G1A output directory already exists")
    if out_dir == ROOT or ROOT in out_dir.parents:
        raise SystemExit("formal G1A output must be outside the Git worktree")
    if sha256_file(data_path) != SOURCE_DATA_SHA256:
        raise SystemExit("G1A source data SHA256 mismatch")
    if sha256_file(raw_db_path) != RAW_DB_SHA256:
        raise SystemExit("G1A raw database SHA256 mismatch")
    if sha256_file(ct_uae_path) != CT_UAE_SHA256:
        raise SystemExit("G1A ct-UAE SHA256 mismatch")
    out_dir.mkdir(parents=True)

    base = CrystalGraphDataset(data_path)
    db = connect(str(raw_db_path))
    pbc_by_index: dict[int, np.ndarray] = {}
    raw_geometry_by_index: dict[int, dict[str, np.ndarray]] = {}
    exact_distances: dict[int, np.ndarray] = {}
    mic_rows = []
    for index, sample in enumerate(base.data):
        raw_atoms = db.get(id=int(sample["id"])).toatoms()
        np.testing.assert_array_equal(sample["numbers"], raw_atoms.get_atomic_numbers())
        np.testing.assert_allclose(sample["positions"], raw_atoms.get_positions(), rtol=0.0, atol=2.0e-5)
        np.testing.assert_allclose(sample["cell"], np.asarray(raw_atoms.get_cell()), rtol=0.0, atol=2.0e-5)
        pbc = np.asarray(raw_atoms.get_pbc(), dtype=bool)
        pbc_by_index[index] = pbc
        raw_geometry_by_index[index] = {
            "numbers": raw_atoms.get_atomic_numbers().astype(np.int64),
            "positions": raw_atoms.get_positions().astype(np.float64),
            "cell": np.asarray(raw_atoms.get_cell()).astype(np.float64),
        }
        assert_legacy_identity_graph(sample, legacy_build_graph(raw_atoms), index)
        exact = _pbc_distance_matrix(sample["positions"], sample["cell"], pbc).astype(np.float32)
        exact_distances[index] = exact
        delta = np.abs(np.asarray(sample["dist_matrix"], dtype=float) - exact)
        degrees = np.bincount(sample["edge_index"][0], minlength=len(sample["numbers"]))
        obliquity, obliquity_bin = cell_obliquity(sample["cell"], pbc)
        mic_rows.append({
            "sample_index": index,
            "defecttype": str(sample.get("metadata", {}).get("defecttype", "")),
            "obliquity": obliquity,
            "obliquity_bin": obliquity_bin,
            "legacy_cap_active": bool(np.any(degrees * (degrees - 1) > 32)),
            "mic_max_delta_A": float(np.max(delta)),
            "mic_mean_delta_A": float(np.mean(delta)),
            "mic_pair_count_gt_1e-6": int(np.sum(delta > 1.0e-6)),
        })

    sample_rows: dict[int, dict[str, Any]] = {}
    with (protocol_dir / "samples.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["sample_index"])
            sample_rows[index] = {
                **row, "sample_index": index, "id": int(row["id"]),
                "target_eV": float(row["target_eV"]), "natoms": int(row["natoms"]),
            }
    canonical_expected = {
        index for index, row in sample_rows.items()
        if str(row.get("canonical_retained", "")).lower() == "true"
    }
    if len(canonical_expected) != 10_224:
        raise ValueError("protocol does not contain exactly 10,224 canonical rows")
    protocol_manifest = json.loads((protocol_dir / "manifest.json").read_text())
    if protocol_manifest.get("data_sha256") != SOURCE_DATA_SHA256:
        raise ValueError("protocol manifest is not bound to the frozen source dataset")

    n_samples = len(base)
    baseline = np.full(n_samples, np.nan)
    identity_predictions = np.full(n_samples, np.nan)
    targets = np.full(n_samples, np.nan)
    exact_predictions = np.full(n_samples, np.nan)
    permutation_predictions = np.full((n_samples, N_PERMUTATIONS), np.nan)
    checkpoint_receipts = []
    roundtrip_deltas = []
    identity_roundtrip_deltas = []

    for fold in range(5):
        split_id = f"pair_cv5_f{fold}"
        run_dir = run_root / "selected/g111/transfer" / split_id / "seed242"
        manifest_path = run_dir / "run_manifest.json"
        checkpoint_path = run_dir / "best.pt"
        archived_path = run_dir / "test_predictions.npz"
        manifest = json.loads(manifest_path.read_text())
        split_path = protocol_dir / "splits" / f"{split_id}.json"
        split = json.loads(split_path.read_text())
        if (
            manifest.get("schema_version") != "prm_run_manifest_v1"
            or manifest.get("status") != "complete"
            or manifest.get("git", {}).get("dirty")
            or manifest.get("git", {}).get("commit") != LEGACY_TRAINING_COMMIT
            or int(manifest.get("seed", -1)) != 242
            or manifest.get("split", {}).get("split_id") != split_id
            or manifest.get("split", {}).get("sha256") != sha256_file(split_path)
            or manifest.get("data", {}).get("data_sha256") != SOURCE_DATA_SHA256
            or manifest.get("config", {}).get("model") != "v2"
            or int(manifest.get("config", {}).get("seed", -1)) != 242
            or Path(manifest.get("config", {}).get("split_path", "")).name
            != f"{split_id}.json"
        ):
            raise ValueError(f"inadmissible legacy checkpoint manifest for {split_id}")
        model_kwargs = manifest["config"].get("model_kwargs", {})
        if not all(
            bool(model_kwargs.get(name))
            for name in (
                "use_gated_pooling", "use_env_enrichment", "use_prenorm_local",
            )
        ):
            raise ValueError(f"legacy checkpoint is not selected g111 for {split_id}")
        assets = manifest.get("assets", {})
        if (
            assets.get("ct_uae", {}).get("sha256") != CT_UAE_SHA256
            or assets.get("pretrained_embed", {}).get("sha256")
            != LEGACY_PRETRAINED_SHA256
        ):
            raise ValueError(f"legacy initialization asset mismatch for {split_id}")
        if (
            manifest.get("outputs", {}).get("checkpoint") != "best.pt"
            or manifest.get("outputs", {}).get("test_predictions")
            != "test_predictions.npz"
            or manifest.get("output_sha256", {}).get("checkpoint")
            != sha256_file(checkpoint_path)
            or manifest.get("output_sha256", {}).get("test_predictions")
            != sha256_file(archived_path)
        ):
            raise ValueError(f"legacy checkpoint/output inventory mismatch for {split_id}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        kwargs = deepcopy(manifest["config"]["model_kwargs"])
        kwargs["ct_uae_path"] = str(ct_uae_path)
        # The archived g111 checkpoints predate the corrected E-module
        # zero-neighbour contract.  G1A is a frozen round-trip diagnostic, so
        # it must reproduce the historical batch-dependent residual exactly.
        kwargs["env_zero_neighbor_mode"] = ENV_ZERO_NEIGHBOR_LEGACY
        model = CrystalTransformerV2(**kwargs).to(device)
        model.load_state_dict(checkpoint["model"])
        normalizer = checkpoint["normalizer"]
        test_indices = [int(index) for index in split["test"]]

        observed_i, observed_p, observed_t = infer(
            model, Subset(base, test_indices), normalizer, device, args.batch_size
        )
        archived_i, archived_p, archived_t = load_archived_predictions(
            archived_path, split_id
        )
        observed_order = np.argsort(observed_i)
        archived_order = np.argsort(archived_i)
        np.testing.assert_array_equal(observed_i[observed_order], archived_i[archived_order])
        np.testing.assert_array_equal(np.sort(archived_i), np.sort(test_indices))
        np.testing.assert_allclose(observed_t[observed_order], archived_t[archived_order], atol=1.0e-6, rtol=0.0)
        np.testing.assert_allclose(
            archived_t,
            np.asarray([sample_rows[int(index)]["target_eV"] for index in archived_i]),
            atol=1.0e-6, rtol=0.0,
        )
        roundtrip_deltas.extend(np.abs(observed_p[observed_order] - archived_p[archived_order]))
        baseline[archived_i] = archived_p
        targets[archived_i] = archived_t

        identity_i, identity_p, _ = infer(
            model,
            DiagnosticDataset(
                base, test_indices, pbc_by_index, raw_geometry_by_index,
                permutation_variant=-1,
            ),
            normalizer, device, args.batch_size,
        )
        identity_predictions[identity_i] = identity_p
        identity_map = dict(zip(identity_i.tolist(), identity_p.tolist()))
        identity_roundtrip_deltas.extend(
            abs(identity_map[int(index)] - float(prediction))
            for index, prediction in zip(archived_i, archived_p)
        )

        exact_i, exact_p, _ = infer(
            model,
            DiagnosticDataset(
                base, test_indices, pbc_by_index, raw_geometry_by_index,
                exact_distances=exact_distances,
            ),
            normalizer, device, args.batch_size,
        )
        exact_predictions[exact_i] = exact_p
        for variant in range(N_PERMUTATIONS):
            variant_i, variant_p, _ = infer(
                model,
                DiagnosticDataset(
                    base, test_indices, pbc_by_index, raw_geometry_by_index,
                    permutation_variant=variant,
                ),
                normalizer, device, args.batch_size,
            )
            permutation_predictions[variant_i, variant] = variant_p

        checkpoint_receipts.append({
            "fold": fold, "split_id": split_id, "seed": 242,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "manifest_sha256": sha256_file(manifest_path),
            "archived_predictions_sha256": sha256_file(archived_path),
            "training_commit": manifest["git"]["commit"],
        })
        del model, checkpoint
        torch.cuda.empty_cache()

    if not (
        np.isfinite(baseline).sum() == 10_224
        and np.isfinite(identity_predictions).sum() == 10_224
        and np.isfinite(exact_predictions).sum() == 10_224
        and np.isfinite(permutation_predictions).sum() == 10_224 * N_PERMUTATIONS
    ):
        raise ValueError("G1A predictions do not cover exactly 10,224 canonical rows")
    canonical = np.flatnonzero(np.isfinite(baseline))
    if set(canonical.tolist()) != canonical_expected:
        raise ValueError("G1A coverage differs from protocol canonical membership")
    if (
        max(roundtrip_deltas, default=float("inf")) > 1.0e-5
        or max(identity_roundtrip_deltas, default=float("inf")) > 1.0e-5
    ):
        raise ValueError("legacy checkpoint/identity graph roundtrip exceeds 1e-5 eV")
    prediction_range = np.ptp(
        np.column_stack([baseline[canonical], permutation_predictions[canonical]]), axis=1
    )
    baseline_mae = float(np.mean(np.abs(baseline[canonical] - targets[canonical])))
    permutation_mae = np.mean(
        np.abs(permutation_predictions[canonical] - targets[canonical, None]), axis=0
    )
    reference_decisions = decision_maps(canonical, baseline[canonical], sample_rows)
    permutation_decisions = [
        decision_delta(
            reference_decisions,
            decision_maps(canonical, permutation_predictions[canonical, variant], sample_rows),
        )
        for variant in range(N_PERMUTATIONS)
    ]
    exact_decisions = decision_delta(
        reference_decisions,
        decision_maps(canonical, exact_predictions[canonical], sample_rows),
    )

    range_stats = quantile_summary(prediction_range)
    max_mae_change = float(np.max(np.abs(permutation_mae - baseline_mae)))
    max_class_flips = max(row["incorporation_class_flips"] for row in permutation_decisions)
    max_site_flips = max(
        max(row["global_site_flips"], row["within_class_site_flips"])
        for row in permutation_decisions
    )
    empirical_pass = (
        range_stats["p99"] < 0.01
        and range_stats["max"] < 0.05
        and max_mae_change < 0.01
        and max_class_flips == 0
        and max_site_flips == 0
    )

    canonical_mic_rows = [
        row for row in mic_rows if int(row["sample_index"]) in canonical_expected
    ]
    grouped: dict[str, dict[str, list[float]]] = {
        "obliquity_bin": defaultdict(list),
        "legacy_cap_active": defaultdict(list),
        "defecttype": defaultdict(list),
    }
    for row in canonical_mic_rows:
        for axis in grouped:
            grouped[axis][str(row[axis])].append(float(row["mic_max_delta_A"]))
    mic_strata = {
        axis: {label: {"n": len(values), **quantile_summary(np.asarray(values))} for label, values in values_by_label.items()}
        for axis, values_by_label in grouped.items()
    }

    summary = {
        "schema_version": "prm_g1a_geometry_diagnostic_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_profile": runtime_gpu["host_profile"],
        "git": git,
        "scope": {
            "legacy_graph": LEGACY_FIRST32,
            "legacy_env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_LEGACY,
            "legacy_inference_batch_size": 64,
            "permutations": list(PERMUTATION_NAMES),
            "exact_mic_intervention": "dense radial matrix only; legacy local graph retained",
            "interpretation": "diagnostic sensitivity only; not a formal invariance waiver",
        },
        "implementation": {
            "diagnostic_repository_path": "scripts/prm_g1_diagnose_geometry.py",
            "diagnostic_sha256": sha256_file(
                ROOT / "scripts/prm_g1_diagnose_geometry.py"
            ),
            "graph_repository_path": "src/graph.py",
            "graph_sha256": sha256_file(ROOT / "src/graph.py"),
            "model_repository_path": "src/models/crystal_v2.py",
            "model_sha256": sha256_file(ROOT / "src/models/crystal_v2.py"),
        },
        "inputs": {
            "dataset": {"alias": "PRM_DATASET_PATH", "sha256": SOURCE_DATA_SHA256},
            "raw_db": {"alias": "PRM_RAW_DB_PATH", "sha256": RAW_DB_SHA256},
            "ct_uae": {"alias": "PRM_CT_UAE_PATH", "sha256": CT_UAE_SHA256},
            "checkpoints": checkpoint_receipts,
        },
        "coverage": {"canonical_rows": len(canonical), "folds": 5},
        "checkpoint_roundtrip": {
            "stored_graph_eV": quantile_summary(np.asarray(roundtrip_deltas)),
            "raw_identity_rebuild_eV": quantile_summary(
                np.asarray(identity_roundtrip_deltas)
            ),
            "hard_max_eV": 1.0e-5,
        },
        "legacy_under_permutations": {
            "prediction_range_eV": range_stats,
            "baseline_mae_eV": baseline_mae,
            "permutation_mae_eV": permutation_mae.tolist(),
            "max_absolute_mae_change_eV": max_mae_change,
            "decision_deltas": permutation_decisions,
            "thresholds": {
                "prediction_range_p99_eV_lt": 0.01,
                "prediction_range_max_eV_lt": 0.05,
                "absolute_mae_change_eV_lt": 0.01,
                "class_and_site_flips_eq": 0,
            },
            "empirical_stability_pass": empirical_pass,
        },
        "old_to_exact_mic": {
            "prediction_absolute_delta_eV": quantile_summary(
                np.abs(exact_predictions[canonical] - baseline[canonical])
            ),
            "mae_change_eV": float(
                np.mean(np.abs(exact_predictions[canonical] - targets[canonical])) - baseline_mae
            ),
            "decision_delta": exact_decisions,
            "distance_sample_summary_A": quantile_summary(
                np.asarray([row["mic_max_delta_A"] for row in canonical_mic_rows])
            ),
            "distance_strata": mic_strata,
            "distance_scope": "10,224 protocol-canonical rows",
            "full_container_rows_audited": len(mic_rows),
        },
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "gpu_name": runtime_gpu["name"],
            "gpu_uuid": runtime_gpu["uuid"],
        },
    }
    np.savez_compressed(
        out_dir / "predictions.npz",
        schema_version=np.asarray("prm_g1a_predictions_v1"), indices=canonical,
        targets=targets[canonical], baseline=baseline[canonical],
        identity_rebuild=identity_predictions[canonical],
        exact_mic=exact_predictions[canonical], permutations=permutation_predictions[canonical],
        permutation_names=np.asarray(PERMUTATION_NAMES),
    )
    with (out_dir / "sample_diagnostics.csv").open("w", newline="") as handle:
        fieldnames = list(mic_rows[0]) + ["permutation_prediction_range_eV", "exact_mic_prediction_delta_eV"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        ranges = dict(zip(canonical, prediction_range))
        for row in mic_rows:
            index = int(row["sample_index"])
            out = dict(row)
            out["permutation_prediction_range_eV"] = ranges.get(index, "")
            out["exact_mic_prediction_delta_eV"] = (
                abs(exact_predictions[index] - baseline[index]) if index in ranges else ""
            )
            writer.writerow(out)
    summary["outputs"] = {
        "predictions_sha256": sha256_file(out_dir / "predictions.npz"),
        "sample_diagnostics_sha256": sha256_file(out_dir / "sample_diagnostics.csv"),
    }
    (out_dir / "summary.json").write_text(strict_json(summary))
    print(strict_json({
        "empirical_stability_pass": empirical_pass,
        "summary_sha256": sha256_file(out_dir / "summary.json"),
    }))


if __name__ == "__main__":
    main()
