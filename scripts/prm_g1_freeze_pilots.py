"""Freeze repaired split membership and two/three G1B pilot YAML contracts."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prm_g1_diagnose_geometry import (  # noqa: E402
    LEGACY_PRETRAINED_SHA256,
    SOURCE_DATA_SHA256,
    sha256_file,
)
from src.graph import GRAPH_BUILDER_VERSION  # noqa: E402
from src.models.crystal_v2 import ENV_ZERO_NEIGHBOR_CORRECTED  # noqa: E402
from src.prm_provenance import config_sha256  # noqa: E402
from src.splits import validate_split  # noqa: E402


SOURCE_SPLIT_REPOSITORY_PATH = Path(
    "artifacts/prm_protocol_v2/splits/pair_cv5_f0.json"
)
SOURCE_SPLIT_SHA256 = (
    "2a27fd4f3e862d2225048b6255ef3c5d9d2e7b88f326e9cd9bd3da5a0223895a"
)
BASE_CONFIG_REPOSITORY_PATH = Path(
    "configs/prm/promoted/g111/transfer/pair_cv5_f0_seed242.yaml"
)
BASE_CONFIG_SHA256 = (
    "04cdb90d48f8e55e0c15b92064ca1af4dc5d64dcc3db012c059e3ea58b642f8e"
)
DERIVED_SPLIT_REPOSITORY_PATH = Path(
    "artifacts/prm_g1/splits/pair_cv5_f0_repaired_v1.json"
)
GENERATED_CONFIG_REPOSITORY_DIR = Path("configs/prm/g1/generated")
FREEZE_MANIFEST_REPOSITORY_PATH = Path("artifacts/prm_g1/pilot_freeze.json")
JARVIS_SOURCE_SHA256 = (
    "41284603e6eeb6920fe132975570e657e69601a414257bb48d2196d7389369fa"
)
EXPECTED_OUTPUT_DIRS = {
    "legacy_init": "g1/pilots/legacy_init/pair_cv5_f0/seed242",
    "no_pretrain": "g1/pilots/no_pretrain/pair_cv5_f0/seed242",
    "corrected_pretrain": "g1/pilots/corrected_pretrain/pair_cv5_f0/seed242",
}


def clean_commit() -> str:
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
    if not commit or dirty or not any(ref.strip() for ref in remote_refs):
        raise SystemExit(
            "pilot freezing requires a clean commit reachable from a fetched remote ref"
        )
    return commit


def require_remote_ancestor(ancestor: str, descendant: str, *, label: str) -> None:
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT, capture_output=True, check=False,
    ).returncode == 0
    remote_refs = subprocess.run(
        ["git", "branch", "-r", "--contains", ancestor], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.splitlines()
    if not is_ancestor or not any(ref.strip() for ref in remote_refs):
        raise ValueError(f"{label} commit is not a pushed ancestor of freeze commit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repaired-dataset-receipt", type=Path, required=True)
    parser.add_argument(
        "--source-split", type=Path,
        default=ROOT / SOURCE_SPLIT_REPOSITORY_PATH,
    )
    parser.add_argument(
        "--base-config", type=Path,
        default=ROOT / BASE_CONFIG_REPOSITORY_PATH,
    )
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--config-output-dir", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--corrected-pretraining-receipt", type=Path, required=True)
    args = parser.parse_args()

    commit = clean_commit()
    if socket.gethostname() != "WHUServer-L40S":
        raise SystemExit("formal G1 pilot freezing requires WHUServer-L40S")
    repaired_receipt_path = args.repaired_dataset_receipt.expanduser().resolve()
    source_split_path = args.source_split.expanduser().resolve()
    base_config_path = args.base_config.expanduser().resolve()
    split_output = args.split_output.expanduser().resolve()
    config_dir = args.config_output_dir.expanduser().resolve()
    freeze_manifest = args.freeze_manifest.expanduser().resolve()
    corrected_path = args.corrected_pretraining_receipt.expanduser().resolve()
    expected_paths = {
        source_split_path: (ROOT / SOURCE_SPLIT_REPOSITORY_PATH).resolve(),
        base_config_path: (ROOT / BASE_CONFIG_REPOSITORY_PATH).resolve(),
        split_output: (ROOT / DERIVED_SPLIT_REPOSITORY_PATH).resolve(),
        config_dir: (ROOT / GENERATED_CONFIG_REPOSITORY_DIR).resolve(),
        freeze_manifest: (ROOT / FREEZE_MANIFEST_REPOSITORY_PATH).resolve(),
    }
    if any(observed != expected for observed, expected in expected_paths.items()):
        raise SystemExit("formal G1 pilot freeze paths are repository-contract fixed")
    if len({
        repaired_receipt_path, source_split_path, base_config_path,
        split_output, config_dir, freeze_manifest, corrected_path,
    }) != 7:
        raise SystemExit("pilot freeze input/output paths must be distinct")
    destinations = [split_output, config_dir, freeze_manifest]
    if any(path.exists() for path in destinations):
        raise FileExistsError("pilot freeze destinations must not already exist")

    repaired = json.loads(repaired_receipt_path.read_text())
    if (
        repaired.get("schema_version") != "prm_g1_repaired_dataset_v1"
        or repaired.get("host_profile") != "WHUServer-L40S"
        or repaired.get("git", {}).get("dirty")
        or repaired.get("inputs", {}).get("dataset", {}).get("sha256")
        != SOURCE_DATA_SHA256
        or repaired.get("graph_builder", {}).get("version") != GRAPH_BUILDER_VERSION
        or float(repaired.get("graph_builder", {}).get("cutoff_A", -1.0)) != 5.0
        or float(repaired.get("graph_builder", {}).get(
            "structure_identity_atol", -1.0
        )) != 2.0e-5
        or repaired.get("container", {}).get("rows") != 10_641
        or repaired.get("container", {}).get("canonical_rows") != 10_224
        or repaired.get("container", {}).get("excluded_rows") != 417
    ):
        raise ValueError("repaired IMP2D receipt is not admissible for pilot freezing")
    require_remote_ancestor(
        str(repaired.get("git", {}).get("commit", "")), commit,
        label="repaired IMP2D",
    )
    repaired_sha = str(repaired["output"]["sha256"])
    if sha256_file(source_split_path) != SOURCE_SPLIT_SHA256:
        raise ValueError("canonical pair-fold source split SHA256 mismatch")
    if sha256_file(base_config_path) != BASE_CONFIG_SHA256:
        raise ValueError("promoted g111 base-config SHA256 mismatch")
    source_split = json.loads(source_split_path.read_text())
    if (
        source_split.get("schema_version") != "prm_split_v1"
        or source_split.get("split_id") != "pair_cv5_f0"
        or source_split.get("data_sha256") != SOURCE_DATA_SHA256
        or int(source_split.get("n_samples", -1)) != 10_641
        or source_split.get("counts")
        != {"excluded": 417, "test": 2045, "train": 6134, "val": 2045}
    ):
        raise ValueError("source pair-fold split contract mismatch")
    validate_split(source_split, 10_641)
    derived_split = deepcopy(source_split)
    derived_split["split_id"] = "g1_pair_cv5_f0_repaired_v1"
    derived_split["data_sha256"] = repaired_sha
    derived_split.setdefault("metadata", {}).update({
        "g1_graph_builder_version": GRAPH_BUILDER_VERSION,
        "membership_source_split": "pair_cv5_f0",
        "membership_source_sha256": SOURCE_SPLIT_SHA256,
        "purpose": "bounded G1 numerical-stability pilot; not paper evidence",
    })
    validate_split(derived_split, 10_641)
    split_output.parent.mkdir(parents=True, exist_ok=True)
    split_output.write_text(
        json.dumps(derived_split, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    base = yaml.safe_load(base_config_path.read_text())
    if not isinstance(base, dict):
        raise ValueError("promoted g111 base config is not a mapping")
    base.update({
        "data_path": "data/processed/cleaned_dataset_g1_repaired_v1.pkl",
        "data_sha256": repaired_sha,
        "split_path": str(split_output.relative_to(ROOT)),
        "epochs": 30,
        "seed": 242,
    })
    base.setdefault("model_kwargs", {})[
        "env_zero_neighbor_mode"
    ] = ENV_ZERO_NEIGHBOR_CORRECTED
    base["g1_pilot_contract"] = {
        "schema_version": "prm_g1_pilot_config_v1",
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "graph_builder_source_sha256": sha256_file(ROOT / "src/graph.py"),
        "env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_CORRECTED,
        "model_source_sha256": sha256_file(ROOT / "src/models/crystal_v2.py"),
        "repaired_dataset_receipt_sha256": sha256_file(repaired_receipt_path),
        "repaired_dataset_builder_commit": repaired["git"]["commit"],
        "source_split_sha256": SOURCE_SPLIT_SHA256,
        "base_config_sha256": BASE_CONFIG_SHA256,
        "derived_split_sha256": sha256_file(split_output),
        "catastrophic_validation_mae_eV_lt": 3.0,
        "test_partition_role": "collected_by legacy trainer but forbidden for pilot decisions",
    }

    arms: dict[str, dict[str, Any]] = {}
    legacy = deepcopy(base)
    legacy["output_dir"] = EXPECTED_OUTPUT_DIRS["legacy_init"]
    legacy["g1_pilot_contract"].update({
        "arm": "legacy_init",
        "claim_boundary": "legacy-initialization-sensitivity only",
    })
    legacy["asset_sha256"]["pretrained_embed"] = LEGACY_PRETRAINED_SHA256
    arms["legacy_init"] = legacy

    no_pretrain = deepcopy(base)
    no_pretrain["output_dir"] = EXPECTED_OUTPUT_DIRS["no_pretrain"]
    no_pretrain.pop("pretrained_embed", None)
    no_pretrain["asset_sha256"].pop("pretrained_embed", None)
    no_pretrain["g1_pilot_contract"].update({
        "arm": "no_pretrain",
        "claim_boundary": "correctness-clean IMP2D graph pilot; no transfer claim",
    })
    arms["no_pretrain"] = no_pretrain

    corrected_receipt = json.loads(corrected_path.read_text())
    if (
        corrected_receipt.get("schema_version") != "prm_g1c_pretraining_receipt_v1"
        or corrected_receipt.get("host_profile") != "WHUServer-L40S"
        or corrected_receipt.get("git", {}).get("dirty")
        or corrected_receipt.get("input", {}).get("frozen_source_dataset_sha256")
        != JARVIS_SOURCE_SHA256
        or corrected_receipt.get("input", {}).get("graph_builder_version")
        != GRAPH_BUILDER_VERSION
        or corrected_receipt.get("input", {}).get("rows") != 19_902
        or corrected_receipt.get("input", {}).get(
            "downstream_env_zero_neighbor_mode"
        ) != ENV_ZERO_NEIGHBOR_CORRECTED
        or corrected_receipt.get("input", {}).get(
            "downstream_model_source_sha256"
        ) != sha256_file(ROOT / "src/models/crystal_v2.py")
        or corrected_receipt.get("dart_g111_compatibility", {}).get(
            "env_zero_neighbor_mode"
        ) != ENV_ZERO_NEIGHBOR_CORRECTED
        or corrected_receipt.get("dart_g111_compatibility", {}).get(
            "model_source_sha256"
        ) != sha256_file(ROOT / "src/models/crystal_v2.py")
        or corrected_receipt.get("training_contract", {}).get("epochs") != 30
        or corrected_receipt.get("training_contract", {}).get("seed") != 42
    ):
        raise ValueError("corrected pretraining receipt is inadmissible")
    require_remote_ancestor(
        str(corrected_receipt.get("git", {}).get("commit", "")), commit,
        label="corrected pretraining",
    )
    corrected_asset_sha = corrected_receipt["outputs"]["corrected_asset_sha256"]
    corrected = deepcopy(base)
    corrected["output_dir"] = EXPECTED_OUTPUT_DIRS["corrected_pretrain"]
    corrected["pretrained_embed"] = "results/g1c/pretrained_embed_corrected.pt"
    corrected["asset_sha256"]["pretrained_embed"] = corrected_asset_sha
    corrected["g1_pilot_contract"].update({
        "arm": "corrected_pretrain",
        "claim_boundary": "end-to-end corrected lineage compatibility pilot",
        "corrected_pretraining_receipt_sha256": sha256_file(corrected_path),
        "corrected_pretraining_commit": corrected_receipt["git"]["commit"],
    })
    arms["corrected_pretrain"] = corrected
    corrected_receipt_hash = sha256_file(corrected_path)

    config_dir.mkdir(parents=True)
    config_records = []
    for arm, config in arms.items():
        path = config_dir / f"{arm}_pair_cv5_f0_seed242.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        config_records.append({
            "arm": arm,
            "repository_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "config_sha256": config_sha256(config),
        })
    payload = {
        "schema_version": "prm_g1_pilot_freeze_v1",
        "git_commit": commit,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "graph_builder_source_sha256": sha256_file(ROOT / "src/graph.py"),
        "env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_CORRECTED,
        "model_source_sha256": sha256_file(ROOT / "src/models/crystal_v2.py"),
        "repaired_dataset_sha256": repaired_sha,
        "repaired_dataset_receipt_sha256": sha256_file(repaired_receipt_path),
        "source_split_repository_path": str(SOURCE_SPLIT_REPOSITORY_PATH),
        "source_split_sha256": SOURCE_SPLIT_SHA256,
        "base_config_repository_path": str(BASE_CONFIG_REPOSITORY_PATH),
        "base_config_sha256": BASE_CONFIG_SHA256,
        "derived_split_repository_path": str(DERIVED_SPLIT_REPOSITORY_PATH),
        "derived_split_sha256": sha256_file(split_output),
        "corrected_pretraining_receipt_sha256": corrected_receipt_hash,
        "configs": config_records,
    }
    freeze_manifest.parent.mkdir(parents=True, exist_ok=True)
    freeze_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"arms": sorted(arms), "freeze_sha256": sha256_file(freeze_manifest)}))


if __name__ == "__main__":
    main()
