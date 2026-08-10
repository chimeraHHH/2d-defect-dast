"""Fail-closed launcher and receipt writer for one preregistered G1B pilot arm."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prm_g1_diagnose_geometry import (  # noqa: E402
    CT_UAE_SHA256,
    LEGACY_PRETRAINED_SHA256,
    gpu_identity,
    git_snapshot,
    sha256_file,
)
from src.graph import GRAPH_BUILDER_VERSION  # noqa: E402
from src.models.crystal_v2 import ENV_ZERO_NEIGHBOR_CORRECTED  # noqa: E402
from src.prm_provenance import (  # noqa: E402
    config_sha256,
    validate_training_completion,
)
from src.splits import validate_split  # noqa: E402
from scripts.prm_g1_freeze_pilots import (  # noqa: E402
    BASE_CONFIG_REPOSITORY_PATH,
    BASE_CONFIG_SHA256,
    DERIVED_SPLIT_REPOSITORY_PATH,
    EXPECTED_OUTPUT_DIRS,
    FREEZE_MANIFEST_REPOSITORY_PATH,
    GENERATED_CONFIG_REPOSITORY_DIR,
    SOURCE_SPLIT_REPOSITORY_PATH,
    SOURCE_SPLIT_SHA256,
)


ALLOWED_ARMS = {"legacy_init", "no_pretrain", "corrected_pretrain"}
EXPECTED_CLAIMS = {
    "legacy_init": "legacy-initialization-sensitivity only",
    "no_pretrain": "correctness-clean IMP2D graph pilot; no transfer claim",
    "corrected_pretrain": "end-to-end corrected lineage compatibility pilot",
}


def require_ancestor(ancestor: str, descendant: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{label} commit is not an ancestor of the pilot commit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--ct-uae", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--pretraining-receipt", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    git = git_snapshot()
    gpu = gpu_identity()
    config_path = args.config.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    dataset_receipt_path = args.dataset_receipt.expanduser().resolve()
    ct_uae_path = args.ct_uae.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    if result_root == ROOT or ROOT in result_root.parents:
        raise SystemExit("formal G1 pilot result root must be outside the Git worktree")
    canonical_config_dir = (ROOT / GENERATED_CONFIG_REPOSITORY_DIR).resolve()
    if not config_path.is_relative_to(canonical_config_dir):
        raise SystemExit("formal G1 pilot config must be in the frozen config directory")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("G1 pilot config must be a mapping")
    pilot = config.get("g1_pilot_contract", {})
    arm = str(pilot.get("arm", ""))
    expected_config_path = (
        canonical_config_dir / f"{arm}_pair_cv5_f0_seed242.yaml"
    ).resolve()
    if (
        pilot.get("schema_version") != "prm_g1_pilot_config_v1"
        or pilot.get("graph_builder_version") != GRAPH_BUILDER_VERSION
        or pilot.get("graph_builder_source_sha256")
        != sha256_file(ROOT / "src/graph.py")
        or pilot.get("env_zero_neighbor_mode")
        != ENV_ZERO_NEIGHBOR_CORRECTED
        or pilot.get("model_source_sha256")
        != sha256_file(ROOT / "src/models/crystal_v2.py")
        or arm not in ALLOWED_ARMS
        or config_path != expected_config_path
        or int(config.get("epochs", -1)) != 30
        or int(config.get("seed", -1)) != 242
        or float(pilot.get("catastrophic_validation_mae_eV_lt", -1)) != 3.0
        or pilot.get("claim_boundary") != EXPECTED_CLAIMS.get(arm)
        or config.get("output_dir") != EXPECTED_OUTPUT_DIRS.get(arm)
        or config.get("split_path") != str(DERIVED_SPLIT_REPOSITORY_PATH)
        or config.get("data_path")
        != "data/processed/cleaned_dataset_g1_repaired_v1.pkl"
        or pilot.get("source_split_sha256") != SOURCE_SPLIT_SHA256
        or pilot.get("base_config_sha256") != BASE_CONFIG_SHA256
    ):
        raise ValueError("G1 pilot config contract mismatch")
    base_config_path = (ROOT / BASE_CONFIG_REPOSITORY_PATH).resolve()
    if sha256_file(base_config_path) != BASE_CONFIG_SHA256:
        raise ValueError("promoted g111 base config SHA256 mismatch")
    base_config = yaml.safe_load(base_config_path.read_text())
    expected_model_kwargs = dict(base_config.get("model_kwargs", {}))
    expected_model_kwargs[
        "env_zero_neighbor_mode"
    ] = ENV_ZERO_NEIGHBOR_CORRECTED
    allowed_changes = {
        "data_path", "data_sha256", "split_path", "epochs", "seed",
        "output_dir", "pretrained_embed", "asset_sha256", "model_kwargs",
        "g1_pilot_contract",
    }
    if (
        any(
            config.get(key) != value
            for key, value in base_config.items()
            if key not in allowed_changes
        )
        or set(config) - (set(base_config) | {"g1_pilot_contract"})
        or config.get("asset_integrity_required") is not True
        or config.get("asset_sha256", {}).get("ct_uae") != CT_UAE_SHA256
        or config.get("model_kwargs") != expected_model_kwargs
    ):
        raise ValueError("G1 pilot diverges from the frozen promoted g111 recipe")

    freeze_path = (ROOT / FREEZE_MANIFEST_REPOSITORY_PATH).resolve()
    freeze = json.loads(freeze_path.read_text())
    config_records = {
        record.get("arm"): record for record in freeze.get("configs", [])
        if isinstance(record, dict)
    }
    config_record = config_records.get(arm, {})
    if (
        freeze.get("schema_version") != "prm_g1_pilot_freeze_v1"
        or freeze.get("graph_builder_version") != GRAPH_BUILDER_VERSION
        or freeze.get("graph_builder_source_sha256")
        != sha256_file(ROOT / "src/graph.py")
        or freeze.get("env_zero_neighbor_mode")
        != ENV_ZERO_NEIGHBOR_CORRECTED
        or freeze.get("model_source_sha256")
        != sha256_file(ROOT / "src/models/crystal_v2.py")
        or freeze.get("source_split_sha256") != SOURCE_SPLIT_SHA256
        or freeze.get("base_config_sha256") != BASE_CONFIG_SHA256
        or freeze.get("derived_split_repository_path")
        != str(DERIVED_SPLIT_REPOSITORY_PATH)
        or config_record.get("repository_path")
        != str(config_path.relative_to(ROOT))
        or config_record.get("sha256") != sha256_file(config_path)
        or config_record.get("config_sha256") != config_sha256(config)
    ):
        raise ValueError("G1 pilot config is not bound to the canonical freeze manifest")
    require_ancestor(str(freeze.get("git_commit", "")), git["commit"], label="freeze")

    dataset_receipt = json.loads(dataset_receipt_path.read_text())
    data_sha = sha256_file(data_path)
    if (
        dataset_receipt.get("schema_version") != "prm_g1_repaired_dataset_v1"
        or dataset_receipt.get("git", {}).get("dirty")
        or dataset_receipt.get("git", {}).get("commit")
        != pilot.get("repaired_dataset_builder_commit")
        or sha256_file(dataset_receipt_path)
        != pilot.get("repaired_dataset_receipt_sha256")
        or dataset_receipt.get("graph_builder", {}).get("version")
        != GRAPH_BUILDER_VERSION
        or dataset_receipt.get("output", {}).get("sha256") != data_sha
        or int(dataset_receipt.get("output", {}).get("size_bytes", 0))
        != data_path.stat().st_size
        or config.get("data_sha256") != data_sha
        or freeze.get("repaired_dataset_sha256") != data_sha
        or freeze.get("repaired_dataset_receipt_sha256")
        != sha256_file(dataset_receipt_path)
    ):
        raise ValueError("actual repaired dataset/receipt/config SHA mismatch")
    require_ancestor(
        str(dataset_receipt.get("git", {}).get("commit", "")),
        git["commit"], label="repaired-dataset builder",
    )
    split_path = (ROOT / config["split_path"]).resolve()
    if split_path != (ROOT / DERIVED_SPLIT_REPOSITORY_PATH).resolve():
        raise ValueError("pilot split path escaped the canonical repository artifact")
    split = json.loads(split_path.read_text())
    source_split_path = (ROOT / SOURCE_SPLIT_REPOSITORY_PATH).resolve()
    if sha256_file(source_split_path) != SOURCE_SPLIT_SHA256:
        raise ValueError("canonical source split SHA256 mismatch")
    source_split = json.loads(source_split_path.read_text())
    if (
        split.get("split_id") != "g1_pair_cv5_f0_repaired_v1"
        or split.get("data_sha256") != data_sha
        or split.get("metadata", {}).get("membership_source_split")
        != "pair_cv5_f0"
        or split.get("metadata", {}).get("membership_source_sha256")
        != SOURCE_SPLIT_SHA256
        or split.get("metadata", {}).get("g1_graph_builder_version")
        != GRAPH_BUILDER_VERSION
        or sha256_file(split_path) != freeze.get("derived_split_sha256")
        or sha256_file(split_path) != pilot.get("derived_split_sha256")
        or any(
            split.get(partition) != source_split.get(partition)
            for partition in ("train", "val", "test", "excluded")
        )
    ):
        raise ValueError("derived G1 split is not bound to the actual repaired dataset")
    validate_split(source_split, 10_641)
    validate_split(split, 10_641)
    if sha256_file(ct_uae_path) != CT_UAE_SHA256:
        raise ValueError("ct-UAE asset SHA256 mismatch")

    pretrained_path = (
        args.pretrained.expanduser().resolve() if args.pretrained is not None else None
    )
    expected_pretrained_sha = config.get("asset_sha256", {}).get("pretrained_embed")
    pretraining_receipt_hash = None
    if arm == "no_pretrain":
        if (
            pretrained_path is not None or expected_pretrained_sha is not None
            or config.get("pretrained_embed") or args.pretraining_receipt is not None
            or set(config.get("asset_sha256", {})) != {"ct_uae"}
        ):
            raise ValueError("no_pretrain arm received a pretraining asset")
    else:
        if arm == "legacy_init" and args.pretraining_receipt is not None:
            raise ValueError("legacy_init must not receive a corrected-pretraining receipt")
        if pretrained_path is None or sha256_file(pretrained_path) != expected_pretrained_sha:
            raise ValueError("pilot pretraining asset SHA256 mismatch")
        if arm == "legacy_init" and expected_pretrained_sha != LEGACY_PRETRAINED_SHA256:
            raise ValueError("legacy_init arm is not bound to the frozen legacy asset")
        if arm == "corrected_pretrain":
            if args.pretraining_receipt is None:
                raise ValueError("corrected_pretrain arm requires its G1C receipt")
            receipt_path = args.pretraining_receipt.expanduser().resolve()
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("schema_version") != "prm_g1c_pretraining_receipt_v1"
                or receipt.get("git", {}).get("dirty")
                or receipt.get("git", {}).get("commit")
                != pilot.get("corrected_pretraining_commit")
                or sha256_file(receipt_path)
                != pilot.get("corrected_pretraining_receipt_sha256")
                or receipt.get("outputs", {}).get("corrected_asset_sha256")
                != expected_pretrained_sha
                or receipt.get("input", {}).get(
                    "downstream_env_zero_neighbor_mode"
                ) != ENV_ZERO_NEIGHBOR_CORRECTED
                or receipt.get("input", {}).get(
                    "downstream_model_source_sha256"
                ) != sha256_file(ROOT / "src/models/crystal_v2.py")
                or receipt.get("dart_g111_compatibility", {}).get(
                    "env_zero_neighbor_mode"
                ) != ENV_ZERO_NEIGHBOR_CORRECTED
                or receipt.get("dart_g111_compatibility", {}).get(
                    "model_source_sha256"
                ) != sha256_file(ROOT / "src/models/crystal_v2.py")
            ):
                raise ValueError("corrected pretraining asset/receipt mismatch")
            pretraining_receipt_hash = sha256_file(receipt_path)

    configured_output = Path(str(config["output_dir"]))
    if configured_output.is_absolute() or ".." in configured_output.parts:
        raise ValueError("G1 pilot output_dir must be a safe relative path")
    output_dir = (result_root / configured_output).resolve()
    if (
        not output_dir.is_relative_to(result_root)
        or output_dir != (result_root / EXPECTED_OUTPUT_DIRS[arm]).resolve()
    ):
        raise ValueError("G1 pilot output_dir escaped its fixed result-root location")
    if output_dir.exists():
        raise FileExistsError("versioned G1 pilot output already exists")
    environment = dict(os.environ)
    environment["PRM_RESULTS_ROOT"] = str(result_root)
    environment["PRM_DATA_PATH"] = str(data_path)
    environment["PRM_CT_UAE_PATH"] = str(ct_uae_path)
    if pretrained_path is not None:
        environment["PRM_PRETRAINED_EMBED"] = str(pretrained_path)
    else:
        environment.pop("PRM_PRETRAINED_EMBED", None)
    subprocess.run(
        [
            sys.executable, "-m", "src.train_enhanced", "--config",
            str(config_path), "--device", "cuda",
        ],
        cwd=ROOT, env=environment, check=True,
    )

    run_manifest_path = output_dir / "run_manifest.json"
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "best.pt"
    run_manifest = json.loads(run_manifest_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    validate_training_completion(run_manifest, run_manifest_path, verify_outputs=True)
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint_payload, dict) or "model" not in checkpoint_payload:
        raise ValueError("pilot checkpoint is not readable as a model checkpoint")
    best_val = float(metrics.get("best_val_mae", float("nan")))
    history = metrics.get("history", [])
    if (
        run_manifest.get("status") != "complete"
        or run_manifest.get("git", {}).get("dirty")
        or run_manifest.get("git", {}).get("commit") != git["commit"]
        or run_manifest.get("config_sha256") != config_sha256(config)
        or run_manifest.get("config") != config
        or run_manifest.get("data", {}).get("data_sha256") != data_sha
        or int(run_manifest.get("data", {}).get("size_bytes", 0))
        != data_path.stat().st_size
        or run_manifest.get("split", {}).get("split_id")
        != "g1_pair_cv5_f0_repaired_v1"
        or run_manifest.get("split", {}).get("sha256") != sha256_file(split_path)
        or int(run_manifest.get("seed", -1)) != 242
        or run_manifest.get("environment", {}).get("hostname") != "WHUServer-L40S"
        or run_manifest.get("environment", {}).get("device") != "cuda"
        or run_manifest.get("environment", {}).get("device_name") != gpu["name"]
        or len(history) != 30
        or not math.isfinite(best_val)
    ):
        raise ValueError("completed pilot output failed provenance/stability validation")
    expected_assets = {"ct_uae": CT_UAE_SHA256}
    if expected_pretrained_sha is not None:
        expected_assets["pretrained_embed"] = expected_pretrained_sha
    manifest_assets = run_manifest.get("assets", {})
    if set(manifest_assets) != set(expected_assets) or any(
        manifest_assets[key].get("sha256") != value
        or manifest_assets[key].get("expected_sha256") != value
        for key, value in expected_assets.items()
    ):
        raise ValueError("pilot run manifest asset inventory mismatch")
    if arm == "no_pretrain":
        if "pretraining" in run_manifest:
            raise ValueError("no_pretrain run unexpectedly loaded pretraining")
    elif run_manifest.get("pretraining", {}).get("checkpoint_sha256") != expected_pretrained_sha:
        raise ValueError("pilot pretraining report checkpoint mismatch")
    with np.load(output_dir / "split_indices.npz", allow_pickle=False) as archive:
        if any(
            not np.array_equal(
                np.asarray(archive[partition], dtype=np.int64),
                np.asarray(split[partition], dtype=np.int64),
            )
            for partition in ("train", "val", "test")
        ):
            raise ValueError("pilot run split indices differ from the frozen membership")
    passed = best_val < 3.0
    receipt = {
        "schema_version": "prm_g1_pilot_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_profile": gpu["host_profile"],
        "gpu": {"uuid": gpu["uuid"], "name": gpu["name"]},
        "git": git,
        "arm": arm,
        "claim_boundary": pilot["claim_boundary"],
        "inputs": {
            "config_repository_path": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256_file(config_path),
            "controlled_config_sha256": config_sha256(config),
            "dataset_alias": "PRM_REPAIRED_DATASET_PATH",
            "dataset_sha256": data_sha,
            "dataset_receipt_sha256": sha256_file(dataset_receipt_path),
            "split_sha256": sha256_file(split_path),
            "source_split_sha256": SOURCE_SPLIT_SHA256,
            "freeze_manifest_sha256": sha256_file(freeze_path),
            "env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_CORRECTED,
            "model_source_sha256": sha256_file(
                ROOT / "src/models/crystal_v2.py"
            ),
            "ct_uae_sha256": CT_UAE_SHA256,
            "pretrained_sha256": expected_pretrained_sha,
            "pretraining_receipt_sha256": pretraining_receipt_hash,
        },
        "acceptance": {
            "epochs_completed": len(history),
            "best_validation_mae_eV": best_val,
            "catastrophic_threshold_eV_lt": 3.0,
            "finite_history": all(
                math.isfinite(float(row[key]))
                for row in history for key in ("train_mae", "val_mae", "val_rmse")
            ),
            "checkpoint_readable": True,
            "passed": passed,
        },
        "outputs": {
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "metrics_sha256": sha256_file(metrics_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "split_indices_sha256": sha256_file(output_dir / "split_indices.npz"),
            "validation_predictions_sha256": sha256_file(
                output_dir / "val_predictions.npz"
            ),
            "test_predictions_sha256": sha256_file(
                output_dir / "test_predictions.npz"
            ),
        },
        "test_metric_policy": "trainer output exists but is excluded from all G1 decisions and claims",
    }
    if not receipt["acceptance"]["finite_history"] or not passed:
        raise ValueError("G1 pilot failed its preregistered numerical-stability gate")
    receipt_path = output_dir / "g1_pilot_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "arm": arm, "passed": passed, "receipt_sha256": sha256_file(receipt_path)
    }, sort_keys=True))


if __name__ == "__main__":
    main()
