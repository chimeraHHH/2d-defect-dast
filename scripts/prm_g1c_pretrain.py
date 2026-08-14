"""Run hash-bound 30-epoch corrected JARVIS source-task pretraining on CUDA."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prm_g1_diagnose_geometry import (  # noqa: E402
    CT_UAE_SHA256,
    gpu_identity,
    git_snapshot,
    sha256_file,
)
from src.graph import GRAPH_BUILDER_VERSION  # noqa: E402
from src.models.crystal_v2 import (  # noqa: E402
    ENV_ZERO_NEIGHBOR_CORRECTED,
    CrystalTransformerV2,
)
from src.prm_assets import load_pretrained_initialization  # noqa: E402
from src.prm_provenance import config_sha256  # noqa: E402


EXPECTED_ROWS = 19_902
JARVIS_SOURCE_SHA256 = (
    "41284603e6eeb6920fe132975570e657e69601a414257bb48d2196d7389369fa"
)
TRAINING_CONFIG = {
    "model": "baseline",
    "seed": 42,
    "batch_size": 64,
    "epochs": 30,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "grad_clip": 5.0,
    "optimizer": {"lr": 3.0e-4, "weight_decay": 1.0e-4},
    "model_kwargs": {
        "atom_fea_len": 9,
        "hidden_dim": 128,
        "n_local_layers": 3,
        "n_global_layers": 2,
        "num_heads": 4,
        "rcut_local": 5.0,
        "dmax_global": 12.0,
        "defect_embedding": False,
        "dropout": 0.1,
    },
}


def require_remote_ancestor(ancestor: str, descendant: str) -> None:
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT, capture_output=True, check=False,
    ).returncode == 0
    remote_refs = subprocess.run(
        ["git", "branch", "-r", "--contains", ancestor], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.splitlines()
    if not is_ancestor or not any(ref.strip() for ref in remote_refs):
        raise ValueError("corrected JARVIS builder commit is not a pushed ancestor")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-data", type=Path, required=True)
    parser.add_argument("--dataset-receipt", type=Path, required=True)
    parser.add_argument("--ct-uae", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    git = git_snapshot()
    gpu = gpu_identity()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("formal G1C pretraining requires assigned CUDA")
    data = args.corrected_data.expanduser().resolve()
    data_receipt_path = args.dataset_receipt.expanduser().resolve()
    ct_uae_path = args.ct_uae.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if len({data, data_receipt_path, ct_uae_path, out_dir}) != 4:
        raise SystemExit("G1C pretraining input/output paths must be distinct")
    if out_dir.exists():
        raise FileExistsError("versioned G1C pretraining output already exists")
    if out_dir == ROOT or ROOT in out_dir.parents:
        raise SystemExit("formal G1C pretraining output must be outside the Git worktree")
    data_receipt = json.loads(data_receipt_path.read_text())
    if sha256_file(ct_uae_path) != CT_UAE_SHA256:
        raise ValueError("G1C compatibility ct-UAE SHA256 mismatch")
    if (
        data_receipt.get("schema_version") != "prm_g1c_jarvis_dataset_v1"
        or data_receipt.get("host_profile") != "WHUServer-L40S"
        or data_receipt.get("input", {}).get("sha256") != JARVIS_SOURCE_SHA256
        or data_receipt.get("container", {}).get("rows") != EXPECTED_ROWS
        or data_receipt.get("container", {}).get("order_preserved") is not True
        or data_receipt.get("container", {}).get(
            "targets_and_non_graph_fields_preserved"
        ) is not True
        or data_receipt.get("graph_builder", {}).get("version")
        != GRAPH_BUILDER_VERSION
        or float(data_receipt.get("graph_builder", {}).get("cutoff_A", -1.0))
        != 5.0
        or data_receipt.get("graph_builder", {}).get("pbc")
        != [True, True, True]
        or data_receipt.get("graph_builder", {}).get("exact_mic_backend")
        != "ase.geometry.find_mic"
        or int(data_receipt.get("graph_builder", {}).get(
            "triplet_cap_per_centre", -1
        )) != 32
        or int(data_receipt.get("graph_builder", {}).get(
            "triplet_centre_column", -1
        )) != 1
        or data_receipt.get("output", {}).get("sha256") != sha256_file(data)
        or int(data_receipt.get("output", {}).get("size_bytes", 0))
        != data.stat().st_size
        or data_receipt.get("git", {}).get("dirty")
    ):
        raise ValueError("corrected JARVIS data/receipt contract mismatch")
    require_remote_ancestor(
        str(data_receipt.get("git", {}).get("commit", "")), git["commit"]
    )

    config = dict(TRAINING_CONFIG)
    config["data_path"] = str(data)
    config["output_dir"] = str(out_dir)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="prm_g1c_pretrain_", delete=False
    ) as handle:
        config_path = Path(handle.name)
        yaml.safe_dump(config, handle, sort_keys=False)
    try:
        subprocess.run(
            [
                sys.executable, "-m", "scripts.pretrain_element_embeddings",
                "--config", str(config_path), "--device", "cuda",
            ],
            cwd=ROOT, check=True,
        )
    finally:
        if config_path.exists():
            config_path.unlink()

    legacy_named_asset = out_dir / "pretrained_embed.pt"
    best_checkpoint = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.json"
    for path in (legacy_named_asset, best_checkpoint, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"G1C pretraining output missing: {path.name}")
    metrics = json.loads(metrics_path.read_text())
    for key in ("best_val_mae", "test_mae", "test_rmse"):
        if not math.isfinite(float(metrics.get(key, float("nan")))):
            raise ValueError(f"non-finite corrected source-task metric: {key}")
    if int(metrics.get("n_params", 0)) <= 0:
        raise ValueError("corrected source-task model parameter count is invalid")
    history = metrics.get("history")
    if (
        not isinstance(history, list)
        or len(history) != 30
        or [int(row.get("epoch", -1)) for row in history] != list(range(1, 31))
        or any(
            not math.isfinite(float(row.get(key, float("nan"))))
            for row in history
            for key in ("train_mae", "val_mae", "val_rmse", "lr")
        )
        or not math.isclose(
            float(metrics["best_val_mae"]),
            min(float(row["val_mae"]) for row in history),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("corrected source-task history is incomplete or inconsistent")

    original_asset = torch.load(
        legacy_named_asset, map_location="cpu", weights_only=True
    )
    corrected_asset = {
        **original_asset,
        "schema_version": "prm_g1c_pretrained_initialization_v1",
        "source_dataset": "jarvis_dft_3d_lite_corrected_g1c",
        "source_dataset_sha256": sha256_file(data),
        "source_dataset_receipt_sha256": sha256_file(data_receipt_path),
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "training_commit": git["commit"],
        "training_seed": 42,
        "training_epochs": 30,
        "downstream_env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_CORRECTED,
        "downstream_model_source_sha256": sha256_file(
            ROOT / "src/models/crystal_v2.py"
        ),
    }
    corrected_path = out_dir / "pretrained_embed_corrected.pt"
    if corrected_path.exists():
        raise FileExistsError(corrected_path)
    torch.save(corrected_asset, corrected_path)
    compatibility_model = CrystalTransformerV2(
        atom_fea_len=9, hidden_dim=128, n_local_layers=3, n_global_layers=2,
        num_heads=4, rcut_local=5.0, dmax_global=12.0,
        defect_embedding=True, dropout=0.1, ct_uae_path=str(ct_uae_path),
        use_gated_pooling=True, use_env_enrichment=True, use_prenorm_local=True,
        env_zero_neighbor_mode=ENV_ZERO_NEIGHBOR_CORRECTED,
    )
    compatibility = load_pretrained_initialization(
        compatibility_model, corrected_path
    )
    compatibility.pop("checkpoint_path", None)
    compatibility["env_zero_neighbor_mode"] = ENV_ZERO_NEIGHBOR_CORRECTED
    compatibility["model_source_sha256"] = sha256_file(
        ROOT / "src/models/crystal_v2.py"
    )

    receipt = {
        "schema_version": "prm_g1c_pretraining_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_profile": gpu["host_profile"],
        "git": git,
        "gpu": {"uuid": gpu["uuid"], "name": gpu["name"]},
        "input": {
            "dataset_alias": "PRM_CORRECTED_JARVIS_DATA_PATH",
            "dataset_sha256": sha256_file(data),
            "dataset_receipt_sha256": sha256_file(data_receipt_path),
            "frozen_source_dataset_sha256": JARVIS_SOURCE_SHA256,
            "graph_builder_version": GRAPH_BUILDER_VERSION,
            "rows": EXPECTED_ROWS,
            "ct_uae_sha256_for_dart_compatibility": CT_UAE_SHA256,
            "downstream_env_zero_neighbor_mode": ENV_ZERO_NEIGHBOR_CORRECTED,
            "downstream_model_source_sha256": sha256_file(
                ROOT / "src/models/crystal_v2.py"
            ),
        },
        "training_contract": TRAINING_CONFIG,
        "training_contract_sha256": config_sha256(TRAINING_CONFIG),
        "split_contract": {
            "algorithm": "python_random_shuffle_v1",
            "seed": 42,
            "train_rows": 15_921,
            "validation_rows": 1_990,
            "test_rows": 1_991,
        },
        "metrics": {
            key: float(metrics[key])
            for key in ("best_val_mae", "test_mae", "test_rmse")
        },
        "training_audit": {
            "epochs_completed": len(history),
            "best_epoch": int(min(history, key=lambda row: row["val_mae"])["epoch"]),
            "finite_history": True,
        },
        "outputs": {
            "corrected_asset_alias": "PRM_CORRECTED_PRETRAINED_PATH",
            "corrected_asset_sha256": sha256_file(corrected_path),
            "source_export_sha256": sha256_file(legacy_named_asset),
            "best_checkpoint_sha256": sha256_file(best_checkpoint),
            "metrics_sha256": sha256_file(metrics_path),
        },
        "dart_g111_compatibility": compatibility,
        "claim_boundary": (
            "corrected source-task lineage only; downstream IMP2D utility "
            "requires the preregistered corrected_pretrain pilot arm"
        ),
    }
    receipt_path = out_dir / "pretraining_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "corrected_asset_sha256": receipt["outputs"]["corrected_asset_sha256"],
        "receipt_sha256": sha256_file(receipt_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
