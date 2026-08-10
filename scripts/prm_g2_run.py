"""Fail-closed launcher for one canonical G2 repaired-OOF training run.

Wraps ``src/train_enhanced.py`` and, after a verified completion, derives the
``prm_g2_run_manifest_v1`` the acceptance builder and canonical G3 consume.
Preconditions are checked before any GPU work starts; every binding is
re-verified from file hashes after the run.  The wrapper refuses to overwrite
anything: the run directory and the emitted manifest are exclusive-create.

Authorization boundary: G2 was approved on 2026-08-10, but this wrapper may
only run after the G1/G1C receipts exist (see
``paper_Q1/review/g2_execution_runbook.md``).  It fails closed without them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prm_g2_contract import (  # noqa: E402
    CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
    CANONICAL_PREDICTION_SEEDS,
    EXPECTED_ATOM_FEATURE_SHA256,
    EXPECTED_SOURCE_DATA_SHA256,
    OOF_INFERENCE_BATCH_SIZE,
    build_g2_run_manifest,
    file_sha256,
    regime_of_split_id,
    require,
    validate_g1_acceptance_receipt,
    validate_g1c_pretraining_receipt,
    validate_repaired_dataset_receipt,
    validate_split_payload,
)

EXCLUDED_PHYSICAL_GPU = "2"


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()


def require_clean_pushed_head() -> str:
    commit = git_output("rev-parse", "HEAD")
    require(len(commit) == 40, "cannot resolve a 40-hex HEAD commit")
    status = git_output("status", "--porcelain")
    require(status == "", "the working tree is dirty; canonical G2 runs "
            "require a clean pushed commit")
    remotes = [
        line.strip()
        for line in git_output("branch", "-r", "--contains", commit).splitlines()
        if line.strip()
    ]
    require(
        bool(remotes),
        "HEAD is not contained in any fetched remote branch; push first",
    )
    return commit


def exclusive_write_json(path: Path, payload: dict) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="versioned controlled YAML for this run")
    parser.add_argument("--split-id", required=True,
                        help="canonical split id, e.g. pair_cv5_f0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol-dir", required=True,
                        help="protocol root containing splits/<id>.json")
    parser.add_argument("--run-dir", required=True,
                        help="new run directory; must not exist")
    parser.add_argument("--repaired-dataset", required=True,
                        help="repaired graph pickle from the G1 rebuild")
    parser.add_argument("--g1-repaired-dataset-receipt", required=True)
    parser.add_argument("--g1-acceptance-receipt", required=True)
    parser.add_argument("--g1c-pretraining-receipt", required=True)
    parser.add_argument("--pretrained-asset", required=True,
                        help="corrected G1C initialization checkpoint")
    parser.add_argument("--atom-feature-table",
                        default=str(ROOT / "data" / "atom_features_ref.pth"))
    parser.add_argument("--cuda-visible-devices", default=None,
                        help="explicit CUDA_VISIBLE_DEVICES; physical GPU 2 "
                             "is refused")
    args = parser.parse_args()

    # ── Git preflight ────────────────────────────────────────────────
    commit = require_clean_pushed_head()

    # ── GPU policy ───────────────────────────────────────────────────
    env = dict(os.environ)
    if args.cuda_visible_devices is not None:
        devices = [d.strip() for d in args.cuda_visible_devices.split(",") if d.strip()]
        require(
            EXCLUDED_PHYSICAL_GPU not in devices,
            "physical GPU 2 is excluded from every canonical queue",
        )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    # ── Receipt chain ────────────────────────────────────────────────
    rebuild_path = Path(args.g1_repaired_dataset_receipt)
    g1_path = Path(args.g1_acceptance_receipt)
    g1c_path = Path(args.g1c_pretraining_receipt)
    rebuild_payload = json.loads(rebuild_path.read_text())
    validate_repaired_dataset_receipt(rebuild_payload)
    g1_payload = json.loads(g1_path.read_text())
    validate_g1_acceptance_receipt(
        g1_payload,
        repaired_receipt_sha256=file_sha256(rebuild_path),
        repaired_output_sha256=rebuild_payload["output"]["sha256"],
    )
    g1c_payload = json.loads(g1c_path.read_text())
    validate_g1c_pretraining_receipt(g1c_payload, g1_payload)
    lineage_sha256 = file_sha256(g1c_path)
    require(
        g1_payload["corrected_pretraining_lineage_sha256"] == lineage_sha256,
        "G1 acceptance lineage does not match the G1C receipt file hash",
    )

    # ── Frozen inputs ────────────────────────────────────────────────
    repaired_path = Path(args.repaired_dataset)
    repaired_sha256 = file_sha256(repaired_path)
    require(
        repaired_sha256 == rebuild_payload["output"]["sha256"],
        "repaired dataset file hash differs from the G1 rebuild receipt",
    )
    asset_path = Path(args.pretrained_asset)
    asset_sha256 = file_sha256(asset_path)
    require(
        asset_sha256 == g1c_payload["outputs"]["corrected_asset_sha256"],
        "corrected initialization asset hash differs from the G1C receipt",
    )
    atom_table_sha256 = file_sha256(Path(args.atom_feature_table))
    require(
        atom_table_sha256 == EXPECTED_ATOM_FEATURE_SHA256,
        "atom feature table hash violates the frozen contract",
    )
    model_source_sha256 = file_sha256(ROOT / "src" / "models" / "crystal_v2.py")
    require(
        model_source_sha256 == g1_payload["model_source_sha256"],
        "src/models/crystal_v2.py differs from the G1-accepted model source",
    )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", g1_payload["code_commit"], commit],
        cwd=ROOT, check=False,
    )
    require(
        ancestry.returncode == 0,
        "HEAD does not descend from the accepted G1 repair commit",
    )

    # ── Controlled config contract ───────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    require(cfg.get("model") == "v2", "config must instantiate model v2")
    require(
        int(cfg.get("batch_size", -1)) == OOF_INFERENCE_BATCH_SIZE,
        "config batch_size must be the frozen inference batch size 64",
    )
    model_kwargs = cfg.get("model_kwargs", {})
    require(
        model_kwargs.get("env_zero_neighbor_mode")
        == CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
        "config model_kwargs must bind env_zero_neighbor_mode=zero_residual_v1",
    )
    require(
        model_kwargs.get("use_env_enrichment") is True,
        "config model_kwargs must enable the frozen E module",
    )
    require(
        cfg.get("data_sha256") == EXPECTED_SOURCE_DATA_SHA256,
        "config data_sha256 must bind the source split identity",
    )
    regime = regime_of_split_id(args.split_id)
    require(
        args.seed in CANONICAL_PREDICTION_SEEDS[regime],
        f"seed {args.seed} violates the frozen {regime} seed law",
    )

    protocol_dir = Path(args.protocol_dir)
    split_file = protocol_dir / "splits" / f"{args.split_id}.json"
    split_payload = json.loads(split_file.read_text())
    validate_split_payload(split_payload, args.split_id)
    split_sha256 = file_sha256(split_file)

    run_dir = Path(args.run_dir)
    require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    g2_manifest_path = run_dir / "g2_run_manifest.json"

    # ── Launch ───────────────────────────────────────────────────────
    env["PRM_PRETRAINED_EMBED"] = str(asset_path)
    command = [
        sys.executable, str(ROOT / "src" / "train_enhanced.py"),
        "--config", args.config,
        "--seed", str(args.seed),
        "--data-path", str(repaired_path),
        "--split-path", str(split_file),
        "--output-dir", str(run_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    require(
        completed.returncode == 0,
        f"trainer exited with status {completed.returncode}; evidence in "
        f"{run_dir} is preserved and no G2 manifest is written",
    )

    # ── Postflight ───────────────────────────────────────────────────
    post_commit = git_output("rev-parse", "HEAD")
    post_status = git_output("status", "--porcelain")
    require(
        post_commit == commit and post_status == "",
        "HEAD or tree state changed during the run; the run is not canonical",
    )
    source_manifest_path = run_dir / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    require(
        source_manifest.get("git", {}).get("commit") == commit,
        "trainer recorded a different commit than the preflight HEAD",
    )
    prediction_path = run_dir / "test_predictions.npz"
    declared = str(
        source_manifest.get("output_sha256", {}).get("test_predictions", "")
    )
    require(
        file_sha256(prediction_path) == declared,
        "prediction file hash differs from the trainer's declared output hash",
    )

    manifest = build_g2_run_manifest(
        source_manifest=source_manifest,
        source_manifest_sha256=file_sha256(source_manifest_path),
        split_payload=split_payload,
        split_sha256=split_sha256,
        repaired_data_sha256=repaired_sha256,
        model_source_sha256=model_source_sha256,
        pretrained_asset_sha256=asset_sha256,
        pretrained_asset_lineage_sha256=lineage_sha256,
        atom_feature_table_sha256=atom_table_sha256,
    )
    exclusive_write_json(g2_manifest_path, manifest)
    print(f"g2_run_manifest written: {g2_manifest_path}")
    print(f"prediction_sha256: {manifest['outputs']['prediction_sha256']}")


if __name__ == "__main__":
    main()
