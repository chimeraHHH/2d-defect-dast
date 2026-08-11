"""Assemble and verify one ``prm_g2_acceptance_v1`` from completed G2 runs.

Reads the ``g2_run_manifest.json`` written by ``scripts/prm_g2_run.py`` for
every canonical fold and seed, re-verifies each binding from file hashes,
enforces the frozen seed law, and writes the acceptance JSON that canonical
G3 consumes.  The builder recomputes the model-recipe hash from each run's
embedded ``config.model_kwargs``; it never trusts manifest prose.

Committed acceptance files retain aliases and hashes only: paths outside the
repository are emitted through explicit ``$VAR``-style aliases that the
consumer resolves with its environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prm_g2_contract import (  # noqa: E402
    CANONICAL_FOLDS,
    CANONICAL_PREDICTION_SEEDS,
    CANONICAL_REGIMES,
    EXPECTED_SOURCE_DATA_SHA256,
    build_g2_acceptance,
    canonical_json_sha256,
    file_sha256,
    require,
    validate_g1_acceptance_receipt,
    validate_g1c_pretraining_receipt,
    validate_g2_run_manifest,
    validate_repaired_dataset_receipt,
    validate_split_payload,
)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()


def contract_path(path: Path, run_root: Path | None, run_root_alias: str | None,
                  explicit_alias: str | None = None) -> str:
    """Return the alias/relative string a committed acceptance may contain."""
    resolved = path.resolve()
    if explicit_alias:
        expanded = os.path.expandvars(explicit_alias)
        require(
            "$" not in expanded,
            f"alias does not resolve in this environment: {explicit_alias}",
        )
        require(
            Path(expanded).expanduser().resolve() == resolved,
            f"alias resolves to a different file: {explicit_alias}",
        )
        return explicit_alias
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        pass
    if run_root is not None and run_root_alias:
        try:
            relative = resolved.relative_to(run_root.resolve()).as_posix()
        except ValueError:
            relative = None
        if relative is not None:
            expanded = os.path.expandvars(run_root_alias)
            require(
                "$" not in expanded
                and Path(expanded).expanduser().resolve() == run_root.resolve(),
                f"run-root alias does not resolve to {run_root}",
            )
            return f"{run_root_alias}/{relative}"
    raise ValueError(
        f"path is outside the repository and no valid alias was supplied: {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True,
                        help="root holding <regime>_cv5_f<fold>/seed<seed>/ runs")
    parser.add_argument("--run-root-alias", default=None,
                        help="alias emitted for paths under --prediction-root, "
                             "e.g. '$PRM_RUN_ROOT/g2'")
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--g1-repaired-dataset-receipt", required=True)
    parser.add_argument("--g1-repaired-dataset-receipt-alias", default=None)
    parser.add_argument("--g1-acceptance-receipt", required=True)
    parser.add_argument("--g1-acceptance-receipt-alias", default=None)
    parser.add_argument("--g1c-pretraining-receipt", required=True)
    parser.add_argument("--g1c-pretraining-receipt-alias", default=None)
    parser.add_argument("--repaired-dataset", required=True)
    parser.add_argument("--repaired-dataset-alias", default=None)
    parser.add_argument("--pretrained-asset", required=True)
    parser.add_argument("--pretrained-asset-alias", default=None)
    parser.add_argument("--output", required=True,
                        help="acceptance JSON path; must not exist")
    args = parser.parse_args()

    output_path = Path(args.output)
    require(not output_path.exists(), f"output already exists: {output_path}")
    prediction_root = Path(args.prediction_root)
    protocol_dir = Path(args.protocol_dir)

    # ── Receipt chain, identical to the run wrapper ──────────────────
    rebuild_path = Path(args.g1_repaired_dataset_receipt)
    g1_path = Path(args.g1_acceptance_receipt)
    g1c_path = Path(args.g1c_pretraining_receipt)
    rebuild_payload = json.loads(rebuild_path.read_text())
    validate_repaired_dataset_receipt(rebuild_payload)
    rebuild_sha256 = file_sha256(rebuild_path)
    g1_payload = json.loads(g1_path.read_text())
    validate_g1_acceptance_receipt(
        g1_payload,
        repaired_receipt_sha256=rebuild_sha256,
        repaired_output_sha256=rebuild_payload["output"]["sha256"],
    )
    g1c_payload = json.loads(g1c_path.read_text())
    validate_g1c_pretraining_receipt(g1c_payload, g1_payload)
    lineage_sha256 = file_sha256(g1c_path)
    require(
        g1_payload["corrected_pretraining_lineage_sha256"] == lineage_sha256,
        "G1 acceptance lineage does not match the G1C receipt file hash",
    )
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

    # ── Collect and verify every run ─────────────────────────────────
    code_commit: str | None = None
    recipe_sha256: str | None = None
    model_source_sha256: str | None = None
    prediction_sources: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for regime in CANONICAL_REGIMES:
        prediction_sources[f"{regime}_cv"] = {}
        for fold in CANONICAL_FOLDS:
            split_id = f"{regime}_cv5_f{fold}"
            split_file = protocol_dir / "splits" / f"{split_id}.json"
            split_payload = json.loads(split_file.read_text())
            validate_split_payload(split_payload, split_id)
            split_sha256 = file_sha256(split_file)
            entries: list[dict[str, Any]] = []
            for seed in sorted(CANONICAL_PREDICTION_SEEDS[regime]):
                run_dir = prediction_root / split_id / f"seed{seed}"
                manifest_path = run_dir / "g2_run_manifest.json"
                require(
                    manifest_path.is_file(),
                    f"missing G2 run manifest: {manifest_path}",
                )
                manifest = json.loads(manifest_path.read_text())
                run_commit = str(manifest.get("git", {}).get("commit", ""))
                run_kwargs = manifest.get("config", {}).get("model_kwargs", {})
                run_recipe = canonical_json_sha256(
                    json.loads(json.dumps(run_kwargs))
                )
                run_source = str(manifest.get("model_source_sha256", ""))
                if code_commit is None:
                    code_commit = run_commit
                    recipe_sha256 = run_recipe
                    model_source_sha256 = run_source
                require(
                    run_commit == code_commit
                    and run_recipe == recipe_sha256
                    and run_source == model_source_sha256,
                    f"run {split_id}/seed{seed} breaks commit/recipe/source "
                    "uniformity across the queue",
                )
                validate_g2_run_manifest(
                    manifest,
                    code_commit=code_commit,
                    recipe_sha256=recipe_sha256,
                    model_source_sha256=model_source_sha256,
                    repaired_data_sha256=repaired_sha256,
                    pretrained_asset_sha256=asset_sha256,
                    pretrained_asset_lineage_sha256=lineage_sha256,
                    split_id=split_id,
                    split_sha256=split_sha256,
                    split_counts=split_payload["counts"],
                    seed=seed,
                )
                prediction_path = run_dir / "test_predictions.npz"
                declared = str(manifest["outputs"]["prediction_sha256"])
                require(
                    prediction_path.is_file()
                    and file_sha256(prediction_path) == declared,
                    f"prediction hash mismatch: {prediction_path}",
                )
                entries.append({
                    "seed": seed,
                    "split_sha256": split_sha256,
                    "prediction_path": contract_path(
                        prediction_path, prediction_root, args.run_root_alias,
                    ),
                    "prediction_sha256": declared,
                    "run_manifest_path": contract_path(
                        manifest_path, prediction_root, args.run_root_alias,
                    ),
                    "run_manifest_sha256": file_sha256(manifest_path),
                })
            prediction_sources[f"{regime}_cv"][str(fold)] = entries

    assert code_commit and recipe_sha256 and model_source_sha256

    # ── Git bindings the consumer will re-check ──────────────────────
    require(
        model_source_sha256 == g1_payload["model_source_sha256"],
        "queue model source differs from the G1-accepted source",
    )
    blob_id = git_output(
        "rev-parse", f"{code_commit}:src/models/crystal_v2.py",
    )
    blob_content = subprocess.run(
        ["git", "cat-file", "blob", blob_id],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout
    require(
        hashlib.sha256(blob_content).hexdigest() == model_source_sha256,
        "model source at the G2 commit differs from the run-recorded hash",
    )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor",
         g1_payload["code_commit"], code_commit],
        cwd=ROOT, check=False,
    )
    require(
        ancestry.returncode == 0,
        "the accepted G1 repair commit is not an ancestor of the G2 commit",
    )
    remotes = [
        line.strip()
        for line in git_output(
            "branch", "-r", "--contains", code_commit
        ).splitlines()
        if line.strip()
    ]
    require(
        bool(remotes),
        "the G2 commit is not contained in any fetched remote branch",
    )

    acceptance = build_g2_acceptance(
        code_commit=code_commit,
        model_source_sha256=model_source_sha256,
        model_recipe_sha256=recipe_sha256,
        pretrained_asset_lineage_sha256=lineage_sha256,
        pretrained_asset={
            "path": contract_path(
                asset_path, prediction_root, args.run_root_alias,
                args.pretrained_asset_alias,
            ),
            "sha256": asset_sha256,
        },
        repaired_dataset={
            "source_data_sha256": EXPECTED_SOURCE_DATA_SHA256,
            "graph_dataset_sha256": repaired_sha256,
            "path": contract_path(
                repaired_path, prediction_root, args.run_root_alias,
                args.repaired_dataset_alias,
            ),
        },
        g1_receipts={
            "repaired_dataset_receipt": {
                "path": contract_path(
                    rebuild_path, prediction_root, args.run_root_alias,
                    args.g1_repaired_dataset_receipt_alias,
                ),
                "sha256": rebuild_sha256,
            },
            "acceptance_receipt": {
                "path": contract_path(
                    g1_path, prediction_root, args.run_root_alias,
                    args.g1_acceptance_receipt_alias,
                ),
                "sha256": file_sha256(g1_path),
            },
            "corrected_pretraining_receipt": {
                "path": contract_path(
                    g1c_path, prediction_root, args.run_root_alias,
                    args.g1c_pretraining_receipt_alias,
                ),
                "sha256": lineage_sha256,
            },
        },
        prediction_sources=prediction_sources,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "x", encoding="utf-8") as handle:
        json.dump(acceptance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"prm_g2_acceptance_v1 written: {output_path}")
    print(f"code_commit: {code_commit}")
    print(f"model_recipe_sha256: {recipe_sha256}")


if __name__ == "__main__":
    main()
