"""Fail-closed helpers for the prm_g2 repaired-OOF evidence chain.

This module is the single producer-side implementation of the
``prm_g2_run_manifest_v1`` and ``prm_g2_acceptance_v1`` schemas consumed by
``scripts/prm_g3_physics_analysis.py``.  Every function raises ``ValueError``
on the first violated binding; nothing is repaired or defaulted silently.

The module is deliberately importable without torch/numpy so its logic is
unit-testable on any machine.  Field semantics follow
``paper_Q1/review/g3_g2_acceptance_contract.md``; this module must never relax
a check that the consumer performs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

G2_RUN_MANIFEST_SCHEMA = "prm_g2_run_manifest_v1"
G2_ACCEPTANCE_SCHEMA = "prm_g2_acceptance_v1"
G2_MILESTONE = "J-R2-repaired-core-oof"
GRAPH_BUILDER_VERSION = "exact_mic_invariant_triplets_v1"
GRAPH_STATUS = "repaired_exact_mic_invariant_triplets"
CANONICAL_ENV_ZERO_NEIGHBOR_MODE = "zero_residual_v1"
INITIALIZATION_ARM = "corrected_pretrain"
MODEL_RECIPE_SCOPE = "config.model_kwargs"
OOF_INFERENCE_BATCH_SIZE = 64
FEATURE_CUTOFF_A = 5.0

EXPECTED_SOURCE_DATA_SHA256 = (
    "1d59cc818d81252d49da525c6d77e2da8549ceb604d54af953d1caa4fb974a9b"
)
EXPECTED_RAW_DB_SHA256 = (
    "3a71db999b477112da248dcf762c4384e455689953679d58b3d71a91e7148fc4"
)
EXPECTED_ATOM_FEATURE_SHA256 = (
    "5fa97d7788ef9b1e10be6d874aa9c6c7d55c532d055c187bb9452659983c5bb4"
)

EXPECTED_ROWS = 10_641
EXPECTED_CANONICAL_ROWS = 10_224
EXPECTED_EXCLUDED_ROWS = 417
EXPECTED_JARVIS_ROWS = 19_902
PRETRAINING_SEED = 42
PRETRAINING_EPOCHS = 30

CANONICAL_PREDICTION_SEEDS: dict[str, frozenset[int]] = {
    "pair": frozenset({242}),
    "host": frozenset({242, 243, 244}),
    "dopant": frozenset({242, 243, 244}),
}
CANONICAL_REGIMES = ("pair", "host", "dopant")
CANONICAL_FOLDS = tuple(range(5))
# Paper-facing repaired reruns outside the G2 acceptance contract: the
# random-interpolation folds and the predefined chemistry block, matching the
# legacy campaign's single-seed law.  The acceptance builder never reads
# these regimes; they exist so the same fail-closed wrapper can produce the
# supplementary core numbers.
PAPER_PREDICTION_SEEDS: dict[str, frozenset[int]] = {
    "id": frozenset({242}),
    "chemistry_block": frozenset({242}),
}
ALL_PREDICTION_SEEDS: dict[str, frozenset[int]] = {
    **CANONICAL_PREDICTION_SEEDS, **PAPER_PREDICTION_SEEDS,
}


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash one JSON value with the frozen contract encoding.

    Must remain byte-identical to the consumer implementation in
    ``scripts/prm_g3_physics_analysis.py``.
    """
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hex64(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def _hex40(value: Any) -> bool:
    text = str(value)
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text)


def validate_repaired_dataset_receipt(payload: Mapping[str, Any]) -> None:
    """Mirror of the consumer's ``prm_g1_repaired_dataset_v1`` checks."""
    require(
        payload.get("schema_version") == "prm_g1_repaired_dataset_v1",
        "repaired-dataset receipt schema mismatch",
    )
    inputs = payload.get("inputs", {})
    require(
        inputs.get("dataset", {}).get("sha256") == EXPECTED_SOURCE_DATA_SHA256,
        "repaired-dataset receipt source-data hash mismatch",
    )
    require(
        inputs.get("raw_db", {}).get("sha256") == EXPECTED_RAW_DB_SHA256,
        "repaired-dataset receipt raw-db hash mismatch",
    )
    container = payload.get("container", {})
    require(
        container.get("rows") == EXPECTED_ROWS
        and container.get("canonical_rows") == EXPECTED_CANONICAL_ROWS
        and container.get("excluded_rows") == EXPECTED_EXCLUDED_ROWS
        and container.get("order_preserved") is True
        and container.get("non_graph_fields_preserved") is True,
        "repaired-dataset receipt container contract violated",
    )
    builder = payload.get("graph_builder", {})
    require(
        builder.get("version") == GRAPH_BUILDER_VERSION
        and builder.get("cutoff_A") == FEATURE_CUTOFF_A,
        "repaired-dataset receipt graph-builder contract violated",
    )
    require(
        _hex64(payload.get("output", {}).get("sha256", "")),
        "repaired-dataset receipt lacks output hash",
    )


def validate_g1_acceptance_receipt(
    payload: Mapping[str, Any],
    repaired_receipt_sha256: str,
    repaired_output_sha256: str,
) -> None:
    """Mirror of the consumer's ``prm_g1_acceptance_v1`` checks."""
    require(
        payload.get("schema_version") == "prm_g1_acceptance_v1"
        and payload.get("status") == "accepted",
        "G1 acceptance receipt is not an accepted prm_g1_acceptance_v1",
    )
    require(
        payload.get("graph_builder_version") == GRAPH_BUILDER_VERSION
        and payload.get("source_data_sha256") == EXPECTED_SOURCE_DATA_SHA256
        and payload.get("env_zero_neighbor_mode")
        == CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
        "G1 acceptance receipt identity fields violate the frozen contract",
    )
    require(
        _hex64(payload.get("model_source_sha256", "")),
        "G1 acceptance receipt lacks model source hash",
    )
    require(
        payload.get("repaired_dataset_receipt_sha256") == repaired_receipt_sha256
        and payload.get("repaired_dataset_sha256") == repaired_output_sha256,
        "G1 acceptance receipt is not bound to the repaired-dataset receipt",
    )
    require(
        bool(payload.get("property_tests_passed"))
        and bool(payload.get("pilot_passed")),
        "G1 acceptance receipt does not record passed property tests and pilot",
    )
    require(
        _hex64(payload.get("corrected_pretraining_lineage_sha256", "")),
        "G1 acceptance receipt lacks corrected-pretraining lineage",
    )
    require(
        _hex40(payload.get("code_commit", "")),
        "G1 acceptance receipt lacks the repair commit",
    )


def validate_g1c_pretraining_receipt(
    payload: Mapping[str, Any], g1_payload: Mapping[str, Any],
) -> None:
    """Mirror of the consumer's ``prm_g1c_pretraining_receipt_v1`` checks."""
    require(
        payload.get("schema_version") == "prm_g1c_pretraining_receipt_v1",
        "G1C receipt schema mismatch",
    )
    git = payload.get("git", {})
    require(
        git.get("dirty") is False and _hex40(git.get("commit", "")),
        "G1C receipt does not record a clean 40-hex commit",
    )
    inputs = payload.get("input", {})
    require(
        inputs.get("rows") == EXPECTED_JARVIS_ROWS
        and inputs.get("graph_builder_version") == GRAPH_BUILDER_VERSION
        and inputs.get("dataset_receipt_sha256")
        == g1_payload.get("corrected_jarvis_dataset_receipt_sha256")
        and _hex64(inputs.get("dataset_sha256", "")),
        "G1C receipt input lineage violates the frozen contract",
    )
    require(
        _hex64(payload.get("outputs", {}).get("corrected_asset_sha256", "")),
        "G1C receipt lacks the corrected asset hash",
    )
    contract = payload.get("training_contract", {})
    require(
        contract.get("seed") == PRETRAINING_SEED
        and contract.get("epochs") == PRETRAINING_EPOCHS,
        "G1C receipt training contract violates the frozen recipe",
    )


def validate_split_payload(payload: Mapping[str, Any], split_id: str) -> None:
    """Identity checks the consumer applies to a protocol split JSON."""
    require(
        payload.get("schema_version") == "prm_split_v1"
        and payload.get("split_id") == split_id
        and int(payload.get("n_samples", -1)) == EXPECTED_ROWS
        and payload.get("data_sha256") == EXPECTED_SOURCE_DATA_SHA256,
        f"split identity mismatch for {split_id}",
    )
    counts = payload.get("counts", {})
    require(
        counts.get("excluded") == EXPECTED_EXCLUDED_ROWS
        and sum(int(counts.get(name, -1)) for name in ("train", "val", "test"))
        == EXPECTED_CANONICAL_ROWS,
        f"split counts violate the canonical population for {split_id}",
    )


def regime_of_split_id(split_id: str) -> str:
    for regime in CANONICAL_REGIMES:
        for fold in CANONICAL_FOLDS:
            if split_id == f"{regime}_cv5_f{fold}":
                return regime
    for fold in CANONICAL_FOLDS:
        if split_id == f"id_cv5_f{fold}":
            return "id"
    if split_id == "chemistry_block_g6x3d":
        return "chemistry_block"
    raise ValueError(f"split id is not a recognized G2 split: {split_id}")


def build_g2_run_manifest(
    *,
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    split_payload: Mapping[str, Any],
    split_sha256: str,
    repaired_data_sha256: str,
    model_source_sha256: str,
    pretrained_asset_sha256: str,
    pretrained_asset_lineage_sha256: str,
    atom_feature_table_sha256: str = EXPECTED_ATOM_FEATURE_SHA256,
    graph_builder_version: str = GRAPH_BUILDER_VERSION,
) -> dict[str, Any]:
    """Derive one ``prm_g2_run_manifest_v1`` from a verified completed run.

    Every value is re-checked against the consumer contract before emission;
    the function trusts nothing that is merely asserted by prose.
    """
    require(
        source_manifest.get("schema_version") == "prm_run_manifest_v1",
        "source run manifest schema mismatch",
    )
    require(
        source_manifest.get("status") == "complete",
        "source run is not complete",
    )
    git = source_manifest.get("git", {})
    require(
        git.get("dirty") is False and _hex40(git.get("commit", "")),
        "source run does not originate from a clean 40-hex commit",
    )
    seed = source_manifest.get("seed")
    require(isinstance(seed, int), "source run seed is not an integer")

    config = source_manifest.get("config")
    require(isinstance(config, Mapping), "source run lacks an embedded config")
    require(
        config.get("model") == "v2",
        "source run does not instantiate the frozen DART model",
    )
    require(
        int(config.get("batch_size", -1)) == OOF_INFERENCE_BATCH_SIZE,
        "source run batch size violates the frozen inference contract",
    )
    model_kwargs = config.get("model_kwargs")
    require(
        isinstance(model_kwargs, Mapping),
        "source run lacks embedded config.model_kwargs",
    )
    require(
        model_kwargs.get("env_zero_neighbor_mode")
        == CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
        "embedded model_kwargs does not bind zero_residual_v1",
    )
    require(
        model_kwargs.get("use_env_enrichment") is True,
        "embedded model_kwargs does not enable the frozen E module",
    )

    split_id = str(split_payload.get("split_id", ""))
    validate_split_payload(split_payload, split_id)
    regime = regime_of_split_id(split_id)
    require(
        int(seed) in ALL_PREDICTION_SEEDS[regime],
        f"seed {seed} is outside the frozen {regime} seed law",
    )
    source_split = source_manifest.get("split", {})
    require(
        source_split.get("split_id") == split_id
        and source_split.get("sha256") == split_sha256,
        "source run is not bound to the expected protocol split",
    )
    source_counts = source_split.get("counts", {})
    require(
        all(
            int(source_counts.get(name, -1)) == int(split_payload["counts"][name])
            for name in ("train", "val", "test")
        ),
        "source run split counts differ from the protocol split",
    )

    assets = source_manifest.get("assets", {})
    require(
        assets.get("pretrained_embed", {}).get("sha256")
        == pretrained_asset_sha256,
        "source run did not load the corrected initialization asset",
    )
    prediction_sha256 = str(
        source_manifest.get("output_sha256", {}).get("test_predictions", "")
    )
    require(_hex64(prediction_sha256), "source run lacks a prediction hash")
    require(
        _hex64(repaired_data_sha256)
        and _hex64(model_source_sha256)
        and _hex64(pretrained_asset_sha256)
        and _hex64(pretrained_asset_lineage_sha256)
        and _hex64(atom_feature_table_sha256),
        "one or more identity hashes are not 64-hex digests",
    )

    return {
        "schema_version": G2_RUN_MANIFEST_SCHEMA,
        "milestone": G2_MILESTONE,
        "status": "complete",
        "git": {"commit": str(git["commit"]), "dirty": False},
        "seed": int(seed),
        "split_id": split_id,
        "split_sha256": split_sha256,
        "split_counts": dict(split_payload["counts"]),
        "config": json.loads(json.dumps(config)),
        "env_zero_neighbor_mode": CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
        "model_recipe_scope": MODEL_RECIPE_SCOPE,
        "model_recipe_sha256": canonical_json_sha256(
            json.loads(json.dumps(model_kwargs))
        ),
        "model_source_sha256": model_source_sha256,
        "graph_builder_version": graph_builder_version,
        "atom_feature_table_sha256": atom_feature_table_sha256,
        "initialization_arm": INITIALIZATION_ARM,
        "pretrained_asset_sha256": pretrained_asset_sha256,
        "pretrained_asset_lineage_sha256": pretrained_asset_lineage_sha256,
        "data": {
            "data_sha256": repaired_data_sha256,
            "source_data_sha256": EXPECTED_SOURCE_DATA_SHA256,
        },
        "outputs": {
            "prediction_path": "test_predictions.npz",
            "prediction_sha256": prediction_sha256,
        },
        "generator": "scripts/prm_g2_run.py",
        "source_run_manifest_sha256": source_manifest_sha256,
    }


def validate_g2_run_manifest(
    manifest: Mapping[str, Any],
    *,
    code_commit: str,
    recipe_sha256: str,
    model_source_sha256: str,
    repaired_data_sha256: str,
    pretrained_asset_sha256: str,
    pretrained_asset_lineage_sha256: str,
    split_id: str,
    split_sha256: str,
    split_counts: Mapping[str, int],
    seed: int,
) -> None:
    """Re-run every consumer-side condition against one G2 run manifest."""
    require(
        manifest.get("schema_version") == G2_RUN_MANIFEST_SCHEMA
        and manifest.get("status") == "complete"
        and manifest.get("git", {}).get("dirty") is False
        and manifest.get("git", {}).get("commit") == code_commit
        and manifest.get("split_id") == split_id
        and manifest.get("split_sha256") == split_sha256
        and manifest.get("split_counts") == dict(split_counts)
        and int(manifest.get("seed", -1)) == int(seed)
        and int(manifest.get("config", {}).get("batch_size", -1))
        == OOF_INFERENCE_BATCH_SIZE,
        "G2 source run is incomplete, dirty, or violates its split/seed binding",
    )
    config = manifest.get("config")
    model_kwargs = config.get("model_kwargs") if isinstance(config, Mapping) else None
    require(
        isinstance(model_kwargs, Mapping),
        "G2 run lacks embedded config.model_kwargs",
    )
    require(
        manifest.get("env_zero_neighbor_mode") == CANONICAL_ENV_ZERO_NEIGHBOR_MODE
        and model_kwargs.get("env_zero_neighbor_mode")
        == CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
        "G2 top-level and embedded zero-neighbor modes must both be zero_residual_v1",
    )
    require(
        config.get("model") == "v2"
        and model_kwargs.get("use_env_enrichment") is True,
        "G2 run does not instantiate the frozen DART E module",
    )
    require(
        manifest.get("model_recipe_scope") == MODEL_RECIPE_SCOPE
        and manifest.get("model_recipe_sha256") == recipe_sha256
        and canonical_json_sha256(json.loads(json.dumps(model_kwargs)))
        == recipe_sha256,
        "G2 model recipe is not the hash of the embedded model_kwargs",
    )
    require(
        manifest.get("model_source_sha256") == model_source_sha256,
        "G2 run model source differs from the accepted source",
    )
    require(
        manifest.get("data", {}).get("data_sha256") == repaired_data_sha256,
        "G2 run repaired-data hash mismatch",
    )
    require(
        manifest.get("graph_builder_version") == GRAPH_BUILDER_VERSION,
        "G2 run graph-builder mismatch",
    )
    require(
        manifest.get("atom_feature_table_sha256") == EXPECTED_ATOM_FEATURE_SHA256,
        "G2 run atom-feature-table mismatch",
    )
    require(
        manifest.get("pretrained_asset_lineage_sha256")
        == pretrained_asset_lineage_sha256,
        "G2 run pretraining-asset lineage mismatch",
    )
    require(
        manifest.get("initialization_arm") == INITIALIZATION_ARM
        and manifest.get("pretrained_asset_sha256") == pretrained_asset_sha256,
        "G2 run is not bound to the corrected initialization asset",
    )
    require(
        _hex64(manifest.get("outputs", {}).get("prediction_sha256", "")),
        "G2 run lacks a prediction hash",
    )


def validate_seed_entries(regime: str, seeds: list[int]) -> None:
    expected = CANONICAL_PREDICTION_SEEDS[regime]
    require(
        len(seeds) == len(expected)
        and len(set(seeds)) == len(seeds)
        and set(seeds) == expected,
        f"canonical {regime} seed ensemble changed",
    )


def build_g2_acceptance(
    *,
    code_commit: str,
    model_source_sha256: str,
    model_recipe_sha256: str,
    pretrained_asset_lineage_sha256: str,
    pretrained_asset: Mapping[str, str],
    repaired_dataset: Mapping[str, str],
    g1_receipts: Mapping[str, Mapping[str, str]],
    prediction_sources: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Assemble one ``prm_g2_acceptance_v1`` from verified components.

    ``prediction_sources`` must already contain per-fold entry dicts with
    ``seed``, ``split_sha256``, ``prediction_path``, ``prediction_sha256``,
    ``run_manifest_path``, ``run_manifest_sha256``.  The caller is responsible
    for having validated every run manifest with
    :func:`validate_g2_run_manifest` first; this function re-checks structure
    and the seed law.
    """
    require(_hex40(code_commit), "acceptance code commit is not 40-hex")
    require(
        _hex64(model_source_sha256)
        and _hex64(model_recipe_sha256)
        and _hex64(pretrained_asset_lineage_sha256),
        "acceptance identity hashes are not 64-hex digests",
    )
    for name in ("repaired_dataset_receipt", "acceptance_receipt",
                 "corrected_pretraining_receipt"):
        spec = g1_receipts.get(name, {})
        require(
            bool(spec.get("path")) and _hex64(spec.get("sha256", "")),
            f"acceptance is missing the {name} binding",
        )
    require(
        repaired_dataset.get("source_data_sha256") == EXPECTED_SOURCE_DATA_SHA256
        and _hex64(repaired_dataset.get("graph_dataset_sha256", ""))
        and bool(repaired_dataset.get("path")),
        "acceptance repaired-dataset identity is incomplete",
    )
    require(
        bool(pretrained_asset.get("path"))
        and _hex64(pretrained_asset.get("sha256", "")),
        "acceptance corrected-asset binding is incomplete",
    )

    entry_fields = (
        "seed", "split_sha256", "prediction_path", "prediction_sha256",
        "run_manifest_path", "run_manifest_sha256",
    )
    sources: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for regime in CANONICAL_REGIMES:
        folds = prediction_sources.get(f"{regime}_cv")
        require(
            isinstance(folds, Mapping)
            and set(folds) == {str(fold) for fold in CANONICAL_FOLDS},
            f"acceptance {regime} folds are incomplete",
        )
        sources[f"{regime}_cv"] = {}
        for fold in CANONICAL_FOLDS:
            entries = folds[str(fold)]
            require(
                isinstance(entries, list) and len(entries) > 0,
                f"acceptance {regime} fold {fold} has no sources",
            )
            for entry in entries:
                require(
                    all(field in entry for field in entry_fields),
                    f"acceptance {regime} fold {fold} entry is missing fields",
                )
            validate_seed_entries(
                regime, [int(entry["seed"]) for entry in entries],
            )
            sources[f"{regime}_cv"][str(fold)] = [
                {field: entry[field] for field in entry_fields}
                for entry in entries
            ]

    return {
        "schema_version": G2_ACCEPTANCE_SCHEMA,
        "milestone": G2_MILESTONE,
        "status": "accepted",
        "graph_status": GRAPH_STATUS,
        "data_sha256": EXPECTED_SOURCE_DATA_SHA256,
        "repaired_dataset": dict(repaired_dataset),
        "g1": {name: dict(spec) for name, spec in g1_receipts.items()},
        "code_commit": code_commit,
        "model_source_sha256": model_source_sha256,
        "model_recipe_scope": MODEL_RECIPE_SCOPE,
        "model_recipe_sha256": model_recipe_sha256,
        "pretrained_asset_lineage_sha256": pretrained_asset_lineage_sha256,
        "initialization_arm": INITIALIZATION_ARM,
        "pretrained_asset": dict(pretrained_asset),
        "atom_feature_table_sha256": EXPECTED_ATOM_FEATURE_SHA256,
        "env_zero_neighbor_mode": CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
        "prediction_sources": sources,
    }
