"""Unit tests for the prm_g2 producer-side contract helpers.

Pure standard-library tests: no torch, numpy, or filesystem fixtures.  They
assert that the producer mirrors every consumer-side condition in
``scripts/prm_g3_physics_analysis.py`` and fails closed on each violated
binding.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prm_g2_contract import (  # noqa: E402
    CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
    EXPECTED_ATOM_FEATURE_SHA256,
    EXPECTED_RAW_DB_SHA256,
    EXPECTED_SOURCE_DATA_SHA256,
    G2_ACCEPTANCE_SCHEMA,
    G2_RUN_MANIFEST_SCHEMA,
    build_g2_acceptance,
    build_g2_run_manifest,
    canonical_json_sha256,
    regime_of_split_id,
    validate_g1_acceptance_receipt,
    validate_g1c_pretraining_receipt,
    validate_g2_run_manifest,
    validate_repaired_dataset_receipt,
    validate_seed_entries,
    validate_split_payload,
)


def hex64(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


SPLIT_SHA = hex64("split")
PRED_SHA = hex64("pred")
ASSET_SHA = hex64("asset")
LINEAGE_SHA = hex64("lineage")
REPAIRED_SHA = hex64("repaired")
MODEL_SRC_SHA = hex64("model-src")
SOURCE_MANIFEST_SHA = hex64("source-manifest")
JARVIS_RECEIPT_SHA = hex64("jarvis-receipt")
REBUILD_RECEIPT_SHA = hex64("rebuild-receipt")
COMMIT = "a" * 40
G1_COMMIT = "b" * 40
G1C_COMMIT = "c" * 40

COUNTS = {"train": 6134, "val": 2045, "test": 2045, "excluded": 417}


def make_split_payload(split_id: str = "pair_cv5_f0") -> dict:
    return {
        "schema_version": "prm_split_v1",
        "split_id": split_id,
        "n_samples": 10_641,
        "data_sha256": EXPECTED_SOURCE_DATA_SHA256,
        "counts": dict(COUNTS),
    }


def make_source_manifest(seed: int = 242,
                         split_id: str = "pair_cv5_f0") -> dict:
    return {
        "schema_version": "prm_run_manifest_v1",
        "status": "complete",
        "git": {"commit": COMMIT, "dirty": False},
        "seed": seed,
        "config": {
            "model": "v2",
            "batch_size": 64,
            "model_kwargs": {
                "env_zero_neighbor_mode": CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
                "use_env_enrichment": True,
                "hidden_dim": 128,
            },
        },
        "split": {
            "split_id": split_id,
            "sha256": SPLIT_SHA,
            "counts": {name: COUNTS[name] for name in ("train", "val", "test")},
        },
        "assets": {"pretrained_embed": {"sha256": ASSET_SHA}},
        "output_sha256": {"test_predictions": PRED_SHA},
    }


def build_manifest(**overrides) -> dict:
    kwargs = dict(
        source_manifest=make_source_manifest(),
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        split_payload=make_split_payload(),
        split_sha256=SPLIT_SHA,
        repaired_data_sha256=REPAIRED_SHA,
        model_source_sha256=MODEL_SRC_SHA,
        pretrained_asset_sha256=ASSET_SHA,
        pretrained_asset_lineage_sha256=LINEAGE_SHA,
    )
    kwargs.update(overrides)
    return build_g2_run_manifest(**kwargs)


def make_rebuild_receipt() -> dict:
    return {
        "schema_version": "prm_g1_repaired_dataset_v1",
        "inputs": {
            "dataset": {"sha256": EXPECTED_SOURCE_DATA_SHA256},
            "raw_db": {"sha256": EXPECTED_RAW_DB_SHA256},
        },
        "container": {
            "rows": 10_641,
            "canonical_rows": 10_224,
            "excluded_rows": 417,
            "order_preserved": True,
            "non_graph_fields_preserved": True,
        },
        "graph_builder": {
            "version": "exact_mic_invariant_triplets_v1",
            "cutoff_A": 5.0,
        },
        "output": {"sha256": REPAIRED_SHA},
    }


def make_g1_acceptance() -> dict:
    return {
        "schema_version": "prm_g1_acceptance_v1",
        "status": "accepted",
        "graph_builder_version": "exact_mic_invariant_triplets_v1",
        "source_data_sha256": EXPECTED_SOURCE_DATA_SHA256,
        "env_zero_neighbor_mode": CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
        "model_source_sha256": MODEL_SRC_SHA,
        "repaired_dataset_receipt_sha256": REBUILD_RECEIPT_SHA,
        "repaired_dataset_sha256": REPAIRED_SHA,
        "property_tests_passed": True,
        "pilot_passed": True,
        "corrected_pretraining_lineage_sha256": LINEAGE_SHA,
        "code_commit": G1_COMMIT,
        "corrected_jarvis_dataset_receipt_sha256": JARVIS_RECEIPT_SHA,
    }


def make_g1c_receipt() -> dict:
    return {
        "schema_version": "prm_g1c_pretraining_receipt_v1",
        "git": {"commit": G1C_COMMIT, "dirty": False},
        "input": {
            "rows": 19_902,
            "graph_builder_version": "exact_mic_invariant_triplets_v1",
            "dataset_receipt_sha256": JARVIS_RECEIPT_SHA,
            "dataset_sha256": hex64("jarvis-data"),
        },
        "outputs": {"corrected_asset_sha256": ASSET_SHA},
        "training_contract": {"seed": 42, "epochs": 30},
    }


# ── recipe hashing ────────────────────────────────────────────────────


def test_canonical_json_sha256_uses_frozen_encoding() -> None:
    value = {"b": 1, "a": [True, None, "x"]}
    expected = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert canonical_json_sha256(value) == expected
    assert canonical_json_sha256({"a": [True, None, "x"], "b": 1}) == expected


def test_canonical_json_sha256_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json_sha256({"x": float("nan")})


# ── run-manifest construction ────────────────────────────────────────


def test_build_g2_run_manifest_happy_path() -> None:
    manifest = build_manifest()
    assert manifest["schema_version"] == G2_RUN_MANIFEST_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["git"] == {"commit": COMMIT, "dirty": False}
    assert manifest["split_id"] == "pair_cv5_f0"
    assert manifest["split_sha256"] == SPLIT_SHA
    assert manifest["split_counts"] == COUNTS
    assert manifest["seed"] == 242
    assert manifest["config"]["batch_size"] == 64
    assert manifest["env_zero_neighbor_mode"] == CANONICAL_ENV_ZERO_NEIGHBOR_MODE
    assert manifest["model_recipe_scope"] == "config.model_kwargs"
    assert manifest["model_recipe_sha256"] == canonical_json_sha256(
        manifest["config"]["model_kwargs"]
    )
    assert manifest["model_source_sha256"] == MODEL_SRC_SHA
    assert manifest["atom_feature_table_sha256"] == EXPECTED_ATOM_FEATURE_SHA256
    assert manifest["initialization_arm"] == "corrected_pretrain"
    assert manifest["pretrained_asset_sha256"] == ASSET_SHA
    assert manifest["pretrained_asset_lineage_sha256"] == LINEAGE_SHA
    assert manifest["data"]["data_sha256"] == REPAIRED_SHA
    assert manifest["outputs"]["prediction_sha256"] == PRED_SHA


def test_built_manifest_passes_consumer_mirror() -> None:
    manifest = build_manifest()
    validate_g2_run_manifest(
        manifest,
        code_commit=COMMIT,
        recipe_sha256=manifest["model_recipe_sha256"],
        model_source_sha256=MODEL_SRC_SHA,
        repaired_data_sha256=REPAIRED_SHA,
        pretrained_asset_sha256=ASSET_SHA,
        pretrained_asset_lineage_sha256=LINEAGE_SHA,
        split_id="pair_cv5_f0",
        split_sha256=SPLIT_SHA,
        split_counts=COUNTS,
        seed=242,
    )


@pytest.mark.parametrize("mutate, message_part", [
    (lambda m: m.update(status="truncated"), "not complete"),
    (lambda m: m["git"].update(dirty=True), "clean"),
    (lambda m: m["config"].update(batch_size=32), "batch size"),
    (lambda m: m["config"].update(model="v1"), "frozen DART model"),
    (lambda m: m["config"]["model_kwargs"].update(
        env_zero_neighbor_mode="legacy_batch_dependent_v0"), "zero_residual_v1"),
    (lambda m: m["config"]["model_kwargs"].update(use_env_enrichment=False),
     "E module"),
    (lambda m: m.update(seed=7), "seed law"),
    (lambda m: m["split"].update(sha256=hex64("other-split")),
     "protocol split"),
    (lambda m: m["split"]["counts"].update(test=2044), "split counts"),
    (lambda m: m["assets"]["pretrained_embed"].update(sha256=hex64("wrong")),
     "corrected initialization asset"),
    (lambda m: m["output_sha256"].pop("test_predictions"), "prediction hash"),
])
def test_build_g2_run_manifest_fails_closed(mutate, message_part) -> None:
    source = make_source_manifest()
    mutate(source)
    with pytest.raises(ValueError, match=message_part):
        build_manifest(source_manifest=source)


def test_build_rejects_host_seed_on_pair_split() -> None:
    source = make_source_manifest(seed=243)
    with pytest.raises(ValueError, match="seed law"):
        build_manifest(source_manifest=source)


def test_build_accepts_host_seed_on_host_split() -> None:
    source = make_source_manifest(seed=243, split_id="host_cv5_f1")
    manifest = build_manifest(
        source_manifest=source,
        split_payload=make_split_payload("host_cv5_f1"),
    )
    assert manifest["split_id"] == "host_cv5_f1"
    assert manifest["seed"] == 243


def test_validate_rejects_recipe_hash_drift() -> None:
    manifest = build_manifest()
    tampered = copy.deepcopy(manifest)
    tampered["config"]["model_kwargs"]["hidden_dim"] = 256
    with pytest.raises(ValueError, match="embedded model_kwargs"):
        validate_g2_run_manifest(
            tampered,
            code_commit=COMMIT,
            recipe_sha256=manifest["model_recipe_sha256"],
            model_source_sha256=MODEL_SRC_SHA,
            repaired_data_sha256=REPAIRED_SHA,
            pretrained_asset_sha256=ASSET_SHA,
            pretrained_asset_lineage_sha256=LINEAGE_SHA,
            split_id="pair_cv5_f0",
            split_sha256=SPLIT_SHA,
            split_counts=COUNTS,
            seed=242,
        )


# ── split and seed law ───────────────────────────────────────────────


def test_regime_of_split_id() -> None:
    assert regime_of_split_id("pair_cv5_f0") == "pair"
    assert regime_of_split_id("dopant_cv5_f4") == "dopant"
    with pytest.raises(ValueError):
        regime_of_split_id("random_s42")


def test_validate_split_payload_rejects_wrong_population() -> None:
    payload = make_split_payload()
    payload["counts"]["test"] = 2044
    with pytest.raises(ValueError, match="canonical population"):
        validate_split_payload(payload, "pair_cv5_f0")


def test_seed_law() -> None:
    validate_seed_entries("pair", [242])
    validate_seed_entries("host", [244, 242, 243])
    for regime, seeds in [
        ("pair", [242, 242]),
        ("pair", [243]),
        ("host", [242, 243]),
        ("host", [242, 243, 244, 244]),
        ("dopant", [242, 243, 245]),
    ]:
        with pytest.raises(ValueError):
            validate_seed_entries(regime, seeds)


# ── receipt validators ───────────────────────────────────────────────


def test_receipt_chain_happy_path() -> None:
    rebuild = make_rebuild_receipt()
    validate_repaired_dataset_receipt(rebuild)
    g1 = make_g1_acceptance()
    validate_g1_acceptance_receipt(
        g1,
        repaired_receipt_sha256=REBUILD_RECEIPT_SHA,
        repaired_output_sha256=REPAIRED_SHA,
    )
    validate_g1c_pretraining_receipt(make_g1c_receipt(), g1)


@pytest.mark.parametrize("mutate", [
    lambda r: r["container"].update(rows=10_640),
    lambda r: r["container"].update(excluded_rows=416),
    lambda r: r["container"].update(order_preserved=False),
    lambda r: r["graph_builder"].update(version="legacy_first32_v0"),
    lambda r: r["graph_builder"].update(cutoff_A=6.0),
    lambda r: r["inputs"]["dataset"].update(sha256=hex64("wrong-src")),
])
def test_rebuild_receipt_fails_closed(mutate) -> None:
    receipt = make_rebuild_receipt()
    mutate(receipt)
    with pytest.raises(ValueError):
        validate_repaired_dataset_receipt(receipt)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(status="draft"),
    lambda r: r.update(env_zero_neighbor_mode="legacy_batch_dependent_v0"),
    lambda r: r.update(property_tests_passed=False),
    lambda r: r.update(pilot_passed=False),
    lambda r: r.update(repaired_dataset_sha256=hex64("other")),
    lambda r: r.update(code_commit="deadbeef"),
])
def test_g1_acceptance_fails_closed(mutate) -> None:
    receipt = make_g1_acceptance()
    mutate(receipt)
    with pytest.raises(ValueError):
        validate_g1_acceptance_receipt(
            receipt,
            repaired_receipt_sha256=REBUILD_RECEIPT_SHA,
            repaired_output_sha256=REPAIRED_SHA,
        )


@pytest.mark.parametrize("mutate", [
    lambda r: r["git"].update(dirty=True),
    lambda r: r["input"].update(rows=19_901),
    lambda r: r["input"].update(dataset_receipt_sha256=hex64("other")),
    lambda r: r["training_contract"].update(seed=43),
    lambda r: r["training_contract"].update(epochs=29),
])
def test_g1c_receipt_fails_closed(mutate) -> None:
    receipt = make_g1c_receipt()
    mutate(receipt)
    with pytest.raises(ValueError):
        validate_g1c_pretraining_receipt(receipt, make_g1_acceptance())


# ── acceptance assembly ──────────────────────────────────────────────


def make_entry(seed: int) -> dict:
    return {
        "seed": seed,
        "split_sha256": SPLIT_SHA,
        "prediction_path": f"$PRM_RUN_ROOT/g2/x/seed{seed}/test_predictions.npz",
        "prediction_sha256": PRED_SHA,
        "run_manifest_path": f"$PRM_RUN_ROOT/g2/x/seed{seed}/g2_run_manifest.json",
        "run_manifest_sha256": hex64(f"manifest-{seed}"),
    }


def make_sources() -> dict:
    sources = {}
    for regime, seeds in [("pair", [242]), ("host", [242, 243, 244]),
                          ("dopant", [242, 243, 244])]:
        sources[f"{regime}_cv"] = {
            str(fold): [make_entry(seed) for seed in seeds]
            for fold in range(5)
        }
    return sources


def acceptance_kwargs(**overrides) -> dict:
    kwargs = dict(
        code_commit=COMMIT,
        model_source_sha256=MODEL_SRC_SHA,
        model_recipe_sha256=hex64("recipe"),
        pretrained_asset_lineage_sha256=LINEAGE_SHA,
        pretrained_asset={"path": "$PRM_G1C_ASSET_PATH", "sha256": ASSET_SHA},
        repaired_dataset={
            "source_data_sha256": EXPECTED_SOURCE_DATA_SHA256,
            "graph_dataset_sha256": REPAIRED_SHA,
            "path": "$PRM_REPAIRED_DATASET_PATH",
        },
        g1_receipts={
            "repaired_dataset_receipt": {
                "path": "artifacts/prm_g1/rebuild_receipt.json",
                "sha256": REBUILD_RECEIPT_SHA,
            },
            "acceptance_receipt": {
                "path": "artifacts/prm_g1/acceptance_receipt.json",
                "sha256": hex64("g1-receipt"),
            },
            "corrected_pretraining_receipt": {
                "path": "artifacts/prm_g1/g1c_receipt.json",
                "sha256": LINEAGE_SHA,
            },
        },
        prediction_sources=make_sources(),
    )
    kwargs.update(overrides)
    return kwargs


def test_build_g2_acceptance_happy_path() -> None:
    acceptance = build_g2_acceptance(**acceptance_kwargs())
    assert acceptance["schema_version"] == G2_ACCEPTANCE_SCHEMA
    assert acceptance["milestone"] == "J-R2-repaired-core-oof"
    assert acceptance["status"] == "accepted"
    assert acceptance["graph_status"] == "repaired_exact_mic_invariant_triplets"
    assert acceptance["data_sha256"] == EXPECTED_SOURCE_DATA_SHA256
    assert acceptance["initialization_arm"] == "corrected_pretrain"
    assert acceptance["env_zero_neighbor_mode"] == CANONICAL_ENV_ZERO_NEIGHBOR_MODE
    assert acceptance["atom_feature_table_sha256"] == EXPECTED_ATOM_FEATURE_SHA256
    assert set(acceptance["prediction_sources"]) == {
        "pair_cv", "host_cv", "dopant_cv",
    }
    assert len(acceptance["prediction_sources"]["host_cv"]["3"]) == 3
    assert len(acceptance["prediction_sources"]["pair_cv"]["0"]) == 1


def test_build_g2_acceptance_rejects_missing_fold() -> None:
    sources = make_sources()
    del sources["host_cv"]["2"]
    with pytest.raises(ValueError, match="host folds are incomplete"):
        build_g2_acceptance(**acceptance_kwargs(prediction_sources=sources))


def test_build_g2_acceptance_rejects_empty_fold() -> None:
    sources = make_sources()
    sources["dopant_cv"]["1"] = []
    with pytest.raises(ValueError, match="no sources"):
        build_g2_acceptance(**acceptance_kwargs(prediction_sources=sources))


def test_build_g2_acceptance_rejects_seed_violation() -> None:
    sources = make_sources()
    sources["host_cv"]["0"] = [make_entry(242), make_entry(242), make_entry(243)]
    with pytest.raises(ValueError, match="seed ensemble"):
        build_g2_acceptance(**acceptance_kwargs(prediction_sources=sources))


def test_build_g2_acceptance_rejects_missing_receipt_binding() -> None:
    receipts = acceptance_kwargs()["g1_receipts"]
    receipts["corrected_pretraining_receipt"] = {"path": "", "sha256": ""}
    with pytest.raises(ValueError, match="corrected_pretraining_receipt"):
        build_g2_acceptance(**acceptance_kwargs(g1_receipts=receipts))
