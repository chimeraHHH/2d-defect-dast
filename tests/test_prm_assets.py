from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from src.models import CrystalTransformerV2
from src.prm_assets import (
    file_sha256,
    load_pretrained_initialization,
    verify_training_assets,
)
from src.prm_provenance import validate_dart_assets


def _model(*, prenorm: bool, ct_uae_path: str | None = None) -> CrystalTransformerV2:
    return CrystalTransformerV2(
        atom_fea_len=9,
        hidden_dim=16,
        n_local_layers=1,
        n_global_layers=1,
        num_heads=4,
        ct_uae_path=ct_uae_path,
        use_gated_pooling=False,
        use_env_enrichment=False,
        use_prenorm_local=prenorm,
    )


def _checkpoint(tmp_path):
    source = _model(prenorm=False)
    path = tmp_path / "pretrained.pt"
    torch.save(
        {
            "embed_weight": source.embed.weight.detach().clone(),
            "embed_bias": source.embed.bias.detach().clone(),
            "local_layers": source.local_layers.state_dict(),
            "source_dataset": "synthetic_pretraining",
            "source_test_mae": 0.125,
        },
        path,
    )
    return source, path


def test_pretraining_load_is_exact_and_reports_seeded_gate_parameters(tmp_path):
    source, checkpoint_path = _checkpoint(tmp_path)
    ct_uae_path = tmp_path / "ct_uae.pt"
    torch.save(torch.zeros(100, 3), ct_uae_path)
    target = _model(prenorm=True, ct_uae_path=str(ct_uae_path))
    seeded_columns = target.embed.weight[:, 9:].detach().clone()

    report = load_pretrained_initialization(target, checkpoint_path)

    assert torch.equal(target.embed.weight[:, :9], source.embed.weight)
    assert torch.equal(target.embed.weight[:, 9:], seeded_columns)
    assert torch.equal(target.embed.bias, source.embed.bias)
    assert report["input_projection"]["checkpoint_weight_shape"] == [16, 9]
    assert report["input_projection"]["model_weight_shape"] == [16, 12]
    assert report["input_projection"]["seeded_model_columns"] == 3
    assert report["local_layers"]["loaded_tensor_count"] == 17
    assert len(report["local_layers"]["seeded_model_keys"]) == 4
    assert all(".edge_gate." in key for key in report["local_layers"]["seeded_model_keys"])


def test_pretraining_load_rejects_input_shape_drift(tmp_path):
    _, checkpoint_path = _checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["embed_weight"] = checkpoint["embed_weight"][:, :8]
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="input weight shape mismatch"):
        load_pretrained_initialization(_model(prenorm=False), checkpoint_path)


def test_required_training_assets_enforce_both_hashes(tmp_path):
    _, checkpoint_path = _checkpoint(tmp_path)
    ct_uae_path = tmp_path / "ct_uae.pt"
    torch.save(torch.zeros(100, 3), ct_uae_path)
    config = {
        "asset_integrity_required": True,
        "asset_sha256": {
            "ct_uae": file_sha256(ct_uae_path),
            "pretrained_embed": file_sha256(checkpoint_path),
        },
        "model_kwargs": {"ct_uae_path": str(ct_uae_path)},
        "pretrained_embed": str(checkpoint_path),
    }

    records = verify_training_assets(config)
    assert records["ct_uae"]["sha256"] == config["asset_sha256"]["ct_uae"]
    assert records["pretrained_embed"]["expected_sha256"] == config["asset_sha256"]["pretrained_embed"]

    invalid = deepcopy(config)
    del invalid["asset_sha256"]["ct_uae"]
    with pytest.raises(ValueError, match="incomplete required asset contract"):
        verify_training_assets(invalid)


def test_collector_asset_validation_accepts_only_the_audited_load(tmp_path):
    _, checkpoint_path = _checkpoint(tmp_path)
    ct_uae_path = tmp_path / "ct_uae.pt"
    torch.save(torch.zeros(100, 3), ct_uae_path)
    target = _model(prenorm=True, ct_uae_path=str(ct_uae_path))
    report = load_pretrained_initialization(target, checkpoint_path)
    hashes = {
        "ct_uae": file_sha256(ct_uae_path),
        "pretrained_embed": file_sha256(checkpoint_path),
    }
    manifest = {
        "config": {
            "asset_integrity_required": True,
            "asset_sha256": hashes,
            "model_kwargs": {
                "atom_fea_len": 9,
                "hidden_dim": 16,
                "n_local_layers": 1,
                "use_prenorm_local": True,
            },
        },
        "assets": {
            "ct_uae": {
                "sha256": hashes["ct_uae"],
                "expected_sha256": hashes["ct_uae"],
                "size_bytes": ct_uae_path.stat().st_size,
            },
            "pretrained_embed": {
                "sha256": hashes["pretrained_embed"],
                "expected_sha256": hashes["pretrained_embed"],
                "size_bytes": checkpoint_path.stat().st_size,
            },
        },
        "pretraining": report,
    }
    validate_dart_assets(manifest, tmp_path / "run_manifest.json")

    invalid = deepcopy(manifest)
    invalid["pretraining"]["local_layers"]["seeded_model_keys"][0] = "0.norm.weight"
    with pytest.raises(ValueError, match="non-gate"):
        validate_dart_assets(invalid, tmp_path / "run_manifest.json")
