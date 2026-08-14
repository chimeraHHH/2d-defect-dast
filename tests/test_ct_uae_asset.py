from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.prm_verify_ct_uae_asset import (
    derive_ct_uae_asset,
    tensor_sha256,
    verify_ct_uae_derivation,
)
from src.models.element_table import lookup_ct_uae, prepare_ct_uae_table
from src.prm_assets import file_sha256


def _assets(tmp_path: Path) -> tuple[Path, Path, torch.Tensor]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    weight = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 10
    bias = torch.tensor([0.25, -0.5, 0.75])
    table = weight.T + bias.unsqueeze(0)
    checkpoint = tmp_path / "checkpoint.pt"
    asset = tmp_path / "table.pt"
    torch.save(
        {
            "state_dict": {
                "atom_embed.weight": weight,
                "atom_embed.bias": bias,
            }
        },
        checkpoint,
    )
    torch.save(table, asset)
    return checkpoint, asset, table


def test_ct_uae_table_is_exact_one_hot_linear_projection(tmp_path):
    checkpoint, asset, table = _assets(tmp_path)
    record = verify_ct_uae_derivation(
        checkpoint,
        asset,
        expected_checkpoint_sha256=file_sha256(checkpoint),
        expected_asset_sha256=file_sha256(asset),
    )

    assert record["derivation"]["formula"] == (
        "atom_embed.weight.T + atom_embed.bias"
    )
    assert record["derivation"]["table_shape"] == list(table.shape)
    assert record["derivation"]["table_stride"] == [1, 5]
    assert record["derivation"]["exact"] is True


def test_ct_uae_asset_generation_preserves_canonical_layout(tmp_path):
    checkpoint, reference, table = _assets(tmp_path / "source")
    asset = tmp_path / "generated" / reference.name

    record = derive_ct_uae_asset(
        checkpoint,
        asset,
        expected_checkpoint_sha256=file_sha256(checkpoint),
        expected_asset_sha256=file_sha256(reference),
        expected_tensor_sha256=tensor_sha256(table),
    )
    generated = torch.load(asset, map_location="cpu", weights_only=True)

    assert file_sha256(asset) == file_sha256(reference)
    assert generated.stride() == table.stride()
    assert record["stride"] == [1, 5]
    assert record["tensor_sha256"] == tensor_sha256(table)


def test_ct_uae_asset_generation_refuses_unrequested_overwrite(tmp_path):
    checkpoint, asset, table = _assets(tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        derive_ct_uae_asset(
            checkpoint,
            asset,
            expected_checkpoint_sha256=file_sha256(checkpoint),
            expected_asset_sha256=file_sha256(asset),
            expected_tensor_sha256=tensor_sha256(table),
        )


def test_ct_uae_verification_rejects_a_different_table(tmp_path):
    checkpoint, asset, table = _assets(tmp_path)
    table[0, 0] += 1
    torch.save(table, asset)

    with pytest.raises(ValueError, match="not the exact"):
        verify_ct_uae_derivation(
            checkpoint,
            asset,
            expected_checkpoint_sha256=file_sha256(checkpoint),
            expected_asset_sha256=file_sha256(asset),
        )


def test_ct_uae_lookup_respects_one_based_atomic_numbers():
    source = torch.arange(300, dtype=torch.float32).reshape(100, 3)
    prepared = prepare_ct_uae_table(source)
    atomic_numbers = torch.tensor([[0, 1, 2, 100]])

    observed = lookup_ct_uae(prepared, atomic_numbers)

    assert prepared.shape == (101, 3)
    assert torch.equal(observed[0, 0], torch.zeros(3))
    assert torch.equal(observed[0, 1], source[0])
    assert torch.equal(observed[0, 2], source[1])
    assert torch.equal(observed[0, 3], source[99])


def test_ct_uae_lookup_rejects_out_of_range_atomic_numbers():
    prepared = prepare_ct_uae_table(torch.zeros(100, 3))
    with pytest.raises(IndexError):
        lookup_ct_uae(prepared, torch.tensor([101]))
