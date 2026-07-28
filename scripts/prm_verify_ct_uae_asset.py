"""Verify the exact derivation of the fixed ct-UAE elemental table."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prm_assets import verify_asset


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def verify_ct_uae_derivation(
    checkpoint_path: Path,
    asset_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_asset_sha256: str,
) -> Dict[str, Any]:
    checkpoint_record = verify_asset(
        checkpoint_path,
        label="ct-UAE source checkpoint",
        expected_sha256=expected_checkpoint_sha256,
    )
    asset_record = verify_asset(
        asset_path,
        label="ct-UAE derived table",
        expected_sha256=expected_asset_sha256,
    )

    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("ct-UAE checkpoint does not contain a state_dict mapping")

    weight = state.get("atom_embed.weight")
    bias = state.get("atom_embed.bias")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError("ct-UAE checkpoint lacks atom_embed weight or bias tensors")
    if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
        raise ValueError(
            "incompatible ct-UAE atom_embed shapes: "
            f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
        )

    table = torch.load(
        asset_path.expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(table, torch.Tensor):
        raise ValueError("ct-UAE derived asset must be a tensor")
    expected = weight.detach().cpu().T.contiguous() + bias.detach().cpu().unsqueeze(0)
    if tuple(table.shape) != tuple(expected.shape):
        raise ValueError(
            "ct-UAE table shape mismatch: "
            f"expected={tuple(expected.shape)}, observed={tuple(table.shape)}"
        )
    if not torch.equal(table, expected):
        max_abs = float((table.float() - expected.float()).abs().max().item())
        raise ValueError(
            "ct-UAE table is not the exact one-hot atom_embed projection; "
            f"maximum absolute difference={max_abs}"
        )

    return {
        "schema_version": "prm_ct_uae_asset_verification_v1",
        "checkpoint": checkpoint_record,
        "asset": asset_record,
        "derivation": {
            "state_dict_weight": "atom_embed.weight",
            "state_dict_bias": "atom_embed.bias",
            "formula": "atom_embed.weight.T + atom_embed.bias",
            "weight_shape": list(weight.shape),
            "bias_shape": list(bias.shape),
            "table_shape": list(table.shape),
            "tensor_sha256": tensor_sha256(table),
            "exact": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-asset-sha256", required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    record = verify_ct_uae_derivation(
        args.checkpoint,
        args.asset,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_asset_sha256=args.expected_asset_sha256,
    )
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
