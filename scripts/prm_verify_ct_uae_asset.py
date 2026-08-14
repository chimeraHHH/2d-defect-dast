"""Verify the exact derivation of the fixed ct-UAE elemental table."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prm_assets import verify_asset


_SHA256_LENGTH = 64


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_projection(
    checkpoint_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # Preserve the transposed output layout used by the canonical asset. Calling
    # contiguous() here leaves values unchanged but changes torch.save bytes.
    table = weight.detach().cpu().T + bias.detach().cpu().unsqueeze(0)
    return weight, bias, table


def derive_ct_uae_asset(
    checkpoint_path: Path,
    asset_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_asset_sha256: str,
    expected_tensor_sha256: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    verify_asset(
        checkpoint_path,
        label="ct-UAE source checkpoint",
        expected_sha256=expected_checkpoint_sha256,
    )
    _, _, table = _load_projection(checkpoint_path)
    observed_tensor_sha256 = tensor_sha256(table)
    if (
        len(expected_tensor_sha256) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_tensor_sha256)
    ):
        raise ValueError(
            f"invalid expected ct-UAE tensor SHA256: {expected_tensor_sha256!r}"
        )
    if observed_tensor_sha256 != expected_tensor_sha256:
        raise ValueError(
            "derived ct-UAE tensor SHA256 mismatch: "
            f"expected {expected_tensor_sha256}, observed {observed_tensor_sha256}"
        )

    resolved = asset_path.expanduser().resolve()
    if resolved.exists() and not overwrite:
        raise FileExistsError(
            f"ct-UAE derived asset already exists; pass --overwrite to replace it: {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ct-uae-",
        dir=resolved.parent,
    ) as temporary_directory:
        staged = Path(temporary_directory) / resolved.name
        torch.save(table, staged)
        record = verify_asset(
            staged,
            label="generated ct-UAE table",
            expected_sha256=expected_asset_sha256,
        )
        staged.replace(resolved)
    record["path"] = str(resolved)
    record["tensor_sha256"] = observed_tensor_sha256
    record["stride"] = list(table.stride())
    return record


def verify_ct_uae_derivation(
    checkpoint_path: Path,
    asset_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_asset_sha256: str,
    expected_tensor_sha256: str | None = None,
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

    weight, bias, expected = _load_projection(checkpoint_path)

    table = torch.load(
        asset_path.expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(table, torch.Tensor):
        raise ValueError("ct-UAE derived asset must be a tensor")
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
    observed_tensor_sha256 = tensor_sha256(table)
    if (
        expected_tensor_sha256 is not None
        and observed_tensor_sha256 != expected_tensor_sha256
    ):
        raise ValueError(
            "ct-UAE tensor SHA256 mismatch: "
            f"expected {expected_tensor_sha256}, observed {observed_tensor_sha256}"
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
            "table_stride": list(table.stride()),
            "tensor_sha256": observed_tensor_sha256,
            "exact": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-asset-sha256", required=True)
    parser.add_argument("--expected-tensor-sha256")
    parser.add_argument(
        "--derive",
        action="store_true",
        help="Generate the fixed table from the source checkpoint before verification.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --derive to replace an existing derived table.",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    if args.overwrite and not args.derive:
        parser.error("--overwrite requires --derive")
    if args.derive:
        if args.expected_tensor_sha256 is None:
            parser.error("--derive requires --expected-tensor-sha256")
        derive_ct_uae_asset(
            args.checkpoint,
            args.asset,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_asset_sha256=args.expected_asset_sha256,
            expected_tensor_sha256=args.expected_tensor_sha256,
            overwrite=args.overwrite,
        )

    record = verify_ct_uae_derivation(
        args.checkpoint,
        args.asset,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_asset_sha256=args.expected_asset_sha256,
        expected_tensor_sha256=args.expected_tensor_sha256,
    )
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
