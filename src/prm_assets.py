"""Integrity checks and auditable DART pretraining initialization."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn


ASSET_KEYS = ("ct_uae", "pretrained_embed")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EDGE_GATE_PATTERN = re.compile(r"^\d+\.edge_gate\.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} asset does not exist: {resolved}")
    expected = expected_sha256.lower() if expected_sha256 else None
    if expected is not None and _SHA256_PATTERN.fullmatch(expected) is None:
        raise ValueError(f"invalid expected SHA256 for {label}: {expected_sha256!r}")
    actual = file_sha256(resolved)
    if expected is not None and actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, observed {actual}"
        )
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected,
    }


def verify_training_assets(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    expected = config.get("asset_sha256", {})
    if not isinstance(expected, Mapping):
        raise ValueError("asset_sha256 must be a mapping")
    unknown = sorted(set(expected) - set(ASSET_KEYS))
    if unknown:
        raise ValueError(f"unrecognized training asset hashes: {unknown}")

    paths = {
        "ct_uae": config.get("model_kwargs", {}).get("ct_uae_path"),
        "pretrained_embed": config.get("pretrained_embed"),
    }
    if config.get("asset_integrity_required"):
        missing_hashes = [key for key, value in paths.items() if value and not expected.get(key)]
        missing_paths = [key for key, value in expected.items() if value and not paths.get(key)]
        if missing_hashes or missing_paths:
            raise ValueError(
                "incomplete required asset contract: "
                f"missing hashes={missing_hashes}, missing paths={missing_paths}"
            )

    records: Dict[str, Dict[str, Any]] = {}
    for key, path in paths.items():
        if path:
            records[key] = verify_asset(
                path,
                label=key,
                expected_sha256=expected.get(key),
            )
    return records


def load_pretrained_initialization(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> Dict[str, Any]:
    """Load the shared checkpoint and reject all unplanned incompatibilities.

    The checkpoint was trained with the nine elemental input features and V1
    local layers. DART appends ct-UAE features to those nine columns. The
    checkpoint therefore initializes only the corresponding input slice. V2
    local blocks may additionally contain edge-gate parameters; these are the
    only model keys allowed to remain at their seeded initialization.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"pretraining checkpoint is not a mapping: {path}")
    required = {"embed_weight", "embed_bias", "local_layers"}
    missing_fields = sorted(required - set(checkpoint))
    if missing_fields:
        raise ValueError(f"pretraining checkpoint is missing fields: {missing_fields}")
    if not hasattr(model, "embed") or not isinstance(model.embed, nn.Linear):
        raise ValueError("model has no linear input projection named embed")
    if not hasattr(model, "local_layers"):
        raise ValueError("model has no local_layers module")

    source_weight = checkpoint["embed_weight"]
    source_bias = checkpoint["embed_bias"]
    if not isinstance(source_weight, torch.Tensor) or not isinstance(source_bias, torch.Tensor):
        raise ValueError("pretrained input projection must contain tensors")
    current_weight = model.embed.weight
    current_bias = model.embed.bias
    if current_bias is None:
        raise ValueError("model input projection has no bias")
    atom_fea_len = int(getattr(model, "atom_fea_len", source_weight.shape[1]))
    expected_source_shape = (current_weight.shape[0], atom_fea_len)
    if tuple(source_weight.shape) != expected_source_shape:
        raise ValueError(
            "pretrained input weight shape mismatch: "
            f"expected {expected_source_shape}, observed {tuple(source_weight.shape)}"
        )
    if tuple(source_bias.shape) != tuple(current_bias.shape):
        raise ValueError(
            "pretrained input bias shape mismatch: "
            f"expected {tuple(current_bias.shape)}, observed {tuple(source_bias.shape)}"
        )
    if current_weight.shape[1] < atom_fea_len:
        raise ValueError("model input projection is narrower than the pretrained feature slice")

    source_local = checkpoint["local_layers"]
    if not isinstance(source_local, Mapping) or not source_local:
        raise ValueError("pretrained local_layers must be a non-empty state mapping")
    current_local = model.local_layers.state_dict()
    unexpected = sorted(key for key in source_local if key not in current_local)
    shape_mismatches = [
        {
            "key": key,
            "checkpoint_shape": list(source_local[key].shape),
            "model_shape": list(current_local[key].shape),
        }
        for key in sorted(set(source_local) & set(current_local))
        if tuple(source_local[key].shape) != tuple(current_local[key].shape)
    ]
    missing = sorted(key for key in current_local if key not in source_local)
    allowed_seeded = [key for key in missing if _EDGE_GATE_PATTERN.match(key)]
    forbidden_missing = sorted(set(missing) - set(allowed_seeded))
    if unexpected or shape_mismatches or forbidden_missing:
        raise ValueError(
            "incompatible pretrained local layers: "
            f"unexpected={unexpected}, shape_mismatches={shape_mismatches}, "
            f"forbidden_missing={forbidden_missing}"
        )

    with torch.no_grad():
        current_weight[:, :atom_fea_len].copy_(source_weight)
        current_bias.copy_(source_bias)
    incompatible = model.local_layers.load_state_dict(source_local, strict=False)
    if sorted(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
        raise RuntimeError("PyTorch pretraining load result differs from the validated key set")

    matched = sorted(set(source_local) & set(current_local))
    return {
        "schema_version": "prm_pretraining_report_v1",
        "checkpoint_path": str(path),
        "checkpoint_sha256": file_sha256(path),
        "source_dataset": checkpoint.get("source_dataset"),
        "source_test_mae_eV": checkpoint.get("source_test_mae"),
        "input_projection": {
            "checkpoint_weight_shape": list(source_weight.shape),
            "model_weight_shape": list(current_weight.shape),
            "copied_rows": [0, int(current_weight.shape[0])],
            "copied_columns": [0, atom_fea_len],
            "bias_copied": True,
            "seeded_model_columns": int(current_weight.shape[1] - atom_fea_len),
        },
        "local_layers": {
            "checkpoint_tensor_count": len(source_local),
            "model_tensor_count": len(current_local),
            "loaded_tensor_count": len(matched),
            "loaded_keys": matched,
            "seeded_model_keys": allowed_seeded,
            "unexpected_checkpoint_keys": unexpected,
            "shape_mismatches": shape_mismatches,
        },
        "all_model_parameters_trainable": all(
            parameter.requires_grad for parameter in model.parameters()
        ),
    }
