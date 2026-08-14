"""Generate transfer and UQ configs after validation-only architecture selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.prm_provenance import load_verified_factorial_selection

COMPONENTS = ("use_gated_pooling", "use_env_enrichment", "use_prenorm_local")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_bits(selection: Dict[str, Any]) -> List[bool]:
    if selection.get("selection_data") != "validation only":
        raise ValueError("architecture promotion requires validation-only selection")
    variant = str(selection.get("selected_variant", ""))
    if len(variant) != 4 or not variant.startswith("g") or set(variant[1:]) - {"0", "1"}:
        raise ValueError(f"invalid selected variant: {variant!r}")
    return [digit == "1" for digit in variant[1:]]


def write_config(path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config, sort_keys=False)
    path.write_text(text)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "split_path": config["split_path"],
        "seed": config["seed"],
        "output_dir": config["output_dir"],
    }


def promoted_config(
    base: Dict[str, Any], bits: Iterable[bool], split_path: Path,
    output_dir: str, seed: int, data_sha256: str,
) -> Dict[str, Any]:
    config = deepcopy(base)
    config["seed"] = seed
    config["split_path"] = str(split_path.relative_to(ROOT))
    config["output_dir"] = output_dir
    config["data_sha256"] = data_sha256
    for name, enabled in zip(COMPONENTS, bits):
        config["model_kwargs"][name] = bool(enabled)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", type=Path,
        default=ROOT / "artifacts/prm_results/factorial/selection.json",
    )
    parser.add_argument("--factorial-bundle", type=Path, default=None)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "configs/prm/base_factorial.yaml",
    )
    parser.add_argument(
        "--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v2",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "configs/prm/promoted",
    )
    args = parser.parse_args()

    selection, factorial_bundle = load_verified_factorial_selection(
        args.selection, args.factorial_bundle
    )
    bits = selected_bits(selection)
    variant = selection["selected_variant"]
    base = yaml.safe_load(args.base.read_text())
    protocol = json.loads((args.protocol_dir / "manifest.json").read_text())
    data_sha256 = protocol["data_sha256"]
    if selection["data_sha256"] != data_sha256:
        raise ValueError("factorial selection does not match the frozen protocol dataset")
    variant_dir = args.out_dir / variant
    records = []

    transfer_splits = [
        *(f"id_cv5_f{fold}" for fold in range(5)),
        *(f"pair_cv5_f{fold}" for fold in range(5)),
        *(f"host_cv5_f{fold}" for fold in range(5)),
        *(f"dopant_cv5_f{fold}" for fold in range(5)),
        "chemistry_block_g6x3d",
    ]
    for split_id in transfer_splits:
        seeds = (242,) if split_id.startswith(("id_cv5", "pair_cv5")) else (242, 243, 244)
        for seed in seeds:
            config = promoted_config(
                base, bits, args.protocol_dir / "splits" / f"{split_id}.json",
                f"selected/{variant}/transfer/{split_id}/seed{seed}", seed,
                data_sha256,
            )
            records.append(
                {
                    "purpose": "transfer",
                    **write_config(
                        variant_dir / "transfer" / f"{split_id}_seed{seed}.yaml",
                        config,
                    ),
                }
            )

    uq_split = "uq_calibration_s62"
    for seed in range(442, 447):
        config = promoted_config(
            base, bits, args.protocol_dir / "splits" / f"{uq_split}.json",
            f"selected/{variant}/uq/{uq_split}/seed{seed}", seed, data_sha256,
        )
        records.append(
            {
                "purpose": "uq_ensemble",
                **write_config(
                    variant_dir / "uq" / f"{uq_split}_seed{seed}.yaml", config,
                ),
            }
        )

    manifest = {
        "schema_version": "prm_promoted_config_manifest_v1",
        "selection_path": str(args.selection.resolve()),
        "selection_sha256": sha256(args.selection),
        "factorial_bundle": str(
            (args.factorial_bundle or args.selection.parent / selection["factorial_bundle"])
            .resolve()
        ),
        "factorial_bundle_sha256": selection["factorial_bundle_sha256"],
        "factorial_training_commit": selection["training_commits"][0],
        "factorial_collector_git": factorial_bundle["collector_git"],
        "selection_data": selection["selection_data"],
        "selected_variant": variant,
        "component_flags": dict(zip(COMPONENTS, bits)),
        "data_sha256": data_sha256,
        "n_configs": len(records),
        "configs": records,
    }
    manifest_path = variant_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "n_configs": len(records)}, indent=2))


if __name__ == "__main__":
    main()
