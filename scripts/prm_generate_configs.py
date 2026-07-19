"""Generate paired configs for the PRM factorial and transfer experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = ("use_gated_pooling", "use_env_enrichment", "use_prenorm_local")


def write_config(path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(config, sort_keys=False)
    path.write_text(config_text)
    return {
        "path": str(path.relative_to(ROOT)),
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "output_dir": config["output_dir"],
        "split_path": config["split_path"],
        "seed": config["seed"],
    }


def component_code(bits: Iterable[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "configs/prm/base_factorial.yaml")
    parser.add_argument("--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v1")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "configs/prm/generated")
    args = parser.parse_args()

    base = yaml.safe_load(args.base.read_text())
    protocol_manifest = json.loads((args.protocol_dir / "manifest.json").read_text())
    data_sha256 = protocol_manifest["data_sha256"]
    generated = []

    # All eight variants use the same split and model seed within each repeat.
    for split_seed in range(42, 47):
        model_seed = split_seed + 100
        split_path = args.protocol_dir / "splits" / f"id_repeat_s{split_seed}.json"
        for mask in range(8):
            bits = tuple((mask >> shift) & 1 for shift in (2, 1, 0))
            code = component_code(bits)
            cfg = deepcopy(base)
            cfg["seed"] = model_seed
            cfg["split_path"] = str(split_path.relative_to(ROOT))
            cfg["data_sha256"] = data_sha256
            cfg["output_dir"] = f"factorial/g{code}/split{split_seed}_seed{model_seed}"
            for name, enabled in zip(COMPONENTS, bits):
                cfg["model_kwargs"][name] = bool(enabled)
            generated.append(
                write_config(
                    args.out_dir / "factorial" / f"g{code}_split{split_seed}_seed{model_seed}.yaml",
                    cfg,
                )
            )

    # These are provisional full-core transfer configs. They are promoted only
    # if the factorial does not select a different architecture.
    transfer_splits = [
        *(f"host_cv5_f{i}" for i in range(5)),
        *(f"dopant_cv5_f{i}" for i in range(5)),
        "chemistry_block_g6x3d",
    ]
    for split_id in transfer_splits:
        for model_seed in (242, 243, 244):
            cfg = deepcopy(base)
            cfg["seed"] = model_seed
            cfg["split_path"] = str(
                (args.protocol_dir / "splits" / f"{split_id}.json").relative_to(ROOT)
            )
            cfg["data_sha256"] = data_sha256
            cfg["output_dir"] = f"transfer/{split_id}/seed{model_seed}"
            for name in COMPONENTS:
                cfg["model_kwargs"][name] = True
            generated.append(
                write_config(
                    args.out_dir / "transfer" / f"{split_id}_seed{model_seed}.yaml",
                    cfg,
                )
            )

    manifest = {
        "schema_version": "prm_config_manifest_v1",
        "data_sha256": data_sha256,
        "base_config": str(args.base.resolve().relative_to(ROOT)),
        "n_configs": len(generated),
        "configs": generated,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "n_configs": len(generated)}, indent=2))


if __name__ == "__main__":
    main()
