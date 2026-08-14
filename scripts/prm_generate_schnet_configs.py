"""Generate SchNet comparator configs for every formal PRM split."""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "configs/prm/base_schnet.yaml")
    parser.add_argument("--protocol-dir", type=Path, default=ROOT / "artifacts/prm_protocol_v2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "configs/prm/generated/schnet")
    args = parser.parse_args()

    base = yaml.safe_load(args.base.read_text())
    protocol_path = args.protocol_dir / "manifest.json"
    protocol = json.loads(protocol_path.read_text())
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_ids = [
        *(f"id_repeat_s{i}" for i in range(42, 47)),
        *(f"id_cv5_f{i}" for i in range(5)),
        *(f"pair_cv5_f{i}" for i in range(5)),
        *(f"host_cv5_f{i}" for i in range(5)),
        *(f"dopant_cv5_f{i}" for i in range(5)),
        "chemistry_block_g6x3d",
    ]
    records = []
    for split_id in split_ids:
        single_seed = (
            split_id.startswith("id_repeat")
            or split_id.startswith("id_cv5")
            or split_id.startswith("pair_cv5")
        )
        seeds = (342,) if single_seed else (342, 343, 344)
        for seed in seeds:
            cfg = deepcopy(base)
            cfg["seed"] = seed
            cfg["data_sha256"] = protocol["data_sha256"]
            cfg["split_path"] = str(
                (args.protocol_dir / "splits" / f"{split_id}.json").relative_to(ROOT)
            )
            cfg["output_dir"] = f"baselines/schnet/{split_id}/seed{seed}"
            path = out_dir / f"{split_id}_seed{seed}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            records.append(str(path.relative_to(ROOT)))
    manifest = {
        "schema_version": "prm_schnet_config_manifest_v2",
        "data_sha256": protocol["data_sha256"],
        "protocol_manifest_sha256": file_sha256(protocol_path),
        "protocol_dir": str(args.protocol_dir.relative_to(ROOT)),
        "n_configs": len(records),
        "configs": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
