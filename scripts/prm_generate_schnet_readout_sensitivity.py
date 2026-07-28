"""Generate the bounded SchNet mean-readout host-CV sensitivity campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
PARENT_PREFIX = "baselines/schnet"
OUTPUT_PREFIX = "sensitivity/schnet_readout/mean"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_mean_readout_config(parent: Mapping[str, Any]) -> dict[str, Any]:
    """Change only output location and SchNet graph readout."""
    config = deepcopy(dict(parent))
    output_dir = Path(str(config.get("output_dir", "")))
    try:
        relative_output = output_dir.relative_to(PARENT_PREFIX)
    except ValueError as exc:
        raise ValueError("parent SchNet output is outside the controlled prefix") from exc
    split_id = Path(str(config.get("split_path", ""))).stem
    if not split_id.startswith("host_cv5_f"):
        raise ValueError("readout sensitivity is restricted to host_cv5 splits")
    model_kwargs = config.get("model_kwargs")
    if not isinstance(model_kwargs, dict) or model_kwargs.get("readout") != "add":
        raise ValueError("parent SchNet config must use additive readout")
    config["output_dir"] = str(Path(OUTPUT_PREFIX) / relative_output)
    config["model_kwargs"] = {**model_kwargs, "readout": "mean"}
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent-dir",
        type=Path,
        default=ROOT / "configs/prm/generated/schnet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "configs/prm/sensitivity/schnet_mean_host",
    )
    args = parser.parse_args()

    parent_paths = sorted(args.parent_dir.glob("host_cv5_f*_seed*.yaml"))
    if len(parent_paths) != 15:
        raise ValueError(f"expected 15 host-CV parent configs, found {len(parent_paths)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for parent_path in parent_paths:
        parent = yaml.safe_load(parent_path.read_text())
        config = make_mean_readout_config(parent)
        output_path = args.out_dir / parent_path.name
        output_path.write_text(yaml.safe_dump(config, sort_keys=False))
        records.append(
            {
                "path": str(output_path.relative_to(ROOT)),
                "sha256": file_sha256(output_path),
                "parent_path": str(parent_path.relative_to(ROOT)),
                "parent_sha256": file_sha256(parent_path),
                "split_id": Path(str(config["split_path"])).stem,
                "seed": int(config["seed"]),
            }
        )

    manifest = {
        "schema_version": "prm_schnet_readout_sensitivity_configs_v1",
        "analysis_role": "post_hoc_exploratory_robustness",
        "question": (
            "Does replacing SchNet atomwise sum readout with mean readout "
            "reduce the host-held-out error under otherwise fixed conditions?"
        ),
        "intervention": {"model_kwargs.readout": {"from": "add", "to": "mean"}},
        "fixed_conditions": [
            "dataset and immutable host-CV splits",
            "five folds and seeds 342-344",
            "SchNet architecture except graph readout",
            "optimizer, schedule, sampling, label noise, and 150 epochs",
        ],
        "stop_condition": "exactly 15 complete runs and one paired host-cluster analysis",
        "n_configs": len(records),
        "configs": records,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
