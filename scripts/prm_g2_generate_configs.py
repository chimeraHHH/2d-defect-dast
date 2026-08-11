"""Generate the frozen G2 repaired-OOF configuration set.

The G2 model recipe is the archived validation-selected g111 transfer recipe
with exactly three controlled changes:

1. ``model_kwargs.env_zero_neighbor_mode: zero_residual_v1`` — the corrected
   E-module contract accepted by G1;
2. ``asset_sha256.pretrained_embed`` — the corrected G1C initialization asset
   (the wrapper supplies its path through ``PRM_PRETRAINED_EMBED``);
3. per-run ``split_path`` / ``seed`` / ``output_dir`` for the canonical
   pair/host/dopant folds under the frozen seed law.

Everything else is copied verbatim from the archived
``pair_cv5_f0/seed242`` manifest, whose 43 siblings were verified to differ
only in those per-run fields.  The generator is deterministic; rerunning it
must reproduce byte-identical YAML.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prm_g2_contract import (  # noqa: E402
    CANONICAL_ENV_ZERO_NEIGHBOR_MODE,
    CANONICAL_FOLDS,
    CANONICAL_PREDICTION_SEEDS,
    CANONICAL_REGIMES,
)

TEMPLATE_MANIFEST = (
    ROOT / "artifacts/prm_results/comparison/runs/selected/g111/transfer/"
    "pair_cv5_f0/seed242/run_manifest.json"
)
CORRECTED_ASSET_SHA256 = (
    "293e1c21991a6c92757e30b776f8aa9aca10c9504e8830f1914916ff7b393333"
)
OUTPUT_DIR = ROOT / "configs" / "prm" / "g2"


def main() -> None:
    template = json.loads(TEMPLATE_MANIFEST.read_text())["config"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for regime in CANONICAL_REGIMES:
        for fold in CANONICAL_FOLDS:
            split_id = f"{regime}_cv5_f{fold}"
            for seed in sorted(CANONICAL_PREDICTION_SEEDS[regime]):
                config = json.loads(json.dumps(template))
                config["model_kwargs"]["env_zero_neighbor_mode"] = (
                    CANONICAL_ENV_ZERO_NEIGHBOR_MODE
                )
                config["asset_sha256"]["pretrained_embed"] = (
                    CORRECTED_ASSET_SHA256
                )
                # Documentation value only: the fail-closed wrapper always
                # overrides the asset path through PRM_PRETRAINED_EMBED and
                # verifies the file hash against the G1C receipt.
                config["pretrained_embed"] = (
                    "g1c_pretrain/pretrained_embed_corrected.pt"
                )
                config["split_path"] = (
                    f"artifacts/prm_protocol_v2/splits/{split_id}.json"
                )
                config["seed"] = seed
                config["output_dir"] = f"g2/{split_id}/seed{seed}"
                path = OUTPUT_DIR / f"{split_id}_seed{seed}.yaml"
                path.write_text(
                    yaml.safe_dump(config, sort_keys=True, width=88),
                )
                written += 1
    print(f"wrote {written} configs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
