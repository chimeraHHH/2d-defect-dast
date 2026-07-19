from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from src.train_schnet import build_scheduler


ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_DART_KEYS = {
    "online_aug_cfg",
    "aux_defect_weight",
    "drop_path_rate",
    "use_swa",
    "swa_start_epoch",
    "swa_lr",
}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _assert_canonical_dart(config: dict) -> None:
    assert config["online_aug"] is False
    assert FORBIDDEN_DART_KEYS.isdisjoint(config)


def _assert_canonical_schnet(config: dict) -> None:
    assert config["online_aug"] is False
    assert "online_aug_cfg" not in config


def test_neural_base_configs_share_the_optimization_contract():
    dart = _load(ROOT / "configs/prm/base_factorial.yaml")
    schnet = _load(ROOT / "configs/prm/base_schnet.yaml")

    _assert_canonical_dart(dart)
    _assert_canonical_schnet(schnet)
    for key in (
        "epochs",
        "grad_clip",
        "host_balanced",
        "samples_per_host",
        "label_noise_std",
        "warmup_epochs",
    ):
        assert dart[key] == schnet[key]
    assert dart["optimizer"]["lr"] == schnet["lr"]
    assert dart["optimizer"]["weight_decay"] == schnet["weight_decay"]
    assert dart["scheduler"]["eta_min"] == schnet["eta_min"]


def test_every_generated_config_inherits_the_canonical_recipe():
    dart_manifest = json.loads(
        (ROOT / "configs/prm/generated/manifest.json").read_text()
    )
    assert dart_manifest["n_configs"] == 83
    for record in dart_manifest["configs"]:
        _assert_canonical_dart(_load(ROOT / record["path"]))

    schnet_dir = ROOT / "configs/prm/generated/schnet"
    schnet_manifest = json.loads((schnet_dir / "manifest.json").read_text())
    assert schnet_manifest["n_configs"] == 48
    for relative_path in schnet_manifest["configs"]:
        _assert_canonical_schnet(_load(ROOT / relative_path))


def test_schnet_scheduler_warms_up_then_decays_to_floor():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-4)
    scheduler = build_scheduler(
        optimizer, epochs=150, warmup_epochs=10, eta_min=1e-6
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    for _ in range(140):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)


@pytest.mark.parametrize("warmup_epochs", [-1, 150])
def test_schnet_scheduler_rejects_invalid_warmup(warmup_epochs):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-4)
    with pytest.raises(ValueError, match="warmup_epochs"):
        build_scheduler(
            optimizer,
            epochs=150,
            warmup_epochs=warmup_epochs,
            eta_min=1e-6,
        )
