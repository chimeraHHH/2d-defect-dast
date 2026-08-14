from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.train_schnet import build_scheduler
from src.train_enhanced import (
    capture_rng_state,
    evaluate as evaluate_dart,
    make_label_noise_generator,
    resolve_runtime_assets,
    restore_rng_state,
    set_seed,
)


ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_DART_KEYS = {
    "online_aug_cfg",
    "aux_defect_weight",
    "drop_path_rate",
    "use_swa",
    "swa_start_epoch",
    "swa_lr",
}
EXPECTED_ASSET_HASHES = {
    "ct_uae": "ac77b2720b7bb8a3b290d6bfbae482857962d55daa7fc3486f2c7f44c41c60dc",
    "pretrained_embed": "5dd085b1393acee4db83422241102595a5d011c18a880a44c09947b29274a3bc",
}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _assert_canonical_dart(config: dict) -> None:
    assert config["online_aug"] is False
    assert config["num_workers"] == 0
    assert FORBIDDEN_DART_KEYS.isdisjoint(config)
    assert config["asset_integrity_required"] is True
    assert config["asset_sha256"] == EXPECTED_ASSET_HASHES


def _assert_canonical_schnet(config: dict) -> None:
    assert config["online_aug"] is False
    assert config["num_workers"] == 0
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


def test_readout_sensitivity_configs_change_only_the_declared_readout():
    sensitivity_dir = ROOT / "configs/prm/sensitivity/schnet_mean_host"
    manifest = json.loads((sensitivity_dir / "manifest.json").read_text())
    assert manifest["analysis_role"] == "post_hoc_exploratory_robustness"
    assert manifest["n_configs"] == 15
    for record in manifest["configs"]:
        parent = _load(ROOT / record["parent_path"])
        sensitivity = _load(ROOT / record["path"])
        assert parent["model_kwargs"]["readout"] == "add"
        assert sensitivity["model_kwargs"]["readout"] == "mean"
        parent.pop("output_dir")
        sensitivity.pop("output_dir")
        parent["model_kwargs"].pop("readout")
        sensitivity["model_kwargs"].pop("readout")
        assert sensitivity == parent


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


def test_rng_state_round_trip_reproduces_all_training_generators():
    set_seed(91)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng_state(state)
    observed = (random.random(), np.random.random(), torch.rand(3))
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])


def test_label_noise_stream_is_independent_of_model_rng_consumption():
    device = torch.device("cpu")
    first = make_label_noise_generator(device, seed=142, epoch=7)
    expected = torch.randn((4,), generator=first)

    torch.manual_seed(999)
    _ = torch.randn(10_000)
    repeated = make_label_noise_generator(device, seed=142, epoch=7)
    observed = torch.randn((4,), generator=repeated)
    next_epoch = make_label_noise_generator(device, seed=142, epoch=8)
    changed = torch.randn((4,), generator=next_epoch)

    assert torch.equal(observed, expected)
    assert not torch.equal(changed, expected)


def test_dart_evaluation_uses_standard_tie_aware_spearman():
    class Model(torch.nn.Module):
        def forward(self, batch):
            return batch["prediction"]

    class IdentityNormalizer:
        @staticmethod
        def denorm(values):
            return values

    batch = {
        "target": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        "prediction": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "sample_index": torch.arange(4),
    }
    metrics = evaluate_dart(
        Model(), [batch], IdentityNormalizer(), torch.device("cpu")
    )
    assert metrics["spearman"] == pytest.approx(0.0)


def test_runtime_asset_resolution_does_not_mutate_controlled_config():
    controlled = {
        "model_kwargs": {"ct_uae_path": "data/ct_uae.pt"},
        "pretrained_embed": "results/pretrained.pt",
    }
    runtime = resolve_runtime_assets(
        controlled,
        ct_uae_override="/runtime/ct_uae.pt",
        pretrained_override="/runtime/pretrained.pt",
    )
    assert controlled["model_kwargs"]["ct_uae_path"] == "data/ct_uae.pt"
    assert controlled["pretrained_embed"] == "results/pretrained.pt"
    assert runtime["model_kwargs"]["ct_uae_path"] == "/runtime/ct_uae.pt"
    assert runtime["pretrained_embed"] == "/runtime/pretrained.pt"
