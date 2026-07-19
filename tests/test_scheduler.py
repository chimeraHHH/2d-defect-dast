from __future__ import annotations

import json

from scripts import prm_scheduler


def test_atomic_json_round_trip(tmp_path):
    path = tmp_path / "state.json"
    prm_scheduler.atomic_json(path, {"status": "running", "count": 2})
    assert json.loads(path.read_text()) == {"status": "running", "count": 2}
    assert not path.with_suffix(".json.tmp").exists()


def test_config_record_reads_output_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(prm_scheduler, "ROOT", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(
        "output_dir: factorial/g000/split42_seed142\n"
        "split_path: artifacts/split.json\n"
        "seed: 142\n"
    )
    record = prm_scheduler.config_record(path)
    assert record["config_relative"] == "config.yaml"
    assert record["output_dir"] == "factorial/g000/split42_seed142"
    assert record["status"] == "pending"
