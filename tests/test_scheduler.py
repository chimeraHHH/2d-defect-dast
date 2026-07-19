from __future__ import annotations

import json

import pytest
import yaml

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
    assert record["config_sha256"] == prm_scheduler.config_sha256(
        yaml.safe_load(path.read_text())
    )
    assert record["status"] == "pending"


def _write_run_manifest(result_root, config, *, status="complete", dirty=False, commit="abc"):
    run_dir = result_root / config["output_dir"]
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "prm_run_manifest_v1",
        "status": status,
        "git": {"commit": commit, "dirty": dirty},
        "config": config,
        "config_sha256": prm_scheduler.config_sha256(config),
        "execution": {"max_steps": 0, "resume_requested": False},
        "seed": config["seed"],
        "split": {"split_id": "id_repeat_s42"},
        "metrics": {
            "split_id": "id_repeat_s42",
            "n_params": 100,
            "best_val_mae": 0.4,
            "history": [{"epoch": 1, "val_mae": 0.4}],
            "validation": {
                "mae": 0.4, "rmse": 0.5, "bias": 0.0,
                "spearman": 0.7, "r2": 0.6,
            },
            "test": {
                "mae": 0.45, "rmse": 0.55, "bias": 0.01,
                "spearman": 0.65, "r2": 0.5,
            },
        },
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_existing_complete_run_requires_the_current_config(tmp_path):
    config = {
        "output_dir": "factorial/g000/split42_seed142",
        "split_path": "split.json",
        "seed": 142,
        "epochs": 1,
    }
    _write_run_manifest(tmp_path, config)
    assert prm_scheduler.existing_run_status(tmp_path, config, "abc") == "complete"

    changed = {**config, "epochs": 2}
    with pytest.raises(ValueError, match="stale run config"):
        prm_scheduler.existing_run_status(tmp_path, changed, "abc")


def test_existing_dirty_run_is_rejected(tmp_path):
    config = {
        "output_dir": "factorial/g000/split42_seed142",
        "split_path": "split.json",
        "seed": 142,
        "epochs": 1,
    }
    _write_run_manifest(tmp_path, config, dirty=True)
    with pytest.raises(ValueError, match="dirty existing run"):
        prm_scheduler.existing_run_status(tmp_path, config, "abc")


def test_partial_run_cannot_resume_across_commits(tmp_path):
    config = {
        "output_dir": "factorial/g000/split42_seed142",
        "split_path": "split.json",
        "seed": 142,
        "epochs": 1,
    }
    _write_run_manifest(tmp_path, config, status="running", commit="old")
    with pytest.raises(ValueError, match="refusing to resume"):
        prm_scheduler.existing_run_status(tmp_path, config, "new")


def test_queue_terminal_status_distinguishes_failure_from_completion():
    complete = [{"status": "complete", "attempts": 1}]
    exhausted = [{"status": "failed", "attempts": 2}]
    retryable = [{"status": "failed", "attempts": 1}]
    assert prm_scheduler.queue_terminal_status(complete, 0, 2) == "complete"
    assert prm_scheduler.queue_terminal_status(exhausted, 0, 2) == "failed"
    assert prm_scheduler.queue_terminal_status(retryable, 0, 2) is None
    assert prm_scheduler.queue_terminal_status(complete, 1, 2) is None


def test_changed_config_paths_detects_runtime_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(prm_scheduler, "ROOT", tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(
        "output_dir: factorial/g000/split42_seed142\n"
        "split_path: artifacts/split.json\n"
        "seed: 142\n"
    )
    record = prm_scheduler.config_record(path)
    assert prm_scheduler.changed_config_paths([record]) == []
    path.write_text(path.read_text() + "epochs: 151\n")
    assert prm_scheduler.changed_config_paths([record]) == [str(path)]


def test_terminate_running_reaps_process_and_closes_log():
    class Process:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            assert timeout == 30
            return -15

    class Log:
        closed = False

        def close(self):
            self.closed = True

    process = Process()
    log = Log()
    record = {"status": "running"}
    running = {0: {"process": process, "log_handle": log, "record": record}}
    prm_scheduler.terminate_running(running)
    assert running == {}
    assert process.terminated is True
    assert log.closed is True
    assert record["status"] == "stopped"
    assert record["return_code"] == -15
