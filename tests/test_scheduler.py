from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from scripts import prm_scheduler
from src.prm_metrics import regression_metrics
from src.prm_provenance import file_sha256


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
    split_id = "id_repeat_s42"
    split_indices = {
        "train": np.asarray([0, 1], dtype=np.int64),
        "val": np.asarray([2, 3], dtype=np.int64),
        "test": np.asarray([4, 5], dtype=np.int64),
    }
    validation_targets = np.asarray([0.0, 1.0])
    validation_predictions = np.asarray([0.2, 0.8])
    test_targets = np.asarray([0.0, 2.0])
    test_predictions = np.asarray([0.5, 1.5])
    validation_metrics = regression_metrics(validation_targets, validation_predictions)
    test_metrics = regression_metrics(test_targets, test_predictions)
    metrics = {
        "split_id": split_id,
        "n_params": 100,
        "best_val_mae": validation_metrics["mae"],
        "history": [{"epoch": 1, "val_mae": validation_metrics["mae"]}],
        "validation": validation_metrics,
        "test": test_metrics,
    }
    outputs = {
        "metrics": "metrics.json",
        "checkpoint": "best.pt",
        "split_indices": "split_indices.npz",
        "validation_predictions": "val_predictions.npz",
        "test_predictions": "test_predictions.npz",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, sort_keys=True))
    (run_dir / "best.pt").write_bytes(b"checkpoint")
    np.savez_compressed(
        run_dir / "split_indices.npz",
        schema_version=np.asarray("prm_split_indices_v1"),
        split_id=np.asarray(split_id),
        **split_indices,
    )
    for name, indices, targets, predictions, split_name in (
        ("val_predictions.npz", split_indices["val"], validation_targets, validation_predictions, "val"),
        ("test_predictions.npz", split_indices["test"], test_targets, test_predictions, "test"),
    ):
        np.savez_compressed(
            run_dir / name,
            schema_version=np.asarray("prm_predictions_v1"),
            split_id=np.asarray(split_id),
            split=np.asarray(split_name),
            indices=indices,
            preds=predictions,
            targets=targets,
        )
    manifest = {
        "schema_version": "prm_run_manifest_v1",
        "status": status,
        "git": {"commit": commit, "dirty": dirty},
        "config": config,
        "config_sha256": prm_scheduler.config_sha256(config),
        "execution": {"max_steps": 0, "resume_requested": False},
        "seed": config["seed"],
        "split": {
            "split_id": split_id,
            "counts": {key: len(value) for key, value in split_indices.items()},
        },
        "metrics": metrics,
        "outputs": outputs,
        "output_sha256": {
            key: file_sha256(run_dir / name) for key, name in outputs.items()
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


def test_existing_complete_run_cannot_be_reused_across_commits(tmp_path):
    config = {
        "output_dir": "factorial/g000/split42_seed142",
        "split_path": "split.json",
        "seed": 142,
        "epochs": 1,
    }
    _write_run_manifest(tmp_path, config, status="complete", commit="old")
    with pytest.raises(ValueError, match="refusing to reuse"):
        prm_scheduler.existing_run_status(tmp_path, config, "new")


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
