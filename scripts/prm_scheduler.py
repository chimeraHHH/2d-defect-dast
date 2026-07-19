"""Resource-aware scheduler for reproducible PRM training queues."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def config_sha256(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_snapshot() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def gpu_status() -> Dict[int, Dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    status: Dict[int, Dict[str, int]] = {}
    for line in result.stdout.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        status[index] = {"memory_used_mib": memory, "utilization_percent": utilization}
    return status


def existing_run_status(
    result_root: Path,
    config: Dict[str, Any],
    current_commit: str | None,
) -> str:
    manifest = result_root / config["output_dir"] / "run_manifest.json"
    if not manifest.exists():
        return "pending"
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable existing run manifest: {manifest}") from exc
    if payload.get("schema_version") != "prm_run_manifest_v1":
        raise ValueError(f"unsupported existing run manifest: {manifest}")
    expected_hash = config_sha256(config)
    if payload.get("config_sha256") != expected_hash:
        raise ValueError(
            f"stale run config at {manifest}; archive its output directory before restarting"
        )
    if config_sha256(payload.get("config", {})) != expected_hash:
        raise ValueError(f"run manifest config/hash mismatch: {manifest}")
    if payload.get("git", {}).get("dirty"):
        raise ValueError(f"dirty existing run is inadmissible: {manifest}")
    status = payload.get("status")
    if status == "complete":
        return "complete"
    if status != "running":
        raise ValueError(f"unsupported existing run status {status!r}: {manifest}")
    run_commit = payload.get("git", {}).get("commit")
    if current_commit is not None and run_commit != current_commit:
        raise ValueError(
            f"refusing to resume {manifest} from commit {run_commit}; "
            f"current commit is {current_commit}"
        )
    return "pending"


def is_complete(result_root: Path, config: Dict[str, Any]) -> bool:
    return existing_run_status(result_root, config, current_commit=None) == "complete"


def config_record(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    return {
        "config_path": str(path.resolve()),
        "config_relative": str(path.resolve().relative_to(ROOT)),
        "output_dir": config["output_dir"],
        "split_path": config["split_path"],
        "seed": config["seed"],
        "config_sha256": config_sha256(config),
        "attempts": 0,
        "status": "pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="configs/prm/generated/factorial/*.yaml")
    parser.add_argument("--queue-id", default="factorial")
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--module", default="src.train_enhanced")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--ct-uae-path", type=Path, required=True)
    parser.add_argument("--pretrained-embed", type=Path, required=True)
    parser.add_argument("--exclude-gpus", default="2")
    parser.add_argument("--max-parallel", type=int, default=7)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-confirmations", type=int, default=2)
    parser.add_argument("--max-memory-mib", type=int, default=1500)
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()

    result_root = args.result_root.expanduser().resolve()
    scheduler_git = git_snapshot()
    if scheduler_git["commit"] is None:
        raise SystemExit("scheduler must run from a Git worktree")
    if scheduler_git["dirty"]:
        raise SystemExit(
            "scheduler requires a clean Git worktree: "
            + ", ".join(scheduler_git["status_porcelain"])
        )
    scheduler_dir = result_root / "_scheduler" / args.queue_id
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (scheduler_dir / "scheduler.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"scheduler {args.queue_id!r} is already running")

    config_paths = sorted(ROOT.glob(args.glob))
    if not config_paths:
        raise SystemExit(f"no configs matched {args.glob!r}")
    records = [config_record(path) for path in config_paths]
    config_by_path = {
        record["config_path"]: yaml.safe_load(Path(record["config_path"]).read_text())
        for record in records
    }
    for record in records:
        record["status"] = existing_run_status(
            result_root,
            config_by_path[record["config_path"]],
            current_commit=scheduler_git["commit"],
        )

    excluded = {int(value) for value in args.exclude_gpus.split(",") if value.strip()}
    state: Dict[str, Any] = {
        "schema_version": "prm_scheduler_state_v1",
        "queue_id": args.queue_id,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "scheduler_git": scheduler_git,
        "settings": {
            "glob": args.glob,
            "module": args.module,
            "excluded_gpus": sorted(excluded),
            "max_parallel": args.max_parallel,
            "poll_seconds": args.poll_seconds,
            "idle_confirmations": args.idle_confirmations,
            "max_memory_mib": args.max_memory_mib,
            "max_utilization": args.max_utilization,
            "max_attempts": args.max_attempts,
        },
        "jobs": records,
    }
    state_path = scheduler_dir / "state.json"
    atomic_json(state_path, state)

    running: Dict[int, Dict[str, Any]] = {}
    idle_counts: Dict[int, int] = {}
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while True:
        for gpu, active in list(running.items()):
            return_code = active["process"].poll()
            if return_code is None:
                continue
            active["log_handle"].close()
            record = active["record"]
            config = config_by_path[record["config_path"]]
            complete = return_code == 0 and is_complete(result_root, config)
            record.update(
                {
                    "finished_at": utc_now(),
                    "return_code": return_code,
                    "status": "complete" if complete else "failed",
                }
            )
            del running[gpu]

        if stop_requested:
            state["status"] = "stopping"
            for active in running.values():
                active["process"].terminate()
            atomic_json(state_path, state)
            break

        pending = [
            record for record in records
            if record["status"] in ("pending", "failed")
            and record["attempts"] < args.max_attempts
        ]
        if not pending and not running:
            state["status"] = "complete"
            state["completed_at"] = utc_now()
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
            break

        try:
            observed = gpu_status()
        except (OSError, subprocess.CalledProcessError) as exc:
            state["last_gpu_error"] = f"{type(exc).__name__}: {exc}"
            observed = {}

        for gpu, metrics in observed.items():
            if gpu in excluded or gpu in running:
                idle_counts[gpu] = 0
                continue
            idle = (
                metrics["memory_used_mib"] <= args.max_memory_mib
                and metrics["utilization_percent"] <= args.max_utilization
            )
            idle_counts[gpu] = idle_counts.get(gpu, 0) + 1 if idle else 0

        available = [
            gpu for gpu in sorted(observed)
            if gpu not in excluded
            and gpu not in running
            and idle_counts.get(gpu, 0) >= args.idle_confirmations
        ]
        slots = max(0, args.max_parallel - len(running))
        for gpu in available[:slots]:
            pending = [
                record for record in records
                if record["status"] in ("pending", "failed")
                and record["attempts"] < args.max_attempts
            ]
            if not pending:
                break
            record = pending[0]
            config = config_by_path[record["config_path"]]
            output_dir = result_root / config["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "stdout.log"
            log_handle = log_path.open("a")
            command = [
                str(args.python), "-u", "-m", args.module,
                "--config", record["config_path"],
            ]
            if (output_dir / "latest.pt").exists():
                command.append("--resume")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PRM_RESULTS_ROOT": str(result_root),
                    "PRM_DATA_PATH": str(args.data_path.resolve()),
                    "PRM_CT_UAE_PATH": str(args.ct_uae_path.resolve()),
                    "PRM_PRETRAINED_EMBED": str(args.pretrained_embed.resolve()),
                }
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.update(
                {
                    "status": "running",
                    "attempts": record["attempts"] + 1,
                    "started_at": utc_now(),
                    "gpu": gpu,
                    "pid": process.pid,
                    "command": command,
                    "log": str(log_path),
                }
            )
            running[gpu] = {
                "process": process,
                "record": record,
                "log_handle": log_handle,
            }
            idle_counts[gpu] = 0

        state["updated_at"] = utc_now()
        state["gpu_status"] = observed
        state["running_gpus"] = sorted(running)
        state["counts"] = {
            status: sum(record["status"] == status for record in records)
            for status in ("pending", "running", "complete", "failed")
        }
        atomic_json(state_path, state)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
