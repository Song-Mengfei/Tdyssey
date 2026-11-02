"""Launch the single frozen-model diagnostic behind sampled Windows guards.

This parent imports neither torch nor simulator code. The real interpreter must
complete the existing nonce/PID/kill-on-close Job Object handshake before imports.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdyssey.frozen_wm_budget import LIMITS, WALL_SECONDS, storage_usage, replace_with_retry
from tdyssey.policy_memory import disk_snapshot, gpu_memory, guard_reason, process_memory, system_memory

GIB = 1024**3


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def append(out, value):
    out.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    out.flush()


def provenance():
    paths = [*(ROOT / "tdyssey").rglob("*.py"), Path(__file__),
             ROOT / "scripts/guarded_c_child.py", ROOT / "scripts/validate_frozen_wm.py"]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def claim_worker(launcher, directory, nonce):
    """Reuse the verified actual-PID ownership handshake, including venv launchers."""
    from tdyssey.policy_memory import OwnedProcess
    began = time.monotonic()
    path = directory / "worker.json"
    while not path.exists():
        if launcher.poll() is not None or time.monotonic() - began >= 10:
            raise RuntimeError("Actual interpreter did not provide its bounded bootstrap handshake")
        time.sleep(.05)
    identity = json.loads(path.read_text(encoding="utf-8"))
    if identity.get("nonce") != nonce:
        raise ValueError("Actual interpreter handshake nonce differs")
    owned = OwnedProcess(identity["pid"], launcher.pid)
    try:
        process = process_memory(owned.pid)
        system = system_memory()
        if not process.get("ok"):
            raise RuntimeError("Actual interpreter memory measurement failed before ACK")
        reason = guard_reason(system, 4 * GIB, 6 * GIB)
        if reason:
            raise RuntimeError(reason)
        write_json(directory / "worker-claimed.json", {"identity": identity, "launcher_pid": launcher.pid,
            "process": process, "system": system, "kill_on_parent_job_close": True})
        temporary = directory / "worker-ack.tmp"
        write_json(temporary, {"pid": owned.pid, "nonce": nonce})
        replace_with_retry(temporary, directory / "worker-ack.json")
        return owned
    except BaseException:
        owned.close()
        raise


def startup_reason(initial):
    reason = guard_reason(initial["system"], 4 * GIB, 6 * GIB)
    if not initial["gpu"].get("ok"):
        reason = reason or "initial_gpu_measurement_failed"
    elif min(device["free_bytes"] for device in initial["gpu"]["devices"]) < 4 * GIB:
        reason = reason or "initial_gpu_free_below_4GiB"
    if not initial["disk"].get("ok") or initial["disk"].get("free_bytes", 0) < 2 * GIB:
        reason = reason or "initial_disk_headroom_below_2GiB_or_measurement_failed"
    return reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue the original diagnostic authority and deadline")
    for name in ("output-dir", "ledger-dir", "official-root", "gpc-root", "checkpoint", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.ledger_dir = args.ledger_dir.resolve()
    if (args.output_dir == args.ledger_dir or args.output_dir.is_relative_to(args.ledger_dir)
            or args.ledger_dir.is_relative_to(args.output_dir)):
        parser.error("Output and independent diagnostic ledger must be separate")
    if not args.resume and args.ledger_dir.exists() and any(args.ledger_dir.iterdir()):
        parser.error("Existing diagnostic ledger is immutable and cannot be restarted")
    for name in ("python", "checkpoint", "protocol"):
        if not getattr(args, name).is_file():
            parser.error("Missing required file: --" + name)
    original_started = None
    if args.resume:
        if not args.output_dir.is_dir() or not (args.ledger_dir / "ledger.json").is_file():
            parser.error("--resume requires the original existing output and ledger")
        saved = json.loads((args.ledger_dir / "ledger.json").read_text(encoding="utf-8"))
        if saved.get("output_dir") != str(args.output_dir) or saved.get("ledger_dir") != str(args.ledger_dir):
            parser.error("--resume paths differ from the original diagnostic authority")
        original_started = saved["started_at"]
        index = 1
        while True:
            telemetry = args.output_dir / "telemetry" / f"recovery-{index}"
            try:
                telemetry.mkdir()
                break
            except FileExistsError:
                index += 1
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        telemetry = args.output_dir / "telemetry"
        telemetry.mkdir()
    initial = {"system": system_memory(), "gpu": gpu_memory(), "disk": disk_snapshot(args.output_dir)}
    reason = startup_reason(initial)
    if original_started is not None and time.time() - original_started >= WALL_SECONDS:
        reason = reason or "original_deadline_expired_during_downtime"
    command = [str(args.python), "-B", str(ROOT / "scripts/validate_frozen_wm.py")]
    for name in ("output_dir", "ledger_dir", "official_root", "gpc_root", "checkpoint", "protocol"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name).resolve())])
    if args.resume:
        command.append("--resume")
    nonce = uuid.uuid4().hex
    launch = [str(args.python), "-B", str(ROOT / "scripts/guarded_c_child.py"),
              "--handshake", str(telemetry / "worker.json"), "--ack", str(telemetry / "worker-ack.json"),
              "--nonce", nonce, "--", *command[2:]]
    declaration = {"schema_version": 1, "kind": "tdyssey.frozen_wm_guard.v1", "command": command,
        "launch_command": launch, "source_sha256": provenance(), "initial": initial,
        "resume_same_attempt": args.resume, "original_started_at": original_started,
        "prestart_reason": reason, "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "limits": {**LIMITS, "wall_seconds": WALL_SECONDS},
        "guards": {"sample_seconds": .2, "gpu_sample_seconds": 5, "gpu_maximum_stale_seconds": 15,
            "startup_floors_bytes": {"ram": 4 * GIB, "commit": 6 * GIB, "gpu": 4 * GIB, "disk": 2 * GIB},
            "runtime_floors_bytes": {"ram": GIB, "commit": 2 * GIB, "gpu": GIB}},
        "sampling_limitations": "Observed minima only; polling does not ensure instantaneous floors. No user apps closed."}
    write_json(telemetry / "declaration.json", declaration)
    if reason:
        write_json(telemetry / "monitor-report.json", {"status": "not_started", "guard_reason": reason,
            "initial": initial, "limits": declaration["limits"], "actual_worker_pid": None})
        print(reason)
        return 2
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", HF_HOME=str(ROOT / "artifacts/hf-cache"),
               SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    stopped = threading.Event()
    latest_gpu = {"value": initial["gpu"], "monotonic": time.monotonic()}

    def gpu_sampler():
        try:
            with (telemetry / "gpu.jsonl").open("x", encoding="utf-8") as out:
                while not stopped.is_set():
                    snapshot = gpu_memory(timeout=3)
                    latest_gpu.update(value=snapshot, monotonic=time.monotonic())
                    append(out, snapshot)
                    stopped.wait(5)
                out.flush()
                os.fsync(out.fileno())
        except Exception as exc:
            latest_gpu.update(value={"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                              monotonic=time.monotonic())

    thread = threading.Thread(target=gpu_sampler, daemon=True)
    began = time.monotonic()
    prior_elapsed = max(0.0, time.time()-original_started) if original_started is not None else 0.0
    deadline_remaining = max(0.0, WALL_SECONDS-prior_elapsed)
    count = 0
    minimum_available = minimum_commit = minimum_gpu = None
    peaks = dict.fromkeys(("working_set_bytes", "private_bytes", "peak_working_set_bytes", "peak_pagefile_bytes"), 0)
    child = owned = worker_exit_code = monitor_error = None
    latest_storage = None
    try:
        with (telemetry / "child-output.log").open("x", encoding="utf-8") as output, \
                (telemetry / "memory.jsonl").open("x", encoding="utf-8") as samples:
            child = subprocess.Popen(launch, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            owned = claim_worker(child, telemetry, nonce)
            thread.start()
            while not owned.poll_exit():
                system, process = system_memory(), process_memory(owned.pid)
                elapsed = time.monotonic() - began
                latest_storage = storage_usage(args.output_dir, args.ledger_dir)
                row = {"elapsed_seconds": elapsed, "system": system, "process": process, "storage": latest_storage}
                append(samples, row)
                count += 1
                if system.get("ok"):
                    available, commit = system["physical"]["available_bytes"], system["commit"]["headroom_bytes"]
                    minimum_available = available if minimum_available is None else min(minimum_available, available)
                    minimum_commit = commit if minimum_commit is None else min(minimum_commit, commit)
                if process.get("ok"):
                    for name in peaks:
                        peaks[name] = max(peaks[name], process[name])
                reason = guard_reason(system, GIB, 2 * GIB)
                if not process.get("ok") and not owned.poll_exit():
                    reason = reason or "actual_worker_process_measurement_failed"
                gpu = latest_gpu["value"]
                if gpu.get("ok"):
                    free = min(device["free_bytes"] for device in gpu["devices"])
                    minimum_gpu = free if minimum_gpu is None else min(minimum_gpu, free)
                    if free < GIB:
                        reason = reason or "sampled_gpu_free_below_1GiB"
                else:
                    reason = reason or "gpu_measurement_failed"
                if time.monotonic() - latest_gpu["monotonic"] > 15:
                    reason = reason or "gpu_measurement_stale"
                if elapsed >= deadline_remaining:
                    reason = reason or "deadline_exceeded"
                for name, amount in latest_storage.items():
                    if amount >= LIMITS[name]:
                        reason = reason or (name + "_storage_limit")
                if reason:
                    break
                stopped.wait(.2)
            samples.flush()
            os.fsync(samples.fileno())
    except BaseException as exc:
        reason = reason or "monitor_exception"
        monitor_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if owned is not None and not owned.poll_exit():
                owned.terminate_and_wait(timeout=10)
            if child is not None and child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.kill()
                    child.wait(timeout=10)
        except Exception as exc:
            reason = reason or "child_shutdown_failed"
            monitor_error = f"{type(exc).__name__}: {exc}"
        finally:
            if owned is not None:
                try:
                    worker_exit_code = owned.exit_code()
                    owned.close()
                except Exception as exc:
                    reason = reason or "owned_process_close_failed"
                    monitor_error = f"{type(exc).__name__}: {exc}"
        stopped.set()
        if thread.ident is not None:
            thread.join(timeout=4)
    final = {"system": system_memory(), "gpu": gpu_memory(), "disk": disk_snapshot(args.output_dir)}
    elapsed = time.monotonic() - began
    if elapsed >= deadline_remaining:
        reason = reason or "deadline_exceeded"
    if not latest_gpu["value"].get("ok"):
        reason = reason or "gpu_sampler_failed"
    if owned is not None and worker_exit_code != 0:
        reason = reason or "actual_worker_exit_failed_or_unconfirmed"
    if child is None or child.returncode != 0:
        reason = reason or "launcher_exit_failed_or_unconfirmed"
    if not final["system"].get("ok") or not final["gpu"].get("ok"):
        reason = reason or "final_resource_measurement_failed"
    try:
        latest_storage = storage_usage(args.output_dir, args.ledger_dir)
        for name, amount in latest_storage.items():
            if amount >= LIMITS[name]:
                reason = reason or (name + "_storage_limit")
        if not reason and child is not None and child.returncode == 0:
            result = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
            ledger = json.loads((args.ledger_dir / "ledger.json").read_text(encoding="utf-8"))
            if result.get("status") not in ("complete", "completed", "complete_with_aborted_branches") or ledger.get("pending"):
                raise ValueError("Diagnostic completion report is incomplete or reservations remain pending")
            if ledger.get("policy_optimizer_updates") != 0 or ledger.get("world_model_optimizer_updates") != 0:
                raise ValueError("Diagnostic ledger reports nonzero or missing optimizer-update counts")
            after = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            if after != declaration["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint changed during diagnostic")
    except Exception as exc:
        reason = reason or "child_acceptance_report_failed"
        monitor_error = monitor_error or f"{type(exc).__name__}: {exc}"
    report = {"schema_version": 1, "kind": "tdyssey.frozen_wm_guard.v1", "status": "stopped" if reason else "exited",
        "guard_reason": reason, "monitor_error": monitor_error, "initial": initial,
        "launcher_pid": None if child is None else child.pid, "actual_worker_pid": None if owned is None else owned.pid,
        "kill_on_parent_job_close": owned is not None, "worker_exit_code": worker_exit_code,
        "returncode": None if child is None else child.returncode, "samples": count, "elapsed_seconds": elapsed,
        "deadline_seconds": WALL_SECONDS, "original_started_at": original_started,
        "prior_elapsed_including_downtime_seconds": prior_elapsed, "remaining_deadline_at_launch_seconds": deadline_remaining,
        "same_attempt_elapsed_seconds": prior_elapsed+elapsed, "resume_same_attempt": args.resume,
        "sampled_minimum_available_bytes": minimum_available,
        "sampled_minimum_commit_headroom_bytes": minimum_commit, "sampled_minimum_gpu_free_bytes": minimum_gpu,
        "process_observed_peaks": peaks, "storage": latest_storage, "final": final,
        "ended_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(telemetry / "monitor-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not reason and child is not None and child.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
