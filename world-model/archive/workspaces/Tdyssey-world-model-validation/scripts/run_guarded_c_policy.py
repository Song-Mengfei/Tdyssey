"""Run one explicitly selected C child with sampled Windows memory guards.

No torch import in this parent. Floors are operational stop rules, not a
guarantee against transient pressure between samples. Only our child is stopped.
Load mode never initializes the permanent C experiment account.
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
from tdyssey.policy_memory import disk_snapshot, gpu_memory, guard_reason, process_memory, system_memory

GIB = 1024**3


def write_json(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def append(out, value):
    out.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    out.flush()


def provenance():
    files = [*(ROOT / "tdyssey").rglob("*.py"),
             ROOT / "scripts/run_c_policy.py", ROOT / "scripts/probe_c_policy_load.py",
             ROOT / "scripts/guarded_c_child.py",
             Path(__file__), ROOT / "configs/policy_c.example.json"]
    pair_entry = ROOT / "scripts/evaluate_c_policy_pair.py"
    if pair_entry.is_file():
        files.append(pair_entry)
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def runtime_sources(sources):
    normalized = {name.replace("\\", "/"): digest for name, digest in sources.items()}
    return {name: digest for name, digest in normalized.items()
            if (name.startswith("tdyssey/") and name != "tdyssey/policy_memory.py")
            or name in ("scripts/run_c_policy.py", "scripts/probe_c_policy_load.py",
                        "configs/policy_c.example.json")}


def claim_worker(launcher, directory, nonce):
    """Claim the real interpreter, including Windows venv redirector children."""
    from tdyssey.policy_memory import OwnedProcess
    path = directory / "worker.json"
    began = time.monotonic()
    while not path.exists():
        if launcher.poll() is not None or time.monotonic() - began >= 10:
            raise RuntimeError("Actual interpreter did not provide its bounded bootstrap handshake")
        time.sleep(0.05)
    identity = json.loads(path.read_text(encoding="utf-8"))
    if identity.get("nonce") != nonce:
        raise ValueError("Actual interpreter handshake nonce differs")
    owned = OwnedProcess(identity["pid"], launcher.pid)
    try:
        snapshot = process_memory(owned.pid)
        if not snapshot.get("ok"):
            raise RuntimeError("Actual interpreter memory observation failed before ACK")
        system = system_memory()
        reason = guard_reason(system, minimum_available_bytes=4 * GIB, minimum_commit_bytes=6 * GIB)
        if reason:
            raise RuntimeError(reason)
        write_json(directory / "worker-claimed.json", {"identity": identity, "launcher_pid": launcher.pid,
            "process": snapshot, "system": system, "kill_on_parent_job_close": True})
        temporary = directory / "worker-ack.tmp"
        write_json(temporary, {"pid": owned.pid, "nonce": nonce})
        os.replace(temporary, directory / "worker-ack.json")
        return owned
    except BaseException:
        owned.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("load", "train", "evaluate"))
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/policy_c.example.json")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id")
    parser.add_argument("--registry", type=Path, default=ROOT / "artifacts/policy-c-registry")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Selected trained delta, required for evaluate")
    parser.add_argument("--training-ledger", type=Path, help="Read-only training identity/counters, required for evaluate")
    parser.add_argument("--load-report", type=Path, help="Successful same-source load report, required before train")
    args = parser.parse_args()
    if args.phase == "train" and (not args.experiment_id or not args.run_dir or not args.load_report):
        parser.error("train requires --experiment-id, --run-dir and --load-report")
    if args.phase == "evaluate" and (not args.experiment_id or not args.run_dir or not args.checkpoint or not args.training_ledger):
        parser.error("evaluate requires --experiment-id, --run-dir, --checkpoint and --training-ledger")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    initial = {"system": system_memory(), "gpu": gpu_memory(), "disk": disk_snapshot(ROOT)}
    reason = guard_reason(initial["system"], minimum_available_bytes=4 * GIB,
                          minimum_commit_bytes=6 * GIB)
    if not initial["gpu"].get("ok"):
        reason = reason or "initial_gpu_measurement_failed"
    elif min(d["free_bytes"] for d in initial["gpu"]["devices"]) < 4 * GIB:
        reason = reason or "initial_gpu_free_below_4GiB"
    if not initial["disk"].get("ok") or initial["disk"].get("free_bytes", 0) < 2 * GIB:
        reason = reason or "initial_disk_headroom_below_2GiB_or_measurement_failed"
    common = ["--config", str(args.config), "--source-root", str(args.source_root),
              "--base-checkpoint", str(args.base_checkpoint)]
    command = [str(args.python), "-B"]
    if args.phase == "load":
        command += [str(ROOT / "scripts/probe_c_policy_load.py"), *common,
                    "--output-dir", str(args.output_dir)]
    elif args.phase == "train":
        command += [str(ROOT / "scripts/run_c_policy.py"), "train", "--execute", *common,
                    "--experiment-id", args.experiment_id, "--registry", str(args.registry),
                    "--run-dir", str(args.run_dir), "--controls", "2048", "--batches", "24"]
    else:
        command += [str(ROOT / "scripts/evaluate_c_policy_pair.py"), "run", "--execute",
                    "--account-id", args.experiment_id, "--registry", str(args.registry),
                    "--run-dir", str(args.run_dir), *common, "--checkpoint", str(args.checkpoint),
                    "--training-ledger", str(args.training_ledger)]
    nonce = uuid.uuid4().hex
    launch_command = [str(args.python), "-B", str(ROOT / "scripts/guarded_c_child.py"),
        "--handshake", str(args.output_dir.resolve() / "worker.json"),
        "--ack", str(args.output_dir.resolve() / "worker-ack.json"), "--nonce", nonce,
        "--", *command[2:]]
    declaration = {"schema_version": 1, "phase": args.phase, "command": command, "launch_command": launch_command,
        "initial": initial, "prestart_reason": reason, "source_sha256": provenance(),
        "config_path": str(args.config.resolve()), "config_file_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "guards": {"sample_seconds": 0.2, "minimum_available_bytes": GIB,
                   "minimum_commit_bytes": 2 * GIB, "gpu_sample_seconds": 5,
                   "minimum_gpu_free_bytes": GIB, "deadline_seconds": 180 if args.phase == "load" else 7200,
                   "telemetry_limit_bytes": 64 * 1024**2},
        "limits": "Observed minima only. Polling cannot guarantee instantaneous floors. No user processes are closed."}
    write_json(args.output_dir / "declaration.json", declaration)
    if args.phase == "train" and not reason:
        try:
            from tdyssey.policy_protocol import CConfig
            loaded = json.loads(args.load_report.read_text(encoding="utf-8"))
            prior = json.loads((args.load_report.parent / "monitor-report.json").read_text(encoding="utf-8"))
            binding = json.loads((args.load_report.parent / "declaration.json").read_text(encoding="utf-8"))
            if (loaded.get("status") != "complete" or loaded.get("kind") != "tdyssey.C.load_only_probe"
                    or loaded.get("config_hash") != CConfig.from_json(args.config).config_hash
                    or loaded.get("actor_shares_official_network") is not True
                    or prior.get("status") != "exited" or prior.get("returncode") != 0 or prior.get("guard_reason")
                    or runtime_sources(binding.get("source_sha256", {})) != runtime_sources(declaration["source_sha256"])):
                raise ValueError("load success/configuration/source/released-child evidence differs")
        except Exception as exc:
            reason = "required_load_evidence_failed: " + str(exc)
    if reason:
        write_json(args.output_dir / "monitor-report.json", {"status": "not_started", "reason": reason, "initial": initial})
        print(reason)
        return 2
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", HF_HOME=str(ROOT / "artifacts/hf-cache"),
               SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    if args.phase == "train":
        # The permanent wall clock begins here, immediately before actual training.
        init_command = [str(args.python), "-S", "-B", str(ROOT / "scripts/run_c_policy.py"),
                        "init", "--execute", "--config", str(args.config),
                        "--experiment-id", args.experiment_id, "--registry", str(args.registry),
                        "--run-dir", str(args.run_dir)]
        with (args.output_dir / "init-output.log").open("x", encoding="utf-8") as out:
            initialized = subprocess.run(init_command, cwd=ROOT, env=env, stdout=out,
                                         stderr=subprocess.STDOUT, timeout=30, check=False)
        if initialized.returncode:
            write_json(args.output_dir / "monitor-report.json", {"status": "init_failed", "returncode": initialized.returncode})
            return 2
    elif args.phase == "evaluate":
        # A separate evaluation account starts only immediately before its worker.
        # Init is stdlib-only and never initializes or resumes the training clock.
        init_command = [str(args.python), "-S", "-B", str(ROOT / "scripts/evaluate_c_policy_pair.py"),
                        "init", *command[4:]]
        with (args.output_dir / "init-output.log").open("x", encoding="utf-8") as out:
            initialized = subprocess.run(init_command, cwd=ROOT, env=env, stdout=out,
                                         stderr=subprocess.STDOUT, timeout=30, check=False)
        if initialized.returncode:
            write_json(args.output_dir / "monitor-report.json", {"status": "init_failed", "returncode": initialized.returncode})
            return 2
    stopped = threading.Event()
    latest_gpu = {"value": initial["gpu"], "monotonic": time.monotonic()}

    def gpu_sampler():
        try:
            with (args.output_dir / "gpu.jsonl").open("x", encoding="utf-8") as out:
                while not stopped.is_set():
                    snapshot = gpu_memory(timeout=3.0)
                    latest_gpu["value"] = snapshot
                    latest_gpu["monotonic"] = time.monotonic()
                    append(out, snapshot)
                    stopped.wait(5)
        except Exception as exc:
            latest_gpu["value"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            latest_gpu["monotonic"] = time.monotonic()

    thread = threading.Thread(target=gpu_sampler, daemon=True)
    began = time.monotonic()
    count = 0
    minimum_available = minimum_commit = minimum_gpu = None
    peaks = {name: 0 for name in ("working_set_bytes", "private_bytes", "peak_working_set_bytes", "peak_pagefile_bytes")}
    child = None
    owned = None
    worker_exit_code = None
    monitor_error = None
    try:
        with (args.output_dir / "child-output.log").open("x", encoding="utf-8") as output, \
                (args.output_dir / "memory.jsonl").open("x", encoding="utf-8") as samples:
            child = subprocess.Popen(launch_command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            owned = claim_worker(child, args.output_dir, nonce)
            thread.start()
            while not owned.poll_exit():
                system, process = system_memory(), process_memory(owned.pid)
                row = {"elapsed_seconds": time.monotonic() - began, "system": system, "process": process}
                append(samples, row)
                count += 1
                if system.get("ok"):
                    available = system["physical"]["available_bytes"]
                    commit = system["commit"]["headroom_bytes"]
                    minimum_available = available if minimum_available is None else min(minimum_available, available)
                    minimum_commit = commit if minimum_commit is None else min(minimum_commit, commit)
                if process.get("ok"):
                    for key in peaks:
                        peaks[key] = max(peaks[key], process[key])
                reason = guard_reason(system, minimum_available_bytes=GIB, minimum_commit_bytes=2 * GIB)
                if not process.get("ok") and not owned.poll_exit():
                    reason = reason or "child_process_measurement_failed"
                gpu = latest_gpu["value"]
                if gpu.get("ok"):
                    free = min(d["free_bytes"] for d in gpu["devices"])
                    minimum_gpu = free if minimum_gpu is None else min(minimum_gpu, free)
                    if free < GIB:
                        reason = reason or "sampled_gpu_free_below_floor"
                else:
                    reason = reason or "gpu_measurement_failed"
                if time.monotonic() - latest_gpu["monotonic"] > 15:
                    reason = reason or "gpu_measurement_stale"
                if row["elapsed_seconds"] >= declaration["guards"]["deadline_seconds"]:
                    reason = reason or "deadline_exceeded"
                if samples.tell() >= declaration["guards"]["telemetry_limit_bytes"]:
                    reason = reason or "telemetry_storage_limit"
                if reason:
                    break
                stopped.wait(0.2)
            samples.flush()
            os.fsync(samples.fileno())
            if time.monotonic() - began >= declaration["guards"]["deadline_seconds"]:
                reason = reason or "deadline_exceeded"
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
    final = {"system": system_memory(), "gpu": gpu_memory(), "disk": disk_snapshot(ROOT)}
    if not latest_gpu["value"].get("ok"):
        reason = reason or "gpu_sampler_failed"
    if time.monotonic() - began >= declaration["guards"]["deadline_seconds"]:
        reason = reason or "deadline_exceeded"
    if owned is not None and worker_exit_code != 0:
        reason = reason or "actual_worker_exit_failed_or_unconfirmed"
    if not final["system"].get("ok") or not final["gpu"].get("ok"):
        reason = reason or "final_resource_measurement_failed"
    if not reason and child is not None and child.returncode == 0:
        try:
            if args.phase == "load":
                ready = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
                if ready.get("status") != "complete" or ready.get("ready", {}).get("stage") != "critic_optimizer_ready":
                    raise ValueError("Load child did not complete its final measured ready stage")
            elif args.phase == "train":
                reports = list(args.run_dir.glob("train-report-*.json"))
                result = json.loads(reports[-1].read_text(encoding="utf-8"))["result"]
                if result.get("first_batch_gate", {}).get("status") != "passed":
                    raise ValueError("Training did not pass its first in-budget batch gate")
            else:
                ready = json.loads((args.run_dir / "evaluation-pair-report.json").read_text(encoding="utf-8"))
                groups = ready.get("groups")
                if (ready.get("status") != "completed" or type(ready.get("training_steps")) is not int
                        or ready["training_steps"] != 0 or not isinstance(groups, list) or len(groups) != 2):
                    raise ValueError("Paired evaluation did not complete exactly two groups with zero training steps")
                if any(not isinstance(group, dict) or group.get("status") not in ("complete", "completed") for group in groups):
                    raise ValueError("Paired evaluation contains an incomplete group")
                if [group.get("method") for group in groups] != ["base_training", "delta_training"]:
                    raise ValueError("Paired evaluation methods/order differ from base_training then delta_training")
                if not isinstance(ready.get("comparison"), dict) or ready["comparison"].get("available") is not True:
                    raise ValueError("Paired evaluation did not provide its completed comparison")
        except Exception as exc:
            reason = "child_acceptance_report_failed: " + str(exc)
    report = {"schema_version": 1, "phase": args.phase, "status": "stopped" if reason else "exited",
        "guard_reason": reason, "monitor_error": monitor_error,
        "launcher_pid": None if child is None else child.pid, "pid": None if owned is None else owned.pid,
        "worker_exit_code": worker_exit_code,
        "returncode": None if child is None else child.returncode, "samples": count,
        "elapsed_seconds": time.monotonic() - began, "sampled_minimum_available_bytes": minimum_available,
        "sampled_minimum_commit_headroom_bytes": minimum_commit, "sampled_minimum_gpu_free_bytes": minimum_gpu,
        "process_observed_peaks": peaks, "final": final,
        "ended_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output_dir / "monitor-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not reason and child is not None and child.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
