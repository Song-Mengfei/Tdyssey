"""Offline audit of completed paired-evaluation telemetry; stdlib only.

This script refuses an absent/nonterminal monitor report. It never imports the
project runtime, queries Windows/GPU resources, polls a PID or starts a child.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


META = Path(__file__).resolve().parent
ROOT = META.parent.parent
GIB = 1024**3
SGT = timezone(timedelta(hours=8))
METHODS = ("base_training", "delta_training")


def stamp(utc):
    return {"utc": utc, "sgt": datetime.fromisoformat(utc).astimezone(SGT).isoformat()}


def size(number):
    return {"bytes": number, "gib": number / GIB}


def extreme(previous, number, at, *, minimum=False, **extra):
    if previous is None or (number < previous["bytes"] if minimum else number > previous["bytes"]):
        return {**size(number), **stamp(at), **extra}
    return previous


class Audit:
    def __init__(self):
        self.sources = {}

    def raw(self, path):
        raw = path.read_bytes()
        self.sources[str(path)] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return raw

    def json(self, path):
        return json.loads(self.raw(path).decode("utf-8-sig"))

    def optional_json(self, path):
        if not path.is_file():
            return {"audit_error": "missing_file", "path": str(path)}
        try:
            return self.json(path)
        except (OSError, ValueError) as error:
            return {"audit_error": type(error).__name__, "message": str(error), "path": str(path)}

    def lines(self, path, errors):
        digest = hashlib.sha256()
        total = 0
        with path.open("rb") as source:
            for number, raw in enumerate(source, 1):
                digest.update(raw)
                total += len(raw)
                try:
                    yield number, json.loads(raw.decode("utf-8-sig"))
                except (ValueError, UnicodeError) as error:
                    errors.append({"line": number, "error": str(error)})
        self.sources[str(path)] = {"bytes": total, "sha256": digest.hexdigest()}


def host_samples(audit, path, guards):
    result = {"count": 0, "invalid_system_count": 0, "invalid_process_count": 0,
              "invalid_examples": [], "json_errors": [], "process_pids": [],
              "minimum_available": None, "minimum_commit_headroom": None,
              "maximum_system_commit_total": None, "process_maxima": {},
              "host_floor_violations": 0, "max_sample_gap_seconds": 0.0,
              "clock_backwards_count": 0}
    pids = set()
    previous = None
    fields = ("working_set_bytes", "private_bytes", "pagefile_bytes", "peak_working_set_bytes", "peak_pagefile_bytes")
    for line, row in audit.lines(path, result["json_errors"]):
        result["count"] += 1
        system, process = row.get("system", {}), row.get("process", {})
        at = system.get("at_utc")
        pids.add(process.get("pid"))
        try:
            now = datetime.fromisoformat(at)
            if "first_sample" not in result:
                result["first_sample"] = stamp(at)
            result["last_sample"] = stamp(at)
            if previous is not None:
                gap = (now - previous).total_seconds()
                result["clock_backwards_count"] += int(gap < 0)
                result["max_sample_gap_seconds"] = max(result["max_sample_gap_seconds"], gap)
            previous = now
            valid = system.get("ok") is True and all(
                type(system.get(group, {}).get(key)) is int and system[group][key] >= 0
                for group, key in (("physical", "available_bytes"), ("physical", "total_bytes"),
                                   ("commit", "headroom_bytes"), ("commit", "total_bytes"), ("commit", "limit_bytes")))
            if not valid:
                raise ValueError("unsuccessful or malformed system memory observation")
            available, commit = system["physical"]["available_bytes"], system["commit"]["headroom_bytes"]
            result["minimum_available"] = extreme(result["minimum_available"], available, at, minimum=True)
            result["minimum_commit_headroom"] = extreme(result["minimum_commit_headroom"], commit, at, minimum=True)
            result["maximum_system_commit_total"] = extreme(result["maximum_system_commit_total"], system["commit"]["total_bytes"], at)
            result["host_floor_violations"] += int(available < guards["minimum_available_bytes"] or commit < guards["minimum_commit_bytes"])
        except (TypeError, ValueError, KeyError) as error:
            result["invalid_system_count"] += 1
            if len(result["invalid_examples"]) < 20:
                result["invalid_examples"].append({"line": line, "kind": "system", "error": str(error), "measurement_error": system.get("error")})
        try:
            if process.get("ok") is not True or any(type(process.get(key)) is not int or process[key] < 0 for key in fields):
                raise ValueError("unsuccessful or malformed process memory observation")
            for key in fields:
                result["process_maxima"][key] = extreme(result["process_maxima"].get(key), process[key], process["at_utc"])
        except (TypeError, ValueError, KeyError) as error:
            result["invalid_process_count"] += 1
            if len(result["invalid_examples"]) < 20:
                result["invalid_examples"].append({"line": line, "kind": "process", "error": str(error), "measurement_error": process.get("error")})
    result["process_pids"] = sorted(pids, key=repr)
    return result


def gpu_samples(audit, path, guards):
    result = {"count": 0, "invalid_count": 0, "invalid_examples": [], "json_errors": [],
              "minimum_free": None, "maximum_used": None, "floor_violations": 0,
              "max_sample_gap_seconds": 0.0, "clock_backwards_count": 0, "devices": []}
    previous, devices = None, set()
    for line, row in audit.lines(path, result["json_errors"]):
        result["count"] += 1
        try:
            at = row["at_utc"]
            now = datetime.fromisoformat(at)
            if previous is not None:
                gap = (now - previous).total_seconds()
                result["clock_backwards_count"] += int(gap < 0)
                result["max_sample_gap_seconds"] = max(result["max_sample_gap_seconds"], gap)
            previous = now
            if row.get("ok") is not True or not row.get("devices"):
                raise ValueError("unsuccessful GPU observation")
            for device in row["devices"]:
                if any(type(device.get(key)) is not int or device[key] < 0 for key in ("index", "total_bytes", "free_bytes", "used_bytes")):
                    raise ValueError("malformed GPU counters")
                devices.add((device["index"], device["name"], device["total_bytes"]))
                result["minimum_free"] = extreme(result["minimum_free"], device["free_bytes"], at, minimum=True, device_index=device["index"])
                result["maximum_used"] = extreme(result["maximum_used"], device["used_bytes"], at, device_index=device["index"])
                result["floor_violations"] += int(device["free_bytes"] < guards["minimum_gpu_free_bytes"])
        except (TypeError, ValueError, KeyError) as error:
            result["invalid_count"] += 1
            if len(result["invalid_examples"]) < 20:
                result["invalid_examples"].append({"line": line, "error": str(error), "measurement_error": row.get("error")})
    result["devices"] = [{"index": index, "name": name, "total": size(total)} for index, name, total in sorted(devices)]
    return result


def log_audit(audit, directory):
    result = {}
    fatal = re.compile(r"Traceback|OutOfMemoryError|MemoryError|CUDA out of memory|CUDA error|device-side assert|^FAIL:|Failure telemetry unavailable", re.I)
    for name in ("child-output.log", "init-output.log"):
        path = directory / name
        if not path.is_file():
            result[name] = {"audit_error": "missing_file"}
            continue
        lines = audit.raw(path).decode("utf-8-sig", errors="replace").splitlines()
        result[name] = {"line_count": len(lines),
            "fatal_pattern_matches": [{"line": index, "text": line} for index, line in enumerate(lines, 1) if fatal.search(line)],
            "warning_lines": [{"line": index, "text": line} for index, line in enumerate(lines, 1)
                              if "UserWarning:" in line or line.startswith("Gym has been unmaintained")]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-dir", type=Path, default=META / "monitor")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/milestone-c-evaluation-20261024")
    parser.add_argument("--output", type=Path, default=META / "resources-review-final.json")
    parser.add_argument("--expected-pid", type=int, default=7996)
    parser.add_argument("--expected-launcher-pid", type=int, default=32552)
    args = parser.parse_args()
    audit = Audit()
    # Mandatory gate before reading any potentially active JSONL or writing.
    monitor_path = args.monitor_dir / "monitor-report.json"
    if not monitor_path.is_file():
        raise FileNotFoundError("Final monitor-report.json is absent; refusing an active-run audit")
    monitor = audit.json(monitor_path)
    if (monitor.get("status") not in ("exited", "stopped") or not monitor.get("ended_at_utc")
            or type(monitor.get("elapsed_seconds")) not in (int, float)):
        raise ValueError("Monitor report is not a terminal resource report; refusing an active-run audit")
    declaration = audit.json(args.monitor_dir / "declaration.json")
    worker = audit.optional_json(args.monitor_dir / "worker.json")
    claimed = audit.optional_json(args.monitor_dir / "worker-claimed.json")
    host = host_samples(audit, args.monitor_dir / "memory.jsonl", declaration["guards"])
    gpu = gpu_samples(audit, args.monitor_dir / "gpu.jsonl", declaration["guards"])
    logs = log_audit(audit, args.monitor_dir)
    groups = {method: audit.optional_json(args.run_dir / (method + ".json")) for method in METHODS}
    releases = {method: audit.optional_json(args.run_dir / (method + "-released.json")) for method in METHODS}
    pair = audit.optional_json(args.run_dir / "evaluation-pair-report.json")
    failure = audit.optional_json(args.run_dir / "evaluation-failure.json") if (args.run_dir / "evaluation-failure.json").exists() else None
    cuda = {method: {"group_end": {key: size(value) for key, value in groups[method].get("resources", {}).items() if type(value) is int},
                     "after_release": {key: size(value) for key, value in releases[method].get("resources", {}).items() if type(value) is int},
                     "network_released": releases[method].get("network_released")}
            for method in METHODS}
    checks = {
        "actual_worker_identity": monitor.get("pid") == worker.get("pid") == claimed.get("identity", {}).get("pid") == args.expected_pid,
        "launcher_identity": monitor.get("launcher_pid") == worker.get("parent_pid") == claimed.get("launcher_pid") == args.expected_launcher_pid,
        "all_memory_samples_target_actual_worker": host["process_pids"] == [args.expected_pid],
        "all_host_samples_valid": host["count"] > 0 and host["invalid_system_count"] == 0 and not host["json_errors"],
        "all_process_samples_valid": host["count"] > 0 and host["invalid_process_count"] == 0 and not host["json_errors"],
        "all_gpu_samples_valid": gpu["count"] > 0 and gpu["invalid_count"] == 0 and not gpu["json_errors"],
        "host_sample_count_matches": host["count"] == monitor.get("samples"),
        "sampled_minimum_available_matches": (host["minimum_available"] or {}).get("bytes") == monitor.get("sampled_minimum_available_bytes"),
        "sampled_minimum_commit_matches": (host["minimum_commit_headroom"] or {}).get("bytes") == monitor.get("sampled_minimum_commit_headroom_bytes"),
        "sampled_minimum_gpu_matches": (gpu["minimum_free"] or {}).get("bytes") == monitor.get("sampled_minimum_gpu_free_bytes"),
        "process_maxima_match_monitor": all(host["process_maxima"].get(key, {}).get("bytes") == value
            for key, value in monitor.get("process_observed_peaks", {}).items()) and bool(monitor.get("process_observed_peaks")),
        "no_observed_floor_violation": host["host_floor_violations"] == gpu["floor_violations"] == 0,
        "worker_and_launcher_exit_zero": monitor.get("worker_exit_code") == monitor.get("returncode") == 0,
        "monitor_clean_exit": monitor.get("status") == "exited" and monitor.get("guard_reason") is None and monitor.get("monitor_error") is None,
        "model_releases_verified": all(releases[method].get("network_released") is True for method in METHODS),
        "both_group_reports_completed": all(groups[method].get("status") == "completed" and groups[method].get("method") == method for method in METHODS),
        "both_group_cuda_records_present": all("peak_allocated_bytes" in groups[method].get("resources", {}) for method in METHODS),
        "pair_completed_without_training": pair.get("status") == "completed" and type(pair.get("training_steps")) is int
            and pair["training_steps"] == 0 and pair.get("comparison", {}).get("available") is True,
        "no_failure_report": failure is None,
        "no_fatal_log_pattern": all("audit_error" not in record and not record["fatal_pattern_matches"] for record in logs.values()),
    }
    ended = datetime.fromisoformat(monitor["ended_at_utc"])
    duration = monitor["elapsed_seconds"]
    report = {"schema_version": 1, "reviewed_at": stamp(datetime.now(timezone.utc).isoformat()),
        "scope": "Offline audit only; no Torch/project runtime imports, live queries, active polling or source changes.",
        "launch_context_from_orchestrator": {"source_commit_prefix": "1c4c66a", "expected_actual_pid": args.expected_pid,
                                             "expected_launcher_pid": args.expected_launcher_pid},
        "checks": checks, "all_checks_passed": all(checks.values()),
        "identity": {"actual_pid": monitor.get("pid"), "launcher_pid": monitor.get("launcher_pid"),
                     "kill_on_parent_job_close_recorded": claimed.get("kill_on_parent_job_close")},
        "timing": {"monitor_start_inferred_from_end_minus_elapsed": stamp((ended - timedelta(seconds=duration)).isoformat()),
                   "monitor_end": stamp(monitor["ended_at_utc"]), "elapsed_seconds": duration, "elapsed_minutes": duration / 60},
        "host_memory_sampling": host, "gpu_driver_sampling": gpu, "cuda_per_group": cuda,
        "post_exit_snapshot": monitor.get("final"), "guards": declaration["guards"],
        "child_logs": logs, "failure_report": failure,
        "group_status": {method: groups[method].get("status", groups[method].get("audit_error")) for method in METHODS},
        "pair_status": pair.get("status", pair.get("audit_error")),
        "limitations": [
            "Polling minima are observed values, not guaranteed instantaneous floors; host target0.2s and GPU target5s can miss transients.",
            "OS lifetime peak counters are maxima observed up to the final successful sample, not a post-exit query. Current private/working maxima are sampled values.",
            "PeakPagefileUsage is process committed/pagefile-accounted memory, not bytes necessarily resident in the page file.",
            "nvidia-smi covers the whole GPU including other applications; PyTorch allocated/reserved counters cover the allocator and are separate evidence.",
            "CUDA allocator peaks may be cumulative across groups unless explicitly reset; each group record is reported without claiming an isolated group-only peak.",
            "System CommitPeak is historical since boot, not this evaluation's peak; sampled system CommitTotal also includes other processes.",
            "Post-exit observations show recorded resource recovery; changes cannot all be attributed solely to the worker.",
            "Log pattern scanning is supplementary evidence, not proof that every possible error is absent."],
        "sources": audit.sources}
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "all_checks_passed": report["all_checks_passed"],
        "checks": checks, "identity": report["identity"], "timing": report["timing"],
        "minimum_available": host["minimum_available"], "minimum_commit_headroom": host["minimum_commit_headroom"],
        "minimum_gpu_free": gpu["minimum_free"], "process_maxima": host["process_maxima"],
        "cuda_per_group": cuda, "samples": {"host": host["count"], "gpu": gpu["count"]}}, indent=2, sort_keys=True))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"AUDIT REFUSED/FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
