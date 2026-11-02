"""Bounded reset-only C scene audit, independent of every training ledger.

Default plan uses stdlib and creates no files. Run requires --execute and a new
diagnostic directory. Only reset's internal settle step is authorized here;
policy loading, sampling, controller steps and optimization are absent.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAX_RESETS = 212
MAX_SECONDS = 300
MAX_BYTES = 8 * 1024**2


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", nargs="?", choices=("plan", "run"), default="plan")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--config", type=Path, default=ROOT / "configs/policy_c.example.json")
    result.add_argument("--source-root", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--preserve-file", type=Path, action="append", default=[])
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")


def write_json(path, value):
    payload = json_bytes(value)
    used = sum(item.stat().st_size for item in path.parent.iterdir() if item.is_file())
    if used + len(payload) > MAX_BYTES:
        raise RuntimeError("Reset diagnostic storage cap would be exceeded; existing evidence retained")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def resource_snapshot(path):
    """Read current host/process resources without importing a tensor library."""
    result = {"at_utc": utc_now(), "platform": platform.platform(),
              "process_id": os.getpid(), "scope": "reset-only process; no policy model loaded"}
    disk = shutil.disk_usage(path)
    result["disk"] = {"path": str(Path(path).resolve()), "total_bytes": disk.total,
                      "free_bytes": disk.free, "used_bytes": disk.used}
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    *[(name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical",
                    "total_pagefile", "available_pagefile", "total_virtual", "available_virtual", "extended")]]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            memory = MemoryStatus()
            memory.length = ctypes.sizeof(memory)
            kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
            kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
            if not kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
                raise ctypes.WinError(ctypes.get_last_error())
            result["ram"] = {"source": "Win32 GlobalMemoryStatusEx",
                "total_bytes": memory.total_physical, "available_bytes": memory.available_physical,
                "load_percent": memory.load}

            class ProcessCounters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in ("peak_working_set", "working_set",
                    "peak_paged_pool", "paged_pool", "peak_nonpaged_pool", "nonpaged_pool",
                    "pagefile", "peak_pagefile", "private")]]

            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessCounters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = ProcessCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            result["process_memory"] = {"source": "Win32 GetProcessMemoryInfo",
                "peak_working_set_bytes": counters.peak_working_set,
                "working_set_bytes": counters.working_set, "private_bytes": counters.private,
                "peak_pagefile_bytes": counters.peak_pagefile}
        except (OSError, ValueError) as exc:
            result["windows_memory_error"] = f"{type(exc).__name__}: {exc}"
    try:
        command = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used",
                   "--format=csv,noheader,nounits"]
        output = subprocess.check_output(command, text=True, timeout=5)
        result["gpu"] = {"source_command": command, "devices": [
            {"name": row[0].strip(), "total_mib": int(row[1]), "free_mib": int(row[2]), "used_mib": int(row[3])}
            for row in csv.reader(output.splitlines())]}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["gpu_error"] = f"{type(exc).__name__}: {exc}"
    return result


def provenance(source_root):
    from tdyssey.official import official_spec
    paths = ("tdyssey/envs/official.py", "tdyssey/collection.py", "tdyssey/official.py",
             "tdyssey/policy_protocol.py", "tdyssey/policy_preflight.py",
             "tdyssey/policy_reset_probe.py", "scripts/preflight_c_scenes.py")
    versions = {}
    for package in ("numpy", "pymunk", "pygame", "gym", "shapely", "opencv-python", "scikit-image"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), "rev-parse", "HEAD"]
    spec = official_spec()
    return {"code_parent_commit": subprocess.check_output(command, text=True).strip(),
        "code_sha256": {path: digest_file(ROOT / path) for path in paths},
        "python": sys.version, "executable": sys.executable, "dependencies": versions,
        "source_root": str(Path(source_root).resolve()), "official_source_commit": spec["source_commit"],
        "official_source_sha256": spec["source_sha256"], "adapter_protocol": "official-t-l-com-v1",
        "legacy": True, "checkpoint_loaded": False, "policy_loaded": False}


class AuditedProbe:
    """Fsync a claim before each probe, and the result before the next one."""

    def __init__(self, probe, output_dir, started):
        self.probe, self.output_dir, self.started = probe, output_dir, started
        self.calls = 0
        self.resets = self.physics = self.controls = 0
        self.pending_calls = {}
        self.journal = output_dir / "attempts.jsonl"

    def _append(self, value):
        payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        used = sum(path.stat().st_size for path in self.output_dir.iterdir() if path.is_file())
        if used + len(payload) > MAX_BYTES:
            raise RuntimeError("Reset diagnostic storage cap exhausted; journal retained")
        with self.journal.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def __call__(self, shape, seed, phase):
        self.calls += 1
        claim = {"event": "attempt_reserved", "id": self.calls, "shape": shape, "seed": seed,
                 "phase": phase, "at_utc": utc_now()}
        self._append(claim)
        self.pending_calls[self.calls] = claim
        if (self.calls > MAX_RESETS or self.resets >= MAX_RESETS or self.physics >= MAX_RESETS
                or time.perf_counter() - self.started >= MAX_SECONDS):
            result = {"valid": False, "issues": [], "error": "Independent reset diagnostic limit exhausted",
                      "reset_attempts": 0, "reset_physics_substeps": 0, "control_steps": 0}
        else:
            try:
                result = self.probe(shape, seed, phase)
            except BaseException as exc:
                self._append({**claim, "event": "attempt_interrupted", "counts_unknown": True,
                              "error": f"{type(exc).__name__}: {exc}"})
                raise
        self.resets += result["reset_attempts"]
        self.physics += result["reset_physics_substeps"]
        self.controls += result["control_steps"]
        self._append({**claim, "event": "attempt_result", "result": result})
        del self.pending_calls[self.calls]
        return result


def declared_plan(config):
    return {"status": "proposed_not_executed", "config_hash": config.config_hash,
        "candidate_order": {"T": [400032, 400063], "L": [500032, 500063]},
        "candidate_order_inclusive_ascending": True, "maximum_reset_attempts": MAX_RESETS,
        "maximum_settle_substeps": MAX_RESETS, "maximum_seconds": MAX_SECONDS,
        "maximum_bytes": MAX_BYTES, "policy_samples": 0, "control_steps": 0,
        "optimizer_steps": 0, "formal_C_experiment_initialized": False, "formal_C_clock_started": False}


def run(args, config):
    from tdyssey.policy_preflight import run_reset_preflight
    from tdyssey.policy_reset_probe import RuntimeResetProbe
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new empty diagnostic output directory; no existing artifact is overwritten")
    for path in args.preserve_file:
        if not path.is_file():
            raise ValueError(f"Preserved evidence file is missing: {path}")
    preserved = {str(path.resolve()): digest_file(path) for path in args.preserve_file}
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    before = resource_snapshot(output)
    write_json(output / "resources-before.json", before)
    write_json(output / "original-config.json", config.to_dict())
    write_json(output / "declared-plan.json", declared_plan(config))
    report = {"status": "blocked", "error": "Reset probe construction did not complete"}
    audit = None
    try:
        with RuntimeResetProbe(args.source_root) as probe:
            audit = AuditedProbe(probe, output, started)
            report = run_reset_preflight(config, audit)
            report["runtime_probe_provenance"] = probe.provenance
    except Exception as exc:
        report = {**report, "status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
    after = resource_snapshot(output)
    unchanged = {path: digest_file(path) == digest for path, digest in preserved.items()}
    report.update(kind="tdyssey.C.reset_only_preflight", at_utc=utc_now(),
        command=[sys.executable, *sys.argv], provenance=provenance(args.source_root),
        limits={"reset_attempts": MAX_RESETS, "reset_physics_substeps": MAX_RESETS,
                "wall_seconds": MAX_SECONDS, "storage_bytes": MAX_BYTES,
                "candidate_probes_per_shape": 32},
        formal_C_experiment_initialized=False, formal_C_clock_started=False,
        formal_evaluation=False, policy_loaded=False, policy_samples=0, optimizer_steps=0,
        tensor_runtime_imported="torch" in sys.modules,
        preserved_evidence_sha256=preserved, preserved_evidence_unchanged=unchanged,
        process_resources_before=before, process_resources_after=after,
        process_peak_scope="Reset-only imports/physics/rendering; no model load or PPO",
        elapsed_seconds=time.perf_counter() - started,
        audit_counts=None if audit is None else {"probe_calls": audit.calls,
            "reset_attempts": audit.resets, "reset_physics_substeps": audit.physics, "control_steps": audit.controls,
            "unresolved_probe_claims": list(audit.pending_calls.values())})
    if not all(unchanged.values()) or (audit is not None and (audit.controls or audit.pending_calls)):
        report.update(status="blocked", error="Preserved evidence changed, a control step occurred, or a probe claim is unresolved")
    if report["status"] == "complete":
        write_json(output / "proposed-config.json", report["proposed_config"])
    write_json(output / "resources-after.json", after)
    write_json(output / "report.json", report)
    if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > MAX_BYTES:
        raise RuntimeError("Final reset diagnostic artifacts exceeded the declared byte cap")
    return report


def main():
    args = parser().parse_args()
    try:
        from tdyssey.policy_protocol import CConfig
        config = CConfig.from_json(args.config)
        if args.mode == "plan":
            value = declared_plan(config)
        else:
            if not args.execute or args.source_root is None or args.output_dir is None:
                raise ValueError("Reset run requires explicit --execute, --source-root and new --output-dir")
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            report = run(args, config)
            value = {"status": report["status"], "report": str(args.output_dir.resolve() / "report.json"),
                     "audit_counts": report["audit_counts"], "proposed_config_hash": report.get("proposed_config_hash")}
        print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
        return 0 if value["status"] in ("complete", "proposed_not_executed") else 1
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
