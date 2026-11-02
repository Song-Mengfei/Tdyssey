"""Fake-resource checks plus one harmless native Windows process-tree check.

No test loads a model, decoder, simulator or GPU runtime, and no test queries
nvidia-smi. Native ownership evidence is explicitly separate from injected data.
"""

from copy import deepcopy
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from tdyssey import wm_compat_guard as guard

ROOT = Path(__file__).resolve().parents[1]
RACE_EVIDENCE = ROOT/"artifacts/wm-input-compat-eval-v3-audit-20261025/guard"
EVIDENCE = RACE_EVIDENCE


def resources():
    return {"available_ram_bytes": 8*guard.GIB, "commit_headroom_bytes": 12*guard.GIB,
        "gpu_free_bytes": 6*guard.GIB, "disk_free_bytes": 20*guard.GIB,
        "gpu_age_seconds": 0., "working_set_bytes": 64*guard.MIB, "private_bytes": 128*guard.MIB}


def status(pid=303, nonce="nonce", phase="setup"):
    backend = phase in ("running", "model_loading")
    return {"actual_worker_pid": pid, "nonce": nonce, "at_unix": 100., "phase": phase,
        "model_instances": int(backend), "model_loads": int(backend), "optimizer_updates": 0,
        "cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0,
        **dict.fromkeys(guard.FORBIDDEN_COUNTERS, 0)}


class FakeOwner:
    def __init__(self, state, pid, launcher):
        self.state, self.pid = state, pid
        state["owned"] = True
    def poll_exit(self):
        return self.state.get("exited", False)
    def exit_code(self):
        return self.state.get("exit_code", 0) if self.poll_exit() else None
    def terminate_and_wait(self, timeout):
        self.state.update(exited=True, exit_code=1, terminated=True)
    def close(self):
        self.state["closed"] = True
    def quiesce(self, timeout):
        self.state["job_empty_verified"] = True
        return 0


class FakeLauncher:
    pid = 202
    def __init__(self, state):
        self.state = state
        self.stdout = io.BytesIO()
    def poll(self):
        return 0 if self.state.get("exited") else None
    def wait(self, timeout):
        return self.state.get("exit_code", 0)
    def terminate(self):
        self.state.update(exited=True, exit_code=1)
    kill = terminate


class FakeProvider:
    def __init__(self, state, *, fail_runtime=False, fail_startup=False):
        self.state, self.fail_runtime, self.fail_startup = state, fail_runtime, fail_startup
        self.reads, self.closed = 0, False
    def startup(self, directory):
        value = resources()
        if self.fail_startup:
            value["gpu_free_bytes"] = None
        return value
    def start(self):
        self.state["provider_started"] = True
    def runtime(self, pid, directory):
        self.reads += 1
        value = resources()
        if self.reads >= 2 and self.fail_runtime:
            value["private_bytes"] = None
        elif self.reads >= 2:
            payload = status(pid, self.state["nonce"], "complete")
            payload["at_unix"] = guard.time.time()
            guard.atomic_json(self.state["directory"]/"worker-status.json", payload, storage=self.state["storage"])
            self.state["exited"] = True
        return value
    def close(self):
        self.closed = True
        if self.state.get("storage") is not None:
            self.state["storage"].close()


class GuardChecks(unittest.TestCase):
    def setUp(self):
        RACE_EVIDENCE.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=RACE_EVIDENCE, prefix="guard-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
    def fake_launch(self, state, *, bad_nonce=False):
        def launch(argv, **kwargs):
            path = Path(argv[argv.index("--handshake")+1])
            nonce = argv[argv.index("--nonce")+1]
            state.update(directory=path.parent, nonce=nonce)
            from tdyssey.wm_compat_storage import Coordinator
            state["storage"] = Coordinator(**json.loads(kwargs["env"][guard.ENV_STORAGE]))
            guard._immutable_json(path, {"pid": 303, "parent_pid": 202, "nonce": "wrong" if bad_nonce else nonce}, state["storage"])
            return FakeLauncher(state)
        return launch
    def supervise(self, state, provider=None, **kwargs):
        return guard.supervise([str(self.directory/"unused_fake_target.py")],
            output_dir=self.directory/"output", lock_path=self.directory/"one-model.lock",
            provider=provider or FakeProvider(state), owner_factory=lambda pid, launcher: FakeOwner(state, pid, launcher),
            launcher_factory=self.fake_launch(state), poll_seconds=.001, **kwargs)

    def test_read_only_plan_and_launcher_default_do_not_touch_disk_or_query_resources(self):
        output = self.directory/"not-created"
        with patch.object(guard.policy_memory, "gpu_memory", side_effect=AssertionError("GPU query in plan")), \
                patch.object(guard.policy_memory, "system_memory", side_effect=AssertionError("resource query in plan")):
            declaration = guard.plan(["future.py", "--execute"], output_dir=output)
            self.assertFalse(declaration["execution_authorized_by_plan"])
            spec = importlib.util.spec_from_file_location("compat_guard_cli_test", ROOT/"scripts/run_guarded_wm_compat.py")
            cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
            with patch("builtins.print"):
                self.assertEqual(cli.main(["--output-dir", str(output)]), 0)
        self.assertFalse(output.exists())

    def test_mock_complete_ack_follows_ownership_and_is_distinct_from_native_evidence(self):
        state = {}
        provider = FakeProvider(state)
        report = self.supervise(state, provider)
        self.assertEqual(report["status"], "exited", report)
        self.assertTrue(state["owned"])
        self.assertTrue(state["closed"])
        self.assertTrue(provider.closed)
        self.assertTrue(report["provider_injected"])
        self.assertTrue(report["owner_injected"])
        guardian = guard._read(Path(report["directory"])/"guardian.json")
        self.assertEqual(guardian["status"], "owned_before_ack")
        self.assertEqual(guardian["job_memory_bytes"], 3*guard.GIB)
        self.assertIsNone(guardian["ledger_dir"])
        self.assertEqual(guardian["started_monotonic"], report["started_monotonic"])
        self.assertEqual(guardian["deadline_monotonic"], guardian["started_monotonic"]+guardian["deadline_seconds"])
        self.assertEqual(guard._read(report["stop_record"])["status"], "exited")
        self.assertFalse((self.directory/"one-model.lock").exists())
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        guard.atomic_json(EVIDENCE/"mock-last.json", {"kind": "mock_only", "native_job_ownership_tested": False,
            "real_gpu_queries": 0, "model_operations": 0, "report": report})

    def test_normal_exit_with_ledger_cannot_skip_missing_declaration(self):
        state = {}
        report = self.supervise(state, ledger_dir=self.directory/"ledger")
        self.assertEqual(report["guard_reason"], "storage_failure", report)
        self.assertFalse(report["final_audit"]["passed"])
        self.assertEqual(report["storage_error"]["exception_type"], "FileNotFoundError")
        self.assertEqual(Path(report["storage_error"]["path"]), self.directory/"ledger/declaration.json")
        self.assertTrue(state["job_empty_verified"])

    def test_child_compute_pools_are_two_threads_before_import_without_changing_parent_environment(self):
        expected = dict(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
            NUMEXPR_NUM_THREADS="2", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
        inherited = {key: "11" if key.endswith("NUM_THREADS") else "TRUE" for key in expected}
        state = {}
        launch = self.fake_launch(state)
        def checked_launch(argv, **kwargs):
            self.assertEqual({key: kwargs["env"].get(key) for key in expected}, expected)
            state["thread_environment_verified"] = True
            return launch(argv, **kwargs)
        with patch.dict(os.environ, inherited):
            report = guard.supervise(["unused.py"], output_dir=self.directory/"output",
                lock_path=self.directory/"one-model.lock", provider=FakeProvider(state),
                launcher_factory=checked_launch, owner_factory=lambda pid, launcher: FakeOwner(state, pid, launcher),
                poll_seconds=.001)
            self.assertEqual({key: os.environ[key] for key in inherited}, inherited)
        self.assertEqual(report["status"], "exited")
        self.assertTrue(state["thread_environment_verified"])

    def test_launcher_counts_fixed_registry_and_ledger_as_telemetry(self):
        spec = importlib.util.spec_from_file_location("compat_guard_cli_telemetry_test", ROOT/"scripts/run_guarded_wm_compat.py")
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        ledger = self.directory/"new-ledger"
        with patch.object(cli.Path, "is_file", return_value=True), \
                patch.object(cli, "supervise", return_value={"status": "exited"}) as supervise, \
                patch("builtins.print"):
            self.assertEqual(cli.main(["--execute", "--output-dir", str(self.directory/"output"),
                "--ledger-dir", str(ledger)]), 0)
        self.assertEqual(supervise.call_args.kwargs["telemetry_dirs"],
            (ledger, ROOT/"artifacts/wm-input-compat-eval-v3-registry"))
        self.assertEqual(supervise.call_args.kwargs["ledger_dir"], ledger)
        self.assertEqual(supervise.call_args.kwargs["attempt_registry"], ROOT/"artifacts/wm-input-compat-eval-v3-registry")
        self.assertFalse(ledger.exists())

    def test_production_preflight_rejects_wrong_v3_identity_before_any_claim_or_setup(self):
        from tdyssey.wm_compat_identity import OUTPUT, LEDGER, AUTHORIZATION_ID
        repo = self.directory
        output, ledger = repo/"artifacts"/OUTPUT, repo/"artifacts"/LEDGER
        registry = repo/"artifacts/wm-input-compat-eval-v3-registry"
        cases = (
            (output, ledger, AUTHORIZATION_ID, None),
            (output, ledger, AUTHORIZATION_ID, repo/"artifacts/wm-input-compat-eval-v1-registry"),
            (output, ledger, AUTHORIZATION_ID, repo/"artifacts/wm-input-compat-eval-v2-registry"),
            (output, ledger, AUTHORIZATION_ID, repo/"artifacts/wm-input-compat-alternate-registry"),
            (output, ledger, "wrong-or-old-authorization", registry),
            (output, ledger, "wm-input-compat-eval-v2-20261025-user-authorized", registry),
            (repo/"artifacts/wm-input-compat-eval-v1-20261025", ledger, AUTHORIZATION_ID, registry),
            (output, repo/"artifacts/wm-input-compat-eval-v1-ledger-20261025", AUTHORIZATION_ID, registry),
            (repo/"artifacts/wm-input-compat-eval-v2-20261025", ledger, AUTHORIZATION_ID, registry),
            (output, repo/"artifacts/wm-input-compat-eval-v2-ledger-20261025", AUTHORIZATION_ID, registry),
        )
        for selected_output, selected_ledger, authority, selected_registry in cases:
            with self.subTest(output=selected_output, ledger=selected_ledger, registry=selected_registry, authority=authority):
                command = [str(repo/"scripts/validate_wm_compat.py"), "--execute",
                    "--output-dir", str(selected_output), "--ledger-dir", str(selected_ledger),
                    "--authorization-id", authority, "--reviewed-commit", "a"*40]
                provider = Mock()
                with patch.object(guard, "ROOT", repo), \
                        patch.object(guard.Path, "mkdir", side_effect=AssertionError("setup before production identity preflight")), \
                        patch.object(guard, "_exclusive", side_effect=AssertionError("claim before production identity preflight")):
                    with self.assertRaisesRegex(ValueError, "fixed v3"):
                        guard.supervise(command, output_dir=selected_output, ledger_dir=selected_ledger,
                            attempt_registry=selected_registry, lock_path=repo/"artifacts/wm-input-compat-test-model.lock",
                            provider=provider)
                provider.startup.assert_not_called()
                self.assertFalse((repo/"artifacts").exists())

    def sealed_supervise(self, state, *, provider=None, output="output", ledger="ledger"):
        command = [str(self.directory/"unused_fake_entry.py"), "--execute",
            "--authorization-id", "synthetic-authority", "--reviewed-commit", "a"*40]
        return command, guard.supervise(command, output_dir=self.directory/output,
            ledger_dir=self.directory/ledger, attempt_registry=self.directory/"registry",
            lock_path=self.directory/"one-model.lock", provider=provider or FakeProvider(state),
            launcher_factory=self.fake_launch(state),
            owner_factory=lambda pid, launcher: FakeOwner(state, pid, launcher), poll_seconds=.001)

    def test_startup_or_source_failure_permanently_seals_attempt_before_another_output_is_created(self):
        for source_failure in (False, True):
            with self.subTest(source_failure=source_failure):
                base = self.directory/("source" if source_failure else "startup"); base.mkdir()
                original = self.directory
                self.directory = base
                try:
                    state = {}
                    class SourceFailure(FakeProvider):
                        def runtime(provider, pid, output):
                            value = super().runtime(pid, output)
                            if provider.reads >= 2:
                                payload = status(pid, state["nonce"], "error")
                                payload.update(at_unix=guard.time.time(), error="synthetic source binding mismatch")
                                guard.atomic_json(state["directory"]/"worker-status.json", payload, storage=state["storage"])
                                state["exit_code"] = 1
                            return value
                    provider = SourceFailure(state) if source_failure else FakeProvider(state, fail_startup=True)
                    command, report = self.sealed_supervise(state, provider=provider)
                    expected = "worker_error:synthetic source binding mismatch" if source_failure else "missing_or_invalid_resource:gpu_free_bytes"
                    self.assertEqual(report["guard_reason"], expected)
                    seal_path = base/"registry/launcher-attempt.json"
                    original_seal = seal_path.read_bytes()
                    seal = json.loads(original_seal)
                    self.assertEqual(seal["started_monotonic"], report["started_monotonic"])
                    self.assertEqual(seal["deadline_monotonic"], report["deadline_monotonic"])
                    self.assertEqual(seal["authorization_id"], "synthetic-authority")
                    self.assertEqual(seal["reviewed_commit"], "a"*40)
                    self.assertFalse(seal["resume_allowed"])
                    changed = command.copy(); changed[0] = str(base/"changed_source.py")
                    unused = Mock()
                    with self.assertRaisesRegex(RuntimeError, "permanently sealed"):
                        guard.supervise(changed, output_dir=base/"replacement-output", ledger_dir=base/"replacement-ledger",
                            attempt_registry=base/"registry", lock_path=base/"one-model.lock", provider=unused)
                    unused.startup.assert_not_called()
                    self.assertFalse((base/"replacement-output").exists())
                    self.assertFalse((base/"replacement-ledger").exists())
                    self.assertFalse((base/"one-model.lock").exists())
                    self.assertEqual(seal_path.read_bytes(), original_seal)
                finally:
                    self.directory = original

    def test_launch_seal_hash_identity_command_roots_and_deadline_are_verified(self):
        state = {}
        command, report = self.sealed_supervise(state)
        directory = Path(report["directory"])
        marker = guard._read(directory/"guardian.json")
        accepted = guard.validate_launch_seal(marker, command=command)
        self.assertEqual(accepted["nonce"], marker["nonce"])
        self.assertEqual(accepted["path"], report["launch_seal"]["path"])
        self.assertEqual(accepted["sha256"], report["launch_seal"]["sha256"])
        self.assertEqual(accepted["attempt_id"], "wm-input-compat-eval-v3")
        for name, value in (("nonce", "changed"), ("guard_pid", -1), ("output_dir", str(self.directory/"other")),
                            ("ledger_dir", str(self.directory/"other-ledger")),
                            ("deadline_monotonic", marker["deadline_monotonic"]+1)):
            with self.assertRaisesRegex(RuntimeError, "launch seal"):
                guard.validate_launch_seal(dict(marker, **{name: value}), command=command)
        with self.assertRaisesRegex(RuntimeError, "command"):
            guard.validate_launch_seal(marker, command=[*command, "--changed"])
        with patch.object(guard.Path, "read_bytes", return_value=b"tampered seal bytes"):
            with self.assertRaisesRegex(RuntimeError, "bytes changed"):
                guard.validate_launch_seal(marker, command=command)
        changed = deepcopy(marker)
        seal_path = Path(marker["launch_seal"]["path"])
        tampered = guard._read(seal_path)
        tampered["attempt_id"] = "wm-input-compat-eval-v1"
        payload = guard._json(tampered)
        changed["launch_seal"]["sha256"] = hashlib.sha256(payload).hexdigest()
        read_bytes = Path.read_bytes
        with patch.object(guard.Path, "read_bytes", lambda path: payload if path == seal_path else read_bytes(path)):
            with self.assertRaisesRegex(RuntimeError, "kind/attempt differs"):
                guard.validate_launch_seal(changed, command=command)
        with patch.dict(os.environ, {guard.ENV_DIRECTORY: str(directory), guard.ENV_NONCE: marker["nonce"],
                                    guard.ENV_PARENT: str(marker["guard_pid"])}), \
                patch.object(guard.os, "getpid", return_value=303), patch.object(guard.os, "getppid", return_value=202), \
                patch.object(guard.sys, "argv", command):
            self.assertEqual(guard.verify_guardian(native_check=False)["launch_seal"], accepted)

    def test_launch_identity_preflight_does_not_create_registry_or_output(self):
        with self.assertRaisesRegex(ValueError, "authorization-id"):
            guard.supervise(["unused.py", "--execute"], output_dir=self.directory/"output",
                ledger_dir=self.directory/"ledger", attempt_registry=self.directory/"registry", provider=Mock())
        self.assertFalse((self.directory/"registry").exists())
        self.assertFalse((self.directory/"output").exists())

    def test_handoff_storage_policy_cannot_make_claims_or_guardian_mutable(self):
        state = {}
        report = self.supervise(state)
        marker = guard._read(Path(report["directory"])/"guardian.json")
        self.assertIsNotNone(guard.from_handoff(marker))
        variants = []
        for path in (self.directory/"registry/attempt.json", Path(report["directory"])/"guardian.json"):
            changed = deepcopy(marker)
            changed["storage_policy"]["append_only"].append(str(path))
            variants.append(changed)
        for key, value in (("pending_roots", [str(self.directory)]), ("timeout", 5),
                           ("telemetry_reserve", 0), ("emergency_headroom", 1)):
            changed = deepcopy(marker); changed["storage_policy"][key] = value; variants.append(changed)
        changed = deepcopy(marker); changed["storage_telemetry_roots"].append(str(self.directory/"unapproved")); variants.append(changed)
        for changed in variants:
            with self.assertRaisesRegex(RuntimeError, "canonical policy"):
                guard.from_handoff(changed)

    def test_storage_scan_error_keeps_first_details_in_emergency_stop_record(self):
        state = {}
        class CorruptImmutable(FakeProvider):
            def runtime(provider, pid, output):
                value = super().runtime(pid, output)
                if provider.reads >= 2:
                    with state["storage"].transaction("synthetic.immutable_corruption"):
                        (state["directory"]/"worker.json").write_bytes(b"corrupt")
                return value
        report = self.supervise(state, CorruptImmutable(state))
        self.assertEqual(report["guard_reason"], "storage_failure")
        self.assertTrue(report["storage_error"]["path"].endswith("worker.json"))
        self.assertIn("phase", report["storage_error"])
        saved = guard._read(report["stop_record"])
        self.assertEqual(saved["guard_reason"], "storage_failure")
        self.assertEqual(saved["storage_error"], report["storage_error"])
        self.assertTrue(saved["job_empty_verified_before_close"])

    def test_launcher_disallows_abbreviated_paths_and_execute_equals_override(self):
        spec = importlib.util.spec_from_file_location("compat_guard_cli_flags_test", ROOT/"scripts/run_guarded_wm_compat.py")
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        for arguments in (["--output", str(self.directory/"output")],
                          ["--output-dir", str(self.directory/"output"), "--", "--execute=true"]):
            with patch("sys.stderr"), self.assertRaises(SystemExit) as stopped:
                cli.main(arguments)
            self.assertEqual(stopped.exception.code, 2)

    def test_old_nonempty_overlapping_and_linked_paths_are_rejected_before_any_write(self):
        old = ROOT/"artifacts"/("frozen-wm-guard-no-write-"+self.directory.name)
        occupied = self.directory/"occupied"
        occupied.mkdir(); (occupied/"sentinel.txt").write_bytes(b"preserve")
        cases = ((old, None), (occupied, None), (self.directory/"output", self.directory/"output/ledger"),
            (self.directory/"output", ROOT/"artifacts/frozen-wm-ledger-no-write"),
            (self.directory/"output", ROOT/"artifacts/wm-input-compat-eval-v3-registry"))
        for output, ledger in cases:
            with patch.object(guard, "_Recorder", side_effect=AssertionError("Recorder before path preflight")), \
                    patch.object(guard.Path, "mkdir", side_effect=AssertionError("mkdir before path preflight")):
                with self.assertRaises(ValueError):
                    guard.supervise(["unused.py"], output_dir=output, ledger_dir=ledger,
                        lock_path=self.directory/"lock", provider=Mock())
        linked = self.directory/"linked"
        original = Path.is_symlink
        with patch.object(guard.Path, "is_symlink", lambda path: path == linked or original(path)), \
                patch.object(guard.Path, "mkdir", side_effect=AssertionError("mkdir before link rejection")):
            with self.assertRaisesRegex(ValueError, "links"):
                guard.supervise(["unused.py"], output_dir=linked/"output", provider=Mock())
        self.assertFalse(old.exists())
        self.assertFalse((self.directory/"output").exists())
        self.assertEqual(list(occupied.iterdir()), [occupied/"sentinel.txt"])
        self.assertEqual((occupied/"sentinel.txt").read_bytes(), b"preserve")

    def test_final_event_and_close_io_failures_are_durable_and_preserve_first_reason(self):
        for failure, existing_reason in (("event", False), ("close", False), ("event", True)):
            with self.subTest(failure=failure, existing_reason=existing_reason):
                state = {}
                base = self.directory/(failure+str(existing_reason)); base.mkdir()
                original_event, original_close = guard._Recorder.event, guard._Recorder.close
                def event(recorder, kind, detail, **kwargs):
                    if failure == "event" and kind in ("exited", "stopped"):
                        raise OSError("synthetic final-event IO failure")
                    return original_event(recorder, kind, detail, **kwargs)
                def close(recorder):
                    original_close(recorder)
                    if failure == "close":
                        raise OSError("synthetic close-after-slot-close IO failure")
                with patch.object(guard._Recorder, "event", event), patch.object(guard._Recorder, "close", close):
                    report = guard.supervise(["unused.py"], output_dir=base/"output", lock_path=base/"lock",
                        provider=FakeProvider(state, fail_startup=existing_reason),
                        launcher_factory=self.fake_launch(state),
                        owner_factory=lambda pid, launcher: FakeOwner(state, pid, launcher), poll_seconds=.001)
                expected = "missing_or_invalid_resource:gpu_free_bytes" if existing_reason else "io_failure"
                self.assertEqual(report["guard_reason"], expected)
                saved = guard._read(report["stop_record"])
                self.assertEqual(saved["guard_reason"], expected)
                self.assertEqual(saved["status"], "stopped")
                self.assertTrue(saved["finalization_errors"])
                self.assertFalse((base/"lock").exists())

    def test_missing_startup_or_runtime_resource_reads_fail_closed(self):
        for name in ("available_ram_bytes", "commit_headroom_bytes", "gpu_free_bytes", "disk_free_bytes", "gpu_age_seconds"):
            value = resources(); value[name] = None
            self.assertIsNotNone(guard.resource_reason(value, startup=True))
        state = {}
        report = self.supervise(state, FakeProvider(state, fail_startup=True))
        self.assertEqual(report["guard_reason"], "missing_or_invalid_resource:gpu_free_bytes")
        self.assertNotIn("owned", state)
        # A different output for this unit fixture does not share a real model/account.
        self.directory = self.directory/"second"; self.directory.mkdir()
        state = {}
        report = self.supervise(state, FakeProvider(state, fail_runtime=True))
        self.assertEqual(report["guard_reason"], "missing_or_invalid_resource:private_bytes")
        self.assertTrue(state["terminated"])
        self.assertTrue(state["closed"])

    def test_os_working_set_peak_still_stops_after_current_usage_recovers(self):
        value = resources()
        value["process_peak_working_set_bytes"] = guard.LIMITS["working_set_bytes"]+1
        self.assertLess(value["working_set_bytes"], guard.LIMITS["working_set_bytes"])
        self.assertEqual(guard.resource_reason(value), "process_memory_limit:process_peak_working_set_bytes")
        value["process_peak_working_set_bytes"] = float("nan")
        self.assertEqual(guard.resource_reason(value), "missing_or_invalid_resource:process_peak_working_set_bytes")

    def test_existing_single_model_lock_is_not_removed_or_acknowledged(self):
        lock = self.directory/"one-model.lock"
        lock.write_bytes(b"another-owner")
        state = {}
        report = self.supervise(state)
        self.assertEqual(report["guard_reason"], "single_model_lock_or_evidence_collision")
        self.assertEqual(lock.read_bytes(), b"another-owner")
        self.assertNotIn("owned", state)

    def test_nonce_failure_never_acknowledges_the_target(self):
        state = {}
        report = guard.supervise(["unused.py"], output_dir=self.directory/"output",
            lock_path=self.directory/"one-model.lock", provider=FakeProvider(state),
            launcher_factory=self.fake_launch(state, bad_nonce=True),
            owner_factory=lambda *args: self.fail("Owner invoked despite bad nonce"))
        self.assertEqual(report["guard_reason"], "bootstrap_nonce_mismatch")
        self.assertFalse((Path(report["directory"])/"worker-ack.json").exists())

    def test_worker_counters_loading_timeout_stale_status_cuda_and_oom_reasons(self):
        base = status()
        self.assertIsNone(guard.worker_status_reason(base, pid=303, nonce="nonce", now=100))
        self.assertEqual(guard.worker_status_reason(base, pid=303, nonce="nonce", now=106), "worker_status_stale_or_invalid")
        for key, value in (("model_instances", 2), ("optimizer_updates", 1), ("cuda_allocated_bytes", 1)):
            wrong = dict(base, **{key: value})
            self.assertIsNotNone(guard.worker_status_reason(wrong, pid=303, nonce="nonce", now=100))
        loading = status(phase="model_loading")
        loading.update(cuda_allocated_bytes=None, cuda_reserved_bytes=None)
        self.assertIsNone(guard.worker_status_reason(loading, pid=303, nonce="nonce", now=100, loading_elapsed=29))
        self.assertEqual(guard.worker_status_reason(loading, pid=303, nonce="nonce", now=100, loading_elapsed=31), "model_loading_deadline")
        running = status(phase="running")
        running["cuda_reserved_bytes"] = guard.GIB+1
        self.assertEqual(guard.worker_status_reason(running, pid=303, nonce="nonce", now=100), "cuda_allocator_limit:cuda_reserved_bytes")
        running.update(phase="error", error="CUDA out of memory")
        self.assertEqual(guard.worker_status_reason(running, pid=303, nonce="nonce", now=100), "worker_error:CUDA out of memory")

    def test_journal_io_failure_uses_preallocated_durable_stop_slot_and_cleans_owner(self):
        original = guard._Recorder.event
        def fail_on_sample(recorder, kind, detail, **kwargs):
            if kind == "sample":
                raise OSError("synthetic journal IO failure")
            return original(recorder, kind, detail, **kwargs)
        state = {}
        with patch.object(guard._Recorder, "event", fail_on_sample):
            report = self.supervise(state)
        self.assertEqual(report["guard_reason"], "io_failure")
        saved = guard._read(report["stop_record"])
        self.assertEqual(saved["guard_reason"], "io_failure")
        self.assertIn("synthetic journal IO", saved["error"])
        self.assertTrue(state["closed"])

    def test_failed_tree_cleanup_keeps_single_model_lock_fail_closed(self):
        state = {}
        class BrokenClose(FakeOwner):
            def close(self):
                raise OSError("synthetic retained-job cleanup failure")
        report = guard.supervise(["unused.py"], output_dir=self.directory/"output",
            lock_path=self.directory/"one-model.lock", provider=FakeProvider(state),
            launcher_factory=self.fake_launch(state), owner_factory=lambda pid, launcher: BrokenClose(state, pid, launcher))
        self.assertEqual(report["guard_reason"], "process_tree_cleanup_failed")
        self.assertTrue(report["single_model_lock_retained_after_cleanup_failure"])
        self.assertTrue((self.directory/"one-model.lock").exists())
        self.assertEqual(guard._read(report["stop_record"])["guard_reason"], "process_tree_cleanup_failed")

    def test_nonempty_job_keeps_lock_even_after_worker_exit_and_successful_handle_close(self):
        state = {}
        class NonemptyJob(FakeOwner):
            def quiesce(self, timeout):
                return 1
        report = guard.supervise(["unused.py"], output_dir=self.directory/"output",
            lock_path=self.directory/"one-model.lock", provider=FakeProvider(state),
            launcher_factory=self.fake_launch(state), owner_factory=lambda pid, launcher: NonemptyJob(state, pid, launcher))
        self.assertEqual(report["guard_reason"], "process_tree_cleanup_failed")
        self.assertFalse(report["job_empty_verified_before_close"])
        self.assertTrue(state["closed"])
        self.assertTrue((self.directory/"one-model.lock").exists())
        self.assertTrue(guard._read(report["stop_record"])["single_model_lock_retained_after_cleanup_failure"])

    def test_job_cleanup_terminates_and_queries_until_whole_job_is_empty(self):
        owner = guard.WindowsWorker.__new__(guard.WindowsWorker)
        owner._job, owner._kernel = 17, Mock()
        active = iter((3, 2, 0))
        def query(job, information, pointer, size, returned):
            self.assertEqual((job, information), (17, 1))
            pointer._obj.ActiveProcesses = next(active)
            pointer._obj.TotalProcesses = 3
            return True
        owner._kernel.QueryInformationJobObject.side_effect = query
        owner._kernel.TerminateJobObject.return_value = True
        with patch.object(guard.time, "sleep"):
            self.assertEqual(owner.quiesce(timeout=1), 0)
        owner._kernel.TerminateJobObject.assert_called_once_with(17, 1)
        self.assertEqual(owner._kernel.QueryInformationJobObject.call_count, 3)

    def test_forbidden_operation_fields_are_required_integer_zero_including_real_renders(self):
        for key in guard.FORBIDDEN_COUNTERS:
            for value in (None, False, 1):
                wrong = status()
                if value is None:
                    del wrong[key]
                else:
                    wrong[key] = value
                self.assertIsNotNone(guard.worker_status_reason(wrong, pid=303, nonce="nonce", now=100), (key, value))

    def test_nonzero_worker_preserves_published_oom_reason_after_fast_exit(self):
        state = {}
        class OOMProvider(FakeProvider):
            def runtime(self, pid, directory):
                value = super().runtime(pid, directory)
                if self.reads >= 2:
                    payload = status(pid, state["nonce"], "error")
                    payload.update(at_unix=guard.time.time(), error="CUDA out of memory")
                    guard.atomic_json(state["directory"]/"worker-status.json", payload, storage=state["storage"])
                    state["exit_code"] = 1
                return value
        report = self.supervise(state, OOMProvider(state))
        self.assertEqual(report["guard_reason"], "worker_error:CUDA out of memory")
        self.assertEqual(guard._read(report["stop_record"])["guard_reason"], "worker_error:CUDA out of memory")

    def _status_continuity_scenario(self, mode, *, winerror=None):
        state, wall = {}, [100.]
        class SequenceProvider(FakeProvider):
            def runtime(provider, pid, directory):
                provider.reads += 1
                if provider.reads == 2 or (provider.reads == 4 and mode == "transient"):
                    complete = mode == "missing_after_complete" or provider.reads == 4
                    wall[0] = 105.001 if provider.reads == 4 else 100.
                    payload = status(pid, state["nonce"], "complete" if complete else "running")
                    payload["at_unix"] = wall[0]
                    if complete:
                        payload.update(model_instances=1, model_loads=1)
                    guard.atomic_json(state["directory"]/"worker-status.json", payload, storage=state["storage"])
                    state["gap"] = False
                    if provider.reads == 4:
                        state["exited"] = True
                elif provider.reads >= 3:
                    state["gap"] = True
                    wall[0] = (105. if provider.reads == 3 else 105.001) if mode == "permanent" else 104.999
                    if mode == "missing_after_complete":
                        state["exited"] = True
                return resources()
        original = guard._locked_read
        def read(path, storage):
            if Path(path).name == "worker-status.json" and state.get("gap"):
                state["gap_reads"] = state.get("gap_reads", 0)+1
                if mode == "permission":
                    raise PermissionError(13, "synthetic status read denied", str(path))
                if winerror is not None:
                    cause = ctypes.WinError(winerror) if sys.platform == "win32" else OSError("synthetic Windows sharing busy")
                    cause.winerror = winerror
                else:
                    cause = FileNotFoundError(2, "synthetic publication gap", str(path))
                raise guard._io_error(path, "guardian.json_read.open", cause) from cause
            return original(path, storage)
        provider = SequenceProvider(state)
        with patch.object(guard.time, "time", side_effect=lambda: wall[0]), patch.object(guard, "_locked_read", read):
            report = self.supervise(state, provider)
        return state, provider, report

    def test_mutable_status_gap_reuses_fresh_snapshot_then_accepts_complete(self):
        state, provider, report = self._status_continuity_scenario("transient")
        self.assertEqual(report["status"], "exited", report)
        self.assertEqual(state["gap_reads"], 1)
        self.assertEqual(provider.reads, 4)
        self.assertEqual(report["last_worker_status"]["phase"], "complete")
        self.assertEqual(report["worker_status_unavailable_samples"], 1)

    def test_permanent_mutable_status_gap_stops_at_original_snapshot_age(self):
        state, provider, report = self._status_continuity_scenario("permanent")
        self.assertEqual(report["guard_reason"], "worker_status_stale_or_invalid", report)
        self.assertEqual(provider.reads, 4)  # Five seconds is valid; the next observation is stale.
        self.assertEqual(report["last_worker_status"]["at_unix"], 100.)
        self.assertTrue(state["terminated"])
        self.assertEqual(report["worker_status_unavailable_samples"], 2)

    def test_explicit_win32_sharing_busy_reuses_fresh_status_then_accepts_complete(self):
        state, provider, report = self._status_continuity_scenario("transient", winerror=32)
        self.assertEqual(report["status"], "exited", report)
        self.assertEqual(provider.reads, 4)
        self.assertEqual(report["worker_status_unavailable_samples"], 1)

    def test_persistent_win32_sharing_busy_stops_without_refreshing_snapshot_age(self):
        state, provider, report = self._status_continuity_scenario("permanent", winerror=32)
        self.assertEqual(report["guard_reason"], "worker_status_stale_or_invalid", report)
        self.assertEqual(provider.reads, 4)
        self.assertEqual(report["last_worker_status"]["at_unix"], 100.)
        self.assertEqual(report["worker_status_unavailable_samples"], 2)

    def test_permission_after_live_status_fails_immediately_without_cache_fallback(self):
        state, provider, report = self._status_continuity_scenario("permission")
        self.assertEqual(report["guard_reason"], "io_failure", report)
        self.assertEqual(provider.reads, 3)
        self.assertIn("PermissionError", report["error"])

    def test_missing_final_status_cannot_pass_using_cached_complete(self):
        state, provider, report = self._status_continuity_scenario("missing_after_complete")
        self.assertEqual(report["guard_reason"], "final_worker_status_missing", report)
        self.assertEqual(report["last_worker_status"]["phase"], "complete")
        self.assertEqual(state["gap_reads"], 2)  # Live read may cache; quiescent success requires a fresh file.

    def test_status_continuity_is_exact_path_and_does_not_hide_corruption_or_other_missing_files(self):
        path = self.directory/"worker-status.json"
        storage = Mock(atomic_targets={path: guard.STATUS_BYTES})
        previous = status()
        wrong_path = self.directory/"storage.lock"
        failures = (json.JSONDecodeError("bad status", "?", 0),
            FileNotFoundError(2, "different file missing", str(wrong_path)),
            guard._io_error(path, "guardian.json_read.read", FileNotFoundError(2, "read failed", str(path))))
        for failure in failures:
            with self.subTest(error=type(failure).__name__), patch.object(guard, "_locked_read", side_effect=failure):
                with self.assertRaises(type(failure)):
                    guard._latest_worker_status(path, storage, previous)
        with patch.object(guard, "_locked_read", side_effect=FileNotFoundError(2, "gap", str(path))):
            self.assertIs(guard._latest_worker_status(path, storage, previous), previous)
        for code in (5, 32, 33):
            cause = PermissionError(13, "synthetic Windows error", str(path)); cause.winerror = code
            failure = guard._io_error(path, "guardian.json_read.open", cause); failure.__cause__ = cause
            with self.subTest(winerror=code), patch.object(guard, "_locked_read", side_effect=failure):
                if code in (32, 33):
                    self.assertIs(guard._latest_worker_status(path, storage, previous), previous)
                else:
                    with self.assertRaises(type(failure)):
                        guard._latest_worker_status(path, storage, previous)
        with patch.object(guard, "_locked_read", return_value=None), self.assertRaises(ValueError):
            guard._latest_worker_status(path, storage, previous)
        for invalid in (wrong_path, self.directory/"other"/"worker-status.json"):
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                guard._latest_worker_status(invalid, storage, previous)

    def test_entry_requires_real_handoff_and_interval_helpers_load_no_runtime_package(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "No verified"):
                guard.verify_guardian()
        self.assertNotIn("torch", {name.split(".")[0] for name in guard.__dict__ if name == "torch"})

    def test_entry_deadline_is_original_parent_deadline_and_expiry_or_extension_is_rejected(self):
        directory = self.directory/"handoff"; directory.mkdir()
        now = guard.time.monotonic()
        marker = dict(status="owned_before_ack", guard_pid=42, actual_worker_pid=os.getpid(), nonce="unit",
            kill_on_job_close=True, job_memory_bytes=3*guard.GIB,
            output_dir=str(self.directory/"output"), ledger_dir=str(self.directory/"ledger"),
            started_monotonic=now-10, deadline_monotonic=now+1790, deadline_seconds=1800)
        guard._exclusive(directory/"worker.json", {"pid": os.getpid(), "parent_pid": os.getppid(), "nonce": "unit"})
        guard._exclusive(directory/"worker-ack.json", {"pid": os.getpid(), "nonce": "unit"})
        guard.atomic_json(directory/"guardian.json", marker)
        with patch.dict(os.environ, {guard.ENV_DIRECTORY: str(directory), guard.ENV_NONCE: "unit", guard.ENV_PARENT: "42"}):
            accepted = guard.verify_guardian(native_check=False)
            self.assertEqual(accepted["deadline_monotonic"], marker["deadline_monotonic"])
            self.assertEqual(accepted["ledger_dir"], marker["ledger_dir"])
            for changed in (dict(marker, deadline_monotonic=now-1), dict(marker, deadline_seconds=1801),
                            dict(marker, started_monotonic=float("inf")), dict(marker, deadline_monotonic=now+1800)):
                # JSON cannot represent infinity; exercise that case through the
                # read adapter without writing nonfinite source evidence.
                if changed["started_monotonic"] == float("inf"):
                    original = guard._read
                    with patch.object(guard, "_read", side_effect=lambda path: changed if Path(path).name=="guardian.json" else original(path)):
                        with self.assertRaisesRegex(RuntimeError, "deadline"):
                            guard.verify_guardian(native_check=False)
                else:
                    guard.atomic_json(directory/"guardian.json", changed)
                    with self.assertRaisesRegex(RuntimeError, "deadline"):
                        guard.verify_guardian(native_check=False)

    def test_entry_handoff_rejects_old_or_overlapping_paths_without_empty_check(self):
        directory = self.directory/"handoff"; directory.mkdir()
        output, ledger = self.directory/"output", self.directory/"ledger"
        output.mkdir(); ledger.mkdir()
        (output/"already-written.txt").write_bytes(b"guardian evidence")
        (ledger/"already-written.txt").write_bytes(b"worker may already have claimed ledger")
        now = guard.time.monotonic()
        marker = dict(status="owned_before_ack", guard_pid=42, actual_worker_pid=os.getpid(), nonce="unit",
            kill_on_job_close=True, job_memory_bytes=3*guard.GIB, output_dir=str(output), ledger_dir=str(ledger),
            started_monotonic=now, deadline_monotonic=now+1800, deadline_seconds=1800)
        guard._exclusive(directory/"worker.json", {"pid": os.getpid(), "parent_pid": os.getppid(), "nonce": "unit"})
        guard._exclusive(directory/"worker-ack.json", {"pid": os.getpid(), "nonce": "unit"})
        with patch.dict(os.environ, {guard.ENV_DIRECTORY: str(directory), guard.ENV_NONCE: "unit", guard.ENV_PARENT: "42"}):
            guard.atomic_json(directory/"guardian.json", marker)
            self.assertEqual(guard.verify_guardian(native_check=False)["ledger_dir"], str(ledger))
            for path in (output/"nested-ledger", ROOT/"artifacts/frozen-wm-ledger-v1"):
                guard.atomic_json(directory/"guardian.json", dict(marker, ledger_dir=str(path)))
                with self.assertRaisesRegex(RuntimeError, "path binding"):
                    guard.verify_guardian(native_check=False)


class StorageRaceEngineeringChecks(unittest.TestCase):
    """Deterministic file-only regression; legacy baseline JSON stays intact."""
    def setUp(self):
        RACE_EVIDENCE.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=RACE_EVIDENCE, prefix="guard-scanner-fixture-")
        self.directory = Path(temporary.name).resolve()
        self.assertTrue(self.directory.is_relative_to(RACE_EVIDENCE.resolve()))
        self.addCleanup(temporary.cleanup)

    def evidence(self, name, detail):
        record = dict(kind="deterministic_synthetic_file_race_fix", source_commit="engineering_working_tree",
            model_operations=0, decoder_operations=0, physics_operations=0, native_process_operations=0,
            production_paths_written=False, **detail)
        (RACE_EVIDENCE/name).write_text(json.dumps(record, indent=2, sort_keys=True)+"\n", encoding="utf-8")

    def partial_append(self, raw, message):
        self.assertIsInstance(raw, io.FileIO, "Production append handles must have no Python write buffer")
        class Partial:
            name = raw.name
            calls = 0
            admitted = None
            def write(stream, payload):
                stream.calls += 1
                if stream.calls == 1:
                    stream.admitted = len(payload)
                    return raw.write(payload[:3])
                raise OSError(message)
            def fileno(stream): return raw.fileno()
            def close(stream): raw.close()
            def __getattr__(stream, name): return getattr(raw, name)
        return Partial()

    def consume_normal_telemetry_space(self, storage, path):
        with storage.local_transaction("synthetic.consume_normal_space"):
            current = storage.usage()["telemetry_bytes"]
            amount = storage.limits["telemetry_bytes"]-current
            self.assertGreater(amount, 0)
            ticket = storage.reserve_bytes(path, amount, phase="synthetic.consume_normal_space")
            with path.open("xb", buffering=0) as stream:
                storage.validate_append(path, stream)
                guard._write_all(stream, b"f"*amount)
                os.fsync(stream.fileno())
                storage.record_append(path, b"f"*amount, stream)
            storage.commit(ticket, actual_bytes=amount)
        return amount

    def test_write_all_completes_positive_short_writes_and_rejects_no_progress(self):
        class Short(io.BytesIO):
            def write(stream, payload):
                return super().write(payload[:2])
        stream = Short()
        guard._write_all(stream, b"bounded payload")
        self.assertEqual(stream.getvalue(), b"bounded payload")
        for result in (None, 0, -1, True, 100):
            with self.assertRaisesRegex(OSError, "no progress|invalid byte count"):
                guard._write_all(Mock(write=Mock(return_value=result)), b"tiny")

    def test_immutable_publication_io_is_outside_quota_lock_and_failure_stays_charged(self):
        from tdyssey.wm_compat_storage import Coordinator, StorageError
        directory = self.directory/"telemetry"; directory.mkdir()
        storage = Coordinator(telemetry_roots=(directory,))
        payload = {"synthetic":"immutable"}; size = len(guard._json(payload))
        exclusive, rename, fsync = guard._exclusive_bytes, os.rename, os.fsync
        observed = []
        def check_then(function, label):
            def call(*args, **kwargs):
                self.assertFalse(storage.accounting_lock_held, label)
                observed.append(label)
                return function(*args, **kwargs)
            return call
        with patch.object(guard, "_exclusive_bytes", check_then(exclusive,"payload.write")), \
                patch.object(guard.os, "rename", check_then(rename,"payload.publish")), \
                patch.object(guard.os, "fsync", check_then(fsync,"payload.fsync")):
            guard._immutable_json(directory/"complete.json",payload,storage)
        self.assertEqual(storage.usage()["telemetry_bytes"], size)
        self.assertEqual(storage.usage()["telemetry_observed_bytes"], size)
        with patch.object(guard.os, "rename", side_effect=OSError("synthetic publish failure")):
            with self.assertRaisesRegex(StorageError,"publish failure"):
                guard._immutable_json(directory/"failed.json",payload,storage)
        self.assertEqual(storage.usage()["telemetry_bytes"], 3*size)
        self.assertFalse((directory/"failed.json").exists())
        self.assertEqual(len(list(directory.glob("failed.json.pending-*"))),1)
        self.evidence("guard-incremental-immutable-publication.json",dict(io_events_outside_quota_lock=observed,
            successful_bytes=size,failed_final_and_temp_charge=2*size,usage=storage.usage()))

    def test_quiescent_journal_manifest_checks_chain_and_rejects_guardian_override(self):
        from tdyssey.wm_compat_storage import Coordinator, StorageError
        output, ledger = self.directory/"output", self.directory/"ledger"
        (output/"results").mkdir(parents=True); ledger.mkdir()
        storage=Coordinator(result_roots=(output,),telemetry_roots=(ledger,))
        path=output/"results/complete.bin"; data=b"synthetic"
        ticket=storage.reserve_bytes(path,len(data));path.write_bytes(data)
        record=storage.remember_file(path);storage.commit(ticket,len(data))
        guard._immutable_json(ledger/"declaration.json",{"synthetic_only":True},storage)
        journal=ledger/"journal.jsonl"
        complete=dict(path=str(path),bytes=record.size,mtime_ns=record.mtime_ns,
            sha256=record.sha256,identity=list(record.identity))
        def journal_bytes(completion, bad_chain=False):
            previous="0"*64;result=[]
            for sequence,(kind,value) in enumerate((("created",{}),("output_reserved",{"path":str(path)}),
                    ("output_complete",completion)),1):
                row=dict(sequence=sequence,kind=kind,data=value,previous_sha256=previous,elapsed_seconds=sequence/10)
                if bad_chain and sequence==3: row["previous_sha256"]="f"*64
                line=(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode()
                result.append(line);previous=hashlib.sha256(line).hexdigest()
            return b"".join(result)
        raw=journal_bytes(complete);ticket=storage.reserve_bytes(journal,len(raw))
        journal.write_bytes(raw);storage.commit(ticket,len(raw))
        expected,summary=guard._completed_output_manifest(ledger,output)
        self.assertEqual(summary["records"],3)
        self.assertEqual(summary["completed_outputs"],1)
        self.assertEqual(storage.audit(expected=expected)["output_observed_bytes"],len(data))
        for altered in (journal_bytes(complete,True),journal_bytes(dict(complete,path=str(output/"guardian.json")))):
            journal.write_bytes(altered)
            with self.assertRaises(StorageError): guard._completed_output_manifest(ledger,output)

    def test_compact_completion_paths_reconstruct_bound_roots_and_require_legacy_agreement(self):
        from tdyssey.wm_compat_storage import StorageError
        output, ledger = self.directory/"output", self.directory/"ledger"
        result, exported = output/"results/nested/sample.bin", ledger/"exports/summary.json"
        for path in (result, exported):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic completed output")
        (ledger/"declaration.json").write_text("{}\n", encoding="utf-8")
        def completion(path, name, category):
            value = path.stat()
            return dict(name=name, category=category, bytes=value.st_size, max_bytes=value.st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), mtime_ns=value.st_mtime_ns,
                identity=[value.st_dev, value.st_ino])
        compact = completion(result, "nested/sample.bin", "results")
        telemetry = completion(exported, "summary.json", "telemetry")
        journal = ledger/"journal.jsonl"
        def publish(records):
            previous, lines = "0"*64, []
            for sequence, record in enumerate(records, 1):
                row = dict(sequence=sequence, kind="output_complete", data=record,
                    previous_sha256=previous, elapsed_seconds=sequence/10)
                line = (json.dumps(row, sort_keys=True, separators=(",",":"))+"\n").encode()
                lines.append(line); previous = hashlib.sha256(line).hexdigest()
            journal.write_bytes(b"".join(lines))
        publish((compact, telemetry))
        expected, summary = guard._completed_output_manifest(ledger, output)
        self.assertEqual(summary["completed_outputs"], 2)
        self.assertEqual(expected[str(result)], dict(compact, path=str(result)))
        self.assertEqual(expected[str(exported)], dict(telemetry, path=str(exported)))
        for legacy in (dict(compact, path=str(result)),
                {key:value for key,value in dict(compact,path=str(result)).items() if key not in ("name","category")}):
            publish((legacy,))
            self.assertEqual(guard._completed_output_manifest(ledger,output)[0][str(result)]["sha256"], compact["sha256"])
        invalid = (dict(compact, name="../sample.bin"), dict(compact, name=str(result)),
            dict(compact, name=""), dict(compact, name="C:sample.bin"),
            dict(compact, category="guardian"), dict(compact, category=None),
            dict(compact, path=str(output/"results/different.bin")), dict(compact, path=None),
            dict(compact, path=str(exported)), dict(compact, name=None, path=str(result)),
            dict(compact, path=str(output/"guardian/worker-status.json")),
            {key:value for key,value in compact.items() if key!="category"},
            dict(path=str(output/"guardian/worker-status.json")),
            dict(path=str(exported),category="results"))
        for record in invalid:
            with self.subTest(record=record):
                publish((record,))
                with self.assertRaises(StorageError):
                    guard._completed_output_manifest(ledger,output)

    def test_terminal_completion_rejects_every_valid_prefix_and_allows_controlled_stop(self):
        from tdyssey.wm_compat_storage import StorageError
        output, ledger = self.directory/"output", self.directory/"ledger"
        terminal = output/"results/completion.json"
        terminal.parent.mkdir(parents=True); ledger.mkdir()
        declaration, journal = ledger/"declaration.json", ledger/"journal.jsonl"
        declaration.write_bytes(b'{"synthetic_only":true}\n')
        outcomes = []
        def terminal_lines(terminal_status):
            payload = guard._json(dict(status=terminal_status,
                stop_reason="T_gate_failed" if terminal_status=="stopped" else None))
            terminal.write_bytes(payload)
            value = terminal.stat()
            complete = dict(name="completion.json", category="results", bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(), mtime_ns=value.st_mtime_ns,
                identity=[value.st_dev,value.st_ino])
            previous, lines = "0"*64, []
            for sequence,(kind,data) in enumerate((("created",{}),
                    ("output_reserved",dict(name="completion.json",category="results")),
                    ("output_complete",complete)),1):
                row = dict(sequence=sequence,kind=kind,data=data,previous_sha256=previous,
                           elapsed_seconds=sequence/10)
                line=guard._json(row);lines.append(line);previous=hashlib.sha256(line).hexdigest()
            return lines
        for terminal_status in ("complete","stopped"):
            lines = terminal_lines(terminal_status); full = b"".join(lines)
            journal.write_bytes(full)
            expected, summary = guard._completed_output_manifest(ledger,output,require_terminal=True)
            self.assertEqual(summary["records"],3)
            self.assertIn(str(terminal),expected)
            for kept in range(len(lines)):
                journal.write_bytes(b"".join(lines[:kept]))
                with self.subTest(status=terminal_status,kept=kept), self.assertRaises(StorageError):
                    guard._completed_output_manifest(ledger,output,require_terminal=True)
                # Failure diagnostics can still inspect a partial valid chain.
                guard._completed_output_manifest(ledger,output)
            journal.write_bytes(full)
            declaration.unlink()
            with self.assertRaises(StorageError):
                guard._completed_output_manifest(ledger,output,require_terminal=True)
            self.assertEqual(guard._completed_output_manifest(ledger,output),({},None))
            declaration.write_bytes(b'{"synthetic_only":true}\n')
            journal.unlink()
            with self.assertRaises(StorageError):
                guard._completed_output_manifest(ledger,output,require_terminal=True)
            journal.write_bytes(full)
            terminal.write_bytes(b'{"status":"tampered"}\n')
            with self.assertRaises(StorageError):
                guard._completed_output_manifest(ledger,output,require_terminal=True)
            outcomes.append(dict(status=terminal_status,valid_terminal_accepted=True,
                proper_prefixes_rejected=len(lines),missing_declaration_rejected=True,
                missing_journal_rejected=True,completion_hash_mismatch_rejected=True))
        journal.write_bytes(b"".join(terminal_lines("active")))
        with self.assertRaises(StorageError):
            guard._completed_output_manifest(ledger,output,require_terminal=True)
        self.evidence("guard-terminal-boundary-"+guard.uuid.uuid4().hex+".json",
            dict(scope="terminal byte fixtures only; complete fake flow was not rerun after this patch",
                 outcomes=outcomes,nonterminal_status_rejected=True))

    def test_monitor_partial_error_then_close_cannot_consume_emergency_headroom(self):
        from tdyssey.wm_compat_storage import Coordinator, StorageError
        directory = self.directory/"telemetry"; directory.mkdir()
        monitor, filler = directory/"monitor.jsonl", directory/"other.log"
        storage = Coordinator(telemetry_roots=(directory,), append_only=(monitor, filler),
            fixed_slots={directory/"stop-record.json":65536}, emergency_headroom=65536,
            limits={"telemetry_bytes":2*65536+512})
        recorder = guard._Recorder(directory, storage)
        self.addCleanup(recorder.close)
        partial = self.partial_append(recorder.stream, "synthetic partial monitor failure")
        recorder.stream = partial
        with self.assertRaisesRegex(StorageError, "partial monitor failure") as primary:
            recorder.event("synthetic", {"tiny":True}, durable=True)
        self.assertEqual(monitor.stat().st_size, 3)
        self.assertEqual(partial.calls, 2)
        filled = self.consume_normal_telemetry_space(storage, filler)
        before = storage.usage()
        recorder.finish({"guard_reason":"synthetic partial monitor failure", "stop_record":str(recorder.path)})
        recorder.close()
        after = storage.usage()
        self.assertEqual(monitor.stat().st_size, 3)
        self.assertEqual(before["telemetry_bytes"], after["telemetry_bytes"])
        self.assertEqual(after["telemetry_bytes"], storage.limits["telemetry_bytes"])
        physical = storage.audit()["telemetry_observed_bytes"]
        self.assertGreaterEqual(after["telemetry_bytes"]-physical, 65536)
        self.evidence("guard-monitor-partial-close.json", dict(admitted_payload_bytes=partial.admitted,
            actual_append_bytes=3, later_admitted_filler_bytes=filled, close_added_bytes=0,
            protected_emergency_bytes=65536, error=primary.exception.details, before=before, after=after))

    def test_stdout_partial_error_then_close_cannot_consume_emergency_headroom(self):
        from tdyssey.wm_compat_storage import Coordinator
        directory = self.directory/"telemetry"; directory.mkdir()
        path, filler = directory/"worker-output.log", directory/"other.log"
        storage = Coordinator(telemetry_roots=(directory,), append_only=(path, filler),
            fixed_slots={directory/"stop-record.json":65536}, emergency_headroom=65536,
            limits={"telemetry_bytes":2*65536+512})
        ready, release = threading.Event(), threading.Event()
        class Source(io.BytesIO):
            def read(source, count):
                ready.set()
                if not release.wait(2): raise TimeoutError("test barrier")
                return super().read(count)
            def close(source):
                super().close()
                raise OSError("synthetic secondary source close")
        pump = guard._OutputPump(Source(b"p"*100), path, storage)
        self.addCleanup(lambda: pump.stream.close())
        self.assertTrue(ready.wait(1))
        partial = self.partial_append(pump.stream, "synthetic partial stdout failure")
        pump.stream = partial
        release.set(); pump.thread.join(timeout=2)
        self.assertFalse(pump.thread.is_alive())
        self.assertEqual(path.stat().st_size, 3)
        self.assertEqual(partial.calls, 2)
        filled = self.consume_normal_telemetry_space(storage, filler)
        before = storage.usage()
        with self.assertRaisesRegex(RuntimeError, "partial stdout failure"):
            pump.close()
        after = storage.usage()
        self.assertEqual(path.stat().st_size, 3)
        self.assertEqual(before["telemetry_bytes"], after["telemetry_bytes"])
        self.assertEqual(after["telemetry_bytes"], storage.limits["telemetry_bytes"])
        physical = storage.audit()["telemetry_observed_bytes"]
        self.assertGreaterEqual(after["telemetry_bytes"]-physical, 65536)
        self.assertTrue(any("secondary source close" in note for note in pump.error.__notes__))
        self.evidence("guard-stdout-partial-close.json", dict(admitted_payload_bytes=partial.admitted,
            actual_append_bytes=3, later_admitted_filler_bytes=filled, close_added_bytes=0,
            protected_emergency_bytes=65536, error=guard._storage_error_details(pump.error),
            secondary_notes=pump.error.__notes__, before=before, after=after))

    def test_public_usage_rejects_linked_roots_before_any_resolve(self):
        from tdyssey import wm_compat_storage as shared
        output, linked = self.directory/"output", self.directory/"linked-root"
        output.mkdir()
        lstat = shared._lstat
        def reject_link(path, phase):
            if Path(path) == linked:
                raise shared.StorageError(path, phase, message="Synthetic symlink/reparse root")
            return lstat(path, phase)
        with patch.object(shared, "_lstat", reject_link), \
                patch.object(guard.Path, "resolve", side_effect=AssertionError("resolve must not precede link validation")):
            for root, extras in ((linked, ()), (output, (linked,))):
                with self.assertRaisesRegex(shared.StorageError, "symlink/reparse root"):
                    guard.storage_usage(root, extras)

    def test_prepaid_status_publication_is_independent_of_global_quota_lock(self):
        from tdyssey.wm_compat_storage import Coordinator, create_lock_file
        output = self.directory/"output"
        directory = output/"guardian"/"synthetic-nonce"; directory.mkdir(parents=True)
        target = directory/"worker-status.json"
        create_lock_file(directory/"storage.lock")
        storage = Coordinator(lock_path=directory/"storage.lock", result_roots=(output,),
            telemetry_roots=(output/"guardian",), atomic_targets=(target,))
        self.addCleanup(storage.close)
        guard.atomic_json(target, {"old":True}, storage=storage)
        attempted, published, errors = threading.Event(), threading.Event(), []
        def publish():
            try:
                attempted.set()
                guard.atomic_json(target, {"new":"n"*101}, storage=storage)
                published.set()
            except BaseException as exc:
                errors.append(exc)
        before = guard.storage_usage(output, storage=storage)
        with storage.transaction("synthetic.held_quota_lock"):
            writer = threading.Thread(target=publish, name="synthetic-publisher")
            writer.start()
            started, completed_while_quota_held = attempted.wait(1), published.wait(1)
        writer.join(timeout=2)
        self.assertFalse(writer.is_alive())
        self.assertFalse(errors)
        self.assertTrue(started)
        self.assertTrue(completed_while_quota_held)
        self.assertTrue(published.is_set())
        after = guard.storage_usage(output, storage=storage)
        self.assertGreaterEqual(before["telemetry_bytes"], 2*guard.STATUS_BYTES)
        self.assertGreaterEqual(after["telemetry_bytes"], 2*guard.STATUS_BYTES)
        self.assertFalse(list(directory.glob("worker-status.json.*.tmp")))
        self.evidence("guard-shared-replacement-window.json", dict(before=before, after=after,
            publisher_finished_while_quota_lock_held=True, payload_bytes_after=target.stat().st_size,
            conclusion="Prepaid status publication performs no quota transaction or recursive snapshot"))

    def test_final_audit_rejects_immutable_disappearance_without_refunding_quota(self):
        from tdyssey.wm_compat_storage import Coordinator, StorageError
        output = self.directory/"output"; output.mkdir()
        immutable = output/"completed-result.bin"
        storage = Coordinator(result_roots=(output,))
        ticket = storage.reserve_bytes(immutable, len(b"immutable"))
        immutable.write_bytes(b"immutable")
        storage.remember_file(immutable); storage.commit(ticket, len(b"immutable"))
        measured = guard.storage_usage(output, storage=storage)
        self.assertEqual(measured["output_bytes"], len(b"immutable"))
        with storage.local_transaction("synthetic.invalid_delete"):
            immutable.unlink()
        self.assertEqual(guard.storage_usage(output, storage=storage)["output_bytes"], len(b"immutable"))
        with self.assertRaises(StorageError) as caught:
            storage.audit()
        self.evidence("guard-shared-immutable-disappearance.json", dict(before=measured,
            vanished_file="completed-result.bin", error=str(caught.exception),
            legacy_evidence="guard-legacy-immutable-disappearance.json",
            conclusion="An observed immutable result disappearing fails closed"))

    def test_atomic_small_slot_and_fixed_slot_reject_before_any_write(self):
        from tdyssey.wm_compat_storage import Coordinator
        directory = self.directory/"telemetry"; directory.mkdir()
        target = directory/"worker-status.json"
        storage = Coordinator(telemetry_roots=(directory,), atomic_targets={target:8})
        with self.assertRaisesRegex(ValueError, "exact declared slot"):
            guard.atomic_json(target, {"larger":"payload"}, storage=storage)
        self.assertEqual(list(directory.iterdir()), [])
        slot_storage = Coordinator(telemetry_roots=(directory,), fixed_slots={directory/"stop-record.json":8})
        with self.assertRaisesRegex(ValueError, "64KiB fixed slot"):
            guard._Recorder(directory, slot_storage)
        self.assertEqual(list(directory.iterdir()), [])

    def test_atomic_publication_diagnostics_preserve_primary_and_retain_orphan_bytes(self):
        from tdyssey.wm_compat_storage import Coordinator, StorageError
        directory = self.directory/"telemetry"; directory.mkdir()
        target = directory/"worker-status.json"
        storage = Coordinator(telemetry_roots=(directory,), atomic_targets=(target,))
        guard.atomic_json(target, {"old":True}, storage=storage)
        unlink = Path.unlink; cleanup_attempts = []
        primary = OSError(22, "synthetic primary publication failure"); primary.winerror = 1175
        def cleanup_failure(path, *args, **kwargs):
            if path.name.startswith("worker-status.json.") and path.name.endswith(".tmp"):
                cleanup_attempts.append(str(path))
                raise PermissionError("synthetic cleanup failure")
            return unlink(path, *args, **kwargs)
        with patch.object(guard, "_publish_json", side_effect=primary) as publication, \
                patch.object(guard.Path, "unlink", cleanup_failure):
            with self.assertRaises(StorageError) as caught:
                guard.atomic_json(target, {"new":True}, storage=storage)
        self.assertEqual(caught.exception.details["phase"], "atomic.replace")
        self.assertEqual(caught.exception.details["exception_type"], "OSError")
        self.assertIn("primary publication", str(caught.exception))
        self.assertIs(caught.exception.__cause__, primary)
        self.assertEqual((caught.exception.details["errno"], caught.exception.details["winerror"]), (22, 1175))
        self.assertEqual(publication.call_count, 1)
        self.assertEqual(cleanup_attempts, [])
        self.assertFalse(any("cleanup failure" in note for note in getattr(caught.exception, "__notes__", [])))
        orphans = list(directory.glob("worker-status.json.*.tmp"))
        self.assertEqual(len(orphans), 1)
        usage = storage.usage()
        self.assertEqual(usage["telemetry_bytes"], 2*guard.STATUS_BYTES)
        with self.assertRaisesRegex(StorageError, "Orphaned temporary"):
            guard.atomic_json(target, {"retry":"forbidden"}, storage=storage)
        self.assertEqual(list(directory.glob("worker-status.json.*.tmp")), orphans)
        self.evidence("guard-orphan-primary-error.json", dict(error=caught.exception.details,
            cleanup_notes=getattr(caught.exception, "__notes__", []), cleanup_attempts=cleanup_attempts,
            retained_orphan_bytes=orphans[0].stat().st_size,
            usage=usage, retry_refused=True))

    def test_atomic_only_retries_explicit_windows_sharing_or_lock_errors(self):
        from tdyssey.wm_compat_storage import Coordinator, StorageError
        replace = guard._publish_json
        for shared_mode in (True, False):
            for winerror in (None, 5, 32, 33, 1175):
                with self.subTest(shared=shared_mode, winerror=winerror):
                    directory = self.directory/(str(shared_mode)+"-"+str(winerror)); directory.mkdir()
                    target = directory/"worker-status.json"
                    storage = Coordinator(telemetry_roots=(directory,), atomic_targets=(target,)) if shared_mode else None
                    guard.atomic_json(target, {"old":True}, storage=storage)
                    old = target.read_bytes()
                    failure = PermissionError(13, "synthetic publication denial")
                    if winerror is not None:
                        failure.winerror = winerror
                    calls = []
                    def maybe_replace(source, destination, **kwargs):
                        calls.append(1)
                        if len(calls) == 1:
                            raise failure
                        return replace(source, destination, **kwargs)
                    with patch.object(guard, "_publish_json", maybe_replace), patch.object(guard.time, "sleep") as sleep:
                        if not shared_mode and winerror in (32, 33):
                            guard.atomic_json(target, {"ok":winerror}, storage=storage)
                            self.assertEqual(len(calls), 2)
                            sleep.assert_called_once_with(.02)
                            self.assertEqual(json.loads(target.read_bytes()), {"ok":winerror})
                        else:
                            with self.assertRaises(StorageError) as caught:
                                guard.atomic_json(target, {"denied":winerror}, storage=storage)
                            self.assertEqual(caught.exception.details["exception_type"], "PermissionError")
                            self.assertIs(caught.exception.__cause__, failure)
                            self.assertEqual(len(calls), 1)
                            sleep.assert_not_called()
                            self.assertEqual(target.read_bytes(), old)
                            self.assertEqual(len(list(directory.glob("worker-status.json.*.tmp"))), int(shared_mode))

    def test_stdout_pipe_quota_rejects_before_append(self):
        from tdyssey.wm_compat_storage import Coordinator
        directory = self.directory/"telemetry"; directory.mkdir()
        path = directory/"worker-output.log"
        storage = Coordinator(telemetry_roots=(directory,), append_only=(path,), limits={"telemetry_bytes":64})
        pump = guard._OutputPump(io.BytesIO(b"x"*100), path, storage)
        pump.thread.join(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "stdout_pump_failed"):
            pump.close()
        self.assertEqual(path.stat().st_size, 0)
        self.assertIsNotNone(guard._storage_error_details(pump.error))
        self.evidence("guard-stdout-quota.json", dict(limit=64, attempted_bytes=100,
            actual_bytes=path.stat().st_size, error=guard._storage_error_details(pump.error)))

    def test_stdout_primary_write_error_survives_secondary_pipe_close_failure(self):
        from tdyssey.wm_compat_storage import Coordinator
        directory = self.directory/"telemetry"; directory.mkdir()
        path = directory/"worker-output.log"
        storage = Coordinator(telemetry_roots=(directory,), append_only=(path,))
        ready, release = threading.Event(), threading.Event()
        class Source(io.BytesIO):
            def read(source, count):
                ready.set()
                if not release.wait(2): raise TimeoutError("test barrier")
                return super().read(count)
            def close(source):
                super().close()
                raise OSError("synthetic secondary pipe close")
        pump = guard._OutputPump(Source(b"tiny"), path, storage)
        self.assertTrue(ready.wait(1))
        original = pump.stream
        pump.stream = Mock(wraps=original)
        pump.stream.write.side_effect = OSError("synthetic primary stdout write")
        release.set(); pump.thread.join(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "primary stdout write"):
            pump.close()
        self.assertTrue(any("secondary pipe close" in note for note in pump.error.__notes__))

    def test_repeated_publications_and_journal_appends_share_one_cross_instance_lock(self):
        from tdyssey.wm_compat_storage import Coordinator, create_lock_file
        output, ledger = self.directory/"output", self.directory/"ledger"
        directory = output/"guardian"/"synthetic"; directory.mkdir(parents=True)
        policy = guard._storage_policy(output, ledger, directory, (ledger,))
        create_lock_file(policy["lock_path"])
        writer, journal, scanner = (Coordinator(**policy) for _ in range(3))
        for storage in (writer, journal, scanner):
            self.addCleanup(storage.close)
        target, journal_path = directory/"worker-status.json", ledger/"journal.jsonl"
        errors, start = [], threading.Barrier(3)
        def publications():
            try:
                start.wait(2)
                for sequence in range(40):
                    guard.atomic_json(target, {"synthetic_sequence":sequence}, storage=writer)
            except BaseException as exc: errors.append(exc)
        def appends():
            try:
                start.wait(2)
                for sequence in range(40):
                    data=(json.dumps({"synthetic_only":True,"sequence":sequence})+"\n").encode()
                    with journal.local_transaction("synthetic.journal"):
                        ticket = journal.reserve_bytes(journal_path,len(data),phase="synthetic.journal")
                        ledger.mkdir(exist_ok=True)
                        with journal_path.open("ab", buffering=0) as stream:
                            journal.validate_append(journal_path, stream)
                            guard._write_all(stream, data)
                            journal.record_append(journal_path, data, stream)
                        journal.commit(ticket, len(data))
            except BaseException as exc: errors.append(exc)
        threads = [threading.Thread(target=publications), threading.Thread(target=appends)]
        for thread in threads: thread.start()
        start.wait(2)
        samples = [guard.storage_usage(output, (ledger,), storage=scanner) for _ in range(40)]
        for thread in threads: thread.join(timeout=5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertFalse(errors, errors)
        self.assertEqual(guard._read(target)["synthetic_sequence"], 39)
        self.assertEqual(len(journal_path.read_text().splitlines()), 40)
        self.assertFalse(list(directory.glob("worker-status.json.*.tmp")))
        self.evidence("guard-concurrent-heartbeat-journal.json", dict(status_publications=40,journal_appends=40,
            snapshots=len(samples), errors=[], final_usage=scanner.audit(), all_temporary_files_cleaned=True))


@unittest.skipUnless(sys.platform == "win32", "native Windows Job Object ownership test")
class NativeWindowsChecks(unittest.TestCase):
    def test_share_delete_reader_keeps_old_status_while_replacement_and_new_read_succeed(self):
        from tdyssey.wm_compat_storage import Coordinator, StorageError
        RACE_EVIDENCE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RACE_EVIDENCE,prefix="shared-reader-") as temporary:
            directory=Path(temporary);status=directory/"worker-status.json"
            storage=Coordinator(telemetry_roots=(directory,),atomic_targets=(status,))
            old,new={"generation":"old"},{"generation":"new"}
            guard.atomic_json(status,old,storage=storage)
            with guard._open_shared_read(status) as held:
                original_identity=os.fstat(held.fileno()).st_ino
                guard.atomic_json(status,new,storage=storage)
                held.seek(0)
                held_value=json.loads(held.read())
                self.assertEqual(held_value,old)
                self.assertEqual(guard._read(status),new)
                self.assertNotEqual(status.stat().st_ino,original_identity)
            immutable=directory/"synthetic-immutable-claim.json"
            guard.atomic_json(immutable,old)
            with self.assertRaisesRegex(StorageError,"Immutable JSON identity/content changed"):
                with guard._open_shared_read(immutable) as held:
                    self.assertEqual(json.loads(held.read()),old)
                    immutable.write_bytes(b'{"modified": true}')
            evidence=dict(kind="native_windows_share_delete_json_reader",old_reader_value=held_value,
                new_reader_value=guard._read(status),replacement_succeeded_while_reader_handle_open=True,
                immutable_replacement_rejected=True,model_operations=0,decoder_operations=0,physics_operations=0,
                gpu_queries=0)
            guard.atomic_json(EVIDENCE/"native-share-delete-reader.json",evidence)

    def test_harmless_sleeping_worker_and_grandchild_are_in_owned_memory_limited_job_and_killed(self):
        RACE_EVIDENCE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RACE_EVIDENCE, prefix="guard-native-") as temporary:
            directory = Path(temporary)
            worker_identity = directory/"output/results/worker-identity.json"
            grand_identity = directory/"output/results/grandchild-identity.json"
            target = directory/"harmless_worker.py"
            target.write_text('''import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from tdyssey.wm_compat_guard import verify_guardian,WorkerHeartbeat,from_handoff,_immutable_json,_write_all
handoff=verify_guardian()
storage=from_handoff(handoff)
identity=Path(sys.argv[1])
with storage.local_transaction("native.worker_identity"):
 identity.parent.mkdir(parents=True,exist_ok=True)
 _immutable_json(identity,{"pid":os.getpid(),"parent":os.getppid(),"native_handoff_verified":True},storage)
code="import os,json,sys,time;from pathlib import Path;from tdyssey.wm_compat_storage import Coordinator;from tdyssey.wm_compat_guard import _immutable_json;storage=Coordinator(**json.loads(os.environ['WM_COMPAT_STORAGE_POLICY']));_immutable_json(Path(sys.argv[1]),{'pid':os.getpid(),'parent':os.getppid()},storage);time.sleep(30)"
subprocess.Popen([sys.executable,"-B","-c",code,sys.argv[2]])
payload=dict(phase="setup",model_instances=0,model_loads=0,optimizer_updates=0,cuda_allocated_bytes=0,cuda_reserved_bytes=0,
 environment_creations=0,resets=0,restores=0,controls=0,physics_substeps=0,real_renders=0,policy_sampling=0)
with WorkerHeartbeat(payload,handoff=handoff):
 while True:
  journal=Path(sys.argv[3])/"journal.jsonl"
  data=(json.dumps({"synthetic_only":True,"at":time.time()})+"\\n").encode()
  with storage.local_transaction("native.synthetic_journal"):
   ticket=storage.reserve_bytes(journal,len(data),phase="native.synthetic_journal")
   journal.parent.mkdir(parents=True,exist_ok=True)
   with journal.open("ab",buffering=0) as stream:
    storage.validate_append(journal,stream)
    _write_all(stream,data)
    storage.record_append(journal,data,stream)
   storage.commit(ticket,len(data))
  print("harmless-alive",flush=True)
  time.sleep(.1)
''', encoding="utf-8")
            kernel, _ = guard.policy_memory._libraries()
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            retained = {"handle": None, "pid": None}
            class NativeProvider:
                def startup(self, output):
                    return resources()
                def start(self):
                    pass
                def runtime(self, pid, output):
                    value = resources()
                    actual = guard.policy_memory.process_memory(pid)
                    value.update(working_set_bytes=actual.get("working_set_bytes"), private_bytes=actual.get("private_bytes"))
                    if grand_identity.exists() and retained["handle"] is None:
                        try:
                            identity = json.loads(grand_identity.read_text())
                        except json.JSONDecodeError:
                            return value
                        retained.update(pid=identity["pid"], handle=kernel.OpenProcess(0x00100000, False, identity["pid"]))
                    return value
                def close(self):
                    pass
            with patch.object(guard.policy_memory, "gpu_memory", side_effect=AssertionError("Native engineering test must not query GPU")):
                report = guard.supervise([str(target), str(worker_identity), str(grand_identity), str(directory/"ledger")],
                    output_dir=directory/"output", ledger_dir=directory/"ledger", lock_path=directory/"one-model.lock",
                    provider=NativeProvider(), python_executable=sys.executable, deadline_seconds=2.5, poll_seconds=.05)
            try:
                report["worker_output"] = (Path(report["directory"])/"worker-output.log").read_text(encoding="utf-8", errors="replace")
                EVIDENCE.mkdir(parents=True, exist_ok=True)
                guard.atomic_json(EVIDENCE/("native-attempt-"+Path(report["directory"]).name+".json"), report)
                self.assertEqual(report["guard_reason"], "deadline_exceeded", report)
                self.assertFalse(report["owner_injected"])
                self.assertTrue(report["provider_injected"])
                self.assertTrue(report["job_empty_verified_before_close"])
                self.assertTrue(report["final_audit"]["passed"])
                identity = json.loads(worker_identity.read_text())
                self.assertEqual(identity["pid"], report["actual_worker_pid"])
                self.assertTrue(identity["native_handoff_verified"])
                self.assertIn("harmless-alive", report["worker_output"])
                self.assertGreater(len((directory/"ledger/journal.jsonl").read_text().splitlines()), 2)
                self.assertIsNotNone(retained["handle"])
                self.assertNotEqual(retained["handle"], 0)
                signaled = kernel.WaitForSingleObject(retained["handle"], 3000)
                self.assertEqual(signaled, 0, "retained grandchild handle was not signaled after Job closure")
                self.assertFalse((directory/"one-model.lock").exists())
                EVIDENCE.mkdir(parents=True, exist_ok=True)
                guard.atomic_json(EVIDENCE/"native-last.json", {"kind": "native_windows_harmless_process_tree",
                    "native_worker_job_handoff_verified": True, "native_job_memory_limit_bytes": 3*guard.GIB,
                    "retained_grandchild_pid": retained["pid"], "retained_grandchild_handle_signaled": signaled == 0,
                    "gpu_resource_provider": "injected fixed counters; no nvidia-smi call",
                    "model_operations": 0, "decoder_operations": 0, "physics_operations": 0,
                    "worker_identity": identity, "report": report})
            finally:
                if retained["handle"]:
                    kernel.CloseHandle(retained["handle"])


if __name__ == "__main__":
    unittest.main()
