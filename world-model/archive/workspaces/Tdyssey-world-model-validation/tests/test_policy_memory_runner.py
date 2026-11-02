"""Fake-process checks for resource refusal and exact-child termination."""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("guarded_c_runner", ROOT / "scripts/run_guarded_c_policy.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
GIB = 1024**3


class Child:
    pid = 12345
    returncode = None
    terminated = False
    waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout):
        self.waited = True
        return self.returncode

    def poll_exit(self):
        return self.returncode is not None

    def terminate_and_wait(self, timeout):
        self.terminate()
        self.wait(timeout)

    def close(self):
        pass

    def exit_code(self):
        return self.returncode


class GuardedRunnerTests(unittest.TestCase):
    def test_source_binding_normalizes_windows_paths_and_keeps_model_code(self):
        selected = RUNNER.runtime_sources({"tdyssey\\policy_ppo.py": "actor",
            "scripts\\run_c_policy.py": "entry", "tdyssey\\policy_memory.py": "monitor",
            "scripts\\run_guarded_c_policy.py": "parent"})
        self.assertEqual(selected, {"tdyssey/policy_ppo.py": "actor", "scripts/run_c_policy.py": "entry"})

    def invoke(self, directory, snapshots, child):
        arguments = ["runner", "load", "--python", "existing-python", "--source-root", "existing-source",
                     "--base-checkpoint", "existing-base", "--output-dir", str(Path(directory) / "probe")]
        gpu = {"ok": True, "devices": [{"free_bytes": 6 * GIB}]}
        with patch.object(RUNNER.sys, "argv", arguments), \
                patch.object(RUNNER, "provenance", return_value={}), \
                patch.object(RUNNER, "claim_worker", return_value=child), \
                patch.object(RUNNER, "system_memory", side_effect=snapshots), \
                patch.object(RUNNER, "gpu_memory", return_value=gpu), \
                patch.object(RUNNER, "disk_snapshot", return_value={"ok": True, "free_bytes": 10 * GIB}), \
                patch.object(RUNNER, "process_memory", return_value={"ok": True, **{k: 100 for k in
                    ("working_set_bytes", "private_bytes", "peak_working_set_bytes", "peak_pagefile_bytes")}}) as measure, \
                patch.object(RUNNER.subprocess, "Popen", return_value=child) as spawn, \
                patch("builtins.print"):
            code = RUNNER.main()
        return code, spawn, measure

    def test_initial_low_memory_refuses_without_spawning_or_initializing(self):
        low = {"ok": True, "physical": {"available_bytes": GIB}, "commit": {"headroom_bytes": 10 * GIB}}
        with tempfile.TemporaryDirectory() as directory:
            code, spawn, measure = self.invoke(directory, [low], Child())
            self.assertEqual(code, 2)
            spawn.assert_not_called()
            measure.assert_not_called()
            self.assertFalse((Path(directory) / "probe/init-output.log").exists())

    def test_runtime_floor_stops_only_spawned_child_and_waits(self):
        high = {"ok": True, "physical": {"available_bytes": 5 * GIB}, "commit": {"headroom_bytes": 10 * GIB}}
        low = {"ok": True, "physical": {"available_bytes": GIB // 2}, "commit": {"headroom_bytes": 10 * GIB}}
        child = Child()
        with tempfile.TemporaryDirectory() as directory:
            code, spawn, measure = self.invoke(directory, [high, low, high], child)
            self.assertEqual(code, 2)
            spawn.assert_called_once()
            measure.assert_called_once_with(child.pid)
            self.assertTrue(child.terminated)
            self.assertTrue(child.waited)

    def test_zero_exit_without_ready_report_is_rejected(self):
        high = {"ok": True, "physical": {"available_bytes": 5 * GIB}, "commit": {"headroom_bytes": 10 * GIB}}
        child = Child()
        child.returncode = 0
        with tempfile.TemporaryDirectory() as directory:
            code, spawn, measure = self.invoke(directory, [high, high], child)
            self.assertEqual(code, 2)
            report = json.loads((Path(directory) / "probe/monitor-report.json").read_text())
            self.assertIn("child_acceptance_report_failed", report["guard_reason"])

    def test_shutdown_error_still_writes_failure_evidence(self):
        high = {"ok": True, "physical": {"available_bytes": 5 * GIB}, "commit": {"headroom_bytes": 10 * GIB}}
        low = {"ok": True, "physical": {"available_bytes": GIB // 2}, "commit": {"headroom_bytes": 10 * GIB}}
        child = Child()
        with tempfile.TemporaryDirectory() as directory, patch.object(child, "terminate", side_effect=OSError("injected shutdown failure")):
            code, spawn, measure = self.invoke(directory, [high, low, high], child)
            self.assertEqual(code, 2)
            report = json.loads((Path(directory) / "probe/monitor-report.json").read_text())
            self.assertIn("injected shutdown failure", report["monitor_error"])
            self.assertIsNone(report["returncode"])

    def evaluate(self, directory, ready, *, init_code=0):
        folder = Path(directory)
        run_dir, output_dir = folder / "evaluated", folder / "monitored"
        arguments = ["runner", "evaluate", "--python", "existing-python", "--source-root", "existing-source",
                     "--base-checkpoint", "existing-base", "--output-dir", str(output_dir),
                     "--experiment-id", "paired-account", "--registry", str(folder / "eval-registry"),
                     "--run-dir", str(run_dir), "--checkpoint", "selected-delta", "--training-ledger", "training-ledger"]
        high = {"ok": True, "physical": {"available_bytes": 5 * GIB}, "commit": {"headroom_bytes": 10 * GIB}}
        gpu = {"ok": True, "devices": [{"free_bytes": 6 * GIB}]}
        child = Child()
        child.returncode = 0
        calls = []

        def init(command, **kwargs):
            calls.append(("init", command))
            return SimpleNamespace(returncode=init_code)

        def spawn(command, **kwargs):
            calls.append(("worker", command))
            if ready is not None:
                run_dir.mkdir()
                (run_dir / "evaluation-pair-report.json").write_text(json.dumps(ready), encoding="utf-8")
            return child

        with patch.object(RUNNER.sys, "argv", arguments), \
                patch.object(RUNNER, "provenance", return_value={}), \
                patch.object(RUNNER, "claim_worker", return_value=child), \
                patch.object(RUNNER, "system_memory", return_value=high), \
                patch.object(RUNNER, "gpu_memory", return_value=gpu), \
                patch.object(RUNNER, "disk_snapshot", return_value={"ok": True, "free_bytes": 10 * GIB}), \
                patch.object(RUNNER.subprocess, "run", side_effect=init), \
                patch.object(RUNNER.subprocess, "Popen", side_effect=spawn), \
                patch("builtins.print"):
            code = RUNNER.main()
        return code, calls, output_dir

    @staticmethod
    def completed_pair():
        return {"status": "completed", "training_steps": 0,
                "groups": [{"status": "complete", "method": "base_training"},
                           {"status": "complete", "method": "delta_training"}],
                "comparison": {"available": True, "report": {"paired_cases": 20}}}

    def test_evaluate_requires_explicit_identity_paths_before_outputs_or_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "must-not-exist"
            arguments = ["runner", "evaluate", "--python", "existing-python", "--source-root", "existing-source",
                         "--base-checkpoint", "existing-base", "--output-dir", str(folder),
                         "--experiment-id", "paired", "--run-dir", "future-evaluation"]
            with patch.object(RUNNER.sys, "argv", arguments), \
                    patch.object(RUNNER, "system_memory") as measure, \
                    patch.object(RUNNER.subprocess, "Popen") as spawn, \
                    contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as refused:
                RUNNER.main()
            self.assertEqual(refused.exception.code, 2)
            self.assertIn("--checkpoint and --training-ledger", errors.getvalue())
            self.assertFalse(folder.exists())
            measure.assert_not_called()
            spawn.assert_not_called()

    def test_evaluate_initializes_separate_account_then_runs_exact_entry_without_load_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            code, calls, output = self.evaluate(directory, self.completed_pair())
            self.assertEqual(code, 0)
            self.assertEqual([name for name, command in calls], ["init", "worker"])
            declaration = json.loads((output / "declaration.json").read_text())
            command = declaration["command"]
            expected = ["existing-python", "-B", str(ROOT / "scripts/evaluate_c_policy_pair.py"), "run", "--execute",
                        "--account-id", "paired-account", "--registry", str(Path(directory) / "eval-registry"),
                        "--run-dir", str(Path(directory) / "evaluated"),
                        "--config", str(ROOT / "configs/policy_c.example.json"),
                        "--source-root", "existing-source", "--base-checkpoint", "existing-base",
                        "--checkpoint", "selected-delta", "--training-ledger", "training-ledger"]
            self.assertEqual(command, expected)
            self.assertEqual(calls[0][1], ["existing-python", "-S", "-B", expected[2], "init", *expected[4:]])
            self.assertEqual(declaration["guards"]["deadline_seconds"], 7200)
            self.assertEqual(declaration["guards"]["telemetry_limit_bytes"], 64 * 1024**2)
            self.assertNotIn("load-report", " ".join(command))
            self.assertEqual(calls[1][1][-len(command[2:]):], command[2:])
            report = json.loads((output / "monitor-report.json").read_text())
            self.assertEqual(report["phase"], "evaluate")
            self.assertEqual(report["worker_exit_code"], 0)
            self.assertIsNone(report["guard_reason"])

    def test_evaluate_init_failure_never_launches_worker_or_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            code, calls, output = self.evaluate(directory, self.completed_pair(), init_code=3)
            self.assertEqual(code, 2)
            self.assertEqual([name for name, command in calls], ["init"])
            report = json.loads((output / "monitor-report.json").read_text())
            self.assertEqual(report, {"status": "init_failed", "returncode": 3})

    def test_evaluate_zero_exit_requires_completed_pair_comparison_and_zero_training(self):
        invalid = [None]
        for key, value in (("status", "incomplete"), ("training_steps", 1), ("training_steps", False),
                           ("groups", [{"status": "complete", "method": "base_training"}]),
                           ("groups", [{"status": "complete", "method": "delta_training"},
                                       {"status": "complete", "method": "base_training"}]),
                           ("groups", [{"status": "failed", "method": "base_training"},
                                       {"status": "complete", "method": "delta_training"}]),
                           ("comparison", {"available": False})):
            ready = self.completed_pair()
            ready[key] = value
            invalid.append(ready)
        for ready in invalid:
            with self.subTest(report=ready), tempfile.TemporaryDirectory() as directory:
                code, calls, output = self.evaluate(directory, ready)
                self.assertEqual(code, 2)
                self.assertEqual(len(calls), 2)
                report = json.loads((output / "monitor-report.json").read_text())
                self.assertIn("child_acceptance_report_failed", report["guard_reason"])


if __name__ == "__main__":
    unittest.main()
