"""Reset-audit CLI IO/gates with stdlib fake callbacks; no real reset/runtime."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tdyssey.policy_protocol import CConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preflight_c_scenes.py"
CONFIG = ROOT / "configs/policy_c.example.json"
SPEC = importlib.util.spec_from_file_location("tdyssey_preflight_cli_checks", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def reset_result():
    return {"valid": True, "issues": [], "error": None,
            "reset_attempts": 1, "reset_physics_substeps": 1, "control_steps": 0,
            "test_only": True}


class PreflightCLIChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "diagnostic"
        self.config = CConfig.from_json(CONFIG)

    def child(self, *arguments):
        return subprocess.run([sys.executable, "-S", "-B", str(SCRIPT), *arguments],
                              cwd=self.directory, capture_output=True, text=True, timeout=20)

    def injected_runtime(self, engine, probe_factory):
        engine_module = ModuleType("tdyssey.policy_preflight")
        engine_module.run_reset_preflight = engine
        probe_module = ModuleType("tdyssey.policy_reset_probe")
        probe_module.RuntimeResetProbe = probe_factory
        return patch.dict(sys.modules, {"tdyssey.policy_preflight": engine_module,
                                        "tdyssey.policy_reset_probe": probe_module})

    def arguments(self, preserve=()):
        return SimpleNamespace(output_dir=self.output, source_root=Path("unused-fake-source"),
                               preserve_file=list(preserve))

    def test_default_plan_and_help_are_read_only_without_site_packages(self):
        plan = self.child("--output-dir", str(self.output))
        self.assertEqual(plan.returncode, 0, plan.stderr)
        value = json.loads(plan.stdout)
        self.assertEqual(value["status"], "proposed_not_executed")
        self.assertEqual(value["config_hash"], self.config.config_hash)
        self.assertEqual(value["maximum_reset_attempts"], 212)
        self.assertEqual(value["maximum_settle_substeps"], 212)
        self.assertEqual(value["candidate_order"], {"T": [400032, 400063], "L": [500032, 500063]})
        self.assertTrue(value["candidate_order_inclusive_ascending"])
        self.assertFalse(value["formal_C_experiment_initialized"])
        self.assertFalse(value["formal_C_clock_started"])
        self.assertEqual((value["policy_samples"], value["control_steps"], value["optimizer_steps"]), (0, 0, 0))
        help_result = self.child("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--execute", help_result.stdout)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_missing_execute_or_runtime_paths_fail_before_runtime_import_and_output(self):
        cases = (["run", "--source-root", "unused-source", "--output-dir", str(self.output)],
                 ["run", "--execute", "--output-dir", str(self.output)],
                 ["run", "--execute", "--source-root", "unused-source"])
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.child(*arguments)
                self.assertEqual(result.returncode, 1)
                self.assertIn("explicit --execute", result.stderr)
                self.assertNotIn("ModuleNotFoundError", result.stderr)
                self.assertNotIn("pymunk", result.stderr)
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_nonempty_diagnostic_directory_rejected_before_probe_or_resource_calls(self):
        self.output.mkdir()
        existing = self.output / "do-not-overwrite.json"
        existing.write_text('{"existing":true}', encoding="utf-8")
        factory, engine = Mock(), Mock()
        with (self.injected_runtime(engine, factory), patch.object(CLI, "resource_snapshot") as resources):
            with self.assertRaisesRegex(ValueError, "new empty"):
                CLI.run(self.arguments(), self.config)
        factory.assert_not_called()
        engine.assert_not_called()
        resources.assert_not_called()
        self.assertEqual(existing.read_text(encoding="utf-8"), '{"existing":true}')
        self.assertEqual([path.name for path in self.output.iterdir()], [existing.name])

    def test_audit_claim_is_visible_before_fake_reset_and_result_precedes_next_claim(self):
        self.output.mkdir()
        fake_resets = []

        def fake_reset(shape, seed, phase):
            records = [json.loads(line) for line in (self.output / "attempts.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["event"], "attempt_reserved")
            self.assertEqual(records[-1]["id"], len(fake_resets) + 1)
            if fake_resets:
                self.assertEqual(records[-2]["event"], "attempt_result")
            fake_resets.append((shape, seed, phase))
            return reset_result()

        audit = CLI.AuditedProbe(fake_reset, self.output, time.perf_counter())
        audit("T", 400000, "original_train")
        audit("L", 500000, "original_train")
        self.assertEqual((audit.calls, audit.resets, audit.physics, audit.controls), (2, 2, 2, 0))
        self.assertEqual(audit.pending_calls, {})
        events = [json.loads(line) for line in audit.journal.read_text().splitlines()]
        self.assertEqual([event["event"] for event in events],
                         ["attempt_reserved", "attempt_result", "attempt_reserved", "attempt_result"])
        self.assertEqual(fake_resets, [("T", 400000, "original_train"), ("L", 500000, "original_train")])

    def test_exception_keeps_the_durable_claim_pending_without_fabricating_zero_cost(self):
        self.output.mkdir()

        def interrupted_reset(shape, seed, phase):
            self.assertTrue((self.output / "attempts.jsonl").is_file())
            raise RuntimeError("fake reset callback interrupted after its reserved claim")

        audit = CLI.AuditedProbe(interrupted_reset, self.output, time.perf_counter())
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            audit("T", 400000, "original_train")
        self.assertEqual(audit.calls, 1)
        self.assertEqual(set(audit.pending_calls), {1})
        self.assertEqual(audit.pending_calls[1]["event"], "attempt_reserved")
        events = [json.loads(line) for line in audit.journal.read_text().splitlines()]
        self.assertEqual([event["event"] for event in events], ["attempt_reserved", "attempt_interrupted"])
        self.assertTrue(events[-1]["counts_unknown"])
        self.assertEqual(audit.controls, 0)

    def test_audit_caps_block_callback_and_json_byte_cap_prevents_file_creation(self):
        self.output.mkdir()
        callback = Mock(side_effect=lambda *args: reset_result())
        audit = CLI.AuditedProbe(callback, self.output, time.perf_counter())
        with patch.object(CLI, "MAX_RESETS", 1):
            audit("T", 400000, "original_train")
            rejected = audit("L", 500000, "original_train")
        self.assertEqual(callback.call_count, 1)
        self.assertIsNotNone(rejected["error"])
        self.assertEqual((audit.resets, audit.physics, audit.controls), (1, 1, 0))
        self.assertEqual(audit.pending_calls, {})
        path = self.output / "too-large.json"
        with patch.object(CLI, "MAX_BYTES", 1), self.assertRaisesRegex(RuntimeError, "storage cap"):
            CLI.write_json(path, {"payload": "must not be written"})
        self.assertFalse(path.exists())

    def test_fake_run_binds_config_plan_provenance_and_preserved_evidence(self):
        evidence = self.directory / "spent-B-evidence.json"
        evidence.write_text('{"spent":true,"controls":48}', encoding="utf-8")
        original_digest = CLI.digest_file(evidence)
        proposed = self.config.to_dict()
        proposed["scenes"]["training"]["T"][0] = 400032
        proposed_config = CConfig.from_dict(proposed)
        instances = []

        class FakeResetProbe:
            def __init__(self, source_root):
                self.source_root, self.calls, self.closed = str(source_root), [], False
                self.provenance = {"source_verified": False, "injected_test_dependencies": True,
                                   "policy_loaded": False, "test_only": True}
                instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True

            def __call__(self, shape, seed, phase):
                self.calls.append((shape, seed, phase))
                return reset_result()

        def fake_engine(config, audited):
            self.assertEqual(config.config_hash, self.config.config_hash)
            for shape, seed, phase in (("T", 400000, "original_train"),
                                       ("L", 500000, "original_train"),
                                       ("T", 400032, "candidate_train")):
                audited(shape, seed, phase)
            return {"status": "complete", "test_only": True,
                    "original_config_hash": config.config_hash,
                    "proposed_config": proposed_config.to_dict(),
                    "proposed_config_hash": proposed_config.config_hash}

        fake_resources = {"scope": "fake reset-only process; no GPU API queried", "process_memory": {"peak_working_set_bytes": 123}}
        fake_provenance = {"code_parent_commit": "fake-unit-commit", "code_sha256": {"fake": "a" * 64},
                           "source_verified": False, "checkpoint_loaded": False, "policy_loaded": False}
        with (self.injected_runtime(fake_engine, FakeResetProbe),
              patch.object(CLI, "resource_snapshot", return_value=fake_resources) as resources,
              patch.object(CLI, "provenance", return_value=fake_provenance)):
            report = CLI.run(self.arguments([evidence]), self.config)
        self.assertEqual(resources.call_count, 2)
        self.assertTrue(instances[0].closed)
        self.assertEqual(len(instances[0].calls), 3)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["audit_counts"], {"probe_calls": 3, "reset_attempts": 3,
                          "reset_physics_substeps": 3, "control_steps": 0, "unresolved_probe_claims": []})
        for field in ("formal_C_experiment_initialized", "formal_C_clock_started", "formal_evaluation", "policy_loaded"):
            self.assertFalse(report[field])
        self.assertEqual((report["policy_samples"], report["optimizer_steps"]), (0, 0))
        self.assertTrue(report["runtime_probe_provenance"]["injected_test_dependencies"])
        self.assertEqual(report["provenance"], fake_provenance)
        self.assertEqual(report["preserved_evidence_sha256"][str(evidence.resolve())], original_digest)
        self.assertTrue(report["preserved_evidence_unchanged"][str(evidence.resolve())])
        self.assertEqual(CLI.digest_file(evidence), original_digest)
        original_file = json.loads((self.output / "original-config.json").read_text())
        proposed_file = json.loads((self.output / "proposed-config.json").read_text())
        self.assertEqual(original_file, self.config.to_dict())
        self.assertEqual(proposed_file, proposed_config.to_dict())
        self.assertEqual(CConfig.from_dict(proposed_file).config_hash, report["proposed_config_hash"])
        self.assertEqual(json.loads((self.output / "declared-plan.json").read_text()), CLI.declared_plan(self.config))
        self.assertEqual(json.loads((self.output / "report.json").read_text()), report)
        self.assertFalse((self.directory / "registry").exists())


if __name__ == "__main__":
    unittest.main()
