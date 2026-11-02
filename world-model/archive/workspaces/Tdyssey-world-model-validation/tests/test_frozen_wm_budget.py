"""Synthetic diagnostic accounting checks; no simulator/model imports or work."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import frozen_wm_budget as module
from tdyssey.frozen_wm_budget import Budget, BudgetBusy, BudgetExceeded, LIMITS

ROOT = Path(__file__).resolve().parents[1]
BINDING = {"protocol_sha256": "a" * 64, "config_sha256": "b" * 64,
           "source_sha256": {"tdyssey/fake.py": "c" * 64}, "checkpoint_sha256": "d" * 64}


class FrozenBudgetChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output, self.ledger = self.directory / "output", self.directory / "ledger"
        patched = patch.object(module, "REGISTRY_ROOT", self.directory / "registry")
        patched.start()
        self.addCleanup(patched.stop)

    def open(self, output=None, ledger=None, binding=None):
        return Budget(output or self.output, ledger or self.ledger, deepcopy(BINDING if binding is None else binding))

    def test_fixed_limits_composite_attempts_and_durable_abort_never_refund(self):
        with self.open() as budget:
            self.assertEqual(budget.snapshot()["limits"], LIMITS)
            operation = budget.reserve(branches=1, reset=1, restore=1, settle=1, controls=7, substeps=70)
            # Failed attempts consume the complete reservation, even before controls.
            budget.abort(operation, {"reason": "invalid_initial_geometry", "actual_controls": 0})
            state = budget.snapshot()
            self.assertEqual(state["reserved"]["controls"], 7)
            self.assertEqual(state["aborted"]["controls"], 7)
            self.assertEqual(state["completed"]["controls"], 0)
            operation = budget.reserve(forecast_images=4, euler_steps=12)
            budget.complete(operation, {"frozen": True})
            self.assertEqual(budget.snapshot()["completed"]["forecast_images"], 4)
            with self.assertRaises(ValueError):
                budget.complete(operation)
        saved = json.loads((self.ledger / "ledger.json").read_text())
        self.assertEqual(saved["reserved"]["branches"], 1)
        rows = [json.loads(line) for line in (self.ledger / "journal.jsonl").read_text().splitlines()]
        self.assertEqual([row["kind"] for row in rows], ["initialize", "reserve", "aborted", "reserve", "completed", "close"])
        self.assertEqual([row["event_id"] for row in rows], list(range(1, 7)))

    def test_same_binding_cannot_restart_by_new_ledger_or_output_or_after_interruption(self):
        with self.open() as budget:
            operation = budget.reserve(reset=1)
        before = (self.ledger / "ledger.json").read_bytes()
        for output, ledger in ((self.output, self.ledger), (self.directory / "other", self.ledger),
                                (self.directory / "other", self.directory / "other-ledger")):
            with self.subTest(output=output, ledger=ledger), self.assertRaises(FileExistsError):
                self.open(output, ledger)
        self.assertEqual((self.ledger / "ledger.json").read_bytes(), before)
        self.assertIn(operation, json.loads(before)["pending"])
        self.assertFalse((self.directory / "other-ledger").exists())
        changed = deepcopy(BINDING)
        changed["config_sha256"] = "e" * 64
        changed["source_sha256"] = {"new.py": "f" * 64}
        with self.assertRaises(FileExistsError):
            self.open(self.directory / "changed-output", self.directory / "changed-ledger", changed)
        claim = json.loads((self.directory / "registry" / (module.ATTEMPT_ID + ".json")).read_text())
        self.assertEqual(claim["binding"], BINDING)
        self.assertFalse((self.directory / "changed-ledger").exists())

    def test_all_fixed_operation_caps_and_optimizer_prohibition(self):
        with self.open() as budget:
            operation = budget.reserve(**{name: cap for name, cap in LIMITS.items()
                if name not in ("output_bytes", "telemetry_bytes", "optimizer_updates")})
            budget.complete(operation)
            for name in LIMITS:
                if name in ("output_bytes", "telemetry_bytes"):
                    continue
                with self.subTest(name=name), self.assertRaises(BudgetExceeded):
                    budget.reserve(**{name: 1})
            for costs in ({}, {"training_steps": 1}, {"reset": True}, {"controls": -1}, {"controls": 0}):
                with self.subTest(costs=costs), self.assertRaises(ValueError):
                    budget.reserve(**costs)
            self.assertEqual(budget.snapshot()["policy_optimizer_updates"], 0)
            self.assertEqual(budget.snapshot()["world_model_optimizer_updates"], 0)

    def test_pending_owner_and_persistence_failure_fail_closed(self):
        with self.open() as budget:
            operation = budget.reserve(calibration=1, reset=1, restore=1, settle=1)
            with self.assertRaises(BudgetBusy):
                budget.reserve(controls=1)
            budget.complete(operation)
            with patch.object(module, "_atomic_json", side_effect=OSError("synthetic disk failure")):
                with self.assertRaises(OSError):
                    budget.reserve(controls=1, substeps=10)
            with self.assertRaises(BudgetBusy):
                budget.reserve(controls=1)
        rows = [json.loads(line) for line in (self.ledger / "journal.jsonl").read_text().splitlines()]
        self.assertEqual(rows[-1]["kind"], "reserve")
        self.assertEqual(rows[-1]["detail"]["costs"], {"controls": 1, "substeps": 10})
        with self.assertRaises(FileExistsError):
            self.open(self.directory / "new-output", self.directory / "new-ledger")

    def test_wall_clock_rollback_cannot_extend_monotonic_deadline_and_abort_is_auditable(self):
        clock = [1000.0, 500.0]
        with patch.object(module.time, "time", side_effect=lambda: clock[0]), \
                patch.object(module.time, "monotonic", side_effect=lambda: clock[1]):
            with self.open() as budget:
                operation = budget.reserve(controls=1, substeps=10)
                clock[:] = [999.0, 2300.0]
                with self.assertRaises(BudgetExceeded):
                    budget.check()
                budget.abort(operation, {"reason": "wall_deadline"})
                self.assertEqual(budget.snapshot()["elapsed_seconds"], 1800)
                self.assertFalse(budget.snapshot()["pending"])
                clock[:] = [999.0, 501.0]
                with self.assertRaises(BudgetExceeded):
                    budget.reserve(controls=1)

    def test_actual_storage_separates_payload_and_telemetry_and_ledger(self):
        self.output.mkdir()
        (self.output / "telemetry").mkdir()
        (self.output / "telemetry/handshake.json").write_bytes(b"{}")
        with self.open() as budget:
            (self.output / "prediction.bin").write_bytes(b"0123456789")
            used = module.storage_usage(self.output, self.ledger)
            self.assertEqual(used["output_bytes"], 10)
            self.assertGreater(used["telemetry_bytes"], 2)
            with patch.object(module, "storage_usage", return_value={"output_bytes": LIMITS["output_bytes"] + 1,
                                                                    "telemetry_bytes": 0}):
                with self.assertRaises(BudgetExceeded):
                    budget.check()
            with self.assertRaises(BudgetExceeded):
                budget.reserve(output_bytes=LIMITS["output_bytes"] + 1)
            with patch.object(module, "storage_usage", return_value={"output_bytes": LIMITS["output_bytes"] - 2,
                                                                    "telemetry_bytes": 0}):
                with self.assertRaises(BudgetExceeded):
                    budget.reserve(output_bytes=3)

    def test_binding_validation_and_protected_outputs_have_no_account_side_effect(self):
        for binding in ({}, {**BINDING, "checkpoint_sha256": "BAD"}, {**BINDING, "source_sha256": {}}):
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                self.open(binding=binding)
        self.assertFalse((self.directory / "registry").exists())
        self.output.mkdir()
        protected = self.output / "training-ledger.json"
        protected.write_bytes(b"do not modify")
        with self.assertRaises(FileExistsError):
            self.open()
        self.assertEqual(protected.read_bytes(), b"do not modify")
        self.assertFalse(self.ledger.exists())

    def test_storage_scan_tolerates_atomic_temporary_file_disappearance(self):
        self.output.mkdir()
        vanished = self.output / "report.json.tmp"
        original_stat = Path.stat
        def racing_stat(path, *args, **kwargs):
            if path == vanished:
                raise FileNotFoundError("atomically replaced temporary")
            return original_stat(path, *args, **kwargs)
        with patch.object(Path, "rglob", return_value=iter([vanished])), \
                patch.object(Path, "stat", racing_stat), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "is_symlink", return_value=False):
            self.assertEqual(module.storage_usage(self.output, self.ledger),
                             {"output_bytes": 0, "telemetry_bytes": 0})

    def test_mutating_caller_binding_or_snapshot_does_not_change_the_authority(self):
        binding = deepcopy(BINDING)
        with Budget(self.output, self.ledger, binding) as budget:
            binding["source_sha256"]["tdyssey/fake.py"] = "e" * 64
            state = budget.snapshot()
            state["reserved"]["reset"] = 88
            self.assertEqual(budget.snapshot()["reserved"]["reset"], 0)
            self.assertEqual(budget.snapshot()["binding"], BINDING)

    def test_same_attempt_resume_replays_journal_ahead_reserve_and_keeps_costs_deadline_and_lineage(self):
        with self.open() as budget:
            completed = budget.reserve(reset=1, settle=1)
            budget.complete(completed)
            started = budget.snapshot()["started_at"]
            with patch.object(module, "_atomic_json", side_effect=PermissionError("synthetic Windows access denial")):
                with self.assertRaises(PermissionError):
                    budget.reserve(reset=1, settle=1)
        ledger_before = (self.ledger / "ledger.json").read_bytes()
        journal_before = (self.ledger / "journal.jsonl").read_bytes()
        claim_path = self.directory / "registry" / (module.ATTEMPT_ID + ".json")
        claim_before = claim_path.read_bytes()
        changed = deepcopy(BINDING)
        changed["source_sha256"] = {"repaired.py": "e"*64}
        with Budget.resume(self.output, self.ledger, changed) as budget:
            state = budget.snapshot()
            self.assertEqual(state["reserved"]["reset"], 2)
            self.assertEqual(state["completed"]["reset"], 1)
            self.assertEqual(state["aborted"]["reset"], 1)
            self.assertEqual(state["operations"]["2"]["status"], "aborted")
            self.assertFalse(state["pending"])
            self.assertEqual(state["started_at"], started)
            self.assertEqual(state["binding"], BINDING)
            self.assertEqual(state["active_binding"], changed)
            self.assertEqual(len(state["source_lineage"]), 1)
            operation = budget.reserve(forecast_images=4, euler_steps=12)
            self.assertEqual(operation, "3")
            budget.complete(operation)
        self.assertEqual(claim_path.read_bytes(), claim_before)
        self.assertTrue((self.ledger / "journal.jsonl").read_bytes().startswith(journal_before))
        self.assertEqual(next(self.ledger.glob("ledger-before-recovery-*.json")).read_bytes(), ledger_before)
        with Budget.resume(self.output, self.ledger, changed) as budget:
            self.assertEqual(budget.snapshot()["reserved"]["reset"], 2)
            self.assertEqual(budget.snapshot()["completed"]["forecast_images"], 4)
            self.assertEqual(len(budget.snapshot()["source_lineage"]), 2)

    def test_resume_rejects_changed_authority_expired_original_clock_and_corrupt_journal(self):
        clock = [1000., 500.]
        with patch.object(module.time, "time", side_effect=lambda: clock[0]), \
                patch.object(module.time, "monotonic", side_effect=lambda: clock[1]):
            with self.open():
                pass
            changed = deepcopy(BINDING)
            changed["checkpoint_sha256"] = "f"*64
            with self.assertRaises(ValueError):
                Budget.resume(self.output, self.ledger, changed)
            with self.assertRaises(ValueError):
                Budget.resume(self.directory / "new-output", self.ledger, BINDING)
            before = (self.ledger / "journal.jsonl").read_bytes()
            clock[:] = [2800., 501.]
            with self.assertRaises(BudgetExceeded):
                Budget.resume(self.output, self.ledger, BINDING)
            self.assertEqual((self.ledger / "journal.jsonl").read_bytes(), before)
            self.assertFalse(list(self.ledger.glob("ledger-before-recovery*")))
            clock[:] = [1001., 501.]
            with (self.ledger / "journal.jsonl").open("ab") as stream:
                stream.write(before.splitlines()[-1]+b"\n")
            with self.assertRaisesRegex(ValueError, "IDs"):
                Budget.resume(self.output, self.ledger, BINDING)

    def test_replace_retries_only_transient_failures_and_stops_with_finite_attempts(self):
        with patch.object(module.os, "replace", side_effect=[PermissionError("busy"), PermissionError("busy"), None]) as replace, \
                patch.object(module.time, "sleep") as sleep:
            module.replace_with_retry("source", "destination")
            self.assertEqual(replace.call_count, 3)
            self.assertEqual(sum(call.args[0] for call in sleep.call_args_list), .06)
        with patch.object(module.os, "replace", side_effect=PermissionError("still busy")) as replace, \
                patch.object(module.time, "sleep"):
            with self.assertRaises(PermissionError):
                module.replace_with_retry("source", "destination")
            self.assertEqual(replace.call_count, 5)
        with patch.object(module.os, "replace", side_effect=OSError("not a sharing violation")) as replace, \
                patch.object(module.time, "sleep") as sleep:
            with self.assertRaises(OSError):
                module.replace_with_retry("source", "destination")
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()


class FrozenGuardChecks(unittest.TestCase):
    def test_startup_floors_fail_closed_and_parent_imports_no_model_or_simulator(self):
        spec = importlib.util.spec_from_file_location("frozen_guard_test", ROOT / "scripts/run_guarded_frozen_wm.py")
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        gib = 1024**3
        initial = {"system": {"ok": True, "physical": {"available_bytes": 4 * gib},
                               "commit": {"headroom_bytes": 6 * gib}},
                   "gpu": {"ok": True, "devices": [{"free_bytes": 4 * gib}]},
                   "disk": {"ok": True, "free_bytes": 2 * gib}}
        self.assertIsNone(guard.startup_reason(initial))
        for group, fields in (("system", {"ok": False}), ("gpu", {"ok": False}),
                              ("disk", {"ok": True, "free_bytes": 2 * gib - 1})):
            bad = deepcopy(initial)
            bad[group] = fields
            self.assertIsInstance(guard.startup_reason(bad), str)
        source = (ROOT / "scripts/run_guarded_frozen_wm.py").read_text()
        self.assertNotIn("import torch", source)
        self.assertNotIn("import pymunk", source)
        self.assertIn("scripts/guarded_c_child.py", source)


if __name__ == "__main__":
    unittest.main()
