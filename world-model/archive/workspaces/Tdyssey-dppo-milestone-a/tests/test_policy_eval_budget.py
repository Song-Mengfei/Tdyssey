"""Evaluation-only authority tests with temporary files, never real operations."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import policy_eval_budget as module
from tdyssey.policy_eval_budget import BudgetBusy, BudgetExceeded, EvaluationBudget, LIMITS


ROOT = Path(__file__).resolve().parents[1]
BINDING = {"training_experiment_id": "original-training", "config_hash": "a" * 64,
           "base_identity": {"bytes": 1095708090, "sha256": "b" * 64},
           "delta_identity": {"bytes": 16003916, "sha256": "c" * 64},
           "case_digest": "d" * 64, "methods": ["base_training", "delta_training"],
           "max_controls": 300}


class EvaluationBudgetChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.registry, self.run = self.directory / "registry", self.directory / "output"

    def open(self, *, create=False, run=None, binding=None, evaluation_id="paired-unit"):
        return EvaluationBudget(self.registry, evaluation_id, deepcopy(binding or BINDING),
                                run or self.run, create=create)

    def claim(self, budget, group="z-base", method="base_training"):
        return budget.claim_evaluation_group(group, method=method,
            protocol=module.PROTOCOL, case_digest=BINDING["case_digest"])

    def test_fixed_caps_and_completed_groups_persist_across_output_directories(self):
        with self.open(create=True) as budget:
            first_started = budget.snapshot()["started_at"]
            public_binding = budget.binding
            public_binding["case_digest"] = "f" * 64
            self.assertEqual(budget.binding, BINDING)
            self.assertEqual(budget.snapshot()["limits"], LIMITS)
            self.assertEqual(set(budget.snapshot()["reserved"]), {"development"})
            self.claim(budget)
            for kind in ("control", "plan", "reset"):
                event = budget.reserve(kind, shape="T", amount=LIMITS[kind] // 2)
                budget.mark_completed(event)
                with self.assertRaises(BudgetExceeded):
                    budget.reserve(kind, shape="T")
            budget.complete_evaluation_group("z-base", "e" * 64)
        with self.open(run=self.directory / "another-output") as budget:
            self.assertEqual(budget.snapshot()["started_at"], first_started)
            self.assertEqual(budget.remaining("control"), 6000)
            self.claim(budget, "a-delta", "delta_training")
            for kind in ("control", "plan", "reset"):
                event = budget.reserve(kind, shape="L", amount=LIMITS[kind] // 2)
                budget.mark_completed(event)
                self.assertEqual(budget.remaining(kind), 0)
            budget.complete_evaluation_group("a-delta", "f" * 64)
            state = budget.snapshot()
            self.assertEqual(state["reserved"], state["completed"])
            self.assertEqual(state["reserved"]["development"], {"control": 12000, "plan": 1520, "reset": 80})
            self.assertEqual(len(state["output_dirs"]), 2)
            self.assertFalse(state["pending"])
            with self.assertRaises(BudgetExceeded):
                self.claim(budget, "third", "delta_training")
        # JSON key order differs from method chronology (a-delta before z-base).
        with self.open() as budget:
            self.assertEqual(len(budget.snapshot()["evaluation_groups"]), 2)
            with self.assertRaises(BudgetExceeded):
                self.claim(budget)

    def test_training_kinds_phases_and_unclaimed_physical_operations_are_rejected(self):
        with self.open(create=True) as budget:
            with self.assertRaises(BudgetBusy):
                budget.reserve("control", shape="T")
            for kind in ("actor_step", "critic_step", "batch", "train", "storage"):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    budget.reserve(kind, shape="T")
            for phase in ("train", "diagnostic", "global"):
                with self.subTest(phase=phase), self.assertRaises(ValueError):
                    budget.reserve("control", phase=phase, shape="T")
            self.claim(budget)
            with self.assertRaises(ValueError):
                budget.reserve("control", shape=None)
            self.assertFalse(any(budget.snapshot()["reserved"]["development"].values()))

    def test_group_order_protocol_cases_and_duplicate_binding_cannot_refill_account(self):
        with self.open(create=True) as budget:
            for method, protocol, digest in (("delta_training", module.PROTOCOL, "d" * 64),
                                             ("base_original", module.PROTOCOL, "d" * 64),
                                             ("base_training", "original_t", "d" * 64),
                                             ("base_training", module.PROTOCOL, "e" * 64)):
                with self.assertRaises(ValueError):
                    budget.claim_evaluation_group("bad", method=method, protocol=protocol, case_digest=digest)
            self.assertEqual(budget.snapshot()["evaluation_groups"], {})
        with self.assertRaisesRegex(FileExistsError, "cannot refill"):
            self.open(create=True, evaluation_id="new-id", run=self.directory / "new-output")
        self.assertFalse((self.directory / "new-output").exists())
        changed = deepcopy(BINDING); changed["delta_identity"]["sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            self.open(binding=changed)
        with self.assertRaises(FileExistsError):
            self.open(create=True, run=self.directory / "third-output")

    def test_failed_physical_attempt_is_never_refunded_or_reclaimed_after_reopen(self):
        with self.open(create=True) as budget:
            self.claim(budget)
            event = budget.reserve("control", shape="L")
            with self.assertRaises(BudgetBusy):
                budget.reserve("plan", shape="L")
        with self.open(run=self.directory / "audit-output") as budget:
            self.assertTrue(budget.snapshot()["audit_only_after_interruption"])
            self.assertEqual(budget.reserved_count("control"), 1)
            self.assertEqual(budget.remaining("control"), 11999)
            self.assertEqual(budget.snapshot()["completed"]["development"]["control"], 0)
            with self.assertRaises(BudgetBusy):
                self.claim(budget, "another")
            with self.assertRaises(BudgetBusy):
                budget.reserve("control", shape="L")
            with self.assertRaises(ValueError):
                budget.mark_completed(event)
            with self.assertRaises(BudgetBusy):
                budget.complete_evaluation_group("z-base", "e" * 64)
            # Failure evidence can still be stored without restarting physics.
            path = self.directory / "audit-output/failure.json"
            written = budget.reserve_storage(2, path=path)
            path.write_bytes(b"{}")
            budget.mark_storage_written(written, 2)
            self.assertIn(str(event), budget.snapshot()["pending"])

    def test_load_failure_consumes_claim_even_without_physical_operations(self):
        with self.open(create=True) as budget:
            self.claim(budget)
        with self.open() as budget:
            state = budget.snapshot()
            self.assertFalse(state["evaluation_groups"]["z-base"]["completed"])
            self.assertTrue(state["audit_only_after_interruption"])
            with self.assertRaises(BudgetBusy):
                self.claim(budget, "retry-base")
            with self.assertRaises(BudgetBusy):
                self.claim(budget, "delta", "delta_training")

    def test_wall_time_persists_and_completions_and_audit_survive_deadline(self):
        clock = [1000.0, 500.0]
        with patch.object(module.time, "time", side_effect=lambda: clock[0]), \
                patch.object(module.time, "monotonic", side_effect=lambda: clock[1]):
            with self.open(create=True) as budget:
                self.claim(budget)
                event = budget.reserve("reset", shape="T")
                clock[:] = [8201.0, 7701.0]
                with self.assertRaises(BudgetExceeded):
                    budget.check_time()
                budget.mark_completed(event)
                path = self.run / "timeout.json"
                storage = budget.reserve_storage(2, path=path)
                path.write_bytes(b"{}")
                budget.mark_storage_written(storage, 2)
                budget.complete_evaluation_group("z-base", "e" * 64)
                self.assertGreaterEqual(budget.snapshot()["elapsed_wall_seconds"], 7201)
            # A backward wall clock/new output directory never revives the job.
            clock[:] = [1001.0, 1.0]
            with self.open(run=self.directory / "late-output") as budget:
                with self.assertRaises(BudgetExceeded):
                    self.claim(budget, "delta", "delta_training")
                with self.assertRaises(BudgetExceeded):
                    budget.check_time()
                self.assertEqual(budget.reserved_count("reset"), 1)

    def test_append_growth_storage_cap_and_pending_write_remain_authoritative(self):
        with self.open(create=True) as budget:
            path = self.run / "events.jsonl"
            for payload, mode in ((b"ab", "wb"), (b"cde", "ab")):
                event = budget.reserve_storage(len(payload), path=path)
                with path.open(mode) as stream:
                    stream.write(payload)
                budget.mark_storage_written(event, len(payload))
            self.assertEqual(budget.snapshot()["written_files"][str(path.resolve())]["bytes"], 5)
            with self.assertRaises(BudgetExceeded):
                budget.reserve_storage(LIMITS["storage_bytes"] + 1)
            with self.assertRaises(BudgetExceeded):
                budget.reserve_storage(LIMITS["storage_bytes"])
            with self.assertRaises(ValueError):
                budget.reserve_storage(1, path=self.directory / "outside.txt")
            event = budget.reserve_storage(10, path=self.run / "unfinished.json")
            with self.assertRaises(BudgetBusy):
                budget.reserve_storage(1, path=self.run / "unfinished.json")
        with self.open(run=self.directory / "audit-output") as budget:
            self.assertEqual(budget.snapshot()["storage"]["pending_bytes"], 10)
            self.assertTrue(budget.snapshot()["audit_only_after_interruption"])
            with self.assertRaises(ValueError):
                budget.mark_storage_written(event, 0)

    def test_cross_process_owner_lock_missing_authority_and_training_registry_fail_closed(self):
        with self.open(create=True):
            code = ("import json,sys; from tdyssey.policy_eval_budget import EvaluationBudget,BudgetBusy\n"
                    "try: EvaluationBudget(sys.argv[1],'paired-unit',json.loads(sys.argv[3]),sys.argv[2])\n"
                    "except BudgetBusy: print('locked')\n"
                    "else: raise RuntimeError('concurrent owner accepted')\n")
            child = subprocess.run([sys.executable, "-B", "-S", "-c", code,
                                    str(self.registry), str(self.run), json.dumps(BINDING)],
                                   cwd=ROOT, capture_output=True, text=True, timeout=20)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(child.stdout.strip(), "locked")
        (self.registry / "paired-unit/ledger.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.open()
        with self.assertRaises(FileExistsError):
            self.open(create=True)
        fake_training = self.directory / "fake-training-registry"
        fake_training.mkdir()
        original = b'{"schema_version":1,"experiments":{},"run_bindings":{}}'
        (fake_training / "index.json").write_bytes(original)
        with self.assertRaises(ValueError):
            EvaluationBudget(fake_training, "eval", BINDING, self.directory / "unused", create=True)
        self.assertEqual((fake_training / "index.json").read_bytes(), original)
        self.assertEqual([path.name for path in fake_training.iterdir()], ["index.json"])

    def test_post_replace_failure_poisoning_never_overwrites_durable_attempt(self):
        with self.open(create=True) as budget:
            self.claim(budget)
            original = module._atomic_json
            def commit_then_fail(path, payload):
                original(path, payload)
                if path == budget.path and payload["reserved"]["development"]["control"] == 1:
                    raise OSError("synthetic failure after durable replacement")
            with patch.object(module, "_atomic_json", side_effect=commit_then_fail):
                with self.assertRaisesRegex(OSError, "after durable"):
                    budget.reserve("control", shape="T")
            self.assertEqual(json.loads(budget.path.read_text())["reserved"]["development"]["control"], 1)
            with self.assertRaisesRegex(BudgetBusy, "poisoned"):
                budget.snapshot()
            budget.close()
            self.assertEqual(json.loads(budget.path.read_text())["reserved"]["development"]["control"], 1)
        with self.open() as budget:
            self.assertEqual(budget.remaining("control"), 11999)
            self.assertTrue(budget.snapshot()["audit_only_after_interruption"])
            with self.assertRaises(BudgetBusy):
                budget.reserve("control", shape="T")

    def test_binding_validation_and_source_only_import_do_not_load_training_runtime(self):
        for key, value in (("max_controls", 299), ("methods", ["delta_training", "base_training"]),
                           ("case_digest", "not-a-digest")):
            binding = deepcopy(BINDING); binding[key] = value
            with self.assertRaises(ValueError):
                self.open(create=True, binding=binding)
        code = ("import sys; import tdyssey.policy_eval_budget; "
                "assert not any(n == 'torch' or n == 'pymunk' or n == 'tdyssey.official' "
                "or n == 'tdyssey.policy_rl' or n == 'tdyssey.policy_checkpoint' for n in sys.modules)")
        subprocess.run([sys.executable, "-B", "-S", "-c", code], cwd=ROOT,
                       capture_output=True, text=True, check=True)


if __name__ == "__main__":
    unittest.main()
