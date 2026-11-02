"""Pure-stdlib config/temporary-ledger tests; no real C experiment or assets."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tdyssey import policy_protocol as protocol
from tdyssey.policy_protocol import BudgetBusy, BudgetExceeded, CConfig, ExperimentBudget


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/policy_c.example.json"


class CConfigChecks(unittest.TestCase):
    def test_proposal_is_fixed_disjoint_canonical_and_does_not_create_outputs(self):
        config = CConfig.from_json(EXAMPLE)
        data = config.to_dict()
        self.assertEqual(data["status"], "proposed_not_executed")
        self.assertEqual(len(data["scenes"]["training"]["T"]), 32)
        self.assertEqual(len(data["scenes"]["training"]["L"]), 32)
        for scenes in data["scenes"]["training"].values():
            self.assertFalse(set(scenes) & set(data["scenes"]["excluded_evaluation"]))
        self.assertEqual(data["budgets"]["diagnostic"], {"control": 0, "plan": 0, "reset": 0})
        # Matched reference T plus requested shape: two settling resets/case.
        cases = len(data["shapes"]) * len(data["evaluation"]["scenes"]) * len(data["evaluation"]["policy_seeds"])
        self.assertEqual(cases, 20)
        self.assertEqual(cases * 300 * 4, data["budgets"]["development"]["control"])
        self.assertEqual(cases * 38 * 4, data["budgets"]["development"]["plan"])
        self.assertEqual(cases * 2 * 4, data["budgets"]["development"]["reset"])
        self.assertEqual(config.config_hash, CConfig.from_dict(dict(reversed(list(data.items())))).config_hash)
        original_seed = data["scenes"]["training"]["T"][0]
        data["scenes"]["training"]["T"][0] = 1
        self.assertEqual(config.data["scenes"]["training"]["T"][0], original_seed)
        self.assertEqual(config.hash, config.config_hash)

    def test_rejects_leakage_nonfinite_caps_and_incompatible_capacity(self):
        for change in ("leakage", "missing_exclusion", "infinite", "none", "too_many_steps", "capacity", "settings"):
            data = CConfig.from_json(EXAMPLE).to_dict()
            if change == "leakage":
                data["scenes"]["training"]["L"][0] = 100003
            elif change == "missing_exclusion":
                data["scenes"]["excluded_evaluation"][-1] = 100005
            elif change == "infinite":
                data["budgets"]["wall_time_seconds"] = float("inf")
            elif change == "none":
                data["budgets"]["train"]["plan"] = None
            elif change == "too_many_steps":
                data["budgets"]["global"]["actor_step"] = 73
            elif change == "capacity":
                data["budgets"]["global"]["batch"] = 1
            else:
                data["optimization"]["std_floor"] = .02
            with self.subTest(change=change), self.assertRaises(ValueError):
                CConfig.from_dict(data)


class ExperimentBudgetChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.registry = self.directory / "registry"
        self.run = self.directory / "run"
        self.config = CConfig.from_json(EXAMPLE)
        self.now = time.time()
        self.monotonic = time.monotonic()
        self.wall_patch = patch.object(protocol.time, "time", side_effect=lambda: self.now)
        self.monotonic_patch = patch.object(protocol.time, "monotonic", side_effect=lambda: self.monotonic)
        self.wall_patch.start()
        self.monotonic_patch.start()
        self.addCleanup(self.wall_patch.stop)
        self.addCleanup(self.monotonic_patch.stop)

    def open(self, *, create=False, run=None, config=None, experiment_id="unit-proposal"):
        return ExperimentBudget(self.registry, experiment_id, config or self.config, run or self.run, create=create)

    def write(self, budget, path, payload):
        identifier = budget.reserve_storage(len(payload), path=path)
        with path.open("ab") as stream:
            stream.write(payload)
        return budget.mark_storage_written(identifier, len(payload))

    def test_init_is_explicit_invalid_config_and_existing_b_artifacts_are_untouched(self):
        with self.assertRaises(FileNotFoundError):
            self.open()
        self.assertFalse(self.registry.exists())
        data = self.config.to_dict()
        data["budgets"]["train"]["control"] = None
        with self.assertRaises(ValueError):
            self.open(create=True, config=data)
        self.assertFalse(self.registry.exists())
        self.run.mkdir()
        (self.run / "budget.json").write_text('{"milestone":"B"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-C"):
            self.open(create=True)
        self.assertEqual((self.run / "budget.json").read_text(), '{"milestone":"B"}')
        self.assertFalse((self.registry / "unit-proposal/ledger.json").exists())

    def test_new_run_directory_and_resume_preserve_attempts_and_full_config_identity(self):
        with self.open(create=True) as budget:
            failed = budget.reserve("control", shape="T")
            completed = budget.reserve("plan", shape="T")
            budget.mark_completed(completed)
            self.assertEqual(budget.remaining("control"), 2047)
            self.assertEqual(budget.snapshot()["completed"]["train"]["control"], 0)
        with self.open(run=self.directory / "second-run") as reopened:
            self.assertEqual(reopened.reserved_count("control"), 1)
            self.assertIn(str(failed), reopened.snapshot()["pending"])
            self.assertEqual(reopened.snapshot()["completed"]["train"]["plan"], 1)
            self.assertEqual(len(reopened.snapshot()["output_dirs"]), 2)
            self.assertTrue(reopened.run_dir.joinpath("c-experiment.json").exists())
        with self.assertRaises(FileExistsError):
            self.open(create=True, run=self.directory / "third-run")
        with self.assertRaises(FileNotFoundError):
            self.open(experiment_id="new-id", run=self.directory / "third-run")
        changed = self.config.to_dict()
        changed["train_seed"] += 1
        with self.assertRaises(ValueError):
            self.open(config=CConfig.from_dict(changed))

    def test_phase_caps_are_separate_global_operations_are_train_only_and_failed_attempts_spent(self):
        with self.open(create=True) as budget:
            for phase, kind, count in (("train", "control", 2048), ("development", "reset", 160),
                                        ("train", "actor_step", 72), ("train", "batch", 24)):
                event = budget.reserve(kind, phase=phase, amount=count)
                self.assertEqual(budget.remaining(kind, phase), 0)
                self.assertIs(type(budget.remaining(kind, phase)), int)
                with self.assertRaises(BudgetExceeded):
                    budget.reserve(kind, phase=phase)
                self.assertFalse(budget.snapshot()["pending"][str(event)]["completed"])
            self.assertEqual(budget.remaining("control", "development"), 24000)
            with self.assertRaises(BudgetExceeded):
                budget.reserve("plan", phase="diagnostic")
            with self.assertRaises(ValueError):
                budget.reserve("critic_step", phase="development")

    def test_wall_deadline_persists_across_resume_and_monotonic_time_cannot_refund_it(self):
        with self.open(create=True) as budget:
            self.now += 20
            self.monotonic += 20
            self.assertAlmostEqual(budget.check_time(), 7180)
            event = budget.reserve("control")
        self.now += 7181  # Downtime counts against this same experiment deadline.
        with self.open() as reopened:
            with self.assertRaises(BudgetExceeded):
                reopened.check_time()
            reopened.mark_completed(event)  # Already-performed work can still be logged.
            self.assertEqual(reopened.snapshot()["completed"]["train"]["control"], 1)
        self.now -= 10000
        with self.open() as reopened:
            with self.assertRaises(BudgetExceeded):
                reopened.reserve("control")

    def test_append_growth_pending_path_exclusion_and_storage_survive_reopen(self):
        path = self.run / "training.jsonl"
        with self.open(create=True) as budget:
            self.write(budget, path, b'{"batch":1}\n')
            self.write(budget, path, b'{"batch":2}\n')
            self.assertEqual(path.stat().st_size, 24)
            self.assertEqual(budget.snapshot()["written_files"][str(path)]["bytes"], 24)
            pending = budget.reserve_storage(1024, path=self.run / "failed-write.bin")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                budget.reserve_storage(1024, path=self.run / "failed-write.bin")
            with self.assertRaises(BudgetExceeded):
                budget.reserve_storage(self.config.data["budgets"]["storage_bytes"] + 1)
        with self.open() as reopened:
            storage = reopened.check_storage()
            self.assertEqual(storage["pending_bytes"], 1024)
            self.assertIn(str(pending), reopened.snapshot()["pending"])
            self.assertEqual(reopened.snapshot()["written_files"][str(path)]["bytes"], 24)
            other = self.directory / "separate-output"
            reopened.add_output_dir(other)
            (other / "unreserved.bin").write_bytes(b"x" * 10)
            self.assertGreater(reopened.check_storage()["current_bytes"], storage["current_bytes"])

    def test_checkpoint_retention_and_training_boundary_allow_independent_development(self):
        with self.open(create=True) as budget:
            checkpoints = [self.run / f"checkpoint-{index}.pt" for index in range(3)]
            markers = []
            for index, path in enumerate(checkpoints):
                completed = budget.reserve("batch")
                budget.mark_completed(completed)
                written = self.write(budget, path, bytes([index]) * 16)
                marker = budget.record_boundary(path, {"batch": index + 1})
                markers.append(marker)
                if index == 2:
                    self.assertEqual(written["retire_paths"], [str(checkpoints[0])])
                    self.assertEqual(marker["retire_paths"], [str(checkpoints[0])])
                    with self.assertRaises(BudgetExceeded):
                        budget.reserve("control")
                    budget.retire_paths(marker["retire_paths"])
            self.assertFalse(checkpoints[0].exists())
            self.assertTrue(checkpoints[1].exists() and checkpoints[2].exists())
            for kind in ("plan", "reset", "control"):
                event = budget.reserve(kind, phase="development", shape="L")
                budget.mark_completed(event)
            budget.claim_evaluation_group("independent-dev", method="base_original", protocol="matched_tl", case_digest="a" * 64)
            self.assertEqual(budget.validate_boundary(checkpoints[2])["metadata"], {"batch": 3})
            # Failed training attempts also invalidate an older recovery marker.
            budget.reserve("plan", shape="T")
            with self.assertRaisesRegex(ValueError, "current exact training boundary"):
                budget.validate_boundary(checkpoints[2])
            foreign = self.run / "checkpoint-foreign.pt"
            foreign.write_bytes(b"never registered")
            with self.assertRaises(ValueError):
                budget.retire_paths([foreign])
            self.assertTrue(foreign.exists())

    def test_evaluation_group_claims_are_permanent_across_output_directories(self):
        with self.open(create=True) as budget:
            for index in range(4):
                token = budget.claim_evaluation_group(f"group-{index}", method="base_training", protocol="matched_tl", case_digest="b" * 64)
                self.assertIsInstance(token, str)
            budget.complete_evaluation_group("group-0", "c" * 64)
            with self.assertRaises(BudgetExceeded):
                budget.claim_evaluation_group("group-0", method="base_training", protocol="matched_tl", case_digest="b" * 64)
        with self.open(run=self.directory / "evaluation-elsewhere") as reopened:
            with self.assertRaises(BudgetExceeded):
                reopened.claim_evaluation_group("new-group", method="base_original", protocol="original_t", case_digest="d" * 64)
            self.assertEqual(len(reopened.snapshot()["evaluation_groups"]), 4)
            self.assertTrue(reopened.snapshot()["evaluation_groups"]["group-0"]["completed"])

    def test_completed_history_is_bounded_and_incomplete_attempts_remain_durable(self):
        with self.open(create=True) as budget:
            failed = budget.reserve("control", shape="T")
            for _ in range(80):
                event = budget.reserve("control", shape="L")
                budget.mark_completed(event)
            state = budget.snapshot()
            self.assertEqual(len(state["recent_events"]), 64)
            self.assertEqual(len(state["pending"]), 1)
            self.assertIn(str(failed), state["pending"])
        with self.open() as reopened:
            self.assertEqual(reopened.reserved_count("control"), 81)
            self.assertEqual(reopened.snapshot()["completed"]["train"]["control"], 80)

    def test_cross_process_lock_and_missing_authority_fail_closed(self):
        with self.open(create=True):
            code = ("import sys; from tdyssey.policy_protocol import CConfig,ExperimentBudget,BudgetBusy\n"
                    "try: ExperimentBudget(sys.argv[1],'unit-proposal',CConfig.from_json(sys.argv[2]),sys.argv[3])\n"
                    "except BudgetBusy: print('locked')\n"
                    "else: raise RuntimeError('concurrent owner accepted')\n")
            child = subprocess.run([sys.executable, "-S", "-B", "-c", code, str(self.registry), str(EXAMPLE), str(self.run)],
                                   cwd=ROOT, capture_output=True, text=True, timeout=20)
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(child.stdout.strip(), "locked")
        (self.registry / "unit-proposal/ledger.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.open()
        with self.assertRaises(FileExistsError):
            self.open(create=True, run=self.directory / "new-run")

    def test_post_replace_failure_poisoning_cannot_refund_a_committed_reservation(self):
        with self.open(create=True) as budget:
            original = protocol._atomic_json

            def commit_then_fail(path, payload):
                original(path, payload)
                if path == budget.path and payload["reserved"]["train"]["control"] == 1:
                    raise OSError("simulated post-replace directory fsync failure")

            with patch.object(protocol, "_atomic_json", side_effect=commit_then_fail):
                with self.assertRaisesRegex(OSError, "post-replace"):
                    budget.reserve("control", shape="T")
            self.assertEqual(json.loads(budget.path.read_text())["reserved"]["train"]["control"], 1)
            with self.assertRaisesRegex(BudgetBusy, "poisoned"):
                budget.reserve("control", shape="T")
            budget.close()  # Unlock only: never flush the stale in-memory zero.
            self.assertEqual(json.loads(budget.path.read_text())["reserved"]["train"]["control"], 1)
        with self.open() as reopened:
            self.assertEqual(reopened.reserved_count("control"), 1)
            self.assertEqual(reopened.remaining("control"), 2047)
            event = reopened.reserve("control", shape="T")
            self.assertEqual(reopened.reserved_count("control"), 2)
            reopened.mark_completed(event)


if __name__ == "__main__":
    unittest.main()
