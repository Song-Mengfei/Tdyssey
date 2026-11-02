"""First-batch gate using scalar reports and fake owners; no policy or physics."""

from copy import deepcopy
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tdyssey import policy_experiment as experiment
    from tdyssey.policy_experiment import train_batches, validate_first_batch
    from tdyssey.policy_protocol import CConfig


def valid_row():
    metrics = {"max_logprob_error": 0.0, "ratio_min": 1.0, "ratio_max": 1.0,
               "sample_approx_kl": 0.0, "clip_fraction": 0.0, "transition_count": 200}
    change = {"changed_names": ["fixture.weight"], "changed_tensor_count": 1,
              "l2": .001, "max_abs": .001}
    counts = {"train": {"control": 2, "plan": 2, "reset": 2},
              "global": {"batch": 1, "actor_step": 1, "critic_step": 1},
              "development": {"control": 0, "plan": 0, "reset": 0},
              "diagnostic": {"control": 0, "plan": 0, "reset": 0}}
    return {"event": "batch", "batch": 1, "requested_passes": 1, "budget_allowed_passes": 1,
            "collection": {"plans": 2, "controls": 2},
            "counts": {"reserved": deepcopy(counts), "completed": deepcopy(counts)},
            "update": {"batch_sequence": 0, "requested_passes": 1, "completed_passes": 1,
                "actor_steps": 1, "critic_steps": 1, "entry": deepcopy(metrics),
                "stop_reason": "completed", "frozen_unchanged": True, "frozen_digest": "a" * 64,
                "passes": [{"index": 0, "actor_loss": 0.0, "critic_loss": .1,
                    "actor_grad_norm": .2, "critic_grad_norm": .3, "critic_mse": .09,
                    "critic_explained_variance": None, "after": deepcopy(metrics),
                    "actor_weight_change": deepcopy(change), "critic_weight_change": deepcopy(change)}]}}


class FakeBudget:
    def __init__(self):
        empty = valid_row()["counts"]["completed"]
        self.completed = {phase: dict.fromkeys(values, 0) for phase, values in empty.items()}
        self.reserved = deepcopy(self.completed)

    def add(self, kind, amount=1):
        phase = "global" if kind in ("batch", "actor_step", "critic_step") else "train"
        self.reserved[phase][kind] += amount
        self.completed[phase][kind] += amount

    def snapshot(self):
        return {"reserved": deepcopy(self.reserved), "completed": deepcopy(self.completed)}

    def remaining(self, kind, phase="train"):
        return 100

    def check_time(self):
        return 100.0


class FakeCollector:
    def __init__(self, budget, *, boundary=False):
        self.budget, self.controls, self.calls = budget, 0, 0
        self.is_episode_boundary = boundary

    def summary(self):
        return {"controls": self.controls, "draining": False,
                "last_batch": {"stop_reason": "fake_only"}}

    def collect_batch(self, plans, controls, actor_version):
        self.calls += 1
        self.controls += 2
        self.budget.add("control", 2)
        self.budget.add("plan", 2)
        if self.calls == 1:
            self.budget.add("reset", 2)
        record = {"n_contacts": 1, "raw_reward": .1, "training_reward": .2,
                  "shaping": .1, "coverage": .1, "coverage_before": 0.0}
        return [[SimpleNamespace(shape=shape, control_records=[deepcopy(record)])] for shape in ("T", "L")]

    def request_boundary(self):
        pass

    def release_boundary(self):
        pass


class FakeUpdater:
    def __init__(self, budget, mutate=None):
        self.budget, self.mutate = budget, mutate
        self.batches = self.actor_steps = self.critic_steps = 0
        self.config = SimpleNamespace(max_passes=3)
        self.buffer_ref = None

    @property
    def actor_version(self):
        return f"fake-only:actor:{self.actor_steps}"

    def prepare(self, fragments):
        self.batches += 1
        self.budget.add("batch")
        buffer = SimpleBuffer()
        self.buffer_ref = weakref.ref(buffer)
        return buffer

    def update(self, buffer, *, passes):
        self.actor_steps += passes
        self.critic_steps += passes
        self.budget.add("actor_step", passes)
        self.budget.add("critic_step", passes)
        report = deepcopy(valid_row()["update"])
        report.update(batch_sequence=self.batches - 1, requested_passes=passes,
                      completed_passes=passes, actor_steps=self.actor_steps, critic_steps=self.critic_steps)
        report["passes"] = [{**deepcopy(report["passes"][0]), "index": index} for index in range(passes)]
        if self.batches == 1 and self.mutate is not None:
            self.mutate(report)
        return report


class SimpleBuffer:
    pass


@unittest.skipIf(torch is None, "Optional torch runtime unavailable")
class FirstBatchGateChecks(unittest.TestCase):
    def test_valid_actual_change_and_zero_actor_loss_pass_without_mutating_input(self):
        row = valid_row()
        original = deepcopy(row)
        evidence = validate_first_batch(row)
        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(evidence["transition_count"], 200)
        self.assertEqual(evidence["frozen_digest"], "a" * 64)
        self.assertEqual(row, original)
        evidence["actor_weight_change"]["changed_names"].append("must_not_alias")
        self.assertEqual(row, original)

    def test_missing_actual_actor_or_critic_change_and_nonfinite_fields_fail(self):
        for owner in ("actor", "critic"):
            for key, value in (("changed_names", []), ("changed_tensor_count", 0),
                               ("l2", 0.0), ("max_abs", 0.0), ("l2", float("nan"))):
                with self.subTest(owner=owner, key=key, value=value):
                    row = valid_row()
                    row["update"]["passes"][0][owner + "_weight_change"][key] = value
                    with self.assertRaises(ValueError):
                        validate_first_batch(row)
            for key in (owner + "_loss", owner + "_grad_norm"):
                for value in (float("inf"), float("nan")):
                    with self.subTest(key=key, value=value):
                        row = valid_row()
                        row["update"]["passes"][0][key] = value
                        with self.assertRaises(ValueError):
                            validate_first_batch(row)

    def test_wrong_entry_probability_transition_count_frozen_or_step_counts_fail(self):
        mutations = [lambda r: r["update"]["entry"].update(max_logprob_error=.001),
                     lambda r: r["update"]["entry"].update(ratio_max=1.01),
                     lambda r: r["update"]["entry"].update(clip_fraction=.01),
                     lambda r: r["update"]["entry"].update(transition_count=199),
                     lambda r: r["update"].update(completed_passes=2),
                     lambda r: r["update"].update(frozen_unchanged=False),
                     lambda r: r["update"].update(frozen_digest="invalid"),
                     lambda r: r.update(requested_passes=2),
                     lambda r: r["counts"]["completed"]["global"].update(actor_step=0),
                     lambda r: r["counts"]["reserved"]["train"].update(plan=3)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                row = valid_row()
                mutate(row)
                with self.assertRaises(ValueError):
                    validate_first_batch(row)

    def run_loop(self, mutate=None, *, boundary=False, checkpoint=None, gate=None):
        budget = FakeBudget()
        collector = FakeCollector(budget, boundary=boundary)
        updater = FakeUpdater(budget, mutate)
        rows = []
        result = train_batches(collector, updater, budget, batch_plans=2, batch_controls=2,
            first_batch_passes=1, passes=2, max_batches=2, control_target=4,
            first_batch_gate=gate or validate_first_batch, log=rows.append, on_checkpoint=checkpoint)
        return result, rows, collector, updater

    def test_passing_gate_runs_once_then_allows_next_collection(self):
        checked_rows = []
        def gate(row):
            checked_rows.append(deepcopy(row))
            return validate_first_batch(row)
        result, rows, collector, updater = self.run_loop(gate=gate)
        self.assertEqual(collector.calls, 2)
        self.assertEqual((result["batches"], result["actor_steps"], result["critic_steps"]), (2, 3, 3))
        self.assertEqual(len(checked_rows), 1)
        self.assertEqual(result["first_batch_gate"]["status"], "passed")
        self.assertEqual(result["loop_state"]["first_batch_gate"], result["first_batch_gate"])
        batch_rows = [row for row in rows if row["event"] == "batch"]
        self.assertEqual(len(batch_rows), 2)
        for row in batch_rows:
            self.assertGreaterEqual(row["collection_seconds"], 0)
            self.assertGreaterEqual(row["batch_seconds"], row["collection_seconds"])
        self.assertEqual([row["event"] for row in rows[:3]], ["ppo_pass", "batch", "first_batch_gate"])
        gc.collect()
        self.assertIsNone(updater.buffer_ref())

    def test_failed_gate_stops_after_exactly_one_update_and_only_terminal_audit(self):
        checkpoints = []
        def checkpoint(**kwargs):
            checkpoints.append(deepcopy(kwargs))
            return {"resumable": kwargs["resumable"], "terminal": kwargs["terminal"]}
        result, rows, collector, updater = self.run_loop(
            lambda report: report["passes"][0]["actor_weight_change"].update(l2=0),
            boundary=True, checkpoint=checkpoint)
        self.assertEqual(result["stop_reason"], "first_batch_gate_failed")
        self.assertEqual((collector.calls, updater.batches, updater.actor_steps, updater.critic_steps), (1, 1, 1, 1))
        self.assertEqual(result["first_batch_gate"]["status"], "failed")
        self.assertIn("weights did not change", result["first_batch_gate"]["error"]["message"])
        self.assertEqual(len(checkpoints), 1)
        self.assertFalse(checkpoints[0]["resumable"])
        self.assertTrue(checkpoints[0]["terminal"])
        self.assertEqual(result["last_update"]["passes"][0]["actor_weight_change"]["l2"], 0)

    def test_nonfinite_report_is_retained_as_explicit_json_safe_failure(self):
        result, rows, collector, updater = self.run_loop(
            lambda report: report["passes"][0].update(actor_loss=float("nan")))
        self.assertEqual(result["stop_reason"], "first_batch_gate_failed")
        self.assertEqual(collector.calls, 1)
        self.assertEqual(result["last_update"]["passes"][0]["actor_loss"], {"nonfinite": "nan"})
        self.assertIn("finite", result["first_batch_gate"]["error"]["message"])
        json.dumps(result, allow_nan=False)
        json.dumps(rows, allow_nan=False)

    def test_terminal_audit_error_preserves_failed_gate_and_diagnostics(self):
        def checkpoint(**kwargs):
            raise RuntimeError("fake audit storage unavailable")
        result, rows, collector, updater = self.run_loop(
            lambda report: report.update(frozen_unchanged=False), checkpoint=checkpoint)
        self.assertEqual(result["stop_reason"], "first_batch_gate_failed")
        self.assertIn("fake audit storage unavailable", result["checkpoint_error"])
        self.assertEqual(collector.calls, 1)
        self.assertEqual(result["first_batch_gate"]["status"], "failed")

    def test_failed_gate_resume_is_rejected_before_new_collection(self):
        result, rows, collector, updater = self.run_loop(lambda report: report.update(frozen_unchanged=False))
        with self.assertRaisesRegex(ValueError, "failed first batch gate"):
            train_batches(collector, updater, updater.budget, max_batches=2, control_target=4,
                          first_batch_gate=validate_first_batch, resume_state=result["loop_state"])
        self.assertEqual(collector.calls, 1)

    def test_failed_gate_boundary_save_never_registers_resumable_marker(self):
        config = CConfig.from_json(Path(__file__).resolve().parents[1] / "configs/policy_c.example.json")
        data = config.to_dict()
        collector_state = {"resumable": True, "episode_limit": data["training"]["episode_controls"],
            "gamma": data["optimization"]["gamma"], "beta": data["optimization"]["beta"],
            "scene_cursor": {"scenes": data["scenes"]["training"],
                             "excluded": sorted(data["scenes"]["excluded_evaluation"])}}
        collector = SimpleNamespace(state_dict=lambda **kwargs: deepcopy(collector_state), generators={"T": None})
        updater = SimpleNamespace(state_dict=lambda: {"actor_steps": 1, "critic_steps": 1})
        official = SimpleNamespace(checkpoint_sha256=data["base"]["sha256"])
        budget = FakeBudget()
        budget.snapshot = lambda: {**deepcopy(valid_row()["counts"]), "experiment_id": "fake-first-gate"}
        budget.reserve_storage = Mock(return_value=1)
        budget.mark_storage_written = Mock(return_value={"retire_paths": []})
        budget.record_boundary = Mock(side_effect=AssertionError("failed gate must never be resumable"))
        budget.check_storage = Mock()
        captured = []
        def fake_save(path, *args, metadata, **kwargs):
            captured.append(deepcopy(metadata))
            return {"path": str(path), "bytes": 1, "sha256": "a" * 64}
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(experiment, "_validate_runtime_configuration"),
              patch.object(experiment, "_validate_training_state"),
              patch.object(experiment, "save_update_checkpoint", side_effect=fake_save)):
            saved = experiment.save_c_checkpoint(Path(directory) / "checkpoint-failed-audit.pt", official,
                None, None, None, None, collector, updater, budget, config, terminal=True,
                metadata={"loop_state": {"first_batch_gate": {"status": "failed"}}})
        self.assertFalse(saved["resumable"])
        self.assertFalse(saved["boundary_recorded"])
        self.assertFalse(captured[0]["resumable"])
        budget.record_boundary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
