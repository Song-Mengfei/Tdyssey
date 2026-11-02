"""Pure decisions and workspace-local server monitor fixtures; no ML imports."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tdyssey.native_rgb_run_control import inspect_decision, monitor_training

ROOT = Path(__file__).resolve().parents[1]
SERVER = dict(server=True, early_stopping_patience=8, early_stopping_min_delta=1e-6,
              max_train_seconds=5400)


def checkpoints(losses):
    return [dict(update=(index+1)*250, denoising_loss=loss) for index, loss in enumerate(losses)]


@contextmanager
def fixture():
    sandbox = (ROOT/"artifacts").resolve()
    sandbox.mkdir(exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="native-rgb-run-control-cpu-test-", dir=sandbox)
    root = Path(temporary.name).resolve()
    if not root.is_relative_to(sandbox):
        raise AssertionError("Run-control fixture escaped workspace")
    try:
        yield root
    finally:
        if not root.resolve().is_relative_to(sandbox):
            raise AssertionError("Refusing cleanup outside owned fixture")
        temporary.cleanup()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n", encoding="utf-8")


class RunControlDecisions(unittest.TestCase):
    def test_patience_counts_unique_actual_denoising_checkpoints(self):
        self.assertFalse(inspect_decision(SERVER, checkpoints([1.]*8))["should_stop"])
        decision = inspect_decision(SERVER, checkpoints([1.]*9))
        self.assertTrue(decision["should_stop"])
        self.assertEqual(decision["reason"], "validation_patience")
        self.assertEqual(decision["consecutive_no_improvement"], 8)
        duplicate = checkpoints([1., 1.])
        duplicate += [dict(duplicate[-1]) for _ in range(20)]
        decision = inspect_decision(SERVER, duplicate)
        self.assertFalse(decision["should_stop"])
        self.assertEqual(decision["unique_validation_checkpoints"], 2)
        self.assertEqual(decision["consecutive_no_improvement"], 1)

    def test_min_delta_has_a_reference_and_preserves_true_observed_best(self):
        decision = inspect_decision(SERVER, checkpoints([1., 1.-.5e-6]))
        self.assertEqual(decision["best_denoising_loss"], 1.-.5e-6)
        self.assertEqual(decision["patience_reference_loss"], 1.)
        self.assertEqual(decision["consecutive_no_improvement"], 1)
        decision = inspect_decision(SERVER, checkpoints([1., 1.-.5e-6, 1.-2e-6]))
        self.assertEqual(decision["consecutive_no_improvement"], 0)
        self.assertEqual(decision["patience_reference_loss"], 1.-2e-6)

    def test_other_model_metrics_do_not_change_the_criterion(self):
        rows = checkpoints([1.]*9)
        for index, row in enumerate(rows):
            row.update(selection_score=-index, unseen_shape_score=index, h4_rgb_loss=1/(index+1))
        decision = inspect_decision(SERVER, rows)
        self.assertEqual(decision["reason"], "validation_patience")
        self.assertEqual(decision["criterion"]["validation_field"], "denoising_loss")
        self.assertFalse(decision["unseen_outputs_read"])

    def test_budget_guard_reserves_exactly_fifty_seconds_without_extension(self):
        self.assertFalse(inspect_decision(SERVER, [], dict(charged_seconds=5349.9, max_seconds=5400))["should_stop"])
        decision = inspect_decision(SERVER, [], dict(charged_seconds=5350, max_seconds=5400))
        self.assertEqual(decision["reason"], "checkpoint_time_reserve")
        self.assertEqual(decision["checkpoint_reserve_seconds"], 50.)
        self.assertEqual(decision["max_train_seconds"], 5400)
        invalid = checkpoints([1.]*9)+[dict(update=3000, denoising_loss=float("nan"))]
        self.assertFalse(inspect_decision(SERVER, invalid)["should_stop"])
        self.assertEqual(inspect_decision(SERVER, invalid, dict(charged_seconds=5351))["reason"], "checkpoint_time_reserve")


class RunControlMonitorChecks(unittest.TestCase):
    def test_local_context_does_not_create_files_or_start_a_thread(self):
        with fixture() as root:
            untouched = root/"untouched"
            with monitor_training(dict(server=False), untouched/"training", artifact_root=untouched) as state:
                self.assertFalse(state.enabled)
                self.assertIsNone(state.thread)
            self.assertFalse(untouched.exists())

    def test_server_signal_and_immutable_evidence_preserve_existing_stop(self):
        with fixture() as root:
            validation = root/"training/validation.json"
            write_json(validation, checkpoints([1.]*9))
            write_json(root/"native-rgb-cost-ledger.json", dict(charged_seconds=100, max_seconds=5400))
            with monitor_training(SERVER, root/"training", artifact_root=root) as state:
                self.assertTrue(state.done.wait(3))
                self.assertIsNone(state.last_error)
                self.assertTrue(state.evidence["stop_created"])
                self.assertFalse(state.thread.is_alive())
            evidence_path = root/"early-stopping.json"
            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(evidence["validation"]["sha256"], hashlib.sha256(validation.read_bytes()).hexdigest())
            source = Path(evidence["source"])
            self.assertEqual(evidence["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            before_stop, before_evidence = (root/"STOP").read_bytes(), evidence_path.read_bytes()
            with monitor_training(SERVER, root/"training", artifact_root=root) as again:
                self.assertTrue(again.done.wait(3))
            self.assertEqual((root/"STOP").read_bytes(), before_stop)
            self.assertEqual(evidence_path.read_bytes(), before_evidence)
        with fixture() as root:
            write_json(root/"training/validation.json", checkpoints([1.]*9))
            (root/"STOP").write_text("user's existing STOP\n", encoding="utf-8")
            with monitor_training(SERVER, root/"training", artifact_root=root) as state:
                self.assertTrue(state.done.wait(3))
                self.assertFalse(state.evidence["stop_created"])
            self.assertEqual((root/"STOP").read_text(), "user's existing STOP\n")

    def test_broken_validation_cannot_block_the_budget_guard(self):
        with fixture() as root:
            path = root/"training/validation.json"
            path.parent.mkdir()
            path.write_text("{broken atomic-read fixture", encoding="utf-8")
            write_json(root/"native-rgb-cost-ledger.json", dict(charged_seconds=5360, max_seconds=5400))
            with monitor_training(SERVER, root/"training", artifact_root=root) as state:
                self.assertTrue(state.done.wait(3))
                self.assertEqual(state.evidence["decision"]["reason"], "checkpoint_time_reserve")
                self.assertIn("validation", state.evidence["read_errors"])

    def test_monitor_cannot_read_another_seed_ledger_and_shuts_down_cleanly(self):
        with fixture() as root:
            bad = dict(SERVER, ledger_path=str(root/"other-seed/native-rgb-cost-ledger.json"))
            with self.assertRaises(ValueError):
                with monitor_training(bad, root/"training", artifact_root=root):
                    pass
            with monitor_training(SERVER, root/"training", artifact_root=root) as state:
                self.assertTrue(state.enabled)
            self.assertFalse(state.thread.is_alive())
            self.assertFalse((root/"STOP").exists())


if __name__ == "__main__":
    unittest.main()
