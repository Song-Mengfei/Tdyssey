"""Focused stage3B CPU contracts; no real model, optimizer or CUDA work."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.native_rgb_stage3_prepare import batch_plan
from scripts import native_rgb_stage3b_run as entry
from tdyssey import native_rgb_training as training
from tdyssey.native_rgb_stage_budget import ModelWorkBudget, UpdateLimitReached

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/native-rgb-supervised-20261029/stage3b"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Parameter:
    requires_grad = True

    def numel(self):
        return 8729827


class Model:
    def parameters(self):
        return [Parameter()]


class Budget:
    elapsed = 1.
    baseline_seconds = None

    def require_operation(self, *args):
        pass


class Stage3BEntryTests(unittest.TestCase):
    def setUp(self):
        groups = {name: [{"window_id": f"{name}:{i}"} for i in range(n)]
                  for name, n in [("train18", 18), ("dev36", 36)]}
        self.plans = batch_plan(groups)
        self.truth = {
            name: dict(
                histories=np.full((len(rows), 4, 3, 96, 96), .25, np.float32),
                actions_absolute=np.full((len(rows), 4, 2), 200., np.float32),
                window_ids=np.array([row["window_id"] for row in rows]))
            for name, rows in groups.items()}

    def invoke(self, root, predict):
        (root / "paired-batches.json").write_text(json.dumps(self.plans), encoding="utf-8")
        context = dict(model=Model(), source=object(), check=lambda: None,
                       budget=Budget(), progress={"updates": 0})
        with patch.object(entry, "_parameter_digest", return_value=("digest", {"p": "sha"})), \
             patch("tdyssey.native_rgb_evaluation.native_sampler_batch_predictor", return_value=predict), \
             redirect_stdout(io.StringIO()):
            return entry.prediction_hook(root, "frozen", self.truth, self.plans)(context)

    def test_finite_both_sides_of_range_remain_raw_failure_status_and_all_batches_continue(self):
        calls = []

        def predict(history, actions, *, batch_seed):
            self.assertEqual(history.shape[1:], (4, 3, 96, 96))
            self.assertTrue((actions == 200).all())
            values = np.full((len(actions), 3, 96, 96), .12345, np.float32)
            if not calls:
                values[0, 0, 0, 0] = -.02
                values[1, 0, 0, 0] = 1.125
            calls.append((len(actions), batch_seed))
            return values

        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            root = Path(directory)
            result = self.invoke(root, predict)
            expected = [(batch["batch_size"], batch["batch_seed"])
                        for cohort in ["train18", "dev36"] for batch in self.plans["cohorts"][cohort]]
            self.assertEqual(calls, expected)
            self.assertEqual(result["native_prediction_frames"], 54)
            self.assertEqual(result["cohorts"]["train18"]["ok"], 16)
            self.assertEqual(result["cohorts"]["dev36"]["ok"], 36)
            with np.load(root / "predictions/frozen-train18.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["prediction"][0, 0, 0, 0], np.float32(-.02))
                self.assertEqual(arrays["prediction"][1, 0, 0, 0], np.float32(1.125))
                self.assertEqual(arrays["status"][:4].tolist(), ["out_of_range_native_RGB"] * 2 + ["ok"] * 2)
                self.assertTrue(np.isfinite(arrays["prediction"]).all())
            metadata = read_json(root / "predictions/frozen-train18.json")
            self.assertIsNone(metadata["failure"])
            self.assertFalse(metadata["postprocessing"])
            self.assertEqual(metadata["range_rows"][0]["negative_values"], 1)
            self.assertEqual(metadata["range_rows"][1]["above_one_values"], 1)

    def test_nonfinite_still_aborts_and_retains_partial_raw_denominator(self):
        calls = []

        def predict(history, actions, *, batch_seed):
            calls.append(batch_seed)
            values = np.full((len(actions), 3, 96, 96), .25, np.float32)
            values[0, 0, 0, 0] = np.nan
            return values

        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            root = Path(directory)
            with self.assertRaises(FloatingPointError):
                self.invoke(root, predict)
            self.assertEqual(len(calls), 1)
            self.assertFalse((root / "predictions/frozen-dev36.npz").exists())
            with np.load(root / "predictions/frozen-train18.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["prediction"].shape, (18, 3, 96, 96))
                self.assertTrue(np.isnan(arrays["prediction"][0, 0, 0, 0]))
                self.assertTrue((arrays["prediction"][1:4] == .25).all())
                self.assertTrue(np.isnan(arrays["prediction"][4:]).all())
                self.assertEqual(arrays["status"][:4].tolist(), ["nonfinite_native_RGB", "ok", "ok", "ok"])
                self.assertTrue((arrays["status"][4:] == "not_attempted").all())

    def test_wrong_shape_still_aborts_without_reshaping_or_fabricating_valid_rgb(self):
        def predict(history, actions, *, batch_seed):
            return np.zeros((len(actions), 96, 96, 3), np.float32)

        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "native sampler shape"):
                self.invoke(root, predict)
            with np.load(root / "predictions/frozen-train18.npz", allow_pickle=False) as arrays:
                self.assertTrue(np.isnan(arrays["prediction"]).all())
                self.assertTrue((arrays["status"][:4] == "wrong_native_output_shape").all())
                self.assertTrue((arrays["status"][4:] == "not_attempted").all())
            self.assertFalse((root / "predictions/frozen-dev36.npz").exists())

    def test_closed_prior_is_not_reset_and_explicit_residual_caps_conserve_600_seconds_200_attempts(self):
        prior = read_json(STAGE / "prior-ledger.json")
        residual = entry.validate_prior_ledger(deepcopy(prior), prior)
        self.assertEqual(residual, dict(max_attempts=200, max_seconds=600-prior["charged_seconds"],
                                       update_cutoff_seconds=420-prior["charged_seconds"]))
        for change in [lambda value: value.update(charged_seconds=0),
                       lambda value: value.update(active_stage={"token": "live"}),
                       lambda value: value["stages"][0].update(phase="different")]:
            value = deepcopy(prior)
            change(value)
            with self.assertRaises(RuntimeError):
                entry.validate_prior_ledger(value, prior)
        clock = type("Clock", (), {"now": 0., "__call__": lambda self: self.now})()
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory, \
             patch.object(training.time, "monotonic", clock):
            ledger_file = Path(directory) / "ledger.json"
            ledger_file.write_text(json.dumps(prior), encoding="utf-8")
            ledger = training.CostLedger(ledger_file, 3000, 5400.)
            budget = ModelWorkBudget(ledger.start("stage3b-visible-microfit"), **residual)
            self.assertEqual(budget.effective_time_cap, residual["max_seconds"])
            self.assertEqual(budget.update_cutoff, residual["update_cutoff_seconds"])
            for _ in range(200):
                budget.reserve_update()
                budget.complete_update()
            with self.assertRaises(UpdateLimitReached):
                budget.reserve_update()
            clock.now = residual["max_seconds"]
            with self.assertRaises(training.BudgetExceeded):
                budget.heartbeat()
            budget.finish("mock_residual_limit")
            value = ledger.snapshot()
            self.assertEqual(value["charged_seconds"], 600.)
            self.assertEqual((value["charged_updates"], value["completed_updates"]), (200, 200))
            self.assertEqual((value["max_updates"], value["max_seconds"]), (3000, 5400.))
            self.assertEqual(value["stages"][0], prior["stages"][0])
            self.assertEqual(value["historical_cancelled_experiment"], prior["historical_cancelled_experiment"])
            self.assertIsNone(value["active_stage"])

    def test_actual_small_input_copies_indices_batches_and_score_are_identical(self):
        reused = read_json(STAGE / "reused-inputs.json")
        self.assertEqual(len(reused["files"]), 8)
        self.assertEqual(sum(row["bytes"] for row in reused["files"]), 1181492)
        for row in reused["files"]:
            original, copied = Path(row["original"]), Path(row["copy"])
            for path in [original, copied]:
                self.assertEqual(path.stat().st_size, row["bytes"])
                with path.open("rb") as stream:
                    self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(), row["sha256"])
        old = read_json(ROOT / "configs/native_rgb_supervised_stage3.json")
        new = read_json(ROOT / "configs/native_rgb_supervised_stage3b.json")
        for key in ["train_indices", "validation_indices", "sampler", "seed", "microbatch", "effective_batch",
                    "lr", "weight_decay", "amp", "max_updates", "max_train_seconds", "score_protocol_sha256"]:
            self.assertEqual(new[key], old[key], key)
        self.assertEqual((len(new["train_indices"]), len(new["validation_indices"])), (18, 36))
        batches = read_json(STAGE / "paired-batches.json")["cohorts"]
        self.assertEqual([batch["batch_size"] for batch in batches["train18"]], [4, 4, 4, 4, 2])
        self.assertEqual([batch["batch_size"] for batch in batches["dev36"]], [4] * 9)


if __name__ == "__main__":
    unittest.main()
