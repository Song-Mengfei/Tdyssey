"""Three actual-predictor accounting regressions; synthetic NumPy/markers only.

TDYSSEY_FOUNDATION_MANUAL_SOURCE selects the sealed before-source fixture. The
old implementation must fail these behavioral checks, not merely an import or
missing-method check. No real Torch/SDK, weights, model, decoder or physics runs.
Root owns execution in the separately bounded before/after commands.
"""
from contextlib import ExitStack, nullcontext
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


def manual_source():
    path = os.environ.get("TDYSSEY_FOUNDATION_MANUAL_SOURCE")
    if not path:
        from tdyssey import manual_world_model
        return manual_world_model
    name = "tdyssey._foundation_baseline_manual"
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the explicitly selected manual source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manual = manual_source()


class ManualPredictionUsageTests(unittest.TestCase):
    def fixture(self, *, horizon=4, fail_sample=None, fail_decode=False):
        from tdyssey import wm_image_measurement_v2 as measurement
        from tdyssey import wm_compat_engine as engine
        request = dict(role="frozen", shape="L", horizon=horizon, noise_seed=5478,
            images=np.full((4, 3, 96, 96), .8, dtype=np.float32),
            past_actions=np.asarray([[0, 511], [100, 200], [220, 330]], dtype=np.float32),
            target=np.asarray([410, 125], dtype=np.float64), goal=object())
        calls, outputs = [], []

        class Sampler:
            def seed(self, value):
                if value != 5478:raise AssertionError("Fixed noise seed changed")
            def sample(self, context, actions):
                calls.append((context.copy(), actions.copy()))
                if len(calls) == fail_sample:
                    raise RuntimeError("sampler fixture H2")
                value = np.full((3, 96, 96), -.1*len(calls), dtype=np.float32)
                outputs.append(value.copy())
                return value

        torch = ModuleType("torch")
        torch.random = SimpleNamespace(fork_rng=lambda **kwargs: nullcontext())
        torch.cuda = SimpleNamespace(current_device=lambda: 0)
        torch.inference_mode = nullcontext
        predictor = manual.PinnedPredictor()
        predictor.load = Mock()
        predictor.sampler = Sampler()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(sys.modules, {"torch": torch}))
        stack.enter_context(patch.object(measurement, "decode",
            side_effect=RuntimeError("decoder fixture") if fail_decode else None,
            return_value={"marker": True}))
        stack.enter_context(patch.object(engine, "compact_measurement", return_value={"marker": True}))
        return predictor, request, calls, outputs

    def assert_usage(self, predictor, *, reservations, returns, euler):
        self.assertEqual(getattr(predictor, "forecast_reservations", 0), reservations)
        self.assertEqual(getattr(predictor, "forecast_returns", 0), returns)
        self.assertEqual(getattr(predictor, "euler_reservations", 0), euler)

    def assert_retained(self, predictor, outputs, calls, *, trace_rows):
        partial = getattr(predictor, "last_partial_predictions", ())
        self.assertEqual(len(partial), len(outputs))
        for retained, expected in zip(partial, outputs):
            np.testing.assert_array_equal(retained, expected)
        trace = getattr(predictor, "last_trace", ())
        self.assertEqual(len(trace), trace_rows)
        # Hash names are optional observational metadata, not new computations
        # required from the production predictor. If present, they must describe
        # the exact sampler inputs rather than clipped display arrays.
        for row, (context, actions) in zip(trace, calls):
            for field, value in (("input_sha256", context), ("action_sha256", actions)):
                if field in row:
                    self.assertEqual(row[field], hashlib.sha256(value.tobytes()).hexdigest())

    def test_h2_sampler_failure_retains_two_reservations_one_return_six_euler(self):
        predictor, request, calls, outputs = self.fixture(fail_sample=2)
        with self.assertRaisesRegex(RuntimeError, "sampler fixture H2"):
            predictor.predict(request)
        self.assertEqual(len(calls), 2)
        self.assert_usage(predictor, reservations=2, returns=1, euler=6)
        self.assert_retained(predictor, outputs, calls, trace_rows=2)

    def test_decoder_failure_retains_all_four_returned_predictions_and_twelve_euler(self):
        predictor, request, calls, outputs = self.fixture(fail_decode=True)
        with self.assertRaisesRegex(RuntimeError, "decoder fixture"):
            predictor.predict(request)
        self.assertEqual(len(calls), 4)
        self.assert_usage(predictor, reservations=4, returns=4, euler=12)
        self.assert_retained(predictor, outputs, calls, trace_rows=4)

    def test_forecast_cap_rejects_before_the_extra_call_without_extra_charge(self):
        predictor, request, calls, outputs = self.fixture(horizon=2)
        # Assignment works on the actual old class too: the before-source then
        # visibly makes the prohibited second call instead of merely lacking a
        # new constructor argument. New source must enforce this same cap.
        predictor.forecast_cap = 1
        with self.assertRaises((RuntimeError, ValueError, MemoryError)):
            predictor.predict(request)
        self.assertEqual(len(calls), 1)
        self.assert_usage(predictor, reservations=1, returns=1, euler=3)
        partial = getattr(predictor, "last_partial_predictions", ())
        self.assertEqual(len(partial), 1)
        np.testing.assert_array_equal(partial[0], outputs[0])


if __name__ == "__main__":unittest.main()
