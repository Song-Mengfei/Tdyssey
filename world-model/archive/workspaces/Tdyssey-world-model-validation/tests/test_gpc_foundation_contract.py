"""Actual wrapper helpers with tiny NumPy arrays and an ordinary Torch marker.

No real Torch/SDK, weights, network, simulator, decoder or subprocess is used.
These tests do not establish learned accuracy or native CUDA parity.
"""
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tdyssey import manual_world_model as manual
from tdyssey.gpc_adaptation import NativeSampler
from tdyssey.gpc_world_model import action_windows


class MarkerTensor:
    def __init__(self, value):self.value = np.array(value, copy=True)
    def __getitem__(self, key):return MarkerTensor(self.value[key])
    def cpu(self):return self
    def numpy(self):return self.value.copy()


class FoundationContractTests(unittest.TestCase):
    def test_actual_manual_loop_has_four_real_transforms_raw_feedback_and_next_action(self):
        from tdyssey import wm_image_measurement_v2 as measurement
        from tdyssey import wm_compat_engine as engine
        images = np.full((4, 3, 96, 96), .8, dtype=np.float32)
        for index in range(4):images[index, :, 20:24, 30+index:34+index] = .25
        past = np.asarray([[0, 511], [100, 200], [220, 330]], dtype=np.float32)
        target = np.asarray([410, 125], dtype=np.float64)
        request = dict(role="frozen", images=images, past_actions=past, target=target,
                       horizon=4, noise_seed=5478, shape="L", goal=object())
        before = images.tobytes(), past.tobytes(), target.tobytes()
        calls, generated, transformed, scopes = [], [], [], []
        actual_transform = manual.transform_t1

        def transform(frame):
            value = actual_transform(frame)
            transformed.append(value[0].copy())
            return value

        class Sampler:
            def seed(self, value):scopes.append(("seed", value))
            def sample(self, context, actions):
                calls.append((context.copy(), actions.copy()))
                value = np.full((3, 96, 96), -.25*(len(calls)), dtype=np.float32)
                value[1, 0, 0] = np.float32(1.25)
                generated.append(value.copy())
                return value

        @contextmanager
        def fork_rng(*, devices):
            scopes.append(("fork_enter", devices))
            try:yield
            finally:scopes.append(("fork_exit", devices))

        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(current_device=lambda: 7)
        torch.random = SimpleNamespace(fork_rng=fork_rng)
        torch.inference_mode = nullcontext
        predictor = manual.PinnedPredictor()
        predictor.load = Mock()
        predictor.sampler = Sampler()
        with patch.dict("sys.modules", {"torch": torch}), \
                patch.object(manual, "transform_t1", side_effect=transform), \
                patch.object(measurement, "decode", return_value={"marker": True}) as decode, \
                patch.object(engine, "compact_measurement", return_value={"compact": True}):
            raw, compact, _ = predictor.predict(request)

        predictor.load.assert_called_once_with("frozen")
        self.assertEqual(len(transformed), 4)  # No transformation of any generated frame.
        context = np.stack(transformed)
        windows = action_windows(past, [target]*4)
        for index, ((given, actions), window) in enumerate(zip(calls, windows)):
            np.testing.assert_array_equal(given, context)
            normalized = np.asarray(window, dtype=np.float32)/np.float32(511)*np.float32(2)-np.float32(1)
            np.testing.assert_array_equal(actions, normalized)
            self.assertEqual(actions.dtype, np.dtype("float32"))
            context = np.concatenate((context[1:], generated[index][None]), axis=0)
        np.testing.assert_array_equal(raw, np.stack(generated))
        self.assertLess(float(raw.min()), 0)
        self.assertGreater(float(raw.max()), 1)
        self.assertEqual(before, (images.tobytes(), past.tobytes(), target.tobytes()))
        np.testing.assert_array_equal(decode.call_args.args[0], generated[-1])
        self.assertEqual(decode.call_args.args[1:], ("L", request["goal"]))
        self.assertEqual(compact, {"compact": True})
        self.assertEqual(scopes, [("fork_enter", [7]), ("seed", 5478), ("fork_exit", [7])])

    def test_actual_native_sampler_wrapper_does_not_rescale_or_clip_inputs_outputs(self):
        context = np.full((4, 3, 96, 96), .75, dtype=np.float32)
        actions = np.asarray([[-1, 1], [-.5, .5], [0, 0], [.75, -.75]], dtype=np.float32)
        raw = np.full((1, 3, 96, 96), -.3, dtype=np.float32)
        raw[0, 2, 0, 0] = np.float32(1.2)
        order = []
        def sample(given, act):
            np.testing.assert_array_equal(given.value, context[None])
            np.testing.assert_array_equal(act.value, actions[None])
            order.append("native_sample")
            return MarkerTensor(raw), []
        torch = ModuleType("torch")
        torch.tensor = lambda value, *, device: MarkerTensor(value)
        torch.inference_mode = nullcontext
        torch.cuda = SimpleNamespace(synchronize=lambda device: order.append("synchronize"))
        wrapper = object.__new__(NativeSampler)
        wrapper.device = "marker-device"
        wrapper.sampler = SimpleNamespace(sample=sample)
        wrapper.check = lambda: order.append("check")
        with patch.dict("sys.modules", {"torch": torch}):
            actual = wrapper.sample(context, actions)
        np.testing.assert_array_equal(actual, raw[0])
        self.assertEqual(order, ["native_sample", "synchronize", "check"])

    def test_actual_manual_loader_uses_selected_role_state_and_serial_eval_only_models(self):
        from tdyssey import gpc_adaptation as adaptation
        from tdyssey import gpc_evaluation_assets as assets_api
        from tdyssey import gpc_supervised_adaptation as objective
        assets = dict(weights={role: dict(path="marker-"+role, sha256=role)
                               for role in ("frozen", "adapted")})
        states = {role: {"marker_weight": role} for role in ("frozen", "adapted")}
        loaded, constructed = [], []
        predictor = manual.PinnedPredictor()
        torch = ModuleType("torch")
        torch.set_num_threads = Mock()
        torch.cuda = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False)

        class Model:
            def eval(self):self.evaluating = True;return self
            def requires_grad_(self, value):
                if value is not False:raise AssertionError("evaluation cannot enable gradients")
                self.frozen = True;return self

        def state_loader(role, supplied_assets):
            self.assertIs(supplied_assets, assets)
            self.assertIsNone(predictor.model)  # Previous model was released first.
            loaded.append(role)
            return states[role]

        def factory(root, checkpoint, device, config, *, state_dict):
            role = state_dict["marker_weight"]
            self.assertEqual(root, Path("marker-source"))
            self.assertEqual(checkpoint, "marker-frozen")
            self.assertEqual(device, "cuda")
            self.assertIs(state_dict, states[role])
            self.assertEqual(config, dict(execute=True, evaluation_only=True,
                validated_checkpoint_identity=assets["weights"][role]))
            constructed.append(role)
            return Model()

        with patch.dict("sys.modules", {"torch": torch}), \
                patch.object(assets_api, "verify_evaluation_assets", return_value=assets) as verify, \
                patch.object(assets_api, "load_model_only_state", side_effect=state_loader), \
                patch.object(objective, "make_real_denoiser", side_effect=factory), \
                patch.object(adaptation, "NativeSampler", side_effect=lambda model, check: object()), \
                patch.object(manual, "source_path", return_value=Path("marker-source")), \
                patch.object(predictor, "check_cuda"):
            predictor.load("frozen")
            self.assertTrue(predictor.model.frozen and predictor.model.evaluating)
            predictor.load("frozen")
            predictor.load("adapted")
            self.assertTrue(predictor.model.frozen and predictor.model.evaluating)
            self.assertEqual(predictor.loads, 2)
            predictor.close()
        self.assertEqual(loaded, ["frozen", "adapted"])
        self.assertEqual(constructed, loaded)
        verify.assert_called_once_with(manual.ROOT)
        self.assertIsNone(predictor.model)
        self.assertIsNone(predictor.sampler)


if __name__ == "__main__":unittest.main()
