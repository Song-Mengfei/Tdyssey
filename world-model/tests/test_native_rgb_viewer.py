"""Mock-only click viewer checks; no real simulator controls, windows or GPU."""

from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.native_rgb_viewer import (
    LEFT, NativeRGBSession, allowed_letters, click_target, main, validate_frozen_viewer_protocol,
)
from tdyssey.native_rgb_env import CONTROLLER
from tdyssey.native_rgb_evaluation import TRAIN_LETTERS


class FakeEnv:
    controller = CONTROLLER
    def __init__(self):
        self.actions, self.closed = [], False
        self.counter = 0
    def image(self):
        image = np.ones((3,96,96), dtype=np.float32)
        image[0,0,0] = self.counter/100
        return image
    def reset_scene(self):
        self.counter = 0
        return self.image()
    def state(self):
        return np.asarray([50,60]+[0]*13, dtype=np.float64)
    def step(self, target):
        self.actions.append(np.asarray(target).copy())
        self.counter += 1
        return self.image(), dict(valid=True, contact=0)
    def close(self):
        self.closed = True


class FakePredictor:
    def __init__(self):
        self.calls, self.closed = [], 0
    def __call__(self, history, actions, seed):
        self.calls.append((history.copy(), actions.copy(), seed))
        image = history[-1].copy()
        image[0,0,0] = .987
        return image
    def close(self):
        self.closed += 1


class NativeRGBViewerTests(unittest.TestCase):
    def test_three_actual_holds_then_one_click_with_clean_history_and_raw_output(self):
        env, predictor = FakeEnv(), FakePredictor()
        session = NativeRGBSession(lambda _: env, predictor, max_controls=8)
        self.assertEqual(session.controls, 3)
        self.assertEqual(len(env.actions), 3)
        np.testing.assert_array_equal(np.stack(env.actions), [[50,60]]*3)
        self.assertEqual(len(predictor.calls), 0)
        session.click_and_step((300,200))
        self.assertEqual(session.controls, 4)
        self.assertEqual(len(env.actions), 4)
        inputs, actions, _ = predictor.calls[0]
        np.testing.assert_allclose(inputs[:,0,0,0], [0,.01,.02,.03])
        np.testing.assert_array_equal(actions, [[50,60]]*3+[[300,200]])
        self.assertAlmostEqual(float(session.last_prediction[0,0,0]), .987, places=6)
        self.assertAlmostEqual(float(session.last_actual[0,0,0]), .04, places=6)
        session.click_and_step((200,200))
        # Next click uses four physical observations, not the prior prediction.
        np.testing.assert_allclose(predictor.calls[1][0][:,0,0,0], [.01,.02,.03,.04])
        self.assertFalse(any(np.all(frame[:,0,0] == [1,0,0]) for frame in session.history))
        session.close()
        session.close()
        self.assertTrue(env.closed)
        self.assertEqual(predictor.closed, 1)

    def test_prediction_failure_executes_no_click_control(self):
        env = FakeEnv()
        class Broken(FakePredictor):
            def __call__(self, *args):
                raise RuntimeError("mock inference failure")
        session = NativeRGBSession(lambda _: env, Broken())
        with self.assertRaisesRegex(RuntimeError, "inference"):
            session.click_and_step((200,200))
        self.assertEqual(len(env.actions), 3)
        self.assertEqual(session.controls, 3)
        self.assertTrue(session.blocked)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            session.click_and_step((200,200))

    def test_budget_checks_precede_model_or_extra_physics(self):
        env, predictor = FakeEnv(), FakePredictor()
        session = NativeRGBSession(lambda _: env, predictor, max_controls=4)
        session.click_and_step((200,200))
        with self.assertRaisesRegex(RuntimeError, "allowance"):
            session.click_and_step((300,200))
        with self.assertRaisesRegex(RuntimeError, "allowance"):
            session.reset()
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(len(env.actions), 4)

    def test_unseen_is_rejected_before_environment_or_predictor_call(self):
        with self.assertRaisesRegex(ValueError, "sealed"):
            NativeRGBSession(lambda _: self.fail("No environment may be created"), FakePredictor(), letter="D")
        self.assertEqual(set(allowed_letters()), set(TRAIN_LETTERS))
        with self.assertRaisesRegex(ValueError, "frozen"):
            allowed_letters(server=True, allow_unseen=True, frozen_validated=False)
        self.assertEqual(len(allowed_letters(server=True, allow_unseen=True, frozen_validated=True)), 26)

    def test_click_coordinates_only_use_actual_panel(self):
        self.assertEqual(click_target(LEFT[:2]), (0.,0.))
        self.assertEqual(click_target((LEFT[0]+LEFT[2]-1,LEFT[1]+LEFT[3]-1)), (510.6666666666667,510.6666666666667))
        self.assertIsNone(click_target((LEFT[0]+LEFT[2],LEFT[1])))
        self.assertIsNone(click_target((500,200)))

    def test_wrong_controller_and_invalid_target_do_not_execute_controls(self):
        env = FakeEnv()
        env.controller = "upstream-kinematic"
        with self.assertRaisesRegex(ValueError, "controller"):
            NativeRGBSession(lambda _: env, FakePredictor())
        self.assertEqual(env.actions, [])
        env.controller = CONTROLLER
        predictor = FakePredictor()
        session = NativeRGBSession(lambda _: env, predictor)
        with self.assertRaisesRegex(ValueError, "absolute"):
            session.click_and_step((np.nan,200))
        self.assertEqual(len(env.actions), 3)
        self.assertEqual(len(predictor.calls), 0)

    def test_default_cli_is_plan_only_and_opens_no_model_or_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"not-created"
            with patch("tdyssey.native_rgb_evaluation.NativePredictor", side_effect=AssertionError("GPU must not load")), \
                 patch("tdyssey.native_rgb_env.make_env", side_effect=AssertionError("Physics must not run")), \
                 redirect_stdout(io.StringIO()):
                result = main(["--physics-root","unused","--gpc-root","unused",
                    "--init-checkpoint","unused","--checkpoint","unused","--output",str(output)])
            self.assertEqual(result["status"], "plan_only")
            self.assertEqual(result["real_controls"], 0)
            self.assertFalse(output.exists())

    def test_full_cli_seal_and_direct_core_identity_are_both_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/"best.pt";checkpoint.write_bytes(b"mock checkpoint")
            expected=dict(kind="native-gpc-rgb-final-protocol.v1",checkpoint_sha256="core-mock")
            seal=dict(kind="native-gpc-rgb-frozen.v1",checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      evaluation_identity=expected)
            # No physical identity is necessary for a raw-only viewer.
            with patch("tdyssey.native_rgb_evaluation.final_protocol_identity",return_value=expected):
                self.assertTrue(validate_frozen_viewer_protocol(dict(server=True),checkpoint,"unused","unused",seal))
                self.assertTrue(validate_frozen_viewer_protocol(dict(server=True),checkpoint,"unused","unused",expected))
            with self.assertRaisesRegex(ValueError,"local"):
                validate_frozen_viewer_protocol(dict(server=False),checkpoint,"unused","unused",seal)

    def test_full_seal_outer_checkpoint_and_nested_identity_are_both_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/"best.pt";checkpoint.write_bytes(b"mock checkpoint")
            expected=dict(kind="native-gpc-rgb-final-protocol.v1",identity="expected")
            seal=dict(kind="native-gpc-rgb-frozen.v1",checkpoint_sha256="wrong",evaluation_identity=expected)
            with patch("tdyssey.native_rgb_evaluation.final_protocol_identity",return_value=expected) as identity:
                with self.assertRaisesRegex(ValueError,"checkpoint SHA"):
                    validate_frozen_viewer_protocol(dict(server=True),checkpoint,"unused","unused",seal)
                identity.assert_not_called()
                seal.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            evaluation_identity=dict(expected,identity="changed"))
                with self.assertRaisesRegex(ValueError,"core frozen"):
                    validate_frozen_viewer_protocol(dict(server=True),checkpoint,"unused","unused",seal)

    def test_full_seal_cli_validation_is_still_plan_only_without_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/"best.pt";checkpoint.write_bytes(b"mock checkpoint")
            expected=dict(kind="native-gpc-rgb-final-protocol.v1",identity="expected")
            seal=root/"frozen-model.json"
            seal.write_text(json.dumps(dict(kind="native-gpc-rgb-frozen.v1",
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),evaluation_identity=expected)),encoding="utf-8")
            with patch("scripts.native_rgb_world_model.load_config",return_value=dict(server=True)), \
                 patch("tdyssey.native_rgb_evaluation.final_protocol_identity",return_value=expected), \
                 patch("tdyssey.native_rgb_evaluation.NativePredictor",side_effect=AssertionError("No GPU model may load")), \
                 patch("tdyssey.native_rgb_env.make_env",side_effect=AssertionError("No controls may execute")), \
                 redirect_stdout(io.StringIO()):
                result=main(["--physics-root","unused","--gpc-root","unused","--init-checkpoint","unused",
                    "--checkpoint",str(checkpoint),"--server","--allow-unseen","--letter","D",
                    "--config","unused","--data","unused","--calibration","unused","--frozen-protocol",str(seal)])
            self.assertEqual(result["status"],"plan_only")
            self.assertEqual(result["real_controls"],0)
            self.assertEqual(len(result["allowed_letters"]),26)


if __name__ == "__main__":
    unittest.main()
