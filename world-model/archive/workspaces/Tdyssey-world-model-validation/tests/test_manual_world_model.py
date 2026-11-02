"""Interaction invariants, without model imports or physical stepping."""
import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from tdyssey.manual_world_model import ManualSession, load_case_bundle, request_from_case, save_case_bundle
from tdyssey.geometry import Pose
from tdyssey.state import BodyState, Snapshot

EVIDENCE = Path(__file__).resolve().parents[1] / "artifacts/gpc-foundation-repair-20261026/case-fixtures"


class FakeEnv:
    def __init__(self, shape="T"):
        self.steps = 0
        self.done_at = 100
        self.shape = shape
        self.last_action = None

    def observation(self):
        return dict(image=np.full((3, 96, 96), self.steps / 255, dtype=np.float32),
                    agent_pos=np.array([256, 405]))

    def step(self, action):
        self.steps += 1
        self.last_action = tuple(float(value) for value in action)
        return self.observation(), 0., self.steps >= self.done_at, dict(n_contacts=int(self.steps % 2 == 0))

    def capture_state(self):
        return Snapshot(shape=self.shape, geometry_hash="0"*64, physics_hash="1"*64,
            goal=Pose((256, 256), 0), object_body=BodyState(Pose((256, 240), 0)),
            pusher_body=BodyState(Pose((256, 405), 0)), control_step=self.steps,
            pusher_target=self.last_action)

    def close(self):
        pass


def fake_environment(shape):
    env = FakeEnv(shape)
    return env, env.observation()


class ManualTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        self.session = ManualSession(fake_environment)
        self.session.select([250, 330])

    def tearDown(self):
        self.session.close()

    def prediction(self):
        req = self.session.request()
        raw = np.stack([req["images"][-1]] * req["horizon"])
        return req, raw

    def test_real_history_actions_and_preview_do_not_step(self):
        self.assertEqual(self.session.controls, 3)
        req, raw = self.prediction()
        np.testing.assert_allclose(req["images"][:, 0, 0, 0], np.arange(4) / 255)
        np.testing.assert_array_equal(req["past_actions"], [[256, 405]] * 3)
        self.session.accept(req, raw, {})
        self.assertEqual(self.session.env.steps, 3)
        self.session.execute()
        self.assertEqual(self.session.env.steps, 4)
        np.testing.assert_array_equal(self.session.actions[-1], [250, 330])
        with self.assertRaises(ValueError):
            self.session.execute()

    def test_target_model_horizon_and_reset_invalidate_preview(self):
        for change in (lambda: self.session.select([251, 330]),
                       lambda: self.session.configure(role="adapted"),
                       lambda: self.session.configure(horizon=4),
                       lambda: self.session.reset("L")):
            self.session.select([250, 330])
            req, raw = self.prediction()
            self.session.accept(req, raw, {})
            change()
            with self.assertRaises(ValueError):
                self.session.execute()
            with self.assertRaises(ValueError):
                self.session.accept(req, raw, {})

    def test_success_stops_before_full_horizon_and_matches_actual_index(self):
        self.session.configure(horizon=4)
        self.session.env.done_at = 5
        req, raw = self.prediction()
        self.session.accept(req, raw, {})
        result = self.session.execute()
        self.assertEqual(result["executed_controls"], 2)
        self.assertEqual(result["matched_horizon"], 2)
        self.assertTrue(result["truth_success"])
        self.assertIsNone(self.session.measurement)
        with self.assertRaises(ValueError):
            self.session.request()

    def test_case_save_after_execute_retains_full_request_and_real_prefix_without_calls(self):
        self.session.configure(horizon=4)
        self.session.env.done_at = 5
        request, raw = self.prediction()
        self.session.accept(request, raw, {"state": {"valid": False, "reason": "synthetic"}})
        result = self.session.execute()
        self.assertIsNone(self.session.pending)
        self.assertIsNotNone(self.session.last_accepted_request)
        self.assertEqual(result["executed_controls"], 2)
        controls = self.session.controls
        with tempfile.TemporaryDirectory(dir=EVIDENCE, prefix="after-execute-") as temporary:
            directory = Path(temporary) / "case"
            self.assertTrue(directory.resolve().is_relative_to(EVIDENCE.resolve()))
            with patch.object(self.session.env, "step", side_effect=AssertionError("save must not step")), \
                    patch.object(self.session.env, "capture_state", side_effect=AssertionError("save uses captured state")):
                saved = self.session.save_case(directory)
                bundle = load_case_bundle(directory)
                replay = request_from_case(bundle, role="adapted")
            self.assertEqual(self.session.controls, controls)
            self.assertEqual(saved["request_key"], request["request_key"])
            self.assertEqual(replay["request_key"], request["request_key"])
            self.assertEqual(replay["role"], "adapted")
            self.assertEqual(replay["initial_state"], request["initial_state"])
            self.assertEqual(len(replay["history_states"]), 4)
            self.assertEqual([state["control_step"] for state in replay["history_states"]], [0, 1, 2, 3])
            self.assertEqual([step["control_step"] for step in replay["history_steps"]], [1, 2, 3])
            np.testing.assert_array_equal(replay["images"], request["images"])
            np.testing.assert_array_equal(replay["past_actions"], request["past_actions"])
            np.testing.assert_array_equal(bundle["arrays"]["prediction_000"], raw)
            self.assertEqual(bundle["arrays"]["actual_raw"].shape, (2, 3, 96, 96))
            np.testing.assert_allclose(bundle["arrays"]["actual_raw"][:, 0, 0, 0], [4/255, 5/255])
            np.testing.assert_array_equal(bundle["arrays"]["actual_t1"], self.session.actual)
            steps = bundle["metadata"]["execution"]["steps"]
            self.assertEqual([step["control_step"] for step in steps], [4, 5])
            self.assertEqual([step["contacts"] for step in steps], [1, 0])
            self.assertEqual(steps[0]["before_state"], request["initial_state"])
            self.assertEqual(steps[0]["after_state"], steps[1]["before_state"])
            self.assertFalse(bundle["metadata"]["live_request_eligible"])
            before = (directory / "case.json").read_bytes()
            with self.assertRaises(FileExistsError): self.session.save_case(directory)
            self.assertEqual((directory / "case.json").read_bytes(), before)

    def test_weight_switch_preserves_same_request_comparisons_without_inference_or_physics(self):
        self.session.configure(horizon=4)
        frozen, raw_frozen = self.prediction()
        self.session.accept(frozen, raw_frozen, {"role": "frozen"})
        steps = self.session.env.steps
        self.session.configure(role="adapted")
        self.assertIsNone(self.session.pending)
        self.assertEqual(set(self.session.comparison_predictions()), {"frozen"})
        adapted, _ = self.prediction()
        self.assertEqual(adapted["request_key"], frozen["request_key"])
        raw_adapted = raw_frozen + np.float32(.01)
        self.session.accept(adapted, raw_adapted, {"role": "adapted"})
        self.assertEqual(self.session.env.steps, steps)
        comparison = self.session.comparison_predictions()
        self.assertEqual(set(comparison), {"frozen", "adapted"})
        comparison["frozen"]["raw"][:] = 99
        np.testing.assert_array_equal(self.session.comparison_predictions()["frozen"]["raw"], raw_frozen)
        self.session.execute()
        self.session.select([251, 330])
        current = self.session.request()
        self.assertNotEqual(current["request_key"], frozen["request_key"])
        retained = self.session.capture_case(frozen["request_key"])
        self.assertEqual(len(retained["metadata"]["predictions"]), 2)
        self.assertEqual(retained["metadata"]["execution"]["role"], "adapted")
        self.assertEqual(retained["metadata"]["execution"]["prediction_index"], 1)
        self.assertEqual(request_from_case(retained)["role"], "adapted")
        np.testing.assert_array_equal(retained["arrays"]["prediction_000"], raw_frozen)
        np.testing.assert_array_equal(retained["arrays"]["prediction_001"], raw_adapted)

    def test_mutated_or_obsolete_fixed_request_is_rejected_before_acceptance(self):
        request, raw = self.prediction()
        request["images"][0, 0, 0, 0] = .25
        with self.assertRaises(ValueError): self.session.accept(request, raw, {})
        self.assertEqual(self.session.forecasts, 0)
        self.assertIsNone(self.session.pending)
        request, raw = self.prediction()
        request["initial_state"].pop("physics_hash")
        with self.assertRaises((TypeError, ValueError)): self.session.accept(request, raw, {})
        self.assertEqual(self.session.forecasts, 0)
        with self.assertRaises(ValueError): self.session.capture_case()

    def test_malformed_bundle_and_payload_identity_fail_without_hidden_replay(self):
        request, raw = self.prediction()
        self.session.accept(request, raw, {})
        original = self.session.capture_case()
        with tempfile.TemporaryDirectory(dir=EVIDENCE, prefix="invalid-case-") as temporary:
            bad_path = Path(temporary) / "bad"
            malformed = deepcopy(original)
            malformed["arrays"]["past_actions"] = np.zeros((2, 2), dtype=np.float32)
            with self.assertRaises(ValueError): save_case_bundle(bad_path, malformed)
            self.assertFalse(bad_path.exists())
            with self.assertRaises(ValueError): request_from_case(original, role="unknown")
            valid_path = Path(temporary) / "valid"
            save_case_bundle(valid_path, original)
            payload_path = valid_path / "arrays.npz"
            payload = payload_path.read_bytes()
            payload_path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
            with patch("numpy.load", side_effect=AssertionError("SHA must be checked before decode")):
                with self.assertRaises(ValueError): load_case_bundle(valid_path)
        self.assertEqual(self.session.env.steps, 3)

    def test_case_cache_is_bounded_and_pruning_preserves_executed_reference_and_runtime_metadata(self):
        request, raw = self.prediction()
        request.update(prediction_trace=[{"probe": -1}], prediction_usage={"probe": -1},
                       checkpoint_identity={"role": "frozen", "sha256": "f"*64})
        self.session.accept(request, raw, {})
        self.session.execute()
        actual = self.session.actual.copy()
        controls = self.session.env.steps
        original_key = request["request_key"]
        latest = {}
        for index in range(10):
            role = "frozen" if index % 2 == 0 else "adapted"
            comparison = self.session.request_for_comparison(role)
            comparison.update(prediction_trace=[{"probe": index}], prediction_usage={"probe": index},
                checkpoint_identity={"role": role, "sha256": ("f" if role == "frozen" else "a")*64})
            self.session.accept_comparison(comparison, raw + np.float32(index/100), {})
            latest[role] = index
            self.assertLessEqual(len(self.session._cases[original_key]["predictions"]), 3)
            self.assertIsNone(self.session.pending)
        self.assertEqual(self.session.env.steps, controls)
        np.testing.assert_array_equal(self.session.actual, actual)
        with self.assertRaises(ValueError): self.session.execute()
        bundle = self.session.capture_case()
        self.assertEqual(bundle["metadata"]["cache_policy"]["maximum_recent_cases"], 8)
        for name in ("prediction_trace", "prediction_usage", "checkpoint_identity"):
            self.assertNotIn(name, bundle["metadata"]["request"])
        executed = bundle["metadata"]["execution"]["prediction_index"]
        for index, prediction in enumerate(bundle["metadata"]["predictions"]):
            probe = -1 if index == executed else latest[prediction["role"]]
            self.assertEqual(prediction["prediction_trace"], [{"probe": probe}])
            self.assertEqual(prediction["prediction_usage"], {"probe": probe})
            self.assertEqual(prediction["checkpoint_identity"]["role"], prediction["role"])
        np.testing.assert_array_equal(bundle["arrays"][bundle["metadata"]["predictions"][executed]["array"]], raw)
        self.assertGreater(bundle["metadata"]["cache_policy"]["discarded_superseded_previews"], 0)
        for index in range(10):
            self.session.select([240+index, 330])
            fresh, fresh_raw = self.prediction()
            self.session.accept(fresh, fresh_raw, {})
        self.assertEqual(len(self.session._cases), 8)
        self.assertEqual(self.session.case_evictions, 3)
        self.assertEqual(self.session.env.steps, controls)
        with self.assertRaises(ValueError): self.session.capture_case(original_key)

    def test_comparison_reuses_archived_input_and_rejects_obsolete_or_changed_live_endpoint(self):
        self.session.configure(horizon=4)
        request, raw = self.prediction()
        self.session.accept(request, raw, {})
        self.session.execute()
        controls = self.session.env.steps
        comparison = self.session.request_for_comparison("adapted")
        self.assertEqual(comparison["request_key"], request["request_key"])
        self.assertEqual(comparison["initial_state"], request["initial_state"])
        np.testing.assert_array_equal(comparison["images"], request["images"])
        np.testing.assert_array_equal(comparison["past_actions"], request["past_actions"])
        self.assertEqual(self.session.env.steps, controls)
        self.assertIsNone(self.session.pending)
        self.session.select([251, 330])
        with self.assertRaises(ValueError): self.session.accept_comparison(comparison, raw, {})
        comparison = self.session.request_for_comparison("adapted")
        self.session._step([251, 330])  # explicit fake control, outside comparison APIs
        with self.assertRaises(ValueError): self.session.accept_comparison(comparison, raw, {})
        with self.assertRaises(ValueError): self.session.request_for_comparison("adapted")
        self.assertEqual(self.session.env.steps, controls+1)
        self.assertIsNone(self.session.pending)


if __name__ == "__main__":
    unittest.main()
