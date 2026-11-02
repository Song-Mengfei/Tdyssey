"""Fixed geometry, score and attempted-cost checks without real physics."""

import math
import importlib.util
import unittest

from tdyssey.collection import reset_issues
from tdyssey.envs.official import official_shape
from tdyssey.evaluation import goal_metrics
from tdyssey.frozen_wm_protocol import anchors, calibration_states, candidate_actions, predicted_scores, rank_scores, wrapped_degrees
from tdyssey.geometry import Pose
from tdyssey.state import BodyState, Snapshot
from scripts.validate_frozen_wm import collect_future, stopped_step, unattempted_slots


class ProtocolChecks(unittest.TestCase):
    def test_recovery_excludes_completed_and_failed_slots_without_replacement(self):
        prior = {"branches": [{"anchor": "L00", "candidate": candidate, "status": "complete"}
                              for candidate in ("hold", "toward", "side", "away")] +
                             [{"anchor": "L01", "candidate": "hold", "status": "aborted"}]}
        remaining = unattempted_slots(prior)
        self.assertEqual(len(remaining), 59)
        self.assertNotIn(("L01", "hold"), remaining)
        self.assertIn(("L01", "toward"), remaining)

    def test_predeclared_geometry_targets_and_threshold_calibration(self):
        self.assertEqual(len(anchors()), 16)
        self.assertEqual(len(calibration_states()), 24)
        for row in anchors() + calibration_states():
            shape = official_shape(row["shape"])
            state = Snapshot(shape.name, shape.geometry_hash, "a" * 64, BodyState(row["pose"]),
                             BodyState(Pose(row["pusher"], 0)), row["goal"], 0)
            self.assertEqual([x for x in reset_issues(shape, state) if x != "already_near_goal"], [])
            if not row["id"].startswith("cal-"):
                self.assertLessEqual(goal_metrics(shape, row["pose"], row["goal"]).coverage, .95)
                for name in ("hold", "toward", "side", "away"):
                    self.assertTrue(all(0 <= v <= 511 for action in candidate_actions(row, name) for v in action))
        self.assertEqual(sum(abs(goal_metrics(official_shape(r["shape"]), r["pose"], r["goal"]).coverage-.95) <= .05
                             for r in calibration_states()), 8)

    def test_strict_predicted_stopping_and_missing_crossings(self):
        def sequence(values):
            return [{"valid": v is not None, "coverage": v} for v in values]
        self.assertEqual(predicted_scores(sequence([.95, .96, .4, .2])), [.95, .96, .96, .96])
        self.assertEqual(predicted_scores(sequence([.9, None, .99, .8])), [.9, None, None, None])
        self.assertEqual(predicted_scores(sequence([.96, None, .2, .1])), [.96] * 4)

    def test_candidate_ties_missing_and_uniform_baseline(self):
        row = rank_scores([0, .1, .2, .3], [0, .9e-6, 1.8e-6, 0])
        self.assertEqual(row["selected"], 1)
        self.assertAlmostEqual(row["uniform_regret"], .15)
        invalid = rank_scores([0, .1, .2, .3], [None, .1, .2, .3])
        self.assertTrue(invalid["selection_abstention"])
        self.assertEqual(invalid["eligible_pairs"], 6)
        self.assertEqual(invalid["correct_pairs"], 3)
        self.assertEqual(invalid["regret_conservative"], 1)

    def test_wrap(self):
        self.assertAlmostEqual(wrapped_degrees(math.radians(-358)), 2)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "installed numpy required")
    def test_attempt_reserved_before_step_and_failure_not_refunded(self):
        events = []
        class Budget:
            def reserve(self, **cost):
                events.append(("reserved", cost))
                return "1"
            def complete(self, operation, detail=None):
                events.append(("complete", operation))
            def abort(self, operation, detail=None):
                events.append(("abort", operation))
        class Env:
            def step(self, action):
                events.append(("step", list(action)))
                raise RuntimeError("fake failure")
        with self.assertRaises(RuntimeError):
            stopped_step(Env(), [1, 2], Budget())
        self.assertEqual(events[0], ("reserved", {"controls": 1, "substeps": 10}))
        self.assertEqual(events[-1], ("abort", "1"))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "installed numpy required")
    def test_success_absorbs_without_extra_controls_or_contact_samples(self):
        import numpy as np
        calls = []
        shape = official_shape("L")
        goal = Pose((250, 250), 0)
        state = Snapshot("L", shape.geometry_hash, "a" * 64, BodyState(goal),
                         BodyState(Pose((100, 100), 0)), goal, 1)
        class Budget:
            def reserve(self, **cost):
                calls.append(cost)
                return "1"
            def complete(self, operation, detail=None):
                pass
        class Env:
            def step(self, action):
                return {"image": np.zeros((3, 96, 96))}, 1, True, {"n_contacts": 2}
            def capture_state(self):
                return state
        steps, images = collect_future(Env(), [[200, 200]] * 4, Budget(), shape)
        self.assertEqual(len(calls), 1)
        self.assertEqual([row["absorbing"] for row in steps], [False, True, True, True])
        self.assertEqual([row["contacts"] for row in steps], [2, 0, 0, 0])
        self.assertIsNone(steps[-1]["action"])
        self.assertTrue(all(row["truth"]["coverage"] == 1 for row in steps))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "installed numpy required")
    def test_partial_future_survives_later_failed_control(self):
        import numpy as np
        shape = official_shape("L")
        state = Snapshot("L", shape.geometry_hash, "a" * 64, BodyState(Pose((150, 150), 0)),
                         BodyState(Pose((50, 50), 0)), Pose((350, 350), 0), 1)
        class Budget:
            def reserve(self, **cost):
                return "1"
            def complete(self, operation, detail=None):
                pass
            def abort(self, operation, detail=None):
                pass
        class Env:
            attempts = 0
            def step(self, action):
                self.attempts += 1
                if self.attempts == 2:
                    raise RuntimeError("second attempted step failed")
                return {"image": np.zeros((3, 96, 96))}, 0, False, {"n_contacts": 0}
            def capture_state(self):
                return state
        steps, frames, saved = [], [], []
        with self.assertRaises(RuntimeError):
            collect_future(Env(), [[200, 200]] * 4, Budget(), shape, steps, frames,
                           lambda: saved.append(len(steps)))
        self.assertEqual(len(steps), 1)
        self.assertEqual(len(frames), 1)
        self.assertEqual(saved, [1])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "installed torch/numpy required")
class FakeGPCChecks(unittest.TestCase):
    def test_raw_history_normalized_actions_rng_and_frozen_fake_network(self):
        import numpy as np
        import torch
        from tdyssey.gpc_world_model import GPCWorldModel
        if not torch.cuda.is_available():
            self.skipTest("Tiny synthetic CUDA fixture requires existing CUDA")
        seen = []
        class FakeSampler:
            def sample(self, images, actions):
                seen.append((images.clone(), actions.clone()))
                return torch.ones((1, 3, 96, 96), device="cuda") * 1.25, None
        model = GPCWorldModel.__new__(GPCWorldModel)
        model.device, model.sampler = torch.device("cuda"), FakeSampler()
        before = torch.cuda.get_rng_state().clone()
        images = np.zeros((4, 3, 96, 96), dtype=np.float32)
        past = [[0, 511], [10, 20], [30, 40]]
        future = [[50, 60], [70, 80]]
        raw = model.predict(images, past, future, seed=5478)
        self.assertEqual(float(raw.max()), 1.25)
        self.assertEqual(float(seen[1][0][0, -1].max()), 1.25)
        np.testing.assert_allclose(seen[0][1].cpu().numpy()[0], np.asarray(past + future[:1])/511*2-1, atol=1e-7)
        self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
