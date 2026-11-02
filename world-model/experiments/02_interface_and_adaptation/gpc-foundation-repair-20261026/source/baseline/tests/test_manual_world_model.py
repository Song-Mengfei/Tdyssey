"""Interaction invariants, without model imports or physical stepping."""
import unittest
from types import SimpleNamespace

import numpy as np

from tdyssey.manual_world_model import ManualSession
from tdyssey.geometry import Pose


class FakeEnv:
    def __init__(self):
        self.steps = 0
        self.done_at = 100

    def observation(self):
        return dict(image=np.full((3, 96, 96), self.steps / 255, dtype=np.float32),
                    agent_pos=np.array([256, 405]))

    def step(self, action):
        self.steps += 1
        return self.observation(), 0., self.steps >= self.done_at, dict(n_contacts=0)

    def capture_state(self):
        return SimpleNamespace(goal=Pose((256, 256), 0),
                               object_body=SimpleNamespace(pose=Pose((256, 240), 0)))

    def close(self):
        pass


def fake_environment(shape):
    env = FakeEnv()
    return env, env.observation()


class ManualTests(unittest.TestCase):
    def setUp(self):
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


if __name__ == "__main__":
    unittest.main()
