"""Opt-in D mid-contact replay checks; no checkpoint or training is involved."""

from dataclasses import replace
import os
from pathlib import Path
import unittest

from tdyssey.envs.official import make_official_shape_env_class, official_shape
from tdyssey.geometry import Pose
from tdyssey.official import load_official_source
from tdyssey.simulator_planning import PlannerConfig, SimulatorReranker, state_error
from tdyssey.state import BodyState, Snapshot


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_ROOT"), "Set TDYSSEY_OFFICIAL_ROOT for real D contact replay")
class SimulatorPlanningRuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                            "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
            os.environ.setdefault(name, value)
        import numpy as np
        cls.np = np
        original = make_official_shape_env_class(load_official_source(Path(os.environ["TDYSSEY_OFFICIAL_ROOT"])))

        class PhysicsOnly(original):
            # Rendering is omitted; the inherited controller/physics are unchanged.
            def _get_obs(self):
                return {}

            def step(self, action):
                return super().step(None if action is None else np.asarray(action, dtype=np.float64))

        cls.environment_class = PhysicsOnly

    def environment(self):
        env = self.environment_class(shape="D", legacy=True)
        env.seed(100000)
        env.reset()
        self.addCleanup(env.close)
        return env

    def test_history_branch_matches_uninterrupted_d_contact_and_is_order_independent(self):
        live = self.environment()
        rect = min(official_shape("D").components, key=lambda r: (r.x0, r.y0))
        start = (256+rect.x0-20, 256+rect.center[1])
        initial = replace(live.capture_state(), object_body=BodyState(Pose((256, 256), 0)),
                          pusher_body=BodyState(Pose(start, 0)), goal=Pose((400, 120), .5),
                          control_step=0, pusher_target=None)
        live.restore_state(initial)
        planner = SimulatorReranker(self.environment, initial,
                                    PlannerConfig(horizon=4, prefix_lengths=(4,), settle_steps=2,
                                                  include_hold=False))
        self.addCleanup(planner.close)
        contacts = 0
        for _ in range(8):
            action = (start[0]+65, start[1])
            _, _, done, info = live.step(action)
            self.assertFalse(done)
            contacts += info["n_contacts"]
            planner.record_action(action)
        self.assertGreater(contacts, 0)
        before = live.capture_state()
        hold = [before.pusher_body.pose.position]*4
        push = [(before.pusher_body.pose.position[0]+30, start[1])]*4
        decision = planner.choose([hold, push], before, 30)
        reverse = planner.choose([push, hold], before, 30)
        self.assertEqual(live.capture_state(), before)
        self.np.testing.assert_allclose(decision.actions, reverse.actions, rtol=0, atol=1e-7)
        for action, expected in zip(decision.actions[:3], decision.report["selected_predictions"][:3]):
            live.step(action)
            planner.record_action(action)
            self.assertLessEqual(state_error(live.capture_state(), Snapshot.from_dict(expected)), 1e-7)
        # Reconstruct a second planning start after more real contact controls.
        planner.choose([hold, push], live.capture_state(), 27)

    def test_physical_hold_decelerates_a_moving_pusher_without_teleporting(self):
        live = self.environment()
        initial = replace(live.capture_state(), object_body=BodyState(Pose((256, 256), 0)),
                          pusher_body=BodyState(Pose((100, 100), 0), velocity=(40, 0)),
                          goal=Pose((400, 120), .5), control_step=0, pusher_target=None)
        live.restore_state(initial)
        planner = SimulatorReranker(self.environment, initial,
                                    PlannerConfig(horizon=2, prefix_lengths=(2,), settle_steps=2))
        self.addCleanup(planner.close)
        decision = planner.choose([], initial, 4)
        self.assertEqual(decision.report["selected"]["kind"], "hold")
        self.assertEqual(decision.actions, ((100., 100.),)*4)
        first = Snapshot.from_dict(decision.report["selected_predictions"][0])
        self.assertGreater(first.pusher_body.pose.position[0], 100)
        # The PD controller may already reverse after overshooting its target;
        # hold must damp the motion, not force a zero velocity or fixed pose.
        self.assertGreater(abs(first.pusher_body.velocity[0]), 0)
        self.assertLess(abs(first.pusher_body.velocity[0]), 40)
        for action, expected in zip(decision.actions, decision.report["selected_predictions"]):
            live.step(action)
            planner.record_action(action)
            self.assertLessEqual(state_error(live.capture_state(), Snapshot.from_dict(expected)), 1e-7)


if __name__ == "__main__":
    unittest.main()
