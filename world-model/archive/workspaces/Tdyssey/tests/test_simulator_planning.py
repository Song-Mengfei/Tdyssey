"""Dependency-free planner protocol tests; fake dynamics are not physics evidence."""

from dataclasses import replace
import math
import time
import unittest
from unittest.mock import patch

from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.simulator_planning import PlannerConfig, SimulatorReranker, score_prediction, state_error
from tdyssey.state import BodyState, Snapshot


def initial_state(goal=(256, 256)):
    return Snapshot("D", official_shape("D").geometry_hash, "a"*64,
                    BodyState(Pose((220, 256), 0)), BodyState(Pose((220, 256), 0)),
                    Pose(goal, 0), 0)


class FakeEnv:
    """Hidden-cache dynamics distinguish history replay from cold restoration."""

    def __init__(self, state, *, cache_effect=0.0, success_x=None, pusher_fraction=1.):
        self.state, self.cache_effect, self.success_x = state, cache_effect, success_x
        self.pusher_fraction = pusher_fraction
        self.cache = 0
        self.closed = False

    def restore_state(self, state):
        self.state, self.cache = state, 0
        return {}

    def capture_state(self):
        return self.state

    def step(self, action):
        self.cache += 1
        x = action[0] + self.cache_effect*self.cache
        old_x = self.state.object_body.pose.position[0]
        pusher = tuple(old + self.pusher_fraction*(target-old)
                       for old, target in zip(self.state.pusher_body.pose.position, action))
        self.state = replace(self.state,
                             object_body=BodyState(Pose((x, action[1]), 0), ((x-old_x)*10, 0)),
                             pusher_body=BodyState(Pose(pusher, 0)),
                             pusher_target=action, control_step=self.state.control_step+1)
        done = self.success_x is not None and x == self.success_x
        return {}, float(done), done, {"n_contacts": 1}

    def close(self):
        self.closed = True


class SimulatorPlanningChecks(unittest.TestCase):
    def planner(self, state=None, **config):
        state = state or initial_state()
        planner = SimulatorReranker(lambda: FakeEnv(state), state,
                                    PlannerConfig(horizon=3, prefix_lengths=(3,), settle_steps=1, **config))
        self.addCleanup(planner.close)
        return planner

    def test_selects_terminal_alignment_instead_of_transient_peak(self):
        state = initial_state()
        live = FakeEnv(state)
        planner = self.planner(include_hold=False)
        overshoot = [(256, 256), (290, 256), (320, 256)]
        improve = [(240, 256), (248, 256), (252, 256)]
        decision = planner.choose([overshoot, improve], live.capture_state(), 20)
        self.assertEqual(decision.report["selected"]["candidate_index"], 1)
        self.assertGreater(decision.report["candidates"][0]["peak_coverage"], .99)
        self.assertLess(decision.report["candidates"][0]["final_coverage"], .5)
        self.assertEqual(live.capture_state(), state)
        self.assertEqual(live.cache, 0)
        self.assertEqual(decision.report["counts"], {"restores": 2, "replay_steps": 0, "candidate_steps": 8})

    def test_replays_history_and_forecast_matches_hidden_contact_continuation(self):
        state = initial_state(goal=(400, 400))
        live = FakeEnv(state, cache_effect=.1)
        planner = SimulatorReranker(lambda: FakeEnv(state, cache_effect=.1), state,
                                    PlannerConfig(horizon=2, prefix_lengths=(2,), settle_steps=0, include_hold=False))
        self.addCleanup(planner.close)
        for action in ((230., 256.), (240., 256.), (245., 256.)):
            live.step(action)
            planner.record_action(action)
        expected = live.capture_state()
        chunk = [(248, 256), (250, 256)]
        decision = planner.choose([chunk], expected, 10)
        self.assertEqual(live.capture_state(), expected)
        self.assertEqual(decision.report["counts"]["replay_steps"], 3)
        live.step(decision.actions[0])
        first = Snapshot.from_dict(decision.report["selected"]["first_state"])
        self.assertEqual(state_error(live.capture_state(), first), 0)
        cold = FakeEnv(expected, cache_effect=.1)
        cold.step(decision.actions[0])
        self.assertGreater(state_error(cold.capture_state(), first), .1)

    def test_candidate_order_does_not_leak_branch_state(self):
        state = initial_state()
        planner = self.planner(include_hold=False)
        candidates = [[(240, 256)]*3, [(250, 256)]*3]
        first = planner.choose(candidates, state, 20)
        second = planner.choose(list(reversed(candidates)), state, 20)
        self.assertEqual(first.actions, second.actions)
        self.assertEqual(first.report["selected"]["score"], second.report["selected"]["score"])
        self.assertEqual(planner.counts["candidate_steps"], 16)

    def test_prefix_then_physical_hold_uses_attained_position_and_real_budget(self):
        state = initial_state()
        planner = SimulatorReranker(lambda: FakeEnv(state, pusher_fraction=.5), state,
                                    PlannerConfig(horizon=3, prefix_lengths=(1,), settle_steps=2, include_hold=False))
        self.addCleanup(planner.close)
        decision = planner.choose([[(240, 256), (290, 256), (320, 256)]], state, 4)
        self.assertEqual(decision.actions, ((240., 256.), (230., 256.), (230., 256.), (230., 256.)))
        self.assertEqual(decision.report["forecast_budget"], 4)
        # This actually advances four controls, including the settling controls.
        self.assertEqual(decision.report["counts"]["candidate_steps"], 4)
        short = planner.choose([[(240, 256), (290, 256)]], state, 2)
        self.assertEqual(len(short.actions), 2)

    def test_native_success_is_absorbing_without_executing_remainder(self):
        state = initial_state()
        planner = SimulatorReranker(lambda: FakeEnv(state, success_x=256), state,
                                    PlannerConfig(horizon=3, prefix_lengths=(3,), settle_steps=2, include_hold=False))
        self.addCleanup(planner.close)
        decision = planner.choose([[(256, 256), (320, 256), (350, 256)]], state, 20)
        self.assertEqual(decision.actions, ((256., 256.),))
        self.assertEqual(decision.report["selected"]["score"], 1)
        self.assertEqual(decision.report["counts"]["candidate_steps"], 1)

    def test_hold_is_real_action_and_does_not_fake_success(self):
        state = initial_state()
        planner = self.planner(include_hold=True)
        decision = planner.choose([[(320, 256)]*3], state, 20)
        self.assertEqual(decision.report["selected"]["kind"], "hold")
        self.assertEqual(decision.actions, ((220., 256.),)*4)
        self.assertFalse(decision.report["selected"]["native_success"])

    def test_far_from_goal_pose_progress_beats_hold_without_any_overlap(self):
        state = initial_state(goal=(400, 400))
        planner = self.planner(state=state, include_hold=True)
        decision = planner.choose([[(240, 256)]*3], state, 20)
        self.assertEqual(decision.report["selected"]["kind"], "policy")
        for candidate in decision.report["candidates"]:
            self.assertEqual(candidate["peak_coverage"], 0)
        moving, holding = decision.report["candidates"]
        self.assertGreater(moving["score"], holding["score"])
        self.assertLess(moving["steps"][-1]["position_error"], holding["steps"][-1]["position_error"])

    def test_replay_mismatch_aborts_and_missing_executed_actions_are_rejected(self):
        state = initial_state()
        planner = self.planner()
        changed = replace(state, object_body=BodyState(Pose((221, 256), 0)))
        with self.assertRaisesRegex(ValueError, "differs from live"):
            planner.choose([[(240, 256)]*3], changed, 20)
        with self.assertRaisesRegex(ValueError, "Record every"):
            planner.choose([[(240, 256)]*3], replace(state, control_step=1), 20)

    def test_invalid_candidate_is_reported_without_clipping(self):
        state = initial_state()
        planner = self.planner(include_hold=False)
        decision = planner.choose([[(-1, 256)]*3, [(250, 256)]*3], state, 20)
        self.assertEqual(decision.report["selected"]["candidate_index"], 1)
        self.assertEqual(decision.report["rejected_candidates"][0]["candidate_index"], 0)
        self.assertEqual(decision.report["completed_candidates"], 1)

    def test_deadline_never_returns_incomplete_forecast(self):
        planner = self.planner()
        with self.assertRaises(TimeoutError):
            planner.choose([[(240, 256)]*3], initial_state(), 20, deadline=time.monotonic()-1)
        self.assertEqual(planner.counts["candidate_steps"], 0)

    def test_deadline_returns_completed_best_and_counts_discarded_partial_work(self):
        planner = self.planner(include_hold=False)

        def deadline(_):
            if planner.counts["candidate_steps"] == 5:
                raise TimeoutError("controlled deadline during second candidate")

        with patch.object(planner, "_check_deadline", side_effect=deadline):
            decision = planner.choose([[(240, 256)]*3, [(256, 256)]*3], initial_state(), 20)
        self.assertTrue(decision.report["timed_out"])
        self.assertEqual(decision.report["completed_candidates"], 1)
        self.assertEqual(decision.report["selected"]["candidate_index"], 0)
        self.assertEqual(decision.report["counts"]["candidate_steps"], 5)

    def test_last_control_finishing_after_deadline_is_not_selectable(self):
        planner = self.planner(include_hold=False)
        clock = [0.0]
        step = planner.env.step

        def slow_step(action):
            result = step(action)
            clock[0] += 1.0
            return result

        with patch.object(planner.env, "step", side_effect=slow_step), \
                patch("tdyssey.simulator_planning.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, "No simulator candidate completed"):
                planner.choose([[(240, 256)]*3], initial_state(), 20, deadline=3.5)
        self.assertEqual(planner.counts["candidate_steps"], 4)

    def test_scoring_penalizes_motion_and_uses_terminal_minimum(self):
        config = PlannerConfig(speed_weight=.5, angular_speed_weight=.2)
        rows = [{"coverage": .9, "speed": 512, "angular_speed": 1, "native_done": False,
                 "position_error": 512, "angle_error": math.pi},
                {"coverage": .8, "speed": 0, "angular_speed": -1, "native_done": False,
                 "position_error": 0, "angle_error": 0}]
        self.assertAlmostEqual(score_prediction(rows, config), .8-.25-.2-.05-.025)


if __name__ == "__main__":
    unittest.main()
