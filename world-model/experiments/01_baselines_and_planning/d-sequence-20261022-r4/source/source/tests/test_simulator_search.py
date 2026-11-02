"""Source-only search protocol tests with explicit fake dynamics and local RNG.

These checks do not establish contact physics or NumPy sampling parity.
"""

from dataclasses import replace
import math
import random
import unittest
from unittest.mock import patch

from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.simulator_planning import state_error
from tdyssey.simulator_search import SearchConfig, SimulatorActionSearch, primitive_actions
from tdyssey.state import BodyState, Snapshot


def initial_state():
    return Snapshot("D", official_shape("D").geometry_hash, "a"*64,
                    BodyState(Pose((220, 256), 0)), BodyState(Pose((100, 100), 0)),
                    Pose((256, 256), 0), 0)


class LocalRngDouble:
    def __init__(self, seed):
        self.generator = random.Random(seed)
        self.normal_calls = 0

    def uniform(self, low, high):
        return self.generator.uniform(low, high)

    def normal(self, mean, std):
        self.normal_calls += 1
        return self.generator.gauss(mean, std)

    def choice(self, items):
        return self.generator.choice(items)


class HistoryEnv:
    def __init__(self, state, success_target=None):
        self.state, self.cache = state, 0
        self.success_target = success_target
        self.closed = False

    def restore_state(self, state):
        self.state, self.cache = state, 0

    def capture_state(self):
        return self.state

    def step(self, action):
        self.cache += 1
        old = self.state.object_body.pose.position
        position = (action[0]+self.cache*.01, action[1])
        self.state = replace(self.state, object_body=BodyState(Pose(position, 0),
                             tuple((new-previous)*10 for new, previous in zip(position, old))),
                             pusher_body=BodyState(Pose(action, 0)), pusher_target=action,
                             control_step=self.state.control_step+1)
        done = action == self.success_target
        return {}, float(done), done, {"n_contacts": 1}

    def close(self):
        self.closed = True


def macro(shape, state, parameters, active_controls, max_target_step):
    path = ((245., 256.), (250., 256.), (255., 256.), (256., 256.))
    return path[:active_controls], {"approach_controls": 1}


class SearchProtocolChecks(unittest.TestCase):
    def planner(self, **kwargs):
        state = initial_state()
        options = dict(horizon=4, population=2, generations=1, elite_count=1,
                       max_evaluations=8, execute_steps=2, settle_steps=2)
        options.update(kwargs)
        planner = SimulatorActionSearch(lambda: HistoryEnv(state), state, SearchConfig(**options))
        planner.rng = LocalRngDouble(7)
        self.addCleanup(planner.close)
        return planner

    def test_history_is_reconstructed_and_live_continuation_matches_prediction(self):
        state, planner = initial_state(), self.planner()
        live = HistoryEnv(state)
        for action in ((160., 180.), (180., 200.)):
            live.step(action)
            planner.record_action(action)
        before = live.capture_state()
        with patch("tdyssey.simulator_search.primitive_actions", side_effect=macro):
            decision = planner.choose(before, 10)
        self.assertEqual(live.capture_state(), before)
        self.assertEqual(decision.report["counts"]["replay_steps"], 6)
        self.assertEqual(decision.report["selected"]["replay_state_error"], 0)
        live.step(decision.actions[0])
        expected = Snapshot.from_dict(decision.report["selected_predictions"][0])
        self.assertEqual(state_error(live.capture_state(), expected), 0)

    def test_replanning_keeps_incumbent_tail_instead_of_restarting_approach(self):
        state, planner = initial_state(), self.planner()
        live = HistoryEnv(state)
        with patch("tdyssey.simulator_search.primitive_actions", side_effect=macro):
            first = planner.choose(state, 20)
            for action in first.actions[:2]:
                live.step(action)
                planner.record_action(action)
            second = planner.choose(live.capture_state(), 18)
        self.assertEqual(second.report["candidates"][0]["kind"], "incumbent_tail")
        self.assertEqual(second.report["selected"]["kind"], "incumbent_tail")
        self.assertEqual(second.actions[0], first.actions[2])

    def test_invalid_proposals_count_toward_hard_evaluation_limit(self):
        planner = self.planner(population=4, generations=3, max_evaluations=3)
        with patch("tdyssey.simulator_search.primitive_actions", side_effect=ValueError("unsafe route")):
            decision = planner.choose(initial_state(), 20)
        self.assertEqual(decision.report["evaluations"], 3)
        self.assertEqual(decision.report["completed_candidates"], 1)
        self.assertEqual(len(decision.report["rejected_candidates"]), 2)
        self.assertEqual(decision.report["counts"]["candidate_steps"], 6)

    def test_common_forecast_and_hold_consume_only_remaining_task_controls(self):
        planner = self.planner()
        with patch("tdyssey.simulator_search.primitive_actions", side_effect=macro):
            decision = planner.choose(initial_state(), 2)
        self.assertEqual(decision.report["forecast_budget"], 2)
        self.assertEqual(len(decision.actions), 2)
        self.assertEqual([r["forecast_steps"] for r in decision.report["candidates"]], [2, 2, 2])

    def test_native_success_stops_forecast_immediately(self):
        state = initial_state()
        planner = SimulatorActionSearch(lambda: HistoryEnv(state, success_target=(245., 256.)), state,
                                        SearchConfig(horizon=4, population=2, generations=1,
                                                     elite_count=1, max_evaluations=3, settle_steps=2))
        planner.rng = LocalRngDouble(1)
        self.addCleanup(planner.close)
        with patch("tdyssey.simulator_search.primitive_actions", side_effect=macro):
            decision = planner.choose(state, 20)
        self.assertTrue(decision.report["selected"]["native_success"])
        self.assertEqual(decision.actions, ((245., 256.),))

    def test_elite_refinement_runs_without_touching_global_random_state(self):
        planner = self.planner(population=4, generations=2, elite_count=2, max_evaluations=9)
        before = random.getstate()

        def simple(shape, state, parameters, active_controls, maximum):
            return ((256.+parameters[1]*.1, 256.),)*active_controls, {"approach_controls": 0}

        with patch("tdyssey.simulator_search.primitive_actions", side_effect=simple):
            decision = planner.choose(initial_state(), 20)
        self.assertEqual(random.getstate(), before)
        self.assertGreater(planner.rng.normal_calls, 0)
        self.assertEqual(decision.report["evaluations"], 9)
        self.assertEqual(decision.report["generations_completed"], 2)

    def test_deadline_discards_late_work_but_preserves_its_cost(self):
        planner = self.planner()
        clock, step = [0.], planner.branch.env.step

        def slow(action):
            result = step(action)
            clock[0] += 1.
            return result

        with patch.object(planner.branch.env, "step", side_effect=slow), \
                patch("tdyssey.simulator_planning.time.monotonic", side_effect=lambda: clock[0]), \
                patch("tdyssey.simulator_search.primitive_actions", side_effect=macro):
            decision = planner.choose(initial_state(), 20, deadline=8.5)
        self.assertTrue(decision.report["timed_out"])
        self.assertEqual(decision.report["selected"]["kind"], "hold")
        self.assertEqual(decision.report["counts"]["candidate_steps"], 9)
        self.assertEqual(decision.report["completed_candidates"], 1)

    def test_missing_real_history_is_a_protocol_error(self):
        planner = self.planner()
        with self.assertRaisesRegex(ValueError, "Record every"):
            planner.choose(replace(initial_state(), control_step=1), 20)

    def test_geometric_primitives_preserve_routes_and_reject_workspace_escape(self):
        state = replace(initial_state(), object_body=BodyState(Pose((256, 256), .6)),
                        pusher_body=BodyState(Pose((120, 120), 0)), goal=Pose((300, 210), .7))
        a = (0., 0., 0., 1., 20., .625, 1., 1)
        b = (*a[:-1], -1)
        one, _ = primitive_actions(official_shape("D"), state, a, 64)
        two, _ = primitive_actions(official_shape("D"), state, b, 64)
        self.assertNotEqual(one, two)
        self.assertEqual(len(one), 64)
        near_wall = replace(state, object_body=BodyState(Pose((30, 30), 0)))
        with self.assertRaises(ValueError):
            primitive_actions(official_shape("D"), near_wall, a, 64)


if __name__ == "__main__":
    unittest.main()
