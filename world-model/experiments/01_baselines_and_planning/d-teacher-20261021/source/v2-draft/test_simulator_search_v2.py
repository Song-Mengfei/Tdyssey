"""Isolated v2 draft checks; explicit doubles are not simulator evidence.

The test module imports the same-directory draft without replacing live modules.
Writing these checks does not run search, physics, optimization or training.
"""

from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose, Rectangle, Shape
from tdyssey.simulator_planning import state_error
from tdyssey.simulator_search import SearchConfig as OriginalSearchConfig
from tdyssey.state import BodyState, Snapshot


SPEC = importlib.util.spec_from_file_location(
    "tdyssey_simulator_search_v2_draft", Path(__file__).with_name("simulator_search_v2.py")
)
draft = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = draft
SPEC.loader.exec_module(draft)


def initial_state():
    return Snapshot(
        "D", official_shape("D").geometry_hash, "a" * 64,
        BodyState(Pose((220, 256), 0)), BodyState(Pose((100, 100), 0)),
        Pose((256, 256), 0), 0,
    )


class LocalRngDouble:
    """Local standard-library random source, avoiding a NumPy dependency."""

    def __init__(self, seed):
        self.generator = random.Random(seed)

    def uniform(self, low, high):
        return self.generator.uniform(low, high)

    def normal(self, mean, deviation):
        return self.generator.gauss(mean, deviation)

    def choice(self, items):
        return self.generator.choice(items)


class HistoryEnv:
    """A hidden counter makes full replay observably different from cold restore."""

    def __init__(self, state):
        self.state, self.cache = state, 0
        self.closed = False

    def restore_state(self, state):
        self.state, self.cache = state, 0

    def capture_state(self):
        return self.state

    def step(self, action):
        self.cache += 1
        old = self.state.object_body.pose.position
        position = (action[0] + self.cache * .01, action[1])
        self.state = replace(
            self.state,
            object_body=BodyState(
                Pose(position, 0),
                tuple((new - previous) * 10 for new, previous in zip(position, old)),
            ),
            pusher_body=BodyState(Pose(action, 0)),
            pusher_target=action, control_step=self.state.control_step + 1,
        )
        return {}, 0., False, {"n_contacts": 1}

    def close(self):
        self.closed = True


def macro(shape, state, parameters, active_controls, max_target_step=32.):
    path = ((235., 256.), (238., 256.), (241., 256.), (244., 256.),
            (247., 256.), (250., 256.), (253., 256.), (256., 256.))
    return path[:active_controls], {"approach_controls": min(4, active_controls)}


def short_macro(shape, state, family, parameters, active_controls, max_target_step=32.):
    actions, diagnostics = macro(shape, state, parameters, active_controls, max_target_step)
    return actions, {**diagnostics, "primitive_family": family}


def candidate(score, *, kind="primitive", native_success=False,
              unsafe=0, status="completed"):
    return {
        "kind": kind, "score": score, "native_success": native_success,
        "unsafe_state_count": unsafe, "status": status,
    }


class ExteriorGeometryChecks(unittest.TestCase):
    def test_partially_shared_seam_is_removed_without_erasing_exposed_remainder(self):
        edges = draft.rectangle_exterior_edges(((0., 0., 2., 2.), (2., 0., 4., 1.)))
        self.assertAlmostEqual(sum(math.dist(edge.a, edge.b) for edge in edges), 12.)
        exposed = []
        for edge in edges:
            x, y = edge.point(.5)
            if abs(x - 2.) < 1e-9:
                self.assertFalse(0. < y < 1., "An internal partial seam became a pushing face")
                if 1. < y < 2.:
                    exposed.append(edge)
        self.assertTrue(exposed, "The exposed upper half of the partial seam was erased")
        self.assertEqual(exposed[0].normal, (1., 0.))

    def test_hole_faces_are_excluded_from_outer_exterior(self):
        ring = ((0., 0., 4., 1.), (0., 3., 4., 4.),
                (0., 1., 1., 3.), (3., 1., 4., 3.))
        edges = draft.rectangle_exterior_edges(ring)
        self.assertAlmostEqual(sum(math.dist(edge.a, edge.b) for edge in edges), 16.)
        for edge in edges:
            x, y = edge.point(.5)
            self.assertTrue(abs(x) < 1e-9 or abs(x - 4.) < 1e-9
                            or abs(y) < 1e-9 or abs(y - 4.) < 1e-9)
            nx, ny = edge.normal
            self.assertAlmostEqual(math.hypot(nx, ny), 1.)
            self.assertGreater((x - 2.) * nx + (y - 2.) * ny, 0.)

    def test_signed_rotation_and_torque_use_the_pose_rotation_convention(self):
        self.assertAlmostEqual(draft.signed_turn(math.radians(179), math.radians(-179)),
                               math.radians(2))
        self.assertAlmostEqual(draft.signed_turn(0., math.pi + .2, symmetry_order=2), .2)
        positive = draft.torque_proxy((1., -1.), (0., 0.), (0., 1.))
        negative = draft.torque_proxy((-1., -1.), (0., 0.), (0., 1.))
        self.assertAlmostEqual(positive, 1.)
        self.assertAlmostEqual(negative, -1.)
        pose = Pose((17., 23.), .7)
        contact, center = pose.transform((1., -1.)), pose.position
        inward = (-math.sin(pose.angle), math.cos(pose.angle))
        self.assertAlmostEqual(draft.torque_proxy(contact, center, inward), positive)
        self.assertGreater(draft.signed_turn(0., .2) * positive, 0.)
        self.assertLess(draft.signed_turn(0., .2) * negative, 0.)

    def test_direct_clear_checks_disk_clearance_in_rotated_object_coordinates(self):
        shape = Shape("fixture", (Rectangle(-10., -5., 10., 5.),), 1, "fixture")
        pose = Pose((256., 256.), math.pi / 2)
        self.assertTrue(draft.direct_clear((230., 220.), (230., 290.), shape, pose, clearance=15.))
        self.assertFalse(draft.direct_clear((210., 256.), (300., 256.), shape, pose, clearance=15.))
        # x=237 is 14 pixels from the rotated rectangle's left surface at251.
        self.assertFalse(draft.direct_clear((237., 240.), (237., 272.), shape, pose, clearance=15.))

    def test_direct_clear_checks_neighboring_components_at_a_concave_corner(self):
        shape = Shape("fixture", (Rectangle(-20., -20., 0., 20.),
                                  Rectangle(0., 0., 20., 20.)), 1, "fixture")
        pose = Pose((256., 256.), 0.)
        self.assertTrue(draft.direct_clear((310., 220.), (285., 220.), shape, pose, clearance=15.))
        self.assertFalse(draft.direct_clear((285., 235.), (260., 241.), shape, pose, clearance=15.))


class IncumbentSelectionChecks(unittest.TestCase):
    def choose(self, records, best, incumbent):
        chosen, reason = draft.select_candidate(records, best, incumbent, switch_margin=.02)
        self.assertIsInstance(reason, str)
        self.assertTrue(reason)
        return chosen

    def test_small_gain_keeps_incumbent_and_material_gain_switches(self):
        incumbent = candidate(.5, kind="incumbent_tail")
        small, material = candidate(.51), candidate(.53)
        self.assertIs(self.choose([incumbent, small], small, incumbent), incumbent)
        self.assertIs(self.choose([incumbent, material], material, incumbent), material)

    def test_native_success_bypasses_margin(self):
        incumbent = candidate(.5, kind="incumbent_tail")
        success = candidate(.51, native_success=True)
        self.assertIs(self.choose([incumbent, success], success, incumbent), success)

    def test_safe_replacement_bypasses_margin_even_with_lower_raw_score(self):
        incumbent = candidate(.8, kind="incumbent_tail", unsafe=1)
        replacement = candidate(.7)
        self.assertIs(self.choose([incumbent, replacement], incumbent, incumbent), replacement)

    def test_no_completed_incumbent_does_not_hold_back_new_plan(self):
        replacement = candidate(.4)
        self.assertIs(self.choose([replacement], replacement, None), replacement)

    def test_safe_native_success_is_preferred_over_unsafe_native_success(self):
        unsafe = candidate(1.9, native_success=True, unsafe=1)
        safe = candidate(1.8, native_success=True)
        self.assertIs(self.choose([unsafe, safe], unsafe, None), safe)

    def test_native_success_incumbent_precedes_safe_nonsuccess(self):
        incumbent = candidate(1.9, kind="incumbent_tail", native_success=True, unsafe=1)
        nonsuccess = candidate(.7)
        self.assertIs(self.choose([incumbent, nonsuccess], incumbent, incumbent), incumbent)


class DraftProtocolChecks(unittest.TestCase):
    def planner(self, **overrides):
        state = initial_state()
        options = dict(horizon=8, population=3, generations=1,
                       elite_count=1, max_evaluations=5,
                       execute_steps=1, settle_steps=1)
        options.update(overrides)
        config = draft.SearchConfig(**options)
        planner = draft.SimulatorActionSearchV2(
            lambda: HistoryEnv(state), state, config=config,
            v2_config=draft.V2Config(switch_margin=.02),
        )
        planner.rng = LocalRngDouble(7)
        self.addCleanup(planner.close)
        return planner

    def test_config_preserves_v1_configuration_type(self):
        self.assertIs(draft.SearchConfig, OriginalSearchConfig)
        self.assertEqual(draft.V2Config().switch_margin, .02)

    def test_remaining_approach_is_decremented_and_reported_for_incumbent(self):
        planner, live = self.planner(), HistoryEnv(initial_state())
        with patch.object(draft, "primitive_actions", side_effect=macro), \
                patch.object(draft, "short_actions", side_effect=short_macro):
            first = planner.choose(live.capture_state(), 30)
            self.assertEqual(first.report["selected"]["approach_controls"], 4)
            action = first.actions[0]
            live.step(action)
            planner.record_action(action)
            self.assertEqual(planner._approach_remaining, 3)
            second = planner.choose(live.capture_state(), 29)
        incumbent = next(row for row in second.report["candidates"] if row["kind"] == "incumbent_tail")
        self.assertEqual(incumbent["approach_controls"], 3)
        self.assertEqual(incumbent["approach_contact_controls"], 3)

    def test_full_history_replay_preserves_exact_selected_prediction(self):
        planner, live = self.planner(), HistoryEnv(initial_state())
        for action in ((160., 180.), (180., 200.)):
            live.step(action)
            planner.record_action(action)
        before = live.capture_state()
        with patch.object(draft, "primitive_actions", side_effect=macro), \
                patch.object(draft, "short_actions", side_effect=short_macro):
            decision = planner.choose(before, 20)
        self.assertEqual(live.capture_state(), before)
        self.assertEqual(decision.report["selected"]["replay_state_error"], 0.)
        self.assertGreater(decision.report["counts"]["replay_steps"], 0)
        live.step(decision.actions[0])
        predicted = Snapshot.from_dict(decision.report["selected_predictions"][0])
        self.assertEqual(state_error(live.capture_state(), predicted), 0.)

    def test_divergent_real_action_clears_incumbent_phase_metadata(self):
        planner = self.planner()
        with patch.object(draft, "primitive_actions", side_effect=macro), \
                patch.object(draft, "short_actions", side_effect=short_macro):
            planner.choose(initial_state(), 20)
        planner.record_action((111., 112.))
        self.assertEqual(planner._approach_remaining, 0)
        self.assertEqual(planner._active_remaining, 0)
        self.assertFalse(planner._tail)

    def test_common_forecast_is_clipped_to_remaining_real_controls(self):
        planner = self.planner()
        with patch.object(draft, "primitive_actions", side_effect=macro), \
                patch.object(draft, "short_actions", side_effect=short_macro):
            decision = planner.choose(initial_state(), 2)
        self.assertEqual(decision.report["forecast_budget"], 2)
        self.assertEqual(len(decision.actions), 2)
        self.assertTrue(all(row["forecast_steps"] <= 2 for row in decision.report["candidates"]
                            if row["status"] == "completed"))

    def test_proposal_generation_failures_consume_the_hard_evaluation_budget(self):
        planner = self.planner(max_evaluations=3)
        with patch.object(planner, "_proposal", side_effect=draft.ProposalRejected("no accessible contact")) as proposal:
            decision = planner.choose(initial_state(), 20)
        self.assertEqual(proposal.call_count, 2)
        self.assertEqual(decision.report["evaluations"], 3)
        self.assertEqual(decision.report["requested_candidates"], 3)
        self.assertEqual(len(decision.report["candidates"]), 3)
        self.assertEqual(decision.report["completed_candidates"], 1)
        self.assertEqual(decision.report["selected"]["kind"], "hold")
        rejected = decision.report["rejected_candidates"]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(all(value == 0 for value in row["counts"].values()) for row in rejected))
        self.assertEqual(decision.report["counts"]["candidate_steps"], decision.report["forecast_budget"])

    def test_composite_recovery_uses_actual_stage_state_and_preserves_phase_flags(self):
        planner = self.planner(horizon=32, population=2, max_evaluations=2)
        live = HistoryEnv(initial_state())
        recovery_states = []

        def proposal(state, family, ordinal, elites=None, **kwargs):
            return {
                "family": "rotation" if state.control_step == 0 else "translation",
                "edge_id": 0, "route": 1,
                "parameters": (0, .5, 2., 10., 0., 100., 1),
                "recover_translation": state.control_step == 0,
            }

        def stages(shape, state, family, parameters, active_budget, max_target_step=32.):
            if family == "rotation":
                return ((245., 256.), (246., 256.)), {"approach_controls": 1}
            recovery_states.append(state)
            self.assertEqual(state.control_step, 2)
            self.assertEqual(state.object_body.pose.position, (246.02, 256.))
            return ((255., 256.), (256., 256.)), {"approach_controls": 1}

        with patch.object(planner, "_proposal", side_effect=proposal), \
                patch.object(draft, "short_actions", side_effect=stages):
            decision = planner.choose(live.capture_state(), 40)

        self.assertEqual(len(recovery_states), 1)
        composite = next(row for row in decision.report["candidates"] if row["kind"] == "primitive")
        self.assertEqual(composite["status"], "completed")
        self.assertTrue(composite["recover_translation"])
        self.assertEqual(composite["approach_controls"], 2)
        self.assertEqual(composite["forecast_steps"], decision.report["forecast_budget"])
        self.assertEqual(composite["counts"]["candidate_steps"], decision.report["forecast_budget"])
        self.assertIs(decision.report["selected"], composite)
        self.assertEqual(planner._approach_remaining, 2)
        for index, (action, expected) in enumerate(zip(decision.actions, decision.report["selected_predictions"])):
            live.step(action)
            planner.record_action(action)
            self.assertEqual(state_error(live.capture_state(), Snapshot.from_dict(expected)), 0.)
            if index in (0, 1):
                self.assertEqual(planner._approach_remaining, 1)
        self.assertEqual(planner._approach_remaining, 0)


if __name__ == "__main__":
    unittest.main()
