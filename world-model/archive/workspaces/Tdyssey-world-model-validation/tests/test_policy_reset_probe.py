"""Reset-probe guards with stdlib fakes only: no assets, real env, or policy."""

from copy import deepcopy
from dataclasses import replace
import gc
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import weakref

from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.policy_reset_probe import (ResetProbeViolation, RuntimeResetProbe,
                                       instrument_space_steps, observation_summary,
                                       validate_matched_reset_pair)
from tdyssey.state import BodyState, Snapshot


ROOT = Path(__file__).resolve().parents[1]
IMAGE = [[[0.25] * 96 for _ in range(96)] for _ in range(3)]


def state(shape, *, object_position=None, pusher=(70, 420), goal=None):
    return Snapshot(shape, official_shape(shape).geometry_hash, "1" * 64,
                    BodyState(Pose(object_position or ((180, 120) if shape == "T" else (180, 165)), 0)),
                    BodyState(Pose(pusher, 0)),
                    Pose(goal or ((320, 300) if shape == "T" else (320, 345)), 0), 0)


class FakeSpace:
    calls = []
    fail = False

    def step(self, dt):
        self.calls.append(dt)
        if self.fail:
            raise OSError("fake physics failure")


class FakeBase:
    controls = []

    def step(self, action):
        self.controls.append(action)
        raise AssertionError("unguarded fake base control")


class FakeEnv(FakeBase):
    mode = "normal"
    instances = []
    sim_hz = 100

    def __init__(self, *, shape, legacy):
        self.shape, self.legacy = shape, legacy
        self.space = FakeSpace()
        self.closed = False
        self.seed_value = None
        self.reset_count = 0
        self.instances.append(self)
        if self.mode == "constructor_control":
            self.step(None)
        if self.mode == "constructor_physics":
            self.space.step(0.01)

    def seed(self, seed):
        self.seed_value = seed
        if self.mode == "seed_physics":
            self.space.step(0.01)

    def reset(self):
        self.reset_count += 1
        if self.mode == "reset_failure":
            raise RuntimeError("fake reset failure")
        if self.mode == "control_none":
            self.step(None)
        if self.mode == "control_action":
            self.step((10, 20))
        if self.mode == "super_control":
            super().step(None)
        if self.mode != "no_settle":
            self.space.step(0.02 if self.mode == "wrong_dt" else 0.01)
        if self.mode == "two_settles":
            self.space.step(0.01)
        if self.mode == "swallow_guard":
            try:
                self.space.step(0.01)
            except ResetProbeViolation:
                pass
        if self.mode == "post_settle_failure":
            raise RuntimeError("fake failure after physics")
        if self.mode == "bad_observation":
            return {"image": [], "agent_pos": [70, 420]}
        return {"image": IMAGE, "agent_pos": [70, 420]}

    def capture_state(self):
        if self.mode == "capture_control":
            self.step(None)
        if self.mode == "outside_walls":
            return state(self.shape, object_position=(1, 1))
        if self.mode == "near_goal":
            return state(self.shape, object_position=(320, 300) if self.shape == "T" else (320, 345))
        if self.mode == "matched_collision" and self.shape == "T":
            return state("T", pusher=(225, 195))
        if self.mode == "wrong_snapshot":
            return replace(state(self.shape), control_step=1)
        return state(self.shape)

    def close(self):
        self.closed = True
        if self.mode == "close_failure":
            raise OSError("fake cleanup failure")
        if self.mode == "close_physics":
            self.space.step(0.01)


def fake_metrics(env):
    return {"coverage": 1.0 if env.mode == "high_coverage" else 0.0, "iou": 0.0}


def probe(**kwargs):
    return RuntimeResetProbe(ROOT, env_factory=FakeEnv,
                             space_step_context=lambda guard: instrument_space_steps(FakeSpace, guard),
                             metrics_fn=kwargs.pop("metrics_fn", fake_metrics), **kwargs)


class ResetProbeChecks(unittest.TestCase):
    def setUp(self):
        FakeEnv.mode = "normal"
        FakeEnv.instances = []
        FakeSpace.calls = []
        FakeSpace.fail = False
        FakeBase.controls = []

    def test_fresh_environment_single_reset_snapshot_observation_and_cleanup(self):
        check = probe()
        first = check("T", 400000, "original_train")
        second = check("L", 400000, "final_train")
        for result in (first, second):
            self.assertTrue(result["valid"])
            self.assertEqual(result["classification"], "valid")
            self.assertIsNone(result["error"])
            self.assertEqual((result["reset_attempts"], result["reset_physics_substeps"],
                              result["reset_physics_substep_attempts"], result["control_steps"]), (1, 1, 1, 0))
            self.assertFalse(result["provenance"]["source_verified"])
            self.assertTrue(result["provenance"]["legacy"])
            self.assertEqual(result["observation"]["image"]["shape"], [3, 96, 96])
            self.assertEqual(result["snapshot"]["control_step"], 0)
            json.dumps(result, allow_nan=False)
        self.assertEqual(len(FakeEnv.instances), 2)
        self.assertIsNot(FakeEnv.instances[0], FakeEnv.instances[1])
        self.assertTrue(all(env.closed and env.legacy and env.reset_count == 1 for env in FakeEnv.instances))
        self.assertEqual(FakeSpace.calls, [0.01, 0.01])
        self.assertEqual(FakeBase.controls, [])

    def test_geometry_invalid_is_replaceable_but_initial_success_is_rejected(self):
        for mode, issue in (("outside_walls", "object_outside_walls"),
                            ("near_goal", "already_near_goal"),
                            ("high_coverage", "initial_training_coverage_exceeds_success_threshold")):
            with self.subTest(mode=mode):
                FakeEnv.mode = mode
                check = probe()
                result = check("T", 400000, "candidate_train")
                self.assertFalse(result["valid"])
                self.assertEqual(result["classification"], "geometry_invalid")
                self.assertIn(issue, result["issues"])
                self.assertIsNone(result["error"])
                self.assertFalse(check.blocked)
                FakeEnv.mode = "normal"
                self.assertTrue(check("T", 400001, "candidate_train")["valid"])

    def test_control_none_action_super_and_capture_are_rejected_before_implementation(self):
        for mode in ("control_none", "control_action", "super_control", "capture_control"):
            with self.subTest(mode=mode):
                FakeEnv.mode = mode
                FakeSpace.calls = []
                check = probe()
                result = check("T", 7, "original_train")
                self.assertEqual(result["classification"], "guard_violation")
                self.assertEqual(result["blocked_control_attempts"], 1)
                self.assertEqual(result["control_steps"], 0)
                self.assertEqual(FakeBase.controls, [])
                self.assertEqual(len(FakeSpace.calls), int(mode == "capture_control"))
                self.assertTrue(FakeEnv.instances[-1].closed)
                count = len(FakeEnv.instances)
                with self.assertRaisesRegex(RuntimeError, "blocked"):
                    check("T", 8, "candidate_train")
                self.assertEqual(len(FakeEnv.instances), count)

    def test_second_substep_and_swallowed_guard_never_execute_extra_physics(self):
        for mode in ("two_settles", "swallow_guard"):
            with self.subTest(mode=mode):
                FakeEnv.mode = mode
                FakeSpace.calls = []
                result = probe()("L", 2, "original_train")
                self.assertEqual(result["classification"], "guard_violation")
                self.assertEqual(FakeSpace.calls, [0.01])
                self.assertEqual(result["space_step_attempts"], 1)
                self.assertEqual(result["space_step_completed"], 1)
                self.assertEqual(result["blocked_space_step_attempts"], 1)
                self.assertTrue(FakeEnv.instances[-1].closed)

    def test_wrong_timestep_and_physics_outside_reset_are_blocked(self):
        for mode in ("constructor_physics", "seed_physics", "wrong_dt", "close_physics"):
            with self.subTest(mode=mode):
                FakeEnv.mode = mode
                FakeSpace.calls = []
                result = probe()("T", 4, "final_train")
                self.assertEqual(result["classification"], "guard_violation")
                self.assertEqual(result["blocked_space_step_attempts"], 1)
                self.assertEqual(len(FakeSpace.calls), int(mode == "close_physics"))
                self.assertLessEqual(result["reset_physics_substeps"], 1)
                self.assertTrue(FakeEnv.instances[-1].closed)

    def test_constructor_control_is_guarded_and_partial_environment_is_closed(self):
        FakeEnv.mode = "constructor_control"
        result = probe()("T", 4, "original_train")
        self.assertEqual(result["classification"], "guard_violation")
        self.assertEqual(result["reset_attempts"], 0)
        self.assertEqual(result["blocked_control_attempts"], 1)
        self.assertEqual(FakeSpace.calls, [])
        self.assertEqual(FakeBase.controls, [])
        self.assertTrue(FakeEnv.instances[-1].closed)

    def test_missing_settle_failure_before_and_after_physics_keep_attempts_and_close(self):
        for mode, attempts, completed, classification in (
                ("no_settle", 0, 0, "guard_violation"),
                ("reset_failure", 0, 0, "runtime_error"),
                ("post_settle_failure", 1, 1, "runtime_error"),
                ("bad_observation", 1, 1, "runtime_error"),
                ("wrong_snapshot", 1, 1, "runtime_error"),
                ("close_failure", 1, 1, "runtime_error")):
            with self.subTest(mode=mode):
                FakeEnv.mode = mode
                result = probe()("T", 4, "original_train")
                self.assertEqual(result["classification"], classification)
                self.assertEqual(result["reset_attempts"], 1)
                self.assertEqual(result["reset_physics_substep_attempts"], attempts)
                self.assertEqual(result["reset_physics_substeps"], completed)
                self.assertTrue(result["blocked"])
                self.assertTrue(FakeEnv.instances[-1].closed)
        FakeEnv.mode = "normal"
        FakeSpace.fail = True
        result = probe()("T", 4, "original_train")
        self.assertEqual((result["reset_physics_substep_attempts"], result["reset_physics_substeps"]), (1, 0))
        self.assertTrue(FakeEnv.instances[-1].closed)

    def test_metrics_error_blocks_and_cleanup_error_is_preserved(self):
        FakeEnv.mode = "close_failure"

        def fail_metrics(env):
            raise ArithmeticError("fake metric failure")

        result = probe(metrics_fn=fail_metrics)("L", 7, "original_train")
        self.assertEqual(result["error"]["type"], "ArithmeticError")
        self.assertEqual(result["cleanup_error"]["type"], "OSError")
        self.assertTrue(result["blocked"])

    def test_evaluation_cache_is_phase_and_seed_specific_and_no_restore_is_needed(self):
        check = probe()
        t = check("T", 9, "original_evaluation")
        l = check("L", 9, "original_evaluation")
        pair = validate_matched_reset_pair(t, l)
        self.assertTrue(l["valid"] and pair["valid"])
        self.assertEqual(l["matched_com"], pair)
        self.assertEqual(pair["snapshots"]["L"]["object_body"]["pose"]["position"], (180.0, 165.0))
        self.assertEqual(pair["physics_substeps"], 0)
        self.assertFalse(pair["state_restored"])
        self.assertEqual(len(FakeSpace.calls), 2)
        before = len(FakeEnv.instances)
        missing = check("L", 9, "final_evaluation")
        self.assertTrue(missing["blocked"])
        self.assertEqual(missing["reset_attempts"], 0)
        self.assertEqual(len(FakeEnv.instances), before)

    def test_raw_valid_matched_invalid_rejects_the_pair(self):
        FakeEnv.mode = "matched_collision"
        check = probe()
        t = check("T", 19, "original_evaluation")
        l = check("L", 19, "original_evaluation")
        self.assertTrue(t["valid"] and l["raw_valid"])
        self.assertFalse(l["valid"])
        self.assertEqual(l["classification"], "geometry_invalid")
        self.assertIn("matched_L:pusher_object_overlap_or_insufficient_clearance", l["issues"])
        self.assertEqual(l["matched_com"]["issues"]["T"], [])
        self.assertEqual(len(FakeSpace.calls), 2)

    def test_production_loader_keeps_full_pinning_and_exact_factory_without_creating_env(self):
        from tdyssey import official

        spec = {"source_commit": "a" * 40, "source_sha256": {"env.py": "b" * 64}}
        with patch.object(official, "official_spec", return_value=spec), \
                patch.object(official, "load_official_source", return_value=FakeBase) as load, \
                patch("tdyssey.policy_reset_probe.make_official_shape_env_class", return_value=FakeEnv) as factory:
            check = RuntimeResetProbe(ROOT)
            load.assert_called_once_with(str(ROOT.resolve()), spec)
            factory.assert_called_once_with(FakeBase)
            self.assertTrue(check.provenance["source_verified"])
            self.assertEqual(check.provenance["source_sha256"], spec["source_sha256"])
            self.assertEqual(FakeEnv.instances, [])
        with self.assertRaisesRegex(ValueError, "may not replace"):
            RuntimeResetProbe(ROOT, metrics_fn=fake_metrics)

    def test_guard_patches_restore_and_probe_retains_no_environment_or_images(self):
        original_space, original_step = FakeSpace.step, FakeBase.step
        check = probe()
        check("T", 1, "original_evaluation")
        env_ref = weakref.ref(FakeEnv.instances.pop())
        gc.collect()
        self.assertIsNone(env_ref())
        self.assertIs(FakeSpace.step, original_space)
        self.assertIs(FakeBase.step, original_step)
        cached = check._evaluation_t[("original_evaluation", 1)]
        self.assertNotIn("image", cached)
        self.assertEqual(cached["observation"]["image"]["shape"], [3, 96, 96])
        check.close()
        self.assertEqual(check._evaluation_t, {})
        with self.assertRaisesRegex(RuntimeError, "closed"):
            check("T", 2, "candidate_train")

    def test_instrumentation_contention_fails_before_reset(self):
        from tdyssey import policy_reset_probe

        with policy_reset_probe._INSTRUMENTATION_LOCK:
            result = probe()("T", 1, "original_train")
        self.assertEqual(result["reset_attempts"], 0)
        self.assertEqual(result["classification"], "runtime_error")
        self.assertTrue(result["blocked"])
        self.assertEqual(FakeEnv.instances, [])

    def test_observation_nonfinite_range_shape_and_dtype_are_rejected(self):
        for observation in ({"image": IMAGE, "agent_pos": [float("nan"), 3]},
                            {"image": IMAGE, "agent_pos": [True, 3]},
                            {"image": [], "agent_pos": [1, 2]},
                            {"image": IMAGE, "agent_pos": [1, 2], "shape_id": "T"}):
            with self.assertRaises(ValueError):
                observation_summary(observation)
        invalid = deepcopy(IMAGE)
        invalid[0][0][0] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            observation_summary({"image": invalid, "agent_pos": [1, 2]})

        class WrongDtype:
            shape = (3, 96, 96)
            dtype = "float64"

        with self.assertRaisesRegex(ValueError, "float32"):
            observation_summary({"image": WrongDtype(), "agent_pos": [1, 2]})

    def test_import_with_no_site_packages_does_not_load_policy_or_runtime(self):
        code = ("import sys; import tdyssey.policy_reset_probe; "
                "assert not any(n == 'torch' or n == 'pymunk' or n == 'tdyssey.policy' "
                "or n.startswith('tdyssey.policy_rl') or n.startswith('tdyssey.policy_checkpoint') "
                "for n in sys.modules)")
        subprocess.run([sys.executable, "-B", "-S", "-c", code], cwd=ROOT, check=True,
                       capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
