"""R2A2 source/mock checks only: no real simulator construction or controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tdyssey import native_rgb_r2a2_env as module
from tdyssey.native_rgb_r2a_env import R2AEnv
from tests.test_native_rgb_r2a_env import FakeBase, wrapped


def candidate(arm):
    base = FakeBase()
    env = module.R2A2Env(base, "T", arm,
        [dict(id=0, ring=0, arc_start=0., arc_end=40.)], [base.local_geometry.exterior])
    return env, base


class R2A2EnvTests(unittest.TestCase):
    def test_exact_constants_and_half_lives(self):
        self.assertEqual(module.ARMS, {
            "D-half-life-0.1s": .0009765625,
            "D-half-life-0.3s": .09921256574801246})
        for arm, damping in module.ARMS.items():
            self.assertAlmostEqual(damping**module.HALF_LIFE_SECONDS[arm], .5, places=15)

    def test_actual_identity_only_damping_differs_from_frozen_p0(self):
        original, _ = wrapped()
        reference = original.initial_identity()
        hashes = set()
        for arm, damping in module.ARMS.items():
            env, base = candidate(arm)
            before = base.state().copy()
            rgb = env.reset_scene()
            identity = env.initial_identity()
            self.assertEqual(base.reset_calls, 1)
            self.assertEqual(base.space.calls, 0)
            self.assertEqual(rgb.shape, (3, 96, 96))
            self.assertEqual(identity["arm"], arm)
            self.assertEqual(identity["initial_state"], before.tolist())
            physics, baseline = deepcopy(identity["physics"]), deepcopy(reference["physics"])
            self.assertEqual(physics["space"].pop("damping"), damping)
            self.assertEqual(baseline["space"].pop("damping"), 0.)
            self.assertEqual(physics, baseline)
            self.assertEqual(identity["candidate_protocol"]["sole_physics_delta"],
                dict(parameter="space.damping", original=0., candidate=damping))
            self.assertEqual(identity["base_protocol_sha256"], reference["protocol_sha256"])
            self.assertEqual(identity["protocol_sha256"], hashlib.sha256(json.dumps(
                identity["candidate_protocol"], sort_keys=True).encode()).hexdigest())
            hashes.add(identity["protocol_sha256"])
            json.dumps(identity, allow_nan=False)
        self.assertEqual(len(hashes), 2)

    def test_no_new_control_or_observer_implementation_and_initial_only(self):
        self.assertIs(module.R2A2Env.step, R2AEnv.step)
        self.assertIs(module.R2A2Env._observe_space_step, R2AEnv._observe_space_step)
        self.assertIs(module.R2A2Env._pre_contact, R2AEnv._pre_contact)
        self.assertIs(module.R2A2Env._post_contact, R2AEnv._post_contact)
        for arm, damping in module.ARMS.items():
            env, base = candidate(arm)
            env.reset_scene()
            _, info = env.step((300., 256.))  # FakeBase only; no physics engine.
            self.assertEqual(base.step_calls, 1)
            self.assertEqual(base.space.calls, 10)
            self.assertEqual(info["substep_count"], 10)
            self.assertEqual(env.initial_identity()["physics"]["space"]["damping"], damping)
            with self.assertRaises(RuntimeError):
                env.reset_scene()
            self.assertEqual(base.reset_calls, 1)
            self.assertEqual(base.space.calls, 10)

    def test_factory_delegates_p0_without_reset_and_rejects_unknown_first(self):
        original, _ = wrapped()
        with patch.object(module, "make_r2a_env", return_value=original) as factory:
            env = module.make_r2a2_env("mock-source", "T", "D-half-life-0.1s")
            factory.assert_called_once_with("mock-source", "T", "P0")
            self.assertIs(env._base, original._base)
            self.assertFalse(env._reset)
            with self.assertRaises(ValueError):
                module.make_r2a2_env("unused", "T", "not-a-candidate")
            self.assertEqual(factory.call_count, 1)

    def test_failed_damping_assignment_is_terminal_without_a_valid_identity(self):
        env, base = candidate("D-half-life-0.1s")
        with patch.object(R2AEnv, "_identity", side_effect=[
                {"old": "P0-initial-placeholder"}, RuntimeError("mock identity read failure")]):
            with self.assertRaisesRegex(RuntimeError, "mock identity read failure"):
                env.reset_scene()
        with self.assertRaises(RuntimeError):
            env.initial_identity()
        with self.assertRaises(RuntimeError):
            env.step((300., 256.))
        self.assertEqual(base.space.calls, 0)

    def test_all_original_physics_and_observer_sources_byte_unchanged(self):
        root = Path(module.__file__).resolve().parents[1]
        for relative, expected in {
            "tdyssey/native_rgb_env.py": "47ee5c53ed3369da7548ea1e2512e5cf1eed315e68b290dcf2fa1ee7dd5500d5",
            "tdyssey/native_rgb_r2a_env.py": "0b2965801c310334ebfa0270eb33abb6da3e731d8f7b7b97ff24c6709c6642e4",
            "tests/test_native_rgb_r2a_env.py": "46f0863ddcbdeb31e6198b8aac4dc51aa4d373fe2bd7a92f73394afaad6e709a",
        }.items():
            self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
