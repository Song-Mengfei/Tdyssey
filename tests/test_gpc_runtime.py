"""Opt-in real simulator checks. Set TDYSSEY_GPC_ROOT to the pinned checkout.

Default source-only discovery skips these tests without importing third-party
packages. Run with the locked CPU/GPC or CUDA/GPC environment when configured.
"""

from dataclasses import replace
import math
import os
from pathlib import Path
import unittest

from tdyssey.envs.gpc import make_shape_env_class
from tdyssey.evaluation import goal_metrics
from tdyssey.geometry import Pose, get_shape
from tdyssey.state import BodyState


@unittest.skipUnless(os.environ.get("TDYSSEY_GPC_ROOT"), "Set TDYSSEY_GPC_ROOT for real simulator checks")
class GPCRuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Headless rendering without opening a window or requiring an audio device.
        for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                           "MPLBACKEND": "Agg", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
            os.environ.setdefault(key, value)
        from examples.gpc_shape_env import load_upstream
        import numpy as np
        import shapely.geometry as geometry
        from shapely.ops import unary_union

        root = Path(os.environ["TDYSSEY_GPC_ROOT"]).resolve()
        cls.upstream, cls.revision = load_upstream(root)
        cls.np, cls.geometry = np, geometry
        cls.union = staticmethod(unary_union)
        cls.env_class = make_shape_env_class(cls.upstream.PushTImageEnv)
        previous_cwd = Path.cwd()
        cls.addClassCleanup(os.chdir, previous_cwd)
        os.chdir(root / "gpc_rank_evaluation")

    def environment(self, shape="T", seed=100000, original=False):
        if original:
            env = self.upstream.PushTImageEnv(domain_filename="push_t", resize_scale=96)
        else:
            env = self.env_class(shape=shape, domain_filename="push_t", resize_scale=96)
        self.addCleanup(env.close)
        env.seed(seed)
        return env

    def assert_observations_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.np.testing.assert_array_equal(first[key], second[key])

    def test_t_matches_original_images_states_and_contact_rollout(self):
        for seed in (100000, 100001):
            with self.subTest(seed=seed):
                original = self.environment(seed=seed, original=True)
                adapted = self.environment(seed=seed)
                obs0, _ = original.reset()
                obs1, _ = adapted.reset()
                self.assert_observations_equal(obs0, obs1)
                self.assertEqual(original.block.mass, adapted.block.mass)
                self.assertEqual(original.block.moment, adapted.block.moment)
                # Drive toward the block, exercising inherited controller/contact code.
                action = self.np.array(original.block.position)
                contacts = 0
                for step in range(12):
                    out0, out1 = original.step(action), adapted.step(action)
                    self.assert_observations_equal(out0[0], out1[0])
                    self.assertEqual(out0[1:4], out1[1:4])
                    for key in out0[4]:
                        self.np.testing.assert_array_equal(out0[4][key], out1[4][key])
                    contacts += int(out0[4]["n_contacts"])
                    self.assertEqual(adapted.control_step, step + 1)
                self.assertGreater(contacts, 0)

    def test_collision_geometry_and_reward_match_project_metrics(self):
        for name in ("T", "I", "L"):
            with self.subTest(shape=name):
                env = self.environment(name)
                env.reset()
                shape = get_shape(name, env.block_scale)
                current = env.capture_state()
                state = replace(current, object_body=BodyState(Pose((250, 250), 0.2)),
                                pusher_body=BodyState(Pose((100, 100), 0)),
                                goal=Pose((260, 250), 0.3))
                observation, _ = env.restore_state(state)
                physical = self.union([self.geometry.Polygon([
                    tuple(env.block.local_to_world(v)) for v in poly.get_vertices()
                ]) for poly in env.block.shapes])
                expected = self.union([self.geometry.Polygon(poly)
                                       for poly in shape.polygons(state.object_body.pose)])
                self.assertLess(physical.symmetric_difference(expected).area, 1e-7)
                self.assertTrue(all(poly.body is env.block for poly in env.block.shapes))
                metrics = goal_metrics(shape, state.object_body.pose, state.goal)
                _, reward, _, _, _ = env.step(None)
                self.assertAlmostEqual(float(reward), min(metrics.coverage / env.success_threshold, 1), places=9)
                self.assertEqual(observation["image"].shape, (3, 96, 96))
                self.assertTrue(self.np.isfinite(observation["image"]).all())
                self.assertGreater(float(observation["image"].std()), 0.01)
                self.assertEqual(env.control_step, 0)  # Scoring alone does not advance time.

    def test_i_l_contact_actually_moves_the_object(self):
        for name in ("I", "L"):
            with self.subTest(shape=name):
                env = self.environment(name)
                env.reset()
                shape = get_shape(name, env.block_scale)
                pusher_x = 256 + min(r.x0 for r in shape.components) - 20
                state = replace(env.capture_state(),
                                object_body=BodyState(Pose((256, 256), 0)),
                                pusher_body=BodyState(Pose((pusher_x, 256), 0)),
                                goal=Pose((420, 100), 0))
                env.restore_state(state)
                contacts = 0
                for _ in range(10):
                    _, _, _, _, info = env.step(self.np.array([pusher_x + 80, 256]))
                    contacts += int(info["n_contacts"])
                self.assertGreater(contacts, 0)
                self.assertGreater(math.dist(env.block.position, (256, 256)), 1)
                self.assertTrue(self.np.isfinite([*env.block.position, env.block.angle]).all())

    def test_motion_force_goal_and_controller_state_restore(self):
        for name in ("T", "I", "L"):
            with self.subTest(shape=name):
                env = self.environment(name)
                env.reset()
                state = replace(env.capture_state(),
                                object_body=BodyState(Pose((250, 250), 0.3), (2, 3), 0.2, (4, 5), 0.6),
                                pusher_body=BodyState(Pose((100, 100), 0), (1, 2)),
                                goal=Pose((350, 300), 0.8), control_step=7, pusher_target=(140, 130))
                env.restore_state(state)
                self.assertEqual(env.capture_state(), state)
                self.assertEqual(env._get_info()["n_contacts"], 0)

    def test_fresh_precontact_replay_is_repeatable(self):
        # Restore twice before first contact. This does not establish replay from
        # a mid-contact state, whose solver cache is deliberately not serialized.
        for name in ("T", "I", "L"):
            with self.subTest(shape=name):
                env = self.environment(name)
                env.reset()
                shape = get_shape(name, env.block_scale)
                x = 256 + min(r.x0 for r in shape.components) - 20
                state = replace(env.capture_state(),
                                object_body=BodyState(Pose((256, 256), 0)),
                                pusher_body=BodyState(Pose((x, 230), 0)),
                                goal=Pose((420, 100), 0))
                results = []
                for _ in range(2):
                    env.restore_state(state)
                    for _ in range(8):
                        env.step(self.np.array([x + 70, 230]))
                    final = env.capture_state()
                    results.append([*final.object_body.pose.position, final.object_body.pose.angle,
                                    *final.object_body.velocity, final.object_body.angular_velocity])
                self.np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
