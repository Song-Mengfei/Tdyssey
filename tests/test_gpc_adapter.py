"""Construction/API tests with explicit doubles; these do not test physical motion."""

from dataclasses import replace
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tdyssey.envs.gpc import make_shape_env_class
from tdyssey.geometry import get_shape


class BodyDouble:
    def __init__(self, mass, moment):
        self.mass, self.moment = mass, moment
        self.position = self.velocity = self.force = self.center_of_gravity = (0.0, 0.0)
        self.angle = self.angular_velocity = self.torque = 0.0
        self.shapes = []


class PolyDouble:
    def __init__(self, body, vertices):
        self.body, self.vertices = body, tuple(vertices)


class SpaceDouble:
    def __init__(self):
        self.damping, self.gravity = 0, (0, 0)
        self.added, self.reindexed = [], []

    def add(self, *objects):
        self.added.extend(objects)
        for item in objects:
            if isinstance(item, PolyDouble):
                item.body.shapes.append(item)

    def reindex_shapes_for_body(self, body):
        self.reindexed.append(body)

    def step(self, dt):
        raise AssertionError("Restoring a state must not advance physics")


class GPCBaseDouble:
    def __init__(self, **kwargs):
        self.constructor_arguments = kwargs
        self.object = "T"
        self.block_scale = 30
        self.block_cog = kwargs.get("block_cog")
        self.damping = kwargs.get("damping")
        self.num_obstacle = kwargs.get("num_obstacle", 0)
        self.num_mislead = 0
        self.sim_hz, self.control_hz, self.k_p, self.k_v = 100, 10, 100, 20
        self.legacy = False
        self.space = None
        self.latest_action = None
        self.upstream_body_calls = []

    def add_object(self, position, angle, object, scale, color):
        self.upstream_body_calls.append((position, angle, object, scale, color))
        body = BodyDouble(2, 1234)
        body.position, body.angle = position, angle
        self.space.add(body)
        return body

    def _setup(self):
        self.space = SpaceDouble()
        self.agent = BodyDouble(0, 0)
        self.agent.shapes = [SimpleNamespace(radius=15)]
        self.block = self.add_object((256, 300), 0, self.object, self.block_scale, "gray")
        self.goal_pose = [256, 256, math.pi / 4]

    def reset(self):
        self._setup()
        return self._get_obs(), self._get_info()

    def step(self, action):
        if action is not None:
            self.latest_action = action
        self.last_step_result = (self._get_obs(), 0.25, True, True, self._get_info())
        return self.last_step_result

    def _get_obs(self):
        return {"image": "image-api-sentinel", "agent_pos": self.agent.position}

    def _get_info(self):
        return {"block_pose": (*self.block.position, self.block.angle)}


class AdapterContractChecks(unittest.TestCase):
    def environment(self, shape="T", **kwargs):
        env_class = make_shape_env_class(GPCBaseDouble,
                                        pymunk_module=SimpleNamespace(Body=BodyDouble, Poly=PolyDouble),
                                        pygame_module=SimpleNamespace(Color=lambda name: name))
        return env_class(shape=shape, **kwargs)

    def test_t_uses_upstream_body_and_preserves_step_result(self):
        env = self.environment(domain_filename="push_t", resize_scale=96)
        obs, _ = env.reset()
        self.assertEqual(env.constructor_arguments, {"domain_filename": "push_t", "resize_scale": 96})
        self.assertEqual(env.upstream_body_calls, [((256, 300), 0, "T", 30, "gray")])
        self.assertEqual(obs["image"], "image-api-sentinel")
        self.assertEqual(env.block.moment, 1234)  # No silent replacement of upstream T inertia.
        result = env.step((100, 120))
        self.assertIs(result, env.last_step_result)
        self.assertEqual(env.control_step, 1)
        env.step(None)
        self.assertEqual(env.control_step, 1)

    def test_i_l_attach_all_components_to_one_body(self):
        for name in ("I", "L"):
            with self.subTest(name=name):
                env = self.environment(name)
                env.reset()
                shape = get_shape(name)
                self.assertEqual(env.upstream_body_calls, [])
                self.assertEqual(len(env.block.shapes), len(shape.components))
                self.assertTrue(all(poly.body is env.block for poly in env.block.shapes))
                self.assertEqual([poly.vertices for poly in env.block.shapes], [r.vertices for r in shape.components])
                self.assertEqual(env.block.mass, 2)
                self.assertEqual(env.block.center_of_gravity, (0, 0))

    def test_capture_restore_preserves_motion_force_goal_and_target_without_stepping(self):
        env = self.environment("L")
        env.reset()
        env.block.velocity, env.block.angular_velocity = (2, 3), 0.7
        env.block.force, env.block.torque = (4, 5), 0.8
        env.agent.velocity = (6, 7)
        env.latest_action, env.control_step = (160, 180), 9
        env.goal_pose[:] = [180, 220, 0.3]
        state = env.capture_state()
        env.block.position = (1, 2)
        env.restore_state(state)
        self.assertEqual(env.capture_state(), state)
        self.assertEqual(env.space.reindexed, [env.block, env.agent])

    def test_incompatible_snapshot_does_not_replace_an_active_body(self):
        env = self.environment("I")
        env.reset()
        state, body = env.capture_state(), env.block
        for field in ("geometry_hash", "physics_hash"):
            with self.assertRaises(ValueError):
                env.restore_state(replace(state, **{field: "b" * 64}))
            self.assertIs(env.block, body)

    def test_unsupported_scene_and_centroid_override_are_rejected(self):
        with self.assertRaises(ValueError):
            self.environment("I", block_cog=(1, 2))
        with self.assertRaises(ValueError):
            self.environment(num_obstacle=1)

    def test_missing_backend_never_causes_an_install(self):
        env_class = make_shape_env_class(GPCBaseDouble)
        with patch.dict(sys.modules, {"pymunk": None, "pygame": None}):
            env_class(shape="T").reset()  # T dispatch does not import a new backend.
            with self.assertRaisesRegex(RuntimeError, "authorized"):
                env_class(shape="I").reset()


if __name__ == "__main__":
    unittest.main()
