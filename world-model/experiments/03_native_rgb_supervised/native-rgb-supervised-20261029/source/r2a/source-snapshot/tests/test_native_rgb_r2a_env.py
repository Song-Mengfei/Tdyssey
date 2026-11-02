"""R2A pure mock/source contracts: never construct or step a real simulator."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import box

from tdyssey import native_rgb_r2a_env as module


class Body:
    def __init__(self, static=False):
        self.body_type = 2 if static else 0
        self.mass = float("inf") if static else 1.
        self.moment = float("inf") if static else 3000.
        self.center_of_gravity = (0., 0.)
        self.position, self.velocity, self.force = (256., 256.), (0., 0.), (0., 0.)
        self.angle = self.angular_velocity = self.torque = 0.

    def world_to_local(self, point):
        return np.asarray(point) - self.position


def shape(body, kind="poly"):
    result = NS(body=body, friction=.17, elasticity=.23, sensor=False,
        collision_type=0, filter=NS(group=0, categories=4294967295, mask=4294967295),
        surface_velocity=(0., 0.), radius=0.)
    if kind == "poly":
        result.get_vertices = lambda: [(-5., -5.), (5., -5.), (5., 5.), (-5., 5.)]
    elif kind == "circle":
        result.offset, result.radius = (0., 0.), 15.
    else:
        result.a, result.b, result.radius = (5., 5.), (506., 5.), 2.
    return result


class FakeSpace:
    """A counter and deterministic state mutator, not a physics implementation."""
    def __init__(self, owner):
        self.owner, self.calls = owner, 0
        self.damping, self.iterations, self.collision_slop = 0., 50, .1
        self.collision_bias, self.collision_persistence = .001, 3
        self.sleep_time_threshold, self.idle_speed_threshold = float("inf"), 0.
        self.gravity, self.threads = (0., 0.), 1
        self.shapes = [owner.agent_shape, owner.block_shape, owner.wall_shape]

    def step(self, dt):
        self.calls += 1
        if self.owner.events:
            for pair in self.owner.events:
                arbiter = Arbiter(pair)
                self.owner.collision_handeler.pre_solve(arbiter, self, {})
                arbiter.post = True
                self.owner.collision_handeler.post_solve(arbiter, self, {})
        self.owner.agent.position = tuple(np.asarray(self.owner.agent.position) + (.01, 0.))
        self.owner.agent.force = (0., 0.)  # Engine-clear observation only.
        if self.owner.nonfinite_at == self.calls:
            self.owner.block.angle = float("nan")
        if self.owner.engine_error_at == self.calls:
            raise RuntimeError("mock engine error")


class Arbiter:
    def __init__(self, shapes):
        self.shapes, self.post = shapes, False
        self.contact_point_set = NS(normal=(1., 0.), points=[
            NS(point_a=(261., 256.), point_b=(261., 256.), distance=-.1)])

    @property
    def total_impulse(self):
        if not self.post:
            raise AssertionError("Impulse must never be queried during pre_solve")
        return (1.25, -.5)

    @property
    def total_ke(self):
        if not self.post:
            raise AssertionError("KE must never be queried during pre_solve")
        return .3


class FakeBase:
    def __init__(self, **kwargs):
        self.agent, self.block, self.wall = Body(), Body(), Body(True)
        self.agent.moment = float("inf")
        self.agent_shape = shape(self.agent, "circle")
        self.block_shape, self.wall_shape = shape(self.block), shape(self.wall, "segment")
        self.local_geometry = box(-5., -5., 5., 5.)
        self.space = FakeSpace(self)
        self.controller, self.k_p, self.k_v = module.CONTROLLER, 100., 20.
        self.reset_calls = self.step_calls = self.render_calls = self.control_step = 0
        self.pre_calls = self.post_calls = 0
        self.penetration_at = self.nonfinite_at = self.engine_error_at = None
        self.events = []
        self.collision_handeler = NS(pre_solve=self.pre, post_solve=self.post)

    def pre(self, *args):
        self.pre_calls += 1
        return True

    def post(self, *args):
        self.post_calls += 1

    def reset_scene(self, **kwargs):
        self.reset_calls += 1
        return np.zeros((3, 96, 96), dtype=np.uint8)

    def state(self):
        return np.asarray((*self.agent.position, *self.block.position, self.block.angle,
            *self.agent.velocity, *self.block.velocity, self.block.angular_velocity,
            *self.agent.force, *self.block.force, self.block.torque), dtype=np.float64)

    def world_geometry(self):
        return box(251., 251., 261., 261.)

    def penetrations(self):
        return np.asarray((0., 2.01 if self.penetration_at == self.space.calls else .1, 0.))

    def step(self, action):
        # Mock bound method with the original control expressions; delegation,
        # observation and stopping are tested, not physical behavior.
        self.step_calls += 1
        for _ in range(10):
            velocity = np.asarray(self.agent.velocity)
            acceleration = self.k_p*(action-np.asarray(self.agent.position))-self.k_v*velocity
            force = self.agent.mass*acceleration
            norm = float(np.linalg.norm(force))
            if norm > 10000.:
                force *= 10000./norm
            self.agent.force = tuple(force)
            speed = float(np.linalg.norm(velocity))
            if speed > 200.:
                self.agent.velocity = tuple(velocity*200./speed)
            self.space.step(.01)
        self.control_step += 1
        self.render_calls += 1
        return np.zeros((3, 96, 96), dtype=np.uint8), dict(contact=0, contact_points=[],
            contact_angles=[], penetration=np.asarray((0., .1, 0.)), valid=True,
            physics_substeps=10, control_step=self.control_step)


def wrapped(arm="P0"):
    base = FakeBase()
    segments = [dict(id=0, ring=0, arc_start=0., arc_end=40.)]
    env = module.R2AEnv(base, "T", arm, segments, [base.local_geometry.exterior])
    env.reset_scene()
    return env, base


class R2AEnvTests(unittest.TestCase):
    def test_pair_identity_only_damping_and_observed_shape_materials(self):
        p0, b0 = wrapped("P0")
        p1, b1 = wrapped("P1")
        left, right = p0.initial_identity(), p1.initial_identity()
        self.assertEqual(left["initial_state"], right["initial_state"])
        lp, rp = deepcopy(left["physics"]), deepcopy(right["physics"])
        self.assertEqual(lp["space"].pop("damping"), 0.)
        self.assertEqual(rp["space"].pop("damping"), 1.)
        self.assertEqual(lp, rp)
        self.assertEqual(left["physics"]["agent"]["moment"], "infinity")
        self.assertTrue(all(row["friction"] == .17 and row["elasticity"] == .23
            for row in lp["shapes"]))
        json.dumps(left, allow_nan=False)
        self.assertEqual(left["protocol_sha256"], module.PROTOCOL_SHA256)
        self.assertEqual(right["candidate_protocol"]["sole_physics_delta"],
            dict(parameter="space.damping", original=0., candidate=1.))
        self.assertNotEqual(right["protocol_sha256"], left["protocol_sha256"])
        self.assertEqual(b0.space.calls + b1.space.calls, 0)
        with self.assertRaises(RuntimeError):
            p0.reset_scene()
        with self.assertRaises(AttributeError):
            p0.set_render_state

    def test_delegation_ten_observations_force_cap_and_original_fields(self):
        env, base = wrapped()
        base.agent.velocity = (300., 400.)
        rgb, info = env.step((511., 511.))
        self.assertEqual(base.step_calls, 1)
        self.assertEqual(base.space.calls, 10)
        self.assertEqual(info["substep_count"], 10)
        self.assertEqual(rgb.shape, (3, 96, 96))
        self.assertEqual(info["physics_substeps"], 10)
        first = env.last_substeps[0]
        self.assertEqual(first["velocity_before_cap"], [300., 400.])
        self.assertEqual(first["velocity_after_cap"], [120., 160.])
        np.testing.assert_allclose(first["clipped_body_force"],
            first["exact_formula_reconstruction"]["clipped_force"])
        self.assertEqual(first["post_state"][10:12], [0., 0.])
        self.assertEqual(env.last_substeps[1]["pre_state"], first["post_state"])
        self.assertEqual(first["integrator_pre_state"][5:7], [120., 160.])

    def test_immediate_penetration_abort_no_remaining_substeps_or_render(self):
        env, base = wrapped()
        base.penetration_at = 3
        with self.assertRaises(module.R2AStepAborted) as raised:
            env.step((300., 256.))
        self.assertEqual(base.space.calls, 3)
        self.assertEqual(len(env.last_substeps), 3)
        self.assertEqual(base.render_calls, 0)
        self.assertEqual(base.control_step, 0)
        self.assertEqual(raised.exception.info["max_penetrations"], [0., 2.01, 0.])
        with self.assertRaises(RuntimeError):
            env.step((300., 256.))
        self.assertEqual(base.space.calls, 3)

    def test_nonfinite_and_engine_exception_preserve_partial_evidence(self):
        for field, index in (("nonfinite_at", 2), ("engine_error_at", 4)):
            with self.subTest(field=field):
                env, base = wrapped()
                setattr(base, field, index)
                with self.assertRaises(module.R2AStepAborted):
                    env.step((300., 256.))
                self.assertEqual(base.space.calls, index)
                self.assertEqual(len(env.last_substeps), index)
                self.assertEqual(base.render_calls, 0)
                if field == "nonfinite_at":
                    self.assertTrue(np.isnan(env.last_substeps[-1]["post_state"][4]))

    def test_callbacks_preserve_original_and_post_only_impulse_and_real_arc(self):
        env, base = wrapped()
        base.events = [(base.agent_shape, base.block_shape),
                       (base.block_shape, base.wall_shape), (base.agent_shape, base.wall_shape)]
        _, info = env.step((300., 256.))
        self.assertEqual(base.pre_calls, 30)
        self.assertEqual(base.post_calls, 30)
        for kind in ("agent_object", "object_wall", "agent_wall"):
            self.assertEqual(info[kind + "_contact_substeps"], 10)
        self.assertEqual(info["actual_contact_arc_ids"], [0])
        events = env.last_substeps[0]["contacts"]
        self.assertEqual(len(events), 6)
        self.assertNotIn("total_impulse", events[0])
        self.assertEqual(events[1]["total_impulse"], [1.25, -.5])
        self.assertEqual(events[1]["points"][0]["object_local"], [5., 0.])

    def test_callback_recording_error_stops_after_current_substep(self):
        env, base = wrapped()
        base.events = [(base.agent_shape, base.block_shape)]
        with patch.object(env, "_contact", side_effect=ValueError("mock callback fault")):
            with self.assertRaises(module.R2AStepAborted) as raised:
                env.step((300., 256.))
        self.assertIn("mock callback fault", str(raised.exception))
        self.assertEqual(base.space.calls, 1)

    def test_postcapture_failure_retains_executed_count_and_unknown_penetration(self):
        env, base = wrapped()
        with patch.object(base, "penetrations", side_effect=ValueError("mock geometry error")):
            with self.assertRaises(module.R2AStepAborted) as raised:
                env.step((300., 256.))
        self.assertEqual(base.space.calls, 1)
        self.assertEqual(raised.exception.info["substep_count"], 1)
        self.assertEqual(len(env.last_substeps[-1]["post_state"]), 15)
        self.assertIsNone(env.last_substeps[-1]["penetrations"])
        self.assertEqual(raised.exception.info["max_penetrations"], [None, None, None])
        self.assertIn("mock geometry error", env.last_substeps[-1]["capture_error"][0])
        self.assertEqual(base.render_calls, 0)

    def test_factory_only_uses_original_native_factory_and_rejects_heldout(self):
        fake = FakeBase()
        with patch.object(module, "make_env", return_value=fake) as factory:
            env = module.make_r2a_env("mock-pinned-source", "T", "P0")
            self.assertIs(env._base, fake)
            factory.assert_called_once_with("mock-pinned-source", letter="T",
                controller=module.CONTROLLER, render_size=96)
            self.assertEqual(fake.reset_calls, 0)
            self.assertEqual(fake.space.calls, 0)
            with self.assertRaises(ValueError):
                module.make_r2a_env("unused", "D", "P0")
            with self.assertRaises(ValueError):
                module.make_r2a_env("unused", "T", "unknown")
            self.assertEqual(factory.call_count, 1)

    def test_frozen_physics_source_unchanged(self):
        source = Path(module.__file__).with_name("native_rgb_env.py")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
            "47ee5c53ed3369da7548ea1e2512e5cf1eed315e68b290dcf2fa1ee7dd5500d5")


if __name__ == "__main__":
    unittest.main()
