"""Extend the pinned GPC PushT image environment through its add_object hook.

No external imports happen until an I/L body is constructed. Pass the upstream
PushTImageEnv class from an authorized, pinned GPC checkout to the factory below.
"""

from __future__ import annotations

import hashlib
import json

from ..geometry import Pose, get_shape, point
from ..state import BodyState, Snapshot


def _read_body(body) -> BodyState:
    return BodyState(Pose(tuple(body.position), body.angle), tuple(body.velocity),
                     body.angular_velocity, tuple(body.force), body.torque)


def _write_body(body, state: BodyState):
    # Angle first: changing angle can move the body origin when CoG is offset.
    body.angle = state.pose.angle
    body.position = state.pose.position
    body.velocity = state.velocity
    body.angular_velocity = state.angular_velocity
    body.force = state.force
    body.torque = state.torque


def make_shape_env_class(base_class, *, pymunk_module=None, pygame_module=None):
    """Return a subclass; inherited controller, physics, image API, and reward stay.

    T delegates its body creation to the upstream implementation. I/L use mass 2
    and area-weighted uniform-density inertia with centroid-based local vertices.
    Module arguments support construction-contract tests, not fake physics claims.
    """

    class PushShapeImageEnv(base_class):
        def __init__(self, *args, shape="T", **kwargs):
            self.project_shape = get_shape(shape).name
            self.control_step = 0
            super().__init__(*args, **kwargs)
            if self.object != "T":
                raise ValueError("Use the upstream T domain config as the common starting point")
            if getattr(self, "num_obstacle", 0) or getattr(self, "num_mislead", 0):
                raise ValueError("First project adapter supports the obstacle-free T domain only")
            if self.project_shape != "T" and self.block_cog is not None:
                raise ValueError("I/L use their geometric centroid; a custom CoG needs a separate protocol")
            self.object = self.project_shape

        def add_object(self, position, angle, object, scale=30, color="LightSlateGray"):
            if object not in ("I", "L"):
                return super().add_object(position, angle, object, scale, color)
            pm, pg = pymunk_module, pygame_module
            if pm is None or pg is None:
                try:
                    import pymunk
                    import pygame
                except ImportError as exc:
                    raise RuntimeError("I/L construction needs the authorized project's Pymunk/Pygame setup") from exc
                pm, pg = pm or pymunk, pg or pygame
            shape = get_shape(object, scale)
            body = pm.Body(2.0, shape.inertia_about_origin(2.0))
            polygons = [pm.Poly(body, rect.vertices) for rect in shape.components]
            for polygon in polygons:
                polygon.color = pg.Color(color)
            body.center_of_gravity = (0.0, 0.0)
            body.position = position
            body.angle = angle
            # Preserve the upstream material-setting convention. Do not silently
            # change shape friction or the contact law while adding new geometry.
            body.friction = 1
            self.space.add(body, *polygons)
            return body

        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            self.control_step = 0
            return result

        def step(self, action):
            if action is not None:
                point(action)  # Validate finite coordinates; no clipping/normalizer change.
            result = super().step(action)
            if action is not None:
                self.control_step += 1
            return result

        def _physics_hash(self):
            if self.space is None:
                raise RuntimeError("Reset the environment before capturing state")
            payload = {
                "shape": self.object, "scale": self.block_scale,
                "sim_hz": self.sim_hz, "control_hz": self.control_hz,
                "k_p": self.k_p, "k_v": self.k_v,
                "damping": self.space.damping, "gravity": list(self.space.gravity),
                "mass": self.block.mass, "moment": self.block.moment,
                "center_of_gravity": list(self.block.center_of_gravity),
                "pusher_radius": next(iter(self.agent.shapes)).radius,
                "legacy": self.legacy,
            }
            return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

        def capture_state(self) -> Snapshot:
            shape = get_shape(self.object, self.block_scale)
            physics_hash = self._physics_hash()
            return Snapshot(shape.name, shape.geometry_hash, physics_hash,
                            _read_body(self.block), _read_body(self.agent),
                            Pose(tuple(self.goal_pose[:2]), self.goal_pose[2]), self.control_step,
                            None if self.latest_action is None else point(self.latest_action))

        def restore_state(self, snapshot: Snapshot):
            shape = get_shape(self.object, self.block_scale)
            if snapshot.shape != shape.name or snapshot.geometry_hash != shape.geometry_hash:
                raise ValueError("Snapshot belongs to a different shape/geometry")
            if self.space is not None and snapshot.physics_hash != self._physics_hash():
                raise ValueError("Snapshot belongs to a different physics configuration")
            # Rebuild without upstream _set_state, which advances an uncounted
            # physics substep and omits velocity/force restoration.
            self._setup()
            if self.block_cog is not None:
                self.block.center_of_gravity = self.block_cog
            if self.damping is not None:
                self.space.damping = self.damping
            if snapshot.physics_hash != self._physics_hash():
                raise ValueError("Snapshot belongs to a different physics configuration")
            _write_body(self.block, snapshot.object_body)
            _write_body(self.agent, snapshot.pusher_body)
            self.space.reindex_shapes_for_body(self.block)
            self.space.reindex_shapes_for_body(self.agent)
            self.goal_pose[:] = (*snapshot.goal.position, snapshot.goal.angle)
            self.latest_action = snapshot.pusher_target
            self.control_step = snapshot.control_step
            self.render_cache = None
            return self._get_obs(), self._get_info()

    return PushShapeImageEnv
