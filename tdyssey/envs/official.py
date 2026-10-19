"""T-preserving official PushT extension with rigid capital-letter bodies."""

from dataclasses import replace
import hashlib
import json

from ..collection import integer, reset_issues
from ..geometry import Pose, Rectangle, Shape, point
from ..state import BodyState, Snapshot
from ..letters import PATTERNS, block_letter

PROTOCOL = "official-t-l-com-v1"
T_REFERENCE_COG = (0.0, 45.0)


def official_shape(name):
    """Local collision coordinates; screen y increases downward.

    T preserves original body coordinates. L has height 120, total foot width 90,
    and stroke thickness 30; its two nonoverlapping rectangles share one body.
    """
    if name == "T":
        return Shape("T", (Rectangle(-60, 0, 60, 30), Rectangle(-15, 30, 15, 120)), 1, "official_dp")
    if name == "L":
        # Before centering: stem [0,30]x[0,120], foot [30,90]x[90,120].
        # The union's area centroid is (30,75).
        return Shape("L", (Rectangle(-30, -75, 0, 45), Rectangle(0, 15, 60, 45)), 1, "official_l_area_centroid")
    if name == "T_narrow":
        # Same physical center and legacy inertia as T; only cap width changes.
        return Shape(name, (Rectangle(-54, -45, 54, -15), Rectangle(-15, -15, 15, 75)),
                     1, "official_t_reference_center")
    if name in PATTERNS:
        return block_letter(name)
    raise ValueError(f"Unsupported official shape: {name}")


def matching_shape_state(reference, template):
    """Match physical centers/velocities and target centers, not different origins."""
    if (reference.shape != "T" or reference.geometry_hash != official_shape("T").geometry_hash
            or template.shape == "T" or template.geometry_hash != official_shape(template.shape).geometry_hash):
        raise ValueError("Expected official T reference and non-T template snapshots")
    object_center = reference.object_body.pose.transform(T_REFERENCE_COG)
    target_center = reference.goal.transform(T_REFERENCE_COG)
    return replace(template, object_body=replace(reference.object_body, pose=Pose(object_center, reference.object_body.pose.angle)),
                   pusher_body=reference.pusher_body, goal=Pose(target_center, reference.goal.angle),
                   control_step=reference.control_step, pusher_target=reference.pusher_target)



def matching_l_state(reference, template):
    """Compatibility name for the original T/L protocol."""
    if template.shape != "L":
        raise ValueError("Expected L template")
    return matching_shape_state(reference, template)


def _read_body(body):
    return BodyState(Pose(tuple(body.position), body.angle), tuple(body.velocity),
                     body.angular_velocity, tuple(body.force), body.torque)


def _write_body(body, state):
    body.angle = state.pose.angle
    body.position = state.pose.position
    body.velocity = state.velocity
    body.angular_velocity = state.angular_velocity
    body.force = state.force
    body.torque = state.torque


def make_official_shape_env_class(base_class):
    class OfficialShapeEnv(base_class):
        def __init__(self, *args, shape="T", **kwargs):
            self.project_shape = official_shape(shape).name
            self.control_step = 0
            super().__init__(*args, **kwargs)
            if self.project_shape != "T" and self.block_cog is not None:
                raise ValueError("Non-T shapes fix their mass center; custom center of gravity is unsupported")

        def add_tee(self, position, angle, scale=30, color="LightSlateGray", mask=None):
            # Preserve every original T construction argument and its legacy inertia.
            if self.project_shape == "T":
                if mask is None:
                    return super().add_tee(position, angle, scale=scale, color=color)
                return super().add_tee(position, angle, scale=scale, color=color, mask=mask)
            if scale != 30:
                raise ValueError("The letter protocol fixes its geometry at scale 30")
            import pygame
            import pymunk

            shape = official_shape(self.project_shape)
            body = pymunk.Body(1.0, 3000.0 if self.project_shape == "T_narrow" else shape.inertia_about_origin(1.0))
            polygons = [pymunk.Poly(body, rectangle.vertices) for rectangle in shape.components]
            for polygon in polygons:
                polygon.color = pygame.Color(color)
                polygon.filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() if mask is None else mask)
            body.center_of_gravity = (0.0, 0.0)
            body.angle, body.position = angle, position
            body.friction = 1  # Same upstream material-setting convention; no contact-law change.
            self.space.add(body, *polygons)
            return body

        def _setup(self):
            super()._setup()
            if self.project_shape != "T":
                # T's goal origin is not its mass center. Align the L target's
                # center with the original T target's mass center.
                goal = Pose(point(self.goal_pose[:2]), float(self.goal_pose[2]))
                self.goal_pose[:2] = goal.transform(T_REFERENCE_COG)

        def _set_state(self, state):
            if self.project_shape != "T":
                import numpy as np
                state = np.asarray(state, dtype=np.float64).copy()
                angle = float(state[4])
                offset = T_REFERENCE_COG if self.legacy else Pose((0, 0), angle).transform(T_REFERENCE_COG)
                state[2:4] += offset
            return super()._set_state(state)

        def reset(self):
            result = super().reset()
            self.control_step = 0
            return result

        def step(self, action):
            if action is not None:
                point(action)
            result = super().step(action)
            if action is not None:
                self.control_step += 1
            return result

        def _physics_hash(self):
            if self.space is None:
                raise RuntimeError("Reset before capturing physical state")
            data = {"protocol": PROTOCOL, "shape": self.project_shape,
                    "geometry_hash": official_shape(self.project_shape).geometry_hash,
                    "mass": self.block.mass, "moment": self.block.moment,
                    "cog": list(self.block.center_of_gravity), "legacy": self.legacy,
                    "sim_hz": self.sim_hz, "control_hz": self.control_hz,
                    "kp": self.k_p, "kv": self.k_v, "damping": self.space.damping,
                    "gravity": list(self.space.gravity), "pusher_radius": next(iter(self.agent.shapes)).radius,
                    "materials": sorted((p.friction, p.elasticity, p.sensor, p.filter.group,
                                         p.filter.categories, p.filter.mask) for p in self.block.shapes)}
            return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()

        def capture_state(self):
            shape = official_shape(self.project_shape)
            return Snapshot(shape.name, shape.geometry_hash, self._physics_hash(),
                            _read_body(self.block), _read_body(self.agent),
                            Pose(point(self.goal_pose[:2]), float(self.goal_pose[2])), self.control_step,
                            None if self.latest_action is None else point(self.latest_action))

        def restore_state(self, snapshot):
            shape = official_shape(self.project_shape)
            if snapshot.shape != shape.name or snapshot.geometry_hash != shape.geometry_hash:
                raise ValueError("Snapshot belongs to different official shape/geometry")
            if self.space is not None and snapshot.physics_hash != self._physics_hash():
                raise ValueError("Snapshot belongs to different official physics")
            self._setup()
            if self.block_cog is not None:
                self.block.center_of_gravity = self.block_cog
            if self.damping is not None:
                self.space.damping = self.damping
            if snapshot.physics_hash != self._physics_hash():
                raise ValueError("Snapshot belongs to different official physics")
            _write_body(self.block, snapshot.object_body)
            _write_body(self.agent, snapshot.pusher_body)
            self.space.reindex_shapes_for_body(self.block)
            self.space.reindex_shapes_for_body(self.agent)
            self.goal_pose[:] = (*snapshot.goal.position, snapshot.goal.angle)
            self.latest_action, self.control_step = snapshot.pusher_target, snapshot.control_step
            self.render_cache = None
            return self._get_obs()

    return OfficialShapeEnv


def matched_reset(environments, first_seed, counters, attempts=20):
    """Reject unsafe common scenes; count both shapes' resets, not just accepted ones."""
    integer(attempts, "reset candidates", 1, 20)
    integer(first_seed, "first reset seed", 0, 2**32 - attempts)
    if "T" not in environments or len(environments) < 2:
        raise ValueError("Matched evaluation requires T and at least one other shape")
    if any(name != env.project_shape for name, env in environments.items()):
        raise ValueError("Environment names must match their shapes")
    records = []
    for candidate in range(first_seed, first_seed + attempts):
        for env in environments.values():
            env.seed(candidate)
            env.reset()
            counters["reset_physics_substeps"] += 1
        reference = environments["T"].capture_state()
        if tuple(environments["T"].block.center_of_gravity) != T_REFERENCE_COG:
            raise ValueError("Matched scenes require the original T center of gravity")
        states = {name: reference if name == "T" else matching_shape_state(reference, env.capture_state())
                  for name, env in environments.items()}
        issues = {name: reset_issues(official_shape(name), state) for name, state in states.items()}
        records.append({"seed": candidate, "issues": issues})
        if not any(issues.values()):
            observations = {name: environments[name].restore_state(state) for name, state in states.items()}
            return observations, records
    raise ValueError(f"No valid common scene after {attempts} candidates: {records}")
