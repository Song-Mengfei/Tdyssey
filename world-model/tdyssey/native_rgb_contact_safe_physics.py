"""Versioned contact-constrained kinematic PushT integration.

The pinned source and OriginalPhysicsEnv remain untouched.  This protocol keeps
the original bodies, zero damping, material, geometry and absolute-target action;
it changes solver iterations and integration/kinematic execution at contact.
There is no object pose projection, velocity clearing, image repair or training.
"""
from __future__ import annotations

import hashlib
import json
import math

from .native_rgb_original_physics import OriginalPhysicsEnv, PROTOCOL as ORIGINAL_PROTOCOL
from .native_rgb_r2a_env import R2AStepAborted

PROTOCOL = dict(ORIGINAL_PROTOCOL)
PROTOCOL.update(
    name="contact-constrained-original-kinematic-rgb.v6",
    solver_iterations=50,
    physics_hz=2000,
    controller_hz=100,
    control_loop="100Hz original PD requested velocity; each integration microstep projects only kinematic velocity into unilateral gap-rate constraints; dynamic body advanced by native solver",
    max_microstep_dt=0.0005,
    microsteps_per_controller_update=20,
    integration_tick_seconds=0.0005,
    integration_step_tick_choices=[1],
    contact_compression_world_px=0.05,
    numerical_overlap_valid_max_world_px=1.0,
    material_overlap_world_px=2.0,
    invalid_center="inside solid or an initially inaccessible closed hole",
    wall_execution="pusher center remains in [22,489]^2; requested target is retained",
    projection="Euclidean velocity projection onto four wall halfplanes and all active convex-component signed-gap rate halfplanes intersected with speed disk of this PD request; contact unit-impulse rigid-body mobility and current point-twist wall budget restrict inward command; never amplify requested speed; infeasible linearized command yields zero velocity and is recorded",
    wall_lookahead="next native point integration plus contact-induced unit-impulse mobility; command-only restriction, no body-state writes",
    post_separation_velocity_reset=False,
    step_guard="supplementary nonfinite/overlap/illegal-center guard; not the collision fix",
    upstream_exact_parity=False,
    render_and_object_pose_repair=False,
)
PROTOCOL_SHA256 = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def project_halfplanes(requested, constraints):
    """Exact 2D projection a.v>=b intersect |v|<=|request|."""
    import numpy as np

    requested = np.asarray(requested, dtype=float)
    constraints = [(np.asarray(a, dtype=float), float(b)) for a, b in constraints]
    radius = float(np.linalg.norm(requested))

    def valid(v):
        return float(v @ v) <= radius * radius + 1e-7 and all(float(a @ v) >= b - 1e-8 for a, b in constraints)

    if valid(requested):
        return requested.copy(), True
    candidates = []
    for a, b in constraints:
        norm = float(a @ a)
        if norm > 1e-20:
            v = requested + (b - float(a @ requested)) / norm * a
            if valid(v):
                candidates.append(v)
            distance = b / math.sqrt(norm)
            if abs(distance) <= radius:
                foot = b / norm * a
                tangent = np.asarray((-a[1], a[0])) / math.sqrt(norm)
                offset = math.sqrt(max(0.0, radius * radius - distance * distance))
                for sign in (-1.0, 1.0):
                    circle_intersection = foot + sign * offset * tangent
                    if valid(circle_intersection):
                        candidates.append(circle_intersection)
    for i, (a, b) in enumerate(constraints):
        for c, d in constraints[i + 1:]:
            det = float(a[0] * c[1] - a[1] * c[0])
            if abs(det) > 1e-10:
                v = np.asarray([(b * c[1] - a[1] * d) / det, (a[0] * d - b * c[0]) / det])
                if valid(v):
                    candidates.append(v)
    if not candidates:
        # An incoming dynamic body may make the current linearized constraints
        # inconsistent against a wall.  This is disclosed, never hidden as pass.
        return np.zeros(2), False
    return min(candidates, key=lambda v: float((v - requested) @ (v - requested))), True


class ContactSafePhysicsEnv(OriginalPhysicsEnv):
    def __init__(self, source_root, letter):
        super().__init__(source_root, letter)
        self.controller = "original-pd-kinematic-contact-constrained-v1"

    def reset_scene(self, *args, **kwargs):
        import numpy as np
        from shapely.geometry import Polygon

        rgb = super().reset_scene(*args, **kwargs)
        self.space.iterations = PROTOCOL["solver_iterations"]
        self._boundary = self.local_geometry.boundary
        geoms = list(self.local_geometry.geoms) if hasattr(self.local_geometry, "geoms") else [self.local_geometry]
        self._holes = [Polygon(ring) for geom in geoms for ring in geom.interiors]
        self._allowed_initial_hole = self._point_region()[0]
        cog = np.asarray(self.block.center_of_gravity)
        self._object_radius = max(float(np.linalg.norm(np.asarray(v) - cog)) for shape in self.block.shapes for v in shape.get_vertices())
        self._wall_vertices_local = np.unique(np.asarray([tuple(v) for shape in self.block.shapes for v in shape.get_vertices()]), axis=0)
        self._last_micro_ticks = 1
        self._initial = self._identity()
        return rgb

    def _identity(self):
        identity = super()._identity()
        identity.update(kind="contact-safe-original-kinematic-native-RGB-identity.v1", protocol_sha256=PROTOCOL_SHA256, protocol=PROTOCOL)
        physics = identity["physics"]
        physics.update(controller=self.controller, substep_dt=0.0005, substeps_per_control=200, controller_dt=0.01)
        identity["physics_sha256"] = hashlib.sha256(json.dumps(physics, sort_keys=True).encode()).hexdigest()
        return identity

    def _physics_hash(self):
        binding = dict(base_body_material_physics_sha256=self._base._physics_hash(), execution_protocol_sha256=PROTOCOL_SHA256)
        return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()

    def capture_state(self):
        from dataclasses import replace

        return replace(self._base.capture_state(), physics_hash=self._physics_hash())

    def _point_region(self):
        from shapely.geometry import Point

        local = Point(*self.block.world_to_local(self.agent.position))
        if self.local_geometry.contains(local):
            return "solid", local
        for index, hole in enumerate(getattr(self, "_holes", ())):
            if hole.contains(local):
                return f"hole-{index}", local
        return "outside", local

    def _gap_constraint(self, dt):
        import numpy as np
        from shapely.ops import nearest_points

        region, point = self._point_region()
        near = nearest_points(self._boundary, point)[0]
        displacement = np.asarray((point.x - near.x, point.y - near.y), dtype=float)
        length = float(np.linalg.norm(displacement))
        if length < 1e-12:
            return None, dict(region=region, signed_gap=-15.0, degenerate_normal=True)
        normal_local = displacement / length
        signed_distance = -length if region == "solid" else length
        if region == "solid":
            normal_local = -normal_local
        angle = float(self.block.angle)
        normal = np.asarray((math.cos(angle) * normal_local[0] - math.sin(angle) * normal_local[1], math.sin(angle) * normal_local[0] + math.cos(angle) * normal_local[1]))
        near_world = self.block.local_to_world((near.x, near.y))
        surface_velocity = np.asarray(self.block.velocity_at_world_point(near_world))
        gap = signed_distance - 15.0
        lower = float(normal @ surface_velocity) - (gap + PROTOCOL["contact_compression_world_px"]) / dt
        return (normal, lower), dict(region=region, signed_gap=float(gap), normal_outward=normal.tolist(), point_world=list(near_world), surface_velocity=surface_velocity.tolist(), normal_velocity_lower=lower)

    def _velocity(self, requested, dt):
        import numpy as np

        x, y = self.agent.position
        constraints = [((1, 0), (22 - x) / dt), ((-1, 0), (x - 489) / dt), ((0, 1), (22 - y) / dt), ((0, -1), (y - 489) / dt)]
        details = []
        speed = float(np.linalg.norm(requested))
        angle = float(self.block.angle)
        rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
        cog_world = np.asarray(self.block.local_to_world(self.block.center_of_gravity))
        offsets = (self._wall_vertices_local - np.asarray(self.block.center_of_gravity)) @ rotation.T
        points = offsets + cog_world
        tangent_offsets = np.column_stack((-offsets[:, 1], offsets[:, 0]))
        point_velocity = np.asarray(self.block.velocity) + float(self.block.angular_velocity) * tangent_offsets
        gaps = np.column_stack((points[:, 0] - 7, 504 - points[:, 0], points[:, 1] - 7, 504 - points[:, 1]))
        outward_velocity = np.column_stack((-point_velocity[:, 0], point_velocity[:, 0], -point_velocity[:, 1], point_velocity[:, 1]))
        wall_budget = gaps + PROTOCOL["contact_compression_world_px"] - np.maximum(outward_velocity, 0) * dt
        for shape_index, shape in enumerate(self.block.shapes):
            query = shape.point_query(self.agent.position)
            normal = np.asarray(query.gradient, dtype=float)
            if float(normal @ normal) < 1e-10:
                continue
            gap = float(query.distance) - 15.0
            surface_velocity = np.asarray(self.block.velocity_at_world_point(query.point))
            lower = float(normal @ surface_velocity) - (gap + PROTOCOL["contact_compression_world_px"]) / dt
            near_contact = gap <= PROTOCOL["contact_compression_world_px"] + 2 * speed * dt + float(np.linalg.norm(surface_velocity)) * dt
            detail = dict(shape_index=shape_index, signed_gap=gap, normal_outward=normal.tolist(), point_world=list(query.point), surface_velocity=surface_velocity.tolist(), normal_velocity_lower=lower, wall_push_speed_limit=None)
            if lower > -speed - 1e-8:
                constraints.append((normal, lower))
                details.append(detail)
            if near_contact:
                # Unit normal impulse gives an unmodified body's translational
                # and angular contact mobility.  Bound only the incoming
                # kinematic command before it can create an infeasible twist.
                inward = -normal
                lever = np.asarray(query.point) - cog_world
                torque = float(lever[0] * inward[1] - lever[1] * inward[0])
                mobility = 1.0 / float(self.block.mass) + torque * torque / float(self.block.moment)
                unit_point_velocity = inward / float(self.block.mass) + torque / float(self.block.moment) * tangent_offsets
                coefficients = np.column_stack((-unit_point_velocity[:, 0], unit_point_velocity[:, 0], -unit_point_velocity[:, 1], unit_point_velocity[:, 1])) / mobility
                mask = coefficients > 1e-10
                if mask.any():
                    limit = max(0.0, float(np.min(wall_budget[mask] / (coefficients[mask] * dt))))
                    detail["wall_push_speed_limit"] = limit
                    if limit < speed + 1e-8:
                        constraints.append((normal, -limit))
                        if detail not in details:
                            details.append(detail)
        velocity, feasible = project_halfplanes(requested, constraints)
        return velocity, dict(object=details, feasible=bool(feasible), constraint_count=len(constraints), delta_norm=float(np.linalg.norm(velocity - requested)))

    def _microstep(self, dt, requested, controller_index):
        import numpy as np

        velocity, intervention = self._velocity(requested, dt)
        self.agent.velocity = tuple(velocity)
        before = self.state().copy()
        self._events = []
        self._callback_error = None
        row = dict(index=len(self.last_substeps), controller_index=controller_index, dt=float(dt), pre_state=before.tolist(), integrator_pre_state=before.tolist(), unconstrained_velocity=np.asarray(requested).tolist(), executed_velocity=velocity.tolist(), intervention=intervention, executed=False)
        try:
            self._native_step(dt)
            row["executed"] = True
            row.update(post_state=self.state().tolist(), contacts=self._events, penetrations=self.penetrations().tolist(), signed_wall_gaps=self.signed_wall_gaps(), pusher_region=self._point_region()[0])
        finally:
            self.last_substeps.append(row)
        if self._callback_error:
            raise RuntimeError(self._callback_error)
        reasons = []
        if not np.isfinite(self.state()).all():
            reasons.append("nonfinite")
        if max(row["penetrations"]) > PROTOCOL["numerical_overlap_valid_max_world_px"] + 1e-8:
            reasons.append("numerical_overlap_limit")
        region = row["pusher_region"]
        if region == "solid" or (region.startswith("hole-") and region != self._allowed_initial_hole):
            reasons.append("illegal_pusher_center_region")
        if reasons:
            row["invalid_reasons"] = reasons
            raise RuntimeError("contact-safe invalid transition: " + ",".join(reasons))

    def step(self, action):
        import numpy as np

        if not self._reset or self._terminal or self._in_step:
            raise RuntimeError("A live contact-safe session is required")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all() or (action < 0).any() or (action > 511).any():
            raise ValueError("Absolute finite targetxy in[0,511]required")
        self._action = action.copy()
        self.latest_action = action.copy()
        self.last_substeps = []
        self._in_step = True
        self._base.contact_points = []
        self._base.contact_angles = []
        self._base.substep_penetrations = []
        try:
            for controller_index in range(10):
                velocity = np.asarray(self.agent.velocity)
                acceleration = self.k_p * (action - np.asarray(self.agent.position)) - self.k_v * velocity
                requested = velocity + acceleration * 0.01
                for _ in range(PROTOCOL["microsteps_per_controller_update"]):
                    self._microstep(PROTOCOL["max_microstep_dt"], requested, controller_index)
            self._base.control_step += 1
            info = self._info(True)
            return self.render_clean(), info
        except BaseException as exc:
            self._terminal = True
            raise R2AStepAborted(str(exc), self._info(False)) from exc
        finally:
            self._in_step = False

    def _info(self, valid):
        import numpy as np

        rows = self.last_substeps
        elapsed = sum(row["dt"] for row in rows if row["executed"])
        intervention_seconds = sum(row["dt"] for row in rows if row["executed"] and row["intervention"]["delta_norm"] > 1e-8)
        info = dict(valid=bool(valid), aborted=not valid, physics_substeps=len(rows), physics_seconds=elapsed, control_step=self.control_step, substeps=rows, penetration=np.max([r.get("penetrations", [0, 0, 0]) for r in rows], axis=0).tolist() if rows else [0, 0, 0], protocol_sha256=PROTOCOL_SHA256, requested_action=self._action.tolist(), executed_target=list(self.agent.position), contact=len(self._base.contact_points), intervention_microsteps=sum(row["intervention"]["delta_norm"] > 1e-8 for row in rows), intervention_seconds=intervention_seconds, intervention_time_ratio=intervention_seconds / elapsed if elapsed else 0.0, infeasible_constraint_microsteps=sum(not row["intervention"]["feasible"] for row in rows))
        for kind in ("agent_object", "object_wall", "agent_wall"):
            pred = lambda r: any(e["kind"] == kind and e["phase"] == "post_solve" for e in r.get("contacts", ()))
            info[kind + "_contact_substeps"] = sum(pred(row) for row in rows)
            info[kind + "_contact_seconds"] = sum(row["dt"] for row in rows if pred(row))
        info["agent_object_positive_impulse_substeps"] = sum(any(e["kind"] == "agent_object" and e["phase"] == "post_solve" and np.linalg.norm(e["total_impulse"]) > 0 for e in row.get("contacts", ())) for row in rows)
        return info


def make_contact_safe_env(source_root, letter):
    return ContactSafePhysicsEnv(source_root, letter)
