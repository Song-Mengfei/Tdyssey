"""R2A-only observations around the unchanged native finite-force step.

No copied integration loop: the native bound ``step`` calls an observed
Space.step.  Formula reconstruction is labelled separately from engine reads.
Importing this module does not import or construct a simulator.
"""
from __future__ import annotations

import hashlib
import json
import math

from .native_rgb_env import CONTROL_PROTOCOL, CONTROLLER, PROTOCOL_SHA256, STATE_COLUMNS, make_env

TELEMETRY_VERSION = "tdyssey.r2a-substeps.v1"
ARMS = {"P0": 0.0, "P1": 1.0}
P1_PROTOCOL = dict(version="finite-force-pd-v1-damping1-candidate-v1",
    base_protocol_sha256=PROTOCOL_SHA256,
    sole_physics_delta=dict(parameter="space.damping", original=0.0, candidate=1.0))
P1_PROTOCOL_SHA256 = hashlib.sha256(json.dumps(P1_PROTOCOL, sort_keys=True).encode()).hexdigest()


class R2AStepAborted(RuntimeError):
    """A terminal control failure; completed substeps remain available."""
    def __init__(self, message, info):
        super().__init__(message)
        self.info = info


def _scalar(value):
    value = float(value)
    if math.isnan(value):
        raise ValueError("NaN in physical identity")
    return value if math.isfinite(value) else ("infinity" if value > 0 else "-infinity")


def _xy(value):
    return [float(value[0]), float(value[1])]


def _body_identity(body):
    return dict(body_type=int(body.body_type), mass=_scalar(body.mass),
                moment=_scalar(body.moment), center_of_gravity=_xy(body.center_of_gravity))


def _shape_geometry(shape):
    if hasattr(shape, "get_vertices"):
        return dict(kind="poly", vertices=[_xy(v) for v in shape.get_vertices()],
                    radius=float(getattr(shape, "radius", 0.0)))
    if hasattr(shape, "a") and hasattr(shape, "b"):
        return dict(kind="segment", a=_xy(shape.a), b=_xy(shape.b), radius=float(shape.radius))
    if hasattr(shape, "offset") and hasattr(shape, "radius"):
        return dict(kind="circle", offset=_xy(shape.offset), radius=float(shape.radius))
    raise TypeError("Unsupported physical collision shape")


class R2AEnv:
    """Single-initialization trajectory wrapper; P0/P1 share all observations."""
    def __init__(self, base, letter, arm, segments, rings):
        if arm not in ARMS:
            raise ValueError("R2A arm must be P0 or P1")
        self._base, self.letter, self.arm = base, letter, arm
        self._segments, self._rings = segments, rings
        self._reset = self._terminal = self._in_step = False
        self.last_substeps = []
        self._events = []
        self._callback_error = None

    def __getattr__(self, name):
        if name in ("_setup", "reset", "set_render_state"):
            raise AttributeError("R2A permits only one initial reset_scene")
        return getattr(self._base, name)

    def reset_scene(self, *args, **kwargs):
        if self._reset:
            raise RuntimeError("R2A initial scene cannot be reset or continued by placement")
        # Mark attempted initialization terminal on any failure, including render.
        self._reset = True
        try:
            rgb = self._base.reset_scene(*args, **kwargs)
            if float(self._base.space.damping) != 0.0:
                raise ValueError("Original P0 space damping must be exactly zero")
            if self.arm == "P1":
                self._base.space.damping = 1.0
            self._native_space_step = self._base.space.step
            self._base.space.step = self._observe_space_step
            handler = self._base.collision_handeler
            self._native_pre, self._native_post = handler.pre_solve, handler.post_solve
            handler.pre_solve, handler.post_solve = self._pre_contact, self._post_contact
            self._initial_identity = self._identity()
            return rgb
        except BaseException:
            self._terminal = True
            raise

    def initial_identity(self):
        if not self._reset or not hasattr(self, "_initial_identity"):
            raise RuntimeError("Initial identity is available only after successful initial reset")
        # Return an independent strict-JSON tree, not live mutable engine objects.
        return json.loads(json.dumps(self._initial_identity, allow_nan=False))

    def _role(self, body):
        if body is self._base.agent:
            return "agent"
        if body is self._base.block:
            return "object"
        if int(body.body_type) == 2:
            return "wall"
        raise ValueError("Unexpected dynamic body in R2A collision space")

    def _identity(self):
        base, space = self._base, self._base.space
        shapes = []
        for shape in space.shapes:
            filt = shape.filter
            shapes.append(dict(role=self._role(shape.body), body=_body_identity(shape.body),
                geometry=_shape_geometry(shape), friction=float(shape.friction),
                elasticity=float(shape.elasticity), sensor=bool(shape.sensor),
                collision_type=int(shape.collision_type),
                filter=dict(group=int(filt.group), categories=int(filt.categories), mask=int(filt.mask)),
                surface_velocity=_xy(shape.surface_velocity)))
        shapes.sort(key=lambda row: json.dumps(row, sort_keys=True, allow_nan=False))
        settings = {name: _scalar(getattr(space, name)) for name in (
            "damping", "iterations", "collision_slop", "collision_bias", "collision_persistence",
            "sleep_time_threshold", "idle_speed_threshold")}
        settings["gravity"] = _xy(space.gravity)
        settings["threads"] = int(getattr(space, "threads", 1))
        physics = dict(controller=base.controller, base_control_protocol=CONTROL_PROTOCOL,
            kp=float(base.k_p), kv=float(base.k_v), control_dt=.1, substep_dt=.01,
            substeps_per_control=10, space=settings, agent=_body_identity(base.agent),
            object=_body_identity(base.block), shapes=shapes)
        return dict(kind=TELEMETRY_VERSION, arm=self.arm, letter=self.letter,
            protocol_version=CONTROLLER if self.arm == "P0" else P1_PROTOCOL["version"],
            protocol_sha256=PROTOCOL_SHA256 if self.arm == "P0" else P1_PROTOCOL_SHA256,
            base_protocol_sha256=PROTOCOL_SHA256,
            candidate_protocol=None if self.arm == "P0" else P1_PROTOCOL,
            physics=physics, physics_sha256=hashlib.sha256(json.dumps(physics,
                sort_keys=True, allow_nan=False).encode()).hexdigest(),
            initial_state=base.state().tolist(), state_columns=list(STATE_COLUMNS),
            force_observation="unclipped exact-formula reconstruction; clipped engine property at integrator entry",
            impulse_convention="native arbiter.total_impulse vector and native shapes order; no sign relabelling",
            arc_mapping="canonical contour_segments(letter,8); nearest exact ring within 2 world pixels")

    def signed_wall_gaps(self):
        minx, miny, maxx, maxy = self._base.world_geometry().bounds
        return dict(left=float(minx-7), right=float(504-maxx), top=float(miny-7), bottom=float(504-maxy))

    def _arc_ids(self, local_point):
        from shapely.geometry import Point
        point = Point(*local_point)
        distances = [float(ring.distance(point)) for ring in self._rings]
        if not distances:
            return []
        ring_id = min(range(len(distances)), key=distances.__getitem__)
        if distances[ring_id] > 2.0 + 1e-6:
            return []
        arc = float(self._rings[ring_id].project(point))
        # Half-open bins except the final endpoint; exactly one canonical bin.
        matches = [row["id"] for row in self._segments if row["ring"] == ring_id
            and row["arc_start"] <= arc < row["arc_end"]]
        if not matches:
            matches = [row["id"] for row in self._segments if row["ring"] == ring_id
                and abs(row["arc_end"] - arc) <= 1e-9]
        return matches[-1:] if matches else []

    def _contact(self, arbiter, phase):
        bodies = [shape.body for shape in arbiter.shapes]
        roles = [self._role(body) for body in bodies]
        kinds = {frozenset(("agent", "object")): "agent_object",
                 frozenset(("object", "wall")): "object_wall",
                 frozenset(("agent", "wall")): "agent_wall"}
        kind = kinds.get(frozenset(roles))
        if kind is None:
            raise ValueError(f"Unexpected contact roles: {roles}")
        cp = arbiter.contact_point_set
        points, arc_ids = [], set()
        for point in cp.points:
            row = dict(point_a=_xy(point.point_a), point_b=_xy(point.point_b), distance=float(point.distance))
            if "object" in roles:
                world = point.point_a if roles[0] == "object" else point.point_b
                local = self._base.block.world_to_local(world)
                ids = self._arc_ids(local)
                row.update(object_world=_xy(world), object_local=_xy(local), arc_ids=ids)
                arc_ids.update(ids)
            points.append(row)
        record = dict(phase=phase, kind=kind, shapes_order=roles, normal=_xy(cp.normal),
            points=points, arc_ids=sorted(arc_ids))
        if phase == "post_solve":
            record["total_impulse"] = _xy(arbiter.total_impulse)
            record["total_ke"] = float(arbiter.total_ke)
        return record

    def _record_contact(self, arbiter, phase):
        try:
            record = self._contact(arbiter, phase)
            self._events.append(record)
            self._assert_finite(record)
        except Exception as exc:
            # CFFI callbacks may swallow Python exceptions: check after Space.step.
            self._callback_error = f"{type(exc).__name__}: {exc}"

    def _pre_contact(self, arbiter, space, data):
        try:
            result = self._native_pre(arbiter, space, data) if self._native_pre else True
        except Exception as exc:
            self._callback_error = f"Original pre_solve failed: {type(exc).__name__}: {exc}"
            return False
        self._record_contact(arbiter, "pre_solve")
        return result

    def _post_contact(self, arbiter, space, data):
        try:
            if self._native_post:
                self._native_post(arbiter, space, data)
        except Exception as exc:
            self._callback_error = f"Original post_solve failed: {type(exc).__name__}: {exc}"
            return
        self._record_contact(arbiter, "post_solve")

    @staticmethod
    def _assert_finite(value):
        if isinstance(value, dict):
            for item in value.values():
                R2AEnv._assert_finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                R2AEnv._assert_finite(item)
        elif isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("Nonfinite R2A substep telemetry")

    def _summary(self):
        completed = [row for row in self.last_substeps if row.get("executed")]
        maxima = ([None, None, None] if any(row.get("penetrations") is None for row in completed)
            else [max((row["penetrations"][i] for row in completed), default=0.0) for i in range(3)])
        result = dict(telemetry_version=TELEMETRY_VERSION, substep_count=len(completed),
            max_penetrations=maxima, actual_contact_arc_ids=sorted({arc for row in completed
                for event in row["contacts"] if event["kind"] == "agent_object" for arc in event["arc_ids"]}))
        for kind in ("agent_object", "object_wall", "agent_wall"):
            result[kind + "_contact_substeps"] = sum(any(event["kind"] == kind
                for event in row["contacts"]) for row in completed)
        return result

    def _abort(self, reason):
        self._terminal = True
        info = self._summary()
        info.update(valid=False, aborted=True, reason=str(reason),
            control_step=int(self._base.control_step), substeps=self.last_substeps)
        return R2AStepAborted(str(reason), info)

    def _observe_space_step(self, dt):
        import numpy as np
        if not self._in_step or float(dt) != .01:
            raise self._abort("Unexpected physics entry or time step")
        base = self._base
        before = self._previous_state.copy()
        entry = base.state().copy()
        velocity = before[5:7].copy()
        force = base.agent.mass * (base.k_p * (self._action-before[:2]) - base.k_v*velocity)
        unclipped = force.copy()
        norm = float(np.linalg.norm(force))
        if norm > CONTROL_PROTOCOL["force_limit"]:
            force *= CONTROL_PROTOCOL["force_limit"] / norm
        speed = float(np.linalg.norm(velocity))
        capped = velocity.copy()
        if speed > CONTROL_PROTOCOL["pre_step_speed_limit"]:
            capped *= CONTROL_PROTOCOL["pre_step_speed_limit"] / speed
        measured_force = np.asarray(base.agent.force, dtype=np.float64)
        measured_velocity = np.asarray(base.agent.velocity, dtype=np.float64)
        if not (np.allclose(force, measured_force, rtol=1e-12, atol=1e-10)
                and np.allclose(capped, measured_velocity, rtol=1e-12, atol=1e-10)):
            raise self._abort("Original PD force/cap disagrees with exact-formula reconstruction")
        row = dict(index=len(self.last_substeps), dt=float(dt), executed=False,
            pre_state=before.tolist(), integrator_pre_state=entry.tolist(),
            exact_formula_reconstruction=dict(unclipped_force=unclipped.tolist(), clipped_force=force.tolist()),
            clipped_body_force=measured_force.tolist(), velocity_before_cap=velocity.tolist(),
            velocity_after_cap=measured_velocity.tolist(), contacts=[])
        self._events, self._callback_error = [], None
        try:
            self._assert_finite(row)
        except Exception as exc:
            raise self._abort(exc) from exc
        try:
            self._native_space_step(dt)
            row["executed"] = True
        except BaseException as exc:
            row["engine_exception"] = f"{type(exc).__name__}: {exc}"
            # Entry into the engine is charged even if the engine itself failed.
            row["executed"] = True
            raise self._abort_with_row(row, exc) from exc
        try:
            row["post_state"] = base.state().tolist()
            row["penetrations"] = base.penetrations().tolist()
            row["signed_wall_gaps"] = self.signed_wall_gaps()
            row["contacts"] = self._events
        except BaseException as exc:
            raise self._abort_with_row(row, exc) from exc
        self.last_substeps.append(row)
        self._previous_state = np.asarray(row["post_state"], dtype=np.float64)
        try:
            self._assert_finite(row)
            if self._callback_error:
                raise ValueError(self._callback_error)
            if max(row["penetrations"]) > 2.0:
                raise ValueError("R2A substep penetration exceeds 2 world pixels")
        except Exception as exc:
            raise self._abort(exc) from exc

    def _abort_with_row(self, row, exc):
        # Best-effort raw evidence after a failing engine call; never integrate again.
        errors = []
        for key, capture in (("post_state", lambda: self._base.state().tolist()),
                ("penetrations", lambda: self._base.penetrations().tolist()),
                ("signed_wall_gaps", self.signed_wall_gaps)):
            if key in row:
                continue
            try:
                row[key] = capture()
            except Exception as capture_exc:
                row[key] = None
                errors.append(f"{key}: {type(capture_exc).__name__}: {capture_exc}")
        row["contacts"] = self._events
        row["capture_error"] = errors
        self.last_substeps.append(row)
        return self._abort(exc)

    def step(self, action):
        import numpy as np
        if not self._reset or self._terminal or self._in_step:
            raise RuntimeError("R2A step requires one valid initial reset and a live trajectory")
        self._action = np.asarray(action, dtype=np.float64)
        if self._action.shape != (2,) or not np.isfinite(self._action).all() or (self._action < 0).any() or (self._action > 511).any():
            self._terminal = True
            raise ValueError("Absolute action must be finite xy in [0,511]")
        self.last_substeps = []
        self._previous_state = self._base.state().copy()
        try:
            self._assert_finite(self._previous_state.tolist())
        except Exception as exc:
            raise self._abort(exc) from exc
        self._in_step = True
        try:
            # The original bound method executes the original PD and 10-step loop.
            rgb, info = self._base.step(self._action)
            if len(self.last_substeps) != 10:
                raise self._abort("Original control did not execute exactly 10 substeps")
            info.update(self._summary(), substeps=self.last_substeps)
            return rgb, info
        except R2AStepAborted:
            raise
        except BaseException as exc:
            raise self._abort(exc) from exc
        finally:
            self._in_step = False


def make_r2a_env(source_root, letter, arm):
    """Construct only when the authorized collector explicitly requests an arm."""
    if arm not in ARMS:
        raise ValueError("R2A arm must be P0 or P1")
    if letter not in tuple("ABCE FHIJLMOPSTUVXY".replace(" ", "")):
        raise ValueError("R2A cannot expose held-out letters")
    from .native_rgb_data import contour_segments
    base = make_env(source_root, letter=letter, controller=CONTROLLER, render_size=96)
    geom = base.local_geometry
    return R2AEnv(base, letter, arm, contour_segments(letter, 8), [geom.exterior, *geom.interiors])
