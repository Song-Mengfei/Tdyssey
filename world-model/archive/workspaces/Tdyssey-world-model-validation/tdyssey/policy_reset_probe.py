"""Reset-only official T/L scene checks; importing this module starts no runtime.

One probe creates and closes one environment.  It never calls ``env.step`` or
restores a state.  The only permitted physics call is the pinned upstream
``reset -> _set_state -> Space.step(1 / sim_hz)`` settling substep.  Dependency
injection is for tests, and is explicitly identified as unverified provenance.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import threading
import time
from unittest.mock import patch

from .collection import integer, reset_issues
from .envs.official import matching_shape_state, make_official_shape_env_class, official_shape
from .evaluation import goal_metrics
from .state import Snapshot


PROBE_VERSION = "official-reset-only-v1"
PHASES = frozenset(("original_train", "original_evaluation", "candidate_train",
                    "final_train", "final_evaluation"))
EVALUATION_PHASES = frozenset(("original_evaluation", "final_evaluation"))
_INSTRUMENTATION_LOCK = threading.Lock()


class ResetProbeViolation(RuntimeError):
    """A forbidden operation was rejected before calling its implementation."""


class _StepGuard:
    def __init__(self):
        self.env = None
        self.reset_in_flight = False
        self.space_step_attempts = 0
        self.space_step_completed = 0
        self.blocked_space_step_attempts = 0
        self.blocked_control_attempts = 0
        self.violations = []

    def _reject(self, message):
        self.violations.append(message)
        raise ResetProbeViolation(message)

    def reject_control(self, *args, **kwargs):
        self.blocked_control_attempts += 1
        self._reject("env.step is forbidden during reset-only preflight, including action=None")

    def space_step(self, original, space, dt, *args, **kwargs):
        reason = None
        if not self.reset_in_flight:
            reason = "Space.step is forbidden outside the single reset call"
        elif self.space_step_attempts:
            reason = "A reset may attempt at most one settling Space.step"
        elif self.env is None or space is not getattr(self.env, "space", None):
            reason = "Reset attempted to step a different physics space"
        else:
            hz = getattr(self.env, "sim_hz", None)
            if (isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz)
                    or hz <= 0 or isinstance(dt, bool) or not isinstance(dt, (int, float))
                    or not math.isfinite(dt) or dt != 1.0 / hz or args or kwargs):
                reason = "Reset settling must be exactly Space.step(1 / sim_hz)"
        if reason is not None:
            self.blocked_space_step_attempts += 1
            self._reject(reason)
        # Reserve the actual call before physics.  An exception never refunds it.
        self.space_step_attempts += 1
        result = original(space, dt)
        self.space_step_completed += 1
        return result


@contextmanager
def instrument_space_steps(space_class, guard):
    """Instrument a Space class; fake tests can supply the same context factory.

    Example injection: ``space_step_context=lambda g:
    instrument_space_steps(FakeSpace, g)``.  The original call and timestep are
    preserved.  The caller holds the process-local instrumentation lock.
    """
    original = space_class.step

    def guarded_step(space, dt, *args, **kwargs):
        return guard.space_step(original, space, dt, *args, **kwargs)

    with patch.object(space_class, "step", guarded_step):
        yield


@contextmanager
def _official_space_steps(guard):
    import pymunk

    with instrument_space_steps(pymunk.Space, guard):
        yield


@contextmanager
def _no_environment_steps(env_class, guard):
    # Patch every defining class, so a super().step(None) also fails closed.
    owners = [cls for cls in env_class.__mro__ if "step" in cls.__dict__]
    patches = []
    try:
        for owner in owners:
            patches.append(patch.object(owner, "step", guard.reject_control))
            patches[-1].start()
        original_init = env_class.__init__

        def guarded_init(env, *args, **kwargs):
            # Keep a cleanup reference even if __init__ raises after allocating
            # native resources.  This wrapper changes no constructor arguments.
            guard.env = env
            return original_init(env, *args, **kwargs)

        patches.append(patch.object(env_class, "__init__", guarded_init))
        patches[-1].start()
        yield
    finally:
        for item in reversed(patches):
            item.stop()


def _array_summary(value, expected_shape, label):
    """Small, dependency-free validation; never retain the observation pixels."""
    dtype = str(getattr(value, "dtype", "nested_numeric_sequence"))
    reported_shape = getattr(value, "shape", None)
    if reported_shape is not None and tuple(reported_shape) != expected_shape:
        raise ValueError(f"{label} must have shape {expected_shape}")
    if label == "image" and dtype != "nested_numeric_sequence" and dtype != "float32":
        raise ValueError("Official image observations must be float32")
    values = value.tolist() if hasattr(value, "tolist") else value
    bounds = [math.inf, -math.inf]

    def visit(node, depth):
        if depth == len(expected_shape):
            if isinstance(node, bool) or not isinstance(node, (int, float)) or not math.isfinite(node):
                raise ValueError(f"{label} must contain finite numeric values")
            bounds[0] = min(bounds[0], node)
            bounds[1] = max(bounds[1], node)
            return
        if not isinstance(node, (list, tuple)) or len(node) != expected_shape[depth]:
            raise ValueError(f"{label} must have shape {expected_shape}")
        for child in node:
            visit(child, depth + 1)

    visit(values, 0)
    if label == "image" and (bounds[0] < 0 or bounds[1] > 1):
        raise ValueError("Official image observations must be in [0, 1]")
    return {"shape": list(expected_shape), "dtype": dtype,
            "min": float(bounds[0]), "max": float(bounds[1])}


def observation_summary(observation):
    """Check the official visual input without importing or constructing a policy."""
    if not isinstance(observation, dict) or set(observation) != {"image", "agent_pos"}:
        raise ValueError("Expected the official image and agent_pos observation mapping")
    return {"image": _array_summary(observation["image"], (3, 96, 96), "image"),
            "agent_pos": _array_summary(observation["agent_pos"], (2,), "agent_pos")}


def _snapshot(record, expected_shape):
    state = Snapshot.from_dict(record["snapshot"])
    if state.shape != expected_shape or state.geometry_hash != official_shape(expected_shape).geometry_hash:
        raise ValueError("Reset snapshot belongs to another shape or geometry")
    if state.control_step != 0 or state.pusher_target is not None:
        raise ValueError("Reset snapshot must precede every control action")
    return state


def validate_matched_reset_pair(t_record, l_record):
    """Check production matched-COM initial geometry, without restore or physics.

    Raw reset records stay intact.  Both raw resets and both matched states must
    pass.  This says nothing about restored contact caches or policy performance.
    """
    if (t_record.get("shape") != "T" or l_record.get("shape") != "L"
            or t_record.get("seed") != l_record.get("seed")
            or t_record.get("phase") != l_record.get("phase")
            or t_record.get("phase") not in EVALUATION_PHASES
            or t_record.get("error") or l_record.get("error")):
        raise ValueError("Expected same-phase, same-seed successful T/L evaluation resets")
    reference, template = _snapshot(t_record, "T"), _snapshot(l_record, "L")
    matched = {"T": reference, "L": matching_shape_state(reference, template)}
    issues = {name: reset_issues(official_shape(name), state) for name, state in matched.items()}
    metrics = {name: asdict(goal_metrics(official_shape(name), state.object_body.pose, state.goal))
               for name, state in matched.items()}
    return {"protocol": "official-t-l-com-v1", "seed": t_record["seed"],
            "valid": bool(t_record.get("raw_valid", t_record["valid"])
                          and l_record.get("raw_valid", l_record["valid"])
                          and not any(issues.values())),
            "raw_valid": {"T": bool(t_record.get("raw_valid", t_record["valid"])),
                          "L": bool(l_record.get("raw_valid", l_record["valid"]))},
            "issues": issues, "metrics": metrics,
            "snapshots": {name: asdict(state) for name, state in matched.items()},
            "physics_substeps": 0, "control_steps": 0, "state_restored": False}


class RuntimeResetProbe:
    """Callable single-reset probe with fail-closed operation instrumentation.

    Production uses the fully pinned official loader and shape extension, always
    with ``legacy=True``.  Test injection requires an environment *class* to let
    guards cover its constructor too.  A runtime error poisons this instance;
    later calls fail before creating an environment.  Geometry rejections do not.
    Evaluation calls must visit T before L for each phase/seed.
    """

    def __init__(self, source_root, *, env_factory=None, space_step_context=None, metrics_fn=None):
        self.source_root = str(Path(source_root).resolve())
        self._blocked = False
        self._closed = False
        self._evaluation_t = {}
        injected = env_factory is not None
        if not injected:
            if space_step_context is not None or metrics_fn is not None:
                raise ValueError("Production probes may not replace physics instrumentation or metrics")
            from .official import load_official_source, official_spec, physical_metrics

            spec = official_spec()
            env_factory = make_official_shape_env_class(load_official_source(self.source_root, spec))
            metrics_fn = physical_metrics
            self._provenance = {"probe_version": PROBE_VERSION, "source_root": self.source_root,
                                "source_commit": spec["source_commit"],
                                "source_sha256": deepcopy(spec["source_sha256"]),
                                "source_verified": True, "factory": "make_official_shape_env_class",
                                "base_class": "diffusion_policy.env.pusht.pusht_image_env.PushTImageEnv",
                                "legacy": True, "injected_test_dependencies": False}
        else:
            if space_step_context is None or metrics_fn is None:
                raise ValueError("Fake probes must provide their Space instrumentation and metrics")
            self._provenance = {"probe_version": PROBE_VERSION, "source_root": self.source_root,
                                "source_verified": False, "legacy": True,
                                "injected_test_dependencies": True}
        if not isinstance(env_factory, type):
            raise TypeError("env_factory must be a class so constructor control calls are guarded")
        self._env_factory = env_factory
        self._space_step_context = space_step_context or _official_space_steps
        self._metrics_fn = metrics_fn

    @property
    def blocked(self):
        return self._blocked

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def close(self):
        self._evaluation_t.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __call__(self, shape, seed, phase):
        if self._blocked or self._closed:
            raise RuntimeError("Reset probe is blocked or closed; no further reset is permitted")
        if shape not in ("T", "L") or phase not in PHASES:
            raise ValueError("Reset preflight requires a T/L shape and a declared preflight phase")
        integer(seed, "reset seed", 0, 2**32 - 1)
        started = time.perf_counter()
        guard = _StepGuard()
        record = {"shape": shape, "seed": seed, "phase": phase, "valid": False,
                  "raw_valid": False, "issues": [], "reset_attempts": 0,
                  "reset_physics_substeps": 0, "reset_physics_substep_attempts": 0,
                  "control_steps": 0, "error": None, "blocked": False,
                  "classification": "pending", "provenance": self.provenance}
        env = None
        acquired = False
        failure = None
        timings = {}
        try:
            if phase in EVALUATION_PHASES and shape == "L" and (phase, seed) not in self._evaluation_t:
                raise RuntimeError("Probe the same-phase, same-seed T reference before evaluation L")
            acquired = _INSTRUMENTATION_LOCK.acquire(blocking=False)
            if not acquired:
                raise RuntimeError("Another reset probe owns process-wide physics instrumentation")
            with _no_environment_steps(self._env_factory, guard), self._space_step_context(guard):
                try:
                    env = self._env_factory(shape=shape, legacy=True)
                    guard.env = env
                    env.seed(seed)
                    reset_started = time.perf_counter()
                    record["reset_attempts"] = 1
                    guard.reset_in_flight = True
                    try:
                        observation = env.reset()
                    finally:
                        guard.reset_in_flight = False
                        timings["reset_seconds"] = time.perf_counter() - reset_started
                    if guard.space_step_attempts != 1 or guard.space_step_completed != 1:
                        raise ResetProbeViolation("Pinned official reset must complete exactly one settling substep")
                    if guard.violations:
                        raise ResetProbeViolation("Reset swallowed a forbidden operation: " + guard.violations[0])
                    state = env.capture_state()
                    if not isinstance(state, Snapshot):
                        raise ValueError("capture_state must return the project Snapshot record")
                    record["snapshot"] = asdict(state)
                    state = _snapshot(record, shape)
                    record["geometry_hash"] = state.geometry_hash
                    record["physics_hash"] = state.physics_hash
                    record["issues"] = reset_issues(official_shape(shape), state)
                    metrics = self._metrics_fn(env)
                    record["metrics"] = {}
                    for name in ("coverage", "iou"):
                        value = metrics[name]
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1e-9 <= value <= 1 + 1e-9:
                            raise ValueError("Physical reset metrics must be finite fractions")
                        record["metrics"][name] = float(value)
                    if phase.endswith("train") and record["metrics"]["coverage"] > 0.95:
                        record["issues"].append("initial_training_coverage_exceeds_success_threshold")
                    record["observation"] = observation_summary(observation)
                    record["raw_valid"] = record["valid"] = not record["issues"]
                    if phase in EVALUATION_PHASES:
                        if shape == "T":
                            # Only primitive snapshots/metadata survive cleanup, never an env or image.
                            self._evaluation_t[(phase, seed)] = deepcopy(record)
                        else:
                            record["matched_com"] = validate_matched_reset_pair(
                                self._evaluation_t[(phase, seed)], record)
                            for name, issues in record["matched_com"]["issues"].items():
                                record["issues"].extend(f"matched_{name}:{issue}" for issue in issues)
                            record["valid"] = record["valid"] and record["matched_com"]["valid"]
                    record["classification"] = "valid" if record["valid"] else "geometry_invalid"
                except Exception as exc:
                    failure = exc
                finally:
                    if env is None:
                        env = guard.env
                    if env is not None:
                        close_started = time.perf_counter()
                        try:
                            env.close()
                        except Exception as exc:
                            if failure is None:
                                failure = exc
                            else:
                                record["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
                        timings["close_seconds"] = time.perf_counter() - close_started
                    if guard.violations and failure is None:
                        failure = ResetProbeViolation("Forbidden operation during cleanup: " + guard.violations[0])
        except Exception as exc:
            failure = exc
        finally:
            if acquired:
                _INSTRUMENTATION_LOCK.release()
            record.update(reset_physics_substeps=guard.space_step_completed,
                          reset_physics_substep_attempts=guard.space_step_attempts,
                          space_step_attempts=guard.space_step_attempts,
                          space_step_completed=guard.space_step_completed,
                          blocked_space_step_attempts=guard.blocked_space_step_attempts,
                          blocked_control_attempts=guard.blocked_control_attempts,
                          guard_violations=list(guard.violations))
            timings["total_seconds"] = time.perf_counter() - started
            record["timings"] = timings
        if failure is not None:
            self._blocked = True
            self._evaluation_t.clear()
            record.update(valid=False, blocked=True,
                          classification="guard_violation" if isinstance(failure, ResetProbeViolation) else "runtime_error",
                          error={"type": type(failure).__name__, "message": str(failure)})
        return record
