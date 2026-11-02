"""Frozen-policy candidate ranking using exact-history simulator branches.

No model is trained. Each branch starts from the episode's common initial state
and replays actual controls, rebuilding contact history rather than cold-restoring
a mid-contact snapshot. The environment factory must return an independent,
reset, compatible official shape environment accepting pixel-coordinate tuples.
"""

from dataclasses import asdict, dataclass
import math
import time

from .envs.official import official_shape
from .evaluation import goal_metrics
from .geometry import finite, point
from .state import Snapshot


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 8
    settle_steps: int = 2
    prefix_lengths: tuple[int, ...] = (8,)
    include_hold: bool = True
    settling_window: int = 2
    speed_weight: float = 0.01
    angular_speed_weight: float = 0.01
    position_weight: float = 0.1
    angle_weight: float = 0.05
    state_tolerance: float = 1e-7

    def __post_init__(self):
        for name, lower in (("horizon", 1), ("settle_steps", 0), ("settling_window", 1)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= 300:
                raise ValueError(f"{name} must be an integer in [{lower}, 300]")
        prefixes = tuple(self.prefix_lengths)
        if (not prefixes or len(set(prefixes)) != len(prefixes)
                or any(type(n) is not int or not 1 <= n <= self.horizon for n in prefixes)):
            raise ValueError("Prefixes must be distinct positive integers within the horizon")
        if type(self.include_hold) is not bool:
            raise ValueError("include_hold must be boolean")
        object.__setattr__(self, "prefix_lengths", prefixes)
        for name in ("speed_weight", "angular_speed_weight", "position_weight", "angle_weight", "state_tolerance"):
            value = finite(getattr(self, name), name)
            if value < 0 or (name == "state_tolerance" and value == 0):
                raise ValueError(f"{name} must be {'positive' if name == 'state_tolerance' else 'nonnegative'}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PlannerDecision:
    actions: tuple[tuple[float, float], ...]
    report: dict


def state_error(actual, expected):
    """Largest numeric snapshot difference; require matching identity/schema."""
    if not isinstance(actual, Snapshot) or not isinstance(expected, Snapshot):
        raise ValueError("Expected Snapshot records")
    for name in ("shape", "geometry_hash", "physics_hash", "control_step", "schema_version"):
        if getattr(actual, name) != getattr(expected, name):
            raise ValueError(f"Replay mismatch in {name}")
    if (actual.pusher_target is None) != (expected.pusher_target is None):
        raise ValueError("Replay mismatch in pusher_target availability")
    differences = []
    for a, b in ((actual.object_body, expected.object_body), (actual.pusher_body, expected.pusher_body)):
        differences.extend(abs(x-y) for x, y in zip(a.pose.position, b.pose.position))
        differences.append(abs(a.pose.angle-b.pose.angle))
        differences.extend(abs(x-y) for x, y in zip(a.velocity, b.velocity))
        differences.extend(abs(x-y) for x, y in zip(a.force, b.force))
        differences.extend((abs(a.angular_velocity-b.angular_velocity), abs(a.torque-b.torque)))
    differences.extend(abs(x-y) for x, y in zip(actual.goal.position, expected.goal.position))
    differences.append(abs(actual.goal.angle-expected.goal.angle))
    if actual.pusher_target is not None:
        differences.extend(abs(x-y) for x, y in zip(actual.pusher_target, expected.pusher_target))
    return max(differences)


def score_prediction(rows, config):
    """Terminal minimum coverage with pose-distance and motion penalties.

    Native success is absorbing because the real runner stops at the same event.
    Other candidates are judged on the last settling_window physical controls,
    never their transient maximum overlap. Pose distance supplies a signal before
    shapes overlap. Distance/speed use pixels and pixels/s; angles use radians.
    """
    if not rows:
        raise ValueError("Cannot score an empty forecast")
    if any(row["native_done"] for row in rows):
        return 1.0
    tail = rows[-config.settling_window:]
    coverage = min(finite(row["coverage"], "coverage") for row in tail)
    speed = sum(finite(row["speed"], "speed") for row in tail) / len(tail)
    angular = sum(abs(finite(row["angular_speed"], "angular_speed")) for row in tail) / len(tail)
    position = sum(finite(row["position_error"], "position error") for row in tail) / len(tail)
    angle = sum(finite(row["angle_error"], "angle error") for row in tail) / len(tail)
    return (coverage - config.position_weight * position / 512.0
            - config.angle_weight * angle / math.pi
            - config.speed_weight * speed / 512.0 - config.angular_speed_weight * angular)


class SimulatorReranker:
    """Rank pixel-action chunks without changing checkpoint weights or live state."""

    def __init__(self, env_factory, initial_state, config=None):
        if not isinstance(initial_state, Snapshot):
            raise ValueError("initial_state must be an episode-initial Snapshot")
        self.initial_state = initial_state
        self.config = config or PlannerConfig()
        if not isinstance(self.config, PlannerConfig):
            raise ValueError("Expected PlannerConfig")
        self.shape = official_shape(initial_state.shape)
        if self.shape.geometry_hash != initial_state.geometry_hash:
            raise ValueError("Initial state geometry is not the official shape")
        self.env = env_factory()
        self.executed_actions = []
        self.counts = {"restores": 0, "replay_steps": 0, "candidate_steps": 0}
        self.closed = False

    def record_action(self, action):
        if self.closed:
            raise RuntimeError("Planner is closed")
        self.executed_actions.append(point(action))

    @staticmethod
    def _check_deadline(deadline):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Simulator planning deadline reached")

    def _replay(self, expected, counts, deadline):
        self._check_deadline(deadline)
        self.env.restore_state(self.initial_state)
        counts["restores"] += 1
        for action in self.executed_actions:
            self._check_deadline(deadline)
            _, _, done, _ = self.env.step(action)
            counts["replay_steps"] += 1
            if done:
                raise ValueError("Recorded action history includes native terminal success")
        self._check_deadline(deadline)
        error = state_error(self.env.capture_state(), expected)
        if error > self.config.state_tolerance:
            raise ValueError(f"Replayed branch differs from live state by {error:.9g}")
        return error

    def _forecast(self, actions, prefix, current, total_steps, counts, deadline):
        rows, commands, predictions = [], [], []
        hold = current.pusher_body.pose.position
        for index in range(total_steps):
            self._check_deadline(deadline)
            command = actions[index] if index < prefix else hold
            _, reward, done, info = self.env.step(command)
            counts["candidate_steps"] += 1
            state = self.env.capture_state()
            if index + 1 == prefix:
                # Hold the attained physical position, not an unachieved target.
                hold = state.pusher_body.pose.position
            metrics = goal_metrics(self.shape, state.object_body.pose, state.goal)
            row = {"coverage": metrics.coverage, "iou": metrics.iou,
                   "position_error": metrics.position_error, "angle_error": metrics.angle_error,
                   "speed": math.hypot(*state.object_body.velocity),
                   "angular_speed": state.object_body.angular_velocity,
                   "native_done": bool(done), "reward": float(reward),
                   "n_contacts": int(info.get("n_contacts", 0))}
            rows.append(row)
            commands.append(point(command))
            predictions.append(state)
            if done:
                break
        # Physics cannot be interrupted mid-control. If the last control or its
        # scoring diagnostics overran the deadline, retain its work counts but
        # do not make that late forecast selectable.
        self._check_deadline(deadline)
        return tuple(commands), rows, tuple(predictions)

    def choose(self, candidates, current_state, remaining_steps, deadline=None):
        """Return a completed best plan; partial forecasts are never selectable.

        All plans use min(horizon + settle_steps, remaining_steps) physical
        controls. After its selected policy prefix, a plan holds the attained
        pusher position. Settling steps consume the actual remaining-task budget;
        they are not a free post-budget diagnostic. A deadline may return the
        best already-completed candidate; it raises if none finished.
        """
        if self.closed:
            raise RuntimeError("Planner is closed")
        if type(remaining_steps) is not int or remaining_steps < 1:
            raise ValueError("remaining_steps must be positive")
        if current_state.control_step != self.initial_state.control_step + len(self.executed_actions):
            raise ValueError("Record every real executed action before planning again")
        if deadline is not None:
            deadline = finite(deadline, "deadline")
        required = min(self.config.horizon, remaining_steps)
        chunks, rejected = [], []
        for index, chunk in enumerate(candidates):
            try:
                actions = tuple(point(a) for a in chunk)
                if len(actions) < required:
                    raise ValueError("Incomplete budget-clipped policy horizon")
                if any(not 0 <= coordinate <= 512 for action in actions for coordinate in action):
                    raise ValueError("Action outside official [0, 512] pixel bounds")
                chunks.append((index, actions))
            except (TypeError, ValueError) as exc:
                rejected.append({"candidate_index": index, "reason": str(exc)})
        if not chunks and not self.config.include_hold:
            raise ValueError("No valid policy candidates and hold is disabled")
        total_steps = min(self.config.horizon + self.config.settle_steps, remaining_steps)
        plans = [(index, min(prefix, required), chunk)
                 for index, chunk in chunks for prefix in self.config.prefix_lengths]
        # Near the task budget several prefix settings can become identical.
        plans = list(dict(((index, prefix), (index, prefix, chunk))
                          for index, prefix, chunk in plans).values())
        if self.config.include_hold:
            plans.append((None, 0, ()))
        counts, counts_before = self.counts, dict(self.counts)
        reports, best, best_score, timed_out = [], None, -math.inf, False
        start = time.monotonic()
        for candidate_index, prefix, chunk in plans:
            try:
                replay_error = self._replay(current_state, counts, deadline)
                commands, rows, predictions = self._forecast(chunk, prefix, current_state, total_steps, counts, deadline)
            except TimeoutError:
                timed_out = True
                break
            score = score_prediction(rows, self.config)
            report = {"candidate_index": candidate_index, "policy_prefix": prefix,
                      "kind": "hold" if candidate_index is None else "policy",
                      "score": score, "native_success": rows[-1]["native_done"],
                      "forecast_steps": len(rows), "replay_state_error": replay_error,
                      "final_coverage": rows[-1]["coverage"], "final_iou": rows[-1]["iou"],
                      "peak_coverage": max(row["coverage"] for row in rows), "steps": rows}
            reports.append(report)
            if score > best_score:
                best_score, best = score, (commands, predictions, len(reports)-1)
        if best is None:
            raise TimeoutError("No simulator candidate completed before the deadline")
        commands, predictions, selected = best
        reports[selected]["first_state"] = asdict(predictions[0])
        return PlannerDecision(commands, {
            "config": asdict(self.config),
            "counts": {name: value-counts_before[name] for name, value in counts.items()},
            "selected_index": selected, "selected": reports[selected], "candidates": reports,
            "selected_predictions": [asdict(state) for state in predictions],
            "selected_actions": [list(action) for action in commands],
            "requested_candidates": len(plans), "completed_candidates": len(reports),
            "rejected_candidates": rejected,
            "timed_out": timed_out, "elapsed_seconds": time.monotonic()-start,
            "forecast_budget": total_steps, "settling_uses_real_budget": True,
            "demonstrations": False, "training": False, "privileged_physics": True,
            "branch_protocol": "initial_snapshot_plus_all_executed_actions"})

    def close(self):
        if not self.closed:
            self.env.close()
            self.closed = True
