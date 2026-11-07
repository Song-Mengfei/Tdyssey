"""Bounded non-expert play; optional NumPy is imported only during collection."""

from dataclasses import dataclass, replace
import hashlib
import json
import math
import random

from .evaluation import goal_metrics
from .geometry import Pose, finite, get_shape, point
from .trajectories import Trajectory, Transition

COLLECTOR_VERSION = "waypoint-play-v1"


def integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class CollectionConfig:
    shape: str = "L"
    episodes: int = 2
    steps: int = 100
    seed: int = 100000
    mode: str = "random-waypoints"
    waypoint_steps: int = 12
    max_target_delta: float = 12.0
    max_bytes: int = 16 * 1024 * 1024
    reset_attempts: int = 20

    def resolve_shape(self):
        return get_shape(self.shape)

    def __post_init__(self):
        object.__setattr__(self, "shape", self.resolve_shape().name)
        integer(self.episodes, "episodes", 1, 20)
        integer(self.steps, "steps", 1, 250)
        integer(self.seed, "seed", 0, 2**32 - 1)
        integer(self.waypoint_steps, "waypoint_steps", 1, 100)
        integer(self.reset_attempts, "reset_attempts", 1, 20)
        integer(self.max_bytes, "max_bytes", 1024 * 1024, 64 * 1024 * 1024)
        delta = finite(self.max_target_delta, "max_target_delta")
        if not 0 < delta <= 30:
            raise ValueError("max_target_delta must be in (0, 30] pixels")
        object.__setattr__(self, "max_target_delta", delta)
        if self.mode not in ("random-waypoints", "object-directed"):
            raise ValueError("Unknown non-expert play mode")
        if self.episodes * self.steps > 1000:
            raise ValueError("A pilot run is limited to 1000 executed control steps")
        # Include conservative JSON/ZIP overhead, not just highly compressible images.
        upper_bound = self.episodes * ((self.steps + 1) * 28000 + self.steps * 8000 + 32000) + 131072
        if upper_bound > self.max_bytes:
            raise ValueError("Worst-case output exceeds byte budget; reduce episodes/steps or raise max_bytes")


def derived_seed(seed, *parts):
    encoded = json.dumps([seed, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def reset_issues(shape, state, pusher_radius=15.0):
    """Check the obstacle-free GPC scene, including rotated geometry and goal.

    Inner wall faces lie at 7 and 504 pixels; retain one pixel of clearance.
    This does not establish that a valid goal is dynamically reachable.
    """
    issues = []
    for label, pose in (("object", state.object_body.pose), ("goal", state.goal)):
        if any(not 8 <= coordinate <= 503 for polygon in shape.polygons(pose)
               for vertex in polygon for coordinate in vertex):
            issues.append(f"{label}_outside_walls")
    x, y = state.pusher_body.pose.position
    if any(not 8 + pusher_radius <= v <= 503 - pusher_radius for v in (x, y)):
        issues.append("pusher_outside_walls")
    # Circle-to-rectangle distance in the object's local coordinate frame.
    dx, dy = x - state.object_body.pose.position[0], y - state.object_body.pose.position[1]
    angle = state.object_body.pose.angle
    local_x = math.cos(angle) * dx + math.sin(angle) * dy
    local_y = -math.sin(angle) * dx + math.cos(angle) * dy
    for rect in shape.components:
        distance = math.hypot(max(rect.x0 - local_x, 0, local_x - rect.x1),
                              max(rect.y0 - local_y, 0, local_y - rect.y1))
        if distance < pusher_radius + 1:
            issues.append("pusher_object_overlap_or_insufficient_clearance")
            break
    if goal_metrics(shape, state.object_body.pose, state.goal).coverage > 0.90:
        issues.append("already_near_goal")
    return issues


class WaypointPlay:
    """No reward optimization. Object-directed mode uses privileged object position."""

    def __init__(self, config, seed):
        self.config, self.rng = config, random.Random(seed)
        self.waypoint = None

    def action(self, observation, state, step):
        position = point(observation["agent_pos"])
        if self.waypoint is None or step % self.config.waypoint_steps == 0:
            if self.config.mode == "object-directed" and self.rng.random() < 0.75:
                # Aim through the object, with noisy offsets. This deliberately
                # disclosed exploration prior uses simulator position, not a policy.
                center = state.object_body.pose.position
                direction = (center[0] - position[0], center[1] - position[1])
                distance = max(math.hypot(*direction), 1e-9)
                self.waypoint = tuple(center[i] + 45 * direction[i] / distance
                                      + self.rng.uniform(-20, 20) for i in (0, 1))
            else:
                self.waypoint = (self.rng.uniform(25, 486), self.rng.uniform(25, 486))
            self.waypoint = tuple(min(486, max(25, v)) for v in self.waypoint)
        dx, dy = self.waypoint[0] - position[0], self.waypoint[1] - position[1]
        factor = min(1, self.config.max_target_delta / max(math.hypot(dx, dy), 1e-9))
        return (position[0] + factor * dx, position[1] + factor * dy)


@dataclass
class CollectedEpisode:
    trajectory: Trajectory
    images: object
    agent_pos: object
    goal_pose: object
    steps: list
    resets: list
    reset_seed: int
    action_seed: int
    stop_reason: str


def collect_episode(env, config, index, counters):
    """Collect T+1 observations for T actions; counters include rejected resets.

    The fixed project goal is applied after upstream reset. GPC's one reset-time
    physics substep is accounted separately from recorded control transitions.
    Counters are updated during work even if a later check or write fails.
    """
    import numpy as np

    integer(index, "episode index", 0, config.episodes - 1)
    if env.object != config.shape or env.sim_hz % env.control_hz:
        raise ValueError("Environment shape/rates do not match the collection protocol")
    shape = get_shape(config.shape, env.block_scale)
    fixed_goal = Pose((256, 256), math.pi / 4)
    resets = []
    for attempt in range(config.reset_attempts):
        reset_seed = derived_seed(config.seed, config.shape, index, "reset", attempt)
        env.seed(reset_seed)
        counters["reset_attempts"] += 1
        env.reset()
        counters["reset_physics_substeps"] += 1
        state = replace(env.capture_state(), goal=fixed_goal, pusher_target=None)
        issues = reset_issues(shape, state)
        resets.append({"seed": reset_seed, "issues": issues})
        if not issues:
            observation, _ = env.restore_state(state)
            break
    else:
        raise ValueError(f"No valid initial state in {config.reset_attempts} reset attempts")

    action_seed = derived_seed(config.seed, config.shape, index, config.mode, "actions")
    controller = WaypointPlay(config, action_seed)
    images, positions, goals, transitions, diagnostics = [], [], [], [], []

    def record_observation(obs):
        image = np.asarray(obs["image"])
        if image.shape != (3, 96, 96) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Expected finite CHW 96x96 images in [0, 1]")
        images.append(np.rint(image * 255).astype(np.uint8))
        positions.append(point(obs["agent_pos"]))
        pose = obs["goal_pose"]
        goals.append((*point(pose[:2]), finite(pose[2])))

    record_observation(observation)
    for step in range(config.steps):
        before = env.capture_state()
        action = controller.action(observation, before, step)
        observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float64))
        counters["control_steps"] += 1
        counters["control_physics_substeps"] += env.sim_hz // env.control_hz
        after = env.capture_state()
        horizon = step + 1 == config.steps and not (terminated or truncated)
        transitions.append(Transition(before, action, after, float(reward), bool(terminated), bool(truncated or horizon)))
        metrics = goal_metrics(shape, after.object_body.pose, after.goal)
        contacts = integer(info["n_contacts"], "n_contacts", 0, 100000)
        diagnostics.append({"n_contacts": contacts, "iou": metrics.iou, "coverage": metrics.coverage,
                            "upstream_terminated": bool(terminated), "upstream_truncated": bool(truncated),
                            "collector_truncated": horizon})
        record_observation(observation)
        if terminated or truncated or horizon:
            reason = "upstream_termination" if terminated else "upstream_truncation" if truncated else "collection_horizon"
            break
    identifier = hashlib.sha256(json.dumps([COLLECTOR_VERSION, config.shape, reset_seed, action_seed,
                                            config.steps, config.waypoint_steps, config.max_target_delta]).encode()).hexdigest()[:20]
    trajectory = Trajectory(f"{config.shape}-{identifier}", config.mode, COLLECTOR_VERSION,
                            action_seed, 1 / env.control_hz, tuple(transitions))
    return CollectedEpisode(trajectory, np.stack(images), np.asarray(positions, dtype=np.float64),
                            np.asarray(goals, dtype=np.float64), diagnostics, resets,
                            reset_seed, action_seed, reason)
