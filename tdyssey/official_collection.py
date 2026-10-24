"""Bounded official PushT play data, separate from policy evaluation and GPC."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import time

from .collection import CollectionConfig, CollectedEpisode, WaypointPlay, derived_seed, integer, reset_issues
from .dataset import load_dataset
from .envs.official import official_shape
from .evaluation import goal_metrics
from .geometry import point
from .state import Snapshot
from .trajectories import Trajectory, Transition, split_trajectories

VERSION = "official-waypoint-play-v1"
ENVIRONMENT_FAMILY = "official-pusht-capitals-v1"
# Current original-T and alphabet development scenes. Extend before evaluating
# more scenes that must be excluded from future adaptation data.
DEVELOPMENT_SEEDS = (100000, 100001, 100002)


@dataclass(frozen=True)
class OfficialCollectionConfig(CollectionConfig):
    episodes: int = 8
    steps: int = 100
    seed: int = 200000
    mode: str = "object-directed"
    max_bytes: int = 32 * 1024 * 1024
    seconds: int = 120
    excluded_reset_seeds: tuple = DEVELOPMENT_SEEDS

    def __post_init__(self):
        super().__post_init__()
        if self.shape not in ("T", "L"):
            raise ValueError("The official collector focuses on T/L")
        integer(self.seconds, "seconds", 1, 600)
        seeds = tuple(self.excluded_reset_seeds)
        for seed in seeds:
            integer(seed, "excluded reset seed", 0, 2**32 - 1)
        if len(set(seeds)) != len(seeds) or not set(DEVELOPMENT_SEEDS).issubset(seeds):
            raise ValueError("Excluded seeds must be unique and include all known development scenes")
        object.__setattr__(self, "excluded_reset_seeds", seeds)
        if self.mode == "contact-balanced" and self.max_target_delta <= 30:
            object.__setattr__(self, "max_target_delta", 200.0)


def _play_controller(config, seed):
    """Keep waypoint play as the default; contact-balanced has its own controller."""
    if config.mode == "contact-balanced":
        from .official_contact_sampler import ContactBalancedPlay

        return ContactBalancedPlay(config, seed)
    return WaypointPlay(config, seed)


def freeze_splits(trajectories, seed):
    """Freeze whole episodes; refuse empty sets and repeated initial scenes."""
    records = list(trajectories)
    initial = [record.transitions[0].before.to_json() for record in records]
    if len(set(initial)) != len(initial):
        raise ValueError("Repeated initial scenes must not be split independently")
    splits = split_trajectories(records, seed)
    if not all(splits.values()):
        raise ValueError("Official collection requires nonempty train/validation/model_test splits; use at least seven episodes")
    return {name: [record.trajectory_id for record in rows] for name, rows in splits.items()}


def collect_official_episode(env, config, index, counters, deadline=None):
    """Original 4-tuple API; T+1 frames; no model, expert policy or reward search."""
    import numpy as np

    integer(index, "episode index", 0, config.episodes-1)
    if env.project_shape != config.shape or env.sim_hz % env.control_hz:
        raise ValueError("Official environment shape/rates differ from collection config")
    shape = official_shape(config.shape)

    def check_time():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Official collection wall-time limit exceeded")

    resets = []
    for attempt in range(config.reset_attempts):
        check_time()
        seed = derived_seed(config.seed, VERSION, config.shape, index, "reset", attempt)
        if seed in config.excluded_reset_seeds:
            counters["excluded_seed_candidates"] = counters.get("excluded_seed_candidates", 0)+1
            resets.append({"seed": seed, "issues": ["excluded_evaluation_scene"], "simulated": False})
            continue
        env.seed(seed)
        counters["reset_attempts"] += 1
        env.reset()
        counters["reset_physics_substeps"] += 1
        upstream = env.capture_state()
        state = replace(upstream, pusher_target=None)
        issues = reset_issues(shape, state)
        resets.append({"seed": seed, "issues": issues, "simulated": True,
                       "upstream_state": asdict(upstream), "accepted_state": asdict(state) if not issues else None})
        if not issues:
            observation = env.restore_state(state)
            break
    else:
        raise ValueError("No accepted official reset within the candidate budget")

    action_seed = derived_seed(config.seed, VERSION, config.shape, index, config.mode, "actions")
    controller = _play_controller(config, action_seed)
    images, positions, goals, transitions, diagnostics = [], [], [], [], []

    def record(obs, state):
        image = np.asarray(obs["image"])
        if image.shape != (3,96,96) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Expected official normalized CHW 96x96 observations")
        images.append(np.rint(image*255).astype(np.uint8))
        positions.append(point(obs["agent_pos"]))
        goals.append((*state.goal.position, state.goal.angle % (2*np.pi)))

    record(observation, state)
    for step in range(config.steps):
        check_time()
        before = env.capture_state()
        action = controller.action(observation, before, step)
        observation, reward, done, info = env.step(np.asarray(action, dtype=np.float64))
        counters["control_steps"] += 1
        counters["control_physics_substeps"] += env.sim_hz // env.control_hz
        after = env.capture_state()
        if hasattr(controller, "observe"):
            controller.observe(before, after, info["n_contacts"])
        cutoff = step+1 == config.steps and not done
        transitions.append(Transition(before, action, after, float(reward), bool(done), cutoff))
        metrics = goal_metrics(shape, after.object_body.pose, after.goal)
        diagnostics.append({"n_contacts": integer(info["n_contacts"], "contacts", 0, 100000),
                            "iou": metrics.iou, "coverage": metrics.coverage,
                            "upstream_terminated": bool(done), "upstream_truncated": False,
                            "collector_truncated": cutoff})
        record(observation, after)
        if done or cutoff:
            reason = "upstream_termination" if done else "collection_horizon"
            break
    identity = [VERSION, shape.geometry_hash, transitions[0].before.physics_hash, seed, action_seed,
                config.steps, config.waypoint_steps, config.max_target_delta]
    identifier = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
    trajectory = Trajectory(f"official-{config.shape}-{identifier}", config.mode, VERSION, action_seed,
                            1/env.control_hz, tuple(transitions))
    return CollectedEpisode(trajectory, np.stack(images), np.asarray(positions, dtype=np.float64),
                            np.asarray(goals, dtype=np.float64), diagnostics, resets, seed, action_seed, reason)


def validate_official_dataset(root):
    """Read back official geometry, reset provenance and evaluation separation."""
    manifest, episodes = load_dataset(root)
    if manifest["provenance"].get("environment_family") != ENVIRONMENT_FAMILY:
        raise ValueError("Expected official collection, not GPC/reference data")
    excluded = set(manifest["config"]["excluded_reset_seeds"])
    if not set(DEVELOPMENT_SEEDS).issubset(excluded) or not all(manifest["splits"].values()):
        raise ValueError("Missing evaluation exclusions or nonempty frozen splits")
    simulated = 0
    for trajectory, record, _ in episodes:
        initial = trajectory.transitions[0].before
        if trajectory.collector_version != VERSION or initial.geometry_hash != official_shape(initial.shape).geometry_hash:
            raise ValueError("Official geometry/collector identity differs")
        if record["reset_seed"] in excluded:
            raise ValueError("Evaluation scene leaked into collection")
        attempts = record["reset_attempts"]
        simulated += sum(item["simulated"] for item in attempts)
        if (attempts[-1]["issues"] or not attempts[-1]["simulated"] or attempts[-1]["seed"] != record["reset_seed"]
                or Snapshot.from_dict(attempts[-1]["accepted_state"]) != initial):
            raise ValueError("Accepted reset provenance differs from initial training state")
        if any(t.shape != initial.shape for t in [x.before for x in trajectory.transitions]):
            raise ValueError("Official trajectory changes object shape")
    if simulated != manifest["counts"]["reset_attempts"]:
        raise ValueError("Simulated reset cost differs from reset records")
    return manifest, episodes
