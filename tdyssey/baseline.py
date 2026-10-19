"""Bounded, matched-scene evaluation of the original unranked visual policy."""

from collections import deque
from dataclasses import asdict, replace
import time

from .collection import integer, reset_issues
from .evaluation import goal_metrics
from .geometry import get_shape
from .policy import action_prefix


def validate_budget(shapes, episodes, steps, seed, seconds):
    if not shapes or len(set(shapes)) != len(shapes) or any(s not in ("T", "I", "L") for s in shapes):
        raise ValueError("Select each of T/I/L at most once")
    integer(episodes, "episodes", 1, 3)
    integer(steps, "steps", 1, 300)
    integer(seed, "seed", 100000, 2**32 - 61)
    integer(seconds, "seconds", 1, 900)
    if len(shapes) * episodes * steps > 900:
        raise ValueError("A baseline run is limited to 900 control steps")


def matched_resets(environments, seed, counters):
    """Transfer T's physical state, keeping each shape's geometry/physics identity.

    A T environment must be included as the scene reference even when only I/L
    are evaluated. Keep upstream's randomized goal. Reject a candidate for every
    shape together if any geometry is unsafe. No reward is used to pick actions.
    """
    attempts = []
    for offset in range(20):
        snapshots = {}
        for name, env in environments.items():
            env.seed(seed + offset)
            counters["reset_attempts"] += 1
            env.reset()
            counters["reset_physics_substeps"] += 1
            snapshots[name] = env.capture_state()
        reference = snapshots["T"]
        states = {name: replace(state, object_body=reference.object_body,
                                pusher_body=reference.pusher_body, goal=reference.goal,
                                pusher_target=None, control_step=0)
                  for name, state in snapshots.items()}
        issues = {name: reset_issues(get_shape(name, environments[name].block_scale), state)
                  for name, state in states.items()}
        attempts.append({"seed": seed + offset, "issues": issues})
        if not any(issues.values()):
            observations = {name: environments[name].restore_state(state)[0]
                            for name, state in states.items()}
            return observations, attempts
    raise ValueError(f"No common valid scene after 20 candidates: {attempts}")


def rollout(env, policy, observation, steps, action_seed, deadline, counters):
    """Execute configured action chunks, recording post-action scores and flags.

    This keeps GPC's original terminal flags, including its simultaneous
    termination/truncation on success. No learning/bootstrap rule is inferred.
    """
    import numpy as np

    config = policy.spec["policy"]
    history = deque([observation] * config["obs_horizon"], maxlen=config["obs_horizon"])
    policy.reset_seed(action_seed)
    shape = get_shape(env.object, env.block_scale)
    initial = env.capture_state()
    initial_metrics = asdict(goal_metrics(shape, initial.object_body.pose, initial.goal))
    result = {"shape": env.object, "action_seed": action_seed, "initial_state": asdict(initial),
              "initial_metrics": initial_metrics, "steps": [], "planning_seconds": [],
              "stop_reason": "control_budget", "upstream_success": False}
    start = time.monotonic()
    while len(result["steps"]) < steps:
        if time.monotonic() >= deadline:
            result["stop_reason"] = "wall_time_budget"
            break
        plan_start = time.monotonic()
        prediction = policy.predict(list(history))
        result["planning_seconds"].append(time.monotonic() - plan_start)
        counters["planning_calls"] += 1
        counters["denoising_steps"] += config["num_diffusion_iters"]
        actions = action_prefix(prediction, config, steps - len(result["steps"]))
        for action in actions:
            if time.monotonic() >= deadline:
                result["stop_reason"] = "wall_time_budget"
                break
            observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float64))
            history.append(observation)
            counters["control_steps"] += 1
            counters["control_physics_substeps"] += env.sim_hz // env.control_hz
            state = env.capture_state()
            result["steps"].append({"action": [float(v) for v in action], "reward": float(reward),
                                    "terminated": bool(terminated), "truncated": bool(truncated),
                                    "n_contacts": int(info["n_contacts"]), "state": asdict(state),
                                    "metrics": asdict(goal_metrics(shape, state.object_body.pose, state.goal))})
            if terminated or truncated:
                result["stop_reason"] = "upstream_termination" if terminated else "upstream_truncation"
                result["upstream_success"] = bool(terminated)
                break
        if result["stop_reason"] != "control_budget":
            break
    metrics = [row["metrics"] for row in result["steps"]]
    result.update({"control_steps": len(metrics), "elapsed_seconds": time.monotonic() - start,
                   "contact_steps": sum(row["n_contacts"] > 0 for row in result["steps"]),
                   "final_metrics": metrics[-1] if metrics else initial_metrics,
                   "max_iou": max([initial_metrics["iou"], *(m["iou"] for m in metrics)]),
                   "max_coverage": max([initial_metrics["coverage"], *(m["coverage"] for m in metrics)])})
    return result
