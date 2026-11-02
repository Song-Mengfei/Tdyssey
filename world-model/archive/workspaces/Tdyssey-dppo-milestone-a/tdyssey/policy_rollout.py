"""Bounded direct-simulator fragments and a persistent, conservative budget.

Reservations are written before physical controls, sampling plans or optimizer
steps. An exception never refunds an attempt. This module only collects data;
it does not optimize a policy or turn a short fragment into a learning claim.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import threading
import uuid

import numpy as np
import torch

from .official import physical_metrics
from .official_collection import DEVELOPMENT_SEEDS
from .policy_returns import ActionBlock, BlockBoundary, training_reward
from .policy_rl import SampledChain


class BudgetExceeded(RuntimeError):
    """A hard limit was reached before the requested operation began."""


class BudgetBusy(RuntimeError):
    """An existing owner lock prevents use of this budget, including stale locks."""


_HARD_LIMITS = {"control": 64, "plan": 8, "actor_step": 1, "critic_step": 1}


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _shape(shape):
    if shape not in (None, "T", "L"):
        raise ValueError("This bounded pilot permits only T/L shape metadata")
    return shape


def _atomic_json(path, payload):
    """Publish same-directory JSON only after flushing and fsyncing its bytes."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # POSIX can also flush directory metadata. Native Windows does not
        # expose directory fsync through os.open; the file itself is fsynced.
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


class HardBudget:
    """Persistent attempt ledger, with one exclusive owner for its whole lifetime.

    Limits can be made smaller for controlled tests, never larger than 64 real
    controls / 8 sampled plans / one actor step / one critic step. Reopening a
    ledger requires exactly its saved limits and preserves every reservation.
    ``reserve`` must return successfully *before* its operation is attempted.
    ``mark_completed`` is separate telemetry and never frees any reservation.

    An O_EXCL sidecar prevents concurrent owners, including other processes.
    Use a context manager or close explicitly. A crash deliberately leaves its
    lock behind: automatic stale-lock takeover could overlap a still-running
    owner, so recovery requires a human to verify the old process has stopped.
    The JSON is atomic and file-fsynced, not a transaction with the simulator;
    reserving first makes crash accounting conservative rather than refundable.
    """

    def __init__(self, path, *, max_controls=64, max_plans=8,
                 max_actor_steps=1, max_critic_steps=1):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        limits = {"control": max_controls, "plan": max_plans,
                  "actor_step": max_actor_steps, "critic_step": max_critic_steps}
        for kind, limit in limits.items():
            _integer(limit, f"max_{kind}", 1, _HARD_LIMITS[kind])
        self._thread_lock = threading.RLock()
        self._closed = True
        self._token = uuid.uuid4().hex
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise BudgetBusy(f"Budget is locked; refusing automatic takeover: {self.lock_path}") from exc
        self._closed = False
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"token": self._token, "pid": os.getpid(), "host": socket.gethostname()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if self.path.exists():
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                self._validate(limits)
            else:
                self._data = {"version": 1, "limits": limits,
                              "reserved": {kind: 0 for kind in limits},
                              "completed": {kind: 0 for kind in limits},
                              "by_shape": {}, "reset_attempts": 0,
                              "reset_completed": 0, "events": []}
                _atomic_json(self.path, self._data)
        except BaseException:
            self.close()
            raise

    def _validate(self, limits):
        data = self._data
        if data.get("version") != 1 or data.get("limits") != limits:
            raise ValueError("Budget version/limits differ from the persisted authorization")
        reserved = {kind: 0 for kind in limits}
        completed = {kind: 0 for kind in limits}
        by_shape = {}
        resets = reset_completed = 0
        for index, event in enumerate(data["events"], start=1):
            if (event["id"] != index or type(event["completed"]) is not bool
                    or event["kind"] not in {*limits, "reset"}):
                raise ValueError("Corrupt budget reservation events")
            shape = _shape(event["shape"])
            kind = event["kind"]
            if kind == "reset":
                resets += 1
                reset_completed += int(event["completed"])
            else:
                reserved[kind] += 1
                completed[kind] += int(event["completed"])
            if shape is not None:
                counts = by_shape.setdefault(shape, {key: 0 for key in (*limits, "reset")})
                counts[kind] += 1
        if (reserved != data["reserved"] or completed != data["completed"]
                or by_shape != data["by_shape"] or resets != data["reset_attempts"]
                or reset_completed != data["reset_completed"]
                or any(reserved[kind] > limit for kind, limit in limits.items())):
            raise ValueError("Budget counters disagree with their reservation events")

    def _ensure_open(self):
        if self._closed:
            raise BudgetBusy("Budget owner has already closed")
        try:
            owner = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BudgetBusy("Budget lock is missing/corrupt; refusing further operations") from exc
        if owner.get("token") != self._token:
            raise BudgetBusy("Budget lock ownership changed; refusing further operations")

    def _reserve(self, kind, shape):
        with self._thread_lock:
            self._ensure_open()
            _shape(shape)
            data = deepcopy(self._data)
            if kind != "reset":
                if data["reserved"][kind] >= data["limits"][kind]:
                    raise BudgetExceeded(f"Hard {kind} budget exhausted; no operation was attempted")
                data["reserved"][kind] += 1
            else:
                data["reset_attempts"] += 1
            if shape is not None:
                counts = data["by_shape"].setdefault(shape,
                         {key: 0 for key in (*data["limits"], "reset")})
                counts[kind] += 1
            event_id = len(data["events"]) + 1
            data["events"].append({"id": event_id, "kind": kind, "shape": shape,
                                   "completed": False,
                                   "reserved_at": datetime.now(timezone.utc).isoformat()})
            _atomic_json(self.path, data)
            self._data = data
            return event_id

    def reserve(self, kind, shape=None):
        if kind not in _HARD_LIMITS:
            raise ValueError("Unknown budget operation")
        return self._reserve(kind, shape)

    def reserve_reset(self, shape):
        """Count a reset's one settling physics substep separately from controls."""
        return self._reserve("reset", shape)

    def mark_completed(self, reservation_id):
        with self._thread_lock:
            self._ensure_open()
            _integer(reservation_id, "reservation_id", 1, len(self._data["events"]))
            data = deepcopy(self._data)
            event = data["events"][reservation_id - 1]
            if event["completed"]:
                raise ValueError("Reservation completion was already recorded")
            event["completed"] = True
            if event["kind"] == "reset":
                data["reset_completed"] += 1
            else:
                data["completed"][event["kind"]] += 1
            _atomic_json(self.path, data)
            self._data = data

    def snapshot(self):
        with self._thread_lock:
            self._ensure_open()
            return deepcopy(self._data)

    def close(self):
        with self._thread_lock:
            if not self._closed:
                self._ensure_open()
                self.lock_path.unlink()
                self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True)
class RolloutPlan:
    sample: SampledChain
    block: ActionBlock
    shape: str
    reset_seed: int
    control_records: list[dict]
    final_observation: dict[str, torch.Tensor]
    features: torch.Tensor


def _cpu(value):
    return value.detach().cpu().clone()


def _observation_copy(observation):
    if not isinstance(observation, dict) or not {"image", "agent_pos"}.issubset(observation):
        raise ValueError("Expected the official image/agent_pos observation")
    result = {key: _cpu(torch.as_tensor(observation[key])) for key in ("image", "agent_pos")}
    if (result["image"].shape != (3, 96, 96) or result["agent_pos"].shape != (2,)
            or any(not torch.isfinite(value).all() for value in result.values())
            or result["image"].min() < 0 or result["image"].max() > 1):
        raise ValueError("Unexpected official observation format")
    return result


def _cpu_sample(sample):
    if not isinstance(sample, SampledChain) or sample.mode != "training" or sample.old_log_prob is None:
        raise ValueError("Collector requires a stochastic training SampledChain")
    if (sample.actions.shape != (1, 8, 2) or sample.chain.shape != (1, 101, 16, 2)
            or sample.old_log_prob.shape != (1, 100)
            or not torch.isfinite(sample.actions).all()):
        raise ValueError("Collector requires one complete official 16/8 training plan")
    tensors = {key: _cpu(getattr(sample, key)) for key in
               ("chain", "timesteps", "old_means", "old_stds", "old_log_prob", "targets", "actions")}
    return replace(sample, observations={key: _cpu(value) for key, value in sample.observations.items()},
                   **tensors)


def _features_and_value(actor, critic, observation):
    features = actor.encode(observation).detach()
    value = critic(features).reshape(-1)
    if (features.ndim != 2 or features.shape[0] != 1 or value.shape != (1,)
            or not torch.isfinite(features).all() or not torch.isfinite(value).all()):
        raise ValueError("Critic must return one finite value from frozen visual features")
    return features, float(value.item())


@torch.no_grad()
def collect_fragment(official, actor, critic, env, *, shape, reset_seed, generator,
                     budget, max_controls, max_plans, episode_limit, actor_version,
                     metrics_fn=physical_metrics, gamma=0.99, beta=1.0):
    """Collect one reset-to-cutoff fragment, with no optimizer or hidden resets.

    Only image and pusher history condition actor and critic. Shape/reset IDs are
    metadata. A fragment ends at its first success, timeout, control cap or plan
    cap. The final observation is captured before any subsequent reset; timeout
    retains its coverage potential and value bootstrap. Success takes precedence
    over a coincident time limit. All returned tensors are detached CPU records.

    The ledger reserves before reset/sample/step, counts failed attempts without
    refunds, and is never reset by this helper. Reset settling substeps are
    recorded separately; only non-None action calls are physical controls here.
    No caller may bypass the same ledger for additional simulator interaction.
    """
    if shape not in ("T", "L") or getattr(env, "project_shape", shape) != shape:
        raise ValueError("Shape metadata must match this pilot's T/L environment")
    _integer(reset_seed, "reset_seed", 0, 2**32 - 1)
    if reset_seed in DEVELOPMENT_SEEDS:
        raise ValueError("Known development reset seeds are excluded from policy updates")
    _integer(max_controls, "max_controls", 1, 64)
    _integer(max_plans, "max_plans", 1, 8)
    _integer(episode_limit, "episode_limit", 1, 100000)
    if generator is None or not isinstance(actor_version, str) or not actor_version.strip():
        raise ValueError("Provide an explicit generator and actor version")
    # Validate reward configuration before any physical/reset budget is spent.
    training_reward(0.0, 0.0, 0.0, gamma=gamma, beta=beta)
    snapshot = budget.snapshot()
    if any(requested > snapshot["limits"][kind] - snapshot["reserved"][kind]
           for kind, requested in (("control", max_controls), ("plan", max_plans))):
        raise BudgetExceeded("Fragment caps exceed the remaining hard budget; reset was not attempted")

    env.seed(reset_seed)
    reservation = budget.reserve_reset(shape)
    observation = env.reset()
    budget.mark_completed(reservation)
    observation = _observation_copy(observation)
    # Optional geometry sanity check has no env.step and never retries resets.
    if hasattr(env, "capture_state"):
        from .collection import reset_issues
        from .envs.official import official_shape
        issues = reset_issues(official_shape(shape), env.capture_state())
        if issues:
            raise ValueError(f"Unsafe reset rejected before any physical control: {issues}")
    metrics = metrics_fn(env)
    coverage = float(metrics["coverage"])
    training_reward(coverage, coverage, 0.0, gamma=gamma, beta=beta)
    if coverage > 0.95:
        raise ValueError("Reset is already successful; no physical control was attempted")
    history = [observation, observation]
    episode_id = f"{shape}:{reset_seed}:{uuid.uuid4().hex}"
    plans = []
    controls = 0
    success_seen = False
    if hasattr(critic, "eval"):
        critic.eval()
    while controls < max_controls and len(plans) < max_plans:
        observations = official.observation_tensors(history)
        features, value = _features_and_value(actor, critic, observations)
        reservation = budget.reserve("plan", shape)
        sample = actor.sample(observations, generator=generator, mode="training", actor_version=actor_version)
        budget.mark_completed(reservation)
        sample = _cpu_sample(sample)
        records = []
        boundary = BlockBoundary.CONTINUE
        for action_index, action in enumerate(sample.actions[0]):
            before_observation = _observation_copy(observation)
            before_coverage = coverage
            reservation = budget.reserve("control", shape)
            action_array = action.numpy().astype(np.float64, copy=True)
            result = env.step(action_array)
            budget.mark_completed(reservation)
            controls += 1
            if not isinstance(result, tuple) or len(result) != 4:
                raise ValueError("Expected the original official four-tuple step interface")
            next_observation, raw_reward, upstream_done, info = result
            observation = _observation_copy(next_observation)
            history = [history[-1], observation]
            metrics = metrics_fn(env)
            coverage = float(metrics["coverage"])
            reward = training_reward(before_coverage, coverage, raw_reward,
                                     success_already_seen=success_seen, gamma=gamma, beta=beta)
            success_seen = success_seen or reward.first_success
            timeout = controls >= episode_limit or bool(info.get("TimeLimit.truncated", False))
            if upstream_done and not reward.terminated and not timeout:
                raise ValueError("Unexpected upstream done below the strict success threshold")
            counts = budget.snapshot()
            records.append({"observation_before": before_observation,
                            "observation_after": _observation_copy(observation),
                            "action": action.clone(), "action_index": action_index,
                            "coverage_before": before_coverage, "coverage": coverage,
                            "iou": float(metrics.get("iou", math.nan)),
                            "raw_reward": float(raw_reward), "upstream_done": bool(upstream_done),
                            "training_reward": reward.reward, "shaping": reward.shaping,
                            "first_success": reward.first_success, "terminated": reward.terminated,
                            "control_index": controls, "global_control_attempt": counts["reserved"]["control"],
                            "reservation_id": reservation,
                            "control_physics_substeps": (env.sim_hz // env.control_hz
                                if hasattr(env, "sim_hz") and hasattr(env, "control_hz") else None)})
            if reward.terminated:
                boundary = BlockBoundary.SUCCESS
            elif timeout:
                boundary = BlockBoundary.TIMEOUT
            elif controls >= max_controls or len(plans) + 1 >= max_plans:
                # A plan cap limits *new* plans; finish this sampled plan first.
                if controls >= max_controls or action_index == 7:
                    boundary = BlockBoundary.BUFFER_CUTOFF
            if boundary != BlockBoundary.CONTINUE:
                break
        final_observation = official.observation_tensors(history)
        if boundary == BlockBoundary.SUCCESS:
            final_value = None
        else:
            _, final_value = _features_and_value(actor, critic, final_observation)
        block = ActionBlock(episode_id=episode_id,
                            rewards=tuple(record["training_reward"] for record in records),
                            value=value, final_observation_value=final_value, boundary=boundary)
        plans.append(RolloutPlan(sample=sample, block=block, shape=shape, reset_seed=reset_seed,
                                 control_records=records,
                                 final_observation={key: _cpu(value) for key, value in final_observation.items()},
                                 features=_cpu(features)))
        if boundary != BlockBoundary.CONTINUE:
            break
    return plans
