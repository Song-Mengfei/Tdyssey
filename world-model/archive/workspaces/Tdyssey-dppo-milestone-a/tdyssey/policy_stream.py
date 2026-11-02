"""Continuous T/L training streams with explicit, non-physical resume boundaries.

A batch boundary cuts GAE, not the environment episode. This collector holds
only two observations per live stream and scalar counters; returned chains and
buffers belong to the caller. No simulator/contact snapshot is saved here.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from numbers import Integral

import numpy as np
import torch

from .official import physical_metrics
from .official_collection import DEVELOPMENT_SEEDS
from .policy_protocol import BudgetExceeded
from .policy_returns import ActionBlock, BlockBoundary, training_reward
from .policy_rollout import RolloutPlan, _cpu, _cpu_sample, _features_and_value, _observation_copy


SHAPES = ("T", "L")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Scene seeds must be uint32 integers")
    return value


class SceneExhausted(RuntimeError):
    """A fixed scene list is exhausted; it cannot wrap or be sampled again."""


class SceneCursor:
    """Consume fixed, unique per-shape scenes exactly once, without wrapping.

    ``excluded`` is the complete development/evaluation exclusion set supplied
    by run configuration, not merely this module's known development seeds.
    The latter must be included. Overlapping training scenes are rejected rather
    than silently skipped. T/L may share a seed for matched scenes, but a shape
    cannot use the same seed twice. Failed reset attempts still consume scenes.
    """

    def __init__(self, scenes: dict[str, list[int]], excluded: set[int]):
        if not isinstance(scenes, dict) or set(scenes) != set(SHAPES):
            raise ValueError("Provide fixed T and L scene lists")
        self.excluded = frozenset(_seed(seed) for seed in excluded)
        if not set(DEVELOPMENT_SEEDS).issubset(self.excluded):
            raise ValueError("The complete excluded set must include known development seeds")
        self.scenes = {}
        for shape in SHAPES:
            values = tuple(_seed(seed) for seed in scenes[shape])
            if not values or len(values) != len(set(values)):
                raise ValueError("Each shape needs a nonempty unique fixed scene list")
            if set(values) & self.excluded:
                raise ValueError("Training scenes overlap excluded development/evaluation scenes")
            self.scenes[shape] = values
        self._indices = {shape: 0 for shape in SHAPES}

    def remaining(self, shape):
        if shape not in SHAPES:
            raise ValueError("Unknown scene stream")
        return len(self.scenes[shape]) - self._indices[shape]

    def next(self, shape):
        if not self.remaining(shape):
            raise SceneExhausted(f"Fixed {shape} scenes are exhausted; no wraparound is allowed")
        seed = self.scenes[shape][self._indices[shape]]
        self._indices[shape] += 1
        return seed

    def state_dict(self):
        return {"version": 1, "scenes": {s: list(self.scenes[s]) for s in SHAPES},
                "excluded": sorted(self.excluded), "indices": dict(self._indices)}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if (not isinstance(state, dict) or set(state) != set(expected)
                or any(state[key] != expected[key] for key in ("version", "scenes", "excluded"))
                or not isinstance(state["indices"], dict) or set(state["indices"]) != set(SHAPES)):
            raise ValueError("Scene state differs from configured scene/exclusion provenance")
        indices = {}
        for shape in SHAPES:
            index = _integer(state["indices"][shape], "scene cursor")
            if not self._indices[shape] <= index <= len(self.scenes[shape]):
                raise ValueError("Scene restoration cannot rewind or exceed a fixed scene list")
            indices[shape] = index
        self._indices = indices


def _statistics():
    return {"controls": 0, "control_physics_substeps": 0, "sampled_plans": 0,
            "executed_plans": 0, "sampled_but_unexecuted_plans": 0,
            "reset_attempts": 0, "reset_completed": 0, "episodes_started": 0,
            "successes": 0, "timeouts": 0, "contact_controls": 0,
            "unknown_contact_controls": 0, "nonzero_raw_reward_controls": 0,
            "nonzero_training_reward_controls": 0, "absolute_coverage_delta": 0.0,
            "max_coverage": 0.0, "last_coverage": None}


@dataclass
class _Stream:
    history: list[dict] | None = None
    active: bool = False
    episode_id: str | None = None
    reset_seed: int | None = None
    episode_controls: int = 0
    coverage: float | None = None
    success_seen: bool = False
    terminal_reason: str | None = None
    statistics: dict = field(default_factory=_statistics)


class ContinuousCollector:
    """Retain physical episodes/history across batches; alternate sampled plans.

    ``generators`` are explicit local T/L torch generators. Shape, scene IDs,
    coverage and contacts are metadata/reward labels and never actor or critic
    inputs. The environments are caller-owned and are never silently replaced.
    Each returned list is one stream/episode fragment, ending with c=0. A live
    episode's final block has b=1,c=0 and resumes physically next batch without
    resetting; its unexecuted action suffix is discarded and the next plan is
    sampled from the actual new history.

    Durable reservations use the C ledger's train phase, including resets as
    separate settling substeps. Time/control/plan exhaustion cuts a nonempty
    action prefix and preserves its bootstrap observation. A sample interrupted
    before its first control remains budgeted but produces no zero-length block.
    Unexpected simulator/model errors taint this instance and prohibit retries.
    """

    def __init__(self, official, actor, critic, environments, generators, scenes,
                 budget, *, episode_limit=300, gamma=0.99, beta=1.0,
                 metrics_fn=physical_metrics):
        if set(environments) != set(SHAPES) or set(generators) != set(SHAPES):
            raise ValueError("Provide exactly T/L environments and local generators")
        if any(getattr(environments[s], "project_shape", s) != s for s in SHAPES):
            raise ValueError("Environment stream labels differ from their actual shapes")
        if any(not isinstance(generators[s], torch.Generator) for s in SHAPES):
            raise ValueError("Each stream needs an explicit torch.Generator")
        if not isinstance(scenes, SceneCursor):
            raise ValueError("Use a fixed, excluded-scene-checked SceneCursor")
        _integer(episode_limit, "episode_limit", 1)
        training_reward(0.0, 0.0, 0.0, gamma=gamma, beta=beta)
        self.official, self.actor, self.critic = official, actor, critic
        self.environments, self.generators = dict(environments), dict(generators)
        self.scenes, self.budget = scenes, budget
        self.episode_limit, self.gamma, self.beta = episode_limit, float(gamma), float(beta)
        self.metrics_fn = metrics_fn
        self._streams = {shape: _Stream() for shape in SHAPES}
        self._next_stream = 0
        self._batch_calls = 0
        self._failed = None
        self._boundary_requested = False
        self._last_batch = {"plans": 0, "controls": 0, "plan_order": [], "stop_reason": "not_started"}

    @property
    def is_episode_boundary(self):
        """True only when every stream is unstarted or genuinely done/timed out."""
        return not any(stream.active for stream in self._streams.values())

    def request_boundary(self):
        """Drain live episodes under normal budgets; pause each finished stream.

        This schedules no controls and never resets or fabricates termination.
        Subsequent batches may need several calls to reach a real boundary. If
        a budget expires first, the remaining episode is still non-resumable.
        """
        if self._failed is not None:
            raise RuntimeError("A failed collector cannot drain unknown physical state")
        self._boundary_requested = True

    def release_boundary(self):
        """Permit new episodes only after all requested streams truly ended."""
        if not self.is_episode_boundary or self._failed is not None:
            raise RuntimeError("Boundary drain is incomplete; active episodes cannot be restarted")
        self._boundary_requested = False

    def _remaining(self, kind):
        return _integer(self.budget.remaining(kind, phase="train"), f"remaining {kind}")

    def _choose_stream(self):
        for offset in range(len(SHAPES)):
            index = (self._next_stream + offset) % len(SHAPES)
            shape = SHAPES[index]
            if self._streams[shape].active:
                return index, shape
            if not self._boundary_requested and self.scenes.remaining(shape) and self._remaining("reset"):
                return index, shape
        return None

    def _start_episode(self, shape):
        stream = self._streams[shape]
        if stream.active:
            raise RuntimeError("A live episode cannot be reset at a batch boundary")
        self.budget.check_time()
        reservation = self.budget.reserve("reset", phase="train", shape=shape, amount=1)
        seed = self.scenes.next(shape)
        stream.statistics["reset_attempts"] += 1
        stream.active = True  # An exception leaves unknown physical state, not a resumable boundary.
        stream.reset_seed = seed
        env = self.environments[shape]
        env.seed(seed)
        observation = env.reset()
        stream.statistics["reset_completed"] += 1
        self.budget.mark_completed(reservation)
        observation = _observation_copy(observation)
        if hasattr(env, "capture_state"):
            from .collection import reset_issues
            from .envs.official import official_shape
            issues = reset_issues(official_shape(shape), env.capture_state())
            if issues:
                raise ValueError(f"Unsafe reset rejected without hidden retry controls: {issues}")
        coverage = float(self.metrics_fn(env)["coverage"])
        training_reward(coverage, coverage, 0.0, gamma=self.gamma, beta=self.beta)
        if coverage > 0.95:
            raise ValueError("A training scene is already successful at reset")
        stream.statistics["episodes_started"] += 1
        stream.episode_id = f"train:{shape}:{seed}:{stream.statistics['episodes_started']}"
        stream.episode_controls = 0
        stream.coverage, stream.success_seen = coverage, False
        stream.terminal_reason = None
        stream.history = [observation, observation]
        stream.statistics["max_coverage"] = max(stream.statistics["max_coverage"], coverage)
        stream.statistics["last_coverage"] = coverage

    def _record_control(self, shape, stream, sample, action_index, reservation,
                        before, before_coverage, result, batch_plan_index):
        if not isinstance(result, tuple) or len(result) != 4:
            raise ValueError("Expected original official four-tuple step output")
        next_observation, raw_reward, upstream_done, info = result
        if not isinstance(info, dict):
            raise ValueError("Step info must preserve original simulator diagnostics")
        observation = _observation_copy(next_observation)
        metrics = self.metrics_fn(self.environments[shape])
        coverage = float(metrics["coverage"])
        reward = training_reward(before_coverage, coverage, raw_reward,
                                 success_already_seen=stream.success_seen,
                                 gamma=self.gamma, beta=self.beta)
        timeout = (stream.episode_controls >= self.episode_limit
                   or bool(info.get("TimeLimit.truncated", False)))
        if upstream_done and not reward.terminated and not timeout:
            raise ValueError("Unexpected upstream done below the strict success threshold")
        contacts = info.get("n_contacts")
        if contacts is not None:
            if isinstance(contacts, bool) or not isinstance(contacts, Integral) or contacts < 0:
                raise ValueError("n_contacts must be a nonnegative integer when provided")
            contacts = int(contacts)
        iou = metrics.get("iou")
        if iou is not None and (not math.isfinite(iou) or not 0 <= iou <= 1):
            raise ValueError("Reported IoU must be finite and in [0, 1]")
        stream.history = [stream.history[-1], observation]
        stream.coverage = coverage
        stream.success_seen = stream.success_seen or reward.first_success
        stats = stream.statistics
        stats["contact_controls"] += int(contacts is not None and contacts > 0)
        stats["unknown_contact_controls"] += int(contacts is None)
        stats["nonzero_raw_reward_controls"] += int(raw_reward != 0)
        stats["nonzero_training_reward_controls"] += int(reward.reward != 0)
        stats["absolute_coverage_delta"] += abs(coverage - before_coverage)
        stats["max_coverage"] = max(stats["max_coverage"], coverage)
        stats["last_coverage"] = coverage
        boundary = (BlockBoundary.SUCCESS if reward.terminated else
                    BlockBoundary.TIMEOUT if timeout else BlockBoundary.CONTINUE)
        if boundary != BlockBoundary.CONTINUE:
            stream.active = False
            stream.terminal_reason = boundary.value
            stats["successes" if reward.terminated else "timeouts"] += 1
        env = self.environments[shape]
        substeps = env.sim_hz // env.control_hz if hasattr(env, "sim_hz") and hasattr(env, "control_hz") else None
        if substeps is not None:
            stats["control_physics_substeps"] += substeps
        record = {"observation_before": before, "observation_after": _observation_copy(observation),
                  "action": sample.actions[0, action_index].clone(), "action_index": action_index,
                  "coverage_before": before_coverage, "coverage": coverage,
                  "coverage_delta": coverage - before_coverage, "iou": None if iou is None else float(iou),
                  "raw_reward": float(raw_reward), "upstream_done": bool(upstream_done),
                  "training_reward": reward.reward, "shaping": reward.shaping,
                  "first_success": reward.first_success, "terminated": reward.terminated,
                  "n_contacts": contacts, "nonzero_raw_reward": bool(raw_reward != 0),
                  "nonzero_training_reward": bool(reward.reward != 0),
                  "control_index": stream.episode_controls, "stream_control_index": stats["controls"],
                  "collector_control_index": sum(s.statistics["controls"] for s in self._streams.values()),
                  "global_control_attempt": self.budget.snapshot()["reserved"]["train"]["control"],
                  "reservation_id": reservation, "control_physics_substeps": substeps,
                  "remaining_train_controls": self._remaining("control"),
                  "batch_index": self._batch_calls, "batch_plan_index": batch_plan_index}
        return record, boundary

    @torch.no_grad()
    def collect_batch(self, max_plans, max_controls, actor_version):
        """Collect an on-policy batch, retaining live episodes for the next call.

        Return independent GAE fragments for each (stream, episode), grouped in
        order of their first sample. ``batch_plan_index`` preserves the actual
        interleaved acquisition order. Caps can exceed remaining authorization:
        collection stops conservatively at the ledger's remaining budget instead
        of resetting environments to fill a nominal batch.
        """
        _integer(max_plans, "max_plans", 1)
        _integer(max_controls, "max_controls", 1)
        if not isinstance(actor_version, str) or not actor_version.strip():
            raise ValueError("Record an explicit actor version for each fresh batch")
        if self._failed is not None:
            raise RuntimeError("Collector failed previously; unknown physical state cannot be retried")
        self._batch_calls += 1
        fragments = OrderedDict()
        controls = sampled = 0
        plan_order = []
        stop_reason = "batch_cap"
        if hasattr(self.critic, "eval"):
            self.critic.eval()
        try:
            while sampled < max_plans and controls < max_controls:
                if self._boundary_requested and self.is_episode_boundary:
                    stop_reason = "episode_boundary"
                    break
                self.budget.check_time()
                if not self._remaining("control") or not self._remaining("plan"):
                    stop_reason = "budget_exhausted"
                    break
                selected = self._choose_stream()
                if selected is None:
                    stop_reason = ("episode_boundary" if self._boundary_requested else
                                   "scenes_exhausted" if all(not self.scenes.remaining(s) for s in SHAPES)
                                   else "reset_budget_exhausted")
                    break
                index, shape = selected
                stream = self._streams[shape]
                if not stream.active:
                    self._start_episode(shape)
                observations = self.official.observation_tensors(stream.history)
                features, value = _features_and_value(self.actor, self.critic, observations)
                self.budget.check_time()
                reservation = self.budget.reserve("plan", phase="train", shape=shape, amount=1)
                sample = self.actor.sample(observations, generator=self.generators[shape],
                                           mode="training", actor_version=actor_version)
                stream.statistics["sampled_plans"] += 1
                self.budget.mark_completed(reservation)
                sample = _cpu_sample(sample)
                sampled += 1
                plan_order.append(shape)
                self._next_stream = (index + 1) % len(SHAPES)
                records = []
                boundary = BlockBoundary.CONTINUE
                interrupted = False
                for action_index, action in enumerate(sample.actions[0]):
                    if controls >= max_controls or not self._remaining("control"):
                        stop_reason = "budget_exhausted" if not self._remaining("control") else "batch_cap"
                        interrupted = True
                        break
                    try:
                        self.budget.check_time()
                        reservation = self.budget.reserve("control", phase="train", shape=shape, amount=1)
                    except BudgetExceeded:
                        stop_reason, interrupted = "budget_or_time_exhausted", True
                        break
                    before = _observation_copy(stream.history[-1])
                    before_coverage = stream.coverage
                    result = self.environments[shape].step(action.numpy().astype(np.float64, copy=True))
                    controls += 1
                    stream.episode_controls += 1
                    stream.statistics["controls"] += 1
                    self.budget.mark_completed(reservation)
                    record, boundary = self._record_control(shape, stream, sample, action_index, reservation,
                                                            before, before_coverage, result, sampled - 1)
                    records.append(record)
                    if boundary != BlockBoundary.CONTINUE:
                        break
                if not records:
                    stream.statistics["sampled_but_unexecuted_plans"] += 1
                    break
                final_observation = self.official.observation_tensors(stream.history)
                final_value = None if boundary == BlockBoundary.SUCCESS else _features_and_value(
                    self.actor, self.critic, final_observation)[1]
                block = ActionBlock(episode_id=stream.episode_id,
                                    rewards=tuple(record["training_reward"] for record in records),
                                    value=value, final_observation_value=final_value, boundary=boundary)
                plan = RolloutPlan(sample=sample, block=block, shape=shape, reset_seed=stream.reset_seed,
                                   control_records=records,
                                   final_observation={key: _cpu(value) for key, value in final_observation.items()},
                                   features=_cpu(features))
                fragments.setdefault(stream.episode_id, []).append(plan)
                stream.statistics["executed_plans"] += 1
                if not stream.active:
                    stream.history = None
                if interrupted:
                    break
        except BudgetExceeded:
            stop_reason = "budget_or_time_exhausted"
        except BaseException as exc:
            self._failed = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._last_batch = {"plans": sampled, "controls": controls, "plan_order": plan_order,
                                "stop_reason": stop_reason}
        # Each GAE fragment ends here, even though its physical episode may live
        # on in the stream. Keep already terminal success/timeout masks intact.
        result = []
        for fragment in fragments.values():
            tail = fragment[-1]
            if tail.block.boundary == BlockBoundary.CONTINUE:
                fragment[-1] = replace(tail, block=replace(tail.block, boundary=BlockBoundary.BUFFER_CUTOFF))
            result.append(fragment)
        return result

    def summary(self):
        per_shape = {shape: deepcopy(self._streams[shape].statistics) for shape in SHAPES}
        total = sum(stats["controls"] for stats in per_shape.values())
        return {"batch_calls": self._batch_calls, "by_shape": per_shape,
                "controls": total, "sampled_plans": sum(s["sampled_plans"] for s in per_shape.values()),
                "control_shares": {shape: stats["controls"] / total if total else 0.0
                                   for shape, stats in per_shape.items()},
                "T_to_L_control_ratio": per_shape["T"]["controls"] / per_shape["L"]["controls"]
                    if per_shape["L"]["controls"] else None,
                "active_shapes": [shape for shape in SHAPES if self._streams[shape].active],
                "is_episode_boundary": self.is_episode_boundary,
                "draining": self._boundary_requested,
                "resumable": self.is_episode_boundary and self._failed is None,
                "scenes": self.scenes.state_dict(),
                "exhausted_shapes": [shape for shape in SHAPES if not self.scenes.remaining(shape)],
                "last_batch": deepcopy(self._last_batch), "failed": self._failed}

    def state_dict(self, require_boundary=True):
        """Save scene/counter/local-RNG state, without claiming physical restart.

        With require_boundary=False, a non-resumable metadata record may be
        saved for audit/weight checkpoints. It still contains no observations or
        physics snapshot and ``load_state_dict`` always rejects active episodes.
        Never execute extra controls or force a reset merely to make a checkpoint.
        A successfully drained boundary saves its gate as released for a future
        process (saved_at_requested_boundary=True); the current instance remains
        paused until its caller explicitly releases it after saving.
        """
        resumable = self.is_episode_boundary and self._failed is None
        if require_boundary and not resumable:
            raise RuntimeError("Checkpoint is not resumable: a live/failed episode has no contact-cache restore")
        return {"version": 1, "resumable": resumable, "episode_boundary": self.is_episode_boundary,
                "episode_limit": self.episode_limit, "gamma": self.gamma, "beta": self.beta,
                "scene_cursor": self.scenes.state_dict(), "next_stream": self._next_stream,
                "boundary_requested": self._boundary_requested if not resumable else False,
                "saved_at_requested_boundary": self._boundary_requested and resumable,
                "batch_calls": self._batch_calls, "failed": self._failed,
                "streams": {shape: {"active": stream.active, "episode_id": stream.episode_id,
                    "reset_seed": stream.reset_seed, "episode_controls": stream.episode_controls,
                    "coverage": stream.coverage, "success_seen": stream.success_seen,
                    "terminal_reason": stream.terminal_reason, "statistics": deepcopy(stream.statistics)}
                    for shape, stream in self._streams.items()},
                "generators": {shape: {"device": str(generator.device), "state": _cpu(generator.get_state())}
                               for shape, generator in self.generators.items()},
                "last_batch": deepcopy(self._last_batch)}

    def load_state_dict(self, state):
        """Restore only into a fresh collector and only at episode boundaries."""
        if self._batch_calls or any(stream.episode_id is not None or stream.active
                                    for stream in self._streams.values()):
            raise RuntimeError("Restore requires a newly constructed collector with untouched environments")
        if (not isinstance(state, dict) or state.get("version") != 1
                or state.get("resumable") is not True or state.get("episode_boundary") is not True
                or state.get("failed") is not None or state.get("boundary_requested") is not False):
            raise ValueError("Refusing mid-episode/failed physical resume without a contact-cache restore")
        if any(state.get(key) != getattr(self, key) for key in ("episode_limit", "gamma", "beta")):
            raise ValueError("Restored reward/episode configuration differs")
        if set(state["streams"]) != set(SHAPES) or set(state["generators"]) != set(SHAPES):
            raise ValueError("Restored stream/generator labels differ")
        next_stream = _integer(state["next_stream"], "next_stream")
        if next_stream >= len(SHAPES):
            raise ValueError("Invalid round-robin cursor")
        batch_calls = _integer(state["batch_calls"], "batch_calls")
        streams = {}
        checked_rng = {}
        for shape in SHAPES:
            saved = state["streams"][shape]
            if saved["active"] is not False:
                raise ValueError("A restored stream must be at an actual episode boundary")
            if set(saved["statistics"]) != set(_statistics()):
                raise ValueError("Unknown stream counter schema")
            stats = deepcopy(saved["statistics"])
            for key, default in _statistics().items():
                if type(default) is int:
                    _integer(stats[key], key)
                elif stats[key] is not None and (not math.isfinite(stats[key]) or stats[key] < 0):
                    raise ValueError("Restored stream metrics must be finite/nonnegative")
            episode_controls = _integer(saved["episode_controls"], "episode_controls")
            if saved["reset_seed"] is not None:
                _seed(saved["reset_seed"])
            if type(saved["success_seen"]) is not bool:
                raise ValueError("Restored first-success state must be explicit")
            if saved["episode_id"] is None:
                if (saved["reset_seed"] is not None or episode_controls or saved["coverage"] is not None
                        or saved["success_seen"] or saved["terminal_reason"] is not None
                        or stats["episodes_started"]):
                    raise ValueError("Unstarted stream contains physical episode state")
            elif (not isinstance(saved["episode_id"], str) or not saved["episode_id"]
                  or saved["terminal_reason"] not in ("success", "timeout")
                  or saved["reset_seed"] is None or not 1 <= episode_controls <= self.episode_limit
                  or saved["success_seen"] != (saved["terminal_reason"] == "success")
                  or saved["coverage"] is None or not math.isfinite(saved["coverage"])
                  or not 0 <= saved["coverage"] <= 1):
                raise ValueError("A boundary restore needs actual success/timeout episode metadata")
            if (stats["successes"] + stats["timeouts"] != stats["episodes_started"]
                    or not stats["reset_attempts"] == stats["reset_completed"] == stats["episodes_started"]
                    or stats["controls"] < episode_controls
                    or stats["executed_plans"] > stats["sampled_plans"]):
                raise ValueError("Boundary episode and operation counters disagree")
            rng = state["generators"][shape]
            current = self.generators[shape]
            if (torch.device(rng["device"]).type != torch.device(current.device).type
                    or not isinstance(rng["state"], torch.Tensor) or rng["state"].dtype != torch.uint8
                    or rng["state"].shape != current.get_state().shape):
                raise ValueError("Local generator algorithm/state differs; cannot promise replay")
            checked_rng[shape] = torch.Generator(device=current.device)
            checked_rng[shape].set_state(rng["state"].cpu())
            streams[shape] = _Stream(active=False, episode_id=saved["episode_id"],
                                    reset_seed=saved["reset_seed"], episode_controls=episode_controls,
                                    coverage=saved["coverage"], success_seen=saved["success_seen"],
                                    terminal_reason=saved["terminal_reason"], statistics=stats)
        scene_copy = SceneCursor({shape: list(self.scenes.scenes[shape]) for shape in SHAPES},
                                 set(self.scenes.excluded))
        scene_copy.load_state_dict(state["scene_cursor"])
        if any(state["scene_cursor"]["indices"][shape] != streams[shape].statistics["reset_attempts"]
               for shape in SHAPES):
            raise ValueError("Scene cursors and consumed reset attempts disagree")
        self.scenes.load_state_dict(scene_copy.state_dict())
        for shape in SHAPES:
            self.generators[shape].set_state(checked_rng[shape].get_state())
        self._streams, self._next_stream = streams, next_stream
        self._boundary_requested = False
        self._batch_calls = batch_calls
        self._last_batch = deepcopy(state["last_batch"])
