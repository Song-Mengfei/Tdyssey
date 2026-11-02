"""Milestone C orchestration, without granting permission to run an experiment.

The loop owns only its current buffer and scalar summaries. Checkpoint resume
requires a real joint episode boundary and the same permanent budget counters;
an audit/weight checkpoint at a live episode is intentionally non-resumable.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

import torch

from .policy_checkpoint import _restore_rng, save_update_checkpoint
from .policy_protocol import BudgetExceeded, CConfig
from .policy_rl import PARAMETER_PREFIXES, SamplerConfig


_CHECKPOINT_RESERVATION_BYTES = 64 * 1024**2


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _primitive(value):
    """Detach logging from buffers/graphs; reject unsupported or nonfinite data."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _primitive(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_primitive(item) for item in value]
    raise ValueError("Loop logs/state must contain finite primitives, never tensors or buffer references")


def _diagnostic_primitive(value):
    """Keep failed first-batch scalar evidence JSON-safe without hiding NaN/Inf."""
    if type(value) is float and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {key: _diagnostic_primitive(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_diagnostic_primitive(item) for item in value]
    return _primitive(value)


def validate_first_batch(row, *, logprob_tolerance=1e-4):
    """Validate completed job batch one from actual updater/ledger evidence.

    This checks reported changes, not merely optimizer invocation. Frozen weights
    and the fixed buffer are checked by PPOUpdater before and after its pass.
    A failure raises ValueError so train_batches can retain the row and stop.
    """
    def require(condition, message):
        if not condition:
            raise ValueError("First batch gate: " + message)

    def finite(value, name, minimum=None):
        require(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
        require(minimum is None or value >= minimum, name + " is below its allowed minimum")
        return value

    def exact(value, expected, name):
        require(type(value) is int and value == expected, name + " differs")

    finite(logprob_tolerance, "logprob_tolerance", 0)
    require(isinstance(row, dict) and row.get("event") == "batch", "expected a completed batch row")
    exact(row.get("batch"), 1, "job batch")
    exact(row.get("requested_passes"), 1, "first batch requested passes")
    exact(row.get("budget_allowed_passes"), 1, "first batch allowed passes")
    update, collection, counts = (row.get(key) for key in ("update", "collection", "counts"))
    require(all(isinstance(item, dict) for item in (update, collection, counts)), "missing update/collection/counts")
    exact(update.get("batch_sequence"), 0, "buffer sequence")
    exact(update.get("requested_passes"), 1, "updater requested passes")
    exact(update.get("completed_passes"), 1, "completed passes")
    exact(update.get("actor_steps"), 1, "actor steps")
    exact(update.get("critic_steps"), 1, "critic steps")
    require(update.get("stop_reason") in ("completed", "target_kl"), "update did not complete")
    frozen_digest = update.get("frozen_digest")
    require(update.get("frozen_unchanged") is True and type(frozen_digest) is str
            and len(frozen_digest) == 64 and all(char in "0123456789abcdef" for char in frozen_digest),
            "frozen preservation evidence missing")
    entries = update.get("passes")
    require(isinstance(entries, list) and len(entries) == 1, "expected exactly one pass report")
    entry = entries[0]
    require(isinstance(entry, dict), "invalid pass report")
    exact(entry.get("index"), 0, "first pass index")
    for owner in ("actor", "critic"):
        finite(entry.get(owner + "_loss"), owner + " loss")
        require(finite(entry.get(owner + "_grad_norm"), owner + " gradient norm", 0) > 0,
                owner + " gradient norm is zero")
        change = entry.get(owner + "_weight_change")
        require(isinstance(change, dict), owner + " weight change evidence missing")
        names = change.get("changed_names")
        require(isinstance(names, list) and names and all(type(name) is str and name for name in names)
                and len(set(names)) == len(names), owner + " changed names missing or invalid")
        exact(change.get("changed_tensor_count"), len(names), owner + " changed tensor count")
        require(finite(change.get("l2"), owner + " weight change l2", 0) > 0,
                owner + " weights did not change")
        require(finite(change.get("max_abs"), owner + " maximum weight change", 0) > 0,
                owner + " weights did not change")
    finite(entry.get("critic_mse"), "critic MSE", 0)
    if entry.get("critic_explained_variance") is not None:
        finite(entry["critic_explained_variance"], "critic explained variance")
    plans, controls = collection.get("plans"), collection.get("controls")
    require(type(plans) is int and plans > 0 and type(controls) is int and controls > 0,
            "first batch has no plans/controls")
    behavior = update.get("entry")
    require(isinstance(behavior, dict), "missing behavior entry metrics")
    require(finite(behavior.get("max_logprob_error"), "behavior log-probability error", 0)
            <= logprob_tolerance, "behavior probabilities differ at entry")
    lower = math.exp(-logprob_tolerance)
    upper = math.exp(logprob_tolerance)
    ratio_min = finite(behavior.get("ratio_min"), "behavior minimum ratio", 0)
    ratio_max = finite(behavior.get("ratio_max"), "behavior maximum ratio", 0)
    require(lower <= ratio_min <= ratio_max <= upper, "behavior ratio differs from one")
    require(finite(behavior.get("sample_approx_kl"), "behavior KL", 0) <= logprob_tolerance,
            "behavior KL differs at entry")
    require(finite(behavior.get("clip_fraction"), "behavior clip fraction", 0) == 0,
            "behavior entry is already clipped")
    exact(behavior.get("transition_count"), plans * 100, "behavior transition count")
    after = entry.get("after")
    require(isinstance(after, dict), "missing post-update probability metrics")
    for name in ("max_logprob_error", "ratio_min", "ratio_max", "sample_approx_kl", "clip_fraction"):
        finite(after.get(name), "post-update " + name, 0)
    exact(after.get("transition_count"), plans * 100, "post-update transition count")
    for status in ("reserved", "completed"):
        require(isinstance(counts.get(status), dict), "missing " + status + " budget counts")
        global_counts = counts[status].get("global")
        train_counts = counts[status].get("train")
        require(isinstance(global_counts, dict) and isinstance(train_counts, dict), "missing global/train counts")
        for kind in ("batch", "actor_step", "critic_step"):
            exact(global_counts.get(kind), 1, status + " " + kind)
        exact(train_counts.get("plan"), plans, status + " training plans")
        exact(train_counts.get("control"), controls, status + " training controls")
        require(type(train_counts.get("reset")) is int and train_counts["reset"] > 0,
                status + " training resets missing")
    require(counts["reserved"] == counts["completed"], "incomplete operation reservations remain")
    return {"status": "passed", "batch": 1, "completed_passes": 1,
            "transition_count": plans * 100, "max_logprob_error": behavior["max_logprob_error"],
            "actor_weight_change": deepcopy(entry["actor_weight_change"]),
            "critic_weight_change": deepcopy(entry["critic_weight_change"]),
            "frozen_preservation": "enforced_by_updater", "frozen_digest": frozen_digest}


def _counts(budget):
    snapshot = budget.snapshot()
    return {key: deepcopy(snapshot[key]) for key in ("reserved", "completed")}


def _training_counts(counts):
    """Independent development/diagnostic operations do not rewind training."""
    return {status: {phase: deepcopy(counts[status][phase]) for phase in ("train", "global")}
            for status in ("reserved", "completed")}


def _validate_training_state(collector_state, updater_state, counts):
    authoritative = _training_counts(counts)
    if authoritative["reserved"] != authoritative["completed"]:
        raise ValueError("Incomplete training reservations cannot be resumed")
    completed = authoritative["completed"]
    expected_updater = {"actor_steps": "actor_step", "critic_steps": "critic_step", "batches": "batch"}
    if any(updater_state.get(name) != completed["global"][kind] for name, kind in expected_updater.items()):
        raise ValueError("Updater counters differ from the authoritative training ledger")
    streams = collector_state["streams"]
    expected_collector = {"controls": "control", "sampled_plans": "plan", "reset_completed": "reset"}
    if any(sum(stream["statistics"][name] for stream in streams.values()) != completed["train"][kind]
           for name, kind in expected_collector.items()):
        raise ValueError("Collector counters differ from the authoritative training ledger")
    if sum(stream["statistics"]["reset_attempts"] for stream in streams.values()) != completed["train"]["reset"]:
        raise ValueError("Collector reset attempts differ from the authoritative training ledger")


def _resources():
    # Query an already initialized allocator only; a CPU fake test must not
    # initialize CUDA merely to produce resource telemetry.
    if not torch.cuda.is_initialized():
        return {"cuda": None}
    return {"cuda": {"allocated_bytes": int(torch.cuda.memory_allocated()),
                     "reserved_bytes": int(torch.cuda.memory_reserved()),
                     "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                     "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}}


def _diagnostic_totals():
    return {"controls": 0, "plans": 0, "contact_controls": 0, "unknown_contact_controls": 0,
            "nonzero_raw_reward_controls": 0, "nonzero_training_reward_controls": 0,
            "nonzero_shaping_controls": 0, "raw_reward_sum": 0.0, "training_reward_sum": 0.0,
            "shaping_sum": 0.0, "coverage_delta_sum": 0.0, "absolute_coverage_delta": 0.0,
            "max_coverage": 0.0, "last_coverage": None}


def _batch_diagnostics(fragments):
    per_shape = {shape: _diagnostic_totals() for shape in ("T", "L")}
    all_contacts_known_zero = True
    all_raw_zero = True
    controls = 0
    for fragment in fragments:
        for plan in fragment:
            stats = per_shape[plan.shape]
            stats["plans"] += 1
            for record in plan.control_records:
                contacts = record.get("n_contacts")
                raw, shaped = float(record["raw_reward"]), float(record["training_reward"])
                shaping = float(record["shaping"])
                coverage, before = float(record["coverage"]), float(record["coverage_before"])
                if any(not math.isfinite(value) for value in (raw, shaped, shaping, coverage, before)):
                    raise ValueError("Collected diagnostics must be finite")
                controls += 1
                stats["controls"] += 1
                stats["contact_controls"] += int(contacts is not None and contacts > 0)
                stats["unknown_contact_controls"] += int(contacts is None)
                stats["nonzero_raw_reward_controls"] += int(raw != 0)
                stats["nonzero_training_reward_controls"] += int(shaped != 0)
                stats["nonzero_shaping_controls"] += int(shaping != 0)
                stats["raw_reward_sum"] += raw
                stats["training_reward_sum"] += shaped
                stats["shaping_sum"] += shaping
                stats["coverage_delta_sum"] += coverage - before
                stats["absolute_coverage_delta"] += abs(coverage - before)
                stats["max_coverage"] = max(stats["max_coverage"], before, coverage)
                stats["last_coverage"] = coverage
                all_contacts_known_zero = all_contacts_known_zero and contacts is not None and contacts == 0
                all_raw_zero = all_raw_zero and raw == 0
    return {"by_shape": per_shape, "controls": controls,
            "plans": sum(stats["plans"] for stats in per_shape.values()),
            "empty_for_optional_stop": controls > 0 and all_contacts_known_zero and all_raw_zero}


def _accumulate(total, batch):
    for shape, stats in batch["by_shape"].items():
        for key, value in stats.items():
            if key == "last_coverage":
                if value is not None:
                    total[shape][key] = value
            elif key == "max_coverage":
                total[shape][key] = max(total[shape][key], value)
            else:
                total[shape][key] += value


def train_batches(collector, updater, budget, *, batch_plans=16, batch_controls=128,
                  first_batch_passes=1, passes=3, max_batches=24, control_target=2048,
                  checkpoint_every=4, stop_after_empty_batches=0, on_checkpoint=None,
                  on_replay=None, log=None, resume_state=None, first_batch_gate=None):
    """Collect fresh continuous batches and consume each fixed PPO buffer once.

    Limits/targets are absolute job totals, including on resume. The first job
    batch uses first_batch_passes; later fresh batches use up to passes. If fewer
    optimizer reservations remain, the actual pass count is explicitly reduced.
    A requested checkpoint drains active episodes under ordinary batch/budget
    limits; finished streams pause until saving and release. Budget tails never
    force extra interactions or pretend an active episode can be restored.

    on_replay(*, fragments, loop_state) runs once for the first valid job batch.
    on_checkpoint(*, resumable, terminal, loop_state) runs after buffers release.
    Callbacks may return finite primitive metadata. log receives primitive rows.
    Only current/last scalar reports survive; no historical buffer is retained.
    Optional first_batch_gate(row) checks completed job batch one before any next
    collection. Failed evidence is retained and only a terminal audit is allowed.
    """
    for value, name, minimum in ((batch_plans, "batch_plans", 1), (batch_controls, "batch_controls", 1),
            (first_batch_passes, "first_batch_passes", 1), (passes, "passes", 1),
            (max_batches, "max_batches", 0), (control_target, "control_target", 0),
            (checkpoint_every, "checkpoint_every", 1), (stop_after_empty_batches, "stop_after_empty_batches", 0)):
        _integer(value, name, minimum)
    maximum_passes = updater.config.max_passes
    if max(first_batch_passes, passes) > maximum_passes:
        raise ValueError("Loop passes exceed the updater's declared fixed-buffer protocol")
    if first_batch_gate is not None and not callable(first_batch_gate):
        raise TypeError("first_batch_gate must be callable")
    started = time.perf_counter()
    per_shape = {shape: _diagnostic_totals() for shape in ("T", "L")}
    empty_streak = 0
    checkpoint_callbacks = 0
    replay_done = updater.batches > 0
    next_checkpoint_batch = (updater.batches // checkpoint_every + 1) * checkpoint_every
    checkpoint_pending = bool(collector.summary().get("draining", False))
    last_checkpoint_signature = None
    last_checkpoint = last_update = last_collection = replay_metadata = None
    diagnostic_scope = "whole_job" if updater.batches == 0 else "this_invocation_without_resume_diagnostics"
    first_batch_evidence = None
    if resume_state is not None:
        resume_state = _primitive(resume_state)
        if (resume_state.get("schema_version") != 1
                or resume_state.get("batches") != updater.batches
                or resume_state.get("actor_steps") != updater.actor_steps
                or resume_state.get("critic_steps") != updater.critic_steps
                or resume_state.get("controls") != collector.summary()["controls"]):
            raise ValueError("Loop resume state disagrees with restored actor/collector counters")
        per_shape = deepcopy(resume_state["diagnostics_by_shape"])
        empty_streak = _integer(resume_state["empty_streak"], "empty_streak")
        checkpoint_callbacks = _integer(resume_state["checkpoint_callbacks"], "checkpoint_callbacks")
        replay_done = bool(resume_state["replay_done"])
        next_checkpoint_batch = _integer(resume_state["next_checkpoint_batch"], "next_checkpoint_batch", 1)
        checkpoint_pending = bool(resume_state["checkpoint_pending"])
        last_checkpoint_signature = resume_state.get("last_checkpoint_signature")
        diagnostic_scope = resume_state["diagnostic_scope"]
        first_batch_evidence = resume_state.get("first_batch_gate")
        if isinstance(first_batch_evidence, dict) and first_batch_evidence.get("status") == "failed":
            raise ValueError("A failed first batch gate cannot resume training")
        if checkpoint_pending:
            collector.request_boundary()

    def emit(row):
        if log is not None:
            log(_primitive(row))

    def signature():
        return [updater.batches, updater.actor_steps, updater.critic_steps,
                collector.summary()["controls"], bool(collector.is_episode_boundary)]

    def loop_state():
        return _primitive({"schema_version": 1, "batches": updater.batches,
            "actor_steps": updater.actor_steps, "critic_steps": updater.critic_steps,
            "controls": collector.summary()["controls"], "diagnostics_by_shape": per_shape,
            "diagnostic_scope": diagnostic_scope, "empty_streak": empty_streak,
            "checkpoint_callbacks": checkpoint_callbacks, "replay_done": replay_done,
            "checkpoint_pending": checkpoint_pending, "next_checkpoint_batch": next_checkpoint_batch,
            "last_checkpoint_signature": last_checkpoint_signature, "first_batch_gate": first_batch_evidence})

    def checkpoint(terminal):
        nonlocal checkpoint_callbacks, last_checkpoint, last_checkpoint_signature, next_checkpoint_batch
        if on_checkpoint is None or signature() == last_checkpoint_signature:
            return
        state = loop_state()
        # A safe checkpoint starts a future process after the completed gate.
        # The live collector stays paused until this callback returns.
        if collector.is_episode_boundary:
            state["checkpoint_pending"] = False
            state["next_checkpoint_batch"] = (updater.batches // checkpoint_every + 1) * checkpoint_every
        state["checkpoint_callbacks"] = checkpoint_callbacks + 1
        state["last_checkpoint_signature"] = signature()
        gate_failed = isinstance(first_batch_evidence, dict) and first_batch_evidence.get("status") == "failed"
        result = on_checkpoint(resumable=bool(collector.is_episode_boundary) and not gate_failed,
                               terminal=terminal, loop_state=state)
        last_checkpoint = _primitive(result)
        last_checkpoint_signature = signature()
        checkpoint_callbacks += 1
        next_checkpoint_batch = state["next_checkpoint_batch"]

    stop_reason = "not_started"
    checkpoint_error = None
    while True:
        if checkpoint_pending and collector.is_episode_boundary:
            try:
                checkpoint(terminal=False)
            except BudgetExceeded as exc:
                stop_reason, checkpoint_error = "checkpoint_budget_exhausted", str(exc)
                break
            collector.release_boundary()
            checkpoint_pending = False
            next_checkpoint_batch = (updater.batches // checkpoint_every + 1) * checkpoint_every
        if collector.summary()["controls"] >= control_target:
            stop_reason = "control_target"
            break
        if updater.batches >= max_batches or not budget.remaining("batch", phase="train"):
            stop_reason = "batch_budget_exhausted"
            break
        if not budget.remaining("control", phase="train") or not budget.remaining("plan", phase="train"):
            stop_reason = "collection_budget_exhausted"
            break
        requested = first_batch_passes if updater.batches == 0 else passes
        actual_passes = min(requested, budget.remaining("actor_step", phase="train"),
                            budget.remaining("critic_step", phase="train"))
        if not actual_passes:
            stop_reason = "optimizer_budget_exhausted"
            break
        try:
            budget.check_time()
        except BudgetExceeded as exc:
            stop_reason = "time_budget_exhausted"
            emit({"event": "budget_stop", "reason": str(exc)})
            break
        if actual_passes < requested:
            emit({"event": "pass_budget_reduction", "requested_passes": requested, "actual_passes": actual_passes,
                  "next_batch": updater.batches + 1})
        fragment_limit = min(batch_controls, control_target - collector.summary()["controls"])
        batch_started = time.perf_counter()
        fragments = collector.collect_batch(batch_plans, fragment_limit, updater.actor_version)
        collection_seconds = time.perf_counter() - batch_started
        if not fragments:
            stop_reason = collector.summary()["last_batch"]["stop_reason"]
            del fragments
            break
        last_collection = _batch_diagnostics(fragments)
        _accumulate(per_shape, last_collection)
        empty_streak = empty_streak + 1 if last_collection["empty_for_optional_stop"] else 0
        update_started = time.perf_counter()
        buffer = None
        interrupted = False
        try:
            if not replay_done:
                if on_replay is not None:
                    replay_metadata = _primitive(on_replay(fragments=fragments, loop_state=loop_state()))
                replay_done = True
            buffer = updater.prepare(fragments)
            fragments = None
            raw_update = updater.update(buffer, passes=actual_passes)
            last_update = (_diagnostic_primitive(raw_update) if first_batch_gate is not None and updater.batches == 1
                           else _primitive(raw_update))
        except BudgetExceeded as exc:
            stop_reason, interrupted = "update_budget_or_time_exhausted", True
            last_update = {"interrupted": str(exc), "batches": updater.batches,
                           "actor_steps": updater.actor_steps, "critic_steps": updater.critic_steps}
        finally:
            # prepare() owns fixed copies. Drop the original diagnostic RGB and
            # all prepared chains/targets before checkpointing or collecting again.
            fragments = None
            buffer = None
        for entry in last_update.get("passes", []):
            emit({"event": "ppo_pass", "batch": updater.batches, **entry})
        row = {"event": "batch", "batch": updater.batches, "collection": last_collection,
              "requested_passes": requested, "budget_allowed_passes": actual_passes,
              "update": last_update, "update_seconds": time.perf_counter() - update_started,
              "collection_seconds": collection_seconds, "batch_seconds": time.perf_counter() - batch_started,
              "resources": _resources(), "counts": _counts(budget)}
        emit(row)
        if interrupted:
            break
        if first_batch_gate is not None and updater.batches == 1:
            try:
                evidence = first_batch_gate(deepcopy(row))
                if evidence is False or (isinstance(evidence, dict) and evidence.get("status") == "failed"):
                    raise ValueError("First batch gate callback rejected the completed update")
                first_batch_evidence = {"status": "passed", "batch": 1, "evidence": _primitive(evidence)}
            except Exception as exc:
                first_batch_evidence = {"status": "failed", "batch": 1,
                    "error": {"type": type(exc).__name__, "message": str(exc)}}
                stop_reason = "first_batch_gate_failed"
            emit({"event": "first_batch_gate", **first_batch_evidence})
            if stop_reason == "first_batch_gate_failed":
                break
        if not checkpoint_pending and updater.batches >= next_checkpoint_batch:
            collector.request_boundary()
            checkpoint_pending = True
        if stop_after_empty_batches and empty_streak >= stop_after_empty_batches:
            stop_reason = "optional_empty_batch_streak"
            break
    if checkpoint_error is None and (updater.batches or collector.summary()["controls"]):
        try:
            checkpoint(terminal=True)
        except BudgetExceeded as exc:
            checkpoint_error = str(exc)
            emit({"event": "terminal_checkpoint_unavailable", "reason": checkpoint_error})
        except Exception as exc:
            if stop_reason != "first_batch_gate_failed":
                raise
            checkpoint_error = f"{type(exc).__name__}: {exc}"
            emit({"event": "terminal_checkpoint_unavailable", "reason": checkpoint_error})
    result = {"stop_reason": stop_reason, "batches": updater.batches,
              "actor_steps": updater.actor_steps, "critic_steps": updater.critic_steps,
              "controls": collector.summary()["controls"], "collector": collector.summary(),
              "diagnostics_by_shape": per_shape, "diagnostic_scope": diagnostic_scope,
              "empty_streak": empty_streak, "checkpoint_callbacks": checkpoint_callbacks,
              "last_checkpoint": last_checkpoint, "checkpoint_error": checkpoint_error,
              "terminal_checkpoint_reused": signature() == last_checkpoint_signature,
              "last_collection": last_collection, "last_update": last_update, "replay": replay_metadata,
              "first_batch_gate": first_batch_evidence,
              "loop_state": loop_state(), "elapsed_seconds": time.perf_counter() - started,
              "resources": _resources(), "counts": _counts(budget)}
    return _primitive(result)


def _config(config):
    return config if isinstance(config, CConfig) else CConfig.from_dict(config)


def _experiment_id(budget):
    value = budget.snapshot().get("experiment_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Checkpointing requires a permanent experiment identity")
    return value


def _validate_runtime_configuration(official, actor, critic, actor_opt, critic_opt,
                                    collector, updater, budget, config):
    """A hash of proposed settings is insufficient unless live owners match it."""
    data, options = config.to_dict(), config.to_dict()["optimization"]
    if official.checkpoint_sha256 != data["base"]["sha256"]:
        raise ValueError("C runtime official base identity differs")
    if (collector.actor is not actor or collector.critic is not critic
            or updater.actor is not actor or updater.critic is not critic
            or updater.actor_optimizer is not actor_opt or updater.critic_optimizer is not critic_opt
            or updater.run_id != _experiment_id(budget)
            or getattr(official, "network", None) is not actor.network):
        raise ValueError("C runtime actors/optimizers/official owner identities differ")
    if actor.selection.subset != options["subset"] or actor.config != SamplerConfig(std_floor=options["std_floor"]):
        raise ValueError("C actor subset or Gaussian sampler configuration/version differs")
    named = dict(actor.network.named_parameters())
    expected = {name: parameter for name, parameter in named.items()
                if name.startswith(PARAMETER_PREFIXES[options["subset"]])}
    if (not expected or set(actor.selection.names) != set(expected)
            or tuple(named[name].numel() for name in actor.selection.names) != actor.selection.counts
            or {name for name, parameter in named.items() if parameter.requires_grad} != set(expected)):
        raise ValueError("C actor trainable whitelist or parameter counts differ")
    ppo, update = updater.config, updater.config.update
    desired = {"max_passes": options["max_passes"], "target_kl": options["target_kl"],
               "cache_features": True, "gamma": options["gamma"], "gae_lambda": options["gae_lambda"]}
    if any(getattr(ppo, name, None) != value for name, value in desired.items()):
        raise ValueError("Live PPO passes/KL/feature-cache/discount configuration differs from C")
    desired_update = {"clip": options["clip"], "max_grad_norm": options["max_grad_norm"],
        "plan_microbatch": options["plan_microbatch"], "step_microbatch": options["step_microbatch"],
        "denoising_discount": options["denoising_discount"], "initial_logprob_tolerance": 1e-4}
    if any(getattr(update, name, None) != value for name, value in desired_update.items()):
        raise ValueError("Live PPO clipping/gradient/microbatch configuration differs from C")
    for optimizer, expected_parameters, learning_rate in (
            (actor_opt, list(expected.values()), options["actor_lr"]),
            (critic_opt, list(critic.parameters()), options["critic_lr"])):
        if type(optimizer) is not torch.optim.Adam:
            raise ValueError("C requires separate Adam optimizers")
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if (len(parameters) != len({id(parameter) for parameter in parameters})
                or {id(parameter) for parameter in parameters} != {id(parameter) for parameter in expected_parameters}
                or any(group.get("lr") != learning_rate or group.get("weight_decay", 0) != 0
                       for group in optimizer.param_groups)):
            raise ValueError("C optimizer parameter ownership, learning rate or weight decay differs")
    runtime_device = data["runtime"]["device"]
    if any(parameter.device.type != runtime_device or parameter.dtype != torch.float32
           for parameter in (*named.values(), *critic.parameters())):
        raise ValueError("Live actor/critic device or FP32 precision differs from C")


def save_c_checkpoint(path, official, actor, critic, actor_opt, critic_opt, collector,
                      updater, budget, config, *, terminal=False, metadata=None):
    """Save C's selected weights/optimizers/RNG and checked episode-boundary state.

    Reserve 64 MiB before writing. Failed writes retain their reservations.
    Non-resumable terminal weights remain useful for separate evaluation but
    never receive a resume marker. Only registered checkpoint paths may retire.
    """
    config = _config(config)
    data = config.to_dict()
    path = Path(path).resolve()
    if not path.match("checkpoint-*.pt"):
        raise ValueError("C checkpoints need an owned checkpoint-*.pt path")
    if official.checkpoint_sha256 != data["base"]["sha256"]:
        raise ValueError("Loaded official base differs from the C protocol's pinned base")
    _validate_runtime_configuration(official, actor, critic, actor_opt, critic_opt,
                                    collector, updater, budget, config)
    updater_state = updater.state_dict()
    collector_state = collector.state_dict(require_boundary=False)
    if (collector_state["episode_limit"] != data["training"]["episode_controls"]
            or collector_state["gamma"] != data["optimization"]["gamma"]
            or collector_state["beta"] != data["optimization"]["beta"]
            or collector_state["scene_cursor"]["scenes"] != data["scenes"]["training"]
            or collector_state["scene_cursor"]["excluded"] != sorted(data["scenes"]["excluded_evaluation"])):
        raise ValueError("Collector scene/reward/episode state differs from the C configuration")
    resumable = bool(collector_state["resumable"] and
                     updater_state["actor_steps"] == updater_state["critic_steps"])
    counts = _counts(budget)
    try:
        _validate_training_state(collector_state, updater_state, counts)
    except ValueError:
        resumable = False
    extra = _primitive(metadata or {})
    gate_evidence = extra.get("loop_state", {}).get("first_batch_gate")
    if isinstance(gate_evidence, dict) and gate_evidence.get("status") == "failed":
        resumable = False
    reserved_keys = {"kind", "schema_version", "experiment_id", "config", "config_hash", "resumable",
                     "terminal", "collector", "updater", "counts"}
    if set(extra) & reserved_keys:
        raise ValueError("Additional checkpoint metadata cannot replace C provenance/state")
    details = {"kind": "tdyssey-C", "schema_version": 1, "experiment_id": _experiment_id(budget),
               "config": data, "config_hash": config.config_hash, "resumable": resumable,
               "terminal": bool(terminal), "collector": collector_state, "updater": updater_state,
               "counts": counts, **extra}
    reservation = budget.reserve_storage(_CHECKPOINT_RESERVATION_BYTES, path=path)
    saved = save_update_checkpoint(path, actor, critic, actor_opt, critic_opt,
                                   base_identity=data["base"], generator=collector.generators["T"],
                                   metadata=details)
    written = budget.mark_storage_written(reservation, saved["bytes"])
    marker = budget.record_boundary(path, {"config_hash": config.config_hash, "counts": details["counts"]}) if resumable else None
    # Retention belongs to the entire permanent experiment, including other
    # associated output directories and non-resumable audit deltas.
    retired = sorted(set((written or {}).get("retire_paths", [])) |
                     set((marker or {}).get("retire_paths", [])))
    if retired:
        budget.retire_paths(retired)
    budget.check_storage()
    return {**saved, "resumable": resumable, "terminal": bool(terminal),
            "retired": [str(Path(item).resolve()) for item in retired], "boundary_recorded": marker is not None}


def restore_c_checkpoint(loaded, collector, updater, budget, config, checkpoint_path):
    """Restore C counters/scenes/RNG only at the exact permanent-ledger boundary.

    The caller first reconstructs the verified actor/critic and restores complete
    optimizer groups through B's strict helpers. No physical state is restored,
    and an old boundary cannot rewind budgeted operations performed after it.
    """
    config = _config(config)
    metadata = loaded.payload.get("metadata")
    if (not isinstance(metadata, dict) or metadata.get("kind") != "tdyssey-C"
            or metadata.get("schema_version") != 1 or metadata.get("resumable") is not True):
        raise ValueError("Only resumable C boundary checkpoints can continue training; B/audit deltas cannot")
    if (metadata.get("experiment_id") != _experiment_id(budget)
            or metadata.get("config_hash") != config.config_hash or metadata.get("config") != config.to_dict()):
        raise ValueError("C checkpoint experiment identity/configuration differs")
    if any(getattr(owner, "actor", None) is not loaded.actor or getattr(owner, "critic", None) is not loaded.critic
           for owner in (collector, updater)):
        raise ValueError("Restored collector/updater must use the reconstructed checkpoint actor/critic")
    _validate_runtime_configuration(collector.official, loaded.actor, loaded.critic,
                                    updater.actor_optimizer, updater.critic_optimizer,
                                    collector, updater, budget, config)
    budget.validate_boundary(checkpoint_path)
    if _training_counts(metadata["counts"]) != _training_counts(_counts(budget)):
        raise ValueError("Permanent operation counters advanced beyond this C checkpoint")
    _validate_training_state(metadata["collector"], metadata["updater"], _counts(budget))
    updater.load_state_dict(metadata["updater"])
    collector.load_state_dict(metadata["collector"])
    # Model/environment construction may have consumed globals after the caller
    # loaded the tensor payload. Restore only after every identity/counter check
    # and local T/L restoration has passed, immediately before future collection.
    _restore_rng(loaded.payload["rng"])
    return {key: deepcopy(value) for key, value in metadata.items() if key not in ("collector", "updater")}
