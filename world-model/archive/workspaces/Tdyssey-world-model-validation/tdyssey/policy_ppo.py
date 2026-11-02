"""C's multi-pass PPO protocol; B's strict single-update API stays separate.

Only fresh buffers enter once. Their behavior probabilities and value targets
remain fixed across passes; no endpoint resampling or supervised actor loss.
"""

from dataclasses import dataclass, replace
import hashlib
import math
import uuid

import torch

from .policy_checkpoint import sample_from_payload, sample_to_payload
from .policy_rl import SampledChain, _canonical_scheduler_signature, transition_ppo_loss
from .policy_training import (UpdateConfig, _grad_norm, _optimizer_parameters,
                              _pack_samples, _weight_change, prepare_batch, state_fingerprint)


@dataclass(frozen=True)
class MultiPassConfig:
    max_passes: int = 3
    target_kl: float = 0.02
    cache_features: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update: UpdateConfig = UpdateConfig()

    def __post_init__(self):
        if type(self.max_passes) is not int or not 1 <= self.max_passes <= 3:
            raise ValueError("C permits one to three PPO passes per buffer")
        if not math.isfinite(self.target_kl) or self.target_kl <= 0:
            raise ValueError("An explicit finite positive KL threshold is required")
        if type(self.cache_features) is not bool:
            raise ValueError("cache_features must be a bool")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid physical discount or plan-boundary GAE lambda")


@dataclass(eq=False)
class PPOBuffer:
    prepared: object
    owner: str
    sequence: int
    actor_version: str
    token: str
    fixed_digest: str
    entry_metrics: dict
    consumed: bool = False


def _fixed_digest(batch):
    digest = hashlib.sha256()
    tensors = [batch.features, batch.old_values, batch.raw_advantages, batch.advantages, batch.returns]
    for plan in batch.plans:
        sample = plan.sample
        tensors.extend((sample.chain, sample.old_log_prob, sample.old_means, sample.old_stds, sample.timesteps,
                        sample.targets, sample.actions, *sample.observations.values()))
    for tensor in tensors:
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _owned_fragments(fragments):
    # Retain exact conditioning/chain and numerical targets, not another copy of
    # every control's float RGB diagnostics. The driver summarizes those first.
    return [[replace(plan, sample=sample_from_payload(sample_to_payload(plan.sample)),
                     features=plan.features.detach().cpu().clone(), control_records=[],
                     final_observation={k: v.detach().cpu().clone() for k, v in plan.final_observation.items()})
             for plan in fragment] for fragment in fragments]


def _validate_behavior_records(actor, batch):
    """Apply the same record contract before either cached/observation path.

    A matching numerical density cannot substitute for the declared sampler
    identity. In particular, the cached-feature path never silently accepts a
    different std floor, scheduler, mode or denoising-time sequence.
    """
    for plan in batch.plans:
        sample = plan.sample
        if (not isinstance(sample, SampledChain) or sample.mode != "training"
                or sample.sampler_config != actor.config
                or _canonical_scheduler_signature(sample.scheduler_signature) != actor.scheduler_signature):
            raise ValueError("Behavior record does not match this training sampler configuration")
        shapes = {"chain": (1, 101, 16, 2), "old_log_prob": (1, 100),
                  "old_means": (1, 100, 16, 2), "old_stds": (100,),
                  "timesteps": (100,), "targets": (1, 16, 2), "actions": (1, 8, 2)}
        for name, shape in shapes.items():
            value = getattr(sample, name)
            if not isinstance(value, torch.Tensor) or value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid behavior record {name}")
        if (sample.timesteps.tolist() != list(range(99, -1, -1)) or not (sample.old_stds > 0).all()
                or set(sample.observations) != {"image", "agent_pos"}):
            raise ValueError("Behavior record must preserve all 100 official training transitions")


def _entries(actor, batch, config):
    size = config.update.plan_microbatch
    for start in range(0, len(batch.plans), size):
        rows = slice(start, min(start + size, len(batch.plans)))
        record = _pack_samples(batch.plans[rows], actor.network.device)
        features = batch.features[rows].to(actor.network.device)
        for first in range(0, 100, config.update.step_microbatch):
            steps = list(range(first, min(first + config.update.step_microbatch, 100)))
            yield rows, record, features, steps


def _log_probs(actor, record, features, steps, cached):
    if not cached:
        return actor.recompute_log_probs(record, step_indices=steps)
    return torch.stack([actor.transition_log_prob_from_features(features, record.chain[:, i],
                         record.chain[:, i + 1], int(record.timesteps[i])) for i in steps], dim=1)


@torch.no_grad()
def _metrics(actor, batch, config):
    differences = []
    for _, record, features, steps in _entries(actor, batch, config):
        new = _log_probs(actor, record, features, steps, config.cache_features)
        differences.append((new - record.old_log_prob[:, steps]).flatten().cpu().double())
    log_ratio = torch.cat(differences)
    if not torch.isfinite(log_ratio).all():
        raise ValueError("Nonfinite PPO log probabilities")
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("Nonfinite PPO probability diagnostics")
    return {"max_logprob_error": float(log_ratio.abs().max()), "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()), "sample_approx_kl": float((log_ratio.expm1() - log_ratio).mean()),
            "clip_fraction": float(((ratio - 1).abs() > config.update.clip).double().mean()),
            "transition_count": log_ratio.numel()}


def explained_variance(targets, predictions):
    variance = float(targets.double().var(unbiased=False))
    if variance == 0:
        return None  # Undefined when targets have no variance.
    return 1 - float((targets.double() - predictions.double()).var(unbiased=False)) / variance


def _statistics(values):
    values = values.detach().cpu().double().reshape(-1)
    return {"count": values.numel(), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std(unbiased=False))}


class PPOUpdater:
    """Own one current buffer and a monotonic actor/version/batch watermark.

    No strong reference to a buffer survives update(); only constant-size
    counters and identifiers remain. Restoring mid-pass is intentionally refused
    by the experiment's checkpoint-boundary protocol.
    """

    def __init__(self, actor, critic, actor_optimizer, critic_optimizer, *, budget, run_id,
                 config=MultiPassConfig()):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("An explicit experiment identity is required")
        self.actor, self.critic = actor, critic
        self.actor_optimizer, self.critic_optimizer = actor_optimizer, critic_optimizer
        self.budget, self.run_id, self.config = budget, run_id, config
        self.actor_steps = self.critic_steps = self.batches = 0
        self.entry_checks = 0
        self._active = None
        self._last_consumed = -1
        self._lifecycle_started = self._state_restored = False
        self._frozen_digest = state_fingerprint(actor.network, exclude=actor.selection.names)
        self._validate_optimizers()

    @property
    def actor_version(self):
        return f"{self.run_id}:actor:{self.actor_steps}"

    def _validate_optimizers(self):
        selected = {n: p for n, p in self.actor.network.named_parameters() if p.requires_grad}
        if set(selected) != set(self.actor.selection.names):
            raise ValueError("Actor trainable weights differ from the whitelist")
        actor_ids = _optimizer_parameters(self.actor_optimizer)
        critic_ids = _optimizer_parameters(self.critic_optimizer)
        if actor_ids != {id(p) for p in selected.values()} or critic_ids != {id(p) for p in self.critic.parameters()}:
            raise ValueError("Optimizer parameters differ from the declared actor/critic")
        if actor_ids & critic_ids:
            raise ValueError("Actor and critic optimizers must not share parameters")

    def prepare(self, fragments):
        if self._active is not None:
            raise ValueError("Finish or abort the current buffer before collecting another")
        self._lifecycle_started = True
        owned = _owned_fragments(fragments)
        if any(p.sample.actor_version != self.actor_version for f in owned for p in f):
            raise ValueError("Fresh buffer actor version does not match this updater")
        reservation = self.budget.reserve("batch", phase="train")
        prepared = prepare_batch(owned, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        _validate_behavior_records(self.actor, prepared)
        self.actor.train()
        self.critic.train()
        if state_fingerprint(self.actor.network, exclude=self.actor.selection.names) != self._frozen_digest:
            raise ValueError("Frozen actor/normalizer changed before batch entry")
        with torch.no_grad():
            # Cache validation uses the original one-plan encoder shape, avoiding
            # a new batching-dependent preprocessing convention.
            for plan, cached in zip(prepared.plans, prepared.features):
                obs = {k: v.to(self.actor.network.device) for k, v in plan.sample.observations.items()}
                actual = self.actor.encode(obs).cpu()
                if not torch.allclose(actual, cached.unsqueeze(0), rtol=0, atol=1e-5):
                    raise ValueError("Cached features differ from frozen visual encoding")
            values = self.critic(prepared.features.to(self.actor.network.device)).cpu()
            if not torch.allclose(values, prepared.old_values, rtol=0, atol=1e-5):
                raise ValueError("Old critic values differ at fresh batch entry")
        metrics = _metrics(self.actor, prepared, self.config)
        if metrics["max_logprob_error"] > self.config.update.initial_logprob_tolerance:
            raise ValueError("Initial behavior probabilities differ; stale buffer")
        sequence = self.batches
        token = uuid.uuid4().hex
        buffer = PPOBuffer(prepared, self.run_id, sequence, self.actor_version, token,
                           _fixed_digest(prepared), metrics)
        self.budget.mark_completed(reservation)
        self.entry_checks += 1
        self._active = token
        self.batches += 1
        return buffer

    def update(self, buffer, *, passes=None):
        passes = self.config.max_passes if passes is None else passes
        if type(passes) is not int or not 1 <= passes <= self.config.max_passes:
            raise ValueError("Requested PPO passes exceed the declared protocol")
        if (not isinstance(buffer, PPOBuffer) or buffer.owner != self.run_id or buffer.consumed
                or buffer.token != self._active or buffer.sequence <= self._last_consumed
                or buffer.sequence != self.batches - 1
                or buffer.actor_version != self.actor_version):
            raise ValueError("Consumed, foreign or wrong-version PPO buffer")
        report = {"batch_sequence": buffer.sequence, "behavior_actor_version": buffer.actor_version,
                  "entry": buffer.entry_metrics, "requested_passes": passes, "passes": [], "stop_reason": "completed"}
        batch = buffer.prepared
        consumed_sequence = buffer.sequence
        try:
            for index in range(passes):
                self.budget.check_time()
                self._validate_optimizers()
                if state_fingerprint(self.actor.network, exclude=self.actor.selection.names) != self._frozen_digest:
                    raise ValueError("Frozen actor/normalizer changed before PPO pass")
                selected = [p for p in self.actor.network.parameters() if p.requires_grad]
                if _fixed_digest(batch) != buffer.fixed_digest:
                    raise ValueError("Fixed behavior records or PPO targets changed")
                if index == 0:
                    report["data_statistics"] = {name: _statistics(getattr(batch, name)) for name in
                                                 ("old_values", "raw_advantages", "advantages", "returns")}
                self.actor.zero_grad(set_to_none=True)
                self.critic.zero_grad(set_to_none=True)
                actor_loss = critic_loss = 0.0
                for rows, record, features, steps in _entries(self.actor, batch, self.config):
                    self.budget.check_time()
                    new = _log_probs(self.actor, record, features, steps, self.config.cache_features)
                    loss = transition_ppo_loss(new, record.old_log_prob[:, steps], batch.advantages[rows].to(new.device),
                        clip=self.config.update.clip, denoising_discount=self.config.update.denoising_discount,
                        timesteps=record.timesteps[steps], reduction="sum") / (len(batch.plans) * 100)
                    actor_loss += float(loss.detach())
                    loss.backward()
                for start in range(0, len(batch.plans), self.config.update.plan_microbatch):
                    rows = slice(start, start + self.config.update.plan_microbatch)
                    prediction = self.critic(batch.features[rows].to(self.actor.network.device))
                    loss = (prediction - batch.returns[rows].to(prediction.device)).square().sum() / len(batch.plans)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite critic objective")
                    critic_loss += float(loss.detach())
                    loss.backward()
                actor_norm, critic_norm = _grad_norm(selected), _grad_norm(self.critic.parameters())
                if any(p.grad is not None for n, p in self.actor.network.named_parameters() if n not in self.actor.selection.names):
                    raise ValueError("A frozen actor parameter acquired a gradient")
                torch.nn.utils.clip_grad_norm_(selected, self.config.update.max_grad_norm, error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.update.max_grad_norm, error_if_nonfinite=True)
                # CPU snapshots cover only the selected actor weights and tiny
                # critic, never another full actor or its frozen encoder.
                actor_before = {name: p.detach().cpu().clone() for name, p in self.actor.network.named_parameters()
                                if name in self.actor.selection.names}
                critic_before = {name: p.detach().cpu().clone() for name, p in self.critic.named_parameters()}
                actor_res = self.budget.reserve("actor_step", phase="train")
                critic_res = self.budget.reserve("critic_step", phase="train")
                self.actor_optimizer.step()
                self.actor_steps += 1
                self.budget.mark_completed(actor_res)
                self.critic_optimizer.step()
                self.critic_steps += 1
                self.budget.mark_completed(critic_res)
                if any(not torch.isfinite(p).all() for p in (*selected, *self.critic.parameters())):
                    raise ValueError("Nonfinite updated actor/critic parameters")
                if state_fingerprint(self.actor.network, exclude=self.actor.selection.names) != self._frozen_digest:
                    raise ValueError("Frozen actor/normalizer changed during PPO")
                after = _metrics(self.actor, batch, self.config)
                with torch.no_grad():
                    predictions = self.critic(batch.features.to(self.actor.network.device)).cpu()
                mse = (predictions - batch.returns).square().mean()
                if not torch.isfinite(predictions).all() or not torch.isfinite(mse):
                    raise ValueError("Nonfinite updated critic diagnostics")
                report["passes"].append({"index": index, "actor_loss": actor_loss, "critic_loss": critic_loss,
                    "actor_grad_norm": actor_norm, "critic_grad_norm": critic_norm, "after": after,
                    "actor_weight_change": _weight_change(actor_before, dict(self.actor.network.named_parameters())),
                    "critic_weight_change": _weight_change(critic_before, dict(self.critic.named_parameters())),
                    "critic_mse": float(mse),
                    "critic_explained_variance": explained_variance(batch.returns, predictions)})
                if after["sample_approx_kl"] > self.config.target_kl:
                    report["stop_reason"] = "target_kl"
                    break
            if _fixed_digest(batch) != buffer.fixed_digest:
                raise ValueError("Fixed behavior records or PPO targets changed")
            report.update(completed_passes=len(report["passes"]), actor_version=self.actor_version,
                          actor_steps=self.actor_steps, critic_steps=self.critic_steps,
                          frozen_unchanged=True, frozen_digest=self._frozen_digest)
            return report
        finally:
            buffer.consumed = True
            self._last_consumed = consumed_sequence
            self._active = None
            self.actor.zero_grad(set_to_none=True)
            self.critic.zero_grad(set_to_none=True)

    def state_dict(self):
        if self._active is not None:
            raise ValueError("An active PPO buffer cannot be restored mid-update")
        return {"schema_version": 1, "run_id": self.run_id, "actor_steps": self.actor_steps,
                "critic_steps": self.critic_steps, "batches": self.batches, "entry_checks": self.entry_checks,
                "last_consumed": self._last_consumed, "frozen_digest": self._frozen_digest}

    def load_state_dict(self, state):
        expected = {"schema_version", "run_id", "actor_steps", "critic_steps", "batches",
                    "entry_checks", "last_consumed", "frozen_digest"}
        if (self._active is not None or self._lifecycle_started or self._state_restored
                or any((self.actor_steps, self.critic_steps, self.batches, self.entry_checks))
                or self._last_consumed != -1 or not isinstance(state, dict) or set(state) != expected
                or type(state.get("schema_version")) is not int or state["schema_version"] != 1
                or state.get("run_id") != self.run_id or state.get("frozen_digest") != self._frozen_digest):
            raise ValueError("Updater state belongs to a different or active experiment")
        for name in ("actor_steps", "critic_steps", "batches", "entry_checks"):
            if type(state.get(name)) is not int or state[name] < 0:
                raise ValueError("Corrupt updater counters")
        batches, steps = state["batches"], state["actor_steps"]
        if (state["critic_steps"] != steps or not batches <= steps <= batches * self.config.max_passes
                or state["entry_checks"] != batches or type(state["last_consumed"]) is not int
                or state["last_consumed"] != batches - 1):
            raise ValueError("Inconsistent completed-batch updater counters")
        # Validate the whole boundary before mutating any version/watermark.
        for name in ("actor_steps", "critic_steps", "batches", "entry_checks"):
            setattr(self, name, state[name])
        self._last_consumed = state["last_consumed"]
        self._state_restored = True
