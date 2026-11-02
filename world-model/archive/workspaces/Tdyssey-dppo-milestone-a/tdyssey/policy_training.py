"""One bounded on-policy actor/critic update, using milestone A likelihoods.

This is an implementation smoke check, not a learning experiment. Physical
rollouts live in policy_rollout; this module never calls an environment.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import math
import time

import torch
from torch import nn

from .policy_returns import plan_gae
from .policy_rl import transition_ppo_loss


class VisualCritic(nn.Module):
    """Small value head on the actor's detached, frozen visual/pusher features."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        if type(input_dim) is not int or input_dim < 1 or type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("Critic dimensions must be positive integers")
        self.config = {"input_dim": input_dim, "hidden_dim": hidden_dim}
        self.head = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.config["input_dim"]:
            raise ValueError("Critic expects [B, frozen visual feature dimension]")
        return self.head(features.detach()).squeeze(-1)


@dataclass(frozen=True)
class PreparedBatch:
    plans: tuple
    features: torch.Tensor
    old_values: torch.Tensor
    raw_advantages: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def prepare_batch(fragments, *, gamma=0.99, gae_lambda=0.95):
    """Freeze GAE/value targets; normalize advantages once over the whole batch.

    A fragment is one ordered stream with an explicit terminal/truncation tail.
    Different T/L fragments are estimated separately, then combined. The critic
    targets use raw GAE plus old value, never normalized advantages.
    """
    plans, estimates = [], []
    for fragment in fragments:
        fragment = tuple(fragment)
        estimates.extend(plan_gae([p.block for p in fragment], gamma=gamma, gae_lambda=gae_lambda))
        plans.extend(fragment)
    if not plans:
        raise ValueError("At least one nonempty on-policy fragment is required")
    versions = {p.sample.actor_version for p in plans}
    if len(versions) != 1:
        raise ValueError("Mixed actor versions are not one on-policy batch")
    for plan in plans:
        if plan.sample.mode != "training" or plan.sample.chain.shape != (1, 101, 16, 2):
            raise ValueError("Each plan must retain one complete training chain")
        if plan.sample.old_log_prob is None or plan.sample.old_log_prob.shape != (1, 100):
            raise ValueError("Missing old transition log probabilities")
        if plan.features.ndim != 2 or plan.features.shape[0] != 1:
            raise ValueError("Each plan needs one frozen conditioning feature vector")
    features = torch.cat([p.features.detach().cpu().clone() for p in plans], dim=0)
    old_values = torch.tensor([p.block.value for p in plans], dtype=torch.float32)
    raw = torch.tensor([e.advantage for e in estimates], dtype=torch.float32)
    targets = torch.tensor([e.value_target for e in estimates], dtype=torch.float32)
    if any(not torch.isfinite(v).all() for v in (features, old_values, raw, targets)):
        raise ValueError("Features, values and GAE targets must be finite")
    advantages = (raw - raw.mean()) / raw.std(unbiased=False).clamp_min(1e-8)
    return PreparedBatch(tuple(plans), features, old_values, raw, advantages, targets)


@dataclass(frozen=True)
class UpdateConfig:
    clip: float = 0.2
    max_grad_norm: float = 1.0
    initial_logprob_tolerance: float = 1e-4
    plan_microbatch: int = 1
    step_microbatch: int = 1
    denoising_discount: float = 1.0

    def __post_init__(self):
        if (not math.isfinite(self.clip) or not 0 < self.clip < 1
                or not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
                or not math.isfinite(self.initial_logprob_tolerance) or self.initial_logprob_tolerance < 0
                or not 0 < self.denoising_discount <= 1):
            raise ValueError("Invalid bounded PPO settings")
        if (type(self.plan_microbatch) is not int or not 1 <= self.plan_microbatch <= 4
                or type(self.step_microbatch) is not int or not 1 <= self.step_microbatch <= 4):
            raise ValueError("Microbatches must be integers in [1,4]")


def make_optimizers(actor, critic, *, actor_lr=1e-5, critic_lr=3e-4):
    if any(not math.isfinite(lr) or lr <= 0 for lr in (actor_lr, critic_lr)):
        raise ValueError("Learning rates must be finite and positive")
    actor_parameters = [p for p in actor.network.parameters() if p.requires_grad]
    return (torch.optim.Adam(actor_parameters, lr=actor_lr, weight_decay=0),
            torch.optim.Adam(critic.parameters(), lr=critic_lr, weight_decay=0))


@contextmanager
def measured_phase(report, name, device):
    """Synchronized wall time and phase-local allocated/reserved CUDA peaks."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {"seconds": time.perf_counter() - start}
        if device.type == "cuda":
            result.update(allocated_bytes=torch.cuda.memory_allocated(device),
                          reserved_bytes=torch.cuda.memory_reserved(device),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        report[name] = result


def state_fingerprint(module, *, exclude=()):
    """Hash exact parameters/buffers one tensor at a time, avoiding an actor copy."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        if name in exclude:
            continue
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode("utf-8"))
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _pack_samples(plans, device):
    first = plans[0].sample
    if any(p.sample.sampler_config != first.sampler_config
           or p.sample.scheduler_signature != first.scheduler_signature for p in plans):
        raise ValueError("Mixed sampling configurations")
    names = ("chain", "old_means", "old_log_prob", "targets", "actions")
    combined = {name: torch.cat([getattr(p.sample, name).detach() for p in plans], dim=0).to(device)
                for name in names}
    observations = {key: torch.cat([p.sample.observations[key].detach() for p in plans], dim=0).to(device)
                    for key in first.observations}
    return replace(first, observations=observations, timesteps=first.timesteps.to(device),
                   old_stds=first.old_stds.to(device), **combined)


def _microbatches(batch, config, device):
    for start in range(0, len(batch.plans), config.plan_microbatch):
        rows = slice(start, min(start + config.plan_microbatch, len(batch.plans)))
        record = _pack_samples(batch.plans[rows], device)
        for step in range(0, 100, config.step_microbatch):
            yield rows, record, list(range(step, min(step + config.step_microbatch, 100)))


@torch.no_grad()
def probability_metrics(actor, batch, config=UpdateConfig()):
    differences = []
    device = actor.network.device
    for _, record, steps in _microbatches(batch, config, device):
        new = actor.recompute_log_probs(record, step_indices=steps)
        differences.append((new - record.old_log_prob[:, steps]).flatten().cpu())
    log_ratio = torch.cat(differences).double()
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("Nonfinite diagnostic ratio")
    return {"max_logprob_error": float(log_ratio.abs().max()),
            "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
            "ratio_mean": float(ratio.mean()),
            "sample_approx_kl": float((log_ratio.expm1() - log_ratio).mean()),
            "clip_fraction": float(((ratio - 1).abs() > config.clip).double().mean()),
            "transition_count": log_ratio.numel()}


def _optimizer_parameters(optimizer):
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("An optimizer parameter occurs more than once")
    if any(group.get("weight_decay", 0) != 0 for group in optimizer.param_groups):
        raise ValueError("This bounded check requires zero weight decay")
    return {id(p) for p in parameters}


def _grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError("Every selected parameter needs a finite gradient")
        total += float(p.grad.detach().double().square().sum())
    return math.sqrt(total)


def _weight_change(before, named_parameters):
    changed, squared, maximum = [], 0.0, 0.0
    for name, previous in before.items():
        current = named_parameters[name].detach().cpu()
        if not torch.isfinite(current).all():
            raise ValueError("Updated weights must be finite")
        delta = (current - previous).double()
        if (delta != 0).any():
            changed.append(name)
        squared += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {"changed_names": changed, "changed_tensor_count": len(changed),
            "l2": math.sqrt(squared), "max_abs": maximum}


def update_once(actor, critic, batch, actor_optimizer, critic_optimizer, *, budget, config=UpdateConfig()):
    """Accumulate all B*100 transition gradients, then step each optimizer once.

    Both durable optimizer reservations must succeed before either step. Any
    failure consumes its reservations; retry must not create a fresh budget.
    Zero advantages are allowed (fresh Adam, no decay: actor remains unchanged).
    """
    actor.train()
    critic.train()
    selected = {n: p for n, p in actor.network.named_parameters() if p.requires_grad}
    if set(selected) != set(actor.selection.names):
        raise ValueError("Trainable actor parameters differ from the declared whitelist")
    if _optimizer_parameters(actor_optimizer) != {id(p) for p in selected.values()}:
        raise ValueError("Actor optimizer must contain exactly the whitelist")
    if _optimizer_parameters(critic_optimizer) != {id(p) for p in critic.parameters()}:
        raise ValueError("Critic optimizer must contain exactly the critic head")
    if any(id(p) in _optimizer_parameters(actor_optimizer) for p in critic.parameters()):
        raise ValueError("Actor and critic optimizers must not share weights")
    report = {"timings": {}, "plans": len(batch.plans), "denoising_transitions": len(batch.plans) * 100}
    device = actor.network.device
    with measured_phase(report["timings"], "before_update_checks", device):
        report["before"] = probability_metrics(actor, batch, config)
        if report["before"]["max_logprob_error"] > config.initial_logprob_tolerance:
            raise ValueError("Stale/nonmatching rollout: initial transition probabilities differ")
        with torch.no_grad():
            values = critic(batch.features.to(device)).cpu()
        if not torch.allclose(values, batch.old_values, rtol=0, atol=1e-5):
            raise ValueError("Critic changed since rollout; old values are fixed")
        actor_before = {n: p.detach().cpu().clone() for n, p in selected.items()}
        critic_before = {n: p.detach().cpu().clone() for n, p in critic.named_parameters()}
        report["frozen_digest_before"] = state_fingerprint(actor.network, exclude=selected)
    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    actor_loss, critic_loss = 0.0, 0.0
    with measured_phase(report["timings"], "backward", device):
        for rows, record, steps in _microbatches(batch, config, device):
            new = actor.recompute_log_probs(record, step_indices=steps)
            loss = transition_ppo_loss(new, record.old_log_prob[:, steps], batch.advantages[rows].to(device),
                                       clip=config.clip, denoising_discount=config.denoising_discount,
                                       timesteps=record.timesteps[steps], reduction="sum") / report["denoising_transitions"]
            actor_loss += float(loss.detach())
            loss.backward()
        for start in range(0, len(batch.plans), config.plan_microbatch):
            rows = slice(start, start + config.plan_microbatch)
            prediction = critic(batch.features[rows].to(device))
            loss = (prediction - batch.returns[rows].to(device)).square().sum() / len(batch.plans)
            critic_loss += float(loss.detach())
            loss.backward()
        report["actor_grad_norm"] = _grad_norm(selected.values())
        report["critic_grad_norm"] = _grad_norm(critic.parameters())
        if any(p.grad is not None for n, p in actor.network.named_parameters() if n not in selected):
            raise ValueError("A frozen actor parameter acquired a gradient")
        torch.nn.utils.clip_grad_norm_(list(selected.values()), config.max_grad_norm, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm, error_if_nonfinite=True)
    actor_reservation = budget.reserve("actor_step")
    critic_reservation = budget.reserve("critic_step")
    with measured_phase(report["timings"], "first_optimizer_steps", device):
        actor_optimizer.step()
        budget.mark_completed(actor_reservation)
        critic_optimizer.step()
        budget.mark_completed(critic_reservation)
    with measured_phase(report["timings"], "after_update_checks", device):
        report["actor_change"] = _weight_change(actor_before, dict(actor.network.named_parameters()))
        report["critic_change"] = _weight_change(critic_before, dict(critic.named_parameters()))
        report["frozen_digest_after"] = state_fingerprint(actor.network, exclude=selected)
        if report["frozen_digest_before"] != report["frozen_digest_after"]:
            raise ValueError("Frozen actor weights or normalizer changed")
        report["after"] = probability_metrics(actor, batch, config)
    report.update(actor_loss=actor_loss, critic_loss=critic_loss, actor_optimizer_steps=1, critic_optimizer_steps=1)
    return report
