"""Milestone A: fixed-record Gaussian denoising likelihoods, without a trainer.

The baseline delegates sampling to the pinned scheduler. The training sampler
changes its noise rule (including t=0); this must be a separate baseline in any
learning experiment. Latents/noise are never clamped. Only predicted x0 uses the
official DDPM clipping. Pixel targets may subsequently be bounded by the physical
controller; that mapping is not part of this latent Gaussian density.
"""

from dataclasses import dataclass
import json
import math

import torch
from torch import nn

from .official import official_spec


PARAMETER_PREFIXES = {
    "output": ("model.final_conv.",),
    "last_up_and_output": ("model.up_modules.1.", "model.final_conv."),
}


@dataclass(frozen=True)
class SamplerConfig:
    """Positive normalized-action std floor; 0.01 is an untuned proposal."""

    std_floor: float = 0.01
    version: str = "tdyssey-ddpm-gaussian-v1"

    def __post_init__(self):
        if not math.isfinite(self.std_floor) or self.std_floor <= 0:
            raise ValueError("std_floor must be finite and strictly positive")
        if self.version != "tdyssey-ddpm-gaussian-v1":
            raise ValueError("Unknown sampler version")


@dataclass(frozen=True)
class ParameterSelection:
    subset: str
    names: tuple[str, ...]
    counts: tuple[int, ...]

    @property
    def total(self):
        return sum(self.counts)


def select_trainable_parameters(network, subset="output"):
    """Explicit name whitelist; normalizer/encoder and all other weights freeze.

    This selects weights, not denoising times: one network participates at all
    100 times. Names and counts are suitable for saving with a future checkpoint.
    """
    if subset not in PARAMETER_PREFIXES:
        raise ValueError(f"Unknown parameter subset: {subset}")
    network.eval().requires_grad_(False)
    selected = []
    for name, parameter in network.named_parameters():
        parameter.grad = None
        if name.startswith(PARAMETER_PREFIXES[subset]):
            parameter.requires_grad_(True)
            selected.append((name, parameter.numel()))
    if any(not any(n.startswith(prefix) for n, _ in selected)
           for prefix in PARAMETER_PREFIXES[subset]):
        raise ValueError("Network lacks a requested official parameter block")
    return ParameterSelection(subset, tuple(n for n, _ in selected), tuple(c for _, c in selected))


def _canonical_scheduler_signature(signature):
    """Canonicalize only diffusers' unordered default-name bookkeeping.

    Legacy A records retain all scheduler values. The private name list is built
    from a Python set, whose order varies by process/hash seed; it does not alter
    DDPM mathematics. Sorting it also permits unchanged legacy records to load.
    """
    config = json.loads(signature)
    if not isinstance(config, dict):
        raise ValueError("Scheduler signature must describe a configuration object")
    defaults = config.get("_use_default_values")
    if defaults is not None:
        if not isinstance(defaults, list) or any(not isinstance(name, str) for name in defaults):
            raise ValueError("Invalid scheduler default-name bookkeeping")
        config["_use_default_values"] = sorted(defaults)
    return json.dumps(config, sort_keys=True)


def _scheduler_signature(scheduler):
    expected = official_spec()["policy_config"]["noise_scheduler"]
    for key, value in expected.items():
        if key != "_target_" and getattr(scheduler.config, key) != value:
            raise ValueError(f"Training requires the locked official DDPM {key}")
    if (scheduler.config.thresholding or scheduler.config.clip_sample_range != 1.0
            or scheduler.config.steps_offset != 0 or scheduler.config.timestep_spacing != "leading"
            or scheduler.config.rescale_betas_zero_snr or scheduler.config.trained_betas is not None):
        raise ValueError("Unsupported change to the locked DDPM scheduler")
    return _canonical_scheduler_signature(json.dumps(dict(scheduler.config), sort_keys=True))


def ddpm_transition(scheduler, epsilon, timestep, x_before, *, std_floor=None):
    """Return official clipped-x0 posterior mean and scalar std on x's device.

    Operation order matches diffusers 0.31.0 fixed_small/epsilon. std_floor=None
    describes baseline noise: exactly zero at t=0 (no Gaussian density there).
    A positive floor defines the training Gaussian at *every* transition.
    """
    if type(timestep) is not int or not 0 <= timestep < 100:
        raise ValueError("timestep must be an integer in [0, 99]")
    if epsilon.shape != x_before.shape:
        raise ValueError("epsilon and x_before shapes differ")
    if std_floor is not None and (not math.isfinite(std_floor) or std_floor <= 0):
        raise ValueError("std_floor must be positive")
    _scheduler_signature(scheduler)
    if scheduler.num_inference_steps != 100:
        raise ValueError("Set all 100 official denoising timesteps first")
    # Scheduler coefficients are evaluated in their original FP32 on CPU, then
    # used as scalars by torch, exactly as in the pinned scheduler.step().
    previous = scheduler.previous_timestep(timestep)
    alpha = scheduler.alphas_cumprod[timestep]
    alpha_previous = scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.one
    beta, beta_previous = 1 - alpha, 1 - alpha_previous
    current_alpha = alpha / alpha_previous
    current_beta = 1 - current_alpha
    predicted_x0 = (x_before - beta ** 0.5 * epsilon) / alpha ** 0.5
    predicted_x0 = predicted_x0.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
    mean = ((alpha_previous ** 0.5 * current_beta) / beta) * predicted_x0
    mean = mean + (current_alpha ** 0.5 * beta_previous / beta) * x_before
    variance = torch.clamp((1 - alpha_previous) / (1 - alpha) * current_beta, min=1e-20)
    std = variance ** 0.5 if timestep > 0 or std_floor is not None else torch.zeros_like(variance)
    if std_floor is not None:
        std = std.clamp_min(std_floor)
    return mean, std.to(device=x_before.device, dtype=x_before.dtype)


def gaussian_log_prob(x_after, mean, std):
    """Joint conditional log density: sum all 16x2 coordinates, never mean."""
    if x_after.shape != mean.shape or mean.ndim != 3 or mean.shape[1:] != (16, 2):
        raise ValueError("Expected matching [B,16,2] transition tensors")
    if not torch.isfinite(x_after).all() or not torch.isfinite(mean).all():
        raise ValueError("Gaussian endpoints and means must be finite")
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("A Gaussian density needs finite positive std")
    terms = -0.5 * ((x_after - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)
    return terms.sum(dim=(-2, -1))


@dataclass(frozen=True)
class SampledChain:
    """Detached rollout record; observations are the exact pre-plan conditioning.

    Shapes: chain [B,101,16,2], timesteps [100], old_means [B,100,16,2],
    old_stds [100], old_log_prob [B,100], targets [B,16,2], actions [B,8,2].
    Baseline old_log_prob is None because its final transition is deterministic.
    actor_version is caller-supplied provenance, not a weight identity guarantee.
    """

    observations: dict[str, torch.Tensor]
    chain: torch.Tensor
    timesteps: torch.Tensor
    old_means: torch.Tensor
    old_stds: torch.Tensor
    old_log_prob: torch.Tensor | None
    targets: torch.Tensor
    actions: torch.Tensor
    sampler_config: SamplerConfig
    scheduler_signature: str
    mode: str
    actor_version: str


class DenoisingPolicy(nn.Module):
    """Reuse one loaded OfficialDiffusionPolicy, without allocating an old actor.

    Construct with ``DenoisingPolicy(official, subset=...)``. Use the official
    ``observation_tensors(history)`` or a batched equivalent. Do not use the shared
    official object as an independent frozen teacher after enabling its weights.
    """

    def __init__(self, official, *, subset="output", config=SamplerConfig()):
        super().__init__()
        self.network = official.network
        self.config = config
        n = self.network
        if (not n.obs_as_global_cond or n.horizon != 16 or n.action_dim != 2
                or n.n_obs_steps != 2 or n.n_action_steps != 8 or n.num_inference_steps != 100):
            raise ValueError("Expected the official two-frame/global-condition 16/8 policy")
        self.scheduler_signature = _scheduler_signature(n.noise_scheduler)
        self.selection = select_trainable_parameters(n, subset)
        self.train(False)

    def train(self, mode=True):
        # Eval is orthogonal to autograd. Keeping the whole upstream module in
        # eval guarantees the original 84x84 center crop even after actor.train().
        super().train(mode)
        self.network.eval()
        return self

    @torch.no_grad()
    def encode(self, observations):
        self.network.eval()
        if set(observations) != {"image", "agent_pos"}:
            raise ValueError("Actor inputs must be only image and agent_pos")
        image, position = observations["image"], observations["agent_pos"]
        if (image.ndim != 5 or image.shape[0] < 1 or image.shape[1:] != (2, 3, 96, 96)
                or position.shape != (image.shape[0], 2, 2)
                or not torch.isfinite(image).all() or not torch.isfinite(position).all()
                or image.min() < 0 or image.max() > 1):
            raise ValueError("Expected finite official [B,2,3,96,96] RGB and [B,2,2] positions")
        normalized = self.network.normalizer.normalize(observations)
        flattened = {k: v[:, :2].reshape(-1, *v.shape[2:]) for k, v in normalized.items()}
        return self.network.obs_encoder(flattened).reshape(image.shape[0], -1).detach()

    @torch.no_grad()
    def sample(self, observations, *, generator, mode="training", actor_version="unspecified"):
        if mode not in {"training", "baseline"}:
            raise ValueError("mode must be training or baseline")
        if generator is None:
            raise ValueError("An explicit torch.Generator is required")
        if _scheduler_signature(self.network.noise_scheduler) != self.scheduler_signature:
            raise ValueError("Scheduler changed after actor construction")
        observations = {k: v.detach().clone() for k, v in observations.items()}
        condition = self.encode(observations)
        n, scheduler = self.network, self.network.noise_scheduler
        scheduler.set_timesteps(100)
        x = torch.randn((condition.shape[0], 16, 2), dtype=n.dtype, device=n.device, generator=generator)
        chain, means, stds, logs = [x.clone()], [], [], []
        for timestep in scheduler.timesteps:
            t = int(timestep)
            epsilon = n.model(x, timestep, local_cond=None, global_cond=condition)
            mean, std = ddpm_transition(scheduler, epsilon, t, x,
                                        std_floor=self.config.std_floor if mode == "training" else None)
            if mode == "baseline":
                x = scheduler.step(epsilon, timestep, x, generator=generator).prev_sample
            else:
                x = mean + std * torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
                logs.append(gaussian_log_prob(x, mean, std))
            chain.append(x.clone())
            means.append(mean)
            stds.append(std)
        targets = n.normalizer["action"].unnormalize(x)
        return SampledChain(observations, torch.stack(chain, dim=1), scheduler.timesteps.clone(),
                            torch.stack(means, dim=1), torch.stack(stds),
                            torch.stack(logs, dim=1) if logs else None,
                            targets, targets[:, 1:9].clone(), self.config,
                            self.scheduler_signature, mode, actor_version)

    def transition_log_prob(self, observations, x_before, x_after, timestep):
        """Fresh forward with fixed recorded endpoints; no chain backpropagation."""
        return self.transition_log_prob_from_features(self.encode(observations), x_before, x_after, timestep)

    def transition_log_prob_from_features(self, features, x_before, x_after, timestep):
        """Same conditional density using verified, detached frozen features.

        C validates feature caches against encode() at batch entry. The standard
        observation path above remains the reference and B's default behavior.
        """
        if _scheduler_signature(self.network.noise_scheduler) != self.scheduler_signature:
            raise ValueError("Scheduler changed after actor construction")
        self.network.eval()
        if features.ndim != 2 or features.shape[0] != x_before.shape[0] or not torch.isfinite(features).all():
            raise ValueError("Expected finite matching frozen condition features")
        self.network.noise_scheduler.set_timesteps(100)
        epsilon = self.network.model(x_before.detach(), timestep, local_cond=None, global_cond=features.detach())
        mean, std = ddpm_transition(self.network.noise_scheduler, epsilon, timestep, x_before.detach(),
                                    std_floor=self.config.std_floor)
        return gaussian_log_prob(x_after.detach(), mean, std)

    def recompute_log_probs(self, record, *, step_indices=None, batch_indices=None):
        """Return [selected B, selected K] likelihoods, retaining only their graphs.

        For the real network choose a small step/batch subset, backpropagate, then
        discard its graph before the next subset. This is not 100-step pathwise
        differentiation. Weight each subset's summed PPO loss by the *full*
        transition count for exact gradient accumulation over unequal subsets.
        """
        if (record.mode != "training" or record.sampler_config != self.config
                or _canonical_scheduler_signature(record.scheduler_signature) != self.scheduler_signature):
            raise ValueError("Record does not belong to this training sampler configuration")
        if record.chain.shape[1:] != (101, 16, 2) or record.timesteps.tolist() != list(range(99, -1, -1)):
            raise ValueError("Malformed official denoising record")
        indices = list(range(100)) if step_indices is None else list(step_indices)
        if not indices or any(type(i) is not int or not 0 <= i < 100 for i in indices):
            raise ValueError("Select at least one valid transition index")
        batch = slice(None) if batch_indices is None else batch_indices
        observations = {k: v[batch].detach() for k, v in record.observations.items()}
        return torch.stack([self.transition_log_prob(observations, record.chain[batch, i],
                                                     record.chain[batch, i + 1], int(record.timesteps[i]))
                            for i in indices], dim=1)


def transition_ppo_loss(new_log_prob, old_log_prob, advantages, *, clip=0.2,
                        denoising_discount=1.0, timesteps=None, reduction="mean"):
    """One ratio per 32D transition; average over samples and denoising times.

    This explicit joint-density PPO objective differs from the public DPPO
    implementation's likelihood aggregation/clipping heuristics; it is not a
    line-by-line reproduction. No product of ratios across the 100-step chain.
    Advantages/old probabilities are fixed. No entropy, retention, weight decay,
    value loss or optimizer is supplied. A zero advantage gives zero actor grad.
    Denoising weights are discount**t (default 1), independent of physical gamma.
    ``sum / full_transition_count`` supports exact unequal microbatch weighting.
    """
    if new_log_prob.ndim != 2 or new_log_prob.shape != old_log_prob.shape or new_log_prob.numel() == 0:
        raise ValueError("Expected nonempty matching [B,K] log probabilities")
    if not math.isfinite(clip) or not 0 < clip < 1 or not 0 < denoising_discount <= 1:
        raise ValueError("Invalid PPO clip or denoising discount")
    if advantages.shape == (new_log_prob.shape[0],):
        advantages = advantages[:, None]
    if advantages.shape not in {(new_log_prob.shape[0], 1), new_log_prob.shape}:
        raise ValueError("Advantages must be [B], [B,1], or [B,K]")
    if any(not torch.isfinite(v).all() for v in (new_log_prob, old_log_prob, advantages)):
        raise ValueError("PPO inputs must be finite")
    ratio = (new_log_prob - old_log_prob.detach()).exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("Likelihood ratio overflow; reject this update rather than clamp its density")
    advantage = advantages.detach()
    surrogate = ratio * advantage
    if not torch.isfinite(surrogate).all():
        raise ValueError("PPO surrogate overflow; reject this update")
    loss = -torch.minimum(surrogate, ratio.clamp(1 - clip, 1 + clip) * advantage)
    if denoising_discount != 1:
        if timesteps is None or timesteps.shape != (new_log_prob.shape[1],):
            raise ValueError("Explicit [K] denoising timesteps required for discount")
        loss = loss * denoising_discount ** timesteps.to(loss.device)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError("reduction must be none, sum, or mean")
