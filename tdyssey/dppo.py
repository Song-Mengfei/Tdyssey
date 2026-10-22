"""Tail-transition PPO for the pinned official visual CNN diffusion policy.

This is a deliberately specified DPPO-style variant, not an upstream reproduction:
update t=K..1; freeze early transitions AND deterministic t=0. No variance floor,
coordinate averaging, proxy marginal likelihood or sampler change is introduced.
"""

from contextlib import contextmanager
import copy
import math

import numpy as np
import torch
from torch import nn


def ddpm_mean_std(scheduler, prediction, timestep, sample):
    """Pinned fixed_small epsilon DDPM conditional, matching diffusers 0.31.0."""
    cfg = scheduler.config
    if (cfg.prediction_type != "epsilon" or cfg.variance_type != "fixed_small"
            or cfg.thresholding or scheduler.num_inference_steps != 100):
        raise ValueError("Only the original 100-step fixed_small epsilon DDPM is supported")
    previous = scheduler.previous_timestep(timestep)
    alpha = scheduler.alphas_cumprod[timestep]
    previous_alpha = scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.one
    beta, previous_beta = 1-alpha, 1-previous_alpha
    current_alpha = alpha / previous_alpha
    current_beta = 1-current_alpha
    original = (sample-beta**.5*prediction) / alpha**.5
    if cfg.clip_sample:
        original = original.clamp(-cfg.clip_sample_range, cfg.clip_sample_range)
    mean = previous_alpha**.5*current_beta/beta*original + current_alpha**.5*previous_beta/beta*sample
    std = scheduler._get_variance(timestep)**.5 if timestep > 0 else torch.tensor(0.)
    return mean, std.to(sample.device)


def gaussian_log_prob(value, mean, std):
    """Joint log density over ALL latent action coordinates, no heuristic mean."""
    if not bool((std > 0).all()):
        raise ValueError("A deterministic transition has no Gaussian density")
    return (-.5*((value-mean)/std)**2-std.log()-.5*math.log(2*math.pi)).flatten(1).sum(1)


class TailPPO:
    def __init__(self, official, config):
        self.official, self.config = official, config
        self.network, self.device = official.network, official.device
        self.network.noise_scheduler.set_timesteps(100)
        if not self.network.obs_as_global_cond:
            raise ValueError("The pilot requires official global image conditioning")
        self.network.requires_grad_(False).eval()
        self.actor = self.network.model.final_conv if config.trainable == "head" else self.network.model
        self.base = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.actor.requires_grad_(True)
        self.names = [name for name, p in self.network.named_parameters() if p.requires_grad]
        self.critic = nn.Sequential(nn.Linear(self.network.obs_feature_dim*2, 128), nn.Tanh(),
                                    nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1)).to(self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)

    @contextmanager
    def module(self, updated):
        target = self.actor if updated else self.base
        if self.config.trainable == "head":
            old = self.network.model.final_conv
            self.network.model.final_conv = target
        else:
            old = self.network.model
            self.network.model = target
        try:
            yield
        finally:
            if self.config.trainable == "head":
                self.network.model.final_conv = old
            else:
                self.network.model = old

    @torch.no_grad()
    def encode(self, history):
        observations = {k: torch.as_tensor(np.stack([o[k] for o in history]), device=self.device).unsqueeze(0)
                        for k in ("image", "agent_pos")}
        normalized = self.network.normalizer.normalize(observations)
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in normalized.items()}
        return self.network.obs_encoder(flat).reshape(1, -1)

    @torch.no_grad()
    def sample(self, condition):
        x = torch.randn((1, 16, 2), device=self.device, generator=self.generator)
        transitions = []
        scheduler = self.network.noise_scheduler
        for t in scheduler.timesteps:
            timestep = int(t)
            updated = 1 <= timestep <= self.config.train_timesteps
            with self.module(updated):
                prediction = self.network.model(x, t, global_cond=condition)
            after = scheduler.step(prediction, t, x, generator=self.generator).prev_sample
            if updated:
                mean, std = ddpm_mean_std(scheduler, prediction, timestep, x)
                transitions.append({"t": timestep, "before": x.clone(), "after": after.clone(),
                                    "old_logp": gaussian_log_prob(after, mean, std)})
            x = after
        actions = self.network.normalizer["action"].unnormalize(x)[0, 1:9].cpu().numpy()
        if not np.isfinite(actions).all():
            raise FloatingPointError("Nonfinite sampled actions")
        return actions, transitions

    def log_prob(self, condition, before, after, timestep):
        with self.module(True):
            prediction = self.network.model(before, timestep, global_cond=condition)
        mean, std = ddpm_mean_std(self.network.noise_scheduler, prediction, timestep, before)
        return gaussian_log_prob(after, mean, std)

    def value(self, condition):
        return self.critic(condition).squeeze(-1)

    def state(self):
        return {"schema_version": 1, "kind": "official_tail_ppo", "config": self.config.__dict__,
                "base_sha256": self.official.spec["compact"]["sha256"],
                "actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load(self, payload):
        if (payload["kind"] != "official_tail_ppo" or payload["base_sha256"] != self.official.spec["compact"]["sha256"]
                or any(payload["config"][k] != getattr(self.config, k) for k in ("trainable", "train_timesteps"))):
            raise ValueError("Adapter base or policy parameterization differs")
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.critic.load_state_dict(payload["critic"], strict=True)


def ppo_update(agent, records, adv, targets, actor_optimizer, critic_optimizer):
    """One fresh physical rollout batch, multiple clipped-surrogate passes.

    Each Gaussian transition ratio is joint over 16x2 coordinates. All K trainable
    transitions receive the same physical-plan advantage (denoising discount=1).
    """
    config, device = agent.config, agent.device
    advantage = torch.tensor(adv, device=device, dtype=torch.float32)
    if len(records) > 1:
        advantage = (advantage-advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    returns = torch.tensor(targets, device=device, dtype=torch.float32)
    metrics = []
    for _ in range(config.epochs):
        order = torch.randperm(len(records)).tolist()
        for start in range(0, len(order), config.minibatch_size):
            indices = order[start:start+config.minibatch_size]
            condition = torch.cat([records[i]["condition"] for i in indices])
            actor_optimizer.zero_grad(set_to_none=True)
            losses, kls, clipped = [], [], []
            for slot in range(config.train_timesteps):
                transitions = [records[i]["transitions"][slot] for i in indices]
                logp = agent.log_prob(condition, torch.cat([t["before"] for t in transitions]),
                                     torch.cat([t["after"] for t in transitions]), transitions[0]["t"])
                old = torch.cat([t["old_logp"] for t in transitions])
                log_ratio = logp-old
                ratio = log_ratio.exp()
                if not torch.isfinite(ratio).all():
                    raise FloatingPointError("Nonfinite likelihood ratio; reduce update size")
                surrogate = torch.minimum(ratio*advantage[indices],
                    ratio.clamp(1-config.clip, 1+config.clip)*advantage[indices])
                loss = -surrogate.mean()/config.train_timesteps
                loss.backward()
                losses.append(float(loss.detach()))
                kls.append(float(((ratio-1)-log_ratio).mean().detach()))
                clipped.append(float(((ratio-1).abs() > config.clip).float().mean()))
            kl = sum(kls)/len(kls)
            if kl > config.target_kl:
                actor_optimizer.zero_grad(set_to_none=True)
                return {"early_stop_kl": kl, "minibatches": metrics}
            norm = nn.utils.clip_grad_norm_(agent.actor.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            actor_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss = .5*(agent.value(condition)-returns[indices]).square().mean()
            value_loss.backward()
            nn.utils.clip_grad_norm_(agent.critic.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            critic_optimizer.step()
            metrics.append({"policy_loss": sum(losses), "value_loss": float(value_loss.detach()),
                            "approx_kl": kl, "clip_fraction": sum(clipped)/len(clipped),
                            "actor_gradient_norm": float(norm)})
    return {"early_stop_kl": None, "minibatches": metrics}
