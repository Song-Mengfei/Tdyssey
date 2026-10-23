"""DDPM endpoint fine-tuning with NFT or regularized advantage regression.

These are epsilon-prediction adaptations, not reproductions of image benchmarks.
The original finite sampler is retained; updated weights apply to all 100 steps.
"""

from contextlib import contextmanager
import copy

import numpy as np
import torch
from torch import nn


def forward_noise(scheduler, clean, noise, timesteps):
    """Analytic corruption without mutating the finite inference scheduler.

    diffusers.add_noise moves scheduler.alphas_cumprod in place. Sampling with
    that GPU buffer changes scalar rounding versus the pinned CPU-buffer sampler.
    Keep this training tensor local so an unchanged actor stays exactly unchanged.
    """
    alpha = scheduler.alphas_cumprod.to(device=clean.device, dtype=clean.dtype)[timesteps]
    alpha = alpha.reshape(-1, *([1]*(clean.ndim-1)))
    return alpha.sqrt()*clean+(1-alpha).sqrt()*noise


def nft_loss(prediction, old, target, optimality, beta, reliability=None):
    weight = optimality if reliability is None else .5+reliability*(optimality-.5)
    delta = prediction-old
    positive = (old+beta*delta-target).square().flatten(1).mean(1)
    negative = (old-beta*delta-target).square().flatten(1).mean(1)
    return (weight*positive+(1-weight)*negative).mean()


def advantage_flow_loss(prediction, old, target, advantage, anchor):
    error = (prediction-target).square().flatten(1).mean(1)
    retention = (prediction-old).square().flatten(1).mean(1)
    return (advantage*error+anchor*retention).mean()


class ForwardPolicy:
    def __init__(self, official, config):
        self.official, self.config = official, config
        self.network, self.device = official.network, official.device
        self.network.requires_grad_(False).eval()
        self.network.noise_scheduler.set_timesteps(100)
        scheduler = self.network.noise_scheduler
        if (not self.network.obs_as_global_cond or scheduler.config.prediction_type != "epsilon"
                or len(scheduler.timesteps) != 100):
            raise ValueError("Expected the original globally conditioned epsilon DDPM")
        self.lora_layers = []
        if config.trainable == "lora":
            from .lora import inject_lora
            target = self.network.model.final_conv if config.lora_target == "head" else self.network.model
            self.lora_layers = inject_lora(target, config.lora_rank, config.lora_alpha)
            self.actor = nn.ModuleDict({str(i): layer.delta for i, layer in enumerate(self.lora_layers)})
            self.reference = None
        else:
            self.actor = self.network.model.final_conv if config.trainable == "head" else self.network.model
            self.reference = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.actor.requires_grad_(True)
        self.old = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.names = [name for name, p in self.network.named_parameters() if p.requires_grad]
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self.noise_generator = torch.Generator(device=self.device).manual_seed(config.seed+1)
        self.order_generator = torch.Generator().manual_seed(config.seed+2)

    @contextmanager
    def use(self, mode):
        if mode not in ("actor", "old", "reference"):
            raise ValueError("Unknown policy view")
        if self.lora_layers:
            saved = [(layer.delta, layer.enabled) for layer in self.lora_layers]
            for index, layer in enumerate(self.lora_layers):
                layer.enabled = mode != "reference"
                if layer.enabled:
                    layer.delta = (self.actor if mode == "actor" else self.old)[str(index)]
            try:
                yield
            finally:
                for layer, (delta, enabled) in zip(self.lora_layers, saved):
                    layer.delta, layer.enabled = delta, enabled
        else:
            target = {"actor": self.actor, "old": self.old, "reference": self.reference}[mode]
            if self.config.trainable == "head":
                saved = self.network.model.final_conv
                self.network.model.final_conv = target
            else:
                saved = self.network.model
                self.network.model = target
            try:
                yield
            finally:
                if self.config.trainable == "head":
                    self.network.model.final_conv = saved
                else:
                    self.network.model = saved

    @torch.no_grad()
    def encode(self, history):
        observations = {k: torch.as_tensor(np.stack([o[k] for o in history]), device=self.device).unsqueeze(0)
                        for k in ("image", "agent_pos")}
        normalized = self.network.normalizer.normalize(observations)
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in normalized.items()}
        return self.network.obs_encoder(flat).reshape(1, -1)

    def predict(self, x, timesteps, condition, mode="actor"):
        with self.use(mode):
            return self.network.model(x, timesteps, global_cond=condition)

    @torch.no_grad()
    def sample(self, condition, mode="actor"):
        x = torch.randn((1, 16, 2), device=self.device, generator=self.generator)
        scheduler = self.network.noise_scheduler
        for t in scheduler.timesteps:
            prediction = self.predict(x, t, condition, mode)
            x = scheduler.step(prediction, t, x, generator=self.generator).prev_sample
        actions = self.network.normalizer["action"].unnormalize(x)[0, 1:9].cpu().numpy()
        if not np.isfinite(actions).all() or not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite sampled plan")
        return actions, x.detach()

    @torch.no_grad()
    def refresh_old(self):
        decay = self.config.rollout_ema
        source_state = self.actor.state_dict()
        for name, value in self.old.state_dict().items():
            source = source_state[name]
            if value.is_floating_point():
                value.mul_(decay).add_(source, alpha=1-decay)
            else:
                value.copy_(source)

    def state(self):
        return {"schema_version": 1, "kind": "official_forward_rl", "config": self.config.__dict__,
                "base_sha256": self.official.spec["compact"]["sha256"],
                "actor": self.actor.state_dict(), "old": self.old.state_dict(),
                "plan_rng": self.generator.get_state(), "noise_rng": self.noise_generator.get_state(),
                "order_rng": self.order_generator.get_state()}

    def load(self, payload):
        keys = ("trainable", "lora_target", "lora_rank", "lora_alpha", "objective")
        if (payload.get("kind") != "official_forward_rl" or payload.get("schema_version") != 1
                or payload["base_sha256"] != self.official.spec["compact"]["sha256"]
                or any(payload["config"][k] != getattr(self.config, k) for k in keys)):
            raise ValueError("Forward checkpoint/base/parameterization differs")
        self.actor.load_state_dict(payload["actor"], strict=True)
        self.old.load_state_dict(payload["old"], strict=True)
        self.generator.set_state(payload["plan_rng"].cpu())
        self.noise_generator.set_state(payload["noise_rng"].cpu())
        self.order_generator.set_state(payload["order_rng"].cpu())


def forward_update(agent, records, optimizer, deadline=lambda: None):
    """A fixed rollout batch; fresh forward noise/time, no trajectory gradients."""
    config, device = agent.config, agent.device
    metrics = []
    for _ in range(config.epochs):
        order = torch.randperm(len(records), generator=agent.order_generator).tolist()
        for start in range(0, len(order), config.minibatch_size):
            deadline()
            batch = [records[i] for i in order[start:start+config.minibatch_size]]
            condition = torch.cat([r["condition"] for r in batch]).to(device)
            clean = torch.cat([r["plan"] for r in batch]).to(device)
            noise = torch.randn(clean.shape, device=device, generator=agent.noise_generator)
            t = torch.randint(0, 100, (len(batch),), device=device, generator=agent.noise_generator)
            noisy = forward_noise(agent.network.noise_scheduler, clean, noise, t)
            with torch.no_grad():
                old = agent.predict(noisy, t, condition, "old")
            prediction = agent.predict(noisy, t, condition)
            optimizer.zero_grad(set_to_none=True)
            if config.objective == "nft":
                optimality = torch.tensor([r["optimality"] for r in batch], device=device)
                loss = nft_loss(prediction, old, noise, optimality, config.nft_beta)
            else:
                advantage = torch.tensor([r["advantage"] for r in batch], device=device)
                anchor = torch.tensor([r["anchor"] for r in batch], device=device)
                loss = advantage_flow_loss(prediction, old, noise, advantage, anchor)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite forward-policy loss")
            loss.backward()
            # T-only teacher plans come from the initial checkpoint, not human data.
            retained = [i for i, r in enumerate(batch) if r.get("reference_plan") is not None]
            retention = torch.zeros((), device=device)
            if retained and config.reference_weight:
                clean_t = torch.cat([batch[i]["reference_plan"] for i in retained]).to(device)
                noisy_t = forward_noise(agent.network.noise_scheduler, clean_t, noise[retained], t[retained])
                with torch.no_grad():
                    reference = agent.predict(noisy_t, t[retained], condition[retained], "reference")
                actual = agent.predict(noisy_t, t[retained], condition[retained])
                # Mean over the whole minibatch, with zero contribution from non-T.
                retention = (actual-reference).square().mean()*len(retained)/len(batch)
                (config.reference_weight*retention).backward()
            norm = nn.utils.clip_grad_norm_(agent.actor.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            metrics.append({"loss": float(loss.detach()), "T_retention_loss": float(retention.detach()),
                            "gradient_norm": float(norm), "noise_timesteps": t.tolist()})
    agent.refresh_old()
    return metrics
