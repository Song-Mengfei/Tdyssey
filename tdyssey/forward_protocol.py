"""Dependency-free experiment contract for forward-process policy learning."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ForwardConfig:
    objective: str = "nft"
    trainable: str = "lora"
    lora_target: str = "action_network"
    lora_rank: int = 8
    lora_alpha: float = 8.
    updates: int = 1
    contexts_per_update: int = 2
    group_size: int = 4
    score_chunks: int = 1
    warmup_controls: int = 0
    epochs: int = 1
    minibatch_size: int = 1
    actor_lr: float = 1e-5
    max_grad_norm: float = 1.
    nft_beta: float = .5
    score_scale: float = .05
    anchor_margin: float = .01
    reference_weight: float = .001
    rollout_ema: float = 0.
    gamma: float = .99
    shaping_beta: float = 1.
    shapes: tuple = ("T", "L")
    eval_shapes: tuple = ("T", "L")
    eval_seeds: tuple = (100000,)
    eval_steps: int = 128
    eval_every: int = 0
    report_detail_updates: int = 20
    episode_steps: int = 300
    seed: int = 42
    seconds: int = 1200
    max_train_controls: int = 512
    max_bytes: int = 128 * 1024**2
    # Sample each candidate group (and each evaluation sweep) in lock-step with one
    # batched denoiser call per plan step; per-candidate/per-episode RNG streams.
    batched_rollouts: bool = False

    def __post_init__(self):
        for name, low, high in (("updates", 1, 10000), ("contexts_per_update", 1, 128),
                ("group_size", 2, 32), ("score_chunks", 1, 64), ("warmup_controls", 0, 499),
                ("epochs", 1, 10), ("minibatch_size", 1, 64), ("lora_rank", 1, 64),
                ("eval_steps", 1, 500), ("episode_steps", 1, 500), ("seconds", 1, 604800),
                ("eval_every", 0, 10000), ("report_detail_updates", 0, 10000),
                ("max_train_controls", 1, 10000000), ("max_bytes", 1, 16*1024**3),
                ("seed", 0, 2**32-1)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
        if type(self.batched_rollouts) is not bool:
            raise ValueError("batched_rollouts must be true or false")
        if self.objective not in ("nft", "advantage_flow"):
            raise ValueError("objective must be nft or advantage_flow")
        if self.trainable not in ("head", "action_network", "lora"):
            raise ValueError("Invalid trainable parameter scope")
        if self.lora_target not in ("head", "action_network"):
            raise ValueError("Invalid LoRA target")
        for name, low, high in (("lora_alpha", 0, 128), ("actor_lr", 0, 1),
                ("max_grad_norm", 0, 100), ("nft_beta", 0, 10), ("score_scale", 0, 100),
                ("anchor_margin", 0, 10), ("gamma", 0, 1)):
            value = getattr(self, name)
            if not math.isfinite(value) or not low < value <= high:
                raise ValueError(f"Invalid {name}")
        for name in ("reference_weight", "shaping_beta"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 100:
                raise ValueError(f"Invalid {name}")
        if not math.isfinite(self.rollout_ema) or not 0 <= self.rollout_ema < 1:
            raise ValueError("Invalid rollout EMA")
        for name in ("shapes", "eval_shapes"):
            values = tuple(getattr(self, name))
            if not values or len(values) > 27 or len(set(values)) != len(values):
                raise ValueError(f"Use distinct {name}")
            if any(v not in tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")+("T_narrow",) for v in values):
                raise ValueError(f"Unknown {name}")
            object.__setattr__(self, name, values)
        if "T" not in self.eval_shapes or len(self.eval_shapes) < 2:
            raise ValueError("Matched evaluation requires T and another shape")
        if self.reference_weight and "T" not in self.shapes:
            raise ValueError("T retention requires T training contexts")
        if self.warmup_controls >= self.episode_steps:
            raise ValueError("Warmup must leave room for candidate trials")
        seeds = tuple(self.eval_seeds)
        if (not seeds or len(seeds) > 20 or len(set(seeds)) != len(seeds)
                or any(type(s) is not int or not 0 <= s < 2**32-20 for s in seeds)):
            raise ValueError("Use 1-20 distinct evaluation seeds")
        object.__setattr__(self, "eval_seeds", seeds)


def group_feedback(scores, scale, anchor_margin=.01):
    """Center within one context; ties stay neutral, never divide by group std."""
    if (len(scores) < 2 or not all(math.isfinite(s) for s in scores)
            or not math.isfinite(scale) or not scale > 0):
        raise ValueError("Finite candidate group and positive scale required")
    mean = scores[0] if all(s == scores[0] for s in scores) else math.fsum(scores)/len(scores)
    advantage = [max(-1., min(1., (s-mean)/scale)) for s in scores]
    return {"mean": mean, "advantages": advantage,
            "optimality": [.5+.5*a for a in advantage],
            "anchor": max(abs(a) for a in advantage)+anchor_margin}
