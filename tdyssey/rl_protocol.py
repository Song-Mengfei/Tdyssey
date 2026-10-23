"""Dependency-free reward and physical-time return rules for the RL pilot."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RLConfig:
    updates: int = 1
    plans_per_update: int = 8
    epochs: int = 1
    minibatch_size: int = 4
    train_timesteps: int = 4
    trainable: str = "head"
    lora_rank: int = 8
    lora_alpha: float = 8.
    lora_target: str = "head"
    gamma: float = .99
    gae_lambda: float = .95
    beta: float = 1.
    clip: float = .2
    actor_lr: float = 1e-5
    critic_lr: float = 3e-4
    max_grad_norm: float = 1.
    target_kl: float = .02
    episode_steps: int = 300
    eval_steps: int = 128
    eval_seeds: tuple = (100000,)
    seed: int = 42
    seconds: int = 600
    max_train_controls: int = 64
    max_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        for name, limit in (("updates", 10000), ("plans_per_update", 1024),
                            ("epochs", 10), ("minibatch_size", 64),
                            ("train_timesteps", 99), ("episode_steps", 500),
                            ("eval_steps", 500), ("seconds", 604800),
                            ("max_train_controls", 10000000), ("max_bytes", 16 * 1024**3)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"Invalid {name}")
        if self.trainable not in ("head", "action_network", "lora"):
            raise ValueError("trainable must be head, action_network or lora")
        if type(self.lora_rank) is not int or not 1 <= self.lora_rank <= 64:
            raise ValueError("Invalid LoRA rank")
        if not math.isfinite(self.lora_alpha) or not 0 < self.lora_alpha <= 128:
            raise ValueError("Invalid LoRA alpha")
        if self.lora_target not in ("head", "action_network"):
            raise ValueError("Invalid LoRA target")
        for name in ("gamma", "gae_lambda", "clip", "actor_lr", "critic_lr", "max_grad_norm", "target_kl"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"Invalid {name}")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("Invalid beta")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed")
        seeds = tuple(self.eval_seeds)
        if not seeds or len(seeds) > 20 or len(set(seeds)) != len(seeds):
            raise ValueError("Use 1-20 distinct development evaluation seeds")
        if any(type(s) is not int or not 0 <= s < 2**32 - 20 for s in seeds):
            raise ValueError("Invalid evaluation seed")
        object.__setattr__(self, "eval_seeds", seeds)


def shaped_reward(before, after, terminated, gamma=.99, beta=1.):
    """Original completion plus potential shaping; timeout is NOT termination."""
    return float(terminated) + beta * (gamma * (0. if terminated else after) - before)


def block_return(rewards, gamma):
    """Physical rewards discounted inside an actually executed action prefix."""
    return sum(gamma**i * reward for i, reward in enumerate(rewards))


def advantages(records, gamma, gae_lambda):
    """Per-plan SMDP GAE with physical-time lambda and explicit episode cuts.

    Records are one sequential environment stream. At a timeout, bootstrap from
    its last real observation but never propagate advantage into the next reset.
    Collection-buffer ends bootstrap too. Genuine success never bootstraps.
    """
    result = [0.] * len(records)
    carry = 0.
    for index in reversed(range(len(records))):
        row = records[index]
        discount = gamma**row["controls"]
        delta = row["reward"] + discount * (0. if row["terminated"] else row["next_value"]) - row["value"]
        carry = delta + (0. if row["episode_end"] else (gamma * gae_lambda)**row["controls"] * carry)
        result[index] = carry
    return result, [a + row["value"] for a, row in zip(result, records)]
