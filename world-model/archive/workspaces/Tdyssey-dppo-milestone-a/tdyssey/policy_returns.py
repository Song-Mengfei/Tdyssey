"""Pure reward and return calculations for the direct-simulator policy pilot.

Coverage/reward labels are privileged simulator supervision. These helpers do
not run a simulator, estimate values, or update a policy. One ``ActionBlock`` is
one plan's *actually executed* physical controls, never denoising iterations.
The caller must retain the per-control ``TrainingReward`` records alongside the
on-policy buffer; reducing them to block rewards must not replace raw metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Iterable, Sequence


SUCCESS_COVERAGE = 0.95
DEFAULT_GAMMA = 0.99
DEFAULT_BETA = 1.0
DEFAULT_GAE_LAMBDA = 0.95


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _unit_interval(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _control_rewards(rewards: Iterable[float]) -> tuple[float, ...]:
    try:
        values = tuple(_finite(reward, "control reward") for reward in rewards)
    except TypeError as exc:
        raise ValueError("rewards must be an iterable of physical-control rewards") from exc
    if not values:
        raise ValueError("An action block must execute at least one physical control")
    return values


@dataclass(frozen=True)
class TrainingReward:
    """One physical transition, including unmodified simulator metrics.

    ``next_coverage`` is measured before any reset. ``next_potential`` differs
    from it only on genuine success termination. ``success_already_seen`` is
    caller-maintained episode state, reset only when a new episode starts.
    """

    current_coverage: float
    next_coverage: float
    upstream_reward: float
    success: bool
    first_success: bool
    terminated: bool
    next_potential: float
    shaping: float
    reward: float


def training_reward(
    current_coverage: float,
    next_coverage: float,
    upstream_reward: float,
    *,
    success_already_seen: bool = False,
    gamma: float = DEFAULT_GAMMA,
    beta: float = DEFAULT_BETA,
) -> TrainingReward:
    """Return first-success bonus plus coverage-potential shaping.

    Success is strictly ``next_coverage > 0.95`` and terminates this task. Its
    terminal potential is zero, even though the original coverage is retained.
    A repeated success record receives no second bonus. Callers normally stop
    execution at the first successful control; this helper does not authorize
    executing the remainder of that plan after termination.

    Timeout/buffer cutoff are not success: use the actual final observation's
    coverage here and their separate ``BlockBoundary`` for value bootstrapping.
    Negative shaping is retained. Gamma/beta are explicit, untuned pilot settings.
    """
    current = _unit_interval(current_coverage, "current_coverage")
    next_coverage = _unit_interval(next_coverage, "next_coverage")
    upstream_reward = _finite(upstream_reward, "upstream_reward")
    gamma = _unit_interval(gamma, "gamma")
    beta = _finite(beta, "beta")
    if beta < 0.0:
        raise ValueError("beta must be nonnegative")
    if not isinstance(success_already_seen, bool):
        raise ValueError("success_already_seen must be a bool")
    success = next_coverage > SUCCESS_COVERAGE
    first_success = success and not success_already_seen
    next_potential = 0.0 if success else next_coverage
    shaping = beta * (gamma * next_potential - current)
    return TrainingReward(
        current_coverage=current,
        next_coverage=next_coverage,
        upstream_reward=upstream_reward,
        success=success,
        first_success=first_success,
        terminated=success,
        next_potential=next_potential,
        shaping=shaping,
        reward=float(first_success) + shaping,
    )


class BlockBoundary(str, Enum):
    """The reason the executed action block ended."""

    CONTINUE = "continue"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    BUFFER_CUTOFF = "buffer_cutoff"


@dataclass(frozen=True, kw_only=True)
class ActionBlock:
    """Fixed samples/values for one executed physical action prefix.

    ``final_observation_value`` must be V of the observation immediately after
    the last executed control, *before resetting* an episode. In particular, a
    timeout uses its final observation, never the next episode's initial value.
    A true success does not bootstrap and may provide None. These numeric helpers
    cannot verify which image was supplied to a critic; the rollout collector
    must preserve that observation/value association.

    Use a unique ``episode_id`` per episode, including for mixed-shape buffers.
    A normal plan boundary retains both masks; success clears both. Timeout and
    buffer cutoff retain bootstrap b=1 but clear GAE continuation c=0. A buffer
    cutoff is the final item of a buffer and may resume that episode in a later
    buffer; advantages never continue beyond the current buffer.
    """

    episode_id: str
    rewards: tuple[float, ...]
    value: float
    final_observation_value: float | None
    boundary: BlockBoundary

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must identify one episode with a nonempty string")
        object.__setattr__(self, "rewards", _control_rewards(self.rewards))
        object.__setattr__(self, "value", _finite(self.value, "value"))
        try:
            boundary = BlockBoundary(self.boundary)
        except (TypeError, ValueError) as exc:
            raise ValueError("Unknown action-block boundary") from exc
        object.__setattr__(self, "boundary", boundary)
        if self.final_observation_value is None:
            if boundary != BlockBoundary.SUCCESS:
                raise ValueError("A bootstrapped block needs its pre-reset final_observation_value")
        else:
            object.__setattr__(self, "final_observation_value",
                               _finite(self.final_observation_value, "final_observation_value"))

    @property
    def num_controls(self) -> int:
        return len(self.rewards)

    @property
    def bootstrap_mask(self) -> int:
        return int(self.boundary != BlockBoundary.SUCCESS)

    @property
    def continuation_mask(self) -> int:
        return int(self.boundary == BlockBoundary.CONTINUE)


def discounted_block_return(
    rewards: Iterable[float], *, gamma: float = DEFAULT_GAMMA,
) -> float:
    """Compute sum_i gamma**i * r_i over a nonempty executed physical prefix."""
    gamma = _unit_interval(gamma, "gamma")
    values = _control_rewards(rewards)
    return math.fsum(gamma ** index * reward for index, reward in enumerate(values))


@dataclass(frozen=True)
class PlanEstimate:
    """Return/advantage targets for the corresponding ``ActionBlock``."""

    discounted_reward: float
    physical_discount: float
    delta: float
    advantage: float
    value_target: float


def plan_gae(
    blocks: Sequence[ActionBlock],
    *,
    gamma: float = DEFAULT_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
) -> tuple[PlanEstimate, ...]:
    """Estimate advantages across one ordered stream of physical plans.

    For m actually executed controls, R=sum_i gamma**i*r_i and
    delta=R+gamma**m*b*V(final_observation)-V(start). Then
    A=delta+gamma**m*gae_lambda*c*A(next plan), and value_target=A+V.
    Lambda is applied once per plan boundary; it is not a per-control lambda
    raised to m. Denoising iterations do not enter this calculation.

    Each episode must be contiguous, resets need a new episode_id, and the last
    block must explicitly mark success, timeout or buffer cutoff (c=0). This
    rejects silently joining episodes or bootstrapping from a following reset.
    Call separately for each rollout stream; no advantage crosses streams.
    """
    gamma = _unit_interval(gamma, "gamma")
    gae_lambda = _unit_interval(gae_lambda, "gae_lambda")
    blocks = tuple(blocks)
    if not blocks or any(not isinstance(block, ActionBlock) for block in blocks):
        raise ValueError("blocks must contain at least one ActionBlock")
    if blocks[-1].continuation_mask:
        raise ValueError("The buffer tail must explicitly clear GAE continuation")
    seen_episodes: set[str] = set()
    for index, block in enumerate(blocks):
        if block.boundary == BlockBoundary.BUFFER_CUTOFF and index != len(blocks) - 1:
            raise ValueError("A buffer cutoff must be the final block")
        if index:
            previous = blocks[index - 1]
            if previous.continuation_mask:
                if block.episode_id != previous.episode_id:
                    raise ValueError("An episode change requires a terminated/truncated boundary")
            elif block.episode_id == previous.episode_id:
                raise ValueError("An episode reset requires a new episode_id")
        if index == 0 or block.episode_id != blocks[index - 1].episode_id:
            if block.episode_id in seen_episodes:
                raise ValueError("Each episode must form one contiguous sequence")
            seen_episodes.add(block.episode_id)

    rewards = [discounted_block_return(block.rewards, gamma=gamma) for block in blocks]
    discounts = [gamma ** block.num_controls for block in blocks]
    deltas = [reward + discount * block.bootstrap_mask *
              (block.final_observation_value if block.bootstrap_mask else 0.0) - block.value
              for block, reward, discount in zip(blocks, rewards, discounts)]
    advantages = [0.0] * len(blocks)
    next_advantage = 0.0
    for index in range(len(blocks) - 1, -1, -1):
        next_advantage = (deltas[index] + discounts[index] * gae_lambda *
                          blocks[index].continuation_mask * next_advantage)
        advantages[index] = next_advantage
    return tuple(
        PlanEstimate(discounted_reward=reward, physical_discount=discount,
                     delta=delta, advantage=advantage, value_target=advantage + block.value)
        for block, reward, discount, delta, advantage in
        zip(blocks, rewards, discounts, deltas, advantages)
    )
