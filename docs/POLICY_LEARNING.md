# Direct policy learning: method survey, derivations and runnable pilot

Updated: 2026-10-23. This page specifies the first implemented policy-update
method. The [whole-project plan](LEARNING_PLAN.md) still governs data exposure,
world-model work and evaluation. Runtime evidence is recorded in [status](STATUS.md)
and [validation](VALIDATION.md); an implementation is not an improvement result.

## Which method fits this project?

Primary-source survey checked 2026-10-23. The fit judgments are our assessment,
not published PushT/alphabet performance claims. No method is established as
best for our geometry, checkpoint or data budget.

| Method | Published mechanism | Fit and limitation for this project |
| --- | --- | --- |
| [DPPO](https://arxiv.org/abs/2409.00588) | PPO on an augmented process whose internal decisions are diffusion transitions | Directly changes the pretrained action network. Strong first baseline; fresh rollouts and many network calls cost time. |
| [Residual PPO / ResiP](https://residual-assembly.github.io/) | Frozen chunked base policy plus per-control learned corrections | Strong precision/reactivity comparison. Improves a combined controller while base weights stay fixed; observation/state choices require matched comparisons. |
| [DSRL](https://arxiv.org/abs/2506.15799) | RL chooses diffusion starting noise while the base policy stays frozen | Useful if better plans are reachable in the base model. Its deterministic-decoder setup needs a sampler compatibility decision for our DDPM. |
| [Q-score matching](https://arxiv.org/abs/2312.11752) | Connect diffusion-policy scores to gradients of a learned action-value function | Reuses data; action-value gradients around contact can be unreliable. Adds critic/replay complexity to the first experiment. |
| [OGPO](https://arxiv.org/abs/2605.03065) | Off-policy critics score generated actions for PPO-style generative-policy improvement | Promising second full-network method if data reuse matters. Requires reliable critics and stabilization; its internal denoising optimization is not a world-model environment rollout. |
| [DiPOD](https://arxiv.org/abs/2606.13795) | Self-distillation/ELBO regularization addresses drift in surrogate diffusion-policy optimization | Relevant caution about proxy likelihoods. We use explicit conditional Gaussian densities, not an ELBO substituted for final-action likelihood; this is not a claim that all its concerns disappear. |

Select a checked DPPO-style baseline first, then residual PPO or OGPO when a
failure identifies the useful comparison. Implementing all six at once would
confound debugging and consume the experiment budget. A new method is not required
to make adapting a visual policy across shape/contact changes substantial work.

## Exact first variant: train selected stochastic tail transitions

The implementation is `tdyssey/dppo.py`, called by
`scripts/train_official_policy.py`. It deliberately differs from the full published
DPPO implementation and must be described as **tail-transition PPO**, not a
reproduction of the authors' benchmark scores or all best practices.

- Keep the pinned 100-step official DDPM, image processing, normalization, 16
  targets and executed indices 1:9. Inference before updates must match the
  official sampler and its following random state exactly.
- Update the network only at sampler indices `K, K-1, ..., 1`. Use a frozen
  reference for earlier transitions and **for deterministic index 0**. Do not
  assign a Gaussian probability to a deterministic transition.
- The image encoder and normalizers remain frozen. `head` updates the final
  convolution block; `action_network` updates the entire action U-Net. A frozen
  copy of the selected module supplies unchanged steps. Both modes use the same
  script and data/reward contract; runtime memory must be measured.
- Use actual fixed-small DDPM variance, with no extra variance floor or noise
  injection. Use joint Gaussian density across all 16x2 latent coordinates.
  Do not average coordinate log probabilities and call that a joint density.
- Use denoising discount 1, physical discount 0.99 by default, and physical-time
  GAE lambda 0.95. No reward-weighted behavior cloning, ELBO proxy, expert replay,
  teacher-retention loss or world model is present in this initial trainer.

The actor that is evaluated and saved follows the same frozen/updated schedule
as training. Loading its delta into the original inference wrapper and applying
it to all 100 transitions would define a different policy. Reload through
`TailPPO.load` together with the pinned base and saved configuration.

The output-head mode is an implementation/memory check and a limited adaptation
baseline. It does not establish that frozen visual features or that small a
parameter subset can solve all shapes. Full action-network training is provided
for a larger GPU; its runtime validation must be reported separately.
The frozen last transition can also restrict adaptation. If it becomes a measured
limitation, test stochastic-final-step fine-tuning as a separate variant, with a
matched untrained sampler baseline. It is not silently enabled in this pilot.

## Derivation 1: why rewards can update denoising weights

Let `h` be the observed history, `z` the recorded denoising chain and `R` the
future discounted task return. The starting noise and frozen transitions have
no trainable parameters. Every selected transition has a known conditional
Gaussian density, written `q_theta`.

```text
joint_chain_density = fixed_factors * product(q_theta for selected transitions)

gradient log(joint_chain_density)
    = sum(gradient log(q_theta) for selected transitions)

gradient expected_return
    = expected[return * sum(gradient log(q_theta))]
```

This is the score-function identity: differentiate the trajectory probability,
then use `gradient p = p * gradient log p`. Physical transition dynamics do not
depend on policy weights and contribute no score term. A frozen deterministic
last step is a parameter-independent mapping, so it also contributes no trainable
score. It is part of the outcome, not a stochastic density to divide by.

An observation-dependent value baseline can be subtracted: conditional on that
observation, the expected score of a normalized distribution is zero. Credit can
therefore depend on whether the result exceeded predicted future return, not
only whether it exceeded the previous overlap.

Assumptions include on-policy samples, differentiable conditional densities,
parameter-independent support and exchanging derivative/integration under suitable
integrability conditions. The identity concerns the complete trajectory return;
GAE/value approximation and finite truncated data introduce estimation choices.
No simulator derivative is needed. This is the same augmented-process principle
used in [DPPO](https://arxiv.org/html/2409.00588v3); it does not assert that a
product of latent densities equals the marginal density of the final action.

We then use [PPO's clipped surrogate](https://arxiv.org/abs/1707.06347) separately
for recorded selected transitions. Its gradient at the old policy connects to
the score-function update; several clipped passes are an approximate practical
optimization, not an unbiased policy gradient at every later minibatch. Clipping
and our sampled approximate-KL stop do **not** prove monotonic policy improvement
or bound every unseen action. Advantages are normalized over the fresh rollout
batch. Equal averaging across the fixed K transitions rescales the objective.

## Derivation 2: reward shaping and completion

The first task still terminates at strict physical coverage > 0.95. The base
reward is 1 at completion and 0 otherwise. Let `Phi` be coverage until genuine
termination, where the absorbing terminal potential is zero. Proposed shaping is:

```text
shaping_t = beta * (gamma * Phi_next - Phi_now)
training_reward_t = completion_reward_t + shaping_t

sum over H controls of gamma**t * shaping_t
    = beta * (-Phi_initial + gamma**H * Phi_final)
```

Intermediate potentials cancel. At true termination `Phi_final=0`; the return
changes by a constant determined by the initial observation. With bounded potential
and discount below one, the infinite-horizon remainder also vanishes. Under these
conditions the shaping preserves the task's policy ordering, rather than rewarding
high-overlap camping. See [Ng, Harada and Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

A time limit is not success. Bootstrap from the final physical observation and
retain its nonterminal potential; cut advantage propagation across the reset.
For a prefix of `m` actually executed controls:

```text
block_reward = r0 + gamma*r1 + ... + gamma**(m-1)*r_(m-1)
temporal_difference = block_reward + gamma**m * next_value - current_value
GAE_carry_multiplier = (gamma * lambda)**m
```

The next value is zero for genuine success, but not for time-limit truncation.
These identities have independent source tests for telescoping, short prefixes
and a timeout followed by a reset into a high-value episode. The last case ensures
that one episode's advantage cannot leak into the previous episode.

Coverage is flat before contact/overlap, so this reward can have poor exploration.
Potential shaping cannot create informative signal everywhere. The value estimator
is visual/history-based; object geometry supplies privileged reward labels only.
The pilot preserves raw upstream rewards separately and has no learned stop head.
Sustained alignment needs its own declared completion rule, not merely this
first-crossing reward. Full-reset success must be evaluated even if a later
training curriculum introduces easier near-goal scenes.

## Derivation 3: what limits imagined learning

A standard coupling argument explains why shorter predicted rollouts can reduce
model bias. Suppose rewards are bounded in absolute value by `Rmax`, conditional
reward prediction error is at most `eps_r`, and conditional transition distributions
differ by total variation at most `eps_p` over the visited histories. Starting from
the same history and following the same policy, the probability that coupled
histories have diverged by control `t` is at most `t*eps_p` (capped at one).

```text
absolute difference in H-control expected return
    <= sum for t=0..H-1 of gamma**t * (eps_r + 2*Rmax*t*eps_p)
```

Proof: before divergence, the same policy can choose the same action and reward
error is at most `eps_r`. After divergence, bounded rewards differ by at most
`2*Rmax`. Apply the union bound over transition mismatches, then sum discounted
per-control errors. This is an explanatory bound under stated assumptions, not
a measured guarantee for our predictor. RGB MSE does not certify total variation;
deterministic systems with slightly different exact predictions can even have
maximal total variation. Partial observations and rare contacts make empirical
validation essential. The current trainer does not use imagined rollouts.

## Promising project-specific extensions, not new-method claims

1. **Finishing and reaction rate:** compare the same trained actor with original
   eight-control execution and shorter prefixes under matched inference/compute
   records. Add a separately specified sustained-alignment objective. This tests
   whether errors arise from stale plans or the chosen contacts, rather than
   assuming every near miss only needs stopping.
2. **Residual corrections:** learn per-control corrections and compare with weight
   updates under the same visual information. It directly tests whether feedback
   precision is sufficient or new long-horizon plans are needed.
3. **Model-assisted practice with selective horizons:** validate contact/reward
   predictions, freeze dynamics, start short and compare physical performance.
   Any confidence gate requires measured calibration; model disagreement alone
   is not a proof of correctness. The bound above motivates the question without
   establishing the benefit.

These are research hypotheses and ablations. No novelty, convergence, alphabet
success or world-model benefit has been established by deriving an update rule.

## Run and inspect

Current evidence: a 64-control GPU update check and a 1,008-control head-only
pilot completed. The latter retained T success on one scene (270 to 297 controls)
and did not improve L. Peak allocated CUDA memory was about 1.11 GiB; no full
action-network update has been runtime-validated. See the complete counts,
negative result and source/report hashes in [validation](VALIDATION.md).

The next experiment should first check full-network gradients/reload on the
larger machine, then compare capacity and data budgets using fixed development
scenes. Do not interpret a longer run as sufficient by itself; inspect L contacts,
reward coverage and T retention. Training alternates T/L plans, so early T success
can make per-shape physical-control counts differ.

Use the already authorized [CUDA/DP environment](ENVIRONMENT.md) and pinned official
source/checkpoint. No new package, dataset or world-model asset is needed.

```bash
# Source-only interface and mathematical boundary checks:
python3 -B scripts/train_official_policy.py --help
python3 -B -m unittest discover -s tests -p 'test_rl_protocol.py' -v

# Small end-to-end GPU update plus paired before/after development rollouts:
.venv/bin/python -B scripts/train_official_policy.py --config configs/dppo_smoke.json --output artifacts/dppo-smoke-001

# Longer bounded head-update pilot; not required for a basic installation check:
.venv/bin/python -B scripts/train_official_policy.py --config configs/dppo_local_pilot.json --output artifacts/dppo-local-pilot-001

# Larger head-update experiment:
.venv/bin/python -B scripts/train_official_policy.py --config configs/dppo_pilot.json --output artifacts/dppo-pilot-001

# First check full action-network updates on an independently authorized larger GPU:
.venv/bin/python -B scripts/train_official_policy.py --config configs/dppo_action_network_smoke.json --output artifacts/dppo-action-network-smoke-001

# Then the full action-network learning pilot:
.venv/bin/python -B scripts/train_official_policy.py --config configs/dppo_action_network.json --output artifacts/dppo-action-network-001
```

Output paths must be new. The smoke profile requests one update / at most 64
training controls, plus before/after evaluation capped at 128 controls per T/L.
The local pilot caps training at 1,024 controls and total time at 20 minutes.
The larger learning profiles cap training at 20,000 controls and four hours, with final
300-control development evaluation. These are runtime bounds, not sufficient
training budgets or expected success rates. The full-network profile allows
8 GiB of output; weights/Adam state are much larger than head deltas. Review
storage/compute settings before long jobs on each machine.

Outputs are `report.json` and `last.pt`, both ignored. The report separates
training/evaluation controls, reset physics, denoising and optimizer work; records
per-control reward/coverage/contact/action traces, trainable names, source hashes,
before/after performance, frozen-weight integrity, actor change and checkpoint
action parity. It records failures/timeouts instead of calling them completed.
The checkpoint contains selected actor weights, critic, configuration and Adam
states; the original pinned base is still required. Automatic resume and standalone
video evaluation of a trained delta are not implemented yet. Interruptions retain
progress reports after completed updates but do not yet save recovery checkpoints
each update. Completed checkpoints are saved after final evaluation.

Compare success/final/peak overlap across matched development scenes. Do not
claim success from policy loss, nonzero gradients or the larger of two noisy
returns. The smoke check is expected to establish mechanics, not learn L from
64 controls. Final reserved scenes and repeated training seeds are later work.
