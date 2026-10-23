# RL for diffusion and flow policies: research and implementation choices

Checked: 2026-10-23, using the primary sources linked below. This is a method
selection note. The selected [NFT/AdvantageFlow adaptations](FORWARD_POLICY_LEARNING.md)
are now implemented alongside [tail-transition PPO](POLICY_LEARNING.md); this
does not imply successful adaptation or that every surveyed method runs here.
No new packages or pretrained assets were needed.

## Recommendation and project scope

Keep the official CNN Diffusion Policy and the existing PPO implementation as
the starting baseline. **Compare DDPM-compatible DiffusionNFT next**:
it offers a substantially different weight-update rule without requiring a new
policy architecture, a differentiable simulator or a learned action-value critic.
Compare AdvantageFlow's regularized regression loss after the common collection
and denoising code works. RAM and QAM are worthwhile alternatives, with additional
approximation or critic/solver work described below.

This is a course project. Implementing established methods, comparing them fairly,
adapting them to this checkpoint and explaining failures are valid project work.
We do not need an original RL algorithm or a claim of algorithmic novelty. This
does **not** reduce the intended scope: direct simulator learning, a validated
shared world model, world-model-assisted learning and shared multi-shape policies
remain in the [whole-project plan](LEARNING_PLAN.md). T/L verifies the loop before
expansion; it is not the final shape scope. A specific extension is proposed
below as an experiment, with no claim that its ingredients are new.

## What changed after looking beyond DPPO?

There are two useful families. Reverse-process methods reward the internal
denoising transitions, using their known Gaussian probabilities. Forward-process
methods generate complete samples, score them, add fresh noise analytically, and
learn from the resulting denoising examples. Here the generated sample is an
**action plan**, not a picture. Its reward must come from physical interaction or
validated predictions of that interaction.

The fit judgments are our assessment. Image-generation results and speedups are
not PushT results, and fewer training network calls do not remove the cost of
generating plans, running physics or predicting futures.

| Method and source version | Main mechanism | Assessment for this project |
| --- | --- | --- |
| [DPPO](https://arxiv.org/abs/2409.00588), 2024 | PPO on stochastic denoising decisions | Keep as the checked baseline. Our tail variant differs from the published implementation. |
| [Flow-GRPO](https://arxiv.org/html/2505.05470v5), Oct 2025 | Converts flow sampling to a stochastic process and uses group-relative rewards | Useful flow baseline; changes the sampler and needs a flow-compatible policy. Group rewards still cost physical or predicted action trials. |
| [ReinFlow](https://arxiv.org/html/2505.22094v7), Jan 2026 | Learns noise injection alongside a flow policy for tractable transition probabilities | Already studies robot policy fine-tuning. Attractive for a later flow-policy comparison; not a direct loader for our released DDPM checkpoint. |
| [FPO](https://arxiv.org/html/2507.21053v2), Aug 2025 | PPO-like update using exponentiated differences in denoising/flow-matching losses | Works with several prediction parameterizations and black-box sampling. Its ratio is an ELBO proxy, not the exact final-action likelihood ratio. |
| [DiffusionNFT](https://arxiv.org/html/2509.16117v2), Feb 2026 | Fits implicit positive and negative denoising branches around a rollout policy | First alternative to implement. Uses complete samples and scalar feedback; no reward gradient or likelihood computation. Negative feedback is handled through bounded quadratic losses. |
| [V-GRPO](https://arxiv.org/html/2604.23380v1), Apr 2026 | ELBO-based GRPO with shared noise/time samples, variance reduction and update controls | Useful comparison if implementing a likelihood surrogate. Monte Carlo variance and surrogate bias are separate issues. |
| [AdvantageFlow](https://arxiv.org/html/2605.26013v2), Sep 2026 | Advantage-weighted prediction loss plus rollout/reference prediction penalties | Simple second objective on the same data. The anchor must dominate negative weights; per-prediction convexity does not make network optimization convex. |
| [RAM](https://arxiv.org/html/2605.10759v2), May 2026 | Reward-corrected regression targets derived from adjoint matching | Needs no reward gradients. The scalable form drops a path-cost correction and uses an approximate bridge; its general fixed point is not exactly the KL optimum. |
| [QAM](https://arxiv.org/html/2601.14234v4), May 2026 | Transforms a learned Q-function's action gradients into adjoint-matching targets | Promising when replay/data reuse matters. Adds critic learning and reference-model derivative calculations; critic errors near contact remain important. |
| [RFPO](https://arxiv.org/html/2609.15123v2), Sep 2026 | Refines generated actions using a critic, then fits the flow policy to those targets | Another useful route to improving plans. Adds critic gradients and finite-step stochastic refinement; reported control results do not validate our visual setting. |
| [DiffusionRFT study](https://arxiv.org/html/2609.33444v1), Sep 2026 | Relates regression methods to divergence-constrained reward optimization and studies target/normalizer choices | A recent preprint worth consulting when choosing losses. Merely adding weighted regression or a different normalization is not automatically a new method. |

The earlier [policy survey](POLICY_LEARNING.md) covers residual PPO, DSRL, Q-score
matching, OGPO and DiPOD. In particular, [DiPOD](https://arxiv.org/html/2606.13795v2)
warns about policy drift when optimizing diffusion likelihood surrogates. A lower
denoising loss is not automatically a higher physical task return.

## Closely related world-model work

[MODIP](https://arxiv.org/html/2606.10825v1) uses a learned world model and planning
to produce supervised targets for a diffusion policy. That supports a simpler
planning-to-policy-fitting comparison alongside direct RL; planning targets need
not be human demonstrations.

[Motus2](https://arxiv.org/html/2608.30237v1), section 3.3.2, already scores imagined
robot action candidates and applies DiffusionNFT while keeping prediction and
evaluation fixed. Its candidate groups also include a ground-truth action/future
anchor. We must not silently import that anchor as new human adaptation data.
For our setting, reference behavior can come from the frozen T policy on automated
T observations; target-shape feedback comes from machine interaction.

[WISE](https://arxiv.org/html/2609.03681v1) already combines bounded imagination,
selective use of prediction, confidence-weighted feedback and policy refinement.
[Dual-Frontier](https://arxiv.org/html/2609.26293v2) explicitly studies when a
predicted improvement exceeds a bound on decision-relevant model error. Earlier
[MBPO](https://arxiv.org/abs/1906.08253) and
[MOPO](https://arxiv.org/abs/2005.13239) also motivate short rollouts and model-error
control. Combining a world model, RL and uncertainty is established prior work.
These are references to adapt and evaluate, not reasons to abandon the project.

## How the proposed DiffusionNFT adaptation would work

Preserve the original RGB/pusher history, saved normalization, 16 generated xy
targets and execution of indices 1:9. Use the original 100-step DDPM initially.
Keep the image encoder frozen for the first comparison; train the whole action
U-Net or distributed LoRA, not only the output head. These are parameter choices,
separate from the learning rule. A flow Transformer or new sampler would require
its own loading checks and frozen-policy baseline.

For one exact physical context/history `h`, sample several full normalized plans
`u_i` from the frozen rollout actor. Evaluate their **executed** prefixes from
the same physical context, using independent policy noise. Store all 16 targets
as the generated sample; do not label an unexecuted tail as an observed future
trajectory. Shared-context comparisons require full prefix replay from a fresh
initial simulator space to reconstruct contacts. Restoring body positions alone
mid-contact is not a verified equivalent branch in our current environment.

Use discounted physical rewards, original success and terminal/truncation rules
from [the implemented protocol](POLICY_LEARNING.md). A score can be one block's
return or a longer return under a fixed continuation actor. Declare which: short
coverage gains are not the same objective as eventual success. A learned terminal
value is a later option with its own errors, not a free accurate future return.
Group centering compares alternative plans from the **same** context; mixing
different starting states in one group would also reward easier scenes.

For the first variant, freeze the rollout actor within each collection/update
round and refresh it from the trained actor before the next round. The paper's
moving-average refresh is a separate configurable comparison, not an invisible
default. Keep the initial T reference fixed throughout.

Map scores with `r_i = 0.5 + 0.5*clip((score_i-group_mean)/scale, -1, 1)`.
Fix and record the positive scale or its estimation rule; all tied scores should
give neutral `r_i=0.5`, without division by zero. For the DDPM noise schedule,
let `a_t = sqrt(alpha_bar_t)` and
`s_t = sqrt(1-alpha_bar_t)`. Draw fresh time/noise and form:

```text
noisy_plan = a_t * u_i + s_t * noise
f         = noise_predictor_theta(noisy_plan, t, h)
f_old     = frozen_rollout_predictor(noisy_plan, t, h)
delta     = f - f_old

positive_prediction = f_old + beta * delta
negative_prediction = f_old - beta * delta

loss = r_i       * squared_error(positive_prediction, noise)
     + (1-r_i)   * squared_error(negative_prediction, noise)
     + lambda_T  * frozen_T_retention_loss_on_separate_T_contexts
```

The optional T term matches the initial checkpoint's noise predictions on noisy
plans sampled from that checkpoint at automated T contexts. It adds no new human
demonstrations and must be compared with the same retention term in other methods.

This is a **proposed epsilon-prediction adaptation** of the published objective,
not a reproduction of its SD3.5 experiments. The paper permits general Gaussian
noising schedules, but weighting and finite sampling still matter. Clean-plan
prediction is `(noisy_plan - s_t*f)/a_t`; at fixed `t`, its squared error is
`s_t**2/a_t**2` times noise-prediction error. Therefore choosing a prediction space
changes time weighting unless corrected. Verify this mapping with the saved
scheduler; do not replace DDPM with a straight-line rectified flow by relabeling
the same network output. Keep rollout/reference outputs detached from gradients.

For one fixed training pair, write `e = f_old - noise`. Ignoring the optional T
term, expand the two branches:

```text
loss = ||e||² + 2*beta*(2*r_i-1)*dot(e,delta) + beta²*||delta||²
best delta in unrestricted prediction space
     = ((2*r_i-1)/beta) * (noise - f_old)
```

The positive quadratic coefficient avoids the unbounded objective obtained by
simply assigning a negative coefficient to squared error. A smaller `beta` means
a larger ideal displacement, not necessarily a more conservative update. This
algebra does not prove convergence of a neural network or physical improvement.
AdvantageFlow's alternative is an advantage-weighted error plus old/reference
prediction penalties; its quadratic coefficient must remain positive as well.

The proposed endpoint trainer would update the denoiser used throughout sampling.
Our implemented PPO instead updates selected stochastic steps and freezes the
last deterministic step. Report this difference and include a matched
parameter-use schedule ablation before attributing an outcome solely to the
objective. Never load an endpoint-trained delta through the tail-PPO wrapper and
assume it represents the same policy. Even solver-flexible objectives need
evaluation of the actual finite sampler used at deployment.

## A specific world-model extension to test

Question: **can reducing the strength of uncertain imagined feedback preserve
useful physical updates when contact changes across shapes?** First implement
the ordinary simulator-scored loss above. Then test a small modification:

```text
c_i in [0,1]: reliability weight for this candidate/group
r_safe = 0.5 + c_i * (r_i - 0.5)
use r_safe in the same positive/negative branch loss
```

At zero confidence this keeps an old-prediction anchor; at full confidence it
recovers the original loss. The expansion above gives the ideal displacement
`c_i * (2*r_i-1)/beta * (noise-f_old)`. Thus the modification reduces the reward
direction while retaining the positive quadratic coefficient. It is an
interpretable loss modification, **not** a theorem that the physical policy
improves, and we claim no novelty for confidence weighting.

Reliability must concern **errors in candidate return differences/ranking**,
including contacts and near-goal moves. Low pixel error or ensemble agreement
alone is insufficient. Estimate it from separate physical validation branches
under the same rollout actor, shapes and horizon; keep evaluation scenes out.
For example, test whether a predicted score difference exceeds a measured
return-difference error allowance. An empirical allowance is a heuristic unless
its statistical assumptions and coverage are established. Policy/model changes,
rare contacts and selecting only optimistic candidates can invalidate calibration.

For context, if every predicted candidate score differs from its true score by
at most `epsilon`, a predicted pairwise improvement exceeding `2*epsilon` has the
correct sign: the two errors can cancel at most that margin. This elementary
bound is conditional on a valid error bound. It does not certify our predictor,
nor does fitting the denoiser exactly realize the desired action distribution.
Fresh physical policy evaluation remains the deciding evidence.

Freeze the world model during each policy phase. Predicted scores must use its
predicted rewards/states and a checked termination interface; future simulator
labels belong only to training, validation or a separately labeled oracle control.
No gradient through physics or the world model is needed for this route. Training
the world model still uses supervised transitions; this actor loss trains the
diffusion policy. They are different learning jobs.

## Implementation order and experiments for three developers

These are proposed work boundaries, not assigned author names. Keep the existing
trainer usable and give the new objective a separate entry point/checkpoint type.

| Workstream | Concrete next work | Acceptance evidence |
| --- | --- | --- |
| Policy | Shared forward-noising utility; detached rollout/reference actor; DDPM-NFT loss; full-U-Net/LoRA updates; periodic checkpoints and standalone reload | Analytic loss/gradient checks, exact unchanged initial actions, GPU updates, unchanged frozen weights and reload parity; then physical learning curves |
| Data/world model | Branch collector with full-prefix replay; context/actor/model IDs; physically scored groups first; compare frozen/adapted predictors on the same candidate plans | Exact branch replay including contacts, disjoint trajectory/scene splits, per-shape return-difference/ranking errors and optimistic-action failures |
| Evaluation/integration | Frozen T/L scene lists, reward/timing checks, cost accounting and per-shape summaries; then lock a diverse shape set for expansion | Matched physical evaluation of frozen policy, tail-PPO and forward training; success/T retention, failure cases and costs across seeds |

Start with a bounded update/reload check, then a declared development budget
(the existing 20,000-control proposal is a starting point, not known sufficient).
Count **every** counterfactual physical branch, prefix replay and continuation;
separate training, validation and final evaluation. For four candidates with
eight controls each, one context costs up to 32 branch controls before replay
and any continuation. Batch scoring creates no free simulator data. Log plan
generation, network-update and world-model time separately.

The main sequence remains:

1. Frozen official policy, direct tail-PPO and direct DDPM-NFT. Use the same
   observation/reward/scene protocol and comparable parameter/control budgets.
   Measure denoising-update coverage differences explicitly.
2. Compare AdvantageFlow on the common physically scored data if objective
   stability or update cost makes it informative. A fresh method does not fix
   absent contact exploration; diagnose reward/coverage before scaling.
3. Once prediction passes action/reward checks, compare physical-scored versus
   model-scored forward training. Add reliability-weighted feedback as an
   ablation; also retain ordinary unweighted model feedback.
4. Expand one shared world model and policy to a fixed diverse development set,
   then more letters. Repeat key methods/seeds; reserve genuinely unseen geometry
   separately. A-Z training measures seen-shape competence, not unseen letters.

Report original success (**coverage > 0.95**), peak/final coverage, T retention,
interaction/compute/storage costs and seed variability. Post-success stability
remains a separate task. Include world-model collection/adaptation costs in the
comparison with cheap physics. Useful failures include no L improvement,
forgetting T, better predicted returns with worse physical returns, or a world
model that costs more than the simulator.

This specification now has a separate [physical forward trainer](FORWARD_POLICY_LEARNING.md).
The code implements both losses, matched branches and recovery without introducing
a new flow-policy architecture. **Imagined RL and reliable learned dynamics remain
unestablished.** Runtime evidence is in [status](STATUS.md) and [validation](VALIDATION.md).
