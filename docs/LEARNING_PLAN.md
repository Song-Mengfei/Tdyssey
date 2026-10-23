# Learning plan and developer handoff

Updated: 2026-10-23. This defines the shared experiments; read
[completed checks and results](STATUS.md) alongside it.
The first direct trainer and its precise variant, survey, derivations and commands
are now in [policy learning](POLICY_LEARNING.md). The ambitious learning stages
below are not established by a successful optimizer check.
The [extended RL research note](DIFFUSION_RL_RESEARCH.md) specifies a
DiffusionNFT alternative and its world-model extension. The
[forward training guide](FORWARD_POLICY_LEARNING.md) covers its implemented
physical NFT/AdvantageFlow path and recovery; imagined feedback remains pending.
DPPO below describes the
first implementation, not a requirement that all later learning use PPO. The
course project can implement and compare established methods without claiming
algorithmic novelty; its shared multi-shape scope remains unchanged.

## The idea in one minute

Start with the released T-trained visual Diffusion Policy. Let one shared policy
learn to push other letters using automated interaction and simulator-derived
rewards, without new human demonstrations. First establish learning on L while
checking T retention; then expand to mixed shapes and potentially A-Z.

A world model predicts what an action will do. A policy-learning method changes
the policy using rewards from trying actions; DPPO is the implemented starting
method. They are different models with different losses.
The reward specifies the task; an image predictor does not automatically supply
a reliable reward. Original expert T pretraining and privileged simulator labels
remain part of the method and must be disclosed.

After training, the intended deployed controller is the updated diffusion policy
acting on real observations. The world model is a training aid; running it for
test-time action planning is a separate comparison, not a hidden requirement for
the learned policy's success.

The next step is **two workstreams that can proceed independently**, followed by
a joint experiment:

```text
Existing T policy + official T/L simulator
  |
  +-- Physical practice --> reward + policy update --> directly trained policy
  |
  +-- Automated data --> assess/adapt world model --> pass prediction checks
                                                   |
                            imagined practice + same update method
                                                   |
                          test policy in physical simulator
```

The direct-simulator route is both a debugging check and a scientific baseline.
Implemented update choices are tail-PPO and the NFT/AdvantageFlow adaptations.
Hold the update method fixed when comparing simulator versus world-model feedback.
It must work before imagined policy improvement is claimed. World-model evaluation
need not wait for it. A negative learning result should be diagnosed and reported,
not hidden by moving to a more complex predictor.

## What is ready, and what must be built

| Component | Current state | Next deliverable |
| --- | --- | --- |
| Official visual CNN policy and T/L physics | CUDA inference and physical checks passed | Reuse this baseline and its saved normalizers |
| Alphabet/near-T environments | Frozen-policy development sweep completed | Extend the T/L training collector before mixed-shape collection |
| Automated letter collection and split-safe batches | Implemented; L pilot and tiny Y path check available locally | More varied, disjoint training/validation scenes and balanced shape exposure |
| Policy inspection | Actual denoising replay, near-miss analysis and 500-control T continuation completed | Review behavior; no training is provided by these tools |
| Custom 1.3M-parameter L predictor | Small supervised pilot; unreliable rewards/drift | Keep as a comparison, not the imagined environment |
| Pretrained 8.73M-parameter GPC predictor | Loading verified; official-domain/contact failures measured | Assess official-domain adaptation against frozen weights |
| Direct DPPO-style trainer and value network | Tail-transition PPO and Conv1d LoRA implemented; initial real GPU update/reload checks passed | Establish physical learning curves and validate full action-network training |
| Forward-process policy learning | NFT/AdvantageFlow, matched physical branches and recovery implemented | Diagnose physical learning curves, compare objectives/capacity, then validate learned-model feedback |
| Shared multi-shape world model / imagined trainer | **Not implemented** | Build only after the relevant interfaces and checks work |

The committed [result index](RESULTS.md) contains selected earlier evidence.
New decision-inspection outputs, raw datasets and weights remain ignored/local;
a hash does not make an artifact accessible to a collaborator. Reproduce locally
or arrange an external artifact location before depending on those files.

## Order of experiments

| Step | Work | Evidence required before advancing |
| --- | --- | --- |
| 1: agree on the pilot | Record T/L scene splits, reward, action timing, trainable weights and interaction/compute/storage caps | A versioned experiment configuration; no guessed successful result |
| 2A: physical learning | Implement and check direct-simulator DPPO on T/L | Correct probabilities/returns, actual CUDA updates, reloadable checkpoint; then separate-scene learning curves |
| 2B: prediction assessment | Compare compatible frozen predictors on the same T/L transitions and candidate actions | Contact/rotation, pusher, reward and action-ranking errors plus memory/latency |
| 3: model adaptation if needed | Adapt the best suitable pretrained model using automated training episodes | Improvement over its frozen version on untouched validation trajectories; no model-test tuning |
| 4: imagined learning | Freeze the validated world model; use short predicted rollouts for DPPO | Physical evaluation against the frozen policy and direct-simulator DPPO |
| 5: shared multi-shape learning | Fix a diverse shape set, extend collection, retrain/adapt shared dynamics and mix policy practice | Per-shape physical improvement, T retention, per-shape costs and failures |
| 6: alphabet and stretch tests | Expand to A-Z if affordable; reserve fresh geometry variants for unseen-shape tests | Separate new-scene competence from genuinely unseen-shape transfer |

Step 2A can establish a working update without showing improvement. A tiny update
smoke check is not a completed learning experiment. If direct learning fails,
check likelihoods, rewards, exploration and control timing before interpreting
imagined learning. Stop or revise the approach when evidence warrants it.

### Initial experiment sizes and settings

The following are **starting proposals to encode and check**, not implemented
trainer defaults, authorized jobs on another machine, or claims of sufficient data:

| Stage | Proposed bounded size | Purpose |
| --- | --- | --- |
| Data plumbing | One existing-style 8-episode, 100-control run per T/L; up to 1,600 controls total | Verify both domains, splits and actual storage/contact distribution |
| First dynamics comparison | Five separate 8-episode, 100-control runs per shape; up to 8,000 controls total | A larger starting pool without raising existing per-run collector caps |
| Policy update check | One rollout batch of 8 plans, at most 64 controls; one optimizer update | Check real gradients, changed weights, timing and serialization |
| Learning pilot | Up to 20,000 physical training controls total across T/L, 32 plans per batch, three PPO passes, minibatches of four plans | Detect whether learning moves in a useful direction; not a final paper-scale evaluation |
| Development evaluation | Ten disjoint scenes per shape, at most 300 controls each, at fixed checkpoints | Select settings without touching the reserved final scene set; account for these controls separately |
| Key final comparisons | At least three training seeds if affordable, same reserved physical evaluation scenes for every method | Report variability, per-shape outcomes and cost; decide scene count before training |

Use distinct run/reset seeds. Preserve each run's existing 6/1/1 episode split:
five runs give 30/5/5 train/validation/model-test episodes per shape. The catalog
must reject duplicate scenes or cross-split leakage. Adding runs requires a new
explicit catalog, not reassignment of old episodes. Training, model validation,
model test, policy development evaluation and final policy evaluation have
different purposes; their initial scenes must not leak into training.

The first 16 episodes contain about 42.6 MiB of uncompressed RGB alone at 96x96;
encoded size is data-dependent. Measure it before scaling. Ten runs have a combined
configured 320 MiB default encoded-data ceiling, not a guaranteed actual size or
a total-project storage budget. Source checkpoints, optimizer state and reports
need separate budgets. Keep uint8 frames once, avoid duplicate float datasets,
retain only selected checkpoints, and measure GPU memory before longer runs.

For the first reward/optimizer trial, propose `gamma=0.99`, `beta=1`, completion
bonus 1, PPO clip 0.2, actor learning rate 1e-5 and critic learning rate 3e-4.
These are unvalidated starting values; do not tune them on final evaluation.
Use a separately declared denoising discount/sampling-noise schedule and exact
trainable-block list after the sampler check; physical discount and denoising
discount are different. Record all changes and the reason. Reduce minibatch size
if measured memory requires it; do not silently change control/inference budgets.

## World model: what to test and when to train it

### Compatibility before model size

We are not committed to a sweep of every available model size. Compare a small
shortlist using identical input histories, action meaning, control rate, prediction
horizons and physical branches. A larger model is worth testing when its interface,
asset availability, reuse terms and resource cost fit the task. Parameter count or
convincing-looking video alone cannot establish action prediction accuracy.

The starting candidates are the already loaded GPC predictor and the custom CNN
comparison. [DINO-WM](https://arxiv.org/abs/2411.04983) is an alternative that
predicts pretrained visual features. Its pretrained image encoder is not itself
an action-dynamics model. Our RGB policy would need validated image reconstruction
or an explicitly compared representation change before consuming its rollouts.
Other larger pretrained candidates need the same compatibility review before
downloads; no additional model or environment is required by this PR.

The chosen imagined-environment interface must produce:

- Predicted RGB observations compatible with the original policy, and predicted
  pusher positions. Preserve the policy's two-frame history and checkpoint scaling;
  a predictor may maintain a separate, longer history.
- Action-dependent motion, including contact, rotation, no-contact stability,
  goal visibility and near-goal corrections.
- Predicted coverage/reward and success information, through validated heads or
  a validated decoder. The released GPC RGB predictor alone does not supply this
  full interface; its T-specific reward geometry cannot be assumed valid for L.

### Data and supervised model training

Collect automated play and frozen/current-policy interactions, including misses,
contacts, rotations, approach, near-goal states and recovery. Successful episodes
are useful but not required to learn dynamics. Pure random motion may miss the
important contacts. Object-directed exploration uses a geometric prior; disclose
it. Match the official physics and retain per-transition actions and labels.

The existing collector supports object-directed and random-waypoint play on T/L.
Policy-driven collection, deliberate near-goal/recovery sampling and mixed-letter
collection require extensions; the frozen-policy evaluation reports are not
automatically training datasets. Start with the existing modes, inspect missing
contact/state coverage, then implement the missing source rather than merely
increasing the number of similar trajectories.

Split by whole trajectory and initial scene, not adjacent frames. Fit all learned
normalization on training data only. Use validation for development, leaving the
final model-test split untouched until the declared comparison.

First test frozen pretrained weights. If measured failures require adaptation,
train on these automated episodes. The model loss depends on the architecture:
future image/feature prediction (or diffusion noise prediction), together with
pusher and coverage/reward/terminal prediction where those outputs are needed.
Auxiliary object motion/contact labels are allowed privileged supervision; they
must not silently become inputs to the RGB policy. Record loss weights and trainable
parts before running. Do not assume the custom CNN loss transfers unchanged to GPC.

The GPC wrapper currently loads and samples frozen weights. Adapting those weights
requires a supervised training path using the model's actual denoising objective;
calling its inference-only sampler is not that training path. Build and check
that separately, preserving a strict frozen-checkpoint comparison.

| Output being learned | Suitable supervised loss to specify for the chosen model |
| --- | --- |
| Future RGB/features | Image/feature prediction error; for a diffusion predictor, its architecture-specific noise/denoising objective |
| Pusher motion | Squared error on training-normalized displacement |
| Object motion, if used as auxiliary supervision | Translation error and wrapped/symmetry-aware rotation error with explicit units |
| Coverage/reward | Coverage or reward regression; inspect errors near success, not just the dataset average |
| Contact/success probability | Binary cross-entropy plus held-out calibration checks |

Define a weighted sum only after deciding which outputs the selected architecture
actually supplies. Weights, normalization and multi-step rollout length belong in
its versioned configuration. Log each component separately so background image
accuracy cannot hide a useless reward head. Train short autoregressive rollouts
using predicted history as well as one-step checks; evaluate error accumulation
without feeding true future frames back in.

Prefer adapting suitable pretrained components over starting a large model from
scratch. Compare before/after weights on the same validation data. One shared
image/action-conditioned model across shapes is the intended endpoint.

### What counts as accurate enough?

Test at horizons 1 and 4 controls initially, adding 8 before trusting a complete
original action chunk. The existing probes cover four controls; longer probes
are work to implement. Any shorter-prefix policy experiment needs a matched
physical baseline with the same replanning rate.

| Check | Required report |
| --- | --- |
| Motion | Pusher/object translation and symmetry-aware rotation error, separately for contact and no contact; compare persistence |
| Reward and success | Coverage error, threshold-crossing errors near 0.95 and calibration; a constant reward head must fail |
| Action choice | Predicted versus physical ranking of alternative action sequences from the same history |
| Optimistic errors | Physically execute actions the model rates highest; measure predicted gain minus actual gain |
| Multi-step stability | Drift, lost goal, inconsistent image/position/reward outputs and error growth |
| Resources | Weight/storage size, peak GPU memory, prediction latency and total experiment time |

Declare numerical pass/fail tolerances and the tested horizons before running the
gate; these tolerances are not yet fixed or passed. Choose them for the accuracy
needed to distinguish useful corrections near the goal, not merely lower image
error. Include held-out near-goal failures and report per-shape results. A model
may pass short rollouts and fail longer ones; use only the validated horizon.

True future simulator states are available to evaluate predictions, never to
secretly calculate imagined rewards. A visually plausible but physically wrong
prediction is a failure. Keep the world model frozen during each policy-learning
phase; later data collection or model refresh is a separate recorded cost.

### Do we train on all A-Z?

Not as the first experiment. Start with T/L, then fix a diverse development set
and a per-shape data budget. Collect across shapes before fitting shared dynamics;
the existing L-only pilot does not validate the alphabet. Expand to all letters
when the basic learning loop and costs are understood. A transferable frozen
model may need no new dynamics training for a shape, but that must be demonstrated.

Training on all letters and testing new starting scenes measures multi-shape
competence. For an unseen-shape claim, exclude that geometry from both world-model
data and policy updates. All existing alphabet shapes have already been explored;
reserve fresh geometry variants before using them as a stronger final test.

## Policy learning: reward, loss and what changes

### Data contract between components

| Record | Required information and boundary |
| --- | --- |
| Physical episode | `N+1` uint8 RGB frames for `N` executed actions; pusher/goal histories, absolute pixel targets, raw rewards/success, cutoff, coverage/IoU, contacts and full state provenance |
| Model training window | Past observations through time `t`, future actions starting at `t`, targets after those actions; no future frame among inputs; see [window timing](TRAINING_DATA.md) |
| Imagined transition | Model/version and initial-history provenance; chosen action, predicted next RGB/pusher/coverage, derived reward and predicted termination; no unseen true future labels |
| On-policy DPPO buffer | Current actor/version, exact conditioning observations, full sampled denoising chain, old conditional log probabilities, executed prefix length, per-control rewards, value estimates and terminal/truncation masks |
| Forward-process buffer | Implemented physical groups include exact context/prefix provenance, rollout round, full normalized sampled plans, executed prefixes, simulator scores and normalization; predicted scoring remains pending; see [guide](FORWARD_POLICY_LEARNING.md) |
| Evaluation record | Frozen method/checkpoint, scene/policy seeds, physical action/state/metric traces, costs and stop reason; no optimizer updates |

For the actor, preserve two 96x96 RGB frames, original cropping and pusher-history
normalization. Do not add object pose, letter identity or reward as actor inputs.
The initial value network should use the same observation information; a
privileged critic is a separate declared variant. The model may use privileged
labels during training without granting that state to the acting policy.

Existing offline play data trains dynamics. Standard on-policy PPO actor updates
require fresh rollouts from the current sampling policy and their old log
probabilities; do not feed arbitrary play windows into PPO as if they met that
requirement. A frozen model can generate those fresh rollouts even when its own
supervised training data is older.

### Reward is the task; loss is the weight-update rule

Use the official strict **coverage > 0.95** completion criterion. Coverage is
intersection area divided by target area. IoU divides by the union; log both,
but do not substitute one threshold for the other. Preserve upstream clipped
coverage reward in reports even when adding a separate training reward.

The proposed first training objective is sparse completion plus potential-based
progress shaping:

```text
success_bonus = 1 on the first successful transition, otherwise 0
potential = current goal coverage
training_reward = success_bonus
                  + beta * (gamma * next_potential - current_potential)
```

`gamma` discounts future physical-control rewards; `beta` scales the progress
signal. Record their values in the pilot configuration before training. At genuine
success termination, use zero terminal potential and no value bootstrap. A time
limit is recorded separately; for the continuing-task pilot, bootstrap from its
final observation and do not zero its potential merely because the runner resets.
Training masks must distinguish these cases. This follows
[potential-based reward shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf);
the theoretical policy-invariance conditions are not a promise that our neural
policy will converge or improve.

Moving from poor toward good alignment usually supplies progress signal; losing
alignment removes it. Do not clip away negative shaping terms. At zero overlap,
coverage offers little guidance: investigate exploration/contact coverage and,
if needed, a declared training-scene curriculum. A curriculum is a comparison
with disclosed geometric priors, not evidence of full-reset task success.

Do not use maximum-ever overlap as the training reward: it hides later loss.
Do not add arbitrary motion or angle penalties before testing the basic reward;
they can discourage necessary contact or penalize valid shape symmetries.

For **sustained alignment**, define a separate task before training: for example,
coverage > 0.95 for ten consecutive physical controls. Continue through the first
crossing, provide completion at the end of the streak, and record the streak in
the task state. Ten is an illustrative proposal, not a validated setting. Evaluate
the original benchmark and retention separately. A stop head is optional;
learning to hold or retreat can also be expressed through ordinary pusher targets.

In physical practice, simulator geometry supplies the reward. In imagined practice,
validated model predictions must supply it. Use the same reward definition on
both paths. Predicted rewards can be exploited by the policy, which is why physical
rechecks are required.

### DPPO and the two losses

[DPPO](https://arxiv.org/abs/2409.00588) applies policy-gradient learning to the
diffusion denoising decisions. Save sampled transitions and their old probabilities,
execute the final action prefix, and use future physical rewards to assign credit.
The starting noise, intermediate denoising samples and physical trajectory are
different things. A denoising iteration is not a control step.

The [PPO objective](https://arxiv.org/abs/1707.06347) can be written without raw
mathematical typesetting:

```text
advantage = estimated future return - value network's prediction
ratio = updated transition probability / recorded old transition probability
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1-epsilon, 1+epsilon) * advantage))
value_loss = mean((predicted_return - return_target)^2)
```

For DPPO, probabilities refer to recorded conditional denoising transitions, not
a guessed density of the final action chunk. Better-than-expected outcomes increase
their probability; worse outcomes reduce it. Clipping discourages large changes
but is not a guarantee of improvement. The value network predicts future reward;
it is not the world model. Original supervised noise-prediction loss alone cannot
provide reward-based adaptation without suitable target actions.

Start with limited trainable action-network blocks and a frozen image encoder;
record exact names/counts. This is separate from choosing how many denoising
iterations use the updated network. Compare an adapter/full fine-tuning later with
matched budgets. Mix T practice, or test a frozen-T-teacher retention term on
automatically collected T observations. Such a term needs its own weight/comparison.
Residual PPO is a useful later corrective-controller comparison
([ResiP](https://residual-assembly.github.io/)); it leaves the base weights frozen.

Correct implementation requires a stochastic training sampler and checked Gaussian
log probabilities, including variance floors/last-step handling. Simply enabling
gradients on the inference wrapper is insufficient. Preserve upstream normalization
and the executed prefix (indices 1:9 of 16 predicted targets). A sampler change
must have a separate matched baseline. PPO does not require differentiating
through simulator or world model, so an inference-only world-model sampler is not
by itself a blocker to this likelihood-ratio learning route.

Sum discounted rewards from every actually executed physical control in a chunk.
After `m` controls, the bootstrap discount is `gamma**m`; handle early success and
partial final chunks. Do not use peak chunk reward or count 100 denoising steps
as 100 interactions. Return estimation should credit useful repositioning even
when its immediate reward is low.

### How a physical training iteration works

```text
load the pinned T checkpoint; select and record trainable parameters
initialize a value network and independent training/evaluation RNG streams
repeat within control, update, time and storage budgets:
    reset only when needed using training scenes; keep the real history
    sample action chunks; save the denoising chain and old log probabilities
    execute indices 1:9, stopping early only for declared termination/budget
    record each real reward, next observation and actual number of controls
    estimate discounted returns and advantages, with correct cutoff bootstrap
    run bounded PPO actor and value updates on this fresh rollout batch
    log gradients, clipping/KL diagnostics, return and per-shape counts
    at fixed checkpoints, run separate physical development evaluation
save selected weights/config/provenance; verify loading them reproduces inference
```

During the update check, verify ratios are approximately one before optimization,
Gaussian log probabilities agree with an independent calculation, and current
versus old policies use identical sampling conventions. Check terminal success,
timeout, partial chunk and mixed episode boundaries with explicit return examples.
Nonzero loss alone does not prove a weight update. Do not count stochastic rollout
variation as learning; compare frozen and updated checkpoints under fixed seeds.

### How an imagined training iteration differs

Start from complete observed histories sampled from training episodes. Keep the
world model frozen, and use the current policy to choose actions. Advance the
model one physical control at a time; its predictions supply subsequent history,
coverage and rewards. Use short validated horizons initially, then bootstrap the
value at an artificial rollout cutoff rather than treating it as genuine success
or failure. Store model-based rollouts separately from physical ones.

Apply the same checked DPPO update. Periodically evaluate the resulting policy
in the real simulator. If predicted gains do not transfer, shorten the rollout,
diagnose reward/contact errors and collect new training data as a separately
recorded model-refresh stage. Discard/recompute policy buffers as needed after a
policy/model change; do not pretend stale samples remain fresh on-policy data.
Uncertainty-based rollout stopping is optional future work, not an implemented
safeguard. No gradient through the world model is required for this PPO route.

Repeated practice can make useful approach/contact/rotation/finishing sequences
more likely, including sequences absent from human adaptation data. It cannot
guarantee their discovery: failed exploration or an inaccurate reward/model can
prevent improvement. Mixed-shape practice and explicit T checks address forgetting,
not a separate policy per letter.

## Three developer work packages that can start now

These are proposed boundaries, not assigned author names. Agree on owners before
simultaneous edits. Each workstream uses a branch; coordinate edits to common
contracts/configuration and let one owner update status/decisions per milestone.
For the next method comparison, the [research handoff](DIFFUSION_RL_RESEARCH.md)
adds concrete forward-trainer, matched-branch collection and evaluation tasks to
these same three workstreams; it does not assign extra developers or require a
new policy architecture.

| Workstream | First build task and suggested file ownership | Acceptance evidence |
| --- | --- | --- |
| Policy learning | Extend existing `tdyssey/dppo.py`, `tdyssey/rl_protocol.py`, `tdyssey/policy_training.py` and `scripts/train_official_policy.py`; start from checked tail-transition PPO | Establish physical learning curves; validate full-network mode, periodic checkpoints/evaluation and a standalone trained-delta evaluator before claiming reliable adaptation |
| Data and world model | Existing `tdyssey/gpc_world_model.py`, world-model collection/training/probe scripts; start with a versioned T/L comparison protocol and frozen/adapted GPC interface assessment | Whole-episode/scene separation; action normalization verified; contact/reward/ranking failures preserved; reproducible resource/error table |
| Evaluation and integration | Official evaluator/metrics, baseline scene lists, experiment configuration and result summaries; coordinate any edits to `tdyssey/official.py` with policy owner | Frozen checkpoint baseline; disjoint learning/evaluation scenes; no accidental reward/success change; original versus learned physical rollouts and interaction counts |

The first policy training script now exists and accepts source/checkpoint paths,
config and a new output directory, with control/update/time/storage limits in
the configuration. See [actual commands and limitations](POLICY_LEARNING.md).
The world-model module remains an inference wrapper; extending it does not make
pretrained-model adaptation or imagined training ready. Existing dynamics-training
commands still train a different model from the acting diffusion policy.

The shared experiment configuration must record: checkpoint/source hashes,
shape exposure, training/validation/evaluation scene IDs, observation/action
conventions, reward and discount, terminal/truncation rules, trainable parameters,
sampling schedule, RNG seeds, update/control/compute/storage budgets, model-selection
rule and output provenance. Exclude known evaluation scenes from training. No
full-alphabet collection or unbounded training is needed to begin these packages.

### Existing commands

Source checks need no installation:

```bash
python3 -B scripts/check_repo.py
python3 -B -m unittest discover -s tests -p 'test_*.py'
```

With an already authorized [CUDA/DP environment](ENVIRONMENT.md) and official assets:

```bash
.venv/bin/python -B scripts/doctor.py --runtime --require-cuda --dp
.venv/bin/python -B scripts/evaluate_official.py --episodes 1 --steps 300 --output artifacts/official-baseline-handoff
```

Choose a new output path if it exists. Use [official collection](OFFICIAL_COLLECTION.md)
for real T/L datasets, [world-model commands](WORLD_MODEL.md) for the existing
supervised pilot and prediction probes, and [inspection commands](POLICY_DIAGNOSTICS.md)
for decision replay. The raw datasets/reports referenced by those examples may
need to be collected or transferred first. Existing world-model training commands
train dynamics, **not** the diffusion policy.

## What the final comparison should answer

### Why use a learned model when physical simulation exists?

Both can provide transitions for RL. The physical simulator applies programmed
collision/contact dynamics; the learned model estimates outcomes from data and
can accumulate errors. A world model may enable batched practice or reduce the
number of additional environment interactions, but it is not inherently faster,
more accurate or necessary. Its benefit in expensive real-world experiments does
not automatically transfer to this cheap 2D simulator.

The core question is whether world-model-assisted adaptation provides a useful
accuracy/data/compute tradeoff **over direct simulator learning**, not whether
it can replace a simulator that already works. Treat simulator-only RL as a full
research comparison, not a disposable setup check. If it wins, report that outcome
and identify model bias, contact errors, reward errors or overhead as supported
explanations. Such a controlled negative result remains a substantive deliverable.

### Comparison protocol

Compare the frozen T policy, direct-simulator DPPO and world-model-assisted DPPO
on the same physical evaluation scenes. Keep any ranking-only planner/residual
controller clearly identified. Report per-shape success, final/peak coverage and
IoU, retention where applicable, T forgetting, multiple seeds and failure cases.

Count all physical collection and policy interactions, including data used to
adapt the world model, separately from validation/replay and imagined transitions.
Compare data efficiency and wall time; a learned model can cost more than this
small physical simulator. Report both an end-to-end budget comparison and any
conditional comparison that reuses a shared model/data pool. Never call model
training data free. Better imagined reward is not evidence of physical improvement.
