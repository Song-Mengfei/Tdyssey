# Train the official policy with forward-process RL

The code implements **DDPM epsilon-prediction adaptations** of DiffusionNFT and
AdvantageFlow, not image-benchmark reproductions or new algorithms. It preserves
the official CNN, RGB/pusher history, normalizers, 100-step DDPM and execution
of targets 1:9. See the [research/loss explanation](DIFFUSION_RL_RESEARCH.md) and
[whole-project plan](LEARNING_PLAN.md). The shared multi-shape scope is unchanged.

The [tail-PPO trainer](POLICY_LEARNING.md) remains available. Forward training
updates the denoiser used at **all** steps, including deterministic t=0; tail-PPO
uses a frozen/updated timestep schedule. Report that difference in comparisons
and load each checkpoint through its own actor class.

## Components and parameters

| Component | Code |
| --- | --- |
| Configuration and within-context feedback | `tdyssey/forward_protocol.py` |
| Original sampler, NFT/AdvantageFlow loss, frozen rollout/reference actors | `tdyssey/forward_policy.py` |
| Full-prefix replay and matched physical trials | `tdyssey/policy_branches.py` |
| Bounded training, periodic checkpoints and source-checked resume | `tdyssey/forward_training.py` |
| Flushed metrics, optional W&B and fixed diagnostic | `tdyssey/tracking.py`, `tdyssey/denoising_probe.py` |
| Commands | `scripts/train_forward_policy.py`, `scripts/evaluate_trained_policy.py` |

The image encoder stays frozen. Distributed LoRA affects ordinary Conv1d kernels
throughout the action U-Net; conditioning linears, transposed convolutions and
normalization stay fixed. `trainable="action_network"` updates the entire action
U-Net. Dense training requires a separate memory/runtime check. The required
[CUDA + official DP environment](ENVIRONMENT.md) already supplies all dependencies.

## Run a check or pilot

Use an existing authorized environment. These commands do not install packages.
Training requires CUDA and runs headless; previews are separate evaluation jobs.
For new machines, longer comparisons and live charts, use the
[training-machine handoff](TRAINING_MACHINE.md). Tracking defaults to local
`metrics.jsonl`; optional `--tracking offline` or `online` mirrors scalars to W&B.

```bash
.venv/bin/python -B scripts/train_forward_policy.py --config configs/nft_smoke.json --output artifacts/nft-smoke-001
.venv/bin/python -B scripts/train_forward_policy.py --config configs/advantage_flow_smoke.json --output artifacts/advantage-flow-smoke-001
.venv/bin/python -B scripts/train_forward_policy.py --config configs/nft_local_pilot.json --output artifacts/nft-local-pilot-001
```

| Profile | Purpose |
| --- | --- |
| `nft_smoke.json`, `advantage_flow_smoke.json` | One update, two contexts, two candidates each, short T/L evaluation; mechanics only |
| `nft_local_pilot.json` | Distributed LoRA, up to eight updates / 4096 training controls, two-chunk scores and policy-only context warmup |
| `nft_pilot.json`, `advantage_flow_pilot.json` | Matched proposed T/L budgets: 20,000 controls, three evaluation scenes, four-chunk scores |
| `nft_full_network_smoke.json`, `nft_full_network.json` | Dense action-U-Net comparison; check the smoke profile on a suitable authorized GPU first |
| `nft_multishape.json` | Explicit T/L/I/Y/near-T development set after the T/L check; not an unseen-shape test |
| `advantage_flow_resume_check.json` | Extend the matching smoke profile to a second update in a new output directory |
| `tracking_smoke.json` | Check fixed probe, periodic evaluation, trace pruning, best/last checkpoints and tracking |
| `nft_extended.json`, `nft_full_network_extended.json` | Unexecuted longer T/L presets for a checked training machine; five development scenes, periodic evaluation and explicit larger caps |

Budgets are starting experiments, not known sufficient data. Dense profiles allow
up to 12 GiB of output; LoRA uses smaller checkpoints. Temporary atomic checkpoint
replacement is included in storage checks. Configurations do not authorize jobs
on another author's machine.

## Data, rewards and fair comparisons

Sample each candidate group from one frozen rollout actor at one exact context.
Restore the **initial** snapshot and replay all earlier actions before each trial.
Check physical state, contacts, rewards and conditioning images/history. Body
positions alone cannot restore a warm contact solver. Optional warmup follows
the policy; it is not a privileged near-goal reset. Advance along the last IID
candidate independently of reward.

Execute up to eight controls from the candidate and optionally continue with the
same frozen actor for `score_chunks-1` plans. Scores are discounted observed
completion-plus-potential rewards. Genuine success zeros terminal potential;
scoring cutoffs do not. This is a **finite scoring horizon without a value
bootstrap**, unlike PPO/GAE. The sampled 16-target plan is the learning example;
its unexecuted tail is not labeled as an observed physical future.

Center scores within their context, divide by the fixed positive `score_scale`
and clip to [-1,1]. NFT maps this to [0,1]; AdvantageFlow adds an old-prediction
anchor greater than the largest absolute signed weight. Ties are exactly neutral.
Zero-gradient batches are recorded rather than presented as successful learning.

All branch, warmup and replay controls count toward `max_train_controls`. Admit
only complete equal-horizon groups, so a run may stop below its cap. Evaluation,
reference queries, denoising and optimizer work have separate counts. The initial
T teacher supplies its own sampled plans at automated T contexts, with no new
human demonstrations. Its weights stay fixed. The rollout actor refreshes after
each update; `rollout_ema=0` makes a hard copy. RGB/pusher histories remain the
only actor inputs; physical state and geometric rewards are privileged feedback.

## Batched lock-step rollouts

Each plan is 100 sequential denoiser calls; at batch 1 they dominate wall time
while most GPU memory sits idle. Set `"batched_rollouts": true` (see
`configs/nft_pilot_batched.json` and D033) to sample a whole candidate group, and
every evaluation sweep, with one denoiser call per step. Each candidate gets its
own fully replayed environment and RNG stream; physics stays sequential on CPU.
The protocol is otherwise unchanged, but random streams differ from sequential
runs, so compare methods within one mode and do not change the flag on resume.
Larger `group_size` then costs little extra sampling time; its replays and
trials still count toward `max_train_controls`.

## Recovery, evaluation and video

`last.pt` stores trainable/rollout weights, Adam, three RNG states and completed
update state. LoRA saves factors, not the original base weights. The pinned base
checkpoint remains required. `report.json` stores source/configuration, episode
prefixes, scalar update summaries, the most recent detailed candidate traces,
physical evaluations and costs. `report_detail_updates` bounds detailed update
retention; active prefixes needed for contact replay are preserved. Keep raw outputs out of Git.

`eval_every=0` retains the original before/after checks; a positive value adds
physical development checks every specified number of updates, counted separately
from training controls. `best.pt` selects trained checkpoints by macro shape
success, then final coverage; it need not beat the frozen baseline. Fixed noise
probes reuse baseline-generated plans/noise and never feed gradients or selection.

Copy the original configuration and increase its update/control caps to extend:

```bash
.venv/bin/python -B scripts/train_forward_policy.py --config PATH/extended-config.json --resume artifacts/nft-smoke-001/last.pt --output artifacts/nft-resumed-001
```

Resume always writes a new directory. Only update/control/time/storage caps may
change; source hashes and other protocol settings must match. Roll back unfinished
updates to the saved boundary. Copy matching `best.pt` beside `last.pt` to preserve
the lineage's selected checkpoint. Each invocation has a separate tracking run.
Available parent reports charge discarded physical
work too. A hard kill may leave unreported work; the cost audit records this
uncertainty. Interrupted reports count completed plans/controls and may omit a
partial model/physics call. Keep parent reports for lineage. RNG is restored exactly; later CUDA
backward reductions need not reproduce every final-roundoff bit of an update.

```bash
.venv/bin/python -B scripts/evaluate_trained_policy.py --adapter artifacts/nft-smoke-001/last.pt --shape L --steps 300 --display --window-size 768 --output artifacts/nft-preview-001.json
.venv/bin/python -B scripts/evaluate_trained_policy.py --adapter artifacts/nft-smoke-001/last.pt --shape L --steps 32 --reference-report artifacts/nft-smoke-001/report.json --video artifacts/nft-preview-001.mp4 --output artifacts/nft-video-check-001.json
```

Reference checks compare actions, reward, overlap and termination with the saved
final evaluation; use the same scene, horizon and checkpoint. Default seed: 100000.
MP4 output is optional and bounded and needs no desktop window. Presentation
frames render physics at 512x512 while policy observations remain 96x96.

## Multi-shape data and the remaining world-model boundary

The official play collector accepts all implemented letters and `T_narrow`:

```bash
.venv/bin/python -B scripts/collect_official_play.py --shape Y --episodes 8 --steps 100 --output artifacts/play-Y-001
.venv/bin/python -B scripts/prepare_training_data.py --datasets artifacts/play-L-001 artifacts/play-Y-001 --output artifacts/mixed-play-001/catalog.json
```

Follow [official collection](OFFICIAL_COLLECTION.md) for seeds and frozen splits.
Visual-dynamics windows now derive labels/masks from each episode's geometry,
without a letter ID in model inputs. The custom CNN trainer can consume mixed
catalogs. This does not fine-tune the released GPC model, whose wrapper remains
inference-only, or establish reliable shared predictions. The existing custom
model's physical candidate-probe script remains L-specific; shared-shape contact
evaluation needs checked per-shape probes. No all-letter dataset
was collected by adding this support.

The forward CLI accepts **physical simulator scores only**. Reliability weighting
exists in the NFT loss for later integration, but imagined learning still needs
a validated predicted observation/reward/termination interface and physical policy
checks. True simulator futures cannot be passed off as model predictions. RAM/QAM
and a new flow architecture remain later comparisons. See [validation](VALIDATION.md)
and [status](STATUS.md) for actual evidence; a GPU update is not shape adaptation.
