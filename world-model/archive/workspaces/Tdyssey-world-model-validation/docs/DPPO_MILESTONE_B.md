# Direct-simulator DPPO milestone B

Implemented 2026-10-23 on `codex/dppo-milestone-a`, building on milestone A commit
`32d13dab29ad029ae3f19b3c4926dffa158c502c`. The branch starts from fetched
`origin/main` `41d99968dc64244f2381a148e74723261db92328` and retains the scoped
[Windows compatibility](ENVIRONMENT.md) used by [milestone A](DPPO_MILESTONE_A.md).

This adds an online T/L fragment collector, visual critic, one actor/critic update,
incremental checkpoint and a separate-process reload check. It is a bounded
implementation check, not a learning-curve experiment. Simulator coverage/reward
remain privileged labels; the initial policy retains expert T pretraining.
The updated policy has not been evaluated through additional physical rollouts.

## Implementation and numerical contract

| File | Interface and responsibility |
| --- | --- |
| [policy_rollout.py](../tdyssey/policy_rollout.py) | `HardBudget`, `collect_fragment`, `RolloutPlan`: persistent attempt accounting and original four-tuple simulator interaction |
| [policy_training.py](../tdyssey/policy_training.py) | `VisualCritic`, `prepare_batch`, `UpdateConfig`, `make_optimizers`, `update_once`: fixed targets and one weighted update |
| [policy_checkpoint.py](../tdyssey/policy_checkpoint.py) | `save_update_checkpoint`, `load_update_checkpoint`, `restore_optimizers`, `sample_to_payload`, `sample_from_payload`: CPU deltas and safe record reconstruction |
| [train_official_policy.py](../scripts/train_official_policy.py) | `run`, `resume-update`, `verify`: bounded orchestration and reports |

The actor keeps the official two-frame RGB/pusher interface, saved normalizers,
84x84 fixed crop and 16 predicted position targets, executing indices 1:9.
Only `model.final_conv.` trains: 1,313,282 parameters in six tensors. Encoder,
normalizer and other actor parameters/buffers stay frozen. The same network
serves all 100 denoising times. No supervised `compute_loss`, world model,
privileged object-state policy input or extra action search is involved.

The critic is a `132 -> 128 -> 1` MLP with Tanh and 17,153 parameters in the
observed checkpoint configuration. Its input is the actor's detached, frozen
visual/pusher encoding; shape and reset IDs are logging metadata. It does not
share trainable weights with the actor. `VisualCritic.config` saves both widths.

Each `RolloutPlan` retains the complete detached CPU `SampledChain`, start feature
and old value, actually executed control rewards/actions, coverage and upstream
reward, before/after image/pusher observations, final history and pre-reset final
value. Plans stop at strict success, timeout or an explicit buffer/plan cap.
Reset settling is counted separately, without hidden retry resets. T and L are
separate ordered fragments; no GAE continuation crosses their boundary.

`prepare_batch` freezes GAE targets using physical gamma=.99 and plan-boundary
lambda=.95, then normalizes actor advantages once over the complete batch.
Critic targets are raw GAE plus old value. First-success bonus and coverage
potential shaping retain [milestone A's terminal/bootstrap rules](DPPO_MILESTONE_A.md).
For a short executed prefix, rewards and bootstrap use its actual length m,
including `gamma**m`; 100 denoising times are never 100 physical interactions.

`update_once` first checks fixed-record probabilities (maximum absolute FP32
log-density difference <=1e-4) and old critic values (absolute tolerance 1e-5).
It accumulates every plan/time entry, using each subset's summed loss divided
by the full `B*100` transition count. Defaults are one plan and one transition
per microbatch. The checked joint 32-dimensional Gaussian PPO objective uses
clip=.2, denoising discount=1 and the explicit untuned std floor=.01 at all
times, including t=0. These probability conventions are inherited from A and
differ from public DPPO's aggregation heuristics.

The actor and critic use separate Adam optimizers, learning rates 1e-5 and 3e-4,
zero weight decay, and gradient norm clipping at 1. Both optimizer reservations
must succeed before either step. Reports contain finite pre-clipping gradient
norms, actual changed parameter names/norms, ratio/KL/clip diagnostics, and exact
frozen-state fingerprints before/after. Only one real step per optimizer is
allowed; a nonzero loss alone is not the acceptance criterion.

## Hard budget and recovery

The limits are 64 physical controls, eight sampled plans **including both
inference-only replays**, one actor optimizer step and one critic optimizer step.
The actual command requests 48 controls: T and L each receive at most 24 controls
and three sampled plans. Six rollout plans leave one reference replay and one
fresh-process replay. Early termination may consume fewer controls/plans; the
script never fills unused capacity by launching another episode.

`HardBudget` persists `budget.json` before each reset, sampling plan, physical
control or optimizer operation. Reserved attempts and completed operations are
separate counters. Failure never refunds a reservation. JSON publication is atomic
and file-fsynced; an exclusive sidecar lock prevents concurrent owners. A process
crash leaves its lock in place and fails closed. These records are conservative
attempt accounting, not a transaction with the physics engine.

Keep the same run directory and ledger for collection, update and verification.
Never delete/reset its ledger or create a replacement budget to repeat this
authorized job. `run` refuses existing attempts. `resume-update` reuses saved
rollouts without new environment controls or sampling, checks base/run provenance
and existing target equality, and is allowed only before any optimizer reservation.
It cannot recover an already-consumed optimizer operation by stepping again.
No mode claims to restore a physical episode, body/contact cache or simulator RNG.

## Saved artifacts and reload

| Local artifact | Contents |
| --- | --- |
| `budget.json` | Durable reservations, completions, reset settling and per-shape counters |
| `rollout.pt` | CPU fragments, initial critic, exact conditioning/chains/rewards/final observations, base identity, run configuration and post-collection local generator state |
| `targets.pt` | Frozen features, old values, raw/normalized advantages and value targets |
| `update.pt` | Selected actor parameters, critic, complete optimizer state/groups, official/sampler/base identities and RNG state |
| `probe.pt` | Original conditioning/chain, updated probabilities/critic value, reference inference chain and its final local RNG state |
| `report.json` | Status, configuration, real counts, update evidence, synchronized timings, CUDA allocated/reserved peaks and storage |

Tensor files use primitive/tensor payloads and `torch.load(..., weights_only=True)`.
New tensor/checkpoint paths refuse overwrite. Actor weights alone occupy about
5 MiB; complete Adam moments and critic state add to the actual checkpoint size.
Frozen actor weights, encoder and normalizer are reconstructed from the same
verified base rather than copied into each update artifact or a second GPU actor.

`load_update_checkpoint` checks schema, recorded base byte count/SHA-256, official
source/configuration, sampler identity and exact selected names/counts/shapes/dtypes.
It rebuilds one official actor, applies only the selected delta and loads the
critic strictly. `restore_optimizers` checks optimizer type and parameter-group
name order before restoring full states. Checkpoints retain torch CPU/CUDA,
local-generator, Python and NumPy RNG states; NumPy keys are encoded as tensors
for safe loading. Cross CPU/CUDA generator algorithms cannot promise replay:
`restore_rng=True` rejects that mismatch; otherwise the returned local generator
is None while its original state remains in the payload.

The checkpoint captures RNG immediately before the nonphysical reference sample.
`verify` requires a different process ID, reloads both optimizers without a step,
checks their step counters equal one, and compares updated probabilities, critic
value and frozen-state hash. It then reserves one fresh-process replay plan
(the eighth in this run) and compares the
entire sampled chain, targets, executed action prefix, log probabilities and
post-sample local generator state with zero tolerance. This is inference replay,
not replaying contacts or collecting evaluation rewards.

## Commands and current evidence

Use existing authorized interpreter/assets. There is no environment sync, asset
download or implicit `uv run`. The two commands below show the recorded job's
configuration; that run directory is already consumed and must not be rerun.
Another physical job needs its own authorized budget.

```powershell
$python = 'path/to/existing/.venv/Scripts/python.exe'
$source = 'path/to/existing/third_party/diffusion_policy'
$base = 'path/to/existing/checkpoints/official-dp/ema_policy.pt'
$runDir = 'artifacts/milestone-b-20261023'
New-Item -ItemType Directory -Force artifacts/test-temp | Out-Null
$env:TEMP = (Resolve-Path artifacts/test-temp).Path
$env:TMP = $env:TEMP
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:HF_HOME = Join-Path (Get-Location) 'artifacts/hf-cache'
$env:HF_HUB_OFFLINE = '1'
$env:SDL_VIDEODRIVER = 'dummy'
$env:SDL_AUDIODRIVER = 'dummy'
$env:PYGAME_HIDE_SUPPORT_PROMPT = '1'
$commonArgs = @('--source-root', $source, '--base-checkpoint', $base,
    '--base-sha256', '37d661ce6a487905fa4ff81ab7d65bf1b56b1c06adf480a961f65338fa0e38a0',
    '--run-dir', $runDir, '--device', 'cuda', '--controls', '48',
    '--max-controls', '64', '--max-plans', '8', '--threads', '4',
    '--seed', '5478', '--t-scene', '3005478', '--l-scene', '3005479')
& $python -B scripts/train_official_policy.py run @commonArgs
# A second Python process, only after run succeeds; no physical controls.
& $python -B scripts/train_official_policy.py verify @commonArgs
```

The existing local compact base has 1,095,708,090 bytes and the digest above.
Its digest differs from the manifest compact file; explicit pinning does not
establish released-EMA tensor equality. Configuration/provenance and strict
loading checks remain in force. All update/reload comparisons refer to this
same local artifact, preserving [A's identity limitation](DPPO_MILESTONE_A.md).

The completed update/reference-replay report recorded:

| Check | Completed observation |
| --- | --- |
| Physical collection | T: 24 controls / 3 plans; L: 24 controls / 3 plans; 480 control physics substeps plus two reset-settling substeps |
| Fragment boundaries | Each shape: continue, continue, buffer cutoff; no success/timeout in these short fragments |
| Pre-update probability check | 600 transitions; maximum log-probability error 0; all ratios 1 |
| Real optimizer steps | Actor 1, critic 1 |
| Actor gradient/change | Gradient norm 2.3561713; all six selected tensors changed; delta L2 .01129891, max absolute 1.00136e-5 |
| Critic gradient/change | Gradient norm .03348601; all four tensors changed; delta L2 .02615701, max absolute 3.00001e-4 |
| Frozen actor state | Exact before/after fingerprints equal |
| Post-update fixed-buffer ratios | Min .9070333, max 1.0361412; sample approximate KL 3.55941e-5; clip fraction 0 |
| Incremental checkpoint | 16,002,578 bytes; SHA-256 `6e89994dce1dfc31f8c24469850327ddd85aa2fc2e064788a64ac04e4d010704` |
| Inference-only reference replay | One plan completed, zero physical controls |
| Fresh-process reload/replay | Passed in PID 9472 (update PID 7984); chain/actions/targets/log-prob/critic value/local RNG bitwise equal; both Adam states at step 1 |
| Final permanent ledger | 48 controls / 8 plans / actor step 1 / critic step 1, all reservations completed |

The first fresh-process load rejected the scheduler signature before any plan
reservation or physical control. Diffusers' private `_use_default_values` list
contained the same names in a different process-dependent order. The fix
canonicalizes only that metadata list, including legacy saved signatures; it
does not change DDPM coefficients, transition probabilities or saved actor
weights. On retry the CLI retains the failed verification in
`verification_failures`, rather than presenting a first-attempt pass. At that point the ledger remained at
48 controls, seven plans and exactly one actor/critic optimizer step. The fixed
loader then passed in another process, consuming the eighth plan for its only
inference replay and no additional physical controls or optimizer steps. Two
regressions check legacy shuffled metadata and independent Python hash seeds;
real beta configuration changes are still rejected.

Synchronized phase timings and CUDA peaks (MiB = 2**20 bytes):

| Phase | Seconds | Peak allocated MiB | Peak reserved MiB |
| --- | --- | --- | --- |
| Original load | 6.1382 | 1,006.579 | 1,030 |
| T/L rollout | 10.4431 | 1,095.773 | 1,198 |
| Before-update checks | 12.0111 | 1,095.182 | 1,198 |
| Actor/critic backward | 12.6546 | 1,100.193 | 1,198 |
| First actor and critic optimizer steps, combined | .0611 | 1,043.177 | 1,198 |
| After-update checks | 12.7296 | 1,118.776 | 1,198 |
| Save | .0461 | 1,038.058 | 1,198 |
| Reference inference replay | 1.4097 | 1,118.914 | 1,198 |
| Successful new-process load | 5.8738 | 1,016.731 | 1,030 |
| Reload probability/frozen-state checks | 3.0512 | 1,105.382 | 1,198 |
| New-process inference replay | 1.2789 | 1,105.740 | 1,198 |

These peaks are allocator observations for this small check, not projected full
trainer memory. The raw report includes end-of-phase allocated/reserved bytes.
The eight sampled chains contain 800 actual denoising transitions; 600 recorded
rollout transitions each receive one actor backward pass. Probability diagnostics
reuse these fixed endpoints and are separately accounted for in the report.
The delta is 15.26 MiB; all persistent run files occupy about 28.75 MiB, including
the 12.97 MiB rollout buffer and .48 MiB probe. No environment or full model was
copied. Exact machine-specific sizes/hashes are in the ignored report.

These are observations on Windows AMD64, Python 3.12.3, torch 2.5.1+cu124,
diffusers 0.31.0 and RTX 3060 Ti, with four CPU threads. Final/peak coverage from
the pre-update fragments is not an updated-policy result. No physical learning
curve, T retention evaluation, multiple-seed improvement or unseen-shape transfer
has been established. Ignored local artifacts must be transferred or reproduced
under a new authorization before another collaborator can depend on them.

Focused checks use tiny models or fake environments. They include persistent
attempt accounting, failed-operation/partial-chunk boundaries, whole-batch target
normalization, unequal microbatch equivalence, stale-buffer rejection, zero
advantages, optimizer group/state restoration and tensor-only RNG/chain round trips.
CLI help/invalid-budget checks run with `-S`, without loading research packages.
Closeout discovery ran 160 tests: installed environment 114 passed/46 skipped
(26.893s); no-site-packages 76 passed/84 skipped (.875s). Focused B coverage is
10 fake collector/budget tests, 8 toy-update tests, 6 checkpoint tests and 5 CLI
tests. The likelihood suite now has 14 tests including the two signature
regressions; the independent reward/GAE suite has 9. Real simulator runtime flags
were disabled in discovery so these regressions consumed no extra physical
controls. The explicit real update/reload evidence above is separate from skips.
Repository source, diff and message checks passed. The spent ledger remains at
48 controls, eight plans and one optimizer step each.
For example, these commands require no real policy asset or simulator rollout:

```powershell
& $python -S -B -m unittest discover -s tests -p test_policy_training_cli.py -v
& $python -B -m unittest discover -s tests -p test_policy_checkpoint.py -v
& $python -B -m unittest discover -s tests -p test_policy_rollout.py -v
& $python -B -m unittest discover -s tests -p test_policy_training.py -v
& $python -S -B scripts/check_repo.py
```

Longer repeated updates and independently budgeted physical evaluation remain
future work. The present CLI intentionally enforces a single-update smoke check.
