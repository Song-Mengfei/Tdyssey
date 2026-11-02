# Direct-simulator DPPO milestone C engineering

Implemented on 2026-10-23, after milestone B commit
`92e285518957b62e61a68004850e372034297c88`, in the existing isolated
`codex/dppo-milestone-a` worktree. C adds continuous collection, repeated PPO
passes, permanent experiment limits, boundary recovery and independent evaluation.
The original research checkout and B's completed ledger are separate.

The initial engineering milestone used source checks, fake environments and small
synthetic CPU models. A subsequent [guarded real runtime](DPPO_C_RUNTIME.md), on
2026-10-24, completed 2,048 controls and 52 updates per optimizer in 59 minutes
35 seconds. Report, ledger and both retained delta files passed offline integrity
checks. A later [fixed paired development evaluation](DPPO_C_EVALUATION.md)
completed two 20-case groups: T success 6/10 to 5/10, L 0/10 to 0/10, with
mixed coverage changes. Broad policy improvement has not been established. This
bounded run does not validate longer training or other machines. No dependency
installation, asset download, environment synchronization or push was needed.

A subsequent [reset-only preflight](DPPO_C_RESET_PREFLIGHT.md) found and replaced
11 invalid training seeds, then verified all 64 training and 10 fixed evaluation
geometries with real resets. It used 160 internal settling substeps and zero
controller steps, policy samples or optimizer updates. This does not validate
the continuous trainer's runtime memory, throughput or learning behavior.

## Interfaces and fixed training contract

| File | Responsibility |
| --- | --- |
| [policy_stream.py](../tdyssey/policy_stream.py) | Two continuous T/L streams, fixed scene cursors, true episode boundaries and local sampling RNGs |
| [policy_ppo.py](../tdyssey/policy_ppo.py) | One entry check per fresh buffer, one to three passes, immutable behavior data and single-consumption tokens |
| [policy_protocol.py](../tdyssey/policy_protocol.py) | Stdlib configuration validation, permanent attempt counters, ownership lock, storage/deadline and evaluation group limits |
| [policy_experiment.py](../tdyssey/policy_experiment.py) | Bounded batch loop, scalar diagnostics, selected-delta checkpoints and boundary-only recovery |
| [policy_evaluation.py](../tdyssey/policy_evaluation.py) | Fixed development cases on independent environments and local RNGs; paired report comparison |
| [run_c_policy.py](../scripts/run_c_policy.py) | Separate plan/init/train/evaluate/compare commands, explicit execution gate and compact outputs |
| [policy_c.example.json](../configs/policy_c.example.json) | Versioned proposal; reading it starts no job |

The actor retains A/B's two RGB/pusher frames, fixed crop, normalizers, full
100-transition denoising chain and target indices 1:9. Only `model.final_conv.`
trains. The critic uses detached frozen visual/pusher features. No world model,
supervised policy loss, privileged state policy input or new human demonstrations
are introduced. Rewards remain simulator-derived privileged supervision and the
base retains expert T pretraining.

T and L take turns sampling plans. Both environments, observation histories,
episode control counts and local RNG states continue across batch boundaries.
Only success or timeout permits another reset. A scene cursor never wraps or
silently substitutes a scene. Exhaustion, failed operations and short budget tails
are explicit. The actual T/L control ratio is reported, rather than assumed 50/50
when successes, timeouts or exhaustion shorten a stream.

Each batch has at most 16 sampled plans and 128 physical controls. All transitions
of every retained plan contribute equally, with each microbatch's summed loss
divided by the full `B*100` count. Plan/time microbatches default to 1/1. Old latent
endpoints, probabilities, values, rewards, GAE targets and once-normalized
advantages stay fixed across passes. The new C updater performs the behavior
consistency check once on entry; it does not call B's strict single-update API
three times. Consumed, foreign, mutated or wrong-version buffers are rejected.

The first job batch uses one pass, subsequent batches up to three. All passes
cover all 100 denoising times. Frozen feature caching is checked against the
original one-plan encoder and uses the same likelihood/gradient calculation.
The std floor remains .01 at every time including t=0, with joint 32-coordinate
Gaussian log densities, PPO clip .2, denoising discount 1, gamma .99, plan lambda
.95 and shaping beta 1. Separate Adam learning rates remain 1e-5/3e-4, zero weight
decay and gradient clipping at 1. Post-pass sample approximate KL above .02 stops
remaining passes without taking another step. This is a sampled diagnostic,
not a guarantee of full-distribution KL.

At every pass, parameter selection and the frozen actor/normalizer fingerprint
must remain valid. Both optimizer reservations precede either step. A failure
after one optimizer step is recorded as a spent, inconsistent attempt and cannot
be treated as a resumable checkpoint. No buffer or computation graph is retained
after its update; only bounded scalar reports and the current batch survive.

## Permanent limits and outputs

After the reset-only scene correction, the example's canonical SHA-256 is
`0c682afbaf03360dc7af77c3c0917c3cfd120d5e7a8da10532c1d3d5251a8db3`.
The original `af09a4a` proposal was
`68a5817ce539facf651aa95934560b5c85b84bd83e90f784c44a391144135b85`.
The proposal separates these attempt budgets; failures are never refunded:

| Account | Controls | Sampled plans | Resets |
| --- | ---: | ---: | ---: |
| Training | 2,048 | 384 | 384 |
| Development evaluation | 24,000 | 3,040 | 160 |
| Other diagnostics | 0 | 0 | 0 |

Global caps are 24 batches, 72 actor steps and 72 critic steps. The first-one,
then-three schedule permits at most 70 paired steps over 24 complete batches;
the ledger caps are separate upper limits. Wall time is at most two hours from
initialization, including downtime, and aggregate experiment storage at most
512 MiB. Hitting any cap stops work; unused control capacity does not authorize
extra batches, passes, scenes or evaluation groups. A 20,000-step configuration
is rejected by C v1.
Time checks precede operations and PPO microbatches; an in-flight library call
can finish after the deadline. There is no independent process-killing watchdog.

Initialize one stable registry and experiment ID. Its authoritative ledger
follows the experiment across output directories, with exclusive crash-closed
ownership and fsynced atomic counter updates. Reopening or choosing another run
directory cannot reset counters or the deadline. An unbound nonempty directory
is rejected, protecting existing artifacts such as B. Preserve the registry;
creating another identity/registry is a new experiment requiring its own scope.
The ledger retains cumulative counters and pending attempts with a bounded recent
event window, rather than an unbounded per-control event history.

Outputs are scalar JSONL batch/pass diagnostics, JSON summaries, at most two
registered selected-weight/optimizer checkpoints across all associated output
directories, and at most one first-batch short uint8 control trace per shape.
Full RGB rollout buffers and full base networks are not repeatedly saved. Pending
write reservations count against storage; actual output and registry bytes are
checked. The shared base checkpoint is read in place and excluded from new output
storage. A conservative 64 MiB reservation precedes each incremental checkpoint.

Logs include contacts (missing contact data remains unknown), raw/training/shaping
rewards, coverage changes, per-shape controls/plans, ratio/KL/clip, gradient/critic
diagnostics, pass counts, timing and initialized CUDA allocator telemetry. The
optional consecutive zero-contact and zero-raw-reward batch stop is disabled by
default; enabling it records the exact condition and stop. Zero contact/raw reward
is a diagnostic; shaping can still be nonzero and does not prove adaptation.

## Checkpoints and recovery

Checkpoints reuse B's strict selected actor delta, critic, optimizer group order
and moments, sampler/source/base identities and Python/NumPy/Torch RNG payload.
C adds its experiment/config hash, updater version/counters, scene cursor, T/L
local RNG states, collector counters, compact loop state and boundary metadata.
Save/restore verify the live actor subset/sampler, PPO settings, optimizer
ownership/rates, device and FP32 precision against that configuration.
Training always starts from the explicitly verified base and a new critic unless
an exact C boundary checkpoint is requested. B's learned delta is never the
implicit starting policy.

Every four batches the loop requests a joint episode boundary: a finished stream
pauses while the other finishes naturally, under the existing batch/control/time
caps. No artificial reset, forced completion or extra budget is allowed. A safe
checkpoint starts a future process with new episodes and the saved next scene/RNG
states. No contact cache or active simulator state is restored. The registry
accepts resume only from its current saved boundary with unchanged training and
global counters. Independent development evaluation advances its own counters
without rewinding them or invalidating an otherwise unchanged training boundary.

A terminal checkpoint saved while either episode is active is an **audit/evaluation
delta with `resumable=false`**. It cannot continue training. For example, stopping
at exactly 1,024 controls may land inside an episode; do not promise a restart
from that file to reach 2,048. Choose a continuous authorized run or a genuine
saved boundary. A crash after additional attempts cannot rewind to an older
boundary. Mid-update replay and recovery from uncommitted physical interaction
remain unsupported.

## Independent development evaluation

One manually selected `evaluate` command runs one group in a separate process,
with no optimizer steps. Methods are `base_original` (official sampler),
`base_training` (positive-floor training sampler), and `delta_training` (same
training sampler plus an explicit C delta). Base and delta hashes are bound to
the report. No training environment/history is accepted, and tests verify global
and local RNGs, module modes, scheduler and parameters remain unchanged.

The proposal has five fixed scenes, two policy seeds and both shapes: 20 cases
per matched group, at most 300 controls per case. The first three scenes were
previously used for development; 100003/100004 are additional development scenes.
All five are excluded from training. They are **not untouched final test data**.
Four groups are the maximum, not an automatically launched evaluation matrix.
Claiming a group permanently consumes its slot even on failure or another output
directory. Repeating the same group ID is rejected.

`matched_tl` names the existing COM-matched T/L initial-state protocol: each case
resets both shapes and checks the common initial state, counting both reset
attempts. `original_t` has its own original T-only protocol and denominator.
They cannot be pooled in paired comparisons. Initial-only matching does not
establish arbitrary mid-contact state restoration.

Reports preserve every requested case, explicit incomplete/unstarted cases,
strict post-control coverage >.95 success, raw rewards, peak/final coverage/IoU,
first success step, failures, sampled plans, denoising steps, physical controls,
reset/physics costs and latency. An incomplete group has no full-denominator
success fraction. Offline comparison requires complete reports with the same
ordered cases, protocol, caps and base. Distinct delta stages retain distinct
labels/hashes; repeated method names do not overwrite pair results. Five scenes
with two policy seeds provide development evidence, not a robust generalization
or improvement claim.

## Commands and engineering evidence

Read-only commands, safe without optional dependencies:

```bash
python -S -B scripts/run_c_policy.py plan
python -S -B scripts/run_c_policy.py --help
python -S -B scripts/check_repo.py
```

Future runtime commands require separately authorized resources and existing
assets. The following are interface examples and **were not executed for C**:

```bash
python -B scripts/run_c_policy.py init --execute --experiment-id EXPERIMENT --registry REGISTRY --run-dir OUTPUT
python -B scripts/run_c_policy.py train --execute --experiment-id EXPERIMENT --registry REGISTRY --run-dir OUTPUT --source-root SOURCE --base-checkpoint BASE --controls 2048 --batches 24
python -B scripts/run_c_policy.py evaluate --execute --experiment-id EXPERIMENT --registry REGISTRY --run-dir EVAL_OUTPUT --source-root SOURCE --base-checkpoint BASE --method base_training --protocol matched_tl --group base-training
python -B scripts/run_c_policy.py compare --reports BASE_REPORT DELTA_REPORT
```

For a safe training continuation add `--resume-checkpoint C_BOUNDARY` with the
same config, experiment, registry and saved loop options. For learned evaluation
use `--method delta_training --checkpoint C_DELTA` and a fresh group label.
Commands have no automatic stage sweep, 20,000-step extension, repeated seed
sweep, background run or scheduled follow-up.

The proposal pins the existing local compact artifact to SHA-256
`37d661ce6a487905fa4ff81ab7d65bf1b56b1c06adf480a961f65338fa0e38a0`,
1,095,708,090 bytes. As in A/B, its file digest differs from the published compact
manifest and identity to released EMA tensors has not been independently proved.
The default published digest acceptance remains unchanged.

All 75 new C tests passed. Full installed discovery: 189 passed/46 skipped;
source-only discovery: 98 passed/137 skipped (235 total). Commands:

```bash
python -B -m unittest discover -s tests
python -S -B -m unittest discover -s tests
```

Disable `TDYSSEY_OFFICIAL_ROOT`, `TDYSSEY_OFFICIAL_CHECKPOINT`,
`TDYSSEY_RL_CHECKPOINT_CHECK`, `TDYSSEY_GPC_ROOT`, `TDYSSEY_DATA_TESTS` and
`TDYSSEY_GPC_WORLD_CHECKPOINT` before this regression. On Windows, place `TMP` and
`TEMP` in an ignored workspace directory for atomic-file tests. Set an ignored
workspace `HF_HOME` with `HF_HUB_OFFLINE=1`; no download is needed.

The tests cover dense-versus-microbatch updates, cached likelihood/gradient
equivalence, multiple passes with fixed data, buffer release, interleaved
episodes, budget tails, true-boundary split-versus-uninterrupted updates,
evaluation state/RNG isolation, append storage and post-publication IO faults.
Source/message/diff checks also passed. See STATUS (external historical reference: `STATUS.md`). C tests use fake environments/tiny
models and temporary ledgers, never B's real ledger. Preserved B artifact hashes:
`budget.json` = `4c1dd1a0c1be31b3fd0217608b2e424303e8df738f12e37adc94749a27c91a7e`;
`report.json` = `193697bbb054520eba033e55073b74be4ba7c2c4c50982429e082d0c97d95350`.
