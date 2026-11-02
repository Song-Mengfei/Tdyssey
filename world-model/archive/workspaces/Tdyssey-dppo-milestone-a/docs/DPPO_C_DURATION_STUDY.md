# Proposed single-factor C exposure study

Status: **proposal only, not executed or authorized** (2026-10-24). Derived from
the [offline diagnosis](DPPO_C_DIAGNOSIS.md). This is one future comparison, not a
queue of experiments. Current C v1 rejects 4,096 controls and its 24-batch/72-step
caps are insufficient. A separately reviewed protocol version, explicit runtime
authorization and new study accounts are prerequisites; none is created here.

## Question and factor

Does doubling physical training exposure from **2,048 to 4,096 controls**, with
the existing learner unchanged, produce successful L behavior without losing T
success on a newly frozen validation set?

This tests whether doubling this small exposure is sufficient; it does not assume
that undertraining caused the observed failure. L saw four episode starts, 932
controls and 247 contact controls, never high alignment. The reward arithmetic
matches the declared objective and real updates occurred. Testing exposure first
keeps the task/reward/representation fixed before changing an unproven mechanism.
Credit horizon, reward/exploration, critic quality and output-only capacity remain
alternative explanations, not additional factors in this study.

## Matched arms

| Setting | Short control | Long treatment |
| --- | --- | --- |
| Physical training target and ceiling, T+L combined | 2,048 | 4,096 |
| Initial actor | Same pinned local base SHA `37d661ce6a487905fa4ff81ab7d65bf1b56b1c06adf480a961f65338fa0e38a0` | Same base |
| Initial critic / Adam | Fresh deterministic initialization, empty optimizer state | Same initialization/state |
| Training seed / T and L local RNGs | 5478, with the existing explicit stream mapping | Same mapping and starting states |
| Training catalog and order | Exact geometry-validated 32 T + 32 L lists in `policy_c.example.json`, no wrapping | Identical lists/order |
| Policy, observation and sampling | Official visual/pusher history and normalizers; FP32; 100 denoising times; floor .01; eight-action prefix | Identical |
| Trainable actor / critic | Existing six output tensors (1,313,282 parameters), frozen remainder; same 17,153-parameter critic | Identical |
| Objective and optimization | Gamma .99, beta 1, lambda .95, PPO clip .2, actor LR 1e-5, critic LR 3e-4, grad cap 1, target KL .02 | Identical |
| Collection/update schedule | Up to 16 plans/128 controls per batch; first batch one pass, later three; same microbatches 1/1 | Identical |
| Episode end / reset / saving | Strict success >.95 or timeout 300; continuous streams, existing checkpoint request every four batches and boundary drain | Identical |
| Optional empty-batch stop / model selection | Disabled; fixed final endpoint only, no selection by validation reward | Identical |

The longer arm naturally receives more updates and encounters more scenes; those
are consequences of the exposure factor, not separately tuned settings. Record
actual T/L exposure rather than assuming 50/50. Both arms run on the same existing
environment, source hashes, device and four-thread setting, sequentially.

The control's final partial batch is collected to its remaining control allowance
and updated normally. The long arm uses the **same remaining-allowance rule**, and
is not artificially stopped/updated at 2,048 just to match a checkpoint. Thus the
short final delta is not an exact resumable prefix of the long arm. Compare complete
shared batches and early control records until the budget-tail branches differ;
do not assert bit-identical endpoints or restore the short tail into the long arm.
Any unexpected earlier mismatch is a matching failure, not a reason to retry.

Both arms must start afresh from base, including repeating all prefix work. The
existing spent training account and nonresumable final delta are not continued.
Total new training cost is **6,144 controls**, not 2,048 additional controls. The
old run remains historical evidence; it is not silently reused as the new control.
Identical training seed gives one matched seed comparison, not two independent seeds.

## Independent validation and leakage control

Use candidates **100005–100009**, policy seeds **5478/5479**, both shapes, ordered
T then L, scene then policy seed: 20 cases per arm, 40 executions. These candidates
have **not** been reset or geometry-validated in this offline task. Before any
training, audit available catalogs and run one separately budgeted matched initial
geometry preflight per scene: ten total T/L resets, zero controls/plans. If any
candidate was previously used for tuning, is invalid, or preflight fails, stop the
study before training. No replacement scene, retry or hidden extra reset is allowed.
An undetected external exposure cannot be ruled out from this machine's records.

Freeze and hash both training/evaluation exclusions and the valid case list before
either arm starts. Old development scenes 100000–100004 remain excluded from training
but are diagnostic only here; no repeated runs or model selection on those cases.
Candidate validation is a new, fixed confirmatory comparison after a development
decision, not a claim of an untouched final paper test. Reserve separate scenes
and multiple training seeds for later final validation, outside this proposal.

Evaluate only final endpoints with the same .01-floor training sampler, matched
initial T/L COM states, 300-control cap and strict first-success >.95 termination.
Do not change reset distribution, success threshold, reward or stopping rule to
help either arm. Load one complete model at a time; no optimizer or training calls
are granted by evaluation. No intermediate policy evaluation or best-case stopping.

## Finite prospective budgets

These are **proposed new account limits**, not changes to existing ledgers. Caps
other than the physical target are operational guards intended not to bind first.
Hitting another cap before target makes that arm incomplete; do not compare it as
a completed dose, extend it, create a refill ID or substitute another run.

| Account | Controls | Plans | Resets | Batches | Actor / critic steps | Wall deadline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Geometry preflight | 0 | 0 | 10 | 0 | 0 / 0 | 300 seconds |
| Short training | 2,048 | 384 | 64 | 24 | 72 / 72 | 5,400 seconds |
| Long training | 4,096 | 768 | 64 | 48 | 144 / 144 | 9,000 seconds |
| Two-group validation | 12,000 | 1,520 | 80 | 0 | 0 / 0 | 5,400 seconds |

Validation resets are 80 because each of the 40 cases resets both T and L to
validate a matched initial state. Preflight's ten resets are additional and consumed
even on failure. Training catalogs do not wrap beyond their 64 possible starts.
All attempts/failures remain spent. Maximum new physical controls are 18,144;
preflight, training and validation are separately attributed.

Each training/evaluation account has a 512 MiB formal artifact cap, at most two
retained deltas per training arm; preflight has 1 MiB. Three guarded jobs have
64 MiB host telemetry each, and offline review has 128 MiB. Storage bounds are
explicit limits, not expected size or permission to duplicate base checkpoints.

Reuse the actual Windows worker handshake/retained process handle/kill-on-close
Job Object guard. Before each guarded job require available RAM 4 GiB, commit
headroom 6 GiB, free GPU 4 GiB and disk 2 GiB; ensure aggregate storage also fits.
Stop on runtime floors RAM 1 GiB, commit 2 GiB, GPU 1 GiB; preserve the 0.2-second
host and five-second GPU cadence. No user applications are closed. Deadline starts
immediately before each account's initialization and includes downtime. No automatic
resume after interruption or ambiguous partial operations.

Existing first-batch/finite-loss/frozen-weight/behavior-probability/accounting gates
apply equally. Stop on gate, identity, resource, storage or runtime failure. Existing
KL stopping can reduce passes equally; log this rather than changing its threshold.
Post-run success requires semantic target completion, not process exit code alone.

## Predeclared interpretation

Use paired success as primary and report every case and the five scene blocks.
An encouraging result must satisfy **all** of these descriptive gates:

1. Long-minus-short L net successful cases is at least two of ten, with new L
   successes in at least two different scene blocks.
2. No T case successful in the short arm becomes a failure in the long arm.
3. T mean final coverage declines by no more than 1 pp; report peak/IoU and
   all paired outcomes even if this gate passes.

No tuning or extra evaluation follows automatically. If L remains unsuccessful,
report that doubling exposure did not establish successful adaptation for this
seed; do not claim all longer training is futile. If L improves but T fails its
gate, report the tradeoff. Coverage-only changes are secondary and cannot replace
the primary success gate. Resource/incomplete/matching failures are inconclusive,
not successes or permission to add runs. One seed/five scene blocks do not establish
statistical significance, population performance or unseen-shape transfer.

Record per-control metrics, contacts, actual pusher/object states and bounded actual
RGB identically in both arms if future implementation adds them, within these storage
caps. Logging must not add policy/physical operations or privileged actor inputs.
Separate per-shape return/advantage diagnostics may use the already collected buffers;
never change the existing mixed normalization or training math in this comparison.

## Cost estimate and execution boundary

Measured source run: 2,048 controls in 59m35s; measured paired 40-run evaluation:
35m42s. Linear exposure extrapolation suggests about one hour for short training,
two hours for long training and 36 minutes for validation: **roughly 3h35m plus
preflight/IO**, with no installations or downloads. Boundary drains, successes,
GPU pressure and optimizer costs can change this substantially. Expected long-arm
updates are roughly 100–110, not guaranteed. Proposed wall ceilings sum to 5h35m.

No runnable long-study config, ledger, launcher or protocol-cap patch is produced
in this task. Future implementation must preserve old v1 semantics/evidence and
bind both fresh arms to one explicitly authorized study budget. A new identifier
alone never grants more work. Present this proposal for review; do not start it.
