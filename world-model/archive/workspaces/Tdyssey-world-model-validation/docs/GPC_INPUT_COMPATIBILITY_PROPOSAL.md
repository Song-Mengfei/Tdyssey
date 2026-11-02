# Frozen input compatibility and early-shape proposal v1

**Design only, not authorized for execution.** Base `994c6ff`, 2026-10-24.
No new experiment, model forward, sampler, decoder, physics or training ran while
writing this document. Implementation of a future harness, review of its guards,
and explicit authorization are prerequisites to any execution. There is no
launch command here. The [static audit](GPC_INPUT_COMPATIBILITY_AUDIT.md) supplies
the contract and provenance evidence; the old ledgers remain permanently spent.

The subsequent [engineering implementation](GPC_INPUT_COMPATIBILITY_ENGINEERING.md)
adds the isolated harness and fake/native process checks. It does not authorize
execution or change the fixed experimental settings below.

Question: under one exact frozen checkpoint, does a single predefined RGB-only
palette transform improve early rigid-shape preservation on the saved official
T/L contexts? Separately, does replacing generated intermediate context with
saved real observations change that outcome? These tests do not assume that
color is the only problem, or that a failed transform requires fine-tuning.

## Fixed assets and exposure

Bind exact hashes before any future run:

- Source/config baseline `994c6ff`, GPC upstream
  `a96146e568826ef46134f73c22fdf59a675e7264`; official source
  `5ba07ac6661db573af695b419a7947ecb704690f`. A future harness has its own reviewed
  source hash; upstream and current runtime files are not patched by this plan.
- Exactly one phase-two checkpoint, 35,043,916 bytes, SHA-256
  `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`, HF revision
  `c16a467ec176af19bba5136564a81d8d61d72af9`. Never substitute the csgo file.
- Existing `artifacts/frozen-wm-validation-v1-20261024/report.json`, hash
  `c7b1e3257468874940ad074d3ffc00c3d3d88f9f6de16531aba6d2c3cc6e3795`;
  63 complete physical NPZ branches: 16 T, 47 L. Each supplies four history RGB,
  three past controls, four future targets, and four saved actual future RGB.
  The 64th planned branch L01/hold stays missing; never reconstruct or replace it.
- All 189 old forecast NPZ (756 images) are development references, not reused
  as the new A arm. Same seed across a new backend/process is not assumed bitwise
  identical. Recompute both future arms under one new, matched implementation.
- Existing 24 calibration RGB plus 252 history and 252 actual future RGB = 528
  real-image rows for a measurement screen. Keep duplicated rows in denominators;
  caching by RGB hash is allowed. The 24 calibration +63 physical +189 old
  forecast files form 276 input NPZ; bind each file's bytes/hash in a manifest.
- Freeze measurement source/config at `994c6ff`: v2 object/pusher/shape/coverage
  interface, strict >.95 termination and the declared +/- .03 development
  assumption. No threshold, geometry, ambiguity or success-rule tuning mid-run.

These assets have already been examined and used for development. There is no
train/test random split over frames or noises. New results remain development
diagnostics, including the four T controls. Exact upstream training shape/domain
exposure is unknown, so even a future held-out scene is not necessarily an
unseen shape to the checkpoint.

## Two input arms; conditioning is a separate factor

| Arm | Real observation input | Everything else |
| --- | --- | --- |
| A | Original official RGB, identity operation | Same frozen checkpoint, actions, sampler, RNG state, decoder, goal/geometry and scoring |
| B | One fixed per-pixel transform T1 below | Identical to A |

T1 means "source-evaluation palette hypothesis", not a proven reconstruction of
the checkpoint's training domain. It does not synthesize the action marker seen
in an upstream evaluation render path. Sampler `x0=randn`, image input [0,1],
sigma schedule, three Euler steps, dtype float32, batch1 and eval/no-grad status
remain unchanged. No image centering to [-1,1], noise-amplitude change, resize,
rotation, translation, blur, target redraw or appearance search is another arm.

### Exact T1 definition to review before implementation

Input is a finite `[3,96,96]` float32 real RGB in [0,1]. Reject malformed input;
do not clip it into compliance. Work independently at each existing pixel,
using RGB values in [0,255]. Fix these swatches from source drawing constants:

| Symbol | RGB | Meaning |
| --- | --- | --- |
| W | (255,255,255) | White background |
| Q | (211,211,211) | LightGray walls |
| P0 / P1 | (65,105,225) / (78,126,255) | RoyalBlue outline / bright pusher interior |
| G0 / G1 | (119,136,153) / (143,163,184) | Gray object outline / bright interior |
| R0 / R1 | (165,28,48) / (198,34,58) | Red object outline / bright interior |
| V / K | (144,238,144) / (0,0,0) | Visible green / black target |

For each source segment `[u,v]`, compute float64 projection
`alpha=clip(dot(p-u,v-u)/dot(v-u,v-u),0,1)` and squared RGB distance to
`u+alpha*(v-u)`. Pick the smallest distance, resolving equality within `1e-9`
by this fixed order. The table maps source endpoints to output endpoints:

| Order | Source segment | Output segment |
| --- | --- | --- |
| 1 | W--Q | W--Q (identity) |
| 2 | W--P0 | W--P0 (identity) |
| 3 | P0--P1 | P0--P1 (identity) |
| 4 | W--P1 | W--P1 (identity) |
| 5 | W--G0 | W--R0 |
| 6 | G0--G1 | R0--R1 |
| 7 | W--G1 | W--R1 |
| 8 | W--V | W--K |
| 9 | V--G0 | K--R0 |
| 10 | V--G1 | K--R1 |

Accept a conversion only if RMS channel residual is <=8 in [0,255] RGB units;
otherwise preserve the original pixel exactly. Identity-segment winners also
preserve original pixels exactly. For conversion winners use the same alpha on
the output segment, convert to float32, then multiply by float32 `1/255` without
uint8 re-quantization. Record winner counts, residuals, changed/unmatched pixels
and range checks; do not select a different mapping by these results.

This fixed color-line approximation preserves some fractional antialias evidence
but is not an exact inverse renderer. It may misclassify mixed boundary colors;
the real-image measurement screen below must detect harmful consequences. It
cannot see an occluded goal, infer a missing arm or restore target geometry.
There is no image ROI, connected-component replacement, true pose, action input,
contact, shape mask, true goal location or future pixel input to T1. It never
moves a pixel, rescales a body or draws a replacement shape. No fitting of T1
parameters on the trial outputs is allowed.

### AR boundary

Apply identity/T1 **once to the four real initial history frames**. Predict H1;
append the raw prediction and drop the oldest frame. Continue through H4 with
raw generated RGB, without further T1, clipping, denoising or recoloring. After
H1/H2/H3 the history contains three/two/one transformed real frames plus raw
predictions; after H4 all four positions have been generated. This boundary is
part of the hypothesis, not an adaptive choice after seeing recovery.

Save H1-H4 raw arrays. Primary reported horizons are H1/H2/H4; H3 is necessary
to generate H4 and remains available in the inventory. Never feed transformed
actual future frames into an AR rollout or use truth termination to repair it.

### Teacher-forced (TF) diagnostic boundary

Run one-step conditional predictions only at H2 and H4, with the same candidate
action window, but replace preceding generated images with **already saved real
past observations**. For history ending at `o(t)`:

- H2 input: `[o(t-2),o(t-1),o(t),actual o(t+1)]` and
  `[a(t-2),a(t-1),a(t),a(t+1)]`.
- H4 input: `[o(t),actual o(t+1),actual o(t+2),actual o(t+3)]` and
  `[a(t),a(t+1),a(t+2),a(t+3)]`.

Apply identity/T1 once to every real frame in those TF windows. The target frame
`actual o(t+h)` is never an input. H1 equals the AR first step and is reused,
with no extra forecast. This privileged use of realized intermediate RGB is an
explicit **conditional diagnostic**. TF results cannot enter candidate planning,
stopping-aware scores, imagined return or policy comparisons.

## Matching, phases and fixed prediction budget

Noise seeds are fixed at 5478/5479/5480. Record and replay each AR sampler-entry
CUDA RNG state for `(branch,seed,horizon)` across A, B and corresponding TF.
Simply reseeding TF with the same seed would reuse H1 noise at H2/H4 and confound
conditioning with noise. Replay all random draws in that sampler call, not just
an initial latent; check entry/exit RNG hashes and fixed call shapes. Restore the
external Python/NumPy/Torch RNGs afterward. Seeds/noises are correlated repeats,
not additional physical cases. Do not pick the best seed or output.

Both arms use one model instance sequentially. Fix order T12..T15 then L00..L11,
candidate hold/toward/side/away, and seed order above. H1-H4 AR precedes TF H2/H4
within each branch/seed block. A/B matching is by the same underlying saved-RGB
file/frame identity, action window and RNG state; B's input is deliberately T1(A).
Record both the source RGB hash and each arm's actual input tensor hash. Both raw
outputs and an unsuccessful attempt remain logged.

| Phase | Scope | Forecast images | Euler/denoiser steps |
| --- | --- | ---: | ---: |
| M | 528 real rows x identity/T1, measurement only | 0 | 0 |
| T-AR | 16 T branches x3 seeds x4 horizons x2 arms | 384 | 1152 |
| T-TF | 16 T branches x3 seeds x2 horizons x2 arms | 192 | 576 |
| L-AR | 47 L branches x3 seeds x4 horizons x2 arms | 1128 | 3384 |
| L-TF | 47 L branches x3 seeds x2 horizons x2 arms | 564 | 1692 |
| Maximum | 1512 AR +756 TF, H1 reused | **2268** | **6804** |

Exactly three denoiser/Euler calls per image; there are no free warm-up forwards,
model-residency probes, reruns, extra seeds, new assets, policy calls or training.
All new environment creations, resets, restores, physical renders, controls,
physics substeps and optimizer updates have caps **0**. Model instances/loads
have cap **1**. Startup/loading and all measurement/export work count toward the
same future 1800-second deadline.

Complete the T phase without performance-driven example filtering. Continue to
L only if M passes and the T phase meets the shape screen below. Otherwise stop
at at most 576 images/1728 Euler steps and mark L **not attempted**, retaining
its planned denominator and explicit stage-block reason. No other transform,
extra T case, changed threshold or weight update follows automatically.

## Measurement, attribution and comparisons

Keep independent full-shape soft IoU, full-template residual, rigid-area ratio,
fragment mass, object state, pusher validity and coverage. Retain rejected
candidate poses only as unvalidated diagnostics. Goal geometry is used after
image pose fitting; actual pose/contact/coverage supply posthoc error labels.
V2's soft-shape threshold is not interchangeable with v1's binary metric.

M checks the **same frozen decoder** on original/T1 real RGB before model loading:
all 528 available rows retain object/shape/pusher/coverage validity; per-shape
position P90 <=2 world pixels, angle P90 <=2 degrees, coverage absolute-error P90
<=.03. For both A and B require zero false positives from the three-state
`termination.status=='success'` classifier against strict true coverage >.95;
also export the paired row IDs of every B-only declared positive. Nominal point
coverage >.95 confusion is diagnostic only and is not an alternative M gate.
Report max errors, all losses, point confusion and unknowns. The eight near-.95 calibration rows cannot validate
success precision when no interval-positive case exists. Failure stops the
forecast proposal as an input-transform/measurement incompatibility, not a
demonstrated dynamics failure. No refit or threshold relaxation is permitted.

The T shape screen uses **AR only**, never TF as a substitute. It requires B
object/full-shape validity >=90% of available
rows separately at H1/H2/H4 (at least44/48 in each cell), pusher validity >=95%
in each of those AR cells (at least46/48),
and no >5 percentage-point validity regression against A in an available
contact/noncontact stratum. Use all available rows, not just paired measurable
ones, for these rates. If L is reached, apply the same >=90% screen per horizon
to its 141 available rows (at least127/141), and report all 144 planned rows.
These are predeclared engineering screens, not statistically validated cutoffs.

A limited saved-scene compatibility claim additionally requires B-AR's measured
position and pusher P90 <=5 world pixels, angle P90 <=5 degrees and absolute
coverage-error P90 <=.05, per shape and H1/H2/H4, alongside the >=90% shape
coverage above. Keep the measured fraction explicit. Cross-arm error differences
use the common measurable subset and are undefined when that subset is empty;
in particular A's old H1 validity cannot be silently imputed. Passing M on real
RGB does not certify measurement uncertainty on generated images. None of these
criteria establishes strict terminal precision or a dynamics/generalization pass.

For every phase report A/B paired validity win/tie/loss; invalid reasons; blur/
outline nuisance estimates; geometry/coverage errors on the **common measurable
subset** and that subset's fraction; unmatched/malformed inputs; RGB range and
clipped fraction for diagnostics. Contact by H4, no-contact, true rotation >5
degrees and near-.95 are posthoc strata. Aggregate noises inside branch, then
weight anchors equally. Provide raw row/branch/anchor counts. Few physical
anchors cannot support a general contact or rotation claim.

Appearance-only checks compare A forecast to original actual RGB and B forecast
to `T1(actual RGB)`; B persistence uses `T1(latest RGB)`. Do not compare different
color-domain RGB MSEs as a pure motion improvement. Principal cross-arm endpoints
are common physical pose/shape/coverage and validity. Inspect fixed case plates
(first six matches per predeclared reason category) with saved RGB and separate
physical labels. A bad area/soft-IoU gate alone is **unresolved attribution**;
actual generated missing arms need visible evidence, while a plausible-looking
shape may still expose a measurement failure. TF-versus-AR changes identify
conditioning dependence, not automatically physical correctness.

AR raw endpoint measurements survive earlier unknown frames. Stopping-aware
scores require an unambiguous predicted prefix, allowing model-only absorption
after a declared success. The strict >.95 rule remains; unknown is never success.
The +/- .03 interval is still only a development assumption and full pose
ambiguity remains unassessed. No true future stop, mask or score repairs it.

Action ranking is **AR only**, separately raw diagnostic and stopping-aware.
Compare A/B/hold/uniform-random expectation on identical complete anchors where
all four candidates and all three noises are measurable for both arms and the
same scoring mode. Report every omitted anchor and abstained pair. Hold and
uniform references also keep independent complete physical denominators.
Uniform-random means the exact average of the four saved candidate outcomes;
it requires no new policy or sampling. Use mean across fixed noises, pair
separation >=.05 and tie tolerance1e-6. Observed correct/scored, abstained and
incorrect pairs are distinct; conservative correct/all eligible is separately
labeled. Never compare selected-subset model regret against full-set baseline
means. TF never contributes a future task score.

## New ledger, Windows guard and stopping rules

If separately authorized, implement a **new evaluation-only** attempt registry
and fsynced journal, for example `artifacts/wm-input-compat-eval-v1-ledger/`.
Do not instantiate the old frozen budget class against old files, reset old
clocks or modify any B/C account. Bind attempt ID, protocol/T1/harness/decoder/
sampler/source/checkpoint hashes, input manifest and the complete planned table
including L01/hold's missing status. Changing an output directory does not create
new authority. Source/config mismatch fails closed; it does not reset counters.

Reserve each sampler call's one image/three Euler costs durably before entry.
Failed/unknown reservations stay charged; no retry or replacement. Record actual
worker PID, nonce, call key, RNG entry/exit identity, wall time, raw output hash
and bytes. An error ends the attempt, with unattempted planned rows explicit.
There is no automatic resume, stage restart, deadline extension or budget refund.

Proposed resource limits, to be checked in a later authorized launch:

| Resource | Launch prerequisite | Runtime stop |
| --- | --- | --- |
| Available host RAM / commit headroom | >=4GiB / >=6GiB | <1.5GiB / <3GiB |
| Free GPU / disk | >=4GiB / >=2GiB | free GPU <1.5GiB; output/disk reserve unavailable |
| Worker process | One batch1 CUDA model, two CPU threads | working set >2GiB or private commit >3GiB |
| CUDA allocator | One float32 model, no parallel policy | allocated >512MiB or reserved >1GiB |
| Wall time | One monotonic 1800s deadline before any setup | deadline reached, including measurement/export |
| Storage | Results <=256MiB; journal/telemetry <=16MiB | combined new files, temporary files or outstanding byte reservations exceed either cap |

These are conservative proposed guard values, not current free-memory readings
or proof that a launch fits. Use a Windows kill-on-close Job Object with the
actual worker PID/nonce and a 3GiB job-memory ceiling, process-tree cleanup,
resource polling and a durable stop reason. Missing/invalid resource measurements,
singleton-lock failure, asset drift, RNG/pair mismatch, nonfinite output, OOM,
unexpected environment/policy/optimizer creation, or failed IO stop immediately.
Shape/coverage abstention itself is recorded through the fixed T phase; it does
not trigger unbudgeted retries. T-stage futility gates determine whether L runs.

Maximum raw forecast float32 payload is 250,822,656 bytes. Reserve that worst
case without assuming compression, plus at most12MiB metadata/real-image screen/
RNG state/previews within the 256MiB result cap. Store inputs by hash/reference,
not duplicate history/weights; use immutable bounded archive chunks with space
reserved for the temporary chunk as well, never repeatedly rewrite an ever-growing
archive. The raw payload plus12MiB leaves 5,029,888 bytes under the result cap for
archive/header headroom; prove all actual headers and temporary reservations fit. The
16MiB journal/telemetry cap is separate (272MiB aggregate). Preflight must prove
the actual serialization bound before any forward; otherwise do not launch.
No checkpoints, videos, downloaded assets or full denoising trajectories are saved.

## Conditional next step; no unconditional adaptation claim

| Future outcome | Supported conclusion and next proposal |
| --- | --- |
| T1 damages real-image measurement or M fails | The transform/measurement pair is unsuitable. Repair or independently assess measurement before attributing prediction errors to physics. No weight-adaptation conclusion. |
| M passes; B restores early shape through the fixed AR horizons | T1 is sufficient for **limited saved-scene input/shape compatibility** at these settings, if the paired physical metrics also remain acceptable. This is not a general world model, precise terminal detector or policy gain. |
| TF improves but AR remains poor | Generated-history distribution/compounding is implicated. Propose a separate temporal/feedback study or data-adaptation comparison, after checking generated-style measurement. Do not claim dynamics alone is wrong. |
| Both arms/conditioning modes retain visible rigid-geometry errors, with measurement independently adequate | Residual generation/shape-domain mismatch supports considering a bounded weight/data-adaptation experiment alongside other representation/interface alternatives. One failed T1 does not prove fine-tuning is necessary or sufficient. |
| Shape is adequate but matched, measurable action effects repeatedly lose to persistence/hold | Stronger evidence for action-conditioned model error, conditional on timing, image measurement and enough independent physical anchors. A separate contact/rotation adaptation proposal may be justified; do not infer this from rejected frames alone. |
| Termination remains unknown or rank subset is tiny | Keep reward/planning blocked even if images look better. No unknown-as-success conversion and no policy training. |

Existing branches contain no successful future endpoint and too few independent
ranking/rotation cases to validate terminal precision or general ranking. After
any development improvement, a separately authorized unseen-scene collection and
fixed final model/transform/decoder evaluation is required, reserving trajectories
and scene seeds before inspecting results. Shape exposure must be disclosed;
genuine unseen-shape claims need verified exclusion from adaptation data and
appropriate checkpoint provenance. This proposal creates neither those data nor
permission to collect them, and does not authorize training or a new experiment.
