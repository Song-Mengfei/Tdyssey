# GPC supervised adaptation preparation

Subsequent authorized work collected [A and the stopped first B attempt](GPC_ADAPTATION_COLLECTION_RUN.md).
The current [frozen train-list/B2 stage](GPC_TRAIN_SCHEDULE_B2.md) binds existing
A data and a train-only list in the configuration. The original preparation
evidence below remains historical; neither stage authorizes real GPC training.

This is new engineering based on `7a807b3`, following the offline L diagnosis.
It is not another frozen M/T/L experiment and does not reuse that run's success
as adaptation evidence. Real GPC/policy loads, forwards, predictions, sampling,
training, collection, resets and controls are zero in this preparation. Existing
weights, raw evidence, six spent claims and budgets remain protected.

## Data and readiness

The [data audit](GPC_ADAPTATION_DATA.md) found no eligible local official T/L
collector catalogs or complete episodes. The old custom CNN's L pilot is
documented, but its source assets are missing; its reviewed validation cannot
become independent validation. GPC fixtures and the 63 reviewed physical branches
are diagnostic/regression data, not official supervised episodes or new validation.
No held model-test payload was opened, hashed, decoded or used for tuning.

`artifacts/wm-l-adaptation-preparation-20261025/` contains `before.json`,
`data-audit.json`, `split-proposal.json`, `collection-plan.json`,
`benchmark-proposal.json` and `objective-source-audit.json`. The split manifest
is deliberately valid but empty; the benchmark has no independent payloads.
The default CLI exposes these gaps. It also requires eight distinct loss windows
per shape in independent validation before real startup, not after training.
Whole trajectories and full initial-scene hashes are isolated across train,
development, independent validation and held model-test roles. Independent
benchmark scenes must be disjoint from every source role, including held metadata,
and all 316 previously reviewed full-state hashes. No held payload is needed to
check those identities. Prior uses must be explicitly declared.

The fixed collection proposal has four eight-episode official runs, 3,200 total
controls, 640 reset candidates, 480 seconds and 128 MiB. Intended source-derived
roles are 24 train, two development, two independent validation and four held.
Accepted scene IDs and source hash-sorted split membership cannot be known before
collection. Existing collector/readback code still needs reviewed-scene exclusion
and held-payload handling before that proposal can run. Near-success/contact/
rotation coverage is not guaranteed; deficiencies stop the proposal without
automatic replenishment. No collection is implemented or executed here.

## Model and supervised objective

The [configuration](../configs/gpc_supervised_adaptation.json) fixes original GPC
source `a96146e568826ef46134f73c22fdf59a675e7264`, phase-two initialization
SHA256 `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`,
architecture, T1, action normalization and the three-Euler inference sampler.
Initialization remains immutable; the adapted product is `adapted-latest.pt`
in a separate, exclusive candidate directory.

Training uses the differentiable native `compute_model_output`, never the
no-gradient inference sampler or custom CNN loss. It is the exact t=5 one-target
subcase of phase-two supervision, not a reproduction of its longer original
training sequences: real images 0..3 and controls 0..3 predict real image 4.
All five real images receive fixed T1 exactly once; actions arrive already
normalized by `a/511*2-1`. No image or action statistics are fitted for rescaling.
Descriptive statistics count unique training controls only, excluding validation
and held episodes. They describe the full eligible training catalog, not actual
optimizer exposure. The initial iteration visited the first 250 windows per shape
in catalog order. The subsequently authorized configuration consumes an explicit
frozen 500-entry train list, balancing all 12 scenes per shape and including the
existing >5-degree rotation targets. Its file/order/rule hashes and cumulative
cursor bind checkpoint recovery. The old prefix remains a comparison only;
audit the actual consumed list rather than infer exposure from corpus totals.
Generated AR feedback stays raw, with no second T1 or clip.

Conditioning and target sigma draws are lognormal `exp(N(-1.2,1.2))`, clipped to
[.002,20]. Independent channel offset noise .1 is added without multiplying by
sigma. Native sigma-data .5 and effective `sqrt(sigma^2+.1^2)` conditioners are
preserved. Plain MSE compares native model output with
`(clean-c_skip*noisy)/c_out`; there is no additional loss weighting, contact head,
reward head or policy network. All `inner_model.*` parameters train; architecture
and buffers remain intact. AdamW is lr1e-4, betas(.9,.999), eps1e-8, weight decay
.01; gradient norm clipping 1.0 is an explicit pilot addition, not an upstream
default. No EMA, augmentation, scheduler or alternative candidates are searched.

## One future bounded candidate

The [entry](../scripts/adapt_gpc_world_model.py) defaults to read-only planning:

```powershell
python -S -B scripts/adapt_gpc_world_model.py
```

`--preflight` is a separate native Windows resource query, implemented but not
executed in this preparation. It loads no model. Future `--execute` additionally
requires complete reviewed assets, a specific execution authorization ID, exact
authorized config SHA256, clean reviewed source commit, explicit existing Python,
pinned source and initialization paths. Changing output directories cannot
refill the single fixed candidate account. There is no automatic retry or CLI
resume. Missing data refuses execute before a claim, output, resource query or
Torch import. No future execution authority is supplied by this document.

Training alternates T/L 1:1, batch 1, seed 5478. Exactly 500 completed updates
(250 examples per shape) are required for selection. The first real feasibility
update is budgeted update 1, with its forward/backward/optimizer work and memory
check included; there is no free warm-up. Training, loading and checkpoint
selection must finish within 900 seconds. If 500 is not reached in time, stop
incomplete and do not select an earlier prefix or evaluate it. Save every 50
updates and at the final 500; only the exact final checkpoint is selected once.
No validation performance or prior L diagnosis selects a checkpoint.

The entire candidate includes setup, loading, training, checkpointing and one
frozen-versus-adapted evaluation in 3,600 seconds. At most three serial model
loads are allowed: train initialization, original frozen, selected adapted.
Release each model and optimizer before loading the next; no policy is loaded.
Original and adapted weights use identical T1 real contexts and fixed noise.
Validation loss uses 16 examples per weight (eight per shape), separate generators
and matched per-example noise; those examples never enter the optimizer.

The forecast comparison has the 63 old regression branches plus 48 future
independent branches from 12 new anchors, four candidates each. For each weight,
three fixed seeds and six predictions per branch mean 1,998 images; total
3,996 images and 11,988 Euler calls. AR H1..4 feeds raw output; TF H2/H4 uses only
preceding saved real observations. H1 is reused, and TF cannot rescue AR ranking.
All seeds share the same physical branches and are not independent scenes.
The independent benchmark proposal separately caps future physical collection at
256 controls, 256 simulated reset attempts, 64 restores, 512 RGB renders,
2,816 physics substeps, 64 MiB and 600 seconds. It requires separate authority
and collection implementation; these operations are not hidden training costs.

Native startup floors are RAM 4/commit 6/GPU-free 4/disk 2 GiB; runtime floors are
RAM 1.5/commit 3/GPU-free 1.5 GiB. Process working-set/private ceilings are 2/3 GiB,
with a retained 3 GiB Windows Job and one active model. Two CPU threads and CUDA
allocated/reserved ceilings 512 MiB/1 GiB apply throughout. These are conservative
proposed limits, not measured training memory or throughput. The real first
update may fail them; stop without increasing them. The native parent observes
resources every .2s, uses the existing asynchronous 5s GPU reader, and can terminate
a blocked worker at the fixed phase/global deadline. New protocol ownership and
actual GPU training remain unvalidated on native Windows.

Output is capped at 1 GiB plus 64 MiB parent telemetry. Two 160 MiB checkpoint slots
reserve old/new atomic overlap; maximum forecast NPY serialization is below 443 MB.
Report metadata has a 16 MiB subcap and another 16 MiB is prepaid for the finite
attempt/highwater ledger. Chunks are immutable, single-owner writes; no new
shared storage scanning/locking framework is introduced. Durable reservations
precede computation; failed units stay spent. Parent telemetry has its own
bounded writer and reuses existing native ownership/resource primitives.

## Evaluation and recovery limits

Fixed engineering screens retain 90% joint object/shape validity, 95% pusher
validity, 5px/5deg position/angle P90, 5px pusher P90 and absolute coverage-error
P90 <=.05, as in the frozen proposal. Coverage MAE remains diagnostic and cannot
substitute for this P90 gate. Report
noncontact drift (mean<=2px), contact error against same-valid-subset persistence,
and actual >5deg rotation direction (>=80%, angle MAE<=5deg and better than zero
rotation). Missing/invalid predictions retain their denominator and explicit
abstentions. Sparse contact/rotation coverage is insufficient evidence, not a pass.
The new summary separates regression from independent data, shape, horizon,
contact and TF. Error means on measurable frames are explicitly conditional.

Ranking uses stopping-consistent AR scores, all three seeds per candidate and
all four candidates for selection; unknown stopping history abstains. Fix tie
tolerance 1e-6, pair separation .05, accuracy>=75%, selected endpoint regret<=.03
and compare hold/uniform expectation on the same anchors. Truth only labels
errors and physical outcomes; it never repairs predicted state/stopping or
selects model actions. Strict physical success remains coverage >.95. Precision
with no predicted positives is undefined; the old +/- .03 development interval
does not certify new near-success stopping. These screens do not authorize
imagined DPPO or establish adaptation benefit/generalization.

The paired retention rules were fixed before any real adaptation execution.
They compare frozen/adapted rows with the exact same stratum, shape, physical
anchor/candidate, noise seed, mode and horizon. The runtime passes the complete
case declaration to reporting; an observed-row union is insufficient to certify
retention. Each T horizon has 48 regression rows (16 branches/four anchors) and 72
independent rows (24 branches/six anchors) per weight before contact stratification.
Missing rows remain in those denominators. AR H1..4 and TF H2/H4 are separate.
The fixed seeds 5478/5479/5480 remain in the declared grid even if a caller omits
one from its observed/requested set; that omission cannot certify retention.
The absolute T AR summary also checks every declared H1/H2/H4 cell, not just
the cells still present after missing rows or an entire horizon disappear.

Allowed validity drop and newly invalid rows are both **zero**, including when
gains on easier rows would offset losses. Report both-valid/frozen-only-valid/
adapted-only-valid/both-invalid transitions, completion/missing counts, lost row
IDs and affected branch/anchor counts. Compare position/angle/pusher/coverage
errors only on exactly common valid rows with consistent source truth and matched
noise. Publish both arms' common-subset distributions, per-row error deltas,
the difference of the two P90s, and common row/branch/anchor counts. Error
increases are allowed only within numerical tolerances: 1e-6 world pixels for
position/pusher, 1e-6 degrees for angle and 1e-9 coverage fraction. Physical
degradation allowance is zero; the 5px absolute gate is not a relative allowance.
Every comparable row must respect its tolerance, not just the average.

A pass additionally needs all declared rows completed with matched labels/noise,
every planned anchor represented, and common valid support >=90% of planned rows
for position/angle/coverage and >=95% for pusher. Empty/mismatched/insufficient
support yields `insufficient`, `passed=null`; known new failures or larger error
increases yield `fail`. Identical rules also publish L changes without declaring
an adaptation benefit. Regression evidence remains separate from independence.

Ranking retention has two separate requirements. First, every planned anchor,
including no-contact controls, must retain both weights' four candidates, all
three noise seeds and every declared AR/TF horizon with consistent physical
truth, contact labels and RNG identities. Missing or mismatched control rows
still block a pass. Second, ranking information is fixed solely by the declared
physical endpoint outcomes: an anchor is informative when at least one candidate
pair differs by>=.05. Contact labels, prediction validity and predicted scores
cannot select that pool. Every informative anchor and all its physical separable
pairs must be comparable; unscoreable informative anchors remain abstentions,
failures or insufficient evidence, never exclusions.

Complete anchors with no such physical pair are explicitly `uninformative`, with
`passed=null`. Report their IDs/count and the total planned anchor/row denominator.
They remain in state, coverage and T retention checks, but contribute neither a
ranking success nor a ranking failure, nor any ranking-support units. Unknown
physical truth is insufficient, not uninformative. Each shape/stratum retains
the existing minimum of one comparable informative anchor and one separable
physical pair; all informative anchors must be covered. Entirely uninformative
or otherwise undersupported groups are insufficient. No previously correct pair may
become incorrect/tied/unscoreable, and selected regret may not rise beyond 1e-9.
Report all planned eligibility/loss counts alongside common-subset accuracy and
regret. No separable pairs means undefined accuracy, not successful retention.
Overall T retention requires AR relative retention, the existing absolute AR
screens, and paired ranking retention separately in regression and independent
strata. TF cannot rescue it. Termination precision remains uncertified even when
retention passes. These are conservative fixed-sample engineering rules.

The ranking-support repair after `ab03ba3` separates complete scene coverage from
this fixed informative subset. It leaves zero-degradation tolerances, absolute
gates, budgets, training/model/sampler settings and all data/native limitations
unchanged. Its new evidence lives under
`artifacts/wm-l-adaptation-ranking-support-repair-20261026/`; earlier evidence is
preserved. This repair executes only synthetic numeric/CPU checks, never real
models, native preflight or physical operations.

This repair passes 19/19 pure stdlib checks with Torch absent and 85/85 combined
CPU/fake checks with CUDA initialization and real GPC imports denied. The first
combined run had 84 passes and one pre-existing guard-fixture equality mismatch
(`3601.0000000000036` versus 3601). Repeating with integer fake-clock origins for
that single fixture passed; guard and guard-test sources were not changed.
Both reports and `clock-fixture-note.json` are retained. Reproduce the fixture
setup with the repair directory's `run-repair-checks.py --report <new-file>`.
All 7,491 protected files / 565,233,078 bytes across 38 roots remain unchanged,
including 13 prior retention-repair files, 18 preparation files and six spent
claims. Real data/independent-benchmark, collector and native/GPU limitations
remain unchanged; no real operation ran.

The coherent checkpoint contains model, optimizer, Python/NumPy/Torch RNG,
explicit training generator, data cursor, completed counters, source/config/
candidate identity and original deadlines in one bounded atomic blob. Partial
serialization or failed replacement retains the previous complete checkpoint.
The spent highwater journal is independent of the blob; mismatched counts,
pending work, partial journals or expired original time refuse recovery rather
than pairing new weights with old counters. A recovered account cannot reserve
work or overwrite with random weights until the full boundary is restored.
Selected evaluation loads weights only and cannot rewind counts/RNG after the
frozen evaluation. The recovery API is tested; automatic continuation and a
budgeted real model reload/resume protocol are deliberately unavailable.

## Verification

Use the existing locked environment, without installs or downloads:

```powershell
python -B scripts/check_gpc_adaptation.py --report artifacts/new-check-name.json
python -S -B scripts/check_repo.py
```

The CPU harness denies CUDA initialization and real GPC source imports. It tests
synthetic official-format data/held isolation, source-equation loss and gradients,
intended parameter updates/frozen buffers, a complete two-update CPU substitute
pipeline, short last batches and caps smaller than a batch, first-error stops,
independent evaluation RNG, raw AR/TF timing, joint checkpoint recovery and save
faults. Native ownership/resource tests use fakes, not an actual process or GPU
preflight. See preparation evidence JSON for exact results and limitations.

The original `92c44cf` combined run passed **69/69 tests**, with no failures, errors or skips,
in 15.131 seconds (`final-cpu-tests-01.json`). This is CPU fixture timing, not GPC
training throughput. It includes 14 fake guardian checks and an entry-level fake
heartbeat/CPU pipeline check. Independent source review found two boundary bugs
(insufficient independent windows discovered too late, and selection fsync
crossing the training deadline); both are repaired and regression-tested.
Source/config/Markdown checks pass. The preservation audit confirms 7,460 old
files / 547,169,548 bytes across 36 roots unchanged, including both initialization
checkpoints and all six spent claims. No real operation or native preflight ran.

The acceptance repair passed 14/14 pure standard-library numeric checks with
Torch absent, and 80/80 combined tiny-CPU/fake checks with CUDA initialization and
real GPC imports denied. Repair evidence is separate in
`artifacts/wm-l-adaptation-retention-repair-20261025/`, including the failed first
synthetic check and its corrections, final reports and exact counterexamples.
All 7,478 protected files / 556,052,086 bytes across 37 roots are unchanged,
including all 18 original preparation evidence files and six spent claims.
No old experimental result or preparation evidence was overwritten. The real
data/independent-benchmark gaps, collector prerequisites, 500-sample exposure audit,
and unvalidated native/GPU path remain unchanged. No real operation was run.
