# Offline L contact, rotation and rollout diagnosis

Base `c8f16ed1b5a8e95a92aa8fd07b8ef4c5efdf4a24`, branch
`codex/wm-l-dynamics-repair`, 2026-10-25. This analyzes the completed
[v3 comparison](GPC_INPUT_COMPATIBILITY_RUN_V3.md), existing RGB/NPY and saved
real branches. New model loads/forwards/predictions, physical simulation/reset/
control, policy calls and training updates are zero. CPU image reading, frozen
measurement and synthetic-array checks are offline work. Original scores,
thresholds, weights, claims and ledgers remain unchanged.

Reviewed pixels support generation failures: A can lose the object or turn L
into a bar; B later grows a foot/crossbar, under-tracks rotation and misplaces
the visible blue pusher. Feedback compounds some errors, but failures persist
with real TF context. Saved input/action reconstruction found no local timing
mismatch. A metadata defect is repaired; these findings do not justify lowering
shape thresholds or replacing rejected poses with truth. Borderline/unreviewed
generated-style cases remain uncertain rather than automatically classified as
dynamics errors.

## Inputs and denominators

Evidence root: `artifacts/wm-l-dynamics-diagnosis-20261025/` (local/ignored, not a
published asset). Before seal: 7309 files / 482,389,522 bytes / 35 roots, including
the earlier 895 files, all 6386 v3 formal files and all 28 v3 audit files, with six
spent claims. Before SHA-256:
`e5f9f202f2a433df1f38c5a9a8ee8ff899393d021f600ae3d4ec92b3ff3f2794`.

L has 48 planned branches / 12 anchors, 47 available, with L01/hold still missing.
Each AR arm/horizon has 144 planned / 141 available rows: three noises repeat
each physical branch, not 141 independent scenes. Errors expose row P90 and
seed-mean within branch, candidate-mean within anchor and equal-anchor summaries.
Missing truth, rejected measurements and missing records have separate counters.

Contact strata mean any true contact **through H4**; H1 need not already touch.
Rotation is strict wrapped actual change >5 degrees relative to latest real pose,
separately at each horizon. The rotating stratum exists at H3/H4 with nine rows /
three branches / three anchors; earlier cells are N/A. Anchor strata overlap.
TF reuses AR-H1, independently samples H2/H4 and has no sampled H3.

## First failure and physical error

Joint object/full-shape valid counts for B:

| Stratum | AR H1 | AR H2 | AR H3 | AR H4 | TF H2 | TF H4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All L: 141 rows/47 branches/12 anchors | 141 | 113 | 91 | 78 | 118 | 103 |
| Contact by H4: 27/9/9 | 27 | 3 | 0 | 0 | 4 | 1 |
| Noncontact: 114/38/12 | 114 | 110 | 91 | 78 | 114 | 102 |
| Rotation >5deg: 9/3/3 at H3/H4 | N/A | N/A | 0 | 0 | N/A | 1 |

Nonrotating cells have 141 rows at H1/H2 and 132 rows/44 branches/12 anchors at
H3/H4. B-AR valid counts are 141/113/91/78; TF H2/H4 counts are 118/102.
The complete machine-readable cells preserve these changing denominators.

B-AR has 63/141 noise streams with first joint rejection: H2=28, H3=22, H4=13,
spanning 21 branches/11 anchors. TF has 38/141: H2=23, H4=15, spanning 13 branches/
9 anchors. First rejection is separate from later endpoint validity; later
recovery, if present, is retained. Three seeds never create new physical scenes.

All nine contact branches first touch at H2. First shape rejection is concurrent
in eight branches/24 streams, one step later in L04/toward/three streams, and
never earlier in this saved set. Among 38 noncontact branches, 12 branches/36
streams also become shape-invalid, while 26 branches/78 streams retain shape
validity through H4.
The original missing branch/three streams is separate. This temporal association
does not establish a contact causal effect or eliminate other failure mechanisms.

B-AR pusher P90 at H1/H2/H3/H4 is 1.744/3.521/5.686/14.383 world pixels despite
valid recognition. Rotating H4 rows have 27.503px P90. TF-H4 reduces these to
1.315px on all 141 rows and .886px on the rotating nine. Full-row TF-H2 P90 is
1.963px; v3's older A/B-common diagnostic reported 1.954px on 140/141, a different
subset. First >5px pusher error affects 16 AR streams/6 branches/3 anchors,
versus 6 TF streams/2 branches/1 anchor. Small surviving object-pose P90 cannot
conceal zero measurable AR contact objects at H3/H4 or only 1/27 TF contact and
1/9 TF rotating objects at H4. Rejected candidate poses remain diagnostic.

Row-level B-AR pusher P90 by physical stratum (world pixels):

| Stratum | H1 | H2 | H3 | H4 |
| --- | ---: | ---: | ---: | ---: |
| Contact through H4 | 1.666 | 4.106 | 7.853 | 12.175 |
| Noncontact | 1.939 | 3.502 | 5.460 | 11.783 |
| Rotation >5deg at that horizon | N/A | N/A | 16.711 | 27.503 |
| Rotation <=5deg | 1.744 | 3.521 | 3.087 | 3.255 |

These are small repeated-noise empirical quantiles, not independent-case
confidence bounds. Branch/anchor means and all object/angle/coverage errors are
available separately; they never substitute errors for rejected measurements.

## Fixed visual evidence and attribution

Selection was frozen before RGB reads: seed 5478, all rejected contact/rotation
branches, first two AR-fail/TF-recover and both-fail branches, fixed noncontact
hold and matched hold controls. Deduplication selects 14 branches/56 pages per
layout. Each has RGB, measurement overlays, original-fit residual and palette
residual, showing latest/H1-H4 real frames, A-AR, B-AR and available B-TF.
Display clipping never enters measurement/feedback. Green truth is posthoc,
separate from fitted magenta geometry, cyan pusher and black goal.

| Case, seed 5478 | Evidence | Interpretation / limit |
| --- | --- | --- |
| L00/hold | A becomes a bar; B stays L | Appearance/input mismatch and stationary control, not contact validation |
| L10/toward | A has no visible object; B-H1 remains L before deformation | Absence is in generated RGB, not just a rejected template |
| L06/toward versus hold | Extra left foot/crossbar and missed rotation versus rigid hold | Saved-case action/contact-associated generation failure; no general causal estimate |
| L08/toward H4 | Blue blob visibly lags truth; pusher fit IoU .91969, error 27.667px | Detector follows the wrong generated blob, supporting prediction error here |
| L01/away, L02/away | AR grows a foot; TF-H4 is closer to L | Generated-context accumulation; TF is privileged diagnostic conditioning |
| L03/toward | Later AR/TF protrusion, geometry residual there, small palette residual | One-step mismatch remains with real context; color agreement does not repair geometry |

All 210 pusher remeasurements in each visual bundle agree with original positions/
validity; original object IoU reconstructs within 1.11e-16. No object refit uses
truth or a new ROI. An 84-frame B residual audit excludes RGB-fitted pusher discs
only diagnostically: L03/L06/L08 AR-H4 have 9.12%/3.37%/1.95% of residual within
the disc plus one image pixel; outside-disc IoU remains .687/.653/.676. Immediate
occlusion masking would not remove those large mismatches. These values never
change acceptance or infer hidden geometry.

No reviewed case conclusively demonstrates a correct rigid L rescued by a
segmentation/fit correction. Unusual color, soft edges, occlusion and optimizer
ambiguity remain possible in borderline/unreviewed images. Independent image/
pose evidence is needed before changing measurement. Residuals are not
probabilities; valid recognition does not guarantee physical accuracy.

## Saved input/action and feedback audit

All 1692 L records match reconstructed input/action/raw and source-provenance
hashes. H1 actions are [past0,past1,past2,future0], sliding through H4; absolute
targets use `a/511*2-1`. TF excludes its target. AR appends saved raw predictions
without clipping/T1 again. Missing AR predecessors block reconstruction and
never fall back to actual future RGB.

For AR, `source_context_sha256` identifies original real-history provenance;
`input_sha256` identifies the evolved input. Confusing them would be an analysis
error. Recorder ordering and checked upstream rollout agree. RGB stays float32
without external `2*x-1`; only actions use that normalization. Internal denoiser
clamp/quantization differs from feedback clamp. Raw-range asymmetry exists
upstream too; preview clipping is separate. All 20 pinned upstream hashes match.
Exact checkpoint training exposure/cadence remains unknown, so code agreement
does not establish training-domain compatibility or dynamics accuracy.

## B-only ranking versus common A/B ranking

All common A/B L anchor subsets are empty. This does **not** mean B cannot score
any complete anchor. Its own raw and stopping-consistent subsets are:

| Horizon | B complete anchors /12 planned | Included | Rejected/incomplete | Inseparable physical pairs within subset |
| --- | ---: | --- | ---: | ---: |
| H1 | 11/12 | All except L01 | 1/12 | 66/66 |
| H2 | 2/12 | L00, L04 | 10/12 | 12/12 |
| H3 | 1/12 | L00 | 11/12 | 6/6 |
| H4 | 1/12 | L00 | 11/12 | 6/6 |

Hold and uniform use exactly those anchors. B/hold/uniform rewards match up to
floating-point roundoff, with zero regret on these physically inseparable
subsets. Pair accuracy is undefined, not 100%. Across all planned L anchors,
H3 has nine physically separable pairs and B abstains on all nine; H4 has 12 and
B abstains on all 12, with zero scored. Each horizon also preserves three pairs
with unknown actual outcomes from the missing branch. T's easy scored pairs do
not validate L. TF is never ranked as deployable AR planning evidence.

## Coverage and termination remain unvalidated

Each primary horizon's 63 available physical outcomes contains no strict >.95
success (maximum .942857); L contributes 47. M has four actual positive
calibration cases per arm, but zero declared interval positives and 38/528
unknowns per arm. Unknowns remain unknown; zero declared positives leaves
precision undefined. These data do not validate successful terminal precision/
recall, generated-domain interval coverage or absorbing-state behavior.
Future checks need independent positive and negative trajectories near the
threshold, fixed interval calibration/validation partitions and reported
abstention coverage. No success label or threshold is changed here.

## Implemented repair and staged repair proposal

One existing measurement-record field changes: `compact_measurement` now retains
the decoder's actual `pusher.fit_iou`, which its old whitelist missed while
expecting `soft_iou`. This fixes evidence loss, not position, segmentation,
thresholds or scores. Old records remain immutable; full offline pusher evidence
is separate. Existing v3 forecast records are <=2633 bytes; one bounded numeric
addition fits their 3072-byte record limit. Future run validation is not claimed.

| Problem / uncertainty | Module and smallest next change | Acceptance / evidence | Execution class |
| --- | --- | --- | --- |
| Pusher fit evidence lost | `wm_compat_engine.compact_measurement`: retain `fit_iou`; archive full offline evidence | Field regression; original positions/validity unchanged; done | Offline interface repair |
| Common-subset-only reporting hides B coverage | `wm_l_diagnostics`: own/common subsets and matching hold/uniform; ties/rejections | Same baseline anchors; original missing identities; no TF ranking; done | Offline analysis repair |
| Possible timing/range wrapper fault | `wm_l_replay_diagnostics` and upstream review | All 1692 actual-input/action/raw hashes match; raw feedback unaltered; done | Offline audit; no preprocessing change justified |
| TF contact/rotation deformation persists | New adaptation module around the pinned phase-two denoiser; one candidate versus frozen B | Retain frozen screens; proposed >=90% shape, >=95% pusher recognition and <=5px/5deg/.05 error targets with full coverage; improve contact/rotation, not just hold | Proposal only; separately authorized training/predictions |
| Drift reduced by real TF context | Later one fixed short-rollout intervention versus one-step baseline | Same branches/noises; AR improves without sacrificing TF/T; no true future deployment input | Proposal only; new model execution required |
| Borderline generated-style segmentation not adjudicated | Keep frozen decoder; obtain independent masks/pose checks before occlusion/palette changes | Freeze on development data, validate untouched partition; old-score comparison and malformed negatives | Offline interface research first; no relaxed gate |
| L ranking and terminal support incomplete | Independent contact/rotation and near-success trajectories with separable candidate outcomes | Branch/anchor coverage; same-subset baselines; both actual outcome classes; no precision claim with zero positives | New data and separate collection authority |

Evidence supports considering weight/data adaptation, not its necessity or a
proved architecture defect. A concrete later feasibility pilot could cap one
candidate at 500 optimizer updates or 15 minutes, whichever comes first, batch1,
with a predeclared native memory ceiling and no automatic restart. One fixed
evaluation would retain at most 2268 predictions/6804 Euler calls for the existing
sampler, with no parameter sweep. These are proposed pilot ceilings, not measured
training feasibility or authorization. Native memory/allocator limits and a
trajectory split must be approved explicitly before that pilot; failure to fit
does not permit silently enlarging them. Fitting reused data would only test
optimization, not generalization.

Contact/rotation and near-success independent validation data are prerequisites
to a scientific repair claim. Training state/coverage labels would be disclosed
privileged supervision, never substituted for predicted states/rewards at test
time. Architecture replacement is a later alternative if a controlled small
adaptation fails on adequate independent cases; this report does not justify it
from threshold rejection alone. Reused 47-branch data and noise repeats cannot
supply an untouched validation set or prove that a large architecture change is
necessary. None of the proposed new training, evaluation or collection runs here.

No measurement algorithm/threshold was tuned, so no new decoder-calibration pass
is claimed. Synthetic fixtures are mathematical regression checks, not an
independent real/generated-image validation partition.

## Reproduction and evidence index

Use the installed project interpreter; no install or download is needed:

```bash
python -B scripts/diagnose_wm_l.py --output-dir artifacts/wm-l-dynamics-new-review --replay-inputs --figures
python -B -m unittest discover -s tests -p 'test_wm_l_*.py'
python -B -m unittest discover -s tests -p 'test_wm_compat_engine.py'
python -S -B scripts/check_repo.py
```

The derived directory must be new and outside immutable input roots. Optional
`--residual-selection` points to the already frozen selection JSON and emits
diagnostic residual partitions without object refits. Evidence paths below are
relative to the new evidence root:

| Artifact | Content |
| --- | --- |
| `before.json`, `baseline-seal.json` | 7309-file/35-root preservation baseline |
| `analysis-final/` | Final numeric-only statistics, first failures, own/common ranking and contact-onset timing |
| `analysis-cli/conditioning-replay.json` | All 1692 saved input/action/raw checks |
| `analysis-cli/residual-localization.json` | 84 B frames; pusher-local residual sensitivity, no changed eligibility |
| `analysis-cli/manifest.json` | Integration run, 20.031s, exact executed-source hashes and zero new real operations |
| `figures-v2/selection-frozen.json`, `visual-cases-v2.json` | Fixed rules, 14 cases, diagnostics, per-PNG identities and bounded visual observations |
| `figures-v2/*-rgb.png` | Raw/display/real context sequences, explicit stream and horizon per panel |
| `figures-v2/*-measurements.png` | Separate original candidate, pusher, target and posthoc truth outlines |
| `figures-v2/*-fit-residual.png`, `*-color-residual.png` | Separate geometry/color mismatch, fixed display scales |

Earlier root/final/final-v2 numeric exports and the first 56 PNGs remain as
intermediate evidence. The first layout was visually ambiguous despite correct
numeric identities; revised titles name each stream/horizon and distinguish H1
pass from H2 rejection. A rendering regression inspects actual axis titles.
The v2 layout preserves all 210 numeric frame diagnostics exactly.

The CLI figure-module hash predates that layout revision and is retained as the
executed version, not falsely relabeled as current code. Numeric/residual
calculation did not change with the layout-only revision. Final numeric CLI and
v2 visual manifests bind their own final sources. Original scientific files
and formal records are never rewritten by these derived exports.

The final focused suite passed 33 tests in 6.744 runner seconds (zero failures,
errors or skips). It includes the new numeric, visual, reconstruction and
residual tests plus the existing fake paired-engine checks; it imports no Torch,
Pymunk, Pygame or policy runtime. Source/link, message and whitespace checks are
recorded with the local commit. The local `review.md` and `evidence-index.json`
provide clickable case pages and exact numeric artifact hashes.

Closing preservation passes for all 7309 files / 482,389,522 bytes / 35 roots:
zero added, missing, changed files or root-membership differences. All six claims
(7031 bytes), both weights and original v3/old audit evidence remain intact.
`after.json` SHA-256 is
`e72af84a54c4d5b5b845d2eb1dcfaf7fcc3cf09dccade69c8878109d646aee04`;
`reconciliation.json` SHA-256 is
`9aa76f169f710017a6ae17753d5024a6b62bb8fd725592201df875c93c3af180`.
The before inventory's own hash is unchanged. The only existing-source change
is the pusher metadata field; forward, decoder/config/thresholds, guardian,
storage, budgets and execution identity are unchanged. Derived CPU diagnostics
are separate from the preserved experiment and its spent authority.
