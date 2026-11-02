# Saved-image interface v2 development results

The interface now separates object, rigid-shape, pusher, coverage and termination
measurements, but **stopping-consistent model action scoring remains blocked**.
Real-image bias is smaller and pusher measurements survive malformed objects.
The generated object passes the frozen v2 shape check in only 104/756 available
forecast images, and in no H1 image, including T. No model, success-precision,
contact-dynamics, adaptation or generalization gate passed.

This is reuse of already inspected development data. The [versioned
protocol](WM_INTERFACE_V2_PROTOCOL.md) preserves the original
[v1 protocol](FROZEN_WM_PROTOCOL_V1.md) and [result](FROZEN_WM_VALIDATION.md).
Base `11cb023`; measured implementation `ab01b9e`. Subsequent changes correct
reporting metadata only, with no new decoding or numerical measurement changes.

## Scope and evidence identity

Only existing saved RGB/NPZ/JSON, pure geometry, and mathematical array fixtures
were used. New environment constructions, resets, restores, real renders,
controls, model/policy inference, sampling and optimizer updates are all **0**.
No install, download, model-residency check, ledger/budget reopen, or B/C work
occurred. The offline process imported none of torch/pymunk/pygame/diffusion_policy.

The fixed inventory retains 1,304 planned image rows and 1,284 available images:

| Source | Planned | Available |
| --- | ---: | ---: |
| Calibration | 24 | 24 |
| Physical history | 256 | 252 |
| Physical future | 256 | 252 |
| Saved forecast | 768 | 756 |

All 20 missing rows belong to the unretried original L01/hold branch. Three
noises share each physical branch; they are not independent physical samples.
RGB hashing cached computation while preserving every row. Actual image-decoder
calls were v1 208, v2 988; these are CPU image fits, not learned-model inference.
The main measurement/aggregation phase used **273.594 CPU seconds / 282.641 wall
seconds**. Preview/serialization and later report-only review are outside this
timing. The immutable main output is **10,999,946 bytes**, including its manifest.

Local, ignored evidence:

- `artifacts/wm-interface-v2-development/`: original per-image JSONL/CSV,
  inventory, strata, sequences, rankings, baselines, 30 case PNGs and manifest.
- `artifacts/wm-interface-v2-review/`: `review-metrics.json`, descriptive
  quantiles/common-subset errors, reviewed CSV exports, saved-RGB evidence plate,
  reproducible report-only scripts and Chinese review.
- `artifacts/wm-interface-v2-audit/`: front/back hashes and independent legacy
  decoder review. All 595 protected old files, 186,686,009 bytes, including both
  checkpoints and all 150 B/C files, are unchanged. V1 decoder/config are intact.

The main output manifest SHA-256 is
`4a9faf997fe75d372fe49115f59fe58b0bc6f415b56591a15a36399ef2f62050`.
It binds the original frozen numerical source/config hashes and every payload.

## What changed in the measurement

V2 retains fractional color boundaries and fits the disclosed rigid L/T geometry
with bounded appearance nuisance terms. The goal enters only after RGB pose
fitting. The final shape check uses the entire image and complete template;
an image-derived optimization ROI cannot hide a missing arm. Shape thresholds
retain v1's numeric limits, but soft IoU is a different metric from binary IoU
and is not validated as equivalent. Geometry is never rescaled to accept a
malformed image. Missing or multiple pushers are independent local results.

The old T10/T11 largest binary object component was identical, but the RGBs
differ at 51 pixels, including 30 object-boundary pixels after RGB-based pusher
exclusion. The old collision demonstrates discarded information, not intrinsic
RGB identity. V2 gives distinct coverage estimates .943795/.938770 for true
.961905/.923810. This distinction still does not resolve strict success.

Pose-fit alternatives are diagnostics from three coarse-survivor local fits,
not a full ambiguity search. `ambiguity_assessed=False`, `ambiguity=None`
explicitly mean unassessed. They do not certify a unique pose. Coverage intervals
are estimate +/- .03, a fixed **development engineering assumption**, without
probability, confidence, full pose ambiguity or generated-domain calibration.

## Real RGB error and near-threshold decisions

Every available real image is measurable: 24 calibration + 252 history + 252
future = 528 rows. These include repeated/correlated states, not 528 independent
validation samples. Errors below weight anchors equally, after branch/frame
aggregation; P90 is a descriptive image-row quantile.

| Calibration metric | L v1 | L v2 | T v1 | T v2 |
| --- | ---: | ---: | ---: | ---: |
| Position P90, pixels | 1.039 | .841 | 1.060 | .883 |
| Angle P90, degrees | .354 | .474 | .668 | .176 |
| Pusher P90, pixels | .821 | 1.054 | 1.175 | .805 |
| Coverage MAE | .008360 | .006988 | .005699 | .004651 |
| Maximum coverage error | .025193 | .019882 | .022651 | .018110 |

Improvement is not uniform: L angle/pusher P90 become worse. History coverage
MAE changes L .007983 to .003651, T .011202 to .009006. H4 physical-future MAE
changes L .006929 to .003750 (47/48 rows), T .009626 to .007177 (16/16).
The fixed +/- .03 interval contains truth in all 528 available real rows; this
is observed reuse-data containment, not a new calibrated guarantee.

For the eight calibration states within .05 of .95 (four actual successes):

| Classifier | TP | TN | FP | FN | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 point > .95 | 0 | 4 | 0 | 4 | 0 |
| V2 point > .95 | 2 | 4 | 0 | 2 | 0 |
| V2 declared development interval | 0 | 2 | 0 | 0 | 6 |

Strict success remains coverage **> .95**. Interval lower > .95 is conditional
success, upper <= .95 conditional non-success, otherwise unknown. All four true
successes remain unknown under this rule. Interval precision is undefined with
no predicted positives. Two correct point positives in reused data do not
validate precision; abstention has not corrected all four original misses.
V1 has no interval classifier: it is **not applicable**, not a second baseline.

## Generated images and attribution

| Horizon | L object v1 / v2, planned 144 | T object v1 / v2, planned 48 | V2 pusher L / T |
| --- | ---: | ---: | ---: |
| H1 | 0 / 0 | 6 / 0 | 139 / 48 |
| H2 | 11 / 3 | 29 / 24 | 141 / 48 |
| H3 | 8 / 5 | 34 / 34 | 141 / 48 |
| H4 | 4 / 4 | 34 / 34 | 141 / 48 |

L has three missing forecast rows per horizon. V2's total object validity is
L 12/564 available, T 92/192; v1 is L 23/564, T 103/192. V2 recovers no forecast
object that v1 rejected, and rejects 22 that v1 accepted. Its independent pusher
validity is L 562/564, T 192/192. This is a useful interface separation, not an
object-decoding pass. The accepted-object set did not grow.

V2 first failure reasons over 756 available forecasts: 453 area mismatch, 195
rigid-shape mismatch, four object missing, 104 accepted. These describe the
measurement gate, not a causal allocation between color, blur, shape deformation
and dynamics. A rigid fit may have interface bias; a rejected fit has no valid
task coverage. Its diagnostic candidate pose must not enter error or reward means.

Visual review of the saved-RGB evidence plate supports specific distinctions:

- L00/hold/H1 is a small red blob versus a full actual L (area ratio .445,
  soft IoU .380). L06/hold/H1 loses much of the vertical arm (.648/.493).
  These examples show generated shape/identity loss beyond the old pusher gate.
- T13/hold/5479/H1 still looks T-like but v1 accepts while v2 rejects at soft
  IoU .784. Appearance mismatch, deformation and interface false rejection are
  unresolved here; rejecting it is not proof of physical dynamics error.
- L06/toward/5480/H3 is measurable: position error 7.223 px, angle error 4.461
  degrees, actual rotation 10.054 degrees. Coverage .262077 versus truth
  **.265205**, error .003128, shows that small reward error can hide pose error.
- T13/toward/5478/H2 is measurable with soft IoU .922, position error 6.041 px,
  coverage .333023 versus .288212. This is one of four valid forecast intervals
  that miss truth; all four are T13/toward H2/H3, seeds 5478/5480. Overall
  forecast containment is 100/104 measured rows, not 100/756 coverage.
- T14/toward/5478/H4 predicts visible rotation: true -11.455 degrees, angle
  error 1.073 degrees. Thus rotation is not uniformly absent from the model.

At H4, L contact object coverage is 0/27, true-rotation >5 degrees 0/9; no
numerical contact/rotation accuracy follows from those rejected images. T has
6/9 contact rows measurable; its rotating subset is 3/3 measured but represents
one physical branch with three noises, not three independent rotation tests.
The evidence cannot isolate appearance domain from dynamics or uniquely assign
the remaining image error to the model versus estimator.

## Endpoint ranks, stopping prefixes and independent references

An unknown earlier predicted stop blocks a stopping-consistent score; later
valid raw endpoint measurements remain recorded. True futures never repair that
prefix. V2 has no eligible stopping-consistent H1..H4 score, so all **16/16
anchors abstain** from complete three-noise candidate selection.

H4 raw-endpoint selection is a separate diagnostic. L completes 0/12 anchors;
all 12 physically separated pairs abstain, with three other actual pairs unknown
because of L01/hold. T completes T12/T13 only (2/4); T13 scores three eligible
pairs correctly, T15's other three eligible pairs abstain. T12 has no separated
pair. Thus observed T accuracy is 3/3 on one anchor; conservative equal-anchor
credit is .5. L observed accuracy is unavailable, conservative credit 0. Zero
credit for abstention is an engineering bound, not an observed model mistake.

Both selectable T anchors choose hold, observed regret 0. Hold on the **same**
two-anchor subset also has regret 0; uniform expected regret is .013205. These
are not superiority over hold. On independent complete physical denominators
(L 11/12 anchors, T 4/4), hold/uniform regret is L .006258/.037622 and T
.010070/.043881. Do not compare these full-denominator means with selected-subset
model regret as a performance claim.

Privileged observed-pose persistence remains available independent of decoder
success: H4 coverage error L .031618 (47/48 branches), T .038846 (16/16).
H1 persistence error is exactly 0 because these saved first steps have no object
motion. H1 also has no >=.05-separated physical candidate pair. The small
measurable model subset cannot establish a gain over persistence or constant
reward. Baseline CSVs retain every physical denominator.

## Reporting correction, verification and next decision

The first automatic export is preserved byte-for-byte. Its v1
`declared_interval` rows merely duplicate point stopping and must not be used as
interval evidence. The reviewed confusion CSV removes those 88 rows. Reviewed
image CSVs replace unassessed `ambiguity=false` with unknown plus an explicit
assessment flag, and label missing inputs explicitly. Neither correction changes
any pose, validity, coverage, sequence score or ranking. The source now emits
the corrected metadata; `reporting-corrections.json` binds both source versions.
`final-source.json` records the final metadata-only source hashes and local commit.

Verification: **30 focused tests pass**, including mathematical RGB fixtures,
strict interval equality/unknown-prefix/absorption, independent validity,
multiple pushers, full-template missing-arm rejection, forbidden imports, goal
separation, noise/anchor aggregation, explicit ranking eligibility and asset
tampering. Repository source/link, commit-message and whitespace checks pass.
No full installed discovery was invoked because old tests include learned-model
prediction calls outside this offline scope.

Reproduce measurement with the existing environment and a fresh permitted output
directory using the CLI command in the protocol. The review scripts in the local
review directory operate on saved outputs only; they never call the decoder.

The repaired API is useful, but generated early-frame shape validity and precise
termination still block imagined rewards. Because H1 also fails for T while T
H4 is often measurable, the next proposal should prioritize official-input
visual-domain compatibility and early shape preservation alongside L geometry.
Separate appearance/estimator checks from action-conditioned contact/rotation
checks, then validate stopping and ranking on independently reserved scenes.
Existing evidence does not determine whether color preprocessing, a different
representation, data adaptation or weight fine-tuning is sufficient. This is a
proposal only: no further sampling, inference, adaptation or experiment began.
