# Saved-image interface revision v2: development protocol

Base: `11cb023`, using the same branch and existing environment. This revision
is exclusively numerical analysis of saved RGB/labels. It neither changes the
[old frozen protocol](FROZEN_WM_PROTOCOL_V1.md) nor retroactively passes its gates.
Every saved image has already been inspected or used in development; no unseen
validation claim, frame/noise random split, or policy-improvement claim is made.

## Hard boundary and asset preservation

New environment creations, resets, restores, real renders, controls, world-model/
policy inference, sampling and optimizer updates are all zero. Do not import
torch, pymunk, pygame or diffusion_policy in the offline entry. Reuse numpy,
scipy, PIL and existing geometry functions; do not install/download anything.
Never open/resume/reinitialize a training, evaluation or diagnostic budget.

Byte-audit all previous NPZ/JSON/PNG, histories, ledger/journal, reports and both
checkpoint files before/after. The audit also includes the 150 protected B/C
files. Old files and directories are immutable, including additions inside them.
Only new `artifacts/wm-interface-v2-*` output directories are allowed. Hashing
weights does not load or inspect a resident model. Shared source documentation
may add a versioned correction; original v1 configuration and decoder remain.

## Revised interface

`decode(rgb, shape_name, fixed_goal)` accepts no truth pose, contact, target action,
true coverage, future ROI or true stopping time. Goal coordinates only enter
coverage after pose fitting; they cannot initialize/crop/select the pose answer.
Known official L/T geometry and palette directions are disclosed priors.

Use continuous opponent-color projections, preserving antialiased fractional
color evidence. Red hue is a per-image nuisance direction estimated from the
strongest quartile of fixed red RGB seeds, with a fixed fallback; no truth or
goal enters this color estimate. Fixed visible-green background unmixing uses RGB alone, never
goal geometry. Distinguish blue chromatic direction from weak gray leakage.
Local image-derived ROIs accelerate deterministic rigid-template fitting, but
the final consistency check includes the **complete template and full image**.
Keep the v1 rigid-area range .72..1.35, consistency threshold .85 and fragment
mass threshold .95. Their soft-evidence metric differs from binary IoU; report
that measurement change, not a claim of equivalent calibration. Red outline/
blur are bounded appearance nuisance parameters; collision geometry is never
rescaled to make a malformed object scoreable.

The output has independent records:

- `state`: valid object pose or unavailable, plus an explicitly unvalidated
  candidate fit for diagnosing rejected images.
- `shape`: rigid consistency/area/fragmentation, full versus local fit scores,
  residual and bounded appearance parameters. Residual is not probability.
- `pusher`: independent position/validity. Multiple meaningful circles abstain;
  weak scattered chromatic leakage is not a second pusher. Missing pusher does
  not erase valid object/coverage; missing object does not erase pusher.
- `coverage`: point estimate from valid rigid pose and fixed goal; malformed
  shapes receive no task score. Its declared interval is separate from validity.
- `termination`: success/non_success/unknown with the exact strict >.95 rule.

Use a fixed .03 absolute coverage half-width as a **development engineering
assumption**, based on the previous 24-state maximum absolute error .025193
rounded up. It is not a confidence interval, a probability or a guaranteed error
bound for generated images. Do not derive confidence from fit residual. Report
actual interval containment and classification on saved real RGB afterwards.
Lower bound >.95 means conditional success; upper bound <=.95 non-success;
overlap/missing/inapplicable bound means unknown. Report the point-estimate
classifier separately so abstention cannot masquerade as corrected FN/FP.

## Offline comparison and denominators

Fixed input inventory: 24 calibration, 256 planned history, 256 future and 768
forecast rows = 1304. Available: 1284; the failed branch contributes 20 missing
rows. Cache identical RGB only to avoid repeated computation; retain every row.
v1 calibration/forecast measurements preferentially use the original stored
results. Numerical v1 measurement of saved real history/future is allowed.

Compare v1/v2 raw pose, angle, pusher, shape consistency, coverage and termination
independently. Retain raw endpoint measurements when earlier frames are missing.
For stopping-aware scores, use predicted termination only: unknown prior history
can block scoring but cannot delete later raw measurements. No truth stop repair.
Raw endpoint rankings are a separate diagnostic, not deployable task rewards.

Require each candidate's three fixed noise rows to have its mode's explicit
validity flag and finite number. Invalid diagnostic values cannot enter ranking.
Report scored, abstained and incorrect pairs separately. Observed pair accuracy
is correct/scored and is undefined with zero scored pairs. Conservative gate
credit correct/all eligible pairs is separately labeled. Missing rows never
disappear. Keep >.05 physical pair eligibility and 1e-6 tie tolerance consistent
with the previous protocol (eligibility at exactly .05 is included).

Aggregate noises within branch/frame, then weight anchors equally. Report L/T,
H1/H4, contact by H4, no contact, true rotation >5 degrees and near-.95. Posthoc
truth strata and error labels never enter measurement or answer selection.
Keep persistence/hold/uniform references on their own physical denominators.
At near .95 report TP/TN/FP/FN, unknown, missing truth and valid denominators;
precision is undefined without predicted positives. No small measured subset
is a passed generalization gate.

Case tables distinguish interface recovery/bias, shape/image ambiguity, measured
model state error, missed rotation and remaining attribution uncertainty. First
six matches in fixed inventory order are explanatory examples only, not selected
test success. Preview labels show actual coverage **and** strict success.

## Development record and execution

Before the full comparison, numerical probes inspected saved L/T calibration
and L06/T14 forecast examples. They motivated retaining color boundary values,
independent validity and full-template/multiple-pusher checks. Synthetic tests
use direct arrays/polygon mathematics, never an environment renderer. These
choices use development evidence and are not an independent model validation.
Freeze v2 source/config during the full offline comparison; record source hashes,
CPU wall time, output bytes, cache calls and all zero-operation counters.

```powershell
& $python -B scripts/diagnose_wm_interface_v2.py `
  --run-dir artifacts/frozen-wm-validation-v1-20261024 `
  --output-dir artifacts/wm-interface-v2-development
```

The next decision is whether the repaired measurement interface remains the
dominant blocker or evidence supports a separately proposed world-model
adaptation. This protocol authorizes neither adaptation nor a new experiment.

The [completed development result](WM_INTERFACE_V2_RESULTS.md) preserves the
first numerical output. Later metadata-only review labels full fit ambiguity
unassessed (unknown, never false) and omits inapplicable v1 interval-classifier
tables. It does not rerun or change any saved numerical measurement.
