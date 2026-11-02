# Independent review of the frozen v1 image interface

This review reads existing JSON/NPZ and source only. It performs no simulator
construction, reset, step, restore, rendering, model load, forecast, download or
decoder optimization. Numeric evidence is in `legacy-decoder-review.json`.
All original files were inventoried first in `before.json`.

## What the near-threshold collision establishes

`cal-T-10` and `cal-T-11` have true coverage 0.961905 and 0.923810, on opposite
sides of strict success. V1 returns the same pose and coverage 0.939254 for both.
Its **largest connected binary object component**, which is the entire object
fit input, is identical. The complete color masks already differ by two pixels.
The complete RGB arrays differ at 51 image pixels, and 35 differing pixels lie
inside a one-pixel dilation of the common RGB-derived object component. Maximum
channel difference is 0.745098; full-image RGB MSE is 0.000174899.
After excluding a two-pixel dilation of both RGB blue-pusher masks, 30 differing
object-boundary pixels remain. Thus the observed RGB difference is not solely
the correlated calibration pusher position. This exclusion uses RGB, not truth.

This establishes information discarded by v1 thresholding/component selection.
It does not establish that the 96-pixel RGB observation lacks the information,
or that a new estimator will necessarily recover it reliably. The latter needs
a frozen algorithm and measured errors, including missed/ambiguous cases. A
claim of RGB observational identity would require comparing complete RGB, not
the estimator's coarsened mask or its final pose alone.

## Failure labels describe a gate, not a physical cause

At H1, L has 128/141 `object_missing_or_area_mismatch`, 12 shape mismatches and
one pusher-circle mismatch. T has 20/48 area mismatches, 22 shape mismatches and
six valid decodes. At H4, L has only 4/141 valid decodes; T has 34/48.
These are the **first failing v1 checks**. They do not separate threshold/color
failure, soft edges, component filtering, generated deformation or actual loss
of object evidence. In particular, a color-mask area failure is not proof that
the generated image contains no object. The original prediction history and
its clip convention must remain available for this distinction.

## Constraints for an offline v2 repair

- Preserve continuous RGB evidence, including subthreshold boundary colors.
  Fixed disclosed gray/red appearance alternatives may be compared by image
  fit, rather than selected by true future state or by desired coverage.
- Keep the physical L/T geometry and goal convention unchanged. Renderer
  outline thickness can enter image fitting; it must not enlarge reward
  geometry. Fit pose from RGB, then compute geometric coverage against the
  fixed goal; avoid a goal-seeking pose prior that manufactures alignment.
- Keep malformedness and fit quality separate from the returned pose. Forcing
  every image onto a known L template can create plausible but unsupported
  rewards. Invalid, ambiguous and occluded cases must remain in denominators.
- True pose, pusher, contact, reward and ROI remain error labels. They must not
  choose a mask, crop, fit initialization, angle neighborhood, optimizer result,
  candidate or terminal crossing. Action targets must not substitute for true
  pusher position. A wiring test should change labels/metadata while preserving
  the RGB and verify that image estimates remain unchanged.
- Continuous pose/coverage error does not establish strict-success precision.
  Validate near-0.95 TP/FP/FN/TN and explicitly uncertain predictions. If coverage
  intervals are introduced, distinguish observed developmental calibration from
  an established statistical or deployment guarantee. Never lower the strict
  physical success threshold.
- These calibration images and model branches have already been inspected.
  A predeclared reuse split is developmental evidence, not an untouched final
  test. Split by physical state/anchor where possible, not by correlated noise
  replicas or neighboring frames from the same branch. Do not choose examples
  after seeing the new decoder's success.

The v1 score rule requires every preceding predicted observation to be
interpretable until a model-only terminal crossing. A repaired H4 pose alone
does not repair an ambiguous H1 stopping history. Report endpoint pose
measurement and terminal sequence scoring separately, preserving abstention.

The audit script uses only the standard library. Six focused tests pass under
`python -S -B`, including opaque non-NPZ/non-pickle assets, same-length rewrites,
deletion/addition, and refusal to write audit reports inside protected roots.
