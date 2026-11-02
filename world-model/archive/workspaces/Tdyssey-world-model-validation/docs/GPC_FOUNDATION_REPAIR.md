# Manual world-model foundation repair

The reported duplicate pusher is reproducible in checkpoint500's raw H1 output.
The initial checkpoint predicts one pusher on the same reconstructed input.
A direct call to the pinned upstream sampler gives exactly the same adapted H1
pixel values as the local wrapper (maximum absolute difference0). This excludes
a UI overlay or this wrapper as the source of that duplication in this case;
it does not establish why adaptation produced the bad learned function. Both
paths share the loaded model, so this is not independent model-construction
validation. The neural prediction problem remains unresolved. No retraining or
conditioning experiment was started.

## Reproducible cases and bounded verification

Evidence is in `artifacts/gpc-foundation-repair-20261026/`. `start.json` and
`baseline/` preserve the actual dirty starting tree atcbb87ef2. Existing unfinished
conditioning files remain byte-identical and outside the change/commit.

The supplied38.2s/382-frame video was decoded at16 predetermined uniform
positions; `video-review/receipt.json` records the actual local file and identity.
The file currently has the basename `121126-971032.mp4`, while the historical log
and user reference retain the `manual-` prefix. No video content was modified.
The contact sheet shows pusher-position and object-pose discrepancies. It cannot
recover unrecorded model inputs or prove a historical sampling-call count.

`real-plan.json` froze two L cases from the manual event log before execution:

| Case | Request/predicted/executed log lines | Captured history endpoint |
| --- | --- | --- |
| Duplicate pusher |440/441/442|control44, historical H1 |
| Noncontact comparison |224/225/226|control7, historical H4 |

Four full logged states were restored and rendered without advancing physics.
These are reconstructed inputs, not a claim that the missing historical image
bytes or source identity were recovered. Both weight roles receive identical
real histories, past actions, target and seed5478. The first case's H2-H4 are a
new extension of the historical H1 action. Each case executes its real future
once; adapted predictions reuse the detached input without further simulation.

One Python diagnostic completed in19.125s: two serial model loads,
16 rollout forecasts plus one direct-native parity forecast,51 reserved Euler
steps,14 real controls including six initialization holds, and two resets/two
reset-settle substeps. There were zero training updates or new adaptation-data
records. Peak private/working-set usage was2177953792/1296449536B; CUDA allocated/
reserved peaks were109831168/136314880B. Both cases' four real futures had zero
contacts and zero object translation, so this run does not validate contact
dynamics, general T retention, or the historical missing-arm example.

The external PowerShell footer then failed because `retry=false` lacked `$`.
Consequently the overall wrapper returned1 and its final timing/peak receipt was
not published. This is an actual engineering failure, not a clean wrapper pass.
`run-real.executed.ps1` preserves the executed source and its frozen identity;
`real-external-failure.json` records missing outer fields as null and conservatively
charges300s against the600s real allowance. The child completion, complete
bundles, PNGs, measurements and native-parity receipt are intact. The footer and
already-spent admission check were corrected offline; there was no real retry.
No Python/Tdyssey process remained after exit.

## Findings

The H1 duplicate consists of a blue component1.81 world pixels from the input
pusher and another2.71 world pixels from the actual H1 pusher. The output retains
a pusher near the old location while also drawing it near the new location.
Both components are present in the raw generated RGB. They are not introduced
by changing gray/green into the fixed red/black display palette.

| Fixed case | Initial H4 pusher error | Adapted H4 pusher error | Adapted H1 pusher |
| --- | --- | --- | --- |
| Duplicate pusher |1.93|21.88|invalid: two components |
| Noncontact |1.66|6.62|valid single pusher |

Distances use the512-unit world coordinate system. Invalid H1-H3 positions remain
null; no candidate position is substituted for a valid estimate.

All sixteen saved rollout records verify four native calls, the requested action
window and raw generated feedback. The UI regression also verifies H4 uses
array index3, or the matched earlier physical-success prefix. Thus a pusher that
looks behind its actual H4 position does not imply a missing sampler iteration.
The adapted model's action effect is wrong in these cases despite four calls.

Initial shape gates pass all four horizons of the noncontact case and fail all
four horizons of the duplicate case (`rigid_shape_mismatch`). Adapted shape gates
fail all eight horizons (`fragmented_object`); adapted local ROI IoU is about
.793-.801 and .851-.857, respectively. Existing diffuse color contamination
affects these gates, but does not explain away the independently observed blue
duplication, pusher displacement errors or initial L shape distortion. Original
thresholds, invalid/null decisions and scientific acceptance remain unchanged.

Static audits found no proven local history-index, action-scale, control-time,
RGB channel, raw-feedback, role-loading or numerical sampler discrepancy.
The one-target local objective versus native multi-target AR training and the
history-noise/inference-conditioning difference remain candidates, not causal
findings. Upstream render-based history can include an action cross, unlike the
observation-image history here; its released-training provenance is unknown.
None of these observations justifies silently changing sigma, colors or physics.
Checkpoint500 remains rejected; reliable dynamics are required before imagined
policy adaptation. A next neural experiment needs a separately fixed causal
comparison, reviewed by the main conversation. It is not started by this repair.

## Repairs and checks

The manual demo retains its English two-panel click-to-predict-and-execute
workflow and double-click Windows launcher. The concrete changes are:

- Save complete immutable JSON/NPZ cases: four raw/T1 histories, full physical/
  controller states, actions, raw H1-H4 predictions, actual prefix and per-role
  checkpoint/RNG/input traces. Keep eight recent cases with at most three previews
  each, preserving the exact preview referenced by execution.
- C compares the other checkpoint on the captured input without another physical
  action. Ordinary checkpoint switches use the current live state, so they are
  not advertised as matched comparisons.
- Reserve model/call/Euler work before invocation, preserve spent work on partial
  sampling/decoding/rejection, and distinguish returned/accepted counts. Save
  failed requests and returned prefixes, including obsolete-result rejection.

Three accounting regressions failed against the sealed old predictor as expected.
The repaired suite passed20 tests, then21 after adding rejected-return retention;
no real Torch/weights/physics are used in those tests. A separate offline check
loaded both paired bundles and verified all sixteen raw/action/context traces,
without importing Torch or advancing physics. Recorded CPU stages including
video decoding use7.584s before final source/message checks, within600s; each
command is below120s. These checks validate the repairs, not learned accuracy.

Useful evidence: `real-01/L-double-pusher-horizons.png`,
`real-01/L-noncontact-horizons.png`, the corresponding `*-paired/` case bundles,
`real-01/summary.json`, `diagnostic-summary.json`, `real-01/native-parity.json`,
`real-01/final.json`, `real-external-failure.json`, and CPU receipts. Each figure's
columns are actual/initial/adapted and rows H1-H4. The new bundles are manual
diagnostic artifacts only, with `training_data=false`.
