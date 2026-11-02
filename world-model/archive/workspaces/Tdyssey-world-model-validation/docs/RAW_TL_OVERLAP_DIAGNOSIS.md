# Saved raw T/L overlap diagnosis

This offline-only task investigates the current raw2328 model's grey object
missing/green replacement at overlap with the goal. It reads the user's saved
H1 cases and existing source-train metadata. No GUI/model/physics operation,
new prediction, H4, training or collection is authorized by this task.

The fixed two-command plan and complete local evidence live under
`artifacts/raw-tl-overlap-diagnosis-20261027/`; the source-only helper is
`scripts/diagnose_raw_overlap.py`. Source inputs, old results, cases, screenshots,
logs and thresholds are not modified. Weight contents are not read or rehashed.

The saved directories `view-001009-813929-case` and `view-001013-708426-case`
under `artifacts/manual-official-world-model/20261026-235943-407588/` contain the
same request/input/output. Both bounded commands are complete. These S captures
show the same class of defect, but are not asserted to be the exact timestamps
of the user's earlier screenshots/video.

## Saved-case localization

Command one completed in1.110s, observed peak working set85,856,256B. Both saves
are one exact case: request key
`261af8c90be57aa6afd77582e485049a8ef43bcc2312438660ba70b7c4aad8c1`,
input hash `79449368d476192a49acda8c65c6974cf36583a7f375e2f9ffd035aaba2e19e9`,
prediction hash `cb777ac7342827618856a02db076b94d12ec184ce7a9813a5be6eceed15db35a`.
The archive, sibling prediction NPY and saved UI right panel agree exactly
(zero differing UI pixels under the source's floor nearest-neighbour scaling).
The second S capture is not a second scene, prediction or statistical sample.

Both archives pass the existing case/request/hash validator. Four real history
frames84–87 and three intervening controls condition one target; execution
performs87→88 once. Raw/model history and truth arrays are identical, T1 is
absent, and the trace binds the actual input/output hashes and four actions.
The recorded checkpoint identity is raw2328 SHA
`79c1fe8b9eaa44b80f8c90ae27f3280b84a01456aa3efd6ebc75dda253558c05`;
weight contents were not reread or independently rehashed in this task.

Concrete panels under the result directory:

- `case-input-truth-prediction.png`: the four actual input frames, actual88 and
  unchanged saved prediction88.
- `case-overlap-crops.png`: actual/prediction crops, separate truth-only regions
  and the locations where actual gray becomes nearest-palette green.
- `actual-96.png` and `prediction-96.png`: direct saved-array previews.

The true L target coverage is.53854401; all four input states are.49559097.
Using known truth geometry and the predeclared fixed palette/margins:

| Region | Actual gray pixels | Predicted gray retained | Predicted nearest green |
| --- | ---: | ---: | ---: |
| Conservative overlap interior | 22 | 14 | 8 |
| Conservative nonoverlap interior | 4 | 4 | 0 |

All26 scored pixels are gray in the true image. The eight green classifications
are96-pixel coordinates `(39,56),(39,57),(38,58),(39,58),(40,58),(40,59),
(41,59),(41,60)`, at the L bend/adjacent stem. The raw prediction already shows
green there. This confirms a generated-image defect, not a UI-only composition
or capture mismatch for this case. It does not identify the learned cause or
prove pure recolouring: displaced/deformed geometry can also reduce gray.

The object centre mask has190 pixels; one-pixel erosion and pusher exclusion
leave88, of which62 near the goal boundary are excluded from the conservative
comparison. The nonoverlap denominator is only4. These are local diagnostic
counts, not population rates, fitted poses or model-predicted coverage. The
pixel-centre/edge convention approximates rasterization; no decoder thresholds
were tuned and no source/prediction pixels were repaired.

## Actual training exposure

All2,328 unique scheduled target states were recomputed from the24 source-train
JSONs using existing official geometry. All2,328 values exactly match the
recorded collection coverage labels (maximum delta0). No new development,
independent-validation or held payload was opened.

| Shape | Targets / scenes | Coverage0 | 0<coverage<.5 | .5<=coverage<.8 | >=.8 | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T | 1,164 / 12 | 888 / 11 scenes | 276 / 6 scenes | 0 / 0 | 0 / 0 | .49420920 |
| L | 1,164 / 12 | 802 / 12 scenes | 362 / 7 scenes | 0 / 0 | 0 / 0 | .36257830 |

Coverage>=.1 occurs in118 T windows/3 scenes and204 L windows/5 scenes. Neither
shape has targets>=.5,.8,.9 or.95; strictly>.95 success is also0. The failed
manual history and target are above every L training target's overlap. This is
a concrete support gap; increasing exposure to all existing windows did not
create any high-overlap or near-success target.

The frozen first-trajectory/anchor selection contains four windows (two per
shape), not eight: both upper bins are empty and are not filled from other
splits. Command two decoded exactly20 selected train frames, not full float
episodes. NPZ streaming inflated/skipped2,156,544 prefix bytes for the selected
late T window; this IO is explicitly disclosed rather than claiming only the
selected compressed bytes were read. Only the selected five frames per window
became arrays; source compressed-file hashes were also verified.

All four five-frame panels were inspected. The four targets have396 scored
actual-gray interior pixels and0 nearest-green pixels:395 nonoverlap pixels and
only1 conservative overlap pixel (in the L.08007-coverage window). This offers
limited empirical support for source-label integrity, alongside the consistent
source draw/collection code; it is not a large-overlap label audit. Empty high
bins cannot be repaired by borrowing held scenes. No evidence here establishes
a wrong training target palette/order, but uninspected archive pixels are not
independently certified by this sample.

## Source evidence and remaining uncertainty

The pinned renderer draws LightGreen goal polygons first, then the agent/block
(`pusht_env.py:197–205`); the gray object should hide green beneath it.
`pymunk_override.py:193–209,245` supplies fixed gray fill/outline colours.
`official_collection.py:116–144` pairs action-after RGB with the transition's
after state; `gpc_adaptation_data.py:247–266` uses the fifth raw frame as target.
The raw mode does not recolour targets. No deterministic incorrect draw order,
T1 application or off-by-one target bug is established by this evidence.

`gpc_supervised_adaptation.py:168–175` minimizes whole-image residual MSE. It
contains no rigid-shape, object/goal layer or occlusion-consistency constraint;
the native sampler returns free RGB. Those facts explain why improved overall
MAE/loss does not guarantee local geometry, but do not establish training-loss
causality. A saved output alone cannot separate recolouring from geometric
deformation or attribute the problem to one optimizer choice.

Two remaining explanations are: (1) overlap-specific appearance/occlusion
generalization failure, with the measured L coverage support gap as direct
evidence; (2) a more general contact-motion/shape prediction error, which can
also expose green where gray should be. Neither is proved causal.

The smallest discriminating follow-up to consider is one matched H1 goal-only
intervention: keep the recorded object/pusher histories, action and seed fixed,
render the original goal and a nonoverlapping relocated goal, and compare two
raw2328 forecasts. A recovered gray bend only when the goal moves would support
an overlap-conditioned error; persistent deformation would support the broader
motion/shape explanation. This requires separately authorized renders/two
forecasts, after any user GUI conflict is cleared; it is a proposal only. No
training, collection, H4, production fix or DPPO admission follows from this task.

## Execution and handoff

The two CPU commands took1.1101767s and1.2293019s (2.3394786s total), each with
two-core affinity; peak observed working set85,856,256B. The second command also
passed source syntax/config/link and prepared-message checks. The complete
diagnostic directory was5,012,552B before the final delivery summary, well below
16MiB, with no numerical temporary full-episode arrays. All installed
Torch/pygame/pymunk/CUDA/SDK imports, model loads, forecasts, backward/optimizer,
physics/collection and GUI operations were0. A blocking import hook rejects the
neural/simulator runtimes. The user's current GUI was not touched or polled.

The current event log was read once, using a bounded byte-length snapshot of
at most64KiB; subsequent GUI activity is not this diagnosis's work or evidence.
No old case, screenshot, log, threshold, checkpoint, claim or training output
was modified. Only this helper, report/status and small ignored diagnostic
artifacts were added. No production/model/loss repair is supported or attempted.
