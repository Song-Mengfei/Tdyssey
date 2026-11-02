# Checkpoint500 offline diagnosis

This is a **partial diagnosis, stopped on its first numerical-command error**.
All3996 JSON summaries and10 synthetic checks completed. The fixed pixel command
saved32 panels, then failed serializing its final numerical report. No retry,
repair execution or fourth command followed. The checkpoint remains rejected;
no model, production code, data, score or gate changes.
Evidence: `artifacts/gpc-checkpoint500-offline-diagnosis-20261026/`.
Base: `a461829c6bdf967c0bfab60e621e8c286465c68a`, same checkout/branch.

## Frozen samples and complete measurements

`analysis-plan.json` precedes pixel decoding. `fixed-samples.json` is252831B,
SHA256 `193096b5f23df1bec4e8f2ed73df5d15fbc4f2ef9c3edc571ad9701eb61a7ed7`.
Selection is lexicographic `(anchor,candidate)` within each shape/stratum/physical
contact cell. All eight cells are nonempty; no outcome-based replacement occurs.

| Shape | Stratum | Noncontact first branch | Contact first branch |
| --- | --- | --- | --- |
| T | Regression | T12/away | T13/toward |
| T | Independent | IT00/away | IT01/toward |
| L | Regression | L00/away | L03/toward |
| L | Independent | IL00/away | IL01/toward |

Each supplies seed5478, AR H1/H4 and TF H4, frozen/adapted:48 existing predictions
and64 real history/future frames. Train inspection selects the first two distinct
windows in original500list order within each shape/contact cell: cursors24,26/0,2
for T noncontact/contact;31,33/1,3 for L. Only five frames per window, at most40,
may be decoded. Output order groups cells; original cursors remain unchanged.
This panel is diagnostic, not a new statistical validation set.

All3996 measurement JSON files match the sealed inventory. The compact
`measurement-summary.json` crosses weight/shape/stratum/AR-or-TF/horizon/contact/
>5-degree wrapped physical turn byH4. Every invalid/null/unknown remains, with
physical anchor/branch denominators and noise-repeat row counts. No candidate
pose becomes an accepted pose and no score is recomputed. For example, adapted
independent L noncontact H1 has60 rows/20branches: component-fraction median.3205,
ROI/full IoU medians.8130/.2997, and0 valid object/coverage estimates.

## Train and inference contract

Define `o_i` before absolute world target `a_i`. Training anchor`t` uses images
`[t-3:t+2]`, actions`[t-3:t+1]`: `o_(t-3)..o_t -> o_(t+1)` with
`a_(t-3)..a_t`. See [dataset](../tdyssey/gpc_adaptation_data.py) line237 and
[objective](../tdyssey/gpc_supervised_adaptation.py) line168. H4 uses actions`t..t+3`;
TF context is real frames`t..t+3`, excluding target`t+4`.
See [rollout](../tdyssey/gpc_adaptation.py) line385.

Real RGB is float32 CHW3x96x96 `[0,1]`; screen/world `(x,y)` actions are normalized
once by`a/511*2-1`. T1 applies once to real train/history/TF frames; raw AR feedback
bypasses T1. Clipping in old MAE/measurement code is diagnostic, not feedback.
Checkpoint500 extracts`payload['model']`, strict-loads keys/shapes/dtypes and uses
`.eval()`. Training uses`.train()` and all`inner_model.` parameters; Fourier
frequencies are buffers. Eval restores no optimizer/RNG/account.
See [loader](../tdyssey/gpc_evaluation_assets.py) line73 and
[constructor](../tdyssey/gpc_supervised_adaptation.py) line254.

Cached GPC remains`a96146e568826ef46134f73c22fdf59a675e7264`. Local loss is the
first`t=5` term of official phase two. Both draw
`sigma=clip(exp(1.2*z-1.2),.002,20)`, add channel offset`.1*z_offset` plus pixel
noise`sigma*z_pixel`, and use native conditioners:
`s=sqrt(sigma^2+.1^2)`, `c_in=1/sqrt(s^2+.5^2)`,
`c_skip=.5^2/(s^2+.5^2)`, `c_out=s*sqrt(c_skip)`, log noise embeddings/4.
The residual target is`(clean-c_skip*noisy)/c_out`; native model output divides
context by`.5`. Cached exact file/line evidence is in`contract-upstream.md` and
`contract-data.md`; source semantics are checked beyond existing test results.

Declared differences remain: local training has one real-history target;
official16-frame training has12 autoregressive targets and quantized generated
contexts for later terms. Local gradient clipping is1; the pinned official loop
has none. Native inference uses three Euler steps, rho7/sigma5 to.002 then0,
churn0/conditional-noise0, returning raw final state. Denoised quantization uses
`[-1,1]`; this does not imply real RGB is converted to that range. These are
exposure/training differences, not established causes of H1/TF failure. No
definite local index/action-normalization/repeated-T1/checkpoint-mapping bug has
been found within the inspected scope.

## Pixel evidence and limits

The existing32 PNGs are under`pixels/`: eight comparison, eight segmentation,
eight history and eight train-window panels. Display explicitly clips/rounds
only for PNG presentation. After the error, only already-exported PNG/text and
source were inspected; no raw NPY/NPZ was reopened or statistic recomputed.

In the eight fixed comparison panels, adapted images retain recognizable red
T/L objects, blue pushers and black T1 goals. Their backgrounds visibly shift
toward pink relative to real T1 targets/frozen output, including H1 and TF H4.
Thus these selected outputs are not blank images. This does not establish valid
physical states or accuracy. Contact H4 panels also show local outline/position
differences; recognizable shape alone cannot pass the fixed fit gates.

Noncontact segmentation panels show one thresholded object seed component while
the continuous map contains weak background evidence outside its ROI. Rounded
printed component fractions for adapted H1 are.3611(T12),.3595(IT00),.3196(L00),
.3238(IL00), versus approximately1 for the corresponding frozen panels. These
are existing PNG captions, not a reconstructed numerical report. The saved
formal measurement JSON remains authoritative for original values. The fraction
uses dilated-largest-component mass over all continuous alpha; it can reject
distributed weak color without multiple disconnected seed pieces.
`outside_roi_template_mass` in old JSON refers to the fitted template, not the
observed alpha mass outside ROI; it must not be used as the latter quantity.

The two supported explanations for the measurement phenotype, in evidence order,
are (1) diffuse color response affecting global chromatic consistency and
(2) local geometric/area/fit deterioration contributing alongside it. The first
is visible in the fixed panels; its full magnitude cannot be reported because
the numerical export failed. The second is consistent with existing ROI IoU/
area diagnostics, not proof of a localized physical-dynamics defect. Neither
establishes why training caused the change. No learning-rate/data-size/500-step
causal explanation or definite production-interface repair is justified here.
One-target/noisy-history conditioning differences remain contract facts and
untested causal possibilities, not additional established explanations.

The command reached`outputs.write('pixels.json', _json(report))` and failed with
`TypeError: Object of type int64 is not JSON serializable`. One statically
identified carrier is NumPy-derived slice bounds stored in`metrics['roi']`.
The exact offending in-memory instance was not preserved. The minimum proposed
fix is four explicit`int(...)` conversions plus a synthetic JSON serialization
check of a realistic record. This offline-helper defect is not evidence of a GPC
training defect. The10 passed checks did not cover final report serialization.
Executed sources/partial files remain unchanged;`pixels/pixels.json` is absent.

Control flow and the33 saved files imply that48 predictions,64 real case frames,
40 selected train frames,32 H1/TF4 input checks,48 action checks and eight
before/after state alignments finished before serialization. This is a source/
traceback inference, not durable counters or per-array metrics. AR H4 input
reconstruction remains unverified for16 rows; no excluded intermediate image
was opened. Lost full-precision ROI/alpha/MAE deltas are not recovered or claimed.

The smallest next step for main review is this isolated serializer fix and one
bounded re-export task with its own authorization; no new model call, data,
threshold or training change is proposed. This round does not perform it.
The complete scientific decision remains
[the rejected evaluation](GPC_CHECKPOINT500_EVALUATION_RUN.md).

## Commands and preservation

The fixed numerical commands are JSON freeze, synthetic checks, selected pixels.
Each has a60s external limit; at most4commands/240s are authorized, this plan uses
at most3. Two CPU cores/library threads2,1GiB working set and32MiB evidence caps
apply. First numerical failure/limit stops remaining commands without retry.
Command1 completed in1.7763734s. Its venv redirector's4956160B peak is not the
actual interpreter peak; original wrapper/receipt and the limitation are retained
in`runtime-observation-note.json`. Later commands directly use the same installed
base CPython3.12.3 with existing venv packages. No command is replayed.

| Fixed numerical command | External seconds | Observed process peak | Result |
| --- | ---: | ---: | --- |
| 1: JSON summary/sample freeze | 1.7763734 | Interpreter unknown; redirector4956160B | Complete |
| 2: ten NumPy/ordinary-object checks | 1.4585270 | 87060480B | 10pass,0skip; test-body.847s |
| 3: selected pixel panels/report | 5.0267723 | 100630528B | Exit1, final JSON serialization |

Total8.2616727s; three of at most four commands spent, unused allowance not reused.
No cap breach was observed; command1 interpreter-peak coverage remains unknown.
All real Torch imports/model constructions/weights-to-tensor loads/CUDA/neural
forward/backward/optimizer/prediction/Euler/policy/physics/reset/control/render/
collection operations are0. Synthetic native-method arithmetic uses only NumPy
objects, not real tensors or a constructed neural model.

Old protection evidence is reused from main's completed review. Actual read
identities are checked, without rehashing all historical assets. Held payloads
receive0opens/hashes; weights receive0opens/deserializations. New diagnostic
scripts/ignored outputs are separate from all spent candidates/claims/ledgers/
failed temporaries. No new production account/config/candidate exists.
