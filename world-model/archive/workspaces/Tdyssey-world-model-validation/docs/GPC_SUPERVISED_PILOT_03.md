# Pilot03: 500 updates saved; evaluation stopped at status publication

The sole attempt on clean `e6b03416cbe4aa50acc95ed9fe8c62efabf1f2b6` completed
500 real updates and selected checkpoint500 in135.766s. It stopped at360.578s
during frozen-weight evaluation: Windows status replacement returned1175.
Adapted evaluation never started, so there is no paired T/L adaptation result.
Main review of [the reference repair](GPC_GRADIENT_LIFETIME.md) on `07616f9`
authorized this separate attempt. Relative to pilot02, releasing local `grads`
was the sole training implementation change; identity changes only isolate costs.
No retry, resumed evaluation, fallback or post-stop production repair followed.

Authorization: `20261026-user-autonomous-gpc-supervised-pilot-03`.
Candidate: `gpc-tl-supervised-one-step-03`. Fixed output:
`artifacts/gpc-supervised-adaptation-pilot03-20261026/`. The
[new configuration](../configs/gpc_supervised_adaptation_pilot03.json) changes only
candidate/output from preserved original config
`1c5683baa16daee11ca419772532cdda95595275cca794b977261188d99abe2c`.
Pilot02 config `bd43e43e28e0f54fe8965bdd6c558f21dcbf5d7b20128b7520a1d3594c8ca0e8`
also stays immutable. Entry accepts only the explicitly matched config/auth/
candidate/output, not arbitrary new names or automatic resume/retry.
The new3,618-byte config SHA is
`5bc6c88efd4023dc27c3718f57655791e0271a4bc8ae8241069fef98352e4937`.

Keep GPC source/initial checkpoint, T1, A/B2 data and the frozen500list
`f7217a2d09393798529b00e72fcd971e8d7be11cab7a9c769704a9b30ddd004a`.
Preserve phase-two one-step loss, AdamW1e-4, batch1, seed5478, clip1, T/L250 each,
500 updates, saves every50/500 and checkpoint500 as the only selectable result.
The first forward/backward/optimizer consumes update1. Fewer than500 is incomplete.

Loading/training/selection share900s, full execution/cleanup3600s, with original
dual clocks, three serial loads/one resident model,32 loss batches/examples,
3,996 forecasts and11,988 Euler calls. Evaluate frozen/adapted on63 regression
and48 independent B2 branches, three matched noise seeds, AR H1..4 and TF H2/H4.
Keep1GiB output/64MiB telemetry including temporary/atomic reservations, original
native floors, process2GiB working set/3GiB private,3GiB Job, allocated/peak512MiB,
reserved/peak1GiB CUDA and5s heartbeat. First error or cap ends this attempt.
No CPU/model/precision/seed fallback, new data, policy/DPPO, physical operation,
installation or download is authorized.

Report real ledger highwater, finite gradients, tensor parameter changes where
measurable, checkpoint500/optimizer/RNG/cursor identity, actual peaks and cleanup.
Checkpoint-file SHA alone is not a parameter-change measurement. Preserve all
declared rows and invalid/abstained denominators. Independent contact4<8 and
rotation2<4, T near-positive above .98, zero H1/H2 separable pairs, one T/two L
informative H4 anchors, one-scene loss and missing train near-success/T CCW
coverage remain limitations. Completion or loss reduction cannot certify
imagined DPPO or physical policy improvement.

The prelaunch gate passed41 stdlib/marker checks in9.562s, with zero failures,
errors or skips and zero real runtime/process/native/network attempts. The
training function AST matches reviewed `07616f9`; the three fixed identities,
old spent isolation, lifecycle regression, schedule/cursor and budget behavior
pass. Source-only checks pass. Evidence is under
`artifacts/gpc-supervised-pilot03-review-20261026/`.

## Actual training and selected contents

Fresh native preflight and pinned source/initial-weight/data/config/list checks
passed. GPC source remains `a96146e568826ef46134f73c22fdf59a675e7264` and the
35,043,916-byte initial checkpoint remains
`4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
The1,714-event ledger hash chain verifies500 reservations/completions in exact
frozen order:500 unique windows,250 T/250 L, cursor0->500. All500 update records
have finite loss/gradient norms and a continuous dedicated noise-generator hash
chain. Training loss means T .00384982/L .00459062 describe changing training
examples/noises, not independent improvement. Checkpoints were committed at
50,100,...,500; the only selection is the final500 boundary.

Selected `adapted-latest.pt` has105,134,565bytes and SHA
`0bfd1a944c8fb01555a7ad2d01aab6f526a4805e2bb5239a96703d356c5741d3`.
After process exit, a stdlib-only ZIP/restricted-pickle metadata reader compared
the actual float32 values at matching named tensor positions against the pinned
initial weights. It instantiated no Torch/tensor/model and used no runtime
fallback. All296 trainable tensors changed;8,729,815 of8,729,827 elements differ,
all finite, with L2 delta3.9458084106 and maximum absolute delta .00985536724.
These are parameter-content comparisons, not inference from checkpoint-file SHA.

The saved logical step, config/attempt/budget identity and schedule cursor all
match500. All296 participating AdamW states have step500. The saved `train_noise`
state SHA `48f485962ec6955c10ba0f4307d33d48baa2b5eac7594bb5ec40f921fc057164`
matches the last update's generator exit; Python/NumPy/Torch CPU/CUDA RNG records
are present. This checks saved content/coherence. Adapted evaluation/reload did
not execute, so native restoration or task quality is not established.
Details: `parameter-contents.json` and `ledger-training-audit.json` in the review
root. Parameter inspection was offline file parsing, not an extra model load.

## Precise failure and partial evaluation

At2026-10-26 05:12:18.032 SGT the first durable heartbeat exception is
`status.atomic.publish`, `OSError`, errno22, WinError1175:
"Unable to remove the file to be replaced" (the OS record is Chinese).
Its traceback reaches `_publish_json`'s `ReplaceFileW` call. Both the prior status
and the failed temporary generation are retained; the temporary counters are
333forecasts/999Euler. The parent stops within .391s of the error with the last
heartbeat still fresh. The5s rule remains unchanged; this is a known publication
failure, not a freshness timeout or observed CUDA/process/resource cap breach.
The files do not identify why deletion failed or which external handle/OS
condition caused it. No new error is retroactively assigned to pilot01.

| Cost/completion | Actual | Original cap |
| --- | ---: | ---: |
| Serial model loads reserved/completed | 2/2 (train/frozen) | 3, one resident |
| Training updates reserved/completed | 500/500 | 500 |
| Loss batches/examples reserved/completed | 16/16 (frozen only) | 32 |
| Forecasts reserved/durably completed | 333/332 | 3,996 |
| Euler steps reserved/durably completed | 999/996 | 11,988 |
| Saved finite prediction/measurement pairs | 333 | 3,996 planned |
| New physical/policy operations | 0/0 | forbidden |

The last pending forecast already has its raw image and measurement: source
ordering writes those before the failing progress publication and before
`complete_forecast`. Therefore333 real outputs and the fixed sampler support999
executed Euler calls, while332/996 remain the durable completion counts. The
pending unit stays spent. All333 are frozen/regression (T288,L45); independent
B2 and adapted prediction counts are zero. Independent frozen loss executed16
batches, but its values were held in memory pending final completion and were
not exported. No completion/screens record exists. Keep the full3,996 denominator;
T retention, L change, AR/TF change, ranking and termination are unavailable.

## Resources, retained evidence and cumulative costs

CUDA allocated peak is510,962,176B (487.292MiB), reserved peak578,813,952B
(552MiB), within the original512MiB/1GiB limits. Allocated peak is34,946,560B
below pilot02's observed peak. This separate-run difference is consistent with
released gradient storage; no per-allocation trace proves its exact source.
Training500 now fits the cap in this run, not necessarily every remaining phase.
Private-memory peak3,170,590,720B leaves only50,634,752B below3GiB; third model
loading was never tested. Working-set/OS peak1,840,005,120B stays below2GiB.
Minimum free RAM/commit/GPU/disk were5,936,214,016 /14,423,961,600 /
6,601,834,496 /28,935,475,200B, above original floors.

The run starts05:06:17.840 SGT and ends05:12:18.418 SGT, within900s training/
3,600s total. Guard21472 owns worker26136 through launcher24964; worker/launcher
exit1, Job active0, global lock released, cleanup errors empty. Observed heartbeat
publication gaps are at most1.141s. The first-error slot and bounded stderr keep
the full exception; no failure temporary file was removed.

There are1,184 retained files/146,085,235B. All333 NPY headers and raw-data hashes
match their saved measurement identities. The selected checkpoint was verified
by its immutable hash and actual parameter contents. Model-output prepaid bound
is390,222,502B<1GiB; telemetry prepaid bound20,167,624B<64MiB. Monitor actual length
2,340,168B includes189B after the stop snapshot's2,339,979B; stdout is1,664B.
`output-inventory.json` records file identities without copying images/weights.

Across pilot01/02/03, totals are3 real attempts,3 model loads,502 reserved updates/
501 durable completions,333 reserved forecasts/332 durable completions,999/996
Euler reserved/completed,16 loss batches and394.485s guarded elapsed. These are
costs, not independent sample counts. The separate SDK diagnostic adds3.860s and
zero model/update/forecast cost. Old failed reservations are not refunded.

Closing preservation passes for7,881 old files/629,907,389B across59 roots plus
the standalone SDK claim. All14 old claim documents/16,616B, original/pilot02
configs,500list, GPC manifest and initial checkpoint stay unchanged. Eight held
files use sealed manifest/stat checks only, with zero content opens/hashes.
Compact before/after/reconciliation reports show no changed/missing/added member.

Main review should address the precise status-publication failure with harmless
filesystem checks. Checkpoint500 is retained for a later separately scoped
decision; this failed attempt cannot resume or spend its remaining capacity.
There is no paired validation benefit or imagined-DPPO approval. Existing
independent-data support and success-boundary limitations remain unchanged.
