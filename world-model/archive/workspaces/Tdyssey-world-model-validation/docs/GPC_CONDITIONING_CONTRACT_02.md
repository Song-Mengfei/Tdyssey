# Resource-restored conditioning attempt02

After attempt01 stopped at startup RAM2.6919GiB<4GiB with zero model work, the
user explicitly said resources should now suffice and authorized one new chance.
Attempt01's claim and evidence remain spent and immutable. This is a new
authorization, not an automatic retry or refund.

## Result: complete pairing, inconsistent quality; stop this sigma direction

The single fresh startup gate passed with6855462912B free RAM (6.3847GiB).
The unchanged guarded diagnostic on clean source67d2fd7e completed exactly
32 H1 forecasts/96 Euler/two serial model-only loads. Guard time was44.593s;
the whole external command took45.2642828s, exit0. There were no loss examples,
training/backward, physics/reset/render, collection or policy operations.
No synthetic suite was repeated. The actual source, fixed inputs and config
passed the existing authority gates, with entry/exit RNG matched across A/B and
weight roles. Recorded conditioner traces show the same c_in/c_out/c_skip/c_noise
and the intended c_noise_cond change, with three native denoiser calls per image.

B changes the pictures, but does not repair checkpoint500. Adapted object-state
validity remains0/8 under both A and B (T0/4 and L0/4); pose and coverage remain
invalid/null. The eight retained panels show recognizable objects and a pink
background in both adapted arms. B increases outside-ROI alpha in all eight,
and raw target MAE is higher in all eight. Local ROI IoU is lower in6/8; fitted
shape failure therefore cannot be treated as a background-only issue or as proof
that the entire object vanished. No original threshold or invalid row was changed.

| TRAIN window | Adapted pusher error A→B | Adapted ROI IoU A→B | Frozen pusher error A→B |
| --- | --- | --- | --- |
| T024 |5.08→5.54|.8293→.8308|1.72→3.88 |
| T026 |5.59→2.23|.8052→.8024|.96→1.99 |
| T000 |7.94→4.96|.8267→.8264|1.92→4.69 |
| T002 |3.84→3.69|.8151→.8126|1.22→2.32 |
| L031 |null→1.40|.8179→.8092|1.41→4.12 |
| L033 |2.74→.73|.8083→.8037|2.79→6.28 |
| L001 |3.78→.10|.7688→.7898|1.49→5.15 |
| L003 |4.90→3.90|.8375→.8259|1.87→7.00 |

Errors use the512-unit world coordinate system and only valid pusher positions.
L031 adapted A fails the pusher size/shape gate, so its error stays null. Among
the seven windows with valid positions in both adapted arms, six improve and
T024 worsens; the other L window becomes measurable. Initial-checkpoint pusher
errors worsen in all eight. Initial L object validity improves1/4→3/4 (L033/L003
recover), but L001 remains a rigid-shape mismatch; initial T remains4/4 valid.
These local gains coexist with side effects rather than establishing a broadly
better setting. All32 predictions contain one meaningful blue component; neither
arm reproduces the earlier duplicate-pusher case in this selected set. This
diagnostic cannot claim that B fixes that separate case or the reported missing
arm, and no extra case was added to answer it.

The prespecified criterion required consistent pusher/structure improvement
across both shapes without clear side effects. It is not met. Stop the sigma
direction without a grid or fixed-case confirmation; do not adopt B in the
manual demo or start training/DPPO. This does not prove that conditioning never
matters or identify the causal training defect. The denominator is eight TRAIN
windows from six trajectories/scenes (T4/2 and L4/4), one sampling seed, with no
independent validation, p-value or general dynamics-success claim.

All original per-window diagnostics, outline/full-IoU/local-fit records, pusher
components, valid pose/coverage, raw means, target MAE and PNGs remain in the
native output. `paired-summary.json` and CSV in preparation contain16 A/B rows,
including the invalid measurement. Eight panels use latest/truth/A/B columns
and initial/adapted rows; key examples are `panels/train-cursor-024.png`,
`panels/train-cursor-033.png` and `panels/train-cursor-001.png`, with the complete
set retained. These are display-only clipping of raw saved outputs, not recoloring.

The guard reports no first error or cleanup error, model_instances0 at completion,
Job active count0 and model lock released; native integration verification is
true. Process WS/private peaks were1276755968/2155356160B; CUDA allocated/reserved
peaks were109832192/136314880B. All caps/floors remained unchanged. The01 claim,
stop record and old config match their recorded byte identities after02. No
retry, additional real task, Windows framework expansion, installation, remote
repository change or migration was performed. Main review receives this result
before any separately authorized Linux preparation.

## Identity-only source binding

The sole changes from the [01 protocol](GPC_CONDITIONING_CONTRACT.md) are:

| Binding |01|02|
| --- | --- | --- |
| Candidate suffix |diagnostic-01|diagnostic-02|
| Authorization |20261026-user-autonomous-gpc-conditioning-contract-diagnostic-01|auth20261026-user-autonomous-gpc-conditioning-contract-diagnostic-02|
| Config |gpc_conditioning_contract.json|gpc_conditioning_contract02.json|
| Output leaf |auth20261026-user-autonomous-gpc-conditioning-contract-diagnostic-01|auth20261026-user-autonomous-gpc-conditioning-contract-diagnostic-02|

The old config SHA remains
`043339f6e1950970b54e3c084311a8c325bfac7ae4bb66540711c7fefff6c327`.
Only `candidate_id` and `output_dir` differ in the new JSON; all scientific fields
are equal. Entry and account edits change only identity assignments. One identity
test's output join follows the new authorization's explicit `auth` prefix.
Data, delegate, sampler, guard algorithm, resources, seed and UI are unchanged.
No prior synthetic suite is rerun; only static identity/config/source/message
checks are performed.

Fixed scope remains32 H1 forecasts/96 Euler, two serial model-only loads,
eight selected TRAIN windows at cursors24,26,0,2,31,33,1,3, T4 windows/2 scenes,
L4/4 scenes. A=None/B=.002 on the same clean history/actions/seed5478, T1 once,
native three-Euler schedule and matching entry/exit RNG. No loss, training,
physics/reset/render/collection, extra case, seed or sigma grid.

The existing guardian's fresh startup gate performs the single resource precheck
before launching any SDK/model worker. Startup floors remain RAM4/commit6/GPU4/
disk2GiB, with all runtime, Job, CUDA, CPU,300s and64MiB limits unchanged. Failure
of that precheck stops immediately; no polling, program closure, lowered limit,
fallback, resume, retry or further identity follows. Passing the precheck
authorizes the already frozen actual run without another confirmation.

Preparation lives in `artifacts/gpc-conditioning-contract-preparation02-20261026/`;
native output is bound to
`artifacts/gpc-conditioning-contract-diagnostic-20261026/auth20261026-user-autonomous-gpc-conditioning-contract-diagnostic-02`.
`start.json` records the unchanged parent evidence and old config;
`identity-validation.json` records exact differences and source proof.
The source-freeze stage preceded the single gate/run. Its completed outcome and
limited scientific interpretation are recorded above; no next experiment follows.
