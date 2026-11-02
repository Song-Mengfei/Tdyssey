# Windows status publication review

## Result and retained change

The pilot03 publication failure remains unresolved. Its real WinError1175 was
not reproduced. A single native comparison rejected `MoveFileExW` flags9:
it failed with WinError5 even while the production share-delete reader held a
valid old status. The final implementation retains `ReplaceFileW`, without a
fallback, changed timestamp, publication retry or relaxed resource/age limit.
No new model, SDK, CUDA, prediction, training, physics or production account ran.

The retained source changes address separately demonstrated contracts:

- Shared publication checks the temporary source is in the target directory,
  ordinary, single-linked, non-reparse and at most64KiB, before initial or later
  publication. Existing target checks also apply to shared publication off Windows.
- `atomic_json(storage=...)` attempts publication once, preserves failed/partial
  temporary bytes and the original error chain, then refuses another publication
  through its existing orphan check. Its private, non-storage path keeps its prior
  policy. GPC already preserved its pilot03 failed temporary; this is no explanation
  for that incident and does not repair its publisher failure.

No reader or training code changed. Path validation uses lexical absolute paths
and `lstat`; the production reader already opens with read/write/delete sharing.
This review did not establish which CPython metadata path or external handle, if
any, participated in pilot03. Do not attribute the failure to antivirus or missing
share-delete. Validation checks do not eliminate filesystem races.

## Frozen native comparison

Evidence lives in `artifacts/gpc-status-publication-review-20261026/`.
`native-cases.json`, `plan.json` and `plan-fixed.json` freeze seven cases and the
exact harness/source hashes. The directory named `native-fixed` contains the
**rejected candidate**, not an accepted fix. Exact original/candidate source
archives match those plan hashes; final source SHA is
`0779a7d25bf1a0214e318390ae34a848b90f2ff945d34e602da32d4542a3cab2`.

Each run allowed at most2060 publications (below5000), JSON at most64KiB
(fixture bound2048B), one writer, two readers and three processes including the
PowerShell timeout parent. Work/cleanup deadlines were45/55s; Python had a58s
watchdog, and an external PowerShell parent enforced58.8s with process-tree cleanup.
Each case stopped at its first fault. No failed publication was retried.

| Case | Original ReplaceFileW | Rejected MoveFileExW flags9 |
| --- | --- | --- |
| A: same-thread production old reader | 2/2 published; old whole, new generation1 | 1/2; WinError5 |
| B: two cross-thread production old readers | 2/2; both old whole, new generation1 | 1/2; WinError5 |
| C: ordinary `Path.open('rb')` old reader | 1/2; WinError32 | 1/2; WinError5 |
| D: explicit GENERIC_READ, share3 old reader | 1/2; WinError32 | 1/2; WinError5 |
| E: FILE_READ_ATTRIBUTES, share0 handle | 2/2; new generation1 | 1/2; WinError5 |
| F: GPC publisher, two actual production readers | 3/3 publications; reader WinError2 | 3/4; publisher WinError5 |
| G: WM publisher, two actual path/metadata readers | 8/8 publications; reader file-not-found | 57/58; publisher WinError5 |

F/G planned1025 publications each, but stopped early as required; neither is a
completed stress pass. Original F/G readers completed16+8 and61+59 reads;
candidate F/G readers completed8+9 and160+161. Each reader had successful reads
overlapping the writer interval. Successful JSON reads were coherent generations;
the original failures were transient name absence, not mixed JSON. Existing parent
`_latest`/`_latest_worker_status` already allow exact-path open disappearance or
32/33 to reuse the unchanged-age snapshot. This raw-reader observation therefore
does not reproduce the writer's1175 or establish a new parent stop mechanism.

Across both runs:93 publication attempts,84 successes, nine failed temporaries
retained with original error records and complete byte copies. All held data
readers finished their old generation, including on failure. Both Jobs ended
empty with exit0/0 and no cleanup errors. Harness status `collected` means evidence
collection completed; it does not mean every case passed. Guarded times were
0.968/1.110s; external-parent times1.321/1.445s, combined2.766s versus120s allowed.
Unused publication allowance is not permission for a third run.

The documented [ReplaceFileW contract](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-replacefilew)
defines1175 as failure to delete the old target with both original names retained;
its metadata/ACL merge differs from a move. The
[MoveFileExW contract](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw)
supports replacement subject to security requirements, but does not justify
assuming live-reader compatibility on this machine. The native comparison rejects
that candidate. Foreign data handles that deny delete sharing can still block
publication. Neither API's broad reliability is proved here, and old-target
preservation is not guaranteed for every possible real filesystem failure.

## Synthetic checks and preservation

Exactly three bounded synthetic commands ran, all without explicit native APIs,
process launch, SDK/model import, network or physics. Baseline two contract cases
produced seven expected failures: failed-temp deletion and six invalid-source
combinations. The candidate then passed20 fake checks but failed native validation;
after reverting its primitive, the final20 checks passed in2.250s inclusive.
Inclusive synthetic times were0.125/2.657/2.250s, total5.032s versus180s allowed.
No fourth synthetic command or third native run followed.

Checks cover actual GPC/WM publisher bodies at injected OS boundaries, complete
generations, links/reparse/size refusal, injected1175 and partial-write evidence,
immediate later heartbeat refusal, separate original5s heartbeat/allocator ages,
and parent first-error/Job-cleanup handling. Link metadata is injected; native
cases use real Windows handles. The final source was not given another native
run; it retains the original primitive plus the tested boundary/error changes.

`before.json` and compact `after.json` preserve all9084 old files/776457089B,
62 roots plus the standalone SDK claim,16 claim documents/20174B and six input
identities. Eight held payloads received stat checks only, no opens/hashes.
No old output, failed temporary, checkpoint, configuration or500list changed.
`summary.json` and `delivery.json` bind the evidence. New evidence stays below32MiB.
Required source/message checks are recorded in the delivery; they are separate
from the three synthetic regression commands and do not execute model/runtime code.

## Future eval-only plan

This original proposal was later implemented without real model execution in
the separate [eval-only engineering](GPC_CHECKPOINT500_EVALUATION.md). The negative
ReplaceFileW/MoveFileExW results above remain unchanged.

This section proposes a separately reviewed evaluation-only mode. It implements
no entry point and authorizes no evaluation, training, SDK or physical operation.
Windows diagnostic results will be recorded separately after that work finishes.

Keep the existing immutable inputs and use exactly these two weight sources:

- Initial phase-two checkpoint: 35,043,916 bytes, SHA256
  `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
- Pilot03 selected checkpoint500: 105,134,565 bytes, SHA256
  `0bfd1a944c8fb01555a7ad2d01aab6f526a4805e2bb5239a96703d356c5741d3`.
  Its saved logical step, selected source account, config and schedule cursor 500
  must match the sealed [pilot03 evidence](GPC_SUPERVISED_PILOT_03.md).

Create a new explicit `eval_only` binding, output and ledger. Its account must
reject training, optimizer construction, update reservations and new checkpoint
selection. Record `source_checkpoint.logical_update=500` independently from
`account.completed_updates=0` and zero spent training examples. Do not reopen the
failed pilot03 ledger, resume its counters, or restore its saved training/RNG state.

The current `run_candidate` and `AdaptationBudget` are not suitable entry points:
they require train_init -> frozen -> adapted loads and a training selection.
The guard also requires 500 updates before `training_complete`. A reviewed mode
and account must represent training as inapplicable, rather than fabricate 500
new updates. Reuse `load_cases`, `evaluate_loss`, `rollout_case`, `NativeSampler`
and `gpc_adaptation_reporting.summarize` through that explicit account boundary.
Do not invoke the existing training flow or its checkpoint restoration path.

Recompute the complete fixed comparison in a fresh attempt:

| Quantity | Eval-only bound |
| --- | ---: |
| Serial model loads / active models | 2 / 1 |
| New updates / training examples | 0 / 0 |
| Regression / B2 branches | 63 / 48 |
| Noise seeds | 5478, 5479, 5480 |
| Modes and horizons | AR H1-H4; TF H2/H4 |
| Forecasts / three-step Euler calls | 3,996 / 11,988 |
| Independent loss batches/examples | 32: 16 per weight role, 8 T + 8 L each |

Load frozen weights, evaluate and release them, then load selected adapted model
weights and evaluate. Rebuild the matched-noise protocol from the declared seeds;
do not restore old checkpoint RNG or splice in pilot03's 333 saved forecasts.
Those 333 forecasts/999 executed Euler calls remain historical spent work.
Keep real histories, aligned controls, T1, native sampler feedback, measurement
thresholds, invalid denominators and per-stratum/seed/AR/TF reporting unchanged.

The original 900s is the launch-to-training-selection window, including startup
and the initial load. It is not 900s per load or a fresh evaluation allowance.
`gpc_adaptation_guard._clocks` separately fixes the full 3600s launch-to-cleanup
deadline; `run_candidate` subtracts elapsed startup from the 900s training window.
An eval-only mode must explicitly review the training deadline as inapplicable
while retaining the 3600s total, starting before either load. No clock reset or
other cap increase follows from removing training. Retain the original 1GiB
output, 16MiB metadata, 64MiB telemetry, native resource floors, 2GiB working set,
3GiB private/Job, 512MiB allocated/peak and 1GiB reserved/peak CUDA limits,
heartbeat deadline, durable pre-call charges and first-error stop behavior.

Pilot03's observed private peak was only 50,634,752 bytes below its 3GiB ceiling.
The current `CheckpointStore.load_selected_for_evaluation` loads the 105MB container and
clones the current model for rollback. A future bounded model-only loader must
validate the selected container/identity, keys, shapes, dtypes and finite values
without restoring optimizer/RNG or retaining an unnecessary 35MB rollback model.
Discard a failed new model and stop; no fallback, retry or resource-cap increase.
Do not assume memory mapping or a file hash alone proves safe loading.

File-complete evaluation would still leave T near-positive support, absent H1/H2
separable pairs, small contact/rotation denominators and limited independent
scenes insufficient. Preserve those abstentions and gaps. Completion, loss
reduction or checkpoint500 selection does not admit imagined DPPO or establish
physical policy improvement. A real eval-only attempt needs separate review and
explicit authorization after the status-publication findings are available.
