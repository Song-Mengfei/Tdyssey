# Milestone C guarded runtime

The reviewed reset config remains `configs/policy_c.example.json`, canonical
SHA-256 `0c682afbaf03360dc7af77c3c0917c3cfd120d5e7a8da10532c1d3d5251a8db3`.
Its base is the explicitly pinned local compact checkpoint described in
[the reset preflight](DPPO_C_RESET_PREFLIGHT.md). Its provenance field does not
independently establish tensor identity with the published EMA release.

## Execution controls

`scripts/run_guarded_c_policy.py load` starts one disposable child through C's
actual `_runtime` and `_new_actor` path. Source-derived trace points observe the
unchanged official loader after mmap, CPU construction, strict state copy and
CUDA transfer. The actor shares that one official network. The child then creates
the small critic and empty Adam shells. No policy forward, sample, simulator reset,
physical control, optimizer step, formal evaluation episode or C ledger is needed.
The official loader's ordinary `.eval()` mode selection remains unchanged.

The parent uses stdlib Windows APIs for physical RAM, actual system commit
(`GetPerformanceInfo`, pages multiplied by page size), and child working set,
private bytes and lifetime peaks. Driver GPU readings run separately from the
0.2-second host sampler. Stage journals are fsynced. Load deadline is 180 seconds.
Startup requires 4 GiB available RAM, 6 GiB commit headroom, 4 GiB free GPU memory
and 2 GiB disk headroom; runtime stop floors are 1 GiB available RAM, 2 GiB commit
headroom and 1 GiB free GPU memory. These are operational guard choices, not a
proof that pressure cannot change between samples. Only the spawned child may
be terminated; user applications are never closed.

Windows virtual-environment `python.exe` can be a redirector: its Popen PID is
not necessarily the interpreter doing the work. The corrected parent first
starts `guarded_c_child.py`, which writes its actual PID and a fresh nonce and
waits for ACK before importing target code. The parent verifies process ancestry,
holds the actual process handle, attaches a kill-on-close Job Object, samples
that process and rechecks RAM/commit before ACK. Monitoring and termination then
use the actual worker. This also makes parent-process death close the job handle.

On a successful, fully released load check, separately authorized `train` starts
the permanent ledger immediately before one formal `run_c_policy.py train` child.
The reviewed protocol keeps 2,048 controls as both target and ceiling, at most
24 batches, 72 actor/critic steps, 7,200 seconds, 512 MiB formal artifact storage
and two retained deltas. No development evaluation group starts automatically.
Monitoring files have a separate 64 MiB host telemetry cap.

Formal C training now gates its first in-budget batch: one complete PPO pass,
matching behavior probabilities, finite losses/gradients, actual actor/critic
weight changes, verified frozen fingerprint and reconciled operation counts.
Failure stops before another collection and forces any saved delta to audit-only.
Successful batches record collection/update time, PPO probability diagnostics,
per-shape reward/contact/coverage summaries and allocator lifetime peaks.
Continuous episodes, KL stopping and genuine-boundary-only recovery stay in force.

## Evidence

Initial 2026-10-24 05:04 SGT read-only snapshot:
`artifacts/milestone-c-load-memory-20261024/resources-initial.json`.
Available RAM was 5.46/15.85 GiB, system commit headroom 16.07 GiB,
GPU free 6.75/8 GiB and disk free 23.30 GiB. This snapshot alone establishes
neither successful loading nor a training peak.

### Real load, 2026-10-24 05:19 SGT

Code `eb4c0e0`, one completed load through the unchanged formal C actor factory:
`artifacts/milestone-c-load-memory-20261024/load/` and `load-acceptance.json`.
All eight stages share actual worker PID 12284. Its Win32 lifetime peak working
set was 2,798,817,280 bytes (2.607 GiB), peak process commit/pagefile counter
4,684,144,640 bytes (4.362 GiB). Sampled private bytes peaked at 4.362 GiB.
CUDA allocator peak was 1,055,474,688 bytes; reserved peak 1,080,033,280 bytes.
Minimum observed available physical RAM was 3.081 GiB, commit headroom 11.520 GiB;
stage-level driver snapshots observed at least 5.629 GiB free GPU memory.

The official network had 262,709,056 parameters; the actor shared it and selected
1,313,282 output parameters. Critic parameters: 17,153. Adam state entries: zero.
Forward, policy sample, reset, control, optimizer step and evaluation counts were
all zero. No C ledger or clock was initialized. The child exited normally in
about 17 seconds; afterward available RAM was 5.588 GiB and GPU free 6.749 GiB.

The original parent had monitored redirector PID 21936 (about 4 MiB), so its
process peaks and targeted termination were invalid evidence. The lifetime peaks
above come from the actual child's own Win32 records. System RAM/commit sampling
remained valid, while the parent's sparse GPU samples missed residency; use the
child's stage GPU records. The PID/Job Object fix must pass a native tiny-process
check before training. That native check now passed all three tiny stdlib cases
(normal completion, exact retained-handle termination, Job Object closure), in
`worker-control-check-3/report.json`. The first check exposed an asynchronous
job-close race and prompted a bounded wait. Job closure can report exit code zero
even though target work never completed; semantic ready/gate reports remain
mandatory. These checks do not initialize C or load models.

No second model load is needed: formal model, loader,
training entry, config and upstream source hashes match the accepted load;
only parent/bootstrap memory-control code changes. Load success does not measure
PPO activation peaks or establish physical task improvement.

### Training launch handoff (historical snapshot)

Source `69742b7` launched permanent experiment `c-tl-2048-20261024` at about
05:30 SGT. The actual worker is PID 10104, redirector PID 15792. Training output:
`artifacts/milestone-c-20261024/`; permanent ledger:
`artifacts/policy-c-registry/c-tl-2048-20261024/ledger.json`; parent resource logs:
`artifacts/milestone-c-load-memory-20261024/train/`.
The launch handoff snapshot recorded 184 completed controls, 23 plans, 2 resets,
one completed batch and one step per optimizer. Development/diagnostic accounts
remain zero. The job was still running at this handoff; these counters were not a
completion or improvement claim. The user requested ending active watching once
the job was started.

The parent and training child continued recording without chat polling.
`training.jsonl` contains batch/gate/pass diagnostics; `train-report-*.json` or
`failure-*.json` records the child outcome, checkpoint deltas retain explicit
resumability, and the parent writes `monitor-report.json` on exit. Guards and
permanent caps stop work automatically. `launch-handoff.json` captures the exact
handoff counters. The separate human-readable first-batch review was pending
when active watching ended; the training process's own gate remains authoritative.

### Completed short training and offline review

The run completed on 2026-10-24, 05:30:45.299 to 06:30:19.814 SGT
(3,574.515 seconds, including model load and cleanup). Both the actual worker and
redirector exited zero, with stop reason `control_target`, no guard reason,
checkpoint error, runtime failure report or pending ledger reservation.

| Quantity | T | L | Total |
| --- | ---: | ---: | ---: |
| Physical controls | 1,116 | 932 | 2,048 |
| Sampled/executed plans | 142 | 118 | 260 |
| Resets / episodes started | 5 | 4 | 9 |
| Successful training episodes | 1 | 0 | 1 |
| Timeout episodes | 3 | 3 | 6 |
| Active episodes at budget tail | 1 | 1 | 2 |
| Controls with contact | 272 | 247 | 519 |
| Highest observed coverage | 0.958976 | 0.486981 | — |
| Sum of training reward (success + shaping) | 0.028190 | -1.884964 | -1.856773 |

There were 20,480 control physics substeps plus nine reset settling substeps.
Shape control shares were 54.49% T / 45.51% L. These are training observations
from changing policies and scenes, not fixed-policy success rates or paired
evidence of improvement. No development/diagnostic controls or evaluation groups
were consumed, and no world model or new human demonstrations were used.

All 18 batch entries had zero log-probability error, ratio exactly one and
`plans * 100` transitions. The first batch gate passed. Its one PPO pass plus
17 later batches with three passes each gave 52 actor steps and 52 critic steps.
Every pass had finite diagnostics and actual changes to all six selected actor
tensors and all four critic tensors; the frozen fingerprint stayed identical.
Maximum sampled approximate KL was 0.000161719 against the 0.02 stop threshold;
maximum clipped fraction was 0.001875. No KL early stop occurred.

Critic quality remains uneven: explained variance was negative in 28/52 pass
diagnostics (range -1.15456 to 0.937685). Decreasing loss on a fixed buffer does
not establish better task performance. Actor preclip gradient norm exceeded one
in 43/52 passes; the declared clipping was applied, with no nonfinite gradients.
Mean collection/update time per batch was 23.03/174.95 seconds, so PPO updates
dominated the roughly one-hour run. L had no successful training episode.

All 17,347 host/process and 706 GPU samples were valid and monitored actual
worker PID 10104. Minimum observed available RAM, commit headroom and free GPU
memory were 3.2098, 10.8084 and 4.8477 GiB. OS lifetime peak working set was
2.6065 GiB; peak process commit/pagefile counter was 4.3618 GiB. CUDA allocation
and reservation peaks were 1.09256/1.16992 GiB. No OOM, traceback or memory guard
was observed; only existing Gym/torchvision maintenance/deprecation notices.
Polling records observed minima, not instantaneous guarantees. On exit free RAM
was 6.769 GiB and free GPU memory 6.727 GiB.

Reserved and completed counters match exactly: controls 2,048/2,048 cap,
plans 260/384, resets 9/384, batches 18/24, and each optimizer 52/72.
Registered storage was 32,721,688/536,870,912 bytes; no evaluation slot was claimed.
The base/config/launch-source identities and B's prior report/budget hashes are
unchanged. The reviewed local base pin retains its original published-EMA
provenance limitation.

Two retained deltas are each 16,003,916 bytes (15.26 MiB):

- `checkpoint-0018-0052-audit.pt`, final controls 2,048 / steps 52;
  SHA-256 `d3128c7fcfb1a23842b4863d96ae88b7f124e93d751b371b23696eb24d6703dd`.
  The tail contains active episodes, so this is explicitly non-resumable.
- `checkpoint-0017-0049-boundary.pt`, saved at controls 1,989 / steps 49.
  It was resumable at save, but the current ledger advanced through another
  59 controls and three updates. Its operation marker 2,430 is stale against
  current 2,511, so it cannot legally resume this experiment now.

Both files passed SHA/byte checks, CPU `weights_only=True` loading, finite-tensor
checks, official/config/base metadata checks, selected actor/critic counts and
Adam step/parameter ownership checks. This read did not reconstruct the base,
initialize CUDA, run inference or consume operations. The files remain useful
for separately scoped evaluation; no retained checkpoint can resume the spent
training ledger. The permanent wall budget includes downtime and must not be
reset for a later request.

Offline closeout evidence under `artifacts/milestone-c-load-memory-20261024/`:
`train-reconciliation-verified.json` (passed, no issues or warnings),
`checkpoint-metadata-verified.json`, `ppo-review-final.json`, and
`resources-review-final.json`. The earlier pending first-batch human review is
superseded by these completed reviews. Raw `train-report-0000.json`,
`training.jsonl`, registry and monitor files remain immutable.

Reproduce the read-only closeout with the existing installed Python, from the
worktree root (no install, full model load, reset, update or evaluation):

```powershell
& $python -S -B artifacts/milestone-c-load-memory-20261024/reconcile_c_run.py `
  --checkpoint-metadata artifacts/milestone-c-load-memory-20261024/checkpoint-metadata-verified.json `
  --output artifacts/milestone-c-load-memory-20261024/reconciliation-new.json
```

Use a new output path; recorded evidence is never overwritten. The next research
step is a fixed-case base-versus-trained physical comparison under an explicit
evaluation budget, with T retention and multiple seeds. This review starts none.

That subsequent [separately authorized paired evaluation](DPPO_C_EVALUATION.md)
has now completed all 40 fixed cases without modifying this training ledger.
T success declined 6/10 to 5/10 and L stayed 0/10; L final coverage increased
slightly while peak coverage decreased. The detailed evaluation record preserves
the negative/mixed result and its small development-set limitations.
