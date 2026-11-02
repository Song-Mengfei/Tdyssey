# R3B stopped before its first training batch

The independent background process started, restored the complete original6000
checkpoint, and then stopped at the first CPU data-loader batch with
`concurrent.futures.process.BrokenProcessPool`. This is a failed integration
attempt, not a completed training epoch or a model-quality result.

## Actual outcome

- R3A collection remains complete:912 trajectories,768TRAIN/144seen-validation.
  Its prior numerical, renderer and contact audits remain valid. CP1/CP2 were
  not merged. The approved distinct TRAIN pool remains23,984 windows.
- Run ID:R3B-firstepoch-0c0e39d5b67940799c2211eca058017f.
  Worker PID18548;launcher PID29080. Existing Explorer broker launched hidden
  direct-base Python with the existing virtual environment. Startup proof and
  launcher live query both observed WindowsJob=false for the real worker.
- Full original6000 model/AdamW/scaler/scheduler/global RNG restoration passed;
  the new conditioning RNGs and data cursor0 were explicit. No original file
  was overwritten.
- The144natural/16stress and exact old4 batch-context specification was frozen.
  t0 evaluation was never entered. Epoch1 training and its evaluation were never
  entered. There are no new prediction panels or learning results.
- Engineering and formal optimizer attempts/successes/AMP skips/draws are all0;
  forward/backward counts are0. Formal cursor/exposures are0;lineage remains6000.
  Neither a temporary engineering checkpoint nor a new formal last.pt exists.
- Model ledger records33.047additional seconds,including29.875restore and2.953
  engineering-section seconds;worker wall time34.255seconds. Global charged and
  completed update counts remain6250,active_stage=null,no pending reservation.
- Worker and launcher have exited. No automatic retry or process remains running.
  Full after-launcher-exit progression proof was not completed before this fault;
  startup independence alone is not presented as a successful running pipeline.
- Original6000 SHA remains
  c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148;
  R3A manifest SHA remains
  0e91e1412f937c7d8f39473243a27a3720317a07474e730a510e628d040c0a19.
- The official completion-check interval was changed to30minutes as requested.
  It remains PAUSED because this run has already failed and is being reported
  directly. No unsupported offline chat callback is claimed.

## Failure and proposed next action

The exception occurs at `next(session._ensure_loader().iterate([indices]))`,
before an optimizer reservation. Child-process initializer stderr/exit details
are absent, so neither data corruption nor CUDA failure is established. The new
second-level Windows multiprocessing spawn boundary and virtual-environment
executable selection are a plausible cause, not a verified diagnosis.

Request approval for a narrow loader-startup recovery:preserve this failed run,
first capture two CPU worker bootstrap identities/exitcodes/stderr in the same
detached environment before constructing a model;only if executable aliasing is
confirmed,bind multiprocessing to the actual base Python PE while retaining the
existing venv. Keep two CPU workers,micro4/effective16,physics,data,conditioning,
evaluation windows and the1499-batch first-epoch boundary unchanged. Any later
controlled restart must retain this33.047s charge and require explicit release;
this report does not authorize it. No code repair or new probe was run here.

The frozen release requires exceptions to save evidence and exit without an
automatic restart. Stop waiting_user_approval after the actual report callback.

## Evidence

- [Worker completion](background/completion.json) and failure traceback (external historical reference: `background/failure.json`)
- Worker stderr (external historical reference: `background/worker.stderr.log`) and stdout (external historical reference: `background/worker.stdout.log`)
- Startup proof (external historical reference: `background/startup-proof.json`) and launcher receipt (external historical reference: `background/launcher-receipt.json`)
- Engineering probe failure (external historical reference: `run/engineering-probe.json`) and zero formal cursor (external historical reference: `run/progress.json`)
- Final read-only verification (external historical reference: `background/failure-verification.json`)
- [Approved execution boundaries](PARENT_RELEASE.md)

The success-only REPORT.md,result.json and background/final-manifest.json were
not created by the failed worker. Paths in completion.json are intended targets,
not evidence those success outputs exist. This failure report is a separate
post-stop record;no successful epoch is claimed.
