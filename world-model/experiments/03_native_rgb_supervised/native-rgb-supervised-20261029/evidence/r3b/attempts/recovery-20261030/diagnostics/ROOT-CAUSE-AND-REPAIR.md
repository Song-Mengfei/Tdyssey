# Confirmed Windows loader-startup cause and CPU repair verification

The default detached launch selected the virtual-environment redirector for
multiprocessing. This created a second actual Python process under each PID that
owned the duplicated queue handles. The first worker completed Python imports
and the original R3A dataset initializer, then failed while releasing the queue
SemLock with `OSError: [WinError 6] invalid handle`. This was neither a renderer
data-read exception nor a CUDA/native-model failure.

The evidence-bound repair uses the public `multiprocessing.set_executable` API
to launch the actual base Python PE directly, while inheriting
`__PYVENV_LAUNCHER__` with the existing venv Python identity. It does not patch
Python's standard library, replace the existing venv, install dependencies, reduce
the two-worker pool, or change data, renderer, physics, caches or sampling.

## Controlled default/repaired comparison

Both diagnostics used the same hidden independent launch path, repository cwd,
existing venv, diagnostic logging hook and original R3A dataset/loader. Each
parent verified its launcher had exited before creating the process pool. The
logging hook records only necessary Python configuration and does not choose
the executable. Only the explicit executable/venv repair distinguishes the
repaired spawn.

| Observation | Default diagnostic | Repaired diagnostic |
| --- | --- | --- |
| Parent PID | 12416 | 23420 |
| Parent actual PE | Base Python | Base Python |
| Parent executable and base-executable aliases | Both venv redirector | Both remain venv redirector |
| CreateProcess target | Venv redirector | Actual base Python |
| Returned process PIDs | 33472,18744 | 33696,31700 |
| Actual CPython PIDs | 34056,34772 | 33696,31700 |
| Queue handle recipient matches actual worker | No | Yes, both |
| Completed batches | 0 | 4 x16 |
| Exact indexed RGB/action comparison | Not reached | All64 windows pass |
| Exit codes | 1;other pool member terminated(-15) | 0,0 |
| Recorded wall time | 5.266s | 4.610s |

The default second actual Python reached early bootstrap but was terminated when
the first failure broke the pool. Its -15 is not reported as an independent
second root cause. The first child's stderr identifies the actual WinError6;
the broad parent BrokenProcessPool is only the resulting wrapper exception.

The repaired run used64distinct selected windows from64different trajectory
families. Both actual Python worker PIDs retained the existing venv and reported
Job=false. Maximum in-flight batches remained2. The original worker dataset
fetched64windows and rendered320frames;the independent parent reference fetched
each window once per input key(128getitem calls) and rendered320frames with
320cache hits. All image/action arrays,shapes and float32 dtypes matched exactly.
These192CPU fetches represent64distinct inspected windows,not192distinct examples
or formal/model exposures. Both diagnostics imported no Torch,initialized no
CUDA,called no model or optimizer,and advanced0physics controls.

## Costs, scope and remaining gates

The two detached CPU diagnostics used9.876recorded seconds in total. Two focused
CPU regression tests passed(core test time0.456s):the diagnostic hook preserves
the exact CreateProcess arguments and return value without recording unrelated
environment entries,and initializer failures retain their original exception
and traceback. These tests do not substitute for the real detached run.

The original failed GPU-restoration attempt remains unchanged with33.047model
seconds and0engineering/formal optimizer attempts. CPU diagnostic reads do not
advance the frozen formal cursor or count as formal training draws. No first
epoch,gradient/update/checkpoint probe or model-quality result is claimed by this
CPU report. The full worker must still verify these same loaders with the CUDA
model resident,pass the approved engineering probe,and execute all1499formal
batches plus the original paired evaluation/report.

## Evidence

- Confirmed cause and raw bindings (external historical reference: `root-cause.json`)
- [Default completion and exit codes](diag01-default/completion.json)
- Actual first-worker error (external historical reference: `diag01-default/child-34056.stderr.log`)
- Default CreateProcess trace (external historical reference: `diag01-default/bootstrap-12416.jsonl`)
- Default source snapshot (external historical reference: `diag01-default/source-snapshot/source-identity.json`)
- [Repaired completion,exact inputs and exit codes](diag02-repaired/completion.json)
- Repaired gate (external historical reference: `diag02-repaired/gate.json`)
- Repair proof (external historical reference: `diag02-repaired/spawn-repair.json`)
- [Structured diagnostic counts and costs](diagnostic-summary.json)
- Source/evidence identity (external historical reference: `diagnostic-identity.json`)
