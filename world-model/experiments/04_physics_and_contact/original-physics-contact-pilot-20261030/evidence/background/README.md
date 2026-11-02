# Finite independent collection supervisor

Engineering implementation only. No formal collection is released by these files.
The root agent freezes the real `runtime-config.json`, reviews calibration/resource
evidence and explicitly releases it before launching. Check-only never starts a
subprocess, executes physics, imports a model or writes a result.

`common.py`, `launcher.py`, `worker.py` use only the Python standard library. Reuse
the existing venv; do not install packages or copy a checkpoint. The worker binds
the unchanged7499 checkpoint, but never loads it. The separate live UI is untouched.

## Independent calibration-v2 entry point

`calibration-config.example.json` is a non-releasable schema example. Root freezes
the real config at `background/calibration-v2/runtime-config.json`, with
`mode:calibration-v2`, `run_label:calibration-v2`, the18 original cal-v2 plan slots,
separate cal-v2 gate/source/resource/policy/notification/checkpoint bindings and an
immutable cal-v1 ledger snapshot. `calibration_policy` replaces formal/supplement
policy keys; no formal/supplement slot release or overview script is required.
Source bindings include shared `common.py`/`worker.py` helpers plus both calibration
entry-point files and actual collection/physics dependencies. The unused formal
launcher and saved-report module are not mandatory calibration dependencies;
binding additional files freezes them too, so include only deliberately chosen
execution dependencies if report formatting must continue during calibration.

Both `calibration-worker.py --config CONFIG --config-sha SHA --check-only` and
`calibration-launcher.py ... --check-only` read/validate only, with0physics,
0model,0subprocesses and0file writes. After root releases and actually dispatches
`calibration-launcher.py` through the verified existing Explorer desktop broker,
its one worker independently proves the same parent-exit/Job/venv conditions and
starts exactly one collector CLI for the original18 slots. It writes independent
status/heartbeat/completion/startup/manifest/failed evidence under
`background/calibration-v2/`; the collector owns `collection/runs/calibration-v2/`.
No alternative label, slot skip, physical rerun, automatic formal phase, model,
training or report generation occurs in this worker. Root/official30min monitor
reads the real terminal result, can build a saved-only report, and reviews cal-v2
before separately freezing/releasing formal. Completion leaves callbacks explicitly
unsent and grants no formal approval. Failure retains actual partial costs and
requests its own collector's boundary stop, never process termination/restart.

## Collector interface contract (must be implemented before freeze)

```
python -X utf8 -B scripts/original_contact_pilot.py collect
  --output COLLECTION --plan PLAN --phase formal|supplement
  --physics-gate GATE --source-root PINNED_DIFFUSION_POLICY
  --storage-roots-json ROOTS --policy-revision STRATEGY
  --resource-policy RESOURCES --letters TLO --run-label LABEL --stop-file STOP
```

The two CLI processes receive disjoint letter sets. Trajectory files retain the
one original path `COLLECTION/trajectories/<slot_id>/`. Each label exclusively owns
`COLLECTION/runs/<label>/{identity.json,ledger.json,events.jsonl,progress.json,completion.json}`.
Actual collector labels must be lowercase `[a-z0-9][a-z0-9_-]{0,63}` and avoid
Windows reserved names. Use `formal-tlo`/`formal-cxb` and
`supplement-tlo`/`supplement-cxb`; the letter selectors themselves remain uppercase
`TLO`/`CXB`. The fixed `calibration-v2` label already meets this actual contract.
Neither shard may write `COLLECTION/ledger.json` or a shared progress/event file.
The supervisor creates each run directory first to hold its stdout/stderr/process
receipt; collector must accept that existing directory with no existing ledger.
An existing scene claim is not permission to retry or silently skip requested work.

Identity and completion receipts must contain `phase`, `run_label`, `plan_sha256`,
`physics_protocol_sha256`, `physics_gate_sha256`, `resource_policy_sha256`,
`policy_revision_sha256`, and zero `model_calls`/`optimizer_updates`. Completion
must contain `completed: true` only after every selected slot was attempted, plus
`graceful_stop: false`; this does not establish that all attempts were valid. Any
substantive physical/config/storage failure must persist its actual partial cost,
stop the shard and write an explicit failed completion / nonzero CLI exit. The
supervisor separately requires all64 actual controls valid before success.

At every trajectory boundary, before constructing the next environment, collector
checks its **own** stop-file. If present it writes `graceful_stop: true`, preserves
all prior actual costs and exits without claiming another slot. No mid-control
termination, process kill, restart, missing-slot fill or replacement scene.

Resource policy JSON must at least bind integer `shared_cap_bytes:3221225472`,
`atomic_reserve_bytes:268435456`, and the explicitly reviewed operational
`collection_cap_bytes`. Root/collector must account for both concurrent shard
atomic allocations within the same hardcap/reserve. Resource policy must never
silently change frozen scene/rules/physics identities.
Root also freezes an explicit `postprocess_reserve_bytes` (planned128MiB) before
collection. Collection guards retain that budget; report/video generation must
check before each bounded write, keep the separate64MiB video limit and respect
the postprocess budget while preserving256MiB shared reserve. Supervisor checks
alone before/after a subprocess do not establish safety during its writes. The
collector/report write guard must be delivered and independently reviewed before
formal launch. Insufficient space means an accurate partial failure, never cap
expansion or unique-evidence deletion. The worst possible72-supplement run is not
promised to fit.

## Runtime config contract

All paths are absolute local paths. Every `{path,sha256}` binding is a SHA256 of
the complete real file bytes. `config.example.json` shows field names; it is a
non-releasable template, never an actual runtime identity. Root generates slot
arrays directly from the original immutable plan, not observed outcomes.

Required scalar keys: `kind:tdyssey-contact-background.v1`, unique `run_id`,
`released:true` only after review, `repo`, `artifact_root`, `collection`,
`python_executable`, `source_root`, `heartbeat_seconds:30`, `auto_retry:false`,
`training_authorized:false`, `model_calls:0`, `optimizer_updates:0`.

Required bound files: `plan`, `physics_gate`, `checkpoint`, `storage_roots`,
`resource_policy`, `formal_policy`, `supplement_policy`, `calibration_ledger`,
`collector_script`, `overview_script`, `notification`. Policy files contain a
`changes` object, including an explicit empty object when unchanged.

`source_bindings` is an array of `{path,sha256}` covering all code used, including
the three coordinator files, collector, report, CLI, overview, v6/original physics
and actual pinned upstream environment dependencies. The gate must independently
bind passed physical evidence and the unchanged protocol/checkpoint. **Every**
dependency in `gate.bindings` is checked; preserve the original calibration gate
and create a separate administrative/formal gate when current report/collector
SHAs differ. Never edit old execution evidence to fit new code.

`calibration_ledger` must reference an immutable frozen snapshot of the current
cal-v1/v2 ledger, never the later merged `COLLECTION/ledger.json`.
`formal_slot_ids` must list all and only the360 plan formal slots. The original
18/36 calibration attempts are retained but never replace any formal slot.
`supplement_slot_ids` maps each of six letters to its original12 plan supplement
slots. `shards` contains exactly two objects, each with unique `formal_label`,
`supplement_label` and compact `letters`; letters are disjoint and cover TLOCXB.

The supervisor refuses duplicate reservations, changed plan/split/letter/protocol,
metadata/ledger disagreement, raw byte/SHA mismatch, cost totals inconsistent
with attempted controls, incomplete formal360 or model/optimizer activity. It
merges frozen calibration+formal shard ledgers only after both shards exit, then
runs `report --videos` on saved arrays. All valid actual formal H1 CSV controls,
including controls without sufficient model history, determine the per-letter
trigger: **strictly >50%** translation<.5 image px, **or no** H1 rotation>=5deg
trajectory. Only one finite supplement phase is possible, with12 original slots
per triggered letter and the separately frozen supplement policy. Calibration,
candidate windows and supplements cannot alter this trigger denominator. There
is no second supplement decision or retry loop.

## Zero-work verification and later launch

Run `python -X utf8 -B background/test_worker.py` for standard-library synthetic
identity/denominator/corruption/failure checks. Fixtures are explicitly synthetic,
created under background and automatically removed. No actual physics/model data
is generated. `preflight.py` inventories real saved calibration and code readiness
without starting any worker. Once root freezes a real config:

```
EXISTING_VENV_PYTHON -X utf8 -B background/worker.py --config CONFIG --config-sha SHA --check-only
EXISTING_VENV_PYTHON -X utf8 -B background/launcher.py --config CONFIG --config-sha SHA --check-only
```

The root launches the real `launcher.py` once through the already approved
Explorer desktop COM broker (hidden ShellExecute), with the exact frozen config
and SHA. Launcher uses actual `GetModuleFileNameW` PE plus
`__PYVENV_LAUNCHER__=EXISTING_VENV_PYTHON`, `DETACHED_PROCESS`,
`CREATE_NEW_PROCESS_GROUP`, `CREATE_BREAKAWAY_FROM_JOB`, hidden startup. It writes
a spawn receipt and exits; no fallback drops breakaway protection. Before first
control, the worker proves `IsProcessInJob=false`, expected venv, and actual
launcher exit via Win32 wait. Child collectors inherit that independent process
context; their actual Job membership is also recorded. No new terminal window.

The verified broker is the **existing Explorer desktop** COM application:
`Shell.Application.Windows().FindWindowSW([ref]0,[ref]0,8,[ref]hwnd,1).Document.Application.ShellExecute(...,0)`.
Creating `New-Object -ComObject Shell.Application` and directly calling its
`.ShellExecute` is a different route and did **not** escape the Job in actual
probe002, despite successful Popen with all breakaway flags. Preserve that failed
receipt in `isolation-probes/isolation-20261031-002/`. Initial probe001 produced
no receipt and is not evidence of a started child. Root's actual probe003 through
the existing desktop broker passed: child3152 saw actual parent34436 exit,
`IsProcessInJob=false` twice and the correct existing venv, with zero physics/model
calls. Actual evidence: `isolation-probes/isolation-20261031-003/child-result.json`.
Formal still independently checks its own Job/parent-exit receipts; the probe does
not substitute for those real-run startup checks.

Before formal launch, root can separately invoke `isolation_probe.py --label UNIQUE
--python-executable EXISTING_VENV_PYTHON` through that same hidden broker. This
fixed standard-library probe starts only its own probe child, never the supervisor,
collector, simulator or model. Its actual parent-exit/Job/venv receipts remain in
`isolation-probes/UNIQUE/`. It has not run merely because the script exists; root
must inspect the real `child-result.json` before claiming isolation is proven.

Existing launch/worker claims or completion refuse automatic resume. If launch
or source validation fails, inspect real receipt/stderr and request an owner
audit; do not delete claims and rerun blindly. This release has no auto retry.

`status.json` and `heartbeat.json` update every30s locally. The **official app
automation** checks at30min and stays quiet unless completion/failure/identity
mismatch requires action. The worker writes `completion.json` and retains costs,
raw data, saved-only reports, hashes and limitation statements. It cannot perform
MCP callback while Codex is closed, so callback remains explicitly `sent:false`.
Only root/app monitor can actually send the parent chat callback, persist the
exact message+real tool receipt, set SUPERVISION/stage-state to
`waiting_user_approval`, and formally pause its automation. No report/failure
automatically authorizes training, further collection, model changes or remote work.
