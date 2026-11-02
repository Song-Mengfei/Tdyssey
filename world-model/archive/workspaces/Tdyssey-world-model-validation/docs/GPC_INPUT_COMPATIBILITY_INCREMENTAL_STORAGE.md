# Incremental storage accounting: engineering verification

This engineering follow-up starts from `fb29cf68969f89898838841876f5d22ac391d011`.
The user requests a simpler runtime and a complete fake-data flow before any
later real experiment. No formal run, saved-image decoder/transform, real model,
physics, training, installation or download is authorized here. Both v1 and v2
claims remain spent. Execution identities and the scientific protocol are unchanged.

## Runtime responsibilities

The worker and guardian share one telemetry cap. A small mapped counter page
coordinates byte reservations so two writers cannot each admit bytes against
the same remaining allowance. The process mutex protects reservation/commit
updates, rather than file contents or model computation. Independent fixed
writer quotas or a single writer could avoid that mutex; this implementation
retains the existing shared cap and permits either owner to use its remaining
capacity.

`Coordinator.usage()` reads the counters. It does not enumerate output trees,
open historical results, recalculate their hashes or reread journal prefixes.
Every growing write first reserves its maximum bytes. IO then runs outside the
shared quota mutex. Successful completion records actual bytes and releases
only a temporary allowance; unused final bounds stay charged. Failed writes
retain their reservations. Duplicate commits are rejected under the same mutex.
`CompatBudget.check()` likewise returns detached status/counters without
rebuilding the 2304-call table or deep-copying the growing output history.
The complete report remains available through explicit `snapshot()` calls.

The mapped page is 4096 bytes and its one-byte mutex file is also charged.
It is admission state, not a durable recovery format. The fixed exclusive
claims and pre-call journal still prevent reopening an attempt after a crash.
The mutex file's byte position is used for OS locking; its byte value is not
application data and needs no read during contention. Its file type, identity
and size are checked. Windows mandatory byte locking is covered by a real
cross-process contention regression.

The worker owns model/image/Euler accounting and journal order under its own
local lock. A reservation event must still be fsynced before a sampler call.
Claim creation, critical journal admission, first-failure and terminal records
retain their durability rules. Ordinary monitor samples and stdout do not
fsync every payload; they are flushed at explicit durable/final boundaries.
Output journal events store the relative name and category under the already
bound roots, rather than repeating absolute destination/temporary paths in
both reservation and completion events. Bounds, SHA-256, file identity, status
and the hash chain remain. Full in-memory output records and explicit snapshots
are unchanged. The final reader reconstructs paths and accepts legacy records
only when their absolute paths agree with the allowed roots; traversal and
root mismatches are rejected.

Status target plus temporary, the guardian stop slot, the worker's cumulative
64 KiB emergency allowance and external model-lock allowance are prepaid.
Heartbeat publication therefore needs no growing-byte admission. The existing
2 MiB scratch allowance covers one outstanding result publication. Another
result publication is refused while it is outstanding; a failed publication
does not release it. Result/telemetry caps remain 256/16 MiB, and the quota-lock
timeout remains 0.5 seconds.

Windows status publication uses share-delete readers and directly selects
`ReplaceFileW` for an existing status target. Byte-only native regressions prove
old readers retain the old complete JSON while a new reader sees the new one.
This uses the documented [Windows replacement API](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-replacefilew),
with no ACL-ignore flags or WinError5 fallback. First publication remains
exclusive. Only a missing exact status path or native sharing/lock codes 32/33
at its open step can reuse the previous live snapshot, without changing its
timestamp or the original five-second freshness limit. Other permission/IO
errors and corrupt data still fail immediately. Successful exit requires a
fresh final `complete` status, so cached data cannot hide a terminal deletion.

Shared status writers now make one publication attempt and retain failed
temporary bytes for the existing orphan refusal. The later
[publication review](GPC_STATUS_PUBLICATION_REVIEW.md) observed transient name
absence under continuous readers and rejected a MoveFileExW candidate under
legal held readers. ReplaceFileW remains; pilot03's1175 is unresolved. The earlier
single held-reader success does not establish uninterrupted name availability.

## Integrity boundaries

Every newly completed immutable file is fingerprinted once, outside the quota
mutex, and recorded in the owner manifest. Completing one file inspects that
file rather than scanning its entire directory. Append owners validate their
current descriptor/path identity, size and change marker; their digest is
updated from new payload bytes only. Path traversal, links/reparse points,
nonordinary files, short writes and current IO failures still stop execution.

Full owner audits run at M, T and successful finish boundaries. They cover the
worker's results, ledger and claims, excluding concurrently mutable guardian
telemetry. The guardian's final full audit runs after the worker tree and stdout
pump are quiescent. It validates the durable journal chain and derives expected
output fingerprints from completion records, rather than trusting another
process's in-memory manifest. It also verifies its own files and physical byte
totals against the reserved quota.

This changes detection timing deliberately: quota snapshots do not promise
continuous detection of unrelated historical file changes. Missing, modified
or unregistered owner files are rejected at the explicit audit boundary. The
original input/weight/source binding checks and final content audits remain;
the code does not accept modified scientific inputs as a new baseline.

## Fake-flow verification and evidence

The constructed-data fixture uses the real evaluation engine, gates, sink,
budget and guardian lifecycle, with synthetic RGB, measurements, backend and
resource values. It exercises the original M and T/L row counts and complete
exports, including the planned missing branch. It never reads saved real RGB
or imports Torch, the real decoder, simulator or policy runtime. Native process
ownership/Job cleanup is distinguished from its injected resource counters.

The fixture deliberately pauses M file flush and forecast file publication
longer than the 0.5-second quota timeout while other-process quota admission,
heartbeat, monitor and stdout must progress. Hot-path tree scans and repeated
old-file hashes are forbidden by instrumentation. Audit-boundary hashes remain
enabled. Final measured results are recorded in the new ignored directory
`artifacts/wm-input-compat-incremental-engineering-20261025/`.

The before inventory seals 806 files / 202,935,362 bytes across 30 roots,
including both stopped formal attempts and all four claims. Earlier regenerated
repair fixture logs are recorded as historical exceptions, not new changes.
This follow-up's raw fixtures are disposable workspace directories; persistent
test evidence stays in the new engineering directory.
Intermediate fake failures are retained: held-reader replacement behavior,
Windows mandatory mutex-byte reads, and the distinction between initial missing
status and temporarily unavailable live status were found before a formal run.
They were repaired in the engineering fixture/source, with regression tests;
their existence is not a claim to have identified every old incident's cause.
A later fake run reached 2167 prediction outputs before the original 16 MiB
telemetry cap stopped it. Its journal alone was 12,354,910 bytes. This exposed
the redundant path fields above. During compact-format verification the cap,
production heartbeat/polling intervals and stdout stress cadence stay fixed.
The failure record is retained as
`fake-flow-4e0413c652a14997b54c5222a02e1a1e.json`.

The compact-format native fake flow passed. Evidence is
`fake-flow-d274878c69bb4df9a5581dd83ced6db0.json` (21,199 bytes; SHA-256
`ffa90aa16eb6b7d0961c506a27b03a11920d951ba94afc90003788305e9ea514`).
The guardian ran for 897.140 seconds within its original 1800-second deadline;
the fixture, including cleanup, took 927.406 seconds. Its native worker PID was
27124 and guardian PID 9488. Resource counters were injected; actual PID/nonce
ownership, the 3 GiB Job and an empty process tree before close were native checks.

| Check | Measured fake-flow result |
| --- | --- |
| M | 1072 planned rows, 1056 measurements, 16 missing; passed |
| Forecasts | T 576 and L 1692 complete; 1512 AR / 756 TF; 36 original missing calls |
| Actual raw files | 2268 files; 250,822,656 payload bytes; 251,112,960 serialized bytes |
| Fake accounting | One fake backend; 2268 sampler and 6804 denoiser reservations/calls; zero real model operations |
| Exports and journal | 6372 completed outputs; 17,286 chained events; 2268 durable call admissions/completions; no pending calls |
| Results | 265,980,928 charged / 256,346,555 physical bytes, below 256 MiB |
| Telemetry | 11,818,428 charged / 11,556,771 physical bytes, below 16 MiB |
| Integrity instrumentation | Three owner audits; zero hot-path tree scans; each new immutable file hashed once |
| Lifecycle | Fresh final status; worker/launcher exit 0; final parent audit and Job cleanup passed |

Charged bytes retain unused maximum bounds, prepaid slots and scratch. Physical
telemetry above is the quiescent audit before the guardian's own final report
and terminal monitor record. It is not the sum of every later evidence file.
The worker's 1799-second local bound is the rounded-down remainder of the
unchanged guardian deadline, rather than a new or extended clock.

M payload fsync and forecast publication were paused for 1.266 and 1.250 seconds.
During each pause, another process admitted quota in 0 / 15 ms at the sampled
monotonic-clock resolution, while heartbeat, monitor and stdout all advanced.
The largest measured counter-update body was 7.485 ms; this excludes acquisition
and setup and is not a universal upper bound on OS lock latency. The original
0.5-second acquisition timeout, 1-second heartbeat, 0.2-second polling interval,
16 MiB telemetry cap and stdout stress cadence were unchanged.

The other 180 focused tests passed in 77.458 runner seconds / 78.109 wall seconds,
with zero failures/errors/skips (`focused-other-02.json`). That run excluded the
active native full-flow test. The four incremental tests separately passed in
928.335 seconds; three overlap the 180-test run. These cover 181 distinct checks
at the full-flow milestone, rather than two independent additive suite totals
or a claimed single-suite wall time. Final terminal-record regressions are
reported separately below.

Subsequent small-file review reproduced a finalization gap: a journal truncated
to a complete valid prefix could pass its own chain check, and missing declaration
could skip that check. Normal exits with a configured ledger now require the
ordinary declaration/journal and their unique final `completion.json` output
record, with its bounded payload/hash checked. A controlled gate stop's `stopped`
completion is valid; it is not required to pretend that M/T/L all passed. Failure
paths retain partial-journal inspection. Six proper prefixes, missing files,
changed payloads and invalid completion status are covered by two new regressions.
All 53 guardian tests passed in 13.414 seconds, including these two; evidence is
`guard/guard-terminal-boundary-1e3828c439064184a2444c466f2f36cd.json` (929 bytes; SHA-256
`5a12ec415e18c3b5f0512dfda155768dd5688c827036237a610d3fc882616cab`).
The native full fake flow above preceded this finalization-only patch and was
not rerun afterward. Its retained last event is the unique final
`output_complete` for `results/completion.json`, sequence 17,286, and its parent
content audit passed; the retained terminal record meets the added rule.
Coverage totals 183 distinct checks after the two additions; the 53-test rerun
overlaps the earlier coverage and is not another 53 unique tests.
The deduplicated IDs, evidence hashes and retained terminal-rule reconciliation
are in `integration-verification-7cbe7db4eabc40d689b4030154a432b2.json`.

The closing byte audit passed for all 806 sealed files / 202,935,362 bytes across
30 roots: zero added, missing, changed files or membership differences. This
includes all four formal claims (4693 bytes), v1's 53 files / 760,099 bytes,
v2's 37 files / 683,850 bytes and all 150 B/C files / 97,364,187 bytes. The before
inventory itself remained unchanged. New `after.json` is 289,056 bytes, SHA-256
`429baf1a9f08a8710a18b405f308f952b771078631c98ee007f3abf19a477565`;
`reconciliation.json` is 2756 bytes, SHA-256
`ccc935abcdb9bfef5f5bff4e9e3469f1add51a3d744691cd9536efea044eea85`.

Reproduce engineering checks only with the existing development environment:

```bash
python -B -m unittest discover -s tests -p 'test_wm_compat*.py'
python -S -B scripts/check_repo.py
git diff --check
```

Passing fake data is evidence about accounting, publication and orchestration.
It does not validate GPC predictions, real image measurement, model residency
or scientific gates, and does not authorize a new formal attempt.
