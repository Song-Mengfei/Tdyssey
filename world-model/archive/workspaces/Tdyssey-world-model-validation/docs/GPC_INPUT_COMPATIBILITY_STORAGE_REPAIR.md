# Input compatibility storage repair

Engineering work based on `e5f2b793379632cf658d18ce541bddfa5c8fa664` repairs
concurrent storage accounting after the [single failed attempt](GPC_INPUT_COMPATIBILITY_RUN_V1.md).
It does not resume that attempt or grant another one. Both permanent claims,
the failed ledger/guardian and all 40 durable calibration measurements remain
unchanged. No actual saved-image transform/decoder, weight load, GPC forward or
sample, simulator operation, training, installation or download ran in this work.

## Reproduced mechanisms and attribution limit

A deterministic fixture uses temporary fake files and the old scanner source:
the producer fsyncs and closes a status temporary; the scanner enumerates its
name; the producer successfully replaces the final status using the actual
atomic writer; then the old scanner's `is_file()` returns false for the vanished
temporary and raises `Only ordinary output files are allowed`. Events/barriers
fix the order without depending on a timing sleep. The final status is present,
the temporary is absent and an unrelated immutable fixture retains its hash.

A second fixture exposes the old guardian's inconsistent snapshot: it counts a
3-byte final file, a legal replacement publishes a 101-byte final, and the scan
reports 3 bytes after swallowing the temporary's disappearance. Another fixture
shows that blanket `FileNotFoundError` handling also hides a missing completed
result. These establish reproducible engineering defects. The original incident
did not log its offending path, so its exact cause remains unproven.

Local evidence is kept in the new ignored directory
`artifacts/wm-input-compat-race-engineering-20261025/`, including
`old-scanner-race.json`, `guard-legacy-replacement-window.json` and
`guard-legacy-immutable-disappearance.json`. Existing run/audit evidence was not
rewritten. Tests that emulate replacement in restricted temporary directories
are labeled separately from workspace fixtures using actual `os.replace`.

## Storage rules

`tdyssey/wm_compat_storage.py` supplies stable file fingerprints and a bounded,
reentrant process lock. The guardian creates its one-byte lock exclusively and
passes an exact storage policy through the verified PID/nonce handoff. Budget
mutations, journal appends, output publication, heartbeat replacement, guardian
records, stdout capture and storage snapshots use that same lock. Handoff
validation rejects changes to roots, mutable paths, reservations, limits or the
0.5-second lock timeout and shared emergency headroom. Contention beyond the
timeout stops the attempt.

| File class | Validation and accounting |
| --- | --- |
| Completed results, claims, declarations and other ordinary files | Identity, size, modification time and SHA-256 stay fixed after observation; disappearance or replacement fails |
| Exact worker-status path | Replacement allowed only for this declared target and one exact `<target>.<32 lowercase hex>.tmp`; both 64 KiB slots are always charged |
| Ledger journal, monitor journal and captured stdout | Same file identity, no shrinkage, and unchanged previously observed byte prefix; growth checked before append |
| Preallocated guardian stop record | Same identity, at most 64 KiB; reserved before failures so the first cause can still be recorded |
| Pending output destination and temporary | Explicit bounded paths within results/ledger exports, at most 1 MiB each; observed bytes plus unused reservation are charged |
| Other `.tmp` files | Ordinary immutable files; suffix alone gives no exemption or deletion permission |

Scanning checks directories and file descriptors before and after reads. Parent
traversal, symlinks, Windows reparse points, multiple hard links, nonordinary
files, missing observed files, permission failures and other IO errors fail
closed. Only an explicitly uncreated root/tail may initially be absent; an
observed root cannot later disappear. No scan exception is broadly ignored.

The guardian can reuse an already verified SHA only when file identity, size,
mtime and a strong OS change marker all agree. On Windows this marker is native
`FILE_BASIC_INFO.ChangeTime`, not the creation-time `st_ctime` value. Even cached
files are opened and restated, so permission failures remain visible. Budget
result checks still read and hash contents; successful budget finish and final
guardian reconciliation force fresh hashing across all shared roots.

Status writes check payload size before creating a temporary. Publication and
cleanup remain inside the lock; an orphan temporary consumes its reservation
and prevents another publication. Retry is bounded to Windows sharing/locking
violations, not generic permission failures. Child stdout is pumped through a
bounded pipe and charged before append, avoiding uncontrolled inherited-file
growth. Append/fsync/replace and scan failures report path, phase, original
exception type and errno. Secondary cleanup/export failures retain the first
reason and record secondary diagnostics. Budget emergency records consume their
previously reserved headroom cumulatively without rescanning a known bad tree.
Every normal telemetry writer preserves the same 64 KiB emergency allowance
before admitting growth. A cached budget snapshot therefore cannot let a
guardian append consume that allowance while a stopped budget waits for the
lock. The allowance is not a new cap or permission to continue operations.
Monitor/stdout appends use unbuffered file handles and write only the admitted
payload under the lock, with positive short writes completed and zero/invalid
progress rejected. Each append is fsynced. Closing a failed append stream cannot
replay a hidden Python write buffer after other writers consume available space.

No experiment/resource limits were enlarged. Result and telemetry caps remain
256 MiB and 16 MiB. The parent conservatively charges a further 2 MiB for a
pending destination plus temporary, even before creation. The closed-form upper
bound is 265,502,720 + 2,097,152 = **267,599,872 bytes**, below 268,435,456 by
835,584 bytes. Actual status overlap, append growth and pending reservations are
also checked against the existing caps; over-cap bytes are rejected before write.
Original model/image/Euler counters, 1800-second deadline, thread settings,
resource floors, 3 GiB Job cap and permanent single-attempt rules remain fixed.

## Verification and preservation

Final focused discovery passed **158/158 tests**, zero failures/errors/skips, in
**211.828 wall seconds** including discovery (158.406 parent CPU seconds;
211.427 seconds in the test runner). The recorded runner found
none of Torch, simulator, policy, GPC model wrapper or v2 decoder imported.
Tests use constructed arrays, fake records/samplers, temporary filesystem faults
and harmless Windows processes. They include both complete fake plan cap loops,
missing/tampered immutable files, restored-mtime changes, reparse/IO failures,
append-prefix corruption, byte boundaries, pending reservations, multi-instance
and cross-process lock contention, first-cause retention and strict handoff policy.
The stale-cache lock-timeout regression checks shared emergency headroom at the
byte boundary. Partial monitor/stdout write failures followed by other admitted
growth and close add zero replayed bytes and retain their first IO cause.
Forty real heartbeat publications, forty synthetic journal appends and forty
snapshots under shared coordination complete without scan errors or leftover
temporaries. A 100-byte stdout attempt against a 64-byte fixture cap writes zero.

The native test verifies the actual worker PID differs from its launcher, native
Job membership, the 3 GiB limit and kill-on-close, a signaled retained grandchild
handle, and an empty Job before model-lock release. Its latest 2.5-second deadline
test completed in 2.547 seconds. OS process ownership is real; resource/GPU
counters are injected and no GPU query runs. This does not validate model memory
residency or real GPC throughput. POSIX locking support has no native POSIX
runtime validation from this Windows check.

`full-focused-03.json` records the final complete suite; earlier 153/155-test
records remain as intermediate evidence. `guard/native-last.json` records native
process evidence. Before/after integrity inventories cover 703
distinct protected files (199,402,329 bytes) and membership of 23 protected
artifact roots: the 595 old files, 42 v2 payloads, all 53 stopped-run files
(760,099 bytes, including both claims), and 13 reference/posthoc metadata files.
The closing audit reports no changed, missing, extra or unreadable protected
files. Source changes are intentionally outside this asset baseline.
The final audit follows `full-focused-03.json`, whose SHA-256 is
`b730b0fea65ec3f92b5a220597a69776ee560088bc40cf98c37b27f50be30da5`.
`after-integrity-02.json` SHA-256 is
`c72a86f14d3fb9675f993cf2a3ebc7cece94f2814be4d64e852f917a29a0cabd`;
`integrity-reconciliation-02.json` SHA-256 is
`41ff97f6ac00628e618e3491da3f2525dfafe1b648b632acbe8405d85263008b`.

Reproduce engineering checks only, using an existing environment with NumPy:

```bash
python -B -m unittest discover -s tests -p 'test_wm_compat*.py'
python -S -B scripts/check_repo.py
git diff --check
```

These checks do not invoke the real experiment entry. Full installed discovery
of unrelated model/physics suites is outside this task. Repeated content checks,
append-prefix verification, fsync and bounded lock contention can consume time;
their fit within a real GPC run remains unvalidated. The repaired implementation
does not establish T1 benefit, visual/contact accuracy, or permission to retry.
The failed attempt remains spent; any later real experiment needs a separate
reviewed decision.
