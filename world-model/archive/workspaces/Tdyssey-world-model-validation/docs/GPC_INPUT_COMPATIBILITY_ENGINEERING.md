# Frozen input compatibility harness engineering

A later [separately authorized attempt](GPC_INPUT_COMPATIBILITY_RUN_V1.md) on
`aecb82a` stopped during M telemetry accounting, before model loading. Its
permanent claims are spent. This page records the earlier engineering checks;
the subsequent failure is preserved separately and was not repaired or retried.
The later [storage repair](GPC_INPUT_COMPATIBILITY_STORAGE_REPAIR.md) changes the
implementation using fake files and harmless processes only; its 158-test result
supersedes the harness test count below without changing that failed attempt.
The original 106-test evidence and limits below remain historical.

The [fixed proposal](GPC_INPUT_COMPATIBILITY_PROPOSAL.md) now has an isolated
implementation, based on `1c16b61`. This is engineering verification using
constructed arrays, fake samplers and harmless sleeping processes. It is not a
compatibility experiment or an adaptation result. No saved RGB was transformed
or decoded, no checkpoint was loaded, and no real GPC forward, simulator,
policy, optimizer, installation or download ran during this work.

## Entry and execution boundary

`scripts/validate_wm_compat.py` defaults to a read-only plan. It works with
Python's `-S` option, imports no NumPy/model/decoder in that mode, writes no
output and claims no attempt. `scripts/run_guarded_wm_compat.py` also defaults
to planning. The source-only commands are:

```bash
python -S -B scripts/validate_wm_compat.py
python -S -B scripts/run_guarded_wm_compat.py --output-dir artifacts/wm-input-compat-plan-only
```

A future execution requires separate human authority, its authorization identity,
an exact reviewed clean commit, explicit new output/ledger paths, and a native
Windows guardian handoff. Both parsers reject abbreviated flags. The worker's
resolved output and ledger paths must equal the guardian's monitored paths.
The permanent registry is independent of those paths:
`artifacts/wm-input-compat-eval-v1-registry/`. Its exclusive, fsynced
`launcher-attempt.json` seals the original deadline/command/authority before
resource checks, output setup or worker binding. The worker verifies that seal
and binds its separate `attempt.json` accounting claim to the same launch.
Startup or source-binding failure therefore cannot be retried with new output,
code or authority strings. Generic harmless-process tests use temporary registries
or no production seal; the real entry requires the fixed production seal path.
This document grants no execution authority and supplies no launch command.

Before a future claim, the entry checks the old sealed inventory's own SHA-256
(`f1e2325e87df1d7dd8ef513da1ee15061ed8787ba0b6ba731d43f9d68dea5871`)
and compares every saved input payload against its original bytes/hash. It binds
the 276 NPZ, report, exact 4084 phase-two checkpoint, provenance inventory,
reviewed harness/decoder/config/protocol and upstream source identities. A
modified NPZ cannot become the new baseline simply by computing a fresh hash.
The decoder/contract/config must still match `994c6ff`; upstream source and
weights are never patched. T1 has a canonical specification hash:
`800751b7a3e32df4fee5660d1ea22c7ee0fb45b51a7b41632d420189238a7838`.

## Protocol mechanics

- `tdyssey/wm_compat_transform.py` implements exactly one RGB-only T1, with
  float64 segment projection/global-minimum tie handling and float32 output.
  Malformed real inputs are rejected; identity and unmatched pixels preserve
  their bits. M records frame hashes plus bounded palette/residual/range totals
  across every real row. These rows cover all real AR/TF context frames.
- `tdyssey/wm_compat_engine.py` separates sampler inputs from posthoc truth.
  A/B AR starts from four original/T1 real frames, then appends each raw generated
  frame unchanged. TF H2/H4 receives the corresponding saved real context,
  excluding its own target; H1 is a reference to the AR result. Actions retain
  the fixed aligned four-target window and `/511*2-1` normalization.
- Each branch/seed captures A-AR sampler-entry states, replays the matching
  horizon state for B and TF, checks entry/exit identities, and restores the
  external CUDA state. It does not reseed TF H2/H4 with H1 noise. Raw output,
  input/action/source hashes, sampler duration, RNG identities and independent
  measurements are stored separately from physical labels. Construction preserves
  CPU RNG state; the transform/array engine does not draw Python/NumPy randomness.
- `tdyssey/wm_compat_evaluation.py` performs M before any real backend import or
  construction, then the fixed 576-image T phase. A failed T screen leaves all
  1692 available L rows not attempted. The reporting module enforces the original
  M/T thresholds, filters TF out of scoring/ranking, preserves missing rows and
  compares A/B on their common complete subset while retaining independent
  physical baseline denominators. Unknown stopping prefixes remain unknown.
- The diagnostics and appearance helpers provide separate AR/TF posthoc error
  cells, validity win/tie/loss, common measurable fractions, L's 127/141 screen,
  and conditional B-AR physical-error screens. Row P90 is explicitly distinguished
  from summaries that average noises within branch and weight anchors equally.
  Contact, rotation above five degrees and true coverage in [.92,.98] are fixed
  development strata. Appearance MSE compares each arm with actual/persistence
  RGB in that arm's own palette; cross-palette MSE is not a motion claim.
  Fixed reason categories select the first six cases each. A bounded RGB332
  thumbnail plate is display-only; its manifest keeps physical labels separate
  from predicted measurements. No preview pixels enter inference or scoring.
- The lazy `TorchBackend` uses the existing unmodified sampler, one float32
  batch-one CUDA model, and its existing two-thread wrapper. It checks allocator
  peaks and a frozen-weight fingerprint. Its real loading/forward path has not
  been exercised in this engineering task.

The plan retains 2304 call rows, including 36 original missing rows, with maximum
2268 new images/6804 Euler steps, one load and zero physical/policy/update calls.
There is no warm-up, automatic rerun, resume or substitute for L01/hold.

## Accounting and guardian

`tdyssey/wm_compat_budget.py` exclusively claims the fixed registry and creates
a hash-chained, fsynced journal. One image/three Euler steps are durably reserved
before each sampler call. Pending and failed work remains charged. Immutable
chunks use exclusive files and bounded temporary space; IO failure retains
partials and permanently stops the attempt. A secondary cleanup/export failure
preserves the first stop reason. Final reports precede the successful finish
transition; `completion.json` carries the final ledger state.

Worst-case raw payload is 250,822,656 bytes. The declared metadata allocations
cover 2268 forecast records, 1072 real records, 756 RNG-state chunks, binding,
M/ranking/diagnostics/summary/status/completion exports and case previews within
12 MiB (12,578,816 reserved bytes, leaving 4,096). NPY headers reserve
580,608 bytes within 1 MiB, with a separate 1 MiB temporary-chunk reserve. The
combined worst-case reservation is 265,502,720 bytes, below 256 MiB; telemetry
has its own 16 MiB cap. No compression of raw predictions is assumed. Each
chunk is at most 1 MiB, and actual files plus outstanding reservations are checked.

`tdyssey/wm_compat_guard.py` starts the monotonic deadline before setup, reserves
one model lock and binds the actual Windows worker PID/nonce to a kill-on-close
Job Object with a 3 GiB memory limit before ACK. It requires current resource
reads and explicit zero forbidden-operation counters, including real renders.
Missing reads, stale heartbeat, invalid load phase, allocator/resource/storage
limits, unexpected operations, IO and worker failure stop the tree. It confirms
the entire Job is empty before releasing the model lock; uncertain cleanup retains
the lock. Final telemetry failures use a preallocated durable stop slot.
Child-only OMP/MKL/OpenBLAS/NumExpr thread settings are fixed to two before
NumPy/SciPy import, with dynamic OMP/MKL pools disabled. The parent environment
is preserved; the GPC wrapper also retains its two Torch compute threads.

The proposal's RAM/commit/GPU/disk floors, process limits and original 1800-second
deadline remain fixed. The worker receives only the remaining portion of that
deadline after setup; measurement, load and export do not restart it. Host/GPU
floors are sampled and are not guarantees against every transient. OS working-set
and CUDA allocator peaks add detection between samples; the Job enforces its
owned-tree memory cap independently.

## Verification and limits

Focused tests use only controlled arrays/RNG, fake backends, temporary ledgers
and deliberate bounded IO/resource faults. They cover transform boundaries,
time windows, paired noise, unmodified generated feedback, pre-call charging,
permanent claims, first-error retention, M/T failures, independent denominators,
metadata size limits and default-entry import/write boundaries.

Final focused discovery passed **106/106 tests**, with no skips, in **41.202s**
(41.516s including discovery; parent CPU 35.766s). One test is native Windows
process ownership/termination; the other 105 use arrays, records, fake backends,
injected faults or read-only/invalid-entry subprocesses. The test process imported
none of Torch, GPC backend, v2 image decoder or simulator/policy runtime modules.
This is evidence about the harness, not real model compatibility.

The complete command uses only this new test family; installed discovery of the
older full suite would include real prediction paths outside this task's scope:

```bash
python -B -m unittest discover -s tests -p 'test_wm_compat*.py'
python -S -B scripts/check_repo.py
git diff --check
```

The native Windows test uses a sleeping worker and sleeping grandchild. It checks
the actual worker differs from its launcher, verifies native Job membership and
the 3 GiB limit, then proves deadline termination and a signaled retained grandchild
handle before lock release. Its GPU/system resource provider is injected; the
test makes no GPU query. Mock resource failures are reported separately from
this native process-ownership evidence. Local evidence remains ignored under
`artifacts/wm-compat-guard-engineering-tests/` and the engineering audit directory.

Local `artifacts/wm-input-compat-engineering-20261025/verification.json` records
the test counts/scope and absence of both production launch/worker claims.
`asset-audit.json` confirms all 595 protected files (186,686,009 bytes) and all
42 v2 output-manifest entries retain their hashes. The mixed-reason long-float
diagnostic fixture occupies 140,401 bytes within its 192 KiB chunk limit; total
metadata reservation is checked independently of those observed payload sizes.

Real GPC loading/sampling, actual GPU-resource queries and saved-image M/T/L
outcomes remain unvalidated by this work. The implementation creates no evidence
that T1 improves images, predicts contact dynamics or supports policy learning.
All old results, checkpoints, interfaces and spent B/C/world-model ledgers remain
immutable. Next review this implementation and the fixed protocol before a
separately authorized guarded experiment.
