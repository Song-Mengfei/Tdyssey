# Frozen input comparison v1: stopped during measurement

The separately authorized single attempt on reviewed source
`aecb82a6507d87879217fb39441131e8a04b47d0` stopped after **25.906 seconds** in
M, before a gate decision or model load. The first error was
`BudgetError: Only ordinary output files are allowed`, raised while accounting
for telemetry files before reserving the next measurement export. This is an
operational failure, not a failed scientific M/T screen. **M is incomplete;
T and L are not attempted.** Model loads, forecast images and Euler steps are
all zero. The permanent launch and worker claims remain spent; there was no
repair, retry, resume, replacement scene or deadline extension.

This report follows the [fixed proposal](GPC_INPUT_COMPATIBILITY_PROPOSAL.md)
and [engineering implementation](GPC_INPUT_COMPATIBILITY_ENGINEERING.md).
Results are development observations on previously examined saved data.

## Authority, identity and actual execution

The launch seal was written at **2026-10-25 05:18:22.519 SGT**, with the original
1800-second monotonic deadline. The guardian finished at **05:18:48.438 SGT**.
The worker received only the remaining deadline, not a fresh 1800 seconds.
Authorization identity: `wm-input-compat-eval-v1-20261025-user-authorized`.

Startup checks confirmed a clean exact `aecb82a`, no existing production
registry/output/ledger/model lock, and no Python processes. The CIM query was
unavailable; the ordinary native process list supplied the process check. Earlier
dispatcher cancellations occurred before dispatch and left no launch claim.
Only **one actual guardian launch** occurred. Later interrupted status-poll calls
did not launch anything; the durable stop record and native process check
confirmed termination.

The run bound 84 code/config/protocol files and 280 asset/provenance/seal files.
All still matched their recorded hashes after stopping. Important identities:

- Protocol SHA-256: `2bfff44111d41f5c54a626b717ec2ea35f4bee5051b9fbeb19926f8fa4dbacf2`.
- T1 specification: `800751b7a3e32df4fee5660d1ea22c7ee0fb45b51a7b41632d420189238a7838`.
- Phase-two checkpoint: 35,043,916 bytes,
  `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
- GPC revision: `a96146e568826ef46134f73c22fdf59a675e7264`; all 20 pinned
  upstream source hashes matched. The legacy csgo weights were not substituted.
- Saved report: `c7b1e3257468874940ad074d3ffc00c3d3d88f9f6de16531aba6d2c3cc6e3795`.

Native guardian/launcher/actual-worker PIDs were **6384 /31200 /16060**. The
actual worker entered the verified kill-on-close Windows Job before ACK, with
a 3 GiB aggregate limit. The final Job query verified zero active processes;
all three PIDs were subsequently absent and the model lock was released.
Resource provider and process owner were both native, with no injected values.
The launched child used the fixed two-thread compute settings.

## Stages and preserved denominators

M planned **1072** arm-specific rows: **1056 available and 16 originally
missing**. Per arm that is 536 planned, 528 available and eight original
missing. Only **40 measurement records were durably exported**: original/T1
views of `cal-L-00..11` and `cal-T-00..07` (20 paired calibration identities).
No history or physical-future measurement record was exported. **1016 available
M rows lack durable measurements.** They are unobserved, not failures.

| Arm / shape | Planned M rows | Available | Original missing | Durable / available | Local object/shape/pusher/coverage valid |
| --- | ---: | ---: | ---: | ---: | --- |
| A / L | 396 | 388 | 8 | 12/388 | 12/12 for each |
| B / L | 396 | 388 | 8 | 12/388 | 12/12 for each |
| A / T | 140 | 140 | 0 | 8/140 | 8/8 for each |
| B / T | 140 | 140 | 0 | 8/140 | 8/8 for each |

The trace shows a measurement returned before the next `sink.record` failed;
canonical order identifies this as `cal-T-08/A`. Its numeric result was not
persisted and was not reconstructed. Therefore **40 is a durable-record count,
not an exact runtime transform/decoder-call count**. The aggregate palette/
residual/range report and full M decision were not exported. Recorded input
hashes match between A/B for the 20 paired identities, and A's input/output
hashes are identical. B changed 8478/184320 pixels, with zero recorded unmatched
pixels, across these 20 persisted real images. This is partial transform evidence.

M requires all 528 available rows per arm to retain local validity, object
position P90 <=2 world pixels, angle P90 <=2 degrees, coverage-error P90 <=.03
per shape, and zero interval-declared false positives. Its journal gate is
**null**, so those requirements were not adjudicated. Pusher error is diagnostic;
local pusher validity remains required. Partial validity cannot pass this gate.

| Forecast scope | Planned | Available | Original missing | Generated | Available not attempted |
| --- | ---: | ---: | ---: | ---: | ---: |
| T AR | 384 | 384 | 0 | 0 | 384 |
| T TF | 192 | 192 | 0 | 0 | 192 |
| L AR | 1152 | 1128 | 24 | 0 | 1128 |
| L TF | 576 | 564 | 12 | 0 | 564 |
| Total | 2304 | 2268 | 36 | 0 | 2268 |

T's B-AR requirements remain >=44/48 joint object/full-shape and >=46/48 pusher
at H1/H2/H4, with no >5pp joint-validity regression in an available contact
stratum. There is no T gate result. L's 144 planned/141 available rows per arm
and primary horizon, with >=127 joint-valid required, likewise remain unknown.
The missing L01/hold branch was not recreated. T/L are not model failures.

## Partial calibration errors and termination

The following values use **only the durable calibration subset above**. Each
cell gives P90 / maximum. Position/pusher use world-coordinate pixels (0..511),
angles use wrapped degrees, and coverage error is an absolute fraction. These
row statistics are not generated-image errors or noise/anchor aggregates.

| Arm / shape / n | Object position | Angle | Pusher position | Coverage absolute error |
| --- | ---: | ---: | ---: | ---: |
| A / L /12 | .841 / .902 | .474 / .818 | 1.054 /1.154 | .017881 /.019882 |
| B / L /12 | .818 / .912 | .363 / .669 | 1.054 /1.154 | .015041 /.019777 |
| A / T /8 | .594 / .633 | .196 / .321 | .242 / .242 | .002273 /.002901 |
| B / T /8 | .577 / .635 | .224 / .344 | .242 / .242 | .002308 /.002862 |

Each arm has **17 non-success, three unknown and zero declared success** interval
decisions. Both true-positive physical calibration states in this subset remain
unknown in both arms. Interval confusion per arm is TP=0, FP=0, TN=17, FN=0,
unknown=3; success precision is **undefined**, because there are no declared
positives. The nominal point diagnostic is TP=1, FP=0, TN=18, FN=1 per arm and
cannot replace the interval rule. There are no B-only interval-positive rows.
The +/- .03 interval remains a development assumption; full pose ambiguity is
unassessed. These partial observations do not establish terminal precision.

No forecast RNG state was recorded and **zero forecast calls/pairs** occurred.
The 20 A/B input-image pairs are not sampler-noise pairing evidence. AR/TF shape,
position, rotation, coverage, appearance/persistence and conditioning effects
are unknown. Model ranking coverage is **0/16 anchors**; no model pair was scored
and ranking accuracy is undefined. No model/hold/uniform comparison was made.
The independent physical source still contains 63 complete branches and 15
complete four-candidate anchors; it was not narrowed to create a model subset.
No physical baseline scores were recomputed for this stopped attempt.

## Resource and accounting evidence

| Quantity | Actual recorded value | Protocol bound |
| --- | ---: | ---: |
| Startup available RAM / commit | 5.001 /14.368 GiB | >=4 /6 GiB |
| Startup free GPU / disk | 6.020 /15.878 GiB | >=4 /2 GiB |
| Sampled minimum RAM / commit | 4.355 /13.784 GiB | >=1.5 /3 GiB runtime |
| Sampled minimum GPU / disk | 6.020 /15.877 GiB | >=1.5 GiB GPU; disk reserve available |
| Worker peak working set / private | 185.078 /260.621 MiB | <=2 /3 GiB |
| CUDA allocated / reserved counters | 0 /0 | <=512 MiB /1 GiB |
| Forecast images / Euler / model loads | 0 /0 /0 | <=2268 /6804 /1 |
| Guardian elapsed | 25.906s | <=1800s |
| Durable result files | 149327 bytes | <=256 MiB |
| Stopped-run telemetry files | 610772 bytes | <=16 MiB |
| Telemetry including all postrun metadata | 1155903 bytes | <=16 MiB |

The guardian recorded 116 resource samples. No resource floor/ceiling triggered.
Actual GPU-resource reads succeeded, but **the real GPC/CUDA model path remains
unexercised**. Zero CUDA counters during M do not validate full-model memory.
Resource floors are sampled, not guarantees between reads. Guardian storage was
610593 bytes before its last log event; the closing byte inventory includes that
event and gives 610772 bytes. Additional postrun audit/partial-statistics metadata
is included in `final-aggregate.json`, separately from the immutable runtime files.
The final aggregate is **63 files /1305230 bytes**, including the aggregate file
itself; all 53 original stopped-attempt files still match their inventory hashes.

Environment creations, resets, restores, physical renders, controls, physics
substeps, policy sampling and optimizer updates all remained **zero**. There
was no training, model warm-up/probe, package installation or asset download.

The ledger has **83 valid hash-chained events**, 41 reserved/complete output
chunks (binding plus 40 real records), and no pending sampler calls. Its last
event hash is `c4f8d66b61b975aca27d9f354f41dd54cdc2e50e485a24fa81d675175eeccdf6`.
The emergency state is stopped in M; model-load status is not attempted and
M/T gates are null. Normal final exports (`M.json`, `summary.json`,
`completion.json`, ranking and forecast diagnostics) are absent because the
worker did not reach that path. Postrun reconciliation files are explicitly
labeled derivations; they do not finish or resume the ledger.

The trace locates the failure at `_files` in `wm_compat_budget.py:118`, reached
through `_telemetry_bytes` and `_check_storage` before the next export reservation.
It does not record the offending path. A transient heartbeat file disappearing
between directory enumeration and `is_file` is a plausible explanation; existing
evidence does not prove it. No reproduction, source patch or retry was attempted.

## Preservation and interpretation

Before/after streaming byte audits confirm **595 old files /186686009 bytes**
unchanged, including **150 B/C files /97364187 bytes**, both checkpoint files
and all **42 v2 manifest outputs**. There are no missing files or new files
inside protected roots. The original registry, ledger, emergency record,
guardian evidence and 40 measurements remain immutable. Neither production claim
is removed or reset after this failure.

This run cannot determine whether T1 improves generated early shape, whether TF
helps, whether action effects are correct or whether weight adaptation is needed.
The partial real-image fit results do not resolve generated-image uncertainty.
An operational stop and one fixed transform provide no basis to require or
validate fine-tuning. The next review concerns the accounting failure; any future
real experiment needs a separately explicit decision and authority, not this
spent attempt.

Local ignored evidence:

- `artifacts/wm-input-compat-eval-v1-20261025/` — immutable runtime output/guardian.
- `artifacts/wm-input-compat-eval-v1-ledger-20261025/` — declaration, journal,
  emergency state; `artifacts/wm-input-compat-eval-v1-registry/` — both claims.
- `artifacts/wm-input-compat-run-audit-20261025/` — before/after byte audits,
  read-only reconciliation, derived full forecast status table, stopped-file
  inventory and final aggregate. No decoder/model is called by reconciliation.
- `artifacts/wm-input-compat-partial-stats-20261025/partial-stats.json` — numeric
  review of the 40 stored rows, with input hashes, exact errors and scope labels.

The [engineering checks](GPC_INPUT_COMPATIBILITY_ENGINEERING.md) and the main
conversation's independent 106-test rerun preceded this launch; they did not
prove this concurrency path. Runtime files were not changed for this report.
Documentation source/link, whitespace and commit-message checks cover this
reporting change; no real runtime test was repeated after the stop.
