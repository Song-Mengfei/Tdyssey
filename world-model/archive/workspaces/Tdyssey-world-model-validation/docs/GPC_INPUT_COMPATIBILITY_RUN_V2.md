# Frozen input comparison v2: stopped on the storage lock deadline

The separately authorized independent attempt on clean source
`706b45bb1ffcbbc0d5234554920b54afad3ef2ac` stopped in M after **22.485 seconds**.
The native guardian could not acquire the shared process storage lock within
its fixed 0.5-second deadline. It recorded `storage_failure`, stopped the owned
process tree and verified an empty Job. **M is incomplete; T and L are not
attempted.** Twenty-four calibration rows are durable. Model loads, forecast
images and Euler steps are zero; neither scientific gate was decided.

Both new permanent claims are spent. There was one actual launch, with no
repair, retry, resume, extra model probe, deadline extension or later experiment.
The [fixed execution identity](GPC_INPUT_COMPATIBILITY_ATTEMPT_V2.md) changes only
the account after the [storage repair](GPC_INPUT_COMPATIBILITY_STORAGE_REPAIR.md).
The [scientific proposal](GPC_INPUT_COMPATIBILITY_PROPOSAL.md), T1, saved inputs,
scenes, noises, sampler, decoder, gates and all budgets remain unchanged.

## Authority, timing and failure evidence

The launch seal was created at **2026-10-25 07:15:50.206 SGT** and the guardian
finished at **07:16:12.706 SGT**. The monotonic 1800-second deadline started before
setup; the worker received only the remaining time. Authorization identity is
`wm-input-compat-eval-v2-20261025-user-authorized`. The new fixed registry is
`artifacts/wm-input-compat-eval-v2-registry/`; output and ledger are
`artifacts/wm-input-compat-eval-v2-20261025/` and
`artifacts/wm-input-compat-eval-v2-ledger-20261025/`. The v1 account was not opened.

Read-only preflight confirmed the exact clean commit, absent v2 paths/claims and
global model lock, matching 35,043,916-byte phase-two checkpoint and sufficient
current native resources. The guardian repeated native startup checks. It used
actual guardian/launcher/worker PIDs **28564 / 9096 / 14928**, nonce
`b1a8a43dfc4549f890b35c629725d0e9`, two-thread child settings and a kill-on-close
3 GiB Job. Both resource provider and process owner were native, not injected.

The first structured failure is:

| Field | Recorded value |
| --- | --- |
| Reason | `storage_failure` |
| Path | `output/guardian/b1a8a43dfc4549f890b35c629725d0e9/storage.lock` |
| Phase | `storage.usage.lock_acquire` |
| Exception | `TimeoutError` |
| Message | `Storage process lock deadline` |
| Deadline | 0.5 seconds, unchanged |

This identifies a bounded process-lock acquisition failure during measurement,
not the old vanished-temporary error or a scientific gate rejection. The exact
lock holder and hold interval were not instrumented; their cause is not inferred
from the final files. No resource floor, memory cap or storage cap was reached.
The repaired implementation's real measurement concurrency did not complete M.
No timeout was relaxed and no performance repair was attempted after stopping.

## Stages and denominators

M has **1072 planned / 1056 available / 16 original missing** arm-specific rows.
Only **24 available rows** were durably exported: `cal-L-00..11`, each in A and
B. **1032 available rows have no durable result**. No T calibration, history or
physical-future measurement is durable. Old v1 measurements are not pooled.

| Arm / shape | Planned M | Available | Original missing | Durable |
| --- | ---: | ---: | ---: | ---: |
| A / L | 396 | 388 | 8 | 12 |
| B / L | 396 | 388 | 8 | 12 |
| A / T | 140 | 140 | 0 | 0 |
| B / T | 140 | 140 | 0 | 0 |

Twenty-four is a durable-file count. Termination may occur while another
transform/decoder call is in progress or before its export; exact runtime
transform/decoder-call counts were not captured. Missing numeric results were
not reconstructed. M requires the complete available set and per-shape error
and interval conditions; its gate is **null**, not passed or failed.

The ledger contains 51 valid hash-chained events: one creation, 25 output
reservations and 25 output completions (binding plus 24 measurements). Every
completed file matches its recorded size, SHA-256, mtime and file identity;
none exceeds its declared maximum. There are no pending output/call reservations
or leftover temporaries. There is no ledger stop/completion event or budget
emergency file because the guardian terminated the worker. The durable guardian
is the effective stop evidence; this does not make the ledger scientifically
complete or permit continuation. The journal tip is
`4521a3c18d42cc50fa420bb6c02e40dd7a9c7a87da8bebaa3b9f9233cc07684b`.

| Forecast scope | Planned | Available | Original missing | Generated | Available unattempted |
| --- | ---: | ---: | ---: | ---: | ---: |
| T AR | 384 | 384 | 0 | 0 | 384 |
| T TF | 192 | 192 | 0 | 0 | 192 |
| L AR | 1152 | 1128 | 24 | 0 | 1128 |
| L TF | 576 | 564 | 12 | 0 | 564 |
| Total | 2304 | 2268 | 36 | 0 | 2268 |

T's gate is also null and L was never entered. There is no A/B AR or TF forecast
comparison, H1/H2/H4 early-shape result, contact/rotation error, generated coverage
or termination result. No branch/anchor has an attempted generated ranking or
selection score. These are unavailable observations, not zero errors or model
failures. The missing L01/hold branch remains missing. TF is not scored as AR.

## Partial L calibration observations

The following are posthoc summaries of only the new durable numeric JSON rows;
no saved image was transformed/decoded during analysis. Each arm has 12 rows.
All local object, full-shape, pusher and coverage validity flags are true. Cells
show **P90 / maximum**, in world pixels, wrapped degrees or absolute fraction.

| Arm | Object position | Object angle | Pusher position | Coverage absolute error |
| --- | ---: | ---: | ---: | ---: |
| A | .841091 / .901629 | .473958 / .817737 | 1.054119 / 1.153920 | .017881 / .019882 |
| B | .818315 / .911643 | .362876 / .669129 | 1.054124 / 1.154078 | .015041 / .019777 |

Each arm has nine interval non-success decisions, three unknowns, and zero
interval positives. Both actual-positive L calibration cases remain unknown.
There are no declared interval false positives, but precision is **undefined**
because there are no positive predictions. This does not validate termination.
No T error statistic is available. The small reused calibration subset cannot
establish the complete M gate, generated-shape benefit or independent accuracy.

A preserves all 12 input/output identities and changes zero pixels. B changes
4392 stored pixels in total (222–464 per frame), with zero stored unmatched
pixels and different output hashes for all 12 frames. RGB range, palette-winner
and residual aggregates were not exported before stopping; they remain unknown.
The original RGB/action/sampler conventions and the +/- .03 development coverage
assumption retain their previous limitations.

## Resources, costs and cleanup

The guardian sampled native resources 52 times. Values below are GiB and are
sampled observations, not guarantees about every transient.

| Quantity | Startup | Minimum sampled | Unchanged startup / runtime floor |
| --- | ---: | ---: | --- |
| Available RAM | 5.998 | 5.417 | 4 / 1.5 |
| Commit headroom | 14.628 | 13.906 | 6 / 3 |
| GPU free | 6.303 | 6.302 | 4 / 1.5 |
| Disk free | 15.635 | 15.629 | 2 / .265625 |

Observed process peaks are **184.980 MiB working set / 261.719 MiB private**;
CUDA allocated/reserved peaks and maximum observed model loads are zero. This
does not measure model residency. Actual costs are 22.485 guardian wall seconds,
24 durable M rows, zero model loads/forecast images/Euler steps and zero new
physics, resets, renders, policy samples or optimizer updates. Worker CPU time
and exact interrupted transform/decoder counts were not recorded.

The last coordinated storage snapshot before final telemetry had 120,164 result
bytes observed and 2,217,316 charged, including the 2 MiB pending allowance.
Telemetry had 563,557 bytes observed and 759,739 charged. Caps remain 256 MiB
results / 16 MiB telemetry; final inventory is recorded separately because the
last monitor event follows this snapshot.
The post-stop physical totals are **120,164 result + 563,686 telemetry = 683,850
bytes**. Including the same reservations, final charges are 2,217,316 result
and 759,868 telemetry bytes. The 129-byte telemetry difference is exactly the
last stopped-monitor event. All totals remain below the original caps.

The guardian's retained handles verified an empty Job before close and model-lock
release. A later native process-list check found all three PIDs absent and the
global lock absent. Cleanup operated only on the owned process tree.

## Verification, integrity and limits

The identity patch passed 87 focused checks (34 budget, 40 guardian, 13 entry/
stage) and source/link/message/whitespace checks. Engineering process tests used
injected resource counters; this actual attempt used native resource reads.
Original sealed-input hashes, exact checkpoint identity and upstream revision
remain required; no changed input was accepted as a new provenance baseline.
All **86 code/config/protocol hashes and 280 asset/provenance/seal hashes**
match the execution binding after the stop. The 66 local source files also
match commit `706b45b` after documented Git line-ending normalization; 45
checkout files have CRLF-only differences from their LF blobs. The GPC revision
remains `a96146e568826ef46134f73c22fdf59a675e7264`, with all 20 upstream source
hashes matching. Checkpoint SHA-256 remains
`4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.

The partial numeric audit SHA-256 is
`a6e1b1d3c972ea85bcaf3135d3dc44e676cb70d1df67e2f4aada0c55bdb7bf12`;
the guardian/ledger reconciliation SHA-256 is
`820c8d03d4594e77fd7a9b092b408accec4bd9005adf0f48ed40a975ddff12b1`.
Its ten identity, accounting, hash-chain, output, cost and cleanup checks pass.

Ignored audit evidence is under
`artifacts/wm-input-compat-eval-v2-audit-20261025/`, including preflight, launcher
console, process cleanup, partial numeric statistics, posthoc forecast statuses
and the stopped-run inventory/reconciliation. The original 703-file/23-root
prelaunch audit matched, including all 53 v1 production files and both claims.
The expanded inventory additionally covered reusable repair-test exports: seven
were regenerated and one native-test record added by the identity tests. Those
exceptions are disclosed in `prelaunch-integrity-review.json`; the expanded
748-file baseline must not be described as wholly unchanged. No original
scientific payload or v1 production evidence was changed.
The closing comparison confirms all original 703 files/23 root memberships and
all 53 v1 production files unchanged, with only those same seven fixture changes
and one known addition in the expanded inventory. It found no further listed
file or directory-membership changes. The newly frozen v2 inventory contains
**37 production files / 683,850 bytes**, SHA-256
`01aeeb7f8e8881701cb34533cd74528b0db4f065452f79259aba99726c80ac61`.
The closing audit and Git EOL reconciliation are `after.json` and
`git-eol-reconciliation.json` in the new audit directory. No new production
file was modified during posthoc analysis.

This is another operationally incomplete comparison. T1's effect on generated
shape, dynamics, action ranking and stopping remains unmeasured. The lock stop
does not establish that fine-tuning is necessary, and partial real-image fits
do not establish a sufficient appearance fix. Both v1 and v2 remain spent.
No follow-up experiment or engineering continuation is automatically authorized.
