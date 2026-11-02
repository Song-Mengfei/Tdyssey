# Same-attempt persistence recovery

The first guarded segment on implementation `11b7cd9` stopped at 09:34:50 UTC
on 2026-10-24, after a Windows `PermissionError: [WinError 5]` replacing the
independent ledger JSON. Its journal append had succeeded. The authority failed
closed; no GPC model had been loaded or forecast generated. Retain the original
report and monitor verbatim. This is an implementation failure, not model evidence.

The 24 independent calibration renders and four L00 physical branches completed.
L01/hold consumed its branch slot and its reset/settle reservation in journal
event 228, although the reserve did not return and the reset was never called.
Conservatively charge the full reservation. Do not retry this failed branch.

Recovery uses the same singleton authorization, original start time and 1800-second
deadline including downtime. Replay every journal event, preserve old binding
and ledger snapshot, add the changed source binding as an explicit lineage event,
and durably abort the unmatched reservation without refund. Require unchanged
protocol/config/checkpoint/asset roots. No calibration, scene, action, decoder,
threshold, seed or physics setting changes are permitted. Continue only the 59
previously unattempted main slots, then forecast the 63 completed branches.

| Before continuation | Conservatively spent | Remaining ceiling |
| --- | ---: | ---: |
| Main branch slots | 5 | 59 |
| Calibration states | 24 | 0 |
| Resets / restores / settles | 29 / 28 / 29 | 59 / 60 / 59 |
| Controls / control substeps | 28 / 280 | 420 / 4200 |
| GPC images / Euler steps | 0 / 0 | 768 / 2304 |

Add bounded retries for transient Windows atomic-file replacement failures only;
these retries perform no simulator or model operations. Original failed telemetry
is retained; resumed monitoring uses a distinct telemetry subdirectory. A partial
physical design keeps all 64 planned rows and explicitly fails or marks coverage
insufficient where the missing physical branch prevents a declared screen.
