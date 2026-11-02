# Fixed T/L data collection

This is the separately authorized data-only stage after `bf5283e`. It never
loads GPC/policy weights, samples a policy, predicts a model image, or trains.
The earlier v3 frozen-model account stays spent. A and B have independent fixed
one-shot accounts; neither supports retry, resume, refill or another output.

The entry is [collect_gpc_adaptation_assets.py](../scripts/collect_gpc_adaptation_assets.py).
Default invocation only prints a plan. `--execute --stage A --source-root PATH`
collects T-A/T-B/L-A/L-B with master seeds 410000/410001/410002/410003. Each run
has eight episodes, at most 100 controls and 20 reset candidates per episode,
120 seconds and 32 MiB. The stage totals stay 3,200 controls, 640 simulated
resets, 32,000 control plus 640 reset substeps, 480 seconds and 128 MiB, including
bounded journal/publication/temporary overhead. Original success ends early.
Source hash-sort splits stay 6/1/1. A-run validation is development; B-run
validation is independent validation, subject to provenance/scene checks.

`--execute --stage B --source-root PATH` requires completed A and freezes the
12 deterministic T/L anchors and four hold/toward/side/away candidates in advance.
It shares three real history controls per anchor, then at most four future
controls per candidate. Its caps stay 256 controls, 256 simulated resets,
64 restores, 512 renders, 2,816 total substeps, 600 seconds and 64 MiB.
An ideal full collection uses 228 controls, 12 resets, 60 restores, 300 renders
and 2,292 substeps. Category labels are intentions; measured weak support is a
gap and cannot trigger replacement. Success uses strict coverage > .95 and
absorbs comparison frames without further controls/renders.

Collection tests each full initial Snapshot SHA before the first control,
alongside the original integer-seed and geometry exclusions. Repeated new
scenes fail freeze; no source split or episode is reassigned. Collection-time
episode metadata binds scene, collector, geometry, timing, seeds, reset costs
and payload identities. Once held, JSON/NPZ payloads receive only manifest/stat
checks. Catalog construction performs no episode readback; the official
validator opens only nonheld episodes. The batch-validation CLI rejects held
sampling. The generic historical dataset loader is not invoked by this flow.

The small collection account reuses the existing native owned-process/3 GiB
Job, bootstrap and atomic publication primitives. Reservations precede physical
calls and remain spent on failure. The accounting lock protects only reservation
and publication bookkeeping; physics, rendering and compression run outside it.
Source hashes are checked at startup and new payload hashes come from in-memory
bytes, without repeated inventory scans or payload readback in the control loop.
Native resource checks query RAM/commit/disk/process, never GPU/model state.
The first error/resource/deadline failure terminates the owned process and stops
the plan. No unrelated process is closed.

New assets and evidence use `artifacts/gpc-data-stage-a-20261026/`,
`artifacts/gpc-data-stage-b-20261026/`,
`artifacts/wm-data-preparation-20261026/` and
`artifacts/gpc-data-stage-engineering-20261026/`. Original proposals and all prior
evidence/claims remain immutable. The exact 500-exposure audit follows the
existing alternating T/L first-250-windows rule, cycling only if short. It
reports full corpus and actual prefix separately; audit gaps never change
sampling order, thresholds or data. Real adaptation still needs main review.
