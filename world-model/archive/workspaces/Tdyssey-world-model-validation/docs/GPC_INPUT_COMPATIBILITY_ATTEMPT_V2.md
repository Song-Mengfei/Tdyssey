# Input compatibility attempt v2: execution identity

The user separately authorizes one new comparison after the storage repair on
`fb0f42de60b68f973fa71411720c701f5778c10c`, including the minimum identity patch,
focused checks and execution on its exact clean committed source. No further
execution confirmation is required for this one attempt. The old v1 launch and
worker claims, failed ledger/guardian and 40 durable rows remain spent and intact.

Only execution identity changes. The [original scientific proposal](GPC_INPUT_COMPATIBILITY_PROPOSAL.md),
T1, original saved inputs, scenes/candidates/noises, sampler, decoder/config,
measurement and T/L gates stay unchanged. The proposal SHA-256 remains
`2bfff44111d41f5c54a626b717ec2ea35f4bee5051b9fbeb19926f8fa4dbacf2`.
Format labels ending in `.v1` still describe the same record schemas.

The fixed identity shared by launcher, worker, budget and guardian is:

| Field | Value |
| --- | --- |
| Attempt | `wm-input-compat-eval-v2` |
| Authorization | `wm-input-compat-eval-v2-20261025-user-authorized` |
| Registry | `artifacts/wm-input-compat-eval-v2-registry/` |
| Output | `artifacts/wm-input-compat-eval-v2-20261025/` |
| Ledger | `artifacts/wm-input-compat-eval-v2-ledger-20261025/` |

These are fixed constants, not new CLI retry/identity selectors. The launcher
exclusively seals this registry before startup checks and the worker exclusively
claims the same account. A startup, gate, pairing, IO, lock, sampling, resource or
guardian failure stops it permanently. Another output path, authority string or
source commit cannot reopen it. The global single-model lock remains shared
with the old harness. The exact new code commit is supplied through
`--reviewed-commit` and verified against clean HEAD; original sealed input hashes
and the exact 4084 phase-two checkpoint remain mandatory.

M measures all 528 available real rows for each arm afresh; no old partial row is
reused. Only an M pass allows one frozen-model load. T is at most 576 forecasts;
only its fixed pass allows L's at most 1692. Maximum totals remain 2268 images,
6804 Euler steps and 1800 seconds from guardian setup, 256 MiB results/16 MiB
telemetry, unchanged resource floors, two compute threads and 3 GiB Job limit.
There is no extra warm-up/probe, deadline extension, retry, resume, replacement
scene, installation/download, simulation/reset/render, policy or model training.

Engineering evidence and old-file before/after audits are stored separately in
`artifacts/wm-input-compat-eval-v2-audit-20261025/`. Actual phase results and costs
are recorded in the [v2 report](GPC_INPUT_COMPATIBILITY_RUN_V2.md). The attempt
stopped in M on the unchanged storage-lock deadline; both claims are now spent.

Prelaunch checks passed 87 focused cases: 34 budget, 40 guardian and 13
entry/stage tests, all using fake data or harmless processes. The two unchanged
long fake cap loops were not repeated. Source/link, message and whitespace
checks also passed. The original 703 protected files/23 roots, including all 53
v1 production files and both claims, match their original identities. During
the guardian tests, seven reusable fixture logs in the older repair-evidence
directory were regenerated and one native-test record was added; the expanded
748-file inventory is therefore not wholly unchanged. The original baseline and
comparison are preserved, with these exceptions listed in
`prelaunch-integrity-review.json`. Future guardian test exports use the new v2
audit directory. No scientific payload, old production evidence or claim changed.
