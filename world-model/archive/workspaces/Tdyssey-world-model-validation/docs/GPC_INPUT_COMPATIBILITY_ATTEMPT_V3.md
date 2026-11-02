# Input compatibility attempt v3: execution identity

After the [incremental accounting verification](GPC_INPUT_COMPATIBILITY_INCREMENTAL_STORAGE.md)
on `df229186c386ac78db31e9bcb3da73d06ccb0067`, the user explicitly authorizes one
formal comparison under the original proposal: unchanged budgets and gates,
stop on any error, and no automatic retry. This includes the minimum fixed
identity patch, necessary checks and one launch on the exact clean committed
source; no further confirmation is needed. Both v1/v2 accounts remain spent.

Only the execution identity changes. The [scientific proposal](GPC_INPUT_COMPATIBILITY_PROPOSAL.md)
remains SHA-256 `2bfff44111d41f5c54a626b717ec2ea35f4bee5051b9fbeb19926f8fa4dbacf2`.
T1, saved inputs, missing L01/hold, scenes/candidates/noises, sampler,
decoder/config, measurement and M/T/L gates remain unchanged. Record format
labels ending in `.v1` still refer to the same schemas.

| Field | Fixed value |
| --- | --- |
| Attempt | `wm-input-compat-eval-v3` |
| Authorization | `wm-input-compat-eval-v3-20261025-user-authorized` |
| Registry | `artifacts/wm-input-compat-eval-v3-registry/` |
| Output | `artifacts/wm-input-compat-eval-v3-20261025/` |
| Ledger | `artifacts/wm-input-compat-eval-v3-ledger-20261025/` |

These are fixed constants, with no selectable retry identity or resume mechanism.
The launcher exclusively seals the new registry before resource/setup work;
the worker independently claims the same account. Any startup, gate, pairing,
IO, lock, sampling, resource or guardian stop spends it. Do not repair and
continue, substitute paths, reset a claim, extend time or run another attempt.
Retain every failure and partial output. No warm-up, extra model probe, new
physical interaction, policy/model training, installation or download is allowed.

M remeasures all 528 available real rows for each arm, keeping 1072 planned /
1056 available / 16 missing rows. Only its pass permits one frozen-model load.
T allows at most 576 forecasts; only the unchanged AR shape screen permits L's
1692. Total caps remain 2268 images, 6804 Euler steps and one monotonic 1800-second
guardian deadline including setup, binding, measurement and exports. Result /
telemetry limits remain 256 / 16 MiB; resource floors, two compute threads, CUDA
allocator caps and native 3 GiB Job limit are unchanged. The global single-model
lock remains shared with the old harness.

The new audit directory is
`artifacts/wm-input-compat-eval-v3-audit-20261025/`. Before checks, it seals 895
files / 204,135,829 bytes across 31 roots: all 806 earlier protected files plus
89 incremental-engineering evidence files, including all four spent formal
claims. Tests write new evidence here, never over those sealed logs. Exact
weights, upstream source, original input hashes, native resources, clean reviewed
HEAD and absent v3 paths are checked before the single launch. The guardian and
worker repeat their own required binding/ownership checks within the deadline.

Prelaunch engineering checks passed all 106 budget, guardian and entry/stage
tests in 51.163 runner seconds, with zero failures/errors/skips and no real
model/decoder/physics imports. These use fake data and the existing harmless
native Windows process checks, with new fixture exports only in this audit
directory. The unchanged full fake flow is not repeated for the identity patch.

This identity document records authority and the fixed account, not an executed
result. Actual stages, measurements, costs and cleanup must be reported from
the new outputs after the launch, including a stop or incomplete experiment.

The [actual v3 run](GPC_INPUT_COMPATIBILITY_RUN_V3.md) completed the full plan on
clean `ff1c5b4`, with both new claims now spent. Its results and limitations are
recorded separately; this identity document grants no additional execution.
