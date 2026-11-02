# SDK startup observability and one bounded diagnostic

This stage follows the preserved [pilot01 startup stop](GPC_SUPERVISED_PILOT_01.md)
on reviewed `1390dcc`. It fixes missing evidence, not a proven Torch/DLL cause.
It does not reopen the spent training candidate or authorize GPC, policy,
forward/backward, optimization, sampling, physics or another training attempt.
Scientific config `1c5683...be2c` and500list `f7217...004a` remain unchanged.

## Minimal observability changes

The shared SDK initializer fsyncs before/after markers for Torch import,
`set_num_threads(2)`, CUDA availability, explicit CUDA initialization, reset of
allocator peaks and allocator sampling. The first marker precedes a potentially
blocking SDK call. Main-thread stage progress has separate wall/monotonic times
from heartbeat publication. A fresh heartbeat does not imply stage advancement.

Heartbeat refresh/publication first exceptions are retained in a preallocated
64KiB emergency file and bounded stderr, with precise step, type, errno/Winerror,
time and traceback. Failed/partial status temporary files are retained; cleanup
does not mask the original write/publication error. The parent checks that first
error before freshness. Background failures are no longer only in lost RAM.
Unknown allocator values are null, with an unavailable state; fresh samples
have their own timestamps. Initial zeros cannot masquerade as measured peaks.
The original5s heartbeat bound and all process/CUDA/resource caps stay fixed.

The SDK-only entry is
[diagnose_gpc_sdk_startup.py](../scripts/diagnose_gpc_sdk_startup.py). Default
invocation prints a plan. Its fixed one-shot identity is
`20261026-gpc-sdk-startup-diagnostic-01`, with output
`artifacts/gpc-sdk-startup-diagnostic-20261026/` and a separate claim file.
It reuses the existing native Job, PID/nonce/ACK ownership, resource observations,
global model slot and bounded publication/telemetry primitives. It creates no
training candidate or optimizer ledger, and never sets `training_complete`.

## Frozen limits and interpretation

Before the one real launch, freeze source/manifest identities and reservations
in `artifacts/gpc-sdk-startup-review-20261026/diagnostic-plan.json`. Total wall
time is at most60s including import/export/cleanup. Source checks share the
48-second work deadline; the remaining12s are reserved for cleanup, with waits
shrinking against the same original deadline. Output including temporary files is at most8MiB:
4MiB monitor,1MiB stdout and3MiB fixed metadata/result/error/temporary allowance.
There is at most one SDK worker and no concurrent model/physical work.

Allowed SDK calls stop at import, original two CPU threads, CUDA query/init and
allocator readings. Startup RAM/commit/GPU/disk floors remain4/6/4/2GiB;
runtime RAM/commit/GPU floors1.5/3/1.5GiB, process working set/private2/3GiB,
native Job3GiB, CUDA allocated512MiB/reserved1GiB. No real Torch smoke import is
performed before this unique attempt. Any first error, stale heartbeat, resource,
output or deadline failure ends it without retry, replacement output or new seed.

An exception record can identify a failed refresh/publication and its Windows
error. A last durable `before` stage localizes uncompleted startup work; absent
an exception it cannot by itself prove GIL or DLL blocking. A successful warm
startup would demonstrate that one SDK path only, not explain an earlier cold
stop, validate GPC training resources, or authorize another real model attempt.
Any broader startup liveness protocol must return for main review; no5s bound
or process architecture is changed here.

The engineering gate passed43 stdlib scalar/JSON/thread checks in9.547s, with
zero failures/errors/skips and real Torch/GPC/NumPy/native/process/physics paths
denied. This includes the original guard's explicit fresh-CUDA fixtures and
shared-deadline Git timeout checks. Evidence is
`artifacts/gpc-sdk-startup-review-20261026/engineering-final-01.json` and its log.
Source-only checks passed. Prior weights, data, claims and eight held payloads
remain protected; held checks use metadata/stat.

## One real SDK result

The sole attempt ran on clean `404028e1c56c22e226caa50f95356c7cb304afda`,
starting2026-10-26 04:08:01.318 Singapore time. It reached `sdk_ready` without
retry in3.860s guarded time; the execution tool returned in4.428s including its
launch. The frozen plan SHA is
`7aaf68e63bd1a50f5221857d071d8d9151051b3c10e5f8821ea5a4eb23ce5cff`;
the now-spent claim SHA is
`4197307771bf5f721bcb70618230990c2e79ad5dcb4d734d74c715c0b27fdc93`.

| Completed stage | Monotonic seconds |
| --- | ---: |
| Torch import | 2.579 |
| Set two CPU threads | .047 |
| CUDA availability | .078 |
| Explicit CUDA initialization | .031 |
| Reset allocator peaks | .016 |
| Allocator sample | .016 |
| SDK result export | .015 |

Durations bracket durable before/after records and their status publication, so
they are stage intervals, not isolated SDK microbenchmarks. All16 journal
records are present with no unmatched before marker. The parent retained17
resource samples,16 with worker status and five distinct observed heartbeat
publications; their maximum observed gap was1.047s. Fourteen sampled statuses
had unavailable/null allocator readings; two had completed fresh samples, with
all four allocator counters zero. No tensors or model weights were allocated.
Heartbeat first-error slot remains an empty preallocated object; stdout is
empty and no failed temporary file remains. These observations exercise the
successful native path; failure durability is supported by the fake checks.

The output contains11 files/162,361bytes, plus the898-byte separate claim,
for163,259 retained bytes. Including temporary/metadata allowances, the prepaid
bound is3,170,271bytes, below8,388,608. Actual monitor length is24,543bytes;
the stop report's24,406-byte counter precedes its137-byte final monitor event.
Both fit the4MiB monitor allowance; stdout uses zero of1MiB. Fixed metadata and
temporary slots remain within their3MiB allowance. No ledger/checkpoint exists.

Minimum observed free RAM/commit/GPU/disk were7,916,236,800 /17,197,592,576 /
7,353,663,488 /29,199,790,080bytes. Observed worker working-set/private peaks
were413,417,472 /712,798,208bytes; the OS working-set peak was413,499,392bytes.
These remain below original limits; allocator peaks were measured zero, not
initial placeholder zeros. Guard16844 owned worker20644 through launcher23364;
both exited0. The retained3GiB Job had zero active processes after cleanup,
the global slot was released, and cleanup had no errors. `sdk_complete=true`
and `training_complete=false`; all GPC/policy/network construction, forward,
backward, optimizer, sampling, training-account and physical counters remain0.

Raw evidence is only in `artifacts/gpc-sdk-startup-diagnostic-20261026/` and its
separate claim. Offline file/phase/budget verification is
`artifacts/gpc-sdk-startup-review-20261026/diagnostic-offline-audit.json`.
The stage journal SHA is
`cd4c06c8eac23c850fe95d41ce4841e59be23d9360512d501937fd82ede92c0c`;
stop-record SHA is
`578875e585cad910bd542ca3a52543dd0f57ce962e2abbf9b95f25377b9227e2`.

Closing preservation passed for7,804 old files/624,531,020bytes across50 roots,
including all10 old spent claims/11,929bytes. This comprises7,796 fresh content
hashes and eight held payloads checked only through sealed manifest/stat
identities, with zero held content opens/hashes. Config,500list, GPC manifest
and initial checkpoint remain unchanged. Compact `after-summary.json` and
`reconciliation.json` in the review root retain empty changed/missing/added
lists; current diagnostic and current review outputs are excluded.

This attempt did not reproduce the old failure. It provides no causal proof of
Torch/GIL/DLL blocking, a publication error, or a cold/warm-cache explanation
for pilot01. No further code repair is justified by this successful result.
Preserve the five-second rule and return for main review; do not run a second
diagnostic, reopen pilot01, or treat SDK readiness as GPC training feasibility
or successful500-step adaptation. The original scientific coverage gaps remain.
