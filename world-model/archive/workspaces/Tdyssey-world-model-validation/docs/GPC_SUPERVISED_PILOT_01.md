# Supervised pilot 01: stopped during startup, no model update

The single authorized attempt on clean
`4867a66eaac286da61178364d4c84b52b486dffb` stopped with
`worker_status_stale_or_invalid` before a model account was created. It did not
reach GPC loading, the first forward/backward/update, checkpoint selection or
paired evaluation. This is a startup engineering failure; it supplies no
evidence about whether supervised adaptation improves L or preserves T.
No retry, resume, replacement output, source repair or new experiment followed.

## Identity and actual cost

Authorization: `20261026-user-autonomous-gpc-supervised-pilot-01`.
Candidate: `gpc-tl-supervised-one-step-01`. The permanent spent output is
`artifacts/gpc-supervised-adaptation-pilot-20261025/`, with its original
`attempt.json`, guardian handoff, monitor, initial status and stop record.
The directory is not reusable even though no model update was spent.

Before launch the fixed source/model/checkpoint/data/config/schedule identities
passed, the candidate directory and global model lock were absent, and native
resource preflight passed. Config remains
`1c5683baa16daee11ca419772532cdda95595275cca794b977261188d99abe2c`;
the 500-entry list remains
`f7217a2d09393798529b00e72fcd971e8d7be11cab7a9c769704a9b30ddd004a`.
The initial35,043,916-byte checkpoint remains
`4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
GPC source stays at `a96146e568826ef46134f73c22fdf59a675e7264`.
A manifest `2a28136c...b069e` and B2 metadata `dde6b5b8...759ea` were bound.

| Operation/result | Actual | Original cap/requirement |
| --- | ---: | ---: |
| GPC model load reservations/instances | 0 | 3 serial, at most1 resident |
| Finite-gradient/optimizer updates | 0 | exactly500 |
| Actual list entries consumed by training | 0 | 500, T/L250each |
| Selected checkpoint | absent | update500 only |
| Forecasts / Euler calls | 0 / 0 | 3,996 / 11,988 |
| Independent loss batches/examples | 0 / 0 | 32 / 32 |
| New physical/policy operations | 0 | forbidden |
| Durable guardian elapsed | 6.188s | 900s training /3,600s total |

The returned guardian elapsed was6.203s, including slightly later finalization.
Start was2026-10-26 03:04:28.699 SGT; the durable end was03:04:34.892 SGT.
Eight retained files total98,145 bytes; stdout is empty. There is no `ledger/`,
`updates/`, `adapted-latest.pt`, forecast directory or completion record.
Telemetry retained the original16MiB ledger and1MiB fixed reservations; its
reported monitor count was27,701bytes (final file27,864), stdout0, within64MiB.
No output or telemetry cap was enlarged. The unused numerical limits cannot
be recovered through this failed one-shot identity.

## Confirmed cause and uncertainty

Only the initial `setup` heartbeat persisted, at03:04:29.742 SGT. It was about
5.15s old at the durable end, exceeding the unchanged5s freshness limit. The
native guardian stopped the owned worker. Its Job was empty after cleanup,
the single-model lock was released and cleanup reported no errors.

The pre-ledger worker region imports Torch, configures threads and initializes
CUDA before `run_candidate`. Existing logs do not identify which substage was
active when status updates ceased. An import/GIL/DLL initialization stall and
a heartbeat-publication exception retained only in the killed process cannot
be distinguished. Empty stdout supplies no traceback. No reproduction, fresh
Torch import, native probe or liveness-rule change was performed afterward.
There is no evidence establishing a CUDA OOM or a training/gradient failure.

Startup resource floors passed:8,142,934,016 bytes free RAM,
17,626,648,576 commit headroom,7,321,157,632 free GPU and29,283,905,536 free disk
at preflight. The final resource sample is larger than the guardian's earlier
peak accumulator; including it yields398,684,160 bytes observed working set/
OS peak working set and701,698,048 sampled private bytes. These are below the
unchanged2GiB/3GiB process ceilings and3GiB Job cap. The only allocator status
contained initial zeros and became stale, so a fresh CUDA allocator peak is
unavailable. Training memory/throughput remains unvalidated.

## Scientific interpretation and next review

Parameter change is not applicable: no adapted checkpoint exists. No loss,
shape/contact/rotation/coverage error, AR-versus-TF difference, H4 ranking or
paired T retention comparison was measured. Those results are `not_run`, not
passing, failing, improved or zero-error estimates. The list/source contract
passed metadata/engineering review, but actual500-step consumption was not
exercised by this run. Loss decrease cannot be claimed, much less policy gain.

The previous physical support limits remain: independent B2 has four contact
branches per shape (minimum8) and two >5deg rotation branches/twoanchors per
shape (branch minimum4). Their independent dynamics screens must remain
`insufficient/null`; three noise seeds and old63 regression branches cannot
expand the physical denominator. T near-positive coverage .981947 is outside
`(.95,.98]`; H1/H2 have no .05-separable pairs, and H4 supports only one T/two L
anchors. Generated-image termination is uncertified. None of these scientific
gaps caused or explains this pre-model startup stop.

The direct next mainline need is a minimal offline investigation of startup
heartbeat observability, followed by bounded engineering validation if the
main reviewer authorizes it. It should distinguish import/progress and publisher
failure while preserving hard budgets, ownership, resource checks and original
evidence. The main conversation decides any next real attempt. Do not reopen
this candidate, train a policy, run DPPO or collect more data to mask this failure.

## Evidence and preservation

Review evidence lives in `artifacts/gpc-supervised-pilot-review-20261026/`:
`prelaunch-identities.json`, `native-preflight.txt`, `guard-static-review.json`,
`interpretation-contract.json`, `stopped-summary.json`, `stop-account-audit.json`,
startup analysis and preservation `before.json`/`after.json`/`reconciliation.json`.
The new candidate output is retained separately and untouched after stopping.

Preservation passed for all7,784 old files/615,095,018 bytes/48roots:7,776 fresh
content hashes, eight held files checked only through sealed manifest/stat
identities, and all nine old claims/10,602bytes unchanged. No held content was
opened/hashed. Initial checkpoint, GPC manifest, config, list and original
baseline identities match. Only failure documentation was committed afterward;
no model/guard/source repair, installation/download, push or PR merge occurred.
