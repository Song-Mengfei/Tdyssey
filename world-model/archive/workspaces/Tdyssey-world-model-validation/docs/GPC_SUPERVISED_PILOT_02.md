# Supervised pilot 02: CUDA peak stop after one completed update

The sole pilot02 on clean `0adf51bff1e9073eb4e635f2f88f7aa2e696c968` stopped
after one durably completed update and a second reserved iteration. Its CUDA
allocated peak exceeded the original512MiB ceiling. It did not reach500 updates,
checkpoint selection or evaluation, so it supplies no adaptation-effect result.
Main review authorized this separate attempt after the
[SDK-only diagnostic](GPC_SDK_STARTUP_DIAGNOSTIC.md) on `846344f`; old pilot01 and
all earlier evidence remain immutable. No second attempt or limit change follows.

Authorization is `20261026-user-autonomous-gpc-supervised-pilot-02`, candidate
`gpc-tl-supervised-one-step-02`, and fixed output
`artifacts/gpc-supervised-adaptation-pilot02-20261026/`. The new configuration is
[gpc_supervised_adaptation_pilot02.json](../configs/gpc_supervised_adaptation_pilot02.json).
Only its candidate ID and output differ from the preserved original config
SHA `1c5683baa16daee11ca419772532cdda95595275cca794b977261188d99abe2c`.
The new3,618-byte config SHA is
`bd43e43e28e0f54fe8965bdd6c558f21dcbf5d7b20128b7520a1d3594c8ca0e8`.
Execution requires the exact reviewed clean source/config/authority combination;
there is no arbitrary candidate generation, retry or automatic resume.

The original initialization, GPC source, T1, machine data and frozen500list
`f7217a2d09393798529b00e72fcd971e8d7be11cab7a9c769704a9b30ddd004a`
stay fixed. Preserve phase-two one-step loss, AdamW1e-4, batch1, seed5478, clip1,
T/L250 each, at most500 updates and checkpoint500 as the only selected result.
Save every50 and500; fewer than500 is incomplete, never an earlier selection.
The first real forward/backward/optimizer belongs to charged update1.

Loading/training/selection share900s; the full run including evaluation/cleanup
shares3600s with the original dual clocks. Use at most three serial model loads,
one resident model,32 loss batches/examples,3,996 forecasts and11,988 Euler calls.
Keep63 regression and48 independent B2 branches, matched seeds5478/5479/5480,
AR H1..4 and TF H2/H4. No policy load, physical interaction or new collection.
Output remains1GiB plus64MiB telemetry with original metadata/checkpoint atomic
reservations. Native resource floors, process2GiB working-set/3GiB private,
3GiB Job, CUDA allocated512MiB/reserved1GiB and five-second heartbeat stay fixed.
Use the shared SDK stages and first-error evidence; do not add a smoke import.
Any first error or cap stops this one attempt without CPU/model/seed fallback.

Report actual finite-gradient updates, parameter/checkpoint changes, manifest
order/cursor, paired T retention and L errors/ranking. Loss is diagnostic.
B2's independent four contact branches per shape are below eight, and two
rotation branches below four. T near-positive .981947 exceeds .98; H1/H2 have
no .05-separable physical pairs; H4 has only one T/two L informative anchors.
Independent loss has one scene/shape; train near-success and >5deg T CCW targets
are absent. Preserve insufficient/null screens and full denominators; noise
replicates or old regression data cannot enlarge independent physical support.
No result alone authorizes imagined DPPO or establishes policy improvement.

The engineering gate passed49 stdlib/fake checks in7.656s with zero failures,
errors or skips. They cover identity isolation, the parent/worker guard, frozen
schedule consumption and account highwater. Tensor/model/native/process/physics
paths were denied, with zero real attempts; marker operations are synthetic.
Source/message/whitespace checks pass. Evidence is under
`artifacts/gpc-supervised-pilot02-review-20261026/`.

## Actual stop and account

Startup resource preflight and fresh source/weight/data/config/schedule checks
passed. GPC source remains `a96146e568826ef46134f73c22fdf59a675e7264`; the
35,043,916-byte initial checkpoint remains
`4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
The one real GPC load completed. Its296 trainable tensors contain8,729,827
parameters, all under the fixed `inner_model.` selection.

| Result/cost | Actual | Original limit/requirement |
| --- | ---: | ---: |
| Serial model loads reserved/completed | 1/1 | at most3, one resident |
| Training updates reserved/durably completed | 2/1 | 500/500 |
| Spent list cursor | 2 | 500, T/L250 each |
| Selected checkpoint/checkpoint saves | 0/0 | checkpoint500 only; every50/500 |
| Loss evaluation/forecasts/Euler calls | 0/0/0 | 32/3,996/11,988 |
| New physical/policy operations | 0/0 | forbidden |
| Guard elapsed | 27.719s | training900s/total3,600s |
| CUDA allocated peak | 545,908,736B (520.619MiB) | 536,870,912B (512MiB) |
| CUDA reserved peak | 614,465,536B (586MiB) | 1,073,741,824B (1GiB) |

The seven-event hash chain records load completion, reservation/completion of
update1, reservation of update2, then the durable first failure:
`training_failed: MemoryError: Fixed training/evaluation CUDA memory ceiling exceeded`.
Allocated peak exceeded the cap by9,037,824B (8.619MiB); this is the explicit
application ceiling check, not evidence of a device CUDA out-of-memory error.
The precise forward/backward/optimizer allocation producing that peak was not
recorded. No CPU/model/precision/seed fallback or new budget was attempted.

Reserved dataset indices1048(T)/2034(L) exactly match the first two frozen-list
entries, with cursor transitions0->1->2 and unchanged list/order/rule identities.
Only update1 has a completion event and `updates/0001.json`: loss .0027586442884,
gradient norm .1472360492, finite gradient/parameter checks passed. Training noise
generator hashes changed as recorded. These are one-example training values,
not a loss-improvement comparison or independent validation.

The last published status reports one reserved update, lagging the ledger's two:
the second call raises inside `progress -> check_progress -> memory_check`,
before publishing the next counter or `complete_update`. In the frozen source,
this check follows the second `optimizer.step()` and post-step finite-parameter
validation. Thus code control flow supports that those operations returned;
there is no full traceback or second metric record, and update2 is not a durable
completion. Its reservation stays spent. No checkpoint/parameter snapshot exists,
so numerical parameter changes and coherent optimizer/RNG reload are unmeasured.
No early checkpoint is selected and no resume is authorized.

## Resources, evidence and remaining uncertainty

All14 SDK stage records are paired. Torch import took5.484s in this attempt,
while observed heartbeat publication gaps stayed at most1.110s; the5s rule was
not extended. The first heartbeat-error slot remains empty. This confirms live
heartbeat during this longer SDK import, without identifying old pilot01's cause.
The worker output also contains a Hugging Face cache-directory warning; model
load completed afterward, and no cache/environment/download repair was made.

Start was2026-10-26 04:28:11.721 SGT; durable end was04:28:39.424 SGT. Guard21292
owned worker600 through launcher20828; worker/launcher exited1. Job cleanup
proved zero active processes, released the global model slot and reported no
cleanup errors. Minimum free RAM/commit/GPU/disk were6,226,677,760 /
14,777,499,648 /6,706,692,096 /29,172,604,928B. Sampled working-set/private peaks
were1,648,201,728 /3,008,393,216B; OS working-set peak was1,648,488,448B, under
original2/3GiB process and3GiB Job limits.

Use the final fresh status's CUDA peak545,908,736B. The parent's earlier sampled
peak433,224,704B is a lower observation, because it breaks on the newer error
status before aggregating that final allocator sample. This sampling difference
does not erase the explicit over-cap reading.

There are17 retained files/347,280B, including the permanent root/ledger attempt
records, first failure, one update metric and bounded guardian evidence. Monitor
length is170,163B; the stop snapshot's169,949B precedes214B of final events.
Stdout is281B. Original two160MiB checkpoint slots plus16MiB model metadata and
16MiB ledger/1MiB fixed telemetry reservations remain in force; no cap is raised.
Raw evidence is only in `artifacts/gpc-supervised-adaptation-pilot02-20261026/`.
Offline journal/order/file verification is
`artifacts/gpc-supervised-pilot02-review-20261026/pilot02-offline-audit.json`.

Closing preservation passed for7,831 old files/629,382,853B across54 roots plus
the standalone spent SDK claim:7,823 fresh content hashes and eight held payloads
checked only through sealed manifest/stat identities, with zero held content
opens/hashes. All12 old claim documents/13,057B remain unchanged (the SDK's two
documents describe one attempt). The original config,500list, GPC manifest and
initial checkpoint are unchanged. Compact before/supplement/after/reconciliation
files in the review root have no changed/missing/added old members.

Offline source inspection found that local `grads` retains the previous gradient
objects through the next `zero_grad(set_to_none=True)` until reassignment after
the next backward. An ordinary-object weakref demonstration confirms that Python
reference lifetime without Torch. Dense float32 gradients for all selected
parameters would total34,919,308B (33.302MiB); actual nonempty coverage and live
bytes were not measured. This is a plausible avoidable contribution, not proof
of the peak's source or a guarantee of fitting512MiB after release. No production
source change was made after the stop. The corrected offline note is
`gradient-reference-offline-corrected.json`; it supersedes an initial note that
used the wrong JSON count-field names, which is retained for transparency.

Return this cap failure and the narrow reference-lifetime finding to main review
before any repair/new attempt. L prediction, AR/TF differences, H4 action ranking,
T retention and termination remain unmeasured; original scientific coverage
limitations remain. The completed update does not establish task improvement or
authorize imagined DPPO.
