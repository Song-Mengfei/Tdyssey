# Checkpoint500 evaluation-only engineering

The evaluation-only entry and append status channel are implemented and tested
without real model/SDK/CUDA/physics work. The sole native file run passed. This
is readiness for main review, not execution authority or paired model results.
Pilot03's1175 cause remains unknown; the new path removes status replacement.

The later [sole real evaluation](GPC_CHECKPOINT500_EVALUATION_RUN.md) completed
the full grid, but checkpoint500 failed the fixed scientific gates. The engineering
claims below remain limited to their original stage.

## Fixed entry and account

`scripts/evaluate_gpc_checkpoint500.py` defaults to a read-only plan. Future
execution requires the exact config, clean reviewed commit, config SHA,
authorization ID, pinned source and initial checkpoint. The worker repeats
source/config/provenance and native ownership checks. No production output,
claim or ledger was created during this engineering stage.

- Config: `configs/gpc_checkpoint500_evaluation.json`, SHA256
  `cd586ed72153ad5d246b4109d18b8ee1faca5dfd7312c721c2c470ca0369e6e2`.
- Future candidate: `gpc-checkpoint500-evaluation-01`; output:
  `artifacts/gpc-checkpoint500-evaluation-20261026`.
- Future authorization: `20261026-user-autonomous-gpc-checkpoint500-evaluation-01`.
- Initial:35043916B, SHA256
  `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
- Selected checkpoint500:105134565B, SHA256
  `0bfd1a944c8fb01555a7ad2d01aab6f526a4805e2bb5239a96703d356c5741d3`.

Preparation rechecks the sealed source claim/config03,500 schedule,1714-event
journal/hash chain,500 complete ordered updates and sole selected boundary at
event1014. The fixed298-key state specification comes from the sealed initial
asset audit. Planning uses no tensor deserializer. The actual read-only plan was
ready with no gaps and `execution_authorized=false`; its assets SHA is
`808d380635f3c5f73d2aa010f38524479468b345d2d8e21e9f7100b55e773c52`.

`EvaluationBudget` reuses the existing append ledger through a fixed new account.
It rejects update, optimizer, checkpoint selection, resume and training-RNG
restoration routes. Source logical step500 is distinct from new updates/examples0
and `training_applicable=false`; it never fabricates500 new updates or calls the
training runner's train_init route.

| Fixed future work | Bound |
| --- | ---: |
| Serial roles, each explicitly released | frozen, adapted |
| Model loads / active models | 2 / 1 |
| New updates / training examples | 0 / 0 |
| Loss batches/examples | 32:16 per role,8 T+8 L each |
| Existing regression / B2 branches | 63 / 48 |
| Noise seeds | 5478,5479,5480 |
| Modes | AR H1-H4; TF H2/H4 |
| Forecasts / Euler calls | 3996 / 11988 |

The engine reuses `load_cases`, `evaluate_loss`, `rollout_case`, `NativeSampler`
and the original reporter. The whole grid and invalid measurement denominators
are required. Matched inference noise follows the unchanged sampler protocol.
No old333 outputs or saved training RNG are spliced in; historical333 forecasts/
999 executed Euler remain spent.

The future loader verifies source identity, keys/shapes/dtypes/finite values and
source provenance, extracts model state and clears container references. It
restores no optimizer/RNG/account state and creates no35MB rollback clone.
`torch.load(weights_only=True,mmap=True)` still parses the105MB container metadata
and creates transient optimizer/RNG tensor wrappers; model storages may retain
the mapping. No real loader was called here. Mmap/hashes do not prove that the
prior50.6MB private-memory margin will suffice.

Keep original3600s wall/monotonic deadlines across startup, loads, evaluation and
cleanup. The900s training-selection deadline is explicitly inapplicable, not an
extra load allowance; account clocks clamp to both original remainders. Retain
1GiB output/16MiB metadata; startup RAM4/commit6/GPU4/disk2GiB; runtime
RAM1.5/commit3/GPU1.5GiB; process working set2/private3/Job3GiB; CUDA
allocated/peak512MiB and reserved/peak1GiB; two CPU threads and five-second
heartbeat/allocator freshness. First error stops without retry/fallback/refund.

## Append status and capacity

Only this GPC mode uses `evaluation-status.journal`. One writer exclusively
creates a fixed ordinary single-link non-reparse file and retains its handle.
Each frame has magic8B, length4B, sequence8B, compact JSON, SHA25632B over the new
header/payload and completion marker4B. Payload<=1536B, frame<=1592B (also below
64KiB), lifetime<=10000 records/16MiB. There is no status rename, replace, delete,
rotation or prior-content scan. Positive short writes reuse `_write_all`; flush
and fsync must succeed before writer success. Failed publications never retry.

The persistent parent reader shares READ|WRITE, not DELETE, reading new bytes in
64KiB chunks. Every frame, including intermediate ones, is checked for sequence,
checksum, completion, PID/nonce, monotonic order and eval-only schema. Observed
truncation, changed path/handle identity, links/types, complete corruption and
capacity violations fail closed. Partial tails retain the original latest status
and timestamps; five seconds still stops the parent. Successful terminal
acceptance additionally requires a clean complete tail.

The marker means visible complete bytes, not proof that concurrent fsync has
returned. Writer durable counts advance only after fsync. The parent checks the
existing64KiB first-error slot before status and terminal acceptance; failed
write/fsync overrides a seemingly fresh frame. Original errno/WinError/traceback
and partial bytes remain evidence. CUDA sample age is independent of heartbeat age.

4035 maximum engine progress calls +3600 one-second heartbeats +128 SDK/lifecycle
allowance =7763 records, below10000. Worst journal10000*1592=15920000B<16MiB.
The evaluation ledger admits rows before write/transition and reserves two64KiB
attempt/failure slots within its16MiB. Normal fixed rows fit1KiB;8064 events plus
those slots fit8MiB.

| Original telemetry allocation | MiB |
| --- | ---: |
| Append status | 16 |
| Evaluation ledger | 16 |
| Fixed identity/startup/error slots | 1 |
| Stdout | 2 |
| Monitor | 29 |
| Total | 64 |

Monitor admission includes the journal reservation; no extra quota is created.
Existing training and old WM status paths retain their prior protocol.

## Verification and boundary

Evidence: `artifacts/gpc-checkpoint-evaluation-engineering-20261026/`.
Two of four allowed synthetic commands ran:47 checks in11.594s with one old
Telemetry-constructor injection compatibility error, then50/50 passed in12.140s
after repair and targeted additions. Both outputs are retained. No forbidden
runtime/native/process/network attempts occurred. Total23.734s<240s; no broad
test suite or extra neural probe ran.

Full markers exercise the actual account and32 loss-loop iterations with ordinary
objects, injected inference rollout and original reporting: two ordered roles,
at most one live marker,3996 forecasts/11988 Euler charges,8064 events, zero
updates and36 invalid measurement rows retained. Fsync is faked; this validates
neither real tensors/RGB export/CUDA nor crash durability. Other checks cover
source/illegal-route rejection, capacity, partial/corrupt/duplicate/out-of-order/
truncated/changed-file journals, independent ages, terminal partial-tail refusal,
first-error priority and Job cleanup. Loader checks use FakeTorch only.

The one frozen Windows file run took18.430s including its external parent
(18.125s before final parent report), within60s. Normal:2048/2048 records,
913-923B payloads,2000771B journal, reader sequence2048/1038 polls/zero partial,
and a long-held ordinary reader with the same whole-file SHA/bytes and initial
prefix. Partial:2 reservations/1 durable,7 failed-tail bytes retained, errno5/EIO
at `status.append.write`, old timestamp unchanged and later update immediately
refused. This was an intentional fault. Replacement/deletion traps stayed empty.
Both cases passed; exit0/0, Job active0, no cleanup errors. No second native run
is allowed by this stage. File-protocol validation does not prove that a future
complete model evaluation will finish or explain the old1175.

`before.json`, `after.json` and `delivery.json` bind preservation and final totals.
All9211 old files/777252748B,16 claim documents, six identities and eight stat-only
held payloads are preserved. New evidence/bounded temporary fixtures stay below
32MiB. Source/message checks are static.

T near-positive support, B2 contact4<8/rotation2<4 per shape, T .981947 outside
the fixed .98 band, no H1/H2 separable pair, only oneT/twoL informative H4 anchors
and one independent loss scene per shape remain insufficient/null. Engineering
readiness establishes no adaptation quality or imagined-DPPO acceptance. Main
review must dispatch real evaluation separately; no pilot04 or retraining follows.
