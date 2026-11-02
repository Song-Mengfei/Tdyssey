# C reset-only scene preflight

On 2026-10-23, continued from engineering commit
`af09a4ad06c4555110988268485d07a6b100dc29` with authorization for real resets and
their internal settling physics only. No policy/checkpoint was loaded; no policy
sample, controller action, training optimizer update or formal evaluation ran.
The original research checkout, Windows support and dependency lock are retained.

The audit uses the exact production
`make_official_shape_env_class(load_official_source(...))`, `legacy=True`,
`seed -> reset -> capture_state -> reset_issues`. The official source commit is
`5ba07ac6661db573af695b419a7947ecb704690f`, with all 18 manifest source hashes
verified. The report records those hashes, seven local code hashes, Python and
dependency versions, individual snapshots/observation summaries and every result.

## Deterministic rule and actual result

The original proposal had 32 training seeds per shape and five evaluation scene
seeds. Check T training in list order, then L training. For each fixed evaluation
seed, check T followed by L. Evaluation also checks the production matched-COM
initial geometry using the actual reset snapshots; this pure calculation performs
no state restore and no additional physics.

Retain every original valid training seed at its original list position. Replace
only rejected positions, T then L, taking the first valid unused candidate from
the **predeclared ascending** sequences T `400032..400063` and L
`500032..500063`. Exclude every original/selected training seed and the complete
evaluation exclusion set, and do not retry rejected candidates. Selection uses
reset validity only, with no policy performance or reward rollout. Evaluation
seeds never change; an invalid evaluation scene would block the audit.

Every rejection below was observed **after the actual reset settle**, rather than
inferred from the earlier static calculation. All have
`pusher_object_overlap_or_insufficient_clearance`: minimum pusher-center distance
to object geometry is below the unchanged 16-pixel threshold.

| Shape | Original seed | Measured distance (px) | Replacement |
| --- | ---: | ---: | ---: |
| T | 400000 | 13.633829 | 400032 |
| T | 400007 | 0 | 400033 |
| T | 400010 | 2.878631 | 400034 |
| T | 400028 | 8.309766 | 400035 |
| L | 500002 | 3.787729 | 500032 |
| L | 500010 | 2.958677 | 500033 |
| L | 500011 | 0 | 500034 |
| L | 500015 | 0 | 500036 |
| L | 500018 | 2.735277 | 500037 |
| L | 500025 | 0 | 500038 |
| L | 500030 | 0 | 500039 |

Candidate L `500035` was also rejected at 12.197279 pixels and remains in the
attempt journal. Twelve candidates were tested to accept eleven replacements.
All original 53 valid training seeds were retained. The complete revised training
list and unchanged evaluation list were then independently reset again.

| Stage | Valid resets | Rejected resets | Total |
| --- | ---: | ---: | ---: |
| Original training | 53 | 11 | 64 |
| Original fixed evaluation geometry | 10 | 0 | 10 |
| Replacement candidates | 11 | 1 | 12 |
| Final training verification | 64 | 0 | 64 |
| Final fixed evaluation geometry | 10 | 0 | 10 |
| Total | 148 | 12 | 160 |

Actual cost: **160 reset attempts, 160 attempted/completed settle substeps,
0 controller steps, 0 policy samples and 0 real policy optimizer updates**.
No runtime errors, blocked control calls, extra physics calls or unresolved
claims occurred. The reset process reported 4.42 seconds; its command completed
in 5.64 seconds. Diagnostic files total approximately 1.71 MiB.

The initial configuration hash was
`68a5817ce539facf651aa95934560b5c85b84bd83e90f784c44a391144135b85`.
The [updated proposal](../configs/policy_c.example.json), saved proposed config and
all 74 final records bind to canonical SHA-256
`0c682afbaf03360dc7af77c3c0917c3cfd120d5e7a8da10532c1d3d5251a8db3`.
Only training seed lists changed. No existing C ledger was migrated or reset;
the formal C experiment has not been initialized.

Final ordered training lists:

```text
T: 400032, 400001, 400002, 400003, 400004, 400005, 400006, 400033,
   400008, 400009, 400034, 400011, 400012, 400013, 400014, 400015,
   400016, 400017, 400018, 400019, 400020, 400021, 400022, 400023,
   400024, 400025, 400026, 400027, 400035, 400029, 400030, 400031
L: 500000, 500001, 500032, 500003, 500004, 500005, 500006, 500007,
   500008, 500009, 500033, 500034, 500012, 500013, 500014, 500036,
   500016, 500017, 500037, 500019, 500020, 500021, 500022, 500023,
   500024, 500038, 500026, 500027, 500028, 500029, 500039, 500031
Evaluation/excluded: 100000, 100001, 100002, 100003, 100004
```

## Implementation, bounds and reproduction

- [policy_preflight.py](../tdyssey/policy_preflight.py) orders probes and proposes
  a config using only stdlib. Unexpected errors or unknown counts block progress.
- [policy_reset_probe.py](../tdyssey/policy_reset_probe.py) creates/closes one fresh
  environment per call. It rejects every `env.step`, including `None`, and permits
  only the pinned reset's one `Space.step(1 / sim_hz)`. Errors block further calls;
  constructor and cleanup failures remain explicit.
- [preflight_c_scenes.py](../scripts/preflight_c_scenes.py) defaults to read-only
  plan mode. Its independent diagnostic bounds are 212 reset attempts/settle
  substeps, 32 candidate probes per shape, 300 seconds and 8 MiB. It fsyncs each
  claim before reset and each result before continuing. An interrupted claim is
  preserved with unknown costs, without a fabricated refund.

The diagnostic never initializes `ExperimentBudget` or its 7,200-second clock.
C's diagnostic account remains zero. These authorized reset costs are separately
recorded, including all rejected originals/candidates. No reset validity rule,
physics setting or evaluation list was weakened to make the check pass.

Actual command, using existing interpreter/source and no installation/download:

```bash
python -B scripts/preflight_c_scenes.py run --execute --source-root SOURCE --output-dir artifacts/milestone-c-preflight-20261023/reset-run --preserve-file artifacts/milestone-b-20261023/budget.json --preserve-file artifacts/milestone-b-20261023/report.json
```

The exact resolved command is in the raw report. It used SDL dummy video/audio
and the existing workspace offline cache. Use `plan` or `--help` without runtime
dependencies. A repeated run requires a new empty diagnostic directory and new
execution scope. The original config is retained alongside the report; supplying
that file reproduces the original rejection/replacement sequence. Using today's
revised proposal instead checks 148 resets when every scene remains valid.

Raw local evidence is under `artifacts/milestone-c-preflight-20261023/reset-run/`:
`report.json`, `attempts.jsonl`, original/proposed configs, declared plan and
resource snapshots. Raw report SHA-256:
`9a227239d4b8e26ceae0b842aff2d4bc568fb873a13f923b4bbf7cc2833aa5e6`.
B's budget/report hashes remain exactly the values recorded in
[milestone C engineering](DPPO_MILESTONE_C.md).

Validation adds 33 fake/stdlib tests for ordering, exclusion/deduplication,
candidate exhaustion, every-attempt accounting, execution guards, cleanup,
failure journaling and config binding. The explicit
`TDYSSEY_RESET_PREFLIGHT_REPORT` acceptance test reads the real report and checks
all snapshots/counters/journal/code/config hashes; it runs no physics and ordinary
discovery skips it. See STATUS (external historical reference: `STATUS.md`) for final full-suite counts.
Final installed discovery passed 222 and skipped 47 (269 total); source-only
passed 131 and skipped 138. The recorded-report acceptance passed separately.
Real simulator/asset flags stayed disabled during full discovery. A stale test
that assumed the original first T seed was changed to test copy isolation against
the actual input value, preserving its original invariant after the scene repair.

## Windows resources and remaining limits

The reset process measured Windows 11 with Python 3.12.3, NumPy 1.26.4, Pymunk
6.9.0, pygame 2.6.1, gym 0.26.2, Shapely 2.0.6 and scikit-image 0.24.0. OpenCV is
the existing `opencv-python-headless` 4.10.0.84 distribution; the raw report's
`opencv-python=null` means that differently named distribution is absent.
No package or source was changed.

| Resource at the reset audit | Observed |
| --- | --- |
| Physical RAM total | 15.85 GiB |
| Available RAM before/after | 2.328 / 2.333 GiB |
| Reset-process peak working set | 93.46 MiB |
| Reset-process peak pagefile/commit usage | 295.34 MiB |
| RTX 3060 Ti total/free | 8.00 / 5.65 GiB |
| C: disk free | 24.89 GiB |

RAM/process peaks use Win32 `GlobalMemoryStatusEx`/`GetProcessMemoryInfo`, disk
uses `shutil.disk_usage`, and GPU figures are driver snapshots from `nvidia-smi`.
CIM was unavailable from the sandbox client; its error and the fallback sources
are preserved. Earlier in this task RAM was 3.81 GiB available and GPU 6.46 GiB
free, so instantaneous headroom varies. No user program was closed.
The post-regression resource review at 22:47 SGT found only **1.21 GiB RAM
available**, 4.93 GiB GPU free and 24.79 GiB disk free; its separate
`resources-review.json` includes the headless OpenCV distribution metadata.

These are **reset-only** process peaks. Torch was not imported and no model was
loaded for a speculative memory check. B's historical CUDA reserved peak was
about 1.17 GiB, but its report contains no host-memory peak. Neither measurement
establishes C's model-loading/PPO peak or Windows long-run throughput. With roughly
2.3 GiB RAM available during reset and only 1.21 GiB at review, host headroom is
limited. Do not start model loading or the 2,048-control job on this evidence.
Recheck resources
before any separately authorized single-instance model-load or training check;
do not infer an OOM margin from the reset result.

Reset validity does not prove goal reachability, policy quality or successful
2,048-control training. The continuous trainer/evaluator still has fake/tiny
validation and B's distinct single-update evidence. Training and formal evaluation
remain stopped pending separate authorization; genuine-boundary-only recovery,
development-scene limitations and local base-identity caveats still apply.
