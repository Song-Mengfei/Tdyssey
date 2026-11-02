# C fixed paired development evaluation

This separately authorized evaluation follows [the completed C short run](DPPO_C_RUNTIME.md)
on `6096e56`. It grants no training operations and does not reopen, alter or
replenish training experiment `c-tl-2048-20261024` or milestone B's accounts.

## Fixed question and cases

Compare `base_training` against the final `delta_training` weights using the
same `tdyssey-ddpm-gaussian-v1` sampler, positive std floor .01, 100 denoising
times, official action prefix, fixed crop and normalization. The strict delta
loader changes only the 1,313,282 selected output parameters. Frozen actor and
normalizer fingerprints must match across methods; complete fingerprints must
differ. The original zero-floor sampler is outside this two-group comparison.

Cases are ordered T then L, scenes 100000 through 100004, then policy seeds
5478 and 5479: 20 cases per method, 40 total. The matched T/L initial-state
protocol is `official-t-l-com-v1`. Each case gets fresh T and L environments,
two reset settling steps, matching initial COM state, and a local generator.
No mid-contact restore, scene substitution, tuning or extra seed group is allowed.
Success is measured after physical control with strict coverage > .95; a case
stops at success or 300 controls, including the last short action prefix.

Final delta: `checkpoint-0018-0052-audit.pt`, 16,003,916 bytes, SHA-256
`d3128c7fcfb1a23842b4863d96ae88b7f124e93d751b371b23696eb24d6703dd`.
Its lack of training resumability does not prohibit reading weights for evaluation.
The base remains the explicit local SHA-256
`37d661ce6a487905fa4ff81ab7d65bf1b56b1c06adf480a961f65338fa0e38a0`;
published-EMA tensor identity has not been independently established.
The reviewed training config canonical SHA-256 remains
`0c682afbaf03360dc7af77c3c0917c3cfd120d5e7a8da10532c1d3d5251a8db3`.

## Evaluation-only authority

`policy_eval_budget.py` uses a distinct registry/ledger kind and exact binding
to the original training ID, config, base/delta bytes/hashes and ordered case
digest. Identical bindings cannot register under a fresh ID; output directories
and process restarts cannot refresh spent attempts. Training, optimizer, batch,
or other phase reservations are rejected. Every physical attempt is durably
reserved before execution, and every group slot is consumed before model loading.
Incomplete claims/pending operations block automatic retries after reopening.

| Account limit | Total | Per method |
| --- | ---: | ---: |
| Evaluation groups | 2 | 1 |
| Physical controls | 12,000 | 6,000 |
| Plans | 1,520 | 760 |
| Resets | 80 | 40 |
| Wall time, including downtime | 7,200 s | Shared |
| Registered results/storage | 512 MiB | Shared |

`run_guarded_c_policy.py evaluate` initializes this account immediately before
the actual worker, with the existing Windows PID/nonce/Job Object handshake and
host/GPU guards. Startup floors are 4 GiB available RAM, 6 GiB system commit
headroom, 4 GiB free GPU memory and 2 GiB disk space. Runtime floors are 1/2/1
GiB for RAM/commit/GPU; parent deadline is 7,200 seconds. Host telemetry has a
64 MiB limit; GPU telemetry has a fixed five-second cadence and finite deadline.
Observed minima cannot guarantee instantaneous floors between samples.

Only one complete model may be resident. Each method has a fresh load; its actor
is frozen and runs in eval/no-grad mode. Model fingerprint, global RNG, modes
and scheduler must be unchanged by evaluation. A weak reference must confirm
the first network is gone, followed by allocator cleanup, before the second load.
No optimizer is constructed by the evaluation entry or stepped; the existing
official constructor's temporary upstream shells are unchanged.

`evaluate_c_policy_pair.py` defaults to a stdlib read-only plan. Runtime requires
explicit source/base/delta, separate account/output paths and `--execute`.
Protected training/B trees and registry paths cannot overlap evaluation outputs;
file inventories and hashes are compared at account initialization, worker start,
before/after groups and closeout. Added/deleted original files are also detected.
Case-start and completed/aborted case records are fsynced to `case-events.jsonl`.
An incomplete group keeps full-denominator success fractions undefined and stops
the second group. A killed process is diagnosed from its ledger, case journal
and parent report; it is never retried invisibly.

## Descriptive comparison and limits

`policy_eval_comparison.py` requires complete, identically ordered fixed cases,
the same positive-floor sampler and pinned base, explicit final delta identity,
and preserved weights. It reports T/L success counts with denominator ten,
paired success wins/ties/losses, final/peak coverage and IoU paired mean/median
differences and sign counts, both-success first-success step differences, and
five per-scene blocks with two seeds each. Differences are delta minus base;
metric ties use an explicit 1e-9 tolerance. Unsuccessful first-success times
remain censored, not assigned a synthetic step count.

This is a fixed development set with five scene blocks per shape and correlated
seeds within each block. It does not establish statistical significance,
untouched-test performance, unseen-shape generalization or broad T retention.
Changes must be reported honestly even if weak or mixed. Training episode
outcomes are not substituted for this independent comparison.

## Completed result, 2026-10-24

Implementation commit `1c4c66ab501d930ae02ffc95db5e9b2eaafd1047` ran at
15:29:57.535–16:05:39.317 SGT (2,141.782 seconds, 35 minutes 41.8 seconds).
Actual worker PID 7996 and redirector PID 32552 exited zero. Both fixed groups
completed all 20 cases, with no abort, replacement, retry or extra group.

**T shows a small-set retention loss; L has no successful adaptation evidence.**
T success declined from 6/10 to 5/10. L remained 0/10. L's final coverage mean
increased slightly while its peak coverage mean decreased, so the result is mixed.
These are descriptive development-set observations, not a population conclusion.

| Metric, delta minus base | T | L |
| --- | ---: | ---: |
| Successes, base → delta | 6/10 → 5/10 | 0/10 → 0/10 |
| Success difference | -10 pp | 0 pp |
| Paired success win / tie / loss | 0 / 9 / 1 | 0 / 10 / 0 |
| Final coverage mean, base → delta | 82.8427% → 81.6031% | 26.7562% → 29.0354% |
| Final coverage paired mean / median difference | -1.2396 / -0.6480 pp | +2.2792 / +0.4665 pp |
| Final coverage win / tie / loss | 5 / 0 / 5 | 5 / 1 / 4 |
| Peak coverage mean, base → delta | 92.4733% → 92.8036% | 40.1132% → 39.2954% |
| Peak coverage paired mean / median difference | +0.3303 / +0.1943 pp | -0.8179 / -0.0277 pp |
| Peak coverage win / tie / loss | 7 / 0 / 3 | 4 / 1 / 5 |
| Final IoU paired mean / median difference | -1.9405 / -1.1970 pp | +1.4434 / +0.3471 pp |
| Peak IoU paired mean / median difference | +0.2455 / +0.3630 pp | -0.6615 / -0.0208 pp |

The one lost T success is scene 100001, policy seed 5478. Base succeeded at
control 240, coverage 0.9805651. Delta reached its 300-control cap without
success: final coverage 0.9019827, peak 0.9493688. Rounding that peak to 95%
cannot change the declared strict `> .95` criterion. No new T success offset it.
Among only the five pairs where both T methods succeeded, delta's first success
was on average 41.2 controls earlier (median four earlier; three earlier, one
equal, one later). Other pairs remain censored, so this is not an overall speed
claim and does not remove the lost success.

The five scene blocks below each contain two correlated policy seeds. A block
mean averages those seeds; it does not create two independent scene samples.

| Scene | T successes base → delta (/2) | T final coverage difference | L successes base → delta (/2) | L final coverage difference |
| --- | ---: | ---: | ---: | ---: |
| 100000 | 0 → 0 | +1.1042 pp | 0 → 0 | -8.3234 pp |
| 100001 | 1 → 0 | +9.4263 pp | 0 → 0 | +17.3535 pp |
| 100002 | 1 → 1 | -15.9084 pp | 0 → 0 | +0.5636 pp |
| 100003 | 2 → 2 | -1.6750 pp | 0 → 0 | +3.7830 pp |
| 100004 | 2 → 2 | +0.8550 pp | 0 → 0 | -1.9807 pp |

The positive L final-coverage mean is concentrated in scene 100001; scenes
100000 and 100004 deteriorated. No claim of consistent L improvement, broad T
retention, statistical significance or generalization follows from this set.
The comparison uses the training sampler for both methods, not the original
zero-floor inference sampler. No experiment was extended in response to outcomes.

## Accounting, preservation and resources

Evaluation-only account: `c-tl-paired-dev-20261024`, registry
`artifacts/policy-c-evaluation-registry/`. Reserved equals completed, with no
pending operations. Both consumed group slots are complete. New training and
optimizer steps are zero; original C remains at 2,048 controls and 52/52 steps.

| Usage | Base training sampler | Final delta | Total / limit |
| --- | ---: | ---: | ---: |
| Controls | 5,393 | 5,247 | 10,640 / 12,000 |
| Plans / completed 100-step chains | 682 | 666 | 1,348 / 1,520 |
| Resets / settling substeps | 40 | 40 | 80 / 80 |
| Control physics substeps | 53,930 | 52,470 | 106,400 |
| Denoising steps | 68,200 | 66,600 | 134,800 |

Thus physical cost was 106,400 control substeps plus 80 reset settling substeps.
T/L control totals across methods were 4,640/6,000; plan totals 588/760. Reset
shape totals are 40/40 because each case resets both shapes before restoring
only its fresh target environment. Fewer than 12,000 controls reflects early
success stopping, not omitted cases. Registered artifact storage was
5,465,520/536,870,912 bytes. The original C/B/base inventory checked 17 files and
three trees: no altered bytes, additions or deletions. Its training ledger was
read offline only, without changing its elapsed deadline or operation counters.

All 10,486 host/process and 423 driver GPU samples were valid, on actual worker
7996. Observed minimum available RAM / commit headroom / free GPU memory were
3.5438 / 11.1452 / 5.2529 GiB. OS lifetime peak working set was 3.1581 GiB and
peak process commit/pagefile counter 5.1749 GiB. CUDA allocation peaks were
1.07005/1.07100 GiB for base/delta; reservation peaks 1.16211/1.15820 GiB.
No memory guard, OOM, traceback or runtime failure was recorded. Only existing
Gym/torchvision maintenance/deprecation notices appeared.

Each full network's weak reference was dead before the next load. Freed CUDA
allocations were 1,055,406,080 and 1,056,425,472 bytes, covering all 262,709,056
FP32 parameters (1,050,836,224 bytes). About 8.125 MiB of CUDA workspace
remained after each release. An initial offline audit incorrectly demanded zero
allocator bytes; the corrected audit checks actual network destruction and
full parameter allocation release while permitting workspaces. The initial
failed audit is retained, and `paired-reconciliation-verified.json` passes with
no issues/warnings. This was a read-only audit correction, with no physical
evaluation retry, new account or altered raw result.

Both groups preserved full model fingerprints within evaluation and identical
frozen fingerprints across groups; their complete fingerprints differ. Global
RNGs, modes and mutable scheduler state were restored exactly. The final delta
was loaded by the strict source/config/selection checker. Forty ordered case
starts and forty full completions match the reports and ledger. The comparison
was independently recomputed from raw cases and matched exactly.

Actual existing environment: Windows 11, Python 3.12.3, torch 2.5.1+cu124,
torchvision 0.20.1+cu124, diffusers 0.31.0, numpy 1.26.4, robomimic 0.2.0,
pymunk 6.9.0, pygame 2.6.1, gym 0.26.2, shapely 2.0.6, scikit-image 0.24.0,
opencv-python-headless 4.10.0.84. No installation or download occurred. This is
one machine's bounded run, not validation of other hardware or longer training.

## Review files and recorded command

Runtime files in `artifacts/milestone-c-evaluation-20261024/`:

- `evaluation-pair-report.json`: complete reports, comparison, dependencies and
  limits; SHA-256 `a9e739b0d24bb6ad65e320d4c85ea3945fa8b09397ba7785cfa32db7844cfeac`.
- `base_training.json`, `delta_training.json`: all cases, initial/final/peak
  coverage and IoU, strict success/first-success step, actions, costs and latency.
- `paired-comparison.json`, `case-events.jsonl`, preserved file/tree inventories
  and individual model-release records.

Review files in `artifacts/milestone-c-evaluation-20261024-meta/`:
`paired-reconciliation-verified.json`, `resources-review-final.json`,
`interpretation-review.json`, and `monitor/` raw telemetry/final report.
`tables/shape_summary.csv`, `tables/paired_cases.csv` and
`tables/scene_blocks.csv` contain two, twenty and five rows respectively;
`tables/summary.md` is the compact reader view. Full raw step records remain in
the two method reports. Generated evidence is ignored/local, not Git-published.

Recorded invocation (existing local Python/assets; the consumed account rejects
re-initialization, so this is provenance, not permission to replay or refill):

```powershell
& $python -S -B scripts/run_guarded_c_policy.py evaluate --python $python `
  --source-root $officialSource --base-checkpoint $baseCheckpoint `
  --checkpoint artifacts/milestone-c-20261024/checkpoint-0018-0052-audit.pt `
  --training-ledger artifacts/policy-c-registry/c-tl-2048-20261024/ledger.json `
  --experiment-id c-tl-paired-dev-20261024 `
  --registry artifacts/policy-c-evaluation-registry `
  --run-dir artifacts/milestone-c-evaluation-20261024 `
  --output-dir artifacts/milestone-c-evaluation-20261024-meta/monitor
```

Before runtime, installed discovery passed 290 and skipped 47 (337 tests);
final focused entry/ledger/evaluator/comparison/monitor checks passed 48, including
the added journal/real-account fake integration. Source discovery passed 189 and
skipped 149 (338 tests). An earlier source attempt using sandbox TEMP hit Windows
atomic-replace permission errors; the final run used workspace TEMP and passed.
These tests used fake physics/tiny models; the 40 real cases are separate evidence.
No additional training, evaluation group or process was launched during closeout.
