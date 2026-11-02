# Fixed T/L collection: A complete, B stopped before reset

The single separately authorized A/B attempt ran on clean
`0aa7c8e3debed957ca1ffc3589a7fd3cdfc8d2a4`. A completed. B failed before its first
reset, immediately stopped, and was not repaired/retried/resumed. No model,
policy, GPU query, forward, prediction, optimizer update, extra episode or
replacement seed was used. Scientific budgets and acceptance gates stayed fixed.
**The next GPC adaptation is not ready.** The independent 48-branch benchmark
is absent, and the actual 500-exposure prefix has weak large-rotation coverage.

## Actual collection and stop

| Account | Episodes/branches | Controls | Simulated resets | Restores | Renders | Physics substeps | Wall seconds | Charged bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A, complete | 32 episodes | 3,200 / 3,200 | 35 / 640 | 32 | 3,267 | 32,000 control + 35 reset | 27.766 / 480 | 19,912,001 / 134,217,728 |
| B, stopped | 0 / 48 branches | 0 / 256 | 0 / 256 | 0 / 64 | 0 / 512 | 0 / 2,816 | 1.390 / 600 | 2,326,370 / 67,108,864 |

All four A runs contain eight 100-control episodes, with no early successes:
T-A/T-B/L-A/L-B used master seeds 410000/410001/410002/410003, respectively.
They used 9/9/9/8 resets and approximately 5.547/5.453/5.375/5.609 collection
seconds. Original source hash-sort splits stayed 6/1/1 in every run.
Charges include the prepaid guardian envelope, durable journal and every
publication generation/temporary overlap; they are not RGB payload size alone.

A has no controls remaining and 605 unused reset slots; B has all physical
slots unused. A/B byte headroom is 114,305,727 / 64,782,494 bytes. The unused
headroom cannot replenish either spent one-shot account. Both owned Jobs were
empty after cleanup, both native owner locks were released, and cleanup reported
no errors. A native integration completed; B's stopped path is not reported as
a successful native collection.

## B failure and test limitation

`gpc_independent_benchmark.py:217` reads `env.success_threshold` immediately
after constructing the environment. The pinned upstream `PushTEnv` initializes
that attribute in `_setup()` (source line 316), called by `reset()`, rather than
in its constructor. The pre-reset check therefore raises `AttributeError`.
This is an implementation error in the initialization check, unrelated to a
lock wait, content scan, hash comparison, resource shortage or scientific gate.

The fake environment supplied the threshold at construction and concealed this
lifecycle difference. The 48 successful synthetic/CPU/fake checks do not prove
the real benchmark path works. The failed implementation/traceback and fixed
plan remain preserved. A subsequently authorized minimal engineering repair
moves the threshold check after the first precharged reset and tests a fake
whose threshold appears only on reset, including genuine post-reset mismatch.
It performs no real reset/control/model operation. The repaired real B path
remains unverified. A new real benchmark attempt still needs separate review
and authorization; zero physical usage does not make the failed account reusable.

## Roles, scenes and held data

The 32 distinct new full initial states produce 24 train, two development,
two independent-validation and four held-model-test episodes. A-run validation
is development; B-run validation is independent validation. There is no source
split reassignment, repeated new scene or exact overlap with the frozen union
of 779 known historical states / 109 excluded seeds. This union includes the
original 316 reviewed states and recognizable old diagnostic/DPPO scenes.
Exclusions are tested before the first control. Source provenance records
official geometry, 100 Hz physics / 10 Hz control, seeds and accepted resets.

Held identities/payload hashes were captured from memory before assignment.
Catalog creation and the actual coverage audit use manifest/stat checks for
held files; held JSON/NPZ opens, content hashes, decoded images, statistics and
selection are zero. The old 63 diagnostic branches remain regression only.
Historical L payload/catalog absence still prevents exhaustive comparison with
every old L state/episode identity; exact hashes also do not exclude nearby or
equivalent scenes represented differently. New collection does not establish
broad independent generalization.

## Exact 500-exposure prefix

The existing sampler is unchanged: alternate T/L, take each shape's first 250
windows, cycle only if short. The 500 recorded indices are an audit of planned
exposures, not 500 executed optimizer updates. All 500 windows are distinct,
with zero repeated-window occurrences. Only three episodes/scenes per shape
enter this prefix, despite 12 train episodes/scenes per shape in the full corpus.

| Statistic | Full T train | Actual T prefix | Full L train | Actual L prefix |
| --- | ---: | ---: | ---: | ---: |
| Unique controls / exposed target windows | 1,200 | 250 | 1,200 | 250 |
| Available four-control windows | 1,164 | 250 | 1,164 | 250 |
| Episodes / scenes | 12 / 12 | 3 / 3 | 12 / 12 | 3 / 3 |
| Contact targets | 577 | 148 | 703 | 109 |
| Absolute rotation > 5 degrees | 9 | 0 | 18 | 1 |
| Mean translation, pixels | 1.418 | 1.570 | 2.164 | 1.807 |
| Mean absolute rotation, degrees | .456 | .217 | .655 | .500 |
| Near-success [.9,.95] / success > .95 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

These contact/motion values describe the target control. Context contains four
aligned controls/observations; counting their overlap as independent exposure
would inflate the sample count. The full training corpus has L rotation that
the original prefix scarcely uses. T has no >5-degree target in its prefix.
No near-success or positive-termination training support was collected.

Each shape has 97 distinct independent loss-validation windows, meeting the
fixed minimum of eight. Each set comes from just one episode/scene, so 97
overlapping windows are not 97 independent experimental scenes. B provides
no physical separability, contact/rotation category or positive/negative
termination support; absent evidence is unknown, not a passing zero-error rate.

Future review can consider a fixed selection/order change to expose more of
the already collected large-rotation windows. Near-success training support
would require separately authorized, disjoint data rather than using an
independent benchmark as training data. Neither change was made here; there
was no resampling, extra collection, threshold relaxation or training.

## Resources, checks and evidence

Read-only startup passed with 8,488,075,264 bytes free RAM,
15,894,212,608 bytes commit headroom and 29,359,063,040 bytes free disk.
The A worker's observed process peak working set was 85,815,296 bytes
(81.84 MiB), with sampled private-memory peak 95,059,968 bytes (90.66 MiB).
B's observed values were 58,855,424 / 74,072,064 bytes (56.13 / 70.64 MiB).
The native Job cap stayed 3 GiB. Minimum machine free RAM was
7,777,976,320 / 8,120,516,608 bytes for A/B. These are worker observations and
machine headroom, not a GPU-memory or future-training feasibility result.

48 focused synthetic/CPU/fake checks and the source/message/whitespace checks
passed before launch. Initial sandbox temporary-directory rename failures and
the subsequent workspace-fixture verification are retained. No real simulator
test suite ran outside A/B. The later lifecycle-only repair passed all 12 fake
benchmark checks, including absent constructor threshold, charged successful
reset, genuine threshold mismatch and reset failure retaining charges. Its
real B path remains untested. The final acceptance/configuration, optimizer,
model, T1, sampler, 500-update checkpoint rule, coverage P90 and T/ranking
retention settings remain unchanged. The future configuration SHA remains
`4a4b9ab34c077dcfb9bebeca1cfbc81f0e58ba3a216acc5fb8954cef262cb376`;
it retains its old preparation paths because B is incomplete.

Closing preservation passed for all 7,505 old files / 574,863,585 bytes across
39 protected roots, including all six old claims. No protected file was added,
removed or changed. The old empty proposals and both checkpoint identities
remain intact. The A audit trapped all eight held payload paths with zero open
attempts. Raw A data, the failed B identity/account and their evidence remain
unchanged during the later synthetic-only repair.

Ignored local evidence:

- `artifacts/gpc-data-stage-a-20261026/`: four raw runs, `catalog.json`,
  `adaptation-data.json`, collection/completion records and native guardian.
- `artifacts/gpc-data-stage-b-20261026/`: fixed plan, failure/journal and native
  stop record/traceback; no complete benchmark is published.
- `artifacts/gpc-data-stage-20261026-registry/`: the two spent claims.
- `artifacts/wm-data-preparation-20261026/`: exclusion/plan identities,
  native preflight, actual budget/stop report, exact 500-index prefix audit,
  asset index and closing preservation evidence.
- `artifacts/gpc-data-stage-engineering-20261026/`: focused checks and retained
  engineering failure notes. No push, PR merge or unrelated checkout edit.
