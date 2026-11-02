# Saved-array contact audit: technical tables and limits

The main findings are in REPORT.md. Definitions were frozen before extraction in
definitions.json. All source trajectory and metadata hashes were checked. No new
physics, checkpoint loading, Torch execution or model calls were needed.

## Denominators and six exclusive outcomes

| 独立统计分母 | 0.1s 区间数 | 区间内曾接触 | 100Hz 接触占比 | 接触且明显运动 | 基本静止 |
| --- | --- | --- | --- | --- | --- |
| 全部成功 TRAIN | 36,864 | 53.82% | 49.54% | 32.51% | 44.31% |
| 合法 TRAIN H1 | 34,560 | 55.15% | 51.40% | 34.67% | 42.83% |
| 选入的 TRAIN 目标 | 23,984 | 65.48% | 61.93% | 49.96% | 31.12% |
| 全部成功 seen-val | 6,912 | 51.62% | 47.39% | 30.05% | 46.74% |
| 合法 seen-val H1 | 6,480 | 52.93% | 49.19% | 32.05% | 45.28% |

| Actual outcome | All TRAIN count / % | Selected TRAIN count / % |
| --- | --- | --- |
| no_contact_nearstatic | 13,120 / 35.59% | 5,941 / 24.77% |
| no_contact_small | 3,894 / 10.56% | 2,329 / 9.71% |
| no_contact_obvious | 9 / 0.02% | 9 / 0.04% |
| contact_nearstatic | 3,214 / 8.72% | 1,523 / 6.35% |
| contact_small | 4,644 / 12.60% | 2,199 / 9.17% |
| contact_obvious | 11,983 / 32.51% | 11,983 / 49.96% |

Each six-class column sums exactly to its denominator and 100%. The 2,304 early
TRAIN controls excluded from H1 are retained in all-acquisition statistics.
Selected targets are a deterministic, biased subset, not an unbiased estimate of
all time. Distinct successful controls total 43,776; no failed prefix enters this
successful-data table. Legal H1 targets total 41,040 across the separate splits.

## Contact duration, endpoint and impulse evidence

| Contacting physics intervals / 10 | All TRAIN count | Selected target count | Seen-val count |
| --- | --- | --- | --- |
| 0 | 17023 | 8279 | 3344 |
| 1-4 | 1553 | 727 | 293 |
| 5-9 | 1706 | 1331 | 333 |
| 10 | 16582 | 13647 | 2942 |

TRAIN terminal contact is 50.41%; selected terminal
contact is 62.10%. Any contact and terminal
contact differ. A one-substep brush contributes 0.01s, not 0.1s of contact.

| Split | Contact runs | Run duration median / P90 / max (s) | Positive-impulse seconds |
| --- | --- | --- | --- |
| train | 2714 | 0.400 / 1.717 / 4.360 | 1785.59 |
| seen_val | 528 | 0.325 / 1.580 / 3.350 | 317.15 |

| Impulse norm epsilon | TRAIN presence substeps / all substeps | TRAIN duty | Selected duty | Seen-val duty |
| --- | --- | --- | --- | --- |
| 0.0 | 178972 / 368640 | 48.55% | 61.16% | 46.43% |
| 1e-08 | 178559 / 368640 | 48.44% | 61.09% | 45.88% |
| 1e-06 | 178329 / 368640 | 48.37% | 61.04% | 45.78% |
| 0.0001 | 177840 / 368640 | 48.24% | 60.92% | 45.61% |

All flagged AO contact substeps have a valid saved impulse row (unknown count 0).
TRAIN has 3,638 known exactly-zero impulse substeps. The recorder repeats event
total_impulse for each point and lacks an event/arbiter ID; neither point impulses
nor arc impulses are summed into force, work or total physical impulse.
core/impulse-provenance.json records source lines and raw examples.

In the 11,983 selected contact+obvious targets: 11,799 have at least one valid
norm>1e-8 substep; 11,789 have >=5/10 contact substeps; 10,962 have 10/10; 11,787
satisfy both nonzero-impulse presence and >=5 contact. Five need not be consecutive.
Observed support is 735 trajectories for any-contact+obvious and 734 for the
impulse-supported form. This does not establish strict pusher causality.

Nearstatic TRAIN controls contain 307.29s of AO contact, 301.11s concurrent with
object-wall flags and 6.18s without a simultaneous object-wall flag. At control
resolution, contact+nearstatic is 3,214 intervals: 3,143 with some object-wall
contact and 71 without. These are observations, not proof that the wall caused
every stalled interval. Agent-wall data remain separate in strata.csv.

## Time, intent and motion

| phase | Controls | Any contact | Contact+obvious | Nearstatic | Planned push / no contact |
| --- | --- | --- | --- | --- | --- |
| approach | 2268 | 34.39% | 0.00% | 66.01% | 0 / 0 |
| push | 27252 | 67.55% | 43.78% | 35.61% | 27252 / 8842 |
| recontact | 3168 | 2.53% | 0.00% | 87.88% | 0 / 0 |
| static | 576 | 0.00% | 0.00% | 100.00% | 0 / 0 |
| withdraw | 3600 | 15.86% | 1.47% | 49.25% | 0 / 0 |

| mode | Controls | Any contact | Contact+obvious | Nearstatic | Planned push / no contact |
| --- | --- | --- | --- | --- | --- |
| contact_push_intent | 20261 | 90.02% | 58.17% | 22.79% | 18380 / 938 |
| free_route | 13507 | 4.10% | 0.38% | 76.24% | 6632 / 6593 |
| hold | 3096 | 33.85% | 4.75% | 45.80% | 2240 / 1311 |

| half | Controls | Any contact | Contact+obvious | Nearstatic | Planned push / no contact |
| --- | --- | --- | --- | --- | --- |
| first24 | 18432 | 64.31% | 43.41% | 29.24% | 13788 / 3256 |
| last24 | 18432 | 43.34% | 21.60% | 59.38% | 13464 / 5586 |

Planned push has 6,632 free_route intervals (6,593 no-contact), 2,240 hold
intervals (1,311 no-contact), and 18,380 contact_push_intent intervals (938
no-contact). No-contact is an actual outcome, not a name for intended withdrawal.
operation-time.csv separates moving pusher with no contact, route/recovery
candidates, withdrawal and static negatives. These operational sets overlap;
their durations must not be added together. Pusher moving means endpoint motion
>=0.1 image96px, not accumulated path length. Each classified interval lasts0.1s;
physical contact seconds instead use the measured100Hz flags.

| Population | Exactly zero translation and rotation | Nearstatic fraction |
| --- | --- | --- |
| 全部成功 TRAIN | 2465 / 6.69% | 44.31% |
| 合法 TRAIN H1 | 943 / 2.73% | 42.83% |
| 选入的 TRAIN 目标 | 337 / 1.41% | 31.12% |
| 全部成功 seen-val | 667 / 9.65% | 46.74% |
| 合法 seen-val H1 | 374 / 5.77% | 45.28% |

TRAIN translation median/P90/max is0.1444/2.2799/3.1003 image96px; signed rotation
min/median/max is -13.4233/0/15.7218degrees; absolute rotation median/P90 is
0.1194/5.8078degrees. Full continuous and intermediate-threshold sensitivity
summaries are in main_summary.json and strata.csv. Exact zero is distinct from
nearstatic. Motion uses area centroid, not body origin or symmetry-folded yaw.

## Recovery and spatial support

| Split | Recontact trajectories | Observed | No recovery / share | Observed median / P90 / max (s) |
| --- | --- | --- | --- | --- |
| train | 756 | 668 | 88 / 11.64% | 0.73 / 1.31 / 2.12 |
| seen_val | 138 | 117 | 21 / 15.22% | 0.71 / 1.37 / 2.11 |

All unrecovered cases have2.2s right-censor windows; they are not zero-delay
recoveries. Static trajectories with no planned recontact are N/A. Per-letter,
scenario, initial region and space delays are in recontact.csv/main_summary.json.

| letter | Controls | Any contact | Contact+obvious | Nearstatic | Planned push / no contact |
| --- | --- | --- | --- | --- | --- |
| B | 6144 | 54.79% | 31.84% | 47.31% | 4542 / 1411 |
| C | 6144 | 53.86% | 29.54% | 44.37% | 4542 / 1490 |
| L | 6144 | 57.52% | 35.60% | 43.00% | 4542 / 1286 |
| O | 6144 | 56.17% | 35.16% | 44.60% | 4542 / 1305 |
| T | 6144 | 49.14% | 32.41% | 44.48% | 4542 / 1757 |
| X | 6144 | 51.46% | 30.50% | 42.09% | 4542 / 1593 |

| regiontype | Controls | Any contact | Contact+obvious | Nearstatic | Planned push / no contact |
| --- | --- | --- | --- | --- | --- |
| center | 4320 | 52.66% | 35.21% | 40.56% | 3230 / 1105 |
| corners | 16128 | 54.84% | 31.29% | 45.74% | 11972 / 3797 |
| edges | 16416 | 53.12% | 32.99% | 43.89% | 12050 / 3940 |

Letter-by-region controls, both rotation signs and independent-trajectory support
are in letter-region-support.csv. X TRAIN corner contact+obvious support is726
controls from54/56 trajectories; positive rotation has261 controls/29 trajectories,
negative79/19. X TRAIN center/edge/corner contact+obvious fractions are34.03%,
33.00%,27.01%. X seen-val corner support is163controls/8trajectories and must
not be added to TRAIN. Initial region describes the reset, not position throughout.

Actual dominant contact arc is assigned once per control by distinct-substep
votes, with lowest-ID ties and an explicit unmapped class. Concentration is in
main_summary.json; multi-arc marginals are never summed as independent time.
Previously measured outer-arc gaps and untested O/B closed inner rings remain;
zero coverage is not proof of physical unreachability.

## Selected-pool exposures and batch distribution

All23,984 targets appear once in the frozen1,499x16 first-epoch plan. Original
window-ID order SHA256 is800655f0dc54234f92c9784dc4b0ff94e44910d488d39f32439ff625a2ee2335.
No short/repeated/dropped target or repeated family within a batch. Actual model
draws/updates remain0. Target contact P10/median/P90 is8/11/13 of16, worst3/16;
contact+obvious is5/8/11 of16, worst1/16. Neither statistic has a zero batch.

Four-action prefixes create95,936 repeated input control exposures over34,836
distinct TRAIN controls;2,028 raw controls have zero prefix exposure. Weights
min/median/max including zeros are0/3/4. Weighted any-contact is63.50%, weighted
contact+obvious46.00%; these are exposures, not additional independent physical
time. input-exposure.npz preserves every weight; batches.csv preserves all batches.
Selected contact+obvious top1/top10 trajectory concentration is0.309%/2.921%.

## Failures, verification and inspection

| Attempt | Charged / complete | Known substeps | Partial substeps | AO contact seconds |
| --- | --- | --- | --- | --- |
| R3A-train-L-000-a0 | 19 / 18 | 180 | 0 | 1.50 |
| R3A-train-T-109-a0 | 34 / 33 | 335 | 5 | 1.70 |
| R3A-train-L-012-a0 | 10 / 9 | 98 | 8 | 0.71 |
| R3A-train-C-012-a0 | 42 / 41 | 418 | 8 | 2.76 |
| R3A-train-X-074-a0 | 1 / 1 | 10 | 0 | 0.00 |
| R3A-train-X-110-a0 | 9 / 8 | 88 | 8 | 0.58 |

The six failures contribute115charged,110complete controls,1,129known substeps,
29partial substeps,zero unknown advances and one unstarted reservation. Combined
successful+failed costs are43,891charged/43,886complete/438,889known substeps.
Failure rows retain actual contact, complete-control motion and partial endpoint
motion; partial durations are not promoted into full controls. Original failure
videos remain in ../r3a/report/failure-videos/.

Six core CPU fixtures and six summary fixtures passed. Independent root checks
passed106/106:8original NPZs with direct rectangle-weighted centers and flag/impulse
checks plus all41,040H1 motions compared to the pre-existing catalogue. The
presentation retains all70original MP4/PNG SHA values, aligns35old examples to
core, and adds13deduplicated choices (six fixed-seed5478 random letter cases plus
measured extremes). Fifty newly rendered saved frames match collection RGB hashes.
Root visually reviewed the cohort and recorded-example overview. No annotations
enter model inputs or ground-truth RGB. B014 shows event391 at interval[3.91,3.92),
delay1.31s from planned2.60s; its saved poststate is3.92s/delay1.32s. Control38 is
contact_push_intent with zero contact;39 has9/10,44 has3/10. Stateframe0 isN/A.

The pre-existing repository publication-evidence check still reports three hash
mismatches (custom-L-summary,pretrained-T-summary,palette-T-summary); those files
were not modified for this audit. No model-quality conclusion follows from this
data audit. Shared source export/server jobs and the eight held-out letters remain
unchanged. final-verification.json and final-storage.json record final bindings.
