# CONTACT-PILOT1 saved-array results

| Population | Cases / failed | Complete / charged | AO duty | Contact+obvious / charged | Empty upper / charged | Late contact+obvious | Recovery median / P90 s | Censored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_old_baseline | 24 / 0 | 1152 / 1152 | 48.13% | 23.96% | 40.71% | 15.28% | 0.6799999999999999 / 2.113 | 6 / 24 (+0not reached) |
| A_new_controller | 24 / 1 | 1109 / 1110 | 57.60% | 32.88% | 22.61% | 28.08% | 0.31 / 0.41000000000000003 | 4 / 23 (+1not reached) |
| B_new_fresh | 24 / 0 | 1152 / 1152 | 56.20% | 38.89% | 28.82% | 36.63% | 0.365 / 0.9249999999999996 | 0 / 24 (+0not reached) |
| all_new_diagnostic_only | 48 / 1 | 2261 / 2262 | 56.89% | 35.94% | 25.77% | 32.45% | 0.36 / 0.42 | 4 / 47 (+1not reached) |

A is difficult matched existing TRAIN scenes; only B contributes new independent families. Do not pool these as48freshfamilies. Six-class CSV has completed and charged denominators, with unclassified charges explicit. Recovery timing uses the frozen task start, observed postsolve flags and interval-start convention; failed/never-reached tasks are not zero-delay recoveries. Contact+motion is co-occurrence, not strict causal force attribution.

| Case | Letter | Role | Old contact+obvious | New contact+obvious | Delta | Old empty | New empty | Old/new wallstall controls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CP1-A-T-01 | T | slow_or_unrecovered_recontact | 12.50% | 47.92% | 35.42% | 83.33% | 22.92% | 0 / 0 |
| CP1-A-T-02 | T | slow_observed_recontact | 29.17% | 64.58% | 35.42% | 64.58% | 25.00% | 0 / 0 |
| CP1-A-T-03 | T | contact_wall_nearstatic | 2.08% | 4.17% | 2.08% | 20.83% | 2.08% | 32 / 41 |
| CP1-A-T-04 | T | weak_exterior_boundary_rotation | 33.33% | 27.08% | -6.25% | 52.08% | 4.17% | 0 / 5 |
| CP1-A-L-01 | L | slow_or_unrecovered_recontact | 12.50% | 10.42% | -2.08% | 66.67% | 41.67% | 5 / 1 |
| CP1-A-L-02 | L | slow_observed_recontact | 29.17% | 68.75% | 39.58% | 64.58% | 25.00% | 0 / 0 |
| CP1-A-L-03 | L | contact_wall_nearstatic | 0.00% | 0.00% | 0.00% | 2.08% | 12.50% | 43 / 37 |
| CP1-A-L-04 | L | weak_exterior_boundary_rotation | 41.67% | 79.17% | 37.50% | 47.92% | 14.58% | 0 / 0 |
| CP1-A-O-01 | O | slow_or_unrecovered_recontact | 8.33% | 14.58% | 6.25% | 6.25% | 43.75% | 0 / 12 |
| CP1-A-O-02 | O | slow_observed_recontact | 29.17% | 47.92% | 18.75% | 64.58% | 27.08% | 0 / 0 |
| CP1-A-O-03 | O | contact_wall_nearstatic | 4.17% | 4.17% | 0.00% | 20.83% | 12.50% | 31 / 0 |
| CP1-A-O-04 | O | weak_exterior_boundary_rotation | 41.67% | 39.58% | -2.08% | 45.83% | 14.58% | 0 / 18 |
| CP1-A-C-01 | C | slow_or_unrecovered_recontact | 18.75% | 45.83% | 27.08% | 2.08% | 29.17% | 0 / 0 |
| CP1-A-C-02 | C | slow_observed_recontact | 37.50% | 68.75% | 31.25% | 56.25% | 25.00% | 0 / 0 |
| CP1-A-C-03 | C | contact_wall_nearstatic | 10.42% | 10.42% | 0.00% | 2.08% | 12.50% | 36 / 0 |
| CP1-A-C-04 | C | weak_exterior_boundary_rotation | 45.83% | 10.42% | -35.42% | 37.50% | 4.17% | 0 / 34 |
| CP1-A-X-01 | X | slow_or_unrecovered_recontact | 29.17% | 64.58% | 35.42% | 31.25% | 25.00% | 0 / 0 |
| CP1-A-X-02 | X | slow_observed_recontact | 12.50% | 27.08% | 14.58% | 64.58% | 31.25% | 0 / 0 |
| CP1-A-X-03 | X | contact_wall_nearstatic | 6.25% | 33.33% | 27.08% | 25.00% | 33.33% | 31 / 0 |
| CP1-A-X-04 | X | weak_exterior_boundary_rotation | 43.75% | 33.33% | -10.42% | 37.50% | 6.25% | 0 / 0 |
| CP1-A-B-01 | B | slow_or_unrecovered_recontact | 31.25% | 22.92% | -8.33% | 66.67% | 52.08% | 0 / 2 |
| CP1-A-B-02 | B | slow_observed_recontact | 16.67% | 25.00% | 8.33% | 72.92% | 45.83% | 0 / 0 |
| CP1-A-B-03 | B | contact_wall_nearstatic | 10.42% | 10.42% | 0.00% | 20.83% | 12.50% | 29 / 34 |
| CP1-A-B-04 | B | weak_exterior_boundary_rotation | 68.75% | 29.17% | -39.58% | 20.83% | 29.17% | 0 / 0 |

Engineering targets are frozen>=40% contact+obvious,<=25% empty upper, observed recovery median<=0.5s/P90<=1s and<=5%unrecovered. Targets are mechanism checks, not population guarantees. Per-letter support and failures must be reviewed alongside them.

| Population | Letter | Cases / failures | Contact+obvious | Empty upper | Late contact+obvious | Wallstall controls |
| --- | --- | --- | --- | --- | --- | --- |
| A_old_baseline | T | 4 / 0 | 19.27% | 55.21% | 13.54% | 32 |
| A_old_baseline | L | 4 / 0 | 20.83% | 45.31% | 16.67% | 48 |
| A_old_baseline | O | 4 / 0 | 20.83% | 34.38% | 10.42% | 31 |
| A_old_baseline | C | 4 / 0 | 28.12% | 24.48% | 19.79% | 36 |
| A_old_baseline | X | 4 / 0 | 22.92% | 39.58% | 16.67% | 31 |
| A_old_baseline | B | 4 / 0 | 31.77% | 45.31% | 14.58% | 29 |
| A_new_controller | T | 4 / 0 | 35.94% | 13.54% | 30.21% | 46 |
| A_new_controller | L | 4 / 0 | 39.58% | 23.44% | 36.46% | 38 |
| A_new_controller | O | 4 / 0 | 26.56% | 24.48% | 20.83% | 30 |
| A_new_controller | C | 4 / 0 | 33.85% | 17.71% | 36.46% | 34 |
| A_new_controller | X | 4 / 1 | 41.33% | 21.33% | 33.33% | 0 |
| A_new_controller | B | 4 / 0 | 21.88% | 34.90% | 12.50% | 36 |
| B_new_fresh | T | 4 / 0 | 37.50% | 25.52% | 29.17% | 20 |
| B_new_fresh | L | 4 / 0 | 59.90% | 22.40% | 57.29% | 0 |
| B_new_fresh | O | 4 / 0 | 44.79% | 31.77% | 42.71% | 0 |
| B_new_fresh | C | 4 / 0 | 45.83% | 24.48% | 43.75% | 1 |
| B_new_fresh | X | 4 / 0 | 26.56% | 35.94% | 32.29% | 1 |
| B_new_fresh | B | 4 / 0 | 18.75% | 32.81% | 14.58% | 11 |

| Population | Letter | Observed / reached | Censored / not reached | Withdrawal released | Observed median / P90 s |
| --- | --- | --- | --- | --- | --- |
| A_old_baseline | T | 3 / 4 | 1 / 0 | 4 | 0.81 / 1.85 |
| A_old_baseline | L | 3 / 4 | 1 / 0 | 3 | 0.63 / 1.814 |
| A_old_baseline | O | 3 / 4 | 1 / 0 | 4 | 1.11 / 1.9180000000000001 |
| A_old_baseline | C | 3 / 4 | 1 / 0 | 3 | 0.62 / 1.492 |
| A_old_baseline | X | 3 / 4 | 1 / 0 | 4 | 0.62 / 1.8040000000000003 |
| A_old_baseline | B | 3 / 4 | 1 / 0 | 4 | 0.73 / 1.842 |
| A_new_controller | T | 3 / 4 | 1 / 0 | 3 | 0.25 / 0.378 |
| A_new_controller | L | 4 / 4 | 0 / 0 | 4 | 0.345 / 0.401 |
| A_new_controller | O | 3 / 4 | 1 / 0 | 4 | 0.37 / 0.402 |
| A_new_controller | C | 3 / 4 | 1 / 0 | 3 | 0.3 / 0.388 |
| A_new_controller | X | 2 / 3 | 1 / 1 | 3 | 0.395 / 0.40700000000000003 |
| A_new_controller | B | 4 / 4 | 0 / 0 | 4 | 0.28 / 0.349 |
| B_new_fresh | T | 4 / 4 | 0 / 0 | 4 | 0.385 / 0.40700000000000003 |
| B_new_fresh | L | 4 / 4 | 0 / 0 | 4 | 0.34500000000000003 / 0.9870000000000002 |
| B_new_fresh | O | 4 / 4 | 0 / 0 | 4 | 0.37 / 0.417 |
| B_new_fresh | C | 4 / 4 | 0 / 0 | 4 | 0.39 / 1.169 |
| B_new_fresh | X | 4 / 4 | 0 / 0 | 4 | 0.365 / 0.9070000000000003 |
| B_new_fresh | B | 4 / 4 | 0 / 0 | 4 | 0.345 / 0.44000000000000006 |

Machine-readable files:summary.json,cases.csv,paired-deltas.csv,paired-letter-deltas.csv,letters.csv,six-classes.csv,recontact.csv,failures.csv. No new physics/model calls were made for reporting. Original R3A data/checkpoints/selected training pool remain unchanged.
