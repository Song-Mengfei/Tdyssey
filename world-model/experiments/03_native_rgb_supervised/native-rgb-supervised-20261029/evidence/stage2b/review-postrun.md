# Stage 2B independent final review

The integrity and cost audit passed with no issues. All 234 new trajectories contain 14,976 controls and 149,760 recorded 100 Hz substeps, with zero invalid transitions. Independent reconstruction of real contact counts and initial planned attempts reproduced every selected bin and complete candidate ranking. This replay constructed no environment.

| Cohort | Trajectories | Controls | Valid H1 | Valid H4 | All 18 H1 targets |
|---|---:|---:|---:|---:|---|
| 2A | 162 | 10,368 | 9,882 | 9,396 | No |
| 2B | 234 | 14,976 | 14,274 | 13,572 | Yes |
| Combined | 396 | 25,344 | 24,156 | 22,968 | Yes |

The original 2A M failure remains: 267 moving and 253 near-static windows out of 549 (46.08% near-static). New 2B M has 478 moving and 287 near-static out of 793. Combined M has 745 moving and 540 near-static out of 1,342 (55.51% moving, 40.24% near-static). Static windows and difficult trajectories were retained.

All 32 frozen source rows match their snapshots and current files. All 417 protected 2A files, 201 earlier protected files, and old best/last weights and training ledger remain intact. The merged view references all 396 original NPZ/metadata files with both initialization versions and copies no NPZ files.

The 149 final checks match the closed reports, including per-letter H1 classes and gates, H4 denominators, coverage H/R/E/K/U/C sets, 234 attempt prefixes, zero saved-coverage differences and zero reachability conflicts. Eight new examples use consecutive saved states, at most eight frames each, with zero physics or model calls.

Initial boundary coverage remains incomplete: ten letters cover 20 of 24 space/gap combinations and eight cover 19 of 24, leaving 80 explicitly listed missing combinations. Each of the nine spatial classes has 44 combined trajectories; that does not establish complete per-letter gap coverage.

Supervised interaction totals 29,312 controls; all native interaction including history totals 63,872. Historical optimizer and training costs are unchanged. The final footprint measured with closed reports is 1,624,746,253 bytes, below the 3 GiB limit.

The reviewer executed zero environment constructions, physical controls, model calls, optimizer updates, training-index draws or server jobs. These data motion targets do not establish world-model prediction quality. Root must report this batch and stop for parent review; validation and training remain unapproved.
