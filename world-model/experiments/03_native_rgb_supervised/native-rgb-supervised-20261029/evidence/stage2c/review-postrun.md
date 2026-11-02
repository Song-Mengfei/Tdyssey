# Stage 2C independent final review

The data, index and reference audit passed with no issues. The fixed seen_val batch contains 162 trajectories, 10,368 controls and 103,680 recorded substeps, with zero invalid transitions. Valid H1/H4 candidates are 9,882/9,396. Independent reconstruction reproduced all 162 zero-baseline hit/attempt prefixes and complete candidate rankings.

| Panel | Unique windows | Independent trajectories | H1 moving/static/intermediate |
|---|---:|---:|---|
| natural | 162 | 162 | 100/50/12 |
| extra | 157 | 157 | 90/49/18 |
| combined | 319 | 162 | 190/99/30 |
| development | 36 | 32 | 31/4/1 |

Natural anchors span 3..60, with mean 30.067901. Each selection matches the frozen per-trajectory minimum SHA rank over all legal anchors. Extras match the fixed rare-first, least-trajectory-load and hash rule. Five blocked-proxy slots (C/F/M/S/T) remain empty. The 36 development windows reuse 32 trajectories; temporal overlap is explicitly reported.

All 558 original references, split/seed/full-scene identities, file hashes and four CPU sample timing/action-normalization checks passed. Each split has a CPU batch of images [2,5,3,96,96] and actions [2,4,2], with bounded uint8 cache and no Torch/model/iterator calls. All 37 frozen sources and 1,027 protected TRAIN files remain unchanged.

One offline coverage guard failed because its exact snapshot path omitted source-snapshot/. The original traceback and frozen source remain preserved. An artifact-only derivative changes that literal and repository-root discovery; normalized full-module AST comparison proves no other change and preserves the SHA check. The repaired replay passed without coverage differences or reachability conflicts. No collection failed and no physical control was added.

The original M blocked example explicitly falls back to moving because M lacks proxy support. The separate E left-top example, seen-val-2c-093:31, has the minimum existing selection hash among 14 frozen extra proxy windows and shows saved states 28..35. It adds no evaluation index, changes no original example and does not establish M blocked support.

Raw motion counts are 6,090 moving / 3,570 near-static / 708 intermediate; H1 counts are 6,022 / 3,408 / 452. Every letter meets the diagnostic 40%/45% thresholds. These describe validation distribution and do not establish model prediction quality. Historical optimizer costs remain 1,899 attempts and 5,352.516 seconds; this stage used zero model/optimizer/training-iterator work. Supervised controls total 39,680 and all native controls 74,240. The remaining 320 controls are not authorized.

Final measured old/new/source footprint is 1,707,918,560 bytes, below 3 GiB. The reviewer constructed no environment and executed no physical step, model, optimizer or training iterator. Root must actively report the evidence and stop for parent review; stage 3 remains unapproved.
