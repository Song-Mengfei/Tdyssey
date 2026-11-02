# Frozen-data contact coverage

This is an offline report. It performs no physics controls, model inference or GPU work, and changes no input data or denominator.

Physical audit passed: **True**. Data: **33,144 controls, 518 independent trajectories**.

!Coverage overview (external historical reference: `coverage-overview.png`)

!Observed reachability lower bound (external historical reference: `observed-reachability-lower-bound.png`)

!Contact and boundary diagnostics (external historical reference: `contact-boundary-diagnostics.png`)

| Split | Letter | All contour hit | Verified reachable hit | Closed-hole bins | Unconfirmed bins | Contact trajectories |
|---|---|---:|---:|---:|---:|---:|
| train | A | 57/92 (62.0%) | 52/52 (100.0%) | 24 (26.1%) | 16 (17.4%) | 23 |
| train | B | 49/98 (50.0%) | 45/45 (100.0%) | 38 (38.8%) | 15 (15.3%) | 23 |
| train | C | 57/75 (76.0%) | 51/51 (100.0%) | 0 (0.0%) | 24 (32.0%) | 23 |
| train | E | 79/90 (87.8%) | 74/74 (100.0%) | 0 (0.0%) | 16 (17.8%) | 23 |
| train | F | 62/70 (88.6%) | 58/58 (100.0%) | 0 (0.0%) | 12 (17.1%) | 23 |
| train | H | 70/81 (86.4%) | 65/65 (100.0%) | 0 (0.0%) | 16 (19.8%) | 23 |
| train | I | 65/75 (86.7%) | 59/59 (100.0%) | 0 (0.0%) | 16 (21.3%) | 23 |
| train | J | 55/74 (74.3%) | 47/47 (100.0%) | 0 (0.0%) | 27 (36.5%) | 23 |
| train | L | 49/53 (92.5%) | 48/49 (98.0%) | 0 (0.0%) | 4 (7.5%) | 23 |
| train | M | 64/85 (75.3%) | 60/60 (100.0%) | 0 (0.0%) | 25 (29.4%) | 23 |
| train | O | 44/93 (47.3%) | 39/40 (97.5%) | 37 (39.8%) | 16 (17.2%) | 23 |
| train | P | 46/75 (61.3%) | 43/44 (97.7%) | 19 (25.3%) | 12 (16.0%) | 23 |
| train | S | 62/85 (72.9%) | 53/53 (100.0%) | 0 (0.0%) | 32 (37.6%) | 23 |
| train | T | 55/60 (91.7%) | 52/52 (100.0%) | 0 (0.0%) | 8 (13.3%) | 23 |
| train | U | 63/81 (77.8%) | 58/58 (100.0%) | 0 (0.0%) | 23 (28.4%) | 22 |
| train | V | 50/77 (64.9%) | 43/43 (100.0%) | 0 (0.0%) | 34 (44.2%) | 22 |
| train | X | 48/84 (57.1%) | 39/39 (100.0%) | 0 (0.0%) | 45 (53.6%) | 22 |
| train | Y | 40/60 (66.7%) | 36/36 (100.0%) | 0 (0.0%) | 24 (40.0%) | 22 |
| seen_val | A | 30/92 (32.6%) | 26/52 (50.0%) | 24 (26.1%) | 16 (17.4%) | 6 |
| seen_val | B | 32/98 (32.7%) | 30/45 (66.7%) | 38 (38.8%) | 15 (15.3%) | 6 |
| seen_val | C | 24/75 (32.0%) | 22/51 (43.1%) | 0 (0.0%) | 24 (32.0%) | 6 |
| seen_val | E | 30/90 (33.3%) | 28/74 (37.8%) | 0 (0.0%) | 16 (17.8%) | 6 |
| seen_val | F | 36/70 (51.4%) | 33/58 (56.9%) | 0 (0.0%) | 12 (17.1%) | 6 |
| seen_val | H | 16/81 (19.8%) | 16/65 (24.6%) | 0 (0.0%) | 16 (19.8%) | 6 |
| seen_val | I | 38/75 (50.7%) | 36/59 (61.0%) | 0 (0.0%) | 16 (21.3%) | 6 |
| seen_val | J | 26/74 (35.1%) | 23/47 (48.9%) | 0 (0.0%) | 27 (36.5%) | 6 |
| seen_val | L | 19/53 (35.8%) | 19/49 (38.8%) | 0 (0.0%) | 4 (7.5%) | 6 |
| seen_val | M | 44/85 (51.8%) | 40/60 (66.7%) | 0 (0.0%) | 25 (29.4%) | 6 |
| seen_val | O | 16/93 (17.2%) | 16/40 (40.0%) | 37 (39.8%) | 16 (17.2%) | 6 |
| seen_val | P | 31/75 (41.3%) | 29/44 (65.9%) | 19 (25.3%) | 12 (16.0%) | 6 |
| seen_val | S | 25/85 (29.4%) | 23/53 (43.4%) | 0 (0.0%) | 32 (37.6%) | 6 |
| seen_val | T | 20/60 (33.3%) | 20/52 (38.5%) | 0 (0.0%) | 8 (13.3%) | 6 |
| seen_val | U | 33/81 (40.7%) | 28/58 (48.3%) | 0 (0.0%) | 23 (28.4%) | 6 |
| seen_val | V | 26/77 (33.8%) | 23/43 (53.5%) | 0 (0.0%) | 34 (44.2%) | 6 |
| seen_val | X | 27/84 (32.1%) | 18/39 (46.2%) | 0 (0.0%) | 45 (53.6%) | 6 |
| seen_val | Y | 17/60 (28.3%) | 14/36 (38.9%) | 0 (0.0%) | 24 (40.0%) | 6 |

## Reachability evidence from recorded contacts

Let **R** be the canonical normal-path reachable-bin set, **H** the distinct actual collision-hit-bin set, and **C** the proven closed-hole-bin set. The physically audited trajectories start in the exterior; each actual hit is evidence that its bin was reachable in at least that observed pose. Repeated touches of one bin add no new member to H.

The global observed-poses reachability lower bound is **R ∪ H**, computed independently within training and validation. Its hit coverage is **|H| / |R ∪ H|**; unresolved bins are **all bins − (R ∪ H) − C**. This is a lower bound of demonstrated reachability, not a complete classification of all physically reachable bins. The original canonical denominator remains unchanged.

| Split | Letter | R ∪ H bins | Hit / observed reachable lower bound | Unresolved bins | H ∩ closed-hole conflict |
|---|---|---:|---:|---:|---|
| train | A | 57 | 57/57 (100.0%) | 11 (12.0%) | none |
| train | B | 49 | 49/49 (100.0%) | 11 (11.2%) | none |
| train | C | 57 | 57/57 (100.0%) | 18 (24.0%) | none |
| train | E | 79 | 79/79 (100.0%) | 11 (12.2%) | none |
| train | F | 62 | 62/62 (100.0%) | 8 (11.4%) | none |
| train | H | 70 | 70/70 (100.0%) | 11 (13.6%) | none |
| train | I | 65 | 65/65 (100.0%) | 10 (13.3%) | none |
| train | J | 55 | 55/55 (100.0%) | 19 (25.7%) | none |
| train | L | 50 | 49/50 (98.0%) | 3 (5.7%) | none |
| train | M | 64 | 64/64 (100.0%) | 21 (24.7%) | none |
| train | O | 45 | 44/45 (97.8%) | 11 (11.8%) | none |
| train | P | 47 | 46/47 (97.9%) | 9 (12.0%) | none |
| train | S | 62 | 62/62 (100.0%) | 23 (27.1%) | none |
| train | T | 55 | 55/55 (100.0%) | 5 (8.3%) | none |
| train | U | 63 | 63/63 (100.0%) | 18 (22.2%) | none |
| train | V | 50 | 50/50 (100.0%) | 27 (35.1%) | none |
| train | X | 48 | 48/48 (100.0%) | 36 (42.9%) | none |
| train | Y | 40 | 40/40 (100.0%) | 20 (33.3%) | none |
| seen_val | A | 56 | 30/56 (53.6%) | 12 (13.0%) | none |
| seen_val | B | 47 | 32/47 (68.1%) | 13 (13.3%) | none |
| seen_val | C | 53 | 24/53 (45.3%) | 22 (29.3%) | none |
| seen_val | E | 76 | 30/76 (39.5%) | 14 (15.6%) | none |
| seen_val | F | 61 | 36/61 (59.0%) | 9 (12.9%) | none |
| seen_val | H | 65 | 16/65 (24.6%) | 16 (19.8%) | none |
| seen_val | I | 61 | 38/61 (62.3%) | 14 (18.7%) | none |
| seen_val | J | 50 | 26/50 (52.0%) | 24 (32.4%) | none |
| seen_val | L | 49 | 19/49 (38.8%) | 4 (7.5%) | none |
| seen_val | M | 64 | 44/64 (68.8%) | 21 (24.7%) | none |
| seen_val | O | 40 | 16/40 (40.0%) | 16 (17.2%) | none |
| seen_val | P | 46 | 31/46 (67.4%) | 10 (13.3%) | none |
| seen_val | S | 55 | 25/55 (45.5%) | 30 (35.3%) | none |
| seen_val | T | 52 | 20/52 (38.5%) | 8 (13.3%) | none |
| seen_val | U | 63 | 33/63 (52.4%) | 18 (22.2%) | none |
| seen_val | V | 46 | 26/46 (56.5%) | 31 (40.3%) | none |
| seen_val | X | 48 | 27/48 (56.2%) | 36 (42.9%) | none |
| seen_val | Y | 39 | 17/39 (43.6%) | 21 (35.0%) | none |

**Evidence conflict risk:** No actual hit bin intersects a proven closed-hole bin in either split. This consistency check does not prove all remaining bins reachable or unreachable.

## Planned contact arc versus actual collision bins

The planned arc is the collector's preselected **object contour segment** (`staging_segment`). This metric is not the pusher click-target arrival rate. Actual hit bins come from real collision-point mappings for that trajectory; repeated contact with one bin contributes one distinct hit-bin member.

| Split | Independent trajectories | Any real contact | Planned arc actually hit | Planned closed-hole arcs | Closed-hole arcs actually hit |
|---|---:|---:|---:|---:|---:|
| train | 410 | 410 | 396 (96.6%) | 0 | 0 |
| seen_val | 108 | 108 | 108 (100.0%) | 0 | 0 |

The original sampler checked the staging center's exterior component without the later normal-contact-path and closed-hole-priority checks. Therefore a planned arc can be in a closed hole while its staging center is outside; contacts elsewhere on the object do not count as hitting that planned arc. These misses remain in the report and were not replaced or recollected.

Per-trajectory planned/actual comparison (external historical reference: `target-contact-comparison.csv`) lists every trajectory, planned arc, any-real-contact flag, planned-arc-hit flag, number and identities of distinct actual hit bins, and the recorded versus reviewed classification. [Target-contact summary](target-contact-summary.json) retains all planned closed-hole cases.

The all-contour denominator includes outer and hole boundaries. Verified reachable coverage uses only canonical bins with conservative exterior-path evidence. Closed holes are inaccessible from the exterior. Unconfirmed paths are not claimed impossible; some have recorded contacts under other approaches/poses.

Training retains three verified-reachable misses: **L bin 2, O bin 55 and P bin 55**. Each has zero mapped contact at that bin despite contacts on the rest of the letter. Their precise failure causes are unconfirmed. The finite collection budget is exhausted; no replacement or extra collection was performed.

Validation uses six independent trajectories per letter and has much lower coverage. This report does not treat the small validation set as complete contour coverage or as an unseen-letter test.

Contact-angle histograms use mapped real collision points and pre-solve relative velocities, not commanded directions. Repeated 100 Hz contact points are correlated; independent contact-trajectory counts are reported separately.

Boundary statistics are independently derived from recorded states and the fixed existing geometry. They measure 16px bounding-box proximity to the four inner walls and simultaneous proximity at each corner, including initial frames. Wall-arbiter contact points were not retained, so actual wall-collision counts are unavailable and are not claimed. Planned wall/corner locations are never substituted for actual collisions.

[Full compact CSV](summary.csv) contains raw counts, percentages, all eight angle bins, all four walls/corners, and remaining reachable-bin IDs. Original per-letter SVGs remain under `../data/coverage/train/` and `../data/coverage/seen_val/`.

Input manifest SHA256: `8fea889c626843deec45bbcebce17f288ab33e33320a31e3433576dde6925dfe`.
