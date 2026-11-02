# Stage1 actual contact and reachability

Derived only from the completed 54 continuous trajectories; no new physics, rendering or model calls. Historical coverage influenced initial selection priority only and is excluded from every actual stage1 hit/contact count below.

H = distinct actually hit arc bins; R = canonical proven reachable bins; E = union of proven reachable bins across each letter’s nine legal initial poses. K = R ∪ E ∪ H is a demonstrated reachability lower bound, not a complete account of all physically reachable contour. Unconfirmed = all − K − closed holes. Repeated contacts do not enlarge H.

| Letter | H / all | R | E | H / K | Unknown | Closed holes | Contact trajectories | Contact controls | Planned arc hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T | 51/60 (85.0%) | 52 | 52 | 51/55 (92.7%) | 5 | 0 | 9/9 | 451/576 | 9/9 |
| L | 44/53 (83.0%) | 49 | 49 | 44/50 (88.0%) | 3 | 0 | 9/9 | 458/576 | 9/9 |
| O | 33/93 (35.5%) | 40 | 40 | 33/42 (78.6%) | 14 | 37 | 9/9 | 429/576 | 7/9 |
| C | 43/75 (57.3%) | 51 | 51 | 43/54 (79.6%) | 21 | 0 | 9/9 | 450/576 | 9/9 |
| X | 42/84 (50.0%) | 39 | 39 | 42/46 (91.3%) | 38 | 0 | 9/9 | 352/576 | 9/9 |
| B | 41/98 (41.8%) | 45 | 45 | 41/47 (87.2%) | 13 | 38 | 9/9 | 424/576 | 9/9 |

Planned-arc hit measures whether the initial selected contact segment was actually hit during that same trajectory; it does not measure click-target arrival or substitute intended targets for real collision evidence. Full per-trajectory target outcomes and actual distinct hit bins are retained in the JSON.

Closed holes and unconfirmed narrow/blocked normal paths remain distinct. Feature counts retain the original overlapping hole, corner_or_concavity and thin_end_or_stroke labels; JSON includes counts and all known-unhit/unconfirmed bin IDs plus actual initial wall gaps for all nine locations.

**Evidence conflicts:** none: H, R and E are disjoint from proven closed holes and no historical trajectory IDs appear in stage1 mappings.

Real object-contact controls come from recorded contact arrays. Contact points repeat at 100 Hz; per-bin contact-control incidences can count one control more than once, so they are separately labeled and do not inflate independent-trajectory or global contact-control counts.

Actual contact-control cube (external historical reference: `contact-controls.csv`): **3,428 distinct control/bin incidences** from **2,564 object-contact controls**. Rows join the frozen real-transition statistics by trajectory/control and include actual arc, requested approach, area-centroid dx/dy, motion class, continuous movement/rotation, and initial/current four-wall gaps. Saved per-bin contact-control counts agree exactly (0 discrepancies). Multiple touched arcs do not create extra independent controls.
