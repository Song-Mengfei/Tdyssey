# Frozen CONTACT-PILOT1 case selection

48 cases were frozen before any physical initialization:24 matched existing
TRAIN families and24 new independent TRAIN families. They are separate groups,
not48new independent scenes. No seen-val outcome or held-out shape selected a
case. Seen-val initial states were read only as duplicate-rejection references.

Matched selection uses TRAIN recorded outcomes:one unrecovered release case,
one slow observed release case,the largest AO+OW+nearstatic count,and a boundary
rotation with the weakest initial-target arc support. Ties use original IDs.
The old48-control trajectory is the baseline without rerunning it. Exact saved
initial full state,initialization and local contact target are preserved.

Fresh selection uses two release tasks,one boundary rotation,and one weakest
geometry-feasible exterior arc at each letter. X focuses on boundaries and
negative rotation. Seed base90481030,regions/yaws/targets and all IDs are fixed
in cases.json. Candidate geometry uses no physics rollout. All24fresh scenes
have unique seeds and no exact/declared near match to the912existing TRAIN/val
initial states or another fresh pilot state. Existing R3A near thresholds stay
8worldpx area-center,5symmetry-degrees and16worldpx local-pusher. No static
candidate was rejected for this frozen list. Geometry feasibility is not proof
of useful future motion or physical reachability.

Each case retains48controls maximum,withdraw start18forrelease or22otherwise,
withdraw end/recontact start26. No retries/replacements are permitted after
physics;allpartial/failure charges count against2304. The goals are frozen in
cases.json but remain small-sample engineering targets,not population claims.
The fixed paired media cases are CP1-A-T-01(oldT040unrecovered release) and
CP1-A-X-04(oldX005weak boundary negative rotation),selected before new results.

The source controller snapshot and selected physics snapshot are preserved.
The fresh controller is a new version owned by another agent;this planning
script never creates an environment,renders,steps physics,loads models or
imports Torch. Source and fullcase hashes bind preflight.json. The direct cold
verification independently checks all48initial placements and24source metadata,
fullstate and target/initialization identities.
