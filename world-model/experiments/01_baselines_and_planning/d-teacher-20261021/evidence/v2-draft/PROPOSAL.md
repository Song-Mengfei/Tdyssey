# Generic teacher v2 isolated draft — focused checks passed

This directory now contains an isolated runnable implementation plus 18 passing
source tests, a tiny real-D protocol check (repeated once after the final
deadline guard), and a complete real-D composite continuation fixture. The active
v1 module, tests, evaluator and matrix settings are
unchanged. No long v2 optimizer or student training was started here. Broad
teacher performance is unvalidated. See CHECKS.json and retained logs for commands,
work counts, source hash and limits. These are generic geometric priors, not
parameters fitted to a particular scene or known successful path.

## Current implementation status

Load simulator_search_v2.py with an isolated module name, registering it in
sys.modules before exec_module (dataclass requirement). Entry class
SimulatorActionSearchV2(env_factory,initial_state,config=None,v2_config=None)
accepts the original SearchConfig unchanged; alias SimulatorActionSearch exists
for a process-local launcher. V2Config exposes switch_margin=0.02 and bounded
short-contact travel/speed/staging ranges. No live package imports this draft.

The implementation uses the pure-Python exterior extractor described below.
Both correction modes allow 2–80 px inward travel; deterministic translation
seeds start at min(60,max(5,goal_distance)). A short rotational variant can compose
one deterministic goal-translation recovery after its actual simulated rotation
state, inside the same forecast budget. Per-control approach flags preserve both
approach phases. Requested and actually used composites are reported separately.

The first tiny real-D fixture completed a simple rotation proposal and exact
selected continuation, but its requested composites did not fit the short stage
budget. A separate H80 staged-contact fixture then forced one generic exterior
rotation plus recovery solely to check protocol: independent initial+history
replay and all 80 forecast controls matched states/rewards/native-done/contact
counts/coverage. It ran in 2.547 seconds, using 2 branch restores, 8 historical
replay controls and 160 candidate controls, plus 4 independent-history and 80
independent-continuation controls. This is protocol evidence, not task-performance
evidence; the fixture and forced selection are declared in its report.

## Motivation and fixed comparison

The live v1 scene-100000/search-42 trace exposed route chatter: after committing
only a prefix of a long approach, a new slightly higher terminal score can reset
the approach. The saved reports must remain intact, including unsuccessful and
interrupted runs. Do not reinterpret a predicted terminal score as actual contact
or success. Finish the v1 matrix before enabling v2.

Keep the same initial scenes, search seeds, actual-control/wall-time budgets,
success threshold and full-history replay protocol for a v2 comparison. Report
both the total v1/v2 difference and the declared changes together. A later
ablation can distinguish hysteresis from the new primitive families; no component
benefit can be claimed from the combined comparison alone.

## 1. Preserve and consistently score incumbent progress

Add `_approach_remaining` alongside `_active_remaining` and `_tail`.

- After selecting a fresh primitive, save its approach-control count.
- Each matching `record_action` decrements both remaining phase counters, floored
  at zero. A divergent actual action clears the tail and phase metadata.
- Reforecast an incumbent with its remaining approach count, clipped to its
  available active tail. Do not reset this count to zero for scoring.
- New `hold` candidates have zero approach controls. Phase data in reports refer
  only to the forecast still ahead, never already-executed contacts.

Introduce a fixed `switch_margin=0.02` in absolute score units. Evaluate all
candidates as before, then retain the completed incumbent unless the best fresh
candidate improves its consistently recomputed score by at least this margin.
Do not apply the margin if no incumbent completed, the fresh candidate reaches
native success while the incumbent does not, or an unsafe incumbent has a safe
replacement. A live/predicted state mismatch remains a hard protocol failure,
not an exception that permits silently dropping the incumbent.

This is a preference for route continuity, not an irrevocable commitment. Large
objective improvements still replace the route. The unchanged incumbent forecast
also reflects any genuine deterioration of its remaining plan. Record raw best
score, incumbent score, score gap, selection rule, margin, bypass reason and
remaining approach controls so the effect is auditable.

## 2. Represent actual exterior contacts

Use the solid union of the existing local rectangle components. Do not use D's
center-to-boundary radial normal: its mass center can lie in the hole. Do not
treat component seams or hole-facing edges as exterior pushing surfaces.

Practical first implementation: lazily build `unary_union(Polygon(rect.vertices))`
with the already-installed Shapely package. Use only polygon exterior rings;
exclude interior rings. Orient each exterior consistently, and derive outward
normals from its directed edges. Alternatively, a pure-Python extractor must
split partially shared rectangle edges before cancelling covered subsegments,
and additionally distinguish outside-connected free space from holes.

For directed edge a→b with solid to its left, tangent t=(b-a)/|b-a| and outward
normal n=(t_y,-t_x). Transform point q and normal through the current object pose.
Reject any proposed staged pusher center q+(15+clearance)*n that is too close to
another solid component or outside the existing safe pusher-center workspace.
Check every rectangle, including neighbors around concave corners. Pusher-radius
inflation is applied once.

For D, COM is the snapshot object position (the adapter fixes local COG=(0,0)).
Original T is a separate case: its COM is pose.transform((0,45)). Keep this
distinction explicit if the planner is later exposed for shapes beyond D.

## 3. Add two small primitive families

Retain the v1 long-route/push family as a broad exploration option. Add families
whose control lengths depend on path distance and speed rather than a mandatory
35–80% approach fraction.

### Short translation/contact correction

Parameters: exterior edge ID; contact fraction u∈[0.15,0.85] on that edge;
staging clearance 0.5–8 px; inward travel 2–60 px; heading bias ±0.6 rad;
target speed 30–140 px/s; fallback route direction ±1. Use a small deterministic
set of edge midpoints/quarter points for seeding, then local refinement.

Prefer accessible contacts whose inward normal has a positive component toward
the goal displacement, while retaining a minority of other-side proposals for
exploration. The simulator, not this preference, decides whether the push works.
Travel is an independent parameter: do not multiply it solely by goal-center
distance, which vanishes during orientation correction.

Approach selection:

1. If the pusher is already staged at the proposed contact, start the push without
   a route. Keep the first target displacement bounded.
2. Otherwise use the direct current→staging segment if its swept pusher disk is
   clear of the stationary object and walls.
3. If starting in contact, first try an explicit short outward retreat along the
   associated exposed-edge normal, then test direct visibility. Count retreat
   controls as approach, and let the physical forecast validate their effect.
4. Use the existing two AABB directions only when a direct approach is blocked.

The approach-clearance filter does not apply to the intentional contact/push
segment. The snapshot geometry is a proposal approximation; actual contact and
wall diagnostics still come from stepping the simulator.

### Signed rotation/contact correction

Use the same contact representation and short path machinery. Set desired turn
delta=wrap(m*(goal_angle-current_angle))/m for symmetry order m. For D, m=1.
For contact q, COM c and nominal inward force f=-n, compute the torque proxy
tau=cross(q-c,f). Prefer delta*tau>0 with nontrivial lever arm; retain a small
opposite-sign exploration allocation rather than forbidding alternatives.

Do not reverse this sign for screen coordinates: Pose.transform and Pymunk use
the same mathematical rotation sign; positive rotation merely appears clockwise
in the rendered image. Use actual edge normals, not the direction toward COM.

Parameters: edge ID, u, travel 2–80 px, heading bias ±0.6 rad, staging clearance
and speed as above. Seed edge points with high desired signed lever arm, and
refine contact fraction and short travel independently. Do not promise a target
rotation from the torque proxy: moving contact, negligible shape friction and PD
tracking are resolved only by the physical forecast. A later arc family would
be a distinct addition, not hidden inside this version.

## 4. Candidate allocation, refinement and bounds

For population 12, first generation allocates 4 long pushes, 4 short translations
and 4 signed-rotation proposals. Within each family include distinct accessible
edges/routes before duplicating one. Rejected proposals still consume the hard
evaluation budget. Hold and incumbent evaluation also count, as in v1.

Keep family and edge ID categorical. Refine continuous parameters within a
family using its own elites; never average an edge ID, route sign or incompatible
parameter vector. If fewer than two completed family examples exist, use the
declared generic seeds/exploration distribution instead of fitting a singular
Gaussian. Preserve local RNG ownership and broad exploration.

Determine approach/push control counts from segment length and sampled speed at
the existing control period; reserve the configured physical settling controls.
Reject overlong routes that cannot fit rather than silently dropping necessary
waypoints. Every candidate still forecasts the same budget
min(horizon+settle, remaining_actual_controls,80), with attained-position hold
after its active sequence. Native success ends the forecast immediately.

Validate all target coordinates, per-control target jumps and physical path
diagnostics. For the first command, explicitly check the previous controller
target when present as well as physical pusher distance, avoiding an ambiguous
definition during PD lag. Existing targets/physics are never clipped or modified
to manufacture feasibility. Do not relax bounds to copy a known successful path.

## 5. Integration boundaries and focused checks

Once enabled, modify only the search module and its focused source tests first.
Leave SimulatorReranker replay, forecast, state tolerance, native termination,
hard deadline and work accounting unchanged. Keep the evaluator API and required
selected_actions/selected_predictions keys; add primitive family, contact edge,
torque proxy, route mode, phase counts and selection-rule diagnostics.

Before runtime: source tests for approach metadata after partial execution;
small-gap incumbent preference and all bypasses; positive/negative torque and
rotated poses; D-hole/seam exclusion; circle clearance near concave corners;
direct versus blocked approach; zero-goal-distance rotation; finite remaining
budget; unchanged exact selected-state/history protocol. Geometry tests can use
small literal rectangle fixtures; they need no actual simulator run. If Shapely
is chosen, import it lazily and gate its geometry tests separately so source-only
test discovery remains dependency-free.

Then repeat the fixed matrix once with v2, preserving every report including
timeouts and failures. Report success per scene/seed, actual first contact,
approach restarts, score-gap switches, actual controls, wall time, candidate
attempts/rejections, full-history replay and forecast controls. No student data
export or training starts merely because this proposal exists.
