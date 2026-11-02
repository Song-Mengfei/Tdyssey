# CONTACT-PILOT1 controller changes frozen before physical execution

Implementation: `tdyssey/native_rgb_contact_pilot_controller.py`.
Entry point: `ContactPilotPolicy(env, slot, candidate)`, with the same
`next_action(index) -> action, decision` and `observe(before, action, after, info)`
contract as R3A. `info['substeps']` must contain the ten known actual substeps.
Root alone constructs/steps the unchanged selected-physics environment.

The historical source `native_rgb_r3a_collection.py` is imported but not edited.
Initial target metadata is copied from each frozen case without substitution.
The original numeric `phase` remains the historical planned task schedule, so
no-contact planned-push intervals remain in the same audit denominator. The
new `planned_task_phase` and `controller_state` fields explicitly separate the
task clock from established contact, acquisition, recovery and wall escape.
The fixed withdrawal windows remain18..25 for release_recontact and22..25 for
the other scenarios, followed by planned recontact at26; no early task removal,
reset, replacement trajectory or horizon extension is performed.

## Actual observations and contact following

Only post_solve AO events count. Valid impulse is finite vector presence, and
nonzero means norm>1e-8. A copied event impulse on several point rows never
increases the substep count or becomes a summed force. Controller contact is
established with at least5/10 contacting substeps, terminal AO contact and at
least one positive valid impulse. Two consecutive controls lacking that stable
evidence demote following to acquisition/recovery. These controller hysteresis
rules do not replace the audit's any-postsolve contact definition or recovery
first-contact time.

After a target change, a remaining touch at the old point cannot revert the new
target: establishment must match the selected point within24worldpx and normal
within45degrees. Once established, the most recent positive contact's local
point and outward normal track the current object transform. Point changes are
capped at12worldpx/control; normal changes above45degrees need two consistent
observations. Current contact-point velocity supplies a0.1s feedforward lead,
capped at16worldpx. This is current-state kinematics, without physics/model
lookahead. Current normal and resultant levers are recomputed for telemetry.

## Recovery routes and wall stalls

A direct normal-entry command is used immediately when the existing exterior
prefix/entry geometry check confirms it. The pusher need not stop at a stale
staging waypoint first. Otherwise a current-pose exterior visibility route is
computed; when starting in the contact shell, a monotone-clearance departure
bridge to the current point's exterior staging position is permitted before
the route around the object. The issued bounded command is checked again.
Unconfirmed commands produce recorded holds; bodies are never repositioned.

The selected local contact remains stable for at least4controls. Three controls
with less than0.5worldpx projected progress along the issued route direction,
or two unconfirmed-route controls, request a different feasible exterior point.
Routes are recomputed using current geometry, while target identity is held
until such a measured failure. ETA is ceil(remainingdistance/(100worldpx/s*0.1s))
plus2controls. It is telemetry, not a fixed deadline that discards a productive
long path; continued progress prevents the old four-control target-refresh
oscillation. Failed alternative searches are also debounced for4controls.

Three consecutive AO+OW controls with area-centroid displacement<.1image96px
and absolute rotation<.1degree request wall escape. Candidate drive directions
must not point into the nearest wall (inward dot>=-.05), have at least16worldpx
current geometric drive room and have a confirmed route. Tangential directions
are preferred before route length. Task-specific leverage is retained if a
candidate exists; a generic wall-escape fallback is explicitly recorded if
needed. The finite search uses exterior arcs only and no simulated outcomes.

## Frozen checks and limitations

Six focused CPU tests passed: post/pre-solve and duplicated impulse presence;
verified direct recontact entry; current-body/contact/velocity tracking and
lever telemetry; contact-loss hysteresis despite zero-impulse touch flags;
productive-route stability and no old-contact target rollback; full unchanged
withdrawal window despite stalls. In addition, all48 frozen case first commands
were checked against state-only stand-ins: initial target identity preserved,
finite bounded action, geometry verified and no false established-contact claim.
No environment, renderer, model or physical step was invoked in these checks.

Rules and source bindings are in `controller-rules.json`; test/source identities
and48 initial-command checks are in `controller-tests.json`. No source changes
may be made in response to this pilot's measured outcome. Static checks establish
command/observation contracts, not successful dynamic recovery. A clipped contact
point near a corner can still fail the unchanged geometry guard and must be
reported as recovery/hold rather than repaired with future-state information.
