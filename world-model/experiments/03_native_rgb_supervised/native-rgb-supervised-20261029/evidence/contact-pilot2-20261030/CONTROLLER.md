# CONTACT-PILOT2 controller mechanisms and prephysics evidence

This is a new controller module. CONTACT-PILOT1 source, rules and data are not
overwritten. It reads current geometry/state and completed past contact traces;
it creates no environment, renders no RGB, loads no model and runs no dynamics.
All physical parameters, the original 48-control task schedule and existing
penetration guard remain unchanged. The controller is a privileged geometric
collector, not an image-only learned policy or a world-model motion interface.

## Contact target and route are a transaction

Candidate selection returns both a target and a verified route packet. The packet
contains the departure source point/normal/arc, waypoints, destination target
identity, target revision and route identity. Committing a target commits this
packet atomically. The first exit segment therefore does not lose the old contact
bridge when the new destination becomes `self.contact`.

Each action recalculates the destination staging point from the current object
pose, revalidates the route using current full-union geometry and retains the
committed departure source. Invalid routes are explicitly invalidated/replanned;
no old world-coordinate path is followed blindly. A normal-entry handoff can occur
only when current geometry confirms it. Its telemetry says that it is a current
geometric handoff, not evidence that unexecuted waypoints were travelled.

The recorded CP1 T03 state10 regression uses its actual previous cap-side target
and action. Production `next_action(10)` selects a replacement with its old-source
departure bridge, emits a non-hold departure command, and retains the same bridge
after the target has changed. At the identical saved state another production
replan still emits a non-hold command. No physics is replayed by this test.

## Raw observations are distinct from executable targets

CompactAttempt still records all raw physical contact points, normals and arc
membership. CP2 never clips or overwrites those raw observations. Its telemetry
retains the representative raw observation separately from the executable target
and the validation/rejection reason.

An executable point/normal/arc must pass a joint validation against the complete
local union exterior: finite values and unit normal, exterior distance <=0.75
worldpx, matching exterior arc projection (0.75px tolerance), normal/arc dot>=0.85,
outward/inside probes, and a radius15.5 normal staging segment accepted by the
current full-union free-space test. An observed target is accepted as a whole
pair; CP2 does not clip the point and then attach an unrelated normal. A legal
pair jump above24px is rejected as a whole. Invalid physical/component observations
leave a still-valid previous target in place or cause an explicit reacquisition
state if the previous target itself is no longer executable.

The CP1 B04 states8,15,31,39 regressions are the matched original B018 case,
not the fresh weak B case. Each recorded CP1 executable pair is rejected for a
normal/arc mismatch. Production observation tracking consumes the preceding real
raw trace and either retains the legal previous target or commits one complete
legal raw pair. The resulting target is jointly valid in all four states. This
addresses a demonstrated integration mismatch; it does not establish that normal
tracking alone caused every B regression.

## Requested arc remains an immutable task objective

`initial_requested_arc` is an immutable integer; `initial_requested_contact`
retains the original point and normal. Mutable contact-following state is separate.
Actual requested-arc contact requires that the post_solve point's recorded arc
membership contains this exact ID. Positive evidence additionally requires a
finite valid impulse vector norm>1e-8. Per-substep sets prevent duplicated event
impulse copies from increasing counts. A stable requested acquisition requires
positive requested membership in at least5/10 substeps including the terminal
substep. The previous loose24px/45-degree association and minimum arc ID are not
used as evidence of requested-arc hit.

Before stable acquisition, entry is normal-only: zero tangential term and zero
velocity feedforward. Lateral displacement from the requested normal line must
be<=2worldpx for direct entry; otherwise current geometry routing is used. Raw
incidental contact can provide a legal departure source without replacing the
requested objective or establishing requested success. Repeated unconfirmed or
incidental acquisition can trigger an explicitly recorded fallback; the original
requested ID and its unhit/not-stable reason remain. Unknown reachability is
never relabelled physical impossibility. Closed O/B inner rings remain out of scope.

The six old CP1 weak cases regress their original requested point and arc. All
first commands retain that target with beta0. Their original first actual contact
traces are explicitly classified incidental: all six have zero positive requested
substeps. Processing those observations cannot mutate the requested identity.
These are bookkeeping/geometry regressions, not evidence of CP2 physical hits.

## Actual actions are gated by current-state wall risk

The separate frozen module `native_rgb_contact_pilot2_safety.py` evaluates the
full-union exterior vertices using current translation and angular boundary
velocity, known damping, control interval and actual bounded pusher target.
Its rules and limitations are in SAFETY.md and safety-rules.json. This is a
conservative heuristic, not a mathematical safety guarantee or future physics
rollout. It does not change force, velocity, inertia or collision parameters.

Every actual bounded command is assessed, including free-route-labelled actions;
a label cannot bypass an inward-action risk. For a verified committed departure
segment the risk contact context is its retained actual departure source rather
than the new destination contact. This context, candidate and final action are
logged. Follow/entry use the current executable target.

A risky command loses normal advance and velocity lead. The controller tries
finite geometry-verified outward retreats or tangential actions under the same
bounds; if none can be verified it records a current-state hold, without claiming
that inertial risk is thereby guaranteed safe. Two repeated wall-risk mitigations
can trigger a debounced alternate exterior target transaction. Progress uses the
actual final issued action, not a cancelled inward raw candidate.

Production-controller tests at CP1 X03 saved states0..5 flag risk before control5
and replace the unsafe candidate. A further regression labels the real control3
inward action `free_route`; the same production gate still cancels it. The safety
module also separately verifies an approaching rotational corner and does not
blanket-freeze the previously improved T040 profile.

## Frozen prephysics verification

Seven main controller regression checks passed using production methods. Evidence
is controller-regressions.json; six separate safety checks are in
safety-regressions.json. It covers T03 bridge commit, B04 four illegal-pair states,
X03 early warning timeline, mode-label non-bypass, six requested-arc traces,
fixed release withdrawal18..25, and all60 draft-case initial geometry/bounds.
Withdrawal tasks are not removed; a full known10-substep no-AO withdrawal interval
is required before contact is labelled separated-and-recovered in controller
telemetry. The evaluator independently checks those raw flags.

The explicit rule/source hashes are in controller-rules.json. No simulator,
renderer, model or physical control was used in these checks. The static evidence
permits freezing one physical case plan; it does not establish physical safety,
requested-arc success or the declared collection-efficiency goals. After the one
authorized physical pass, preserve this version unchanged and report any failure.
