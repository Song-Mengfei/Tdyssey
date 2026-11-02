# CONTACT-PILOT2 current-state wall-risk gate

The new safety module examines each actual bounded follow/entry target using the
current full union exterior. Each boundary vertex gets the physical COM velocity
plus angular velocity crossed with its current COM offset. T retains its physical
local COM(0,45); this is distinct from its area centroid used by motion scoring.

For each wall, the gate subtracts a conservative unforced inertial displacement,
a rotation-curvature remainder and an active-command allowance from the current
vertex clearance. The unforced integral uses the unchanged damping0.0009765625
and0.1s control interval:effective time0.07213475s. Active allowance is capped by
the actual pusher target distance and200worldpx/s x0.1s,less current free distance
to the contact shell. Current contact torque sign and lever add a boundary sweep
allowance. Remaining drive room below a frozen8worldpx margin warns before the
original2worldpx penetration guard is reached. The guard itself is not changed.

The original X-03 recorded candidates first warn at control3,then4and5. At
control3 the top gap is32.47165worldpx and remaining room is7.515worldpx. This
precedes the old control5 failure. A zero-COM-velocity rotating-corner fixture
warns from its boundary velocity,which a COM-only check would miss. Mode labels
cannot bypass the gate:an inward action remains inward when called route or
withdraw. A real outward target can cancel added pushing,but its telemetry still
shows any existing inertial risk. Geometry-valid fallback selection belongs to
the controller; the risk module never edits any state or commands an environment.

Six focused tests use the production gate,actual X-03 states0..5 and T040 commands,
a rotating-corner current-state fixture,physical COM/damping checks and finite
JSON evidence. The T040 initial0/3and later30commands remain allowed; this is not
a rule that freezes all normal pushing. See safety-regressions.json for exact
inputs,outputs and hashes; all checks required0physics/env/render/model calls.

This is a conservative collector heuristic,not a mathematical safety proof or
a learned motion model. It does not predict contact forces,collision torque or
future physics. A late retreat cannot promise to stop already accumulated body
momentum. The one frozen physical pilot,not these static checks,will determine
whether the change actually avoids the recorded guard failure and retains useful
pushing. Any new guard/unknown advance must stop the parent run under its scope.
