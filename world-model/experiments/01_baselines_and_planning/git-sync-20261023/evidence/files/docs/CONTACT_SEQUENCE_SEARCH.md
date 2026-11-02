# Contact-sequence simulator search

This is a privileged, policy-free D control experiment on native Windows/Pymunk.
It uses known geometry, full physical state and the original simulator. It loads
no checkpoint or Torch model and performs no policy, student or world-model
training. It is not image-only deployment or learned cross-shape adaptation.

## Representation and search

`tdyssey/simulator_sequence.py` provides a separate `ContactSequenceSearch`
backend. The original search and frozen-policy reranker remain unchanged.
`tdyssey/contact_geometry.py` promotes the exterior-boundary and clearance helpers
from the preserved V2 prototype. Exterior extraction excludes enclosed-hole
contact proposals; collision checks retain every actual shape component.

A stage stores a material edge ID, position on that edge, staging clearance,
push distance, yaw, push/approach speeds, route choice, bend and bounded press
dwell. A candidate starts with one or two such stages, with bounded third-stage
precision extensions described below, physical release/reapproach and final holding. A visibility graph
around actual exterior corners supplies approach paths. At an existing corner
contact, release candidates must clear the entire disc footprint, not merely
one arbitrarily selected face normal. The complete generated route is physically
simulated; intermediate poses never set object state.

The search considers hold and the incumbent, eight initial contact proposals,
four diverse retained prefixes with four second-contact proposals each, eight
joint refinements, and bounded spare exploration. Rejected and timed-out
proposals consume the same 40-attempt hard cap. Prefix retention protects angle
progress and different contact locations/routes rather than ranking by overlap
alone. The second contact is generated from the first push's actual predicted
endpoint. Joint refinements mutate both stages and resimulate both in sequence.

Continuous elite statistics are conditional on the same edge/route sequence.
Only a minority of previous elites is reused and variance has a positive floor.
Near the goal, deterministic stroke lengths and refinement variance scale down
with position/angle error. This is iCEM-inspired memory and refinement, not a
reproduction of the paper's complete optimizer or a learned sampler.

Fixed material-point RMS error is the base non-success endpoint score. Points
are the unique component vertices, each matched to the same point in the target
pose, respecting declared shape symmetry. The maximum RMS over the final three
controls prevents selection by a transient good frame. Orientation influence
does not fade with coverage. Hard trajectory/workspace checks precede native
success and endpoint scoring. Precision mode adds the disclosed access/control
cost below; coverage and terminal motion remain secondary comparisons. Native
coverage greater than 0.95 is absorbing success.
The pusher still receives absolute PD targets; forces or object poses are not
assigned directly. Sticking/sliding behavior is produced by Pymunk, not solved
through explicit analytic contact-mode constraints.

## Replay, execution and budgets

Every hypothetical sibling reconstructs initialization plus all real executed
actions and its entire hypothetical prefix. Saved prefix states are geometry
metadata, never cold contact restores. Hypothetical-prefix replay counts as
forecast work; actual-history replay is counted separately. Geometrically invalid
root/child proposals are rejected before replay when endpoint metadata suffices.

All completed paths use the same `min(80, remaining episode controls)` horizon,
including physical holding at the attained pusher position. A single-contact fallback can use the full active budget;
only a genuine pair reserves controls for its second stage. No candidate gets
free post-budget controls. A candidate that overruns its deadline cannot win,
but all of its work remains counted.

The selected plan retains exact commands, phase boundaries and all predicted
snapshots. Execution checks every actual control against prediction. It consumes
at most eight approach/hold controls or two push controls per returned block,
without crossing a phase boundary unchecked. A matching continuation performs
zero new branch simulation. Full search occurs at plan completion or a bounded
stall trigger, with at most eight global searches per episode.

For one attempted rollout, history length `h` plus total hypothetical length `H`
is at most 300 controls. Thus 40 attempts and eight global searches give a
conservative 96,000-control search bound, separate from actual execution, reset,
initial live-history reconstruction and independent video audit work. The CLI
also enforces its wall deadline. Search-count exhaustion is a recorded failure,
not success or an unreported hold loop.

## Precision and budget revision (2026-10-22)

The third development revision addresses the two failed second-revision cases.
The pilot reuses their original control-160 states and all caps, rather than
granting extra controls after control 300. Changes were fixed before either
new outcome was observed; this is a combined mechanism change, not an ablation.

1. Settling reserve is `min(configured maximum, horizon // 12)` for a slow
   object, or `horizon // 8` above 5 px/s or 0.1 rad/s. Five remaining controls
   therefore do not become one push/approach control plus four mandatory holds.
   Any available forecast tail still uses physical holding and counts against
   the common horizon. Native success stops immediately.
2. Joint RMS at most 25 px, or at most 40 episode controls left, activates
   precision proposals. Approach targets use 220 px/s while push targets begin
   at 35 px/s and can be refined within 12--80 px/s. Fine stroke lengths can be
   0.5 px. Up to three repeated terminal PD targets allow the pusher to establish
   contact; they are **push controls**, not zero-force waits or simulator holds,
   and are shortened if the remaining budget cannot fit them. Nearby same-face
   proposals can maintain contact rather than release and reapproach. Every
   target still passes the original actual physics and target continuity checks.
3. Fine proposals include the pusher's projected position on an exterior edge,
   stronger access preference and candidate pairs from opposite exterior faces.
   The second push is generated after the first actual simulated endpoint, with
   translation compensation and desired torque in its proposal priority. Actual
   cancellation is never assumed. Elite memory does not substitute a coarse
   stage for a precision root. There are still at most two stages and 40 total
   attempts, including rejects; no new seed sweep or larger beam is used.
4. Precision ranking adds a modest cost in RMS-pixel units for approach controls,
   active controls, access to a useful next contact and an estimated remaining
   control deficit. The estimate uses next-contact distance/22, position error/3.5,
   material radius times angle error/3.5 and three lag controls. It is a geometric
   heuristic, not a feasibility guarantee or physical value function. Native
   success and safety remain ahead of this cost. Reports retain raw RMS and
   coverage separately from the cost and split approach versus push contacts.

Presses are sampled target sequences with exact-state execution checks and
replanning at their end, not a new force-feedback controller. Original dynamics,
control frequency, native threshold and policy checkpoints are unchanged. The
same-budget pilot results are recorded in `docs/VALIDATION.md`.

## Native Windows commands

Run from the repository root in the existing environment:

```powershell
.\.venv\Scripts\python.exe -B scripts/check_repo.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_sequence.py' -v
$env:TDYSSEY_OFFICIAL_ROOT='third_party/diffusion_policy'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_sequence_runtime.py' -v
.\.venv\Scripts\python.exe -B scripts/evaluate_contact_sequence.py --resume-report artifacts/d-teacher-20261021/matrix-v2/scene-100100-search-42/report.json --resume-step 160 --steps 300 --seconds 600 --horizon 80 --max-evaluations 40 --max-global-searches 8 --search-seed 42 --output artifacts/new-contact-sequence-run
```

Use a fresh output directory. `--steps` includes imported history; the example
allows at most 140 new controls. Resume resolves complete recursive source
histories and checks every replayed state, reward, contact count and nonterminal
flag. The initial state, source references and all actual actions are preserved.

## Scope of the first results

The 2026-10-22 pilot resumed three previously named V2 scene/search-42 reports
at control 160. First-revision failures were retained; after corner-release,
short-budget and near-goal proposal fixes, the same starts/seed/caps were rerun.
One of three corrected recoveries reached native success; two exhausted the
300-total-control budget. This is development-state rescue evidence, not reset
success, a general success rate or multiple demonstrated successful strategies.
All six runs have independently checked, fixed-pause videos. See
[complete results and costs](VALIDATION.md#two-contact-search-recovery-pilot-2026-10-22).

## Research references

The graph/sequence formulation borrows from [Graesdal et al., RSS 2024](https://www.roboticsproceedings.org/rss20/p132.pdf),
contact-guided branching from [Cheng et al., ICRA 2021](https://arxiv.org/abs/2011.01454),
and elite memory from [Pinneri et al., CoRL 2020](https://proceedings.mlr.press/v155/pinneri21a.html).
This implementation uses simulator sampling, not their convex relaxations or
quasistatic projection solvers, and has no inherited optimality/completeness
guarantee. The deterministic geometry/torque priors are disclosed supervision.

## Guarded joint correction revision (2026-10-22)

Revision 4 addresses observed center-for-angle tradeoffs in revision 3. Coarse
search still uses material-point RMS. Once precision mode starts it remains
active. Non-native precision selection requires separate terminal position and
angle limits, each evaluated over the last three forecast states. At a macro
boundary they are `max(3 px, current position error + 0.25 px)` and
`max(2 degrees, current angle error + 0.25 degrees)`, ratcheted down against
the previous limits. A smaller angle cannot justify violating the center limit.
These planning scales never replace the original coverage-based success rule.

The predicted intermediate trajectory may exceed the terminal center limit by
up to 3 px and angle limit by up to 8 degrees (relative to the larger of current
error and terminal limit), permitting finite release/compensation moves. Each
real executed control checks that corridor as well as exact prediction agreement.
Among feasible non-native plans, ranking first minimizes the worse of terminal
position error / 3 px and angle error / 2 degrees, plus 0.02 times the existing
access/control penalty. Native success has priority over these heuristic pose
guards, with original workspace safety and immediate stopping preserved. A
precision search with no eligible completed plan stops explicitly, not by silently
executing an infeasible plan.

Second contacts use the first push's actual simulated center-to-goal vector.
New candidates project the center onto accessible edges, favor translation along
that vector with small or beneficial torque, and scale travel by the vector's
magnitude and normal alignment. Opposite-face and rotation alternatives remain.
Actual compensation is logged only if the whole multi-stage candidate reduces
center error relative to the attained first-stage state; an edge label or the
presence of two pushes is not sufficient.

Prefix pruning preserves a position-improving state as well as orientation and
joint progress. Continuous refinement retains separate position, angle, joint
and actual-compensation frontier representatives rather than refining only the
two lowest scalar scores. Precision memory occupies at most one root slot.
Press dwell is sampled in 0--3 controls: half of the configured eight refinement
slots are joint parameter changes, half explicit dwell variants. Dwell consumes
actual controls; ending a press transitions to a physically simulated hold or
subsequent stage, not instantaneous velocity reset.

If available two-stage paths do not attain both 3 px / 2 degree planning scales
or native success, at most four **spare** attempt slots extend two diverse pair
endpoints with a third contact. All siblings rebuild the complete initial,
real and entire hypothetical history, now including both earlier stages. No
cold endpoint restore or direct pose assignment is used. The ordinary defaults
still cap a global search at 40 attempts and an episode at eight searches,
80 forecast controls, 300 total actual controls and the CLI wall deadline.
Three-contact rejection, dwell variants and refinements all consume that same cap.

Revision 4 is a combined development change on the same two original control-160
states and search seed 42, not a held-out evaluation or a single-mechanism ablation.
Its pilot and paused videos are recorded separately under
`artifacts/d-sequence-20261022-r4/`; prior outcomes remain untouched. No policy,
world model or student training is part of this revision.

### Executed terminal correction

The first two revision-4 runs exposed an execution/scoring mismatch: the score
used the long forecast's held endpoint, while execution could replan immediately
after the last press target, before executing the holds that produced that score.
Those runs and videos remain preserved. A bounded correction is frozen separately
as revision 5 in `artifacts/d-sequence-20261022-r4-settle/`, using the same two
starts, seed and every original cap.

Precision selection now scores and constrains the exact prefix it intends to
execute: active targets plus up to the configured four physical settling controls,
clipped by the same forecast/remaining budget. The full common-horizon forecast
is still performed and counted. Reports distinguish `execution_*` endpoint
metrics and `execution_controls` from the full forecast's legacy `final_*` fields.
The executor retains those exact hold targets and predictions, caps returned
blocks at the execution boundary and verifies the terminal position/angle limits
against the last three actual states when the boundary is reached. Native success
still stops immediately, with no settling after success. Settling is real control
work, not free post-budget simulation, and does not spend a new global search.

The revision-5 contract audit also found the old scalar RMS stall trigger
interrupting two precision macros before their execution boundary. Revision 6
disables that coarse-search heuristic once precision mode is active: separate
pose constraints and per-control corridor/state checks govern those macros.
A dedicated plateau regression verifies zero-work continuation. The final
same-cap correctness rerun is frozen under
`artifacts/d-sequence-20261022-r4-final/`; versions 4 and 5 remain recorded.
