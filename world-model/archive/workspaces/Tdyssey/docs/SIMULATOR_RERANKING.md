# Frozen policy with physical-simulator action ranking

This optional inference baseline keeps the official T-trained CNN checkpoint
frozen. It needs no D demonstrations, learned world model, optimizer, or model
updates. The current two real observations still condition the original policy.
A separate D simulator predicts candidate effects and ranks the actions before
execution. This uses privileged simulator state, geometry and dynamics; it is
not an image-only policy or evidence of learned cross-shape adaptation.

Native Windows AMD64 validation on 2026-10-21 passed 13 planner protocol tests,
two real D contact/hold tests and three candidate tests including actual CUDA
checkpoint parity, with no skips in those focused suites. The 20-control smoke
run completed both T and D in 95.203 seconds; neither succeeded. These are
correctness/integration checks, not a performance claim. The earlier 2026-10-20
handoff only ran source tests on another machine. See
[actual commands and matched A/B/C results](VALIDATION.md#windows-simulator-reranking-validation-2026-10-21).

The existing environment was extended with the locked official-policy dependencies
and the published checkpoint. No D demonstrations or training were used.
Checkpoint file/content hash gates are disabled; configuration, complete state
keys/shapes, frozen inference and direct before/after weight equality checks
remain. See [checkpoint preparation](OFFICIAL_BASELINE.md).

## Control loop

1. `OfficialDiffusionPolicy.predict_action_candidates(history, K)` repeats the
   current two-observation history across a batch. Independent diffusion noise
   produces K action chunks through the unchanged official sampler. They are
   already unnormalized pixel targets and time-aligned to `action_pred[:, 1:9]`.
2. Each candidate branch restores the accepted episode-initial snapshot and
   replays every control actually executed in the live environment. It compares
   the resulting poses, velocities, forces, goals, controller targets and control
   count with the live state before predicting any future action. A mismatch
   aborts; it is not silently accepted as a valid forecast.
3. The branch executes a proposed policy prefix, then holds the attained pusher
   position for the rest of the forecast and optional settling controls. An
   immediate physical hold is also a candidate. All plans receive the same
   remaining-task-clipped forecast budget. Invalid/out-of-bounds samples are
   reported and rejected, without modifying their coordinates.
4. Rank completed forecasts using the terminal score below. Execute only the
   selected short command prefix in the live environment, update the real
   observation history, record the executed controls, and replan. The default
   executes one control. Original upstream success still terminates immediately;
   choosing hold below that threshold does not mean success.

`tdyssey/simulator_planning.py` implements the simulator branch/ranking protocol.
`scripts/evaluate_shapes.py` connects it to the existing matched-shape evaluator.
Omitting `--planner simulator` preserves the eight-action baseline by default.
The original `scripts/evaluate_official.py` is unchanged.

## Score and stopping

At each forecast control, coverage is computed from the actual D collision
polygons and their D-shaped goal, including the empty hole:

`coverage = intersection_area / goal_area`.

For a forecast that does not reach native success, the default score is:

```text
minimum coverage over the final 2 controls
  - 0.10 * mean object position error / 512
  - 0.05 * mean symmetry-aware object angle error / pi
  - 0.01 * mean object speed / 512
  - 0.01 * mean absolute object angular speed
```

Speed is in pixels/second and angular speed in radians/second. These fixed
engineering weights are not learned or validated for performance. A forecast
that reaches original `coverage > 0.95` success stops there and scores 1, matching
the actual runner's absorbing termination. Stable-overlap diagnostics do not
replace that task definition.

Position and angle error provide a progress signal before the shapes overlap;
coverage alone would tie all such plans, and motion penalties could favor holding
over approaching the goal. Near the goal, coverage remains the largest score
term. This is an engineering objective, not a guarantee of monotonic improvement.

By default each sample contributes its full eight-control policy prefix, followed
by two hold controls. `--prefix-lengths 1 4 8` additionally evaluates shorter
prefixes followed by hold. For example, if the object is closest after the first
four actions and actions five through eight push it away, the four-action prefix
can be selected. Hold uses the pusher's attained physical position as a fixed PD
target, allowing the simulator to model deceleration and ongoing contact.
`[0, 0]` is a move command and `step(None)` does not advance upstream physics;
neither implements physical stopping.

The total forecast is `min(planning_horizon + settle_steps, remaining_controls)`.
Hold/settling consumes that budget; no plan is ranked using a task outcome that
would require controls after the episode limit. At each actual action the runner
checks native success and stops executing any remaining commands when it occurs.

## Why replay the full history?

The existing `Snapshot` includes physical/controller state, but not Chipmunk's
internal contact solver cache. `restore_state()` builds a fresh space. Restoring
the current moving/contacting object would therefore not establish an identical
prediction start.

The accepted matched initial state is already restored in a fresh space for the
live episode. Replaying the same executed controls in a separate space reconstructs
its contact history. The branch skips rendering observations; its controller,
geometry and physics remain inherited. Only the live observations enter the
policy. Numerical state comparison checks reconstruction, and the opt-in D test
checks the following contact continuation as well. These runtime checks must pass
before claiming accurate physical predictions.

This first implementation costs approximately K times the elapsed episode length
in replay work per planning call, multiplied again when several prefixes are
enabled. A separate hold branch also replays history. It is a bounded correctness
baseline, not a real-time implementation.
Do not replace replay with a faster contact-state copy without validating it.

## Bounded runs and matched controls

On a machine with the existing official source/checkpoint/profile, first check a
small inference run. This evaluates both T and D on a common accepted scene:

```bash
.venv/bin/python -B scripts/evaluate_shapes.py \
  --shapes T D --planner simulator --candidates 4 \
  --planning-horizon 8 --prefix-lengths 1 4 8 --settle-steps 2 \
  --execute-steps 1 --steps 20 --seconds 300 \
  --output artifacts/td-simulator-smoke
```

A run-wide wall-time limit can stop before D or before 20 controls. Check report
status, the `requested_episodes` census and per-episode stop reasons. Unstarted
shapes are explicitly marked `not_executed`; an incomplete run is not a completed failure
or a success. Each output directory must be new. Increase the bounded run budget
deliberately after measuring replay and inference costs. `--display --video` uses
the existing viewer/video path and does not change policy observations.

Compare these three conditions with the same shapes, accepted scenes, policy seed
and actual-control budget. A shorter execution prefix alone can change behavior:

```bash
# Existing eight-control baseline.
.venv/bin/python -B scripts/evaluate_shapes.py --shapes T D \
  --planner none --execute-steps 8 --output artifacts/td-baseline-eight
# Replanning-frequency control: no candidate ranking.
.venv/bin/python -B scripts/evaluate_shapes.py --shapes T D \
  --planner none --execute-steps 1 --output artifacts/td-baseline-one
# Add simulator ranking, retaining one-control execution.
.venv/bin/python -B scripts/evaluate_shapes.py --shapes T D \
  --planner simulator --candidates 8 --prefix-lengths 1 4 8 \
  --execute-steps 1 --seconds 7200 --output artifacts/td-simulator-rank
```

Default episode limit remains 300 controls. Report wall-time truncation separately.
The Windows comparison used an explicitly smaller 64-control budget per shape
and 900 seconds per condition; reaching 64 controls is not task success.
`K=1` uses the exact baseline sampling path/RNG consumption. Batched K>1 samples
do not preserve the sequential single-sample RNG ordering, so candidate zero is
not claimed to reproduce the baseline chunk. A ranking gain must be separated
from compute, replanning frequency, and the privileged hold/geometry prior.

The report retains actual actions, states, native done flags, maximum/final IoU
and coverage. Each decision adds candidate scores, selected prefix, predicted
states/actions, rejected samples, timeout status and per-call simulator counts.
Episode `simulator_planning_counts` separately records restores, historical replay
controls and forecast controls, including interrupted work. Each such simulator
control uses the original ten physics substeps. Actual controls remain in the
existing top-level counts. Policy-only latency and total planning latency are
recorded separately, as are batch denoising iterations and sample-iterations.

## Verification

No third-party dependencies are needed for the protocol tests:

```bash
python3 -B scripts/check_repo.py
python3 -B -m unittest discover -s tests -p 'test_simulator_planning.py' -v
```

They check history-sensitive fake dynamics, isolation between candidates and the
live state, overshoot rejection, physical hold/prefix behavior, real task budgets,
native success and deadline accounting. Fake dynamics are not simulator evidence.
On the installed official environment, additionally run:

```bash
TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy \
  .venv/bin/python -B -m unittest discover -s tests -p 'test_simulator_planning_runtime.py' -v
TDYSSEY_DATA_TESTS=1 .venv/bin/python -B -m unittest discover \
  -s tests -p 'test_official_candidates_runtime.py' -v
TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy \
TDYSSEY_OFFICIAL_CHECKPOINT=checkpoints/official-dp/ema_policy.pt \
  .venv/bin/python -B -m unittest discover -s tests -p 'test_official_candidates_runtime.py' -v
```

The first needs only the simulator/source; the last needs the actual CUDA policy
and compares batched sampling with a direct unchanged official-class call.
No test command trains a model or obtains new D demonstrations.

If all T-policy samples are poor for D, selecting among them may still stall or
fail. Hold can preserve a near miss without completing the task. This version
does not add geometric search, candidate perturbations, CEM, or policy learning;
those are separate future experiments, not hidden parts of a reranking result.
