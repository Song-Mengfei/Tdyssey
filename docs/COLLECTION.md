# Bounded non-expert interaction collection

For the primary official T/L pipeline, use [official collection](OFFICIAL_COLLECTION.md).
The remainder of this page describes the separate historical GPC collector.

The collector uses the existing pinned GPC simulator and T/I/L adapter. It does
not load an expert policy, optimize reward, install packages, download assets, or
train a model. Use the installed CPU/GPC profile for small runs.

## Run a small collection

From the project root after the setup in `docs/ENVIRONMENT.md`:

```bash
.venv/bin/python -B scripts/collect_play.py --shape L --episodes 2 --steps 60 --mode object-directed --max-mib 8 --output artifacts/play-L-001
.venv/bin/python -B scripts/collect_play.py --validate-only --output artifacts/play-L-001
```

Output must be a new directory. Existing data is never overwritten or resumed.
`artifacts/` is ignored; choose a configurable path on another disk when needed.
`--help` also works with system Python without third-party packages.

| Setting | Default | Hard pilot limit |
| --- | --- | --- |
| Episodes | 2 | 20 |
| Control actions per episode | 100 | 250 |
| Total control actions | 200 | 1,000 |
| Published file contents | 16 MiB | 64 MiB |
| Reset attempts per episode | 20 | 20 |

A conservative preflight estimate rejects configurations that could exceed the
byte budget. The writer also checks actual encoded size. Filesystem allocation,
a temporary manifest replacement, and one episode's in-memory compression buffer
are additional small costs. Larger collection budgets require a deliberate
protocol change.

## Exploration and reset protocol

- `random-waypoints` is the default. A seeded controller samples workspace points
  and moves its position target toward them by at most 12 pixels per control step.
  It does not use object position or reward to choose actions.
- `object-directed` samples noisy targets through the object's current position
  75% of the times a waypoint changes. It uses **privileged simulator position**
  to encourage contact. This is a geometric exploration prior, not an expert
  demonstration or learned policy. Neither mode optimizes reward.

Reset and action randomness use separate recorded seed streams. Reset candidates
are checked using rotated collision geometry, circle-to-object clearance, wall
bounds for the pusher/object/goal, and initial goal overlap. Rejected candidates
are recorded. This changes the initial-state distribution: use the same accepted
reset protocol and comparable seeds for collector comparisons. These checks do not
establish dynamic goal reachability or safety during motion.

The collector fixes the goal at `(256, 256, pi/4)` after upstream reset, replacing
GPC's random goal for this data protocol. It restores a fresh space without an
additional physics step and clears the prior action target. This is a collection
condition, not a change to the original T baseline evaluator.

## Files and timing

Each run contains a `manifest.json` and two files per episode:

| File | Contents |
| --- | --- |
| `episode-NNNN.npz` | `images`: `(T+1,3,96,96)` uint8; `agent_pos`: `(T+1,2)` float64; `goal_pose`: `(T+1,3)` float64 |
| `episode-NNNN.json` | State/action trajectory, reset/action seeds, rejected reset diagnostics, contact/overlap observations, and stop reason |
| `manifest.json` | Config, package/source versions and hashes, file hashes/sizes, executed-step accounting, and frozen whole-episode splits |

Observation 0 precedes action 0. Observation `t+1` follows action `t` and matches
the recorded `after` state. Images are rounded from `[0,1]` to uint8 with at most
half a byte level of quantization error. Store each image once; construct histories
when loading training windows. The original goal overlay is retained. Actions are
absolute world-frame position targets in pixels, not normalized displacements.

Physical states, velocities, contact labels, and geometric scores are privileged
labels. Training with them must be distinguished from image-only learning.
The episode loader validates bounded pilot data. `docs/TRAINING_DATA.md` describes
the windowed/batched loader and frozen combined catalogs. Conversion to the original
GPC training format remains pending.

The last transition is truncated if the collector reaches its action horizon.
Diagnostics separately retain the original upstream terminal flags and collector
cutoff. GPC may set both upstream flags on success; those flags are preserved.
Training must define terminal/timeout bootstrapping deliberately.

Contact counts preserve GPC's `ceil(mean contact-point count over the control
step)`. They are not unique contact pairs, impulses, or a contact-coverage score.
Accepted and rejected resets each execute one upstream physics substep; this work
is counted separately from control actions and their physics substeps.

## Integrity and use in experiments

The command reads saved data back before reporting completion. The loader checks
hashes, trajectory continuity, T+1 alignment, image dtype/shape, observation/state
agreement, terminal diagnostics, split coverage, and interaction accounting.
Interrupted or failed runs remain labelled `failed` or `collecting` and are rejected
by the normal loader. Inspect them explicitly rather than using partial data as a
completed dataset.

Splits use whole episodes, never windows from the same episode in multiple sets.
Membership is saved when the run finishes. Tiny verification runs can have empty
validation/test sets and cannot support model-quality claims. Held-out shapes and
policy-test initial states need separate manifests. Do not merge per-run splits
blindly: deduplicate trajectory IDs and freeze a combined manifest first.

```bash
python3 -B -m unittest discover -s tests -p 'test_collection.py' -v
TDYSSEY_GPC_ROOT=third_party/gpc .venv/bin/python -B -m unittest discover -s tests -p 'test_collection_runtime.py' -v
```

The runtime tests cover real collection/read-back, quantization, rejected resets,
reproducibility, horizon handling, overwrite refusal, and corruption detection.
One terminal-handling test injects a terminal flag after a real step; it tests
recorder behavior, not task success.
