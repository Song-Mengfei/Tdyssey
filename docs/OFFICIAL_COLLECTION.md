# Official T/L interaction data

The primary collector uses the original official PushT environment plus the tested
capital-L adapter. It collects non-expert interactions, not expert demonstrations
or frozen-policy evaluation traces. No policy checkpoint, new packages, downloads
or model training are involved. Existing CPU physics/rendering performs collection;
GPU use is a separate requirement for policy/model training.

## Collect and verify

Use the installed official profile from [environment setup](ENVIRONMENT.md).
The defaults produce a bounded 8-episode / at most 800-control pilot:

```bash
.venv/bin/python -B scripts/collect_official_play.py --shape L --episodes 8 --steps 100 --mode object-directed --output artifacts/official-play-L-001
.venv/bin/python -B scripts/collect_official_play.py --validate-only --output artifacts/official-play-L-001
.venv/bin/python -B scripts/prepare_training_data.py --datasets artifacts/official-play-L-001 --output artifacts/official-play-L-001.catalog.json
.venv/bin/python -B scripts/prepare_training_data.py --catalog artifacts/official-play-L-001.catalog.json --split validation
.venv/bin/python -B scripts/prepare_training_data.py --catalog artifacts/official-play-L-001.catalog.json --split model_test
```

Choose a new dataset directory and catalog filename; existing output is refused.
Raw datasets and generated catalogs remain outside Git. CLI help needs only Python:
`python3 -B scripts/collect_official_play.py --help`.

| Limit | Default | Maximum |
| --- | --- | --- |
| Episodes | 8 | 20; at least 7 for nonempty frozen splits |
| Actions per episode | 100 | 250 |
| Total actions | 800 | 1000 |
| Encoded dataset byte budget | 32 MiB | 64 MiB |
| Total wall time including loading | 120 seconds | 600 seconds |
| Reset candidates per episode | 20 | 20 |

Conservative preflight and actual encoded-byte checks bound storage. Filesystem
allocation and temporary manifest replacement need a little additional space.
Failure/time-budget exhaustion leaves an incomplete dataset rejected by normal
validation. These pilot limits are not a claim that 800 interactions suffice for
reliable learned dynamics. Larger budgets need an explicit protocol decision.

## Exploration and reset conditions

`object-directed` aims through the object's current position with noisy offsets
75% of waypoint changes; other waypoints sample the workspace. It uses privileged
simulator position as a disclosed geometric exploration prior. It does not select
actions using reward, goal alignment, a learned model or an expert policy.
`--mode random-waypoints` removes the object-position prior. Both retain the
existing 12-pixel maximum target displacement and 12-control waypoint interval.
Actions are absolute world-position targets in pixels.

Use the [official geometry/physics](OFFICIAL_SHAPES.md): original T or mass-1
capital L with height 120, foot width 90, stroke 30 and inertia 1875. The original
T goal stays `(256,256,pi/4)`; L's goal origin is shifted to match T's target mass
center, not copied blindly from T. The collector uses this fixed official goal,
unchanged pusher/controller/contact convention, and legacy reset semantics.

Reset/action streams are versioned and independent. Candidate resets undergo
rotated object/goal wall checks, pusher clearance and not-already-solved checks.
Each simulated reset, accepted or rejected, contributes its original one physics
substep. The full post-reset physical state and accepted restored state are saved.
Clear the previous pusher target before recording observation zero. Restore into
a fresh space without extra physics; cached contact impulses/future-reset RNG
remain outside the snapshot.

Known development seeds **100000, 100001 and 100002** are always excluded. Add
future evaluated scenes with repeated `--exclude-seed NUMBER`; extend the shared
list before using more development scenes in the project. An excluded seed
candidate does not call the simulator or consume a reset-physics step; it is still
recorded with `simulated: false` and counted under `excluded_seed_candidates`.
These checks cover the recorded seed list, not every possible near-duplicate scene
or an unspecified future evaluation set.

## Timing, splits and data identity

The original official environment returns `(observation, reward, done, info)`.
Here `done` denotes original success. The collector's action horizon is recorded
as truncation, separately from success; the unwrapped source has no timeout flag.
A later RL trainer must bootstrap time limits deliberately. No learning mask is
silently inferred by the collector or [window loader](TRAINING_DATA.md).

Store exactly T+1 uint8 image frames for T actions, with matched agent/goal poses,
full physical/controller states, action targets, original rewards, contact counts
and coverage/IoU diagnostics. Object state, rewards and contact/overlap labels are
privileged supervision. Original goal overlays remain in the policy images.
Frame quantization is `round(image*255)` and each image is stored once.

The shared [episode format](COLLECTION.md#files-and-timing) is reused. Each reset
record additionally identifies whether simulation occurred and stores full reset
state provenance. Counts refer to actual simulated resets, not skipped seeds.
Whole-episode splits use the fixed seed and 80/10/10 allocation, then refuse an
empty set or repeated initial scene. Adding episodes later requires a new explicit
freeze; windows never cross episode or split boundaries. Nonempty episodes can
still give zero windows when a chosen history/future exceeds episode length, so
check all batch reports rather than assuming model-quality evidence.

Official datasets carry environment family `official-pusht-capitals-v1`. Catalog
creation/loading refuses to mix them with GPC reference data even when control
rates and shape names match. Historical GPC records/catalogs retain their existing
meaning; missing legacy family tags default to `gpc-reference-v1`. Do not relabel
old GPC fixtures as official data.

## Evidence and next task

The first retained L pilot collected 800 controls across eight 100-step episodes,
with 354 contact steps, 12 simulated resets and 3,279,264 bytes of data. Frozen
train/validation/model-test membership is 6/1/1 episodes. History 2/future 4 gives
576/96/96 windows, and two batches of four examples passed in every split. These
are data-path checks, not learned-model accuracy or an adaptation benefit. See
[validation details](VALIDATION.md#official-l-collection-and-training-data-2026-10-19).

The [first visual model](WORLD_MODEL.md) now implements training-only normalization
and validation contact/noncontact comparisons against simple baselines. Its first
GPU pilot still has drift and action-ranking failures. Reserve model-test data for final model comparisons;
use validation for development choices. Require contact-dynamics and selected-action
checks, then verify direct physical-simulator policy optimization before relying
on imagined updates. More data, repeated seeds and a fixed final policy-evaluation
set are still needed for research conclusions.
