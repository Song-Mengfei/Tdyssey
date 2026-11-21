# Goal overlay

An optional rendering aid for the camera policy, off by default. It paints the part of the goal that the block does
not yet cover in red, and thickens thin slivers so that they still show in the 96 px camera image. It follows the
professor's suggestion at the final talk: most failures are near misses (two in three reach 90% coverage), and one
96 px camera pixel spans about 5.3 simulator pixels, so a 94.6% and a 95.1% end state look almost the same to the
official camera. The experiment that used it and its result are below.

## What is drawn

1. Uncovered region: goal polygon minus block polygon, from the same collision polygons as the coverage metric
   (`physical_metrics`). Pieces under 1 simulator px^2 are ignored (floating-point slivers).
2. Thin parts: what a morphological opening with radius R/2 removes from that region (parts thinner than R),
   cleaned of numerical hairlines under 0.1 px, keeping pieces at least R long (convex-hull diameter). These are
   slivers along an edge and wedges from a small rotation; the small pieces an opening also cuts from corners of
   about 53 degrees or more are skipped, so the goal's letter shape is unchanged far from the block.
3. Red (255, 0, 0) = uncovered region plus the thin parts grown by R. It is painted on the 512 px canvas after the
   goal and before the block and pusher (both stay exactly as before and are drawn on top), and the official resize
   to 96 px (or 192 px) follows unchanged. Round joins use 4 segments per quarter circle (within 0.17 px of the arc);
   the region is filled at 1/8 px precision.

The experiment uses R = 9 simulator px. Code: `goal_overlay_region` and the `goal_overlay` option of
`make_official_shape_env_class` in `tdyssey/envs/official.py`. Changing the definition changes what a checkpoint was
trained on, so a changed definition needs a new name.

The overlay is drawn the same way above 95% coverage (pure geometry, no success threshold in the picture). Episodes
end at success in evaluation and in collection, so the policy only sees such frames in the human T demonstrations.

## Why simulator geometry and not image processing

The alternative was to find the still-visible goal-green pixels in the 512 px render. It cannot see the slivers that
matter: the official renderer draws every block edge as a 4 px wide line (2 px past the true edge) on integer-rounded
vertices, so an uncovered sliver up to about 2 px wide is hidden even in the sharp 512 px image, and at 90-95% coverage
the slivers are typically 1-4 px wide. The geometric region is exactly the region the success metric scores.

Disclosure: the overlay uses privileged simulator geometry (exact block and goal poses) in every image the policy
sees, in training and in testing. A policy trained with it is not a camera-only policy; results are reported
separately from the official 96 px camera, like the 192 px camera (a sensor change). It tests whether the missing
information limits the last nudge, not a deployable sensor.

## Choice of R

The block's drawn outline covers about 2-2.5 px of a sliver next to it, so a grown sliver shows at least R - 2.5 px of
red. The official 512 -> 96 px bilinear resize samples about every 5.3 px with a 2 px window, so a band needs roughly
7 px to cover a whole camera pixel sample at any position. R = 9 gives at least 6.5 px for a zero-width sliver plus
the sliver's own width. `scripts/learning/overlay_check.py` measured R = 5-13 on 250 random near-goal states per
coverage band: with R = 9 every state showed at least 64 (coverage 0.90-0.95) and 59 (0.95-0.99) fully red
camera pixels, against a median of 8 and 0 goal-green pixels in the official image of the same states (2 or fewer in
18% and 75% of them). Every R tried showed at least 25 fully red pixels; by eye R = 5 draws thin, partly broken bands,
R = 9 continuous bands, and R = 11-13 start to fill the letters' inner gaps (`radius_compare.png` in the check output).

Known property: a sliver just thinner than R is grown and shows a band of about its width + R, while a strip just
wider than R is drawn as it is. The switch happens at strip widths of about 9 px (coverage about 0.70-0.85); below
that, red area grows with the uncovered sliver (shifting the block step by step from the goal in 8 directions for 8
letters: median rank correlation 0.99, no drop above 15% in 576 steps). Pooled over letters the correlation is low
(0.41), because the band's length follows each letter's edges, not the uncovered area.

## Using it

| Step | How |
| --- | --- |
| Environment | `cls(shape=..., legacy=True, render_size=96, goal_overlay=9)`; 0 (default) is the official image, bit-identical |
| Checkpoint | `bc_train.py --goal-overlay 9` stores `arch["goal_overlay"] = 9`; `base_rollouts.build` sets `agent.goal_overlay`, and evaluation (`evaluate_batched`), collection (`mpc_dagger.py`), `record_letters.py` and the last-mile scripts render with it |
| Teacher frames | `overlay_frames.py demos --goal-overlay 9 DIR...` replays every demonstration (checked step by step, as `replay_frames.py`) into `DIR/frames-ov9/` |
| Human T frames | `overlay_frames.py human --goal-overlay 9 ZARR` renders each dataset step from its recorded state into `ZARR's folder/frames-ov9/` |
| Correction records | collected by an overlay student; files carry `goal_overlay` and, from now on, `block_pose` (both history frames) |
| Training | `bc_train.py --goal-overlay 9 [--mixed-records]`; record folders rendered differently stop the run unless `--mixed-records` |
| Checks | `CUDA_VISIBLE_DEVICES= .venv/bin/python -B scripts/learning/overlay_check.py --out DIR` |

## The experiment (settings fixed before any run)

Two arms continue from round 50 on one GPU, concurrently and with the same practice scenes, steps and data rules:
the overlay arm (R = 9) and a control arm without the overlay, so a gain cannot come from extra rounds. Round 0 adapts
each arm to its data (54k steps); then 14 correction rounds per arm (`scripts/learning/experiments/chain_goal_overlay.sh`). No
round starts later than 13.5 h before the job ends, which keeps at least 10 h for the last validation, the test, the
report and slack. Every round's student of both arms, and round 50 once, is validated in the background on the
selection scenes (seeds 400000 + 20*i, 50 per shape), paired by scene: a progress curve for the report
(`goal_overlay_report.py validation`), never used to pick a checkpoint. The test compares the last round both arms finished,
overlay against control (primary), on the final-test scenes (200 per shape, seeds 15000000 + 20*i) with round 50's
stored episodes (`goal_overlay_report.py`).

## Data that cannot carry it

The training set of round 50 has 3,205,500 samples. The teacher demonstrations and the human T demonstrations
(694,012 samples, 50.6% of the sampling weight) store simulator states and are re-rendered with the overlay. The 116
earlier correction folders (2,511,488 samples, 49.4% of the weight) store only their two camera frames, the pusher
position and the labels: no block pose and no executed actions, so they cannot be re-rendered. Re-running their
collection would not reproduce them (GPU sampling, earlier students). The experiment therefore keeps them with their official
frames in both arms (`--mixed-records` for the overlay arm), and new records store block poses so later runs can
re-render them. The human T dataset images come from an older renderer: re-rendering with the overlay off differs from
them on about 0.5% of the pixels (block fill and edges), so overlay runs use project-rendered human T frames while
official runs keep the dataset's images.

Rendering cost on one CPU core: along teacher trajectories 2.6-3.0 ms per frame officially
and 5.0-7.1 ms with the overlay (shapely operations and painting); replaying a demonstration with physics 13 ms per
frame officially and 21 ms with the overlay, so the 705k teacher frames take about 4 CPU hours (about 10-15 minutes
with 24 worker processes).

## Checks (CPU only)

`scripts/learning/overlay_check.py` passed every check: with the overlay off, replayed teacher frames equal the stored
`frames/` files and stepped episodes of six letters equal the code before the overlay was added (366 frames,
bit-identical); with the block exactly on the goal the overlay image equals the official one; the fast region equals
the coverage metric's to 5e-12 px^2; all 652 shifted near-goal states below 95% show red; the loader builds the same
samples and labels with overlay frames, its parallel and serial builds agree, and record tags are enforced. Evaluation
on one validation scene (seed 400000, T and G) with `base_rollouts.py eval --device cpu`: the plain round-50 student
solved both (T in 171 controls, G in 229); the same student given overlay images without training never covered any
of either goal in 300 controls, so the overlay arm needs its adaptation round. A one-scene `mpc_dagger.py` collection
on CPU wrote overlay-tagged records whose frames re-render from the stored block poses within one pixel.

