# Story outline: talk (10 minutes) and paper

Numbers come from `presentation/data/` (built from the saved test files) and `docs/LEARNING_FINDINGS.md`. Figure,
image and video names refer to `presentation/figures/`, `presentation/images/` and `presentation/videos/`.

## One-sentence story

A camera policy that only knew how to push a T now pushes all 26 capital letters: we never collected new human
demonstrations; a planner that cheats with exact simulator state teaches it, and the camera policy learns to imitate
that planner and then corrects its own mistakes.

## Key numbers (all on fresh test scenes never used for training or model selection)

| Claim | Number | Source |
| --- | --- | --- |
| Original policy on the 27 trained shapes | 3.7% success | final test, 300 scenes per shape |
| Our 26-letter policy, same camera (96 px) | 33.4% (round 44); 34.8% after six more rounds (round 50) | final test; separate fresh test |
| Our 26-letter policy, sharper camera (192 px) | 33.8% | final test |
| 17 letters added last | 0.4% -> 33.2% | final test |
| Ten earlier shapes | 9.3% -> 33.7% (no loss from adding 17 letters) | final test |
| Six never-seen shapes | 1.2% -> 2.8% (10 shapes) -> 4.7-5.9% (27 shapes); digit 7 2% -> 17-23% | final test |
| Goal covered at the end, 27 shapes | 0.35 -> 0.82 | final test |
| Ten-shape model on its ten shapes | 9.1% -> 34.3% (96 px), 38.2% (192 px) | 1000 scenes per shape |
| Teacher on validation scenes | 390/500 (10 shapes); 36-49/50 on most new letters | validation |
| Forward RL (what we tried first) | T success 0.55 -> 0.05 with larger steps; only 16-36% of the training score comes from the plan itself | learning findings, sections 1-2 |
| Intended-plan labels vs executed pushes | 129 vs 77 successes of 1000 (largest single gain) | label A/B |
| Bigger per-shape budget (54k steps, 60 scenes per new letter) | plateau ~24% -> ~35% validation | 26-letter run |
| Executing 12 or 15 of 16 planned steps instead of 8 | 18.6% and 12.2% vs 34.8% | steps test |
| T after adapting | 46.0% -> 40.7% (96 px; interval includes 0) and 44.0% for round 50 | final tests |

## Talk plan (10 minutes, about 12 slides)

| # | Slide | Time | Message | Visual |
| --- | --- | --- | --- | --- |
| 1 | Title | 0:20 | One camera policy for every letter | `images/shape_gallery_33_upright.png` |
| 2 | The problem | 0:50 | The official policy only knows T: 3.7% on the other shapes | `videos/compare/K.mp4` (original side), `figures/final_*` original bars |
| 3 | Rules we set ourselves | 0:40 | No new human demos; the deployed policy sees only the camera; privileged state allowed only for training, and disclosed | `images/camera_views_512_192_96.png` |
| 4 | First attempt: RL on the simulator | 0:50 | Training signal mostly noise; larger steps destroyed T | `figures/rl_*` |
| 5 | The teacher | 1:00 | A learned world model + a sampling planner that reads exact state solves most letters | `videos/teacher/teacher_9shapes.mp4`, `figures/method_teacher_success*` |
| 6 | From teacher to camera policy | 1:00 | Copy the teacher's intended 16-step plans, then correction rounds (policy drives, teacher labels) | method diagram (TikZ on the slide) |
| 7 | Training story | 1:00 | 5 -> 10 -> 27 shapes; plateau fixed by a per-shape budget | `figures/curves_all_stages*`, `figures/curves_26letters*` |
| 8 | Final test | 1:10 | 3.7% -> 33.4-34.8% over 27 shapes; new letters catch up with old ones | `figures/final_grouped_bars*`, `figures/final_per_shape_bars*` |
| 9 | Demos | 1:00 | Same scene, original vs ours; all 33 shapes | `videos/compare/*.mp4`, `videos/all_shapes/l26_r50_33shapes.mp4` |
| 10 | Generalization and limits | 1:00 | Unseen shapes improve a little with more training shapes; the policy often gets close but misses the final alignment | `figures/heldout_scaling*`, `figures/coverage_*`, `videos/failures/*` |
| 11 | What we learned | 0:40 | Labels and per-shape budget mattered most; executing more steps per plan hurts | `figures/analysis_*` |
| 12 | Conclusion and next steps | 0:30 | Sharper alignment (camera resolution, last-mile corrections), real robot | summary table |

Backup slides: test protocol and seeds; the 192 px side result; round 50 vs 44; T and near-T trade-off; teacher details
(about 1,150 candidate pushes per step, 1 px / 1 degree state noise); the full experiment log.

## Paper outline (one week after the talk)

1. Introduction: shape transfer for visuomotor diffusion policies without new human demonstrations.
2. Related work: Diffusion Policy; RL fine-tuning of diffusion policies (DPPO, DSRL, forward-process methods);
   privileged teachers and DAgger; world models for planning; generalist manipulation.
3. Setup: official PushT, letter shapes with matched physics, held-out shapes, metrics, fresh-scene protocol.
4. What did not work: forward-process RL (signal-to-noise analysis), simulator best-of-N, gradient-based planning.
5. Method: structured world model, sampling planner teacher, intended-plan labels, correction rounds, per-shape
   budget.
6. Results: final tests (5, 10, 27 shapes), held-out shapes, coverage analysis, ablations (labels, camera, data,
   network size, steps executed).
7. Limitations and disclosure: privileged teacher, simulation only, final alignment, T trade-off.
8. Conclusion.

## Likely questions

- *Is the teacher cheating?* It reads exact simulator state, but only to create training data. The tested policy sees
  only camera images and the pusher position, exactly like the original.
- *Were the test scenes fair?* Every final test used a seed pool never used before, fixed before the run, with all
  models on the same scenes and 95% intervals on paired differences.
- *Why not reinforcement learning?* We tried first (sections 1-2 of the findings): the score that drives the update is
  mostly noise, and larger steps destroyed the T skill.
- *Does it generalize to new shapes?* Only a little (4.7-5.9% on six never-seen shapes), but it grows with the number
  of training shapes; it is adaptation with machine-generated data, not zero-shot transfer.
- *Did T get worse?* Slightly at 96 px in the final test (46.0% -> 40.7%, interval includes zero); round 50 reaches
  44.0% on its own fresh test.
- *Why is success only about 35%?* Blocks usually end close (82% coverage on average, 91% at the best moment), but
  success needs 95%: the last millimetres of alignment are the open problem (one 96 px camera pixel is about 5
  simulator pixels).
- *How much compute?* Single GPUs (H100, A100), several 3-day jobs; about 54k training steps per correction round.
