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

## Talk plan (10 minutes; matches slides/main.tex)

| # | Slide | Time | Message | Visual |
| --- | --- | --- | --- | --- |
| 1 | Title | 0:15 | One camera policy for every letter | `images/shape_gallery_33_upright.png` |
| 2 | The problem | 0:35 | The official policy only knows T: 3.7% on our 27 shapes | `videos/compare/P.mp4` |
| 3 | Rules we set ourselves | 0:25 | No new human demos; camera-only policy; privileged state only for training data | `images/camera_views_512_192_96.png` |
| 4 | First attempt: RL | 0:35 | Training signal mostly noise; larger steps destroyed T | `figures/rl_t_collapse`, `rl_signal_variance` |
| 5 | What else did not work | 0:30 | Best-of-N re-ranking, weight blends, back-propagating through the world model | `figures/oracle_bon`, `final_blends_all` |
| 6 | The teacher | 0:40 | World model + sampling planner with exact state solves most letters | `videos/teacher/teacher_9shapes.mp4`, `figures/method_teacher_success_26letters` |
| 7 | How good is the world model | 0:30 | Accurate over a push; errors grow open-loop, so the teacher re-plans | `images/world_model/imagined_vs_real_E.png`, `figures/wm_accuracy` |
| 8 | From teacher to camera policy | 0:40 | Copy intended 16-step plans, then correction rounds | `figures/method_pipeline` |
| 9 | How the training data is made | 0:30 | Teacher labels states the policy reaches; intended plans are smooth | `images/training_data/*.png` |
| 10 | Training story | 0:35 | 5 -> 10 -> 27 shapes | `figures/curves_all_stages` |
| 11 | 26 letters: budget | 0:30 | Per-shape budget was the bottleneck (24% -> 35%) | `figures/curves_26letters` |
| 12 | Final test | 0:45 | 3.7% -> 33.4% (96 px), 33.8% (192 px); round 50 +1.9 on another fresh pool | `figures/final_grouped_bars` |
| 13 | Per shape | 0:25 | New letters improve most; T and near-T dip | `figures/final_dumbbell` |
| 14 | Demos | 0:40 | Same scene, original vs ours | `videos/compare/A.mp4`, `M.mp4`, `S_scene2.mp4` |
| 15 | Never-seen shapes and the gap | 0:35 | Small transfer that grows with training shapes; misses are within a camera pixel or two | `figures/heldout_scaling`, `images/last_mile/near_miss_vs_solved_zoom.png` |
| 16 | What mattered | 0:30 | Intended-plan labels; executing more steps hurts | `figures/analysis_label_ab`, `analysis_steps_executed` |
| 17 | Limitations and disclosure | 0:30 | Privileged teacher, hand-designed candidates, human T reuse, one run per setting, T/near-T dip | text |
| 18 | Conclusion | 0:20 | One policy for all letters, no new human demos | text |
| 19 | Try it (demo Space) | 0:15 | Screenshot or recording of the Space | TODO |

Backup slides: protocol and headline table; per-shape table; round 50 vs 44 and 192 vs 96 px; T and near-T; ten
shapes; teacher vs camera policy (video); teacher per letter and state noise; the same scene through the project;
digit 7 solved; data and compute; project timeline. Ongoing work (D058/D059: last-mile RL and a new flow-matching
policy) can be added as one slide once it has fresh-test numbers.

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

## Figure plan for the paper

Draft in `paper/main.tex` (build notes in `paper/README.md`). Paths are relative to `presentation/`; tables are
`\input` from `data/tables/`. Figures are slide-sized, so regenerate them at column width before submission.

| Paper section | Figure / table files | Why |
| --- | --- | --- |
| 1 Introduction | `images/shape_gallery_33_upright.png` (Fig. 1, full width); `data/tables/final_headline.tex` (cited) | Shows the whole shape scope (27 trained, 6 held out) at a glance; the intro quotes the headline numbers |
| 2 Related work | none | Text only |
| 3 Setup | `images/camera_views_512_192_96.png`; scene-pool table written in `main.tex` | What the policy sees and why final alignment is hard (one camera pixel is about 5 simulator pixels); which seeds each test used |
| 4 What did not work | `figures/rl_t_collapse.pdf`, `figures/rl_signal_variance.pdf` | Larger RL steps destroyed T, and the cause: most of the training score was noise |
| 5 Method: pipeline and world model | TikZ diagram in `main.tex`; `images/world_model/imagined_vs_real_D.png` | The whole recipe in one picture; the world model tracks the simulator over 8-32 controls |
| 5 Method: teacher and labels | `figures/method_teacher_success_26letters.pdf`; `images/training_data/labels_intended_vs_executed.png` | How strong the teacher is per letter; why intended-plan labels beat executed pushes |
| 6 Results: training | `figures/curves_all_stages_panels.pdf`; `figures/analysis_budget.pdf` | 5 -> 10 -> 27 shapes on one page; the per-shape budget that fixed the 27-shape plateau |
| 6 Results: final tests | `data/tables/final_headline.tex`; `figures/final_dumbbell.pdf`; `data/tables/final_10shapes.tex` | The pre-registered headline; every shape improves except T and near-T; the ten-shape test on 1000 scenes per shape |
| 6 Results: held-out shapes | `figures/heldout_scaling.pdf`; `data/tables/unseen_capitals.tex` (appendix) | Never-seen shapes improve a little as training shapes grow (1 -> 10 -> 27) |
| 6 Results: remaining failures | `figures/coverage_near_misses.pdf`; `images/last_mile/near_miss_vs_solved_zoom.png` | Most failures end close; the gap between 94.9% and 95.2% is smaller than one camera pixel |
| 6 Results: ablations | `data/tables/label_ab.tex`; `figures/analysis_steps_executed.pdf` | Labels are the largest single gain; executing 12 or 15 of 16 steps is much worse |
| 6 Results: camera and extra rounds | `figures/analysis_192_vs_96.pdf`; `data/tables/round50_vs_44.tex` | 192 px helps on ten shapes but not on 27; six more rounds add +1.9 points |
| 6 Results: T and near-T | `figures/analysis_t_tradeoff_diff.pdf` | States the T and near-T loss with intervals instead of hiding it |
| 7 Limitations and disclosure | `figures/analysis_selection_bias.pdf`, `figures/analysis_plateau_noise.pdf` (appendix) | Validation picks are biased upward and late rounds sit inside chance, which is why only fresh-scene tests carry claims |
| 8 Conclusion | none | Text only |
| Appendix: per-shape detail | `data/tables/final_per_shape_compact.tex`; `data/tables/steps_executed.tex`; `figures/curves_26letters_small_multiples.pdf` | Full per-shape numbers and per-shape training curves for readers who want them |
| Appendix: extra comparisons | `figures/final_forest_shapes.pdf`, `figures/heldout_scaling_per_shape.pdf`, `figures/final_5shapes_bars.pdf` | Paired change per shape; each held-out shape; the five-shape stage tests |
| Appendix: supporting analysis | `figures/final_blends_5shapes.pdf`, `figures/analysis_commitment.pdf`, `figures/coverage_cdf.pdf` | The early blend attempt; committing to more steps hurts teacher and student alike; full coverage distributions |

Swaps if space is short: `figures/final_grouped_bars.pdf` can replace the headline table (it is the slide version),
`figures/final_per_shape_bars.pdf` can replace the dumbbell, and `figures/curves_26letters_groups.pdf` can replace the
budget figure.
