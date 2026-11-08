# Story outline: talk (10 minutes) and paper

Numbers come from `presentation/data/` (built from the saved test files) and `docs/LEARNING_FINDINGS.md`. Figure,
image and video names refer to `presentation/figures/`, `presentation/images/` and `presentation/videos/`.

## One-sentence story

A camera policy that only knew how to push a T now pushes all 26 capital letters: we never collected new human
demonstrations; a planner that cheats with exact simulator state teaches it, and the camera policy learns to imitate
that planner and then corrects its own mistakes.

## Key numbers (fresh test scenes never used for training or model selection, unless marked otherwise)

| Claim | Number | Source |
| --- | --- | --- |
| Original policy on the 27 trained shapes | 3.7% success | final test, 300 scenes per shape |
| Our 26-letter policy, same camera (96 px) | 33.4% (round 44); 34.8% after six more rounds (round 50) | final test; separate fresh test |
| Our 26-letter policy, sharper camera (192 px) | 33.8% | final test |
| 17 letters added last | 0.4% -> 33.2% | final test |
| Ten earlier shapes | 9.3% -> 33.7% (no loss from adding 17 letters) | final test |
| Six never-seen shapes | 1.2% -> 2.8% (10 shapes) -> 4.7% (27 shapes, 96 px) / 5.9% (192 px); digit 7 2.0% -> 19% (96 px) / 23% (192 px), same scenes | final test |
| Goal covered at the end, 27 shapes | 0.35 -> 0.82 | final test |
| Ten-shape model on its ten shapes | 9.1% -> 34.3% (96 px), 38.2% (192 px) | 1000 scenes per shape |
| Teacher on validation scenes | 1192/1350 = 88% on all 27 shapes vs 35% for the camera policy on the same scenes; 390/500 in the ten-shape labelling run | validation (not fresh) |
| Forward RL (what we tried first) | T success 0.55 -> 0.05 with larger steps; only 16-36% of the training score comes from the plan itself | learning findings, sections 1-2 |
| Intended-plan labels vs executed pushes | 129 vs 77 successes of 1000 (largest single gain) | label A/B (validation) |
| Bigger per-shape budget (54k steps, 60 scenes per new letter) | plateau ~24% -> ~35% | 26-letter run (validation) |
| Executing 12 or 15 of 16 planned steps instead of 8 | 18.6% and 12.2% vs 34.8% | steps test |
| T and near-T after adapting | T 46.0% -> 40.7% (96 px, interval includes 0), 38.3% (192 px, -7.7 [-15.1, -0.2]); near-T 37.3% -> 29.0% (96 px, -8.3 [-15.9, -0.7]). Separate pool (no original run there): round 50 44.0% vs round 44 38.0% on T | final tests |

## Talk plan (10 minutes; matches slides/main.tex and the course's required parts)

The course asks for: problem and motivation; approach and implementation; results and demonstration (with a brief
demo or recorded video); discussion and conclusions; team number and all members' names on the title slide. About
2.5 minutes are left for the image world-model path, presented by the teammates who built it (placeholder slides).

| # | Part | Slide | Time | Visual (builds = overlay steps) |
| --- | --- | --- | --- | --- |
| 1 | Title | Team 28: Liang Wenzhong, Song Mengfei, Su Shenyi | 0:15 | 33-shape gallery |
| 2 | Problem and motivation | A pushing policy that only knows the T | 0:30 | original vs ours on P (build), 3.7% |
| 3 | Problem and motivation | The question | 0:15 | one question, the 27 trained shapes |
| 4 | Approach | Approach at a glance | 0:45 | Figure 1 (`figures/tikz/overview.tex`), builds in 6 steps |
| 5 | Approach | First attempt: RL on the simulator | 0:25 | RL chart, 55% -> 5%, 16-36% |
| 6 | Approach | How the world model works | 0:30 | `figures/tikz/worldmodel.tex` (4 steps), 0.2-0.3 px vs 2.3 px |
| 7 | Approach | How the teacher plans | 0:35 | `figures/tikz/planner.tex` (6 steps), 88% |
| 8 | Approach | Teaching the camera policy: correction rounds | 0:25 | TikZ loop + correction-round examples |
| 9 | Approach | Copy what the teacher intends, not what it did | 0:20 | intended vs executed labels, 77 -> 129 (build) |
| 10 | Results | Training: 5 -> 10 -> 27 shapes | 0:25 | training curves (build per stage) |
| 11 | Results | 26 letters: training per shape | 0:15 | budget curve |
| 12 | Results | Final test: 300 new scenes per shape | 0:25 | 3.7% -> 33.4% |
| 13 | Results | Final test by shape group | 0:15 | bars (build: original, then ours) |
| 14 | Results | Per shape: the new letters improve most | 0:15 | dumbbell; T and narrow T drop 5-8 points |
| 15 | Demo | Same scene, original vs ours | 0:30 | A, M, S grid; play the 40 s demo video |
| 16 | Demo | The teacher and the policy it taught | 0:15 | K: teacher vs policy, 88% vs 35% |
| 17 | Demo | One scene through the whole project | 0:15 | E across six models |
| 18 | Results | Never-seen shapes | 0:15 | held-out chart, digit 7 |
| 19 | Results | The last millimetres | 0:20 | near miss, then close-ups (build) |
| 20-21 | Image world-model path | placeholders for the teammates | 2:30 | TODO (PRs #6, #7, #9-#11) |
| 22 | Discussion and conclusions | What we achieved | 0:15 | three stat tiles |
| 23 | Discussion and conclusions | Takeaways and limitations | 0:25 | three takeaways, limitations panel |
| 24 | Close | Thank you | 0:10 | 33-shape grid, code and model links |

Slide wording avoids internal names: "our policy" (the headline test used the model after 44 correction rounds; the
videos show it after 6 more), "narrow T" for near-T, "new test scenes" for fresh pools, "selection scenes" for the 50
validation scenes per shape used to pick models.

Backup (18 slides): test protocol and headline table, per-shape table, what mattered, simulator re-ranking, weight
blends, the world model's accuracy, imagined vs simulator videos, round 50 vs 44 and 192 vs 96 px, T and narrow T, ten
shapes, ten-shape policies on unseen capitals, teacher vs policy per letter, how the teacher was fixed, every model we
trained, how to read the videos, full disclosure, data and compute, timeline.

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
- *Did T get worse?* Yes, by 5-8 points: T 46.0% -> 40.7% at 96 px (interval includes zero) and -7.7 points at
  192 px; near-T -8.3 points at 96 px (both intervals exclude zero). On a separate fresh pool, round 50 scored 44.0% on
  T against round 44's 38.0% (the original was not run on that pool).
- *Why is success only about 35%?* Blocks usually end close (82% coverage on average, 91% at the best moment), but
  success needs 95%: the last millimetres of alignment are the open problem (one 96 px camera pixel is about 5
  simulator pixels).
- *How much compute?* About 120 GPU-hours of policy training over 128 rounds (single H100 or A100 GPUs), plus
  practice scenes and validation (about 36 h logged for the 27-shape rounds) and the teacher's demonstrations.

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
| 6 Results: final tests | `data/tables/final_headline.tex`; `figures/final_dumbbell.pdf`; `data/tables/final_10shapes.tex` | The pre-registered headline; every trained shape improves except T and near-T; the ten-shape test on 1000 scenes per shape |
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
