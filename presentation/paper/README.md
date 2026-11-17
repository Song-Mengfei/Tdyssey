# Report (`paper/` folder)

`main.tex` is the final CS5478 course project report: two-column `article`, natbib with `plainnat`, figures from
`../figures/` (mostly the deck graphs in `../figures/deck/` and the TikZ diagrams in `../figures/tikz/`), images from
`../images/`. `refs.bib` holds the references. The main text follows the content and order of the final talk (the
37-slide deck sent to the professor on 2026-11-12) and is written for the professor as an account of what we did,
what worked, what failed and why: the 26 letters (the narrow T is an extra training shape left out of every average),
the main model after 50 correction rounds, nine never-seen shapes, and Flow-DiT (the default 16x16 version) as the
larger network. It answers the professor's questions (why RL fine-tuning, re-ranking and back-propagation failed, in
Section 4, and the red-overlay suggestion, at the end of Section 8.1 and in Appendix K). Everything that is not in the talk lives in
the appendix, with a pointer from the main text where needed.

Outline: 1 Introduction (with the approach figure and Algorithm 1), 2 Related work, 3 Setup (task, shapes, camera
policy network with its diagram, test protocol), 4 Early attempts and why they failed (summary table, RL fine-tuning,
re-ranking, back-propagation, DPPO in the physical simulator, image world models), 5 The state world model (its
parts, its speed against the simulator, what it imagines), 6 The planner teacher, 7 From copying to correction rounds
(copying the teacher, correction rounds, 5 to 10 to 26 shapes), 8 Results (final test, per shape, never-seen shapes,
the last millimetres with the goal-overlay result), 9 A larger network on the same data (Flow-DiT with its diagram),
10 Limitations (every use of privileged information),
11 Conclusion. Appendix: A shapes and settings, B early attempts in detail (RL score table, timeline), C state world
model details (accuracy, examples, timing, error chart), D how the teacher was designed (table of stalls and changes),
E distillation and weight blends, F intended plans as labels and other design tests (77 to 129), G training stages and
how strict the success test is, H more results of the final test (narrow T, digits, 192 px, group bars, per-shape
table, earlier tests), I more tests of the last millimetres (noise steering D058, steps executed D056), J Flow-DiT
variants (Flow-DiT against the main model by group, 8x8, ViT-L D062, pose head D064, joint-attention head pilot D065,
earlier 8x8 tests D059/D060), K making the misalignment visible (goal overlay D066, with the 96 px images, the result
and a reading of the last-millimetre tests together), L compute and a data incident.

Terms (defined in Section 3 and used everywhere): a "step" is one control step of the simulator (0.1 s), optimizer
steps are "training steps", "px" are simulator pixels of the 512 px arena unless written as camera pixels (one 96 px
camera pixel is about 5.3 simulator pixels), "points" are percentage points, "examples" are (observation, label) pairs,
"teacher demonstrations" are teacher episodes on "teacher scenes", a "line" is a sequence of correction rounds from
one start, an "arm" is a branch of the overlay experiment, and plans consist of "targets". One name per thing: the
"original policy" (never the baseline or the released checkpoint in prose; pi_orig in Algorithm 1), the "main model"
(round 50 of the 26-letter stage), the "ten-shape model", the "192 px model", the "state world model" (it "reuses the
simulator's exact pusher rule and contact geometry"; "copy" means imitation only), the "image world model", the
"selection scenes", "development scenes" and the "monitor test". The camera policy is never called a student. Decision
numbers (D0xx) and PR numbers appear only in the appendix. The rendered text says "our final presentation", never
"the talk".

Style rules for edits: plain words and short sentences, no hedging words, and no semicolons, colons or em dashes in
prose, captions and table text (colons only where LaTeX or math needs them, ranges as "4 to 11%"). The team's
writing checklist (`report_assets/writing_checklist.md`, outside Git) adds the rest: no copula avoidance, no trailing
"-ing" clauses, no "X, not Y" endings or slogans, one term per thing, old information before new in each sentence.

## Numbers

Every number from the round-50 test on new scenes (seeds 15,000,000 + 20i, 200 scenes per shape, five policies on the
same scenes: original, ten-shape model, main model round 50, 192 px model, Flow-DiT; decisions D061-D064) is a macro
from `numbers.tex`. `../scripts/paper_numbers.py` writes `numbers.tex` and the tables in `tables/` from the CSV files of
`../scripts/build_round50_data.py`. The numbers of the later experiments (Flow-DiT variants D062, D064, D065 and the
earlier 8x8 tests D059/D060, the goal overlay D066, the near-miss share of round 50 on the final test) are typed into
`../data/paper_late_numbers.json`, each group with its source, and the same script writes them as macros. Numbers that
were final before the round-50 test (validation curves, teacher 90%, 77 to 129, world-model error, copying 30 to 41%,
RL 55% to 5%, data sizes, parameter counts, last-mile RL) are typed into `main.tex` with their source in a comment.

Refresh after the test (from the repository root, no GPU):

```bash
python3 presentation/scripts/build_round50_data.py && python3 presentation/scripts/build_experiment_data.py && \
python3 presentation/scripts/paper_numbers.py
```

Then regenerate the deck graphs the report includes (`final_bars_2`, `final_dumbbell`, `heldout_success`,
`arch_letters`) with `make_figures.py --variant deck` and recompile. The legends of `final_bars_1/2`, `final_dumbbell`,
`wm_error_bar` and `rl_t_collapse` use the report's names (original policy, ten-shape model, main model, 192 px model,
state world model, AdvantageFlow, DiffusionNFT) since the second review round; regenerating them with
`--variant deck --only <name>` reproduces the committed files pixel for pixel. While any shape has fewer than 200 paired scenes,
`numbers.tex` defines `\paperdatanote` as a red line under the abstract; it disappears once the test is complete.
`--data <folder>` reads the CSV files from another folder (to check a running test without writing to `../data/`).

Macro families (one per number, each with a source comment in `numbers.tex`):

| Macros | What |
| --- | --- |
| `\num<Group><Model>` | success %, groups `Head` (all 26 letters), `Early` (9 earlier letters), `New` (17 new letters), `Unseen` (9 never-seen shapes), `T`; models `Orig`, `Ten`, `Main`, `HiRes` (192 px), `Flow` (Flow-DiT) |
| `\numCov<Model>`, `\numCovUnseen<Model>`, `\numPeak<Model>` | mean final coverage (26 letters, never-seen), mean peak coverage (26 letters) |
| `\numDiff<Group><Model>` + `Lo`, `Hi`, `Sig` | paired change against the original, 95% interval, `excludes`/`includes` zero |
| `\numVsMain<Group><Model>` + `Lo`, `Hi`, `Sig` | paired change against the main model (Flow-DiT, 192 px, ten-shape) |
| `\numMainOverTen<Group>` + `Lo`, `Hi`, `Sig` | main model minus the ten-shape model |
| `\numLettersUpMain`, `\numLettersUpFlow`, `\numMainLetterMin(Name)`, `\numMainLetterMax(Name)` | per-letter counts and extremes |
| `\numDigitSeven<Model>`, `\numNarrowT<Model>`, `\numTDirMain`, `\numUnseenTrend` | digit 7, narrow T, wording that follows the data |
| `\numTestScenes`, `\numTestLetterEpisodes`, `\numTestStatus`, `\paperdatanote` | test size and completeness |
| `\numHist...`, `\numSteps...` | final data from `../data/final_tests_by_shape.csv` and `final_tests_episodes.csv` (earlier tests, the D054 paired change of round 50 against round 44, D056 steps test over the 26 letters) |
| `\numOtherOrig`, `\numOtherOrigMax(Name)` | the original policy on the 25 letters other than the T, and its best letter among them |
| `\numNearMissMain`, `\numFlow...`, `\numPrefix...`, `\numOverlay...`, `\numControl...` | later experiments, typed in with sources in `../data/paper_late_numbers.json` (Flow-DiT variants, goal overlay D066, near misses of round 50 on the final test) |

Tables written by the script: `tables/main_test.tex`, `tables/flowdit.tex`, `tables/per_shape.tex` (round-50 test),
`tables/steps.tex` and `tables/history.tex` (final earlier data).

## Build

```bash
cd presentation/paper
latexmk -pdf main.tex                         # TeX Live
tectonic -X compile main.tex --outdir <dir>   # or Tectonic
```

Packages: fontenc, lmodern, microtype, geometry, amsmath, amssymb, graphicx, booktabs, float, array, adjustbox, xcolor, tikz
(arrows.meta, positioning, calc), algorithm, algpseudocode, natbib, hyperref. Overleaf: upload `presentation/` and set
`paper/main.tex` as the main document (or import it from a root file with
`\import{paper/}{main.tex}`).

## State (2026-11-17)

- Revised after three blind reviews and the lead's writing checklist: consistency fixes (selection pools, best-of-16
  wording, the 192 px model's two differences, image-model errors in 96 px camera pixels, one set of overlap numbers
  per named model and case set, the copying figure, "parent" model as in the talk, one finishing tolerance, "main
  model" naming, the DPPO score against the published average 0.84 and best 0.91 of Chi et al. 2023, Table 2).
- Restructured to the talk: Section 5 opens with images to states, the network diagrams of slides 4 and 33 are in
  Sections 3 and 9, accuracy details, the D058 steering test, the Flow-DiT group table and the disclosures of the seed
  incident and compute moved to the appendix. Appendix K has the overlay images and a reading of the last-millimetre
  tests together. Framed as a course project report (no contributions list, short related work).
- Second review round (2026-11-17): figure captions match the plotted values (training history uses the final test),
  the DPPO mean score is defined and set against both published numbers, every pool is named once in Section 3, the
  re-ranking, RL and back-propagation causes are stated as far as the data go, the goal-overlay result moved from
  Limitations to the end of Section 8.1, and the style pass cut most "X, so Y", "therefore" and "which is" sentences.
  An exploratory overlay-off check (the overlay-trained round-11 policy with official images, decided after the D066
  result, macros `\numOverlayOff...`) shows that the policy does not use the red slivers. Section 8.1 and Appendix K
  report it, and Appendices I and K read the last-millimetre tests together in the same way.
  The appendix floats now float (`[!htbp]`) with a `\FloatBarrier` after Appendix A and before Appendix K.
- Compiles with Tectonic 0.15.0 to 24 pages: main text on pages 1 to 12, references to page 14, appendix A to L on
  pages 15 to 24. No undefined references or citations, no overfull boxes. Tectonic reports "TeX
  rerun seems needed, but stopping at 6 passes" because of its main.bbl consistency check (the previous version does
  the same when built into a fresh folder), and all labels and citations resolve.
- References are from memory of the published record (Lipman et al. 2023, Peebles and Xie 2023, Oquab et al. 2024 as
  before). The published PushT numbers (0.91/0.84) were checked against arXiv 2303.04137.
