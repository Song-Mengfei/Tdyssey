# Paper

`main.tex` is the final course report: two-column `article`, natbib with `plainnat`, figures from `../figures/` (mostly
the deck graphs in `../figures/deck/` and the TikZ diagrams in `../figures/tikz/`), images from `../images/`. `refs.bib`
holds the references. The main text matches the content and order of the final talk (the 37-slide deck sent to the
professor on 2026-11-12): the 26 letters (the narrow T is an extra training shape left out of every aggregate), the
main model after 50 correction rounds, nine never-seen shapes, and Flow-DiT (the default 16x16 version) as the
architecture experiment. It answers the professor's questions (why RL fine-tuning, re-ranking and back-propagation
failed, in Section 4, kept tight, and the red-overlay suggestion, in two sentences in Section 10). Everything that is not in the
talk lives in the appendix, with a one-sentence pointer from the main text where needed.

Outline: 1 Introduction, 2 Related work, 3 Setup (task, shapes, camera policy network, test protocol), 4 Early
attempts and why they failed (RL fine-tuning, re-ranking with the simulator, back-propagation through a learned model,
the team's DPPO run and image world models, with a summary table), 5 From images to states (state world model, what it
imagines, its speed against the simulator), 6 The planner teacher, 7 From copying to correction rounds (copying alone
is not enough, correction rounds, 5 to 10 to 26 shapes), 8 Results (final test, per shape, never-seen shapes, the last
millimetres), 9 Does the network matter (Flow-DiT), 10 Limitations and future work, 11 Conclusion. Appendix: A settings
and networks, B details of the early attempts (RL score table, timeline), C how the teacher was designed, D
distillation and weight blends, E intended plans as labels and other design tests (77 to 129), F stage details and why
the selection scores look noisy, G more results of the final test (narrow T, digits, 192 px, per-shape table, earlier
tests), H more tests of the last millimetres (noise steering D058, steps executed D056), I Flow-DiT variants (8x8,
ViT-L D062, pose head D064, joint-attention head pilot D065, earlier 8x8 tests D059/D060), J making the misalignment
visible (goal overlay D066, result).

Style rules for edits: plain words and short sentences, no hedging words, and no semicolons, colons or em dashes in
prose, captions and table text (colons only where LaTeX or math needs them, ranges as "4 to 11%").

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

Then regenerate the deck graphs the paper includes (`final_bars_2`, `final_dumbbell`, `heldout_success`,
`arch_letters`) with `make_figures.py --variant deck` and recompile. While any shape has fewer than 200 paired scenes,
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
| `\numHist...`, `\numSteps...` | final data from `../data/final_tests_by_shape.csv` and `final_tests_episodes.csv` (earlier tests, D056 steps test over the 26 letters) |
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
(arrows.meta, positioning, calc), algorithm, algpseudocode, natbib, hyperref. Overleaf: upload `presentation/` without
`videos/` and set `paper/main.tex` as the main document (or import it from a root file with
`\import{paper/}{main.tex}`).

## State (2026-11-16)

- Main text matched to the final talk: everything not in the deck moved to clearly titled appendix sections (distillation
  and weight blends, the intended-plan design test, teacher design steps, stage budget and selection-scene noise, the
  192 px and narrow-T details, earlier tests, the last-mile steering and steps-executed tests, the RL score table and
  the timeline of early attempts). Flow-DiT in the main text is the default model only (ViT-B/14, all 16x16 tokens).
- New appendix sections: I Flow-DiT variants (8x8 tokens 44.2% on the final test, ViT-L/14 D062 42.9% and -2.7
  [-4.5, -0.8], pose head D064 +0.2, joint-attention head pilot D065 -7.3 [-10.9, -3.7] on the selection scenes at
  60k steps, earlier 8x8 tests D059/D060), J the goal overlay D066 (whole-goal version collapsed, slivers-only version
  -1.0 [-2.7, +0.8] against its control after 11 rounds, control +0.6 against round 50). Section 10 states the result in
  two sentences. The near-miss share now comes from the final test (63.6% of round-50 failures reach 90% coverage).
- Compiles with Tectonic 0.15.0 to 23 pages: main text on pages 1 to 13, references to page 15, appendix A to J on
  pages 16 to 23. No undefined references or citations, no overfull boxes. Tectonic reports "TeX rerun seems needed,
  but stopping at 6 passes" because of its main.bbl consistency check (the previous version does the same when built
  into a fresh folder), and all labels and citations resolve.
- References are from memory of the published record (Lipman et al. 2023, Peebles and Xie 2023, Oquab et al. 2024 as
  before).
