# Paper

`main.tex` is the course paper: two-column `article`, natbib with `plainnat`, figures from `../figures/` (mostly the
deck graphs in `../figures/deck/` and the TikZ diagrams in `../figures/tikz/`), images from `../images/`. `refs.bib`
holds the references. The text follows the current talk deck (`../slides/main.tex`): the 26 letters (the narrow T is an
extra training shape left out of every aggregate), the main model after 50 correction rounds, nine never-seen shapes,
and Flow-DiT (the 16x16 version) as the architecture experiment.

Style rules for edits: plain words and short sentences, no hedging words, and no semicolons, colons or em dashes in
prose, captions and table text (colons only where LaTeX or math needs them, ranges as "4 to 11%").

## Numbers

Every number from the round-50 test on new scenes (seeds 15,000,000 + 20i, 200 scenes per shape, five policies on the
same scenes: original, ten-shape model, main model round 50, 192 px model, Flow-DiT; decisions D061-D064) is a macro
from `numbers.tex`. `../scripts/paper_numbers.py` writes `numbers.tex` and the tables in `tables/` from the CSV files of
`../scripts/build_round50_data.py`. Numbers that were final before that test (validation curves, teacher 90%, 77 to
129, world-model error, copying 30 to 41%, RL 55% to 5%, data sizes, parameter counts, last-mile RL) are typed into
`main.tex` with their source in a comment.

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

Tables written by the script: `tables/main_test.tex`, `tables/flowdit.tex`, `tables/per_shape.tex` (round-50 test),
`tables/steps.tex` and `tables/history.tex` (final earlier data).

## Build

```bash
cd presentation/paper
latexmk -pdf main.tex                         # TeX Live
tectonic -X compile main.tex --outdir <dir>   # or Tectonic
```

Packages: fontenc, lmodern, microtype, geometry, amsmath, amssymb, graphicx, booktabs, adjustbox, xcolor, tikz
(arrows.meta, positioning, calc), algorithm, algpseudocode, natbib, hyperref. Overleaf: upload `presentation/` without
`videos/` and set `paper/main.tex` as the main document (or import it from a root file with
`\import{paper/}{main.tex}`).

## State (2026-11-12)

- Compiles with Tectonic to 15 pages (about 10 pages of text, 2 of references, 3 of appendix), no overfull boxes, no
  undefined references or citations. Checked with a partial test (25 of 200 scenes per shape), so every round-50 test
  number and the red note are placeholders until the refresh above.
- `final_bars_2`, `final_dumbbell` and `heldout_success` in `../figures/deck/` still show the earlier round-44 test
  until they are regenerated; `arch_letters` does not exist yet, so the paper shows a framed note in its place.
- The validation-curve figure (`curves_stages_3`) marks an earlier test (original 2.4%, round 50 34.7%); its caption
  says so.
- New references (from memory of the published record): Lipman et al. 2023 (flow matching), Peebles and Xie 2023
  (DiT), Oquab et al. 2024 (DINOv2).
