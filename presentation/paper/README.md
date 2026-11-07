# Paper (draft skeleton)

`main.tex` is the course paper, due one week after the talk: two-column `article`, natbib with `plainnat`, figures
from `../figures/`, images from `../images/` and tables from `../data/tables/` (nothing is copied into this folder).
`refs.bib` holds the references. The outline follows `../STORY.md` ("Paper outline"); the figure plan is at the end of
that file.

## Build

**Not compiled yet.** The training machine has no LaTeX. Only static checks have been run (see "What was checked").

Local (TeX Live):

```bash
cd presentation/paper
latexmk -pdf main.tex        # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Overleaf: upload the whole `presentation/` folder **without `videos/`** (and without `scripts/__pycache__/`).

- **Option A:** set `paper/main.tex` as the main document (Menu > Main document). This works if Overleaf compiles in
  the main file's folder.
- **Option B (Overleaf's recommended setup):** if figures, tables or references are not found, add a file in the
  project root with a unique name, for example `paper-root.tex` (do **not** call it `main.tex`: a root `main.tex` would
  be compiled instead of `slides/main.tex` and `paper/main.tex`), and set it as the main document:

  ```latex
  \RequirePackage{import}
  \import{paper/}{main.tex}
  ```

  `\import` makes the `../figures/...` and `../data/tables/...` paths resolve from `paper/`. The bibliography line
  already switches to `paper/refs` when it is compiled from the root.

Packages used (all in TeX Live and on Overleaf): fontenc, lmodern, microtype, geometry, amsmath, amssymb, graphicx,
booktabs, adjustbox, xcolor, tikz (arrows.meta, positioning), natbib, hyperref.

## What was checked (static, 2026-11-08)

- Every `\includegraphics` and `\input` path exists (25 graphics, 7 tables).
- Every `\cite` key is in `refs.bib` (39 keys, none unused, no duplicates); every `\ref` has a `\label`.
- `\begin`/`\end` environments, braces and `$` signs balance; no non-ASCII characters.
- Not checked: actual compilation, float placement, page count, overfull boxes, how the TikZ diagram looks.

## Draft vs TODO

Red `[TODO: ...]` markers in the PDF show every open item; `grep -n 'todo{' main.tex` lists them.

| Section | State |
| --- | --- |
| Abstract | Drafted; final pass once the page budget is known |
| 1 Introduction | Drafted (problem, rules, recipe, contributions); tighten |
| 2 Related work | Drafted, five paragraphs; add one sentence on action chunking; decide on WiSE-FT and PPO mentions |
| 3 Setup | Drafted (task, success rule, shapes, policy, scene pools table); add architecture sentence and the data-hygiene incident |
| 4 What did not work | Drafted (forward RL, best-of-N, BPTT, blending); fill two one-sentence TODOs; may move to appendix |
| 5 Method | Drafted; fill planner settings, learning rate, batch size, demo counts, rounds per stage |
| 6 Results | Drafted with numbers (training, 27-shape test, earlier stages, held-out, failures, ablations, T trade-off) |
| 7 Limitations and disclosure | Drafted; add compute cost and the list of earlier pre-registered blend tests |
| 8 Conclusion | Drafted; add next steps and links if allowed |
| Appendix | Figures and tables placed; trim to the course limit; add an experiment-log summary |

Also still open:

- **Authors and date** (`\author`, `\date`): placeholders on purpose.
- **Page limit**: unknown; the draft has 21 floats in the main text (16 figures, 5 tables) plus an appendix. Cut floats first
  (candidates: `fig:camera`, `fig:wm`, `fig:192`, `tab:r50`, `tab:ten`).
- **Figure size**: the figures in `../figures/` are drawn for slides, so text gets small at column width. Add a paper
  style (about 3.4 in wide, 8 pt fonts) to `../scripts/make_figures.py` and regenerate, or use the `figure*` versions.
- **New images**: `../images/world_model/`, `../images/training_data/` and `../images/last_mile/` appeared on 2026-11-08
  (made by another session). If they are renamed or removed, fix the three paths in `main.tex`.
- **Planner count**: the slides say "about 1,150 candidate pushes per step"; `mpc_teacher.py` defaults are 256 samples
  x 4 CEM iterations, with 128 of the first iteration's samples replaced by push candidates (about 1,000 plans scored
  per control). Settle the number before writing it.
- **References**: entries with `note = {TODO: verify ...}` print that note in the bibliography until it is removed:
  `zheng2026diffusionnft` (ICLR 2026 venue), `qi2025gpc` (RA-L volume and year; earlier arXiv title),
  `liao2026supermimic` (CoRL 2026 proceedings), `octo2024` (RSS 2024 venue), `kim2024openvla` (CoRL 2024 venue),
  `black2025pi0` (RSS 2025 venue). Citations placed inside `\todo{...}` (PPO, ResNet, U-Net, WiSE-FT) also create
  bibliography entries; drop them if the sentence is cut.

## Numbers to re-verify before submission

Check each against `../data/` (or the doc named) after the last data rebuild. "Paired" rows are in
`final_tests_paired.csv` (columns `test`, `model`, `group`, `difference`, `ci95_low`, `ci95_high`).

| # | Claim in the paper | Value | Source |
| --- | --- | --- | --- |
| 1 | Original, 27 trained shapes | 3.7% | `tables/final_headline.tex`; `final_tests_by_shape.csv` test D051 |
| 2 | 27-shape student 96 px (round 44), change | 33.4%, +29.8 [+28.7, +30.9] | `final_headline.tex`; paired D051, group "27 trained shapes" |
| 3 | 27-shape student 192 px (round 2), change | 33.8%, +30.1 [+29.0, +31.2] | same |
| 4 | Ten earlier shapes / 17 newer letters | 9.3 -> 33.0 / 33.7 / 34.2; 0.4 -> 6.4 / 33.2 / 33.5 | `final_headline.tex` |
| 5 | Held-out success and change | 1.2 / 2.8 / 4.7 / 5.9; +3.5 [+2.4, +4.6], +4.7 [+3.5, +5.9] | `final_headline.tex`; paired D051 "6 held-out shapes" |
| 6 | Digit 7; other held-out shapes | 2.0 -> 19.0 / 23.0; digit 3, digit 4, plus, arrow at or below 3.0 | `tables/final_per_shape.tex` |
| 7 | Final coverage, 27 shapes / held-out | 0.35 -> 0.82 (0.81 at 192 px) / 0.43 -> 0.56, 0.58 | `final_headline.tex`; by-shape CSV `mean_final_coverage` |
| 8 | Mean peak coverage, 27 shapes, 96 px | 0.91 (0.907) | mean of `mean_peak_coverage`, D051, 26-letter 96 px, 27 shapes |
| 9 | Round 50 vs 44 | 32.9 -> 34.8, +1.9 [+0.1, +3.6] | `tables/round50_vs_44.tex`; paired D054 |
| 10 | Ten-shape test | 9.1 -> 34.3 / 38.2; T +0.7 [-3.6, +5.0]; near-T -7.8 [-12.0, -3.6] | `tables/final_10shapes.tex` |
| 11 | 192 vs 96 px, ten shapes | +3.9 [+2.6, +5.2] over 10,000 paired scenes | LEARNING_FINDINGS section 7; `figures/analysis_192_vs_96` |
| 12 | 192 vs 96 px, 27 shapes | +0.3 [-1.1, +1.8] | computed for this draft from `final_tests_episodes.csv` (D051, paired by `shape`, `scene_seed`); check against `analysis_192_vs_96` |
| 13 | Ten-shape students on 17 untrained capitals | 0.2 -> 6.6 / 10.6 | `tables/unseen_capitals.tex`; paired D052 |
| 14 | T and near-T, final test | T 46.0 -> 40.7 (-5.3 [-13.5, +2.9]) / 38.3 (-7.7 [-15.1, -0.2]); near-T 37.3 -> 29.0 (-8.3 [-15.9, -0.7]) / 32.3 (-5.0 [-12.2, +2.2]) | paired D051, groups `T`, `T_narrow` |
| 15 | T mean final coverage | 0.73 -> 0.86 | `final_tests_episodes.csv`, D051, shape T |
| 16 | Round 50 vs 44 on T | 44.0 vs 38.0 (200 scenes) | paired D054, group `T` |
| 17 | Steps executed 8 / 12 / 15 | 34.8 / 18.6 / 12.2 | `tables/steps_executed.tex` |
| 18 | Five-shape tests | round 4: 18.5 -> 32.2; round 7: 18.6 -> 34.2; Y, D, L to 29-39% | paired D046, D047 "5 first shapes"; by-shape CSV |
| 19 | Original T and near-T across pools (intro) | T 42-46%, near-T 37-42% | by-shape CSV, all tests' `original` rows |
| 20 | Label and design A/B tests | 77 vs 129; 111 vs 129; 104 vs 111; 138 vs 146 (of 1000) | typed in: LEARNING_FINDINGS section 7; `tables/label_ab.tex` |
| 21 | Teacher | 390/500 (ten shapes); 36-49/50 (17 letters); T 5 -> 34, L 11 -> 39 with state noise; 303/390 = 78% when committing to 8 | LEARNING_FINDINGS sections 5 and 7; DECISIONS D048 |
| 22 | Budget | 190k vs 512k samples per shape per round; 5-8x fewer corrections; ~24% -> ~35% validation; picks 470 and 511 of 1350 | DECISIONS D053, D051 outcome |
| 23 | Plateau noise | 179/500 mean, spread 9.6 vs 10.7 chance; 211/500 flips | LEARNING_FINDINGS section 7 |
| 24 | Forward RL | T 0.55 -> 0.05 by update 75; 1e-4 collapsed by update 25; dense >= 1e-5 ends at T 0.00-0.30; plan share 16-36%; split-half 0.14-0.29 | LEARNING_FINDINGS sections 1-2; `typed_in_numbers.json` |
| 25 | Best-of-N; BPTT | Y 3/20 -> 8/20, sample spread 0.02; gap 21 -> 3.7 px, coverage 0.04-0.12 after 500+ iterations | LEARNING_FINDINGS sections 3 and 5 |
| 26 | Blend result | Y 8.6 -> 14.8%, D 0.3 -> 1.1% on 2000 scenes; replicated | DECISIONS D042, D043 |
| 27 | World model | 1500 episodes, 431k controls; median 8-control error 0.22-0.32 px; pusher match 1e-4 px | LEARNING_FINDINGS section 4 |
| 28 | Alignment tolerance | 1.3-2.7 px = 0.24-0.51 of a 96 px camera pixel | DECISIONS D048 |
| 29 | Label jumps (Fig. labels) | executed 59-153 px vs intended 20-47 px | panel titles of `images/training_data/labels_intended_vs_executed.png` |
| 30 | Scene pools table | seeds and scene counts per test | DECISIONS D040-D056 |

Snippet for rows 8, 12 and 15 (standard library only):

```bash
python3 -I - <<'EOF'
import csv, math
ep = [r for r in csv.DictReader(open('presentation/data/final_tests_episodes.csv')) if r['test'] == 'D051']
held = {'digit3', 'digit4', 'digit7', 'plus', 'hash', 'arrow'}
by = {}
for r in ep: by.setdefault(r['model'], {})[(r['shape'], r['scene_seed'])] = r
a, b = by['26-letter 96 px (round 44)'], by['26-letter 192 px (round 2)']
d = [float(b[k]['success']) - float(a[k]['success']) for k in a if k[0] not in held]
m = sum(d) / len(d); h = 1.96 * math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
print('192-96, 27 shapes: %+.1f [%+.1f, %+.1f]' % (100 * m, 100 * (m - h), 100 * (m + h)))
for name, rows in by.items():
    t = [float(r['final_coverage']) for k, r in rows.items() if k[0] == 'T']
    print(name, 'T final coverage %.2f' % (sum(t) / len(t)))
EOF
```
