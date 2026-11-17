# Course report

`main.tex` is our CS5478 final project report, a two-column `article` of 24 pages. The main text runs to page 12,
the references to page 14 and the appendix from page 15. The report is submitted on its own, so the rendered text
names no repository files and no internal labels. The only links are the code and the live demo. The main text
follows the order of our final presentation (`../talk/`), and everything else is in the appendix.

## Contents

| Part | What it covers |
| --- | --- |
| 1 Introduction | The question, the four-part method (Fig. 1, Algorithm 1) and the headline results |
| 2 Related work | Diffusion Policy, flow matching, RL for diffusion policies, privileged teachers, world models |
| 3 Setup | Task and simulator, shapes, the camera policy network (Fig. 3), test protocol and scene pools |
| 4 Early attempts | Summary table, RL fine-tuning, re-ranking, back-propagation, DPPO, image world models |
| 5 State world model | Its parts, its speed against the simulator, how far its predictions drift |
| 6 Planner teacher | The sampling planner (Algorithm 2) and its 90% on the selection scenes |
| 7 Copying to correction rounds | Copying the teacher, correction rounds, the three training stages |
| 8 Results | Final test, per letter, never-seen shapes, the last millimetres and the goal overlay |
| 9 Larger network | Flow-DiT on the same data |
| 10 Limitations | Every use of privileged information, single runs |
| 11 Conclusion | Summary and the code link |
| A | Shapes and all training and planning settings (Table 3) |
| B | Early attempts in detail, with the RL signal table and the time line |
| C | State world model accuracy, example windows and timing |
| D | How the teacher was designed, stall by stall |
| E | Distillation and weight blends |
| F | Intended plans as labels and the other matched design tests |
| G | Training per shape, noise of the selection scenes, how strict the success test is |
| H | More final-test results (digits 2, 5 and 8, the 192 px model, group bars, every shape, earlier tests) |
| I | Noise steering and the number of targets executed per plan |
| J | Flow-DiT variants (8x8 tokens, ViT-L encoder, pose head, joint-attention head) |
| K | The goal overlay (design, training, result, overlay switched off) |
| L | Compute and a test-seed incident |

## Build

```bash
cd presentation/paper
latexmk -pdf main.tex                          # TeX Live
tectonic -X compile main.tex --outdir build    # or Tectonic
```

The report reads `../figures/` and `../images/`, so on Overleaf upload the whole `presentation/` folder and set
`paper/main.tex` as the main document. Packages are listed at the top of `main.tex`.

## Where the numbers come from

- **Final test.** Every number of the final test (five policies on the same 200 new scenes per shape, seeds
  15,000,000 + 20i) is a macro such as `\numHeadMain` in `numbers.tex`. `../scripts/paper_numbers.py` writes
  `numbers.tex` and the five tables in `tables/` from the CSV files in `../data/` (`d064_test_*.csv` for the final
  test, `final_tests_by_shape.csv` and `final_tests_episodes.csv` for the earlier tests and the steps test).
- **Later experiments.** The Flow-DiT variants, the goal overlay and the near-miss share are typed into
  `../data/paper_late_numbers.json`, one group per experiment with its source. The same script turns each entry into
  a macro.
- **Earlier results.** Numbers that were final before the final test (training curves, the teacher's 90%, the label
  test 77 to 129, world-model error, copying, RL, data sizes, parameter counts) are typed into `main.tex`, each with
  its source in a LaTeX comment.
- **Graphs.** The PDFs in `../figures/deck/` come from `../scripts/make_figures.py` and the same data folder.

Every macro in `numbers.tex` carries a comment that names its source. The data files also hold an extra training
shape (`T_narrow`). The report leaves it out, and the script keeps it out of every group, table and macro. The
five-shape and ten-shape stage names count it.

## Changing a number safely

1. Never edit `numbers.tex` or `tables/` by hand. The script overwrites them.
2. Change the CSV or JSON file in `../data/` and keep its source field current. Change `paper_numbers.py` only for
   labels or a new macro.
3. From the repository root run `python3 presentation/scripts/paper_numbers.py` (standard library only).
4. Check that `git diff presentation/paper/numbers.tex presentation/paper/tables/` shows only the intended change.
5. Recompile and search the text for the old value with `pdftotext -layout build/main.pdf - | grep`. A few numbers
   are also written out in prose (12.7 in Section 9 repeats `\numVsMainHeadFlow`), and words such as "includes" or
   "excludes" zero come from macros (`\num...Sig`), so reread those sentences.
6. If a graph shows the number, regenerate it with `make_figures.py --variant deck --only <name>` (needs matplotlib).

## Writing rules

Plain words, short sentences and no hedging. No semicolons, colons or em dashes in prose, captions or table text,
and ranges as "4 to 11%". Use one name per thing, namely the original policy, the main model (round 50 of the
26-letter stage), the ten-shape model, the 192 px model, Flow-DiT, the state world model, the image world model, the
selection scenes, the development scenes and the final test. A step is one 0.1 s control step, optimizer updates are training steps,
px are simulator pixels unless the text says camera pixels, and points are percentage points. Keep the rendered text
free of repository file names and internal labels.
