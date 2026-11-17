# Presentation and course report

The final talk and the course report for the CS5478 project, with the data and scripts behind every number and figure
in the report.

| Folder | What is in it |
| --- | --- |
| [`paper/`](paper/README.md) | The course report (`main.tex`, `refs.bib`, generated `numbers.tex` and `tables/`) |
| `talk/` | `tdyssey-talk.pptx`: the final 37-slide deck as presented, with its demo videos embedded |
| `figures/` | The graphs (`deck/`, PDF) and the TikZ diagrams (`tikz/`) that the report includes |
| `images/` | The images the report includes: shape gallery, intended-plan labels, image-model frames and near-miss close-ups |
| `data/` | The tables every number and graph comes from, built from the saved test files: final tests per shape and per episode, paired differences with 95% intervals, validation rounds, early attempts, world-model accuracy and the later experiments (`paper_late_numbers.json`, with a source for each number) |
| `scripts/` | `build_data.py` and `build_round50_data.py` (the CSV tables), `paper_numbers.py` (`paper/numbers.tex` and `paper/tables/`), `make_figures.py` (graphs), `make_videos.py` and `slide_images.py` (rollout images and crops), `build_experiment_data.py` (architecture experiments) |

All success rates in the report come from tests on new scenes. The final test uses 200 scenes per shape (seeds
15000000 + 20*i). Selection scenes picked the models and are not reported as results.

The demo videos, their stills, the Beamer version of the slides and the PowerPoint generator were used to make the talk
and are not kept on main. They remain in the history of PR #12. The live demo is the
[Hugging Face Space](https://tinyurl.com/tdyssey).

## Rebuilding

```bash
python3 presentation/scripts/paper_numbers.py        # numbers.tex and tables/ from data/ (Python 3 only)
MPLBACKEND=Agg python presentation/scripts/make_figures.py --variant deck --only <name>   # one graph
cd presentation/paper && latexmk -pdf main.tex       # or: tectonic -X compile main.tex --outdir build
```

`make_figures.py` needs matplotlib from the project environment. `make_videos.py` also needs the simulator and the
checkpoints.
