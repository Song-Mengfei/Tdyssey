# Presentation materials

Everything for the talk and the paper: slides (LaTeX Beamer), graphs, images, demo videos of every model we trained,
the data behind each graph, and the scripts that rebuild them. Most of the trained models are not public, so this
folder is how the whole team can see and show them.

| Folder | What is in it |
| --- | --- |
| [`STORY.md`](STORY.md) | Talk plan (10 minutes, slide by slide with timing and visuals), paper outline, key numbers, likely questions |
| [`slides/`](slides/) | `main.tex`: the Beamer deck (16:9). Build with `pdflatex main.tex` (twice) in `slides/`, or on Overleaf (see below) |
| [`figures/`](figures/README.md) | 72 graphs, each as PDF (Beamer) and PNG (PowerPoint); [`figures/README.md`](figures/README.md) lists what each shows and a suggested slide |
| `images/` | Shape galleries, camera views, last-frame grids for every model, filmstrips (five moments of one episode) |
| [`videos/`](videos/INDEX.md) | Demo videos; [`videos/INDEX.md`](videos/INDEX.md) lists every clip with its outcome; `videos/posters/` has a still of each (its last frame) |
| `data/` | Everything the graphs are drawn from, so anyone can restyle them: `validation_rounds.csv` (every validation round of every stage, per shape), `final_tests_by_shape.csv`, `final_tests_paired.csv` (differences with 95% intervals), `final_tests_episodes.csv` (every final-test episode), `training_rounds.csv` (training examples by source and loss per round), `typed_in_numbers.json` (the few numbers taken from the docs, with sources) and `tables/*.tex` (booktabs tables for `\input`) |
| `scripts/` | `build_data.py` (CSV tables from the saved test files), `make_figures.py` (all graphs), `make_videos.py` (episodes, videos, images) |

## Which model is which

| Name in figures and videos | Model | Tested in |
| --- | --- | --- |
| Original | the official RSS 2023 checkpoint (human T demonstrations only) | every test |
| 10-shape model (96 px / 192 px) | after the ten-shape correction rounds | ten-shape final test (1000 scenes per shape) |
| "Ours, 26 letters" in the final-test figures | 26-letter 96 px, **round 44** (the pre-registered pick) | headline final test (300 scenes per shape) |
| "Ours, 26 letters, 192 px" | 26-letter 192 px, round 2 | headline final test |
| "main model" / "round 50" in the videos | 26-letter 96 px, **round 50** (six more rounds) | separate fresh test against round 44 (+1.9 points) |
| Teacher | the planner with exact simulator state (training data only) | validation scenes |

## Videos

Playback: one frame per control at 30 fps, i.e. **3x real time** (the simulator runs 10 controls per second); the
world-model clips run at half that speed. Tiles are 384 px (side-by-side clips) or 160-256 px (grids). Legend card:
`images/video_legend.png` (green outline = goal, grey = block, blue dot = pusher, red dots = the next targets the policy
will execute, pink dots = the rest of its 16-step plan, top-left inset = the camera image the policy sees).

| Folder | Clips |
| --- | --- |
| `videos/models/` | Every model we trained, on the same 9 shapes and scene (T, near-T, Y, D, L, E, K, R, digit 7): original, forward RL after it degraded, 5-shape stage (copying the teacher; correction rounds 4 and 7), mid-project encoder blend, 10-shape stage (executed-push labels; final 96 px; 192 px), 26-letter stage (rounds 0, 13, 44, 50; 192 px) |
| `videos/all_shapes/` | All 33 shapes at once (6x6 grids, and `_wide` 11x3 grids that fit 16:9) for the original, 5-shape round 7, 10-shape 96 px, 26-letter round 50 and 192 px models |
| `videos/compare/` | Original vs main model side by side, every shape, two scenes |
| `videos/teacher_vs_student/` | Teacher vs main model on the same scene, all 27 trained shapes |
| `videos/teacher/` | The planner teacher (9-shape and 27-shape grids, one clip per shape) |
| `videos/progression/` | One scene through the project: original, 5 shapes, 10 shapes, 26 letters rounds 0, 13 and 50 |
| `videos/world_model/` | What the world model imagines (red outline) vs the simulator, open-loop over 32 controls |
| `videos/training_data/` | The world model's training data: the original policy with random noise on its targets |
| `videos/near_misses/` | Main model episodes that covered over 90% at some point but did not succeed |
| `videos/held_out/` | Main model on the six never-seen shapes (successes and failures) |

The clips are examples from validation scenes (seeds 400000 and 400020), which were also used to pick checkpoints;
success rates come only from the fresh-scene tests in `data/`. `videos/INDEX.md` gives every clip's outcome so you can
choose honestly (for instance one success and one near miss).

## Overleaf and PowerPoint

- Overleaf limits uploads to about 50 MB, so upload `presentation/` **without** `videos/` except `videos/posters/`
  (about 35 MB): `slides/`, `figures/`, `images/`, `data/tables/`, `videos/posters/`. Set `slides/main.tex` as the main
  file; it finds the figures whether Overleaf compiles from the project root or from `slides/`.
- In the PDF, clicking a video still opens the clip only in local viewers that allow it (for example Acrobat, Okular)
  and only with `videos/` next to `slides/`; it does not work on Overleaf, in browsers or in macOS Preview. In
  PowerPoint, insert the MP4 files directly.
- Large-font versions of the graphs for slides: `figures/slide/` (full width) and `figures/slide_half/` (two per
  slide); the deck uses them automatically when they exist.

## Rebuild

```bash
# from the repository root
python3 presentation/scripts/build_data.py --raw <folder with the run store's runs/>        # tables (no GPU)
MPLBACKEND=Agg .venv/bin/python presentation/scripts/make_figures.py                         # graphs from data/ only (no GPU); edit styles here
.venv/bin/python -B presentation/scripts/make_videos.py rollouts                             # episodes (GPU, checkpoints)
.venv/bin/python -B presentation/scripts/make_videos.py teacher                              # teacher episodes (GPU)
.venv/bin/python -B presentation/scripts/make_videos.py stills                               # galleries and camera views
.venv/bin/python -B presentation/scripts/make_videos.py compose                              # videos, images, stills
```

The video and data scripts need the run store's checkpoints and test files (kept outside Git); everything they produce
is already in this folder.

## Not here yet

- A screen recording and a screenshot of the interactive demo (needs a browser): record them from the Hugging Face
  Space and add them to `videos/` and `images/`; the deck has a placeholder slide.
- The slides have not been compiled on the training machine (no LaTeX there); compile on Overleaf or locally and fix
  anything that breaks.
