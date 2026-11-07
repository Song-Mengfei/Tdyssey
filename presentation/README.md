# Presentation materials

Everything for the talk and the paper: slides (LaTeX Beamer), graphs, images, demo videos of every model we trained,
the data behind each graph, and the scripts that rebuild them. Most of the trained models are not public, so this
folder is how the whole team can see and show them.

| Folder | What is in it |
| --- | --- |
| [`STORY.md`](STORY.md) | Talk plan (10 minutes, slide by slide with timing and visuals), paper outline, key numbers, likely questions |
| [`slides/`](slides/) | `main.tex`: the Beamer deck (16:9). Build with `pdflatex main.tex` (twice) in `slides/`, or on Overleaf (upload the whole `presentation/` folder, main file `slides/main.tex`) |
| [`figures/`](figures/README.md) | 72 graphs, each as PDF (Beamer) and PNG (PowerPoint); [`figures/README.md`](figures/README.md) lists what each shows and a suggested slide |
| `images/` | Shape galleries, camera views, last-frame grids for every model, filmstrips (five moments of one episode) |
| [`videos/`](videos/INDEX.md) | Demo videos; [`videos/INDEX.md`](videos/INDEX.md) lists every clip with its outcome; `videos/posters/` has a still of each (its last frame) |
| `data/` | CSV tables behind the graphs (every validation round of every stage, every final test per shape and per episode, paired differences) and `data/tables/*.tex` (booktabs tables for `\input`) |
| `scripts/` | `build_data.py` (CSV tables from the saved test files), `make_figures.py` (all graphs), `make_videos.py` (episodes, videos, images) |

## Videos

| Folder | Clips |
| --- | --- |
| `videos/models/` | Every model we trained, on the same 9 shapes and scene (T, near-T, Y, D, L, E, K, R, digit 7): the original policy, forward RL after it degraded, the 5-shape stage (copying the teacher; correction rounds 4 and 7), the mid-project encoder blend, the 10-shape stage (executed-push labels; final 96 px; 192 px) and the 26-letter stage (rounds 0, 13, 44, 50; 192 px) |
| `videos/all_shapes/` | All 33 shapes at once for the original, 5-shape round 7, 10-shape 96 px, 26-letter round 50 and 26-letter 192 px models (two scenes each) |
| `videos/compare/` | Original vs our main model side by side, every shape, two scenes |
| `videos/progression/` | One scene through the project: original, 5 shapes, 10 shapes, 26 letters rounds 0, 13 and 50 |
| `videos/teacher/` | The planner teacher (reads exact state; training data only), with its planned path |
| `videos/failures/` | Near misses (over 90% covered at some point but not solved) and the never-seen shapes |

The clips are examples from the validation scenes (seeds 400000 and 400020), recorded with the standard 100-step
sampler. Success rates come from the tests in `data/`, not from these clips; `videos/INDEX.md` says which clips
succeeded so you can choose honestly (for instance, show one success and one failure).

In every clip: green outline = goal, grey = block, blue dot = pusher, red dots = the next targets the policy will
execute, pink dots = the rest of its 16-step plan, top-left inset = the camera image the policy actually sees.

## Rebuild

```bash
# from the repository root
python3 presentation/scripts/build_data.py --raw <folder with the run store's runs/>        # tables (no GPU)
MPLBACKEND=Agg .venv/bin/python presentation/scripts/make_figures.py                         # graphs (no GPU)
.venv/bin/python -B presentation/scripts/make_videos.py rollouts                             # episodes (GPU, checkpoints)
.venv/bin/python -B presentation/scripts/make_videos.py teacher                              # teacher episodes (GPU)
.venv/bin/python -B presentation/scripts/make_videos.py stills                               # galleries and camera views
.venv/bin/python -B presentation/scripts/make_videos.py compose                              # videos, images, stills
```

The video and data scripts need the run store's checkpoints and test files (kept outside Git); everything they produce
is already in this folder.

## Not here yet

- A screen recording of the interactive demo (needs a browser): record it from the Hugging Face Space and add it to
  `videos/`.
- The slides have not been compiled on the training machine (no LaTeX there); compile on Overleaf or locally and fix
  anything that breaks.
