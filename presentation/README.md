# Presentation materials

Everything for the talk and the paper: slides (LaTeX Beamer), graphs, images, demo videos of every model we trained,
the data behind each graph, and the scripts that rebuild them. Most of the trained models are not public, so this
folder is how the whole team can see and show them.

| Folder | What is in it |
| --- | --- |
| [`STORY.md`](STORY.md) | Talk plan (10 minutes, slide by slide with timing and visuals), paper outline, key numbers, likely questions |
| [`slides/`](slides/) | `main.tex`: the Beamer deck (16:9, 41 main slides and 15 backup; 56 final-build pages by default, optional step-by-step builds; standard LaTeX font). Slides 4-14 are the early-attempt image-model story in `slides/image_world_model_48m.tex`, placed after the opening three frames; its summary is slide 41 in `slides/image_world_model_summary.tex`, matching the current PPTX order. All 17 media slides use the current PPTX posters and reviewed videos; slides4/9/14 are saved-output displays (the two48M clips are discrete snapshots); the earlier 8.73M story remains in `slides/image_world_model.tex` as historical source. `slides/playable_video_appendix.tex` retains three recorded clips with dedicated playback space. Build with `pdflatex main.tex` (twice) in `slides/`, on Overleaf (see below), or download the PDF that GitHub builds on every push (Actions -> Presentation PDFs -> `presentation-pdfs`) |
| [`figures/`](figures/README.md) | Every graph as PDF (Beamer) and PNG (PowerPoint); `figures/deck/` holds the versions drawn at the exact size the deck uses them (including the build steps), `figures/slide/` and `figures/slide_half/` older large-font versions, `figures/tikz/` the diagrams (`overview.tex`: Figure 1, used by the deck and the paper; `worldmodel.tex` and `planner.tex`: how the world model and the teacher work; `architecture.tex`: the architecture experiment); [`figures/README.md`](figures/README.md) lists what each shows |
| `images/` | Shape galleries, camera views, last-frame grids for every model, filmstrips (five moments of one episode); `images/slide/` holds the crops the deck uses (built by `scripts/slide_images.py`) |
| [`videos/`](videos/INDEX.md) | Demo videos; [`videos/INDEX.md`](videos/INDEX.md) lists every clip with its outcome; `videos/posters/` has a still of each (its last frame) |
| `data/` | Everything the graphs are drawn from, so anyone can restyle them: `validation_rounds.csv` (every validation round of every stage, per shape), `final_tests_by_shape.csv`, `final_tests_paired.csv` (differences with 95% intervals), `final_tests_episodes.csv` (every final-test episode), `training_rounds.csv` (training examples by source and loss per round), `typed_in_numbers.json` (the few numbers taken from the docs, with sources) and `tables/*.tex` (booktabs tables for `\input`) |
| [`paper/`](paper/README.md) | The paper draft (`main.tex`, `refs.bib`) and what is still to do |
| `scripts/` | `build_data.py` (CSV tables from the saved test files), `make_figures.py` (all graphs), `make_videos.py` (episodes, videos, images, demo reel), `slide_images.py` (crops for the deck), `build_experiment_data.py` (architecture experiment and last-mile RL tables), `overleaf_zip.py` (small zip for Overleaf) |

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
| `videos/talk/` | `demo_reel.mp4`: a 40-second recording for the talk (legend, original vs ours on P, A, M, S, then all 33 shapes) that plays without PDF links |
| `videos/models/` | Every model we trained, on the same 9 shapes and scene (T, near-T, Y, D, L, E, K, R, digit 7): original, forward RL after it degraded, 5-shape stage (copying the teacher; correction rounds 4 and 7), mid-project encoder blend, 10-shape stage (executed-push labels; final 96 px; 192 px), 26-letter stage (rounds 0, 13, 44, 50; 192 px) |
| `videos/all_shapes/` | All 33 shapes at once (6x6 grids, and `_wide` 11x3 grids that fit 16:9) for the original, 5-shape round 7, 10-shape 96 px, 26-letter round 50 and 192 px models |
| `videos/compare/` | Original vs main model side by side, every shape, two scenes |
| `videos/teacher_vs_student/` | Teacher vs main model on the same scene, all 27 trained shapes |
| `videos/teacher/` | The planner teacher (9-shape and 27-shape grids, one clip per shape) |
| `videos/progression/` | One scene through the project: original, 5 shapes, 10 shapes, 26 letters rounds 0, 13 and 50 |
| `videos/world_model/` | What the world model imagines (red outline) vs the simulator, open-loop over 32 controls |
| `videos/training_data/` | The world model's training data: the original policy with 0/10/25 px random noise on its targets (the clip adds a 50 px row for illustration only) |
| `videos/near_misses/` | Main model episodes that covered over 90% at some point but did not succeed |
| `videos/held_out/` | Main model on the six never-seen shapes (successes and failures) |

The clips are examples from validation scenes (seeds 400000 and 400020), which were also used to pick checkpoints;
success rates come only from the fresh-scene tests in `data/`. `videos/INDEX.md` gives every clip's outcome so you can
choose honestly (for instance one success and one near miss).

## Current PPTX, LaTeX and video playback

The editable source now follows the reviewed 62-slide offline PPTX: the image-model
section is at 4-14, the closing slide is 40, its image-model summary is 41, and the 21
backup slides follow. Printed counters and progress bars follow the current order.
The source PPTX SHA256,17 exact MP4/poster hashes, bounding boxes and scientific
qualifications are in [`data/pptx_slide_media.json`](data/pptx_slide_media.json).
The text, tables and TikZ diagrams remain editable; media posters are extracted
from the PPTX without recompression. No training, simulation or inference was run.

- Compile `slides/main.tex` with `pdflatex` twice, or use `latexmk -pdf main.tex`
  from `presentation/slides/`. Default output is exactly62 pages: the final build
  of each PPTX slide. The original overlay builds remain available with
  `pdflatex -jobname=tdyssey-builds "\def\tdanimated{1}\input{main.tex}"`.
- Clicking a video poster in the default PDF opens its existing public Drive
  video. All 17 videos were verified as anyone-with-link viewers on 2026-11-11;
  the deck itself and the containing media folder retain their separate access.
  Eight comparison slides each open one synchronized MP4, so all panels start
  together. Saved-snapshot qualifications remain in the source and media index.
- For local video launches, compile from `presentation/slides/` with
  `pdflatex -jobname=tdyssey-local "\def\tdlocalvideos{1}\input{main.tex}"`
  twice. Keep the PDF in `slides/` and the included `videos/` tree beside it.
  Acrobat/Okular-style local-launch support depends on the viewer. A PDF is not
  an offline video container and does not provide PowerPoint-style inline media.
- Build a portable Overleaf/source zip from the repository root:
  `python -B presentation/scripts/overleaf_zip.py --out presentation-overleaf.zip`.
  It includes the 17 current MP4 files and posters, plus the deck, paper, diagrams
  and figures. Add `--without-videos` for a smaller online-only bundle. Keep
  generated zips/PDFs outside Git. On Overleaf, choose `slides/main.tex` or
  `paper/main.tex` as the main document.
- The standalone offline PPTX supplied separately embeds all 17 videos. A new
  Google Slides export need not retain those embedded files. See
  [`videos/slides/README.md`](videos/slides/README.md) for extraction/verification
  and [`videos/slides/INDEX.md`](videos/slides/INDEX.md) for the current media map.
- Graphs remain in `figures/deck/`; `make_figures.py --variant deck` rebuilds them.

## Rebuild

```bash
# from the repository root
python3 presentation/scripts/build_data.py --raw <folder with the run store's runs/>        # tables (no GPU)
MPLBACKEND=Agg .venv/bin/python presentation/scripts/make_figures.py                         # graphs from data/ only (no GPU); edit styles here
.venv/bin/python -B presentation/scripts/make_videos.py rollouts                             # episodes (GPU, checkpoints)
.venv/bin/python -B presentation/scripts/make_videos.py teacher                              # teacher episodes (GPU)
.venv/bin/python -B presentation/scripts/make_videos.py stills                               # galleries and camera views
.venv/bin/python -B presentation/scripts/make_videos.py compose                              # videos, images, stills
.venv/bin/python -B presentation/scripts/make_videos.py reel                                 # demo reel (from the videos)
```

The video and data scripts need the run store's checkpoints and test files (kept outside Git); everything they produce
is already in this folder.

## Not here yet

- A screen recording and a screenshot of the interactive demo (needs a browser): record them from the Hugging Face
  Space and add them to `videos/` and `images/`; the demo slide has a TODO.
- Speakers per section.
