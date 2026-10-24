# Official T world-model training and use

This page is the shared handoff for the **GPC image denoiser** continued on
official PushT T data. It is not the custom L CNN in [WORLD_MODEL.md](WORLD_MODEL.md).
It does not train or update the pushing policy. Weights, batches and reports
stay in `artifacts/` and are **not in Git**. After a pull, collect and train
locally, or copy a finished `denoiser.pt` outside Git.

Use the installed CUDA profile from [ENVIRONMENT.md](ENVIRONMENT.md) plus pinned
`third_party/gpc` and `third_party/diffusion_policy`. Training and the click
viewer need CUDA. Source-only `--help` needs no packages:

```bash
python3 -B scripts/collect_official_contact.py --help
python3 -B scripts/train_official_contact.py --help
python3 -B scripts/visualize_world_model.py --help
```

CLI help and the commands below use `.venv/bin/python -B`. On Windows, wrap them
in the machine's WSL launcher; do not put author-specific paths here.

## What this model is

The network is the pinned GPC phase-two denoiser: five RGB frames and four
pusher targets predict the next official 96×96 frame. Inputs are RGB and
actions only. Official T occupancy, pusher disks, contact counts and the true
next frame are **privileged labels** used to weight the loss and score
validation. Do not mix this path with GPC fixtures or the custom CNN catalog.

Current default loss is a case split (D031):

- **Hit / useful** window (`moved` or `n_contacts > 0`): weight motion pixels,
  the true next T, and the pusher. Table weight is 0.
- **Miss** window: weight the still T and the pusher, so empty pushes still
  teach “T stays”. Table weight is 0.
- Replay still mixes about 60% moved windows.

Judge **useful versus miss** slices, not full-image MSE. A still-frame
comparison is a hint only. Image error is not contact, rotation, or policy
evidence.

## Collect official T play

Each `batch-*` directory stays inside the 1000-control / 64 MiB cap (D017).
`contact-balanced` play is a disclosed geometry prior, not reward search.
`--resume` skips finished batches. `--preserve-init` copies the current angles
denoiser to `artifacts/official-wm-T-init/` once.

```bash
.venv/bin/python -B scripts/collect_official_contact.py \
  --output artifacts/official-wm-T-contact-2026-10-24 \
  --batches 40 --preserve-init --resume
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--output` | required | Root for `batch-*` directories. Do not train into this path. |
| `--batches` | 40 | How many finished batches to have. 40 is a small start; a larger local run used 1250. |
| `--episodes` | 8 | Episodes per batch. At least 7 so frozen splits are nonempty. |
| `--steps` | 100 | Controls per episode. |
| `--seconds` | 300 | Wall-time budget per batch. |
| `--workers` | 2 | Collector processes. |
| `--seed` | 240000 | Collection seed. |
| `--preserve-init` | off | Copy angles weights to `artifacts/official-wm-T-init/`. |
| `--resume` | off | Continue until `--batches` without rewriting finished dirs. |
| `--source-root` | `third_party/diffusion_policy` | Official PushT source. Leave default. |

This does not train a model. `model_test` stays unread later.

## Train one denoiser

Continue **one** denoiser. New-run setup copies `--checkpoint` weights only and
writes a fresh sidecar `{updates: 0, init_sha256}`. It does not inherit an old
update count. Each chunk samples 64 train windows from the full train-split
index, then scores a frozen 32 useful / 32 miss validation subset. Ctrl+C keeps
the last finished chunk; the same command with `--resume` continues.

A new machine should start from the copied init weights:

```bash
.venv/bin/python -B scripts/train_official_contact.py \
  --datasets artifacts/official-wm-T-contact-2026-10-24 \
  --output artifacts/official-wm-T-case-loss \
  --init-from-best --resume
```

If a local k=8 checkpoint already exists, start a **new** output directory from
those weights. Do not `--resume` the old k=8 report, and do not write into an
earlier from-init run:

```bash
.venv/bin/python -B scripts/train_official_contact.py \
  --datasets artifacts/official-wm-T-contact-2026-10-24 \
  --output artifacts/official-wm-T-case-loss-from-k8 \
  --checkpoint artifacts/official-wm-T-contact-train-2026-10-24/denoiser.pt \
  --resume
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--datasets` | required | Root of finished official `batch-*` dirs. Official family only. |
| `--output` | required | Train directory: `denoiser.pt`, `denoiser.json`, `report.json`. Must not be the collection root. |
| `--checkpoint` | released GPC `denoiser.pth` | Weights copied on a **new** run. Ignored for resume once `output/denoiser.pt` exists. |
| `--init-from-best` | off | Use `artifacts/official-wm-T-init/denoiser.pt` instead of `--checkpoint` on a new run. |
| `--resume` | off | Continue `output` if it exists; start if it does not. Without this, an existing output is refused. |
| `--updates` | 8 | AdamW steps per chunk (1–20). |
| `--max-updates` | 0 | Stop after this many total steps. 0 = no cap. |
| `--budget-seconds` | 0 | Stop after this many seconds. 0 = no cap. |
| `--pool-windows` | 64 | Train windows sampled from every batch each chunk. |
| `--validation-windows` | 64 | Frozen validation size (32 useful / 32 miss when labels exist). |
| `--batch-size` | 4 | Windows per optimizer step (1–8). |
| `--train-seconds` | 900 | Per-chunk wall-time limit. |
| `--seed` | 250000 | Sampling and frozen-validation seed. |
| `--gpc-root` | `third_party/gpc` | Pinned GPC source. Leave default. |

Look at printed `useful_*` and `miss_*` fields. Useful motion is compared with
still-frame about 0.160 on the current frozen subset; miss ROI with about 0.023.
Do not treat a small full-image MSE as calibration. Close the visualizer before
training so the job has the GPU.

## Watch a checkpoint

Left: live official T. Right: predicted next frame. Click the left panel to
choose a pusher target, Space executes it, Escape closes. Needs a real window;
do not set `SDL_VIDEODRIVER=dummy`. Stop training first.

```bash
unset SDL_VIDEODRIVER
.venv/bin/python -B scripts/visualize_world_model.py \
  --checkpoint artifacts/official-wm-T-case-loss/denoiser.pt
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | an older hour-run path | `denoiser.pt` to load. Must exist. |
| `--seed` | 100003 | Official T start scene. This default pose missed the T on all 16×3 angle-probe headings. |
| `--panel` | 512 | Left/right panel size in pixels. Window width is `2 * panel`. |
| `--gpc-root` | `third_party/gpc` | Denoiser source. Leave default. |
| `--source-root` | `third_party/diffusion_policy` | Official environment source. Leave default. |

This is a qualitative viewer. It is not contact, rotation, or policy evidence.

## Optional one-scene probe

```bash
.venv/bin/python -B scripts/probe_push_angles.py \
  --checkpoint artifacts/official-wm-T-case-loss/denoiser.pt \
  --output artifacts/official-wm-T-case-loss-angles-probe
```

`--output` must not already exist. Default `seed=100003` is an all-miss scene:
object overlap there means “the unmoved T still looks like a T”, not contact.
See D030.

## Limits

- No policy update from these weights.
- `model_test` is never opened.
- Privileged geometry and contact counts must stay disclosed.
- A later author cannot replay a local `artifacts/` run from Git alone.
- Action-sensitive imagined rewards are still unproven; see `docs/STATUS.md`.
