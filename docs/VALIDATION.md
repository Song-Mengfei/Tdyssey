# Runtime validation

Checked on 2026-10-19 with Linux x86_64, Python 3.12.3, uv 0.13.0, and the locked
**CPU + GPC** profile. Evidence covers simulator/data development and the later
pretrained-policy check below. No adaptation or model training has been completed.
CUDA 12.4/Ada checks are recorded below; actual project training remains unvalidated.

## Versions and source

- PyTorch `2.5.1+cpu`, torchvision `0.20.1+cpu`, NumPy `1.26.4`, Pymunk `6.9.0`.
- All direct CPU/GPC requirements matched `pyproject.toml`; 85 packages installed.
- GPC commit: `a96146e568826ef46134f73c22fdf59a675e7264`.
- Simulator source SHA-256: `c1a8746343ba0f38eee5753d7a4d49f6723abe50b04a6ee68c4cf9da9e255953`.
- The initial simulator milestone materialized only simulator/T-domain source.
  The later policy milestone below added selected model source and one checkpoint.

See `docs/ENVIRONMENT.md` for installation and sparse-checkout commands. Detailed
reports with machine paths belong in ignored `artifacts/`, not shared Git history.

## Reproduce checks

From the project root after CPU/GPC setup:

```bash
.venv/bin/python -B scripts/doctor.py --runtime --gpc
TDYSSEY_GPC_ROOT=third_party/gpc .venv/bin/python -B -m unittest discover -s tests -p 'test_gpc_runtime.py' -v
```

| Check | Observed result |
| --- | --- |
| Direct package versions | All matched the shared pins |
| CPU tensor and torchvision operator | Expected sum 30 and NMS result `[0]` |
| Pymunk free-body integration | x = 0.1 after 0.1 seconds at unit velocity |
| T/I/L interface examples | Five control steps each; finite 3×96×96 image observations |
| Original versus adapted T | Matching images, rewards, flags, and state/info during 12-step rollouts at two seeds |
| T/I/L geometry and reward | Physical polygon union matched project geometry within 1e-7 pixel²; reward matched coverage-based formula to nine decimal places |
| I/L contact | Nonzero reported contact and object displacement greater than one pixel under pushing |
| T/I/L state restore | Pose, velocities, forces, goal, action target, and control-step records restored |
| Fresh pre-contact replay | Two restored rollouts agreed within 1e-7 for object pose/velocity values |

The five runtime tests use real Pymunk/GPC. They are opt-in and skipped by default
so source-only checks remain usable without third-party packages. The existing
32 source-only tests cover geometry, metrics, state/data records, construction/API
doubles, and message checks. No runtime packages are installed by test discovery.
All 37 tests also passed together with the installed CPU/GPC profile and runtime
checks enabled.

## Measured storage

Allocated sizes after installation without a persistent package cache:

| Item | Bytes | Approximate size |
| --- | --- | --- |
| CPU/GPC `.venv` | 1,423,056,896 | 1.33 GiB |
| Local uv executable | 39,718,912 | 38 MiB |
| Sparse GPC checkout including Git metadata | 290,816 | 284 KiB |

Filesystem allocation and future package versions can change these numbers.
Environments, tools, upstream checkouts, and raw reports are ignored. Checkpoints,
new interaction data, videos, and CUDA libraries are additional storage costs.

## Remaining checks

- Other GPU generations, full model-training memory/throughput, and world-model throughput.
- Repeated T policy evaluation and published-baseline performance reproduction.
- Initial-state penetration, boundary coverage, and goal reachability across seeds.
- Restoring during established contact, whose cached solver impulses are not saved.
- Original GPC training-array conversion, training, and repeated evaluation.

Legacy Gym prints its maintenance warning; NumPy remains pinned to 1.26.4 for
upstream compatibility. Do not reinterpret these short simulator checks as
evidence of policy success, exact mid-contact replay, or training compatibility.

## Interaction collector checks (2026-10-19)

The bounded collector and its loader were checked with the same CPU/GPC profile.
All **48 tests passed** with `TDYSSEY_GPC_ROOT` set; source-only discovery passed
38 tests and skipped ten runtime tests. Six new source-only checks cover limits,
seed/action reproducibility, the object-state exploration prior, and rotated reset
geometry. Five new runtime checks cover persistence/timing, uint8 quantization,
reset rejection accounting, reproducibility, terminal handling, overwrite refusal,
file corruption, byte limits, and failed-run labels.

Retained small example outputs, all verified by reading their files back:

| Shape / mode | Episodes | Control steps | Reset substeps | Contact steps | File bytes |
| --- | --- | --- | --- | --- | --- |
| T / random-waypoints | 1 | 10 | 1 | 0 | 54,412 |
| I / random-waypoints | 1 | 10 | 1 | 0 | 53,418 |
| L / object-directed | 2 | 120 | 2 | 15 | 482,194 |

The four episodes above use 590,024 bytes total across three run directories.
Counts describe these retained collector examples, not total project interaction
cost. A prior 120-step prototype was removed after retaining its accounting summary;
runtime tests and earlier smoke/replay checks are separate diagnostic work.

Commands for the L example and fresh-process read-back:

```bash
.venv/bin/python -B scripts/collect_play.py --output artifacts/collection-check-L --shape L --episodes 2 --steps 60 --mode object-directed --max-mib 8
.venv/bin/python -B scripts/collect_play.py --validate-only --output artifacts/collection-check-L
```

Outputs and detailed provenance remain under ignored `artifacts/`. Runs recorded
the working-tree source hashes and dirty state, not a claimed clean training
release. Object-directed exploration uses simulator object position. Tiny examples
do not establish exploration coverage, learned dynamics quality, or policy success;
their validation/test splits can be empty. The later milestones below add
windows/catalogs and pretrained-policy loading; training remains pending.

## Training-data loader checks (2026-10-19)

All **60 tests passed** with `TDYSSEY_GPC_ROOT` set. Source-only discovery passed
43 tests and skipped 17 installed/runtime checks. Five new source-only tests cover
window boundaries, invalid sizes, duplicate/initial-scene leakage, split names,
and metadata accounting. Seven new installed tests use independent frame/action
marker fixtures to check alignment, split isolation, short episodes, stale/corrupt
data, duplicate scenes, rate compatibility, and actual PyTorch batching.

The existing real T/I/L example data was indexed and read without additional
simulation or copied image files:

```bash
.venv/bin/python -B scripts/prepare_training_data.py --datasets artifacts/collection-check-T artifacts/collection-check-I artifacts/collection-check-L --output artifacts/training-check/catalog.json --history 2 --future 4 --batch-size 4 --max-batches 2
```

Observed results:

- Four training episodes, 140 original actions, and 124 complete windows.
- Two CPU batches of four examples; history images `(4,2,3,96,96)`, future images
  `(4,4,3,96,96)`, and action targets `(4,4,2)`.
- Image/position/goal batch tensors were finite float32; image range remains [0,1].
- Source contact-step fraction was 15/140, about 10.7%. This is not a spatial
  contact-coverage or policy-success metric.
- Catalog file: 1,766 bytes. Saved batch report: 1,587 bytes. No window cache or
  new image dataset was written.
- Validation and model-test splits remained empty, as in the source examples.
  Synthetic marker fixtures separately test split separation, not model quality.

The catalog/read-back command also passed in a fresh process. The raw dataset
files are unchanged; no new packages, weights, published data, or GPU setup were
needed at the loader milestone. The next section records later pretrained-policy
integration. Original GPC training-array conversion, meaningful held-out datasets,
and actual training remain pending.

## Pretrained-policy CPU check (2026-10-19)

The pinned GPC EMA checkpoint was downloaded anonymously and verified against
its immutable revision, size (364,554,150 bytes), and SHA-256 in
`manifests/gpc_policy.json`. It contains 208 parameter tensors covering 91,123,778
parameters and records optimization step 243900. All shapes/dtypes/values matched
the pinned architecture, with no buffers needing separate component weights.
Four original domain18 Zarr schema files were inspected: 55,480 image/action/state
records and 500 episode ends. Original training chunks were not downloaded.

Commands:

```bash
python3 -B scripts/fetch_policy_assets.py --download --max-mib 400
.venv/bin/python -B scripts/evaluate_policy.py --output artifacts/policy-baseline-2026-10-19 --steps 300 --seconds 360 --device cpu
.venv/bin/python -B scripts/replay_policy.py --report artifacts/policy-baseline-2026-10-19/report.json --output artifacts/policy-baseline-2026-10-19/T-preview.mp4
TDYSSEY_GPC_ROOT=third_party/gpc .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
```

All **68 tests passed**: 48 source-only and 20 installed/runtime checks. Five new
source-only tests cover prefix alignment/budgets, file hash/size checks, failed
download cleanup, reuse, and preservation of existing files. Three new installed
checks cover complete/atomic EMA tensor loading, rejection of bad tensors/buffers,
chunk histories, partial final chunks, terminal flags, and wall-time cutoffs.
The actual checkpoint and simulator run below provide separate integration evidence.

One matched initial scene at seed 100000 passed all T/I/L geometry checks without
rejection. Full object/pusher initial states and goals were identical across
shapes; geometry/inertia remained shape-specific. Goal distribution was randomized
upstream behavior, unlike fixed-goal collection. Each shape executed 300 controls:

| Shape | Contact steps | Maximum coverage | Maximum IoU | Final IoU | Upstream success |
| --- | --- | --- | --- | --- | --- |
| T | 240 | 0.675492 | 0.509994 | 0.018979 | No |
| I | 184 | 0.369118 | 0.226330 | 0.068892 | No |
| L | 260 | 0.373109 | 0.229339 | 0.000000 | No |

Total: 900 controls / 9,000 control-physics substeps, three reset substeps, 114
policy calls / 11,400 denoising steps, and 139.51 seconds including loading on
CPU with four threads. Median planning times for T/I/L were 1.81/1.01/0.95 seconds
per sampled chunk; these one-run timings include contention and warm-up effects.
Report size: 1,914,148 bytes. Report SHA-256:
`973f87cd6dadb8550096ffc3a13cdb6936d1c04181b40a3213e31569e5124e1b`.
Reports record the working-tree source hashes and dirty state; they are not a
claimed clean training release. Earlier synthetic-input loading checks and tests
are additional diagnostics, outside the 900 evaluation controls.

T video replay used 300 additional diagnostic controls and one reset substep,
with **no new policy predictions**. Every body pose/velocity agreed with the saved
trajectory within absolute tolerance 1e-7; rewards within 1e-9 and flags matched.
All 301 encoded frames decoded at 10 fps. Video size: 201,055 bytes; SHA-256:
`68beb0e3a765a674904629d0c0acfd28452891568c5eac2dc5eaf5ea83d25e1a`.
This verifies replay starting before contact, including later contact dynamics;
it does not test restoring a state while contact is already established.

Additional allocated checkpoint storage was 364,560,384 bytes. Expanded sparse
source checkout measured 450,560 bytes including Git metadata. No duplicate
component weights, full datasets, extra environment, or CUDA libraries were needed.
Raw reports/video remain ignored; commands and portable findings are shared here.

This is an **unranked single-sample policy**, using configured eight-action chunks,
not upstream's 50-candidate world-model ranking/spreading evaluator or a paper-score
reproduction. The T failure must be investigated with trajectory inspection and
repeated seeds. One scene provides no reliable success-rate comparison, adaptation
benefit, or generalization claim. See `docs/POLICY_BASELINE.md` for the full contract.

## CUDA and visible-window check (2026-10-19)

Switched the existing `.venv` to the locked CUDA/GPC profile, replacing the two
CPU torch packages; no second environment was created and dependency pins were
unchanged. The resulting 98 packages passed consistency checks. CUDA 12.4,
PyTorch `2.5.1+cu124`, and torchvision `0.20.1+cu124` passed on an Ada RTX 4060
Laptop GPU with 8 GiB VRAM. This records a tested configuration, not a hardware
requirement for every author. Other GPU generations require separate verification.

```bash
uv sync --locked --extra cu124 --extra gpc --python /path/to/python3.12 --no-python-downloads --no-cache --link-mode copy
.venv/bin/python -B scripts/doctor.py --runtime --require-cuda --gpc
uv pip check --python .venv/bin/python --no-cache
.venv/bin/python -B scripts/evaluate_policy.py --output artifacts/cuda-visible-large-2026-10-19 --steps 300 --seconds 240 --display --hold-seconds 60
```

Doctor reported CUDA available, sum of squares 30, gradients `[2,4,6,8]` on
`cuda:0`, a working GPU torchvision NMS operator, and no package/runtime errors.
This verifies basic backward computation; it is not a trained project model.

The policy process was independently observed using 530 MiB of GPU memory during
inference. X11 inspection showed a real **768×768** pygame window. The viewer
enlarges the 96×96 RGB display without changing model input, is resizable, and
honors Escape/window close. Two new installed tests verify no CPU fallback when
CUDA is absent and actual pygame drawing/resize/close behavior with an offscreen
test driver. All **70 tests passed**; source-only discovery passed 48 and skipped 22.

The visible GPU run used seed 100000 with identical initial physical states/goals
across shapes:

| Shape | Controls | Contact steps | Maximum IoU | Final IoU | Upstream success |
| --- | --- | --- | --- | --- | --- |
| T | 300 | 250 | 0.341957 | 0.184368 | No |
| I | 300 | 89 | 0.417260 | 0.417260 | No |
| L | 300 | 269 | 0.311426 | 0.029454 | No |

Total: 900 controls / 9,000 control-physics substeps, three reset substeps, 114
policy calls / 11,400 denoising steps. Synchronized median planning times were
about 0.47 seconds per chunk for each shape. Visible elapsed time also includes
10 Hz display pacing and final-frame hold; it is not a throughput benchmark.
CPU/CUDA generators can produce different noise samples for the same seed, so
the earlier CPU trajectories are not expected to match these GPU trajectories.
These counts exclude an earlier 300-control small-window preview, interrupted
during final-frame hold, plus unit/doctor diagnostics. No training was performed.

CUDA/GPC allocated storage: **5,985,271,808 bytes**, about **5.57 GiB**, including
torch and its required CUDA libraries. No project or uv download cache, retained
wheel archive, or duplicate CPU environment remained after installation. Task-owned
temporary message files and empty installer directories were removed. Installed
libraries and the verified inference checkpoint remain necessary runtime assets.
Other projects' environments/caches were left alone. Detailed device/path reports
stay in ignored `artifacts/cuda-validation-2026-10-19` and rollout directories.
