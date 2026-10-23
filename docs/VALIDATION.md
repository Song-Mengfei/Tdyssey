# Runtime validation

Newest work: [portable training and live monitoring](#portable-training-and-live-monitoring-2026-10-23).

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

## Official RSS visual CNN baseline (2026-10-19)

The primary baseline now uses `real-stanford/diffusion_policy` commit
`5ba07ac6661db573af695b419a7947ecb704690f`, with 18 relevant source files hashed
in `manifests/official_dp.json`. No upstream source was patched. It loads the
authors' `image/pusht/diffusion_policy_cnn/train_0` checkpoint named
`epoch=0500-test_mean_score=0.884.ckpt` in the original `PushTImageEnv(legacy=True)`.
The earlier GPC runs use different weights, body construction/coordinates, and
randomized goals; they are not official baseline measurements or a fair method
comparison against this run.

The full published checkpoint was 4,293,295,813 bytes; recorded SHA-256:
`bac7221f7e34cd51162dc1972e1a39ffcddc87de1dc1780c44ffa61b88c4ff76`.
Its embedded policy configuration exactly matched the independently downloaded
publisher configuration. The retained EMA file has all 300 state entries,
including encoder/denoiser parameters, normalizers, and buffers. Size:
1,095,708,090 bytes; SHA-256:
`102fd63b53e9e2ba8261c843e770725740ebdddebe4b1e96dbd92089477ba7c4`.
The original training checkpoint was removed after compact-load and physical
execution checks. No expert demonstration arrays were downloaded.

The unchanged official class constructed and loaded strictly, with 262,709,056
parameters including saved normalizer parameters. Original preprocessing retained
the 84×84 evaluation crop, GroupNorm, checkpoint normalization, two-observation
history, DDPM with 100 reverse steps, and eight-action prefix of 16 predictions.
Original environment reset distribution, fixed goal, T geometry/dynamics, rewards,
and terminal threshold were retained. No GPC wrapper, geometry rejection, world
model, action ranking, or training was involved.

Commands, with the source/asset setup in `docs/OFFICIAL_BASELINE.md`:

```bash
uv sync --locked --extra cu124 --extra gpc --extra dp --no-cache --link-mode copy
.venv/bin/python -B scripts/doctor.py --runtime --require-cuda --gpc --dp
.venv/bin/python -B scripts/evaluate_official.py --output artifacts/official-cnn-baseline-2026-10-19 --display --video --hold-seconds 30
TDYSSEY_GPC_ROOT=third_party/gpc TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy TDYSSEY_OFFICIAL_CHECKPOINT=checkpoints/official-dp/ema_policy.pt .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
```

All **76 tests passed**: 51 source-only and 25 opt-in installed/runtime tests.
New tests cover official resource identity, modified-source rejection, namespace
isolation, history/budget agreement with the official `MultiStepWrapper`, actual
collision-based metrics, and CUDA inference parity against a direct call to the
unchanged official class. The wrapper and direct call produced identical actions
with the same observations and CUDA RNG seed; the returned eight actions exactly
matched prediction indices 1:9. Saved normalizer keys were action/agent_pos/image.

Three development test scenes, all on CUDA:

| Scene seed | Policy RNG seed | Controls to success | Contact steps | Maximum/final IoU | Maximum clipped reward |
| --- | --- | --- | --- | --- | --- |
| 100000 | 3486894925 | 128 | 74 | 0.924145 | 1.0 |
| 100001 | 3690532503 | 157 | 66 | 0.908241 | 1.0 |
| 100002 | 2649433489 | 273 | 86 | 0.923627 | 1.0 |

Total evaluation cost: **558 controls / 5,580 control-physics substeps**, three
reset substeps, 71 policy calls / 7,100 denoising steps. Evaluation elapsed time
was 163.03 seconds including loading and display pacing, excluding the 30-second
final-frame hold. Median synchronized planning times were about 1.48 seconds per
chunk; concurrent tests and display work mean this is not an isolated throughput
benchmark. Report size: 784,788 bytes; SHA-256:
`9188295f1ec2fe3fd45b80aa8d5e2315824d5a577a2bed1f286bafbe34112f12`.
The report records its dirty working tree and source hashes rather than claiming
a clean experiment release.

Video read-back verified 129/158/274 frames for 128/157/273 actions, including the
initial frame. Three 512×512 previews occupied 176,503/215,592/341,647 bytes.
The videos were streamed during evaluation, with no additional replay controls.
Source-only checks and help/plan commands require no optional runtime packages.

The combined CUDA/DP/GPC environment has 113 packages and allocated 6,138,802,176
bytes, about **5.72 GiB**. Fifteen added official-policy dependencies did not change
torch/torchvision/CUDA pins or create another environment. Doctor and package
consistency checks passed. The trimmed official checkout used 12,898,304 bytes
including Git metadata; retained official asset directory used 1,095,725,056 bytes.
No source checkpoint, wheel archive, project/uv download cache, or dataset remains
from setup beyond the deliberately retained inference asset and small metadata.
Other projects' caches/environments were untouched.

These are **three successful development scenes**, not a 50-scene/multiple-training-
seed paper reproduction, an estimated population success rate, or an adaptation
result. Mean maximum clipped reward and observed success fraction were both 1.0
on these three scenes; they are different metrics. Local CUDA RNG streams are
recorded and differ from historical evaluation scheduling. Tests/compatibility
probes are separate diagnostics, outside the 558 evaluation controls. At this
checkpoint L porting, matched target-shape data and learning were pending; the
following sections record the later completed shape port and evaluation.

## Official capital L and matched physical centers (2026-10-19)

The official adapter preserves T and adds the capital-L geometry described in
[shape protocol](OFFICIAL_SHAPES.md): height 120, bottom width 90, stroke 30, one
mass-1 body with area 5400 and uniform-density inertia 1875. Physics and geometry
hashes reject incompatible restores. T retains the original mass, inertia and
center convention. Equal mass does not match area, density or inertia.

```bash
.venv/bin/python -B scripts/evaluate_shapes.py --output artifacts/official-capital-L-2026-10-19 --display --video --hold-seconds 30
```

The first candidate, seed 100000, passed both shapes' rotated wall/clearance and
not-already-solved checks. Object/target mass centers and full pusher state match;
L's local body/goal origins differ from T's. This explicitly selected distribution
is separate from the original benchmark resets.

| Shape | Controls | Contact steps | Success | Best IoU | Final IoU |
| --- | --- | --- | --- | --- | --- |
| T | 128 | 74 | Yes | 0.924145 | 0.924145 |
| L | 300 | 288 | No; control budget | 0.301521 | 0 |

Total: **428 controls / 4280 control-physics substeps**, two reset substeps,
54 policy calls / 5400 denoising steps. Evaluation elapsed: 125.28 seconds,
including loading/display pacing and excluding the 30-second final-frame hold.
All T actions, contacts, full body-state traces and flags matched its earlier
canonical run exactly (maximum state difference zero). T rewards differed by at
most `4.44e-16` from polygon arithmetic. Its video was byte-identical. L's maximum
clipped coverage reward was 0.487723, separate from IoU. This is a real baseline
failure on one scene, not a population performance estimate or a training result.

Ignored output directory: `artifacts/official-capital-L-2026-10-19`:

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| `report.json` | 850661 | `ebf959e353459c2c7a71dc4732012b5cfe6c21899c5dc888682dba1a5994d3ad` |
| `T-100000.mp4` (129 frames) | 176503 | `a433c2ab03f8597505af1a7672cd9865e25c9348c87e98c0bcbbb3d4e55938c2` |
| `L-100000.mp4` (301 frames) | 368564 | `24b7aa1864b805afaca73acc8520ee32359327ae7312d787b2b46a8f7894cb51` |

Both videos decoded fully; verification used saved traces/videos without extra
policy or physics execution. `verification.json` records the exact T comparison.
Reports record their dirty working tree and source hashes. Initial nine shape
tests passed (three source-only, six installed), including original T agreement,
actual contact motion, collision/goal/rendered-foot agreement, full-state restore,
fresh pre-contact replay, common-center matching and rejected-reset accounting.
Diagnostic test controls are separate from the 428 evaluation controls. Contact
solver caches and future-reset RNG state remain outside snapshots; mid-contact
replay is not established. No additional environment, weights, dataset or training
was required. At this milestone official L collection and learned adaptation were
pending; the collection section below records the subsequent data milestone.

## Capital-letter sweep implementation checks (2026-10-19)

All **90 tests passed** in the installed CUDA/DP/GPC environment: 57 source-only
checks and 33 opt-in installed/runtime checks. Source-only discovery passed 57
and skipped 33. `scripts/check_repo.py`, CLI help, and whitespace checks passed.
The full suite includes the added capital-letter geometry, symmetry, collision,
contact, partial-overlap reward agreement, holes, common 27-shape reset and gallery
presentation checks. Original T parity and direct official-class CUDA inference
parity continue to pass. No manifest/lock change or installation was needed.

```bash
python3 -B scripts/check_repo.py
python3 -B -m unittest discover -s tests -p 'test_*.py'
TDYSSEY_GPC_ROOT=third_party/gpc TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy TDYSSEY_OFFICIAL_CHECKPOINT=checkpoints/official-dp/ema_policy.pt .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -B scripts/evaluate_shapes.py --alphabet --seconds 3600 --output artifacts/official-alphabet-2026-10-19 --video
python3 -B scripts/summarize_shapes.py --report artifacts/official-alphabet-2026-10-19/report.json --output artifacts/official-alphabet-2026-10-19/summary.html
```

The full alphabet run completed separately from tests and the earlier T/L probe.
After adding peak-overlap diagnostics/screenshots, the final installed suite passed
**all 92 tests**: 58 source-only and 34 opt-in runtime checks. Source-only discovery
passed 58 and skipped 34. The two additions verify earliest-peak/tie/lost-overlap
handling and the correct decoded post-action frame after initial video frame zero.
Source/whitespace checks passed. No additional policy evaluation was needed for
these presentation changes.

## Completed frozen-policy alphabet sweep (2026-10-19)

All 27 shapes shared accepted development scene **100000** and CUDA noise seed
**3486894925**. The first candidate passed every shape's checks. Original T and Y
succeeded; all other letters and the 10%-shorter-cap T exhausted 300 controls.
The policy/checkpoint remained frozen, with no target-shape data or training.

| Shape | Success | Controls | Best IoU | Final IoU | Contact steps |
| --- | --- | --- | --- | --- | --- |
| T | Yes | 128 | 0.9241 | 0.9241 | 74 |
| A | No | 300 | 0.5433 | 0.1848 | 81 |
| B | No | 300 | 0.3695 | 0.2728 | 160 |
| C | No | 300 | 0.2755 | 0.0645 | 118 |
| D | No | 300 | 0.8154 | 0.4172 | 127 |
| E | No | 300 | 0.3129 | 0.2453 | 151 |
| F | No | 300 | 0.1749 | 0.1542 | 138 |
| G | No | 300 | 0.3782 | 0.2302 | 109 |
| H | No | 300 | 0.2261 | 0.1311 | 181 |
| I | No | 300 | 0.4058 | 0.1898 | 139 |
| J | No | 300 | 0.1876 | 0.1165 | 82 |
| K | No | 300 | 0.7512 | 0.0620 | 175 |
| L | No | 300 | 0.3015 | 0.0000 | 288 |
| M | No | 300 | 0.3322 | 0.1233 | 129 |
| N | No | 300 | 0.4352 | 0.2607 | 134 |
| O | No | 300 | 0.3384 | 0.1099 | 128 |
| P | No | 300 | 0.2332 | 0.2291 | 159 |
| Q | No | 300 | 0.2599 | 0.1725 | 82 |
| R | No | 300 | 0.4032 | 0.2456 | 205 |
| S | No | 300 | 0.5755 | 0.5755 | 149 |
| U | No | 300 | 0.3023 | 0.1137 | 184 |
| V | No | 300 | 0.3422 | 0.1713 | 133 |
| W | No | 300 | 0.3919 | 0.2577 | 221 |
| X | No | 300 | 0.2896 | 0.1684 | 171 |
| Y | Yes | 235 | 0.9121 | 0.9121 | 120 |
| Z | No | 300 | 0.4648 | 0.1616 | 153 |
| T_narrow | No | 300 | 0.8717 | 0.7751 | 69 |

Cost: **7863 controls / 78630 control-physics substeps**, **27 reset substeps**,
**996 policy calls / 99600 denoising steps**. Elapsed: **1553.45 seconds** (25.89
minutes), including loading, rendering and streamed encoding, without display
pacing or a final-frame hold. This is an execution measurement, not an isolated
GPU throughput benchmark. Test/diagnostic controls and the earlier probes are
separate from these counts.

All **7890 frames across 27 videos** decoded at 512×512, including one starting
frame per episode. Videos total 12,025,483 bytes. Read-back also verified per-step
counts, final/max scores, contacts, success/cutoff consistency, common seeds and
current source hashes. T and L actions, full states, contacts, flags, rewards and
video bytes match the earlier paired probe **exactly**. Verification performed
no inference, physical replay, training or downloading.

Ignored output: `artifacts/official-alphabet-2026-10-19`. The full report is
15,589,791 bytes, SHA-256:
`e320ff873951cb6d676e23ca0df560ab9d9610fbe66adb6b17871166b8745a8f`.
`verification.json` records all video sizes/frame counts/hashes and prior T/L
comparisons. `summary.html`/`summary.csv` preserve all outcomes and peak/final
measurements; 27 `*-best.png` screenshots show the earliest best-IoU post-action
frames. Total local output including the gallery and screenshots is about **28
MiB**, with no new environment, model asset, dataset or persistent download cache.
The run started with a dirty tree at parent `7f0d457`; all recorded evaluation
source hashes match the subsequently committed implementation `6b98c90`.

Near misses are visible in the saved data: D peaked at control 201 with IoU
0.8154 / 89.83% target coverage, then finished at IoU 0.4172. K peaked at 163 with
IoU 0.7512 / 85.79% coverage, then finished at 0.0620. The near-T control peaked
at 262 with IoU 0.8717 / 93.14% coverage, then finished at 0.7751. These missed
the unchanged `coverage > 0.95` rule. The gallery records best control, coverage
and best-to-final IoU loss rather than relabeling them successes. Y reached 95.40%
coverage at control 235 and terminated successfully. Peak screenshots were selected
after evaluation; they are descriptive diagnostics, not a learned stopping rule.

These are one-scene observations for one T-trained checkpoint and one specified
block alphabet. They neither establish Diffusion Policy's general limit nor a
reliable success rate or ranking. Near-T failure here does not establish a 10%
tolerance boundary. Repeat fixed evaluation scenes before making those claims.
Area/inertia/density, contact contours, component seams and common reset rejection
are disclosed in the [shape protocol](OFFICIAL_SHAPES.md). There is still no
adaptation result. The following collection section records the subsequent data
milestone; learned optimization remains pending.

## Published alphabet video snapshot

All 27 unchanged MP4 files are committed under
`docs/media/official-alphabet-2026-10-19/`, totaling 12,025,483 bytes. Their copied
sizes/SHA-256 values match the decoded originals in `verification.json`.
[Browse the video table](ALPHABET_RESULTS.md); `videos.json` beside the clips
records scene/noise seeds, original report hash, control costs and per-video hashes.
This explicit curated snapshot is an exception to the default raw-artifact policy.
No inference, replay, re-encoding, download or dependency installation was needed.

## Official L collection and training data (2026-10-19)

Implemented the [official collector](OFFICIAL_COLLECTION.md), keeping the original
four-value step API, fixed goal, capital-L physics and correct post-action image
alignment. Known evaluation seeds 100000–100002 are excluded; skipped seed
candidates do not simulate, while rejected physical resets count their actual
substeps. Full accepted/rejected reset-state provenance is retained. The loader
now distinguishes simulated resets from unsimulated exclusions and refuses mixed
official/GPC environment-family catalogs. Historical GPC data checks still pass.

```bash
.venv/bin/python -B scripts/collect_official_play.py --shape L --episodes 8 --steps 100 --mode object-directed --output artifacts/official-play-L-2026-10-19
.venv/bin/python -B scripts/collect_official_play.py --validate-only --output artifacts/official-play-L-2026-10-19
.venv/bin/python -B scripts/prepare_training_data.py --datasets artifacts/official-play-L-2026-10-19 --output artifacts/official-play-L-2026-10-19.catalog.json --split train
.venv/bin/python -B scripts/prepare_training_data.py --catalog artifacts/official-play-L-2026-10-19.catalog.json --split validation
.venv/bin/python -B scripts/prepare_training_data.py --catalog artifacts/official-play-L-2026-10-19.catalog.json --split model_test
```

The retained run completed **eight 100-control episodes / 800 controls / 8000
control-physics substeps**, with **12 simulated reset substeps** (eight accepted,
four rejected), **808 stored frames**, and **354 contact steps**. All eight
trajectories have contacts. The 44.25% contact-step fraction is not a spatial
contact-coverage metric, demonstration quality or task success rate. This is
object-directed non-expert play using a disclosed privileged object-position prior.

Encoded data: **3,279,264 bytes** (about 3.13 MiB), below the 32 MiB limit. Elapsed
from source loading through collection/writing: **2.0131 seconds**; final read-back
is additional. Concurrent tests and filesystem work mean this is not an isolated
throughput benchmark. No expert demonstrations, checkpoint inference, GPU learning,
downloads or new environment were required. Data remains in ignored artifacts.

| Frozen split | Episodes | Controls | Contact steps | Windows, history 2 / future 4 |
| --- | --- | --- | --- | --- |
| train | 6 | 600 | 275 | 576 |
| validation | 1 | 100 | 40 | 96 |
| model_test | 1 | 100 | 39 | 96 |

Fresh-process dataset validation passed. All three splits produced two verified
PyTorch batches of four examples: history images `(4,2,3,96,96)`, future images
`(4,4,3,96,96)`, actions `(4,4,2)`. These are CPU data-loader checks, not training;
GPU training remains a later task. Images stay uint8 on disk and windows are not
duplicated. There are no shared episode IDs/initial states across frozen splits.

The source dataset manifest SHA-256 is `37629a76a0cc20ec33f9dfb7c8cd945d076e0bbb81668cfffe9fb7594f7251b5`.
The 2634-byte frozen catalog SHA-256 is `d3b4b49bad025e4d235844857fecd8ad5d1963f00cd352416c9505bfc4bf42c6`.
Batch reports are ignored files named `official-play-L-2026-10-19-*-batches.json`
under `artifacts/`. Recorded source hashes match the collection code; provenance
records dirty development at parent `3012a72`. Diagnostic/unit-test physics is
separate from the retained pilot's 800 controls and 12 resets.

All **100 tests passed** together in the installed CUDA/DP/GPC environment:
**61 source-only checks and 39 opt-in runtime checks**. Eight new checks cover
resource/exclusion limits, nonempty/repeatable whole-episode splits, repeated-scene
rejection, real collection/persistence/window timing, skipped/rejected reset costs,
reproducibility, original terminal handling, time-budget exhaustion and environment-
family isolation. Source-only discovery, CLI help, repository and whitespace checks
also passed. The injected terminal case checks recorder behavior, not task success.

This is a useful contact-bearing pilot for starting model development, not enough
evidence of world-model quality, physical policy improvement or generalization.
No learned world model/policy update has run. Future evaluated scenes must be added
to exclusions explicitly; near-duplicate scene detection, broad contact coverage,
mid-contact restoration and final multi-seed policy evaluation remain pending.

## First visual world-model CUDA pilot (2026-10-19)

Implemented the [visual model protocol](WORLD_MODEL.md) on the existing official
L data. Inputs are two RGB frames, pusher history, goal and future actions; object
motion, geometry-derived ROI, reward and contact are privileged **targets**, not
network inputs. The official diffusion policy/representation/weights are unchanged.
A residual CNN with 1,302,570 parameters predicts images, pusher/object motion,
reward and contact autoregressively for four controls. Output-motion normalization
uses 600 unique training transitions only. No expert data or model download was
needed, and the existing CUDA environment/lock were unchanged.

```bash
.venv/bin/python -B scripts/train_world_model.py --catalog artifacts/official-play-L-2026-10-19.catalog.json --output artifacts/visual-world-model-L-2026-10-19
.venv/bin/python -B scripts/evaluate_world_model.py --catalog artifacts/official-play-L-2026-10-19.catalog.json --checkpoint artifacts/visual-world-model-L-2026-10-19/best.pt --output artifacts/visual-world-model-L-2026-10-19-validation-recheck.json
.venv/bin/python -B scripts/probe_world_model.py --catalog artifacts/official-play-L-2026-10-19.catalog.json --checkpoint artifacts/visual-world-model-L-2026-10-19/best.pt --output artifacts/visual-world-model-L-2026-10-19-candidate-probe.json
python3 -B scripts/summarize_world_model_probe.py --probe artifacts/visual-world-model-L-2026-10-19-candidate-probe.json --output artifacts/visual-world-model-L-2026-10-19-score-summary.json
```

The actual run completed **10 epochs / 720 CUDA optimizer updates**, presenting
5760 overlapping training windows / 23040 forecast steps from six training episodes.
These repeated forecasts are not new simulator interactions. Fixed seed 42, AdamW
0.0003 and recorded loss weights were used. Best trained checkpoint: epoch 6,
selected on last-horizon validation contact position error. Initial untrained
results were separate and ineligible for checkpoint selection. **The real model-test
images were never loaded**, and no policy update or imagined-policy learning ran.

Training/loading/validation/preview elapsed: **14.83 seconds**; maximum measured
CUDA allocation: **215,508,480 bytes** (about 205.5 MiB). This is a small-prototype
measurement, not a full-policy training-memory requirement or isolated throughput
benchmark. Only one checkpoint is retained; no optimizer state/epoch archives.

Validation is one frozen 100-control episode, giving 96 overlapping windows. At
each listed horizon there are 40 contact and 56 noncontact forecast labels:

| Metric / horizon | Learned model | Persistence baseline |
| --- | --- | --- |
| Object position error (pixels), contact, +1 | 3.1486 | 4.0354 |
| Object position error (pixels), no contact, +1 | 0.9633 | 0.0100 |
| Object position error (pixels), all, +1 | 1.8738 | 1.6873 |
| Full-image MSE, all, +1 | 0.000562 | 0.000531 |
| Object-region MSE, all, +1 | 0.009600 | 0.009692 |
| Pusher position error (pixels), all, +1 | 4.176454 | 4.776382 |
| Object position error (pixels), contact, +4 | 12.0175 | 15.2505 |
| Object position error (pixels), no contact, +4 | 3.9425 | 0.6733 |
| Object position error (pixels), all, +4 | 7.3071 | 6.7471 |
| Full-image MSE, all, +4 | 0.001851 | 0.001693 |
| Object-region MSE, all, +4 | 0.022306 | 0.023158 |
| Pusher position error (pixels), all, +4 | 15.919136 | 17.987750 |

Persistence repeats the last RGB/pusher pose and zero object motion; reward/contact
use the constant training means. Learned contact position error improved on this
checkpoint-selection episode, but **overall position/image errors worsened and
noncontact false motion grew**. The image/motion heads are not constrained to agree.
Reward/contact predictors remained close to constants. The report preserves all
horizons/groups, not just the improved metric. A fresh-process checkpoint load
reproduced the complete validation metric dictionary exactly. The generated
`prediction-preview.png` shows the first validation window with contact, selected
without reference to prediction quality; `validation-errors.png` plots both the
improvement and deterioration. These are not model-test or policy-success results.

### Matched physical action probe

Only two disjoint eligible validation starts existed (anchors 15 and 47). Each
branch cold-restored the same full saved state and compared recorded, hold,
privileged-object-directed toward, and away candidates over four controls.
Restored images matched the current stored frame within the uint8 quantization
bound. Work: **8 state restores / 32 controls / 320 control-physics substeps**,
**one initialization-reset substep**, and **32 model forecast steps**. This is
additional diagnostic interaction, separate from the 800 collected controls.
There were no model/policy updates and no model-test access.

The primary max-reward block score tied physically (range zero within roundoff)
because an early high-overlap moment hid the later differences. Its near-zero
regret is therefore **uninformative**, not successful action ranking. Predicted
maximum reward ranges were only 1.75e-7 / 1.19e-7. The selected predictions were
pessimistic here, rather than evidence of a useful optimistic-action search.

A separately labelled **posthoc** final/sum analysis used saved traces with zero
new inference/physics. Actual final reward ranges were 0.07728 / 0.10080; predicted
final ranges were only 6.71e-7 / 3.99e-7. With final reward, the predictor preferred
toward pushes that lost alignment, instead of holding/retreating. No rank-quality
claim is supported. Future ranking comparisons must freeze block scoring and
terminal conventions in advance. Cached contact impulses are absent; these are
matched fresh-space branches, not proof of exact warm mid-contact replay.

### Artifacts, checks and limits

Local ignored artifacts:

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| `prediction-preview.png` | 60665 | `abe450891237a4c95aa3b7a2de7ecffe1159a9f9369eee8e9018550f6db978f8` |
| `best.pt` | 5223938 | `9d66fd8ec8071554c68ec3173ee4bdf2b169a6ae5d44344839fc9308b67dc81a` |
| `validation-errors.png` | 65129 | `871079a34b7a2a4450ae9dc87c24ace7dd7b3d98f4c8f7b2ce868eaec4aef4a8` |
| `report.json` | 156113 | `108ee651ad6a61790e142eb8d67cb7240202ed64c92a3133cbf507ce04b6b747` |
| `visual-world-model-L-2026-10-19-candidate-probe.json` | 78948 | `ed62daffbef433c24ca1a9d4cfe90db962f0e9364956f61e65ff42c3963df6bd` |
| `visual-world-model-L-2026-10-19-score-summary.json` | 5793 | `5890b99f1367a36b967e801a4283edb9451bb78cbeeebf28c93eb8e60fca3334` |
| `visual-world-model-L-2026-10-19-validation-recheck.json` | 11952 | `e9c78319e795277d785c9261f24f0538aa94ae2c401a8d173f04e7afb3ec60d7` |

The training directory uses about **5.3 MiB**, including the checkpoint, report
and previews. The source hashes match the actual training implementation; the
report records dirty development at parent `cb87a67`. The evidence sidecar also
records the original probe-runner hash and exact fresh-load comparison. No policy
checkpoint, new data/environment, download or cache archive was created.

All **110 tests passed** in the installed CUDA/DP/GPC profile: 67 source-only and
43 opt-in runtime checks. New checks cover limits, episode-local sampling, metric
weighting/empty groups, angle wrapping, loss-of-progress score views, training-only
normalization, independent marker timing, causal/autoregressive action gradients,
physical metric units and an actual CUDA optimizer update/checkpoint load. A
synthetic test deliberately corrupts its model-test file and verifies training
never reads it. Those synthetic gradients/updates are diagnostics, separate from
the retained 720-update real-data pilot. Source discovery skips runtime checks;
repository/help/whitespace checks passed.

This completes an executable first training experiment, **not the learned-dynamics
reliability gate**. One training seed/validation episode, selection bias, sparse
scene diversity, false noncontact drift, image/motion inconsistency and flat reward
action sensitivity remain blockers to imagined control. Improve data/action
coverage and prediction quality, then repeat physical candidate checks and direct
simulator policy optimization before imagined optimization. Final model-test and
multi-seed evaluation remain pending.

## Focused automated suite (2026-10-19)

The earlier counts above record the suites that ran at each milestone. The
current suite has **98 tests: 55 source-only and 43 opt-in runtime checks**.
Twelve peripheral/repeated checks were removed as recorded in decision D019;
all installed physics, video alignment, data-integrity and GPU checks remain.
No model implementation, experiment split, evaluation budget or saved result
changed. The actual commit/PR message checker remains in use.

```bash
python3 -B scripts/check_repo.py
python3 -B -m unittest discover -s tests -p 'test_*.py'
TDYSSEY_GPC_ROOT=third_party/gpc TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy TDYSSEY_OFFICIAL_CHECKPOINT=checkpoints/official-dp/ema_policy.pt .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
```

Repository checks passed. Source-only discovery passed 55 and skipped 43;
the installed suite passed all 98 in 12.674 seconds. Its synthetic CUDA optimizer
update is a diagnostic, not another real-data model training experiment.

## Pretrained GPC world-model compatibility check (2026-10-19)

Pinned source: `a96146e568826ef46134f73c22fdf59a675e7264`. Downloaded only
`world_model_checkpoint/phase_two_checkpoint/denoiser.pth` from checkpoint revision
`c16a467ec176af19bba5136564a81d8d61d72af9`: **35,043,916 bytes**, SHA-256
`4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
The selected phase-two checkpoint has **8,729,827 parameters / 298 state entries**;
strict loading of all entries and finite-value checks passed on CUDA. Required
source hashes/configuration are in `manifests/gpc_world_model.json`.

No package/environment/lock change, demonstration dataset, reward checkpoint or
duplicate model cache was needed. The sparse GPC source checkout measured about
596 KiB afterward; the one new checkpoint is about 33.42 MiB. Licensing terms
were not stated in the pinned top-level source/model metadata; upstream code and
weights remain outside Git. See [compatibility details](WORLD_MODEL.md#pretrained-gpc-compatibility-check).

```bash
python3 -B scripts/fetch_gpc_world_model.py --download
.venv/bin/python -B scripts/probe_gpc_world_model.py --output artifacts/pretrained-gpc-T-2026-10-19
TDYSSEY_GPC_ROOT=third_party/gpc TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy TDYSSEY_OFFICIAL_CHECKPOINT=checkpoints/official-dp/ema_policy.pt TDYSSEY_GPC_WORLD_CHECKPOINT=checkpoints/gpc-world-model/denoiser.pth .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
```

The CUDA wrapper matched unchanged upstream sampling exactly, including two-step
autoregressive feedback. Independent action markers check the four-frame/control
offset. Future actions do not affect earlier predictions; global inference RNG
and pre-existing `data`/`diffusion`/`utils` modules are restored. Images use the
actual training range [0,1], not the range suggested by quantization comments.
Actions use the source dynamics scale [0,511]. Released sampling disables
gradients and rounds pixels; differentiable policy optimization is unvalidated.

The physical check used two hand-constructed pre-contact scenes (bottom/left)
in each native/official T environment, three hold controls to obtain true history,
and four future controls for hold/push/away/side candidates. Initial collision/goal
geometry is aligned, while palettes and mass/inertia remain different. The same
full physical/controller anchor is cold-restored per branch; true history and
restored images matched exactly. All history contacts were zero and each scene's
push branch later contacted the object. Scene/action construction uses privileged
geometry; this is neither collected training data nor a policy evaluation.

Counts: **76 controls / 760 control physics substeps / two reset substeps /
20 restores / 64 forecast steps / 192 denoising steps**. Elapsed: **6.3919 seconds**;
peak allocated GPU memory **110,936,064 bytes (105.8 MiB)**. Diagnostics/tests and
the earlier CNN pilot are separate work. No stored dataset was opened, no
normalization was fitted, and no research model/policy weights were updated.

At horizon four, object-region image MSE (lower is better):

| Family / group | Samples | Pretrained GPC | Persistence |
| --- | --- | --- | --- |
| Native / contact | 2 | 0.091607 | 0.121502 |
| Native / no contact | 6 | 0.000228 | 0 |
| Official / contact | 2 | 0.153614 | 0.044012 |
| Official / no contact | 6 | 0.134629 | 0 |

The native contact mean hides a failure: bottom-push ROI MSE was 0.007846 vs
0.068212, but left-push was 0.175368 vs 0.174792 and missed rotation. Native left
physical object-origin movement was 15.344 pixels, versus 7.834 in the official
branch; the different physics must not be described as matched trajectories.
Official predictions turn the object red and remove the green target, so the
gray-color mask finds zero predicted pixels. That mask failure does not mean
the image has no T. Color-centroid/IoU metrics are appearance-sensitive proxies,
not a verified pose/reward decoder. Geometry masks/true states are evaluation
labels only. Two hand-constructed scenes do not establish general performance.

Artifacts stay ignored under `artifacts/pretrained-gpc-T-2026-10-19/`:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `report.json` | 188408 | `e69055e8ada899e5381d3fe3d3779ea7b4ad94fad7b676cd799b26f8da00d2b1` |
| `predictions.png` | 171927 | `b191d1d9d74bb70224276eed35f442d697dcd5d2754c62f70716a62d417e052b` |

The report fingerprints exactly match the predictor/probe/spec source files;
development parent was `a03b72d`. The checkpoint and report/preview add about
33.76 MiB total, without duplicate downloads. All **100 tests passed** in the
installed profile in **13.680 seconds**: 56 source-only and 44 opt-in runtime.
Source-only discovery passed 56/skipped 44; repository/help/whitespace checks
passed. The existing synthetic CUDA optimizer test remains separate from the
frozen pretrained diagnostic.

Loading and bounded execution are complete. Appearance/domain compatibility and
native contact reliability fail the official learned-dynamics gate. Next isolate
appearance conversion from T physics adaptation with a frozen matched validation
protocol; preserve official policy inputs and test target retention/contact effects
before L adaptation. No imagined-policy or physical-policy improvement is claimed.

## Matched pretrained T palette experiment (2026-10-19)

Fixed local protocol before the first run:
`configs/gpc_appearance_probe.json`, SHA-256
`90909f757e117118cfb8ea6d8d9f8569d344eab285ea4e084f594dff3fd32b5a`.
It specifies six development scenes, four candidate sequences, four future controls
and three model-noise repeats. Bottom/left were previously inspected; right/top
and two rotated scenes extend this check, not the final evaluation set. The
checkpoint is unchanged from the preceding section; no new model assets,
dependencies, stored datasets or research training were needed.

```bash
.venv/bin/python -B scripts/probe_gpc_appearance.py --output artifacts/gpc-appearance-T-2026-10-19-v2
TDYSSEY_GPC_ROOT=third_party/gpc TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy TDYSSEY_OFFICIAL_CHECKPOINT=checkpoints/official-dp/ema_policy.pt TDYSSEY_GPC_WORLD_CHECKPOINT=checkpoints/gpc-world-model/denoiser.pth .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
```

Only the object/goal renderer colors change between the official views. Every
auxiliary render verifies that full physical/controller state, render cache and
original policy pixels remain unchanged. Color/screen cleanup also passes when
rendering raises an exception. Native GPC is a separate physics control; matching
initial geometry does not equate later physical trajectories. All views/noise
repeats share each actual physical branch. Future images/geometry are labels,
not prediction inputs. Auxiliary rendering needs privileged simulator access and
is not an image-only deployment adapter or learned adaptation.

The probe uses fixed known-color masks accepting both object palettes and both
target palettes. Full/ROI RGB errors compare within each palette; they cannot
alone compare motion across palettes. Color-mask IoU/centroids and target recall
are image proxies, not independent physical-state/reward predictions. Missing
centroids have explicit counts, and each control horizon is summarized separately.
The current experiment has no early success; all rows reach horizon four.

| Official physics, horizon 4 | Original input | Source palette | Persistence in source palette |
| --- | --- | --- | --- |
| Contact object-mask IoU | 0.73144 | 0.86106 | 0.51158 |
| Contact centroid error (pixels) | 10.90145 | 3.27174 | 14.56811 |
| No-contact centroid error (pixels) | 6.99619 | 0.20519 | 0 |
| All visible-target recall | 0.03314 | 0.95403 | 0.93595 |
| All visible-target IoU | 0.03123 | 0.85901 | 0.88894 |

Each official view has 24 physical branches × three model-noise repeats: 18
contact and 54 noncontact horizon-four predictions. These are six unique contact
branches, not 18 independent interactions or trained seeds. Across the three noise
repeats, source-palette contact object IoU means were 0.85945/0.86122/0.86251;
centroid errors were 3.368/3.293/3.154 pixels. Native source-palette contact IoU
was 0.85764 and centroid error 3.09212. Repeats are not used to pick a best seed.

The palette-conditioned **left push remains a failure**: official contact angle
changed -9.835 degrees while the predicted object largely stayed still. Object
IoU was 0.451–0.456, worse than original-input 0.887–0.904. Target recall/IoU also
remain imperfect. The mean improvement therefore cannot be presented as a full
fix, reliable planning, generalization or policy success. No reward ranking or
gradient through the quantized sampler was validated.

Completed run: **228 controls / 2280 control physics substeps / two reset
substeps / 60 restores / 288 auxiliary renders / 864 forecasts / 2592 denoising
steps**, **34.5147 seconds**, peak GPU allocation **110,936,064 bytes**. All 48
physical branches are archived once as compressed uint8 history/future frames
with exact actions, plus state labels in the report. The 48 archives use 370,834
bytes. Total retained output: **2,233,203 bytes (2.13 MiB)**. All archived file
hashes and report fingerprints match the completed source/protocol at parent
`9a47f67`. Original policy and world-model weights remain unchanged.

The first run at `artifacts/gpc-appearance-T-2026-10-19/` is explicitly **failed**:
it stopped at the native image-identity check after 114 controls, two resets,
31 restores, 145 auxiliary renders and 576 forecasts / 1728 denoising steps
(23.4650 seconds). NumPy division versus torchvision float32 multiplication
differed by at most `5.9604645e-8`, with identical uint8 pixels. Matching the
source conversion fixed the implementation; the protocol/thresholds were unchanged.
The failed report and 24 source archives remain, using 1,209,688 bytes. Together,
failed/completed experiments used **342 controls and four reset substeps**, with
unit tests and a one-reset/zero-control conversion diagnostic counted separately.

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| Completed `report.json` | 1648899 | `6905628627c00984140585cf5980137900b1512b82aee2018d37b2a411bca536` |
| `appearance-comparison.png` | 213470 | `40eafde5f16f6d366757a7db282ed9a7e217baab2dafcaca1b37d0f851dbb33f` |
| Failed first `report.json` | 965099 | `8b32f0868ec32bee6e4259a023bfe33b8245c31e965f37a8d8cece4a32a66072` |

All **101 tests passed** together in **13.842 seconds**: 56 source-only and 45
opt-in runtime checks. One focused new test checks palette cleanup, state/policy
image preservation and image-proxy sensitivity to physical displacement/rotation.
Source-only discovery passed 56 and skipped 45; repository, CLI-help, protocol JSON
and whitespace checks passed. Existing synthetic CUDA optimizer updates remain
separate from this frozen-model experiment.

Next use bounded official-RGB T training episodes with varied contact/rotation
to calibrate the pretrained predictor. Compare initial/adapted weights on disjoint
validation scenes; preserve the final model-test split and original policy inputs.
Keep the renderer-only palette result as an explanation/comparison, then assess
L adaptation and actual physical action effects before imagined-policy claims.

## Selected evidence publication (2026-10-19)

Published four original PNG figures, three compact JSON report selections and a
manifest under `docs/media/world-model-2026-10-19/`: **775,943 bytes (0.74 MiB)**.
See [the review index](RESULTS.md). Images were copied byte-for-byte from the
recorded runs. Each selected report names the exact source report with its size
and SHA-256; the palette summary includes failed first-run accounting and per-scene
push predictions as well as aggregate results. Successes and remaining failures
are visible without runtime setup. The 27 alphabet MP4s remain published separately.

The source checker verifies file membership, sizes, hashes, report provenance
and the 1 MiB snapshot cap using the standard library. Selected metrics/fields and
all image bytes were compared with the actual local originals before publication.
No experiment, model, protocol, environment or dependency changed, and no additional
unit tests were added for presentation. Full raw reports, data/frame archives,
weights and caches remain ignored/local; no shared raw archive is configured.

## Exact policy decision replay and near-miss inspection (2026-10-20)

Command (existing CUDA/DP installation, no downloads or training):

```bash
.venv/bin/python -B scripts/inspect_official_policy.py \
  --report artifacts/official-alphabet-2026-10-19/report.json \
  --shapes T D T_narrow \
  --output artifacts/policy-decisions-2026-10-20 \
  --seconds 420 --video
```

The completed run took **143.085 seconds** for model/replay computation and
**147.960 seconds** including export. It verified **616 original physical controls,
77 CUDA policy calls and 7700 denoising steps**: T through 128, D through 216,
near-T through 272. All sampled executed actions matched the original saved
float32 actions exactly. Numeric body/controller states and geometry metrics
matched within `1e-7`; rewards within `1e-9`; contacts and termination matched.
Eight selected plans retain all 101 sampler states (initial noise plus 100 steps).

The posthoc hold intervention separately replayed **463 prefix controls** and
executed **16 hold controls**. All experiment branches together used **1095 control
steps / 10950 physics substeps, five reset substeps and five initial-state restores**.
No mid-contact restore, object teleport or velocity reset was used. Test work is
separate from these counts.

| Shape | Best control | Best IoU / coverage | Position in 8-action chunk | First IoU loss before replan |
| --- | --- | --- | --- | --- |
| T | 128 | 0.924145 / 0.960577 | 8, controls 121-128 | None; upstream success |
| D | 201 | 0.815351 / 0.898285 | 1, controls 201-208 | 202 |
| Near-T | 262 | 0.871654 / 0.931426 | 6, controls 257-264 | 263 |

Neither failing shape reaches the strict coverage > 0.95 stopping rule. Its first
loss happens while executing a plan sampled before that best state, before the
policy can condition a new plan on the improved image. This identifies the timing
of lost progress, not the feature-level cause of the unsuitable plan.

| Posthoc intervention after known best moment | IoU after holding 8 controls | Original continuation at same endpoint |
| --- | --- | --- |
| D, control 201 to 209 | 0.825425 | 0.680307 |
| Near-T, control 262 to 270 | 0.860574 | 0.758507 |

Holding preserves more overlap in these two cases; **neither hold branch succeeds**.
It is privileged posthoc analysis, not a usable stop detector, a modified official
success score or evidence of adaptation. More frequent replanning remains an
untested follow-up, with additional compute and changed noise consumption to report.

The final local directory uses **5,865,462 bytes (5.59 MiB)**, including the offline
HTML viewer, eight compressed NPZ traces, three overview PNGs, three MP4s, the raw
report and an inspected video-preview PNG. Generated outputs remain ignored and
are not a new publication exception. `report.json` is **29,631 bytes**, SHA-256
`020f3a7de55e3bd86606786b34fb7c987b714b9e7c3c1dd062373a36d6ac10f6`.
It identifies the original full alphabet report and all source/trace hashes.

Verification:

- Source-only discovery: **58 passed, 45 runtime checks skipped** (103 discovered).
- Three focused official simulator/checkpoint tests passed in **17.534 seconds**.
  Existing CUDA parity coverage now checks traced actions, unchanged next-plan RNG,
  displayed final-prefix equality, and scheduler restoration. Two new source-only
  checks cover within-chunk peak timing and cleanup after a sampling error.
- All three exported MP4s decoded completely: **179 frames each**. Overview and
  video-preview images were visually inspected; embedded viewer data parsed.
- Generated JavaScript syntax passed. Plan selection, both sliders and both
  animation handlers passed an eight-view DOM/canvas-stub smoke check. This is
  **not a browser rendering check**; no browser surface was available in the session.
- Repository source/manifest/link and whitespace checks passed. No environment,
  dependency, upstream source or checkpoint was changed.

See [interpretation and controls](POLICY_DIAGNOSTICS.md). The official checkpoint
is already trained on T; these are frozen-policy diagnostics, not an untrained
network or a policy-learning experiment.

## Original T continued for 500 controls after ignoring the success stop (2026-10-20)

```bash
.venv/bin/python -B scripts/evaluate_official.py \
  --episodes 1 --steps 500 --continue-after-success \
  --seed 100000 --policy-seed 42 --seconds 300 --video \
  --output artifacts/official-T-500-continuation-2026-10-20
```

Completed exactly **500 physical controls**, including the pre-success prefix,
at 10 Hz simulated control time. Counts: **5000 control physics substeps, one
reset substep, 63 CUDA planning calls and 6300 denoising steps**. The last chunk
executes four controls. Evaluation took **121.926 seconds**, total including the
plot/export **123.065 seconds**. No training, asset download or dependency change.

Initial state, derived policy seed and **all 128 original pre-stop rows matched
the prior official baseline exactly**, including actions, physical states,
contacts, reward, metrics and raw done flags. No reset, velocity change, artificial
hold or history/RNG restart occurred at success. All **501 MP4 frames decoded**.
The coverage/IoU figure and extracted success/later frames were visually inspected.

| Control | Coverage | IoU | Raw original success |
| --- | --- | --- | --- |
| 128 | 0.960577 | 0.924145 | True, first crossing |
| 129 | 0.921633 | 0.854656 | False |
| 200 | 0.808109 | 0.678005 | False |
| 300 | 0.760216 | 0.613184 | False |
| 400 | 0 | 0 | False |
| 500 | 0 | 0 | False |

Only controls **128 and 182** exceed the original strict coverage > 0.95 rule.
There are 372 controls after first success; only one of those retains/re-enters
success (1/372 = 0.002688). Coverage and IoU first reach zero after success at
**363**, and remain zero through 500. Final success is false. The report's
`success: true` and `success_fraction: 1.0` mean **ever reached success**; the
`post_success` fields separately record final success and retention. The report
has a distinct continuation-diagnostic kind and is not a paper-score evaluation.

This gives an original-T control case for lost progress. Here first success is
the last action in controls 121-128 and the first loss occurs under the next plan
at 129. That differs from the D/near-T within-chunk loss documented above.
The observation does not prove that the policy fails generally, that chunk length
alone causes either failure, or that unseen-shape adaptation is unnecessary. The
original task normally stops on success; stability beyond that stop is a different
diagnostic question. This experiment covers one development scene/policy seed.

Local outputs include the annotated `T-100000.mp4`, `continuation.png`, raw report,
verification record and extracted frame comparisons. They remain ignored, with
no new publication exception. Raw report: **698,228 bytes**, SHA-256
`3bec784ac818cad45ddfc1fc7c1ae77e208a585f8c0aa0939320f28a421b745b`.
Final output directory: **2,444,337 bytes (2.33 MiB)**.

Focused verification: three policy-diagnostic source tests passed, including loss,
reentry and exclusion of the first success from retention counts. Two installed
official simulator checks passed (CUDA parity was skipped in that focused call);
the real 500-control CUDA run and exact historical-prefix comparison supply the
inference evidence. The simulator check also confirms that raw stepping after
success preserves the original reward/flag instead of resetting or absorbing.
Source/CLI-help and whitespace checks passed. A subsequent empty-episode plotting
guard and early diagnostic metric labeling do not change this completed rollout.
Final dependency-free discovery passed **59 tests** and skipped **45 runtime
checks**, with repository source/link and whitespace checks passing as well.

## Learning-plan and diagnostic PR verification (2026-10-22)

The shared handoff now documents existing versus planned components, proposed
data/budgets, prediction checks, reward/DPPO losses, developer work packages and
the simulator-only versus world-model-assisted comparison. This is documentation
of future experiments, not a new policy-training result.

Verification in the existing installed environment, with no new packages/assets:

```bash
python3 -B scripts/check_repo.py
python3 -B -m unittest discover -s tests -p 'test_*.py'
TDYSSEY_GPC_ROOT=third_party/gpc TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy TDYSSEY_OFFICIAL_CHECKPOINT=checkpoints/official-dp/ema_policy.pt TDYSSEY_GPC_WORLD_CHECKPOINT=checkpoints/gpc-world-model/denoiser.pth .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
git diff --check
```

Source discovery passed **59**, skipped **45** (104 discovered). Installed
discovery passed **all 104 tests**, no skips, in **34.468 seconds**. It covers
official CUDA inference/current-next-plan trace parity, original simulator
continued stepping, data/physics checks, pretrained prediction compatibility and
the existing tiny supervised world-model update check. It does not implement or
validate a DPPO update. The earlier measured diagnostic rollouts were not repeated
for this documentation update. Repository source/link and whitespace checks passed.

## Direct tail-transition PPO: first CUDA update (2026-10-23)

This is the DPPO-style variant specified in [policy learning](POLICY_LEARNING.md),
not an upstream DPPO reproduction. No world model or new human demonstrations
are used. New files run in the existing CUDA/DP environment without installation.

```bash
python3 -B -m unittest discover -s tests -p 'test_rl_protocol.py' -v
TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy TDYSSEY_OFFICIAL_CHECKPOINT=checkpoints/official-dp/ema_policy.pt SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -B -m unittest discover -s tests -p 'test_dppo_runtime.py' -v
.venv/bin/python -B scripts/train_official_policy.py --config configs/dppo_smoke.json --output artifacts/dppo-smoke-2026-10-23
```

Three independent source checks passed: discounted shaping telescopes with correct
terminal potential, timeout bootstrap does not leak GAE across a reset, and short
prefixes use physical discount/lambda. Two installed CUDA checks passed in 19.249
seconds: DDPM conditional means/variance match the pinned scheduler exactly,
joint log density matches `Independent(Normal(...), 2)`, deterministic t=0 is
excluded, original actions/following RNG match exactly, recorded likelihoods
recompute exactly, gradients update intended weights, frozen weights remain
unchanged, and reloaded deltas reproduce actions under the same RNG.

The end-to-end smoke run completed 64 training controls / eight plans and two
optimizer minibatches. Before/after physical evaluation used 512 controls / 64
plans total, capped at 128 controls per shape per evaluation. Checkpoint action
verification used two additional plans. Total: 576 physical controls / 5,760
control-physics substeps, six reset-physics substeps and 7,400 denoising steps.
Elapsed: 118.643 seconds; peak allocated CUDA memory: 1,194,854,912 bytes.

The 1,313,282-parameter final convolution head changed by L2 norm 0.0161563;
all frozen parameters were byte-identical. Saved actor/critic/Adam state is
16,176,490 bytes, with exact reloaded-action parity. Output totals 16,345,516 bytes.

| 128-control development evaluation | Frozen initial peak/final coverage | Updated peak/final coverage |
| --- | --- | --- |
| T | 0.841069 / 0.773810 | 0.839500 / 0.807566 |
| L | 0.410331 / 0 | 0.410384 / 0 |

Neither evaluation succeeds under this short budget. The training reward was
zero for these first 64 controls; updates used the initial value estimates and
their bootstrapped advantages. This checks mechanics, not meaningful adaptation.
The evaluation uses a derived policy seed different from the earlier canonical
alphabet rollout and a shorter horizon; do not compare their success counts.

Raw report remains ignored: 169,026 bytes, SHA-256
`ef2c63d4518e9caf59293c093c090736b45d175df0f93c3ba5e40002edc0578f`.
Checkpoint SHA-256:
`37f38e81ecce57e6bbc6082059a79db0b5f106a4c9da242e168484583fbca8f1`.
Source hashes in the report identify the exact uncommitted implementation used.
Full action-network updates and automatic resume have not been runtime-validated.

### Short learning pilot, same day

```bash
.venv/bin/python -B scripts/train_official_policy.py --config configs/dppo_local_pilot.json --output artifacts/dppo-local-pilot-2026-10-23
```

The executed config was an identical temporary JSON; the checked-in local profile
preserves its values. Completed 16 update rounds / 64 optimizer minibatches, with
1,008 training controls: T 500, L 508. Early completion/short prefixes leave this
below the 1,024-control cap. Two T training episodes succeeded; no L training
episode did. Rewards became nonzero after the first batch, including positive
progress and lost progress. All minibatches had finite nonzero actor gradients;
maximum sampled approximate KL was 0.000330605, with no KL early stops.

| 300-control paired development scene 100000 | Frozen initial | Updated |
| --- | --- | --- |
| T success / controls | True / 270 | True / 297 |
| T peak/final coverage | 0.950411 / 0.950411 | 0.957375 / 0.957375 |
| L success / controls | False / 300 | False / 300 |
| L peak/final coverage | 0.410331 / 0 | 0.409394 / 0 |

No L adaptation benefit is established; T completion is retained on this case
but is slower. One training seed, one physical development scene and a small
head-only budget cannot establish broader learning or retention. These scenes
are known development cases, not untouched final tests.

Before/after evaluation used 1,167 controls / 148 plans. Including training and
two checkpoint verification plans: 2,175 physical controls / 21,750 control-physics
substeps, ten reset-physics substeps and 27,800 denoising steps. Elapsed 431.609
seconds; peak allocated CUDA memory 1,194,854,912 bytes. Actor change L2: 0.121798;
all frozen weights unchanged and saved/reloaded actions exactly equal.

Raw output stays ignored: report 770,949 bytes, SHA-256
`f50cd13f41901ebed50ce0f7d4d4d110793533c7834d67c5b4027394b8566532`;
checkpoint 16,176,490 bytes, SHA-256
`874057b195187f3710b5202ea5174f36d7845b9d21d8f35cf5d6e858a3aec95e`.
Combined output: 16,947,439 bytes. Together with the smoke run, retained outputs
are 33,292,955 bytes. No new environment, dataset, model download or cache was
created. The full-network profile is a next runtime check on a suitable machine.

Final verification: dependency-free discovery passed 62 / skipped 47 (109 total).
The full installed command from the preceding PR-verification section passed all
109 tests, no skips, in 30.882 seconds. Source/link/syntax, CLI help, all five
configuration profiles and whitespace checks passed. No full-network performance,
multi-seed adaptation, imagined RL or theorem of policy improvement is claimed.

## LoRA and trained-policy desktop previews (2026-10-23)

Added native-PyTorch Conv1d low-rank factors, with a head or distributed action-U-Net
target. Zero-initialized B preserves original inference; the head profile trains
25,604 parameters. It does not train the visual encoder or change the PPO objective.

```bash
.venv/bin/python -B scripts/train_official_policy.py --config configs/dppo_lora_smoke.json --output artifacts/dppo-lora-smoke-2026-10-23
.venv/bin/python -B scripts/evaluate_trained_policy.py --adapter artifacts/dppo-local-pilot-2026-10-23/last.pt --reference-report artifacts/dppo-local-pilot-2026-10-23/report.json --shape L --steps 300 --display --window-size 768 --hold-seconds 30 --output artifacts/dppo-local-pilot-2026-10-23-L-preview.json
# The T preview used --shape T, --hold-seconds 60 and a new T-preview.json path.
```

The LoRA smoke completed 64 training controls, two optimizer minibatches and
512 evaluation controls. Eight training / 64 evaluation / two checkpoint plans,
six reset substeps and 7,400 denoising steps. Elapsed 118.994 seconds; peak allocated
CUDA 1,166,400,512 bytes. Base/frozen parameters were unchanged and checkpoint
reload reproduced actions exactly. All initial baseline traces match the earlier
head-only smoke's initial traces exactly. The first 64 training rewards were zero;
this is an update check, not an adaptation benefit.

Checkpoint including factor/critic/Adam state: 722,458 bytes; total output 891,599
bytes. Checkpoint SHA-256
`af13f4ebbdec2f0f37360039e464d87a60c8b5eef0d624e3bdaba13b94c29e50`;
report SHA-256 `dfd4284d4df4acbf6bd95038dc95f1c87b4c8518a6fa337b4d831b000509c7bf`.
The earlier head run took 118.643 seconds: no overall speed gain is established
by reducing trainable parameters. Sampling and evaluation are unchanged.

Four focused CUDA checks passed in 28.130 seconds. They include independent
low-rank-versus-dense kernel equivalence (with stride/padding), zero-initialized
parity, gradients on both factors with frozen base kernels, and distributed LoRA
original-sampler equality, actual optimizer changes and compact saved-action reload.
The distributed check uses synthetic advantages, not physical task learning.

Separate desktop previews loaded the trained head checkpoint. All L 300 and T 297
actions, rewards, coverage/IoU and raw success flags matched the saved final
evaluations. T succeeded, L did not. Extra preview work: 597 controls / 5,970
control-physics substeps, four reset substeps, 76 plans / 7,600 denoising steps.
Elapsed L 118.555 / T 149.150 seconds includes real-time display and 30/60-second
final holds, so it is not a training-throughput measurement. The requested
768-pixel desktop display was enabled; no MP4 was exported. Output reports are
58,369 / 66,267 bytes, ignored separately from the original training outputs.

Final LoRA/preview verification: source discovery passed 62 / skipped 49 (111
total); full installed CUDA/DP/GPC discovery passed all 111 in 42.356 seconds.
All eight RL configurations, source/link/syntax checks, CLI help and whitespace
checks passed. Full dense action-network execution, distributed-LoRA physical
learning, visual-encoder updates and any novel tuning algorithm remain unestablished.

## Forward RL, recovery and mixed-shape checks (2026-10-23)

The [runnable forward guide](FORWARD_POLICY_LEARNING.md) describes the separate
DDPM-NFT/AdvantageFlow implementation. Checks used the existing Linux/Python
3.12.3, PyTorch 2.5.1+cu124, official/GPC environment and RTX 4060 Laptop GPU.
No package, environment, base checkpoint or dataset download was added. Runs
recorded source hashes and dirty working-tree provenance based on `71a8050`;
that commit alone is not claimed to contain the implementation.

Verification commands:

```bash
python3 -B scripts/check_repo.py
python3 -B -m unittest discover -s tests -p 'test_*.py'
TDYSSEY_GPC_ROOT=third_party/gpc TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy TDYSSEY_OFFICIAL_CHECKPOINT=checkpoints/official-dp/ema_policy.pt TDYSSEY_GPC_WORLD_CHECKPOINT=checkpoints/gpc-world-model/denoiser.pth SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
```

All **120 installed tests passed in 61.286 seconds**, with no skips. Source-only
discovery passed **64** and skipped **56** runtime tests. A subsequent focused
four-test forward run passed in 25.679 seconds, explicitly loading resume state
on CPU before Adam restores CUDA moments, preserving CPU step scalars. The
checkpoint-continuation comparison isolates serialization/RNG from convolution
backward nondeterminism; normal training does not force deterministic kernels.
All nine forward configurations parse. Source/links/AST and whitespace checks
pass; none of these checks installs dependencies.

Coverage includes independent loss derivatives/positive curvature, neutral
feedback, official unchanged initial actions/RNG, immutable scheduler buffers,
real distributed-LoRA updates for both losses, frozen-base integrity, refreshed
rollout weights, optimizer/RNG restoration and fresh-model action parity. Real
physics tests replay established contacts and detect altered contact records;
native presentation rendering preserves physical state and 96px observations.
Mixed-shape mask/label and train-only normalization checks also passed.

### Corrected NFT development pilot

```bash
.venv/bin/python -B scripts/train_forward_policy.py --config configs/nft_local_pilot.json --output artifacts/nft-local-pilot-v2-2026-10-23
```

This updates **1,652,824 distributed Conv1d LoRA parameters** across the action
U-Net, with frozen vision and base weights. It completed six updates / 88
optimizer minibatches and stopped when no complete candidate group fit the
remaining 4096-control cap. Actual training cost: **3712 controls**, comprising
704 candidate controls, 2944 full-prefix replay controls and 64 policy warmup
controls. Eleven groups compare four candidates each with up to 16 scored
controls. Evaluation used another 512 controls / 64 plans; checkpoint parity
used two plans. Total: 4224 physical controls / 42,240 control-physics substeps,
six reset substeps and 16,800 denoising steps including six T-reference queries.

Elapsed **295.393 seconds**; peak allocated CUDA **1,194,914,816 bytes** (about
1.11 GiB). Actor-factor L2 change **0.0489364653**; frozen weights unchanged and
checkpoint action reload passed. Results below use one paired development scene
(seed 100000) and a **128-control evaluation cap**, not an untouched final test.

| Shape | Before peak / final coverage | After peak / final coverage | Success before / after |
| --- | --- | --- | --- |
| T | 0.841069 / 0.773810 | 0.842750 / 0.767869 | No / no within 128 controls |
| L | 0.410331 / 0.000000 | 0.409805 / 0.000000 | No / no |

**No L adaptation benefit is established.** The short T cutoff is not evidence
that the original policy cannot solve T. Four of five L candidate groups had
zero contact controls and exactly tied scores; only the first L group supplied
differentiated scores. This identifies a data/score-signal limitation to investigate,
not proof that one particular exploration change will solve it. More data,
longer-horizon feedback and repeated held-out scenes remain necessary.

Ignored pilot outputs:

- `last.pt`: 27,053,605 bytes; SHA-256 `e8d19198bf71c39902016b63a8720402855659bae80510cc4e08534c5e3c0007`.
- `report.json`: 1,879,922 bytes; SHA-256 `01646ebfdfabe6cc4fd92f7cd38c016eab35eeb96d5b45ed16cf515c763031aa`.

### Corrected AdvantageFlow and resume checks

```bash
.venv/bin/python -B scripts/train_forward_policy.py --config configs/advantage_flow_smoke.json --output artifacts/advantage-flow-smoke-v2-2026-10-23
.venv/bin/python -B scripts/train_forward_policy.py --config configs/advantage_flow_resume_check.json --resume artifacts/advantage-flow-smoke-v2-2026-10-23/last.pt --output artifacts/advantage-flow-resume-2026-10-23
```

The first run used 32 candidate controls, 128 evaluation controls and 2300
denoising steps in **43.181 seconds**. Scores tied, gradients and factor change
were zero, and before/after physical actions and outcomes matched exactly.
This is a meaningful no-signal case, not an improvement result.

Resume added a second update in a new directory. The actual run used a temporary
configuration identical to the tracked resume-check profile. New-session cost:
64 training controls (32 candidate / 32 replay), 64 evaluation controls, two
resets and 1500 denoising steps in **30.434 seconds**. Cumulative lineage: 96
training controls, 192 evaluation controls, eight reset substeps, eight optimizer
minibatches and 3800 denoising steps. The original baseline was reused, not rerun.
Parent cost audit found no discarded controls; checkpoint parity and unchanged
neutral-policy physical traces passed. Peak CUDA matched the NFT pilot.

Parent checkpoint: 26,580,709 bytes, SHA-256
`0673f691f9324e1b2e9911c13dd275062254ce22f37dd5e2bcb1568d398968e3`.
Resumed checkpoint: 26,606,565 bytes, SHA-256
`614ed095449448d46ba88527a005bcc8ee2398e10e2afaa6fd079cadad62c5dd`.
The resumed report is 228,663 bytes, SHA-256
`dab2520fe032a2327c26acdbf4b4b2457fdd83be779463176f1ccef5d294edd3`.

### Fresh-process video/evaluation

```bash
.venv/bin/python -B scripts/evaluate_trained_policy.py --adapter artifacts/nft-local-pilot-v2-2026-10-23/last.pt --shape L --steps 128 --reference-report artifacts/nft-local-pilot-v2-2026-10-23/report.json --video artifacts/nft-trained-L-native-2026-10-23.mp4 --output artifacts/nft-trained-L-native-preview-2026-10-23.json
```

The fresh process reproduced every saved L action, reward, coverage, IoU and
termination value. Native video decoded **129 frames at 512x512**, 283,177 bytes,
SHA-256 `ea7547ad61f962c2170c34b78287076e1aa97862234b30b2ed4cf23f153e4067`.
Elapsed 31.608 seconds. Policy observations remain 96x96. This preview added 128
physical controls / 1280 substeps, two reset substeps and 1600 denoising steps;
it trained no weights. An earlier 96x96 export also matched all 128 actions and
decoded 129 frames (55,853 bytes, 31.235 seconds), adding the same separate work.
Neither preview was a GUI visual inspection or a successful L rollout.

### Multi-shape primary data

```bash
.venv/bin/python -B scripts/collect_official_play.py --shape Y --episodes 7 --steps 8 --max-mib 8 --output artifacts/official-play-Y-check-2026-10-23
.venv/bin/python -B scripts/collect_official_play.py --validate-only --output artifacts/official-play-Y-check-2026-10-23
.venv/bin/python -B scripts/prepare_training_data.py --datasets artifacts/official-play-L-2026-10-19 artifacts/official-play-Y-check-2026-10-23 --output artifacts/official-mixed-LY-check-2026-10-23.catalog.json --history 2 --future 4 --max-batches 1
```

Y collected/read back **56 controls**, 560 physics substeps, seven reset substeps
and five contact controls. Frozen splits: 5/1/1 episodes; **272,344 bytes**.
Manifest SHA-256 `c5ceff4677c41eddc96decd0ffeaa8b362bd4d9a16d948cef2e5df013827704e`.
The mixed real L/Y catalog gives 596/100/100 windows and 11/2/2 episodes across
train/validation/model-test. Actual L and Y visual-dynamics samples loaded with
their own geometry; fitted normalization used only **640 unique train controls**.
Model-test pixels were not used for this check. This does not establish reliable
multi-shape dynamics. The pretrained GPC wrapper still has no adaptation trainer.

### Superseded prototype work

The earlier `nft-smoke-2026-10-23`, `advantage-flow-smoke-2026-10-23` and interrupted
`nft-local-pilot-2026-10-23` used the library's mutating noising helper. Moving the
scheduler's alpha buffer changed sampling rounding even without weight changes.
They are **excluded from policy comparisons**; the latter is explicitly failed.
Their reports remain ignored for accounting. The two prototypes each used 32
training and 128 evaluation controls; the interrupted one recorded 1712 training
and 256 evaluation controls. Its partial in-flight sampling work is not fully
counted. Elapsed: 65.678 / 43.112 / 162.547 seconds respectively.

Only these three superseded generated checkpoints were removed to save space;
their hashes/sizes and cost records are retained in ignored
`artifacts/forward-prototype-lifecycle-2026-10-23.json`. Valid corrected checkpoints
and original reports remain available locally. No raw output or new video is
published by this milestone; portable commands/results are shared in Git.

The three removed checkpoints freed **79,992,687 bytes**. Retained new training,
resume, data and preview outputs plus prototype reports/lifecycle metadata total
**84,493,136 bytes (80.58 MiB)**. Existing environments/base assets are unchanged.

## Portable training and live monitoring (2026-10-23)

Implementation and reproducible machine commands are in
[TRAINING_MACHINE.md](TRAINING_MACHINE.md). No package/environment installation or
model/data download was performed for this milestone. Metadata-only uv resolution
added cu128 and the independent tracking extra to the shared manifest/lock:

```bash
third_party/uv/uv lock --python .venv/bin/python --no-python-downloads --no-build-isolation --cache-dir /tmp/tdyssey-training-lock-cache
third_party/uv/uv lock --check --offline --python .venv/bin/python --no-python-downloads --no-build-isolation --cache-dir /tmp/tdyssey-training-lock-cache
python3 -B scripts/check_repo.py
python3 -B -m unittest discover -s tests -p 'test_*.py'
```

All 136 lock entries/source/link checks passed. Source discovery ran 123 tests:
67 passed, 56 runtime checks skipped. All 20 policy configurations parsed.
The task-owned 29 MiB metadata cache was removed after lock verification.

```bash
TDYSSEY_GPC_ROOT=third_party/gpc TDYSSEY_OFFICIAL_ROOT=third_party/diffusion_policy TDYSSEY_OFFICIAL_CHECKPOINT=checkpoints/official-dp/ema_policy.pt TDYSSEY_GPC_WORLD_CHECKPOINT=checkpoints/gpc-world-model/denoiser.pth SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -B -m unittest discover -s tests -p 'test_*.py'
```

All **123 installed tests passed in 63.353 seconds**. The added fixed probe checks
unchanged training RNG, original-policy drift zero before updates, nonzero drift
after an update and repeatable measurements. Source tests check flushed local
events, remote-service failure isolation, event byte caps, per-shape metrics,
duplicate job refusal and preservation of evaluation settings across methods/seeds.

The custom world-model fixture still corrupts model-test images to prove they are
not decoded by training/validation. Its one-step checkpoint resumes inside the
same partial epoch, reaching two steps/four windows. A separate uninterrupted
two-step run matches loss/weights within strict tolerance and exact CPU RNG/cursor.
The comparison temporarily enables deterministic GPU operations and restores the
flag; ordinary training does not force these slower paths. CUDA bilinear resize
backward can otherwise differ, as described by [PyTorch](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.interpolate.html).
The short-horizon metric uses explicit ordered sums so it also works in strict
deterministic mode. This is a synthetic recovery check, not learned contact accuracy.

### Physical forward tracking and recovery

```bash
.venv/bin/python -B scripts/check_training_setup.py --config configs/nft_smoke.json --backend cu124 --output-root artifacts --tracking
.venv/bin/python -B scripts/train_forward_policy.py --config configs/tracking_smoke.json --output artifacts/portable-training-check-2026-10-23 --tracking offline --run-group portable-validation --log-every 1
```

The read-only preflight passed package pins, CUDA/torchvision/convolution
gradients, every upstream source hash, the base checkpoint and available disk.
It performed no policy training or installation.

For recovery, copy `configs/tracking_smoke.json` to a new JSON file and set only
`updates=2`, `max_train_controls=1024`, then run:

```bash
.venv/bin/python -B scripts/train_forward_policy.py --config PATH/portable-resume.json --resume artifacts/portable-training-check-2026-10-23/last.pt --output artifacts/portable-training-resume-2026-10-23 --tracking offline --run-group portable-validation --log-every 1
```

| Run | Training controls | Evaluation controls | Optimizer minibatches | Session time |
| --- | --- | --- | --- | --- |
| Initial tracking check | 160 = 32 candidate + 64 prefix replay + 64 warmup | 128 | 2 | 57.417 s |
| New work during recovery | 192 = 32 candidate + 160 prefix replay | 64 | 2 | 32.186 s |
| Cumulative lineage | 352 | 192 | 4 | Two sessions; parent audit also charges 0.095 s finalization |

Peak allocated CUDA was **1,194,923,008 bytes (about 1.11 GiB)**. Actor factors
changed; frozen-parameter and saved-checkpoint action checks passed. Every update
ran fixed diagnostics and physical evaluation. With `report_detail_updates=0`,
reports pruned candidate/minibatch detail while preserving scalar summaries,
episode prefixes and recovery. Both runs' recorded training source hashes match
the final source files. The fresh process retained fixed probe data and cumulative
counts; the parent audit found no discarded controls/minibatches. Best checkpoint
carry-forward and current-invocation selection completed.

This is a **32-control development evaluation**, not evidence of useful learning:
T and L both remained unsuccessful at that short cutoff. Final L coverage changed
from 0.158075 to 0.158456 between the two trained evaluations. Do not interpret
that tiny one-scene difference as an adaptation benefit.

Offline W&B runs `d756f991` and `8a3c7a96` recorded per-shape, fixed-noise, training
and resource events. The two tracking invocations have distinct IDs. No online
authentication/upload was attempted. The local viewer's HTML and `/events` routes
returned 200, arbitrary report paths returned 404, and its JavaScript parsed in
V8. Browser appearance could not be inspected because no browser was connected.
Port 8765 was already occupied; the local check used 8897 without altering that
service. View the retained run with:

```bash
python3 -B scripts/view_training.py --run artifacts/portable-training-resume-2026-10-23 --port 8897
```

The six-job planner produced NFT/AdvantageFlow × seeds 42/43/44 with identical
evaluation settings, without creating its proposed output directory or starting
training. The full-network preset kept its intended dense scope. Extended jobs,
Blackwell runtime, dense full-network training and shared cloud access remain
unverified. No world-model-scored policy learning is added.

| Local artifact | Bytes | SHA-256 |
| --- | --- | --- |
| `artifacts/portable-training-check-2026-10-23/last.pt` | 26,608,509 | `0a48439edbbec7c3bcb98d18e713753bd8a69bce5c7cb7142be96a8b45a4d0c7` |
| `artifacts/portable-training-check-2026-10-23/report.json` | 177,703 | `750168d6cb1ea34ce8c155c6cb58d0612d848b79aaa22ddd0c354a073da2d120` |
| `artifacts/portable-training-resume-2026-10-23/last.pt` | 26,623,677 | `a47ff7ccf550cfe565a954f55ded0bd26984f8fba35d8b727e32342170af2ca0` |
| `artifacts/portable-training-resume-2026-10-23/report.json` | 224,989 | `5f3a48361fb4b54ae4b70238eab2c7d8a489e7bed2bb5fa4dac090204476ea04` |

Raw outputs remain ignored/local. Monitoring outputs retained about **102.20 MiB**
after removing two exclusively task-owned checkpoints from an earlier monitoring
check, freeing 53,209,658 bytes. Original reports/scalars/offline logs remain; the
hash-verified removals are recorded in ignored
`artifacts/portable-training-lifecycle-2026-10-23.json`. The earlier check cost
160 training / 128 evaluation controls and 57.273 seconds and is not part of the
final resumed lineage. Existing base weights, environments and older assets
were untouched. A Git hash alone does not transfer these local artifacts.
