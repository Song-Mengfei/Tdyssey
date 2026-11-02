# Runtime validation

The dated sections preserve separate machines and milestones. For the current
native Windows handoff, see [2026-10-21 validation](#windows-simulator-reranking-validation-2026-10-21).
Skipped checks are never included in passed counts.

The earlier checks began on 2026-10-19 with Linux x86_64, Python 3.12.3, uv 0.13.0, and the locked
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
shape protocol (external historical reference: `OFFICIAL_SHAPES.md`): height 120, bottom width 90, stroke 30, one
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
are disclosed in the shape protocol (external historical reference: `OFFICIAL_SHAPES.md`). There is still no
adaptation result. The following collection section records the subsequent data
milestone; learned optimization remains pending.

## Published alphabet video snapshot

All 27 unchanged MP4 files are committed under
`docs/media/official-alphabet-2026-10-19/`, totaling 12,025,483 bytes. Their copied
sizes/SHA-256 values match the decoded originals in `verification.json`.
Browse the video table (external historical reference: `ALPHABET_RESULTS.md`); `videos.json` beside the clips
records scene/noise seeds, original report hash, control costs and per-video hashes.
This explicit curated snapshot is an exception to the default raw-artifact policy.
No inference, replay, re-encoding, download or dependency installation was needed.

## Official L collection and training data (2026-10-19)

Implemented the official collector (external historical reference: `OFFICIAL_COLLECTION.md`), keeping the original
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

Implemented the visual model protocol (external historical reference: `WORLD_MODEL.md`) on the existing official
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
weights remain outside Git. See compatibility details (external historical reference: `WORLD_MODEL.md#pretrained-gpc-compatibility-check`).

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
See the review index (external historical reference: `RESULTS.md`). Images were copied byte-for-byte from the
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

## Frozen-policy physical-simulator ranking source milestone (2026-10-20)

Added optional batched official-policy candidates and simulator reranking; see
protocol and commands (external historical reference: `SIMULATOR_RERANKING.md`). The policy/checkpoint remain
frozen, and no D demonstrations, world model or optimizer are used. Separate
branches reconstruct the actual control history from a common initial snapshot.

Python 3.12 source-only discovery completed **117 tests: 68 passed, 49 skipped**.
The twelve new dependency-free planner tests cover history-sensitive fake dynamics,
candidate/live isolation, overshoot, progress before overlap, physical hold and
prefixes, remaining-control budgets, native termination, rejected samples,
reconstruction mismatch and partial deadline accounting. Repository source checks,
`evaluate_shapes.py --help` and `git diff --check` also passed.

The new real D contact-continuation test and three tensor/checkpoint candidate
tests were skipped along with the other installed-runtime checks. This source
checkout has no project PyTorch/Pymunk/Gym runtime, pinned upstream source or
checkpoint. No dependency installation, model asset download, new physical
experiment or training occurred. These results establish software protocol
behavior, not simulator/contact parity, batch policy parity, latency or improved
Push-D performance. Run the documented opt-in checks and the matched three-way
comparison on the authorized Linux/CUDA profile before making those claims.

## Windows simulator reranking validation (2026-10-21)

This is a new local runtime check, separate from the earlier host's 68 passed /
49 skipped source-only handoff. The complete patch was inspected and applied on
baseline `71d0dfc2b88ab49165e683197258980402179d8b` on branch
`codex/simulator-reranking-windows`. The previous Windows branch was preserved.
No policy/world model was trained and no D demonstration data was collected.

Environment: native Windows 11 AMD64, Python 3.12.3, uv 0.13.0, Torch
2.5.1+cu124, torchvision 0.20.1+cu124, CUDA runtime 12.4, NumPy 1.26.4,
Pymunk 6.9.0 and pygame 2.6.1. Actual inference used `cuda` on an NVIDIA
GeForce RTX 3060 Ti (8 GiB), driver 610.88, with four CPU Torch/OMP/MKL threads.
The existing environment now contains 98 packages using the combined cu124/dp/gpc
profile; fixed upstream source is commit `5ba07ac6661db573af695b419a7947ecb704690f`.
The published EMA checkpoint retains all 300 tensors/normalizers/buffers.

The shared lock explicitly supports Linux x86_64 and Windows AMD64. Robomimic's
unused MuJoCo EGL probe is retained on Linux only because it does not build on
native Windows. `uv pip check` therefore reports one expected missing
`egl-probe` dependency from Robomimic's unconditional metadata; it is not an
all-dependencies-compatible result. The tested PushT/CNN path does not import
those optional MuJoCo renderers. See environment details (external historical reference: `ENVIRONMENT.md`).

Checkpoint file/content hash comparison was removed at the user's request.
The loader retains tensor-only safe loading, published configuration, all 300
state entries, strict keys/shapes and frozen evaluation. Candidate tests compare
all weights/buffers directly before and after inference, without fingerprints.
Conversion also compares each saved tensor directly with its source. Historical
manifest checkpoint digests are provenance only, not acceptance gates.

### Commands and focused checks

Commands were run from the repository root using the existing Windows interpreter:

```powershell
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -B scripts/check_repo.py
.\.venv\Scripts\python.exe -B scripts/doctor.py --runtime --require-cuda --dp --gpc
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_planning.py' -v
$env:TDYSSEY_OFFICIAL_ROOT = 'third_party/diffusion_policy'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_planning_runtime.py' -v
Remove-Item Env:TDYSSEY_OFFICIAL_ROOT
$env:TDYSSEY_DATA_TESTS = '1'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_official_candidates_runtime.py' -v
$env:TDYSSEY_OFFICIAL_ROOT = 'third_party/diffusion_policy'
$env:TDYSSEY_OFFICIAL_CHECKPOINT = 'checkpoints/official-dp/ema_policy.pt'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_official_candidates_runtime.py' -v
.\.venv\Scripts\python.exe -B scripts/fetch_official_assets.py --download
```

| Check | Actual result |
| --- | --- |
| Repository source/configuration/docs | Passed; 119 locked package entries |
| Runtime doctor | Passed direct pins, actual CUDA calculations/gradients, torchvision NMS and Pymunk |
| Planner protocol | 13 passed, zero skipped |
| Real D simulator/contact/physical hold | 2 passed, zero skipped; 1.484 seconds |
| Torch candidates without checkpoint paths | 2 passed, 1 checkpoint test skipped; the skip is not a pass |
| Full candidate suite with actual CUDA checkpoint | 3 passed, zero skipped; 15.795 seconds |
| Existing compact checkpoint preparation path | `already_readable`; no download or hash gate |
| Broader source-only discovery | 119 discovered: 69 passed, 50 skipped; no installed-runtime claim for those skips |

The real D tests replay every actually executed control from the common initial
snapshot, reproduce continued contact and leave the live state unchanged.
Physical hold uses the attained pusher position as a fixed PD target; a moving
pusher decelerates through real physics. The test does not require velocity to
remain positive, since the PD controller can overshoot and reverse.

The actual CUDA test matches K=4 output exactly to the unchanged official
sampler and its unnormalized `action_pred[:, 1:9]`, yielding `(4,8,2)`.
Independent candidate noise, local RNG isolation, K=1 baseline/RNG equality,
unchanged 300-entry state, evaluation mode and disabled gradients all passed.

Runtime fixes included explicit UTF-8 reads, preserving existing reviewed JSON
bytes across Windows checkouts with `-text`, exact per-command Git safe-directory
settings, and rejecting forecasts whose final control finishes after the deadline.
Requested but unstarted shapes now appear explicitly in the report census.
Initial failed source/PD assertions and the first integration startup failure
(Git ownership while recording provenance, zero episode controls) remain in local
logs. Repaired checks were rerun; they are not presented as initial successes.

### Real T/D smoke

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluate_shapes.py --shapes T D --planner simulator --candidates 4 --prefix-lengths 1 4 8 --execute-steps 1 --steps 20 --seconds 300 --output artifacts/td-simulator-smoke-20261021-r2
```

Report status was **complete** in **95.203 seconds**. T and D each executed
20 actual controls; both requested episodes were completed, with no timeout.
Neither succeeded, and maximum/final coverage and IoU were all zero. T had three
contact controls and D had thirteen. Each shape used 240 branch restores,
2,114 historical replay controls and 1,971 forecast controls. Together there were
40 actual controls / 400 actual physics substeps, two reset substeps, 40 policy
calls, 160 sampled chunks and 16,000 denoising sample-iterations.

All 480 completed forecast branches reconstructed the live state exactly.
All 40 selected first predictions matched the following real full-state snapshot,
action, reward, done flag, contact count, coverage and IoU. No planner timed out.
These recorded predictions were checked against actual controls, not just a
second cold-restored snapshot.

### Matched A/B/C controls

Each condition uses T and D, one matched scene, `--seed 100000`,
`--policy-seed 42` (derived policy seed 3486894925), four CPU threads,
64 actual controls per shape and a run-wide 900-second cap. Initial physical
snapshots match per shape across conditions. The 64-control budget is only
6.4 seconds of simulated time, not a completed 300-control task evaluation.

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluate_shapes.py --shapes T D --pairs 1 --seed 100000 --policy-seed 42 --threads 4 --steps 64 --seconds 900 --planner none --execute-steps 8 --output artifacts/td-abc-20261021-A
.\.venv\Scripts\python.exe -B scripts/evaluate_shapes.py --shapes T D --pairs 1 --seed 100000 --policy-seed 42 --threads 4 --steps 64 --seconds 900 --planner none --execute-steps 1 --output artifacts/td-abc-20261021-B
.\.venv\Scripts\python.exe -B scripts/evaluate_shapes.py --shapes T D --pairs 1 --seed 100000 --policy-seed 42 --threads 4 --steps 64 --seconds 900 --planner simulator --execute-steps 1 --candidates 4 --planning-horizon 8 --prefix-lengths 1 4 8 --settle-steps 2 --output artifacts/td-abc-20261021-C
```

All three reports were **complete**: 64 controls per shape, no successful
episodes, no wall-time truncations and no planner timeouts/rejected samples.
Coverage is target-area overlap; IoU is intersection over union, a separate metric.

| Condition | Shape | Max / final coverage | Max / final IoU | Contact controls |
| --- | --- | --- | --- | --- |
| A: no ranking, execute 8 | T | 0.170203 / 0.170203 | 0.093018 / 0.093018 | 44 |
| A: no ranking, execute 8 | D | 0.206047 / 0.120604 | 0.114856 / 0.064172 | 33 |
| B: no ranking, execute 1 | T | 0.540012 / 0.275225 | 0.369874 / 0.159572 | 40 |
| B: no ranking, execute 1 | D | 0.146233 / 0.133399 | 0.078884 / 0.071466 | 52 |
| C: simulator, execute 1 | T | 0.553329 / 0.550897 | 0.382484 / 0.380164 | 36 |
| C: simulator, execute 1 | D | 0.188766 / 0.188766 | 0.104220 / 0.104220 | 30 |

| Condition | Policy calls T / D | Mean policy seconds T / D | Mean total planning seconds T / D | Run wall seconds |
| --- | --- | --- | --- | --- |
| A | 8 / 8 | 1.565 / 1.558 | 1.565 / 1.558 | 34.359 |
| B | 64 / 64 | 1.484 / 1.439 | 1.484 / 1.439 | 198.125 |
| C | 64 / 64 | 1.768 / 1.700 | 2.145 / 2.721 | 322.719 |

A and B used no additional simulator branches. C used **812 restores, 24,972
historical replay controls and 7,691 forecast controls per shape**: 65,326 extra
simulator controls / 653,260 physics substeps total, separate from its 128 actual
controls. C sampled 512 chunks with 51,200 denoising sample-iterations, versus
128 chunks / 12,800 sample-iterations for B and 16 / 1,600 for A. Terminal-horizon
clipping deduplicated equivalent short prefixes near the 64-control limit.
All 1,624 C branch reconstructions had zero state error; all 128 selected first
predictions matched the next real state/action/metrics/flags exactly. C selected
hold on 20 T controls and 13 D controls. Holding did not create a success.

Shortening execution alone (B versus A) improved T overlap but reduced D's peak.
Adding the K=4 simulator controller (C versus B) retained more final overlap on
both shapes, yet D's peak remained below A and no condition solved either task.
This single development scene does not establish a reliable benefit. C adds
candidate noise/compute, privileged hold and physical scoring together; K>1 does
not retain the sequential K=1 random stream. It is not an isolated compute-matched
ablation of only the ranking score, learned adaptation or real-time control.

### Interpreting the earlier D video

The committed D example (external historical reference: `media/official-alphabet-2026-10-19/D-100000.mp4`)
contains 301 frames at 10 fps: 300 controls plus frame zero, 30.1 seconds total.
The earlier report records peak target coverage **89.83%** / IoU **0.8154** at
control **201**, or video time **20.1 seconds**; final IoU was 0.4172 at 300
controls. It never crossed the original success threshold. The A64 maximum
20.6% above only describes the first 6.4 simulated seconds, not that full attempt.
Both use scene 100000 and derived policy seed 3486894925. Historical baseline
sampling, reset and geometry code match the current path; the earlier full raw
report is unavailable here, so exact old/current numeric traces cannot be asserted
from video alone, especially across GPU/OS changes.

Actual A64 T/D replay was displayed locally and exported with 65 decodable frames
per shape. Every replay control matched recorded state within 1e-7 and reward
within 1e-9; this diagnostic adds 128 replay controls and two reset substeps,
separate from the experiment totals. Local videos are raw artifacts, not Git content.

### Full-budget baseline follow-up

After the matched comparison, the original eight-action baseline was rerun with
the full 300-control budget and a visible window to investigate the old D video:

```powershell
.\.venv\Scripts\python.exe -B scripts/evaluate_shapes.py --shapes T D --planner none --execute-steps 8 --seed 100000 --policy-seed 42 --steps 300 --seconds 600 --threads 4 --display --video --hold-seconds 30 --output artifacts/td-baseline-300-20261021
```

The report completed in **124.531 seconds** excluding the final display hold,
without timeout: 428 real controls / 4,280 physics substeps, two reset substeps,
54 policy calls / 5,400 denoising sample-iterations. This is a supplementary
baseline, not an extension of the 64-control B/C comparison.

| Shape | Controls / result | Max / final coverage | Max / final IoU |
| --- | --- | --- | --- |
| T | 128 / native success | 0.953273 / 0.953273 | 0.910718 / 0.910718 |
| D | 300 / control-budget failure | 0.389221 / 0.249621 | 0.241636 / 0.142609 |

T stopped immediately at native success. D had 136 contact controls and reached
its peak at control 88. At control 201 its coverage was only **0.206887**
(IoU 0.115379), so the historical **89.83% near miss was not reproduced**.
The 64-control truncation explained why the first comparison was incomplete;
it does not explain the remaining full-budget difference.

Both shapes' initial snapshots and first 64 actual actions, full physical states,
done flags and contacts exactly match A64. D metric roundoff across these two
local runs is at most 1.11e-16; no physical mismatch was found. New videos decode
to 129 T frames and 301 D frames at 10 fps. The old/new D videos' decoded frames
0, 1 and 8 are pixel-identical. Frame 16 differs by mean absolute intensity 0.0331,
frame 64 by 0.3057 and frame 201 by 4.7061 (0–255 channel units). Their visible
trajectories gradually diverge despite the matching baseline specification.

The published checkpoint configuration, baseline sampler, geometry/reset source,
scene and noise seed match the historical setup. The old raw per-control actions
and states are not available on this machine. Small numerical differences in
inference or contact dynamics across the different GPU/OS builds are plausible,
but their causal role is **not established**. Exact cross-machine reproduction of
the historical D trajectory remains unresolved; do not call this a recovered
89.83% result or tune seeds to hide the discrepancy.

Detailed reports/logs and `review-summary.json` are local under
`artifacts/reranking-20261021/`; experiments are in
`artifacts/td-simulator-smoke-20261021-r2/`, `artifacts/td-abc-20261021-{A,B,C}/`
and `artifacts/td-baseline-300-20261021/`. The audit checks native thresholds,
budgets, independent geometry metrics, branch counts and selected next-state
predictions against real controls. Checkpoints, environment, data and raw videos
remain ignored and are not part of the commit. This is privileged physical
simulator inference, not learned cross-shape adaptation or image-only deployment.

## D teacher-search development pilot (2026-10-21)

The user requested teacher validation before any student training. These runs
search absolute pusher targets using known D state, geometry and simulator
dynamics. No T-policy sample, policy checkpoint, learned world model or Torch
inference is used. The existing native Windows environment is reused. This is
not an image-only controller or evidence of learned cross-shape adaptation.

Four requested initial-scene seeds were fixed before searching:
100000, 100100, 100200 and 100300. All passed the existing matched T/D geometric
reset checks immediately. Scene 100000 is already a development case; none of
this pilot is a held-out generalization benchmark. Each scene uses search seeds
42 and 43, at most 300 actual controls and 600 wall seconds. At most three runs
execute concurrently. Search uses population 12, three generations, three elites,
at most 40 proposals per decision, execution prefixes of eight and four settling
controls. Forecasts, including settling, are capped at 80 controls and the actual
remaining episode budget. Native coverage greater than 0.95 stops execution.

### Original generic teacher: completed

Actual commands executed on this machine, from the repository root:

```powershell
.\.venv\Scripts\python.exe -B scripts/check_repo.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_search.py' -v
$env:TDYSSEY_OFFICIAL_ROOT='third_party/diffusion_policy'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_search_runtime.py' -v
.\.venv\Scripts\python.exe -B artifacts/d-teacher-20261021/run_matrix.py
```

The source check passed; all nine search protocol tests and four installed real
D runtime tests passed, with zero skips. The matrix launcher and experimental
helpers are local ignored artifacts; its protocol records every exact evaluator
command and a copy of the implementation used.

| Scene | Search seed | Actual controls | Outcome | Max/final coverage | Max/final IoU | Final angle error |
| --- | --- | --- | --- | --- | --- | --- |
| 100000 | 42 | 300 | Control budget, no success | 61.2637% | 0.441583 | 100.090 degrees |
| 100000 | 43 | 280 | Wall timeout, no success | 61.7792% | 0.446960 | 101.464 degrees |
| 100100 | 42 | 300 | Control budget, no success | 61.5526% | 0.444592 | 60.615 degrees |
| 100100 | 43 | 300 | Control budget, no success | 63.5755% | 0.466012 | 47.661 degrees |
| 100200 | 42 | 248 | Wall timeout, no success | 66.0001% | 0.492538 | 143.905 degrees |
| 100200 | 43 | 264 | Wall timeout, no success | 59.1090% | 0.419537 | 70.635 degrees |
| 100300 | 42 | 256 | Wall timeout, no success | 20.3467% | 0.113256 | 109.035 degrees |
| 100300 | 43 | 272 | Wall timeout, no success | 63.5646% | 0.465896 | 105.721 degrees |

All eight runs failed the native task-success condition. Peak and final metrics
happen to coincide in these finished runs; this does not mean every intermediate
control improved overlap. The 100000/42 final center error is only 1.274 pixels,
but its angle error is 100.090 degrees. Translation to the target center alone
does not make a useful successful demonstration.

Costs include every failed/timed-out run: 2,220 actual controls, 9,040 branch
restores, 1,296,623 historical replay controls and 672,676 forecast controls;
16 initial reset substeps are separate. Summed planning time was 4,503.594 seconds;
concurrent matrix wall time was 1,799.2 seconds. These time totals are different
quantities. Deadline attempts used 92 restores, 23,303 replay and 3,516 forecast
controls, including their discarded partial-work subsets; they are already in
the totals, not added again. There were 297 actual contact-bearing controls.

All eight source/configuration/initial-state checks passed. Independent initial-
snapshot-plus-entire-action-history replay checked all 2,220 actual controls,
including states, rewards, contacts, native done flags, metrics and selected
predictions. Decoded videos were checked against rendered reference frames.
All eight audits passed; this independent verification adds 2,220 replay controls,
eight reset substeps and 63.250 summed seconds, separate from search costs.
Videos hold every actual control for 0.3 seconds at 10 fps, plus initial/final
holds. There is no interpolated motion. This fixed playback timing does not
represent actual search latency; per-decision search times remain in JSON.

### Experimental contact-search revision: completed

The isolated revision retains the original replay/forecast/native-success
protocol and all initial states and budgets. It adds a 0.02 incumbent-switch
margin, remaining approach flags, exterior edge contacts, short corrections,
signed rotation proposals and a rotation followed by translation recovery from
the actual predicted branch state. Both stages count against the same forecast
budget. The v1 source and reports remain unchanged.

All 18 focused source tests passed without skips. A real D contact continuation
and a separate staged composite fixture passed; the latter independently matched
all 80 forecast controls, including states, rewards, native termination, contacts
and coverage. The composite fixture deliberately selects one generated proposal
to test the protocol; it is not a task-performance trial. A four-control launcher
smoke also passed. Exact commands and costs are in
`artifacts/d-teacher-20261021/v2-draft/CHECKS.json` and `v2-launcher-smoke/`.

```powershell
.\.venv\Scripts\python.exe -B artifacts/d-teacher-20261021/run_matrix_v2.py
```

The full revision comparison completed in 1,775.000 concurrent wall seconds:
zero native successes, three control-budget failures and five wall-time
truncations. No requested case remains queued/running and no protocol error was
reported. Coverage and IoU below are from actual execution, not candidate scores.

| Scene | Search seed | Controls | Outcome | Max / final coverage | Max / final IoU | Final angle error |
| --- | --- | --- | --- | --- | --- | --- |
| 100000 | 42 | 288 | Wall timeout | 63.3663% / 63.3663% | 0.463768 / 0.463768 | 104.865 degrees |
| 100000 | 43 | 300 | Control budget | 60.5258% / 60.5258% | 0.433957 / 0.433957 | 98.232 degrees |
| 100100 | 42 | 300 | Control budget | 61.8698% / 61.8698% | 0.447909 / 0.447909 | 61.118 degrees |
| 100100 | 43 | 300 | Control budget | 56.0724% / 56.0724% | 0.389588 / 0.389588 | 89.862 degrees |
| 100200 | 42 | 272 | Wall timeout | 63.2170% / 63.2170% | 0.462170 / 0.462170 | 103.659 degrees |
| 100200 | 43 | 264 | Wall timeout | 63.0873% / 63.0873% | 0.460785 / 0.460785 | 107.238 degrees |
| 100300 | 42 | 256 | Wall timeout | 63.6089% / 63.2959% | 0.466371 / 0.463014 | 128.180 degrees |
| 100300 | 43 | 256 | Wall timeout | 62.6546% / 62.6481% | 0.456182 / 0.456114 | 117.419 degrees |

V2 used 2,236 actual controls, 9,053 restores, 1,226,293 historical replay
controls and 677,710 forecast controls; 16 initial reset substeps are separate.
Summed planning time was 4,520.091 seconds. The five deadline attempts used
59 restores, 14,869 replay and 1,909 forecast controls, including their partial
subsets of five restores / 597 replay / zero forecast controls. These attempts
are already counted. Independent replay checked every actual control and
selected prediction, with zero physical-state error; all eight replay and decoded
video audits passed. Audit work adds 2,236 replay controls, eight reset substeps
and 63.357 summed seconds separately. All 13 frozen source copies and final
report source/configuration/initial-state checks passed. No Torch was loaded.

### Teacher readiness and comparison

Both versions together contain 16 actual development runs: **zero successes,
six control-budget failures and ten timeouts**. All 4,456 actual controls and
all 16 paused videos passed independent audits. Source/protocol correctness
passed; generic task completion and multiple successful-route qualification
failed within these declared budgets. The previous case-specific rescue from a
selected mid-episode state is separate and is not included as a reset success.

The revision's mean final coverage rose from 57.1489% to 61.7603%, but four
matched pairs improved and four worsened. The 100300/42 increase from 20.3467%
to 63.2959% accounts for most of that mean change. Success stayed at zero and
median final angle error increased from 100.777 to 104.262 degrees. These are
matched initial states and resource caps, not equal completed control/forecast
counts or a single-mechanism ablation; timeouts and concurrent machine load
affect the work completed. No reliable teacher improvement is established.

V2's final center errors are 0.18--4.96 pixels while angle errors remain
61.12--128.18 degrees. The traces support a joint-pose planning limitation:
translation reaches stable partial overlap, while useful rotation can displace
the center and reduce the current terminal-overlap score. For example, at
100100/42 control 200, a physically safe candidate reduced predicted angle error
from 61.22 to 36.26 degrees but increased center error from 6.92 to 49.09 pixels
and reduced coverage from 61.39% to 19.10%. This is a candidate diagnostic,
not an executed success or proof that every better-angle route must fail.

The revision did simulate composites: 837 such forecasts completed. Among
49 fresh executed-plan selections there were 31 long routes, 12 translations
and six rotation/translation composites, with no standalone rotation. The six
selected composites allocate 74.6--90.3% of planned active controls to approach;
these are full-plan allocations, not fractions of actually executed controls.
Selecting an eight-control prefix does not prove its whole composite executed.
There were zero native-success completed candidates among 9,013 forecasts.
The evidence therefore does not justify labelling these traces successful
expert demonstrations or starting student distillation. No student was trained.

Local `matrix-v1/` and `matrix-v2/` contain frozen source snapshots, exact
commands, all reports and summaries. `videos-v1/` and `videos-v2/` contain all
eight audited fixed-pause videos per version and final censuses. The artifact
root's `gallery.html` links all 16 cases; `matrix-comparison-audit.json`
independently verifies raw reports, totals and media, while
`matrix-comparison.csv` preserves every paired difference. Both matrix costs
sum to 3,877,758 simulated controls and 9,023.685 planning seconds, excluding
preview capture, tests and 4,456 independent audit replay controls. These remain
ignored local artifacts; no environment, weights, data or raw media are committed.

## Two-contact search recovery pilot (2026-10-22)

Implemented the separate backend described in
[CONTACT_SEQUENCE_SEARCH.md](CONTACT_SEQUENCE_SEARCH.md). This native Windows
experiment uses known D geometry, state and actual Pymunk dynamics, without
Torch, a policy checkpoint, demonstrations supplied to the search, or student
training. It combines diverse contact-sequence expansion, joint fixed-material-
point pose error, conditional elite memory and exact-history branch replay.
It is not a reproduction of the referenced papers' analytic solvers or proofs.

### Fixed inputs and checks

Before inspecting new search outcomes, fixed the V2 scene/search-42 reports for
100000, 100100 and 100200 at control 160. Start coverage was 60.95%, 57.53%
and 62.22%; center errors were 5.242, 11.612 and 1.120 pixels and angle errors
103.041, 64.952 and 102.088 degrees. These are previously inspected development
states with full recorded contact histories, not newly selected easy resets.

Every run imported and checked all 160 old controls, then allowed at most 140
new controls, 300 total episode controls, 600 wall seconds, eight global searches
and 40 attempts per search. Forecasts, approaches, releases, second pushes and
physical holding share one 80-control horizon clipped to actual remaining time.
At most two pilot runs execute concurrently. Native coverage greater than 0.95
stops immediately. Original source reports are unchanged and each revision has
its own frozen source copies and exact command list.

Actual check commands, from repository root:

```powershell
.\.venv\Scripts\python.exe -B scripts/check_repo.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_sequence.py' -v
$env:TDYSSEY_OFFICIAL_ROOT='third_party/diffusion_policy'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_sequence_runtime.py' -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_search.py' -v
```

The static source check passed. The final focused suites passed 16 source
protocol tests, two installed real D runtime tests and nine original-search
regression tests, all with zero skips. The physics suite includes a deliberately
forced two-contact fixture after eight real contact-bearing history controls:
all 16 selected controls matched actual continuation exactly and search left
live state unchanged. This fixture tests the replay protocol, not performance.
The CLI test saves a report and resumes it through its initial history without
loading Torch. Source checks alone are not physical verification.

### First revision retained, then ordinary fixes

The initial three runs were all unsuccessful at the eight-search limit. One
reached 92.30% coverage, but never native success. Independent replay and paused
video checks for all three completed before changing source. Frozen copies and
all original reports/videos remain under `artifacts/d-sequence-20261022/`.

Observed failures motivated general corrections: at a corner, an arbitrary
nearest-face normal can leave the circular pusher inside the inflated obstacle;
release now validates radial/directional alternatives and the connecting segment.
Single-contact fallbacks no longer reserve 12 controls for a nonexistent second
push. Near-goal strokes and travel-variance floors now scale with residual pose
error. Geometric rejection also occurs before expensive history reconstruction
when the known endpoint suffices. No scene coordinates or successful action
sequence were encoded. The same starts, seed and caps were rerun unchanged.

Actual pilot and video commands:

```powershell
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022/run_pilot.py
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022/audit_and_video.py --report artifacts/d-sequence-20261022/scene-100000-resume160-search42/report.json --report artifacts/d-sequence-20261022/scene-100100-resume160-search42/report.json --report artifacts/d-sequence-20261022/scene-100200-resume160-search42/report.json --output artifacts/d-sequence-20261022/videos-v1
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r2/run_pilot.py
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r2/audit_and_video.py --report artifacts/d-sequence-20261022-r2/scene-100100-resume160-search42/report.json --output artifacts/d-sequence-20261022-r2/videos/scene-100100
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r2/audit_and_video.py --report artifacts/d-sequence-20261022-r2/scene-100000-resume160-search42/report.json --output artifacts/d-sequence-20261022-r2/videos/scene-100000
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r2/audit_and_video.py --report artifacts/d-sequence-20261022-r2/scene-100200-resume160-search42/report.json --output artifacts/d-sequence-20261022-r2/videos/scene-100200
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r2/summarize.py
```

The two exports for 100000/100200 in the corrected revision ran concurrently.
All output paths are single-use; reuse requires choosing a fresh destination.
Launchers/helpers and raw artifacts are ignored local files, not shared assets.

### Actual results: all six cases

| Revision | Scene | Outcome | New / total controls | Peak / final coverage | Final IoU | Center error start / final px | Angle error start / final deg | Joint RMS start / final px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Initial | 100000 | Search-count budget, no success | 89 / 249 | 61.2145% / 56.6064% | 0.394762 | 5.242 / 9.556 | 103.041 / 48.283 | 82.206 / 43.993 |
| Initial | 100100 | Search-count budget, no success | 123 / 283 | 92.3029% / 89.9393% | 0.817178 | 11.612 / 2.700 | 64.952 / 3.138 | 57.946 / 3.870 |
| Initial | 100200 | Search-count budget, no success | 88 / 248 | 62.2184% / 60.6365% | 0.435096 | 1.120 / 5.760 | 102.088 / 47.434 | 81.613 / 42.616 |
| Corrected | 100000 | Total-control budget, no success | 140 / 300 | 81.9795% / 81.9795% | 0.694621 | 5.242 / 0.931 | 103.041 / 9.973 | 82.206 / 9.160 |
| Corrected | 100100 | Native success | 69 / 229 | 96.0907% / 96.0907% | 0.924755 | 11.612 / 0.833 | 64.952 / 1.391 | 57.946 / 1.540 |
| Corrected | 100200 | Total-control budget, no success | 140 / 300 | 75.0035% / 75.0035% | 0.600044 | 1.120 / 4.620 | 102.088 / 12.559 | 81.613 / 12.210 |

The corrected pilot has one of three actual native recoveries, with zero
timeouts or protocol errors. The successful run stopped at control 229, with
no post-success controls. Both other runs exhausted the original total episode
budget. Position and angle improved together in 100000 and 100100; 100200's
angle improved substantially while its center error increased. A lower joint
RMS is not a claim that both component errors always improve monotonically.

### Costs, video checks and limitations

| Revision | New controls | Restores | Historical replay | Forecast incl. hypothetical-prefix replay | Summed planning seconds | Concurrent matrix wall seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Initial | 300 | 960 | 202,160 | 29,964 | 465.935 | 293.360 |
| Corrected | 349 | 561 | 119,661 | 35,497 | 345.347 | 219.265 |

Each revision separately imported 480 controls to establish its three live
start states; this is additional to the search counts above. Reset substeps
are also separate. The two revisions change several proposal/execution details
together and have different actual stopping points; they are not a single-factor
or equal-work ablation. Their time sums and concurrent wall times are distinct.

All six independent simulator replay/video audits passed: 649 new actual
controls plus 960 historical controls were replayed, totaling 1,609 audit
controls and six additional reset substeps. Every recorded state, contact,
reward, done flag and selected prediction matched. Videos were decoded and
checked against actual rendered frames. Each new actual control is held for
0.3 seconds at 10 fps, with no interpolation; playback omits actual planning
latency, which remains in reports. The successful video starts at old control
160, not at a reset or a fabricated final pose.

The independent saved-report summary recomputes every RMS using separate
material-point transforms, checks all attempt/branch totals, caps, continuation
zero-work records, native termination, immutable old source reports and frozen
source copies for each revision. All checks passed. Its six-case gallery is
`artifacts/d-sequence-20261022-r2/gallery.html`; detailed rows are in
`summary.json`, `summary.csv` and `summary.md`. Original and corrected failures
remain visible. No weight, environment, dataset or raw media is committed.

This establishes one successful rescue from a known partial-overlap state.
It does not establish success from reset, reliability across seeds, multiple
successful contact strategies, student readiness or learned generalization.
The remaining 10--13 degree residual angles in two failures remain meaningful
limitations. No further seed search, extended budget or student training was
performed to replace these results.

## Same-budget precision follow-up (2026-10-22)

### Fixed scope and implementation

The user requested the two failed recovery states, with the same budgets and
fixed-pause videos, before any student learning. Revision 3 uses exactly the
original V2 scene 100000 and 100200 control-160 states used by revision 2, not
extra controls after their old control-300 endpoints. Search seed remains 42.
Each run has at most 140 new controls within 300 total, 600 wall seconds,
80 total forecast controls including holding, 40 attempts per global search and
eight global searches. At most two processes run concurrently. The original
native coverage-greater-than-0.95 threshold and immediate success stop are intact.

The changes were made in the agreed order: adaptive settling reserve, separate
approach/slow push speeds with bounded physical press dwell, nearby-contact and
opposite-face compensation proposals, then a small access/remaining-budget cost
added to joint material-point RMS. Reports retain raw RMS, completion cost and
phase-specific actual forecast contact counts. There are still at most two
contact stages. Full real and hypothetical histories are replayed for every
physical branch; the environment, checkpoint and dynamics files are unchanged.
See [mechanism and exact parameter rules](CONTACT_SEQUENCE_SEARCH.md#precision-and-budget-revision-2026-10-22).

All production source was frozen in the new pilot's `source/` directory before
either outcome. `protocol.json` stores both requests, exact commands, source
versions and immutable input references. The prior two failures and their source
snapshots are preserved. This is a combined development revision, not an ablation
or a held-out test. No further search seeds, performance revisions, extra episode
controls, policy checkpoints or student training were used to replace outcomes.

### Executed checks and commands

From the repository root, in the existing native Windows environment:

```powershell
.\.venv\Scripts\python.exe -B scripts/check_repo.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_sequence.py' -v
$env:TDYSSEY_OFFICIAL_ROOT='third_party/diffusion_policy'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_sequence_runtime.py' -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_search.py' -v
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r3/run_pilot.py
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r3/audit_and_video.py --report artifacts/d-sequence-20261022-r3/scene-100000-resume160-search42/report.json --report artifacts/d-sequence-20261022-r3/scene-100200-resume160-search42/report.json --output artifacts/d-sequence-20261022-r3/videos
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r3/summarize.py
```

Source checks passed. The 21 sequence protocol tests, three real D runtime tests
and nine original-search regression tests all passed, with **zero skips**.
New checks cover available pushing in the final five controls, separate approach
and push speed, repeated physical pressing targets with budget truncation,
opposite-face pair generation and native-success priority over access cost.
A generated fine press made real D contact and matched all live predictions.
The two-contact protocol fixture still forces selection only to verify history
continuation; neither fixture is task-performance evidence.

### All matched outcomes

| Revision | Scene | Stop | New / total controls | Peak / final coverage | Final IoU | Position error px | Angle error degrees | Joint RMS px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prior 2 | 100000 | control budget | 140 / 300 | 81.98% / 81.98% | 0.69462 | 0.931 | 9.973 | 9.160 |
| Precision 3 | 100000 | control budget | 140 / 300 | 73.00% / 70.04% | 0.53896 | 8.629 | 8.314 | 11.334 |
| Prior 2 | 100200 | control budget | 140 / 300 | 75.00% / 75.00% | 0.60004 | 4.620 | 12.559 | 12.210 |
| Precision 3 | 100200 | search budget | 138 / 298 | 78.41% / 76.45% | 0.61872 | 6.502 | 7.856 | 9.524 |

Both new recoveries failed. Their first three coarse plans and attained states
match the prior revision; precision mode begins at control 245 and 248. Final
orientation improves against the prior version in both cases, but final position
error increases. Scene 100000 loses coverage and joint accuracy; scene 100200 has
a small coverage and joint-RMS gain but still fails. There is no basis to claim
that the joint task was solved or that teacher readiness improved reliably.

For scene 100000, precision global searches at controls 245, 260 and 273 select
plans with respectively 15/13/12 active controls, of which only 4/2/2 are approach
controls. Their predicted push-contact controls are 9/11/8. For scene 100200 the
corresponding plans at 248, 262 and 275 use 14/13/7 controls, with 4/2/1 approach
and 8/11/6 push-contact controls. Thus long noncontact approach is reduced, and
slow presses do establish contact, but improved contact persistence does not
guarantee appropriate coupled object motion.

Both final two global searches prefer the incumbent physical hold over all
available sampled pushes. With less budget, many contact proposals still cannot
fit their travel plus actual push: scene 100000's final search rejects 23 for
stage duration and four for inaccessible contacts; scene 100200's rejects 20 and
five respectively. Settling reserve is now zero in each final search, so the
earlier mandatory-four-hold problem is removed but not sufficient to solve these
states. The contact-pair prior and fixed RMS/cost ranking remain too limited.

Scene 100200 then reaches the eight-search cap at control 298. Its ninth request
is explicitly rejected with zero branch work; no ninth search or additional two
controls are supplied. It is recorded as `stopped_early / search_budget`, not a
full-control-budget completion or a runtime crash. Neither run timed out or had
a physics protocol error. Every saved selected-to-actual state error is zero.

### Work, media and limitations

| Matched cases | New controls | Restores | History replay | Forecast including hypothetical-prefix replay | Summed planning seconds |
| --- | --- | --- | --- | --- | --- |
| Prior revision 2 | 280 | 411 | 90,595 | 25,009 | 258.518 |
| Precision revision 3 | 278 | 522 | 121,368 | 27,015 | 328.028 |

The new two-process pilot took 177.218 wall seconds. Individual elapsed times
were 166.250 and 175.594 seconds, within their 600-second caps. Timing is
descriptive on a shared CPU, not a controlled speed benchmark. Each revision
also reconstructs 320 actual live prefix controls, separate from search work,
and has two additional reset substeps.

Both new independent replay/video audits passed: 278 new controls plus 320
historical controls, totaling **598** audited controls and two reset substeps.
They check states, contact counts, rewards, done flags, predictions and rendered
video frames. Playback holds each new actual step for **0.3 seconds at 10 fps**,
with no interpolation; initial and final holds are one and two seconds. The
videos start at old control 160 and omit planning latency, which is recorded
separately. They end at their actual stopping controls without fabricated actions.

The saved-report summary also passed: exact initial-state and seed/cap equality
with the prior two cases, independent per-step material-point RMS recomputation,
all branch/attempt totals, zero-work continuations, source snapshots, immutable
source reports and native termination. The four compared videos and all outcomes
are in `artifacts/d-sequence-20261022-r3/gallery.html`, with `summary.json`,
`summary.csv`, `summary.md` and individual raw/audit reports alongside it. Its
1,198 audit controls combine 600 previously audited baseline controls and 598
newly audited controls; do not count baseline audits as new execution.

This is privileged simulator-state control on two repeatedly inspected development
states. It remains insufficient evidence for reset reliability, multiple successful
routes, a trustworthy successful teacher set or student learning. All raw media,
environment assets and frozen local source snapshots remain outside Git.

## Guarded joint correction follow-up (2026-10-22)

### Scope, changes and preserved corrections

The user requested joint center/angle control on the same two failed states,
with identical budgets, every success/failure retained and paused videos, before
student learning. All runs below resume the same original V2 scene 100000/100200
control-160 snapshots and search seed 42. Each permits at most 140 new controls
within 300 total, 600 wall seconds, 80 forecast controls including hold, 40 total
attempts per global search and eight searches. Concurrency is at most two. The
original coverage-greater-than-0.95 success rule and immediate stop are unchanged.

Revision 4 adds independent terminal center/angle limits, a bounded intermediate
corridor, worst-component ranking within feasible plans, compensation proposals
from actual first-stage displacement, separate position/angle/joint/compensation
refinement banks, press dwell variants and up to four spare third-contact attempts.
The third contact replays the full initial + real + both hypothetical prefixes.
No direct pose assignment or cold endpoint restoration is used. Exact parameters
and native-success precedence are in [the implementation notes](CONTACT_SEQUENCE_SEARCH.md#guarded-joint-correction-revision-2026-10-22).

The first pair of actual runs exposed an execution/scoring mismatch: a plan could
use a future held endpoint for scoring but replan before those holds executed.
Both failures were preserved and independently audited before source changes.
Revision 5 fixes that timing: precision execution includes its bounded physical
settling prefix, scoring/constraints use the same prefix and actual terminal
limits are checked at its boundary. The common full forecast remains counted.
Its two failures and videos are also retained. An independent saved-contract
check then found the old scalar RMS stall rule interrupting two still-active
precision macros in scene 100000 (273 -> 274 and 274 -> 275). Revision 6 disables
that coarse-search rule in precision mode while retaining all physical/state
and corridor checks. This final correctness rerun uses the same requests/caps.

Each revision was frozen in its own `source/` directory, including the production
source and focused tests, before observing its outcomes. The three local roots
are `artifacts/d-sequence-20261022-r4/`, `artifacts/d-sequence-20261022-r4-settle/`
and `artifacts/d-sequence-20261022-r4-final/`. Their `protocol.json` files contain
all actual CLI commands and source/input provenance. No seed search, budget
increase, additional performance revision or student training replaces a failure.

### Checks and actual commands

The final source checker passed. Twenty-eight focused sequence tests, four
actual D runtime checks and nine original-search regressions passed, **zero
skipped**. New source checks cover independent constraints rejecting a lower-RMS
center-drift plan, ratcheted limits with compensated intermediate motion,
family retention, counted dwell variants, three-stage complete-history replay,
execution of the scored settling prefix and immunity to old scalar plateau
interruptions. Two legacy fake-hold fixtures were positioned at their fake object
so their artificial target-following dynamics did not confound budget tests.

The real triple fixture forces the selected stage count and uses wide planning
tolerances only to isolate physical history reconstruction; actual collision,
goal and native threshold are unchanged. This fixture is protocol evidence,
not a successful task demonstration. The actual pilot uses unmodified new limits.

Executed from the repository root in the existing Windows environment:

```powershell
.\.venv\Scripts\python.exe -B scripts/check_repo.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_sequence.py' -v
$env:TDYSSEY_OFFICIAL_ROOT='third_party/diffusion_policy'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_sequence_runtime.py' -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_search.py' -v
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r4/run_pilot.py
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r4-settle/run_pilot.py
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r4-final/run_pilot.py
```

Each pilot runs both original resume reports with `--resume-step 160 --steps 300
--seconds 600 --horizon 80 --max-evaluations 40 --max-global-searches 8
--search-seed 42`, using a separate output path. After each revision completed,
its copied `audit_and_video.py` was executed with the two `--report` arguments
and a fresh `--output` under that revision's `videos/`. The final exact command is:

```powershell
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r4-final/audit_and_video.py --report artifacts/d-sequence-20261022-r4-final/scene-100000-resume160-search42/report.json --report artifacts/d-sequence-20261022-r4-final/scene-100200-resume160-search42/report.json --output artifacts/d-sequence-20261022-r4-final/videos
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r4-settle/verify_joint_contract.py
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r4-final/verify_joint_contract.py
.\.venv\Scripts\python.exe -B artifacts/d-sequence-20261022-r4-final/summarize.py
```

### All six new outcomes

| Revision | Scene | New / total controls | Peak / final coverage | Final IoU | Position error px | Angle error degrees | Joint RMS px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Guarded 4 | 100000 | 111 / 271 | 72.13% / 71.98% | 0.56228 | 5.061 | 16.430 | 15.795 |
| Guarded 4 | 100200 | 134 / 294 | 75.29% / 75.26% | 0.60331 | 3.574 | 14.872 | 14.064 |
| Settling fix 5 | 100000 | 115 / 275 | 72.13% / 71.98% | 0.56228 | 5.061 | 16.430 | 15.795 |
| Settling fix 5 | 100200 | 138 / 298 | 75.29% / 75.26% | 0.60331 | 3.573 | 14.872 | 14.063 |
| Final 6 | 100000 | 127 / 287 | 72.13% / 71.77% | 0.55974 | 5.158 | 16.035 | 15.457 |
| Final 6 | 100200 | 138 / 298 | 75.29% / 75.26% | 0.60331 | 3.573 | 14.872 | 14.063 |

Every row is a failure, explicitly `stopped_early / search_budget` after eight
searches. None timed out or crashed. Do not relabel these as 300-control
completions. Against revision 3's 8.63/6.50 px and 8.31/7.86 degree final errors,
the final centers are closer but residual angles are larger. This is still a
failure to jointly finish the task, not teacher qualification or reliable improvement.

The final independent contract audit passes: five precision macros per scene
reach their intended execution boundaries, without early interruption. It checks
the position/angle ratchet, predicted executed prefix, worst-component score,
actual corridor checks on all 92 precision controls and actual terminal checks.
The corresponding revision-5 audit preserves its two interrupted macro records
instead of hiding them. All saved selected-to-actual state errors remain zero.

Final search attempts include 28 third-contact proposals: 12 complete physical
forecasts and 16 geometric/duration rejections. None satisfies the independent
pose constraints, so none is selected. Thus third-stage implementation and replay
are verified, but these runs do not demonstrate a successful three-stage push.
The final searches prefer feasible holding when other sampled plans violate a
pose constraint or cannot fit remaining controls; feasibility restricts drift
without producing the missing jointly effective contact sequence.

### Costs, videos and limits

| Revision | New controls | Restores | History replay | Forecast including hypothetical replay | Summed planning seconds | Concurrent matrix wall seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 245 | 571 | 133,003 | 30,531 | 352.672 | 183.218 |
| 5 | 253 | 557 | 130,169 | 29,197 | 354.249 | 185.219 |
| 6 | 265 | 556 | 130,035 | 29,031 | 361.487 | 191.250 |

All three revisions additionally reconstruct 320 imported live controls and two
reset substeps each, separate from branch counts. Final individual elapsed times
are 189.266 and 182.547 seconds, below 600 seconds. Time sums and concurrent wall
times are different quantities and descriptive on a shared CPU, not a benchmark.

All six new replay/video audits passed: **763 new actual controls + 960 historical
controls = 1,723 audited controls**, plus six reset substeps. States, rewards,
contacts, termination, selected predictions and decoded rendered frames match.
Each actual new control is held for **0.3 seconds at 10 fps**, without interpolation;
initial/final holds are one/two seconds. Playback starts at old control 160,
omits measured planning latency and ends at the actual stopping control, with
no fabricated recovery or post-budget actions.

The final summary independently rechecks immutable source snapshots/input
reports, exact matched starts/seeds/caps, per-step material-point RMS, work totals,
continuation zero-work and native termination. Its gallery places revision 6 first
and retains all ten matched cases from revisions 2--6. `summary.json`, CSV/Markdown
rows and videos are under `artifacts/d-sequence-20261022-r4-final/`. Its 2,921 audit
controls include 1,198 previously audited baseline controls; only 1,723 belong to
this milestone. No raw media, weights, environments or source snapshots enter Git.

These repeatedly inspected states remain development cases. The teacher still
does not supply successful, diverse demonstrations for these two starts. No
student/policy/world-model training, asset loading, dependency changes or learned
cross-shape/visual-deployment claim follows from protocol or constraint checks.
