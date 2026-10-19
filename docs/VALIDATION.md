# Runtime validation

Checked on 2026-10-19 with Linux x86_64, Python 3.12.3, uv 0.13.0, and the locked
**CPU + GPC** profile. These results describe simulator development, not a trained
policy or a successful adaptation experiment. CUDA remains unvalidated.

## Versions and source

- PyTorch `2.5.1+cpu`, torchvision `0.20.1+cpu`, NumPy `1.26.4`, Pymunk `6.9.0`.
- All direct CPU/GPC requirements matched `pyproject.toml`; 85 packages installed.
- GPC commit: `a96146e568826ef46134f73c22fdf59a675e7264`.
- Simulator source SHA-256: `c1a8746343ba0f38eee5753d7a4d49f6723abe50b04a6ee68c4cf9da9e255953`.
- Only the simulator and T-domain source files were materialized from upstream;
  no dataset, checkpoint, or policy evaluation was downloaded or run.

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

- CUDA profile, GPU operators, and policy/world-model throughput.
- Loading the original T policy checkpoint and its observation/action conventions.
- Initial-state penetration, boundary coverage, and goal reachability across seeds.
- Restoring during established contact, whose cached solver impulses are not saved.
- Original GPC dataset conversion, pretrained-policy integration, training, and repeated evaluation.

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
their validation/test splits can be empty. The training-loader milestone below
adds windows/catalogs; pretrained-policy loading and training remain pending.

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
needed. Pretrained policy transforms/normalizers, original GPC data conversion,
meaningful held-out datasets, and actual training remain pending.
