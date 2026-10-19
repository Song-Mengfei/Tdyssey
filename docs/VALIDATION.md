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
- Dataset conversion, visual histories, training, and repeated policy evaluation.

Legacy Gym prints its maintenance warning; NumPy remains pinned to 1.26.4 for
upstream compatibility. Do not reinterpret these short simulator checks as
evidence of policy success, exact mid-contact replay, or training compatibility.
