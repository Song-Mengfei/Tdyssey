# Shared environment and storage

## Source-only work

The current laptop must not acquire an environment, research packages, weights,
datasets, or containers. `python3 -B scripts/check_repo.py` uses the standard library
only. Plain `uv run` may install dependencies, so it is not a source-only check.

The repository contains a metadata-resolved candidate lock, not a runtime-tested
environment. Upstream code is not vendored, and neither doctor nor static checker
downloads resources. The realsense ML environment is left untouched.

## Reference platform and profiles

Use uv **0.13.0**, Python **3.12.3**, and **Linux x86_64** as the first shared target.
The manifest limits resolution to that platform. The CPU profile is for development
and small correctness checks; it is not a practical substitute for all training.

| Profile | Contents | Later setup command |
| --- | --- | --- |
| CPU | Base dependencies + torch/torchvision CPU wheels | `uv sync --locked --extra cpu` |
| CUDA | Base dependencies + torch/torchvision cu124 wheels | `uv sync --locked --extra cu124` |
| GPC adapter | Legacy Gym, diffusers, image/video utilities, logging | Add `--extra gpc` to the chosen profile |

CPU and CUDA extras conflict intentionally. Pass the same extras to `uv run` as to
`uv sync`. Verify NVIDIA driver compatibility separately; driver versions need not
be identical between authors. cu124 is an initial profile for the known Ada GPUs,
not a promise for every GPU. **Blackwell requires a newer compatible PyTorch/CUDA
profile and a new lock/validation decision.** Other operating systems/architectures
need an explicit supported-profile change before training.

Versions target older GPC APIs: NumPy 1.x, Zarr 2, Pymunk 6, and an officially matched
PyTorch 2.5.1 / torchvision 0.20.1 pair. This reduces avoidable API churn but does
not establish compatibility. Legacy Gym is only installed with the GPC extra;
new simulator code should use Gymnasium. Headless OpenCV supports image/video IO;
do not depend on its GUI functions. Video export may additionally need a system
`ffmpeg` executable; SDL/display libraries depend on the chosen host.

Install uv only on the authorized target machine, following its official guide.
Pin its version to 0.13.0. Use the commands in `README.md` after agreeing on disk
budget. Save doctor output alongside the first actual run, then perform T baseline
and checkpoint/schema checks. Doctor alone is not a reproduction result.

## Updating dependencies

Change direct requirements deliberately, regenerate `uv.lock` with the pinned uv
version, review the diff, and record the reason in `docs/DECISIONS.md`. Both authors
sync the same profile after pulling. Never replace the shared lock with a freeze
of an unrelated environment.

Gym 0.26.2 is source-only. Its dependency metadata is recorded in the manifest
from a checksum-verified published archive; see `manifests/gym-metadata.json`.
This lets uv resolve it without executing its setup code. A blanket `--no-build`
also excludes this source-only candidate, so metadata locking instead disables
build isolation and uses the supplied static metadata. Installing the GPC profile
later may build Gym on the authorized machine.

Metadata-only locking, with uv already available and a temporary cache:

```bash
uv lock --python /path/to/existing/python3.12 --no-python-downloads --no-build-isolation --cache-dir /tmp/tdyssey-lock-cache
uv lock --check --offline --python /path/to/existing/python3.12 --no-python-downloads --no-build-isolation --cache-dir /tmp/tdyssey-lock-cache
```

These commands resolve/check metadata, not install the environment. Inspect output
for unexpected build activity. If another package needs a build to inspect its
metadata, stop and investigate; do not install build dependencies on a source-only
machine. Remove only task-owned temporary files.

## Storage planning

Source/configuration is small. On a training machine reserve roughly 40–60 GB for
the course pilot: 8–12 GB environment, about 1.2 GB published GPC assets as stored,
5–15 GB new data/preprocessing, 5–15 GB selected checkpoints, and 5–10 GB temporary
files/logs/videos. These are planning ranges, not a cap or measured run footprint.

At 96×96 RGB, 50k raw uint8 frames use about 1.38 GB; float32 uses four times that.
Do not store repeated sliding windows. Upstream WM code saves every epoch; retain
selected checkpoints after agreeing on retention, rather than accumulating every
epoch from every seed. Avoid duplicate cache/output copies and record actual usage.

Configure data/checkpoint/run roots on the selected machine; do not hardcode an
author's home directory. Large data can live on another disk while small manifests
remain in Git. Each asset manifest should contain source revision, filename, size,
hash, and download/verification status. Do not invent hashes before downloading.

References: [uv lock/sync](https://docs.astral.sh/uv/concepts/projects/sync/),
[PyTorch indexes](https://docs.astral.sh/uv/guides/integration/pytorch/),
[published PyTorch builds](https://pytorch.org/get-started/previous-versions/).
