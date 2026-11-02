# Shared environment and storage

## Choose the smallest profile for the task

Use uv **0.13.0** and Python **3.12.3**. **Linux x86_64** remains the training
reference; the shared manifest and lock also support **native 64-bit Windows**
(`win32`, `AMD64`). `pyproject.toml` and `uv.lock` define package versions for all three authors.
Source-only development remains available with the commands in `README.md`.
Installation and larger downloads follow the permissions and disk budget of the
machine being used; keep those local details out of shared instructions.

| Profile | Purpose | Setup command |
| --- | --- | --- |
| CPU | Development and small correctness checks | `uv sync --locked --extra cpu --no-cache --link-mode copy` |
| CPU + GPC | Real PushT simulator and policy-development dependencies | `uv sync --locked --extra cpu --extra gpc --no-cache --link-mode copy` |
| CUDA + GPC | NVIDIA GPU policy/model work | `uv sync --locked --extra cu124 --extra gpc --no-cache --link-mode copy` |
| CUDA + official DP | Primary official CNN baseline | `uv sync --locked --extra cu124 --extra dp --no-cache --link-mode copy` |

Select both `--extra dp --extra gpc` to retain both runners in one environment.
The earlier Linux combined CUDA/DP/GPC profile measured **5.72 GiB / 113 packages**. The `dp`
extra adds the official encoder/config dependencies and includes its simulator
requirements independently of GPC. See [official baseline setup](OFFICIAL_BASELINE.md).

The CPU/GPC environment measured about **1.33 GiB** after installation, plus about
38 MiB for a local uv executable and less than 1 MiB for the sparse GPC checkout.
This is a measured development footprint, not a training storage budget. The
CUDA profile includes additional GPU libraries. It measured about **5.57 GiB**
after replacing the CPU profile in the same environment; no duplicate CPU
environment was retained. CUDA 12.4 calculations, gradients, torchvision, and
pretrained inference passed on an Ada GPU. This does not prove full training
memory requirements or compatibility with every GPU generation.
See `docs/VALIDATION.md` for the checks that actually ran.

CPU and CUDA extras conflict intentionally. Keep one `.venv` per checkout and
switch its profile with an explicit `uv sync`, rather than retaining duplicate
environments. Use `.venv/bin/python` on Linux or `.venv\Scripts\python.exe` on
Windows directly to avoid an implicit sync. If using
`uv run`, pass the same extras used for setup; plain `uv run` can change the
installed package set. Do not add `--compile-bytecode` for storage-limited work.

## Install without keeping download caches

Install uv 0.13.0 using its [official installation options](https://docs.astral.sh/uv/getting-started/installation/).
An ignored local executable such as `third_party/uv/uv` is also suitable; substitute
that path for `uv` in the commands if it is not on `PATH`. No tool binary belongs
in Git. Use an existing Python 3.12.3 interpreter to avoid another Python download:

```bash
uv sync --locked --extra cpu --extra gpc --no-cache --link-mode copy --python /path/to/python3.12 --no-python-downloads
.venv/bin/python -B scripts/doctor.py --runtime --gpc
```

`--no-cache` uses temporary installation storage without retaining a download
cache. `--link-mode copy` keeps the resulting environment independent of that
temporary cache. Do not clear another project's shared package/model caches to
make room. Ignore `.venv/`, `third_party/`, and `artifacts/`; these are already in
`.gitignore`. Dependencies are isolated from system Python and other projects.

## Native Windows profile

The combined `cu124` + `dp` + `gpc` profile installed **98 packages** on Windows
AMD64 with Python 3.12.3. Its pins include torch **2.5.1+cu124**, torchvision
**0.20.1+cu124**, NumPy **1.26.4**, Pymunk **6.9.0**, and pygame **2.6.1**.
The recorded machine used an RTX 3060 Ti with 8 GB VRAM and driver 610.88;
these are evidence about that machine, not hardware requirements for collaborators.
The doctor and actual CUDA candidate-sampler checks passed. These checks do not
validate policy training or every optional dependency integration; see
[runtime evidence and limits](VALIDATION.md).

From the repository root in PowerShell, after setup is authorized:

```powershell
uv sync --locked --extra cu124 --extra dp --extra gpc --no-cache --link-mode copy
.\.venv\Scripts\python.exe -B scripts\doctor.py --runtime --require-cuda --dp --gpc
$env:TDYSSEY_OFFICIAL_ROOT = (Resolve-Path 'third_party\diffusion_policy').Path
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_planning_runtime.py' -v
```

The source checkout must already be available; these test commands do not fetch it
or train a policy. Use PowerShell `$env:NAME = 'value'` for environment variables;
the inline `NAME=value command` examples elsewhere use POSIX shell syntax.
Repository checks and doctor decode project text as UTF-8 instead of using the
Windows locale. `.gitattributes` marks the published evidence JSON as `-text`,
preserving its byte-size/SHA records through Windows checkouts.

There is one explicit dependency exception. `robomimic==0.2.0` declares
`egl-probe>=1.0.1` unconditionally, but its inspected uses are in the optional
Robosuite and iGibson/MOMART renderers. Native PushT/Pygame and the tested CNN
path do not use those renderers. The shared uv override retains `egl-probe==1.0.2`
on Linux only. On Windows, `uv pip check` therefore reports one expected metadata
incompatibility: robomimic's declared egl-probe dependency is missing. This is not
a clean dependency check or a claim that those optional renderers work on Windows.

## Fetch only the simulator source

The interface and runtime checks need two source files from the pinned GPC
revision, not its datasets, checkpoints, or full working tree. From the project
root, use a new destination:

```bash
git clone --filter=blob:none --no-checkout --depth=1 https://github.com/han20192019/gpc_code.git third_party/gpc
git -C third_party/gpc sparse-checkout set --no-cone /gpc_rank_evaluation/pusht_env.py /gpc_rank_evaluation/domains_yaml/push_t.yml /LICENSE
git -C third_party/gpc fetch --depth=1 --filter=blob:none origin a96146e568826ef46134f73c22fdf59a675e7264
git -C third_party/gpc checkout --detach a96146e568826ef46134f73c22fdf59a675e7264
```

The revision comes from `manifests/upstream.json`. This sparse checkout supports
simulator checks; fetch additional source paths deliberately when integrating the
policy. The example verifies the revision and unchanged simulator/domain bytes.
Keep fixed upstream checkouts at `core.autocrlf=false` on Windows so their source
bytes remain identical to the pinned commit.
Keep upstream modifications separately recorded rather than silently editing the
baseline files.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MPLBACKEND=Agg .venv/bin/python -B examples/gpc_shape_env.py --gpc-root third_party/gpc --shape L --steps 5
TDYSSEY_GPC_ROOT=third_party/gpc .venv/bin/python -B -m unittest discover -s tests -p 'test_gpc_runtime.py' -v
```

The first command holds the pusher still; it is an interface check, not a policy.
The tests additionally exercise actual pushing contacts, compare the adapted T
environment with the original, and check geometry/reward/state behavior. Without
`TDYSSEY_GPC_ROOT`, standard-library test discovery skips real simulator tests.
Reports containing absolute executable paths belong in ignored `artifacts/`,
while portable commands and findings belong in shared documentation.

## Platform and compatibility

Versions target older GPC APIs: NumPy 1.x, Zarr 2, Pymunk 6, and a matched
PyTorch 2.5.1 / torchvision 0.20.1 pair. Legacy Gym is installed with the GPC or DP
extra for upstream compatibility; new environment code should use Gymnasium.
Gym prints its upstream maintenance warning even with the pinned NumPy 1.26.4.
Headless OpenCV supports image/video IO, not its GUI functions. Video export may
also need a system `ffmpeg`; SDL/display requirements depend on the host.

Check NVIDIA driver compatibility before CUDA setup. The cu124 profile has
recorded Ada and Ampere checks; **Blackwell requires a newer compatible PyTorch/CUDA
profile and a new lock/validation decision**. Other operating systems or CPU
architectures need a documented profile change before matched experiments.
CPU simulator checks do not prove GPU or training compatibility. CPU/CUDA
checkpoint loading and GPU gradient checks have separate evidence in
`docs/VALIDATION.md`; a first small visual world-model CUDA pilot is recorded in `docs/WORLD_MODEL.md`.
Full policy training and GPU generations outside the recorded checks remain unvalidated.

## Updating dependencies

Change direct requirements deliberately, regenerate `uv.lock` with the pinned uv
version, review the diff, and record the reason in `docs/DECISIONS.md`. All authors
pull the manifest and lock together, then sync the selected profile. Never freeze
an unrelated environment into this project.

Gym 0.26.2 is distributed as source. Its static dependency metadata is recorded
from a checksum-verified archive in `manifests/gym-metadata.json`, allowing lock
resolution without executing the build backend. Installing the GPC profile builds
Gym and asciitree wheels in temporary storage. This succeeded for CPU/GPC.

Metadata-only locking, with uv available and a task-owned temporary cache:

```bash
uv lock --python /path/to/existing/python3.12 --no-python-downloads --no-build-isolation --cache-dir /tmp/tdyssey-lock-cache
uv lock --check --offline --python /path/to/existing/python3.12 --no-python-downloads --no-build-isolation --cache-dir /tmp/tdyssey-lock-cache
```

A blanket `--no-build` excludes Gym's source-only candidate even with static
metadata. For source-only work, inspect resolver output for unexpected builds and
remove task-owned temporary tooling/cache afterward.

## Training and data storage

The primary official CNN inference asset is about **1.02 GiB**, retaining all EMA
policy/encoder/normalizer state. Its published full training checkpoint is about
4.0 GiB; conversion needs about 5.1 GiB temporarily and removes that source after
verification. No original demonstration arrays are needed for inference.
`scripts/fetch_official_assets.py` defaults to a download plan. The checkpoint is
on the authors' website, so a Hugging Face account is unnecessary.

The separate GPC reference-policy runner needs one verified checkpoint of 364,554,150 bytes
(about 348 MiB), beyond the CPU environment. It does not need the full original
dataset, duplicate component checkpoints, world-model weights, or CUDA libraries.
See [policy resource commands](../../Tdyssey-raw-tl-portable-main/docs/POLICY_BASELINE.md). Public GPC assets need no
Hugging Face account/token; `huggingface-hub` is already in the shared environment.
Future private/gated resources or publishing may use individual accounts, with
credentials stored locally outside Git.

The pretrained GPC **world model** is a separate verified **33.42 MiB** checkpoint;
its loader/probe uses the existing CUDA/GPC/DP profile without dependency changes.
The first diagnostic added about 352 KiB of reports/images. Use the plan-only
fetch command and bounded check documented in [WORLD_MODEL.md](../../Tdyssey-raw-tl-portable-main/docs/WORLD_MODEL.md#pretrained-gpc-compatibility-check).
No full dataset, reward checkpoints, second environment or persistent cache is
required. Loading/inference is verified; target-shape adaptation and policy
optimization are not. See that document for source licensing and compatibility limits.

Do not reserve the full training budget just to develop the simulator. A course
pilot may eventually need roughly 40–60 GB: 8–12 GB for a CUDA environment, about
1.2 GB for published GPC assets as stored, 5–15 GB for new data/preprocessing,
5–15 GB for selected checkpoints, and temporary logs/videos. These are planning
ranges, not measured training requirements. Download assets only as needed and
record their exact revision, filename, size, and hash after verification.

At 96×96 RGB, 50k raw uint8 frames use about 1.38 GB; float32 uses four times that.
Store each frame once, construct history windows during loading, bound collection
runs, and retain selected checkpoints. Upstream world-model training saves every
epoch, so set a retention policy before long runs. Use configurable output roots
or another disk for large artifacts; never hardcode an author's home directory.

References: [uv caching](https://docs.astral.sh/uv/concepts/cache/),
[uv lock/sync](https://docs.astral.sh/uv/concepts/projects/sync/),
[PyTorch profiles](https://docs.astral.sh/uv/guides/integration/pytorch/),
[published PyTorch builds](https://pytorch.org/get-started/previous-versions/).
