# Official Diffusion Policy baseline

The primary baseline is the RSS 2023 authors' **visual CNN policy**, using their
released checkpoint and original planar PushT environment. The implementation
instantiates the unchanged official policy class and environment. It uses no GPC
network, weights, world-model ranking, image transforms, or normalizers.

The official CNN and Transformer generate actions; their architecture choice is
separate from the visual encoder. Start with CNN and one new shape, L. Transformer
comparison is optional later. Real-robot tabletop demonstrations and the separate
2024 point-cloud method DP3 do not change the current 2D simulation scope.

## Setup and resources

On an authorized Linux x86_64 machine, use the existing shared environment:

```bash
uv sync --locked --extra cu124 --extra dp --no-cache --link-mode copy
.venv/bin/python -B scripts/doctor.py --runtime --require-cuda --dp
```

Add `--extra gpc` to sync/`--gpc` to doctor if also developing the older GPC
comparison. Keep all selected extras consistent; do not create another environment
or manually upgrade its packages. `dp` uses the official Robomimic 0.2.0 encoder
dependency with the project's validated modern Python/PyTorch profile, rather than
installing the full historical Conda environment for unrelated benchmarks.

Fetch the pinned official source, using a new destination:

```bash
git clone --filter=blob:none --no-checkout --depth=1 https://github.com/real-stanford/diffusion_policy.git third_party/diffusion_policy
git -C third_party/diffusion_policy fetch --depth=1 --filter=blob:none origin 5ba07ac6661db573af695b419a7947ecb704690f
```

Materialize only the verified source paths:

```bash
python3 -B - <<'PY'
import json
from pathlib import Path
import subprocess
spec = json.loads(Path('manifests/official_dp.json').read_text())
subprocess.run(['git', '-C', 'third_party/diffusion_policy', 'sparse-checkout',
                'set', '--no-cone', *('/' + p for p in spec['source_sha256']),
                '/LICENSE', '/README.md'], check=True)
PY
git -C third_party/diffusion_policy checkout --detach 5ba07ac6661db573af695b419a7947ecb704690f
```

Preview/download the selected official checkpoint:

```bash
python3 -B scripts/fetch_official_assets.py
.venv/bin/python -B scripts/fetch_official_assets.py --download
```

The published training checkpoint is **4,293,295,813 bytes**, about 4.0 GiB. The
command verifies its recorded SHA-256, extracts `state_dicts.ema_model` with all
300 tensor entries and saved normalizers/buffers, and verifies the compact file.
It removes the full staging checkpoint only after successful conversion, retaining
**1,095,708,090 bytes**, about **1.02 GiB**. Allow about 5.1 GiB free during conversion.
The original file contains optimizer state and duplicate policy weights that are
unneeded for inference. It uses the publisher's dill/OmegaConf serialization during
conversion; subsequent inference loads the compact tensor/native-dictionary asset
with `weights_only=True`. Source/config/full/compact hashes are recorded in
`manifests/official_dp.json`. No Hugging Face account, login, demonstration dataset,
or second download cache is required. Existing mismatched/partial files are refused.

The retained EMA asset is for inference and initialization; resuming the original
optimizer would require the full training checkpoint. This does not prevent later
project fine-tuning with a new optimizer and an explicitly recorded protocol.

## Run and inspect

```bash
# GPU-only, original T benchmark setup; at most three episodes / 900 controls:
.venv/bin/python -B scripts/evaluate_official.py --output artifacts/official-baseline
# Visible 768x768 resizable window and small video previews:
.venv/bin/python -B scripts/evaluate_official.py --output artifacts/official-visible --display --video --hold-seconds 30
```

CUDA is required, with no automatic CPU fallback. The CPU runs physics/rendering.
Use a new output directory. Defaults are test seeds 100000–100002, at most 300
controls each, 100 denoising steps, and a run-wide 360-second wall-time limit.
Escape/close stops visible execution. Display pacing and hold time are separate
from synchronized planning latency. Reports preserve source hashes, device/package
versions, scene/policy seeds, per-step actions/states/scores/contacts, and costs.

The policy retains its published 84×84 evaluation crop, GroupNorm/encoder,
checkpoint normalizers, two-observation history, 16-action prediction, and eight
executed actions at indices 1:9. The environment keeps `legacy=True`, fixed goal,
original random reset distribution, original T geometry/mass/inertia, reward, and
termination. No geometry rejection or GPC state restoration is inserted. Histories
repeat the first observation and update after every control. Runtime tests check
these histories against the official `MultiStepWrapper`, including a partial final
chunk, and compare inference with a direct official-class call using the same RNG.

The paper's `test/mean_score` averages each episode's maximum clipped upstream
reward. It is **not** the success fraction or polygon IoU. This command reports all
three separately. Its local per-episode CUDA RNG is recorded; it does not reproduce
the historical runner's precise random stream or multi-process scheduling.

## Evidence and next work

The first three development scenes all reached upstream success, in 128/157/273
controls. Maximum IoU was 0.9241/0.9082/0.9236. These are three development checks,
not a full paper-score reproduction or a generalization/adaptation result.
See [validation evidence](VALIDATION.md) for commands, storage, and costs.

An official capital-L extension and paired probe now exist; see
[shape geometry and comparison protocol](OFFICIAL_SHAPES.md). The older I/L adapter
and play fixtures remain **GPC-specific**, with different coordinates/geometry and
physical choices. Do not pass official weights through the GPC runner or silently
mix its datasets with official L data. Retain GPC as a separate reference/comparison.

Our research contribution remains new-shape interaction data, world-model
adaptation/validation, and measured physical improvement from policy updates.
Compare the same initial policy without adaptation, world-model ranking, direct
simulator fine-tuning, and imagined fine-tuning. Training loops remain pending.

Sources: [official repository](https://github.com/real-stanford/diffusion_policy),
[released CNN configuration](https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml),
[released checkpoints](https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/).
