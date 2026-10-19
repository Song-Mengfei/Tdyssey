# Pretrained policy baseline

This command loads GPC's published T-trained visual diffusion policy and executes
one sampled action chunk at a time in the physical simulator. It uses no world
model, reward predictor, or action ranking. It does not train or install anything.
See [validation evidence](VALIDATION.md) for actual observations and limits.

## Resources

Start with the locked CUDA/GPC environment and pinned sparse simulator checkout
from [environment setup](ENVIRONMENT.md). Add these small source files:

```bash
git -C third_party/gpc sparse-checkout add --no-cone /gpc_rank_evaluation/models.py /gpc_rank_evaluation/utils.py /gpc_rank_evaluation/eval_baseline.py /gpc_rank_evaluation/configs/gpc_rank_evaluation_config.yml /diffusion_policy_training/models.py /diffusion_policy_training/configs/train_config.yml
```

The pinned source hashes, policy settings, immutable Hugging Face revisions,
checkpoint size/hash, and original dataset schema are in
`manifests/gpc_policy.json`. Training and evaluation model definitions are identical
at this revision. The original domain18 dataset metadata describes 55,480 frames
in 500 demonstrations: uint8 HWC 96×96 images, float32 two-coordinate actions,
float32 five-coordinate states, and int64 episode ends. Only four schema files
were inspected; training arrays have not been downloaded or converted.

Only `action_policy_checkpoint/checkpoint_epoch_300/ema_nets.pth` is needed for
inference: **364,554,150 bytes**, about **348 MiB**. It contains all 208 averaged
parameter tensors, covering 91,123,778 parameters in the encoder and policy.
The separate component checkpoints duplicate these weights and are unnecessary
here. Buffers are absent after GroupNorm conversion. The loader verifies every
tensor's count, shape, dtype, and finite values before copying in upstream order,
and uses PyTorch's restricted `weights_only=True` load.

Preview the download with standard-library Python; downloading requires an
explicit flag and an authorized machine/storage budget:

```bash
python3 -B scripts/fetch_policy_assets.py
python3 -B scripts/fetch_policy_assets.py --download --max-mib 400
```

The download streams to one ignored file, verifies size/SHA-256, and retains no
second model cache. It refuses mismatched existing files and preserves preexisting
partial downloads. `--output` selects another checkpoint directory.

These public assets need **no Hugging Face account or token**. `huggingface-hub`
is already pinned in the shared environment; do not install an additional copy.
An account is optional for publishing models/data and required if a future chosen
resource restricts access. Each author keeps any future token in their own local
credential store or environment, outside Git. No login is performed by these scripts.

## Run a bounded comparison

```bash
.venv/bin/python -B scripts/evaluate_policy.py --output artifacts/policy-smoke --steps 32
# Longer single-scene development comparison, at most 900 actions total:
.venv/bin/python -B scripts/evaluate_policy.py --output artifacts/policy-baseline --steps 300 --seconds 360
```

Defaults are CUDA for the neural policy, four CPU threads, one episode per shape,
and seed 100000. CUDA unavailability is an error; there is no automatic CPU
fallback. The simulator physics and rendering remain on CPU. An explicit
`--device cpu` remains available for collaborators doing CPU debugging.
Use `--shapes T` for T alone. `--checkpoint` and `--gpc-root` accept other paths.
The command refuses existing output directories. Reports contain source/asset
hashes, seeds, full initial states, per-action state/metrics/contact counts,
planning latency, reset substeps, control steps, and denoising work. It records
failed or time-limited runs explicitly. A currently running denoising call can
finish after the wall-time limit; no further action is executed after that limit.
No videos or duplicated image datasets are saved.

Export a selected recorded episode when a visual inspection is useful:

```bash
.venv/bin/python -B scripts/replay_policy.py --report artifacts/policy-baseline/report.json --shape T --output artifacts/policy-baseline/T-preview.mp4
```

This replays the recorded actions from the saved pre-contact initial state,
checks every pose/velocity/reward/terminal flag, and decodes the saved video to
verify its frame count. It needs no checkpoint or further policy predictions.
The video report counts diagnostic replay controls and reset work separately.
The first T preview was about 200 KB. Video export renders offscreen using the
installed headless OpenCV build.

## Open a visible simulator window

```bash
.venv/bin/python -B scripts/evaluate_policy.py --output artifacts/visible-policy --shapes T --steps 300 --display --hold-seconds 60
```

This uses pygame to open a real **768×768 resizable window**. The underlying model
image stays 96×96; display enlargement does not change policy input. Resize or
maximize the window, or select `--window-size 1024`. Escape/close stops evaluation
and is recorded distinctly. `--hold-seconds` optionally keeps the final frame for
up to 60 seconds. Without `--display`, evaluation stays offscreen. A desktop
session/display is required for visible mode; no OpenCV GUI package is needed.

Visible runs pace control steps for inspection. Their elapsed time includes
display pacing and optional final-frame hold, so use the separately recorded
planning times or offscreen runs to measure inference speed. CUDA planning times
include synchronization. Recorded device/runtime/GPU fields distinguish actual
GPU runs from CPU debugging. Geometry, scoring, and policy inputs do not change.

Each scene uses T's complete initial physical state and goal across the selected
shapes, while retaining each shape's geometry/inertia identifiers. All shapes
must pass rotated wall/goal/pusher-clearance checks, or the candidate is rejected
for every shape. A T reference environment is reset even when evaluating only
I/L. Every actual reset substep, including rejected candidates, is counted.
Goals retain GPC's randomized distribution; the play collector currently uses
a fixed goal. These are explicitly different development protocols. Geometry
rejection changes the evaluation scene distribution from unrestricted upstream
seeds; the output is not a published benchmark score.

## Observation and action contract

- Two CHW float32 image observations in [0,1], initially repeating the reset
  observation; no ImageNet normalization. Images include the rendered goal.
- Two pixel-coordinate pusher positions, normalized with the published domain18
  minima/maxima. Object state and explicit goal vectors are not policy inputs.
- DDPM with 100 reverse steps, squared-cosine variance schedule, epsilon
  prediction, and scheduler clipping, matching upstream settings.
- Predict 16 actions, reverse the published action normalization, and execute
  indices `[1:9]`: eight pixel-coordinate targets. Shorten the final chunk to the
  remaining control budget; update observation history after every executed action.
- Use a local seeded PyTorch random generator for initial and denoising noise.
  Shape comparisons start with the same noise seed. CPU and CUDA generators can
  produce different samples for the same integer seed; cross-device trajectories
  are not expected to match.

GPC's file named `eval_baseline.py` actually ranks 50 candidates using a world
model and reward heads, includes candidate spreading, and uses a different slice
that can execute nine actions. This implementation deliberately uses one unranked
sample and the configured eight-action prefix. It is a plain-policy baseline
using verified GPC weights, not an exact reproduction of that evaluator or paper.

Keep upstream rewards and both termination flags. Report its success separately
from project IoU metrics: upstream stops immediately at coverage above 0.95, so
this runner cannot establish the project's five-step sustained-success condition.
A single scene can establish executable integration but cannot establish average
performance, an adaptation benefit, or generalization. Review representative
trajectories and repeated T seeds before treating the baseline as reproduced.
