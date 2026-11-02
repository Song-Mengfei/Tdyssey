# Static data and model contract review

Scope: source text and small declared metadata only. No Python, arrays, weights,
model/measurement statistics, SDK, native experiment or physical operation was run.
This checks implemented indexing and boundaries; it does not establish actual RGB
quality, predictive accuracy, or a cause for the evaluation results.

## Five real frames and four executed controls

Define `o_i` as the observation before control `a_i`; `a_i` produces `o_(i+1)`.
The collector saves the initial observation, then executes a control, records its
before/action/after transition, and appends the resulting observation
(collector:124 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/official_collection.py:124`),
collector:128 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/official_collection.py:128`),
collector:133 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/official_collection.py:133`),
collector:138 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/official_collection.py:138`),
collector:144 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/official_collection.py:144`)).
`Transition` additionally requires one control-step increment and `after.pusher_target == action`
(transitions:26 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/trajectories.py:26`)).

At anchor `t`, the dataset slices images `[t-3:t+2]` and transition actions
`[t-3:t+1]`: five frames `o_(t-3)..o_(t+1)` and four controls `a_(t-3)..a_t`.
The objective uses the first four images as history and image4 as the clean target;
the last action therefore advances the latest history frame to that target
(data:237 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation_data.py:237`),
loss:168 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:168`)).
For `t=3`, this is `o0,o1,o2,o3 -> o4` conditioned on `a0,a1,a2,a3`, without padding.
The first frozen schedule declaration is cursor0/index1048/anchor81/T, so its
specified window is `o78..o82`, actions `a78..a81`, target `o82`; the next is
cursor1/index2034/anchor97/L, giving `o94..o98`, actions `a94..a97`, target `o98`
(schedule:418 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/artifacts/wm-sampling-b-only-20261026/training-schedule.json:418`)).
These are metadata/index deductions, not decoded-frame verification.

The schedule maps `dataset_offset + anchor - 3`, alternates T/L by cursor, and the
consumer returns exactly its recorded indices; training reserves that cursor and
those indices before its forward
(schedule source:123 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_training_schedule.py:123`),
consumer:238 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_training_schedule.py:238`),
train:319 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:319`)).
The rule explicitly uses privileged train contact/rotation labels for selection;
the model batch itself contains only image/action, not poses, masks or future truth
(rule:14 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_training_schedule.py:14`),
batch:280 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:280`)).
Held role selection and mutated episode declarations are rejected before payload reads
(data:171 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation_data.py:171`),
data:194 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation_data.py:194`)).

## Pixel and action contract

Training converts saved uint8 RGB to float32 `[0,1]`, applies the same fixed T1 to
all five frames, and normalizes absolute `(x,y)` controls once as `a/511*2-1`.
`batch` only stacks/moves these arrays; `_batch` validates and passes the normalized
actions unchanged (data:238 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation_data.py:238`),
loss input:109 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:109`)).
Pinned phase-two training explicitly supplies action min `[0,0]`, max `[511,511]`
to its dataset, whose normalization is the same affine map; its image conversion
ends in `[0,1]`, with the optional image normalization commented out
(upstream train:63 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/train.py:63`),
upstream dataset:97 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/utils.py:97`),
upstream images:196 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/utils.py:196`)).
This source agreement does not identify every dataset/domain used for the released checkpoint.

Controls are world-frame pusher position targets, not displacement, object-local
coordinates or 96-pixel image coordinates. The official environment is a 512-unit
window, applies PD control to `action-agent.position`, renders HWC then resizes,
and only changes image layout/range to CHW `[0,1]`
(official action:118 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/diffusion_policy/diffusion_policy/env/pusht/pusht_env.py:118`),
official render:216 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/diffusion_policy/diffusion_policy/env/pusht/pusht_env.py:216`),
image observation:37 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/diffusion_policy/diffusion_policy/env/pusht/pusht_image_env.py:37`)).
The local drawing helper declares positive-y-down; T1 declares no spatial operation,
so it does not rotate, flip or rescale control coordinates
(drawing:53 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/diffusion_policy/diffusion_policy/env/pusht/pymunk_override.py:53`),
T1:58 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/wm_compat_transform.py:58`)).

## Rollout alignment and feedback boundary

For latest real frame `o_t`, real history is `o_(t-3)..o_t`, past controls end at
`a_(t-1)`, and future controls start at `a_t`.

| Prediction | Four action indices | AR context; TF replaces predictions with corresponding real frames |
| --- | --- | --- |
| H1: `o_(t+1)` | `t-3..t` | `o_(t-3),o_(t-2),o_(t-1),o_t` |
| H2: `o_(t+2)` | `t-2..t+1` | `o_(t-2),o_(t-1),o_t,pred_(t+1)` |
| H3: `o_(t+3)` | `t-1..t+2` | `o_(t-1),o_t,pred_(t+1),pred_(t+2)` |
| H4: `o_(t+4)` | `t..t+3` | `o_t,pred_(t+1),pred_(t+2),pred_(t+3)` |

The implementation takes the last four concatenated controls and normalizes them
once. T1 is applied only to real history/TF frames; raw AR output is appended
directly. TF H2/H4 sees only real frames through `t+1`/`t+3`, excluding its current
target. Physical truth is attached to the result after sampling
(rollout:385 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:385`),
rollout:396 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:396`),
rollout:419 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:419`),
feedback:428 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:428`)).
The MAE's `clip(raw,0,1)` is a reporting operation, not the AR feedback.

One important scope distinction remains: the native denoiser clamps/quantizes its
denoised estimate to `[-1,1]`; the sampler returns its final Euler state, which
local `NativeSampler` returns without converting to `[0,1]`. Thus subsequent AR
contexts are not guaranteed to remain in the real-training RGB range. No negative
values are asserted here without array inspection. This preserves the pinned
native feedback convention; the local five-frame one-step objective has no later
target trained on its own generated feedback
(native wrap:86 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/denoiser.py:86`),
native return:64 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/diffusion_sampler.py:64`),
local return:490 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:490`)).

## Checkpoint and mode mapping

`frozen` maps to the exact phase-two initial SHA `4084...1892`; `adapted` maps to
pilot03 checkpoint500 SHA `0bfd...41d3`, extracting its `model` mapping. The loader
checks source logical500/config/account/schedule boundary and fixed model keys,
shapes, dtypes and finite values; it restores no optimizer/RNG/account state
(pins:14 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_evaluation_assets.py:14`),
loader:73 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_evaluation_assets.py:73`)).
Training selects named parameters under `inner_model.` and calls `model.train()`;
the eval-only constructor instead freezes gradients and selects `.eval()` without
training setup. The evaluation engine repeats that mode boundary after construction
and serially releases each model
(selection:194 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:194`),
constructor:266 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:266`),
engine:289 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_checkpoint_evaluation.py:289`)).

Conclusion: source inspection found no local off-by-one window/action shift or
double action normalization/T1 feedback application. It exposes a real-versus-AR
conditioning/exposure distinction, not a measured failure cause. Array alignment,
actual ranges and outcome attribution remain outside this static review.
