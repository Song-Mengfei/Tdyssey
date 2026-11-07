# First visual world-model pilot

Selected original figures and compact reports are [published with the source](RESULTS.md).
Full raw reports, frames and weights remain local/ignored; the index states access limits.

This page records experiments already run. The [shared learning plan](LEARNING_PLAN.md)
explains the next model comparisons, automated data, prediction gate, policy rewards
and how validated dynamics will support DPPO. Direct simulator learning can be
implemented alongside model assessment; it does not wait for alphabet dynamics.

The first model predicts the next scene from **two RGB frames, two pusher
positions, the fixed goal pose and future pixel-position actions**. It is a new,
small CNN predictor, separate from the unchanged official Diffusion Policy.
It uses no object pose, velocity, contact or reward as model input. The original
experiments below used only L. The loader now supports mixed official-letter
catalogs, deriving labels/masks from each episode's geometry without adding a
letter ID to inputs. This enables shared-model experiments; it does not establish
shared accuracy or adapt pretrained GPC. See the
[forward-learning/data handoff](FORWARD_POLICY_LEARNING.md).

**This custom CNN pilot uses no pretrained world-model checkpoint.** Its first run
trains from scratch on six L play episodes (600 unique control steps).
It has not been pretrained on T or initialized from GPC or the official policy.
The official T-trained Diffusion Policy is a separate, frozen model. This pilot
checks the learning pipeline; it does not yet demonstrate transferring a world
model from T to L. A transfer comparison would need a recorded T training stage
and a matched L-only baseline before claiming a benefit from pretraining.
The later GPC compatibility experiments below do load a separate pretrained world
model; that model has not supplied an imagined policy-training environment.

This is a functioning training/validation baseline, **not yet a reliable simulator
for policy optimization**. The first GPU run improved one validation contact-motion
metric but made noncontact drift and overall position error worse. Candidate
reward predictions were nearly constant. Preserve these failures as evidence.

## Reproduce with an authorized environment

Use the existing [CUDA profile](ENVIRONMENT.md) and [official L data](OFFICIAL_COLLECTION.md).
No dependency changes, pretrained world-model download or second environment is
needed. Training and saved-model evaluation require CUDA and fail without it.
Source-only CLI help needs no third-party packages.

```bash
.venv/bin/python -B scripts/train_world_model.py --catalog artifacts/official-play-L-2026-10-19.catalog.json --output artifacts/visual-world-model-L-001
.venv/bin/python -B scripts/evaluate_world_model.py --catalog artifacts/official-play-L-2026-10-19.catalog.json --checkpoint artifacts/visual-world-model-L-001/best.pt --output artifacts/visual-world-model-L-001-validation.json
# Small matched physical check on validation states, not a policy improvement run:
.venv/bin/python -B scripts/probe_world_model.py --catalog artifacts/official-play-L-2026-10-19.catalog.json --checkpoint artifacts/visual-world-model-L-001/best.pt --output artifacts/visual-world-model-L-001-probe.json
# Explicitly posthoc score views from existing traces; zero new physics/inference:
python3 -B scripts/summarize_world_model_probe.py --probe artifacts/visual-world-model-L-001-probe.json --output artifacts/visual-world-model-L-001-probe-scores.json
```

Use new output paths. Generated checkpoints/reports/images remain outside Git.
The evaluator defaults to validation; `--split model_test` exists for a later
explicit final model comparison. **The real model-test episode was not read by
this pilot.** Do not use it to tune architecture, losses or stopping decisions.

| Training limit | Default | Maximum |
| --- | --- | --- |
| Epochs | 10 | 10,000 |
| Optimizer updates | 750 | 10,000,000 |
| Batch size | 8 | 1,024 |
| Predicted future controls | 4 | 4 |
| Wall time including loading | 300 seconds | 604,800 seconds |
| Retained output byte budget | 64 MiB | 16 GiB |
| Recovery interval in optimizer steps | 250 | 1,000,000 |

An epoch uses all valid training windows unless the update/time cap interrupts
it. The final partial epoch can be validated at the update cap; an epoch number
alone does not imply all its windows were visited. A wall-time failure is labelled
failed rather than a completed result. Reports include actual updates, windows,
forecast steps, selected epoch and measured peak GPU allocation. `best.pt`
retains the selected inference model. The newer `last.pt` recovery checkpoint
saves optimizer/RNG, report and next training-window cursor at configured step
intervals and validation boundaries. Resume into a new directory with matching
sources/catalog and the matching parent `best.pt`; partial epochs continue from
the cursor. Larger caps are configuration bounds, not measured resource needs.
The earlier published pilot used the original shorter caps and did not save
optimizer recovery. See [machine and logging guide](TRAINING_MACHINE.md) for
commands and shared W&B/local tracking. The report records completed validation passes.

## Model and training protocol

| Part | Architecture |
| --- | --- |
| Image encoder | Stack two RGB frames into 6 channels; three 3×3 stride-2 convolutions, 16 → 32 → 64 channels, each with GroupNorm and SiLU. Spatial size: 96 → 48 → 24 → 12. |
| Action/goal conditioning | Ten values: two pusher positions, relative action target, goal position and sine/cosine of goal angle. Fully connected layers 10 → 64 → 32; tile over the 12×12 map. |
| Shared features | Concatenate 64 image and 32 conditioning channels; 3×3 convolution to 64 channels. |
| Image decoder | Bilinear upsampling with encoder skip connections, 32 → 16 → 16 channels; predict an RGB change added to the latest image. |
| Motion output | Flatten the shared map; fully connected layers 9216 → 128 → 5 predict pusher dx/dy and object dx/dy/angle change. |
| Contact/reward output | Spatial average; fully connected layers 64 → 32 → 2. |

Hidden weights use PyTorch's default random initialization. The final image and
motion layers start at zero; the scalar output starts at training-mean reward
and contact frequency. These are starting values, not pretrained dynamics.

- Encode the two 96×96 RGB images together using three stride-2 convolution blocks
  with 16/32/64 channels, GroupNorm and SiLU. Condition the 12×12 feature map on
  pusher history, action target relative to current pusher, goal position and
  sine/cosine of goal angle.
- Decode with bilinear upsampling, skip connections and a residual RGB prediction.
  Separate heads predict pusher/object motion, contact probability and reward.
  The 1,302,570-parameter model starts from scratch. It does not reuse GPC weights
  or change the official T-trained policy's inputs or weights.
- Predict one action at a time. Append predicted images and pusher positions to
  the history for later steps. Future true images/poses are never fed into the
  rollout. The next action in the supplied sequence is the only new control input.
- Motion heads predict world-coordinate pusher dx/dy and object dx/dy/wrapped angle
  change. Object motion is an auxiliary output, not a hidden state rendered back
  into the predicted image. Image/motion/reward heads can disagree; this is a
  limitation to check before imagined policy learning.
- Normalize input coordinates with the known 512-pixel workspace (`x/256-1`);
  express action targets relative to current predicted pusher in units of 256.
  Fit output-motion means/scales on **600 unique training transitions** only,
  not duplicated windows or validation/test trajectories. Standard-deviation floors
  are 0.25 pixels for translations and 0.005 radians for angle change.
- Optimize AdamW, learning rate 0.0003, weight decay 0.0001, gradient norm capped
  at 1.0. Default seed 42. Shuffle episode order and then windows within each
  episode to preserve the one-decoded-episode cache without duplicating data.
- Loss weights are fixed for the first run: full-image MSE 1, object-region MSE 2,
  normalized pusher motion MSE 0.1, normalized object motion MSE 0.1, contact
  binary cross-entropy 0.05, reward MSE 0.1. No loss was tuned on model-test data.
- Pick the trained checkpoint with the lowest validation contact object-position
  error at the last forecast horizon; use the all-step value only if there is no
  contact group. Initial untrained results are reported separately and cannot
  win selection. Validation is used for model selection, not an unbiased test.

Object motion, reward/contact labels and the **object-region loss mask use
privileged simulator supervision**. The mask is the union of true before/after
collision geometry, sampled at image-pixel centers and dilated by one image pixel
for antialiasing. It is an approximate object-region diagnostic, not exact rendered
segmentation. It is used only in loss/evaluation and is never a network input.
RGB frames retain their original goal overlays. No image-only-supervision or
state-only-policy claim is made.

## Evaluation and simple baselines

For each horizon 1–4, report full-image/object-region MSE, pusher/object position
error in pixels, wrapped object-angle error in degrees, reward MAE and contact
Brier score/threshold-0.5 accuracy. Split results by whether the corresponding
future control has observed contact, as well as all controls. Counts are overlapping
window/forecast-step counts, not additional unique simulator interactions.

The baseline repeats the latest image and pusher position, predicts zero object
motion, and predicts constant training-mean reward/contact frequency. These heads
use the same training labels available to the learned model. Do not interpret low
background image error or a constant reward head as contact-dynamics validation.

The candidate probe restores the same full physical/controller state into a fresh
solver space for recorded/hold/toward/away action sequences (up to four controls).
Toward/away deliberately use privileged object position to generate candidates.
All candidates receive identical visual history; that position is not passed to
the model. It checks the current restored image against the saved quantized frame,
counts restores/control/reset work, and stops physical execution on source success.
Previous contact must be absent and upcoming recorded contact present; eligible
windows are the first disjoint validation windows, at most four.

The original probe's block score is maximum clipped reward in the action block.
At both selected starts this gave physical ties because an early near-goal position
hid later loss. **Near-zero max-score regret is uninformative here.** A separate,
explicitly posthoc final/sum analysis reads the same saved traces with no extra
physics; terminal success is absorbing for the sum calculation. It shows real
action differences that the almost-constant reward predictor failed to distinguish.
Any future ranking protocol must freeze its intended block/terminal scoring in
advance. Cached contact impulses/future-reset RNG are absent, and results apply
to these matched cold-restored branches rather than exact warm mid-contact replay.

## First run and next work

Ten epochs completed 720 GPU optimizer updates, with 5,760 overlapping windows /
23,040 forecast steps. Best validation epoch: 6. On its single validation episode:

| Four-control prediction | Visual model | Persistence baseline |
| --- | --- | --- |
| Object position error, contact (pixels) | 12.017 | 15.250 |
| Object position error, no contact (pixels) | 3.943 | 0.673 |
| Object position error, all (pixels) | 7.307 | 6.747 |
| Full-image MSE, all | 0.001851 | 0.001693 |

This is mixed/negative evidence for overall reliability, with validation-selection
bias and only one training seed/validation episode. The contact-only result cannot
be presented as a generalization or policy improvement. The physical probe used
two eligible states, eight branches, 32 controls and one reset substep. Predicted
reward ranges across candidates were below `2e-7`, while final physical reward
ranges were about 0.077 and 0.101. See [complete evidence](VALIDATION.md).

The first pretrained GPC assessment is now completed below. It loads unchanged
released weights and exposes an appearance mismatch and mixed native contact
prediction. It does not yet provide a reliable official-environment world model.
DINO-WM remains an alternative, with an additional image-decoding question for our
existing visual policy. See [current next task](STATUS.md#next-task).

Keep this CNN as the L-only pipeline baseline. Any future variants need more
diverse contact data, reduced false noncontact motion and consistent image/motion
predictions. Model-test evaluation, multiple seeds and physical policy learning
remain pending. Verify direct simulator policy optimization before imagined-policy
claims; this checkpoint is not ready to supply the imagined environment.

## Pretrained GPC compatibility check

The released phase-two world model is **8,729,827 parameters / 298 state entries**,
not the separate GPC action policy. The verified file is 35,043,916 bytes (33.42
MiB), SHA-256 `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`,
at Hugging Face revision `c16a467ec176af19bba5136564a81d8d61d72af9`.
Exact source hashes, model settings and checkpoint provenance live in
`manifests/gpc_world_model.json`. No dataset or reward-predictor weights are needed
for the prediction check. No package, lockfile or environment change was needed.

The image diffusion model uses a four-level U-Net with 96 channels per level,
two residual blocks per level and attention in the two lowest-resolution levels.
Four RGB images and four continuous 2D actions condition each next-image prediction.
Each action is projected from 2 to 64 dimensions; four projections form the
256-dimensional condition, combined with noise embeddings. The input convolution
receives 12 history channels plus 3 noisy-next-image channels. Released sampling
uses three Euler denoising steps. The release describes single-step training and
then multi-step training on demonstration/exploration trajectories; exact training
membership/seed provenance is not embedded in this weights-only file.

| Convention | GPC model / native scene | Official baseline |
| --- | --- | --- |
| Image | RGB 96×96, **[0,1] in actual training code** | Same shape/range; different rendering/preprocessing |
| History | Four images, three executed actions plus next proposed action | Policy itself uses two frames; maintain a separate four-frame dynamics history |
| Action | Absolute workspace pixel target, normalize with `a/511*2-1` | Same physical target meaning; do not substitute policy normalizers |
| Timing | 10 Hz control, 100 Hz physics | Same rates |
| Appearance | Red T, black target in domain 18 | Gray T, green target |
| T body | Mass 2, inertia 5700, center-of-gravity offset (0,0) | Mass 1, inertia 3000, offset (0,45); different local coordinates |
| Goal | Source setup uses randomized goals | Official baseline fixes its goal |

The sampler internally clips/quantizes to image values derived from [-1,1], but
that is **not** the training input scale. Preserve actual [0,1] RGB inputs. Keep
raw generated images in autoregressive history; clip only for observation metrics
and previews, recording the clipped fraction. Upstream `sample`, `denoise` and
output quantization disable gradients. This verified inference path does not
establish differentiable imagined-policy optimization. No upstream file is patched.
Legacy `diffusion`, `data` and `utils` imports are isolated during loading and
restored afterward so they do not replace other modules in the process.

The pinned repository has no top-level LICENSE and its model README states no
explicit weight license. The release is public, but redistribution terms remain
unresolved; retain only source/hash references in Git, not copied upstream code
or weights. This observation does not establish permission for redistribution.

With an authorized installed CUDA/GPC/DP profile, extend the existing pinned
sparse GPC checkout with the required small Python files:

```bash
git -C third_party/gpc sparse-checkout add --no-cone '/gpc_rank_evaluation/diffusion/*.py' '/gpc_rank_evaluation/data/*.py' /gpc_rank_evaluation/utils.py /gpc_rank_evaluation/eval_baseline.py /world_model_train_phase_one/train.py /world_model_train_phase_one/utils.py /world_model_train_phase_two/train.py /world_model_train_phase_two/utils.py /environment.yml /README.md
python3 -B scripts/fetch_gpc_world_model.py
# Explicitly fetch only the verified 33.42 MiB checkpoint, without a cache:
python3 -B scripts/fetch_gpc_world_model.py --download
.venv/bin/python -B scripts/probe_gpc_world_model.py --output artifacts/pretrained-gpc-T-001
```

The fetch command defaults to a plan, reuses a verified existing file and refuses
overwrites, corrupt streams or downloads above its budget. The probe defaults to
both simulator families, two hand-constructed pre-contact scenes (bottom/left),
four candidate sequences (hold/push/away/side), four future controls, a 120-second
wall budget and 4 MiB of output. It restores identical full physical/controller
states per candidate and uses matched noise seeds. Initial T/goal collision
geometry is aligned across families; appearance and physics remain different.
Scene placement/action directions use disclosed geometry knowledge. This is a
diagnostic, not a policy rollout or a population performance estimate.

The first run completed in 6.39 seconds: **76 controls, 760 control physics
substeps, two reset substeps, 20 restores, 64 forecast steps and 192 denoising
steps**. Peak allocated GPU memory was about 105.8 MiB; report/preview used
360,335 bytes, beyond the single checkpoint. No stored training/validation/model-test
dataset was opened and no weights were updated. Full evidence: [VALIDATION.md](VALIDATION.md).

At four controls, native bottom-push object-region MSE was **0.00785 vs 0.06821**
for persistence; native left-push was **0.17537 vs 0.17479**, with missed rotation.
On official inputs, generated T becomes red and the green target disappears.
Official bottom/left push object-region MSE was **0.15135/0.15588**, versus
**0.02801/0.06001** for persistence. The gray-color mask then finds no predicted
object; that is an appearance failure, not proof the generated image has no T.
Color-based centroids/IoU are only appearance-sensitive diagnostics, not a verified
pose or reward decoder. Geometry-derived ROI masks and true physical states are
evaluation labels, never model inputs. Do not claim a passed planning gate.

The matched palette experiment below isolates much of the appearance failure;
it also preserves the native left-push failure. Preserve those findings when
adapting the pretrained model; more data alone does not explain every error.

## Matched palette experiment

Run with the same authorized environment and existing checkpoint:

```bash
.venv/bin/python -B scripts/probe_gpc_appearance.py --output artifacts/gpc-appearance-T-001
```

`configs/gpc_appearance_probe.json` fixes six scenes, three model-noise seeds,
hold/push/away/side candidates and four predicted controls, with a 180-second /
8 MiB output cap. Two scenes were previously examined; four add other directions
and rotations. All six are development diagnostics, not an untouched final test.
No training data, model weights, packages or downloads are added. Original and
source-palette predictions share the same official physical branches and noise.
Native GPC physics is a separate control with aligned starting geometry.

`tdyssey/appearance.py` temporarily changes only object/goal render colors,
renders an auxiliary image and restores the colors/screen even on an exception.
The probe verifies full state, policy-image and render-cache preservation at every
frame. Source-view conversion matches torchvision's float32 multiplication by
1/255. This **requires simulator rendering access**; it is a privileged experimental
control, not an image-only deployed adapter or learned T-to-L adaptation. Future
images/poses remain evaluation labels. True physical future branches are recorded
once and reused across both views and all noise seeds.

Fixed known-color masks accept both gray and red T pixels, and green/black target
pixels. Report object-mask IoU, centroid error, visible-target IoU/recall, false
noncontact movement and RGB/ROI MSE by horizon/contact. The masks are appearance-
sensitive proxies, not a learned pose or reward decoder. Cross-palette RGB errors
also reflect color contrast and cannot alone establish a motion benefit. Missing
centroids remain absent with explicit counts. Persistence is evaluated per view.

At horizon four, on the unchanged official physics:

| Image-proxy diagnostic | Original input | Source palette | Persistence, source palette |
| --- | --- | --- | --- |
| Object-mask IoU, contact | 0.731 | 0.861 | 0.512 |
| Object centroid error, contact (pixels) | 10.901 | 3.272 | 14.568 |
| Object centroid error, no contact (pixels) | 6.996 | 0.205 | 0 |
| Visible-target recall, all | 0.033 | 0.954 | 0.936 |

Contact rows contain six unique physical branches × three model-noise repeats,
not 18 independent trials. The complete probe has 48 physical candidate branches
across both simulator families. It used 228 controls, 2280 control substeps, two
resets, 60 restores, 288 auxiliary renders and 864 forecast steps / 2592 denoising
steps, in 34.51 seconds. Peak GPU allocation remained about 105.8 MiB; saved
source frames, report and comparison image use about 2.13 MiB. Raw frames are
compressed once per branch for later analysis without new physical interaction.

The mean hides a remaining failure: source-palette left-push object IoU is only
0.451–0.456, compared with 0.887–0.904 under original inputs. Actual official
rotation is about -9.84 degrees; the palette-conditioned prediction largely
remains still. Most other directions improve, so palette conversion cannot be
described as uniformly better or as a complete dynamics fix. Target shape/recall
also remain imperfect. The failed first run and its separate resource counts are
preserved in [VALIDATION.md](VALIDATION.md).

Next calibrate the pretrained model on bounded **official-RGB T training data**
with varied contact/rotation, then compare unchanged/adapted weights on disjoint
scenes and record action effects. This can address appearance without requiring
an auxiliary simulator renderer. Keep the palette control as a diagnostic and
the untouched model-test split for a fixed final comparison. Check prediction and
physical action ranking before L adaptation; policy optimization still needs its
separate direct-simulator baseline.
