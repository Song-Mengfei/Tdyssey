# First visual world-model pilot

The first model predicts the next scene from **two RGB frames, two pusher
positions, the fixed goal pose and future pixel-position actions**. It is a new,
small CNN predictor, separate from the unchanged official Diffusion Policy.
It uses no object pose, velocity, contact or reward as model input. It currently
accepts official capital-L datasets only.

**There is no pretrained world-model checkpoint in use.** The first run trains
this predictor from scratch on six L play episodes (600 unique control steps).
It has not been pretrained on T or initialized from GPC or the official policy.
The official T-trained Diffusion Policy is a separate, frozen model. This pilot
checks the learning pipeline; it does not yet demonstrate transferring a world
model from T to L. A transfer comparison would need a recorded T training stage
and a matched L-only baseline before claiming a benefit from pretraining.

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
| Epochs | 10 | 50 |
| Optimizer updates | 750 | 3000 |
| Batch size | 8 | 32 |
| Predicted future controls | 4 | 4 |
| Wall time including loading | 300 seconds | 900 seconds |
| Retained output byte budget | 32 MiB | 64 MiB |

An epoch uses all valid training windows unless the update/time cap interrupts
it. The final partial epoch can be validated at the update cap; an epoch number
alone does not imply all its windows were visited. A wall-time failure is labelled
failed rather than a completed result. Reports include actual updates, windows,
forecast steps, selected epoch and measured peak GPU allocation. Only one best
checkpoint is retained, with weights/config/normalization; optimizer/resume state
is deliberately not saved. The report records all completed validation passes.

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

Next assess the released pretrained GPC world model and its compatibility with
the official T environment before spending more effort tuning this custom model.
The assessment must identify exact assets, resource costs and image/action/timing
differences; published availability alone does not establish compatibility.
Then check bounded T predictions against real contact/noncontact action effects
before attempting L adaptation. DINO-WM is an alternative candidate, with an
additional image-decoding question for our existing visual policy. See decision
D020 and [current next task](STATUS.md#next-task).

Keep this CNN as the L-only pipeline baseline. Any future variants need more
diverse contact data, reduced false noncontact motion and consistent image/motion
predictions. Model-test evaluation, multiple seeds and physical policy learning
remain pending. Verify direct simulator policy optimization before imagined-policy
claims; this checkpoint is not ready to supply the imagined environment.
