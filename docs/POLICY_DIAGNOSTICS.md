# Inspecting the frozen policy's decisions

The official policy was trained on T demonstrations; it is **not an untrained
network**. The alphabet probe uses that checkpoint without further shape training.
This diagnostic observes its actual sampler and physical actions. It does not
change the sampler, optimize actions, train a policy or use a world model.

## What can be inspected

- Initial Gaussian action noise and every one of the 100 actual DDPM transitions.
- The checkpoint's inverse action normalization, mapping samples to workspace
  coordinates. Intermediate samples can lie outside the scene; the viewer includes
  an unclipped full-range plot rather than silently clamping them.
- All 16 predicted pusher targets, with indices 1 through 8 marked as the executed
  prefix. Index 0 and the later unused targets are not executed by the runner.
- Separate playback of denoising against the fixed input image and subsequent
  physical execution. Denoising iterations are not simulator time steps.
- Coverage/IoU over the full saved episode, the selected chunk, earliest best
  IoU, and whether decline starts before the next observation-based replan.

Dots are **pusher action targets**, not particles, object locations, predicted
future images or attention weights. The policy receives two images and pusher
positions; the encoder retains its original 84-pixel evaluation crop. It does
not receive measured overlap, reward or a success flag as inputs. The simulator
stops execution when coverage is strictly above 0.95. Sampling itself imitates
T demonstrations; it does not explicitly maximize measured overlap during inference.

This reveals an observable decision sequence, not a complete causal account of
the network's internal features. Image masking with fixed noise, multiple sampled
plans and shorter executed prefixes would test additional explanations. They
must be separately labeled experiments, not inferred from a visualization.

## Reproduce without installing anything new

Use the already-installed CUDA/DP profile and pinned official checkpoint. The
input is a complete `scripts/evaluate_shapes.py` report with saved physical states
and actions. It must contain one episode per selected shape, at most 300 controls
each. The original report remains outside Git; obtain it from its producer if it
is not present. The published MP4s alone do not contain enough information to
recover denoising or exact policy RNG state.

```bash
.venv/bin/python -B scripts/inspect_official_policy.py \
  --report artifacts/official-alphabet-2026-10-19/report.json \
  --shapes T D T_narrow \
  --output artifacts/policy-decisions-2026-10-20 \
  --seconds 420 --video
```

Choose a new output directory for each run. Source/checkpoint locations can be
overridden with `--source-root` and `--checkpoint`. The script refuses to overwrite
outputs, bounds GPU inference to three episodes through the chunk after the peak,
and checks wall time between policy and physical steps. Rendering/export follows
the compute phase. No assets, packages, environments or persistent caches are created.

Open `index.html` from the output directory in a browser; it works offline and
needs no server. Select a plan, move the denoising slider, then play physical
execution. The output also contains one overview PNG and, with `--video`, one
compact MP4 per shape. NPZ files retain full-precision normalized and workspace
samples, conditioning images/positions and executed frames. The HTML uses rounded
display coordinates. Full raw outputs remain ignored; no new publication exception
is introduced by this feature.

## Fidelity and interpretation

The scheduler observer copies samples from the unchanged upstream DDPM. It never
consumes random numbers or replaces outputs, and restores the scheduler even if
sampling fails. The existing CUDA parity check now verifies the current action
chunk, the following chunk's RNG stream, and exact equality of final displayed
prefix versus executed actions.

The diagnostic rebuilds each original episode from its initial state and original
policy seed, rather than restoring a mid-contact snapshot. Every replayed action
must match the saved float32 action exactly; body/controller states, metrics,
rewards, termination and contacts are checked too. The report stores source and
trace hashes and separate control, reset and denoising counts.

The optional hold comparison is a privileged, posthoc intervention: replay the
entire original prefix, then keep the pusher target at its position at the known
best moment for up to eight controls. Replaying the full prefix preserves contact
history. It does not teleport/freeze the object, zero velocities, select an action
using a world model, establish a deployable detector, or change official success.
Its interaction cost is recorded separately. Results belong to development-case
analysis, not a generalization rate or a policy-improvement claim.

See [validation evidence](VALIDATION.md) and [current project status](STATUS.md).

## Continue original T pushing after success

The standard official runner stops as soon as **coverage > 0.95**, which is not
the same as 95% IoU. To inspect retention after reaching that threshold, explicitly
enable continuation and request a fixed horizon:

```bash
.venv/bin/python -B scripts/evaluate_official.py \
  --episodes 1 --seed 100000 --policy-seed 42 \
  --steps 500 --continue-after-success --seconds 300 --video \
  --output artifacts/official-T-500-continuation-2026-10-20
```

This executes up to **500 physical controls (50 seconds at 10 Hz)**, including
the controls before the first success, not 500 additional controls afterward.
The last action chunk is shortened to the remaining budget. A wall-time limit
or closed viewer can still stop a run early; inspect report status and counts.

The raw simulator's reward and success flags are unchanged and recorded at every
control. Only the runner's decision to stop on a successful control is disabled.
There is no reset, artificial hold, object freeze, velocity change, new policy
seed or observation-history reset at success. The policy keeps choosing actions
normally. With `--video`, the MP4 includes control count, coverage and IoU;
`continuation.png` plots both metrics with the first-success marker.

This has its own report kind, `official_cnn_t_success_continuation_diagnostic`.
`success` means the episode ever crossed the original threshold; the
`post_success` summary separately records first crossing, first later loss,
fraction/minimum overlap after crossing, and final success/coverage. If there is
no crossing, post-success fractions are null. It is a development diagnostic,
not the paper's terminate-on-success evaluation or evidence of policy adaptation.
Without `--continue-after-success`, the existing 300-control bound and immediate
success stop remain in force.
