# First code milestone: extend the existing PushT environment

The primary policy baseline is now the [official CNN implementation](OFFICIAL_BASELINE.md).
The GPC extension and data pipeline described here remain a separate reference.
The [official capital-L extension](OFFICIAL_SHAPES.md) is now implemented; its
geometry and physical choices differ from the older GPC L fixtures.

PushT is an existing benchmark. Reuse the pinned GPC `PushTImageEnv`, its Pymunk
physics, controller, renderer, and diffusion-policy observation/action interface.
This change does not implement another physics engine or a replacement policy.

## Implemented

| Code | Behavior |
| --- | --- |
| `tdyssey/geometry.py` | T/I/L components, geometry hashes, shape-specific symmetry, area and inertia calculations |
| `tdyssey/envs/gpc.py` | GPC subclass factory: delegate T; build I/L through the existing object hook; capture/restore physical and controller state |
| `tdyssey/evaluation.py` | Separate coverage/IoU/pose metrics and consecutive-step success tracking |
| `tdyssey/state.py` | Versioned state records including velocity, angular velocity, force, torque, goal, and latest pusher target |
| `tdyssey/trajectories.py` | Validated control-step records, JSON round trips, and deterministic whole-trajectory splits |
| `examples/gpc_shape_env.py` | An interface check using a pinned, locally available GPC checkout |
| `tests/test_gpc_runtime.py` | Opt-in real physics, rendering, T comparison, and state/replay checks |
| `tdyssey/collection.py` | Bounded waypoint play with reset checks, seed records, contact diagnostics, and T+1 observations |
| `tdyssey/dataset.py` | Compact episode persistence, byte limits, file hashes, frozen splits, and read-back validation |
| `tdyssey/world_model_data.py` | Visual windows with separate privileged motion/mask labels and train-only statistics |
| `tdyssey/models/visual_dynamics.py` | Small action-conditioned visual CNN with autoregressive RGB/motion/reward/contact heads |
| `tdyssey/world_model_training.py` | Bounded CUDA training, persistence baselines, horizon/contact metrics and compact checkpoints |
| `scripts/train_world_model.py` | First visual L training pilot; see `docs/WORLD_MODEL.md` |
| `scripts/evaluate_world_model.py` | Frozen-catalog checkpoint evaluation; validation default, model-test explicit |
| `scripts/probe_world_model.py` | Bounded matched physical branches for candidate-action diagnostics |
| `scripts/summarize_world_model_probe.py` | Posthoc max/final/sum diagnostics from saved branches without new physics |
| `tdyssey/gpc_world_model.py` | Verified released GPC dynamics with isolated imports, native sampling and explicit four-frame/action alignment |
| `scripts/fetch_gpc_world_model.py` | Plan-only default; fetch just one verified 33.42 MiB world-model checkpoint without a cache |
| `scripts/probe_gpc_world_model.py` | Bounded native/official T contact prediction branches, persistence comparisons and previews |
| `tdyssey/appearance.py` | Temporary source-palette rendering with cleanup and known-color object/target image proxies |
| `scripts/probe_gpc_appearance.py` | Fixed six-scene palette experiment, shared physical branches, noise repeats and saved source frames |
| `tdyssey/official_collection.py` | Official T/L play with evaluation-scene exclusions, full resets and nonempty frozen splits |
| `scripts/collect_official_play.py` | Bounded official collection/read-back without model weights or training |
| `scripts/collect_official_contact.py` | Many official T batches with privileged contact-balanced play |
| `scripts/train_official_contact.py` | Resumable GPC denoiser updates from all finished contact batches |
| `scripts/visualize_world_model.py` | Click viewer: live official T versus predicted next frame |
| `tdyssey/official_contact_sampler.py` | Contact/miss headings and init-weight copy |
| `tdyssey/official_contact_trainer.py` | Global window sampling, frozen useful/miss validation, Ctrl+C resume |
| `tdyssey/region_weight.py` | Privileged case-split foreground masks for the denoiser loss |
| `tdyssey/motion_replay.py` | Moved-T replay mix for contact-train chunks |
| `scripts/collect_play.py` | Collection and validation command; see `docs/COLLECTION.md` |
| `tdyssey/training_data.py` | Frozen combined catalogs, duplicate/split checks, and lazy episode-safe training windows |
| `scripts/prepare_training_data.py` | Verify real CPU batches without training; see `docs/TRAINING_DATA.md` |
| `tdyssey/policy.py` | Verified GPC-published EMA loading and reference-policy inference |
| `tdyssey/baseline.py` | Matched initial scenes, bounded eight-action chunks, scores and accounting |
| `scripts/evaluate_policy.py` | CPU/CUDA-selectable unranked policy runner; see `docs/POLICY_BASELINE.md` |
| `scripts/fetch_policy_assets.py` | Plan-only default; explicitly download one size/hash-verified checkpoint |
| `scripts/replay_policy.py` | Export recorded actions to a small video and check physical replay |
| `tdyssey/viewer.py` | Optional 768×768 resizable pygame desktop viewer, independent of model input size |
| `tdyssey/official.py` | Pinned official CNN/environment loader, saved normalizers, actual-collision metrics |
| `scripts/fetch_official_assets.py` | Verify official training checkpoint, retain compact EMA, remove staging data |
| `scripts/evaluate_official.py` | Bounded CUDA T baseline with original environment, optional window/video, and real costs |
| `tdyssey/envs/official.py` | Preserve official T; add rigid capitals/near-T, matched centers, and checked snapshot restore |
| `tdyssey/letters.py` | Versioned edge-connected block capitals with geometry-specific symmetry |
| `scripts/preview_letters.py` | Small dependency-free SVG catalog of actual collision rectangles |
| `scripts/summarize_shapes.py` | HTML/video gallery and CSV with peak/final overlap and loss diagnostics |
| `scripts/extract_peak_frames.py` | Best-overlap screenshots decoded from saved videos, without physical replay |
| `scripts/evaluate_shapes.py` | Frozen official CNN on matched selected-letter scenes with geometry checks and explicit control/reset/denoising costs |

The geometry/evaluation/data code runs with the standard library. Source-only
adapter tests use explicit construction/API doubles. Separate opt-in runtime tests
have passed with real Pymunk and GPC on the locked CPU profile; see
`docs/VALIDATION.md`. CPU/CUDA checkpoint loading and bounded policy rollouts have
also passed. Policy inference defaults to CUDA without automatic CPU fallback.
A first visual world-model trainer is implemented with mixed validation results;
pretrained GPC loading and prediction checks now expose an official-scene appearance
mismatch and mixed native contact accuracy; see `docs/WORLD_MODEL.md`.
Policy optimization remains unimplemented.
The collector now stores RGB frames once as uint8 alongside physical-state labels
and exact observation timing. History-window loading and PyTorch batches are now
implemented; conversion to the original GPC training-data format remains pending.
Training with physical labels must be reported as using privileged simulator
supervision.

## Run the core code without installation

The `python -m tdyssey` preview below retains the older GPC geometry definitions.
Use `docs/images/official-L.svg` for the current capital-L dimensions.

```bash
python3 -B -m unittest discover -s tests -p 'test_*.py'
python3 -B -m tdyssey
# Optional small geometry preview, not a simulation dataset:
python3 -B -m tdyssey --svg /tmp/tdyssey-shapes.svg
```

At scale 30, T/L area is 6,300 pixel² and I area is 3,600 pixel². The half-turn
preview correctly gives I full overlap but not T/L. These are geometric checks,
not robot success rates.

## Use the existing environment

Obtain the sparse GPC checkout and install a profile using `docs/ENVIRONMENT.md`.
With the CPU/GPC environment installed, run from this repo:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MPLBACKEND=Agg .venv/bin/python -B examples/gpc_shape_env.py --gpc-root third_party/gpc --shape T --steps 5
```

Then check I and L. The command verifies the Git revision and unchanged upstream
environment/domain files, reuses the source class, and checks image shape/range.
It holds the pusher still; it does not load a policy or demonstrate task success.
It does not clone code or download assets automatically.

For integration with GPC evaluation code, the change at the construction point is:

```python
from pusht_env import PushTImageEnv              # Existing pinned GPC module
from tdyssey.envs.gpc import make_shape_env_class

ShapeEnv = make_shape_env_class(PushTImageEnv)
env = ShapeEnv(shape="I", domain_filename="push_t", resize_scale=96)
```

The original image/proprioception and pixel position-target actions stay inherited.
When starting inside GPC's evaluation directory, make Tdyssey importable via the
process's `PYTHONPATH`; do not install it into an unrelated project environment.
The GPC-published T checkpoint loads through `tdyssey/policy.py`; the initial
unranked T/I/L comparison and its protocol are documented in
`docs/POLICY_BASELINE.md`. Repeated T performance checks remain necessary before
treating that reference performance as reproduced. The official policy instead
loads through `tdyssey/official.py` and has separate evidence.

## Geometry and protocol decisions

T preserves the pinned upstream local vertices and its actual body-construction
path, including the upstream mass/inertia/center convention. Do not silently fix
its dynamics while reproducing the baseline. I/L each use one rigid body with
nonoverlapping convex components, mass 2, centroid-based local coordinates, and
uniform-density, area-weighted inertia. These are explicit new-shape conditions;
their density and inertia differ from each other and from the inherited T rule.

GPC already renders and scores goals from the body's collision shapes, so I/L
reuse that path. Its separate learned pose/reward heads and the OPT evaluator's
T-specific geometry still require adaptation; this patch does not make those
models generalize. Preserve upstream reward/termination behavior for reproduction.
GPC sets both termination flags on success; a later RL wrapper must define true
terminal versus timeout semantics consistently before direct or imagined PPO.

Our additional metric uses actual polygon IoU, distinguished from upstream area
coverage and clipped reward. I's 180° equivalence is shape-specific. The initial
success tracker uses 0.90 IoU held for five post-action control steps as a provisional
project metric; it does not override upstream task termination. To measure a hold
after upstream termination, a later evaluation wrapper needs an explicit protocol.

State restoration rebuilds a fresh space and restores bodies without an extra
unrecorded simulation step. It covers the obstacle-free scene's physical/controller
state, but does not preserve Chipmunk's cached contact impulses or random-generator
state for future resets. Repeated rollouts restored before first contact passed
the documented tolerance, but restoration during contact remains unverified.
Do not claim exact replay solely from JSON round-trip tests. Check initial shape
penetration, boundaries, and goal reachability before collecting target-shape data.

Record the source revision and local patches separately from the physics hash.
The hash records current controller/body configuration, not proof of code identity.
Freeze/export data splits: adding trajectories can change split membership. Policy
test initial states and held-out geometry require separate manifests.

Sources: [GPC](https://github.com/han20192019/gpc_code),
[Diffusion Policy PushT](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/pusht/pusht_env.py).
