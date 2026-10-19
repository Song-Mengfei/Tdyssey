# First code milestone: extend the existing PushT environment

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

The geometry/evaluation/data code runs with the standard library. Source-only
adapter tests use explicit construction/API doubles. Separate opt-in runtime tests
have passed with real Pymunk and GPC on the locked CPU profile; see
`docs/VALIDATION.md`. CUDA and checkpoint validation remain pending. There is no
world-model or policy training implementation yet.
Trajectory records currently hold physical-state labels for diagnostics, not RGB
frames or the original GPC training-data format. Visual frame/history storage and
data conversion remain collection work. Training with these state labels must be
reported as using privileged simulator supervision.

## Run the core code without installation

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
The original T checkpoint and evaluator still need the resource-loading checks
described in M0 before comparing policy performance.

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
