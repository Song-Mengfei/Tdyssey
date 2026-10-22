# Tdyssey

A three-author course research project: can a policy trained to push a T-shaped
object learn useful pushing behavior for new shapes through non-expert interaction
data and reinforcement learning inside a learned world model?

The intended outcome is one shared policy across multiple shapes, using no new
human demonstrations for adaptation. The released T policy already uses expert
pretraining. T-to-L is the first learning check before expanding to mixed shapes;
it is not the final project scope.

The project is graded on effort. Substantial implementation, controlled
experiments, reproducibility, and an honest explanation of failures are useful
deliverables even if the ambitious adaptation/generalization goals fail.

**Current stage:** the official RSS 2023 visual CNN policy runs on CUDA and solved
three original T development scenes. T/L data collection, alphabet evaluation and
actual denoising/continuation inspection work. In one common alphabet scene, only
T and Y succeeded. World-model pilots expose contact/reward failures; reliable
imagined rollouts remain pending. A direct-simulator tail-PPO trainer now passes
an initial GPU update/reload check; policy improvement is not established. GPC reference work stays
separate from the official baseline. See [results](docs/RESULTS.md),
[status](docs/STATUS.md) and [validation evidence](docs/VALIDATION.md).

## Start here

- [Full learning plan: data, world models, rewards, DPPO and developer tasks](docs/LEARNING_PLAN.md)
- [Run policy training; method survey and mathematical derivations](docs/POLICY_LEARNING.md)
- [Project scope and milestones](docs/PROJECT.md)
- [Inspect denoising, near misses and post-success continuation](docs/POLICY_DIAGNOSTICS.md)
- [Implemented code and existing PushT integration](docs/IMPLEMENTATION.md)
- [Collect official T/L interaction data](docs/OFFICIAL_COLLECTION.md)
- [Separate GPC reference collector](docs/COLLECTION.md)
- [Prepare episode-safe training batches](docs/TRAINING_DATA.md)
- [Visual training and pretrained world-model checks](docs/WORLD_MODEL.md)
- [Browse published experiment figures and results](docs/RESULTS.md)
- [Run and inspect the official Diffusion Policy baseline](docs/OFFICIAL_BASELINE.md)
- [Watch the published alphabet videos](docs/ALPHABET_RESULTS.md)
- [Capital letters, near-T geometry and matched comparisons](docs/OFFICIAL_SHAPES.md)
- [Separate GPC reference-policy comparison](docs/POLICY_BASELINE.md)
- [Current status and next work](docs/STATUS.md)
- [Coauthor onboarding and task ownership](docs/ONBOARDING.md)
- [Shared environment and storage](docs/ENVIRONMENT.md)
- [Codex project guidance](docs/CODEX.md) and [agent instructions](AGENTS.md)
- [Decisions](docs/DECISIONS.md)
- [Commit and pull request rules](docs/GIT.md)
- [Original detailed research schedule](跨形状机器人世界模型研究_详细执行时间表.md)

## The plan at a glance

**Goal:** improve one T-trained visual diffusion policy across multiple letters,
using automated interaction and rewards, with no new human adaptation demonstrations.
The original T policy was trained on expert demonstrations. T/L is the first
learning check; A-Z is a later expansion, not a prerequisite for getting started.

| Order | What we build | How we know it works |
| --- | --- | --- |
| Now, policy work | A bounded DPPO learning loop in the physical T/L simulator | Correct probabilities/rewards, actual GPU weight updates, then separate-scene physical improvement and T retention |
| Now, world-model work | Compare compatible pretrained prediction models; adapt one if necessary using automated data | Predict contact, rotation, pusher position and reward; correctly rank actions, including near the goal |
| Join the two | Freeze validated dynamics and let DPPO practice in short predicted rollouts | Updated policy improves in the physical simulator, compared with direct-simulator training |
| Expand | Train shared dynamics/policy on a fixed diverse shape set, then A-Z if affordable | Per-shape results, costs, failures and retained T performance |

**What does the reward teach?** The proposed training reward combines completion
at the original **coverage > 95%** threshold with a carefully discounted coverage
progress term. Coverage is target area occupied by the object; it differs from
IoU. DPPO increases the likelihood of denoising decisions that lead to better
future rewards. Maintaining alignment after success is a separate training/evaluation
task; stopping the original episode at first success does not train that behavior.

**Why not just choose the biggest world model?** It must accept our continuous
actions and predict their physical consequences, not merely generate plausible
video. Pretrained GPC loading works, but its official-domain contact prediction
is not reliable yet. A-Z training is needed only as justified by the expansion;
training every letter cannot demonstrate unseen-letter generalization.

**Why not just use the physical simulator?** We can, and it may be the better
choice for cheap 2D PushT physics. The central comparison is frozen policy vs
simulator-trained DPPO vs world-model-assisted DPPO. Count world-model data and
training costs too; lower physical interaction use, faster training or better
physical performance must be measured rather than assumed. A useful negative
result is that the learned model adds cost or errors without improving the policy.

**Ready today:** official policy/environment, T/L collection and batches, alphabet
evaluation, world-model pilots, denoising/continuation inspection and a bounded
direct-simulator tail-PPO trainer. **Still to establish:** useful learned adaptation,
a reliable shared world model and imagined policy updates.
The world model is intended as a training aid; the updated policy can act on real
observations without using it for test-time planning.

Read the [detailed implementation and experiment plan](docs/LEARNING_PLAN.md) for
data sizes/splits, model losses, reward equations, training steps, proposed starting
settings and three developer work packages. Proposed settings are not tested defaults.

To check a small real GPU policy update using an existing authorized CUDA/DP setup:

```bash
.venv/bin/python -B scripts/train_official_policy.py --config configs/dppo_smoke.json --output artifacts/dppo-smoke-001
```

The [policy-learning guide](docs/POLICY_LEARNING.md) explains the DPPO-style variant,
rewards, small/full-network configurations, mathematical limits and current evidence.
The smoke run tests mechanics; it is not an L-learning result.
For the main capacity comparison, use full action-network or distributed LoRA
profiles in the [training guide](docs/POLICY_LEARNING.md), rather than assuming
head-only updates suffice. LoRA changes which parameters learn; the PPO objective
remains the same.

Training runs without a window. To watch a saved trained policy, use
[`scripts/evaluate_trained_policy.py`](scripts/evaluate_trained_policy.py) with
`--adapter PATH/last.pt --display`; see the [preview command](docs/POLICY_LEARNING.md#run-and-inspect).

## Work without installing anything

Source and documentation work needs only an existing editor and Python 3.11+.
These commands create no environment or package cache:

```bash
python3 -B scripts/check_repo.py
python3 -B -m unittest discover -s tests -p 'test_*.py'
python3 -B -m tdyssey
```

## Install a development environment

Reference platform: Linux x86_64, Python 3.12.3, uv 0.13.0. The manifest offers
CPU simulator-development and CUDA 12.4 policy/training profiles. Choose one profile based on the
machine and task. See [environment notes](docs/ENVIRONMENT.md) for storage-saving
setup, the small upstream checkout, and platform limitations.

With uv already installed on that machine, choose one profile:

```bash
# Storage-saving CPU setup, including the existing GPC simulator
uv sync --locked --extra cpu --extra gpc --no-cache --link-mode copy
.venv/bin/python -B scripts/doctor.py --runtime --gpc
```

For GPU work, choose CUDA instead:

```bash
uv sync --locked --extra cu124 --extra dp --no-cache --link-mode copy
.venv/bin/python -B scripts/doctor.py --runtime --require-cuda --dp
```

An existing environment can be used directly through `.venv/bin/python`; activation
is optional. If using `uv run`, pass the same extras to preserve the chosen profile.
Doctor checks packages, torch calculations/gradients, and Pymunk; it does not reproduce a policy.
No upstream repository, checkpoint, or dataset is downloaded by these checks.

Policy runs default to CUDA with no automatic CPU fallback. After fetching the
selected checkpoint using [official setup](docs/OFFICIAL_BASELINE.md), open PushT:

```bash
.venv/bin/python -B scripts/evaluate_official.py --output artifacts/official-visible --display --video --hold-seconds 30
```

The window starts at 768×768 and can be resized. `--display` controls window
visibility independently of GPU use; physics remains on CPU.

## Collaboration

Each author uses a branch or separate checkout. Pull shared documents before a
substantial task, agree on file ownership, and finish with a handoff in
`docs/STATUS.md`. Dependency changes include both `pyproject.toml` and `uv.lock`.
Use the same locked profile for matched experiments and record differences in
hardware and numerical behavior.

Keep datasets, environments, checkpoints, and raw videos outside Git. The
[curated alphabet clips](docs/ALPHABET_RESULTS.md) are the documented exception. The tracked
manifests record upstream revisions and verified asset hashes. Configurable
paths allow moving the project to another machine or disk.
