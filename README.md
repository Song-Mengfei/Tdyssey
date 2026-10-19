# Tdyssey

A three-author course research project: can a policy trained to push a T-shaped
object learn useful pushing behavior for new shapes through non-expert interaction
data and reinforcement learning inside a learned world model?

The project is graded on effort. Substantial implementation, controlled
experiments, reproducibility, and an honest explanation of failures are useful
deliverables even if the ambitious adaptation/generalization goals fail.

**Current stage:** first implementation of T/I/L geometry, an adapter for the
existing GPC PushT simulator, evaluation metrics, state/trajectory records, and a
bounded image/state interaction collector with episode-safe training batches,
and verified pretrained-policy loading with bounded physical T/I/L rollouts.
Core checks run without installing anything. The locked CPU/GPC profile and real
T/I/L simulator and CPU/CUDA checkpoint checks are validated. The first
single-scene policy runs did not solve the task; repeated baseline evaluation and
training remain pending. A resizable desktop viewer is available. See
[validation evidence](docs/VALIDATION.md).

## Start here

- [Project scope and milestones](docs/PROJECT.md)
- [Implemented code and existing PushT integration](docs/IMPLEMENTATION.md)
- [Collect bounded interaction data](docs/COLLECTION.md)
- [Prepare episode-safe training batches](docs/TRAINING_DATA.md)
- [Run and inspect the pretrained policy](docs/POLICY_BASELINE.md)
- [Current status and next work](docs/STATUS.md)
- [Coauthor onboarding and task ownership](docs/ONBOARDING.md)
- [Shared environment and storage](docs/ENVIRONMENT.md)
- [Codex project guidance](docs/CODEX.md) and [agent instructions](AGENTS.md)
- [Decisions](docs/DECISIONS.md)
- [Commit and pull request rules](docs/GIT.md)
- [Original detailed research schedule](跨形状机器人世界模型研究_详细执行时间表.md)

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
uv sync --locked --extra cu124 --extra gpc --no-cache --link-mode copy
.venv/bin/python -B scripts/doctor.py --runtime --require-cuda --gpc
```

An existing environment can be used directly through `.venv/bin/python`; activation
is optional. If using `uv run`, pass the same extras to preserve the chosen profile.
Doctor checks packages, torch calculations/gradients, and Pymunk; it does not reproduce a policy.
No upstream repository, checkpoint, or dataset is downloaded by these checks.

Policy runs default to CUDA with no automatic CPU fallback. After fetching the
selected checkpoint using [policy setup](docs/POLICY_BASELINE.md), open PushT:

```bash
.venv/bin/python -B scripts/evaluate_policy.py --output artifacts/visible-policy --shapes T --steps 300 --display --hold-seconds 60
```

The window starts at 768×768 and can be resized. `--display` controls window
visibility independently of GPU use; physics remains on CPU.

## Collaboration

Each author uses a branch or separate checkout. Pull shared documents before a
substantial task, agree on file ownership, and finish with a handoff in
`docs/STATUS.md`. Dependency changes include both `pyproject.toml` and `uv.lock`.
Use the same locked profile for matched experiments and record differences in
hardware and numerical behavior.

Keep datasets, environments, checkpoints, and videos outside Git. The tracked
manifests record upstream revisions and verified asset hashes. Configurable
paths allow moving the project to another machine or disk.
