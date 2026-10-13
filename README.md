# Tdyssey

A three-author course research project: can a policy trained to push a T-shaped
object learn useful pushing behavior for new shapes through non-expert interaction
data and reinforcement learning inside a learned world model?

The project is graded on effort. Substantial implementation, controlled
experiments, reproducibility, and an honest explanation of failures are useful
deliverables even if the ambitious adaptation/generalization goals fail.

**Current stage:** shared research and engineering scaffold. No simulator,
training loop, or benchmark result has been implemented here. The dependency
lockfile is a candidate specification, not a validated training environment.

## Start here

- [Project scope and milestones](docs/PROJECT.md)
- [Current status and next work](docs/STATUS.md)
- [Coauthor onboarding and task ownership](docs/ONBOARDING.md)
- [Shared environment and storage](docs/ENVIRONMENT.md)
- [Codex project guidance](docs/CODEX.md) and [agent instructions](AGENTS.md)
- [Decisions](docs/DECISIONS.md)
- [Original detailed research schedule](跨形状机器人世界模型研究_详细执行时间表.md)

## Work without installing anything

This laptop is currently source-only. Edit source and documentation using an
existing editor/Codex installation. The following check needs only Python 3.11+
and creates no environment or cache:

```bash
python3 -B scripts/check_repo.py
```

Do not run `uv sync`, `uv run`, package installers, training, or asset downloads on
this laptop unless the owner later authorizes them.

## Setup later on an authorized machine

Reference platform: Linux x86_64, Python 3.12.3, uv 0.13.0. The manifest offers
CPU development and CUDA 12.4 training profiles. Compatibility with GPC is still
pending runtime verification; see [environment notes](docs/ENVIRONMENT.md),
particularly for Blackwell GPUs and other operating systems.

With uv already installed on that machine, choose one profile:

```bash
# NVIDIA training / GPC reproduction candidate
uv sync --locked --extra cu124 --extra gpc
uv run --locked --extra cu124 --extra gpc python -B scripts/doctor.py --runtime --require-cuda --gpc
```

Or choose CPU development:

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu python -B scripts/doctor.py --runtime
```

Pass the same extras to subsequent `uv run` commands. Runtime verification here
checks basic packages, torch, and Pymunk, not a reproduced GPC result. No upstream
repository, checkpoint, or dataset is downloaded automatically by these scripts.

## Collaboration

Each author uses a branch or separate checkout. Pull shared documents before a
substantial task, agree on file ownership, and finish with a handoff in
`docs/STATUS.md`. Dependency changes include both `pyproject.toml` and `uv.lock`.
Use the same locked profile for matched experiments and record differences in
hardware and numerical behavior.

Keep datasets, environments, checkpoints, and videos outside Git. The tracked
manifests record upstream revisions and, later, actual asset hashes. Configurable
paths allow moving the project to another machine or disk.
