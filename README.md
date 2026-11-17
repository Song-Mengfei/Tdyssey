# Tdyssey

[![Model weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model%20weights-yellow)](https://anonymous-hf.com/a/uq38ysup3kuj/)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-live%20demo-blue)](https://tinyurl.com/tdyssey)

**One camera policy that pushes all 26 capital letters.** We start from the official RSS 2023
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) for PushT, which was trained on human demonstrations
of the **T only**, and adapt it to every capital letter with machine-generated data. **No new human demonstrations
were used.** The policy still sees only the camera image (two 96x96 frames) and the pusher position.

- **Model weights:** [Hugging Face](https://anonymous-hf.com/a/uq38ysup3kuj/) (main model, a 192 px variant, the
  planner teacher's world model, and single-file demo models).
- **Live demo:** [Hugging Face Space](https://tinyurl.com/tdyssey): pick a letter, drag the block, the goal and the
  pusher, and watch the original policy and ours on the same scene.

## Results

Success means the block covers more than 95% of its goal outline within 300 controls. Every test uses fresh scenes
never used for training or model selection, and every model in a table plays the same scenes.

**Final test (fixed in advance; 300 fresh scenes per shape):**

| Shapes | Original policy | Ten-shape model | 26-letter 96 px | 26-letter 192 px |
| --- | --- | --- | --- | --- |
| Ten earlier shapes (T, near-T, Y, D, L, E, F, J, P, U) | 9.3% | 33.0% | 33.7% | 34.2% |
| 17 other letters | 0.4% | 6.4% | 33.2% | 33.5% |
| **All 26 trained shapes** | **3.7%** | 16.3% | **33.4%** | 33.8% |
| 6 never-seen shapes (digits 3, 4, 7, plus, hash, arrow) | 1.2% | 2.8% | 4.7% | 5.9% |
| Goal covered at the end, 26 shapes | 0.35 | 0.63 | 0.82 | 0.81 |

Six more correction rounds then lifted the main model from 32.9% to 34.8% on 200 other fresh scenes per shape
(+1.9 points, 95% interval +0.1 to +3.6). Details, intervals and per-shape results:
[learning findings](docs/LEARNING_FINDINGS.md) and [decisions](docs/DECISIONS.md) (D051, D054).

## How it works

1. **World model.** A model of the pushing physics for all letters: the pusher's motion and the contact geometry are
   copied from the simulator, and how the letter moves when pushed is learned from machine-generated pushes.
2. **Teacher.** A sampling planner that reads the exact simulator state and tries about 1,000 candidate push plans per
   control inside the world model, re-planning after every push. It uses privileged information, so it only makes
   training data; it is never deployed.
3. **Copying.** The official training recipe on the teacher's demonstrations, labelled with the teacher's *intended*
   16-step plans (much better than its executed pushes), plus the original human T demonstrations, with an equal
   share for every letter.
4. **Correction rounds (DAgger).** The policy drives new scenes, the teacher labels every situation it reaches, and
   the policy retrains on everything so far; 5 -> 10 -> 26 letters.

**What did not work** (all documented with numbers in the [learning findings](docs/LEARNING_FINDINGS.md) and the
[experiment log](docs/LEARNING_EXPERIMENT_LOG.md)): reinforcement-learning fine-tuning of the diffusion policy in the
simulator (the training signal was mostly noise, and larger steps destroyed the T skill), re-ranking the original
policy's plans with the true simulator, back-propagating through the world model, weight blends, and a world model
learned directly from images ([world-model notes](docs/WORLD_MODEL.md)).

## Run the trained policy

Set up the CUDA environment and the official Diffusion Policy as described below and in
[official setup](docs/OFFICIAL_BASELINE.md), download the weights, then from the repository root:

```bash
# 10 scenes per shape on six letters with the main model
.venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 400000 --scenes 10 --execute 8 \
    --shapes T A K M R Z --student /path/to/96px/model.pt --out my_test.json
```

Shape names: any capital letter, `T_narrow`, and the never-seen `digit3 digit4 digit7 plus hash arrow`. The tools
build the network and the camera size from the checkpoint. The training and correction-round scripts, with the seeds
of every run, are in [scripts/learning](scripts/learning/README.md) and
[scripts/learning/experiments](scripts/learning/experiments/README.md).

## Documentation

- [Main results: one camera policy for all 26 letters, and what did not work](docs/LEARNING_FINDINGS.md)
- [Every learning run, including failed ones](docs/LEARNING_EXPERIMENT_LOG.md) and [the run scripts with their seeds](scripts/learning/experiments/README.md)
- [Decisions, each recorded before its result](docs/DECISIONS.md)
- [Inspect denoising, near misses and post-success continuation](docs/POLICY_DIAGNOSTICS.md)
- [Capital letters, near-T geometry and matched comparisons](docs/OFFICIAL_SHAPES.md)
- [Run and inspect the official Diffusion Policy baseline](docs/OFFICIAL_BASELINE.md)
- [Watch the published alphabet videos](docs/ALPHABET_RESULTS.md) and [browse published figures](docs/RESULTS.md)
- [Visual world-model experiments](docs/WORLD_MODEL.md)
- [Policy training methods and derivations](docs/POLICY_LEARNING.md), [forward-process RL](docs/FORWARD_POLICY_LEARNING.md) and [the RL research survey](docs/DIFFUSION_RL_RESEARCH.md)
- [Original learning plan](docs/LEARNING_PLAN.md) and [project scope](docs/PROJECT.md)
- [Implemented code and PushT integration](docs/IMPLEMENTATION.md)
- [Collect official interaction data](docs/OFFICIAL_COLLECTION.md), [prepare training batches](docs/TRAINING_DATA.md) and [the separate GPC reference collector](docs/COLLECTION.md)
- [Move to a training machine](docs/TRAINING_MACHINE.md) and [shared environment and storage](docs/ENVIRONMENT.md)
- [Current status](docs/STATUS.md), [coauthor onboarding](docs/ONBOARDING.md), [validation evidence](docs/VALIDATION.md)
- [Commit and pull request rules](docs/GIT.md), [agent instructions](AGENTS.md) and [Codex guidance](docs/CODEX.md)
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
CPU simulator-development and CUDA 12.4/12.8 policy/training profiles. Choose one profile based on the
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

For Blackwell use `--extra cu128` and doctor `--backend cu128`; target-machine
model checks are still required. Add `--extra tracking` for W&B. Training commands
accept `--tracking online --wandb-entity TEAM_OR_ACCOUNT`; local scalar logging
is always enabled. No account is needed for the local live viewer:

```bash
python3 -B scripts/view_training.py --run PATH/TO/TRAINING/RUN
```

Open http://127.0.0.1:8765. Charts show loss, fixed noise diagnostics, contact and
physical results per letter. See the [handoff](docs/TRAINING_MACHINE.md) for
authentication, remote viewing and repeatable longer runs.

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
