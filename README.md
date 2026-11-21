# Tdyssey

[![Model weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-model%20weights-yellow)](https://anonymous-hf.com/a/uq38ysup3kuj/)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-live%20demo-blue)](https://tinyurl.com/tdyssey)

**One camera policy that pushes all 26 capital letters.** We start from the official RSS 2023
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) for PushT, which was trained on human demonstrations
of the **T only**, and adapt it to every capital letter with machine-generated data. **No new human demonstrations
were used.** The policy still sees only the camera image (two 96x96 frames) and the pusher position.

- **Talk and results data:** the final 37-slide deck is `presentation/talk/tdyssey-talk.pptx`, and the data and
  scripts behind every reported number and graph are in [`presentation/`](presentation/README.md).
- **Model weights:** [Hugging Face](https://anonymous-hf.com/a/uq38ysup3kuj/) (main model, a 192 px variant, the
  planner teacher's state world model, and single-file demo models).
- **Live demo:** [Hugging Face Space](https://tinyurl.com/tdyssey): pick a letter, drag the block, the goal and the
  pusher, and watch the original policy and ours on the same scene.

## Results

Success means the block covers more than 95% of its goal outline within 300 steps of 0.1 s. The final test uses 200
new scenes per shape that were never used for training or for picking models, and every model plays the same scenes.

| Shapes | Original policy | Ten-shape model | Main model (96 px) | 192 px model | Flow-DiT |
| --- | --- | --- | --- | --- | --- |
| **All 26 letters** | **2.0%** | 16.7% | **32.8%** | 34.0% | 45.5% |
| 9 earlier letters (T, Y, D, L, E, F, J, P, U) | 5.2% | 36.8% | 33.9% | 34.2% | 44.1% |
| 17 new letters | 0.3% | 6.1% | 32.2% | 33.9% | 46.3% |
| 9 never-seen shapes (digits 2, 3, 4, 5, 7, 8, plus, hash, arrow) | 1.0% | 4.3% | 8.0% | 9.4% | 17.3% |
| Goal covered at the end, 26 letters | 0.33 | 0.62 | 0.82 | 0.81 | 0.89 |

- The **main model** keeps the original network, the 96 px camera and the original inputs, after 50 correction rounds.
- The **192 px model** sees a sharper camera image (a sensor change) and comes from 2 rounds of its own line. Its
  +1.2 points over the main model has a 95% interval that includes zero.
- **Flow-DiT** is a larger network (a pretrained DINOv2 encoder and a 418M flow-matching transformer head, 505M in
  total) trained on the main model's final data: +12.7 points over the main model (95% interval +10.9 to +14.5).
- About two in three failures of the main model are near misses that reach 90% coverage at some point.

## How it works

1. **State world model.** A model of the pushing physics for all letters: the pusher's motion and the contact
   geometry use the simulator's exact rules, and how the letter moves when pushed is learned from machine-generated
   pushes. It runs about 200 times faster than one simulator process.
2. **Teacher.** A sampling planner that reads the exact simulator state and tries 1,024 candidate push plans inside
   the state world model at each decision. It uses privileged information, so it only makes training data and is
   never deployed.
3. **Copying.** The official training recipe on the teacher's demonstrations, labelled with the teacher's *intended*
   16-step plans (much better than its executed pushes), plus the original human T demonstrations, with an equal
   share for every letter.
4. **Correction rounds (DAgger).** The policy runs on new scenes, the teacher labels every situation it reaches, and
   the policy retrains on everything so far. The stages train on 5 shapes, then 10 shapes, then 26 letters.

**What did not work:** reinforcement-learning fine-tuning of the diffusion policy in the simulator (the training
signal was mostly noise, and larger steps lowered T success from 55% to between 0 and 30%), re-ranking the original policy's plans with the true
simulator, back-propagating through a learned model, weight blends, and world models learned directly from images
([image world model notes](docs/WORLD_MODEL.md)). After the talk we also painted the uncovered goal red in the camera
image to help the last nudge, and it did not raise success ([goal overlay](docs/GOAL_OVERLAY.md)).

## Run the trained policy

Set up the CUDA environment and the official Diffusion Policy as described below and in
[official setup](docs/OFFICIAL_BASELINE.md), download the weights, then from the repository root:

```bash
# 10 scenes per shape on six letters with the main model
.venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 400000 --scenes 10 --execute 8 \
    --shapes T A K M R Z --student /path/to/96px/model.pt --out my_test.json
```

Shape names: any capital letter and the never-seen `digit2 digit3 digit4 digit5 digit7 digit8 plus hash
arrow`. The tools build the network and the camera size from the checkpoint. The training and correction-round
scripts, with the seeds of every run, are in [scripts/learning](scripts/learning/README.md) and
[scripts/learning/experiments](scripts/learning/experiments/README.md).

## Documentation

- [The run scripts with their seeds](scripts/learning/experiments/README.md)
- [Inspect denoising, near misses and post-success continuation](docs/POLICY_DIAGNOSTICS.md)
- [Goal overlay: a rendering aid that shows the camera the uncovered goal](docs/GOAL_OVERLAY.md)
- [Train and watch the official T image world-model denoiser](docs/OFFICIAL_WORLD_MODEL.md)
- [Capital-letter geometry and matched comparisons](docs/OFFICIAL_SHAPES.md)
- [Run and inspect the official Diffusion Policy baseline](docs/OFFICIAL_BASELINE.md)
- [Watch the published alphabet videos](docs/ALPHABET_RESULTS.md) and [browse published figures](docs/RESULTS.md)
- [Image world-model experiments](docs/WORLD_MODEL.md)
- [Policy training methods and derivations](docs/POLICY_LEARNING.md) and [forward-process RL](docs/FORWARD_POLICY_LEARNING.md)
- [Implemented code and PushT integration](docs/IMPLEMENTATION.md)
- [Collect official interaction data](docs/OFFICIAL_COLLECTION.md), [prepare training batches](docs/TRAINING_DATA.md) and [the separate GPC reference collector](docs/COLLECTION.md)
- [Shared environment and storage](docs/ENVIRONMENT.md)

## Work without installing anything

Source and documentation work needs only Python 3.11+. These commands create no environment or package cache:

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
physical results per letter.

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

Each author uses a branch or separate checkout and merges through pull requests. Dependency changes include both
`pyproject.toml` and `uv.lock`. Use the same locked profile for matched experiments and record differences in
hardware and numerical behavior.

Keep datasets, environments, checkpoints, and raw videos outside Git. The
[curated alphabet clips](docs/ALPHABET_RESULTS.md) are the documented exception. The tracked
manifests record upstream revisions and verified asset hashes. Configurable
paths allow moving the project to another machine or disk.
