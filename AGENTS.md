# Tdyssey agent instructions

## Purpose and priorities

This is a three-author course research project on adapting a T-trained pushing
policy to new object shapes using non-expert interaction data and a learned world
model. The goal is one shared policy across multiple letter shapes, with no new
human demonstrations for adaptation. The released T checkpoint already uses
expert demonstrations. T-to-L is the first implementation check, not the final
shape scope; see `docs/PROJECT.md` and decision D026. It changes the pushed object,
not the robot embodiment. The course values
substantial implementation, experiments, and analysis; useful negative results
are acceptable. Do not reduce the project to a trivial demo just to guarantee success.
The primary baseline is the official RSS 2023 visual CNN Diffusion Policy and
original PushT environment; see `docs/OFFICIAL_BASELINE.md`. GPC is a separate
reference/comparison. Do not treat their checkpoints, geometry, or data as interchangeable.
Official capital-letter/near-T shape choices and matched-state rules live in `docs/OFFICIAL_SHAPES.md`.
Use `docs/OFFICIAL_COLLECTION.md` for primary training data; never mix it with GPC fixtures.
The custom visual model and pretrained GPC compatibility failures are documented
in `docs/WORLD_MODEL.md`. Pretrained loading is verified; official-domain adaptation
and reliable contact prediction are not. The released GPC sampler disables gradients.
The T palette comparison uses privileged auxiliary rendering, not an image-only
adapter; it explains much of the appearance gap but retains a left-push failure.
Use plain English in chat with the team. Avoid jargon where ordinary words work;
briefly explain technical terms when they are needed.

For substantial project work, read `docs/STATUS.md` and the relevant sections of
`docs/PROJECT.md`. For learning work, also read `docs/LEARNING_PLAN.md` for data,
interfaces, rewards, DPPO details and work packages; proposed settings are not
implemented defaults. Read `docs/POLICY_LEARNING.md` before modifying the implemented
tail-transition PPO variant; it is not an upstream DPPO reproduction or evidence
of successful adaptation. Use `docs/ENVIRONMENT.md` for dependency changes and setup;
`docs/ONBOARDING.md` for collaboration. The original Chinese research schedule is
background, not evidence that anything has been implemented or validated.

## Setup and resource use

Source-only work and installed development are both supported. Install packages,
create environments, download research assets, or start training only within the
scope authorized for the machine being used. Existing authorization persists in
the session; do not ask again for the same setup step. Authorization on one
author's machine does not authorize setup on another. `uv sync` and `uv run` can
install packages automatically. Source checks need no environment or downloads.

Prefer the locked CPU profile for storage-limited simulator development. Use the
CUDA profile when GPU work is needed. Avoid duplicate environments, persistent
download caches, and unneeded assets. Keep per-machine paths, installation consent,
disk limits, and personal preferences in local settings or ignored artifacts.
Shared documentation describes reproducible options for all three authors.

Do not modify another project's environment. Do not treat hardware or package
versions observed on one machine as requirements for every collaborator.

## Shared environment and code

- `pyproject.toml`, `uv.lock`, and `.python-version` define the shared environment.
  See `docs/VALIDATION.md` for which profiles and behaviors have actually passed
  runtime checks. Update manifest and lock together.
- Use the documented CPU or CUDA profile consistently when setup is authorized.
  Do not install both profiles or manually upgrade packages inside `.venv`.
- Reference training platform: Linux x86_64, Python 3.12. Other hardware/platforms
  need a documented compatibility decision, not silent dependency substitutions.
- Upstream commit IDs live in `manifests/upstream.json`; models and datasets need
  separate revision/hash records once actually downloaded.
- Keep generated data, checkpoints, caches, and raw video outside Git. The curated
  MP4 snapshot under `docs/media/official-alphabet-2026-10-19/` and reviewed PNG/JSON
  snapshot under `docs/media/world-model-2026-10-19/` are explicit publication
  exceptions; see `docs/RESULTS.md`. The latter has a manifest and 1 MiB cap.
  Keep full raw reports/frames and model weights out of these snapshots. Use configurable paths,
  never author-specific absolute paths.
- Use a branch or isolated checkout per author. Inspect `git status` before edits
  and preserve unrelated work. Split file ownership before any parallel work.

## Research correctness

- Verify the learning loop on one new shape, then expand to a shared multi-shape
  world model and policy. Keep genuine unseen-shape transfer and continual learning
  as separate stretch questions. Training on every letter cannot establish
  unseen-letter generalization; use disjoint scene splits and explicit shape exposure.
- Adaptation uses machine-generated interaction, not new human target-shape
  demonstrations or human success labels. Simulator-derived rewards/geometry are
  privileged supervision and must be disclosed. Do not use true future simulator
  states to score imagined transitions as if the world model had predicted them.
- A state-based mechanism study is valid but must be distinguished from a visual
  system. Changing representation requires matched policy training and baselines.
- Target-shape play data means adaptation, not zero-shot generalization.
- Keep rendering, collision geometry, target geometry, inertia, and symmetry-aware
  evaluation consistent. Restore full simulator state for matched comparisons.
- Align action normalization, observation histories, executed action prefixes,
  control-step counting, block rewards, and termination/bootstrap handling.
- Validate contact dynamics and model-selected optimistic actions, not just image
  reconstruction. Evaluate policy improvement in the physical simulator.
- First verify direct simulator policy optimization. Imagined reward gains alone
  are not evidence of physical improvement or new behavior.
- Disclose geometric exploration priors and privileged state/reward supervision.
- Separate training/validation/test trajectories and held-out shapes. Count real
  interaction and compute costs; report failures and seed variability honestly.

## Verification and handoff

Source-only check (Python 3.11+; no third-party packages, network, or file writes):

```bash
python3 -B scripts/check_repo.py
```

For implementation changes, add and run focused checks for meaningful behavior
when the required environment is available. Otherwise report exactly which checks
were static and what runtime verification is pending. Never claim a dependency
lockfile proves CUDA, simulator, or checkpoint compatibility.

Keep tests focused on meaningful behavior. Reuse existing coverage instead of
adding repeated input/default checks. Documentation wording and presentation-only
changes normally need the source/message checks, not new unit tests. Preserve
checks for physics, action timing, data separation, metrics and actual GPU updates.

After a substantive milestone, update `docs/STATUS.md` with completed work,
evidence, next task, and unresolved blockers. Record scope/dependency/protocol
decisions in `docs/DECISIONS.md`. Keep these short; chat history is not shared
project memory. Include reproducible commands and artifact references for results.
Do not mark an experiment completed without actual outputs.

## Commit and pull request rules

Follow `docs/GIT.md` and the shared templates when preparing commits or PRs.
Use Conventional Commit titles (`type(scope): description`, scope optional,
lowercase type/scope, at most 72 characters). For substantive commits include
Why, Changes, and Validation information; reference only real issues/PRs.
PRs use the template's Summary, Changes, Validation, Limitations, and Related issues
sections. Keep messages specific, concise, and accurate; never invent results,
signatures, or author credit. Check prepared messages with
`python3 -B scripts/check_messages.py` using the documented arguments.
Use message/body files for scripted Git and GitHub commands to preserve newlines.
Do not rewrite published commits or merge a PR without explicit authorization.
