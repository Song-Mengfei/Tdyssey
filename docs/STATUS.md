# Shared project status

Updated: 2026-10-19. Stage: bounded interaction collection; no training experiments completed.

PR #1 merged the shared setup and first code milestone. PR #2 contains CPU setup,
real simulator tests, shared documentation cleanup, and bounded interaction
collection together. Review branch: `build/minimal-runtime-setup`.
Main changes continue through pull requests.

## Completed

- Defined the three-author course project's research question, staged milestones,
  candidate dependency lock, upstream revisions, and contribution rules.
- Added shared Codex instructions, onboarding, experiment-record templates, and
  dependency-free source/message checks.
- Implemented T/I/L geometry, coverage/IoU and symmetry-aware metrics, sustained
  success tracking, physical/controller state records, trajectory validation,
  JSON round trips, and whole-trajectory splits.
- Added a GPC adapter that delegates T construction, physics, control, rendering,
  reward, and step results to upstream. I/L components attach to one rigid body.
- Installed and checked the locked CPU/GPC profile using Python 3.12.3 and uv
  0.13.0 without a persistent download cache. The environment uses about 1.33 GiB.
- Fetched a sparse, pinned upstream simulator checkout without datasets or weights.
- Added opt-in real simulator tests and portable setup/validation instructions.
  Shared documents describe options for all authors; private machine details and
  runtime reports stay in ignored local artifacts.
- Added a bounded real interaction collector with random or explicitly privileged
  object-directed waypoint play, geometry-checked fixed-goal resets, and separate
  reset/control-step accounting. Stores uint8 frames once alongside state/actions,
  contact diagnostics, source/file hashes, and frozen whole-episode splits.
- Added read-back validation and refusal to overwrite earlier datasets. Collection
  and validation commands are in `docs/COLLECTION.md`.

## Verification status

See `docs/VALIDATION.md` for commands and limits.

- CPU/GPC doctor: all direct versions matched; PyTorch tensor, torchvision operator,
  and Pymunk integration checks passed. 85 packages installed from the shared lock.
- T/I/L five-control-step interface checks passed with 3×96×96 images.
- Five real simulator tests passed: original/adapted T rollout agreement, actual
  I/L contacts, collision geometry/reward agreement, state restoration, and fresh
  pre-contact replay. This is not a trained-policy performance result.
- All 48 tests passed together in the installed environment: 38 source-only and
  ten opt-in runtime tests. Source-only discovery passed and skipped runtime tests.
- Small collector examples passed for T/I/L: 140 control steps across four episodes,
  four reset substeps, 15 L-shape contact steps, and 590,024 bytes of saved output.
  These retained example counts exclude earlier prototype and diagnostic/test work.
  Raw data is ignored; portable findings are in `docs/VALIDATION.md`.
- CUDA, checkpoint loading, original dataset conversion, policy/world-model
  training, reset-safety coverage, and replay restored during contact remain pending.
- No datasets, model checkpoints, CUDA libraries, or training runs were acquired
  as part of the CPU simulator setup. Other project environments were untouched.

## Next task

Build a training-data loader that constructs observation histories/transition
windows without crossing episode boundaries. Freeze combined dataset splits and
measure contact coverage before expanding collection or training a world model.

In parallel with source work, prepare the original T policy's resource-loading
and observation/action compatibility checks. Load only the required pinned asset
when ready, then compare T/I/L performance with matched initial states. Reproduce
that baseline before choosing visual route A or state route B and before training.

## Open questions

- Course deadline, rubric details, and named workstream owners.
- Training machine, GPU profile, shared artifact location, and compute budget.
- CUDA and pretrained-checkpoint runtime compatibility.
- Reset distribution coverage/dynamic goal reachability and mid-contact replay tolerance.

## Handoff convention

Replace stale next-task entries when work advances. Record task owner/branch,
changed behavior, exact commands, portable findings, and remaining issues.
Machine paths and detailed runtime reports stay in ignored artifacts. Keep failed
experiments visible; private chat history is not a shared project record.
