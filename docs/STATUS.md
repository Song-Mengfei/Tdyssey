# Shared project status

Updated: 2026-10-19. Stage: official CNN T baseline verified; no model training completed.

PR #1 merged the shared setup and first code milestone. PR #2 contains CPU/CUDA setup,
real simulator tests, shared documentation cleanup, bounded interaction collection,
episode-safe training batches, the official CNN T baseline, and a separate GPC
reference-policy runner together.
Review branch: `build/minimal-runtime-setup`.
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
  0.13.0 without a persistent download cache. That profile measured about 1.33 GiB.
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
- Added frozen combined catalogs with source hashes, duplicate-episode and initial-scene
  leakage checks, matching control rates, and preservation of existing splits.
- Implemented on-demand visual history/future windows with one decoded episode
  cached, plus real CPU PyTorch batches. No image windows are duplicated on disk.
  Commands and timing/label assumptions are in `docs/TRAINING_DATA.md`.
- Verified GPC-published policy/data resource schemas and pinned immutable revisions.
  Added compact EMA checkpoint loading, a cache-free selected-file downloader,
  and bounded matched-scene T/I/L policy evaluation. Public assets need no account.
- Added action/state-checked video replay without loading the model again.
  Commands, normalization/horizons, and evaluator differences are in
  `docs/POLICY_BASELINE.md`.
- Validated the locked CUDA/GPC profile in the same environment, replacing the
  CPU build. CUDA policy execution is now the default with no automatic fallback.
  Added a resizable 768×768 desktop viewer, independent of policy image size.
- Added the official RSS 2023 visual CNN source/checkpoint as the primary baseline,
  with its original T environment, saved normalizers, crop, and legacy reset setup.
  The older GPC runner/results remain separate; see `docs/OFFICIAL_BASELINE.md`.
- Added a locked self-contained `dp` extra, compact EMA asset preparation, bounded
  official evaluation, optional video, and official-wrapper/inference parity tests.

## Verification status

See `docs/VALIDATION.md` for commands and limits.

- CPU/GPC doctor: all direct versions matched; PyTorch tensor, torchvision operator,
  and Pymunk integration checks passed. 85 packages installed from the shared lock.
- T/I/L five-control-step interface checks passed with 3×96×96 images.
- Five real simulator tests passed: original/adapted T rollout agreement, actual
  I/L contacts, collision geometry/reward agreement, state restoration, and fresh
  pre-contact replay. This is not a trained-policy performance result.
- All 76 tests passed together in the installed environment: 51 source-only and
  25 opt-in runtime tests. Source-only discovery passed and skipped runtime tests.
- Small collector examples passed for T/I/L: 140 control steps across four episodes,
  four reset substeps, 15 L-shape contact steps, and 590,024 bytes of saved output.
  These retained example counts exclude earlier prototype and diagnostic/test work.
  Raw data is ignored; portable findings are in `docs/VALIDATION.md`.
- Existing example data produced 124 training windows with history 2/future 4;
  two batches of four examples passed CPU shape/dtype/finite checks. The catalog
  uses 1,766 bytes and its report 1,587 bytes; no new interaction data was needed.
  Validation/test splits remain empty in these tiny examples, so no model-quality
  evaluation or learned improvement is claimed.
- GPC CPU policy loading passed for 208 EMA tensors / 91,123,778 parameters. One
  verified 364,554,150-byte checkpoint was downloaded; no duplicate component
  weights or original training arrays were acquired.
- A GPC matched-scene run at seed 100000 executed 300 controls per shape: 900
  controls, three reset substeps, 114 policy calls, 11,400 denoising steps, about
  140 seconds on CPU. Maximum IoU T/I/L: 0.510/0.226/0.229; none succeeded.
  This validates execution, not baseline quality or an adaptation benefit.
- A 301-frame T video decoded correctly; its 300 diagnostic replay steps matched
  recorded poses/velocities within 1e-7 and rewards within 1e-9. Replay accounting
  is separate from evaluation, and does not establish mid-contact restoration.
- CUDA 12.4 package/operator checks, calculations, gradients, and actual policy
  inference passed on an Ada GPU. The CUDA/GPC environment measured about 5.57
  GiB / 98 packages, replacing CPU PyTorch without retaining a download cache.
- The GPC visible CUDA comparison completed 300 controls per shape / 900 total;
  maximum IoU T/I/L was 0.342/0.417/0.311, with no success on this scene. GPU and
  CPU random streams differ. The enlarged desktop window was verified at 768×768.
- The official CNN policy solved all three development T scenes (100000–100002)
  in 128/157/273 controls. Total: 558 controls, three reset substeps, 71 policy calls,
  7,100 denoising steps. Maximum IoU: 0.9241/0.9082/0.9236. These checks establish
  a working baseline, not reproduction of the full paper evaluation.
- Official loading/parity checks passed with the unchanged source class. All 300
  EMA state entries, including normalizers, were retained in a 1.02 GiB asset;
  the redundant 4.0 GiB training checkpoint was removed after verification.
- The combined CUDA/DP/GPC environment has 113 packages and measured 5.72 GiB.
  Torch/CUDA pins did not change. No demonstration dataset or second environment
  was acquired for official inference.
- Broader baseline evaluation, original dataset conversion, policy/world-model
  training, reset-safety coverage, and replay restored during contact remain pending.
- Original dataset schema inspection did not acquire image/action chunks.
  Training runs remain absent. Other project environments were untouched.

## Next task

Port L to the official PushT setup while preserving original T behavior and
recording the L coordinate/mass/inertia/reset decisions. Check matched T rollouts,
collision/render/goal consistency, and safe L starts before comparing shapes.
Do not load official weights through the GPC runner or silently use GPC play data
as matched official-environment training data.

Then evaluate the frozen official T policy on L and collect meaningful target-shape
dataset with nonempty frozen validation/test splits and contact diagnostics.
Use the visual CNN route initially; train/validate the first world model and verify
direct simulator policy optimization before imagined optimization. Additional GPU hosts still
need their own authorized compatibility/setup checks.

## Open questions

- Course deadline, rubric details, and named workstream owners.
- Training machine, GPU profile, shared artifact location, and compute budget.
- Compatibility on additional GPU generations and reproduced pretrained-policy performance.
- Reset distribution coverage/dynamic goal reachability and mid-contact replay tolerance.

## Handoff convention

Replace stale next-task entries when work advances. Record task owner/branch,
changed behavior, exact commands, portable findings, and remaining issues.
Machine paths and detailed runtime reports stay in ignored artifacts. Keep failed
experiments visible; private chat history is not a shared project record.
