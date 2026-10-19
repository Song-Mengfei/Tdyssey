# Shared project status

Updated: 2026-10-19. Stage: pretrained GPC dynamics checked; official-domain adaptation, dynamics reliability and policy learning remain pending.

PR #1 merged the shared setup and first code milestone. PR #2 contains CPU/CUDA setup,
real simulator tests, shared documentation cleanup, bounded interaction collection,
episode-safe training batches, the official CNN T baseline, and a separate GPC
reference-policy runner together.
The official letter extension and matched frozen-policy probes continue in that PR.
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
- Added official capital L: height 120, shorter 90-pixel bottom foot, 30-pixel
  stroke, one rigid mass-1 body. Original T is delegated unchanged. Added matched
  physical-center resets and full-state capture/restore; see `docs/OFFICIAL_SHAPES.md`.

- Added all remaining 24 uppercase block letters and a near-T control with a 10%
  shorter cap, preserving original T and the specified capital L. All letters are
  edge-connected rigid bodies; openings are real empty geometry. Added a bounded
  CUDA sweep, exact SVG catalog, and report-driven HTML/video gallery and CSV.
  Alphabet exploration is separate from the core T-to-L learning study.

- Published the 27 curated alphabet MP4 files directly in Git (about 11.5 MiB),
  with a linked index and exact video hashes: `docs/ALPHABET_RESULTS.md`.
  Complete raw reports, generated galleries and future videos remain outside Git.

- Ported bounded non-expert play to official T/L, with full reset provenance,
  known evaluation-scene exclusions, correct original done/horizon handling and
  mandatory nonempty whole-episode splits. Catalogs refuse official/GPC mixtures.
  Commands and semantics: `docs/OFFICIAL_COLLECTION.md`.

- Implemented a small visual/action-conditioned CNN world model, four-step
  autoregressive training, train-only normalization, contact/noncontact physical
  metrics, persistence baselines, compact checkpoints and physical candidate probes.
  This visual model uses privileged labels, not physical-state inputs. Its first
  training experiment and limitations are in `docs/WORLD_MODEL.md`.
  It was trained from scratch on L; no T world-model pretraining or transfer
  comparison has been performed. The frozen T policy has separate pretrained weights.
- Added verified loading of the released GPC phase-two world model (8.73M
  parameters, 33.42 MiB) with unchanged upstream sampling, isolated legacy imports,
  four-frame/action alignment and a bounded native/official T prediction probe.
  No new environment, dependency or dataset was needed. Compatibility findings
  and reproducible commands are in `docs/WORLD_MODEL.md`.

## Verification status

See `docs/VALIDATION.md` for commands and limits.

- CPU/GPC doctor: all direct versions matched; PyTorch tensor, torchvision operator,
  and Pymunk integration checks passed. 85 packages installed from the shared lock.
- T/I/L five-control-step interface checks passed with 3×96×96 images.
- Five real simulator tests passed: original/adapted T rollout agreement, actual
  I/L contacts, collision geometry/reward agreement, state restoration, and fresh
  pre-contact replay. This is not a trained-policy performance result.
- All 100 current tests passed together: 56 source-only and 44 opt-in runtime
  checks. Source-only discovery passed 56 and skipped 44.
  Twelve peripheral/repeated checks were removed; simulator and GPU checks remain.
  Pretrained integration adds only two focused timing/sampler-parity checks.
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
- Six real shape tests passed: unchanged T images/state/contact and roundoff-close
  rewards, L collision/goal/render agreement, real pushing, full-state restore and
  pre-contact replay, matched reset centers, and rejected-reset accounting.
- First matched scene 100000: T succeeded in 128 controls; L failed at the 300-step
  budget. Max IoU T/L: 0.9241/0.3015; L contacted the pusher in 288 steps and ended
  at zero overlap. T actions/body-state traces and video matched its earlier
  canonical run exactly. Total: 428 controls, two reset substeps, 54 policy calls.
  One pair establishes an executable probe, not a population performance estimate.
- Completed a common-scene 27-shape CUDA sweep: T/Y succeeded in 128/235 controls;
  other capitals and near-T failed at 300. Near-T best/final IoU: 0.8717/0.7751.
  Total 7863 controls, 27 reset substeps, 996 policy calls, 1553.45 seconds.
  All 7890 video frames decoded; T/L traces, rewards and videos exactly reproduced
  the earlier paired run. Local gallery/CSV/27 best frames total about 28 MiB.
- Added near-miss diagnostics after observing lost progress: D best/final IoU
  0.8154/0.4172; K 0.7512/0.0620. Record peak step/coverage and final loss without
  changing original success. One scene/checkpoint/font is not a generalization
  limit or reliable success-rate estimate. Full results/hashes: `docs/VALIDATION.md`.
- First official L play pilot: 8 episodes, 800 controls, 12 simulated resets,
  354 contact steps and 3,279,264 bytes. Frozen train/validation/model-test: 6/1/1
  episodes and 576/96/96 windows (history 2/future 4). Fresh read-back and two
  batches of four examples passed in all three splits. Raw data stays local/ignored.
  This is data-path evidence, not model training or an adaptation benefit.
- First visual L model: 10 epochs / 720 CUDA updates / 5760 windows / 23040
  forecast steps. Epoch 6 was selected on validation contact position error.
  At horizon 4, contact error was 12.017 vs 15.250 pixels for persistence, but
  all-step error was worse (7.307 vs 6.747) and noncontact drift grew (3.943 vs
  0.673). The real model-test split was not loaded. No policy learning occurred.
- Fresh checkpoint validation reproduced exactly. Training used 14.83 seconds,
  peak allocated GPU memory about 205.5 MiB, and about 5.3 MiB of saved output.
- Candidate diagnostic: two validation starts / eight matched fresh-space branches,
  32 controls and one initialization reset. Reward predictions barely changed
  across actions. Primary max scores tied and were uninformative; posthoc final/
  sum views exposed bad choices. No useful planning or adaptation is established.
- Pretrained GPC loader: all 298 state entries loaded, CUDA wrapper predictions
  matched the released sampler exactly, and imports/random-state isolation passed.
  Native/official T diagnostic: 76 controls, two resets, 64 forecast steps, 6.39
  seconds and about 105.8 MiB peak allocated GPU memory. Native bottom pushing
  improved image prediction over persistence, but native side rotation was missed.
  Official inputs were recolored red and lost the green target; no reliable
  official-world-model transfer or policy improvement is established.
- Broader baseline evaluation, original demonstration conversion, reliable visual
  dynamics, policy optimization, reset coverage and mid-contact replay remain pending.
  Other project environments were untouched.

## Next task

Resolve the pretrained GPC appearance/domain mismatch on T before L adaptation.
Freeze a matched validation protocol that separates appearance conversion from
physics adaptation; preserve the official policy's observation stream. Check
target retention, no-contact stability, contact/rotation and action effects, with
both persistence and the unchanged pretrained model as comparisons. The native
side-push failure needs investigation alongside the official red/gray mismatch.
Use explicit training-only data if adaptation is needed; leave the real model-test
split untouched. Freeze action-ranking/terminal scoring before comparisons.

DINO-WM remains an alternative if GPC cannot pass physical prediction checks.
Retain the custom CNN as an L-only comparison. Loading a public checkpoint does
not establish adaptation, reliable planning or differentiable policy training;
the released sampler rounds pixels and disables gradients. See decision D021.

Verify direct physical-simulator policy optimization before imagined updates.
Repeat a small fixed T/L/Y/near-T evaluation set across additional scenes before
generalization claims. Additional GPU hosts need their own setup authorization
and compatibility checks.

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
