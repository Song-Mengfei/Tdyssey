# Shared project status

Updated: 2026-10-22. Stage: physical-simulator reranking now runs on native
Windows/CUDA. Real D contact replay, physical hold and actual frozen-checkpoint
candidate parity passed. Pretrained calibration, reliable learned dynamics and
policy learning remain separate unfinished work.

## Guarded joint correction follow-up (2026-10-22)

Added separate ratcheted position/angle limits, bounded intermediate motion,
position/angle/joint/actual-compensation refinement banks, searched press dwell
and up to four spare third-contact attempts within the old 40-attempt cap.
Fixed two execution issues found in retained runs: terminal scoring before its
physical holds were executed, and the old scalar stall rule interrupting guarded
macros. The final implementation executes and checks the scored precision prefix.

Final same-budget results remain 0/2: scene 100000 stops at control 287 with
71.77% coverage, 5.16 px position error and 16.04 degrees angle error; scene
100200 stops at 298 with 75.26%, 3.57 px and 14.87 degrees. Both exhaust eight
searches, without timeouts or physics errors. Relative to revision 3, centers
are closer but residual angles are larger. Joint task completion is not achieved.

The initial implementation and two correctness reruns retain all six failures
and paused videos: 763 new controls, all independently replayed with 960 imported
controls. Final checks pass 28 protocol, four real D runtime and nine old-search
regressions, zero skipped. Independent final contract checks confirm ten completed
precision macros, 92 guarded controls and no early interruption. Of 28 final
third-contact attempts, none is jointly feasible; none is selected. No student,
policy or world-model training, extra seeds, larger caps or asset changes occurred.
See [complete records](VALIDATION.md#guarded-joint-correction-follow-up-2026-10-22)
and `artifacts/d-sequence-20261022-r4-final/gallery.html` for all ten matched
historical/current cases, with the final two first. The teacher remains unqualified.

## Same-budget precision follow-up (2026-10-22)

Implemented adaptive settling reserve, separate fine approach/push speeds,
bounded terminal press dwell, nearby-contact continuation, opposite-face
compensation proposals and an access/remaining-controls ranking cost. Fixed the
revision before repeating only the two failed control-160 states with the same
seed, 300 total controls, 600 seconds, 80 forecast controls, 40 attempts and
eight global searches. No additional seeds or student training were run.

Neither recovery succeeded. Scene 100000 ended at control 300 with 70.04%
coverage, 8.63 px position error and 8.31 degrees angle error; the prior revision
had 81.98%, 0.93 px and 9.97 degrees. Scene 100200 stopped at control 298 after
eight searches, with 76.45%, 6.50 px and 7.86 degrees, versus prior 75.00%,
4.62 px and 12.56 degrees. Fine pushes established sustained actual contact and
reduced residual angles, but did not compensate translation sufficiently.
Do not call this a successful improvement or supply these as successful teachers.

All 278 new controls plus 320 imported controls passed independent actual-action
replay and paused-video checks. Twenty-one protocol, three real D runtime and
nine original-search tests passed, zero skipped. Reports, frozen source and
four-video matched comparison are in `artifacts/d-sequence-20261022-r3/`.
See [full evidence](VALIDATION.md#same-budget-precision-follow-up-2026-10-22).

## Two-contact D recovery pilot (2026-10-22)

Added a separate geometric contact-sequence search with joint material-point
pose error, diverse two-contact expansion, conditional elite memory and exact
history replay. Continuations reuse checked predictions rather than repeating
full search. Fixed corner release, short single-stage budgets and near-goal
stroke scaling after retaining the first three unsuccessful recovery runs.

The corrected, same-budget three-state pilot succeeded once: scene 100100,
resumed at control 160, reached 96.09% coverage at control 229 after 69 new
controls. Center/angle errors fell from 11.61 px / 64.95 degrees to 0.83 px /
1.39 degrees. Scenes 100000 and 100200 ended at 81.98% and 75.00% coverage
without success at control 300. All six runs and 649 new controls passed
independent replay and paused-video checks. Sixteen focused source tests,
two real D runtime checks and nine original-search regressions passed with
zero skips. No checkpoint or student was trained/loaded.

This verifies one rescue from a known development state, not reliable reset
completion, multiple successful routes or learned shape adaptation. All first-
revision failures remain recorded. See [implementation](CONTACT_SEQUENCE_SEARCH.md)
and [full results](VALIDATION.md#two-contact-search-recovery-pilot-2026-10-22).

## D teacher-search pilot (2026-10-21)

Student training remains deferred after evaluating a policy-free D simulator teacher.
Four predeclared initial scenes (100000/100100/100200/100300), each with search
seeds 42/43, use at most 300 actual controls and 600 wall seconds per run.
The original generic search completed all eight requested runs: zero native
successes, three control-budget failures and five wall-time truncations.
All 2,220 actual controls passed independent replay and video checks. Several
runs reached the goal center but retained large orientation errors; overlapping
outlines and different contact routes are not successful teacher demonstrations.

A separate experimental revision adds route-continuity preference, short
exterior contacts, signed rotation proposals and a rotation/translation composite.
Its 18 source tests, real contact continuation and complete 80-control composite
protocol check passed, but these are not task-success evidence. The same eight-run
comparison also finished with zero native successes, three control-budget failures
and five timeouts. V1/V2 peak coverage was 66.00%/63.61%; V2 final center errors
were only 0.18--4.96 pixels but orientation errors were 61.12--128.18 degrees.
All 16 runs and 4,456 actual controls passed independent replay/video audits;
all failure and timeout reports remain included. Different failed routes do not
establish multiple successful strategies. The generic teacher is not ready to
supply successful student demonstrations under these budgets.
Local protocols, source snapshots, reports and fixed-pause videos are under
`artifacts/d-teacher-20261021/`. See the teacher pilot in `docs/VALIDATION.md`.
No policy checkpoint, learned world model or student training is used.

## Windows runtime handoff (2026-10-21)

Applied the complete handoff patch onto baseline `71d0dfc` on
`codex/simulator-reranking-windows`, preserving the prior Windows branch.
The shared lock now supports Windows AMD64 with the existing Python 3.12.3,
Torch 2.5.1/cu124 profile. See environment compatibility (external historical reference: `ENVIRONMENT.md`).
Fixed UTF-8/source-snapshot handling, per-command Git ownership checks, final
forecast deadline accounting and explicit reporting of unexecuted requested shapes.
Checkpoint hash gates were removed; strict structure/configuration loading,
frozen evaluation and direct unchanged-tensor checks remain.

This machine passed the source checker, 13 planner tests, two real D physics
tests and all three actual CUDA candidate tests, with no focused-suite skips.
Broader source-only discovery passed 69 and skipped 50 runtime checks; those skips
are not passes. The T/D simulator smoke completed 20 real controls each on CUDA
in 95.203 seconds. D had contact on 13 controls; neither shape reached any goal
overlap or success in that short run. Full-history branches matched live state
exactly, and every selected first forecast matched the following real control.
See [commands, experiment results and costs](VALIDATION.md#windows-simulator-reranking-validation-2026-10-21).

The matched 64-control A/B/C comparison completed all T/D episodes without
timeouts. No episode succeeded. D max/final coverage was A (execute 8)
20.6%/12.1%, B (execute 1) 14.6%/13.3%, C (simulator K=4, execute 1)
18.9%/18.9%. C improved retained overlap versus B on this scene, but its D peak
was below A. Run times A/B/C were 34.359/198.125/322.719 seconds; C required
49,944 historical and 15,382 forecast controls beyond the 128 actual controls.
These are short integration probes, not task-success evidence. The earlier D
video's 89.83% peak occurs at control 201 (20.1 seconds), beyond this 64-control
budget; do not compare that peak with this first-6.4-second maximum as equal runs.

A separate full 300-control original-policy run then solved T at control 128
(95.33% coverage). D completed 300 controls without success, reaching 38.92%
maximum / 24.96% final coverage. The old D 89.83% near miss was **not reproduced**.
The new run's first 64 physical controls exactly reproduce A64, and old/new video
frames initially match, then diverge. The old action/state trace is unavailable
locally; numeric GPU/OS differences are plausible but not a demonstrated cause.
Exact historical trajectory reproduction remains unresolved. No training or seed
search was used to improve the result.

## Earlier source-only handoff (2026-10-20)

Branch `codex/simulator-reranking` adds batched frozen official-policy candidates
and an optional `evaluate_shapes.py --planner simulator` path. It uses the current
shape's known simulator, with no D demonstrations, world-model training or policy
updates. Branches replay actual actions from the common episode-initial snapshot
to reconstruct contact history and check full state before forecasting. Optional
prefixes plus physical hold are scored on terminal coverage, pose error and
motion; only a short prefix executes before observing and replanning. Native
success/episode budgets stay unchanged. The canonical original-T runner and
default eight-action shape baseline are preserved.

Twelve focused source tests cover hidden-history reconstruction, overshoot,
pre-overlap progress, real hold, budgets, success, deadlines and branch isolation.
Full source-only discovery passed 68 tests and skipped 49 installed-runtime
checks (117 discovered); repository checks, CLI help and whitespace checks passed.
Real D contact continuation and actual CUDA checkpoint parity tests are provided
but remain unexecuted here: this macOS source checkout has no project runtime,
upstream checkout or checkpoint. No new environment, assets or dependencies were
installed. Code/tests do not establish prediction accuracy or D improvement.
See implementation, scoring, commands and verification (external historical reference: `SIMULATOR_RERANKING.md`).

The runtime work requested after that handoff is recorded above and in
`docs/VALIDATION.md`; the 68/49 counts here describe only the earlier source host.
Compare baseline-eight, baseline-one and simulator-ranking-one on matched scenes
with explicit replay/inference costs.
Do not combine shorter-prefix and ranking gains. This is a privileged simulator
control for the existing research, not a replacement for the T-to-L learned-model
adaptation study or evidence that its dynamics gate has passed.

PR #1 merged the shared setup and first code milestone. PR #2 contains CPU/CUDA setup,
real simulator tests, shared documentation cleanup, bounded interaction collection,
episode-safe training batches, the official CNN T baseline, and a separate GPC
reference-policy runner together.
The official letter extension and matched frozen-policy probes continue in that PR.
Review branch: `build/minimal-runtime-setup`.
Main changes continue through pull requests.
Review milestone: baseline, data/training infrastructure and world-model diagnostics
are complete enough for coauthor review. Selected figures/reports are now in
`docs/RESULTS.md`. Finish review of PR #2 before starting the next training milestone
on a new branch after merge; the PR does not claim completed adaptation or policy learning.

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
- Added a fixed six-scene T palette comparison with shared physical branches,
  three inference-noise repeats, palette-independent image-mask proxies and
  compressed source frames. Auxiliary rendering preserves original policy images
  and state; it requires simulator access and is not a deployed image adapter.
- Published four unchanged world-model figures and three compact report summaries
  with source hashes in a 775,943-byte reviewed snapshot. The result index links
  these and all 27 alphabet MP4s without setup. Raw reports/data/checkpoints remain
  local/ignored, with no shared raw archive configured; see `docs/RESULTS.md`.

## Verification status

See `docs/VALIDATION.md` for commands and limits.

- CPU/GPC doctor: all direct versions matched; PyTorch tensor, torchvision operator,
  and Pymunk integration checks passed. 85 packages installed from the shared lock.
- T/I/L five-control-step interface checks passed with 3×96×96 images.
- Five real simulator tests passed: original/adapted T rollout agreement, actual
  I/L contacts, collision geometry/reward agreement, state restoration, and fresh
  pre-contact replay. This is not a trained-policy performance result.
- All 101 current tests passed together: 56 source-only and 45 opt-in runtime
  checks. Source-only discovery passed 56 and skipped 45.
  Twelve peripheral/repeated checks were removed; simulator and GPU checks remain.
  Pretrained integration adds only two focused timing/sampler-parity checks.
  The palette experiment adds one rendering-isolation/image-proxy check.
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
- Matched palette run: 228 controls/two resets, 864 forecasts, 34.51 seconds.
  Official contact centroid proxy error fell from 10.901 to 3.272 pixels; visible
  target recall rose from 0.033 to 0.954. Left-push rotation still failed, and the
  original-input prediction was better on that case. No weights/policy were updated.
  A failed first run retained separate counts (114 controls/two resets) before
  matching native float32 conversion. Complete evidence: `docs/VALIDATION.md`.
- Broader baseline evaluation, original demonstration conversion, reliable visual
  dynamics, policy optimization, reset coverage and mid-contact replay remain pending.
  Other project environments were untouched.

## Next task

Calibrate the pretrained GPC world model on bounded official-RGB T training data
with varied contact/rotation. This addresses appearance without requiring the
diagnostic's auxiliary renderer. Compare original/adapted weights and persistence
on disjoint validation scenes with fixed data/compute budgets. Keep the left-push
failure and target/no-contact metrics explicit; image averages alone are insufficient.
The palette result explains much of the gap but does not pass a full dynamics or
planning gate. Preserve official policy inputs and the final model-test split.
Freeze action-ranking/terminal scoring before comparisons, then proceed to L
adaptation after validating actual action effects. See decision D022.

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
