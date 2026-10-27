# Shared project status

Updated: 2026-10-27. Stage: forward-process RL stopped (noisy signal, D035); a
privileged world-model planning teacher solves most D/Y/L scenes in the simulator
(D039); a distilled image student blended with the base improves Y and D
significantly on 1000 fresh scenes but loses 4.8 T points, so the pre-registered final
test did not pass (D040) and the base remains the result. A secondary test of an
encoder-only blend nearly doubled Y (18.1% vs 9.4%) with T -2.5 [-6.2, +1.2] and
near-T -2.0 [-5.8, +1.8], narrowly missing the margin. Work continues on
`feat/wm-planning-teacher`.

PRs #1 and #2 are merged. The baseline, data/training infrastructure and
world-model diagnostics are on main; selected evidence is in `docs/RESULTS.md`.
New decision-inspection work uses `feat/policy-decision-replay`, based on merged
main. It adds no policy training or successful adaptation claim. Continue shared
changes through pull requests; do not rewrite or append to the closed PR branch.

Scope clarified on 2026-10-20: the goal is one shared policy across letter shapes,
using world models and RL with no new human adaptation demonstrations. T-to-L is
the first functional check, not the final scope. The initial expert-trained T
checkpoint and privileged simulator reward labels must remain disclosed. See
`docs/PROJECT.md` and D026. This is a plan update, not completed multi-shape training.

The [learning plan](LEARNING_PLAN.md) now defines the two concurrent workstreams,
world-model data/validation, proposed reward and DPPO losses, physical/imagined
training steps, bounded starting experiments and concrete three-developer tasks.
README gives a short human overview; agents should read the full plan before
learning work. Settings and proposed trainer files are not implemented by this
documentation. See D027. The later training implementation now exists on
`feat/direct-dppo-pilot`; read `docs/POLICY_LEARNING.md` and D028. It is a specified
DPPO-style variant, not an upstream reproduction. PR #4 remains the separate
inspection/plan review milestone.

## Completed

- Learning findings, 2026-10-24 to 2026-10-26 (H100 NVL, `feat/wm-planning-teacher`;
  full tables in [learning findings](LEARNING_FINDINGS.md), code in
  `scripts/learning/`). Base on the 1000-scene final test pool: T 44.4%, near-T
  41.9%, Y 9.4%, D 0.3%, L 0.1%. Long NFT/AdvantageFlow runs did not learn and
  degraded T with drift; only 16-36% of score variance comes from the candidate's
  own plan (D035). Oracle best-of-N gave small gains (Y 3/20 -> 8/20), and
  imitating it stayed within noise (D037). The structured state world model `wm2`
  reaches 0.22-0.32 px median 8-control contact error (D038). BPTT teachers through
  it failed; the sampling planner `mpc_teacher.py` solves D 50/50, Y 36/50, L 21/50
  validation scenes but only T 5/50 and near-T 12/50 (D039). Round-1 teacher data
  (350 scenes, 14.6k records with T retention) was distilled into three image
  students; all were worse than the base on every shape (validation T 35 ->
  15-25), so naive distillation of the closed-loop planner fails. Round 2 with chunk-consistent
  labels failed the same way. Training the encoder too, 60 epochs and DAgger gave the
  first image students with D/L successes but 10-15 points lower T. Blending a
  student with the base (`interpolate.py`) recovered most of T: the selected 70%
  blend on the final-test pool had T 39.6% vs 44.4%, near-T 41.0% vs 41.9%, Y 12.3%
  vs 9.4%, D 4.9% vs 0.3%, L 0.2% vs 0.1% (not passed: T lower bound -8.7 < -5).
  Every run is listed in the [experiment log](LEARNING_EXPERIMENT_LOG.md). New
  physics checks: `tests/test_learning_wm_runtime.py`.

- H200 NVL session (2026-10-24, commit 8cba20e, cu124, MPS): doctor, preflight,
  `tracking_smoke` and `nft_full_network_smoke` passed (dense peak 12.25 GB; its
  tiny smoke had tied scores, so no dense update). Eight completed sequential
  20k-control T/L pilots: NFT/AdvantageFlow with LoRA seeds 42-44 and dense seed
  42 (8 updates each; changed/frozen/reload-parity checks passed). L stayed 0/3 on
  every run; T moved between 1/3 and 3/3, within evaluation noise. Feedback was
  informative (L contact about 0.3-0.5, near-zero tied groups), unlike the earlier
  pilot; 8 updates is too few to judge adaptation. Runs and the verified base asset
  are kept in the author's private HF store; raw outputs remain outside Git.
- Throughput: plans dominate time (a dense pilot sampled about 900 batch-1 plans
  in 22 min; replay physics is minor). Eight concurrent smokes ran 1.4x faster in
  aggregate than one, 2.4x with MPS. Added opt-in `batched_rollouts` (D033):
  batch 32 cost 2.62 s versus 2.15 s for batch 1; the matched batched LoRA pilot
  took 7.4 min versus 29.6 min sequential under similar 6-job load; group size 16
  (8 updates, 77,603 controls) took 9.8 min. Prefix replays were 81% of that run's
  counted controls, so large groups mostly spend interaction budget on replay.
  Same-seed batched/sequential evaluations agree to about 0.02 px, then diverge
  chaotically; one near-threshold T success flipped. Use many more scenes for claims.
  Installed suite with official checks enabled: 126 ran, OK, 28 skipped (GPC not
  installed); `scripts/check_repo.py` passed.

- Added [training-machine handoff](TRAINING_MACHINE.md), optional W&B plus flushed
  local scalar logs and a no-account live viewer for all three trainers. Forward
  runs now record fixed baseline-plan/noise diagnostics, contact/tied-score
  feedback, periodic physical development evaluations and selected `best.pt`.
  Recovery preserves the best checkpoint and starts a new tracking run with
  cumulative axes. Detailed update traces are bounded; active replay prefixes remain.
- Added a no-install read-only GPU/asset/disk preflight, source-only multi-seed
  comparison planner/sequential launcher, bounded longer T/L presets and a locked
  cu128 candidate profile (torch 2.7.1 / torchvision 0.22.1) plus independent W&B
  extra. The current cu124 environment was unchanged. Blackwell and dense extended
  training remain untested; presets do not establish useful settings or permission.
- Custom visual dynamics now save rolling optimizer/RNG/window-cursor recovery,
  including partial epochs, and accept explicit longer budgets. A synthetic CUDA
  check kept model-test images unread, restored the exact cursor/RNG, and matched
  uninterrupted updates under deterministic GPU operations. Normal training is
  not promised bit-identical across GPU executions. GPC adaptation remains absent.
- A real forward monitoring check and fresh-process resume completed: cumulative
  352 training controls, 192 physical evaluation controls, four optimizer
  minibatches, about 1.11 GiB peak CUDA, frozen-weight/reload checks passed.
  Offline W&B and local fixed/per-shape events were recorded; live-viewer HTTP
  routes and JavaScript parsing passed. This is a mechanics check with a 32-control
  evaluation cap, not useful adaptation. Online W&B authentication/upload and
  browser appearance were not verified. See [validation evidence](VALIDATION.md).

- Implemented the [forward training path](FORWARD_POLICY_LEARNING.md): NFT and
  AdvantageFlow epsilon-prediction adaptations, full action-U-Net/head/distributed
  LoRA scopes, frozen rollout/T references, matched physical candidate trials,
  periodic atomic checkpoints and source/config/optimizer/RNG-checked resume.
  The distributed rank-8 actor has 1,652,824 trainable factors. The actor uses RGB
  and pusher history; geometric rewards/physical traces are privileged feedback.
- Added full-prefix warm-contact replay checks and all-step original-sampler
  parity. Analytic training noising preserves the scheduler's original CPU buffer;
  the library's mutating noising helper caused a prototype mismatch and is no
  longer used. Fresh-model checkpoint loading is checked, not only same-object
  reload. Small tied-score batches correctly report no weight change.
- Extended the trained-policy evaluator to forward checkpoints, every registered
  shape and optional bounded MP4 output. The official play collector now supports
  A-Z/near-T; custom visual-dynamics windows use each episode's geometry. A tiny
  Y path check collected 56 controls in seven episodes (272,344 bytes); a mixed
  L/Y catalog read real windows and fitted stats on 640 training controls only.
  This is data support, not learned shared-world-model accuracy or GPC adaptation.
- Completed the corrected distributed-LoRA NFT pilot: six updates, 88 optimizer
  minibatches, 3712 training controls including replay, 295.393 seconds and about
  1.11 GiB peak allocated CUDA. Factors changed and frozen weights/reload parity
  passed. On one 128-control development evaluation, L remained unsuccessful
  (peak coverage 0.410331 to 0.409805, final zero); four of five L training groups
  had zero contact and tied scores. No useful L adaptation is established.
- Corrected AdvantageFlow smoke and a fresh-directory resume passed. Neutral
  batches kept weights and before/after actions identical. The resumed lineage
  accounts for 96 training / 192 evaluation controls without double-counting its
  parent. Fresh-process L previews matched all 128 saved actions/outcomes, and a
  native 512x512 MP4 decoded all 129 frames. Raw outputs remain ignored/local.
  Full installed verification: 120 passed; source discovery: 64 passed / 56
  skipped. See `docs/VALIDATION.md` for commands, hashes and prototype exclusions.
- Surveyed newer image/flow RL objectives and closely related world-model policy
  work; added `docs/DIFFUSION_RL_RESEARCH.md` with primary sources, a DDPM-NFT
  specification, loss algebra, model-reliability limits and three-developer tasks.
  Recommend it as the next objective comparison alongside the existing PPO;
  that research milestone added no trainer; the later forward implementation is
  documented above. No imagined trainer or useful adaptation is established. Course
  scope remains shared multi-shape adaptation; algorithmic novelty is not required.
- Added a direct-simulator visual tail-PPO trainer, physical-time shaping/GAE,
  value estimator and compact checkpoint/reload path. Supports output-head or
  full action-network updates with frozen encoder/normalizers and identical
  original 100-step sampling before updates. Full-network runtime is pending.
- Recorded a six-method primary-source survey and derivations of latent-chain
  policy gradients, potential shaping, chunk returns and an assumption-dependent
  model-bias bound. Independent source and CUDA checks passed; see
  `docs/POLICY_LEARNING.md`. A bounded 64-control GPU run updated 1,313,282 head
  parameters, preserved all frozen parameters and reloaded identical actions.
  It produced zero training reward and establishes no adaptation benefit.
- Completed a short head-only pilot: 16 update rounds / 64 optimizer minibatches,
  1,008 physical training controls (T 500 / L 508). On one paired development
  scene, T remained successful but slowed from 270 to 297 controls; L remained
  unsuccessful with essentially unchanged peak coverage (0.410331 to 0.409394).
  Frozen weights and checkpoint action parity passed. This establishes no L
  adaptation benefit. Elapsed 431.609 seconds, peak allocated CUDA about 1.11 GiB,
  with about 16.2 MiB of ignored output; commands/hashes in `docs/VALIDATION.md`.
- Added native Conv1d LoRA with head or distributed action-U-Net targets and
  separate trained-checkpoint desktop evaluation. A head-LoRA 64-control update
  trained 25,604 factors, preserved base weights and reloaded exact actions.
  Distributed LoRA passed original-sampler parity, real GPU updates and compact
  reload checks; a physical learning improvement is not established. Visible
  L/T previews matched all 300/297 saved evaluation actions/rewards/overlaps.
  See D029 and `docs/POLICY_LEARNING.md`; training itself remains headless.
  Latest checks: all 111 installed tests passed in 42.356 seconds; source-only
  discovery passed 62 and skipped 49 runtime checks. No new dependency or asset.

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
- Added read-only capture of actual official DDPM samples, an offline interactive
  viewer, overview PNGs and compact MP4s. Replayed T/D/near-T decisions exactly,
  including the action chunks containing their best overlap. A posthoc physical
  hold comparison preserves more overlap on D and near-T but solves neither.
  Commands, interpretation and limits: `docs/POLICY_DIAGNOSTICS.md`.
- Added explicit fixed-duration T continuation, preserving normal stop-on-success
  defaults. A 500-control run matches the original successful 128-control prefix
  exactly, then loses success at 129 and ends at zero coverage/IoU. The policy
  and simulator were not reset or frozen at success; see `docs/VALIDATION.md`.

## Verification status

See `docs/VALIDATION.md` for commands and limits.

- CPU/GPC doctor: all direct versions matched; PyTorch tensor, torchvision operator,
  and Pymunk integration checks passed. 85 packages installed from the shared lock.
- T/I/L five-control-step interface checks passed with 3×96×96 images.
- Five real simulator tests passed: original/adapted T rollout agreement, actual
  I/L contacts, collision geometry/reward agreement, state restoration, and fresh
  pre-contact replay. This is not a trained-policy performance result.
- The previous full suite passed all 101 tests together: 56 source-only and 45 opt-in runtime
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
- Decision replay: 616 original controls and all 77 CUDA plans matched saved
  actions/states/contacts/rewards. Captured eight actual 100-step denoising chains.
  Hold checks separately used 463 prefix-replay controls and 16 intervention
  controls; five reset substeps and five initial-state restores in total.
  D's best IoU occurs at action 1 of controls 201-208 and declines at 202; near-T
  peaks at action 6 of controls 257-264 and declines at 263. Both remain below
  0.95 coverage. These are development-case timing/intervention results.
- Decision-replay source discovery: 58 passed / 45 runtime checks skipped. Three focused
  official runtime tests passed, including current/next-plan RNG and trace parity.
  All three exported MP4s decoded (179 frames each); figures were visually checked.
  Viewer JavaScript parsed and control handlers passed a DOM/canvas-stub smoke
  check. Actual browser rendering was not checked because no browser surface was
  available. No dependencies, downloads, training or checkpoint changes were needed.
- T continuation: 500 controls / 5000 physics substeps, one reset, 63 CUDA plans /
  6300 denoising steps; 501 video frames decoded. The first success occurs at 128,
  drops below threshold at 129, briefly recurs at 182, and is absent at the end.
  Coverage first reaches zero after success at control 363. This is one scene and
  one policy seed, not a population stability rate or a replacement benchmark.
  Retention accounting and continued raw-simulator stepping passed focused checks.
  Latest source discovery: 59 passed / 45 runtime checks skipped (104 total).
- PR preparation on 2026-10-22 reran all 104 tests successfully with the existing
  CUDA/DP/GPC environment and pretrained assets enabled. Source-only discovery
  passed 59 and skipped 45. This includes real inference/parity, simulator and
  existing world-model update checks, not a diffusion-policy RL update.
- Training milestone checks on 2026-10-23: source discovery passed 62 / skipped
  47 (109 total); installed CUDA/DP/GPC discovery passed all 109 in 30.882 seconds.
  This includes independent reward/return checks and real denoising likelihood,
  action/RNG parity, actor update, frozen-weight integrity and delta reload checks.

## Next task

Keep T while learning the new letters (D040 failed on T alone). Candidates: more T
retention data from the base's own plans (`mpc_dagger.py --label policy`), blending
the action network and encoder separately, the large students now training, or a
disclosed shape gate that routes T to the base. Any further final-test candidate is a
secondary result under D040. Reduce the planner's leftover T rotation.

The earlier next-task notes below predate these findings.

Run long forward comparisons with `batched_rollouts` on the batched branch, keep
sequential and batched results apart, and evaluate on more than three scenes per
shape before comparing methods. Decide whether prefix-replay controls should
dominate the interaction budget at larger group sizes.

Use the [machine handoff](TRAINING_MACHINE.md) to recreate the selected profile
and pass the short preflight/smoke run on the training GPU. Then review live
per-shape feedback on the bounded pilot before launching repeated longer jobs.
Fix contact-free/tied trials if they persist; extra GPU time alone is insufficient.

Measure the implemented NFT/AdvantageFlow methods against PPO using
[the runnable guide](FORWARD_POLICY_LEARNING.md). Diagnose contact coverage,
finite-horizon score signal and physical learning curves before scaling; retain
the original success criterion. Validate dense action-network mode on its intended
GPU. World-model scoring still follows contact/reward/ranking checks.
The first forward pilot identifies mostly contact-free/tied L groups. Improve
automated context/contact coverage and assess score horizons before simply running
more copies of the same uninformative interaction. Disclose any play-based context
selection or geometric curriculum and retain full-reset physical evaluation.

Measure physical learning on separate development scenes, diagnose reward/contact
coverage, and validate full action-network updates on an authorized larger GPU.
The first tail-PPO sampler/likelihood/update/serialization checks now pass;
an actual learning improvement has not been established. Tail-PPO still needs its
own periodic recovery before long experiments; the forward trainer has it.
A standalone trained-delta desktop evaluator and bounded MP4 export now exist.
In parallel, define the frozen/adapted predictor comparison and
collect bounded T/L data; do not wait for a reliable A-Z model to start policy work.
Follow [the detailed handoff](LEARNING_PLAN.md) for proposed file ownership,
interfaces, reward/return rules and starting budgets.
Use the extended official collector on a fixed, diverse multi-shape development set,
using machine-generated play/current-policy interaction and per-shape episode
splits. Existing substantial data remains L-only; the tiny Y check is not an
alphabet training set or enough data for shared dynamics.
Assess a shared pretrained predictor on those domains; adapt it when measured
errors justify doing so, and require physical action/reward checks before imagined
RL. Compare partial updates/adapters with full updates under matched budgets,
then expand shared-model/policy training to more letters while checking T retention.

Diagnostic follow-up: predeclare a small comparison of eight-action execution
against more frequent replanning on the same development scenes. Repeat policy
noise seeds and report extra inference work; changing prefix length also changes
RNG consumption. Do not turn the posthoc best-state hold into an official stopping
rule or treat it as completed adaptation. This diagnostic does not expand the
first learning milestone beyond T-to-L.
Include the successful-T continuation as a control when investigating goal loss;
its first loss follows a new plan at 129, whereas D/near-T first decline within
an existing chunk. Keep these different timings explicit.

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
the released sampler rounds pixels and disables gradients. That blocks direct
backpropagation through its predictions, not likelihood-ratio policy gradients
such as DPPO, or reward-based forward regression. See decision D021 and
`docs/POLICY_DIAGNOSTICS.md` for the separate policy-sampling inspection. The direct
PPO trainer now exists; a world-model policy-learning loop remains pending.

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
