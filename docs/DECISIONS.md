# Decisions

## D001 — Course research scope (2026-10-13)

The user clarified that grading emphasizes effort and ambitious goals may fail.
Keep cross-shape world-model adaptation as the research objective. Assess progress
through implementation, controlled experiments, and failure analysis. Focus on H1
before expanding to H2/H3; do not quietly redefine success as a trivial demo.

## D002 — Separate source work from machine setup (2026-10-13)

Initial development used source-only checks. Keep that workflow available to all
authors, alongside optional installed development. Setup and training require
authorization for the selected machine. Minimize storage and remove temporary
tooling/cache. See D008 for the first installed development profile.

## D003 — Shared candidate environment (2026-10-13)

Use uv 0.13.0, a committed lockfile, Python 3.12.3, and a Linux x86_64 reference
platform with mutually exclusive CPU/CUDA profiles. Initial candidates use
PyTorch 2.5.1 / torchvision 0.20.1, NumPy 1.26.4, Zarr 2, and Pymunk 6 to target
older upstream APIs. The CUDA profile uses the officially published cu124 wheels.
These versions are a conservative starting point, not an experimentally verified
combination. Revisit them for Blackwell hardware or concrete compatibility failures.
Do not export or modify unrelated project environments to define this project.
Gym's static dependency metadata is parsed from its checksum-verified published
source archive so locking does not require a temporary build environment.

## D004 — Repository-based Codex context (2026-10-13)

Root `AGENTS.md` supplies shared instructions; status, decisions, and onboarding
provide durable handoffs. Minimal `.codex/config.toml` supplies project preferences
where supported/trusted. Account credentials, model choice, and per-machine trust
stay outside the repo. No automatic setup hooks or agent orchestration are added.

## D005 — Three authors and shared review (2026-10-13)

The project has three coauthors. Propose one person per main workstream,
with names and ownership agreed by the team. Shared setup and later changes go
through branches and pull requests before becoming defaults on `main`. Use plain
English in team chat and explain unfamiliar technical terms only when needed.

## D006 — Structured Git messages (2026-10-13)

The user requested consistent commit and PR messages instead of improvised formats.
Adopt Conventional Commit titles and shared commit/PR templates. Add a dependency-free
message checker and GitHub PR-format check. Document actual changes, reasons,
validation, and limitations. Existing published commits are preserved; use a
formatted squash-merge title when the team later authorizes merging.

## D007 — Reuse PushT for the first code milestone (2026-10-13)

The user emphasized that PushT and diffusion-policy implementations already exist.
Reuse GPC's pinned simulator/controller/image interface, preserve the T construction
path, and add I/L through its existing object hook. Do not build a replacement
physics engine. Implement and test geometry, metrics, and state/data contracts with
the standard library before installing the runtime. Physical simulation and
policy reproduction require separate runtime verification. The GPC optional
dependency set includes IPython because the upstream environment imports it.

## D008 — Storage-saving runtime and shared documentation (2026-10-19)

Use the existing locked CPU/GPC profile for initial real simulator validation.
Avoid installing CUDA libraries until GPU work is needed. Install without a
persistent package cache and fetch only pinned GPC simulator/domain source files.
Keep environments, source checkouts, local tools, and machine-specific reports
ignored. Write shared instructions for all three authors; machine-specific setup
consent, paths, and limitations belong in local records. CPU validation does not
establish CUDA, checkpoint, or training compatibility.

## D009 — Bounded play data protocol (2026-10-19)

Collect small, explicitly bounded non-expert runs before building training loops.
Store one uint8 image per observation and T+1 observations per T actions, plus
physical labels, source hashes, reset/action seeds, and contact diagnostics.
Use fixed-goal, geometry-checked resets and count rejected reset substeps as well
as control actions. Default random waypoint exploration has no object-state action
prior; optional object-directed exploration uses simulator position and must be
reported as privileged exploration. Preserve upstream flags separately from the
collector's horizon cutoff. Freeze whole-episode splits and verify files on read-back;
small verification data is not a training or policy-performance result.

## D010 — Frozen catalogs and episode-safe training windows (2026-10-19)

Preserve source episode splits when combining runs, pin source manifest hashes,
and reject duplicate trajectory IDs or exact initial scenes crossing splits.
Do not silently create validation data from a tiny all-training collection.
Emit only complete history/future windows within one episode, with actions at a
predicting observations at a+1. Keep one decoded episode cached and construct
windows on demand rather than duplicating images on disk. Images become float32
in [0,1]; position/action units remain pixels. Privileged labels stay separate
from visual model observations. Pretrained-policy transforms/normalizers and
actual model training remain separate integration tasks.

## D011 — Compact pretrained policy and explicit baseline protocol (2026-10-19)

Use only the hash-verified averaged policy checkpoint for original visual policy
inference: it covers the encoder and action network with no remaining buffers.
Avoid duplicate component weights and full training-array downloads for this task.
Pin source/model/data revisions and inspect original array schema separately.
Public resources do not require a Hugging Face account; never share tokens in Git.

Evaluate one unranked policy sample, using the published domain18 normalization,
two-observation history, 16 predicted actions, and configured eight executed actions
at indices 1:9. Upstream `eval_baseline.py` instead uses world-model ranking,
candidate spreading, and different slicing; do not claim exact evaluator/paper
reproduction. Retain upstream randomized goals for this initial check, unlike
fixed-goal play collection. Transfer full T initial states across shapes, reject
unsafe scenes together, count reset/control/denoising work separately, and retain
original terminal flags. Video replay is separately counted diagnostic interaction.
The first one-scene T/I/L run is execution evidence; failed T performance requires
behavior inspection and repeated seeds before interpreting adaptation results.

## D012 — CUDA policy execution and an optional desktop view (2026-10-19)

Use CUDA by default for policy inference; fail clearly if it is unavailable rather
than silently executing on CPU. Keep explicit CPU debugging available to authors
without a GPU, and keep simulator physics on CPU. Switch the existing environment
between locked profiles instead of keeping duplicate environments. Install without
a persistent download cache. GPU setup authorization remains machine-specific.

Separate desktop visibility from compute device: `--display` opens a resizable
768-pixel pygame window, enlarging only the display of the 96-pixel policy image.
Escape/window close stops the run. Display pacing and final-frame hold count in
elapsed time; synchronized planning times measure GPU inference separately.
Record selected device, CUDA runtime, and GPU identity in local reports. CPU/CUDA
random streams can differ for the same integer seed. Basic CUDA gradients and
inference establish runtime support, not completed project training.

## D013 — Official Diffusion Policy baseline and GPC's separate role (2026-10-19)

Use the official RSS 2023 visual CNN checkpoint in its unchanged original PushT
environment as the primary baseline. The earlier schedule preferred GPC reuse;
that is a reference route, not evidence that GPC weights reproduce the original
policy. Retain its code/results as explicitly separate comparisons. Start with
one T-to-L adaptation question; Transformer and more shapes are later options.

Preserve the official checkpoint's crop, encoder, saved normalizers, horizons,
fixed goal, legacy reset convention, geometry/mass/inertia, reward, and scoring.
Do not silently insert GPC resets, coordinates, weights, or datasets. First port
and test L in the official setup. Reuse appropriate GPC world-model components
after compatibility checks; our contribution is non-expert shape adaptation and
physically evaluated policy improvement, not reproducing unchanged GPC.

Keep one CUDA environment and add a self-contained locked `dp` extra. Retain only
the official EMA state/config for inference after verifying the original training
checkpoint and compact output. Source/checkpoint/config hashes and actual protocol
are recorded separately. Three successful development scenes establish a working
baseline, not reproduction of the full published evaluation or adaptation success.

## D014 — Upright capital L and matching by physical centers (2026-10-19)

Use an upright capital L with height 120 pixels, total bottom-foot width 90 pixels,
and stroke thickness 30 pixels. Its connected nonoverlapping rectangles form one
rigid body. Center local vertices at the area centroid. Keep total mass 1, matching
official T, and use uniform-density inertia 1875 for L. L area is 5400 versus T's
6300 pixel²; density/inertia are not matched. Preserve original T mass, inertia
3000, local coordinates, and legacy center-of-gravity convention unchanged.

Match T/L mass-center poses, velocities/forces, pusher state, and target mass
centers rather than copying different local origins. This shifts L's goal origin
relative to T's, while aligning their physical target centers. Use common rotated
geometry/clearance rejection, count both reset substeps per candidate, and restore
accepted full states into fresh spaces without extra stepping. Keep this paired
development protocol separate from the canonical official benchmark reset path.

Check source T image/state/contact equivalence, scalar rewards within roundoff,
actual L contacts, collision/render/goal agreement, incompatible-state rejection,
and pre-contact replay. Cached contact impulses and future-reset RNG are not saved.
Retain older GPC L definitions/data as historical references rather than changing
their hashes. A frozen-policy failure on one L scene is a development observation,
not a generalization rate or an adaptation result.

## D015 — Frozen-policy alphabet exploration and a near-T control (2026-10-19)

Add the remaining 24 uppercase block letters and one near-T geometry as a separate
exploratory frozen-policy sweep. Keep T-to-L learning as the core research task;
these are development evaluation shapes, not evidence of learned adaptation or a
frozen held-out shape set. Full alphabet exposure cannot later be called unseen
model-selection evidence. Training/test shape choices need a later explicit freeze.

Preserve official T and the specified shorter-foot L. Define the other letters as
versioned edge-connected 5×7 blocks of total size 100×120, with holes, one rigid
mass-1 body and area-centroid uniform-density inertia. Record exact rectangles,
symmetry and hashes, not just a font name. Density/inertia/contact geometry vary;
the sweep measures the combined task shift, not isolated visual generalization.

For `T_narrow`, change only cap width 120→108 and retain original mass 1, inertia
3000 and physical-center convention. Match all shapes by physical centers, pusher
state, orientation and target centers. Common reset rejection applies to the full
selected set, including every rejected reset in interaction accounting. Preserve
identical per-scene policy noise seeds, budgets and original success rule.

Allow 1–3 common scenes, at most 300 controls per shape and a bounded wall time.
Keep CUDA-only inference and stream small videos. Generate HTML/CSV from saved
reports with the standard library; do not download fonts, new weights or datasets.
A single scene per shape is an exploratory observation, not a success-rate estimate
or proof of the policy's overall limit. Retain honest failures and partial-run flags.

Near-miss analysis records the earliest best-IoU control, target coverage there,
final coverage and best-to-final IoU loss. Decode peak frames from saved videos;
run no additional inference/physics. These diagnostics were added after observing
lost alignment and do not replace the original success threshold. Post-run best
frame selection is not an implementable learned stopping policy. Any future
shorter-prefix/stopping experiment needs a separate matched comparison and explicit
disclosure of privileged geometric reward or state used to decide when to stop.

## D016 — Commit one curated alphabet-video snapshot (2026-10-19)

Publish the 27 existing alphabet/near-T MP4 files directly under
`docs/media/official-alphabet-2026-10-19/`, with a linked result table and exact
size/frame/hash manifest. This bounded 11.5 MiB snapshot makes the actual failures
and successes reviewable with the source. Preserve original video bytes. Keep the
ignore exception scoped to this directory; future raw videos, datasets, checkpoints
and complete reports remain outside Git. No release or external host is required.

## D017 — Official non-expert data and environment-family separation (2026-10-19)

Port bounded waypoint play to the official T/L four-value step API, keeping the
fixed official target and physical-center conventions. Record success as terminal
and only the collector horizon as timeout truncation. Preserve T+1 uint8 frames,
full physical labels and original goal overlays. Directed play discloses its
75%-probability privileged object-position prior; it does not optimize reward or
use an expert/checkpoint policy. Physics-only collection needs no policy weights.

Version reset/action seeds separately. Exclude known development seeds
100000–100002 and support additional exclusions. Save all candidate records with
full post-reset/accepted states. Count each simulated accepted/rejected reset;
record excluded unsimulated candidates separately. Restore accepted fresh states
without extra steps. Future evaluation scenes need their own explicit exclusions.

Keep existing pilot resource caps (1000 controls/64 MiB) and add a wall-time cap.
Default to 8×100 controls/32 MiB/120 seconds. Freeze whole-episode 80/10/10 splits,
refusing empty sets or repeated initial scenes; the CLI requires at least seven
episodes. Check actual window availability independently. Official dataset family
is `official-pusht-capitals-v1`; legacy untagged GPC is `gpc-reference-v1`.
Reject mixed families in combined catalogs. Preserve legacy file/split semantics.

The first 800-control pilot validates storage/batches, not learned dynamics or
policy improvement. Next implement the visual world-model training/validation
path, then direct physical-simulator policy optimization before imagined updates.

## D018 — First visual dynamics pilot and failed planning gate (2026-10-19)

Train a separate 1.3-million-parameter CNN from scratch on official L data. Keep
the official diffusion policy frozen and its visual representation unchanged.
Inputs are two RGB/proprioception frames, goal and actions. Supervise image/agent
prediction plus privileged object-motion, geometry ROI, reward and contact targets.
Roll four controls autoregressively with predicted image/pusher history, never
future ground truth. Fit motion normalization on unique training transitions only;
use fixed workspace-coordinate scaling. Shuffle within shuffled episodes to avoid
repeated full-episode decoding and duplicate disk windows.

Use bounded CUDA-only training, fixed recorded losses, one best checkpoint and
validation contact-position error for selection. Keep initial-model/persistence
baselines separate and the real model-test arrays untouched. Record all/contact/
noncontact metrics by horizon; global image reconstruction is insufficient.
This is a visual system with privileged supervision, not a state-policy study or
image-only-supervised model. Independent output heads can disagree physically.

The first run is an implementation experiment, not a passed dynamics/planning
gate: contact error improved on the selected validation episode, but noncontact
motion and overall errors worsened. A matched fresh-space candidate probe found
almost constant reward predictions. Its primary max-reward scores tied physically,
so zero regret is uninformative. Separate posthoc final/sum diagnostics expose the
lost-alignment action differences without rerunning physics. Do not silently
reinterpret those score views as a preregistered improved planning result.

Improve data/action coverage and suppress false motion; freeze future ranking/
terminal protocols before comparisons. Require physical action checks and direct
simulator policy optimization before imagined updates. This checkpoint is not
ready to supply imagined policy training. Preserve the negative findings and
single-seed/episode limitations rather than tuning against the model-test split.

## D019 — Keep automated checks focused on research behavior (2026-10-19)

Remove twelve peripheral or repeated unit checks: message-validator examples,
gallery HTML/CSV formatting, fixed manifest values, seed-hash bookkeeping, basic
window/split argument rejection and an angle helper already exercised through
physical metrics. Keep the source checker and real commit/PR message validation.
Retain all installed simulator, video-frame alignment, data-integrity, causal
gradient and CUDA-training tests, plus source checks of geometry, accounting,
resource caps and split leakage. Prefer focused behavior checks over growth in
test count; changing the suite does not change any recorded experiment protocol
or justify fewer final research evaluations.

## D020 — Assess pretrained dynamics before extending the custom pilot (2026-10-19)

The custom L-only CNN establishes a working training/evaluation path, but does
not establish T-to-L model transfer or reliable imagined control. Keep it as a
comparison. Prioritize a compatibility/resource assessment of the released GPC
world model, with DINO-WM as an alternative. This is an investigation order, not
a claim that either checkpoint has been loaded or validated in the official scene.

Use metadata/source inspection first: pinned revisions, exact required assets and
byte sizes, licenses/dependencies, image appearance/history, action normalization,
goal conventions and control timing. Then make a bounded T prediction comparison
against real physics before L adaptation, recording storage and GPU costs. Preserve
the official policy baseline, frozen episode splits and real model-test quarantine.
Large generalist models remain optional extensions after a compatible first model.
Do not enlarge the present foundation milestone just to include every candidate;
its actual completed work and negative pilot results are independently reviewable.

## D021 — Preserve pretrained inference and expose its domain mismatch (2026-10-19)

Load the released phase-two GPC checkpoint with pinned file/source hashes and an
unchanged upstream sampler. Reuse the existing environment and one 33.42 MiB
checkpoint; do not acquire its demonstration data or T-specific reward predictors.
Use four RGB [0,1] frames, three past actions and the next proposed action;
normalize absolute targets with the source dynamics scale, not the policy scale.
Isolate legacy absolute imports and fork/restore CUDA inference randomness.

Check two hand-constructed pre-contact scenes in each native/official T family,
with matched cold-restored branches and noise. Count history/branch/reset work;
disclose geometry-based scene/action priors. Keep generated reports/images ignored.
ROI/physical labels are privileged evaluation only; appearance-based centroid/IoU
metrics are proxies, not verified reward or pose decoding. Native side rotation
fails and official predictions recolor the object/remove the goal. These findings
do not pass the official dynamics gate despite successful loading and a good
native bottom-push prediction. Separate appearance and physics adaptation on T
before L. Keep DINO-WM as an alternative and the L-only CNN as a comparison.

The released quantized/no-grad sampler is inference-only; no differentiable
imagined optimization is validated. Pinned code/model metadata contains no explicit
license grant; do not redistribute upstream code or weights without clarification.

## D022 — Isolate palette effects with shared physical branches (2026-10-19)

Fix `configs/gpc_appearance_probe.json` before running the six-scene T diagnostic:
three model-noise repeats, four candidate sequences and four predicted controls.
Render official physics in both its original palette and an auxiliary red/black
palette; keep original policy observations, physical state and render cache intact.
Use each actual branch once for all views/noise repeats, with native GPC physics
as a separate control. This is a privileged simulator-rendering control, not a
deployable image-only converter or learned model adaptation. No future geometry
or true future image is passed into the predictor. Preserve raw generated history.

Use fixed known-color object/target masks across both palettes, with occupancy
IoU, centroid error, visible-target recall and within-palette image/ROI errors.
They are image proxies, not verified physical pose/reward estimates. Do not
interpret cross-palette MSE differences as pure motion improvement or count three
noise repeats as independent physical trajectories. Six development scenes do
not establish generalization; two were already examined. The numerical image
conversion fix preserves the same protocol; retain failed-run counts/evidence.

Palette control markedly improves most predictions and target retention, but
the left push still misses rotation and its raw-input prediction is better on
that case. Keep these failures visible. Next calibrate pretrained GPC on explicit
official-RGB T training episodes with contact/rotation coverage, comparing frozen
initial and adapted weights on disjoint scenes. Keep this palette control as an
explanation/comparison, preserve the official policy and final model-test data,
then move to L after checking actual action effects. Do not claim policy learning.

## D023 — Publish a small reviewable evidence snapshot (2026-10-19)

Commit four original world-model PNGs and three selected JSON report summaries
under `docs/media/world-model-2026-10-19/`, with a size/hash manifest and a 1 MiB
cap. Preserve image bytes, recorded source-report hashes, negative results,
privileged-rendering caveats and failed-run accounting. Summaries are selections,
not complete raw reports. Link the figures/reports and existing 27 alphabet MP4s
from `docs/RESULTS.md` so coauthors can review without installing the environment.

Keep datasets, source-frame archives, weights, caches and full raw reports out of
Git. State that these remain local and no shared raw archive exists; hashes are
identifiers, not access or backup. The selected 775,943-byte snapshot is a bounded
publication exception. Use the existing source checker for manifest integrity;
no presentation-only unit tests or new runtime experiment are needed.

This completes a foundation/diagnostic review milestone for PR #2. Review/merge
does not require successful adaptation. Continue the next training milestone on
a new branch after coauthor review and an authorized merge; do not merge automatically.

## D024 — Inspect real denoising and near-miss action timing (2026-10-20)

Observe the official scheduler instance temporarily, copying its initial sample
and actual DDPM outputs without consuming RNG, replacing outputs or patching
upstream files. Verify both current and following action chunks against unchanged
inference, and mark the original executed indices 1:9 separately from the full
16-target prediction. Intermediate inverse-normalized action samples are display
coordinates, may be outside the scene, and are not object predictions.

Replay the original alphabet report from its initial snapshots and policy seeds,
checking saved actions, physical/controller states, contacts, metrics and terminal
flags. Capture the plan containing earliest best IoU and its neighbors for T, D
and near-T. This is posthoc development analysis, not a new held-out-shape result.
The offline viewer separates fixed-scene denoising from physical execution.

Compare continuation with a privileged eight-control hold at the known best
moment. Recreate contact history by replaying the complete initial prefix; do not
restore a mid-contact snapshot or zero object velocities. Preserve the original
strict coverage > 0.95 success rule and record all extra interactions separately.
Holding preserves more overlap in the two examined failures but does not achieve
success. It supports a future predeclared prefix-length study, not a deployable
stop detector or a full explanation of the encoder's internal representation.

No dependency or asset change. Keep generated HTML, NPZ, PNG, MP4 and raw reports
ignored. Source and reproducible commands are shared; generated diagnostics are
not included in the existing publication exceptions. PR #2 is already merged;
this feature starts on a new branch from merged main.

## D025 — Fixed-duration original T continuation diagnostic (2026-10-20)

Add an explicit `--continue-after-success` mode to the official T runner, allowing
at most 500 controls while preserving its normal 300-control, stop-on-success
default. Keep the exact original coverage > 0.95 reward/success calculation.
Continuing means ignoring the raw flag only as a runner stopping condition;
never reset/freeze the simulator or reset policy history/RNG at that crossing.
Use a distinct diagnostic report kind and preserve every raw flag. Distinguish
ever reaching success from final success and post-success retention. Label the
video with controls, coverage and IoU, and keep outputs ignored.

The first requested experiment is one 500-control rollout using the original
development scene 100000 and policy seed derived from base seed 42. Compare the
prefix through first success against the existing baseline, then inspect all
remaining controls. This is a development-case stability observation, not an
adaptation result or a new estimate of overall policy performance. See the
actual outputs and limits in `docs/VALIDATION.md`.

## D026 — Shared multi-shape learning without new human data (2026-10-20)

The team clarified the intended outcome: improve one T-initialized diffusion
policy across multiple letter shapes using world models and RL, without new
human demonstrations or human success labels. The released policy's original
expert T training remains a dependency; simulator-derived reward/geometry labels
are privileged supervision, not a claim of learning without any supervision.

Keep T-to-L as the first functional learning check, then train/validate a shared
world model and policy on a predeclared, diverse development set and expand to
the alphabet as resources permit. Machine-generated data can mix automated play
and frozen/current-policy interaction, including contact, rotation, near-goal
and recovery states. A pretrained model need not be trained on every letter if
its predictions transfer, but each imagined-training domain must pass physical
action/reward validation. Freeze model weights during policy updates. Do not
query true future simulator states to score imagined transitions.

Compare limited action-network updates, optional small adapters and full
fine-tuning with matched data/control/compute records. First establish direct
simulator RL, then compare imagined RL. Include T practice or frozen-T-policy
imitation using automatically collected observations, and measure forgetting.
No RL trainer, adapter or multi-shape training result is implemented by this
scope decision.

Keep official coverage > 0.95 success and separately report final/peak alignment
and retention. Any learned hold/retreat or end-zone objective requires a declared
task/reward comparison. Training all letters tests multi-shape competence on new
scenes; unseen-shape claims require exclusions from both model and policy data.
Previously explored letters are development shapes, not an untouched final test.
This clarification supersedes L-only final-scope wording; it does not justify
starting unbounded all-letter collection or training before the basic loop works.

## D027 — Specify the learning handoff and separate its two workstreams (2026-10-22)

Use `docs/LEARNING_PLAN.md` as the detailed implementation/experiment handoff,
with a short README overview and links from agent instructions/onboarding. It
defines data sources/splits, world-model compatibility and contact/reward checks,
supervised versus policy losses, action timing, proposed reward, physical/imagined
iterations and three developer work packages. Proposed settings/file names are
not implemented defaults or completed training. Record the actual configuration,
owners, numerical prediction tolerances and resource caps before each experiment.

Develop direct-simulator DPPO and pretrained prediction assessment independently;
require a checked physical learning loop and validated dynamics before imagined
policy claims. Model compatibility/action effects precede size comparisons.
Start with T/L, then a declared diverse subset, then A-Z if justified. No new
large-model download or all-letter training is required by this plan. Favor one
shared predictor; adapt pretrained components when measured failures justify it.

Propose sparse original completion plus discounted coverage-potential shaping,
with explicit success/time-limit handling. Keep raw upstream rewards and original
success in reports; sustained completion is a separate task. Freeze dynamics
during policy phases, derive imagined rewards from predictions and test learned
policies physically. Include all model-data collection in end-to-end costs.

## D028 — Implement a checked direct-simulator tail-PPO pilot (2026-10-23)

The method survey and derivations are in `docs/POLICY_LEARNING.md`. Start with a
specified DPPO-style variant rather than claiming a new algorithm or reproducing
all published DPPO settings. Keep the official 100-step DDPM and update Gaussian
transitions t=K..1; freeze early steps and deterministic t=0. Use joint conditional
densities over all 16x2 latent coordinates, no ELBO/marginal-likelihood proxy,
coordinate averaging, extra noise or changed inference prefix. Denoising discount
is 1. The evaluated/saved policy must follow this same parameter-use schedule.

Support final-convolution-head or full action-network updates through one script;
keep the image encoder/normalizers frozen. These modes differ in trainable capacity
and memory and need separate evidence. Retain original expert pretraining and
privileged coverage reward disclosure. Use physical-time potential shaping and
GAE, success-only zero bootstrap and explicit timeout/reset boundaries. No new
human demonstrations, learned dynamics, teacher term or stopping head is added.

First verify probability calculations, unchanged initial actions/RNG, real CUDA
weight updates, frozen-weight integrity and checkpoint action parity. A 64-control
run verifies mechanics, not L learning. Run a bounded longer development pilot
before considering bigger experiments. Keep checkpoints/reports ignored. Initial
checkpoints contain actor deltas, critic and Adam states but have no automatic
resume; original pinned weights and configuration remain required. Keep PR #4's
inspection/plan milestone reviewable; training uses `feat/direct-dppo-pilot`.

## D029 — Separate parameter scope, LoRA and visible evaluation (2026-10-23)

The initial head-only runs check the learning system, not sufficient capacity
for shape adaptation. Make full action-U-Net updates versus distributed Conv1d
LoRA the main capacity comparison; retain head-only variants as limited controls.
Image encoder updates need a separate gradient/data implementation. LoRA is an
established weight parameterization, not a new RL algorithm; the current work
does not establish algorithmic novelty. Keep the original CNN primary. The
authors also released a Transformer PushT policy; it requires its own asset and
baseline compatibility checks before a comparison, not a silent replacement.

Add native-PyTorch Conv1d factors, zero-initialized B, configurable rank/scaling
and head/action-network targets. Freeze base kernels/biases/normalization and
disable factors on frozen denoising steps. Save only factors/critic/Adam state.
Check dense-kernel equivalence, original-policy parity, distributed gradients,
unchanged frozen parameters and compact reload. No extra dependency is needed.

Keep training headless. Add a separate trained-delta desktop evaluator, with
optional action/reward/overlap checks against the saved development rollout.
Label previews as evaluation, not ongoing training. Count their physical work
separately. A small LoRA check saves parameters/storage but did not make the
100-step sampling/evaluation run faster; optimize measured stages and preserve
matched baselines for sampler/precision changes.

## D030 — Compare forward-process RL without requiring novelty (2026-10-23)

The course project does not require an original algorithm. Keep the intended
world-model/RL and shared multi-shape scope; assess established methods through
substantial implementation and controlled experiments. Use
`docs/DIFFUSION_RL_RESEARCH.md` for the extended primary-source survey and next
handoff. Retain tail-PPO and build a separately identified DDPM-compatible
DiffusionNFT comparison; AdvantageFlow is a later loss comparison on common data.
This decision adds a specification, not an alternative trainer or learning result.

Preserve the official checkpoint's representation/schedule initially and verify
the epsilon-prediction adaptation. Disclose differences in parameter use across
denoising steps versus tail-PPO. Generate candidate groups from the same context;
replay the full physical prefix to reconstruct contacts and count all branch,
replay and continuation controls. Use physical feedback first, then frozen-model
feedback only after action/reward validation. Confidence-weighted feedback is an
optional measured ablation, not a novelty or physical-improvement guarantee.
Motus2, WISE and model-error-aware RL already establish related combinations.

## D031 — Implement physical forward learning and multi-shape data (2026-10-23)

Add a separate `official_forward_rl` actor/checkpoint and trainer for NFT and
AdvantageFlow. Use the original 100-step epsilon DDPM, with actor weights at all
steps; preserve tail-PPO as a distinct baseline. Train full action U-Net or
distributed Conv1d LoRA with a frozen image encoder. Forward-noising tensors must
not mutate the inference scheduler: diffusers' helper moves its alpha buffer and
changes scalar rounding, breaking unchanged-policy/fresh-load parity.

Group candidates by exact RGB/pusher context, rebuild contacts through initial
state plus full action-prefix replay, and verify state/contact/history agreement.
Score actual finite-horizon shaped returns with frozen-policy continuation and
no critic bootstrap. Keep true-success versus horizon-cutoff handling explicit.
Use fixed positive score scale, neutral ties and a positive AdvantageFlow anchor;
T retention uses initial-policy-generated plans, never new human data. Contexts
advance via the last IID trial, not a reward-selected oracle trajectory.

Save complete optimizer-update boundaries atomically. Resume into a new output
directory with matching source/config and restored optimizer/RNG/episode ledgers.
Charge recorded discarded work from the parent report; flag unknown partial work
after interruption. Keep physical, replay, evaluation and inference counts separate.

Extend official collection to the existing alphabet/near-T registry without
changing T/L physics or GPC data identity. Shared visual-model windows use each
episode's geometry for labels/masks; actor/model inputs gain no letter ID.
T/L remains the learning check; the explicit T/L/I/Y/near-T profile is a proposed
development expansion. This does not supply a reliable predictor, pretrained GPC
training or imagined RL. Keep all new raw reports/deltas/video ignored.

## D032 — Portable training, live scalars and a Blackwell profile (2026-10-23)

Always keep flushed local scalar logs; mirror to W&B only with an explicit
offline/online tracking choice. Separate monitoring settings from learning
settings. Use a new tracking run for each recovery invocation, preserving
cumulative optimizer/control axes and checkpoint lineage. Do not upload raw
weights/data/code/console through the tracking adapter. W&B failure falls back to
local events and is recorded. A local standard-library viewer needs no account.

Add fixed baseline-plan/noise denoising diagnostics without changing training RNG
or using those examples for optimization/selection. Physical per-shape development
success/final coverage selects `best.pt`; retain the original frozen comparison.
Add regular forward evaluations, bounded detailed-update retention, sequential
multi-seed/objective plans and a read-only GPU/asset/disk preflight. Custom visual
dynamics gain step/cursor/optimizer recovery and explicit longer caps. Tail-PPO
logging is supported, but its older recovery limits remain unchanged.

Keep cu124 unchanged. Add mutually exclusive cu128 (torch 2.7.1 / torchvision
0.22.1) for Blackwell candidates, plus an independent W&B 0.18.7 tracking extra;
remove torch pins from `dp` so callers choose exactly one backend. Manifest and
lock change together. Metadata resolution does not validate Blackwell, dense
full-network training, or useful adaptation; perform machine-specific runtime
checks. No new environment or package installation is part of this change.

## D033 — Opt-in batched lock-step forward rollouts (2026-10-24)

Denoiser sampling, not physics, dominated forward training: every plan was a
batch-1, 100-step DDPM call. Add `batched_rollouts` (default false). When true,
every candidate in a group has its own environment, restored from the initial
snapshot and replayed through the full prefix with the existing verification,
and its own RNG stream derived from the plan RNG; each plan step is one batched
denoiser call. Batched evaluation steps all seed/shape episodes in lock-step and
seeds each episode's generator exactly as the sequential path does.

Contexts, candidates, horizons, rewards, feedback and success are unchanged, but
the random streams differ: batched and sequential runs are statistically, not
bitwise, comparable, and a saved run cannot be resumed with the flag changed.
Convolution roundoff depends on batch size (cuDNN TF32 default). Identical-seed
evaluations agree to about 0.02 px for dozens of controls, then diverge
chaotically, which can flip a near-threshold success. Keep existing presets
sequential, compare methods within one mode, and use more evaluation scenes than
three for claims. `torch.compile` of the U-Net was measured (about 1.3x) but not
adopted because it adds a further numerical change for a small gain.

## D034 — Near-miss shapes as the first learning-loop check (2026-10-24)

Every T/L pilot left L at 0/3 with peak coverage near 0.45, far below the 0.95
success threshold. At the author's direction, also train shapes the frozen
baseline nearly solves, so physical improvement can show up as new successes:
`configs/nft_nearmiss_batched.json` trains T (retention), D, `T_narrow` and Y and
evaluates those plus L. L is untrained there but is a fixed development shape,
so it is a visibility check, not unseen-shape generalization. The near-miss
choice came from the single-scene alphabet sweep (D best IoU 0.815, near-T 0.872,
Y solved); the run's five-scene baseline must confirm it. T/L lineages continue
as the original comparison.

## D035 — Stop long forward-process RL until the training signal is reliable (2026-10-25)

Long NFT/AdvantageFlow runs either left the policy unchanged (small steps) or
degraded T in proportion to drift (dense lr 3e-5: T 0.55 -> 0.05 by update 75).
Only 16-36% of a candidate's 32-control score variance comes from its own plan
(split-half reliability 0.14-0.29); the remaining follow-up plans dominate, and
updates anchor only to the previous policy. A privileged value function did not
rank candidates better. Longer runs of the same recipe were therefore stopped at
about update 150. Revisit forward RL only with a measured reliable score and an
explicit anchor to the start policy. See `docs/LEARNING_FINDINGS.md`.

## D036 — Fixed scene pools and a pre-registered acceptance test (2026-10-25)

Evaluation noise is large: rerunning the same policy flips 37-40% of T scene
outcomes, and validation gains failed to replicate because of checkpoint
selection. Scene pools are disjoint seed ranges (development 100000, validation
400000, monitor test 300000, value 600000, oracle 700000, teacher data 800000,
final test 900000). Select on validation, track on the 100-scene monitor test, and
use the 1000-scene final test only once per candidate policy: ship a policy only
if T and near-T are non-inferior and the new letters improve, with margins fixed
before that test. Otherwise the base policy remains the result.

## D037 — Simulator-oracle best-of-N is an upper bound, not a method (2026-10-25)

Choosing among the frozen base's own plans with exact simulator replay raised Y
from 3/20 to 8/20 and new-letter coverage, but imitating those choices stayed
within evaluation noise. Base samples at one observation are nearly identical,
so re-ranking cannot create new behavior. Keep `oracle_bon.py` as a diagnostic.
Its decision records serve only as T/near-T retention data.

## D038 — A structured privileged state world model (2026-10-25)

A generic MLP state model was too coarse (8-control contact error 2.3 px) for a
task that needs about 1.5 px. `wm2` copies the simulator's known parts, the
kinematic pusher PD law and the letters' rectangle geometry, and learns only the
object's motion in contact, in the object frame, behind a no-contact gate. It is
trained on 1500 non-expert base-policy rollouts with action noise; median 8-control
contact error is 0.22-0.32 px. This state-based mechanism study is privileged and
is not the visual world model of PRs #6/#7; it does not replace that work.

## D039 — Sampling planner teacher, distilled into the image policy (2026-10-26)

SuperMimic-style BPTT through `wm2` did not learn to push (real coverage stayed
below 0.12). A sampling planner inside `wm2` with structured push candidates
succeeds in the real simulator on most D, Y and L scenes, but not on T/near-T.
Use it as a privileged teacher: collect its closed-loop demonstrations, distill
them into the image policy with the policy's denoising loss, and anchor only T and
near-T records to the base. Push candidates, privileged state and geometric
rewards are disclosed priors. Only the distilled image policy, tested under D036,
can count as adaptation.

## D040 — Blend selection and the pre-registered final test (2026-10-27)

Distilled students gained on the new letters but lost 10-15 points on T. Blending
their weights with the base, theta = (1-s) base + s student for the action network
and image encoder (`interpolate.py`, s = 0 is exactly the base), recovered T on
validation while keeping part of the gain. Six blends (two students x s = 0.3, 0.5,
0.7) were compared on validation; the top three go to the 100-scene monitor test.

Selection, fixed before the final test: among the monitor-tested blends whose T and
near-T successes are at least the base's lower monitor-test values (38 and 33),
choose the largest Y + D + L success total; ties go to the smaller s. If none
qualifies, no blend goes to the final test.

Final test: the chosen blend on the 1000-scene final-test pool (900000 + 20*i, policy
noise seed 42), paired by scene with the existing base run. Per shape, the 95%
interval of the success-rate difference uses the paired per-scene differences
(normal approximation). Pass only if (1) T and (2) near-T each have a lower bound of
at least -5 points, (3) at least one of Y, D, L has a lower bound above 0, and (4) no
new letter has an upper bound below 0. A pass is reported as image-policy adaptation
with every disclosure (privileged planner teacher, selection on validation and the
monitor test). Otherwise the base remains the result. Other candidates may be tested
later, but only this first one is the primary result; any later ones are reported
as secondary, with the multiple comparisons stated.

Outcome (2026-10-27): the rule selected the r4e60 student at s = 0.7. Final test: T
39.6% vs 44.4% (difference -4.8, 95% interval -8.7 to -0.9), near-T -0.9 [-4.7, +2.9],
Y +2.9 [+0.2, +5.6], D +4.6 [+3.2, +6.0], L +0.1 [-0.2, +0.4]. Condition 1 failed, so
the base remains the result; see `LEARNING_FINDINGS.md`.

Secondary result (2026-10-27): after the part-swap diagnosis, the same rule applied to
three monitor-tested encoder-heavy blends selected the large student (lr 1e-4) with the
base action network and a 70% student encoder. Final test: T -2.5 [-6.2, +1.2], near-T
-2.0 [-5.8, +1.8], Y +8.7 [+5.7, +11.7], D +1.1 [+0.3, +1.9], L +0.0. Conditions 1 and
2 failed (lower bounds below -5); reported as the second look at the final pool.

## D041 — Confirmatory test of the secondary candidate on a fresh pool (2026-10-27)

The secondary candidate from D040 (large student, lr 1e-4: base action network plus
70% student encoder) is fixed. It is tested once more, against the base, on a pool
never used for training, selection or evaluation: 2000 scenes per letter with seeds
2000000 + 20*i (policy noise seed 42), base and candidate paired by scene. The same
four conditions and 95% intervals as D040 apply. Because the candidate was chosen
before any of this pool existed, the test is free of selection bias. It is reported
together with both earlier looks at the 900000 pool, not instead of them.

Outcome (2026-10-27): T 46.0% vs 42.8% (-3.1 [-5.9, -0.4]), near-T +0.4 [-2.2, +3.1],
Y 8.8% vs 16.8% (+8.0 [+6.0, +9.9]), D +0.8 [+0.3, +1.3], L 0. Condition 1 failed: the
Y gain is confirmed without selection bias, and so is a T loss of about 3 points.

## D042 — Two encoder-share candidates on a second fresh pool (2026-10-27)

On the 100-scene monitor test, blends that keep the base action network and take 30% or
50% of the large student's encoder (`big4-unet0-enc0.3`, `big4-unet0-enc0.5`) scored T
41/39, near-T 41/43, Y 13/16 against base 38-40, 33-38, 7-11. The 70% share lost about 3 T
points on 1000 and 2000 scenes, so smaller shares may keep T. Both candidates are fixed
now and tested once against the base on a pool never used before: 2000 scenes per letter,
seeds 3000000 + 20*i, policy noise seed 42, paired by scene. Because two candidates are
tested, intervals are 97.5% (z = 2.24, Bonferroni); each candidate passes only if D040's
four conditions hold at that width. Results are reported for both, with every earlier
look.

Outcome (2026-10-27): both candidates passed. 50% encoder: T +1.8 [-1.3, +4.8], near-T +1.4
[-1.6, +4.3], Y +6.2 [+4.0, +8.4] (8.6% -> 14.8%), D +0.9 [+0.2, +1.5], L +0.1. 30% encoder:
T +3.0 [+0.1, +6.0], near-T +2.6 [-0.3, +5.6], Y +3.2 [+1.1, +5.3], D 0, L 0 (all 97.5%
intervals, 2000 scenes per letter). This is the first image policy that passes the
no-regression rule; it is reported with the three earlier tests that did not pass.

## D043 — Replicate the passing blend on a third fresh pool (2026-10-28)

The 50% encoder blend from D042 (`big4-unet0-enc0.5`) is tested once more against the base on
a third pool never used before: 2000 scenes per letter, seeds 4000000 + 20*i, policy noise
seed 42, paired by scene. With one candidate, intervals are 95%; D040's four conditions
apply. This is an independent replication: candidate and rule are fixed before the pool
exists. The result is reported whether or not it passes.

Outcome (2026-10-28): passed. T 43.4% -> 45.1% (+1.8 [-0.9, +4.4]), near-T +0.9 [-1.7, +3.6],
Y 9.8% -> 14.1% (+4.3 [+2.4, +6.3]), D 0.1% -> 0.9% (+0.9 [+0.4, +1.3]), L -0.1 [-0.1, +0.0].

## D044 — Target a full new policy, accept a reported T trade-off (2026-10-30)

The encoder blends that passed D042/D043 keep the base's action network, so they cannot carry
most of the teacher's skill (camera policy D 1% vs teacher D 100%). Decided with the project
lead: the remaining work trains one full new camera policy for all letters instead of
encoder-only changes. A small-to-medium drop on T is acceptable; the final report must state
the T loss against the new-letter gains, and T should be recovered at the end if possible.
First step: find out why copying the teacher fails (a student that reads exact state, like the
teacher, separates perception from imitation), then train with the improved noisy-state
teacher and more student-driven correction data.

## D045 — Copy the teacher like a demonstrator, then correct the student's own states (2026-10-30)

Decided with the project lead. Students are trained with the official Diffusion Policy recipe on every step of the
improved teacher's episodes (plus the original human T demonstrations, which are not new demonstrations), not with
an anchor to the original policy. A state-input student showed that most of the gap is in copying, and a literature
check (DAgger, Ross et al. 2011; Learning by Cheating, Chen et al. 2019; PLATO, Kahn et al. 2017; Diffusion Policy,
Chi et al. 2023) points to teacher relabelling of the student's own states, longer training, and closed-loop
execution. Next: 3-4 DAgger rounds (student drives on training scenes, teacher labels), long training on the 3-day
machine, and tests with 1-8 controls per plan, always on the same 50 validation scenes as the teacher.

## D046 — Fair test of the correction-round-4 student (2026-10-31)

Candidate fixed now: `runs/dagger-loop/round4/ema-step20000.pt` (official recipe on teacher demonstrations and the
original human T demonstrations, then four teacher-relabelled DAgger rounds). On the teacher's 50 validation scenes it
scored T 19, near-T 16, Y 17, D 21, L 15 (original: 20, 17, 4, 0, 0). Test: candidate and original policy on a pool never
used before, 1000 scenes per letter (seeds 7000000 + 20*i), 8 controls per plan, policy noise seed 42, paired by scene.
Reported per letter: success rates, the paired difference with a 95% normal-approximation interval, and mean final
coverage. Following D044 this is a trade-off report, not a pass/fail gate: the T and near-T changes are stated next to
the Y, D and L gains, whatever they are.

Outcome (2026-10-31, 1000 fresh scenes per letter, seeds 7000000 + 20*i): T 42.6% -> 36.3% (-6.3 [-10.5, -2.1]),
near-T 40.2% -> 29.3% (-10.9 [-15.1, -6.7]), Y 9.9% -> 33.6% (+23.7 [+20.3, +27.1]), D 0.0% -> 33.1% (+33.1
[+30.2, +36.0]), L 0.0% -> 28.8% (+28.8 [+26.0, +31.6]); mean final coverage rises on every letter (T 0.70 -> 0.86,
near-T 0.68 -> 0.86, Y 0.56 -> 0.85, D 0.32 -> 0.79, L 0.32 -> 0.85). Average success over the five letters 18.5% ->
32.2%. The student brings T and near-T closer to the goal but finishes them less often: a medium T/near-T loss traded
for large gains on Y, D and L (the old D040 no-regression conditions are not met; under D044/D046 this is reported as
the trade-off).


## D047 — Fair test of the correction-round-7 student (2026-10-31)

Candidate fixed now: `runs/dagger-loop/round7/ema-step20000.pt` (plain teacher-relabelled DAgger, seven rounds; best
so far on the teacher's 50 validation scenes: T 28, near-T 14, Y 16, D 28, L 14, total 100 of 250). Chosen with the
project lead because rounds after round 4 appeared to recover T and near-T. Test exactly as D046 but on a new pool:
candidate and original policy, 1000 scenes per letter, seeds 8000000 + 20*i, 8 controls per plan, policy noise seed
42, paired by scene, success differences with 95% normal-approximation intervals and mean final coverage per letter,
reported as a trade-off (D044). Round 7 is compared with round 4's D046 result only descriptively (different pools).

Outcome (2026-10-31, 1000 fresh scenes per letter, seeds 8000000 + 20*i): T 43.4% -> 37.2% (-6.2 [-10.5, -1.9]),
near-T 40.3% -> 28.2% (-12.1 [-16.2, -8.0]), Y 9.1% -> 33.7% (+24.6 [+21.2, +28.0]), D 0.2% -> 38.9% (+38.7
[+35.7, +41.7]), L 0.0% -> 33.1% (+33.1 [+30.2, +36.0]); mean final coverage up on every letter. Average over the
five letters 18.6% -> 34.2%. Compared descriptively with round 4 (D046), round 7 is better on D and L and has the
same T/near-T loss; the apparent T recovery on the 26-letter pool did not hold on fresh scenes.


## D048 — Copyable teacher; 96 px main line, 192 px side experiment (2026-11-01)

Context: student A/B tests on 2026-11-01 (official recipe, 30k steps from the official checkpoint, teacher demos only,
100 validation scenes per letter for ten letters, successes of 1000): intended-plan labels 129 vs executed pushes 77
(same episodes); 192 px camera 129 vs 96 px 111; all demos 111 vs half 104; 990M vs 251M action U-Net from scratch
146 vs 138 (50k steps). Only the label change is clearly beyond test noise (about +-30). Success (coverage > 0.95)
allows only 1.3-2.7 px of shift, which is 0.24-0.51 of a 96 px camera pixel (E tightest, T loosest).

Decisions, agreed with the project lead:
- Copyable teacher: test the teacher re-planning every push (current) against committing to 8 or 4 pushes like the
  student, on the 50 validation scenes per letter. Rule fixed before results: use the better 8-push variant if it
  keeps at least 85% of the every-push teacher's successes, else the 4-push one under the same test, else the
  current teacher. Its demonstrations (training seeds 800000 + 20*i, i = 56200-56299) are compared with the every-push
  teacher's in the A/B setting above; the lines below use only the copyable demos if they win by at least 30 of 1000,
  otherwise all intended-plan demos.
- 96 px is the main line and carries every headline result, because it is the official policy's camera. 192 px is a
  labelled side experiment asking whether the camera limits final alignment: it is a sensor change, disclosed as such,
  and only compared with its matched 96 px control. Both lines start from the same 96 px student and follow the same
  schedule (round 0: 30k steps on all data; then correction rounds of 20k steps). For 192 px, the human T demos and
  earlier corrections exist only at 96 px and are enlarged; its own corrections are native 192 px. Training seeds:
  main line i = 53000 + 100*k, side line i = 57000 + 100*k (30 scenes per letter per round).
- The final candidate is the main-line student with the most validation successes among rounds finished by
  2026-11-02 14:00. Its fair test on fresh scenes is fixed in a separate entry before it starts.

Teacher outcome (2026-11-01, 50 validation scenes per letter, ten letters, successes of 500): re-planning every push
390; committing to 4 pushes 317 (81%); committing to 8 pushes, best sample 303 (78%), elite mean 123. Neither
commitment variant kept 85%, so the rule kept the every-push teacher. Even with exact state, committing to 8 pushes
like the student costs the planner 22% of its successes (near-T 30 -> 9), which bounds what an 8-push student can
copy. The new demos (i = 56200-56299) are therefore more every-push demos (`teacher-plan-demos3`); the planned C1 A/B
would only have repeated L1 and was skipped.

## D049 — Fair test of the final ten-letter student (fixed 2026-11-01 21:50, before any test run)

Test: the original policy and the candidate on 1000 fresh scenes per letter for the ten training shapes (T, near-T,
Y, D, L, E, F, J, P, U), seeds 9000000 + 20*i (i < 1000, never used before), 8 controls per plan, the evaluation
tool's fixed policy noise seed, paired by scene; success differences with 95% normal-approximation intervals and mean
final coverage per letter, reported as a trade-off (D044). The original policy's half runs first (2026-11-01 night);
it cannot influence the choice of candidate.

Candidate: the 96 px main-line student (D048) with the most successes on the 50 validation scenes per letter among
students tested by 2026-11-02 16:00 (ties: the later round). This moves D048's 14:00 selection time to 16:00 because
the original policy's half no longer has to run then. The 192 px side line is compared with its matched 96 px rounds
on the validation scenes only; it is not part of this test.

D049 amendment (2026-11-02 11:00, before any candidate test run): the main line was stopped early with the project
lead after a stored-data diagnosis showed it flat since round 2 (21 validation tests averaging 179 of 500, spread +-10,
as expected from chance alone; the same student tested twice flips 211 of 500 scene outcomes). The selection rule is
unchanged but applied now: the best validation test is round 14's (196), i.e. the student trained in round 13,
`runs/dagger10m/round13/train/ema-step20000.pt`. Given the diagnosis, every main-line student since round 2 is expected
to perform alike; 196 is a lucky draw, and the fair test measures its true level.

## D050 — Fresh-scene test of the 192 px side line (fixed 2026-11-02 11:30, before it runs)

At 15:00 the 192 px side line (D048) stops; its candidate is the student with the most successes on the 50
validation scenes per letter among tests finished by then (ties: the later round). It runs on exactly the D049 scenes
(1000 per letter, ten shapes, seeds 9000000 + 20*i), so it is paired by scene with the original policy's half already
run and with the 96 px main candidate. It is reported as a sensor-change side result next to the 96 px headline.

## D051 — 26-letter run and its final test (fixed 2026-11-02, before the run)

Agreed with the project lead for the next 3-day job. Shapes trained: the 26 capitals and near-T (27). Held-out test
shapes, approved the same day: `digit3`, `digit4`, `digit7`, `plus`, `hash`, `arrow` (tdyssey/letters.py
`NOVEL_PATTERNS`: same 5 x 7 block system as the capitals, one connected piece, none matching a capital). They are never
used for training, data collection, world-model fitting or tuning; the only contact before the final test was a
loading check of the original policy on 3 validation scenes each (2026-11-02).

Preparation (2026-11-02, chain_26prep.sh): dynamics data for the 17 new letters (value pool i = 1400-1799), world
model wm2 v3 on 27 shapes, teacher check on 50 validation scenes, every-push teacher demos with plans (training seeds
i = 50600-50749) for letters whose teacher solves >= 30/50.

Training: both D048 lines continue from their own best ten-letter students (96 px: D049 candidate; 192 px: D050
candidate). Round 0 retrains on all data for the 27 shapes (intended-plan labels); then correction rounds of 30 training
scenes per shape (96 px: i = 58000 + 50*k, 192 px: i = 52100 + 50*k), tested on 50 validation scenes per trained shape.

Final test: fresh seeds 10000000 + 20*i (i < 300), 300 scenes per shape for all 33 shapes, four policies: the original,
the ten-letter 96 px student (D049 candidate), and the 26-letter 96 px and 192 px students (each the best validation
test among those finished 12 hours before the job ends; ties: the later round). Parts of 100 scenes run in lockstep
across policies; if time runs out, every policy is reported on the same completed parts. Primary measures: mean success
over the 27 trained shapes, and pooled success and mean final coverage over the 6 held-out shapes; per-shape paired
differences against the original with 95% intervals; T and near-T reported separately.

D049 and D050 outcomes (2026-11-02, 1000 fresh scenes per letter, seeds 9000000 + 20*i, paired by scene; success %,
original -> 96 px main candidate (D049) / 192 px candidate (D050), differences with 95% intervals in
`runs/fairtest-d049/report-10letters.txt`):

| Shape | Original | 96 px | 192 px |
| --- | --- | --- | --- |
| T | 42.2 | 42.9 (+0.7 [-3.6, +5.0]) | 44.5 (+2.3 [-2.1, +6.7]) |
| near-T | 39.8 | 32.0 (-7.8 [-12.0, -3.6]) | 36.0 (-3.8 [-8.0, +0.4]) |
| Y | 8.4 | 39.4 | 39.0 |
| D | 0.1 | 42.5 | 45.7 |
| L | 0.0 | 34.7 | 37.8 |
| E | 0.0 | 26.7 | 30.4 |
| F | 0.5 | 26.1 | 33.1 |
| J | 0.0 | 23.8 | 24.5 |
| P | 0.4 | 47.3 | 58.1 |
| U | 0.0 | 27.4 | 32.5 |
| Average | 9.1 | 34.3 | 38.2 |

One image policy now covers ten shapes at 3.8x the original's average success, with no loss on T. Every gain on
Y-U is far outside its interval (smallest: J +23.8 [+21.2, +26.4]). Near-T is the only shape below the original
(96 px). The 192 px camera adds +3.9 points [+2.6, +5.2] over the 96 px student across all 10,000 paired scenes,
most on P (+10.8), F (+7.0) and U (+5.1); it is a sensor change and is reported as a side result.

## D052 — Ten-letter students on the 17 letters none of them trained on (fixed 2026-11-02 17:30, before it runs)

Agreed with the project lead. Policies: the original, the 96 px student (D049 candidate) and the 192 px student (D050
candidate). Shapes: the 17 capitals outside the ten training shapes (A B C G H I K M N O Q R S V W X Z) plus T, which
the evaluation needs as its reference shape and is reported separately. 200 fresh scenes per shape, seeds 11000000 +
20*i (i < 200, never used before), 8 controls per plan, paired by scene; parts of 50 scenes run in lockstep across the
three policies. Primary measures: pooled success and mean final coverage over the 17 letters, with paired differences
against the original. These letters become training letters in the 26-letter run (D051); the six held-out shapes stay
untouched.

D052 outcome (2026-11-02, 200 fresh scenes per letter, seeds 11000000 + 20*i, paired by scene;
`runs/unseen17-d052/report.txt`): on the 17 untrained capitals pooled (3400 scenes per policy) success rises from 0.2%
(original) to 6.6% (96 px, +6.3 [+5.5, +7.2]) and 10.6% (192 px, +10.4 [+9.3, +11.4]); mean final coverage from 0.33 to
0.51 and 0.57. Best transfers: B 20.0% / 39.5%, O 18.5% / 22.5%, W 13.5% / 19.5%, R 11.5% / 15.0%. T on these 200 scenes:
48.5%, 44.5%, 41.0% (within this test's +-7 points; the 1000-scene D049 test is the T result).

## D053 — Larger per-round budget for the 26-letter lines (2026-11-03, before use)

Agreed with the project lead after the 96 px 26-letter line (A100, rounds 1-12; validation 283 -> 358 of 1350) was
diagnosed: the recipe is unchanged (intended-plan labels, the same teacher, correction rounds, 1e-4), but 20k steps
shared by 27 shapes give each shape 190k training samples per round instead of 512k with ten shapes, and the 17 new
letters had about 11k correction samples each against 53-93k for the earlier shapes. From then on each round trains
54k steps (each shape again gets about 512k samples per round) and collects 60 practice scenes for each of the 17 new
letters (30 for the ten earlier shapes);
`scripts/learning/experiments/chain_dagger_loop26c.sh`. Training seeds: 96 px i = 57700 + 70*k (rounds
14-32), then i = 48440 + 70*k (rounds 33-50; 50750-51999); 192 px i = 52100 + 70*k (rounds 1-12); all below i = 60000.
Machines (option A): the 96 px line on the H100 job from round 14 (start: the A100's round-13 student); the 192 px
line on the A100 job from round 1 (start: `runs/dagger26h/round0`), with bc_train streaming its frames to local disk
(`TDYSSEY_FRAMES_MMAP`; peak host memory 9.8 GB instead of 222 GB) because that job had 250 GB.

D051 outcome (2026-11-05, 300 fresh scenes per shape, seeds 10000000 + 20*i, paired by scene;
`runs/fairtest-d051/report-final.txt`). Candidates picked at 19:53 by the fixed rule: 96 px round 44 (validation 470 of
1350), 192 px round 2 (511). Success %, original -> ten-letter 96 px -> 26-letter 96 px -> 26-letter 192 px:

| Shapes | Original | Ten-letter 96 px | 26-letter 96 px | 26-letter 192 px |
| --- | --- | --- | --- | --- |
| Ten earlier shapes | 9.3 | 33.0 | 33.7 | 34.2 |
| 17 new letters | 0.4 | 6.4 | 33.2 | 33.5 |
| All 27 trained shapes | 3.7 | 16.3 | **33.4** (+29.8 [+28.7, +30.9]) | **33.8** (+30.1 [+29.0, +31.2]) |
| 6 held-out shapes | 1.2 | 2.8 | 4.7 (+3.5 [+2.4, +4.6]) | 5.9 (+4.7 [+3.5, +5.9]) |
| Final coverage, 27 shapes | 0.35 | 0.63 | 0.82 | 0.81 |

One image policy covers all 26 capitals and near-T at 33-34% success with no loss on the ten earlier shapes; on the
six never-seen shapes success rises with the number of training shapes (1.2% -> 2.8% -> 4.7-5.9%; digit7 2% -> 19-23%).
T 46.0% -> 40.7% (96 px, -5.3 [-13.5, +2.9]) and 38.3% (192 px, -7.7 [-15.1, -0.2]); near-T 37.3% -> 29.0% / 32.3%.
The 192 px pick's validation lead (511 vs 470) did not hold on fresh scenes (33.8% vs 33.4%).

## D054 — 96 px line to round 50, then a fresh-scene check (fixed 2026-11-06, before it runs)

Lead's decision: the 96 px line continues on the next A100 job from round 45 (its practice data was collected before
the D051 pick) to round 50 with the D053 recipe, then stops (seeds i = 48440 + 70*k, rounds 46-50). Validation trend
over rounds 14-44: +0.17 points per round overall, +0.06 over the last ten. Check afterwards: the round-50 student
against the D051 pick (round 44) on 200 fresh scenes per shape for all 33 shapes, seeds 12000000 + 20*i (i < 200,
never used before), paired by scene; reported as the difference with a 95% interval. It replaces the D051 result only
if better; either way D051 stays the pre-registered headline. (The first attempt's node ran out of host memory while
loading round 45 in memory, about 145 GB of frames; the job was requeued and the line restarted with frames streamed
to disk.)

D054 outcome (2026-11-07; 6600 paired episodes, `runs/fairtest-d054/report.txt`). Round 50 against round 44, success %:
ten earlier shapes 34.0 -> 34.4 (+0.4 [-2.5, +3.3]); 17 new letters 32.3 -> 35.0 (+2.7 [+0.6, +4.9]); all 27 trained
shapes 32.9 -> 34.8 (+1.9 [+0.1, +3.6]), final coverage 0.823 -> 0.827; six held-out shapes 4.8 -> 4.4 (-0.4 [-2.0,
+1.2]). T 38.0 -> 44.0 and near-T 32.5 -> 37.5 (200 scenes each). Round 50 is better on the trained shapes, so it
becomes the 96 px model; D051 stays the pre-registered headline. Validation of rounds 45-49: 437, 443, 480, 453, 452
of 1350 (round 44: 470); the line stops here as decided.

## D055 — 192 px line continues on the next H100 job (2026-11-06)

Lead's decision: when the queued H100 job starts, the 192 px line continues from its newest student
(`runs/dagger26h/round11/train/ema-step54000.pt`, next round 12) with the D053 recipe. Training seeds (60 scenes per
round, all below i = 60000 and unused): round 12 i = 52100 + 70*k; rounds 13-19 i = 48590 + 70*k (49500-49979);
rounds 20-29 i = 53900 + 70*k (55300-55989); rounds 30-39 i = 54200 + 70*k (56300-56989).
The line stops 8 hours before that job ends; its best validation student then gets the D054 check on the same
fresh scenes (seeds 12000000 + 20*i, 200 per shape, 33 shapes) against the D051 192 px pick (round 2).

## D056 — Steps executed per plan: 8 against 12 and 15 (fixed 2026-11-07, before it runs)

The lead noticed in the demo that executing more of each 16-step plan (12 or 15 steps instead of 8 before planning
again) may solve more often. Test: the 96 px round-50 student with the tested sampler (100 DDPM steps) executing 12 and
15 steps, on the D054 scenes (200 per shape, 33 shapes, seeds 12000000 + 20*i, same policy-noise seed), paired by scene
with its existing 8-step results; reported as differences with 95% intervals for every setting, whatever the outcome.
The network always plans 16 steps and the first lines up with the current moment, so 15 is the most it can execute.
The policy was trained with 8; these scenes already decided round 50 against round 44, so a setting that wins here
needs a confirmation on unused scenes (seeds 13000000 + 20*i) before it replaces 8 in any headline result.

D056 outcome (2026-11-07; 6600 paired episodes per setting, `runs/fairtest-d054/report-d056.txt`). Executing more of
each plan is much worse on every shape. Success over the 27 trained shapes: 8 steps 34.8%, 12 steps 18.6% (-16.1
[-17.1, -15.2]), 15 steps 12.2% (-22.5 [-23.7, -21.4]); final coverage 0.83 -> 0.76 -> 0.72; six held-out shapes
4.4% -> 2.2% -> 1.3%; T 44.0% -> 23.5% -> 17.5%. The trained setting (8) stays; no confirmation run is needed.

## D058 — 192 px line stopped; last-mile RL on the frozen policy (2026-11-08, before any result)

Lead's decision: the 192 px line stops at round 12 (validation 476/1350, flat since round 2's 511; round 2 stays its
tested model). The GPU goes to two new experiments, both kept separate from the D051 headline.

Last-mile RL on top of the frozen 96 px round-50 policy. Start states: near-goal states recorded from that policy on
training scenes (seeds 800000 + 20*i, i in [20000, 30000), reserved here; block coverage >= 0.80, episode not yet
solved; exact simulator state plus the previous control's state for the two-frame camera input;
`scripts/learning/lastmile_states.py`). Episodes: at most 48 controls (6 decisions of 8 executed targets) from a
restored state. Reward per decision: coverage gain, minus a penalty on coverage lost, plus a one-time bonus when
coverage first exceeds 0.95, which ends the episode; time-limit cut-offs bootstrap. Two methods on the same states and
reward, both off-policy with the base frozen: (1) noise steering (DSRL): the actor chooses the 32-number starting noise
of the deterministic 10-step DDIM sampler (shown equal to the 100-step sampler, 254 vs 244 of 675); (2) a bounded
residual on the 8 executed targets. The actors read only the frozen policy's camera features and the pusher position;
the critics may read exact simulator state (training only, disclosed). Tests: finishing rate from near-goal states of
held-out scenes, then full episodes on an unused fresh pool (seeds 13000000 + 20*i), paired with round 50, with T and
near-T reported.

D058 amendment (2026-11-08, after the first run, before any v2 result). First run (v1, 150k decisions each, start
states from scenes 20000-20019): on 1080 near-goal states of held-out scenes 20030-20039, finishing within 48 controls
fell from 14.1% (base) to 8.0% with noise steering (-6.1 points [-8.5, -3.7]) and 12.8% with the residual (-1.3
[-3.8, +1.2]). Mean training reward rose to about zero while success did not: the doubled penalty on lost coverage
rewarded small, safe moves. v2 rebalances the reward (gain 10, extra loss penalty 2, success bonus 20, cost 0.2 per
decision), allows 12 decisions (96 controls) with gamma 0.95, trains 300k decisions on start states from scenes
20000-20029, and is evaluated on new held-out scenes 20040-20049 (20030-20039 are now used for tuning).

## D059 — New policy architecture (2026-11-08, before any result)

Lead's decision: a new camera policy built with current methods at about 500M parameters, trained on the same data as
the 26-letter line (teacher demonstrations with intended-plan labels, all correction records, the human T
demonstrations; equal share per shape). Design, following the literature check of 2026-11-08: a DINOv2 image encoder
(chosen over SigLIP: better on PushT in published comparisons and built for spatial detail), fine-tuned, on the two
camera frames upsampled from 96 px, pooled to a small grid of spatial tokens; a transformer action head (DiT-style:
self-attention over the 16 action tokens, cross-attention to image and pusher tokens, time by adaptive layer norm)
trained with flow matching (pi0-style Beta time sampling), 10 Euler steps at inference; 16 predicted targets, 8
executed, as before. Same fresh-scene test as D058, paired with round 50; correction rounds with the new policy only if
time allows before the talk.
