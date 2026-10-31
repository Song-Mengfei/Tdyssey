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
