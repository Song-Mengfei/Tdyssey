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

## D028 — Retain scoped native Windows compatibility (2026-10-23)

Carry only the established Windows AMD64 resolver/wheel markers, Linux-only
egl-probe override, UTF-8/platform checks, command-scoped Git ownership handling
and published-JSON newline protection into main-based milestone A. Keep existing
package pins; colorama 0.4.6 is the additional Windows transitive dependency.
Do not import simulator reranking or later research features. Existing Windows
doctor, official simulator and bounded CUDA inference/gradient checks passed;
this does not validate Windows online training or other renderers. Robomimic's
missing-egl-probe metadata warning remains an explicit Windows exception.

## D029 — Bound direct DPPO milestone A to checked numerical interfaces (2026-10-23)

Implement the detached-chain training sampler, conditional Gaussian likelihoods,
explicit action-weight white lists and pure physical reward/block/GAE interfaces.
Keep the official visual inputs, normalizers, fixed crop, 16 predicted targets and
1:9 execution prefix. One actor serves all 100 denoising times; freeze everything
except final_conv alone or up_modules.1 plus final_conv, recording actual names.

Preserve the original scheduler as a baseline. Training uses the exact locked
DDPM mean/clipping and a positive std floor at every transition including t=0;
.01 is untuned. Never clamp sampled Gaussian latents/noise. Sum 32 coordinate
log densities and use separate transition ratios averaged across samples/times.
The supplied PPO math differs from public DPPO aggregation heuristics and must
not be called a line-by-line reproduction. Reject numerical overflow explicitly.

Use first strict-coverage success bonus 1 plus discounted potential shaping;
success has zero terminal potential/no bootstrap. Timeout/buffer cutoff retain
the final observation and bootstrap but stop GAE. Discount by actual physical
prefix length; lambda applies once per plan boundary. Critic/image association
must be enforced by milestone B's collector, beyond these numeric helpers.

The existing local compact checkpoint differs from the manifest file digest.
Keep default digest acceptance unchanged; allow a caller-declared SHA-256 to pin
a separate local artifact while retaining size/config/provenance/strict-state
checks. This does not establish identity to released EMA tensors. Record the
actual digest and scope inference parity to the tested artifact.

No online trainer, real policy optimizer update, production data collection, large download
or learning curve belongs to A. Controlled toy updates and real-network finite
backward checks are numerical evidence only. See commands/results (external historical reference: `DPPO_MILESTONE_A.md`).

## D030 — Verify one real update within a permanent interaction budget (2026-10-23)

Extend A with official T/L fragments, CPU on-policy records, a small critic on
the same frozen visual/pusher features, whole-batch fixed GAE targets/advantages,
and one actor/critic optimizer step. Update all 100 recorded transitions per
plan with equal B*100 weighting; normalize actor advantages once for the full
batch, while critic targets retain raw GAE. Keep only final_conv trainable,
original normalization/crop and A's Gaussian semantics. This is a smoke check,
not evidence of task improvement or an authorization for longer training.

The permanent fsynced ledger caps all attempts at 64 physical controls, eight
sampled plans including inference replay, and one step per optimizer. Reserve
before each operation; failures do not refund, existing budgets never reset,
and a crash lock fails closed. Use 48 actual controls (24 each T/L), six rollout
plans and two nonphysical replay plans. Keep 480 control physics substeps and
two reset settles distinct. No further physical evaluation uses this budget.

Preserve per-control raw rewards/coverage/done and before/after observations,
the full sampled chain, old values and pre-reset final observation. Estimate T/L
fragments separately with explicit terminal/timeout/cutoff masks. Scene/shape
IDs are metadata only. Exclude existing evaluation scenes from collection.

Save selected actor deltas plus critic, full optimizer group order/moments,
torch/Python/NumPy RNGs, sampler/official/base identity and probes. Reconstruct
one network from the same base in another process, restoring optimizer state
without a second step; check weights, probabilities, actions and RNG replay.
This does not restore simulator contacts or prove released-EMA identity for the
locally pinned compact file. First reload exposed process-dependent ordering
in diffusers' private default-name list: canonicalize that list only, preserving
all actual scheduler settings and legacy records. The failed load consumed no
sampling/control budget and remains visible in the report.

The actual Windows update and reload passed; B's report and commands (external historical reference: `DPPO_MILESTONE_B.md`)
record finite gradients, real weight changes, exact frozen-state equality,
memory/time, storage and failure history. Long training, updated-policy physical
evaluation, multi-seed improvement and imagined training remain separate work.

## D031 — Implement bounded continuous DPPO before authorizing its run (2026-10-23)

Add a separate C trainer/evaluator; preserve B's strict one-update interface and
completed budget. Keep T/L environment history and local RNG across batches,
reset only on success/timeout, use fixed non-wrapping training scenes disjoint
from all development scenes, and retain actual executed-prefix discounts/masks.
For each fresh actor-version buffer, check behavior probabilities once, freeze
all records/targets and normalize advantages once; use first-batch one pass,
then up to three full B*100 equally weighted PPO passes with sample-KL early stop.
Validate frozen caches, whitelist/fingerprint and optimizer ownership before
updates; consume each buffer once and release its tensors/graphs afterward.

Pin the original base explicitly; a new C experiment does not silently start
from B's learned delta. The versioned proposal caps training at 2,048 controls,
384 plans, 24 batches and 72 steps per optimizer; development has distinct
24,000-control/3,040-plan/160-reset and four-group caps. Diagnostic interaction
capacity is zero. Permanent registry counters, failed reservations, two-hour
deadline and aggregate 512-MiB output limit survive output-directory changes.
Keep compact scalar logs, at most two incremental checkpoints and two selected
short traces. No automatic larger run or seed/evaluation sweep is permitted.

Save resumable state only at a genuine joint episode boundary after current
buffer consumption. Drain episodes under existing caps; never spend extra
controls to manufacture a boundary. Active-episode final deltas are explicitly
non-resumable. Recovery requires the exact current training/global ledger
watermark; independent development spending remains authoritative and separate.
Do not claim simulator contact-cache restoration or mid-update replay.

Evaluate original base sampling, training-sampler base and explicit C deltas in
separate processes with fresh environments and per-case local RNGs. Original T
and matched T/L protocols have separate labels and cannot be pooled. The fixed
five-scene/two-policy-seed list is development data, including two additional
development scenes; no untouched-test claim. Preserve incomplete cases/costs,
compare only aligned complete groups, and distinguish delta stage identities.

C is engineering only: source/fake/tiny checks, no real environment steps or
policy optimizer updates, 2,048/20,000-step training, formal evaluation, install
or download. C's interfaces, limitations and future commands (external historical reference: `DPPO_MILESTONE_C.md`)
do not authorize execution. Its runtime memory/throughput/recovery and policy
improvement still require separately scoped evidence.

## D032 — Validate fixed C resets without starting its training job (2026-10-23)

Authorize a separate bounded reset-only diagnostic based on `af09a4a`. Use the
pinned production official adapter, legacy reset, captured physical snapshot and
unchanged reset-validity checks. Count every reset and its actual single settle
substep; reject all controller calls and extra physics. Do not load a policy,
sample actions, optimize weights, run formal evaluation or initialize C's ledger/
clock. C's diagnostic budget remains zero; this diagnostic has its own explicit
212-reset/settle, 64-candidate, 300-second and 8-MiB limits and fsynced journal.

Keep original valid seeds in place and replace only observed invalid training
slots, using ascending predeclared T400032..400063/L500032..500063 sequences.
Exclude all original/selected training and fixed evaluation seeds. Use geometry
validity only, never policy performance. Keep evaluation fixed and fail closed
on invalid evaluation scenes, runtime errors, unknown costs or exhausted choices.
Re-reset the complete proposed list and bind results to its canonical hash.

The real diagnostic observed 11 invalid originals and one invalid candidate.
Eleven replacements preserve 32 unique seeds per shape; final 64 training and 10
evaluation resets all passed. Cost is 160 resets/settles and zero controls,
samples or real policy optimizer updates. Formal C remains uninitialized and
B's hashes unchanged. Full evidence and revised config (external historical reference: `DPPO_C_RESET_PREFLIGHT.md`)
record source/dependency/code identities, all rejected seeds, resource figures
and remaining limits. Reset-only memory is not C training memory; available
Windows RAM near 2.3 GiB is not proof of adequate model/PPO headroom. No model was
loaded speculatively and no user program was closed. Training requires separate
review and authorization.

## D033 — Freeze a separately budgeted world-model diagnostic (2026-10-24)

Use protocol v1 (external historical reference: `FROZEN_WM_PROTOCOL_V1.md`), independent of all B/C training and
evaluation accounts: 12 geometry-constructed L anchors and four official T
controls, four fixed candidates, three common-noise GPC repeats, at most four
future controls. Calibrate an RGB-only geometry decoder on 24 independent
render states before forecasting. Calibration failure blocks pose/coverage/
ranking claims; malformed predictions remain abstentions. Preserve strict
coverage >0.95 stopping, raw autoregressive RGB, endpoint scores, and explicit
physical versus predicted absorption. No training or optimizer construction.

The existing upstream-tracked csgo denoiser is a different checkpoint: 296/298
tensor entries differ from the manifest-pinned HF phase-two model. Downloaded
only the exact 35,043,916-byte pinned asset, without modifying the csgo file or
manifest. The old custom best.pt/catalog was not found; that historical model
cannot be run. Retain the initial DNS failure and full tensor audit locally.

The authorization is a single diagnostic attempt regardless of source/config/
output/ledger changes. Reserve and fsync costs before physical/model operations;
failures cannot refund them. Caps are 88 resets/restores, 448 controls, 768 GPC
images/2304 Euler steps, 1800 seconds, 256 MiB results and 64 MiB telemetry.
Reuse actual Windows worker PID/nonce/Job Object guards. Existing environments,
weights, data and the separate DPPO offline analysis remain untouched. Passing
these engineering screens would not itself authorize imagined policy training.

The first segment aborted on a Windows atomic-replace permission failure after
24 calibration states and four complete branches. Same-attempt journal recovery
kept the original deadline and conservatively charged the journal-ahead reset/
settle reservation, with no retry/replacement of failed L01/hold. The remaining
59 slots completed; total cost was 441 controls/756 forecasts and zero updates.
Offline v1.2 corrects gate classification and independent persistence-reference
availability from saved records only, preserving the original raw report. No
ranking gate passed; results and limits (external historical reference: `FROZEN_WM_VALIDATION.md`) are recorded.

## D034 — Repair image measurement using saved development data only (2026-10-24)

Use interface protocol v2 (external historical reference: `WM_INTERFACE_V2_PROTOCOL.md`) on the same saved RGB,
without any new environment, render, control, inference, sampling, update,
installation or download. Keep the old v1 protocol/decoder/config, source assets,
checkpoint bytes and spent ledgers immutable. Front/back audit covers all 595
old assets, including 150 B/C files. New versioned output directories hold results.

Preserve fractional color evidence and rigid complete-template checks, separate
object/shape/pusher/coverage validity, and retain strict coverage >.95. Goal is
used only after RGB pose fitting. A +/- .03 development error assumption enables
conditional success/non-success/unknown labels; it is not generated uncertainty
calibration or a guarantee. Full pose ambiguity remains explicitly unassessed.
Separate raw endpoints from predicted stopping prefixes. Unknown earlier stops
block sequence scores; future truth never repairs them. Keep missing rows, nest
noise within branches/anchors, and evaluate baselines on independent denominators.

The completed comparison (external historical reference: `WM_INTERFACE_V2_RESULTS.md`) improves real RGB bias and
pusher availability but leaves 652/756 generated object images unmeasurable.
All H1 object fits fail, including T; no stopping-consistent selection succeeds.
Near-threshold interval decisions leave all four true successes unknown. These
are reused development data, not independent validation. Appearance/domain,
deformation, dynamics and interface rejection remain partly confounded; do not
infer a sufficient color fix or weight-training method from this run. No new
experiment or adaptation is authorized by the result.

## D035 — Audit input compatibility before proposing adaptation (2026-10-24)

Complete a static source/provenance audit (external historical reference: `GPC_INPUT_COMPATIBILITY_AUDIT.md`),
without new model/decoder execution, physics, sampling, training, installation or
download. Keep runtime code, weights and all old accounts unchanged. Existing
RGB/action conventions agree with the checked GPC paths, but rendering palettes
and the rank-evaluator's action-marker path differ. Checkpoint training membership
is not established by domain18's evaluation YAML or generated red pixels. The old
source-palette probe used the same exact phase-two weights and a privileged
auxiliary renderer; it does not prove an RGB-only adapter works.

The design-only proposal (external historical reference: `GPC_INPUT_COMPATIBILITY_PROPOSAL.md`) fixes one RGB-only
palette transform versus original input, one frozen model, matched per-horizon
RNG states, AR plus separately labeled TF diagnostics on saved branches, and
staged T-then-L gates. Maximum new authority, **if later granted**, would be
2268 forecast images/6804 Euler steps and zero physical/policy/optimizer calls,
with a new independent ledger, fixed deadline/storage and Windows process caps.
No implementation or execution is authorized by this decision. A single failed
transform cannot establish that fine-tuning is necessary; unresolved measurement
cannot establish a dynamics error. Limited development compatibility, action
effects, termination and independent validation have separate conditions.

## D036 — Implement the frozen comparison without running it (2026-10-25)

Implement the fixed proposal (external historical reference: `GPC_INPUT_COMPATIBILITY_PROPOSAL.md`) in separate
T1, paired-sampling, reporting, accounting and Windows-guardian modules, keeping
the old model wrapper, sampler, decoder/config, artifacts and ledgers unchanged.
Default entry is a read-only plan. Future execution requires separate human
authority, an exact clean reviewed commit, original sealed-input hashes, new
output/ledger paths and native actual-PID/nonce/Job ownership before ACK.
An exclusive launch seal precedes resource/setup/binding work and binds the
same original deadline to the worker claim; startup failure cannot create a
second opportunity by changing paths or source.

Use constructed arrays/fake samplers and temporary ledgers for engineering tests;
native verification is limited to harmless sleeping worker/grandchild cleanup.
Preserve pre-call charges and first stop reasons, all planned/missing denominators,
raw generated feedback, per-horizon paired RNG and TF's scoring exclusion.
No actual saved RGB transform/decoder, checkpoint load, real model sampling,
physical interaction, policy update, installation or download is authorized by
this implementation task. Evidence and remaining limits (external historical reference: `GPC_INPUT_COMPATIBILITY_ENGINEERING.md`)
distinguish mock resource checks from native process ownership and future GPU/model
verification. The real independent registry remains unclaimed.

## D037 — Preserve the single guarded attempt's measurement-stage stop (2026-10-25)

Execute the fixed proposal once on independently reviewed `aecb82a`, with its
original 1800-second deadline, native guardian and permanent independent claims.
Continue beyond M/T only on their predeclared gates; never repair/retry a stopped
attempt, replace a scene, extend time or reset a claim. No physics, policy/model
training, warm-up, extra model probe, installation or download is authorized.

The actual attempt (external historical reference: `GPC_INPUT_COMPATIBILITY_RUN_V1.md`) stopped after 25.906s
in telemetry-file enumeration during M. It retained 40 calibration records,
zero model loads/images/Euler steps and null M/T gates. M is incomplete; T and
L are not attempted, not model failures. Native Job cleanup and byte audits pass;
the spent claims and all original evidence remain. No runtime source was fixed
or retried. The offending path was not logged, so a transient-file race is only
a candidate cause. Partial fits and an operational failure cannot establish T1's
generated-shape benefit, a dynamics error, terminal precision or a need for
fine-tuning. Any later engineering review or real run is a separate decision.

## D038 — Repair storage coordination without retrying the attempt (2026-10-25)

Use temporary fake files and harmless Windows workers to reproduce and repair
the storage race after `e5f2b79`. Keep both spent claims, all 40 durable rows,
failed journals/guardian, old assets and reports immutable. No real saved-image
transform/decoder, model execution, physics, training, download or installation
is authorized. A reproduced old mechanism does not prove the original incident's
cause when its offending path was not recorded.

The repair (external historical reference: `GPC_INPUT_COMPATIBILITY_STORAGE_REPAIR.md`) gives snapshots and legal
writes one bounded shared process lock. Exact status targets have fixed overlap
slots, journals enforce immutable prefixes, and completed results/claims remain
immutable. Pending output bytes are reserved before creation. Missing files,
reparse points, path traversal, permission and IO errors fail closed with path,
phase and exception details while preserving the first cause. Existing storage,
resource, deadline, model/image/Euler and single-attempt limits stay fixed.
All 158 focused engineering checks pass and the 703-file preservation audit is
unchanged. Native Job cleanup is verified with harmless processes and injected
resource counters; real model performance remains unvalidated. No repaired-code
run, alternate claim or changed-directory continuation is authorized.

## D039 — Authorize one independent comparison after storage repair (2026-10-25)

The user separately authorizes the minimum fixed-v2 identity change, necessary
checks, and then one guarded run on its exact clean committed source without
another confirmation. Preserve both spent v1 claims and all failed/partial
evidence. Execution identity (external historical reference: `GPC_INPUT_COMPATIBILITY_ATTEMPT_V2.md`) fixes the
new attempt, registry, output, ledger and authorization; no generic retry or
resume mechanism is introduced. Scientific proposal/T1, saved inputs, cases,
noises, sampler, decoder, gates and every prior budget/resource cap stay fixed.
Measure M completely afresh, load at most once only after its pass, and enter L
only after T's pass. Any stop spends the new account; no warm-up, extension,
installation/download, physics, policy/model training or later experiment is
authorized. Report actual gates, denominators, effects, costs and cleanup with
limited conclusions, including any operational failure.

The actual v2 attempt (external historical reference: `GPC_INPUT_COMPATIBILITY_RUN_V2.md`) on clean `706b45b`
stopped after 22.485s in M on the unchanged 0.5s process storage-lock deadline.
It saved 24 L calibration rows and loaded no model; gates remain null and T/L
are not attempted. Native Job cleanup completed. Both new claims remain spent,
no source repair/retry followed, and no generated-shape or adaptation inference
is supported. Original v1 evidence remains intact.

## D040 — Keep quota transactions short and verify with fake data (2026-10-25)

After the v2 stop, the user authorizes accounting simplification and a complete
constructed-data flow, with no further formal experiment. Keep both attempts'
spent claims and failed evidence. Scientific inputs, identity, gates, call counts,
resource/storage caps and the original deadlines remain fixed. Use only temporary
fake ledgers, synthetic images/measurements/backend and harmless native workers;
do not load real weights or run the saved-image decoder, physics or training.

The incremental design (external historical reference: `GPC_INPUT_COMPATIBILITY_INCREMENTAL_STORAGE.md`) reserves
maximum bytes with a short shared counter transaction, then performs publication,
hashing and durable IO outside that mutex. Remove hot-path tree scans, historical
prefix rehashing and full call/output-history copies. Keep one new-file hash,
incremental append digests and explicit M/T/final integrity audits. A single writer
or fixed per-process quotas could remove the mutex; shared admission retains it.
Critical pre-call journal fsync remains; normal monitor/stdout writes flush at
durable/final boundaries. Compact output events omit redundant absolute paths,
while final readers strictly reconstruct their bound destinations. No cap,
heartbeat/polling cadence, stdout stress or timeout is increased to fit the test.

The full constructed flow completed all 2268 raw files, exports and native cleanup;
183 distinct checks cover accounting and a subsequent terminal-record fence.
The finalization-only fence had a separate 53-test guardian pass; the full native
flow was not repeated afterward. All 806 protected files and four formal claims
remain unchanged. Real GPC throughput/compatibility is still unvalidated and no
new formal run is authorized.

## D041 — Authorize one comparison after incremental verification (2026-10-25)

The user explicitly authorizes the original formal comparison once, with every
budget and gate unchanged, stop on any error and no automatic retry. Introduce
only the fixed v3 account (external historical reference: `GPC_INPUT_COMPATIBILITY_ATTEMPT_V3.md`), necessary checks
and exact clean committed source; preserve all spent v1/v2 claims, partial
results and engineering evidence. Run M afresh, load at most one frozen model
only after M passes, and enter L only after T passes. Keep 2268-image / 6804-step /
1800-second and all resource/storage limits, paired RNG, missing denominators,
measurement source/config and original scientific proposal fixed. A startup or
runtime stop spends this new account too. No repair-and-continue, alternate
directory, warm-up, extra model probe, extension, new physical operation,
installation/download or policy/model training follows automatically. Report
actual outcomes and cleanup, including a failed or incomplete attempt.

The actual v3 run (external historical reference: `GPC_INPUT_COMPATIBILITY_RUN_V3.md`) on clean `ff1c5b4` completed
M/T/L in 1760.913s including audit, with 2268 predictions, 6804 Euler calls and
one model load. M/T gates passed; L-H2/H4 shape screens failed despite complete
sampling. Native resource checks, final audit and Job cleanup passed, with no
error, repair or retry. Both new claims are spent; all 895 protected files and
four old claims remain unchanged. T1 addresses the early appearance mismatch on
these saved contexts, while L's rollout degradation remains unresolved. No new
physical interaction, training or later experiment is authorized by this result.

## D042 — Diagnose L offline before any further model execution (2026-10-25)

The user authorizes switching from clean `c8f16ed` to the already created branch
`codex/wm-l-dynamics-repair` and diagnosing existing L contact, rotation and
multi-step failures, including CPU RGB/NPY decoding, offline measurements,
synthetic tests and necessary diagnostic changes. No new simulation/reset/control,
model load/forward/prediction, policy call, training or download is authorized.
Preserve all raw outputs, weights, ledgers and six spent claims; new derived
evidence lives separately. Do not expand guardian/storage engineering or relax
scientific thresholds. Any changed measurement method would require old-method
comparisons and an honestly separated calibration/validation basis.

The diagnosis (external historical reference: `WM_L_DYNAMICS_DIAGNOSIS.md`) separates recorded-window consistency,
visible generation defects, feedback accumulation and unresolved measurement
limits. It repairs lost pusher-fit metadata and adds reproducible statistics,
fixed-case visual evidence, residual localization and own/common ranking subsets.
No fit/acceptance method changes: all original scores remain. A bounded
weight/data-adaptation or architecture proposal is not execution authority;
new independent contact/rotation/near-success evidence is needed for validation.

The offline follow-up passed 33 focused checks and all 1692 saved L conditioning
identities, with 14 fixed cases/56 revised visual pages. First rejection aligns
with H2 contact in eight of nine branches and follows in the ninth; noncontact
failures also remain. B-only scoring coverage is reported separately from the
empty A/B L intersection and does not establish action advantage. Closing audit
preserves all 7309 old files and six claims. New real model, physical, policy and
optimizer operations remain zero; proposed later work was not executed.

## D043 — Prepare one supervised GPC candidate without real execution (2026-10-25)

The user-authorized engineering continues from `7a807b3`; preserve its offline
diagnosis and all frozen run accounts. Implement the pinned phase-two t=5
supervised residual objective, not a custom CNN or inference-sampler loss.
Use four real histories/four aligned controls/one target with T1 once on every
real frame and unchanged action normalization. Train all upstream inner-model
parameters under fixed AdamW settings, T:L 1:1, seed 5478 and explicit gradient
clipping 1.0. Compare only original frozen and supervised-adapted weights.

The fixed preparation (external historical reference: `GPC_SUPERVISED_ADAPTATION.md`) selects only completed
update 500 within 900 seconds including loading and selection; failure stops
incomplete without prefix selection or retry. First feasibility update is
budgeted update 1. A future total 3600-second candidate includes exactly three
serial model loads, 32 loss batches and at most 3996 forecasts/11988 Euler calls
on 63 regression and 48 new independent branches. These are proposed ceilings,
not measured training feasibility. Preserve native single-model memory floors,
retained Job ownership and original spent highwater across any recovery.
Only a coherent atomic model/optimizer/RNG/count/config boundary may restore;
evaluation-only weight loading cannot rewind later counts or RNG.

No eligible local official T/L source catalogs/episodes are available; old CNN
validation and reviewed GPC/DPPO scenes cannot become independent data. Freeze
whole trajectories, scene hashes, prior uses and explicit train/development/
independent/held roles; never inspect held model-test payloads for this work.
The data/collection audit (external historical reference: `GPC_ADAPTATION_DATA.md`) declares missing coverage
and bounded future collection with unresolved collector exclusion/held-readback
prerequisites. Do not collect automatically. Default CLI is read-only and missing
assets refuse execute before creating a claim or loading a model. Actual GPC,
policy, sampling, physics, training and native preflight are not authorized by
this preparation; only synthetic data, tiny CPU tensors and fake guard tests run.
Engineering screens retain original thresholds and do not authorize imagined
DPPO. No PR merge, push, environment change or old account reset follows.

## D044 — Restore frozen coverage gate and predeclare T retention (2026-10-26)

Repair the acceptance layer of `92c44cf` without a new experiment. Absolute
coverage-error P90<=.05 is the frozen gate; the preparation's MAE substitution
was an unintended relaxation. Retain MAE solely as a diagnostic. Original
experimental outputs and all existing evidence stay unchanged.

Predeclare conservative paired nondegradation on the full physical-case/noise/
mode/horizon grid, independently by shape and regression/independent stratum.
Validity drop and newly invalid rows are forbidden. Common-valid per-row error
increase limits are zero with numerical tolerances 1e-6 pixels/degrees and 1e-9
coverage fraction, full label/noise/completion checks, all anchors represented
and 90% object/coverage or 95% pusher common support. Disclose loss transitions,
common counts and both paired-delta P90 and difference of arm P90s. Ranking
requires the same complete anchors and physical separable pairs; newly lost
eligibility or correct pairs fail, and regret cannot increase beyond 1e-9. No separable
pairs or insufficient support is `insufficient`, never a pass. Overall T also
requires the original absolute AR screens; TF is diagnostic and cannot rescue AR.
Publish L deltas under the same rules; certify no termination precision.

Only pure synthetic/CPU/fake checks are authorized for this repair. Training
loss, architecture, candidates, sampler, budgets and resource limits are unchanged.
Do not implement collection, run real models/physics or native preflight, inspect
held payloads, reset spent accounts, install/download, push or merge PR #6/#7.

## D045 — Separate complete coverage from ranking information (2026-10-26)

Correct the ranking aggregation of `ab03ba3` without a new experiment. Every
predeclared anchor/candidate/seed/AR-TF horizon still requires complete matched
records and identities. The old requirement that every anchor also have a
physical candidate difference>=.05 incorrectly blocked complete no-contact
controls. Replace it with full-grid completeness plus coverage of every truly
informative anchor/pair, determined only by declared physical outcomes.

Complete zero-separation anchors are uninformative, never ranking successes or
ranking failures; keep their IDs/counts and total planned denominator. They still
participate in all state/coverage/T retention checks. Unknown truth or missing
records is insufficient. All informative anchors must be scoreable on the same
pairs, with the existing one-anchor/one-pair minimum per shape/stratum; otherwise
do not pass. Whole-group no-information remains insufficient. Existing zero
degradation, numerical tolerances, absolute gates and uncertified termination
remain unchanged. Only numeric/CPU/fake tests run; preserve prior evidence and
accounts. Do not change budgets/models/samplers/collection/guarding or run real
models, native preflight or physical operations; do not install, push or merge.

## D046 — Freeze train-scene exposure before a separate B2 attempt (2026-10-26)

After review of `fb1bb20`, replace the first-250-per-shape training prefix with
one frozen train-only 500-entry list: T/L alternate, all 12 scenes per shape get
20 or 21 windows, rotation >5-degree targets come first, then fixed contact/other
fill. This intentionally changes scene/target weighting using privileged train
labels. It never uses validation, held payloads or benchmark outcomes to select
windows. Bind list/order/rule identities and spent cursor to the existing
checkpoint/account contract. Model, T1, objective, optimizer, 500-update budget,
checkpoint500 and evaluation gates remain fixed; no training is authorized.

Keep A and failed B immutable. One separately authorized B2 account may collect
the unchanged 12-anchor/48-branch plan with the original B caps, native resource
checks and precharged reset/restore/control/render calls. No A/v3 rerun, balance
reuse, unaccounted smoke reset, automatic retry, replacement seed or added sample
is permitted. Freeze the training list before B2 physics and return its actual
coverage, limitations and resource observations for main review before any GPC
load or adaptation. Collection resource checks do not validate training resources.

## D047 — Separate SDK startup evidence from model training (2026-10-26)

After the preserved pilot01 stop before any model ledger, add durable SDK
before/after stages and the first heartbeat refresh/publication exception with
its exact step, time, errno/Winerror and traceback. Keep main-thread progress,
heartbeat age and allocator sample freshness distinct; unavailable CUDA readings
are null. Retain partial failed status publications. The original five-second
heartbeat rule and resource ceilings do not change.

After harmless tests and a clean source/plan freeze, authorize one separately
claimed SDK-only diagnostic: 60 seconds including cleanup, 8 MiB including
temporary output, one worker, existing Job and resource limits. It may import
Torch, set the original two CPU threads, query/initialize CUDA and read allocator
statistics. It cannot construct/load models, sample, train or run physics, and
must not mark training complete. First failure ends the attempt without retry.
Return evidence and remaining uncertainty for main review before any second real
diagnostic or model attempt; do not infer a GIL/DLL cause from a last stage alone.

## D048 — Give pilot02 a separate fixed identity (2026-10-26)

After main review of the single SDK result on `846344f`, permit one new bounded
GPC supervised candidate using the original scientific protocol. Preserve the
old config bytes and spent pilot01; a new config changes only candidate ID and
output directory. Bind the explicit pilot02 authority/config/output at parent
and worker entry, without a general candidate registry or automatic retry.

Keep initialization, T1, frozen500list, loss/optimizer, checkpoint500 selection,
900s training/3600s total clocks, compute/output/resource limits and five-second
heartbeat unchanged. Startup diagnostics add evidence, not training feasibility.
One real attempt stops at first failure and returns for main review, with no new
collection, policy/DPPO work, fallback or second real execution. Existing sparse
independent coverage and uncertified termination remain explicit limitations.

## D049 — One fixed pilot03 after reference-lifetime repair (2026-10-26)

Main review of `07616f9` authorizes one new explicitly bound pilot03 identity.
Relative to pilot02, only the accepted `del grads` implementation change may
affect training; identity/output changes separate costs and cannot refund old
reservations. Keep source weights, data/order, loss/optimizer,500 updates,
checkpoint500, evaluation and every resource/time/output/liveness limit fixed.
No SDK smoke or free update is permitted. First failure stops the new attempt;
return actual ledger/results and preservation for review without retry, fallback,
new collection or policy work. The repair has not yet established CUDA saving.

## D050 — Preserve status failure evidence without claiming a repair (2026-10-26)

After pilot03, restrict diagnosis to bounded harmless Windows file cases. Keep
ReplaceFileW: the one MoveFileExW flags9 comparison failed with WinError5 under
production share-delete readers and is rejected. Original raw-reader name gaps
do not reproduce writer1175; existing snapshot-age handling stays unchanged.
The two native runs are spent, with no third run or real model follow-up.

Tighten shared source type/link/reparse/size and same-directory checks. Shared
WM atomic publication makes one attempt, retaining first error and failed bytes
for orphan refusal. GPC already retained its failed bytes. These auxiliary fixes
do not establish reliable publication or explain1175. Preserve all old accounts,
inputs and the selected checkpoint500. A separately reviewed eval-only mode is
only a plan: no implementation, new account, load or training is authorized here.

## D051 — Narrow append delivery for checkpoint500 evaluation (2026-10-26)

For the new fixed GPC eval-only identity, replace no status path: use one bounded
append file and incremental reads of complete validated records. Keep first-error
slots, native Job/resources and independent five-second ages. Partial tails cannot
refresh age or pass terminal completion. Reserve16MiB inside the original64MiB
telemetry cap, reducing monitor admission. Old WM/training protocols are not migrated.

Reuse source checkpoint500 through a separately bound two-role evaluation account
with zero new updates/examples. Reject training, selection, resume and optimizer/
training-RNG restoration. Keep the full original paired grid and support gates.
Engineering and one harmless file run prove no neural loading/resource or quality
claim. Real execution needs new main review and exact source/config authority,
without retraining or refunding historical spent333 forecasts.

## D052 — Deliver a manual prediction/physics comparison before further research (2026-10-26)

The user explicitly authorized building and verifying a manual T/L demonstration
while training and automatic experiments remain stopped. Reuse the existing
official simulator, native GPC sampler, pinned model-only loaders, fixed T1 and
measurement. Add one local window and a double-click launcher: select a target,
preview without stepping physics, then execute that same request and compare.
Expose initial weights and the known degraded checkpoint honestly. Allow local
screenshots and recording; manual observations are not an independent benchmark
or a new training dataset. Do not add a heartbeat, service or experiment framework.
The hands-on interface is a required delivery, not something to defer until the
model passes scientific acceptance. Existing unfinished conditioning work stays
stopped and is outside this implementation.
