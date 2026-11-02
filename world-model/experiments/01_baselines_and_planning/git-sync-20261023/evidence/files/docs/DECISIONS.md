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

## D024 — Add a frozen-policy simulator-ranking control (2026-10-20)

Implement an optional inference-only controller around the official T checkpoint,
without new-shape demonstrations or model updates. It samples several action
chunks from the same real history and ranks them using the target shape's known
physical simulator. Keep this privileged control distinct from learned-world-model
adaptation, policy fine-tuning and image-only deployment. Do not change the
canonical original-T baseline or the default eight-action shape runner.

Reconstruct each branch from the accepted episode-initial snapshot plus all real
executed controls. Current-state cold restore lacks Chipmunk contact caches and
must not be presented as an exact contact continuation. Check reconstructed
physical/controller state and fail on mismatch; provide a real mid-contact D test.
Skip branch image rendering only, never physics. Forecast completed policy
prefixes followed by physical hold over equal, remaining-task-clipped budgets.
Score terminal coverage with pose/motion terms so pre-overlap progress is visible;
do not select transient peak overlap and then execute its worsening suffix.
Preserve absorbing native success and count real, replay and forecast work separately.

Compare the unchanged eight-action baseline, single-action replanning without
ranking, and single-action simulator ranking. Include independent candidate
noise/compute, hold priors, deadlines and accepted scene distribution in reports.
Source-only tests do not establish simulator accuracy, checkpoint compatibility,
real-time operation or improved D performance. Validate on the existing Linux/CUDA
profile before running comparisons; no dependency/profile changes or new assets
were needed for this source milestone. Commands and limitations are in
SIMULATOR_RERANKING.md (external historical reference: `SIMULATOR_RERANKING.md`).

## D025 — Validate the frozen simulator control on Windows (2026-10-21)

Extend the shared uv lock to native Windows AMD64 as well as Linux x86_64,
retaining Python 3.12.3, Torch 2.5.1/cu124 and all existing direct pins. Keep
Robomimic's optional MuJoCo EGL probe on Linux only: its build requires Linux
rendering headers, while the Windows PushT/CNN path does not use those renderers.
Document the resulting expected `uv pip check` metadata warning rather than
claiming all dependencies are compatible. Linux remains the training reference.

Remove official-checkpoint file/content hash acceptance gates at the user's
request. Identical tensor values can have different PyTorch ZIP serialization
bytes on different platforms. Keep published configuration, complete strict
state loading, frozen evaluation and direct tensor equality during conversion
and inference checks. Historical manifest hashes remain provenance; pinned
official source checks remain unchanged.

Use initial-snapshot plus complete executed-action replay for actual D contact
verification. Reject final forecast controls that finish after the deadline,
preserve their work counts, and explicitly report requested shapes never started
before a wall-time stop. Do not interpret a 64-control integration comparison as
300-control task performance: 10 Hz controls make those 6.4 and 30 seconds of
simulated time respectively. Report a full-budget baseline separately when
comparing with the published alphabet video. No training, D demonstrations,
learned dynamics or image-only deployment claim is part of this milestone.

## D026 — Qualify a simulator teacher before student learning (2026-10-21)

Evaluate policy-free D action search before creating student training targets.
Use known D geometry, complete physical state and the original simulator, with
geometric approach/contact priors disclosed. This is privileged control, not
visual deployment, learned dynamics or cross-shape learning. Keep the original
T checkpoint unchanged and do not load it for these searches.

Freeze four development-scene requests and two search seeds before inspecting
outcomes. Keep 300 actual controls, 600 wall seconds per run, an 80-control total
forecast cap and the native coverage-greater-than-0.95 success rule. An isolated
contact-search revision uses the same scenes and limits; preserve both versions,
all failures and timeouts. The revised search changes several mechanisms together
and is not an ablation of any single improvement. These explored scenes cannot
later serve as untouched student evaluation cases.

Validate selected predictions against actual execution, then independently replay
every control from the initial snapshot with complete history. Export videos with
a fixed pause at each actual control and distinguish playback timing from measured
planning latency. Count forecast, history reconstruction and independent audit work
separately. Protocol correctness alone does not qualify a teacher: require actual
task success and distinct successful contact routes before claiming useful route
diversity. The bounded results and teacher-readiness decision are recorded in
`docs/VALIDATION.md`; do not relabel failed routes as successful demonstrations.

## D027 — Search contact sequences before training students (2026-10-22)

Add an independent Windows/Pymunk two-contact search backend, leaving the old
search and frozen-policy reranker intact. Jointly refine both pushes and rebuild
the second contact from the first push's attained state. Rank completed routes
using fixed material-point pose error with non-vanishing orientation influence;
keep contact/rotation diversity during prefix pruning. These are geometric
priors, not learned skills or an implementation of convex-relaxation guarantees.

Reuse exact remaining commands/predictions between global searches. Reconstruct
every sibling through complete initial + real + hypothetical action history;
never reuse cold endpoint snapshots as physical continuations. Record all
rejected and interrupted work. Keep native success, 80 total forecast controls,
40 attempts per search, eight searches, 300 actual episode controls and explicit
wall-time caps. Imported histories and independent replay audits are separate.

First validate three previously inspected partial-overlap states at control 160.
Keep failed prototypes and the same starts/caps after ordinary correctness fixes.
One corrected recovery success does not establish reset completion, successful
route diversity, student readiness or image-only generalization. Student training
remains unstarted. Detailed evidence is in `docs/VALIDATION.md`.

## D028 — Preserve negative same-budget precision results (2026-10-22)

Refine the two-contact backend in the agreed order: adaptive remaining-budget
settling, slow sustained presses with independent approach speed, opposite-face
translation-compensation proposals, then explicit access/control-cost ranking.
Keep the original physical dynamics, native success threshold, complete-history
replay and all previous search/control caps. Precision contacts are geometric
priors with sampled target sequences, not learned experts or force control.

Repeat only the two failed prior control-160 starts with search seed 42, freeze
the implementation before outcomes, and retain both failures. Smaller angular
errors alone are not successful joint alignment. One case loses coverage and
both retain excessive position error; the other stops two controls early after
the unchanged eight-search cap. Record that termination explicitly instead of
extending search or relabeling it as control-budget completion. Do not append
post-budget or unreported recovery actions to videos. The teacher remains
unqualified for student learning. No student training, extra seeds, third-stage
expansion or additional performance revision was performed to replace either outcome.

## D029 — Separate joint pose feasibility from scalar progress (2026-10-22)

In precision control, independently constrain terminal center and angle errors
with ratcheted tolerances and a bounded intermediate corridor. Among feasible
plans, rank the worst normalized pose component rather than allowing lower RMS
to buy unbounded center drift. Keep native success and the original physical
safety rules authoritative. These privileged geometric planning constraints
are not learned controllers or new task-success thresholds.

Generate compensation from the actually simulated first-stage displacement,
preserve distinct position/angle/joint/compensation refinements, and search press
dwell in the original attempt budget. Up to four spare slots may extend two
contact stages with a third; history reconstruction must include both earlier
stages. Keep 40 attempts, eight global searches, 80 forecast controls, 300 total
episode controls, 600 wall seconds and search seed 42 for the two fixed old
control-160 states.

Preserve the first guarded implementation and both correctness reruns. Fix the
scoring/execution endpoint mismatch by executing and evaluating the same bounded
settling prefix; verify actual terminal constraints at its boundary. Disable the
coarse RMS stall rule during a still-active precision macro so it cannot abort
a joint-feasible plan for improving by less than its old scalar threshold.
Retain every negative result, without seed selection, increased caps or student
training. Task success and teacher readiness still require actual completion;
passing protocol/constraint checks alone is insufficient. See `docs/VALIDATION.md`.
