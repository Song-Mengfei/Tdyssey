# Decisions

## D062 — Approve O/B attribution and a fixed region-weighted C/D pair (2026-11-01)

The human approves D061's exact saved plan. First complete the bounded768-frame
same-case real-history diagnosis on saved O/B, with no optimizer updates. A
demonstrated action/interface/timing fault stops training for a repair proposal;
model prediction error alone is the experiment's subject. C/D restore the same
full epoch2 and each traverse the unchanged7568segments once,1892batches. Both
use the approved0.50/0.25/0.25 full/region/detail residual loss; C uses real history,
D assigns two real/two generated segments per batch. Keep original physics,
data splits, native sampler and all other recipe settings. Preserve A/B evidence.
The existing owner reports diagnosis, throughput and final results proactively,
then stops for human approval. No parent polling or automatic additional work.
Exact authorization/plan binding: artifacts/region-weighted-cd-20261101.

## D061 — Analyze regressions and review the next image-only comparison (2026-11-01)

The human requests explanation and a next-step proposal, explicitly withholding
further training. Preserve original physics/data and the existing A/B evidence.
Saved-only analysis corrects the displayed O failure from mere underrotation to
opposite visible-axis change, with symmetry/deformation/calibration limits kept.
The proposal supplements same-case real-history O/B inference, then compares
region-weighted native residual training on real versus partly generated history.
Masks are target-side image-derived loss weights, not model inputs or output
repairs. No new motion head, data, physical settings or automatic horizon increase.
See artifacts/generated-history-next-step-review-20261101/ANALYSIS_AND_PLAN.txt.
This is not execution authorization: no new inference/training or owner dispatch.

## D060 — Approve one background generated-history A/B comparison (2026-11-01)

Direct human approval releases D059's exact plan, bound by hash in
artifacts/generated-history-ab-20261101/USER_AUTHORIZATION.json. Both arms restore
the same full epoch2 state and traverse the same 7568 four-target segments once;
A uses real history, B detached online samples from the native sampler. Retain
the original physics, data splits and native objective/settings. The first 32
actual updates of each arm are included in its 1892 batches and must be reported
with measured throughput and remaining-time estimates before completion. No new
physical sampling or server work; preserve the shared 10GB cap with atomic reserve.
The existing owner runs in background, actively reports throughput and final O/C/B
paired results, then stops for human approval. The parent does not poll or assign
another experiment automatically. A dispatch receipt is not startup evidence.

## D059 — Review detached four-step generated-history training (2026-11-01)

The human requests a concrete plan on existing data with unchanged physics.
Propose one matched A/B pass over non-overlapping four-target segments, both
from the preserved epoch2 state. Only the candidate replaces later historical
RGB with pure-noise sampler outputs; future truth stays outside that generation
path. Retain the native residual objective and detach generated context to bound
memory. Match targets/updates, disclose extra inference cost and short-horizon
limitations. Improve and measure engineering throughput without silently changing
scientific settings. See artifacts/generated-history-four-step-review-20261101.
This records a proposal only; no new sampling, model calls, training or dispatch.

## D058 — Approve two more frozen-data epochs and O/C diagnostics (2026-10-31)

The human explicitly approved D057's presented execution plan. Restore AR1's
completed epoch1 full state, then traverse the same30935-window pool once per
epoch for two additional epochs, preserving the model/objective/optimizer recipe.
Evaluate fixed real-history and raw recursive moving-sequence diagnostics and
the bounded paired sampler check. Preserve old artifacts and the10GBshared cap.
No new sampling, fourth epoch, scientific configuration change or server work.
The owner runs in background and proactively reports startup/completion; parent
presents results for human review without automatic continuation. Authorization
and exact plan binding are in artifacts/oc-continuation-20261031.

## D057 — Diagnose O/C using existing data before further collection (2026-10-31)

The user requests analysis and a next-step proposal, not execution. Current
evidence does not distinguish insufficient new-domain training from recursive
history errors or objective/optimization limitations. Preserve the original
physics and image-only model. Proposed work reuses the frozen data for two more
epochs and paired real-history/recursive diagnostics; any new loss or architecture
remains a separately reviewed change. No new sampling, inference, training or
owner dispatch occurred during this saved-only review. See the O/C review in
artifacts/oc-training-review-20261031/ANALYSIS_AND_NEXT_STEP.txt.

## D056 — Freeze one new-data epoch and service-owned background execution (2026-10-31)

Implement D055 as one finite pipeline:1536initial independent families, at most
2000attempts/102000controls/1020000regular substeps including any same-strategy
supplement, then clean-prefix selection and one full unique-sample epoch. Keep
the old complete7499weights/Adam/scaler/RNG as explicit new-data initialization,
not same-distribution continuation. Physical targets remain0..512 unchanged;
the old model's0..511normalization stays unchanged, with incompatible windows
excluded and counted. Recovered suspicious events are not admitted automatically.

Use one shared10GBcap with256MiB atomic reserve plus prospective-write and bounded
concurrency allowances. A one-shot Windows Task Scheduler supervisor owns the
background process. Preserve actual Job membership; any Job membership alone
does not establish Codex ownership. The pre-start lifecycle repair consumed no
physical/model calls and retained its previous frozen identity. Scientific
sources/configuration remain frozen under the final runtime hash. The owner
may end its active turn after actual startup proof and callback; a quiet owner
heartbeat handles actual completion/failure reporting while app tools are available.

## D055 — Accelerate useful-data sampling and first training result (2026-10-31)

The latest human direction permits collecting more and selecting useful data
for training, with a 10G shared experiment cap. Use 10,000,000,000 bytes as a
conservative implementation. Replace repeated small acceptance gates with one
bounded collection/selection/local-epoch owner task. The parent's <=2000-attempt
budget is an execution allocation, not a number quoted from the human. Keep
original physics and the existing image-only model/training recipe; do not
introduce unapproved scientific configuration changes. Report actual before/
after H1/H4/H8/H16 results and failures, then await the next human decision.

## D054 — Review contact outcomes and purposeful task blocks (2026-10-31)

The user requests a concrete revision based on the four saved continuation
cases, with original physics unchanged and execution gated by human approval.
T supports observed recovery during continuous contact; O remains unresolved;
L separation does not establish healthy recovery; C has an actual outer-wall
crossing. Do not infer a universal replacement depth threshold or retroactively
rewrite old validity labels. The proposed observer, 19/35/51-control task
organization, exposure indices and <=60-attempt acceptance are review-only in
artifacts/contact-event-revision-review-20261031/PROPOSAL.txt. No implementation,
owner dispatch, collection, training or monitor activation is authorized now.

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

## D028 — Cancel structured prediction; use original GPC RGB (2026-10-28)

The user cancelled the explicit motion/SE2 head and template-based prediction
pipeline and authorized deletion of its code and experimental artifacts. Preserve
original GPC sources, released weights, native RGB entry points, existing 26-letter
geometry, base physics and earlier experiments.

The next study maps RGB history and actions directly to RGB using original GPC,
without explicit pose prediction, template transport, output shape repair or
letter-ID lookup. Initialize only from the original released weights, not any
all-letter or T/L-adapted checkpoint. Split by whole letters for unseen-letter
tests, and by independent trajectories for seen-letter validation. T was already
seen during original pretraining and is ineligible for a fully unseen claim.
Keep the existing 26 shapes; letter allocation and training settings remain pending.
This cleanup starts no collection, training or server jobs. Previous structured
model memory measurements must not be reused for full GPC resource claims.

## D029 — Original GPC and whole-letter holdout (2026-10-28)

Use the complete released 96px GPC with its original H1 residual objective and
sampler; optimize all original weights from the original release only. No explicit
motion head, template transport, repaired RGB output, letter-ID lookup or geometry
loss is allowed. Train on ABCEFHIJLMOPSTUVXY; hold DGKNQRWZ for frozen final testing.
Seen validation/test use independent complete trajectories. T is not unseen.

Use real contact-contour coverage to guide bounded sampling. Version and disclose
the finite-force dynamic pusher controller used to prevent kinematic wall crushing;
keep all RGB interfaces on the same action/control protocol. Local collection is
capped at34,560 controls and has been spent, including probes; local training is
capped at5,000 attempted optimizer updates or90min including engineering/tiny fits.
No optimizer update has started. Preserve the compact data and incomplete sources
while paused for the user's restart. Server execution remains user-only.

## D030 — Honest RGB measurement and I/O-only recovery (2026-10-29)

Visible-object centroids can move when the blue pusher changes occlusion. Keep
raw RGB/visible-mask metrics, and measure physical pose only through a separate,
renderer-calibrated offline instrument reusing identical raw sampler outputs.
Fitted geometry never enters prediction or training. Unmeasurable outcomes stay
in denominators; require gains above finite measurement uncertainty before calling
physical dynamics verified. Freeze both evaluation instruments and calibrations
before either final test split. No local unseen-letter prediction is permitted.

Transient Windows atomic-file replacement failures receive bounded retries.
Recovery across this change is allowed only when all trainer AST outside the I/O
writer and the remaining13 source hashes are identical, and model/optimizer/scaler/
scheduler/RNG/order/progress/config fingerprints do not change. Preserve original
checkpoints/failure evidence and charge all replayed and interrupted attempts;
logical checkpoint progress is not aggregate cost. No budget reset is allowed.

## D031 — Retain the negative RGB pilot and stop scale-up (2026-10-29)

The bounded run ended at logical1668, with best validation checkpoint1500 and
aggregate1,899 charged attempts/5,352.516 seconds. Paired H1/H4 seen validation
failed physical motion improvement in contact, rotation and boundary groups;
shape failures and unavailable predictions remain explicit. Preserve this result,
complete checkpoints and the verified from-scratch delivery. Do not enlarge the
local budget or start server training to bypass the failed gate.

Server qualification requires zero assessed shape failures and at least one
assessable shape at both required horizons, in addition to the motion, physical,
data and engineering gates. Tightening this report-only condition after the
negative result does not alter model inference, calibration, measured rows or the
already-failed outcome. Any new collection/training protocol needs a separately
agreed experiment; final unseen predictions remain sealed.

## D032 — Parent-supervised resampling with mandatory stage callbacks (2026-10-29)

The human user authorized a new correction study in the parent chat. Its direct
userMessage records and chat IDs are persisted in
`artifacts/native-rgb-supervised-20261029/SUPERVISION.md` and stage-state.json.
Read that memo on every continuation. Only stage1 is released: sampling/statistics
checks followed by at most54 six-letter/nine-location real trajectories,64 controls
each, with all probe/failure attempts capped at4,000 controls. No gradient updates,
model predictions or server execution belong to this stage. Preserve all old
evidence and keep old plus new native persistent artifacts below3GiB.

Motion strata use single0.1s area-centroid translation and unwrapped angular
change: moving>=2 world pixels or>=1 degree; near-static<=0.5pixels and<=0.005rad;
all other transitions are intermediate. Inspect continuous motion distributions,
contact cross-tabs and true wall gaps as well as actual contact-arc coverage.
The unchanged finite-force-pd-v1 physics and native GPC interface remain fixed.

At each stage end or blocker, persist evidence/state, actively message the recorded
parent chat, then stop and wait for a new stage instruction. A final reply here
alone is insufficient. Later data collection, microfit, segmented training, final
evaluation and packaging are proposals awaiting separate parent releases.

The stage2B release on2026-10-29 authorizes only the remaining234 formalTRAIN
trajectories/14,976 attempted controls. The parent's fixed9+4 spatial quotas and
cyclic gap tiers are frozen before execution. Only initial target ranking adds
an independent started-trajectory attempt count after the existing motion-room
tier; the continuous stage1B controller and physical protocol remain frozen.
Actual hits and planned attempts inherit formal2A evidence separately, while the
historicalTRAIN prior stays distinct. Formal2A originals and their failed M
motion target remain preserved. A reference-only mergedTRAIN manifest retains
both selector versions and original file hashes. Validation, training, model
calls, server execution and packaging still require later parent releases.

The stage2C release authorizes162 independent seen-letter validation trajectories
and CPU reference-data/index preparation. Validation hit/attempt counters start
at zero and never update the immutable396-trajectory TRAIN. The fixed2B selector
and1B continuous controller remain frozen. All evaluation-index rules/seed are
captured before collection: one uniformly hash-ranked H4-legal anchor per scene,
plus rare-condition slots with explicit missing support and a36-window development
subset. H1 conditions and H4 endpoint/path labels stay separate; exact duplicate
windows are inferred once in future work and temporal overlap is disclosed.
NativeRGBDataset may read a strictly hash/audit-bound, zero-copy joint reference
view while preserving original RGB/action timing, normalization and uint8 LRU.
Validation40/45 targets diagnose distribution and do not authorize resampling.
No model output or optimizer work is allowed until a later explicit release.

The stage3B release corrects only the finite native-RGB range abort: retain raw
out-of-range values and failure status, continue the bounded experiment, and
keep nonfinite/structural/OOM stops. No clipping, RGB remapping, calibration,
score or learning mathematics change is allowed. The exact original18/36
windows and batch seeds are reused from a fresh published initialization. The
old28.344-second pretraining abort is preserved and charged;3+3B jointly stay
within200 attempts/600 model-work seconds. One execution completed129 updates
and the full54+54 H1 comparisons. Actual gradients/parameter changes and color
adaptation are established, while the frozen20%/12-of18 motion-fit diagnostic
is not met. A lower denoising loss does not override that result. Later training
requires a new parent release after the mandatory result callback.

The new stage3C parent release permits one diagnostic continuation on the same
18 windows, from complete update129 to at most cumulative250. Its independent
121-attempt/600-model-second cap does not rewrite or refund the old3+3B cap or
444.625-second use. Original baseline and129 predictions are referenced without
new inference. Preserve the original strict resume validator and trainer bytes;
a separate narrow adapter must validate the exact parent payload and allowed
stage/path/budget/source metadata changes, restore every optimizer/RNG/order
state, and save the true current identity. Training mathematics, samples,
normalization, scheduler horizon and acceptance thresholds remain frozen.

The one3C continuation reached250 complete updates. TRAIN meets the unchanged
18-window diagnostic at20.3848% improvement/15 wins, while DEV still loses to
hold and shape defects/unknowns remain. Treat this as evidence that further
optimization of the same small sample improves its raw fit, not as formal
training or generalization acceptance. The423.859-second stage is separately
charged, bringing new-round use to868.484 seconds/250 attempts. Preserve the
old444.625 record and stop after the mandatory result callback.

## D033 — Motion amplitude and real videos must pass before training resumes (2026-10-29)

The user explicitly suspended training and reprioritized investigation and
sampling redesign. Save the incomplete stage4A interfaces without continuing
the refactor. R1 is read-only existing-data/source diagnosis plus a proposed
bounded R2 protocol; it authorizes no new physics, model or optimizer work.
The current H1 dataset has no>=1 image-pixel centroid translations, despite
60.82% moving under the original lower thresholds. Do not equate that proportion
or the small-sample microfit pass with adequate visible translation or rigid
shape prediction. Further optimization is not guaranteed to fix rounded shapes.

Any proposed damping/force/speed/timing change needs an explicit new protocol,
separate data directory and parent release. Old data/results remain separate.
R2 must submit full actual amplitude distributions, true contact and dynamic
wall/corner-gap coverage, unchanged2px substep quality, all failure denominators
and normal1x continuous96px videos with clearly labeled display enlargement.
Only after parent acceptance may formal TRAIN/independent validation be frozen
and training separately authorized. Do not consume the unused320 old controls.


## D034 — Selected physics and split-safe small motion diagnostic preparation (2026-10-29)

The user selected a 0.1s velocity half-life, exact Space.damping=0.0009765625,
using the existing selected protocol identity. All new native RGB generation
must route through the selected factory; historical protocols/data keep their
identities. Further damping comparisons and validation are stopped.

Parent review admitted R2B53 trajectories for a small world-model learning
check, while full contact/direction/boundary coverage remains incomplete.
R2C0 only prepares whole-family TRAIN41/seen-val12 references, visible-motion
sampling (H1 translation>=2 image96 pixels OR abs rotation>=5degrees), fixed
TRAIN/natural-val/motion-val evaluation indices and raw saved-RGB previews.
The 80/20 draw target changes sampling probabilities, not the saved source set.
This is a deliberately motion-enriched seen-shape diagnostic split, not an
unbiased natural test or unseen-letter generalization.

No new physics, model inference, GPU work or optimizer attempts are authorized
in R2C0. A proposed next bound of300 attempts/1200 total model-work seconds
requires a later parent instruction; the closed250/868.484s ledger and all old
physical counts remain unchanged. Data and evaluation use explicit dependencies
and serializable draw state, without private-global replacement. Full evidence
is retained locally in artifacts/native-rgb-supervised-20261029/r2c0 and actively
reported under the user's standing callback authorization before waiting.


## D035 — One bounded fresh native diagnostic and honest pretraining abort (2026-10-29)

After R2C0 acceptance,the parent authorized one fresh original-GPC full-parameter
run on frozen R2B-derived TRAIN/seen-val data:<=300 new optimizer attempts and
<=1200 all-model seconds,latest newattempt900 with>=300 tail,linked to the
existing3000/5400 ledger. Native math and frozen data/evaluation stayed intact.
The sole run loaded the original weights and saved t0,but checkpoint metadata
collided with a logger parameter before baseline or training. The stage charged
27.359 seconds and0 attempts,closed,and retained all missing denominators.
A logging-only CPU-reproduced patch is an artifact proposal,not an applied fix
or permission to restart. The actual failure,fees,t0 and all missing pages were
actively reported; no model learning or unseen-shape conclusion is available.


## D036 — Model time is statistics, not a human-set stopping budget (2026-10-29)

The human explicitly corrected the supervisor: no model-work time budget was
specified. Earlier1200/1172.641 limits,900/872.641 cutoffs,300s tails and the
global5400s time guard were supervisor trial arrangements and are revoked.
Before any R2C1R1 model operation,version record-only-v2 accounting that preserves
all historical fees,without fake large limits,shifted clocks or paused timers.
Elapsed model work is reported and cannot prevent completing this batch or its
frozen evaluation. The current<=300attempt batch is a finite diagnostic plan,not
a human lifetime training limit; no automatic unlimited continuation is implied.
Storage,device,numerical,identity and one-use protections remain. Future scope
follows phase objectives/results; do not repeatedly request extra time afterdebug.

## D037 — Seen-motion improvement does not establish static or shape fidelity (2026-10-29)

R2C1R1 completed its300-update frozen diagnostic and real checkpoint reload.
All12 motion-enriched seen-val windows beat copy-last at H1/H4, with H4 gray
IoU0.258134 to0.531987. However natural-val lost6/12 at each horizon, all4
near-static auxiliary windows regressed, and H4 shape had12 failures/24 unknown.
Keep these failures alongside average gains; zero numeric/range failures and
unknown shape outcomes cannot establish rigidity or unseen-shape transfer.
Natural/motion cohorts reuse the same12 seen trajectories with temporal overlap.

The run used record-only model-time accounting from before launch;486.375s is
a statistic, not a stopping budget. All earlier costs and old failed artifacts
remain. The full report, raw predictions, runtime scores, unchanged-source/data
audits and full galleries were actually sent to the supervising chat under the
human's standing authorization. Wait for review without automatic extra work.

## D038 — Fixed-weight native denoising-step comparison (2026-10-29)

After independent R1 acceptance of seen-motion learning but not shape/static
retention, R2C2 holds the300-update checkpoint,36windows,batch4,paired seeds,
native normalization/precision/denoiser quantization and all other sampler
settings fixed. Reuse the existing3-step outputs and scores; newly evaluate8
and16 native denoising iterations with raw four-step feedback. This is an
inference diagnostic, with no parameter updates or duplicate checkpoint.
Report all cohorts,static auxiliaries,shape failures and unknowns alongside
motion metrics. A higher iteration count is not assumed to fix rounding.

Model time remains record_only_v2, never a stopping budget. Keep all previous
costs and artifacts; new persistent files<=128MiB/shared<=3GiB including256MiB
atomic reserve. Complete actual outputs, cold checks, readable galleries and
active parent callback, then stop. No new data,physics,held-out shapes,servers
or automatic training. Later server training is executed manually by the user.


## D040 — A full6000 learning curve supports training effects,not shape convergence (2026-10-29)

Under direct human continuation authorization,C3 restored the full300 state and
completed5700 additional successful native updates without resetting data/RNG.
Fixed3 H4 motion IoU grew0.532 to0.635;TRAIN grew0.563 to0.867. Natural-val
slowed near0.728 and fixed-plan val loss rose3000→6000 while TRAIN loss fell.
Keep per-window/static/shape tradeoffs and unknown proxy semantics visible.
More training is a supported factor;neither convergence nor a unique root cause
for rounding is established. O/C requested raw fixed-window images still show
unrecovered corners/shape regions despite local improvements. Unknown does not
make visible deformation pass. Native3/16 matrix,physics,data/loss stayed fixed.

Normal AMP overflow is a separately charged skip and continues after scale
reduction,with no LR/success increment;the actual run had0skips. The old stage300,
global3000 and stream4800 scopes were explicitly superseded. Cosine stops at3000
then holds eta_min;time is only statistics. All states and periodic recovery
points are preserved within shared storage.1000/3000 and final6000 were actually
reported;wait for curve review before10000+,with no automatic extra training.

## D041 — R3A independent scenes, saved-state RGB and clean/weak history (2026-10-30)

The next authorized step replaces repeated exposure to41 trajectories/861 TRAIN
windows with768 new independent TRAIN and144 new seen-validation trajectories,
48 controls each, across the same six letters. Freeze scene families/seeds and
splits before collection. Preserve selected0.1s half-life physics, real executed
actions and compact substep/contact evidence; no dynamics lookahead or heldout
letters. Render saved true state15 on demand with the original renderer, using
bounded caches and two CPU workers. Old1325 saved frames were pixel-exact with
zero physics evolution. Count full legal windows, selected distinct windows and
actual exposure separately; half distinct obvious-motion is a collection/pool
target, not permission to repeatedly sample a few rare windows.

Implement native residual FP32 MSE with50% truly clean history(sigma_cond=None)
and50% weak log-uniform[.002,.05], no history offset. Keep original target noise
unchanged, with independently saved branch/target/weak RNG streams. Joint data
and conditioning changes are authorized; their individual causal attribution
is not established. R3A is CPU-only. The6000 full checkpoint remains the R3B
lineage; new library/cursor/noise-stream migration must be explicit, and the
next observation schedule must reflect unique data exposures rather than300
updates. Real movement/video review precedes R3B training.

The human also authorized removal of documented failed/redundant historical
binaries and rebuildable caches, performed by the planning chat. Keep original
audits immutable and record exact deleted paths/hashes and a fresh storage
baseline. Fourteen historic binaries no longer exist; do not claim otherwise.
All formal models and original data remain. R3A still has96MiB plus144MiB R3B
and256MiB atomic reserves under3GiB. Implementation evidence is in
artifacts/native-rgb-supervised-20261029/r3a/; collection results remain pending.

## D042 — A larger independent library is real data evidence, not model validation (2026-10-30)

R3A collected912 independent48-control trajectories with a pre-acquisition family
split:768TRAIN and144new seen-validation acrossT/L/O/C/X/B. Selected0.1s half-life
physics stayed fixed. All6failed attempts, partial substeps and one unstarted
reservation remain charged. Two engineering interruptions used audited versioned
ownership/I/O recovery; completed trajectories were never recollected. The
user-requested independent background process completed with both return codes0.

All44,589 numerical and94failed-prefix checks passed. Training selection keeps
11,992 distinct obvious-motion windows and11,992 distinct negatives; full legal
motion fraction34.699% is reported separately from selected50%. Epoch sampling
is without replacement, with trajectory separation per batch. The originalRGB
renderer is pixel-exact and state/geometry/motion metadata do not become network
features. Missing hole/outer-arc coverage, later stalls and weak joint-motion
validation support remain explicit; no unseen-shape or model-quality claim follows.

The clean/weak native-loss interface and8epoch,actual-library R3Bconfiguration
passed CPU contracts only. R3B requires explicit dataset/cursor/noise migration
from the complete6000checkpoint,actual GPU integration and subsequent review.
Actual reports and local paths were sent to the planningchat under standing
human authorization. Stop waiting_parent without polling or automatic training.


## D043 — Audit physical contact before resuming R3B (2026-10-30)

Pause the prior first-epoch release and distinguish all acquired TRAIN time,
legal H1 targets,selected targets and repeated four-action input exposures. Use
saved post_solve AO flags plus valid per-substep impulse presence; never infer
contact from planned push or sum recorder-duplicated point impulses. Preserve
motion thresholds,physics,negative examples,failed prefixes and sealed holdouts.

The completed cold audit finds real pushing support across735 independent TRAIN
trajectories,but40.964% moving-pusher/no-contact intervals and11.64% unrecovered
TRAIN recontact cases. This supports retaining the library while withdrawing an
unqualified continuous-contact interpretation. It does not establish model
quality or authorize new collection. Require the actual audit callback and a
new parent decision before R3B or the proposed targeted pilot. No model-time cap
is introduced;record_only_v2 remains in force. See the2026-10-30contact audit.


## D044 — Freeze one contact controller pilot and retain its failed targets (2026-10-30)

After independently accepting the contact audit,the planning chat released one
48case mechanism pilot:24same-initialstate TRAIN matches with saved baselines,
24new independent TRAIN scenes,max2,304charged controls and16MiB. Keep all physics,
rendering,motion thresholds and withdrawal tasks unchanged. Report A/B separately,
charged/partial failures and fixed paired media;do not retune after the pilot.

The actual pass produced47complete trajectories and one existing2px penetration
guard failure,cost2,262charged controls. Recovery speed and Acontact/motion improved,
but35.942%overall contact+obvious,25.774%empty upper and10.417%unrecovered/notreached
missed frozen goals. Bletter regression,remainingwallstalls and six unreached weak
arc targets prevent promotion. Preserve controller version and negative evidence;
keep the R3A library valid/frozen,do not merge pilot into its selected pool,and
require a new parent decision before further collection or R3B. Known controller
route-bridge/normal-following limitations guide a future proposal,not a silent
post-result implementation change. Model work remains0 and record_only_v2 remains.


## D045 — Preserve CP2's safety repair and negative data-efficiency result (2026-10-30)

Version route+target transactions,whole-union observedpoint/normal/arc validation,
current-state wall-risk constraints and immutable requestedarc evidence separately
fromCP1. Pass13real-savedstate production regressions before freezing60cases:
48seen CP1initial scenes plus12new TRAIN families. Keep allphysics/thresholds,
withdrawtasks and originaldatasets;stop thequeue onanyguard/unknown/systemfail.

The actual one-pass experiment completed60/60 withoutguards,cost2880controls,
butcontact+obvious10.764%,empty68.889%and25%unrecoveredmissedfrozengoals. Every
requestedweakarc washit,mostlybriefly,andB04/T040motionregressed. Preserveversion
andallrawnegativeevidence;do not promote/mergeor retuneuntilpassing. Zero guards
andlocalregressionpassesdonotestablishusefulpushing. Keep validR3Adata available
foraseparatelyauthorizedmodel-validation decision ratherthanendlesscollectorwork.

## D046 — Human approval gates every subsequent implementation stage (2026-10-30)

Thehuman explicitlyrequires approvalafterthecurrentCP2report beforethenext
implementation release. Finish currentauthorizedreportandproactivecallback,save
itsactualreceipt,andSTOP waiting_user_approval. Theplanningchatmayreadonlyreview
andpresentrecommendationsbutcannotautonomouslydispatchnewrepair/retest/collection,
R3Bresumeorexpansion. Reports,includingearlyfailurecallbacks,are notauthorization
tocontinue. Thisoverridespriorparent-autonomousstage-releasepractice. Standing
humanconsentforproactivecross-chatstage reports andlocalpaths remainsvalid.

## D047 — Keep first-epoch authorization separate from startup success (2026-10-30)

Explicit human approval releases one complete R3A epoch,1499batches/23984distinct
targets,full6000 model/AdamW/scaler restore,held1e-5 LR and clean/weak conditioning,
with frozen natural144/stress16 and old4 paired native3-step RGB evaluation.
The old8epoch proposal is provenance only. One hidden independent pipeline must
finish its own evaluation/report;an official30minute completion check may read
terminal evidence and send the authorized report,but cannot train or restart it.

The actual first launch restored6000 and then failed at the CPU loader before any
optimizer attempt. Retain33.047s and all failure evidence;do not report a trained
epoch,model regression or a successful running pipeline. A job-independent PID
is insufficient if work then terminates. Preserve the original library/checkpoint,
stop without automatic retry,and report the exception. Missing child diagnostics
prevent a definitive root-cause claim. A minimal detached-worker bootstrap/venv
diagnosis and controlled recovery require explicit human approval;they must not
silently change data,worker count,microbatch,physics or evaluation protocol.

## D048 — Repair only the evidenced Windows loader startup boundary (2026-10-30)

The human approved diagnostic attempts,minimal repair,verification and then the
original independent first epoch. Retain the failed run/cost;newattempt outputs
cannot overwrite it. Real detached CPU evidence shows the venv redirector PID
differs from actual Python and receives duplicated multiprocessing synchronization
handles;dataset init succeeds butqueue.get fails WinError6. Use the public
multiprocessing executable selection with actualbasePE and existingvenv identity.
Do not change renderer/data,worker count,prefetch,physics,model or sampling.

Verify exact multi-batch RGB/actions/two real PIDs after launcher exit,and repeat
the boundary with the complete6000GPUmodel resident before the real optimizer/
save-reload probe. Both passed. Keep CPUdiagnostic,engineering/formal draws and
time distinct,with old33.047s retained and no model-time cap. Preserve original
validation spec bytes and its full cohorts/context/seeds. Only the first1499batch
epoch is released;its pipeline must create paired evaluations and reports itself.
Use the existing formal30minute monitor for quiet completion checks and actual
callback,not training/restart. Startup handoff ends this chat turn;final callback
pauses monitoring and waits for new human approval before any subsequent stage.

## D049 — First-epoch mean improvement does not pass the dynamics/rigidity gate (2026-10-30)

The authorized1499batch epoch and paired evaluation completed with exact unique
order and no skipped update. Preserve the complete7499lineage checkpoint, prior
failure costs and immutable data/spec. Mean RGB/object-overlap gains are real but
not a replacement for motion strata, matched measurable pose and full shape
denominators. Only75/144 natural H4windows beatcopy-last; moving shape failures
increase5to8 and overall83/144remainunknown. Missing-blue-pusher failures are
part of the frozen generation status, so overall failure reduction alone cannot
prove gray-object rigidity. Keep this distinction and the static persistence
advantage explicit; old regressions and16stress do not enter natural means.

Read-only owner audit adds no model calls or new scoring and keeps original
outputs intact, including their stale1475status cache. Authoritative completion,
durable progress, checkpoint metadata and actual event order confirm1499. Send
the actual final callback with its receipt, finish/delete the obsolete completion
monitor and stop waiting_user_approval. The report authorizes no further epoch,
data collection, sampling/architecture change, repair/restart or server work.


## D051 — Gate contact-constrained original-physics candidate collection (2026-10-30)

Direct human approval now permits repair, bounded candidate collection and unused
asset deletion. Kinematic pusher/damping0, object geometry/mass/inertia/materials,
rendering and10Hz requested absolute targets remain fixed. Solver/integration and
infeasible-command execution changes receive new protocol identities; UI and
collector must share the same factory and record requested commands separately
from actual endpoints/velocities/interventions. No manual object pose/velocity
repair or hidden frame filtering. An invalid-data guard alone is not a repair.
Freeze geometry-based tolerances before tests and retain failed versions/costs.

Use complete initial construction/action prefixes for the B regression; arbitrary
midcontact snapshot restore excludes solver cache. Export all additional numerical
controller state when present. Require broad free/contact/rotation/wall/corner
regressions and actual UI integration before the six-letter pilot. Full scene
families split before simulation; target48train-candidate+12seen-val per letter,
64continuous controls, with separately bounded calibration/supplements. Report
all attempts, physical time, valid candidate windows and motion/contact coverage.
The image+action-to-image route is unchanged and training updates remain zero.

Delete only inventoried obsolete experiment assets using exact resolved literal
paths and hashes. Keep current7499/necessary6000/unique original initialization,
user recordings/regression/test/teammate/source and provenance. Disclose removed
historical files without rewriting old results. After actual parent callback,
stop waiting_user_approval; no further stage or training is released.

## D050 — Original preview physics and separately scoped diagnosis (2026-10-30)

Later direct human instruction supersedes the historical preview default below.
Use pinned original kinematic-pusher step/damping0/solver10 and original body,
geometry and material properties. Observe separation through continued steps;
do not hand-clear velocities, pin poses or repair pixels. Keep7499 frozen and
label cross-physics results. Source/substep parity proves implementation alignment,
not model accuracy. Old screenshots lack actions/history/RNG, so only same-class
reproduction is supported. Keep corrected H1/raw-feedback H4 and paired3/8/16
sampling separate from original epoch metrics, retaining failures and unknowns.

Authorized recording saves actual visible canvases as WebM plus complete same-ID
raw. Start/stop/timers cannot advance simulation or model calls; resets and mode
changes remain labelled. Verify actual downloads and two browser rounds. Public
state restoration excludes internal solver caches. Zero optimizer updates; after
actual report callback, wait for explicit human approval before further training,
collection, model changes or remote work.

## Historical R3B checkpoint-specific interactive preview (2026-10-30)

The user requested a playable interface for the just-completed R3B model, explicitly excluding the teammate dataset. Bind this preview to its verified 7499-step checkpoint and historical physics hash; do not substitute the newly proposed original-kinematic protocol. Keep H1 reconditioning and four-step open-loop inference visibly distinct. Mouse targets are UI overlays, never model inputs; raw predicted RGB is not repaired before autoregression. The server is local-loopback only, inference-only, and explicitly stoppable. This preview does not authorize dataset integration or further training.
# D052: restore strict original physics; sample through requested actions only

Date:2026-10-31 Singapore. The direct human correction overrides D051's expanded
numerical/contact-controller release. Active UI and new collection must execute
the pinned original100Hz step with solver10, original100Hz PD100/20, KINEMATIC
pusher and damping0, unchanged shapes/materials/mass/inertia/walls. Added velocity
projection, speed disks, compression/wall budgets and extra integrator steps are
excluded. Observing and ending an invalid trajectory is permitted; repairing its
physical transition or claiming sampling rejection repaired original physics is not.

Retain modified v1-v6 runs/calibration as diagnostic evidence only, ineligible for
training. Old launch guards and automation continuation are revoked. New data
use an independent root and frozen original-protocol/source membership manifest.
Sampling-only recollection is bounded to18 initial calibration scenes, at most
one distinct18-scene strategy revision,360 original-size candidate scenes and
at most72 once-only conditional supplements. No retry/replacement or new letters.
Shared3GiB and256MiB atomic reserve remain, plus allocated report reserve.
Report attempted and valid denominators, real H1/H4 motion/contact/invalidity,
coverage and real-time videos. Actively return actual milestones and paths to
the planning chat with MCP receipts. Final state waits for human training approval.

## D053 — Review direct absolute-target sampling without custom caps (2026-10-31)

The user requests a revised sampling plan using original absolute position
targets directly, without the custom push12/entry2/target-distance24/target-change24
rules. Do not carry those rules into the new plan as defaults or hide equivalent
clipping inside a controller or path smoother. Original physical settings remain
unchanged. The proposal also removes the extra [23,488] sampler clipping and
discloses the existing original [0,512] versus wrapper/model [0,511] interface
difference; resolving that difference requires an explicit reviewed specification.

Preserve continuous reachable contact coverage, visible translation and rotation,
wall/corner cases, negative examples, and complete H1/H4/H8/H16 action/image
sequences. Distinguish intended events from observed events, planned future
actions from feedback-recorded actions, and independent families from overlapping
windows. See artifacts/direct-target-sampling-review-20261031/PLAN.txt.

This decision records a design direction only. New numeric quotas, sampling
distributions, validity/measurement changes, model settings and budgets are not
approved execution defaults. The present task authorizes documentation, not
implementation, new sampling, inference, training or automatic owner dispatch.
The parent presents returned results for human approval before further work.


## 2026-11-01: approved existing high-motion training-fit experiment

The human approved artifacts/high-motion-fit-proposal-20261101/PLAN.txt
(SHA256 138d641c88cafe9a82e3c805aa2bf539219684135615e551fa8d862562b9e0d0).
Use 48 independent O/C/B training clips, original splits and physics, old-A
true-history optimization from epoch2, and at least 75-percent obvious-motion
targets per batch. One background branch may run up to 3000 updates. Evaluate
training fit and original seen-val separately at H1/4/8/16. No new collection,
physics/model/loss changes, server execution, UI switch or automatic extension.
The owner actively reports selected clips, actual throughput/ETA and final
comparisons to the parent. Human review is required before further experiments.
Actual approval and dispatch receipts are stored alongside the plan.

## 2026-10-31: approved direct original-domain acceptance sampling

Actual parent human authorization permits implementing direct absoluteXYsampling and <=120newphysicalattempts (96independentfamilies+24branches, <=64controls each) in unchanged original100Hz/solver10/PD100,20/KINEMATIC/damping0. No custom12/2/24/24 or23..488targetclip. Raworiginal0..512domain is stored exactly; oldmodel0..511normalization andliveUI remain unchanged. Allactions frozenbeforeexecution, fulloriginal reconstruction forsharedhistories, invalidstatesobserved+terminated withoutrepairs, noextra physics/replacements. Whole newroot256MiB underexistingshared3GiB/256MiBatomicreserve. Implement/reporttruthonly; nocurrenttraining/formaldata/serverrelease. Actualverifiedapproval,codebindings,tests and process independence are in artifacts/direct-target-acceptance-20261031. Completion must activelycallbackparent thenstopwaiting_user_review.
# 2026-11-02: Bounded loss-alignment comparison and failed preflight

Direct human authorization selected the planning chat to implement one paired loss comparison and report to the independent analysis chat. The frozen choice was native H1 residual MSE versus native H1 plus `4 * mean(AR4_RGB - GT4_RGB)^2`, with an exact native quantization forward and explicitly approximate straight-through backward. Both arms were to restore E2 full state and use the same first 256 HMF batches; extra B supervision/compute was disclosed, not called an equal-compute comparison. Original physics/data/splits/evaluation thresholds remained fixed. The preflight stopped after 33 denoiser forwards due to a non-reusable logging context during repeated checkpoint backward. No A/B updates occurred. Do not infer scientific efficacy from this engineering failure. Preserve frozen sources, report the failure and unapplied repair, and wait for approval before another numerical execution. Full frozen recipe and failure evidence: `artifacts/loss-rollout-ab-20261102/`.
