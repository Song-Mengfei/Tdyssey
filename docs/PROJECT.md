# Project scope

For the step-by-step implementation, data/reward/loss definitions and developer
handoff, read [the learning plan](LEARNING_PLAN.md). That document distinguishes
existing code from proposed experiments; this page defines the research scope.

## Research question

Start from the official RSS 2023 visual CNN Diffusion Policy trained on T expert
demonstrations and verified in its original PushT environment. Collect
machine-generated non-expert interactions across object shapes, reuse or adapt
a shared world model, freeze that model during each policy-learning phase, and
improve one shared policy through imagined experience. The goal is broader
multi-shape pushing and accurate completion without new human demonstrations for
adaptation. Evaluate learned policies in physical simulation, independently of
the learned predictor. The original T checkpoint's expert pretraining remains
part of the method and must be disclosed.
Record the pretraining-data provenance of any reused world model as well; the
no-new-human-data claim concerns project adaptation, not all prior asset training.

H1: contact-rich interaction data plus a reliable world model can increase useful
contact behaviors that the initial policy rarely produces, improving physical
task performance. H2: a shared world model and mixed-shape policy updates extend
this improvement to multiple shapes while retaining T performance. H3: accumulated
experience improves zero-shot or fixed-budget adaptation to held-out shapes.
H1 is the first learning check; H2 is the broader project objective. Genuine
unseen-shape transfer and sequential continual learning remain stretch questions.

The robot remains a circular 2D pusher. Shape variation concerns the pushed object.
Start with T and an upright capital L with a shorter bottom foot, one rigid
object, fixed physical/control parameters, and
fixed target pose for the adaptation study. Prior GPC reference runs used a
different environment and randomized goals; they are not the official baseline.
Use T-to-L to verify the learning loop, then expand to a fixed multi-shape
development set and eventually the alphabet as resources allow. L alone is not
the intended final scope. Do not train every letter before validating the loop.
Symmetry and shape-dependent mass/inertia must be explicit.

## Data, generalization and completion

No new human demonstrations, human teleoperation or human success annotation are
required for adaptation. Collect random play, frozen/current-policy interaction
and automated exploration. Log simulator coverage, contacts and other physical
labels with explicit privileged-supervision disclosure. The data need not solve
the task to teach dynamics; it must cover contact, rotation, near-goal corrections
and failure/recovery states. The existing 800-control L pilot is not such a
multi-shape training set.

Prefer one image/action-conditioned world model shared across training shapes,
not one model per letter or a mandatory letter-ID lookup. First test a frozen
pretrained model; train/adapt it only where compatibility and physical prediction
checks justify doing so. A genuinely transferable model need not see every letter,
but each shape used for imagined policy learning needs validated action effects.
A model trained only on L does not establish that validation for the alphabet.
For broad training, collect a balanced set of diverse shapes before scaling to
all 26. Freeze the chosen set and per-shape interaction budgets before collection.

World-model validation must include pusher positions, contact/rotation, goal
visibility, reward and termination consistency, not just RGB reconstruction.
Reward labels can come from the physical simulator during collection. Imagined
rewards must come from validated predicted state/reward outputs; they cannot be
replaced by unseen true future simulator states. Keep the official policy's RGB
and pusher-history interface unless a representation change is explicitly compared.

The current 2D policy outputs position targets, with no separate stop output;
the normal runner ends when coverage is strictly above 0.95. Its continuation
failure does not establish that its training demonstrations lacked finishing
behavior. Close alphabet frames suggest partial transfer, not proof that finishing
is the only missing skill. Preserve official success as the primary benchmark.
Report final/peak alignment and post-success retention separately. If optimizing
sustained alignment, hold/retreat or an end-zone finish, define a separate task and
reward/termination rule before training. Do not lower the original success
threshold or introduce a privileged best-frame stop to manufacture improvement.

Training on all letters and testing new starting scenes measures shared
multi-shape competence. Genuine unseen-shape evaluation must exclude the shape
from both world-model data and policy training. If the world model sees a target
shape's play data, disclose that exposure even when policy updates exclude it.
The explored alphabet cannot be described as an untouched final shape test;
reserve new geometry variants before further development for a stronger test.

## Course deliverables and gates

| Milestone | Substantial deliverable | Evidence needed to advance |
| --- | --- | --- |
| M0: resources | Reproducible environment, fixed upstream code, verified resources | T policy rollout and resource/throughput report |
| M1: environment | T/L geometry, collision, reset/replay, reward, contact logging; I later | Boundary cases and matched-state replay |
| M2: dynamics | Automated collectors and shared dynamics, first L then a fixed mixed-shape set | Per-shape held-out contact errors, candidate ranking, optimistic-action replay |
| M3: optimization | Direct simulator diffusion-policy fine-tuning, then imagined fine-tuning | Correct likelihood/gradient checks and physical learning curves |
| M4: analysis | Shared multi-shape policy, T retention, baseline comparisons and cost/failure analysis | Per-shape repeated runs and reproducible tables/videos |
| Stretch | Continual adaptation, forgetting, held-out shapes | Matched-data comparisons and clean shape splits |

Milestone numbers group deliverables, not a requirement to finish dynamics before
starting optimization. Direct-simulator RL and pretrained-model assessment can
proceed independently. Validated dynamics and a checked direct learning loop are
both required before the joint imagined-learning comparison; see D027.

Engineering gates guide debugging and resource use; missing a performance target
does not erase a completed implementation or useful negative result. Document
the failure and competing explanations instead of inventing an improvement.

## Implementation routes

Route A starts with the official visual CNN policy. Reuse suitable GPC world-model
components as implementation references or comparisons after checking compatibility.
GPC and DPPO/DiWA are not plug-and-play: align model inputs, histories, normalizers,
action horizons, rewards, and diffusion likelihoods. GPC learned reward geometry
is T-specific. Running unchanged GPC is not the adaptation contribution.

First implement direct simulator policy updates using the existing checkpoint's
inputs and correct denoising likelihoods. A limited set of trainable action-network
blocks is the simplest initial comparison. Compare a small added adapter and full
fine-tuning only with explicit trainable-parameter/memory budgets and matched data,
control and inference budgets. Freezing the image encoder is an initial option,
not proof its features suffice for every shape. Preserve T through mixed-shape
training and/or imitation of the frozen T policy on automatically collected T
observations; this adds no new human demonstrations. Measure retention physically.

Once shared dynamics pass their physical prediction checks, compare direct
simulator updates with imagined RL using the same policy-update mechanism. Freeze
the world model within a policy-learning phase; any later data/model refresh is
separately budgeted and recorded. DPPO/DiWA support the mechanism, but DiWA's use
of expert skill/reward data is not evidence that our no-new-human-data variant
has already been solved. No RL trainer or trained adapter exists in this repo yet.

Route B uses explicit state and shape geometry with a smaller learned dynamics
model. Retrain a compatible diffusion policy on the same T demonstrations.
This is a substantial mechanism study, with matched state-observation baselines;
it cannot be reported as visual-policy adaptation. Select A/B after resource and
throughput verification on the actual training machine. The visual CNN route is
the current starting point; a representation change needs an explicit decision.

## Minimum comparisons

Original T policy; world-model ranking/planning; direct simulator fine-tuning;
imagined fine-tuning. Add GPC-OPT, successful-rollout behavior cloning, or a
planner without the T policy prior when they resolve an important explanation.
Develop on one seed; repeat the key methods when functional and affordable.

Count underlying control steps, not action chunks or denoising steps. Keep real
training interactions, diagnostic replay, validation, final evaluation, imagined
transitions, and compute in separate records. Report final/max overlap, a frozen
success rule, contact behavior, latency, and failures with matched initial states.

## Expected engineering layout

The first implementation lives in `tdyssey/`, with existing-simulator adapters
under `tdyssey/envs/`. Add collection/models/training modules there when needed;
do not create empty framework layers to imply progress. Tracked configuration lives in `configs/`; small provenance records in
`manifests/`; large outputs in ignored or external artifact directories.

The original 16-week schedule assumes a full-time researcher and an unverified
32 GB GPU. It is a reference plan, not the course deadline or a hardware requirement.
The team has three authors. Deadline, individual owners, available
training machine, and compute budget remain to be recorded in onboarding.

A separate frozen-policy development sweep covers all capital letters plus a
10%-shorter T cap; see [shape protocol](OFFICIAL_SHAPES.md). This supports failure
analysis without expanding the first learning milestone beyond T-to-L. These
explored letters also inform the planned multi-shape phase, and cannot later be
presented as an untouched held-out shape set.
