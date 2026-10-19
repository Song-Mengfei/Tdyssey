# Project scope

## Research question

Start from the official RSS 2023 visual CNN Diffusion Policy trained on T expert
demonstrations and verified in its original PushT environment. Collect
non-expert interactions with a new shape, adapt a world model, freeze that model,
and improve the policy through imagined experience. Evaluate the resulting policy
in an independent physical simulator.

H1: contact-rich interaction data plus a reliable world model can increase useful
contact behaviors that the initial policy rarely produces, improving physical
task performance. H2: experience across shapes reduces later adaptation cost.
H3: accumulated experience improves zero-shot or fixed-budget adaptation to held-out
shapes. H1 is the initial focus; H2/H3 are stretch questions, not promised results.

The robot remains a circular 2D pusher. Shape variation concerns the pushed object.
Start with T and an upright capital L with a shorter bottom foot, one rigid
object, fixed physical/control parameters, and
fixed target pose for the adaptation study. Prior GPC reference runs used a
different environment and randomized goals; they are not the official baseline.
Prioritize T-to-L adaptation before adding I or architecture comparisons.
Symmetry and shape-dependent mass/inertia must be explicit.

## Course deliverables and gates

| Milestone | Substantial deliverable | Evidence needed to advance |
| --- | --- | --- |
| M0: resources | Reproducible environment, fixed upstream code, verified resources | T policy rollout and resource/throughput report |
| M1: environment | T/L geometry, collision, reset/replay, reward, contact logging; I later | Boundary cases and matched-state replay |
| M2: dynamics | Non-expert collectors and learned dynamics | Held-out contact errors, candidate ranking, optimistic-action replay |
| M3: optimization | Direct simulator diffusion-policy fine-tuning, then imagined fine-tuning | Correct likelihood/gradient checks and physical learning curves |
| M4: analysis | Baseline comparisons, behavior diagnostics, failures, cost accounting | Repeated runs and reproducible tables/videos |
| Stretch | Continual adaptation, forgetting, held-out shapes | Matched-data comparisons and clean shape splits |

Engineering gates guide debugging and resource use; missing a performance target
does not erase a completed implementation or useful negative result. Document
the failure and competing explanations instead of inventing an improvement.

## Implementation routes

Route A starts with the official visual CNN policy. Reuse suitable GPC world-model
components as implementation references or comparisons after checking compatibility.
GPC and DPPO/DiWA are not plug-and-play: align model inputs, histories, normalizers,
action horizons, rewards, and diffusion likelihoods. GPC learned reward geometry
is T-specific. Running unchanged GPC is not the adaptation contribution.

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
explored letters cannot later be presented as an untouched held-out shape set.
