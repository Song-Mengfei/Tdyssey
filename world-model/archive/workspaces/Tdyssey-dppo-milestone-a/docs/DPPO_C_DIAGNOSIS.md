# C offline failure diagnosis

Reviewed 2026-10-24 from `e0842d7`, using only saved evidence from the
[completed training](../../Tdyssey-world-model-validation/docs/DPPO_C_RUNTIME.md) and [paired evaluation](DPPO_C_EVALUATION.md).
No new control, reset, policy sample, training update or evaluation was performed.

## Findings and their strength

**L failed before high alignment; the lost T success is a trajectory/finishing
failure in one development case. Neither observation identifies its physical
cause.** All 20 saved L trajectories, across both methods, peak below 0.491.
None enters the diagnostic near-success band `[.90, .95]`. Their 300-control caps
therefore do not establish that a small extension would solve them. The lost T
case reaches .9493688 at control 287, then falls to .9019827 at 300; insufficient
remaining time is plausible there, but success after 300 is unknown.

| Evidence class | Observation | What it supports / cannot establish |
| --- | --- | --- |
| Direct | L training: 932 controls, four episode starts, 247 contact controls, zero successes; evaluation max .490487 | Limited exposure and no high-alignment examples; neither zero contact nor proven insufficient training |
| Direct | L scene 100002 / seed 5479 stays at zero coverage in both methods | No target overlap, not proof of no contact or object motion |
| Direct | T loses one success and has larger mean peak-to-final loss | A development-set retention concern, not broad catastrophic forgetting |
| Direct | Reward/return source and pure arithmetic agree with the declared objective | No discovered terminal-potential or timeout-bootstrap bug; does not prove reward suitability or runtime labels everywhere |
| Supported hypothesis | Sparse success plus long flat-coverage sections can limit useful learning | Exploration and credit assignment deserve attention; their individual causal effects are unmeasured |
| Supported hypothesis | 28/52 negative mixed-buffer explained variances can impair advantages | Critic quality is uneven; saved logs cannot identify per-shape value errors or prove that they caused failure |
| Unresolved | Only 1,313,282 of 262,709,056 actor parameters train, with frozen visual features | Representation/output-range limits remain possible; an actual capacity failure was not measured |

Only five scene blocks per shape were evaluated, with two correlated policy seeds
per scene and one training seed. These are descriptive development findings.

## What was actually saved

| Source | Available | Missing or restricted |
| --- | --- | --- |
| `training.jsonl`, training report | 18 batch summaries, 36 shape/batch summaries, 52 pass reports; contacts, rewards, coverage; mixed value/advantage moments | Full control/episode trajectories; per-shape values/advantages; later RGB and numeric object poses |
| `replay-first-T.pt`, `replay-first-L.pt` | Two pre-update `actor:0` plans: controls 1–8 per shape, 18 actual 96x96 RGB observations, observed pusher positions, commanded actions and contacts | Only 16/2,048 controls (0.78125%); cannot show final-policy behavior or later failures |
| Two evaluation JSON reports | All 10,640 executed action/coverage/IoU/reward/done rows, complete initial physical states, counts and planning latencies | No per-control contacts, object/pusher poses or RGB; no explicit plan IDs |
| Existing verified checkpoint JSON | Identity, selected parameter counts, finite-tensor/optimizer metadata, historical resume flags | No new model load or weight-space attribution in this diagnosis |

The saved first-plan frames show pusher motion while the visible object silhouettes
remain essentially unchanged. Both eight-control records report zero contacts;
T coverage is zero and L coverage stays .0512973. This narrow pre-update sample
cannot establish how later L episodes failed. No frames exist for the lost T case.

Control 0 below is the initial measurement. Positive coverage at 0 is existing
overlap, not policy-created progress. Controls 1..N are executed-control ordinals.
Plan boundaries in derived plots are **inferred** from the locked eight-action
prefix and matching planning counts; they were not explicitly logged.

## Every L pair

All cases below execute 300 controls per method without success. Coverage entries
and positive executed-control fractions are percentages. `never` means no overlap
at the initial measurement or any saved control. Peak indices are the first maxima.

| Scene / seed | First positive base / delta | Peak base @ control | Peak delta @ control | Final base / delta | Positive fraction base / delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100000 / 5478 | 20 / 20 | 44.56% @ 41 | 44.08% @ 41 | 37.21% / 0.00% | 93.7% / 12.0% |
| 100000 / 5479 | 22 / 22 | 45.40% @ 43 | 44.52% @ 43 | 6.34% / 26.91% | 93.0% / 93.0% |
| 100001 / 5478 | 0 / 0 | 41.24% @ 145 | 40.99% @ 72 | 35.57% / 36.50% | 100.0% / 100.0% |
| 100001 / 5479 | 0 / 0 | 46.91% @ 35 | 46.95% @ 35 | 0.81% / 34.58% | 100.0% / 100.0% |
| 100002 / 5478 | 27 / 27 | 47.27% @ 84 | 49.05% @ 110 | 37.74% / 38.87% | 91.3% / 91.3% |
| 100002 / 5479 | never / never | 0.00% @ 0 | 0.00% @ 0 | 0.00% / 0.00% | 0.0% / 0.0% |
| 100003 / 5478 | 0 / 0 | 43.43% @ 72 | 43.86% @ 300 | 36.12% / 43.86% | 100.0% / 100.0% |
| 100003 / 5479 | 0 / 0 | 47.45% @ 66 | 47.69% @ 66 | 40.64% / 40.47% | 100.0% / 100.0% |
| 100004 / 5478 | 0 / 0 | 36.59% @ 72 | 36.54% @ 72 | 35.13% / 35.07% | 99.7% / 99.7% |
| 100004 / 5479 | 0 / 0 | 48.28% @ 134 | 39.27% @ 92 | 37.99% / 34.09% | 98.3% / 98.3% |

The mean final gain (+2.2792 pp) is not consistent progress. The positive-coverage
fraction falls from 87.6000% to 79.4333%. Mean peak-to-final loss decreases
13.3571 to 10.2600 pp, but case 100000/5478 loses 44.08 pp after its delta peak
and finishes at zero. Conversely 100001/5479 preserves about .346 coverage while
base falls to .008; this one pair contributes +33.7739 pp to the final difference.
Neither method gets close to successful L alignment. Positive overlap alone does
not tell whether contacts were useful or the orientation was correct.

## Every T pair and the lost success

An endpoint above 95% denotes success and immediate stopping; other cases stop
at 300. No curve is extended after success, and later ordinals cannot be paired
against an already stopped method. Peak-to-final comparisons reflect this rule,
not a separate post-success retention experiment.

| Scene / seed | Peak base @ control | Peak delta @ control | Final base / delta | Controls base / delta |
| --- | ---: | ---: | ---: | ---: |
| 100000 / 5478 | 91.12% @ 133 | 94.33% @ 169 | 69.04% / 59.04% | 300 / 300 |
| 100000 / 5479 | 93.70% @ 218 | 93.77% @ 124 | 62.76% / 74.97% | 300 / 300 |
| 100001 / 5478 | 98.06% @ 240 | 94.94% @ 287 | 98.06% / 90.20% | 240 / 300 |
| 100001 / 5479 | 64.12% @ 81 | 68.47% @ 255 | 41.68% / 68.39% | 300 / 300 |
| 100002 / 5478 | 94.07% @ 216 | 94.18% @ 249 | 73.22% / 41.09% | 300 / 300 |
| 100002 / 5479 | 95.54% @ 160 | 95.86% @ 161 | 95.54% / 95.86% | 160 / 161 |
| 100003 / 5478 | 98.05% @ 252 | 96.27% @ 154 | 98.05% / 96.27% | 252 / 154 |
| 100003 / 5479 | 97.00% @ 175 | 95.43% @ 171 | 97.00% / 95.43% | 175 / 171 |
| 100004 / 5478 | 97.34% @ 78 | 97.62% @ 78 | 97.34% / 97.62% | 78 / 78 |
| 100004 / 5479 | 95.74% @ 288 | 97.17% @ 183 | 95.74% / 97.17% | 288 / 183 |

Lost case 100001/5478 starts with identical saved states and coverage .16896148.
The first eight commanded targets differ by only .5488 pixels on average, but
mean distance over the shared 240-control prefix is 107.9252 pixels. At ordinal
240, base succeeds at .980565 while delta is at .289955. Delta later recovers to
.9493688 at 287; controls 286–300 are its only near-success segment (15 controls).
It never meets strict `> .95`. The later coverage loss is 4.7386 pp.

These are early command differences followed by accumulated closed-loop trajectory
divergence. Same initial seed or control ordinal does not imply the same later
physical state. At ordinals 200–202 base coverage rises while delta coverage
falls, but missing pose/contact data prevent attributing this to a particular
collision, rotation, overshoot or network component.

The other nine pairs matter: 100000/5478 and 100002/5478 also improve peak but
lose final coverage; the latter's final loss is 32.1323 pp. Meanwhile five
both-success pairs remain successful, three finish earlier, one at the same
control and one later. Small sampled denoising KL therefore coexists with useful
and harmful behavior changes; it cannot establish T preservation or justify a
learning-rate increase.

## Scene-level contrast

Each row averages two correlated policy seeds; differences are delta minus base.

| Scene | T successes base → delta (/2) | T final coverage difference | L successes base → delta (/2) | L final coverage difference |
| --- | ---: | ---: | ---: | ---: |
| 100000 | 0 → 0 | +1.1042 pp | 0 → 0 | -8.3234 pp |
| 100001 | 1 → 0 | +9.4263 pp | 0 → 0 | +17.3535 pp |
| 100002 | 1 → 1 | -15.9084 pp | 0 → 0 | +0.5636 pp |
| 100003 | 2 → 2 | -1.6750 pp | 0 → 0 | +3.7830 pp |
| 100004 | 2 → 2 | +0.8550 pp | 0 → 0 | -1.9807 pp |

L's mean final gain is concentrated in scene 100001; two scene blocks decline.
That scene's positive T mean also hides its lost success. Endpoint means cannot
replace paired success outcomes or per-case review.

## Training signals, rewards and value estimates

L contacts occupy 247/932 controls (26.5021%); T contacts occupy 272/1,116
(24.3728%). Unknown contact counts are zero. These counts refute "no L contacts
were ever observed", not "contacts were effective". L has 869 nonzero training
rewards but no success bonus; a nonzero shaping signal is not necessarily useful
exploration. Only four of the fixed 32 L training scenes were started.

| Batch-index stage | Actual controls | T / L contacts | T / L maximum coverage | Mean mixed pass explained variance |
| --- | ---: | ---: | ---: | ---: |
| 1–6 | 761 | 182 / 124 | .958976 / .447624 | -.020976 |
| 7–12 | 628 | 40 / 60 | .926399 / .486981 | -.281146 |
| 13–18 | 659 | 50 / 63 | .067382 / .459739 | +.697518 |

These stages contain different scenes, partial batches and boundary drains. Their
declining T coverage or rising value fit cannot be called a policy learning curve.
L batches 15–17 contain 172 controls with zero contacts, zero coverage change and
constant coverage .3723345744. Their reward sum is -.640415468, exactly the
flat-coverage shaping formula. Mixed final-pass explained variance nevertheless
rises .88227 → .89431 → .93769 across those batches. Fitting stationary values
or between-shape offsets is plausible; no saved per-shape values prove that mechanism.

Returns, raw/normalized advantages, old values, MSE and explained variance are
**mixed T/L buffer** statistics. Across 260 fresh plans counted once, value-target
mean/std is -.0750233/.1182569; raw advantage mean/std -.0168354/.1112223.
In batches 14 and 15 every raw advantage is negative, yet centered normalized
advantages include positive values. Those are relative preferences after mixed
batch normalization, not proof of positive absolute reward or an observed sign bug.

All 52 passes change the intended actor and critic tensors; all 18 fresh entries
have exact behavior probabilities and preserve frozen fingerprints. Actor
pre-clipping norm exceeds one in 43/52 passes; clipping is applied and no nonfinite
gradient occurs. Actor per-step L2 changes range .002996–.011336; their sum is
not the final displacement from base. Maximum sampled post-pass KL is .000161719,
maximum PPO clipping fraction .001875. The output subset is about .5% of the full
actor; these facts show updates happened, not that capacity was sufficient.

The declared optimized reward is
`first_success_bonus + beta*(gamma*next_potential - current_coverage)` with
`gamma=.99`, `beta=1`. Upstream coverage reward is reported separately, not added.
At true success (`coverage > .95`), next potential and value bootstrap are zero.
Timeout/buffer cutoff preserve actual final coverage and bootstrap from the
pre-reset final observation. For an m-control block, physical discount is
`gamma**m`, and GAE continuation is cut at success/timeout/buffer boundaries;
there is no cross-shape or cross-episode continuation.

Consequences verified by pure arithmetic: zero-coverage stagnation earns zero;
flat coverage .4 earns -.004/control; a small positive coverage change can still
have negative local shaping. T batch 3 contains the sole success bonus 1, although
its undiscounted training reward sum is -.0736889 because shaping sums -1.0736889.
Negative sums do not mean the bonus was missing. Potential differences telescope
under the declared discount/boundaries; practical finite-bootstrap and exploration
effects remain uncertain. This review finds no discrepancy with the planned reward
semantics, and cannot re-audit unsaved per-control training labels.

## Reproduction, figures and preservation

Three reusable read-only analyzers are committed: `scripts/analyze_c_pair_traces.py`,
`scripts/analyze_c_training_diagnostics.py`, `scripts/inspect_c_saved_replays.py`.
Their `--help` paths need only stdlib. Scalar trace/training analysis uses stdlib;
optional plotting uses the existing Matplotlib 3.9.2. Replay export CPU-loads only
the two small replay payloads with `weights_only=True`; no complete model is
reconstructed and CUDA stays uninitialized. Use **new** output directories.

Read-only reproduction with the existing environment (`$python` is its Python
executable; the output root below must not exist):

```powershell
& $python -S -B scripts/analyze_c_training_diagnostics.py `
  --output-dir artifacts/c-diagnosis-new/training
& $python -B scripts/analyze_c_pair_traces.py `
  --baseline artifacts/milestone-c-evaluation-20261024/base_training.json `
  --delta artifacts/milestone-c-evaluation-20261024/delta_training.json `
  --output-dir artifacts/c-diagnosis-new/traces --plots
& $python -B scripts/inspect_c_saved_replays.py `
  --output-dir artifacts/c-diagnosis-new/replays
```

Eighteen focused stdlib tests passed with
`python -S -B -m unittest discover -s tests -p 'test_c_*.py'`.
Repository source/link and Git whitespace checks also passed. Independent reviews
reconstructed all case tables and checked training/reward/budget statements.

Local derived root: `artifacts/milestone-c-offline-diagnosis-20261024/`.

- `traces-final/`: 40 case rows, 20 pair rows, 10 shape/scene rows, 10,680
  measurements including initial states, 5,206 ordinal-aligned prefix rows.
- `training-final/`: 18 batch, 36 shape/batch and 52 pass rows; reward arithmetic,
  mixed-moment analysis and 13 source/input hash records.
- `traces/figures-reviewed/`: PNG/PDF coverage and action-target panels for every
  T/L case, plus lost-T detail. These draw only saved metrics/commands.
- `replays/reviewed/`: actual first-plan PNGs, sheets and frame/control mappings.
- `review-final/`: training diagnostic PNG/PDF, all-case Markdown tables and
  early-command distance contrasts; no fabricated object or contact paths.
- `inputs-before.json`, `inputs-after.json`: full original evidence inventory,
  including B/C/evaluation registries, both monitor trees, base weights and
  original research-checkout changes. `closeout.json` records final hashes,
  checks, zero physical/training operations and unchanged input verification.

Earlier render/analysis drafts remain local; the final directories above are
authoritative. An initial training-figure helper encountered an empty shape batch;
the reviewed plot leaves missing shape observations blank, rather than assigning
zero contact. No source experiment was repeated. Generated images/data stay ignored,
and are not a new published evidence snapshot or automatically shared with coauthors.

The [single proposed follow-up](DPPO_C_DURATION_STUDY.md) varies only physical
training exposure. It is not authorized or executable under C v1. Other explanations
above remain unexcluded; no new experiment starts from this document.
