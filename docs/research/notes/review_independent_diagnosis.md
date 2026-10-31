# Independent diagnosis: why forward RL fine-tuning degrades the official Diffusion Policy

Reviewer note, 2026-10-25. Read-only review of the training checkout (branch
`perf/batched-rollouts`, commit `efbd594`) and the live run files under
`runs/long-s2/` and `logs/score_snr.json`. I did not
run training, GPU jobs or simulators. I re-analysed the stored reports and metrics
with standard-library Python. Because the long runs are still writing their reports,
numbers taken from the "last 10 detailed updates" changed slightly between my
snapshots; where that happened I give ranges.

## Bottom line

1. **No run shows learning.** Weak runs (LoRA 3e-4, dense 3e-6) leave every
   shape where it started. Fast runs (dense 1e-5 and above) lose T and T_narrow.
   No setting finds a window where improvement outpaces damage.
2. **Most likely main cause: the learning signal per optimizer step is far too
   weak, but the optimizer moves the weights at a fixed rate anyway.** Candidate
   scores are mostly noise from the follow-up draws. Many groups carry no signal
   at all. Each record gets only one or two noisy loss samples, and half of each
   plan is never executed. Adam takes steps of roughly fixed size whether or not
   the gradient carries information. The result is undirected drift, and any
   undirected change hurts a finely tuned imitation policy.
3. **Nothing holds the policy near its starting point across updates.** The NFT
   and AdvantageFlow "trust region" terms hardly act. The rollout policy is
   copied every update and only 4 small Adam steps follow. The T anchor has
   weight 0.001, starts with zero gradient, and covers T only. T_narrow, D, Y
   and L have no anchor at all.
4. **The score does not measure what success needs.** Within a group, the
   potential shaping cancels, so the score is just "coverage after 32 controls".
   The candidate's own plan controls only the first 8 of those 32. Coverage is
   zero or unchanged in many contexts, and success inside the scoring window is
   rare (0-4.5% of candidates). The failures we want to fix stall at a peak
   coverage of 0.92-0.94, which this score barely tells apart.
5. **Some headline measurements mislead.** The T_narrow "fall" and the D "rise"
   at near-zero drift are mostly baseline luck plus threshold effects. L is not a
   training shape in any long run. The t=99 probe blow-up does not change
   behaviour. The score diagnostic overstates the share of score variance that
   comes from the plan, especially for Y and T_narrow.
6. **Recommended order:** first fix measurement (more scenes, repeated baselines).
   Then run two cheap decisive diagnostics: a shuffled-advantage null control, and
   a simulator-lookahead best-of-N upper bound. Only then lower the noise in both
   the score and the loss, and add a real anchor to the initial policy. Use the
   cheaper rollouts this frees up to collect many more groups.

## 1. What the data show after re-analysis

### 1.1 Evaluation outcomes pooled by how far the policy moved

"Drift" is the probe's `initial_policy_drift_mse` divided by the shape's initial
`fixed_noise_mse`, which is the definition `sweep_report.py` uses. Rows pool all
post-baseline evaluations across the 10 long runs. Each evaluation has 20 scenes
per shape.

| T drift bin | evals | T success / final / peak | T_narrow success / final / peak | Y success / final | D final | L final |
|---|---|---|---|---|---|---|
| baseline, seed 42 | 1 | 0.55 / 0.78 / 0.87 | 0.55 / 0.70 / 0.85 | 0.05 / 0.55 | 0.27 | 0.34 |
| baseline, seed 43 | 1 | 0.55 / 0.78 / 0.85 | 0.65 / 0.75 / 0.87 | 0.15 / 0.58 | 0.37 | 0.31 |
| < 0.01 | 9 | 0.55 / 0.77 / 0.87 | 0.40 / 0.72 / 0.85 | 0.06 / 0.59 | 0.32 | 0.33 |
| 0.01-0.03 | 3 | 0.53 / 0.77 / 0.86 | 0.43 / 0.76 / 0.86 | 0.15 / 0.65 | 0.33 | 0.35 |
| 0.03-0.1 | 10 | 0.40 / 0.72 / 0.85 | 0.31 / 0.74 / 0.84 | 0.05 / 0.58 | 0.33 | 0.34 |
| 0.1-0.3 | 3 | 0.23 / 0.64 / 0.84 | 0.37 / 0.74 / 0.83 | 0.05 / 0.61 | 0.39 | 0.34 |
| > 0.3 | 7 | 0.11 / 0.49 / 0.68 | 0.12 / 0.49 / 0.63 | 0.01 / 0.37 | 0.24 | 0.24 |

What this shows:

- Two evaluations of the **same frozen policy** with different sampling seeds
  give T_narrow 0.55 versus 0.65, Y 0.05 versus 0.15, and D final coverage 0.27
  versus 0.37. That gap is the noise floor of a single 20-scene baseline.
- Below drift 0.03 nothing changes in the continuous metrics (peak and final
  coverage) for any shape. The policy has neither improved nor degraded.
- For drift between 0.03 and 0.1, T success drops (0.55 to 0.40) while T peak
  coverage barely moves (0.87 to 0.85). These are near-threshold successes
  flipping to failures, not lost pushing skill. Real skill loss (peak coverage
  0.87 to 0.68) appears only above drift 0.3.

### 1.2 Per-scene changes at near-zero drift (drift < 0.01, 9 evaluations)

| Shape | baseline successes lost | baseline failures gained | peak coverage of the lost episodes |
|---|---|---|---|
| T | 26 | 26 | mostly 0.92-0.95, a few at 0.44-0.64 |
| T_narrow | 49 | 18 | almost all 0.88-0.95 |
| Y | 9 | 7 | 0.83-0.94 |

For T, losses and gains balance: the chaotic simulator re-draws near-threshold
outcomes. For T_narrow, losses outnumber gains. Each single baseline is only 1-2
standard deviations above the later mean (11 successes against an expected
~8.7 for seed 42), but both seeds' baselines being high is jointly unlikely
(roughly p of 0.01-0.05). Final and peak coverage are unchanged, so most of the
"T_narrow falls in nearly every run" effect at low drift is a lucky baseline plus
the strict 0.95 threshold. A small real T_narrow-specific sensitivity cannot be
excluded. Section 3 gives the test.

The D seed-42 baseline final coverage (0.27) is the **lowest** of all seed-42
evaluations at low or moderate drift, which range from 0.28 to 0.43. Any apparent
D gain is mostly regression to the mean. Y final coverage drifts up slightly in
some LoRA runs (0.58 to 0.70 in seed 43) while peak coverage stays flat. That is
a weak hint that the policy stops undoing coverage after its peak, but it is
within noise.

The frozen baseline itself shows many near misses. T failures peak at 0.92-0.95,
T_narrow at 0.92-0.94, and Y at 0.81-0.94 in most scenes. Y is a fine-alignment
problem, not a reach-the-goal problem.

### 1.3 How much signal the training scores carry

`score_snr.py` reports `plan_share = between / (between + within)`, where
`between` is the variance of K=8 draw means. That estimate is biased upward,
because the variance of a mean still contains within/K of noise. The bias-corrected
share of variance due to the plan is
`(between - within/K) / (between - within/K + within)`:

| Shape | reported plan share | **corrected plan share** | split-half r (reliability of one draw) | per-context range (corrected) |
|---|---|---|---|---|
| T | 0.33 | **0.25** | 0.27 | about 0 to 0.97 |
| T_narrow | 0.17 | **0.08** | 0.18 | about 0 to 0.31 |
| Y | 0.16 | **0.06** | 0.14 | about 0 to 0.14 |
| D | 0.36 | **0.29** | 0.29 | about 0 to 0.63 |

Only 6 contexts per shape went into this diagnostic, so these estimates are
rough. Still, the two near-miss shapes we most want to improve, Y and T_narrow,
have the weakest signal.

Training groups, taken from the last 10 detailed updates of each run in two
snapshots (about 100 groups per shape each):

| Shape | all 8 scores tied | coverage zero for all candidates | tied although the object moved | candidates that succeed in window | corr(coverage after own 8 controls, score) |
|---|---|---|---|---|---|
| T | 19-39% | 9-27% | about 8% | 2.0-4.1% | 0.26-0.31 |
| T_narrow | 30-39% | 17-29% | about 21% | 3.4-4.5% | 0.22-0.31 |
| Y | 18-21% | 10-14% | about 12% | 0.3-1.8% | 0.22 |
| D | 12-14% | 0% | 0% | 0% | 0.14-0.26 |

The net clipped advantage per group is close to zero (±0.01), so clipping does
not introduce a meaningful bias. Gradient norms are 0.006-0.37, well below
`max_grad_norm = 1`, so gradient clipping never acts.

### 1.4 Budget and update counts

- Prefix replay makes up **84-85%** of counted training controls in every run.
  Candidate branches make up about 15%. For example, one nft-dense-lr1e-5
  snapshot had 644,259 training controls, of which 545,216 were replay and 95,395
  were branches.
- 100 updates give about 416 groups (about 104 per shape), 3,300 scored plans and
  about 400 Adam steps of 16 records each. That is about 4 optimizer steps per
  update.
- Every long run trains T, D, T_narrow and Y. **L is evaluation-only**
  (`configs/nft_nearmiss_batched.json`, D034). No L improvement is expected, and
  the lack of one says nothing about learning.

### 1.5 The probe

`tdyssey/denoising_probe.py` uses **one plan per shape and one noise draw per
timestep** (t = 0, 25, 50, 99). Baseline epsilon errors are 0.256, 0.004, 0.0008
and 0.00008 at those timesteps. The t=99 ratio (5-50x, up to 450x) is large
because its denominator is tiny. Absolute changes at t = 0, 25 and 50 are similar
or larger. For example, AF dense 1e-4 rises by 0.039 at t=0, 0.033 at t=25 and
0.008 at t=99.

The t=99 change has almost no effect on behaviour. Using the saved schedule
(`squaredcos_cap_v2`, `clip_sample=true`), ᾱ₉₉ ≈ 2.4e-7. The predicted clean plan
at t=99 is (x - ε̂)/4.9e-4, which is clipped to ±1 both before and after training.
It enters the next sample with weight 0.016, against added noise with standard
deviation 1.0. Every probe shape, including the untrained L, drifts by a similar
absolute amount (about 5-9e-3 for nft-dense-1e-5). The change is global, not
specific to the shapes being trained.

## 2. Ranked causes

### C1. The signal per optimizer step is too weak, and the optimizer converts noise into drift at a fixed rate. Confidence: high for "no improvement", medium-high for "degradation".

Where the noise comes from:

- **Score noise from follow-up draws.** In `policy_branches.py:147-152`, each
  candidate's follow-up plans come from its own random stream (`generators[i]`).
  Candidates therefore differ by independent continuation noise as well as by
  their own plan. Corrected plan share is 0.06-0.29 (§1.3).
- **Groups with no signal.** 12-39% of groups are fully tied (§1.3). Under NFT, a
  tied group contributes only the β²‖δ‖² term, which is zero at the start of each
  update.
- **Loss estimated from one sample.** In `forward_policy.py:196-197`, each record
  gets one (t, ε) draw per epoch (two in total), and t is uniform over 0-99. A
  single draw is a very noisy estimate of a plan's likelihood gradient.
- **Half of each plan is never executed.** In `policy_branches.py:164`, the record
  stores the full 16-step plan (`clean[i:i+1]`), and the loss averages error over
  all 16×2 coordinates (`forward_policy.py:30-31,36`). Only indices 1-8 are
  executed (`forward_policy.py:118,144`). Index 0 and indices 9-15 add gradient
  variance and carry no information about the score.
- **Very few comparisons.** About 100 groups per shape per run give, at a plan
  share of 0.06-0.29, the equivalent of only tens of "clean" comparisons per
  shape. That is tiny next to 1.65M LoRA parameters or 251.5M dense parameters.
- **Adam's step size.** In `forward_training.py:226`, Adam (constant learning
  rate, β₂=0.999) moves each parameter by about 0.2-1 × lr per step whatever the
  gradient's information content. Even with zero signal the weights keep moving.
  Drift grows with learning rate × number of updates, as observed in both
  objectives.

Evidence for:

- No improvement in any run.
- Drift grows with learning rate and number of updates, in both objectives.
- Drift is shape-agnostic (§1.5), which fits undirected movement.
- Signal is weakest on Y and T_narrow, the shapes where gains are hoped for.

Evidence against or open:

- There is no null control yet. A systematic harmful direction (C4, C5) could add
  to the noise. Experiment R2 separates the two.

### C2. Nothing keeps the policy near the initial policy. Confidence: high that the anchor is missing; it is a necessary fix, not a sufficient one.

- **The per-update trust region hardly acts.** `rollout_ema = 0`
  (`forward_protocol.py:27`; `refresh_old`, `forward_policy.py:155-163`), so the
  old policy equals the actor at the start of every update. Only 4 Adam steps
  follow, so δ = actor − old stays small (about 0.007 per coordinate per update
  for dense 1e-5) compared with a·(old − ε), which is about 0.01-0.15. NFT's β²δ
  restoring term and AF's (a+A)δ term are therefore weak. In practice both
  objectives reduce to the same advantage-weighted epsilon regression,
  ∝ Σ aᵢ(oldᵢ − εᵢ)·∇prediction, and nothing limits how far the policy moves
  over many updates.
- **The T anchor is too weak to act.** `reference_weight = 0.001`
  (`forward_protocol.py:26`, used at `forward_policy.py:214-224`). Its gradient is
  exactly zero at the start, because actual equals reference, and it grows only
  with drift. To balance the main gradient, ‖actual − reference‖ would have to
  reach about 1000·a·‖old − ε‖. The ratio of 1e-4 between the losses shows it
  never does.
- **The anchor covers T only.** It uses T contexts and reference plans only, so
  T_narrow, D, Y and L have no anchor at all.

Evidence:

- Drift rises monotonically with updates.
- The dense AF 3e-5 run jumps from drift 0.10 at update 50 to 1.35 at update 75,
  with no restoring force.
- NFT's own loss, which is dominated by the old model's denoising error on its
  own samples, roughly doubles in the faster runs (0.010 to 0.022 for dense 3e-5;
  0.013 to 0.030 for dense 1e-4). The model is getting worse at denoising its own
  samples.

### C3. The score is misaligned with success and is often uninformative. Confidence: medium. The plumbing is correct, but the target is weak.

- **The shaping cancels inside a group.** `shaped_reward` and `block_return`
  (`rl_protocol.py:67-73`) sum to γᴺ·c_N − c₀, and c₀ is shared by the whole
  group. So the advantage is just the coverage after 32 controls, or the success
  bonus γᵏ⁻¹. Progress along the way does not count.
- **The candidate controls only a quarter of the window.** Its own plan sets 8 of
  the 32 controls (`policy_branches.py:143-162`). The correlation between coverage
  after those 8 and the final score is only 0.14-0.31.
- **The coverage potential is flat in many contexts.** It is zero or constant
  whenever the object does not overlap the goal or is not touched (9-29% of
  groups on T and T_narrow). In about 8-21% of tied groups the object moved, but
  coverage could not register it (§1.3).
- **Success is rarely seen.** Candidates succeed inside the window 0-4.5% of the
  time. The failures that matter are near misses at 0.92-0.94 peak, which need
  fine alignment. Between a 0.94 near miss and a success at step 31, the score
  differs by only about 0.06.

Evidence:

- The tie, zero-coverage and success rates in §1.3.
- The near-miss peaks in the baselines.

Evidence against:

- We do not yet know whether choosing candidates by this score would raise
  success at all. Experiment R3 answers this directly.

### C4. Timestep weighting spends most of the gradient on parts that matter least. Confidence: low-medium; an inefficiency, not a bug.

- **Small timesteps dominate the gradient.** With uniform t in epsilon space,
  each sample's NFT/AF gradient scales with ‖old − ε‖. That is about 0.5 at t=0,
  0.06 at t=25, 0.03 at t=50 and 0.01 at t=99, so samples with t < 25 supply
  roughly 70% of the gradient magnitude. Those are the samples whose residual is
  least informative about plan quality.
- **High timesteps cannot matter.** Clipping makes t ≥ ~95 behaviourally inert
  (§1.5), yet those timesteps still share all their weights with the rest.
- **What would help.** Reweighting t, or restricting it to the range where the
  plan's shape is decided (roughly 5-90), is a cheap variance reduction. I would
  not expect it to fix the problem on its own.

### C5. AdvantageFlow's push on the worst candidate is almost unbounded. Confidence: low-medium. The mechanism is certain; its contribution is unproven.

- **The mechanism.** In `advantage_flow_loss` (`forward_policy.py:35-38`), the
  per-sample curvature is a + A. For the most negative candidate,
  a = −max|a| and A = max|a| + 0.01, so the curvature is
  `anchor_margin` = 0.01 (`forward_protocol.py:25`). Its unconstrained optimum
  sits about 100·|a|·(old − ε) away from the old prediction. The loss pushes
  "make my own bad samples harder to denoise" with almost no restoring force.
  Raising denoising error is easy to achieve by damaging the model globally.
- **Why it is mild here.** With 4 steps per update the effect is small, but it
  accumulates over updates.
- **Weak evidence (one seed each).** At matched learning rates, dense AF lost
  more T than dense NFT. At 1e-5, AF reached T 0.15-0.20 by updates 50-100,
  against NFT's 0.40-0.55. At 3e-5, AF's drift reached 2.05 against NFT's 0.76.
- **NFT behaves similarly at the start.** In the regime where δ ≈ 0, NFT's
  negative branch has the same gradient direction at the start of each update.
  Its curvature β² = 0.25 bounds it, but its target displacement is still
  2|a|·(ε − old).

### C6. The adaptable parameters cannot change how the policy uses its observation, and the shapes are coupled. Confidence: low.

- **LoRA skips the observation pathway.** `inject_lora` (`lora.py`) wraps only
  Conv1d layers. The FiLM conditioning Linear layers, which take the observation,
  stay frozen. Shape-specific changes can only arrive through kernel edits that
  apply in every context. That fits the shape-agnostic drift (§1.5).
- **Shapes share the same updates.** Each update holds one group per shape, and
  every minibatch mixes shapes whose groups differ in how informative they are.
  D has the largest mean |advantage| (about 0.36) but among the lowest
  coverage-at-8 correlation.
- **Minor scaling quirk.** The final Conv1d has 256 inputs and 2 outputs, so the
  LoRA rank is clipped to 2 (`lora.py:12-13`) and its scale becomes 8/2 = 4. The
  output layer's adapter therefore moves about 4 times faster than the others.
- I cannot separate C6 from C1 with the existing data. Experiment R10 does.

### C7. Measurement artefacts distort the picture. Confidence: high. These do not cause degradation, but they cause misreadings.

- **Small, chaotic evaluations.** With 20 scenes per shape, a single baseline draw
  and strict 0.95 success, a single evaluation can only resolve differences of
  about ±0.15. `best.pt` is chosen on these same 20 scenes, so it mostly selects
  noise.
- **A misleading probe.** The one-plan probe and its t=99 ratio are misleading
  (§1.5), and it does not report action-space drift.

### Minor issues found

- **Warmup success drops contexts.** If the policy succeeds during warmup, the
  slot is consumed with no records (`policy_branches.py:193-195`). This biases T
  training toward harder scenes and skips some update slots.
- **Contexts are serially correlated.** There are about 15 training scenes per
  shape per run, each supplying about 7 contexts.
- **Higher-lr dense runs.** The dense 1e-4 runs did not fail numerically; they
  were stopped (SIGTERM) after collapsing.

### Checked and found correct

- Update signs: NFT and AF raise the likelihood of candidates with positive
  advantage.
- Records are aligned with candidate order and scores.
- Plans execute indices 1:9.
- The reference plan does not disturb the candidate random streams.
- Old, actor and reference views work in both LoRA and dense modes.
- The image encoder stays frozen.
- `forward_noise` uses the same schedule as the sampler.
- Tied groups get exactly neutral weights.
- The fixed score scale avoids amplifying noise in near-tied groups. This is
  better than dividing by the group's standard deviation.

## 3. Ranked changes

Cost basis: about 900 simulator controls per second for the whole machine
(about 3.2M controls per hour). One 100-step denoising pass takes about 2.2 s at
batch 1 and 2.6 s at batch 32 (`docs/STATUS.md`). The current 8-job mix achieves
about 1,600 groups per hour in total.

| # | Change | Where | Expected effect | Cost | ~1 h experiment that confirms or refutes |
|---|---|---|---|---|---|
| R1 | **Fix measurement.** Use 100 scenes per shape. Evaluate the frozen baseline with 3-5 sampling seeds. Report peak/final coverage, success at 0.90/0.93/0.95 and time to reach 0.9. Use paired per-scene tests. Pick `best.pt` on separate scenes. | `forward_training.py` evaluation config; `policy_training.evaluate_batched` | Makes ±0.05 changes visible; settles the T_narrow and D baseline questions | About 140k controls (about 3 min of simulation) plus batched denoising: about 5-10 min per checkpoint | Frozen policy × 5 seeds × 100 scenes for T, T_narrow, Y, D gives the baseline spread. Re-evaluate `advantage_flow-dense-lr3e-6` (update 50) and `nft-dense-lr3e-6` (update 75) with the same seeds. If T_narrow matches the baseline, its "fall" was noise. |
| R2 | **Null controls.** Shuffle advantages within each group (same sizes, no information). Also run positive-only and negative-only versions. | `policy_branches.py:244-247` (feedback assignment) | Shows whether the signal does anything and whether the negative branch drives the damage | 4 concurrent runs × 50 updates ≈ 40-60 min | Dense 3e-5 (or 1e-5): real, shuffled, positive-only and negative-only, with 100-scene T and T_narrow evaluations at updates 25 and 50. If shuffled tracks real on drift and T, C1 is confirmed. If negative-only causes most of the drift, C5 matters; drop or cap the negatives. |
| R3 | **Upper bound from lookahead.** At evaluation, sample N=8 plans at every decision, score each by simulator lookahead with the same 32-control rule, and execute the best. Only lookahead copies restore from a snapshot; the executed episode never restores. | New evaluation script reusing `BranchCollector` scoring | One group-relative step can only move the policy toward this distribution, so this bounds what NFT or AF can learn from the current score | About 32× evaluation controls: 50 scenes × 3 shapes ≈ 1.5M controls ≈ 30 min, plus about 10 min of denoising | Y, T_narrow, D. If best-of-8 does not raise success or peak coverage, the score is the problem: do R6 before any learning. If it does, use it as a teacher (R9). |
| R4 | **Less noisy scores.** Share follow-up random streams across the candidates of a group (common random numbers). Optionally use a deterministic DDIM-16 continuation, which also cuts continuation sampling about 6×. Average K ≥ 2 follow-ups. Later, add a privileged state-based critic and score as 8 controls + γ⁸V(s₈). | `policy_branches.py:136-152` | Plan share could rise roughly from 0.06-0.29 to 0.4+ (to be measured). This attacks C1 directly. | Common random numbers are free. K follow-ups cost K × 24 controls plus replay unless R8 is done. A critic is moderate engineering. | Copy `tools/score_snr.py` and make the follow-up generators shared per draw index across candidates (and a deterministic-continuation variant). About 20 min per variant. Compare the corrected plan share and split-half r per shape. |
| R5 | **Less noisy loss.** Use the same (t, ε) for all 8 candidates of a group, so the part that does not depend on the plan cancels when advantages sum to about 0. Take M = 8-16 (t, ε) draws per record per step. Count loss only on executed indices 1-8. Restrict t to about 5-90 or reweight it. | `forward_policy.py:185-224`, `nft_loss` / `advantage_flow_loss` | Several times less gradient noise per step at negligible GPU cost (the U-Net runs on 16×2 inputs) | GPU only, minutes | On one stored batch of groups (regenerate conditions by replaying stored prefixes), compare the cosine similarity of gradients from two disjoint halves of the draws, and from two disjoint sets of groups, before and after. A higher split-half gradient agreement means a usable signal. |
| R6 | **A score that tracks success.** Add a privileged, disclosed potential based on pose error (position plus angle), or combine coverage with distance, plus a term for closeness to success. Concentrate contexts near the goal or in contact, taken from the policy's own episodes (a disclosed selection, not a geometric reset). | `rl_protocol.shaped_reward`; context choice in `BranchCollector.collect` | Recovers signal in 8-21% of currently tied groups. Rewards the fine alignment that Y and T_narrow lack. | Free to compute offline: rows already store object and goal poses | Re-score the stored candidate rows offline (zero simulator cost) to get tie rates. Re-run `score_snr` with the new potential to get plan share. An arbitrary 50 px ≈ 30° pose potential already cuts T_narrow ties from about 39% to 18% and Y ties from 21% to 9% (my offline check). |
| R7 | **Real anchor and step control.** Add a retention term ‖ε_actor − ε_ref‖² on every training input for all shapes, at weight 1-10 relative to the main loss (it acts like a KL penalty). Optionally set `rollout_ema` to 0.9-0.99 so NFT's quadratic term acts. Adapt the learning rate to a target action-space drift per update, measured in pixels on fixed contexts and noise. Take fewer, larger steps per update. | `forward_policy.py:214-224`, `refresh_old`, optimizer loop | Stops T and T_narrow degradation at learning rates high enough to learn. Needed in any case. | Negligible | Dense 3e-5 for 50 updates with retention weight 1 on all shapes, against the current 0.001. Compare drift and 100-scene T/T_narrow results. Add a probe that reports drift per timestep and in pixels on 32+ plans. |
| R8 | **More groups per hour.** Batch all 4 contexts' candidates in one denoiser call per chunk (32 rows cost about 1.2× one row). Replace full-prefix replay with an exact process fork, or a common deterministic snapshot restore for all candidates. Skip rendering during replay except the last 2 frames. Use DDIM for continuations. | `BranchCollector`, `replay_prefix`, training loop | 3-6× more groups per hour; replay share drops from about 85% toward 0 | Engineering | Benchmark one update each way. Check that G restored lanes produce identical rows for identical actions; matched comparison only needs a common start, not a trajectory-exact replay. Compare score distributions under replay and under restore at the same contexts. |
| R9 | **A learning rule with a cleaner signal: filtered self-imitation.** Keep the best candidate only when its lead exceeds a noise-calibrated margin, and fit it with the plain denoising loss plus the R7 anchor, with no negative branch. Optionally use R3 lookahead plans as targets (planner-to-policy). | New objective beside NFT and AF | Avoids "make bad samples harder to denoise" and needs far fewer comparisons | Same data, cheaper loss | Do only after R3 shows gains: 50 updates on Y alone, then 100-scene Y and T evaluations. |
| R10 | **Debug on one shape; let the adapter reach the observation pathway.** Train Y or T_narrow alone with the T anchor before mixing shapes. Add the FiLM `cond_encoder` Linear layers to LoRA, and fix the final layer's rank-2 scale of 4. | `config.shapes`; `lora.inject_lora` | Separates interference between shapes (C6) from C1, and lets adaptation depend on the observation | Same as a normal run | Single-shape run against the 4-shape run at the same learning rate, comparing T drift and T success. |

Suggested order: R1 → R2 and R3 in parallel → R4, R5, R6 and R7 together (all
cheap) → R8 to scale up → R9 if R3 is positive → R10 for the multi-shape stage.
Only R4, R5, R6 and R8 actually create more learnable signal per hour. R7 only
prevents damage. Without R1, no result can be read reliably.

## 4. Evidence that contradicts the obvious story

The obvious story: "RL fine-tuning makes the policy drift and forget T; use a
smaller learning rate or LoRA."

1. **The T_narrow decline at near-zero drift is mostly baseline luck plus the
   strict threshold.** Coverage metrics do not change. Lost successes peaked at
   0.88-0.95. The asymmetry (49 lost against 18 gained) leaves some room for a
   small real effect, so test it with R1.
2. **The D "gain" is the same effect in reverse.** The seed-42 D baseline is
   unusually low.
3. **L is never trained** in these runs, so "no L improvement" is not evidence
   about learning.
4. **The "largest change at t=99" headline comes from dividing by a tiny
   number,** and t=99 cannot affect behaviour because of clipping. Absolute drift
   is spread over all timesteps.
5. **The two objectives perform almost the same update** at 4 steps per update
   with a hard copy of the rollout policy. Their differences are weak evidence
   about loss design, even though AF looked somewhat worse (C5).
6. **The score diagnostic overstated the signal.** The corrected plan share is
   0.06 for Y and 0.08 for T_narrow.
7. **The drift is global, not specific to the trained shapes.** The untrained L
   drifts as much as the trained shapes. This looks less like "learning D erased
   T" and more like undirected movement shared by all shapes.
8. **Moderate drift loses T successes without losing pushing skill.** Peak
   coverage stays flat until drift exceeds about 0.3.
9. **Rising mean training scores** (for example T in the AF LoRA runs) are not
   learning evidence. The mean score contains −(1−γ³²)·c₀ ≈ −0.28·c₀, so it
   mostly tracks which contexts happened to be sampled.

## 5. Where my evidence is weak

- I did not measure action-space drift. How much a given epsilon drift moves the
  executed actions in pixels is inferred, not observed.
- The score diagnostic has 6 contexts per shape. The tie rates come from the last
  10 detailed updates of live runs and moved between snapshots.
- Every NFT-versus-AF and LoRA-versus-dense comparison uses one seed.
- C4 and C5 rest on how the losses work plus weak empirical support. R2 and R5
  are the tests.
- The pose potential I tried offline used an arbitrary 50 px ≈ 30° weighting and
  ignores rotational symmetry. It only shows that ties can be recovered, not that
  this potential is the right target.
