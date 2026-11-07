# Red-team review: value-scored candidates, anchoring and the no-regression test

Date: 2026-10-25. Role: skeptical reviewer of the five-part plan proposed after the
failed forward-process (DiffusionNFT / AdvantageFlow) runs.

What I read (read-only): `tdyssey/forward_policy.py`, `forward_protocol.py`,
`forward_training.py`, `policy_branches.py`, `rl_protocol.py`, the upstream
`pusht_env.py`/`pusht_image_env.py`, `manifests/official_dp.json`, the docs
`DIFFUSION_RL_RESEARCH.md`, `POLICY_LEARNING.md`, `FORWARD_POLICY_LEARNING.md`,
`LEARNING_PLAN.md`, and the session-2 outputs under `logs`
(`score_snr.json`, `lrsweep-s2-final.txt`) and `runs/long-s2/*/metrics.jsonl`.
No GPU jobs were run. Numbers marked **re-analysis** are my own arithmetic on
those saved files, not new experiments. Power numbers come from a small
simulation, with its assumptions stated where they are used.

---

## Bottom line

1. **The plan targets the wrong main problem.** It is built around stopping T from
   getting worse. The evidence says the bigger problem is that **nothing has learned
   anything on Y, D or L**. Runs that moved the policy a lot lost T and gained no target
   performance beyond the noise you get from evaluating the *same* frozen policy twice.
   Stronger anchoring can stop forgetting, but it cannot create a learning signal.
2. **Part 1 contains a mathematical trap.** The shaped reward is a difference of
   coverages (a "potential-based" reward), so its sum over time cancels down to the
   end values. If V is trained on the same reward, the candidate's own 8-control reward
   **cancels exactly** against V's coverage term. The ranking then depends only on the
   base policy's *future success*, plus a heavily discounted time-out coverage term. The
   base never succeeds on D or L, so on those shapes V carries almost no signal by
   construction.
3. **Parts 1, 2 and 4 can only reweight plans the base policy already proposes.**
   Nobody has checked whether those proposals contain behavior that solves L or D. A
   no-learning test answers this for all three parts at once: at each step, pick the best
   of N base samples using the real simulator. Run it first.
4. **Part 5 has the right idea at the wrong size.** Evaluation costs about 1% of the
   budget. At 100 scenes, the proposed "all three must pass" rule has only about a
   20–35% chance of accepting a candidate that is truly as good on T and clearly better
   on Y. Use about 300 scenes × 2 noise seeds, with a stated non-inferiority margin.

---

## Evidence used, including two corrections to current numbers

**The logs already contain an A/A test, and it is noisy.** "A/A" here means the same
frozen policy, the same 20 development scenes and only a different policy-noise seed
(the update-0 rows of seed-42 and seed-43 long runs):

| Shape | Seed 42 success / final cov. | Seed 43 success / final cov. |
| --- | --- | --- |
| T | 0.55 / 0.78 | 0.55 / 0.78 |
| T_narrow | 0.55 / 0.70 | 0.65 / 0.75 |
| Y | 0.05 / 0.55 | 0.15 / 0.58 |
| D | 0.00 / 0.27 | 0.00 / 0.37 |
| L | 0.00 / 0.34 | 0.00 / 0.31 |

So the baseline "Y 1/20" could just as easily have been 3/20, and D's mean final
coverage swings by 0.10 from noise alone. The lrsweep reading "AdvantageFlow 3e-4,
Y 7/20 at update 10" was one of about 16 noisy 20-scene readings and did not persist
in the long run (0.05–0.15 afterwards). Treat it as a multiple-comparisons mirage:
with enough noisy readings, one will look good by chance.

**Long runs (re-analysis):** across 11 runs and up to 4 periodic evaluations each, Y
ranged 0.00–0.25, D and L never succeeded, and D/L final coverage ranged 0.20–0.43.
That is inside or barely outside the A/A spread. One run (AdvantageFlow, dense,
1e-5) drifted D coverage to 0.40 while T fell to 0.15–0.20. That might be a weak
trade, but 20 scenes cannot separate it from noise.

**The "plan share" is overstated (correction 1).** `score_snr.py` measures "between"
as the variance of candidate means, each averaged over K = 8 follow-ups. Those means
still contain within/K of follow-up noise. After removing that bias (between − within/K;
re-analysis of `logs/score_snr.json`), the single-draw plan share is:

| Shape | Corrected plan share, mean (per-context values) | True between-plan sd |
| --- | --- | --- |
| T | 0.26 (0.00, 0.10, 0.15, **0.97**, 0.09, 0.22): median ≈ 0.12 | 0.000–0.014 |
| T_narrow | 0.09 (0.31, 0.01, 0, 0, 0, 0.24) | 0.000–0.027 |
| Y | **0.06** (0.06, 0.05, 0, 0.10, 0.14, 0.02) | 0.000–0.019 |
| D | 0.29 (bimodal: ~0.05 or ~0.5–0.6) | 0.001–0.049 |

The split-half correlations (0.14–0.29) are already unbiased and roughly agree. Only
6 contexts per shape were measured, so all of these numbers are rough. The
important point is the size of the signal: the score difference that a candidate's own
plan causes has a standard deviation of only about **0.005–0.015** in score units, and
it is often near zero. The configured `score_scale` is 0.05.

**The failed updates look mostly random (correction 2: a mechanism, not just "too
weak an anchor").** The fixed-probe drift (`probe/T/initial_policy_drift_mse`) grows
roughly in proportion to the number of updates: the fitted log-log slope is 0.6–1.2
for all NFT runs and for AdvantageFlow dense 1e-4/1e-5. Steps in random directions
produce exactly that pattern (squared drift ∝ updates). Steps in a consistent
direction would give squared drift ∝ updates² (slope ≈ 2). Two AdvantageFlow runs
(LoRA 3e-4: 1.6–1.8; dense 3e-5: 2.1) show some consistent direction, but no target
gain came with it. Caveat: this drift is measured on fixed probes in noise-prediction
space, so it is suggestive, not proof.

**The update design explains the failure on its own.** Each update used 4 contexts ×
8 candidates (about 6k controls, mostly replay), followed by Adam steps. Adam takes
steps of roughly fixed size however noisy the gradient is. With 4 low-reliability
contexts per step, the policy therefore moves at a speed set by the learning rate, in
a direction set mostly by noise. That predicts exactly what was seen: T falls in
proportion to drift, and the targets do not move.

**Replay dominates cost.** Prefix replay was 81% of counted controls. Every replayed
control renders a 96×96 image and computes a shapely overlap
(`PushTEnv.step` → `_get_obs`), although only the last two frames are ever needed.

---

## Failure modes, ranked by how likely they are to sink the plan

### 1. Wrong diagnosis: missing target signal, not only forgetting (very likely)

- **Why it fails:** parts 2–5 protect T. None of them makes the target-shape signal
  more reliable than the one that already failed. If the gradient on Y/D/L is mostly
  noise, a large anchor gives "base policy everywhere": safe and uninformative.
- **Cheapest decisive tests (no new method needed):**
  - *Gradient consistency:* at fixed weights, compute the update direction from two
    independent sets of contexts and report their cosine similarity, versus the number
    of contexts per update. A value near zero at the planned batch size means the
    updates are noise. This takes minutes of GPU time and no physical evaluation.
  - *Shuffled-score control:* the same pipeline and learning rate, with scores randomly
    permuted within each group. If T falls and the targets stay flat just as in the
    real runs, the real runs had no usable signal.
  - *Target-only run:* train on Y (or D) alone, with no T in training. If Y does not
    improve even when forgetting cannot matter, anchoring is irrelevant.
- **Change:** make scores more reliable (average follow-ups, give all candidates the
  same follow-up noise seeds, or use a validated scorer). Use far more contexts per
  update (≥64). Take fewer, better-informed steps.

### 2. Cancellation: V removes the candidate's own reward, and D/L get no signal (very likely for D/L)

With coverage Φ, the code's reward is `r_t = 1[success] + γ·Φ_{t+1}·(not done) − Φ_t`.
Over the candidate's 8 controls (assuming no success):

```text
sum_{t<8} γ^t r_t      = γ^8 Φ_8 − Φ_0                      (it cancels down to the ends)
V(s_8)                 = −Φ_8 + W(s_8, time left)
   W = E[γ^τ · 1(success)] + E[γ^(time left) · Φ_final · 1(no success)]
score = sum + γ^8 V    = −Φ_0 + γ^8 W(s_8, time left)
```

- `−Φ_0` is the same for every candidate in a group, so the candidate's own 8-control
  coverage gain **drops out entirely**. Only W ranks the candidates. Any
  potential-based reward has this property once you add a consistent value
  bootstrap. The "dense" signal of the current 32-control score exists only because
  that score is cut off without a value bootstrap.
- **D and L (0/20 successes):** W ≈ 0.99^~200 × E[final coverage | s_8] ≈ 0.13 × E[final
  coverage]. The ranking becomes "where the base policy ends up at the 300-control
  limit", discounted about 8×. That depends on the arbitrary time limit, so V needs
  time-remaining as an input. If the time limit is instead bootstrapped (treated as not
  terminal), W becomes success-only, ≈ 0 on D/L, and **the ranking is just V's error**.
- **Y (5–15%):** some success signal, but sparse.
- A learned V that represents −Φ imperfectly feeds its own error into the ranking.
  **Parameterize V = −Φ + Ŵ** (Φ is known exactly from privileged state) and learn only Ŵ.
- **Change:** if dense signal is wanted together with a value function, it must come
  from a reward that is *not* a potential difference. Each option below changes the
  objective and must be disclosed:
  - a fixed-horizon target, e.g. expected coverage 64 controls later;
  - a success-threshold curriculum for shapes the base never solves (coverage ≥ 0.6
    or 0.8 first);
  - time-integrated coverage, which rewards holding high coverage (the current shaping
    deliberately avoids rewarding that).
- **Test (proxy validity):** within each context, correlate each candidate's
  short-horizon score with its long-horizon outcome averaged over many follow-ups
  (final coverage, and success where it occurs). Report this per shape, with a
  threshold declared in advance. A proxy that does not predict the eventual outcome
  within a context should not be optimized.

### 3. The base policy's proposals may not contain the needed behavior (likely; untested)

- Part 1 ranks base samples. Part 2's KL anchor has a best policy of the form
  π_base · exp(advantage/λ) / Z, which cannot put probability where the base has none.
  Part 4's DSRL and best-of-N both pick among base outputs. All three reweight the
  base's own 8-control proposals. Recombining them across ~35 decisions *can*
  produce new episode-level behavior, so this is not hopeless. But nobody has measured it.
- **Decisive test: simulator-oracle best-of-N** (a privileged, training-time-only
  upper bound; label it as such). At each chunk boundary, sample N base plans, score
  each in the real simulator and execute the best. Run it in two forms: greedy (an
  8-control score, no follow-ups, fully deterministic) and lookahead (K follow-ups,
  using the horizon chosen in Phase A). If this oracle cannot clearly raise Y/D/L above
  the base, **no V, anchor or noise policy in parts 1, 2 or 4 can**. Its branch data is
  also exactly what part 1's validation needs.
- **Second oracle, outside the base's proposals:** a privileged sampling planner over
  raw pusher targets (CEM or MPPI, physics only, on CPU; the H100 node has 64 cores).
  It tells you whether L and D can be solved in 300 controls with this pusher at all.
  If the base's proposals turn out to be the limit, it also provides machine-generated
  (non-human) demonstrations for distillation.

### 4. V must be more accurate than the small signal it ranks (likely)

- To capture about half of the available best-of-8 gain, the error in **V differences
  between candidates from the same context** must be smaller than about the
  between-plan standard deviation: ≈ 0.005–0.015 (Y: 0.001–0.02). Global accuracy
  (MSE, R²) does not matter here. A V can explain 90% of the variation *across* scenes
  (which ones are easy) and still have no ranking skill *within* a context.
- Candidates differ exactly where V is weakest: the pusher ending 2 px on one side of
  an edge or the other, at a corner, sliding versus sticking. Outcomes jump there, and
  a smooth network on poses averages across the jump.
- Single-episode Monte Carlo targets are dominated by chaotic continuation noise:
  identical-seed rollouts diverge after 30–150 controls. A success target on T has
  variance of about 0.25, so resolving a 0.01 difference needs on the order of
  thousands of samples from nearly the same state. Full-episode base rollouts give one
  sample per visited state.
- **Optimizer's curse:** taking the argmax of 8 noisy V values picks candidates where
  V errs high. With normally distributed errors, the expected gain is about
  1.42·σ_Q²/√(σ_Q² + σ_V²), which collapses to about 1.42·σ_Q²/σ_V when V's error
  σ_V is much larger than the true spread σ_Q.
- **Is V of the base the right continuation value?** Only for *one* greedy step. The
  policy-improvement theorem says that picking the argmax of the base's exact action
  values over base samples is at least as good as the base. After the actor changes,
  V of the base is the value of the wrong policy. It also undervalues "setup" moves on L
  that only pay off if the follow-up is good, and the base's follow-up on L is not.
  Refit V on rollouts of the current policy each phase; that is cheap because ordinary
  rollouts need no replay. A *biased* V (for example, one that likes the pusher near the
  object) can make T worse even as a reranker.
- **Gaps in the proposed validation:**
  - The averaged Monte Carlo reference is itself noisy. With single-draw reliability ρ
    and K follow-ups, its reliability is Kρ/(1+(K−1)ρ): at ρ = 0.1 that is 0.47 for
    K = 8 and 0.78 for K = 32. Correct for this, or use K ≥ 32.
  - The horizons do not match. V predicts a full-episode return, whereas a 32-control
    Monte Carlo score measures something else. Validate against what V claims to
    predict, or against the oracle's selection gain.
  - Use **selection regret**, not a Spearman correlation over 8 items: the fraction of
    the oracle's gain captured by argmax V, where the oracle picks using half of the
    follow-ups and is scored on the other half, so that its own luck is not counted.
  - Report results separately for contact and no-contact contexts, near-goal contexts
    and each shape. Declare the pass threshold in advance (e.g. ≥ 50% of the oracle
    gain on every target shape).

### 5. Statistics: the rule is underpowered, and checkpoints are selected on 20 noisy scenes (likely to waste the result)

Simulated power, paired by scene. Per-scene success probability varies across scenes
(intra-scene correlation 0.2–0.5); tests are one-sided at α = 0.05:

| Design | T non-inferiority, margin 0.10, true difference 0 | margin 0.05 | Y 5%→15% | D/L 0%→5% (exact sign test) |
| --- | --- | --- | --- | --- |
| 20 scenes × 1 seed | 0.19–0.26 | 0.11–0.14 | 0.31 | ≈ 0 |
| 100 × 1 | 0.47–0.64 | 0.17–0.26 | 0.85 (5→10%: 0.48) | 0.56 |
| 100 × 3 seeds | 0.87–0.96 | 0.38–0.54 | | |
| 300 × 1 | 0.87–0.97 | 0.43–0.53 | 1.00 (5→10%: 0.84) | 1.00 (0→2%: 0.72) |
| 300 × 2 | 0.99–1.00 | 0.61–0.77 | | |

- The rule requires T, T_narrow and the targets all to pass. That conjunction
  (an intersection-union test) is valid at α without correction, but the powers
  multiply: at 100 scenes with a 0.10 margin, roughly 0.5 × 0.5 × 0.85 ≈ **0.2–0.35**.
  The rule will almost always "ship the base". That is safe, but it makes a real
  improvement nearly undetectable.
- "Not worse" must mean **non-inferiority with a stated margin**. If it is read as "no
  significant drop", a true 10-point T loss slips through about half the time at
  100 scenes.
- Evaluation is cheap: 300 scenes × 5 shapes × ≤ 300 controls ≈ 0.45M controls per
  policy and seed, roughly 10–20 minutes of the machine. There is no reason to stop
  at 100.
- On D and L, use **mean final coverage** (or another coverage endpoint declared in
  advance) as the primary endpoint. On shapes with 0% baseline success, success is a
  cliff that may not move even when behavior improves.
- Selecting `best.pt` from 20-scene evaluations every 25 updates across many runs is
  noise-driven, and the selected score is inflated. Select on ≥ 100 development
  scenes. Test only candidates chosen in advance (≤ 2–3), with a Holm correction
  across candidates.
- Fix in advance: sampler mode (batched vs sequential, which changes random streams),
  noise seeds per scene, and the test-seed range. Matched resets give **all shapes the
  same initial pose for a given seed**, so exclude test seeds from training and
  development resets for every shape.
- Run the decision rule on base vs base on the test set. Its target-superiority part
  should pass at most about α of the time.
- One checkpoint passing the test shows that *the checkpoint* is better. A claim that
  *the method* works needs ≥ 3 training seeds.

### 6. Anchoring (part 2): it freezes target learning and weights denoising steps poorly (likely)

- Anchoring the *target shapes* to the base pulls against the change you want. With
  advantages of a few hundredths and a "much larger" weight λ, the exp(advantage/λ)
  tilt is negligible, so learning freezes.
- **An anchor limits how far the policy moves, not how good the direction is.** In a
  simple noisy-gradient model with an anchor, the useful steady shift scales as 1/λ
  while the random spread scales as 1/√λ. Stronger anchoring shrinks both and caps
  what more updates can achieve. Only better scores, more contexts per step or
  smaller steps improve the ratio of useful movement to random movement.
- **A uniform noise-prediction penalty is a poorly calibrated trust region.** The
  checkpoint uses `squaredcos_cap_v2`, `fixed_small` variance and `clip_sample=True`.
  An error in the noise prediction ε turns into a clean-plan error multiplied by
  √((1−ᾱ_t)/ᾱ_t): 0.04 at t = 1, 1.0 at t = 50, 2.5 at t = 75, 16 at t = 95, and about
  2000 at t = 99 (where clipping then saturates it). The exact per-step Gaussian KL
  weight, β_t / (2·α_t·(1−ᾱ_{t−1})) before clipping, is U-shaped: ≈ 0.89 at t = 1,
  ≈ 0.03 at t = 50, ≈ 1.5 at t = 98, ~500 at t = 99. The final step t = 0 is
  deterministic, so its KL weight is infinite; use a squared-error penalty there. A
  uniform-ε penalty therefore under-penalizes the high-noise steps, which is exactly
  where the observed drift sits. A correctly computed per-step KL (including clipping)
  is the better of the two options the plan lists.
- **Penalize what predicts T loss.** Measure action-space drift of sampled plans under
  fixed noise on T and T_narrow probe contexts; the team has already observed that T
  loss is proportional to drift. Set an explicit drift budget and stop or reject updates
  that exceed it.
- **Recommendation:** anchor T *and T_narrow* strongly on contexts the *current* actor
  visits (the code does this for T only). Give target shapes only a trust region
  relative to the previous iterate. Sweep at least 2–3 anchor weights and report the
  trade-off curve (T success versus target gain), not one hand-picked weight.
- **Retention upper bound to report:** per-shape adapters chosen by an image-based
  shape classifier (the letter is obvious in the image). This keeps T intact by
  construction and is still one image-based policy. It separates "can we learn
  targets at all?" from "can one set of weights hold every shape?"

### 7. Frozen-base methods (part 4) are not safe by construction, and DSRL does not fit this sampler as-is (likely)

- "Cannot degrade the base" holds only for the base *weights*. A residual policy, a
  noise-steering policy or a reranker also changes the *executed actions on T* unless
  it is gated off for T. These methods need the same T non-inferiority test.
- **DSRL with the original DDPM sampler barely uses the 16×2 starting noise.** At the
  first step (t = 99, β = 0.999):
  `x_98 = 0.0316·x_99 + 0.0156·clip(x̂_0) + 0.9994·z`.
  The starting noise contributes about 3%, fresh noise about 100%, and 99 more fresh
  noise injections follow. Steering only the starting noise requires a deterministic
  sampler (DDIM with η = 0), which is a *different* policy needing its own
  T/Y/D/L baseline. The alternative is to steer all 100 × 32 injected noise values.
- **Expressivity:** under DDIM, the 32-dimensional noise maps onto the base's own plan
  distribution. It can recombine base chunks across decisions, but it cannot produce
  plans the base gives negligible density; pushing the noise outside its usual range
  produces junk. The oracle best-of-N test bounds what it can achieve. DSRL also needs
  its own critic: more machinery, more budget.
- **Best-of-N reranking at deployment, with a privileged V, is not an image-based
  actor.** It needs object pose and a simulator or world model to predict each
  candidate's next state. A deployable version needs an image-based Q(image history,
  plan) that passes the same within-context ranking test, which is unlikely within
  budget. Keep reranking as a training-time teacher or a labeled oracle.
- Residual RL on frozen encoder features is possible but needs many samples in a
  chaotic contact task. It is out of budget unless the oracle shows that small
  corrections are what is missing.

### 8. The frozen image encoder may not see L/D pose well enough (plausible; missing test)

- The encoder was trained only on T images and stays frozen in every variant. If its
  features do not encode L/D/Y pose precisely, no amount of action-network training
  can fix that.
- **Test (minutes):** fit a small probe from frozen features to object pose (x, y, θ),
  on held-out episodes from play or base rollouts, and compare T with L/D/Y. If L/D
  pose errors are much larger than T's, the encoder is a bottleneck. Then either
  fine-tune the encoder (a separate implementation) or restrict the action-only study
  to shapes the encoder already represents well.

### 9. The cost model: V saves little while replay dominates (certain; affects everything)

- Per context, V-based scoring costs 8·(p+8) controls versus 8·(p+32) now. At p ≈ 150
  that is only **about 15% fewer controls**. The main benefit of V is lower variance,
  not lower cost. Averaging K Monte Carlo follow-ups multiplies the replay cost K-fold.
- **The highest-leverage engineering change before any new method is cheaper
  branching:**
  - *Physics-only replay:* skip rendering and overlap computation until the last two
    controls, and verify state at the end. Profile first.
  - *Approximate restore:* restore positions, velocities and the pusher target, which
    loses the warm-started contact impulses. This is acceptable **for training scores
    only**, and only if its divergence is measured to be well below the divergence that
    policy noise already causes. Identical seeds diverge within 30–150 controls anyway,
    so exactness buys little for scoring. Never use it for evaluation.
  - *Measurement:* compare the score distribution under exact replay versus
    approximate restore at the same contexts.
- Keep counting every replay, branch and plan in every comparison, as the reports
  already do.

### 10. Gaps in the V specification (likely to cause silent errors)

- **Time remaining.** Episodes are capped at 300 controls and truncated without a
  bootstrap, so V changes with time. Include time as an input, or define a target that
  does not depend on it.
- **Velocities** (object linear and angular, pusher), the pusher's PD target (last
  action) and the previous pose. The base policy conditions on two frames.
- **Chunk phase.** Train and use V only at chunk boundaries. States in the middle of a
  chunk have a different continuation (the rest of a stale plan).
- **Contact features:** pusher position in the object's frame and distance to the
  nearest edge. Use an ensemble of at least 5 networks to flag uncertainty.
- A shape one-hot is fine for adaptation, but it rules out any claim about unseen
  shapes.
- Separate the success component, and use V = −Φ + Ŵ (§2).

### 11. Training only the executed plan steps (part 3): small upside, real risk (low priority)

- The U-Net denoises all 16 positions together through temporal convolutions, so
  unexecuted positions influence the executed ones during sampling. Removing them
  from the loss leaves them free to drift, which changes the executed actions at
  sampling time.
- If tried, mask only the advantage term and keep the anchor (to the old or base
  policy) on all 16 positions. The expected benefit is small, because the dominant
  noise is in the scores, not in the labels of the unexecuted tail.

---

## Missing baselines and controls

| Control | Why it matters | Cost (≈900 controls/s, 8 jobs) |
| --- | --- | --- |
| A/A: base vs base, different noise seeds, same scenes | Calibrates noise and test size; checks the decision rule's false-positive rate | ~0.5M controls, < 15 min |
| Inference-only settings: execute 4 of 16 (replan more often); lower DDPM noise or DDIM | A trivial change that helps targets must be beaten by any learning method | ~0.5–1M controls each |
| Shuffled-score training | Separates learning from noise-driven drift | one run |
| Anchor-only (zero advantage) | Checks that the training machinery alone does not damage T | one short run |
| Target-only training (Y or D, no T) | Tests whether the method learns targets when forgetting cannot matter | one run per shape |
| Oracle best-of-N with the simulator | Upper bound for parts 1, 2 and 4; validation data for V | 2–8 h |
| Privileged planner outside the base's proposals (CEM/MPPI) | Shows whether L/D are solvable in 300 controls; provides a machine teacher | CPU only |
| Per-shape adapters + image shape gate | Upper bound for retention | one run |
| Privileged-critic tail-PPO (existing trainer + the new V) | Standard asymmetric actor-critic; needs no replay; uses the same V | one run |
| Frozen-encoder pose probe | Rules out a perception bottleneck on L/D | minutes |
| ≥ 3 training seeds for the chosen method | Measures seed-to-seed variability | 3× one run |

## Minimum experiment set for credible course-project conclusions

If the gates below fail, items 1–3 alone still make a solid, honest negative-result report.

1. A/A comparison and 200-scene development baselines for all five shapes, with
   confidence intervals.
2. Plan share, bias-corrected, versus scoring horizon (8/16/32/64 controls), with
   shared versus independent follow-up noise and 30+ contexts per shape. Include the
   proxy-validity check (§2).
3. Oracle best-of-N upper bound on Y, D and L, with T as a check.
4. Only if item 3 passes: one scorer (a validated V or averaged Monte Carlo scores) plus
   one distillation method, versus the shuffled-score control, with 3 seeds, a trade-off
   curve over at least 2 T-anchor weights, and the per-shape-adapter bound.
5. One test registered in advance: about 300 fresh scenes × 2 noise seeds per shape,
   paired, with a non-inferiority margin on T and T_narrow and a target endpoint stated
   before the run.

---

## Revised plan for about 60 GPU-hours, with go/no-go checkpoints

Hours are wall-clock time on the H100 running about 8 concurrent jobs. Estimates
assume about 900 controls/s overall and exact replay unless stated otherwise.

**Phase 0 (development and CPU time; before GPU hours).** Implement cheaper
branching and profile it (§9). Write the registered decisions into `docs/DECISIONS.md`:
endpoints, margins, α, the test-seed range excluded for every shape, sampler mode and
the decision rule.

**Phase A: measure before training (~4 h).**
- A1: A/A comparison and 200-scene baselines for T, T_narrow, Y, D and L (~0.5 h).
- A2: inference-only settings, replanning every 4 controls and lower noise or DDIM
  (~1 h).
- A3: plan share versus horizon; shared versus independent follow-up noise; exact
  replay versus approximate restore; 30 contexts per shape (~1.5 h). Proxy validity
  uses the same data.
- A4: encoder pose probe, plus gradient consistency of NFT/AdvantageFlow at the base
  weights, versus contexts per update (~0.5 h).
- **Gate A:**
  - If an inference-only setting improves targets while T stays non-inferior, it
    becomes the baseline to beat (and is itself a finding).
  - Choose the horizon and noise sharing that give the highest reliability.
  - If the encoder probe fails on L/D, drop L/D from the action-only study and focus
    on Y and T_narrow.
  - Do not start any long run until the gradient cosine is clearly positive at the
    chosen batch size.

**Phase B: upper bound (~8 h).**
- B1: greedy oracle, best of 16, on 30 development scenes × {T, Y, D, L}: ≈ 3–4 h with
  exact replay, ≈ 1 h with physics-only replay.
- B2: lookahead oracle (N = 16, K = 4, horizon from A3) on 30 scenes × {Y, D}. This
  needs cheaper branching: about 1.5 h with approximate restore, otherwise about 8 h,
  in which case run Y only on 20 scenes.
- B3, in parallel on CPU: the privileged planner's feasibility on L and D.
- **Gate B** (declared in advance): GO if the oracle raises Y success by ≥ 15 points
  or target mean final coverage by ≥ 0.10 (paired scenes) without hurting T.
  - If the oracle fails but B3 solves L/D, switch to **planner → image-policy
    distillation** (DAgger-style: the image policy acts, the planner labels the states
    it visits). This uses machine-generated labels and no human demonstrations; the
    privileged teacher must be disclosed.
  - If both fail, stop and write the negative result, with the upper bound, the
    encoder probe and the reliability measurements as evidence.

**Phase C: scorer (~6 h).**
- C1: about 1500 base episodes per shape (≈ 1 h). Train an ensemble V = −Φ + Ŵ with
  time and velocity inputs, or with the alternative target declared in §2.
- C2: within-context selection regret on held-out Phase-B contexts, broken down by
  contact, near-goal and shape.
- C3: physical check, a simulator best-of-N reranked by V on 30 development scenes.
- **Gate C:** use V if it captures ≥ 50% of the oracle's selection gain on each target
  shape *and* reproduces ≥ 50% of B's physical gain. Otherwise use averaged Monte
  Carlo scores with shared follow-up noise, which B has already validated.

**Phase D: learn into the image actor (~24 h).**
- D1 (6 h), target-only, on Y and D:
  - Fit the denoiser only to the top 1–2 of N plans per context (reward-filtered
    regression, the simplest stable form of distillation), and compare with NFT on the
    same scores.
  - Use ≥ 64 contexts per update, seeded with Phase-B branch data.
  - **One-step improvement test first:** after one large update, sample from the
    updated and base actors at 64 held-out contexts and compare averaged Monte Carlo
    scores, using the same noise for both. If they do not improve, stop: the update
    step itself is not working.
  - **Gate D1:** a target gain on 100 development scenes beyond the A/A noise, worth
    ≥ 30% of the oracle gain. Otherwise report the gap between oracle and learned
    policy.
- D2 (12 h), retention:
  - Shared LoRA with a strong on-policy T and T_narrow anchor, at 2–3 weights, versus
    per-shape adapters with a shape gate, plus the shuffled-score control.
  - Monitor action drift on T probes throughout. Stop early if T on 100 development
    scenes falls below the non-inferiority margin.
- D3 (6 h): 3 training seeds of the chosen configuration. Try the executed-step
  masking ablation only if time is left.

**Phase E: final test (~3 h).** About 300 fresh scenes × 2 noise seeds × 5 shapes, for
the base and at most 2 candidates. Apply the registered rule (a non-inferiority margin
on T/T_narrow, the target endpoint, Holm across candidates). Report confidence
intervals and the A/A result on the test set.

**Reserve: ~15 h** for reruns and failures (4 + 8 + 6 + 24 + 3 = 45 h planned).

Why this order: Phase B costs less than building a value function and decides whether
parts 1, 2 and 4 can work at all. It also produces the data needed to validate V.
Phase A protects every later conclusion from evaluation noise.

## Checklist to register before running

- The primary endpoint for each shape, including a coverage endpoint for D/L; the
  non-inferiority margin (e.g. 0.10 absolute on T and T_narrow); one-sided α.
- The conjunction rule plus a Holm correction across candidates; what "ship the base"
  means.
- The test-seed range, excluded from training and development resets for **all**
  shapes.
- Sampler mode, batch composition and noise seeds per scene; checkpoint selection on
  development scenes only.
- Gate thresholds for A–D, written down before the gated result is seen.

## What the plan gets right

- It validates V against within-context Monte Carlo scores before using it. Make the
  metric selection regret with a split-half oracle.
- It uses a privileged critic while keeping the deployed actor image-based. That is
  allowed if disclosed.
- It uses a fresh, paired test set with "ship the base" as the default. It just needs
  enough power and a stated margin.
- It considers frozen-base methods. They are valuable as upper bounds and baselines,
  but not as "safe by construction" policies.
