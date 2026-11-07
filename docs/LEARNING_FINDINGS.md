# Policy learning findings, 2026-10-24 to 2026-11-06

Status: in progress. Every individual run, including abandoned ones, is listed in the
[experiment log](LEARNING_EXPERIMENT_LOG.md). A learned-world-model planning teacher solves all D, most Y and
about 40% of L validation scenes in the real simulator. Distilled into the image
policy and blended into the base, it gives the first image policy that passes the
pre-registered no-regression test: on 2000 fresh scenes per letter, Y success rises from
8.6% to 14.8% and D from 0.3% to 1.1%, while T and near-T stay within the margin (D042,
97.5% intervals), and an independent replication on a third fresh pool passed again
(D043: Y 9.8% -> 14.1%, D 0.1% -> 0.9%, T +1.8, near-T +0.9). Three earlier tests of other
blends did not pass because of T; all are reported. Negative results are
kept below on purpose. Code is in `scripts/learning/` (see its
[README](../scripts/learning/README.md)); decisions are D035-D040 in
[DECISIONS.md](DECISIONS.md). A 25-minute narrated explainer video was made from
these documents (kept outside Git).

## Summary

1. Forward-process RL (NFT/AdvantageFlow, PR #5) on the physical simulator did not
   learn. Small steps left the policy unchanged; larger steps degraded T (dense
   3e-5: T 0.55 -> 0.05 by update 75). Training stopped at about 150 updates.
2. Cause: the training score is mostly noise. Only 16-36% of a candidate plan's
   score variance comes from the plan itself (6-29% after a bias correction;
   split-half reliability 0.14-0.29);
   the random follow-up plans dominate. Updates had no anchor to the start policy.
3. Re-ranking the base policy's own plans with the true simulator helps a little
   (Y 3/20 -> 8/20), and imitating those choices stays within evaluation noise.
   The base policy's proposals are too similar to each other (sample spread about
   0.02 in normalized units) to contain new behavior.
4. A structured state world model trained only on non-expert machine rollouts
   predicts contact motion to about 0.2 px median over 8 controls.
5. Gradient-based training through that model (SuperMimic-style BPTT) failed.
   A sampling planner inside the model succeeds in the real simulator on
   50 validation scenes per letter: D 50/50, Y 36/50, L 21/50, against base rates
   of 0.3%, 9.4% and 0.1%. It is worse than the base on T (5/50) and near-T (12/50).
6. Distilling the planner into the image policy first made it worse on every
   shape. Training the image encoder too, longer training and DAgger produced the
   first students with real D and L successes (100-scene test: D up to 5, L up to 6,
   base 0) but 10-15 points lower T.
7. Blending a student's weights with the base's recovered most of T. The selected
   70% blend, on 1000 fresh scenes paired with the base: T 39.6% vs 44.4% (-4.8,
   95% interval -8.7 to -0.9), near-T 41.0% vs 41.9%, Y 12.3% vs 9.4% (+2.9, +0.2 to
   +5.6), D 4.9% vs 0.3% (+4.6, +3.2 to +6.0), L 0.2% vs 0.1%. T failed the
   pre-registered non-inferiority bound of -5 points, so the result is a measured
   trade-off, not a pass.
8. The T loss comes mostly from the student's action network. A secondary final test
   of an encoder-only blend (base action network + 70% student encoder): Y 18.1% vs
   9.4% (+8.7, +5.7 to +11.7), D 1.4% vs 0.3%, T -2.5 [-6.2, +1.2], near-T -2.0
   [-5.8, +1.8]. Not significantly worse on T, but the margin was narrowly missed, so
   it did not pass either. A confirmatory test on 2000 fresh scenes per letter (D041)
   confirmed Y +8.0 [+6.0, +9.9] and D +0.8, near-T +0.4, but also a real T loss of
   -3.1 [-5.9, -0.4].
9. Passed (D042): keeping the base's action network and blending in 50% of the large
   student's image encoder, on 2000 fresh scenes per letter: Y 8.6% -> 14.8% (+6.2,
   97.5% interval +4.0 to +8.4), D 0.3% -> 1.1% (+0.9, +0.2 to +1.5), T +1.8 [-1.3,
   +4.8], near-T +1.4 [-1.6, +4.3], L unchanged. A 30% share also passed (Y +3.2, T
   +3.0). This was the fourth round of testing; the earlier three are listed above.
   Replicated on a third fresh pool (D043, 95% intervals): Y 9.8% -> 14.1% (+4.3, +2.4
   to +6.3), D 0.1% -> 0.9% (+0.9, +0.4 to +1.3), T +1.8 [-0.9, +4.4], near-T +0.9
   [-1.7, +3.6], L -0.1 [-0.1, +0.0]: passed again.
10. One policy for ten shapes (section 7). Students copy the teacher's intended 16-step plans (+68% over copying its
   executed pushes) and learn from correction rounds. On 1000 fresh scenes per letter (D049): average success 9.1% ->
   34.3%, T 42.2% -> 42.9%, near-T 39.8% -> 32.0%, the eight other letters from 0-8% to 24-47%. A 192 px camera
   (D050, a sensor change) reaches 38.2% (+3.9 points [+2.6, +5.2] over 96 px on 10,000 paired scenes).
11. Why the student stays below the teacher (78% on validation): success allows only 1.3-2.7 px of error (0.24-0.51 of
   a 96 px camera pixel); committing to 8 pushes costs even the exact-state teacher 22% of its successes; after about
   two plan-label rounds more corrections no longer change behaviour beyond test-to-test chance.
12. The ten-letter students also transfer to the 17 untrained capitals (D052): pooled success 0.2% -> 6.6% (96 px)
   and 10.6% (192 px), final coverage 0.33 -> 0.51 / 0.57.
13. One policy for all 26 capitals and near-T (section 8, D051): on 300 fresh scenes per shape, success over the 27
   trained shapes rises from 3.7% to 33.4% (96 px) and 33.8% (192 px), with the ten earlier shapes unchanged (33-34%)
   and the 17 new letters from 0.4% to 33%. Final coverage 0.35 -> 0.82.
14. More training shapes help on shapes never seen: on six held-out shapes success rises 1.2% (original) -> 2.8%
   (ten letters) -> 4.7% / 5.9% (26 letters), coverage 0.43 -> 0.53 -> 0.56 / 0.58.

## Evaluation protocol

Every pool uses scene seeds `base + 20*i`; pools never overlap. "Base" is the frozen
official RSS 2023 CNN diffusion policy (DDPM-100, 8 of 16 actions executed).

| Pool | Seeds | Use |
| --- | --- | --- |
| Development | 100000+ | trainer's periodic evaluation (PR #5) |
| Validation | 400000+ | checkpoint and planner selection |
| Monitor test | 300000+ | 100 scenes; repeated progress checks |
| Value data | 600000+ | value-function and world-model data |
| Oracle data | 700000+ | best-of-N decision records |
| Teacher data | 800000+ | planning-teacher demonstrations |
| Final test | 900000+ | 1000 scenes; reserved for the acceptance test |
| Fresh pool 1 | 2000000+ | 2000 scenes; D041 confirmatory test |
| Fresh pool 2 | 3000000+ | 2000 scenes; D042 test (passed) |
| Fresh pool 3 | 4000000+ | 2000 scenes; D043 replication (passed) |

Base policy on the 1000-scene final test pool (Wilson 95% interval in brackets):
T 44.4% [41.3, 47.5], near-T (`T_narrow`) 41.9% [38.9, 45.0], Y 9.4% [7.7, 11.4],
D 0.3% [0.1, 0.9], L 0.1% [0.0, 0.6]. On the 100-scene monitor test: T 38-40, near-T 33-38, Y 7-11, D 0,
L 0 across two policy-noise seeds.

Evaluation noise is large. Rerunning the same policy with another noise seed flips
37-40% of T scene outcomes, so pairing scenes barely helps and a T non-inferiority
claim needs about 1000 scenes. Validation gains often failed to replicate on the
monitor test because of checkpoint-selection bias.

## 1. Forward-process RL on the physical simulator

Batched lock-step rollouts (D033) made long runs affordable on one H100.

- 2026-10-24, eight lineages of about 1M controls each (LoRA, lr 1e-5): L stayed 0/5
  at every evaluation and T moved without trend. Fixed-probe drift was about 1e-6,
  so the policy had barely changed.
- 2026-10-25, long runs with larger steps: dense lr 1e-4 collapsed by update 25;
  dense 3e-5 took T from 0.55 to 0.05 by update 75; every dense run at 1e-5 or
  above ended with T between 0.00 and 0.30. LoRA and dense 3e-6 stayed flat.
  Near-T dropped in almost every run. Stopped at about update 150.

## 2. Learning-signal diagnosis

`scripts/learning/score_snr.py` re-scores trainer-like candidate groups with
independent follow-up noise (frozen policy, 8 candidates x 8 follow-up draws):

| Shape | Plan share of score variance | Split-half reliability |
| --- | --- | --- |
| T | 0.33 | 0.27 |
| near-T | 0.17 | 0.18 |
| Y | 0.16 | 0.14 |
| D | 0.36 | 0.29 |

Other measured causes: the weighted T-retention term was about 1e-4 of the main
loss, and each update anchored only to the previous policy, so drift accumulated
like a random walk. Changes concentrated at high-noise denoising steps, which
means strategy changes rather than refinements. The shaped coverage reward
telescopes, so a short-horizon score mostly measures the follow-up plans. A
privileged value function (R^2 0.64 on held-out scenes) did not rank candidates
better than the noisy score. Literature review: the
[research report](research/diffusion-policy-rl-learning-signal.md) and its notes.

## 3. Simulator-oracle best-of-N and imitation

`oracle_bon.py` samples N base plans per decision, tries each in a replayed copy
of the episode (8 own controls plus 16 base controls in round 1, 24 later), and
commits the best. This uses the true simulator at decision time, so it is an upper
bound, not a policy.

| Shape (paired scenes) | Base success | Oracle success | Final coverage |
| --- | --- | --- | --- |
| T (10) | 6 | 7 | 0.78 -> 0.80 |
| near-T (20) | 10 | 13 | 0.74 -> 0.77 |
| Y (20) | 3 | 8 | 0.55 -> 0.77 |
| D (5) | 0 | 0 | 0.17 -> 0.44 |
| L (5) | 0 | 0 | 0.31 -> 0.43 |

`distill.py` imitated the oracle's choices with the policy's own denoising loss plus
an anchor to the base. Students from three data rounds (oracle plus two DAgger
rounds) on the monitor test:

| Checkpoint | T | near-T | Y | D | L |
| --- | --- | --- | --- | --- | --- |
| Base, noise seed 42 / 43 | 38 / 40 | 38 / 33 | 7 / 11 | 0 | 0 |
| dense, chosen, round 2 | 40 | 39 | 13 | 0 | 0 |
| dense, chosen, round 3 | 49 | 31 | 10 | 0 | 0 |
| dense, chosen, round 4 | 45 | 34 | 4 | 0 | 0 |
| dense, softmax, round 3 | 42 | 39 | 10 | 0 | 0 |

No consistent gain. The largest T gain (49) came with the lowest near-T (31). With
37-40% of scene outcomes flipping between noise seeds, paired differences of about
6 successes per 100 scenes are expected from noise alone, before choosing among
checkpoints. Only Y coverage rose consistently (0.58-0.59 against 0.49-0.55).

## 4. Privileged world models

Data: 1500 episodes (431k controls) of the frozen base policy on all five shapes
with Gaussian action noise of 0, 10 and 25 px (`base_rollouts.py collect`). This
is machine-generated, non-expert interaction; no human demonstrations.

- `wm_state.py`, a generic MLP over the 10-dimensional state: 8-control contact error
  2.32 px / 1.62 deg median, 32-control 4.21 px / 3.12 deg. Too coarse for pushing
  that needs about 1.5 px.
- `wm2.py`, structured: exact pusher dynamics (10 PD substeps per control, checked
  against the simulator to 1e-4 px), exact signed distance from the pusher to the
  letter's rectangles, a contact gate (no motion without contact, which matches the
  simulator's zero damping), and a learned object-frame MLP for motion in contact.

| `wm2` held-out contact error, median / p90 | 8 controls | 32 controls |
| --- | --- | --- |
| T | 0.23 / 1.02 px, 0.12 / 0.70 deg | 0.42 / 2.82 px |
| near-T | 0.22 / 1.84 px, 0.11 / 0.97 deg | 0.47 / 8.54 px |
| Y | 0.24 / 1.24 px, 0.13 / 0.81 deg | 0.40 / 2.03 px |
| D | 0.32 / 2.34 px, 0.21 / 1.72 deg | 0.56 / 6.31 px |
| L | 0.23 / 1.12 px, 0.13 / 0.75 deg | 0.39 / 2.73 px |

Without contact the predicted motion is exactly zero. Near the goal the one-step
error is 0.08 px with no directional bias (base-policy transitions, training and
validation scenes together). For the planner's own best push from
5 px off the goal, the model's final pose matched the real simulator within
1 px in 13 of 15 trials (0.2-0.9 px; the other two 1.6 and 2.6 px).

## 5. World-model teachers (privileged state, simulator never consulted)

### BPTT through the model: negative

`teacher_bptt.py` (generic model) and `teacher2.py` (`wm2`, SuperMimic-style BPTT
over 32 imagined steps with a GAE critic) were trained from logged start states.
Behaviour cloning the base first failed, because a deterministic network averages
the diffusion policy's modes. Trained from scratch, the teacher learned to approach
the object (gap 21 -> 3.7 px), but real-simulator coverage stayed 0.04-0.12 after
500+ iterations for every reward tried, including a dense goal-distance term.
Gradients through contact appear to trap it in local optima, as reported for
planar pushing by Suh et al. (ICML 2022).

### Sampling planner: works on the new letters

`mpc_teacher.py` reads the real simulator's state each control, plans pusher
targets inside `wm2`, and executes the first target of the best plan. The
simulator only supplies the current state and the outcome. The first version (CEM over
smooth random paths) moved letters near the goal but stalled 2-4 px short; success
needs about 1.5 px and 2 deg (a 2 px shift alone drops coverage to 0.92-0.95). Each
fix below targeted one measured stall cause:

| Change | Stall cause it removed |
| --- | --- |
| Push candidates: leave on a safe circle, go around, push one boundary point | random paths almost never go around the letter |
| Execute the best sample, horizon 48 | the elite average blurred distinct pushes; payoff fell past 32 steps |
| Contact gate margin 0 while planning | phantom 0.03 px "pushes" while hovering within 3 px |
| No pusher-to-object distance penalty | going around cost an estimated 2.3 reward units, against about 1 for finishing |
| Pose error over area points, not rectangle corners | corner error read 1.7 px while the centre was 3.4 px off |
| Sharp final kernel `exp(-d2/2^2)` | the last 1-2 px earned almost no reward |

| Validation scenes | First planner (20) | Final planner (20) | Final planner (50) | Base (final test) |
| --- | --- | --- | --- | --- |
| D | 4 | 20 | 50 (100%) | 0.3% |
| Y | 2 | 16 | 36 (72%) | 9.4% |
| L | 0 | 7 | 21 (42%) | 0.1% |
| near-T | 1 | 3 | 12 (24%) | 41.9% |
| T | 0 | 3 of 24 | 5 (10%) | 44.4% |

Final-planner final coverage on the 50 scenes: D 0.967, Y 0.954, L 0.943, near-T
0.938, T 0.932.

On the 350 teacher-data scenes (800000+): D 144/150, Y 74/100, L 40/100. On T and
near-T it fixes position but leaves about 2.5 deg of rotation; removing it needs
two well-placed pushes, which the planner rarely finds. Plans only work
closed-loop: executing 8 controls per plan, as the image policy does, cut T
coverage from 0.70 to 0.32 in the first version.

## 6. Distilling the planner into the image policy

`mpc_dagger.py` runs the planner with rendering and saves, every 4 controls, the
2-frame observation, 8 samples from the image policy and a label: the planner's
actually executed next 16 targets. Round 1: 11.5k records from Y, D and L, plus
3.0k T/near-T retention records (earlier oracle records of base plans). `distill.py
--anchor-shapes T T_narrow` anchors only those records to the base, so Y/D/L can
follow the teacher. Three dense students (lr 3e-5, 1e-4, and 3e-5 with a 0.1
anchor on all shapes), 100 validation scenes each:

| Policy | T | near-T | Y | D coverage | L coverage |
| --- | --- | --- | --- | --- | --- |
| Base | 35 | 37 | 9 | 0.30 | 0.32 |
| lr 3e-5, epoch 10 / 20 | 15 / 21 | 17 / 14 | 3 / 1 | 0.12 / 0.08 | 0.21 / 0.15 |
| lr 1e-4, epoch 10 / 20 | 18 / 18 | 14 / 12 | 1 / 5 | 0.20 / 0.12 | 0.20 / 0.22 |
| lr 3e-5 + 0.1 anchor, epoch 10 / 20 | 20 / 25 | 17 / 13 | 3 / 7 | 0.14 / 0.09 | 0.24 / 0.20 |

Distillation failed: every student is worse than the base, including on the
letters the teacher solves, and the T anchor did not protect T. The imitation
loss stayed about ten times higher than on oracle data (0.08 against 0.007).
Likely reasons, not yet separated: labels stitched from a planner that replans
every control are inconsistent as 8-control open-loop chunks (its targets jump
20-150 px between controls while going around the letter); the policy's states
drift away from the teacher's (no DAgger yet); and the imitation gradient
overwhelms the anchor on shared weights.

The final planner still succeeds when it executes 8-control open-loop chunks like
the student (20 validation scenes: D 20, Y 14, L 7, against 19-20, 16-18 and 6-7
when replanning every control). Round 2 therefore labels each record with the plan
the teacher actually executes (`mpc_dagger.py --execute 8 --every 8`); in a smoke
test the largest jump inside a label fell from a median of 56 px to 31 px. Round 2
(13.4k records from 450 new scenes, plus T retention) failed in the same way: every
student stayed below the base, with validation T at 21-31 against 35 and D coverage
at 0.18-0.26 against 0.30 (all epochs in the experiment log). The imitation loss
stayed high, which suggests the student cannot represent the teacher's behavior.
One candidate cause: the image encoder stays frozen during distillation and was only
ever trained on T. Round 3 trains the encoder too (`distill.py --train-encoder`) on
20.6k planner records plus 3.0k T retention records, with a frozen-encoder control.

Rounds 3-4 and DAgger, 100-scene monitor test (base T 38-40, near-T 33-38, Y 7-11,
D 0 with coverage 0.31, L 0 with 0.27-0.30):

| Student | T | near-T | Y | D (coverage) | L (coverage) |
| --- | --- | --- | --- | --- | --- |
| Round 3, encoder trained, 30 epochs | 24-26 | 23-24 | 5-6 | 1 (0.25-0.36) | 0 |
| Round 3, frozen encoder, stronger anchor | 25 | 27 | 2 | 0 (0.28) | 0 |
| Round 4, encoder trained, 60 epochs (r4e60) | 24 | 39 | 10 | 5 (0.54) | 5 (0.51) |
| DAgger iteration 1 | 27 | 21 | 10 | 2 (0.51) | 4 (0.35) |
| DAgger iteration 2 | 29 | 27 | 16 | 0 (0.44) | 6 (0.45) |

Training the encoder, more epochs and DAgger (the student drives 70% of decisions;
the planner labels the states it reaches, `mpc_dagger.py --beta 0.3`) each helped the
new letters, giving the first image policies with more than a stray D or L success.
T stayed 10-15 points low. A chaining bug (a process-list wait that matched its own
launcher) stopped DAgger iterations 3-4 overnight.

### Weight blending and the pre-registered final test

`interpolate.py` blends a student with the base, theta = (1-s) base + s student,
for both the action network and the encoder (WiSE-FT style; s = 0 is the base).
On validation (100 scenes; base T 35, near-T 37, Y 9, D 0 with coverage 0.30):

| Blend | T | near-T | Y | D (coverage) | L (coverage) |
| --- | --- | --- | --- | --- | --- |
| r4e60, s = 0.3 | 45 | 41 | 15 | 1 (0.39) | 0 (0.33) |
| r4e60, s = 0.5 | 42 | 41 | 17 | 0 (0.50) | 0 (0.36) |
| r4e60, s = 0.7 | 39 | 42 | 12 | 4 (0.58) | 0 (0.40) |
| DAgger 2, s = 0.3 / 0.5 / 0.7 | 42 / 45 / 39 | 35 / 40 / 44 | 10 / 15 / 15 | 1 / 0 / 1 | 0 |

The top three went to the monitor test, and the rule fixed in D040 before any final
test (T at least 38 and near-T at least 33 there, then the most Y + D + L successes)
selected r4e60 at s = 0.7 (monitor test: T 47, near-T 38, Y 13, D 3, L 0; the
r4e60 s = 0.5 blend had T 36 and the DAgger 2 s = 0.5 blend near-T 31). Final test,
1000 scenes per letter (900000 pool), paired by scene with the base:

| Letter | Base | Blend | Difference [95% interval] | Final coverage |
| --- | --- | --- | --- | --- |
| T | 44.4% | 39.6% | -4.8 [-8.7, -0.9] | 0.692 -> 0.730 |
| near-T | 41.9% | 41.0% | -0.9 [-4.7, +2.9] | 0.696 -> 0.761 |
| Y | 9.4% | 12.3% | +2.9 [+0.2, +5.6] | 0.578 -> 0.688 |
| D | 0.3% | 4.9% | +4.6 [+3.2, +6.0] | 0.314 -> 0.581 |
| L | 0.1% | 0.2% | +0.1 [-0.2, +0.4] | 0.317 -> 0.398 |

Conditions: T non-inferior (lower bound at least -5) no; near-T yes; a new letter
better yes (Y and D); no new letter worse yes. Not passed, so the base remains the
result. The blend's T of 47 on the monitor test was selection luck; on 1000 scenes
T is about 5 points below the base, while coverage rises on every letter.

### Where the T loss comes from, and a secondary final test

Swapping parts between the base and the r4e60 student on validation: base action
network + 70% student encoder gave T 51, near-T 45, Y 12, D 2; student action network
+ base encoder gave T 30, near-T 23, Y 2, D 1; 70% student action network + base
encoder gave T 33, near-T 40, Y 7, D 0. The T loss comes mostly from the student's
action network, while its retrained image encoder helps every letter. Eight
encoder-heavy blends across four students followed (all in the experiment log); the
three best on validation went to the monitor test:

| Blend (monitor test) | T | near-T | Y | D | L |
| --- | --- | --- | --- | --- | --- |
| large student (lr 1e-4): base action network + 70% encoder | 39 | 33 | 16 | 1 | 0 |
| DAgger 2: 30% action network + full encoder | 43 | 39 | 14 | 0 | 0 |
| r4e60: 30% action network + 70% encoder | 39 | 35 | 11 | 2 | 0 |

The unchanged D040 rule selected the first. Its final test is a **secondary result**
(the second candidate evaluated on the final pool, after the primary one failed;
two looks at the same pool make a lucky pass more likely):

| Letter | Base | Blend | Difference [95% interval] | Final coverage |
| --- | --- | --- | --- | --- |
| T | 44.4% | 41.9% | -2.5 [-6.2, +1.2] | 0.692 -> 0.701 |
| near-T | 41.9% | 39.9% | -2.0 [-5.8, +1.8] | 0.696 -> 0.709 |
| Y | 9.4% | 18.1% | +8.7 [+5.7, +11.7] | 0.578 -> 0.635 |
| D | 0.3% | 1.4% | +1.1 [+0.3, +1.9] | 0.314 -> 0.432 |
| L | 0.1% | 0.1% | +0.0 [-0.3, +0.3] | 0.317 -> 0.334 |

Not passed: neither T nor near-T is significantly worse, but 1000 scenes cannot rule
out drops of up to 6.2 and 5.8 points, beyond the 5-point margin. Y nearly doubles and
D improves significantly.

Confirmatory test (D041): the same fixed candidate against the base on 2000 fresh
scenes per letter (seeds 2000000 + 20*i, never used before), paired by scene:

| Letter | Base | Blend | Difference [95% interval] | Final coverage |
| --- | --- | --- | --- | --- |
| T | 46.0% | 42.8% | -3.1 [-5.9, -0.4] | 0.714 -> 0.719 |
| near-T | 42.3% | 42.8% | +0.4 [-2.2, +3.1] | 0.698 -> 0.718 |
| Y | 8.8% | 16.8% | +8.0 [+6.0, +9.9] | 0.561 -> 0.640 |
| D | 0.1% | 0.9% | +0.8 [+0.3, +1.3] | 0.318 -> 0.439 |
| L | 0.0% | 0.0% | +0.0 | 0.319 -> 0.330 |

Free of selection bias, this confirms the Y gain (success nearly doubles) and a small D
gain, with near-T unchanged, but also a real T loss of about 3 points. Not passed.

### The passing result: smaller encoder shares on a second fresh pool (D042)

Monitor-test results suggested that smaller encoder shares keep T (base action network
plus 30%, 50%, 70%, 100% of the large student's encoder: T 41, 39, 39, 40; near-T 41, 43,
33, 39; Y 13, 16, 16, 10). Two candidates were fixed in D042 before any data from a second
fresh pool existed (2000 scenes per letter, seeds 3000000 + 20*i), with 97.5% intervals
for the two comparisons:

| Letter | Base | 30% encoder | Difference | 50% encoder | Difference |
| --- | --- | --- | --- | --- | --- |
| T | 43.6% | 46.7% | +3.0 [+0.1, +6.0] | 45.4% | +1.8 [-1.3, +4.8] |
| near-T | 40.4% | 43.0% | +2.6 [-0.3, +5.6] | 41.8% | +1.4 [-1.6, +4.3] |
| Y | 8.6% | 11.8% | +3.2 [+1.1, +5.3] | 14.8% | +6.2 [+4.0, +8.4] |
| D | 0.3% | 0.3% | +0.0 [-0.4, +0.4] | 1.1% | +0.9 [+0.2, +1.5] |
| L | 0.0% | 0.0% | +0.0 | 0.1% | +0.1 [-0.1, +0.3] |

Both pass all four conditions. The 50% blend is the stronger result: Y success rises by
about 70% relative and D improves, with T and near-T unchanged within the margin. Context
for interpretation: this was the fourth round of testing (D040 primary and secondary,
D041, D042); the share was chosen after the earlier rounds, but the pool itself was never
used before, so the D042 comparison is unbiased. On the other pools the 70% share lost
about 3 T points, so the encoder share is the knob that trades T for the new letters.

Replication (D043): the 50% blend, fixed before the pool existed, on a third fresh pool
(2000 scenes per letter, seeds 4000000 + 20*i, 95% intervals):

| Letter | Base | 50% encoder | Difference [95% interval] | Final coverage |
| --- | --- | --- | --- | --- |
| T | 43.4% | 45.1% | +1.8 [-0.9, +4.4] | 0.699 -> 0.710 |
| near-T | 40.0% | 41.0% | +0.9 [-1.7, +3.6] | 0.690 -> 0.705 |
| Y | 9.8% | 14.1% | +4.3 [+2.4, +6.3] | 0.572 -> 0.608 |
| D | 0.1% | 0.9% | +0.9 [+0.4, +1.3] | 0.306 -> 0.416 |
| L | 0.1% | 0.0% | -0.1 [-0.1, +0.0] | 0.318 -> 0.331 |

All four conditions hold again. Two independent fresh pools agree: the 50% encoder blend
improves Y and D significantly without losing T or near-T. L is not improved.

All 26 letters (descriptive, not pre-registered): the base and the 50% blend on 100 fresh
scenes per letter (seeds 5100000 + 20*i, paired by scene; `runs/all-letters/summary.txt`).
On the 22 letters never used in adaptation (2200 episodes), success is rare for both
(0.23% vs 0.59%, +0.36 points [+0.03, +0.70]) but final coverage rises from 0.306 to 0.330
(+0.024 [+0.015, +0.032]) and is higher on 18 of the 22 letters (for example B 0.39 ->
0.46, Q 0.37 -> 0.46, O 0.42 -> 0.48). So the new encoder helps the policy push unseen
letters closer to their goals, but rarely to the 95% success bar. With 100 scenes per
letter, single-letter differences are within noise (Y here 13% vs 10%, against +4 to +6
points on 2000 scenes; T 43% vs 52%).

Students that train only the encoder with the base action network frozen
(`distill.py --freeze-actor`) kept T (monitor test 39-44) but collapsed on the new
letters (Y, D, L final coverage 0.00-0.05): the useful encoder is the one trained
together with the action network and then blended.

### Teacher-relabelled DAgger works (2026-10-31, D045, D046)

The official-recipe camera student was improved by rounds of correction (DAgger): the student drives on new
training scenes with probability 0.7 per 8-control decision, the improved teacher labels every visited state, and
the student is fine-tuned on all teacher demonstrations, the original human T demonstrations and every correction
round so far (`chain_dagger_loop.sh`, 20k steps per round). On the teacher's 50 validation scenes per letter
(successes; teacher 34, 22, 46, 50, 39):

| Student | T | near-T | Y | D | L | mean final coverage |
| --- | --- | --- | --- | --- | --- | --- |
| Original | 20 | 17 | 4 | 0 | 0 | 0.52 |
| Official recipe, no corrections, best of 30k-150k steps | 23 | 20 | 8 | 15 | 8 | 0.84 |
| Correction round 2 | 18 | 11 | 18 | 17 | 13 | 0.84 |
| Correction round 4 | 19 | 16 | 17 | 21 | 15 | 0.86 |
| Correction round 5 / 6 | 19 / 18 | 19 / 14 | 17 / 19 | 21 / 20 | 17 / 12 | 0.83 / 0.84 |

Longer training without corrections levels off (and the exact-state copy trained 10x longer does not improve);
corrections lift Y, D and L clearly, with a plateau after round 4. Fair test of round 4 (D046, 1000 fresh scenes
per letter): T 42.6% -> 36.3% (-6.3 [-10.5, -2.1]), near-T 40.2% -> 29.3% (-10.9 [-15.1, -6.7]), Y 9.9% -> 33.6%
(+23.7 [+20.3, +27.1]), D 0.0% -> 33.1% (+33.1), L 0.0% -> 28.8% (+28.8); coverage rises on every letter. On all
26 letters (100 scenes each, descriptive): the 22 letters never used in training go from 5 to 57 successes of 2200
and coverage 0.306 -> 0.454 (+0.148 [+0.137, +0.159]), mostly on rounded letters (O 15, Q 12, B 11).

Correction rounds continued (50 validation scenes, successes T / near-T / Y / D / L, total of 250): round 5
19/19/17/21/17 (93), round 6 18/14/19/20/12 (83), round 7 28/14/16/28/14 (100); rounds differ by about +-11 from
chance alone, so the trend (about +5 per round) is slow. On the 26-letter pool (100 scenes per letter), plain
round 6 and round 7 recover T and near-T: T + near-T vs the original -2.5 (-11.7 to +6.7) and -3.0 (-12.3 to +6.3)
points, with Y + D + L +28.7 (D047 tests round 7 on fresh scenes).

Fair test of round 7 (D047, fresh pool): T -6.2 [-10.5, -1.9], near-T -12.1 [-16.2, -8.0], Y +24.6, D +38.7,
L +33.1 points; the T/near-T trade-off is the same as round 4's. Rounds 9 and 10 (validation, of 250): 86 and 89.

Ten letters (E, F, J, P, U added; teacher on 50 validation scenes 49, 30, 47, 49, 44 with world model wm2 v2,
which keeps D 50/50 and L 35/50): one student trained from round 7 on all teacher demonstrations (the new letters'
from training seeds i = 46000-46099, frames by replay), then correction rounds over all ten letters
(`chain_dagger_loop10.sh`). Successes of 50 per letter, T / near-T / Y / D / L / E / F / J / P / U (total of 500):
first training 15/16/17/17/8/3/3/5/4/6 (94); after ten-letter round 1 7/12/18/16/13/11/3/7/9/8 (104); after
round 2 17/14/17/20/12/4/6/9/7/8 (114). The original policy scores 0 on all five new letters.

Negative results from a parallel 3-hour H200 session (2026-10-31, `runs/lastmile/`, logs `logs/lastmile/`):
- "Last-mile" variant from round 4: teacher steps with coverage >= 0.9 sampled 3x more often and twice as many
  corrected scenes per round. Paired with matched plain rounds on 100 validation scenes per letter: -3.8, -4.8, +0.0
  points (all intervals include zero); on the 26-letter pool, +2.0 (-4.0 to +8.0) and +1.8 (-3.9 to +7.5) on the
  training letters. No detectable gain.
- Replanning more often hurts these students: 4 controls per plan instead of 8, -13.6 points (-21.3 to -5.9) for
  round 7; 2 controls per plan was also worse throughout. The remaining gap is final alignment (mean final coverage
  0.8-0.9 vs the teacher's 0.95); candidates are image resolution (one 96x96 pixel is about 5 simulator pixels) and
  the teacher's state noise making near-goal labels partly random.

### Can the student copy the teacher? (2026-10-30, D044)

Same 50 validation scenes per letter (seeds 400000 + 20*i), successes out of 50:

| Policy | T | near-T | Y | D | L |
| --- | --- | --- | --- | --- | --- |
| Teacher: planner reading exact state with 1 px / 1 deg noise | 34 | 22 | 46 | 50 | 39 |
| Original camera policy, 8 controls per plan | 20 | 17 | 4 | 0 | 0 |
| Original camera policy, 2 controls per plan | 16 | 11 | 1 | 0 | 0 |
| Diffusion student reading exact state, 8 controls per plan | 18 | 13 | 5 | 17 | 4 |
| Diffusion student reading exact state, 1 control per plan | 14 | 11 | 16 | 26 | 11 |
| Camera student, official recipe, 10k / 20k / 30k steps | 16 / 15 / 21 | 9 / 13 / 11 | 6 / 5 / 5 | 9 / 6 / 8 | 0 / 3 / 3 |

- The state-input student (`state_student.py`; same 16-step interface, trained from scratch on 79k steps of
  500 teacher episodes, privileged input, a diagnostic only) recovers only a third to a half of the teacher's
  successes, so most of the skill is lost in copying, not in perception. Replanning every control instead of
  executing 8 helps it on Y, D and L (5 -> 16, 17 -> 26, 4 -> 11), consistent with a teacher that replans
  every control.
- The camera student trained with the official Diffusion Policy recipe (`bc_train.py`: every step of 500 teacher
  episodes plus the 206 original human T episodes, letters balanced, random-crop augmentation, EMA, cosine
  learning rate, from the official checkpoint) levels off far below the teacher after about 5M samples, while
  its training loss still falls (30k steps, about 7.7M samples); with 2 controls per plan at 30k: T 11, near-T 9,
  Y 5, D 4, L 10. The official policy saw about 78M samples. The no-anchor older-recipe run reached T 23,
  near-T 23, Y 7, D 3, L 0 of 100 after 10 epochs. A first DAgger round (student drives with probability 0.7,
  teacher labels; 250 training scenes) was collected for the next session (`runs/mpc-data/dagger-bc1`).
- The older recipe (`distill.py`, records every 8 controls, anchor to the original policy) after 5 epochs, 100
  validation scenes: with anchor T 14, near-T 17, Y 7, D 8, L 1; without anchor T 22, near-T 23, Y 6, D 3, L 0
  (original: T 35, near-T 37, Y 9, D 0, L 0).
- Teacher variants on T / near-T / L (50 scenes): 0.5 px 35 / 28 / 38; position-only 1 px 28 / 21 / 39;
  angle-only 1 deg 36 / 34 / 34; a repeat of 1 px with another noise seed 33 / 26 / 39 (run-to-run spread about
  4). One 50-scene rate has a 95% interval of about +-14 points.
- Next (literature summary in the decision log): teacher-relabelled DAgger (the student drives, the teacher
  labels the visited states), much longer training, and fewer controls per plan.

### How accurate must an image-based state estimator be?

The planner reads exact simulator state, which an image-to-state model (the teammates'
direction) would replace. To set an accuracy target, the planner (final settings, 20
validation scenes per letter, same scenes in every row) read noisy estimates instead:
independent Gaussian noise each control on object x/y, pusher x/y (simulator pixels,
512-px frame) and object angle, with velocities from consecutive noisy readings
(`mpc_teacher.py --state-noise PX --angle-noise DEG`).

| State the planner reads | Y | D | L | Final coverage Y / D / L |
| --- | --- | --- | --- | --- |
| Exact simulator state | 18/20 | 19/20 | 6/20 | 0.955 / 0.965 / 0.941 |
| Velocities from consecutive readings, no noise | 16 | 20 | 6 | 0.957 / 0.967 / 0.939 |
| 0.5 px, 0.5 deg | 18 | 20 | 14 | 0.954 / 0.967 / 0.946 |
| 1 px, 1 deg | 19 | 20 | 15 | 0.957 / 0.965 / 0.944 |
| 2 px, 2 deg | 16 | 18 | 16 | 0.951 / 0.963 / 0.943 |
| 4 px, 4 deg | 6 | 4 | 10 | 0.890 / 0.880 / 0.910 |

Up to about 2 px and 2 degrees per reading the planner loses nothing on Y and D; at 4 px
it breaks down. At the policy's 96x96 image resolution, 2 simulator pixels are about 0.4
of an image pixel, so an estimator needs sub-pixel precision there or higher-resolution
input. Small noise raised L from 6 to 14-16 of 20, probably by knocking the planner out of
stalls (a hint worth testing, not a settled result). Caveats: 20 scenes per letter, and
the noise is independent per step, while a real estimator's errors are often systematic
and cannot average out, so the real estimator must be tested in the planner directly.

Follow-up on 50 validation scenes per letter (same scenes in each row): small noise is a
large improvement where the planner used to stall just short of the goal.

| State the planner reads | T | near-T | L | Final coverage T / near-T / L |
| --- | --- | --- | --- | --- |
| Exact simulator state | 5/50 | 13/50 | 11/50 | 0.933 / 0.941 / 0.940 |
| 1 px, 1 deg noise | 34/50 | 22/50 | 39/50 | 0.941 / 0.919 / 0.946 |
| 2 px, 2 deg noise | 35/50 | 22/50 | 31/50 | 0.939 / 0.896 / 0.928 |

With 1 px noise the teacher is strong on every letter: on the same 50 validation scenes,
T 34/50, near-T 22/50, Y 46/50, D 50/50, L 39/50 (68%, 44%, 92%, 100%, 78%). Likely cause: the exact-state planner repeats the same
nearly-right push near the goal, and the noise varies it. This makes the noisy planner the
better teacher for training a new policy. It still reads privileged simulator state.

### Data package for image world models

`scripts/learning/image_wm_data/` packages continuous pushing episodes for training an image
world model (image + action -> next image): the base-policy dynamics data (2000 episodes,
581,845 transitions, seeds 604000-611980) plus 500 planner episodes recorded with
`mpc_teacher.py --save-traj` (seeds 6000000-6001980), with the exact 96x96 camera frame of
every step. Frames are made by replaying the recorded actions in the official simulator
(`replay_frames.py`), because the camera image also draws contact markers that only exist
during physics steps; every replayed pose matched the record. The packed archive (150 MB,
`archives/tdyssey_pushing_data.tar.gz` in the private run store) includes
`PACKAGE_README.md` and a numpy-only loader (`load_example.py`).

### Diffusion-policy samples as planner candidates

The planner can also score plans sampled from the image diffusion policy
(`mpc_teacher.py --dp-candidates`), 20 validation scenes per letter:

| Planner variant | T | near-T | Y | D | L |
| --- | --- | --- | --- | --- | --- |
| Final planner (push candidates + CEM) | 1/24 | 4 | 18 | 19 | 6 |
| Final planner + 32 diffusion samples (hybrid) | 0 | 3 | 15 | 18 | 4 |
| Diffusion policy alone, inside the same loop (no selection) | 10 | - | - | - | - |
| World-model best of 32 diffusion samples, previous plan kept | 0 | 0 | 0 | 0 | 0 |
| World-model best of 32 diffusion samples, without it (15-step scoring) | 11 | 9 | 7 | 1 | 0 |

Adding the policy's samples does not fix the planner's T weakness: it still prefers its own
moves. Selecting among the policy's samples with the world model first froze the policy:
keeping the previous plan (which ends by holding still) as a candidate let "do nothing" win
every time, because holding is always predicted safe; final coverage fell to 0.19-0.36. Without
that candidate, model-based best-of-32 keeps T at the policy's level and raises Y (7/20,
coverage 0.80, against about 9% for the base on validation): about the same gain the
true-simulator oracle gave (Y 3/20 -> 8/20), now with a learned model. It still needs the true
state to run the model, so it is a privileged test-time method.

Two students with low-rank (LoRA) action-network updates plus a retrained encoder
(`--trainable lora --train-encoder`) reached validation T 31-43, near-T 28-38, Y 2-11 with
new-letter coverage below the base: no better than the blends.

## 7. Ten letters: plan labels, correction rounds and fair tests (2026-10-31 to 2026-11-02)

Shapes: T, near-T, Y, D, L and the added letters E, F, J, P, U. World model wm2 v2 learned all ten from the original
policy's noisy rollouts; the teacher (1 px / 1 degree state wiggle) solves 390 of 500 validation scenes. Evaluations are
50 validation scenes per shape unless marked fresh.

**What helps (matched A/B tests, official recipe, 30k steps from the official checkpoint, 100 validation scenes per
shape, successes of 1000; tools `bc_train.py`, `policy_builder.py`):**

| Change | Without | With |
| --- | --- | --- |
| Labels: the teacher's intended plan instead of its executed pushes | 77 | **129** |
| Camera 192 px instead of 96 px | 111 | 129 |
| All teacher demonstrations instead of half | 104 | 111 |
| Action network 990M instead of 251M parameters (both from scratch, 50k steps) | 138 | 146 |

Only the label change is far beyond test noise. A teacher that commits to 8 pushes like the student keeps 303 of 500
(commits to 4: 317; re-plans every push: 390), so the student's 8-push rhythm itself costs precision.

**Correction rounds.** Ten-letter rounds with executed labels climbed from 94 to about 160 of 500 over 16 rounds;
switching to intended-plan labels reached about 180-190 after two rounds and then stayed flat: 21 later tests average
179 with a spread of 9.6, which is what chance alone gives (10.7). The same student tested twice on the same scenes
flips 211 of 500 outcomes, and every validation scene was solved in some tests and failed in others: success at the
finish is largely luck. Training loss keeps falling each round while the student's plans on new states stay about 39 px
from the teacher's (`scripts/learning/diagnose_plateau.py`). Success requires 1.3-2.7 px of position error, 0.24-0.51 of
a 96 px camera pixel.

**Fair tests (D049, D050; 1000 fresh scenes per shape, seeds 9000000 + 20*i, paired by scene; success %):**

| Shape | Original | 96 px student | 192 px student |
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
| **Average** | **9.1** | **34.3** | **38.2** |

The 192 px camera adds +3.9 points [+2.6, +5.2] over 96 px across all 10,000 paired scenes. Mean final coverage rises on
every shape (T 0.69 -> 0.87). Chart: `fairtest_results.png` (made with `scripts/learning/` tools; not in Git).

**Unseen letters (D052, 200 fresh scenes per letter, seeds 11000000 + 20*i, paired by scene).** On the 17 capitals that
none of these policies trained on, pooled success rises from 0.2% (original) to 6.6% (96 px, +6.3 [+5.5, +7.2]) and
10.6% (192 px, +10.4 [+9.3, +11.4]); mean final coverage from 0.33 to 0.51 and 0.57. Letters that share strokes with the
trained ones transfer best: B 20.0% / 39.5%, O 18.5% / 22.5%, W 13.5% / 19.5%, R 11.5% / 15.0%; A, M and X stay
below 4%. Chart: `unseen_letters_results.png` (not in Git).

**26-letter preparation (D051).** Dynamics data for the 17 other capitals; world model wm2 v3 for 27 shapes; teacher
on 50 validation scenes: A 47, B 49, C 42, G 44, H 49, I 47, K 49, M 49, N 46, O 49, Q 49, R 36, S 49, V 44, W 47,
X 48, Z 48 (T 30, D 49, L 34 re-checked). Six held-out shapes (`digit3 digit4 digit7 plus hash arrow`) are reserved for
the 26-letter final test.

## 8. All 26 letters (2026-11-02 to 2026-11-06)

Preparation (D051): dynamics data for the 17 other capitals, world model wm2 v3 for 27 shapes, teacher 36-49 of 50 on
the new letters, intended-plan demonstrations (150 scenes per new letter). Six held-out shapes (`digit3 digit4 digit7
plus hash arrow`, same block system) were fixed before training and used only in the final test.

**Budget, not method, was the bottleneck (D053).** With the ten-letter recipe (20k steps per round) the 96 px line
levelled at about 24% of validation (rounds 1-12, A100). The same 20k steps shared by 27 shapes give each shape 2.7x
less training per round, and the new letters had 5-8x fewer corrections. With 54k steps per round and 60 practice
scenes per new letter it rose to 30% within two rounds and about 34-35% by round 44; the 17 new letters caught up with
the earlier ten. The 192 px line (A100, frames streamed to disk for its 250 GB limit) reached 38% in its second round
and then held at 34-37%.

| Validation (of 1350; 50 scenes x 27 shapes) | 96 px | 192 px |
| --- | --- | --- |
| Start of the 26-letter run | 283 (21%) | 336 (25%) |
| End of the old budget | 314-358 (23-27%) | - |
| Best | 470 (35%, round 44) | 511 (38%, round 2) |

**Final test (D051; 300 fresh scenes per shape, seeds 10000000 + 20*i, paired by scene; success %):**

| Shapes | Original | Ten-letter 96 px | 26-letter 96 px | 26-letter 192 px |
| --- | --- | --- | --- | --- |
| Ten earlier shapes | 9.3 | 33.0 | 33.7 | 34.2 |
| 17 new letters | 0.4 | 6.4 | 33.2 | 33.5 |
| All 27 trained shapes | 3.7 | 16.3 | **33.4** | **33.8** |
| 6 held-out shapes | 1.2 | 2.8 | 4.7 | 5.9 |
| Final coverage, 27 shapes | 0.35 | 0.63 | 0.82 | 0.81 |

Per held-out shape (26-letter 192 px): digit7 23.0% (original 2.0%), hash 7.3%, arrow 2.7%, digit3 1.7%, digit4 0.7%,
plus 0.3%; mean final coverage 0.58 (original 0.43), and 20% of episodes reach 90% coverage at some point (original
6%). T 46.0% -> 40.7% / 38.3% and near-T 37.3% -> 29.0% / 32.3% (wide intervals at 300 scenes). The 192 px pick's
validation lead did not hold on fresh scenes. Six more 96 px rounds (45-50, D054) added +1.9 points [+0.1, +3.6] on
the 27 trained shapes in a separate fresh-scene check (32.9% -> 34.8%; 17 new letters +2.7; held-out shapes
unchanged), so round 50 is the 96 px model. Executing more of each 16-step plan before planning again is much
worse (D056: 12 steps 18.6%, 15 steps 12.2%, against 34.8% with the trained 8). Charts: `training_26letters_full.png`, `final_test_26letters.png`
(not in Git).

## Privileged signals and priors (disclosure)

- Teachers read the simulator's exact object and pusher state. The deployed policy
  must stay image-based; only distilled image policies count as results.
- Rewards use privileged geometry: letter rectangles, goal pose, coverage.
- Push candidates are a hand-designed geometric exploration prior.
- The world model's pusher dynamics and contact geometry are copied from the
  simulator. Only the object's motion in contact is learned.
- The oracle uses exact simulator replay at decision time; it is an upper bound only.
- No human demonstrations beyond the released T checkpoint's expert training data; the students reuse that same
  dataset (206 human T episodes) beside the machine-generated data.
- The 192 px students use a sharper camera than the official policy (a sensor change), reported separately.

## Reproduce

From the repository root, with `RUNS` set to any writable directory:

```bash
P=.venv/bin/python
$P -B scripts/learning/base_rollouts.py eval --scenes 100 --out $RUNS/test-eval/base.json
$P -B scripts/learning/base_rollouts.py collect --start 0 --scenes 50 --action-noise 10 --out $RUNS/dyn-data/part0.npz
$P -B scripts/learning/wm2.py train --data $RUNS/dyn-data --out $RUNS/wm2/v1 --steps 30000
T="--select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
$P -B scripts/learning/mpc_teacher.py --wm $RUNS/wm2/v1/model.pt --shape D --scenes 20 $T --out $RUNS/mpc/D.json
$P -B scripts/learning/mpc_dagger.py --wm $RUNS/wm2/v1/model.pt --shape D --scenes 50 $T --out $RUNS/mpc-data/r1
$P -B scripts/learning/distill.py --data $RUNS/mpc-data/r1/data $RUNS/retain-T/data --trainable action_network \
    --lr 3e-5 --anchor 1.0 --anchor-shapes T T_narrow --epochs 30 --eval-every 10 --val-scenes 100 --out $RUNS/distill/r1
```

Raw runs, checkpoints and logs stay outside Git in the author's private store.
Approximate costs on one H100 NVL: `wm2` training 35 min; a 20-scene planner
evaluation 5-10 min per shape; teacher data for 350 scenes about 15 min with 7
parallel jobs.

## Open problems

- Keeping T while learning the new letters: the blend costs about 5 T points.
  Options: more T retention data (`mpc_dagger.py --label policy` records the base's
  own plans), blending the action network and encoder separately, or a disclosed
  shape gate that routes T to the base.
- The student copies only part of the teacher (planner D 93-100%, students up to
  6%): 96x96 images and 8-control chunks make the final 1-2 px pushes hard.
- The planner's leftover T rotation.
- A planner bug found by the new test: push candidates jumped onto the safe circle
  when the pusher started outside it. The numbers above predate the fix; on the
  same 20 validation scenes the fixed planner gave Y 18, D 19, L 6, near-T 4 and
  T 1/24 against 16, 20, 7, 3 and 3/24, a difference within noise.
- The planner's L successes are marginal: re-running 20 L validation scenes that
  had succeeded gave 6/20 (coverage 95.0-96.3% in the earlier successes).
- The world model's rare large errors: over 500 16-control windows with real motion,
  the median error is 0.26 px but 1 in 20 is off by 14 px or more (worst 84 px),
  mostly when the letter touches the arena wall, which the model has no input for.
- Data hygiene: overnight, an idle-guard collector used scene seeds of the reserved
  final-test pool for seven rounds; they were quarantined before any training, and
  the collector now refuses reserved seeds.
