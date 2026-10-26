# Policy learning findings, 2026-10-24 to 2026-10-26

Status: in progress. A learned-world-model planning teacher now solves all D, most Y
and 42% of L validation scenes in the real simulator. Distilling it into the deployable image policy has
started; no image-policy improvement has been established yet. Negative results are
kept below on purpose. Code is in `scripts/learning/` (see its
[README](../scripts/learning/README.md)); decisions are D035-D039 in
[DECISIONS.md](DECISIONS.md).

## Summary

1. Forward-process RL (NFT/AdvantageFlow, PR #5) on the physical simulator did not
   learn. Small steps left the policy unchanged; larger steps degraded T (dense
   3e-5: T 0.55 -> 0.05 by update 75). Training stopped at about 150 updates.
2. Cause: the training score is mostly noise. Only 16-36% of a candidate plan's
   score variance comes from the plan itself (split-half reliability 0.14-0.29);
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
6. Next: distill the planner into the image policy, anchored to the base on T,
   then a pre-registered acceptance test on 1000 fresh scenes.

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
of the episode (8 own controls plus 24 base controls), and commits the best. This
uses the true simulator at decision time, so it is an upper bound, not a policy.

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

## 6. Distilling the planner into the image policy (in progress)

`mpc_dagger.py` runs the planner with rendering and saves, every 4 controls, the
2-frame observation, 8 samples from the image policy and a label: the planner's
actually executed next 16 targets. Round 1: 11.5k records from Y, D and L, plus
3.0k T/near-T retention records (earlier oracle records of base plans). `distill.py
--anchor-shapes T T_narrow` anchors only those records to the base, so Y/D/L can
follow the teacher. Three dense students (lr 3e-5, 1e-4, and 3e-5 with a 0.1
anchor on all shapes) are training; results will be added here.

## Privileged signals and priors (disclosure)

- Teachers read the simulator's exact object and pusher state. The deployed policy
  must stay image-based; only distilled image policies count as results.
- Rewards use privileged geometry: letter rectangles, goal pose, coverage.
- Push candidates are a hand-designed geometric exploration prior.
- The world model's pusher dynamics and contact geometry are copied from the
  simulator. Only the object's motion in contact is learned.
- The oracle uses exact simulator replay at decision time; it is an upper bound only.
- No human demonstrations beyond the released T checkpoint's expert training data.

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

- Whether the image policy can absorb a closed-loop planner's behavior in 8-control
  chunks. If not, collect DAgger labels on student-visited states (`--beta < 1`),
  or execute fewer controls per plan.
- The planner's leftover T rotation.
- A planner bug found by the new test: push candidates jumped onto the safe circle
  when the pusher started outside it. All planner numbers above predate the fix;
  the fixed version is being re-evaluated.
- Final claims need the 1000-scene acceptance test against the base, with T and
  near-T non-inferiority and new-letter superiority fixed in advance.
