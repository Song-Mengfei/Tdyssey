# Adapting SuperMimic to the PushT letter-adaptation project

Date: 2026-10-25. Status: design note and experiment plan. **Nothing in this note has been run.
No training was done and no GPU job was started for it.** Paper facts come from the arXiv
2610.28480 TeX source (`sections/method.tex`, `results.tex`, `appendix.tex`, `introduction.tex`,
`related-work.tex`, `conclusion.tex`). Project facts come from
[project_measurements.md](project_measurements.md), the
[RL learning-signal report](../../reports/Diffusion%20policy%20RL%20learning%20signal.md), the oracle and
distillation results the team reported on 2026-10-25, the teammates' `pr-7` world-model documents,
and three small CPU-only analyses made for this note (Appendix A).

Terms used throughout:

- **BPTT** (backpropagation through time) means differentiating the summed reward of an imagined
  rollout with respect to the policy weights, through every model step.
- **Critic** means a learned value function that estimates the return after the imagined horizon.
- **GAE** (generalized advantage estimation) is the λ-weighted return the critic is trained on.
- **Teacher/student** means a policy that sees privileged simulator state (the teacher), whose
  behaviour is then copied into the deployed image policy (the student).
- **DAgger** is the standard way of doing that copying: label the states the *student* visits with
  the teacher's actions.

## 0. Short answer

1. **The method's core transfers, but not as a drop-in.** SuperMimic's recipe is a small
   deterministic state-space world model fit from logged simulator transitions, plus short imagined
   rollouts (H = 32 steps) differentiated end to end, plus a critic that covers the return after
   the horizon. PushT suits this better than an image world model does. Its object is
   **quasi-static**: the official environment sets `space.damping = 0`, and in our 282,855 logged
   transitions the object's velocity is exactly 0.0 after every control without contact. The
   state is about ten numbers, the physics is the same for every letter, and the pusher's own
   motion is a known linear PD law that contact does not change.
2. **What does not transfer** is the deployed actor and the reward. SuperMimic's actor reads
   proprioception, which its world model predicts, so the actor runs closed-loop inside imagination.
   Ours reads two 96×96 RGB frames that a state model cannot produce. SuperMimic's tracking reward
   is dense and smooth at every step. Our coverage is flat whenever the pusher is not touching the
   object, which is 60% of controls, and success needs about 1.4–2 simulator pixels of alignment.
3. **Recommended route (Design A):** train a privileged *state* teacher with SuperMimic-style BPTT
   and a GAE critic through a learned PushT state model, initialized by imitating the base. Check
   the teacher in the real simulator, then copy it into the image diffusion policy with the
   existing distillation code, run as DAgger. The teacher runs on CPU in the real simulator in
   about 0.2 s per episode with rendering off; an oracle episode takes minutes. So
   teacher labels are roughly 1000× cheaper than oracle labels, and they are a deterministic
   function of state. That directly attacks the measured failure, in which only 6–29% of score
   variance came from the plan itself.
4. **Honest caveat:** SuperMimic's own limitations list "No terrain or object interaction", and
   PushT is entirely object interaction. Because our simulator is cheap on CPU, plain PPO on the
   state teacher in the real simulator ("PPO-Sim") is a strong, cheap baseline. BPTT's advantage
   has to be measured here, not assumed. Either way, the teacher-then-distill pipeline is usable.
5. **Design B** fine-tunes the image policy through its DDIM-10 sampler, one executed chunk at a
   time, into the state model, with the teacher's critic for the rest (DRaFT-style). It is a later
   refinement step, not a first experiment. **Design C** uses the teammates' GPC image world model.
   It is not viable for gradient-based policy learning now: one 96-pixel image pixel is 5.3
   simulator pixels, which is wider than the whole success tolerance, and the model is not
   contact-calibrated.
6. **First experiments:** five gated steps, ordered by information per GPU-hour. Steps 1–3 are
   mostly CPU and decide whether a state model can rank plans and give gradients that point the
   right way in the *real* simulator. Total is about 8–13 GPU-hours out of the remaining ~55.

## 1. What SuperMimic does

### 1.1 The training loop

One goal-conditioned controller is trained over a large motion library. Three phases alternate
inside every iteration (Algorithm A.1):

1. **Collect.** The current policy runs in a GPU simulator (MuJoCo Warp, 4096 environments per
   GPU) with action noise. Transitions go into a fixed-size replay buffer that overwrites the
   oldest data.
2. **Fit the world model.** The model is fit on k = 8-step windows sampled from the buffer.
3. **Improve the policy.** The policy is improved by BPTT through the *frozen* world model over
   H = 32 imagined steps. Each rollout is seeded from a replay-buffer state ("Dyna style"), and a
   critic trained in the same phase bootstraps the return after step H.

The authors stress that old data stays valid: "Transitions from earlier policies remain valid for
fitting the world model, because the dynamics do not depend on the policy that collected them."
Action noise is for the model fit only: "The noise widens the range of actions on which the world
model is fit; it serves only the model fit and is optional." The replay buffer stores the
executed noisy action.

### 1.2 World model

- **Interface.** Deterministic. The simulator's randomness has known sources, so the model
  "conditions on the per-episode sim context ξ (friction, calibration offsets, and torso
  center-of-mass) and predicts the noise-free observation". Sensor noise is added outside the
  model.
- **Heads.** ŝ_{t+1} = f_ω(ŝ_t, a_t, ξ); r̂_{t+1} = r_ω(ŝ_t, a_t, g_t); ô_t = g_ω(ŝ_t, ξ);
  d̂_t = d_ω(ŝ_t, g_t). "Each head is either learned or computed analytically". The TeX does not
  say which heads were analytic in the reported runs.
- **Architecture** (Table A.4). The dynamics network is a plain MLP with hidden widths (2048, 2048,
  2048). Its inputs are ŝ_t, a_t and ξ, and its output is ŝ_{t+1}. Nothing in the model is
  contact-specific.
- **Loss** (Eq. 1). An autoregressive k = 8 window is seeded at the logged state and rolled forward
  on logged actions:
  Σ_j ‖ŝ_j − s_j‖² + (r̂_j − r_j)² + ‖ô_j − o_j‖² + BCE(d̂_j, d_j), and analytic heads drop
  their term. The authors claim the model "remains stable over indefinitely long rollouts, because
  the k-step autoregressive fit trains it to correct its own prediction errors."
- **Data and refresh.** The buffer holds 1000 steps per environment × 4096 environments, about
  4.1M transitions. The update-to-data ratio is 0.4, with one policy update per world-model update.
  Action noise is N(0, 0.1²) per normalized action dimension.
- **What was removed to keep the model learnable.** BeyondMimic's root-velocity pushes were
  dropped: "such pushes inject state discontinuities that the deterministic world model cannot
  capture".

### 1.3 Policy objective and critic

- **Policy.** Deterministic, a_t = π_θ(ô_t + η_t, g_t), and the actor sees the model's predicted
  observation plus re-added sensor noise. Survival weight: c_0 = 1, c_{t+1} = c_t(1 − d̂_{t+1}).
  The predicted termination probability keeps it differentiable.
- **Objective** (Eq. 2). Ĵ(θ) = E[Σ_{t<H} γ^t ĉ_t r̂_{t+1} + γ^H ĉ_H V_ψ(ŝ_H, g_H, ξ)], with H = 32
  steps at 50 Hz (0.64 s). "The critic V_ψ estimates the return beyond step H, so the model must be
  accurate only within these H steps."
- **Critic** (Eq. 3). Fit to GAE returns computed backward over the *imagined* transitions only,
  with λ = 0.95 and γ = 0.99, starting from R̂_H = V_H. Targets are frozen per rollout. There is
  no target network, and training runs 5 epochs × 4 minibatches. The critic is an MLP (256, 128)
  whose inputs include privileged ŝ, g and ξ.
- **Relation to SHAC.** The method is a SHAC-style short-horizon actor-critic, except that SHAC's
  hard termination becomes the survival weight. "BPTT instead differentiates the return with
  respect to each action, so over short horizons its gradient has much lower variance" than PPO's
  likelihood-ratio gradient.

### 1.4 Gradients and non-smooth contacts

The paper has **no explicit contact smoothing**, ensemble or uncertainty penalty. Contact is
handled by the following:

- The learned MLP surrogate is smooth by construction. "Because these gradients come from the
  learned model rather than from differentiating the simulator, SuperMimic remains compatible with
  rigid-body simulators, including those with non-smooth contacts and non-differentiable
  implementations."
- The horizon is short and backed by the critic.
- Policy gradients are clipped at global norm 1.0, then element-wise at 0.1. Adam uses
  β = (0.9, 0.95). The world-model gradients are clipped at global norm 1.0.
- The world model is held fixed (stop-gradient) inside the policy's backward pass.
- The policy is deterministic in imagination. That is why the self-collision penalty could be
  removed: "Our policy is deterministic in imagined rollouts, so the exploit has no payoff."
- Discontinuous perturbations were removed from training (§1.2).

### 1.5 Reward, observation, action and context

- **Reward** (Table A.2). This is BeyondMimic's tracking reward, a sum of
  exp(−mean‖error‖²/σ²) kernels on body position (σ = 0.3 m), orientation (0.4 rad), linear and
  angular velocity (1.0, 3.14), and root position and orientation (weight 0.5). Action smoothness
  is −0.1 and the joint limit is −2. The self-collision term is removed. Every term is multiplied
  by Δt = 0.02 s. The reward is dense at every step because a reference pose exists at every step.
- **Observation.** 29 joint positions, 29 joint velocities, 3 IMU angular velocities, 3
  projected-gravity components and the 29 previous actions, with a 10-frame history. The goal is
  K = 10 future reference frames.
- **Action.** A joint setpoint with low PD gains, effectively a torque proxy.
- **Termination.** Root height error > 0.25 m or tilt > 0.8.
- **Context ξ.** Friction U[0.3, 1.2], joint offsets ±0.01 rad and torso centre-of-mass offsets.

### 1.6 Hyperparameters (Table A.5)

| Item | Value |
| --- | --- |
| World-model window k | 8; Adam lr 2e-4; 10 batches of 2048 windows per update; clip 1.0 |
| Imagination horizon H | 32 (50 Hz) |
| Policy optimizer | Adam β (0.9, 0.95), lr 2e-4, 4096 imagined rollouts per update, clip 1.0 then 0.1 element-wise |
| Discount, GAE | γ 0.99, λ 0.95 |
| Data | 4096 sim envs per GPU, noise N(0, 0.1²), buffer 1000 steps/env |
| Schedule | UTD 0.4; 1 policy update per WM update; cosine lr decay to 0.03× over 250k iterations; 320k iterations |
| Networks | dynamics MLP (2048×3); motion encoder (2048, 1024, 512, 256); shared decoder (2048, 1024, 512, 256); critic (256, 128); 32-d tanh latent |

### 1.7 Results and ablations

- **Against SONIC** (a 128-GPU PPO pipeline). In **3 h on one H100**, SuperMimic surpasses
  the return of SONIC's released checkpoint: 128 L40 GPUs × 7 days ≈ 21,000 GPU-h, which the
  paper reports as >50× wall-clock and >1000× GPU-hours (~56× and ~7200× by its own accounting).
  On the 129,514-motion benchmark it fails on 25 motions, against SONIC's 179, and on 0 after
  9.8 h on two GPUs. SONIC fails all 44 flips and all 22 side rolls. The benchmark motions are
  training motions: "the counts measure how fully each policy covers these training motions, not
  generalization".
- **Controlled comparison on LAFAN1** (40 clips, 2.45 h of motion). SuperMimic reaches return 0.82
  in 1.2 h on 1 GPU. PPO-Sim, with the same observations, reward, networks and 8192 envs/GPU, ends
  at 0.60, 0.70 and 0.72 after 18 h on 1, 4 and 8 GPUs.
- **Dataset sweep** from 1 to 264 h of motion. GPU-h to the target return were 0.54, 1.35, 1.91
  and 1.89. The 8-step prediction error (×10⁻³) was 6.8, 7.4, 11.0 and 9.9. That error "counts the
  state only, is normalized by a predefined nominal error, and is measured online on policy
  rollouts".
- **Speed.** The world model runs 11× faster than MuJoCo Warp at 2k environments, 26× at 16k and
  62× at 64k.
- **First-order versus zeroth-order gradients on the same model.** SuperMimic reaches the best
  return of PPO-on-world-model with about **64× fewer imagined steps**, "so the gain is not just
  rollout speed".
- **Horizon × critic ablation** (LAFAN1, training return):

  | H | 8 | 16 | 32 | 64 | 128 |
  | --- | --- | --- | --- | --- | --- |
  | Final return, critic | 0.392 | 0.843 | **0.866** | 0.863 | 0.823 |
  | Final return, no critic | 0.087 | 0.25 | 0.84 | 0.862 | 0.835 |
  | Hours to 0.82, critic | not reached | 1.19 | **0.88** | 1.70 | 4.62 |
  | Hours to 0.82, no critic | not reached | not reached | 1.37 | 1.34 | 3.80 |

  Without the critic, training fails at H ≤ 16. With it, H = 16 matches the critic-free H = 32.
  H = 128 is the longest horizon that fits on an H100. **At matched return the critic still
  changes behaviour:** under 0.5–4 m/s pushes, the critic policy settles in 0.5–2.4 s, while the
  critic-free policy "never settles".
- **Real robot.** The policy transfers to a Unitree G1 (skip, side roll, 90° and 360° flips).
- **Downstream velocity interface** (preliminary, one seed each). An RL-trained encoder beats a
  distillation encoder on 650 of 672 step commands, with median relative errors of 0.16 versus
  0.49.
- **No direct model-accuracy ablation.** The paper reports the prediction error across dataset
  sizes but never degrades model accuracy to see how return changes. It reports no seed variation:
  "We ran several random seeds and found training very stable, which we attribute to the dense
  reward ... The figures therefore do not report seed variation."

### 1.8 Limitations

The authors list three limitations:

- Multi-interface training is understudied.
- The dynamics come from the simulator only, with no real-robot data.
- **"No terrain or object interaction. SuperMimic trains only on flat ground."**

A reader should add five gaps:

- Stability is attributed to a dense reward; our reward is not dense.
- Model exploitation is not analysed.
- There is no generalization test, since the benchmark motions are training motions.
- The normalizer for the prediction-error metric is not given.
- All gradients flow through a proprioceptive actor that is cheap to differentiate.

## 2. What transfers to PushT

### 2.1 PushT facts that decide the design

| Fact | Value | Source |
| --- | --- | --- |
| Physics | pymunk, 100 Hz substeps, 10 per control; `space.damping = 0`; pusher is a **kinematic** disk (r = 15) driven by PD k_p = 100, k_v = 20; object mass 1; friction 1 | `pusht_env.py`, `tdyssey/envs/official.py` |
| Object quasi-static | object velocity is **exactly 0.0** after every no-contact control; it carries over for one substep only | Appendix A.1 (282,855 transitions) |
| Contact sparsity | contact in 39.5% of controls; object moves in 38.8%; contact runs median 9, mean 13.5 controls; 42% of controls have no contact and the pusher >80 px from the object centre | A.1 |
| Motion scale per control | object (when moving) median 2.4 px, p99 10.5 px; rotation median 1.2°, p99 7.2°; pusher median 7 px, p99 25 px | A.1 |
| Success tolerance | 1 px of shift costs 0.025 (T) to 0.037 (Y) coverage; **0.95 needs ≤ 2 px (T, L, T_narrow), ~1.4 px (Y), ~1.5 px (D) of pure shift, or ≤ ~2.5° (T) / ~1.7° (Y) of pure rotation**; one 96-px image pixel = 5.33 sim px | A.2 |
| Near-goal data | states with coverage ≥ 0.85 *and* contact: T 1676, T_narrow 1570, Y 710, D 8, L 0 | A.3 |
| Existing data | `runs/value-data` (1000 episodes) has **state, coverage and contacts but no actions and no pusher velocity**, so it cannot fit dynamics. `runs/dyn-data` (6 × 250 episodes, action noise 0/10/25 px, records `action` and `pusher_velocity`) is being collected now | `base_rollouts.py`, `queue.txt` |
| Simulator cost | ~3.7 ms per rendered control, ~82% of it rendering, so physics plus coverage is ≈0.7 ms per control per CPU core with rendering off (estimate; measure). The image policy is the expensive part (~900 controls/s aggregate on the GPU) | `fast_replay.py`, measurements |
| Geometry | every letter is a union of non-overlapping axis-aligned rectangles in the body frame; T's body origin is not its centre of mass (COG (0, 45)); T and T_narrow use legacy moment 3000 | `official_shape`, `block_letter` |

### 2.2 Transfer table

| Aspect | SuperMimic | PushT | Consequence |
| --- | --- | --- | --- |
| State | ~90-d proprioception plus 10-frame history | ~10-d: object pose and carried-over velocity, pusher position and velocity | A *small* MLP world model is enough. The pusher part can be an exact analytic head. |
| Action → state | every action moves the robot directly | the object moves only through contact | true ∂object/∂action = 0 in 60% of controls. Without contact inside H, the only gradient is through the pusher into the critic. |
| Reward | dense exp-kernel tracking of a per-step reference | fixed-goal coverage: piecewise smooth, zero gradient with no overlap, threshold at 0.95 | need an analytic coverage head plus a goal-pose keypoint exp-kernel (the BeyondMimic form, with the goal as the "reference") plus a soft success term. Disclose it as privileged geometry. |
| Termination | fall is bad, so the survival weight shrinks the return | success is good and absorbing | invert: continuation weight c_{t+1} = c_t(1 − d̂_success), bonus at success, no bootstrap after success, a time limit that is bootstrapped (LEARNING_PLAN rule) |
| Horizon / stability | tracking feedback contracts errors | chaotic: near-identical policies flip 29–38% of scenes, and outcomes diverge within 30–150 controls | H = 32 controls (3.2 s, about 3 pushes) is at the edge. Sweep H and expect a shorter optimum. |
| Actor input | proprioception, which the model predicts, so the actor runs closed-loop in imagination | 2 RGB frames that a state model cannot produce | the image actor cannot run inside a state model past the first chunk. Hence a state teacher plus distillation (A), or one-chunk truncation (B). |
| Policy class | deterministic MLP | 251M-parameter U-Net with a 100-step DDPM sampler (DDIM-10 within noise) | BPTT through the actor costs 10 U-Net passes per decision. Use an MLP teacher, or BPTT through DDIM-10 only for one chunk. |
| Context ξ | domain-randomized physics | fixed physics; *shape geometry and moment* vary | condition on geometry features (contact point, normal, moment), not a letter ID. That fits the PROJECT.md preference against mandatory letter-ID lookups. |
| Amortization | one model across 129k motions | one model across letters | contact mechanics are shared and SE(2)-invariant away from walls. D and L lack near-goal data, but physics does not depend on distance to the goal, so object-frame models can transfer from T/Y contact data. |
| Data | 4.1M on-policy transitions, refreshed | ~0.42M base transitions (dyn-data) | refresh with teacher rollouts is cheap: the CPU simulator needs no rendering for a state teacher. |
| Compute advantage | the world model beats an expensive simulator by 10–62× | pymunk is cheap on CPU for state policies | **PPO-Sim on the state teacher is a cheap, strong baseline.** BPTT must win on precision or sample efficiency, not on simulator cost. |

### 2.3 What the analogy buys and what it does not

- **The noise problem that sank forward-process RL disappears inside the model.** A deterministic
  model and a deterministic teacher give a gradient that is a deterministic function of the start
  state, with no random continuation. Real-simulator evaluation is still chaotic, so final claims
  still need the report's paired, pre-registered test.
- **Fine alignment is where analytic gradients should help most and where model error hurts most.**
  The Y and T_narrow near misses (peaks 0.81–0.94) need corrections of about 1–3 px. That is the
  local, smooth regime where a gradient beats sampling, provided the model's 8-control error is
  well under 1 px in those contact configurations.
- **D and L are bottlenecked by reaching contact and by multi-push strategy.** That needs the
  critic, or a disclosed goal-proximal start-state curriculum. SuperMimic's push-recovery result
  is the closest analogue: the critic matters when the reward lies beyond the horizon.

## 3. Candidate designs, ranked

### 3.0 Shared building blocks

#### 3.0.1 A privileged PushT state world model

- **State** at a control boundary: object centre-of-mass pose (x, y, θ) and object velocity
  (vx, vy, ω), plus pusher (x, y, vx, vy). For T, convert `physical_state`'s body-origin position to
  the centre of mass with `T_REFERENCE_COG`. Context: the shape's rectangles in the
  centre-of-mass frame, plus the *simulator's* moment of inertia.
- **Action**: the executed absolute pusher target in pixels. `action[t]` maps `state[t]` to
  `state[t+1]` in `base_rollouts.py`.
- **Pusher head (analytic, exact).** Ten substeps of v ← v + 0.01·(100(a − p) − 20v), then
  p ← p + 0.01·v. Contact does not affect a kinematic body, so this should reproduce the logged
  pusher to float precision. Step 1 checks that. The head also returns the 10 substep positions.
- **Object head (learned).** Work in the object frame, so the model is invariant to where the
  object is. Inputs: the 10 substep pusher positions and velocities relative to the object; the
  signed distance from the pusher disk to the object polygon at each substep; the closest point and
  normal at the first-contact substep; the carried-over object velocity; the moment; optionally a
  16-d embedding of a 32×32 body-frame occupancy mask, for multi-contact global geometry.
  Outputs: Δx, Δy, Δθ in the object frame, plus the end-of-control velocities. Size: MLP 4×512
  with SiLU activations, about 1M parameters, as an ensemble of 5.
- **Hard no-contact rule.** If the swept disk stays more than 3 px from the polygon and the
  carried-over velocity is zero, output exactly zero motion. Step 1 must show the rule never zeroes
  a step that actually moved. This removes "phantom pushes": the teammates' image model predicts
  ~0.85 px of motion on miss windows where the actual motion is 0.008 px.
- **Fit.** SuperMimic Eq. 1, with k = 8 (exactly one executed chunk), seeded at logged states and
  using logged noisy actions, on the state term only (the other heads are analytic). Contact windows
  get ×3 weight, since no-contact steps are exact by rule. The 25-px noise parts of dyn-data match
  SuperMimic's 0.1 normalized-action noise: the official action normalizer scale is
  0.00401/0.00412 per px (x 12–511 px and y 25–511 px mapped to [−1, 1]), so 0.1 normalized unit
  is ≈ 24–25 px.
- **Walls.** Measure how often the object touches a wall (Step 1). Add wall-distance inputs only
  if it exceeds ~1% of contact steps; otherwise exclude near-wall starts from imagination.
- **Irreducible error floor.** pymunk keeps cached contact impulses that a state snapshot does not
  carry; this is why `restore_state` and exact replay can differ mid-contact. Measure the floor as
  "restore the logged state, step the logged action" against the logged next state. If the floor
  exceeds the accuracy targets below, add the previous action and the previous pusher velocity as
  inputs.

#### 3.0.2 Smooth reward surrogate for coverage

All of these are computed from (predicted) state, so they are *analytic heads* and must be
disclosed as privileged geometry and reward.

- **Exact differentiable coverage.** Both the goal and the object are unions of disjoint
  rectangles, so overlap area = Σ_{i,j} area(goal rectangle i ∩ object rectangle j). Each term
  clips one convex rectangle against another (≤ 8 vertices, shoelace area). T has 2×2 pairs, Y
  8×8 and D 12×12. The result equals the shapely coverage exactly and is differentiable almost
  everywhere. It has zero gradient without overlap, so it cannot be the only term.
- **Goal-pose keypoint kernel.** This is the BeyondMimic body-position term with the goal pose
  as the reference. d² = mean_k ‖T_s k − T_goal k‖² over rectangle corners, and
  φ_kp = ½[exp(−d²/30²) + exp(−d²/4²)]. The coarse scale guides from far away; the fine scale is
  about the size of the 0.90–0.95 band (A.2). The five study letters have rotational-symmetry order
  1 (`block_letter` computes it), so there is one minimum. Letters with 2-fold symmetry need the
  minimum over symmetric goal poses.
- **Soft success.** d̂ = σ((cov − 0.95)/0.01) gives a continuation weight c_{t+1} = c_t(1 − d̂)
  and a bonus B·d̂. After success: no reward and no bootstrap. The 300-control limit is truncation
  and gets bootstrapped (LEARNING_PLAN rule). The critic, not the actor, may see the time remaining.
- **Default reward.** r_{t+1} = 0.5·cov + 0.5·φ_kp + B·d̂, with B ≈ 10. These are level rewards,
  as in SuperMimic.
  - **Ablation.** Run LEARNING_PLAN's potential-based version, success + β(γΦ' − Φ). In BPTT a
    potential-based term telescopes to γ^H Φ(ŝ_H) − Φ(s_0). It therefore keeps giving an exact
    analytic gradient early on, but with a perfect critic it cancels to "probability of later
    success". That is flat on D and L, which is the red team's cancellation argument again. Level
    rewards change the optimal policy toward "be near the goal early", which is what the D/L
    coverage endpoint wants.
  - **Approach term.** Add −w·relu(distance(pusher, object) − 15) only if Step 4 shows the teacher
    stalls without contact. It is a disclosed geometric prior.

#### 3.0.3 How to validate the model

Each check is in Section 4 with numbers.

- **V1, offline held-out scenes.** Split by scene seed, with all five shapes of a seed together,
  because matched resets share poses. Measure position, angle and coverage error at 1, 8 and 32
  controls, per shape and per slice (contact, near-contact, free), against persistence (zero
  motion), which is a strong baseline when 60% of steps have no motion.
- **V2, amortization.** Train without L and test on L, compared with the all-shape model.
- **V3, irreducible floor** (§3.0.1).
- **V4, physical decision test.** Restore real contexts by exact prefix replay (`fast_replay.py`).
  Check that model-scored ranking of base plans matches exact-simulator ranking, and that model
  gradients improve the *real* outcome. This is AGENTS.md's "validate model-selected optimistic
  actions".
- **V5, online during policy training.** Track SuperMimic's metric, the 8-step error on the
  teacher's own real rollouts, and the **imagination gap**: imagined minus real return from the
  same start states.

### 3.1 Design A (rank 1): privileged state teacher via SuperMimic BPTT, distilled into the image policy

**Teacher.**

- **Inputs.** An MLP (3×512) receives the object pose relative to the goal (x, y, cos, sin), the
  pusher position and velocity in both the world and object frames, and the geometry embedding.
  It does not get the time remaining.
- **Outputs.** One **8-control open-loop chunk** of bounded pusher offsets (tanh × 40 px,
  cumulative from the current pusher position). Executing 8 controls open-loop matches what the
  student can do. A teacher that re-decides every control may be impossible for an open-loop
  student to copy.
- **Ablation.** Compare against a per-control teacher and measure the distillation gap.

**Initialization.** Fit the teacher to the base's executed chunks in dyn-data's noise-0 parts,
with mean-squared error. The base is multimodal, so mean-squared imitation may average opposite
pushes into bad ones. Measure the result in the real simulator first. If it is far below the base,
use a 3–5 component Gaussian-mixture head and act with the mean of the most likely component, or
skip imitation and start imagination from in-contact states.

**SuperMimic loop.**

- **Imagination.** H = 32 controls (4 teacher decisions), batch 2048, γ = 0.99 per control. The
  critic is an MLP (256, 128) on state, geometry and time remaining, trained with GAE λ = 0.95 on
  imagined transitions. Adam β (0.9, 0.95), lr 2e-4 with cosine decay, clip 1.0 then 0.1.
- **Start states.**
  - 50% uniform from the replay buffer.
  - 40% from SuperMimic's return-based sampler. Our version favours states with low critic value
    and near-miss states, which emphasizes Y, T_narrow, D and L.
  - 10% synthetic goal-proximal states: goal pose plus 2–30 px and 2–20° offsets, with the
    pusher placed near the object. This is possible because the physics does not depend on
    distance to the goal. It is a declared curriculum (LEARNING_PLAN), and its evaluation uses
    `reset_to_state` in the real simulator.
- **Refresh.** Every ~2k updates, run 500 teacher episodes in the real simulator on CPU with
  rendering off (`env._get_obs = lambda: None`, as `fast_replay.py` does), 50% of them with 25-px
  action noise, then refit the model for ~1k steps. Each refresh is a recorded model version: the
  model is frozen within a phase and refreshed between phases, which is how PROJECT.md's "freeze
  the world model within a policy-learning phase" rule meets SuperMimic's joint loop.

**Controls.** These make the result interpretable and mirror the paper's controlled comparisons:

- **PPO-Sim.** The same teacher network, reward and initialization, trained by PPO in the real
  simulator on CPU at matched wall-clock. This is also AGENTS.md's "first verify direct simulator
  policy optimization".
- **PPO-WM**, matching the paper's Fig. 5b. Same model, gradient type changed.
- **H ∈ {8, 16, 32, 64} × critic on/off**, one seed, matching the paper's Fig. 4b/c on a contact
  task. This is cheap with small networks and is analysis the course values.
- **MPPI-in-WM.** A sampling-based planner in the model, used as a non-amortized teacher.

**Distillation.**

- **Data format.** Render teacher rollouts on CPU. Each record holds the two-frame image history,
  `agent_pos`, and a 16-step label in the official layout: `x[0]` is the previous executed action,
  `x[1:9]` the teacher's chunk, `x[9:16]` the first 7 actions of the teacher's *next* decision,
  obtained by simulating the chunk.
- **Training.** Train with `tools/distill.py`. A new loader writes records with N = 1 candidate
  (choice 0 = teacher plan), and base samples feed the existing anchor term. Prefer DDPM samples
  at T states; DDIM-10 is acceptable elsewhere if cost matters.
- **DAgger rounds.** Run the student in the real simulator, capture its states at decisions,
  relabel them with the teacher (exact replay to restore), aggregate and retrain.
- **Evaluation.** Use the existing validation pool (400000+20i) and the report's test protocol.

**Fit with code and data.**

- **Reused as is:** dyn-data (fit), value-data (start states, critic warm-up, coverage-head
  test), `envs/official.py` (geometry, inertia, `capture_state`/`restore_state`, `reset_to_state`),
  `fast_replay.py` (exact contexts), `distill.py` and `base_rollouts.py eval --student`.
- **New code:** `tools/state_wm.py` (windows, model, analytic heads, V1–V3),
  `tools/state_teacher.py` (imitation start, BPTT plus critic, refresh, CPU evaluation) and
  `tools/teacher_data.py` (rendered DAgger records).
- **Focused checks:** pusher-head parity, coverage-head parity with shapely, and window timing.

**Compute estimates** (H100 shared with running jobs; CPU has ~64 cores, mostly free):

| Part | Estimate |
| --- | --- |
| Model fit, 425k transitions, ~20k steps | 10–20 GPU-min, or ~1 h on CPU |
| BPTT, 20k updates × 2048 × 32 controls | ~65k small-MLP steps per update, ~20–60 ms each, so ~0.3–1 GPU-h; the H sweep (8 short runs) adds ~2–3 GPU-h |
| Refresh and evaluation | 500 episodes × 300 controls ≈ 150k controls ≈ 3 s on 40 cores (at the ≈0.7 ms/control estimate) |
| PPO-Sim control | ~60k controls/s on 40 cores, so ~200M controls/h (estimate; measure) |
| Distillation, 3 DAgger rounds | ~70k records per round; ~1–2 GPU-h per round including 200-scene validations, so ~4–8 GPU-h |

**Failure modes and mitigations.**

1. **Model exploitation.** The teacher finds grazing pushes or corner rotations that the model
   overrates. *Detect:* V5's imagination gap, online 8-step error, ensemble disagreement.
   *Mitigate:* refresh with the teacher's own data (SuperMimic's main defence), an ensemble
   disagreement penalty or rollout truncation (a disclosed deviation), and the hard no-contact rule.
2. **Flat gradients without contact.** *Mitigate:* the critic's gradient with respect to pusher
   position, an imitation start that already makes contact, in-contact and goal-proximal start
   states, and the optional approach term.
3. **Contact discontinuities.** First-order gradients can be biased or high-variance at contact
   switches (Suh et al. 2022). *Mitigate:* the smooth model, averaging gradients over a few
   action-noise copies in imagination (randomized smoothing), shorter H, clipping, and comparison
   with PPO-WM and MPPI-WM.
4. **Compounding error and chaos at long H.** *Detect:* gradient norm and the imagination gap
   against H. *Mitigate:* use the H sweep result; k = 8 training.
5. **Mean-squared imitation of a multimodal base gives a weak start.** *Mitigate:* the mixture head,
   or starting from contact states.
6. **The student cannot copy the teacher.** Alignment to 1.4–2 px is about 0.26–0.38 image
   pixels; the image encoder is frozen and was trained on T only; LoRA capacity is limited.
   *Detect:* the frozen-encoder pose probe (Step 2b) before any distillation. The same probe tells
   whether the *oracle* distillation's small gains were limited by perception.
7. **T regression.** Distill on all shapes with the base anchor. If the teacher is not at least
   the base on T, keep the base as T's label source. Monitor T with the report's 60-episode alarm.
8. **Scope deviation.** The project prefers an image world model. This route uses a privileged
   state model and teacher at training time only, and the deployed actor stays image-only. Record
   it in DECISIONS.md and report it as privileged-teacher adaptation, not visual-world-model RL.

### 3.2 Design B (rank 2): BPTT through the image policy's DDIM-10 sampler for one chunk, critic for the rest

**Objective.** For a *real* state s_t, render its real two-frame observation and fix a sampling
noise z. The executed actions are a_{1:8} = DDIM-10_θ(z; o_t), unnormalized at indices 1..8. Roll
the state model 8 controls. Maximize Σ_{k≤8} γ^k r(ŝ_{t+k}) + γ^8 ĉ V_ψ(ŝ_{t+8}) − λ·(anchor),
where the anchor is `distill.py`'s denoising-MSE to the frozen base on base samples at all shapes.
This is DRaFT ("directly fine-tuning diffusion models on differentiable rewards"). DRaFT-K
backpropagates through only the last K = 1–3 sampler steps, which cuts cost and is reported to
work well. Train with LoRA. Diffusion-QL similarly backpropagates a critic through a short reverse
chain.

**Which critic.** The student cannot be rolled in a state model past one chunk, so its own value
can only come from noisy real-simulator Monte Carlo, which is what failed before (R² 0.64, no
ranking gain). The coherent choice is the **Design A teacher's critic**, V^teacher: "push so that
the teacher could finish from here". So B is a refinement stage *after* A. It is useful when the
DAgger student is left with a gap that shows up as a consistent bias the critic can see.

**Why images after the first chunk are the obstacle.**

1. **The official renderer is pygame on CPU and not differentiable.** Re-rendering imagined
   states for every rollout and every chunk boundary costs about 3 ms per frame. With
   stop-gradient observations that is still about 1–2 s per update at batch 2048 × 3 boundaries.
2. **A differentiable re-renderer is feasible but fragile.** The scene is flat polygons and a
   disk, so one could be written. But it must match the official images closely enough for the
   frozen encoder, and the teammates' palette study shows this policy is appearance-sensitive. Its
   gradients exist only at edges.
3. **Resolution.** One image pixel is 5.3 simulator pixels, so the observation pathway barely
   encodes the 1–2 px corrections the reward cares about.
4. **Cost.** Each extra chunk adds 2 encoder passes and 10 U-Net passes, forward and backward.

Truncating at one chunk and bootstrapping with a state critic avoids all four, because the only
image is the real o_t.

**Deployment.** The deployed sampler would become **DDIM-10**: 35/34/6 against DDPM-100's 38/38/7
on T, T_narrow and Y over 100 test scenes. That is within noise, but it is a separate policy with
its own baseline. Training through DDIM and deploying DDPM is a mismatch to avoid.

**Compute.** One update is 10 (or K) U-Net passes forward and backward at batch ~128. The U-Net
has 251M parameters but works on 16×2 sequences, so activations are small. The estimate is
~0.2–0.5 s per update, so 3–5k updates take ~0.5–1 GPU-h. A pilot including validations is
~3–5 GPU-h.

**Failure modes.**

- Over-optimizing the model and critic: sampler gradients find actions where V^teacher is
  wrong. AlignProp and DRaFT both report reward hacking and loss of diversity.
- The sampler chain amplifies gradients, so clip as SuperMimic does.
- Multimodality collapses, which is acceptable for success but must be reported.
- T forgetting; the anchor and the alarm guard it.

### 3.3 Design C (rank 3): the teammates' GPC image world model

What it would take:

- A gradient-enabled rewrite of the EDM sampler. Several denoiser passes per frame, H frames and
  backward through all of them.
- A differentiable coverage estimate from predicted pixels, such as a soft colour mask of the gray
  object against the goal.
- An image actor in the loop. That at least avoids Design B's re-rendering problem.

Why it is not viable now, from the `pr-7` measurements:

- The released sampler disables gradients and rounds pixels.
- The model is not contact-calibrated: 0 of 48 real probe steps made contact, and every prediction
  was worse than a still frame.
- Useful-motion error (0.035 against 0.160 for a still frame) stopped improving after the first
  training chunk of the 400-update run.
- Miss windows still predict ~0.85 px of motion against 0.008 px actual. Read as 96×96 frame
  pixels, that is ~4.5 simulator px, or 0.1–0.15 coverage for Y by A.2, which is larger than the
  whole near-miss gap.
- Even a perfect 96×96 frame model must resolve about 0.3 image px to tell 0.93 from 0.96 coverage.

Possible use: a *descriptive* comparison. Rank the same Step-3 base plans with the image model,
the state model and the exact simulator (~1 GPU-h). That would show quantitatively why the project
moved to a state teacher. It is not on the critical path.

### 3.4 Options set aside

- **Differentiating pymunk:** not possible.
- **Writing a differentiable PushT physics engine:** feasible for quasi-static pushing, but a
  different project, and it loses SuperMimic's simulator-agnostic point.
- **DPPO or PPO inside an image model:** keeps the critic and score-noise problems the report
  measured.
- **Retraining a state-based diffusion policy (PROJECT.md Route B):** a separate mechanism study
  that needs the original T demonstrations. It is not visual adaptation.

### 3.5 Ranking summary

| Rank | Design | What learns | GPU-h to first verdict | Main risk | Payoff if it works |
| --- | --- | --- | --- | --- | --- |
| 1 | A: state model plus BPTT teacher plus DAgger | state MLP teacher, then image student (LoRA) | ~1–2 (model, gradients, teacher), ~8–13 total | contact gradients and model exploitation; the student cannot copy fine alignment | unlimited deterministic labels; can reach behaviour outside the base's proposals; interpretable H/critic/PPO comparisons |
| 2 | B: DRaFT-style one-chunk BPTT through DDIM-10 | image student directly | ~3–5 (after A) | critic and sampler hacking; DDIM deployment | closes a residual student–teacher gap without new labels |
| 3 | C: GPC image model | image student | blocked (accuracy) | resolution and calibration | none for success in the current budget |

## 4. First-experiment plan

The steps are ordered by information per GPU-hour. Steps 1–3 can run beside the oracle and
distillation jobs now occupying the GPU, since the networks need under 2 GB. Hours are estimates.

### Step 1: data audit and analytic heads (CPU only, 0 GPU-h)

Use dyn-data parts 0–1 as soon as they finish, and value-data for state-only checks.

- **Pusher head.** Rolling the PD recursion from (p_t, v_t, a_t) must reproduce the logged
  pusher_{t+1} and velocity_{t+1}, including during contact. *Pass:* max error < 1e-3 px. *Fail:*
  learn the pusher too, and treat the "kinematic and unaffected by contact" assumption as wrong.
- **Coverage head.** Convex clipping must match `physical_metrics` coverage on all logged states.
  *Pass:* max |Δ| < 1e-4.
- **No-contact rule and wall statistics.** The hard rule must have zero false negatives on moved
  steps. This also decides whether walls need inputs (§3.0.1).
- **Splits.** Hold out 20% of dyn-data scene seeds (all shapes together) for validation, plus an
  L-free training set for V2. The training scenes 604000–609980 must not overlap validation
  (400000+), test (300000+, 900000+) or oracle (700000+) scenes.

### Step 2: fit the state model and validate offline (≤ 0.3 GPU-h, or ~1 h CPU)

Run the frozen-encoder pose probe (Step 2b) in parallel.

- **Fit** as in §3.0.1, k = 8, ensemble of 5.
- **Gate 2 (go)** on the contact slice of held-out scenes, for every shape:
  - 8-control open-loop error median ≤ 0.5 px and 0.3°, p90 ≤ 2 px and 1°;
  - |coverage error| p90 ≤ 0.01 at 8 controls and ≤ 0.03 at 32;
  - moved/not-moved agreement ≥ 98%;
  - persistence beaten ≥ 5× on contact error;
  - the L-held-out model within 1.5× of the all-shape model on L.

  The thresholds come from A.2: 1 px ≈ 0.025–0.037 coverage, and the near misses sit 0.01–0.03
  below 0.95.
- **No-go:** errors still > 2× the thresholds after one architecture revision (more inputs, or the
  floor check of V3). Then stop A and B, keep the oracle route, and report the measured contact
  error as the limit.
- **Step 2b, pose probe (≤ 0.3 GPU-h).** Render ~20k states per shape on CPU and extract frozen
  official-encoder features of the two-frame observation. Fit an MLP probe to object (x, y, cos θ,
  sin θ). *Pass:* held-out error ≤ 2 px and 2° on Y, D and L, which is the 0.95 tolerance. *Fail:*
  no distillation route, oracle or teacher, can deliver fine alignment with a frozen encoder.
  Record a decision on encoder LoRA, or restrict claims to coverage. This cheap check also explains
  the small oracle-distillation gains.

### Step 3: physical decision test (CPU heavy, ≤ 0.5 GPU-h)

No policy training happens in this step.

- **Contexts.** 300 decision contexts from held-out noise-0 episodes, 60 per shape, at least half
  in or within 30 px of contact, restored by exact prefix replay.
- **(i) Ranking.** Take 16 base plans per context (DDPM-100, batched). Score each plan's 8
  controls followed by Φ = cov + φ_kp, with the model and with the exact simulator. Report
  Spearman ρ, top-1 agreement and regret.
- **(ii) Gradient direction.** From each base plan, take one step of ascent on the model's Φ at
  1, 2 and 4 px per control. Execute the original and refined plans in the exact simulator. Compare
  against random directions of the same size.
- **Gate 3 (go):**
  - the model's pick keeps ≥ 75% of the exact oracle's selection gain on Y, T_narrow and D contact
    contexts;
  - gradient steps improve the true Φ in ≥ 70% of contact contexts, against ≤ 50% for random
    directions;
  - imagined over real improvement lies in [0.5, 2].
- **Partial.** If ranking passes but gradients fail, use the model for sampling-based planning
  (MPPI-in-WM teacher) instead of BPTT.
- **Both fail:** stop. This is exactly the "validated optimistic actions" check AGENTS.md requires
  before imagined RL.

### Step 4: state teacher (~2–4 GPU-h plus CPU)

- **4a, imitation start.** Evaluate it in the real simulator (CPU, 200 validation scenes per shape,
  deterministic). This is the reference, together with the base and the simulator oracle (Y 8/20,
  coverage 0.77; D 0.44; L 0.43).
- **4b, SuperMimic loop** with H = 32, critic, refresh every 2k updates and three refreshes.
- **4c, controls.** PPO-Sim at matched wall-clock; H ∈ {8, 16, 32, 64} × critic on/off; and,
  if affordable, PPO-WM and MPPI-WM.
- **Gate 4 (go to distillation):**
  - the best teacher beats the base by ≥ 15 points of Y success *or* ≥ 0.10 of Y/D mean final
    coverage;
  - T and T_narrow are not below the base by more than 5 points;
  - it at least matches the simulator oracle on Y or D;
  - imagination gap ≤ 20% of the imagined return.
- **If PPO-Sim beats BPTT:** distill the better teacher and report the comparison honestly. That is
  a real finding about SuperMimic outside its domain.
- **No-go:** no teacher beats the base on Y or D. Then distill the oracle instead (the existing
  path) and report the BPTT negative result with its imagination-gap and H-sweep diagnosis.

### Step 5: distill the teacher into the image policy (~5–8 GPU-h)

- **Procedure.** Round 0 uses ~2000 rendered teacher episodes. Rounds 1–2 are DAgger on student
  states, with LoRA and the base anchor on all shapes. The T monitor uses the report's 60-episode
  alarm, and validation uses 200 scenes per shape at 400000+20i.
- **Gate 5 (go to the final test):**
  - the student closes ≥ 30% of the base-to-teacher gap on Y or T_narrow success, or on D
    coverage;
  - that gain lies beyond the base-versus-base (A/A) noise band;
  - no T alarm.

  Then register at most two candidates for the report's paired, pre-registered test (T and
  T_narrow non-inferiority on 1000 scenes; Y McNemar on 600; D and L coverage). The deployed actor
  stays image-only.
- **Optional (Design B).** If Gate 5 shows a large, consistent student–teacher gap and the
  teacher critic passes a Step-3-style gradient check, run a 3–5 GPU-h one-chunk DDIM-10 refinement.

### Budget

| Step | GPU-h | CPU / wall-clock | Decides |
| --- | --- | --- | --- |
| 1 Audit and analytic heads | 0 | ~1 h compute | whether the pusher and coverage are exact and the no-contact rule is safe |
| 2 Model fit and validation, plus 2b pose probe | ≤ 0.6 | ~1–2 h | whether a state model is accurate enough; whether *any* image distillation can reach fine alignment |
| 3 Physical ranking and gradient test | ≤ 0.5 | ~2 h | whether model gradients point the right way physically |
| 4 Teacher, with controls | 2–4 | ~4–6 h | whether BPTT or PPO-Sim yields a teacher better than base and oracle |
| 5 DAgger distillation | 5–8 | ~8–12 h | whether the image policy inherits it |
| Total | **~8–13** | | leaves ≥ 40 h for the report's A/A and final-test protocol and the oracle line |

## 5. Disclosures and decisions to record

These are proposed entries for DECISIONS.md, to be added when the work starts.

1. **Privileged state teacher route.** The state model, its analytic coverage, keypoint and success
   heads, the geometry and inertia inputs, the teacher and the critic all use simulator state at
   training time only. The deployed actor keeps the official RGB and pusher-history interface, with
   no pose, letter ID or reward inputs. Report this as privileged-teacher adaptation.
2. **Model refresh cadence.** Freeze the model within each policy phase; each refresh is a recorded
   version with its own data budget.
3. **Reward surrogate.** Coverage, keypoint kernel and soft success, with weights fixed before
   training. The potential-based version runs as an ablation. The approach term and goal-proximal
   curriculum, if used, are geometric priors.
4. **Splits and scene pools**, including the L-held-out model. Shapes seen by the model or the
   teacher count as *adaptation*, not zero-shot.
5. **If Design B runs:** DDIM-10 becomes the deployed sampler, with its own baseline.
6. **Counting.** Count real controls, imagined controls, rendered frames and GPU-h separately for
   every method, including PPO-Sim.

## 6. References

- Liao et al., "SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
  Model", arXiv 2610.28480 (CoRL 2026); TeX source read locally.
- Xu et al., "Accelerated Policy Learning with Parallel Differentiable Simulation" (SHAC), ICLR
  2022, arXiv 2204.07137.
- Suh, Simchowitz, Zhang, Tedrake, "Do Differentiable Simulators Give Better Policy Gradients?",
  ICML 2022, arXiv 2202.00817: bias and variance of first-order gradients at contact
  discontinuities, and an interpolated estimator.
- Clark et al., "Directly Fine-Tuning Diffusion Models on Differentiable Rewards" (DRaFT), arXiv
  2309.17400; Prabhudesai et al., "AlignProp", arXiv 2310.03739.
- Wang, Hunt, Zhou, "Diffusion Policies as an Expressive Policy Class for Offline RL"
  (Diffusion-QL), arXiv 2208.06193.
- Chen et al., "Learning by Cheating", CoRL 2019, arXiv 1912.12294; Lee et al., "Learning
  Quadrupedal Locomotion over Challenging Terrain", arXiv 2010.11251 (privileged teacher, then
  sensor student).
- Ross, Gordon, Bagnell, "A Reduction of Imitation Learning ... to No-Regret Online Learning"
  (DAgger), arXiv 1011.0686.
- Janner et al., "When to Trust Your Model" (MBPO), arXiv 1906.08253; Chua et al. (PETS
  ensembles), arXiv 1805.12114.
- Schulman et al., "High-Dimensional Continuous Control Using GAE", arXiv 1506.02438.
- Chi et al., "Diffusion Policy", arXiv 2303.04137; Ren et al., "DPPO", arXiv 2409.00588.

## Appendix A. CPU analyses made for this note

These read existing files only. Interpreter: `$REPO/.venv/bin/python -I` (`env.sh` convention); data paths are relative to `$W`. No GPU was
used.

**A.1 Transition statistics.** Input: `runs/value-data/part{0..3}.npz`, 1000 episodes and 282,855
valid transitions. A transition t→t+1 is valid when both rows are masked in. Contact means
`contacts[t+1] > 0`. Moved means |Δposition| > 1e-3 px or |Δangle| > 1e-4 rad.

| Statistic | Value |
| --- | --- |
| Contact share; moved share | 0.395; 0.388 (moved without contact 0.009; contact without motion 0.017) |
| max object \|v\| at the end of no-contact controls | **0.0** |
| Object Δposition when moved (px), p50/p90/p99 | 2.41 / 6.15 / 10.53 |
| Object Δangle when moved (rad), p50/p90/p99 | 0.021 / 0.071 / 0.126 |
| \|Δcoverage\| when moved, p50/p90/p99 | 0.006 / 0.037 / 0.083 |
| Contact-run length (controls), p50 / p90 / mean | 9 / 29 / 13.5 (8311 runs) |
| Pusher move per control (px), p50/p90/p99 | 7.0 / 16.5 / 24.8 |
| No contact and pusher > 80 px from the object position | 0.42 |
| Per shape: contact share, mean final coverage, success | T 0.43, 0.71, 0.395; T_narrow 0.41, 0.71, 0.375; Y 0.40, 0.53, 0.065; D 0.40, 0.32, 0; L 0.35, 0.32, 0 |

**A.2 Coverage sensitivity.** Shapely overlap of each `official_shape` with itself, after rotating
about the centre of mass (T: (0, 45); others: origin) and translating. Shift values average 64
random directions.

| Shape (area px²) | shift 1 / 2 / 3 / 5 px | rotation 1 / 2 / 3 / 5° |
| --- | --- | --- |
| T (6300) | 0.975 / 0.953 / 0.927 / 0.882 | 0.980 / 0.960 / 0.940 / 0.901 |
| T_narrow (5940) | 0.975 / 0.952 / 0.926 / 0.880 | 0.980 / 0.961 / 0.942 / 0.905 |
| Y (4114) | 0.963 / 0.928 / 0.891 / 0.822 | 0.970 / 0.941 / 0.913 / 0.856 |
| D (6857) | 0.966 / 0.934 / 0.898 / 0.833 | 0.978 / 0.957 / 0.937 / 0.899 |
| L (5400) | 0.975 / 0.951 / 0.925 / 0.877 | 0.981 / 0.961 / 0.943 / 0.906 |

**A.3 Near-goal coverage of the logged data.** Same input as A.1. States with coverage ≥ 0.85, and
those also in contact: T 5219 / 1676; T_narrow 6171 / 1570; Y 3187 / 710; D 12 / 8; L 0 / 0. Mean
episode peak coverage: T 0.815, T_narrow 0.796, Y 0.693, D 0.540, L 0.431. The `value-data` files
contain keys `state, coverage, contacts, mask, success, shape, seed, goal` only.

Core of A.2, for reproduction (run with `sys.path` including the repo):

```python
from shapely.geometry import box; from shapely.ops import unary_union; from shapely import affinity
from tdyssey.envs.official import official_shape, T_REFERENCE_COG
s = official_shape("Y"); poly = unary_union([box(r.x0, r.y0, r.x1, r.y1) for r in s.components])
c = T_REFERENCE_COG if s.name == "T" else (0.0, 0.0)
cov = lambda dx, dy, th: affinity.translate(affinity.rotate(poly, th, origin=c, use_radians=True), dx, dy
                                            ).intersection(poly).area / poly.area
```
