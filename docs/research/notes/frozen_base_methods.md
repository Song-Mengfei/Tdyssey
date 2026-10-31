# Frozen-base RL improvement of pretrained diffusion robot policies (DSRL, residual RL, best-of-N / value reranking, gradient guidance)

Scope note: research current to 2026-10-25. "Project brief" refers to the coordinator's project description (official RSS 2023 PushT image Diffusion Policy CNN; 20-scene baselines T 11/20, narrow-T 11/20, Y 1/20, D 0/20, L 0/20; 16-step chunks, 8 executed; DDPM 100 steps; 16x2 initial noise per plan; ~900 controls/s aggregate; contact state restorable only by full action-prefix replay). Items labelled Inference are my reasoning, not source claims.

## Q1. DSRL / latent-noise steering: mechanism, algorithms, critic inputs, reported steps and success, pixel support, code, failure modes

### Takeaway
DSRL keeps the diffusion policy's weights frozen and runs off-policy RL (SAC) over the *initial noise* fed to a deterministic DDIM sampler. The paper reports large real-robot and π0 gains in 3.5k–11k online steps and a claimed 5–10x sample-efficiency advantage over diffusion fine-tuning methods. It needs DDIM (deterministic) sampling, and the authors say plainly that it can only exploit what the base policy can already produce. Its 32-dim noise space for π0 has the same dimension as this project's 16x2 initial noise.

### Cited Findings
**Mechanism**
- DSRL treats the frozen policy π_dp as part of the environment. The agent picks the initial noise w instead of sampling w ~ N(0, I), and a = π_dp^W(s, w) is computed by deterministic DDIM denoising. This defines a "latent-action MDP" M^W that standard RL can solve. Only forward passes through the denoiser are needed: no weights, gradients, or intermediate steps. — [DSRL arXiv 2506.15799](https://arxiv.org/html/2506.15799)
- **DSRL-SAC** runs SAC directly in M^W. **DSRL-NA (noise-aliased)** trains two critics. Q^A(s, a) is learned by TD on real executed actions (offline and online data), with target actions a' = π_dp^W(s', π^W(s')). Q^W(s, w) regresses Q^A(s, π_dp^W(s, w)) for w ~ N(0, I), which distills action-space value into noise space. The latent actor π^W maximizes Q^W. Because Q^A works on actions, NA can learn from any action data, including offline data. — [DSRL arXiv](https://arxiv.org/html/2506.15799)
- The latent actor's output is a box [−b_W, b_W]^d. The paper reports b_W in [1, 3] online and [0.5, 1.5] offline. Per-task values: Robomimic Lift/Can/Square 1.5, Transport 1; π0 Libero 1.0, Aloha 2.0, π0 real-world 2.5; WidowX 1.5–2. — [DSRL arXiv (appendix)](https://arxiv.org/html/2506.15799)
- For π0 the noise is shared across the action chunk, so actor and critic work in a **32-dimensional** space (π0: 3.3B params, chunk 50, 32-dim actions). — [DSRL arXiv](https://arxiv.org/html/2506.15799)
- DSRL-NA is used for all environments except Robomimic Transport, which uses DSRL-SAC; π0 uses DSRL-SAC. DSRL-NA needed about 2x fewer samples than DSRL-SAC on Square. — [DSRL arXiv](https://arxiv.org/html/2506.15799)

**Hyperparameters and inputs**
- Online SAC settings: lr 3e-4, batch 256, target entropy 0, τ=0.005, 3-layer actor and critic, 2 critics, hidden 2048 for most Robomimic/Gym tasks. Discount 0.99 (0.999 Square, 0.97 WidowX). 20–30 gradient steps per parallel-environment step, 4 parallel sim environments, 10 Q^W update steps (Lift/Can/Square/Gym). Initial random-noise rollouts: Lift/Can 24k steps, Square 32k, Transport 320k. Compute: a single NVIDIA A5000 in sim; π0 real-robot DSRL trained on an 8 GB RTX 3070. — [DSRL arXiv (appendix)](https://arxiv.org/html/2506.15799)
- Critic inputs: the state, plus image and proprioceptive features in real-world tasks. π0-DROID inputs were proprioception, the VLM's final hidden token (2,048-d), and visual features from three cameras. — [DSRL arXiv (appendix)](https://arxiv.org/html/2506.15799)
- Denoising steps (train / inference): Lift and Can 20/8; Square 100/8; Transport 100/100; Franka and WidowX 100/8. — [DSRL arXiv (appendix)](https://arxiv.org/html/2506.15799)
- Action chunk sizes: Robomimic 4 (Transport 8), Franka/WidowX 1, π0 Libero/Aloha 50, π0-DROID 10. — [DSRL arXiv (appendix)](https://arxiv.org/html/2506.15799)

**Pixel support**
- Pixel policies are supported. The real Franka base is a Chi et al. CNN diffusion policy with a ResNet-18 image encoder (images plus proprio). WidowX uses a DiT with ResNet-34, and π0 Libero/Aloha use visual observations. — [DSRL arXiv (appendix)](https://arxiv.org/html/2506.15799)

**Reported results**
- Real Franka pick-and-place (10 trials): base 2/10, RLPD 0/10, RLPD + interventions 0/10, **DSRL 9/10** after **3,500 online steps (~40 episodes)**. — [DSRL arXiv, Table 2](https://arxiv.org/html/2506.15799)
- π0 real world: turn on toaster 5/20 → 18/20 (80 episodes, ~11,000 steps); spoon on plate 15/20 → 19/20 (65 episodes, ~10,000 steps). π0 Libero: ~20% → ~100% after ~10,000 online samples. In the same Libero comparison, **ResiP and V-GPS did not significantly improve π0**. — [DSRL arXiv](https://arxiv.org/html/2506.15799)
- Online Robomimic (Can, Square, Transport) and Gym results appear only in figures. The text claims "at least a 5–10x improvement in efficiency" over DPPO, IDQL, DQL, DIPO and QSM. Offline-to-online on Can/Square: about 2x better sample complexity with offline data; RLPD and Cal-QL "are not able to learn at all." — [DSRL arXiv](https://arxiv.org/html/2506.15799)
- Offline OGBench (4 seeds) is mixed. DSRL vs FQL: antmaze-large 40±29 vs 80±8, cube-double 53±14 vs 36±6, scene 88±9 vs 76±9, puzzle-3x3 0±0 vs 16±5, puzzle-4x4 37±13 vs 11±3. — [DSRL arXiv, Table 1](https://arxiv.org/html/2506.15799)

**Code**
- Official code: [github.com/ajwagen/dsrl](https://github.com/ajwagen/dsrl), "Official implementation for DSRL … (CoRL 2025)". It builds on forks of DPPO and Stable-Baselines3 (`train_dsrl.py`, per-task YAML; `SACDiffusionNoise` for NA; `DiffusionPolicyEnvWrapper` for SAC). The README says: "if using a diffusion policy it must be sampled from with DDIM sampling"; suggested `action_magnitude` about 1.5. — [DSRL GitHub](https://github.com/ajwagen/dsrl); venue string via [gittrend mirror](https://gittrend.io/repo/ajwagen/dsrl)
- JAX π0 version: [nakamotoo/dsrl_pi0](https://github.com/nakamotoo/dsrl_pi0). RLinf documents a DSRL setup with a small (~500K-param) SAC agent on a frozen π0. — [RLinf docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)

**Failure modes stated by authors**
- "We do not have a guarantee that all diffusion policies are steerable." Exploration is limited by the base policy, and a highly concentrated action distribution "might not provide enough 'options'". The method needs rewards, online rollouts and resets. — [DSRL arXiv](https://arxiv.org/html/2506.15799)
- Ablation: a worse-quality base-data split slowed DSRL, but it recovered quickly. — [DSRL arXiv](https://arxiv.org/html/2506.15799)

**2026 follow-ups**
- **DLSRL** (arXiv 2609.11270, 10 Sep 2026) argues that steering only the noise cannot modulate intermediate action representations, which causes "performance degradation and inefficiency." Its actor also outputs a representation latent that feeds adapters added residually to intermediate hidden states of a frozen π0, tested on LIBERO. The abstract gives no numbers. Code: github.com/xianchaoxiu/DLSRL. — [DLSRL abstract](https://arxiv.org/abs/2609.11270)
- **Social-navigation DSRL** (arXiv 2609.24317, 21 Sep 2026) uses DSRL specifically to preserve the base's core navigation performance during on-site adaptation. The abstract gives no formal never-worse guarantee and no numbers. — [arXiv 2609.24317](https://arxiv.org/abs/2609.24317)
- **OGPO** (arXiv 2605.03065, May 2026; Levine, Gupta, Simchowitz et al.) is an off-policy *full* fine-tuning method for diffusion/flow policies. Its abstract claims it "drastically outperforms" policy-steering and residual-correction alternatives, and can bring poorly initialized BC policies to near-full success without expert data in the replay buffer. The abstract gives no numbers, and I did not verify that DSRL is the named steering baseline. — [OGPO abstract](https://arxiv.org/abs/2605.03065)

### Inferences
- **Dimension match.** The project's per-plan initial noise is 16x2 = 32 dims, the same as DSRL-π0's shared-noise latent space, where DSRL-SAC worked in about 10k steps. Dimensionality should not be the obstacle.
- **DDIM requirement.** The official PushT checkpoint was trained as epsilon-prediction DDPM with 100 steps. The original Diffusion Policy paper shows the same family run with DDIM: 100 train / 10 inference steps, ~0.1 s on a 3080 in its real-world setup ([Diffusion Policy arXiv 2303.04137](https://arxiv.org/pdf/2303.04137)). A Crossway Diffusion sweep reports quality dropping below ~40 inference steps on some tasks ([Crossway Diffusion arXiv 2307.01849](https://arxiv.org/pdf/2307.01849); sampler type unverified). The first concrete step is to measure T / narrow-T success of the frozen checkpoint under DDIM (e.g. 10, 16, 50 and 100 steps) on the same 20 scenes. If DDIM-k matches the DDPM 11/20, DSRL applies directly and steps can be cut, which raises rollout throughput.
- **No state restoration needed.** DSRL is ordinary online RL with resets, so the "replay the full prefix to restore contact" limitation does not apply. The decision step is one plan (8 controls), giving about 38 latent decisions per 300-control episode.
- **Privileged critic, image actor.** Privileged simulator state (object pose, shape, goal) can feed Q^A/Q^W as an asymmetric critic, disclosed as privileged training signal. The noise actor can take the base's own frozen ResNet18 features plus pusher xy, so the deployed system stays image-based. The project's existing machine-generated interaction data can pre-train Q^A, since NA accepts offline actions.
- **Budget.** At ~900 controls/s, the DSRL budgets above (~3.5k–320k initial steps, reported gains within ~10k steps for π0) are minutes to hours of simulator time. The likely bottleneck is SAC update compute at UTD 5–20 per transition, not environment samples.

### Gaps
- Exact Robomimic success-vs-steps numbers for DSRL are only in figures; not extracted.
- Whether DSRL's Robomimic experiments used state or pixel DPPO checkpoints was not confirmed.
- No DSRL result on PushT (or any planar pushing task) was found.
- No published test of DSRL on a policy that never succeeds on the target task (0% base) was found.
- The fetched ablation sentence on denoising-step count was garbled ("virtually any impact"); the direction is unverified.
- DLSRL, OGPO and the social-navigation paper were checked only at abstract level; no numbers verified.

## Q2. Residual RL on frozen chunked diffusion policies (Policy Decorator, ResiP, ResFiT, EXPO): parameterization, exploration control, sample efficiency, failure modes

### Takeaway
A residual adds a bounded correction to the frozen base's action at every executed step. With a small, zero-initialized residual this is the most direct "start exactly at base, improve locally" method. Reported gains on precision tasks are large: ResiP 5–54% → 88–99% with PPO; ResFiT 14–23% → 64% on a real humanoid. Off-policy versions are about 200x more sample-efficient than PPO residuals. All three papers say the residual cannot invent strategies the base does not already have.

### Cited Findings
**Policy Decorator** (Yuan et al., arXiv 2412.13630, Dec 2024; UCSD)
- The executed action is base + residual. The residual is a small MLP/CNN Gaussian policy taking **observations only**; this worked better than also feeding the base action. The residual is bounded to (−α, α) via SAC tanh, with α set near the demonstration action scale. The critic takes the summed action. — [Policy Decorator arXiv](https://arxiv.org/html/2412.13630)
- Controlled exploration: with probability ε the behavior policy adds the residual, and ε rises linearly 0→1 over H steps. Examples: H = 1M (StackCube, BeT), 8M (PegInsertionSide, BeT), 30K (TurnFaucet, DP visual). Setting H=0 or α=1 removes controlled exploration and degrades toward vanilla residual RL, whose unconstrained residuals "can grow larger than base actions and disrupt the base behavior." — [Policy Decorator arXiv](https://arxiv.org/html/2412.13630)
- Base DP: U-Net, prediction horizon 16, observation horizon 2. The text says 8 executed actions but Table 6 says action horizon 4, which is internally inconsistent. SAC backbone, and the paper reports PPO also works. — [Policy Decorator arXiv](https://arxiv.org/html/2412.13630)
- Results: "near-perfect" success on all ManiSkill/Adroit tasks with state observations, and it "also performs well with high-dim visual observations" (two tasks per benchmark). On DPPO's Square/Transport checkpoints it reached 80%+ on Transport in ~4M steps vs ~16M for DPPO (~4x), with early termination on success added. Vanilla residual RL helped on only 3 of 6 DP tasks, DIPO got no success signal on any task, and JSRL beat base in about half the cases. — [Policy Decorator arXiv](https://arxiv.org/html/2412.13630)
- Wall-clock: StackCube (BeT, 5M steps) took 7h23m vs 33h52m for SAC fine-tuning. — [Policy Decorator arXiv](https://arxiv.org/html/2412.13630)
- Stated purpose: the residual is "designed to make minor corrections to the base policy." — [Policy Decorator arXiv](https://arxiv.org/html/2412.13630)

**ResiP — "From Imitation to Refinement"** (Ankile et al., arXiv 2407.16677, v4 Dec 2024; MIT)
- Base DP from 50 teleop demos, chunk 32, 8 executed at 10 Hz. The residual is a Gaussian MLP taking (s, a_base) and correcting **every timestep inside the chunk**, which the authors call key to reactivity. Action scale 0.1, orthogonal init with a small final-layer gain so corrections start near zero. PPO with 1024 parallel environments and a 500M-step budget, sparse 0/1 reward. — [ResiP arXiv](https://arxiv.org/html/2407.16677)
- Table I success (%), BC DP → ResiP, with full-RL chunked PPO-C and IDQL for comparison:

  | Task | DP | ResiP | PPO-C | IDQL |
  |---|---|---|---|---|
  | one_leg | 54 | 98 | 70 | 57 |
  | round_table | 12 | 94 | 38 | 18 |
  | lamp | 7 | 97 | 32 | 11 |
  | mug-rack | 26 | 88 | — | — |
  | peg-in-hole | 5 | 99 | — | — |
  | biman-insert | 33 | 93 | — | — |

  — [ResiP arXiv](https://arxiv.org/html/2407.16677)
- Failure mode: under medium randomization performance plateaus at 70–77% (one_leg 26→76, round_table 4→77, lamp 2→70). Placements far from nominal put the base out of distribution, and "local corrections cannot recover." The sim teacher used privileged part poses and was distilled into an RGB student. — [ResiP arXiv](https://arxiv.org/html/2407.16677)

**ResFiT — "Residual Off-Policy RL for Finetuning BC Policies"** (Ankile et al., arXiv 2509.19301, Sep 2025; ICLR 2026 listing)
- Frozen chunked BC base. A per-step residual π_θ(s_t, a_t^base) is learned with an off-policy TD3/DDPG-style learner: delayed actor, target smoothing, randomized ensembled double-Q, n-step=3, UTD > 1 (UTD 4 captures most of the benefit), LayerNorm critic, and 50/50 demo/online batch sampling. The critic sees the full summed action. Vision uses a shallow ViT with DrQ shift augmentation. — [ResFiT arXiv](https://arxiv.org/html/2509.19301v2); [ICLR 2026](https://iclr.cc/virtual/2026/10012529)
- Sample efficiency: BoxCleanup converges in ~200k steps vs ~40M for a PPO residual (~200x). Can reaches high success in ~75k steps vs ~150k for baselines. On Square only ResFiT and Tuned RLPD exceed 90% within 150k steps. — [ResFiT arXiv](https://arxiv.org/html/2509.19301v2)
- Real humanoid (29-DoF, ACT base): WoollyBallPnP 14% → 64% with 134 rollouts (~15 min); PackageHandover 23% → 64% with 343 episodes (~76 min). — [ResFiT arXiv](https://arxiv.org/html/2509.19301v2)
- Stated limitation: the method "cannot discover fundamentally different strategies or skills beyond what the base policy already encodes." Resets and success labels still came from humans. — [ResFiT arXiv](https://arxiv.org/html/2509.19301v2)

**EXPO** (Dong, Li, Sadigh, Finn; arXiv 2507.07986; ICLR 2026)
- Combines an expressive base policy, trained with a stable imitation objective, with a small Gaussian "edit" policy constrained to stay close to base actions. The "on-the-fly" policy picks the highest-Q action among base and edited samples, for both acting and TD backups. Reports "up to 2–3x" sample-efficiency gains over prior methods. Code: github.com/pd-perry/EXPO. Note: EXPO's base keeps training, so a frozen-base version is an adaptation, not the paper's setting. — [EXPO arXiv](https://arxiv.org/html/2507.07986v2); [ICLR 2026 poster](https://www.iclr.cc/virtual/2026/poster/10008723)

**Cross-paper conflict**
- Policy Decorator found residuals work better *without* the base action as input. ResiP and ResFiT condition on the base action. This is unresolved across papers. — [Policy Decorator](https://arxiv.org/html/2412.13630); [ResiP](https://arxiv.org/html/2407.16677); [ResFiT](https://arxiv.org/html/2509.19301v2)

### Inferences
- **PushT parameterization.** Apply the residual to each of the 8 executed controls. Input: the base's frozen image features + pusher xy + the base action for that step (ResiP/ResFiT style). Use a tanh-bounded output in normalized action units with zero-init last layer, plus Policy Decorator's ε-ramp. With α=0 or ε=0 the system is exactly the base. PushT actions are absolute pusher targets on a 512-px workspace; the exact normalizer in the official checkpoint needs checking before choosing α.
- **Sample cost.** With ResFiT-style off-policy training (UTD ~4, n-step 3), the reported 75k–200k-step regimes are under an hour of the project's simulator throughput. PPO residuals (ResiP: 500M budget) are probably too expensive for 60 GPU-hours across several shapes.
- **Fit to the failing shapes.** Per-step residuals fix small misalignments. Y/D/L failures at 0–1/20 likely need different contact strategies, such as pushing on different faces, so the base macro behavior may be wrong rather than imprecise. The ResiP plateau and the ResFiT statement both predict limited gains in that case.

### Gaps
- Policy Decorator's per-task success numbers and visual-observation results are in figures/appendix; not extracted.
- None of these papers test PushT or a target-object change (new shape, same robot); all adapt to the same task with a new robot or task precision.
- No source quantified how large α can grow before the "residual overrides base" failure appears on a contact-rich planar task.

## Q3. Best-of-N / value-guided sampling and reranking (V-GPS, GPC world-model ranking, IDQL-style, FM-Steer, VERITAS): reported gains from selection only, and inference cost

### Takeaway
Selecting among base samples gives moderate, verifier-dependent gains. Exploiting verifier error at large N is a documented failure, and selection cannot rescue a policy whose samples never contain a good action. The only PushT numbers found come from GPC. On vision PushT: DP 0.642 vs V-GPS 0.620 (worse), GPC-RANK with a learned world model and K=100 0.739, and RANK+OPT 0.882. On state PushT, ranking with the true simulator reaches 0.952 vs BC 0.812. All of these are IoU/score metrics, not success at ≥0.95 coverage.

### Cited Findings
**V-GPS** (Nakamoto, Mees, Kumar, Levine; CoRL 2024, PMLR v270)
- Samples K actions from a black-box generalist and scores them with a Cal-QL (or IQL) Q-function: ResNet-34 + FiLM language, 25.6M params, reward −1 except 0 on the last 3 steps, γ=0.98. Executes the argmax (real robot: K=50, greedy), or a softmax sample in simulation. — [V-GPS arXiv](https://arxiv.org/html/2410.13816v1); [PMLR](https://proceedings.mlr.press/v270/nakamoto25a.html)
- Real WidowX (Octo-small-1.5, 20 trials/task, 6 tasks): average 0.24 → 0.44 (+82.8% relative). Per task, e.g. mushroom on cloth 0.35 → 0.70, sweet potato 0.30 → 0.35. — [V-GPS arXiv](https://arxiv.org/html/2410.13816v1)
- SIMPLER averages, base → V-GPS: Octo-small 0.27 → 0.34, Octo-base 0.17 → 0.18, Octo-small-1.5 0.02 → 0.18, RT1-X 0.12 → 0.14, OpenVLA 0.24 → 0.27. Some tasks got worse (Octo-base spoon-on-towel 0.25 → 0.21; RT1-X put-near 0.44 → 0.42). — [V-GPS arXiv](https://arxiv.org/html/2410.13816v1)
- Over-optimization with K: with IQL, eggplant 0.59 at K=10 falls to 0.35 at K=100, which the paper attributes to exploiting value-function errors. Cal-QL is 0.84 at K=50 but 0.63 at K=100. Controls: random selection among base samples gives 0.49 (no gain); a random policy with the value function gives 0.00. — [V-GPS arXiv](https://arxiv.org/html/2410.13816v1)
- Inference cost (Octo-small, batched): base 0.0752 s/step; K=10 1.28x, K=50 1.59x, K=100 2.12x. — [V-GPS arXiv](https://arxiv.org/html/2410.13816v1)

**GPC — "Strengthening Generative Robot Policies through Predictive World Modeling" / "Inference-Time Enhancement …"** (Qi, Yin, Du, Yang; Harvard; arXiv 2502.00622)
- **State PushT** (IoU, 100 seeds, Table I): BC 0.812. GPC-RANK K=100 0.898, K=1000 0.932, K=5000 0.934. GPC-OPT (K=1, M=30) 0.886. RANK+OPT K=20/M=30 0.914. Ground-truth simulator ranking 0.952 (column mapping inferred from a flattened table). — [GPC arXiv v4](https://arxiv.org/html/2502.00622v4); [GPC v1](https://arxiv.org/html/2502.00622v1)
- **Vision PushT** (ResNet18+UNet DP; Table II score, 50 seeds): Diffusion Policy 0.642, **V-GPS 0.620**, LaDi-WM 0.683, DreamerV3 0.649, GPC-RANK (learned reward, K=100) 0.739. Table III IoU ablation: BC 0.642; RANK K=50 0.698; RANK K=100 0.739; OPT (K=1, M=25) 0.791; RANK+OPT (K=10, M=25) 0.882. — [GPC arXiv v4](https://arxiv.org/html/2502.00622v4)
- Cost per decision cycle (vision PushT): BC 0.457 s; RANK K=50 5.835 s; RANK K=100 11.745 s; OPT M=25 39.061 s; RANK+OPT 374.102 s. World-model rollouts take ~90–95% of runtime. Planners without the diffusion prior (MPPI, CEM, gradient ascent) reach success below 0.2. GPC-OPT fails without a frozen world-model noise seed. — [GPC arXiv v4](https://arxiv.org/html/2502.00622v4)
- World-model data for vision PushT: 500 expert demos, 6x5000 perturbations, and 7 long-horizon demos. Exploration data improved results ~10%. No new-shape results. — [GPC arXiv v4](https://arxiv.org/html/2502.00622v4)
- Venue: a search summary says GPC was accepted to IEEE RA-L; the v4 page I read did not name the journal. — [GPC project page](https://computationalrobotics.seas.harvard.edu/GPC/)

**Other selection evidence**
- In DynaGuide's CALVIN MovableObjects setting, "sampling-based GPC falls back to base-policy levels." — [DynaGuide arXiv](https://arxiv.org/html/2506.13922v2)
- In DSRL's π0-Libero comparison, V-GPS "did not significantly improve π0" while DSRL went ~20% → ~100%. — [DSRL arXiv](https://arxiv.org/html/2506.15799)
- IDQL (resampling a diffusion actor with an IQL critic), used as an online baseline in ResiP, gave only small gains: one_leg 54 → 57, round_table 12 → 18, lamp 7 → 11. — [ResiP arXiv](https://arxiv.org/html/2407.16677)
- **FM-Steer** (CVPR 2026): samples several noisy action proposals from a flow VLA, keeps the one with the highest value from a flow-based intermediate verifier, and runs a light denoiser asynchronously to keep the control rate. The poster page has no numbers. — [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/40101)
- **VERITAS** (Zhang & Shah, arXiv 2606.18247, Jun 2026): a gradient-free visual verifier for inference-time steering. Fine-tuning on verified self-generated rollouts is claimed to be about as efficient as expert demos. The abstract gives no numbers. — [VERITAS abstract](https://arxiv.org/abs/2606.18247)
- Q-chunking (Li, Zhou, Levine; NeurIPS 2025) runs TD RL in chunked action space. A 2026 follow-up describes chunk TD backups that take the max over N behavior-policy samples (EMAQ-style best-of-N bootstrap). — [Q-chunking](https://arxiv.org/html/2507.07969v1); [Adaptive Q-chunking arXiv 2605.05544](https://arxiv.org/pdf/2605.05544)

### Inferences
- **Inference cost here.** Per the project brief, batched sampling of 32–64 plans costs about the same as 1, so best-of-N on the policy side is nearly free. The cost is in the scorer. A learned Q (V-GPS style) is cheap. A world-model scorer is expensive (GPC ~0.1 s per sample on vision PushT) and the project's world model is not validated for contact. A true-simulator scorer is privileged, and because contact can only be restored by full-prefix replay, each candidate costs (prefix + lookahead) controls.
  - Rough estimate: N=32, 32-control lookahead, 38 decisions per episode ≈ 32 × Σ(t+32) ≈ 2.2e5 controls per episode, about 4 min at 900 controls/s, or about 80 min per shape for 20 scenes.
  - That is affordable as a diagnostic, but it is a privileged oracle, not a deployable actor.
- **Reachability test (most useful single diagnostic).** Run simulator-oracle best-of-N with the frozen base at N = 1, 8, 32, 64 on Y/D/L. This bounds what any frozen-base selection or noise-steering method can reach: GPC's state PushT ground-truth ranking reached 0.952 IoU vs 0.812 for BC. If oracle best-of-64 still gives 0/20 on L, initial-noise steering and best-of-N cannot solve L without larger corrections.
- **Learned-Q reranking risk.** A learned Q over 32–64 samples is the cheapest deployable variant. The V-GPS K-ablation and the GPC PushT result (V-GPS below base) show it can hurt, so it must be gated by held-out evaluation.

### Gaps
- No source reported best-of-N *success rate* (coverage ≥ 0.95) on PushT; only IoU/score metrics.
- No source tested best-of-N on a changed object shape.
- FM-Steer and VERITAS numbers not verified (pages/abstracts only).
- Whether the original Q-chunking paper itself uses best-of-N backups was not verified.

## Q4. Classifier / value gradient guidance during denoising as a frozen-base alternative

### Takeaway
Gradient guidance adds ∇(critic or objective) to the noise prediction at each denoising step. Scale 0 recovers the base exactly. In robot studies it is fragile: it shifts samples off the learned distribution (more constraint violations, incoherent trajectories at high scale) and yields modest gains on manipulation tasks (PPGuide ~0–10 points). Self-labelled variants need some base successes to start from, which rules them out for L/D at 0/20.

### Cited Findings
**ITPS** (Wang, Shah et al., arXiv 2411.16627 v2, Mar 2025; MIT/NVIDIA)
- Guided diffusion (GD) adds β_i∇ξ to ε_θ at each step: τ_{i−1} = α_i(τ_i − γ_i(ε_θ + β_i∇ξ)) + σ_iη. The authors say this approximates sampling from a *weighted sum* rather than the product of the policy and objective distributions, and "can result in out-of-distribution samples." Annealed-MCMC stochastic sampling (SS) found the closest in-distribution mode instead. — [ITPS arXiv](https://arxiv.org/html/2411.16627v2)
- Real kitchen (DP), random sampling vs GD vs SS:
  - Alignment: 38% / 37% / 71%
  - Constraint satisfaction: 90% / 82% / 73%
  - Aligned success: 34% / 32% / 55%

  GD gave no gain. Block stacking: GD at a large guide ratio reached 86% alignment but only 15% constraint satisfaction. — [ITPS arXiv](https://arxiv.org/html/2411.16627v2)

**DynaGuide** (Du & Song, NeurIPS 2025)
- The gradient of a separate DINOv2-latent dynamics model's objective guides DDIM denoising of an off-the-shelf DP. Stochastic sampling repeats each step M times so the guidance scale can be raised. — [DynaGuide arXiv](https://arxiv.org/html/2506.13922v2); [NeurIPS 2025](https://nips.cc/virtual/2025/poster/117515)
- Results: CALVIN ArticulatedParts ~70% average steering success (8.7x boost of the target behavior). 5.4x better than goal-conditioning on low-quality objectives. With only 1% of the data showing a behavior (Switch-On), steering reached ~40% success, consistently beating sampling. — [DynaGuide arXiv](https://arxiv.org/html/2506.13922v2)
- High guidance scale can make trajectories "less smooth or incoherent" because the gradient pushes noise predictions out of distribution. The task is to choose among base behaviors (which object to interact with), not to change the dynamics strategy. — [DynaGuide arXiv](https://arxiv.org/html/2506.13922v2)

**PPGuide** (Wang, Jha, Qureshi, Romeres; arXiv 2603.10980; ICRA 2026 per MERL)
- Labels chunks from the policy's own rollouts as success- or failure-relevant using attention-based multiple-instance learning on binary episode outcomes. It then trains a 3-class classifier and adds w_sr∇log P(SR) − w_fr∇log P(FR) to the noise estimate on alternating denoising steps. — [PPGuide arXiv](https://arxiv.org/html/2603.10980); [MERL TR2026-066](https://www.merl.com/publications/TR2026-066)
- Success (epoch 500): Square 62 → 72%, Transport 60 → 68%, Coffee D2 54 → 58%, Stack Three 28 → 34%, Kitchen 52 → 52%. Some cells fall below DP (Stack Three at epoch 550: 30 → 28%). — [PPGuide arXiv](https://arxiv.org/html/2603.10980)
- Stated limitations: cold start ("a policy that rarely succeeds gives poor self-labels"), spurious correlations, sensitivity to the z-score threshold and guidance strength, and too much guidance degrading sample quality. — [PPGuide arXiv](https://arxiv.org/html/2603.10980)

**GPC's gradient variant**
- GPC-OPT optimizes action proposals by gradient through a world model. On vision PushT it raised IoU 0.642 → 0.791, but at 39 s per decision, and it needed a frozen world-model noise seed to avoid unstable gradients. — [GPC arXiv v4](https://arxiv.org/html/2502.00622v4)

### Inferences
- **Cost.** For the official DP, guidance needs a critic differentiable in the noisy action, or in the predicted clean action, at up to 100 DDPM steps per plan. That is affordable, about 100 backward passes through a small critic. But the evidence (ITPS GD no gain; DynaGuide incoherence at high scale; PPGuide small gains) says guidance mainly reweights among base modes, at the cost of off-distribution artifacts.
- **Compared with DSRL.** DSRL optimizes the start of a deterministic sampler and always stays on the sampler's output manifold. Guidance perturbs every step and can leave it. That makes DSRL the more conservative frozen-base option for contact-rich pushing.

### Gaps
- No guidance paper evaluated PushT or object-shape transfer.
- DynaGuide's quantitative guidance-strength trade-off (its Appendix A.4) was not read.
- PPGuide's rollout counts and the DP variant (CNN vs transformer, step count) were not stated in the fetched text.

## Q5. Can each method represent behaviors the base policy rarely or never samples (e.g. L at 0/20), compared with fine-tuning?

### Takeaway
None of these methods is designed to create behaviors outside the base's support. They differ in how far into the tails they can reach:

- **Best-of-N:** only what appears in N samples.
- **Gradient guidance:** reweights modes, with off-distribution risk.
- **DSRL:** any action the deterministic sampler produces from bounded noise; b up to 2.5–3 reaches low-density tails, but not regions the model maps no noise to.
- **Bounded residual:** a ±α neighbourhood of base actions at every step.

Full fine-tuning can in principle move the support, which matches the project's observed failure: reach comes with drift.

### Cited Findings
- DSRL: exploration is limited by the base, not all diffusion policies are steerable, and a concentrated distribution may not provide enough options. — [DSRL arXiv](https://arxiv.org/html/2506.15799)
- ResFiT "cannot discover fundamentally different strategies or skills beyond what the base policy already encodes." — [ResFiT arXiv](https://arxiv.org/html/2509.19301v2)
- ResiP: when objects are far from nominal, the base is out of distribution and "local corrections cannot recover"; performance plateaus at 70–77%. — [ResiP arXiv](https://arxiv.org/html/2407.16677)
- Policy Decorator's residual is "designed to make minor corrections." — [Policy Decorator arXiv](https://arxiv.org/html/2412.13630)
- V-GPS: a random policy with the learned value function scores 0.00. Selection cannot create good proposals. — [V-GPS arXiv](https://arxiv.org/html/2410.13816v1)
- DynaGuide amplified a behavior present in only 1% of training data to ~40% success, so guidance can surface rare but present modes. — [DynaGuide arXiv](https://arxiv.org/html/2506.13922v2)
- PPGuide cold start: rare successes make self-labels poor. — [PPGuide arXiv](https://arxiv.org/html/2603.10980)
- DLSRL (2026) argues noise-only steering is limited because it cannot modulate intermediate representations, and adds frozen-base feature adapters to widen reach. No numbers verified. — [DLSRL abstract](https://arxiv.org/abs/2609.11270)
- Full fine-tuning counterpoint: OGPO claims fine-tuning poorly initialized BC policies to near-full success without expert data, outperforming steering and residual methods (abstract claim). ResiP's chunked full-RL PPO-C baseline improved base but trailed the residual (one_leg 54 → 70 vs 98). — [OGPO abstract](https://arxiv.org/abs/2605.03065); [ResiP arXiv](https://arxiv.org/html/2407.16677)

### Inferences
- **Y (1/20)** is the best frozen-base candidate: the base demonstrably produces some successful sequences. **Narrow-T** is already at T level. **D and L (0/20)** are the real test of whether success exists in the base's tails.
- A 0/20 rate under normal sampling does not prove absence. With 38 plan decisions per episode, a bounded noise actor can pick a rarely sampled plan at every step, and the compound probability of such a sequence under random sampling can be tiny even when every single plan is in support. So the oracle best-of-N reachability test from Q3 (per-plan selection with simulator lookahead) is the right check, not raw pass@k.
- **If oracle selection reaches L:** DSRL is likely to learn an image-based noise policy that does the same, without touching T.
- **If oracle selection cannot reach L:** possible routes are a larger bounded residual on top of DSRL, frozen-base adapters (DLSRL-style), or controlled fine-tuning with a base fallback gate. Any of these must be disclosed as needed for a 0% base.

### Gaps
- No paper reports frozen-base steering on a task where the base has 0% success and the required contact strategy differs. The closest are DSRL's Franka (2/10) and π0 Libero (~20%).
- No quantitative study links DSRL's b_W to how far into the tails the steered policy can reach.

## Q6. Exact fallback to the base policy, "never worse than base", and fit to this project

### Takeaway
All frozen-base methods have a parameter setting that recovers the base exactly, and keep its weights untouched:

- **DSRL:** sample w ~ N(0, I) instead of the learned noise.
- **Residual:** α = 0 or ε = 0.
- **Best-of-N:** N = 1.
- **Guidance:** scale 0.

None gives a formal never-worse guarantee. Never-worse has to come from a held-out acceptance gate that routes back to the base. For this project, DSRL-NA plus a bounded per-step residual, both behind a per-shape gate, fits best. The cheap reachability diagnostic should come first.

### Cited Findings
- DSRL replaces only the noise source of the frozen policy, which normally draws w ~ N(0, I). Weights are unchanged. — [DSRL arXiv](https://arxiv.org/html/2506.15799)
- The DSRL social-navigation follow-up adopts DSRL specifically to preserve base performance, but gives no formal guarantee. — [arXiv 2609.24317](https://arxiv.org/abs/2609.24317)
- Residual: ε-ramp from 0 and bounded α (Policy Decorator); residual init near zero (ResiP). — [Policy Decorator](https://arxiv.org/html/2412.13630); [ResiP](https://arxiv.org/html/2407.16677)
- Selection and guidance can fall below base without gating: V-GPS lost on some SIMPLER tasks; V-GPS on vision PushT scored 0.620 vs DP 0.642; PPGuide fell below DP in some cells. — [V-GPS](https://arxiv.org/html/2410.13816v1); [GPC v4](https://arxiv.org/html/2502.00622v4); [PPGuide](https://arxiv.org/html/2603.10980)

### Inferences
Synthesis across Q1–Q4:

| Method | What is learned | Base weights | Exact fallback | Reach beyond common samples | Reported efficiency | Deploy cost (this project) | PushT evidence | Fit |
|---|---|---|---|---|---|---|---|---|
| DSRL-NA / DSRL-SAC | Noise actor π^W(obs) in a 32-d box; Q^A(s,a), Q^W(s,w) | Frozen | Yes: N(0, I) noise (under the same DDIM sampler) | Tails of the sampler's output set; not outside it | 3.5k–11k steps real/π0; claimed 5–10x vs DPPO/IDQL | 1 sampler pass + tiny MLP | None found | **High**, if DDIM-k keeps base T at 11/20 |
| Bounded per-step residual (Policy Decorator / ResiP / ResFiT) | Δa per executed control, \|Δa\| ≤ α | Frozen | Yes: α = 0 / ε = 0 | ±α around base actions | Off-policy ~75k–200k steps; PPO ~10^7–10^8 | Negligible | None found | High for precision (Y, narrow-T); low for new strategies (L) |
| Best-of-N with learned Q / V | Q or V only | Frozen | Yes: N = 1 | Only the N samples | Zero RL steps for the actor; Q needs data | Near-free sampling (batched) + Q | V-GPS 0.620 vs DP 0.642 (vision, GPC table) | Medium; must be gated |
| Best-of-N with world model (GPC-RANK) | World model + reward | Frozen | Yes | Only the N samples | Offline WM data | ~0.1 s/sample; 11.7 s at K=100 | 0.642 → 0.739 (K=100); RANK+OPT 0.882 | Low: project WM not contact-reliable |
| Best-of-N with simulator oracle | Nothing | Frozen | Yes | Only the N samples | n/a | Prefix replay ~4 min/episode (estimate) | State: 0.952 vs 0.812 IoU | Diagnostic only (privileged) |
| Gradient guidance (ITPS-GD / DynaGuide / PPGuide) | Critic or classifier | Frozen | Yes: scale 0 | Reweights modes; off-distribution risk | PPGuide +0–10 pts | ~100 critic backward passes per plan | None | Low–medium; cold start for L/D |
| Full / LoRA fine-tuning (current; OGPO, DPPO) | Denoiser weights | Changed | Only by keeping a base copy and gating | Can move support | OGPO claims strong; DPPO ~16M steps (Transport) | Base cost | Project: T drifts 0.55 → 0.05 | Use only behind a gate |

Recommended plan for this project (inference):

1. **Check DDIM.** Confirm the frozen official checkpoint under DDIM-k reproduces T 11/20 and narrow-T 11/20 on the fixed 20 scenes (k ∈ {10, 16, 50, 100}). Keep DDPM-100 as the reference fallback if DDIM loses success.
2. **Reachability diagnostic.** Run simulator-oracle best-of-N (N ∈ {8, 32, 64}) per plan on Y/D/L, using prefix replay. This bounds what DSRL and best-of-N can reach, and tells whether L needs more than steering.
3. **Train DSRL-NA.** One shared image-conditioned noise actor (frozen base ResNet18 features + pusher xy) across all shapes including T. Critics may use privileged state and a dense coverage reward (disclosed). Bound b_W ≈ 1–2.5, as in DSRL. Use separate training/validation/test scenes.
4. **Add a residual if needed.** If step 2 shows the base's tails are insufficient for some shapes, add a ResFiT-style bounded residual (zero-init, ε-ramp) on top.
5. **Gate per shape.** Deploy the improved controller for a shape only if it beats the base on held-out scenes by a paired comparison with the same seeds and full simulator state restore; otherwise use base sampling. Shape identity must come from the image at deployment.

This gives never-worse-than-base by construction on the evaluated scene distribution, but not a statistical guarantee on 20 scenes. Report confidence intervals.

### Gaps
- No primary source offers a formal "never worse than base" certificate for any of these methods.
- No source reports multi-shape or object-shape transfer with frozen-base steering. The project would be first-of-kind evidence; this is not a literature finding.
- Exact action normalization of the official PushT checkpoint (needed to pick α) and its DDIM success were not verified here.
