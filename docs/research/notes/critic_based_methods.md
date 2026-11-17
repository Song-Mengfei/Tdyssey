# Critic-based learning signals for RL fine-tuning of pretrained diffusion/flow robot policies

Scope: learned value functions (V or Q critics) used to give a reliable learning signal when fine-tuning a pretrained visual Diffusion Policy with simulator rewards, judged against Tdyssey (official RSS 2023 PushT image DP, ~251M ConditionalUnet1D + frozen ResNet18-GN encoder, 2x96x96 RGB (84 crop) + pusher xy, 16-step chunks / 8 executed, DDPM-100 epsilon prediction; adaptation T -> narrow-T/Y/D/L from machine interaction; current forward-process DiffusionNFT/AdvantageFlow trainer with 8 candidates scored by a 32-control Monte Carlo (MC) return whose candidate-specific share of variance is only 15-35%, split-half r 0.14-0.29, and drift-proportional success loss). Research date: 2026-10-25. Primary sources were arXiv abstract/HTML pages and the official DPPO GitHub code/configs; numbers below come from those pages as read by a fetch tool. Anything read only from a search-result summary is labelled as such.

## Q1. DPPO (Ren et al., arXiv 2409.00588): critic definition, inputs, GAE, reported sample counts/success, pixel actors with state critics

### Takeaway
DPPO is on-policy PPO over a two-level MDP (environment step outside, denoising chain inside). Its critic is an environment-level value V(s_t), evaluated at each chunk decision. It does not take the denoising step or the noisy action as input. GAE (gamma_env 0.999, lambda 0.95 on robomimic/furniture) runs over chunk-level environment steps. The single environment-level advantage is then copied to every fine-tuned denoising step and discounted by gamma_denoise. On pixel robomimic the official critic is a ViT on the same image plus 9-D proprio that the actor sees. It is not a privileged-state (asymmetric) critic, so DPPO gives no direct evidence for a state critic paired with a pixel actor. DPPO also fine-tunes only the low-noise tail of the chain, and clips and discounts the high-noise steps more tightly. That speaks directly to Tdyssey's observation that drift concentrates at high-noise steps.

### Cited Findings
- Metadata: "Diffusion Policy Policy Optimization", Ren, Lidard, Ankile, Simeonov, Agrawal, Majumdar, Burchfiel, Dai, Simchowitz. arXiv v1 1 Sep 2024, v2 7 Nov 2024, v3 9 Dec 2024. Project/code: diffusion-ppo.github.io and github.com/irom-lab/dppo. The arXiv page does not list a venue (ICLR 2025 is commonly cited but was not verified here) — [arXiv abs](https://arxiv.org/abs/2409.00588)
- Two-layer MDP: denoising state s̄ = (s_t, a_t^{k+1}), denoising action ā = a_t^k. Reward is 0 for k > 0 and R(s_t, a_t^0) at k = 0. Each denoising step is a Gaussian N(a^k; μ(a^{k+1}, ε_θ(a^{k+1}, k+1, s_t)), σ²_{k+1} I), so the whole chain has a tractable likelihood — [DPPO HTML v3](https://arxiv.org/html/2409.00588v3)
- Critic: "V̂(s̄_{t̄(t,0)}) := Ṽ(s_t)", i.e. the value "only depends on the environment state". The authors report this trains "more efficient and stable" than also estimating the value of the denoised/noisy action — [DPPO HTML v3](https://arxiv.org/html/2409.00588v3)
- Advantage: the main text gives Â = γ_denoise^k (r̄ − V̂). A footnote says GAE is used in practice. Hyperparameters: γ_env 0.99 (Gym) and 0.999 (Robomimic, Furniture-Bench); GAE λ = 0.95 throughout; γ_denoise 0.99 (Robomimic state) and 0.9 (ViT pixel, Furniture-Bench); PPO clip ε 0.01 (Gym, Robomimic state, pixel) and 0.001 (Furniture-Bench). Critic MLP [256,256,256], critic LR 1e-3; actor LR 1e-4 for DPPO and 1e-5 for baselines (down to 1e-5 → 1e-6 for Transport) — [DPPO HTML v3, appendix](https://arxiv.org/html/2409.00588v3)
- Chunk handling in code: the env wrapper executes the whole action chunk per `step()` and aggregates rewards with `reward_agg_method="sum",  # never use other types`. It stops at termination/truncation inside the chunk, still counting the terminal sub-step's reward. GAE therefore runs over chunk decisions with summed in-chunk rewards — [DPPO multi_step.py](https://raw.githubusercontent.com/irom-lab/dppo/main/env/gym_utils/wrapper/multi_step.py)
- Denoising-step handling in code: the clip coefficient goes from `clip_ploss_coef_base` (1e-3) at denoising index 0 to `clip_ploss_coef` (1e-2) at the last index, on an exponential schedule (rate 3). Advantages are multiplied by `gamma_denoising ** (ft_denoising_steps - i - 1)`. Earlier (higher-noise) fine-tuned steps therefore get both tighter clipping and a smaller advantage weight, and the final step gets weight 1. Advantages are normalized by default (`norm_adv`). The value loss is unclipped by default. An optional BC term (`use_bc_loss`, default False) uses log-likelihood of base-policy actions — [DPPO diffusion_ppo.py](https://raw.githubusercontent.com/irom-lab/dppo/main/model/diffusion/diffusion_ppo.py)
- Partial-chain fine-tuning: robomimic state "pre-trains with 20 denoising steps and then fine-tunes the last 10"; pixel "pre-trains with 100 denoising steps and then fine-tunes 5 DDIM steps". DDIM can be "as few as 5 steps", with η = 1 for training and η = 0 for evaluation. The exploration std floor is ~0.01–0.1, and σ_min = 0.1 when evaluating likelihoods — [DPPO HTML v3](https://arxiv.org/html/2409.00588v3)
- Official pixel config (robomimic Can, `ft_ppo_diffusion_mlp_img.yaml`): critic `model.common.critic.ViTCritic` on RGB [3,96,96] (1 image frame) plus state `cond_dim` = 9 (robot proprio). ViT patch 8, depth 1, embed 128; MLP [256,256,256]; critic augmentation off; `n_critic_warmup_itr: 2`. Actor `VisionDiffusionMLP`, LR 5e-5. 151 iterations, 50 envs, n_steps 300, γ 0.999, λ 0.95, 10 update epochs, denoising_steps 100, ft_denoising_steps 5 (DDIM), horizon_steps/act_steps 4/4, gamma_denoising 0.99, min sampling/logprob std 0.1, running reward scaling, max_episode_steps 300 — [DPPO config](https://raw.githubusercontent.com/irom-lab/dppo/main/cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp_img.yaml)
- Reported results (text only; the per-task curves are in figures not captured here). Robomimic state: "faster convergence to ∼100% success rate in Lift and Can"; Transport ">90%". Pixel: DPPO "is the first RL algorithm to solve Transport from either state or pixel input to high (>50%) success rates", while Gaussian-ViT "does not improve from its 0% pre-trained success rate". Baselines are capped at 1e6 samples on Robomimic. Pre-training used 300 state demos / 100 pixel demos; Furniture-Bench used 50 demos and T_a = 8. Seeds: 3 for Robomimic/Kitchen/Furniture, 5 for Gym — [DPPO HTML v3](https://arxiv.org/html/2409.00588v3)
- Per-iteration sample counts in the appendix tables: Robomimic 60k/80k/160k steps per iteration (Lift-Can/Square/Transport); Furniture-Bench 700k/1M/1M (One-leg/Lamp/Round-table, 1000 IsaacGym envs). Wall-clock per iteration: DPPO-MLP state 35.6/43.3/65.0/350.5 s and DPPO-UNet state 83.6/92.7/130.4/431.1 s (Lift/Can/Square/Transport); DPPO-ViT-MLP pixel 194.9/202.5/328.5/871.3 s; DPPO-UNet Furniture 148.4/258.2/188.6 s. From-scratch DPPO costs "about 6×" the wall-clock of Gaussian PPO — [DPPO HTML v3, appendix](https://arxiv.org/html/2409.00588v3)
- Furniture-Bench: DPPO "improves policy performance in all six settings". Gaussian-MLP "collapses to zero success rate" on all three tasks at Med randomness. Sim-to-real One-leg: DPPO 16/20 (80%) on hardware; Gaussian 88% in sim but 0% on hardware; BC-regularized Gaussian 53% sim / 50% real, where BC regularization "significantly limits fine-tuning" — [DPPO HTML v3](https://arxiv.org/html/2409.00588v3)
- Off-policy diffusion baselines as DPPO implemented them: DIPO, DQL and QSM use double-Q with 1M/1M/250k replay buffers; IDQL uses Q and V with an expectile loss (250k buffer); DAWR uses a TD(λ = 0.95) critic (120k buffer); DRWR has no critic. Q-learning baselines show "training collapse" on sparse-reward manipulation, "dropping to zero success after good pre-training". IDQL/DRWR are strong on Lift/Can/Square but weaker on Transport — [DPPO HTML v3](https://arxiv.org/html/2409.00588v3)
- No PushT experiment appears in the DPPO text read here (benchmarks: Gym, Robomimic, Franka-Kitchen, Furniture-Bench, D3IL-style avoid tasks were not checked) — [DPPO HTML v3](https://arxiv.org/html/2409.00588v3)

### Inferences
- DPPO's "last K' steps" means the low-noise end of the chain (the step producing a^0 has discount 1 and the loosest clip). By construction, the high-noise steps where Tdyssey's drift concentrates are frozen (state) or heavily damped (γ_denoise 0.9 and clip 1e-3 on pixel). That is a direct, cheap mitigation for the current trainer even without adopting PPO.
- DPPO's critic is at chunk-decision granularity: one V(s_t) per executed chunk, summed in-chunk reward, γ applied per chunk. For Tdyssey that means a value every 8 controls, reward = sum of the 8 per-control rewards, bootstrap γ_chunk·V(s_{t+8}). With 300 controls there are ~37.5 decisions per episode, comparable to robomimic's 300 controls / 4 = 75.
- The pixel Can config implies roughly 151 iterations × 50 envs × 300 chunk-steps × 4 controls ≈ 9.1M simulated controls per run. That is ~2.8 h of Tdyssey simulator time at 900 controls/s. Simulator throughput is therefore not the binding constraint. Forward/backward passes of the 251M UNet over 5–10 fine-tuned DDIM steps for ~1.1M chunk decisions are. DPPO-UNet is already ~2.4× slower per iteration than DPPO-MLP on state robomimic. (Arithmetic from the config, not a number reported by DPPO.)
- The official pixel critic sees only what the actor sees (image + proprio). Swapping in a privileged simulator-state critic for PushT needs only a separate observation key for the critic, because V is used only at training time for GAE. This is a small code change, but unvalidated in DPPO itself.
- Tdyssey's DP uses DDPM-100 epsilon prediction. DPPO's pixel recipe (pretrain 100 steps, fine-tune a 5-step DDIM chain with η = 1 and σ_min 0.1) is the closest published match. It needs a likelihood-defining stochastic sampler, which the current forward-process trainer does not use.

### Gaps
- Exact per-task success numbers and sample-to-threshold counts for DPPO (state and pixel robomimic, Furniture-Bench) are only in figures. They were not extractable from the HTML text.
- I did not verify whether the DPPO repo already has a switch for a state-only critic with an image actor.
- Venue (ICLR 2025) not verified from a primary page.
- No DPPO PushT result found.

## Q2. Q-function-based diffusion policy improvement (Diffusion-QL, IDQL, QSM, Q/advantage-weighted regression, PA-RL, RFPO, QAM, FQL/QC-FQL, DSRL): requirements and on/off-policy

### Takeaway
All of these learn Q(s,a) off-policy (replay-based TD) and extract a policy in one of four ways:
- Gradient of Q through the sampler (Diffusion-QL).
- Score/adjoint matching to ∇_a Q, which avoids backprop through the chain (QSM, QAM).
- Reweighting or resampling base-policy samples (IDQL, QVPO, AWR-style).
- Building improved "target" actions from Q and regressing the policy onto them with the ordinary denoising/flow loss (PA-RL, RFPO).

For pretrained-robot-policy fine-tuning, the methods with real manipulation evidence keep a hard anchor to the base policy: PA-RL (best-of-N + local Q ascent + BC distillation), DSRL (frozen DP, RL on input noise) and QAM (KL to the base policy, π* ∝ π_β exp(τQ)). DPPO's authors found the unanchored off-policy diffusion baselines (DQL, QSM, DIPO) collapse on sparse-reward manipulation.

### Cited Findings
- Diffusion-QL (Wang, Hunt, Zhou; ICLR 2023; arXiv Aug 2022): offline RL. A learned action-value term is added to the diffusion BC loss, so the policy "seeks high-value actions while staying near the behavior policy". State of the art on most D4RL tasks — [arXiv abs](https://arxiv.org/abs/2208.06193). As DPPO implemented it, DQL uses double-Q with a 1M-sample buffer — [DPPO HTML](https://arxiv.org/html/2409.00588v3). (The abstract page does not itself confirm the full-chain backprop detail.)
- IDQL (Hansen-Estruch, Kostrikov, Janner, Kuba, Levine; arXiv Apr 2023): reinterprets IQL as actor-critic. A diffusion behavior policy generates samples that are weighted by the critic and importance-resampled; the Q-function is trained "using only dataset actions" (offline/implicit). Code: github.com/philippe-eecs/IDQL — [arXiv abs](https://arxiv.org/abs/2304.10573). Used as a fine-tuning baseline, IDQL on Furniture one_leg reached 57%, because "even with maximum denoising stochasticity, the base produces actions with too little variance for effective exploration" — [ResiP HTML](https://arxiv.org/html/2407.16677v4)
- QSM (Psenka, Escontrela, Abbeel, Ma; ICML 2024; arXiv Dec 2023): matches the diffusion policy's score to the action gradient of Q. It is off-policy and "only needs to differentiate through the denoising model", not the whole sampling chain — [arXiv abs](https://arxiv.org/abs/2312.11752). DPPO's QSM baseline used double-Q with a 250k buffer and collapsed with the other Q-learning baselines on sparse manipulation — [DPPO HTML](https://arxiv.org/html/2409.00588v3)
- QVPO (Ding et al.; NeurIPS 2024; arXiv May 2024): online, model-free diffusion RL with a Q-weighted variational loss (a lower bound on the policy objective under conditions), Q-weight transforms, an entropy term and a variance-reducing behavior policy. MuJoCo only — [arXiv abs](https://arxiv.org/abs/2405.16173)
- Weighted regression vs gradient extraction: Park, Frans, Levine, Kumar (NeurIPS 2024) find the policy-extraction method often matters more than the value objective. AWR-style value-weighted BC "do[es] not fully use the learned value function", and behavior-constrained policy gradient (DDPG+BC) gives substantial gains — [arXiv abs](https://arxiv.org/abs/2406.09329)
- PA-RL (Sobol Mark, Gao, Sampaio, Srirama, Sharma, Finn, Kumar; arXiv 9 Dec 2024). Steps: (1) sample candidates from the base (diffusion) policy; (2) keep the top by Q; (3) gradient-ascend each kept action on Q, accepting only Q-increasing steps; (4) distill with the policy's own supervised loss (DDPM noise prediction). Defaults: 32 samples, top 10, 10 gradient steps, step 3e-4 (16 samples for OpenVLA). Critic: Cal-QL (offline → online) or IQL; with Cal-QL the optimized actions are also used as TD-backup targets — [PA-RL HTML](https://arxiv.org/html/2412.06685)
- PA-RL results. Aggregate offline→online over AntMaze/Kitchen/CALVIN (CALVIN is image-based): PA-RL + Cal-QL 568→711, Cal-QL 309→629, IDQL 497→540, DPPO 215→442, DQL 322→373. Real WidowX cup-to-rack: Cal-QL + PA-RL 55%→90% (DDPM offline 50%); pot-to-sink 80%→100%, in 40–110 min of fine-tuning. OpenVLA 40%→70% after 1 h of zero-shot trials + 40 min fine-tuning. Ablation on antmaze-large-diverse: removing global (best-of-N) selection drops 73→93 to 0→0. Stated weaknesses: sampling cost for big policies; CEM from random actions overestimates Q; CEM initialized from a fixed DP averages modes — [PA-RL HTML](https://arxiv.org/html/2412.06685)
- RFPO, "Refinement-Based Flow Policy Optimization" (Park, Yang, Lee; arXiv v1 14 Sep 2026, v2 27 Sep 2026; no code link). Sample a_1 = Φ_θ(s, x_0); run K MALA steps toward p_α(a|s) ∝ exp(Q(s,a)/α) with an acceptance test, score clipping (norm 10) and adaptive step size (target 0.6 acceptance). Pair the refined action with its own source noise x_0 and regress the flow velocity a_1′ − x_0, with no gradient through the sampler. Critic: clipped double-Q (TD3-style, no entropy), 1e6 replay, batch 256, 1 critic update and G = 5 flow updates per env step. Off-policy, from scratch, MuJoCo only — [RFPO HTML](https://arxiv.org/html/2609.15123)
- RFPO results (1M steps, 5 seeds, final smoothed return): RFPO beats SAC on HalfCheetah 13,269 vs 10,228, Walker2d 5,456 vs 4,710, Ant 5,854 vs 4,061 and HumanoidStandup 158.4k vs 149.5k. It loses on Hopper (2,110 vs SAC 2,969, DPMD 3,038). DIPO is higher on Ant (6,033). Temperature is fixed per environment (e.g. HalfCheetah 0.1, Hopper 0.01). The authors say the experiments "do not fully isolate the contribution of each component". No pretrained-policy or fine-tuning experiments — [RFPO HTML](https://arxiv.org/html/2609.15123)
- QAM, "Q-learning with Adjoint Matching" (Li, Levine; arXiv v1 20 Jan 2026, v4 18 May 2026; no code link on the abs page). A TD-based critic's action gradient enters only as the terminal condition −τ∇_{a_1}Q of a lean adjoint computed by VJPs under the frozen base flow. There is no backprop through the denoising chain, and the optimum is π* ∝ π_β exp(τQ) (a KL anchor to the base policy). Critic: 10-critic ensemble with pessimistic target mean − 0.5·std, EMA 0.005; action chunks h = 5 on cube/scene/puzzle; 10 flow steps; element-wise grad clip 1 — [QAM HTML](https://arxiv.org/html/2601.14234)
- QAM results (OGBench, 50 tasks / 10 domains, 12 seeds, 1M offline steps; the aggregate "all" column was read from a garbled rendering): QAM 44, QAM-FQL 45, QAM-EDIT 46, vs ReBRAC 40, QSM 39, DSRL 38, FQL 36, DAC 36, IFQL 34, CGQL 30, FAWAC 8. QAM-EDIT is reported as the most robust offline-to-online fine-tuner. Stated limits: an ill-conditioned critic gives unstable action gradients; it struggles when optimal actions have very low base-policy probability; τ is the most sensitive hyperparameter; real robots are future work — [QAM HTML](https://arxiv.org/html/2601.14234)
- DSRL (Wagenmaker, Nakamoto, Zhang, Park, Yagoub, Nagabandi, Gupta, Levine; arXiv 2506.15799, June 2025). Keeps the pretrained DP frozen and runs SAC over its input noise (the "latent-action MDP"); the noise for a whole chunk is one RL action. DSRL-NA trains Q^A on real actions by TD (so it can use offline data) and distills a noise-space Q^W from it. Black-box forward passes only — [DSRL HTML](https://arxiv.org/html/2506.15799)
- DSRL settings and results. Chunk 4 on Lift/Can/Square, 8 on Transport; UTD 20–30; 2 critics; noise bound 1.0–1.5; γ 0.99 (0.999 Square); base DPs are DPPO's checkpoints for Lift/Can and new 100-step DPs for Square/Transport. Claims "at least a 5-10× improvement in efficiency" over DPPO/IDQL/DQL/DIPO/QSM (figure-only). Real Franka pick-and-place base 2/10 → 9/10 (RLPD 0/10). π0 on Libero ~20% → ~100% in ~10k samples. Limitation: exploration "inherently determined by the underlying diffusion policy", so a concentrated base distribution may offer too few "options" — [DSRL HTML](https://arxiv.org/html/2506.15799)
- Q-chunking with FQL extraction (QC-FQL): a one-step noise-conditioned policy distilled from the flow behavior policy, with loss α‖z¹ − μ_ψ(s,z⁰)‖² − Q (a W2-style anchor) — [Q-chunking HTML](https://arxiv.org/html/2507.07969)

### Inferences
- On-policy vs off-policy:
  - On-policy, critic used only as a baseline/advantage: DPPO.
  - Off-policy, replay plus TD Q: Diffusion-QL, QSM, DIPO, QVPO, RFPO, QAM, Q-chunking, DSRL and PA-RL online.
  - Offline/implicit: IDQL, offline phases of PA-RL and QAM.
  - Tdyssey's "8 candidates per state" trainer is closest in structure to PA-RL/RFPO-style "critic-refined targets": a Q critic would replace the noisy MC scores, and the top or refined chunks would be distilled with the DP's own epsilon loss.
- Methods that backprop through the sampler (Diffusion-QL) or need ∇_a Q of a chunk critic over 16×2 = 32 action dims (QSM, QAM, PA-RL local steps) are feasible for a small state-based critic. Diffusion-QL through 100 DDPM steps of a 251M UNet is expensive, while QAM's adjoint avoids that cost.
- Every method with positive manipulation evidence anchors to a fixed base: frozen weights (DSRL), the base's sample support via best-of-N (PA-RL, QC), or an explicit KL/W2 to π_β (QAM, QC-FQL). Tdyssey's current update is anchored only to the previous iterate, so drift accumulates. This is the clearest design gap relative to the literature.
- RFPO and QAM have no evidence yet on pixel actors, pretrained robot policies or contact-rich pushing. Treat them as mechanisms to borrow (Q-guided target refinement; base-anchored adjoint objective), not validated recipes.

### Gaps
- No primary-source numbers for QSM, Diffusion-QL or IDQL on robomimic/PushT beyond DPPO's qualitative statements and ResiP's 57% IDQL figure.
- The PA-RL page does not say whether PA-RL itself uses action chunks.
- Whether DSRL's 5–10× efficiency claim holds for pixel actors was not checked (robomimic runs were state-based).
- Code availability for QAM and RFPO is not listed on arXiv.

## Q3. Asymmetric (privileged-state) critics with image actors: evidence and risks

### Takeaway
State critics with image actors are a long-standing, effective practice in simulated manipulation (Pinto et al. 2017), and PushT is close to fully observable from two frames plus pusher xy, so the bias risk is modest. Theory shows a state-only critic is biased whenever the actor's observation does not determine the state (Baisero & Amato 2022). For diffusion-policy RL fine-tuning, I found no paper that pairs an image DP actor with a privileged-state critic. The closest evidence is "do RL in privileged state, then distill to vision" (ResiP), plus same-observation image critics (DPPO ViTCritic, DSRL on real robots, ResFiT).

### Cited Findings
- Pinto, Andrychowicz, Welinder, Zaremba, Abbeel (arXiv Oct 2017): critic trained on full simulator state, actor on rendered RGBD images. Asymmetric inputs "significantly improve performance" on simulated tasks, and with domain randomization the policies transfer to real picking, pushing and block-moving with no real training data. No numbers on the abstract page — [arXiv abs](https://arxiv.org/abs/1710.06542)
- Baisero & Amato (AAMAS 2022): state-based critics for history-based policies have theoretical problems "that undermine a common variant" under partial observability. They propose an unbiased asymmetric actor-critic that keeps the policy-gradient theorem valid, and it converges to better policies and/or faster than symmetric and biased-asymmetric baselines — [arXiv abs](https://arxiv.org/abs/2105.11674)
- From a search-result summary only (not verified on the primary page): Ebi et al. (ICML 2026) "Informed Asymmetric Actor-Critic" conditions the critic on chosen privileged signals rather than full state, and claims unbiased gradients and matching or beating full-state critics — [PMLR page](https://proceedings.mlr.press/v306/ebi26a.html); APRiL aligns attention of jointly trained state and image agents for faster image learning — [arXiv PDF](https://export.arxiv.org/pdf/1911.08363)
- DPPO pixel critic = ViT(image) + 9-D proprio, i.e. the same observation as the actor, not privileged object state — [DPPO config](https://raw.githubusercontent.com/irom-lab/dppo/main/cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp_img.yaml)
- ResiP (Ankile et al., arXiv 2407.16677) does RL in privileged state and then distills to vision. A PPO residual on a frozen chunked DP (Ta = 32, 8 executed) uses privileged state (EE pose, velocity, gripper, all part poses) + base action, with residual scale 0.1, 1024 envs and up to 500M env steps. It reaches 98% on one_leg, then distills into a ResNet18 image DP student (~400 re-rendered synthetic demos + 10–40 real). Image and state students show "similar performance gaps from the teacher's 98%" (exact vision number not in text) — [ResiP HTML v4](https://arxiv.org/html/2407.16677v4)
- ResFiT (Ankile et al., arXiv 2509.19301, Sep 2025): an off-policy residual on a frozen chunked BC policy. The Q-ensemble takes images through a shallow ViT with DrQ random-shift augmentation, plus joint state. n-step 3, UTD ~4, layer norm "crucial", 50/50 demo/online batches. BoxCleanup converges at 200k steps vs 40M for on-policy PPO residual (~200×) — [ResFiT HTML](https://arxiv.org/html/2509.19301v2)
- DSRL on the real WidowX gives the noise-space RL agent proprio + features from the DP's pretrained ResNet-34 encoder; π0 simulation actor/critic use 4-layer convnets on 64×64 images — [DSRL HTML](https://arxiv.org/html/2506.15799)
- RL-100 has all critics/components share the frozen imitation-learned visual encoder, so the critic sees the same encoded observation as the actor; critic inputs are not otherwise stated — [RL-100 HTML v1](https://arxiv.org/html/2510.14830v1)

### Inferences
- PushT observability: two 96×96 frames plus pusher xy determine object pose and roughly its velocity. The Baisero–Amato bias (critic conditioning on information the actor lacks) is therefore expected to be small, but not zero. At 84-pixel crops of a 512-unit workspace (~5–6 units per pixel), sub-pixel pose differences that decide coverage ≥ 0.95 are visible to a state critic but not to the actor. Near the success threshold the critic can credit actions for distinctions the actor cannot perceive.
- A privileged-state critic for Tdyssey would be tiny (pusher xy/vel, object x, y, cos θ, sin θ, velocities, shape id), cheap to train on all collected transitions, and usable only at training time. That satisfies "deployed actor must remain image-based", but it is privileged supervision and must be disclosed (per AGENTS.md).
- For multi-letter training, the critic must condition on shape identity or geometry. The image actor sees the shape; the state critic otherwise would not.
- Lower-risk variants: (a) give the critic the frozen ResNet18 features of the same two frames plus privileged state, an "informed" critic; (b) report a matched comparison of state-only vs image-feature critics on the ranking-validation protocol in Q4.

### Gaps
- No primary source found for an asymmetric state critic paired with an image diffusion actor in RL fine-tuning.
- Pinto et al. and Baisero–Amato quantitative results were not extracted (abstract pages only).
- The Ebi et al. ICML 2026 and APRiL claims come from search summaries, not page reads.

## Q4. Scoring candidate chunks with a learned value (r over executed chunk + γ^8 V(s')) instead of Monte Carlo follow-ups: required accuracy, training (MC vs TD, n-step) and validating ranking quality

### Takeaway
Using r_{0:8} + γ^8 V(s′) is exactly the chunk-level TD target used by Q-chunking, DPPO's chunk-step GAE and RL-100. Its main benefit for Tdyssey is replacing the 24 random follow-up controls with their expectation. That removes the variance source responsible for 65–85% of current score variance, and evaluates V only on states reached by policy samples (in-distribution). The literature gives no formal accuracy threshold. Published practice ranks only N ≈ 10–50 candidates per state with conservative or ensembled critics, and reports critic exploitation as N grows (V-GPS: IQL value success fell from 0.59 at K = 10 to 0.35 at K = 100). Ranking quality should be validated with within-state rank correlation and regret against high-sample MC references before the critic is trusted.

### Cited Findings
- Q-chunking TD target: Σ_{j<h} γ^j r_{t+j} + γ^h Q̄(s_{t+h}, a_{t+h:t+2h}) with the critic taking the whole chunk. This avoids n-step off-policy bias because the critic "takes in the exact same actions that are taken to obtain the n-step rewards" (Prop. A.1). Non-chunked n-step baselines do worse: on OGBench, BFN-n 25→50 and FQL-n 27→57, vs QC 52→86 offline→online — [Q-chunking HTML](https://arxiv.org/html/2507.07969)
- RL-100 chunk formulation: R_chunk = Σ_j γ^j R_{t+j} with inter-chunk discount γ^{n_c}. Offline advantage Q_ψ − V_ψ comes from IQL-style critics, and online advantage from GAE(λ, γ) with V_ψ. One shared environment-level advantage is applied in a PPO-style clipped loss summed over all K denoising steps; denoising std is clipped to [0.01, 0.8] — [RL-100 HTML v1](https://arxiv.org/html/2510.14830v1)
- DPPO: GAE λ = 0.95, γ = 0.999 per chunk-step, summed in-chunk rewards — [DPPO HTML](https://arxiv.org/html/2409.00588v3); [multi_step.py](https://raw.githubusercontent.com/irom-lab/dppo/main/env/gym_utils/wrapper/multi_step.py)
- V-GPS (Nakamoto, Mees, Kumar, Levine; CoRL 2024) re-ranks K samples from a frozen generalist policy with a Q-function. It uses greedy argmax (real robot K = 50) or a softmax with β; the Cal-QL critic was trained offline with rewards 0 on the last 3 demo steps and −1 otherwise, γ 0.98. Gains: real Octo-small-1.5 0.24→0.44; SIMPLER Octo-small 0.27→0.34, Octo-small-1.5 0.02→0.18, OpenVLA 0.24→0.27. Exploitation: IQL critic eggplant 0.59 (K = 10) → 0.35 (K = 100), attributed to exploiting value errors; Cal-QL held 0.84 at K = 50 and 0.63 at K = 100. No formal accuracy analysis — [V-GPS HTML](https://arxiv.org/html/2410.13816v1)
- Best-of-N as a regularizer: argmax over N base samples has an implicit KL bound of log N − (N−1)/N (Q-chunking's QC extraction) — [Q-chunking HTML](https://arxiv.org/html/2507.07969)
- PA-RL keeps only Q-increasing local gradient steps, and its "global" best-of-N selection is essential (antmaze-large-diverse 0→0 without it) — [PA-RL HTML](https://arxiv.org/html/2412.06685)
- OPE validation metrics (Fu et al., ICLR 2021). Absolute error |V^π − V̂^π|; regret@k = true value of the best policy minus true value of the best among the top-k by estimate; Spearman rank correlation. "No evaluated algorithm attains near-oracle performance under any metric." Median rank correlation was 0.90 (model-based AR), 0.70 (FQE-L2) and 0.69 (FQE-D) on RL Unplugged, but mean ≈ 0.25 on D4RL. "A method that does poorly on absolute error may still do well on rank correlation" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.16596)
- RL-100 gates each candidate policy update with offline policy evaluation (AM-Q: Σ Q_ψ along rollouts of a learned transition model) and accepts only if J̃(π) − J̃(π_i) ≥ 0.05·|J̃(π_i)| — [RL-100 HTML v1](https://arxiv.org/html/2510.14830v1)
- ResFiT uses n-step returns with n = 3 and says critic layer norm is crucial; QAM uses a 10-critic ensemble with mean − 0.5·std pessimism — [ResFiT HTML](https://arxiv.org/html/2509.19301v2); [QAM HTML](https://arxiv.org/html/2601.14234)

### Inferences
- Required accuracy (variance algebra, not from a paper):
  - After group-centering, error common to the 8 candidates cancels; only the candidate-specific error variance σ_e² competes with the true within-group spread σ_Q² of Q^π(s, chunk).
  - The current MC score has signal fraction f ≈ 0.15–0.35, i.e. noise/signal variance ratio (1 − f)/f ≈ 1.9–5.7.
  - A critic matches today's signal if σ_e² ≲ 1.9–5.7·σ_Q². To reach f ≈ 0.8 (rank reliability far above today's split-half 0.14–0.29), σ_e must be about half the within-group true-Q std or less.
  - Systematic errors correlated with candidate features (e.g. over-valuing near-goal poses) bias rankings rather than adding noise. The optimizer will seek them out, as in V-GPS at large K.
- Why V(s′) rather than Q(s, a): candidates are current-policy samples and V^π is the exact expectation that the 24 random follow-up controls estimate. A state-value at chunk boundaries is queried on in-distribution states, avoiding Q's out-of-distribution-action overestimation (PA-RL's CEM finding). V must track the current policy: retrain or fine-tune it on recent rollouts each iteration (on-policy), as DPPO does.
- Training recipe options with privileged state and a deterministic simulator:
  - (a) MC regression on full-episode returns from every collected rollout (unbiased, noisy, cheap given 900 controls/s).
  - (b) TD(λ)/GAE at chunk granularity with γ_chunk = γ^8 and λ ≈ 0.95 (DPPO).
  - (c) h-step chunk TD (Q-chunking).
  - Use ensembles plus layer norm (ResFiT, QAM) and report the ensemble spread.
  - Keep the shaping consistent: the current reward 1[success] + 0.99·cov_after − cov_before is potential-style shaping, and its discount should match the critic's per-control γ (standard potential-based shaping result; not re-verified here).
- Validation protocol before trusting V (adapted from Fu et al. metrics):
  - On held-out scenes and states, for each state take the 8 candidates and compute a high-sample MC reference Q̂ (mean of M follow-up continuations from each s′, restored by prefix replay).
  - Report within-group Spearman between the critic score and Q̂, regret@1 (true value of the reference-best candidate minus that of the critic-chosen one), and the same metrics for the current single-sample MC score.
  - Correct the reference's own noise using its split-half reliability.
  - Gate updates on simulator-measured improvement on validation scenes. This is a stronger analogue of RL-100's 5% OPE gate, since Tdyssey has the true simulator.

### Gaps
- No paper found giving a quantitative critic-accuracy threshold for candidate re-ranking in policy fine-tuning.
- No head-to-head MC-follow-up vs learned-V scoring study for diffusion fine-tuning found.
- The value of N used in Q-chunking's QC best-of-N was not in the text read.

## Q5. Action chunking and value functions: how papers handle 4–16-step chunks (SMDP / chunk-level Q)

### Takeaway
The dominant practice treats a chunk as one semi-MDP decision. Rewards are summed (discounted) over the executed controls, the bootstrap is discounted by γ^h, and either V(s) at chunk boundaries (DPPO, RL-100) or Q(s, full chunk) (Q-chunking, DSRL noise-chunk, QAM h = 5) is learned. Reported chunk sizes are small (4–10). Q-chunking finds h = 10 best asymptotically, h = 25 faster early but lower, and h = 50 fails. Residual methods instead add a per-step corrector to keep closed-loop reactivity.

### Cited Findings
- Q-chunking (Li, Zhou, Levine; arXiv 2507.07969, v4 May 2026; venue not stated on the page): critic Q_θ(s_t, a_{t:t+h}), h = 5 for main experiments. Sweep on cube-triple h ∈ {1, 5, 10, 25, 50}: h = 10 peaks asymptotically, h = 25 learns faster early but ends lower, h = 50 gets no success. Budget 1M offline + 1M online steps. OGBench overall offline→online: QC 52→86, QC-FQL 38→86, RLPD 67 (online), RLPD-AC (Gaussian chunked) 61; cube-quadruple QC 73 vs RLPD 0. Limitations: fixed h, may be poor "where a high-frequency control feedback loop is essential", best-of-N cost; UTD 5 did not improve QC — [Q-chunking HTML](https://arxiv.org/html/2507.07969)
- DPPO: env wrapper executes the chunk and sums rewards; robomimic pixel uses horizon 4 / act 4 and Furniture-Bench T_a = 8 — [multi_step.py](https://raw.githubusercontent.com/irom-lab/dppo/main/env/gym_utils/wrapper/multi_step.py); [DPPO config](https://raw.githubusercontent.com/irom-lab/dppo/main/cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp_img.yaml); [DPPO HTML](https://arxiv.org/html/2409.00588v3)
- DSRL: the whole chunk's noise is one RL action (chunk 4 robomimic Lift/Can/Square, 8 Transport, 10 π0-DROID, 50 π0 sim). For π0's 1,600-dim noise they train a single-step actor/critic and repeat the noise across the chunk — [DSRL HTML](https://arxiv.org/html/2506.15799)
- QAM uses action chunking h = 5 on cube/scene/puzzle domains — [QAM HTML](https://arxiv.org/html/2601.14234)
- ResiP: a PPO on the chunked MLP treating 8 actions as one output (PPO-C) was "unstable" and needed careful KL regularization (one_leg 70%), vs a per-step residual on the frozen chunked DP (98%) — [ResiP HTML v4](https://arxiv.org/html/2407.16677v4)
- ResFiT: chunked large policies give ~870-dim actions, motivating a single-step residual on the frozen chunked base — [ResFiT HTML](https://arxiv.org/html/2509.19301v2)
- RL-100: the chunk is one decision with R_chunk = Σ_j γ^j R_{t+j} and inter-chunk discount γ^{n_c} — [RL-100 HTML v1](https://arxiv.org/html/2510.14830v1)

### Inferences
- For Tdyssey (predict 16, execute 8, 2-D actions), the natural critic is V(s) at 8-control boundaries, or Q(s, executed 8-step chunk) with 16 action dims. Q-chunking's evidence (h = 5–10 works, 25+ degrades) puts h = 8 in the validated range. Conditioning Q on the unexecuted second half (controls 9–16) would make the critic depend on actions that never affect the reward, so the executed prefix is the right input.
- Because contact states can only be restored by replaying the full action prefix, chunk-boundary states s′ are reached by one replay per candidate. The V(s′) score costs 8 controls per candidate instead of 32 (4× fewer simulated controls per score), plus one critic forward.

### Gaps
- No paper found that evaluates value functions specifically for 16-predicted / 8-executed receding-horizon chunks on PushT.
- No study found comparing V-at-boundary vs Q-of-chunk critics for diffusion fine-tuning.

## Q6. Overall comparison and fit to the Tdyssey setup (including any PushT evidence)

### Takeaway
No published simulation PushT result for critic-based RL fine-tuning of a pretrained diffusion policy was found. The only PushT RL evidence is RL-100's real-robot "dynamic Push-T" insertion variant (DP-2D 40% → iterative offline RL 90% → online 100%, 50 trials), a different task and observation setup. Three options best match Tdyssey's diagnosed failures (noisy candidate scores, drift from the T policy, high-noise-step change):
- (1) A privileged-state chunk-boundary value that replaces the MC follow-ups in the existing candidate-group trainer, validated by within-state ranking metrics.
- (2) A fixed-base anchor via best-of-N / KL / frozen-base residual or noise steering.
- (3) DPPO-style restriction of updates to the low-noise tail of a short DDIM chain.

### Cited Findings
- RL-100 (arXiv 2510.14830, Oct 2025; v1–v4): Dynamic Push-T on a real UR5 with a stick end-effector, where the T must be inserted into a slot ~3 mm larger per side; 55 cm × 60 cm randomization. 50 trials: DP-2D 40%, DP3 64%, iterative offline RL 90%, online DDIM 100%, online consistency-model 100%. Overall "900 out of 900 episodes" — [RL-100 HTML v1](https://arxiv.org/html/2510.14830v1); [search listing](https://arxiv.org/pdf/2510.14830)
- PushT supervised reference: the LeRobot DP PushT checkpoint reports 65.4% success over 500 episodes (coverage ≥ 0.95), from a search-result summary — [HF model card](https://huggingface.co/lerobot/diffusion_pusht)
- FDPP (preference-reward fine-tuning of DP) on Push-T: "performance slightly decreases" with longer rollouts under preference rewards (success threshold 90% coverage), from a search-result summary — [FDPP PDF](https://arxiv.org/pdf/2501.08259)
- Sample-efficiency evidence across families:
  - Off-policy residual ResFiT ~200× fewer steps than on-policy PPO residual (200k vs 40M) — [ResFiT HTML](https://arxiv.org/html/2509.19301v2)
  - DSRL claims 5–10× over DPPO — [DSRL HTML](https://arxiv.org/html/2506.15799)
  - PA-RL beats DPPO in aggregate (711 vs 442 after fine-tuning) — [PA-RL HTML](https://arxiv.org/html/2412.06685)
  - DPPO is the only one with pixel-robomimic Transport >50% — [DPPO HTML](https://arxiv.org/html/2409.00588v3)
- Anchoring/failure modes:
  - DPPO's off-policy diffusion baselines collapse to 0 after good pre-training — [DPPO HTML](https://arxiv.org/html/2409.00588v3)
  - Without a base-policy constraint, ResFiT converges to "faster but more aggressive behaviors", and learned behavior "may remain constrained around the base policy" with it — [ResFiT HTML](https://arxiv.org/html/2509.19301v2)
  - DSRL cannot exceed what the base can be steered to — [DSRL HTML](https://arxiv.org/html/2506.15799)
  - QAM struggles when optimal actions have low base probability — [QAM HTML](https://arxiv.org/html/2601.14234)

### Inferences
Fit table. "Fit" is my judgement against Tdyssey's constraints: image actor at deployment; 251M UNet; 8-candidate trainer; 60 H100-hours; prefix-replay restore; baselines T 11/20, narrow-T 11/20, Y 1/20, D 0/20, L 0/20.

| Option | Critic | Data regime | Main requirement | Fit / risk for Tdyssey |
|---|---|---|---|---|
| Replace MC follow-ups with r_8 + γ^8 V_priv(s′) in current trainer | privileged-state V at chunk boundaries, MC or TD(λ) | on-policy, retrained each iteration | small MLP; validation of within-group ranking | Highest fit: targets the measured 65–85% follow-up noise directly; keeps trainer; 4× fewer controls per score. Risk: critic exploitation, asymmetric bias near the 0.95 threshold |
| DPPO (PPO + GAE, fine-tune last 5–10 DDIM steps) | V(s_t) per chunk (image+proprio in official code; could be privileged) | on-policy | likelihood via DDIM η = 1, σ_min 0.1; many UNet passes | Proven on pixel DP; addresses high-noise drift by design. Cost is UNet compute, not simulator. No KL to base by default (BC term optional) |
| PA-RL-style critic-refined targets (best-of-N + Q ascent + DDPM-loss distillation) | Q(s, chunk) via Cal-QL/IQL/TD | off-policy replay | Q over 16-dim executed chunk; sampling N ≈ 32 | Good structural match to candidate groups; supervised loss keeps DP on its manifold. Needs base support for Y/D/L successes (1/20, 0/20 starts) |
| RFPO-style MALA refinement toward exp(Q/α), regress with own noise | double-Q TD | off-policy | ∇_a Q; temperature per task | Mechanism only (MuJoCo, from scratch, Sept 2026); untested on pretrained or pixel DP |
| QAM (adjoint matching, π* ∝ π_β exp(τQ)) | ensemble TD Q, chunk h = 5 | off-policy, offline→online | ∇_a Q; VJPs through frozen base; τ tuning | Principled fixed-base anchor; no robot or pixel evidence yet; flow-matching formulation, so epsilon-DDPM adaptation is non-trivial |
| DSRL (SAC on DP input noise) | Q on noise (and Q^A on actions) | off-policy, UTD 20–30 | frozen DP; image features for noise actor | Zero base drift; per-shape steering possible. Limited if base cannot produce L/D successes at all |
| Residual (ResiP on-policy / ResFiT off-policy) | state (ResiP) or image ViT ensemble (ResFiT) | on-/off-policy | small per-step residual on frozen DP | Zero base drift; ResiP needed up to 500M steps; ResFiT far cheaper. Deployed actor becomes DP + residual (image residual or distillation needed) |

- The weakest fit is unanchored off-policy Q-gradient methods (Diffusion-QL/QSM/DIPO as in DPPO's baselines). DPPO observed collapse, and the measured Tdyssey failure is already drift.
- A cheap first experiment, using existing infrastructure:
  - Train a privileged-state, ensembled V on logged rollouts.
  - Measure within-group Spearman/regret@1 vs an M-sample MC reference on held-out scenes.
  - Swap V-based scores into the 8-candidate trainer only if the critic beats the single-sample MC score's reliability (split-half 0.14–0.29).
  - Combine with low-noise-tail-only updates and a fixed-base anchor.
  - Judge everything by simulator success on held-out scenes (per AGENTS.md: imagined or critic-predicted gains are not evidence).

### Gaps
- No simulated PushT (or letter-shape PushT) critic-based RL fine-tuning numbers found in any primary source.
- RL-100's Push-T observation type (2-D image vs point cloud) and demo count were not stated in the text read.
- Compute cost of DPPO-style PPO for a 251M UNet on an H100 was not reported anywhere. DPPO-UNet timings are on different hardware and model sizes, so the Tdyssey cost estimate is my extrapolation.
