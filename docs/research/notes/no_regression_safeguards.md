# No-regression safeguards for RL fine-tuning of a pretrained diffusion policy, and statistical acceptance testing of the final policy

Scope: (a) keeping a RL-fine-tuned PushT Diffusion Policy (DP) close enough to the released base that it does not get worse, and (b) deciding, with a statistical test on fresh held-out scenes, whether a candidate beats the base. Everything is judged against the project: 251M-parameter ConditionalUnet1D denoiser plus frozen image encoder; DDPM, 100 steps, epsilon prediction, squared-cosine schedule with scheduler clipping (confirmed in `docs/POLICY_BASELINE.md`); LoRA rank 8 or full denoiser; forward-process RL (DiffusionNFT-style); binary success (coverage ≥ 0.95 within 300 controls); 20 dev scenes per shape; chaotic rollouts; measured failure = updates anchored only to the previous iterate plus a T-retention term at ~1e-4 of the loss, with T success drifting 0.55 → 0.05.

Conventions: "[computed]" marks numbers I calculated for this note (pure-Python exact binomial/hypergeometric enumeration and Monte Carlo with fixed seed; formulas are given inline so the report writer can reproduce them). They are not quoted from a paper. "AdvantageFlow" could not be identified as a published method; nothing below is about it specifically.

---

## Q1. Anchoring to the base: KL / behavior-cloning regularization toward the initial policy, how KL is estimated for diffusion policies, and the reported effect on retaining original-task performance

### Takeaway
Methods that must keep old abilities anchor to a **fixed** reference: InstructGPT uses a per-token KL to the SFT model (β = 0.02) **plus** a pretraining-likelihood term with γ = 27.8; DPOK and Flow-GRPO use per-denoising-step Gaussian KLs to the pretrained model. RL's Razor finds that forgetting tracks the **forward KL to the base measured on the new task** (R² = 0.96 in its toy study). For an epsilon-prediction DDPM, the per-step KL is a weighted squared difference of noise predictions. A denoising (distillation) loss on **base-policy samples**, noised by the forward process, gives an upper bound on forward KL(π_base‖π_θ) when weighted by the likelihood weighting. That fits a forward-process RL method like DiffusionNFT naturally. In robot fine-tuning, behavior cloning on states with the pretrained policy's actions was the best retention method (Meta-World RoboticSequence). The project's ~1e-4 retention weight is several orders of magnitude weaker than these working recipes.

### Cited Findings
**RLHF-style KL and pretraining-mix anchors**
- InstructGPT's RL objective is `objective(φ) = E_{(x,y)~D_πRL}[ r_θ(x,y) − β log(π_RL(y|x)/π_SFT(y|x)) ] + γ E_{x~D_pretrain}[ log π_RL(x) ]`. It is a per-token KL penalty "from the SFT model at each token" plus pretraining gradients ("PPO-ptx"), added "to fix the performance regressions on public NLP datasets." — [Ouyang et al. 2022, InstructGPT, arXiv 2203.02155](https://arxiv.org/pdf/2203.02155)
- InstructGPT used β = 0.02 and γ = 27.8. Setting the pretraining coefficient "to greater or equal 20" recovered the regressions on the 1.3B model, and a single value of 27.8 worked from 1.3B to 175B parameters. "Mixing in pretraining updates performs better than the simpler solution of increasing the KL coefficient." Some regressions (DROP, SQuADv2, translation) were reduced but not removed. — [InstructGPT](https://arxiv.org/pdf/2203.02155)
- Ziegler et al. found that runs "trained with different seeds and the same KL penalty β sometimes end up with quite different values of KL(π, ρ)". They therefore set β with a log-space proportional controller: `e_t = clip((KL(π_t, ρ) − KL_target)/KL_target, −0.2, 0.2)`, `β_{t+1} = β_t (1 + K_β e_t)`, with K_β = 0.1. — [Ziegler et al. 2019, arXiv 1909.08593](https://arxiv.org/pdf/1909.08593)

**Diffusion-model KL anchors (per denoising step)**
- DPOK (Lemma 4.2) bounds the KL between final samples by the sum of per-step KLs along the denoising chain (data-processing inequality): `E[KL(p_θ(x0|z) ‖ p_pre(x0|z))] ≤ E[ Σ_{t=1}^T E_{p_θ(x_t|z)}[ KL(p_θ(x_{t−1}|x_t,z) ‖ p_pre(x_{t−1}|x_t,z)) ] ]`. — [Fan et al. 2023, DPOK, arXiv 2305.16381](https://arxiv.org/html/2305.16381)
- DPOK's objective is `E_z[ α E_{p_θ}[−r(x0,z)] + β Σ_t E_{p_θ(x_t|z)}[KL_t] ]`, with a PPO-style per-step ratio clip. Appendix values: ε = 1e-4, α = 10, β = 0.01 (single prompt), β = 0.001 (multi-prompt). Without KL, RL "can generate lower-quality images (e.g., over-saturated colors and unnatural shapes)." The KL ablation numbers appear only in figures. — [DPOK](https://arxiv.org/html/2305.16381)
- With shared variance σ_t², the per-step Gaussian KL is `‖μ_θ − μ_pre‖²/(2σ_t²)`. Under epsilon parameterization this becomes `KL_t = β_t²/(α_t(1−ᾱ_t)) · ‖ε_θ − ε_pre‖²/(2σ_t²)`. This is a standard derivation (the fetch tool derived it; DPOK's Lemma 4.3 uses the same ‖μ_pre − μ_θ‖²/(2σ_t²) structure). — [DPOK](https://arxiv.org/html/2305.16381)
- Flow-GRPO's closed-form per-step KL for its SDE sampler is `D_KL(π_θ‖π_ref) = ‖x̄_{t+Δt,θ} − x̄_{t+Δt,ref}‖²/(2σ_t²Δt) = (Δt/2)(σ_t(1−t)/(2t) + 1/σ_t)² ‖v_θ(x_t,t) − v_ref(x_t,t)‖²`. It uses β = 0.04 (GenEval, text rendering) and 0.01 (PickScore). — [Liu et al. 2025, Flow-GRPO, arXiv 2505.05470](https://arxiv.org/html/2505.05470)
- Flow-GRPO, without vs with KL: GenEval task reward was the same (0.95 vs 0.95) but quality metrics fell without KL (ImageReward 0.44 vs 1.03; DeQA 2.77 vs 4.01; Aesthetic 4.93 vs 5.25). The text-rendering task showed the same pattern (OCR 0.93 vs 0.92; ImageReward 0.58 vs 0.95). The authors tune KL to "keep the divergence small and nearly constant during training, keeping the model close to its pretrained weights," and state that KL "is not empirically equivalent to early stopping." — [Flow-GRPO](https://arxiv.org/html/2505.05470)
- Score-matching bound: `D_KL(p ‖ p_θ^SDE) ≤ J_SM(θ; g(·)²) + D_KL(p_T ‖ π)`. The proof uses the data-processing inequality plus Girsanov's theorem on the path measures, and λ(t) = g(t)² is called the "likelihood weighting." — [Song et al. 2021, arXiv 2101.09258, Theorem 1](https://arxiv.org/pdf/2101.09258)

**Forgetting is predicted by KL to the base on the new task**
- RL's Razor: "the degree of forgetting is determined by the distributional shift," measured as forward KL `E_{x~τ}[KL(π0‖π)]` on the **new** task. Quadratic-fit R² for forgetting: forward KL 0.96 ± 0.01; reverse KL 0.93 ± 0.01; total variation 0.80. Weight-change measures do worse (L1 0.34; Fisher-weighted L2 0.58; spectral 0.58), as do activation change (0.52–0.55) on ParityMNIST. The LLM fit is R² = 0.71. — [Shenfeld, Pari & Agrawal 2025, arXiv 2509.04259](https://arxiv.org/html/2509.04259)
- RL's Razor also reports that on-policy RL forgets less than SFT at matched new-task performance (representation similarity to base, CKNNA: RL 0.94 vs SFT 0.56). It includes an OpenVLA-7B / SimplerEnv experiment, but no forgetting numbers for it appear in the text. Its recommendation that algorithms should "explicitly aim to minimize KL divergence from the base model" is a proposal it does not test. — [RL's Razor](https://arxiv.org/html/2509.04259)

**Robot / RL retention methods compared**
- Wołczyk et al. (ICML 2024) compare four retention methods: EWC, `Σ_i F_i(θ*_i − θ_i)²`; BC on stored pre-training states, `E_{s~B_BC}[KL(π*(s)‖π_θ(s))]`; kickstarting (KS), the same KL but with `s ~ π_θ` (current-policy states); and episodic memory. — [Wołczyk et al. 2024, arXiv 2402.02868](https://arxiv.org/html/2402.02868)
- Results by domain: in RoboticSequence (Meta-World), BC was best (all four stages solved 80% of the time), then EM, then EWC; **KS failed**. In NetHack, KS was best, then BC, and EWC degraded. In Montezuma's Revenge, BC improved speed and final return. Coefficients (NetHack): BC 2.0 with no decay; KS 0.5 with decay 0.99998 per step; EWC 2e6. The authors note retention can be harmful if the pretrained policy is suboptimal. Mechanism: the model "deteriorates on parts of the state space" not visited early in fine-tuning ("Far" states). — [Wołczyk et al.](https://arxiv.org/html/2402.02868)
- DPPO: "DPPO does not apply regularization at fine-tuning." Its Furniture-Bench Gaussian baseline with BC regularization α_BC = 0.1 "significantly limits fine-tuning," saturating at 53% (sim) / 50% (real). — [Ren et al. 2024, DPPO, arXiv 2409.00588](https://arxiv.org/html/2409.00588)
- DiffusionNFT has no KL penalty to the pretrained model (only an EMA "old" policy; see Q2). It notes that training exclusively on OCR "impairs generalization to other metrics." — [Zheng et al. 2025, DiffusionNFT, arXiv 2509.16117](https://arxiv.org/html/2509.16117)
- RL-100 (real-world diffusion-policy RL) has no BC or KL term in the RL objective. It relies on IL re-training on the expanded dataset each iteration, a frozen visual encoder, and an offline OPE gate (Q3). — [Lei et al. 2025, RL-100, arXiv 2510.14830](https://arxiv.org/html/2510.14830v1)

**Structural anchors (base weights frozen or change limited)**
- DSRL runs RL over the **latent noise input** of a frozen diffusion policy, "obviating the need to modify the weights of the base policy at all." — [Wagenmaker et al. 2025, DSRL, arXiv 2506.15799](https://arxiv.org/abs/2506.15799)
- Policy Decorator adds a model-agnostic residual policy on a large IL model (tested with Diffusion Policy and Behavior Transformer, 8 ManiSkill/Adroit tasks). It claims to preserve the base model's smooth motion and avoid "erratic behaviors of pure RL policies." — [Yuan et al. 2024, arXiv 2412.13630](https://arxiv.org/abs/2412.13630)
- PA-RL samples candidate actions from the base policy, re-ranks and locally optimizes them with a Q-function, and **distills** them into the policy with a supervised loss. Reported: up to 2× gains; OpenVLA real-world success "40% to 70% … in 40 minutes" (with Cal-QL). — [Sobol Mark et al. 2024, PA-RL, arXiv 2412.06685](https://arxiv.org/abs/2412.06685)
- Cal-QL learns a conservative but "calibrated" Q-function (lower-bounds the learned policy's value, upper-bounds a reference policy's value) for offline-to-online fine-tuning. It was best on 9/11 tasks. — [Nakamoto et al. 2023, Cal-QL, NeurIPS 2023, arXiv 2303.05479](https://arxiv.org/abs/2303.05479)
- LoRA "learns less and forgets less": it underperforms full fine-tuning on the target domain but better preserves source-domain ability. Full fine-tuning perturbations have 10–100× higher rank than typical LoRA. — [Biderman et al. 2024, TMLR, arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- WiSE-FT ensembles zero-shot and fine-tuned **weights**. It gains 4–6 points under distribution shift on ImageNet shifts (+1.6 in-distribution) and 2–23 points on six further shifts, at no extra cost. — [Wortsman et al. 2022, CVPR, arXiv 2109.01903](https://arxiv.org/abs/2109.01903)
- WARP uses an **EMA of the policy as the KL anchor**, SLERP-merges independent runs, then linearly interpolates toward the initialization ("LITI") to recover pre-trained abilities. It claims better reward at fixed KL. — [Ramé et al. 2024, arXiv 2406.16768](https://arxiv.org/abs/2406.16768)

**[computed] Per-step KL weight on ‖ε_θ − ε_ref‖² for the project's schedule**

Assumptions: diffusers `squaredcos_cap_v2`, 100 steps, `max_beta = 0.999`, posterior ("fixed_small") variance; ignores sample clipping. Weight w_k in `KL_k = w_k ‖Δε‖²`. With σ_k² = β̃_k = β_k(1−ᾱ_{k−1})/(1−ᾱ_k), w_k = β_k / (2 α_k (1 − ᾱ_{k−1})). Formula from the DPOK derivation above:

| step k (99 = noisiest) | β_k | ᾱ_k | σ_k | w_k | w_k with σ ≥ 0.1 floor (DPPO σ_min^prob) |
|---|---|---|---|---|---|
| 99 | 0.999 (capped) | ~0 | 0.999 | ≈ 500 | ≈ 500 |
| 90 | 0.189 | 0.020 | 0.43 | 0.12 | 0.12 |
| 75 | 0.075 | 0.13 | 0.27 | 0.047 | 0.047 |
| 50 | 0.032 | 0.48 | 0.17 | 0.032 | 0.032 |
| 25 | 0.013 | 0.84 | 0.11 | 0.045 | 0.045 |
| 10 | 0.0055 | 0.97 | 0.068 | 0.10 | 0.046 |
| 2 | 0.0016 | 0.997 | 0.029 | 0.46 | 0.038 |
| 1 | 0.0011 | 0.998 | 0.020 | 0.89 | 0.036 |

### Inferences
- **The current retention weight is effectively zero.** InstructGPT's γ = 27.8 and Wołczyk's BC coefficient 2.0 (coefficients, not measured loss ratios) are 4–5 orders of magnitude larger than a term at ~1e-4 of the loss. Such a weak term cannot plausibly resist accumulated drift, which matches the measured random-walk decay of T success.
- **Recommended anchor (matches RL's Razor and the forward-process RL style).** Add a fixed-reference distillation loss on base samples, computed on both T states and target-shape states:
  `L_anchor = λ · E_{s ∈ S_T ∪ S_new, a0 ~ π_base(·|s), k, ε} [ w_k ‖ ε_θ(√ᾱ_k a0 + √(1−ᾱ_k) ε, k, s) − ε_base(same input) ‖² ]`.
  Per Song et al. (Theorem 1) with likelihood weighting, plus DPOK's chain bound, this upper-bounds forward KL(π_base‖π_θ) up to the weighting. Forward KL is the predictor RL's Razor found best. The cost is one extra frozen-base forward pass per sample and no reverse-chain backpropagation. With LoRA, the base is the same network with adapters disabled, so no second copy is needed. Base samples a0 can be cached per state from base rollouts.
- **Alternative (on-policy KL, DPOK/kickstarting style).** Use `Σ_k w_k ‖ε_θ − ε_base‖²` evaluated at the reverse-chain states x_k actually visited by π_θ. Wołczyk found the on-policy-state variant (KS) failed in their robotic benchmark while replayed-state BC worked. So prefer the base-sample/replay version, or use both.
- **Weighting matters for the observed high-noise drift.** Under true per-step KL weights, the noisiest step is extremely sensitive (w ≈ 500, because β is capped at 0.999), while mid-chain weights are ~0.03–0.12. Uniform ε-MSE (the DDPM "simple" loss) under-penalizes the high-noise steps relative to KL. The project's measured drift concentrates at high-noise steps, so either use KL-style weights there or freeze the highest-noise steps to the base (DPPO fine-tunes only the last K′ steps; see Q2). Caveat: the official scheduler clips the predicted x0, which bounds the real mean sensitivity at k = 99, so the 500 figure is an upper bound in the unclipped model. Verify empirically by measuring anchor loss per k.
- **Set λ by a KL target, not a fixed number.** Ziegler et al. found the same β gives different KL across seeds. Use the K_β = 0.1 controller on the measured anchor loss, with a budget chosen by checking how much anchor loss the base tolerates before T success drops.
- **Cheap post-hoc safeguard (WiSE-FT/LITI).** LoRA's delta is additive, so scaling the adapter by s ∈ [0, 1] interpolates weights between the base (s = 0, exactly the base policy) and the candidate (s = 1). Choosing s on a validation split adds a no-regression knob at zero training cost. This is an inference from WiSE-FT/WARP; it has not been tested on diffusion policies.

### Gaps
- No source found that measures a KL/BC-to-base anchor's effect on **original-task success** for an RL-fine-tuned robot diffusion policy. DPPO and RL-100 use none, and DSRL/Policy Decorator use structural anchors instead.
- Flow-GRPO's and DPOK's KL ablation effects are on image-quality metrics, not on retaining a task. RL's Razor robotics numbers were not in the fetched text.
- The exact `variance_type` of the official checkpoint was not checked (only "squared-cosine … scheduler clipping" is documented). The w_k table assumes `fixed_small`.
- "AdvantageFlow" could not be identified as a published method.

---

## Q2. Trust regions and clipping for diffusion policies, EMA / slow reference models, and which constrain cumulative drift from the base rather than per-update drift

### Takeaway
PPO/DPPO ratio clipping, TRPO's average-KL constraint, KL-based batch early stopping and DiffusionNFT's EMA "old" policy all measure change **relative to the previous iterate**. They bound per-update change, not total distance from the base, so over many updates drift can accumulate like a random walk. Only a **fixed-reference** term (KL/BC to base, Q1), **structural limits** (frozen base with steered noise or a bounded residual; low LoRA rank; freezing some denoising steps), or **weight interpolation back to the base** limit cumulative drift. An EMA anchor (WARP) is in between: a slow-moving reference, not a fixed one.

### Cited Findings
- **Conservative policy iteration (CPI):** the update is the mixture `π_new(a|s) = (1−α)π_old(a|s) + α π′(a|s)`, with lower bound `η(π_new) ≥ L_{π_old}(π_new) − 2εγ α²/(1−γ)²`, where `ε = max_s E_{a~π′}[A_π(s,a)]`. — [Kakade & Langford 2002, as restated in Schulman et al. 2015, TRPO, arXiv 1502.05477](https://arxiv.org/pdf/1502.05477)
- **TRPO** Theorem 1: with α = D_TV^max(π_old, π_new), `η(π_new) ≥ L_{π_old}(π_new) − 4εγ α²/(1−γ)²`, where ε = max|A|. The KL form is `η ≥ L − C·D_KL^max` with `C = 4εγ/(1−γ)²`. The practical algorithm maximizes the surrogate subject to `E_{s~ρ_old}[D_KL(π_old(·|s)‖π_θ(·|s))] ≤ δ`, with δ = 0.01 in all experiments. All bounds are relative to π_old. — [TRPO](https://arxiv.org/pdf/1502.05477)
- **DPPO** treats denoising as an inner MDP. The PPO clipped ratio is computed **per denoising step**, using the step Gaussian π̄_θ(ā|s̄), with advantage `Â = γ_denoise^k (r̄ − V̂(s_t))`. Clip ε was swept over {0.1, 0.01, 0.001}, choosing "the highest value that satisfies the clipping fraction target" of 10–20%. Values used: ε = 0.01 for Gym/Robomimic state and pixel (annealed), 0.001 for Furniture-Bench. — [DPPO](https://arxiv.org/html/2409.00588)
- DPPO's clip schedule over denoising steps: "using a higher ε for earlier denoising steps in DPPO further improves training stability"; "in practice we set ε_{k=(K−1)} = 0.1 ε_{k=0}, and it follows an exponential schedule among intermediate k." **Flag:** as written, the formula gives the earliest (noisiest) step the *smaller* clip, which contradicts the sentence before it. Check the DPPO code before copying it. — [DPPO Appendix B](https://arxiv.org/html/2409.00588)
- DPPO stops a batch of updates "when the KL divergence between π_θ and π_θold reaches 1," "although in practice we find this never happens." This is again relative to the previous iterate. — [DPPO Appendix B](https://arxiv.org/html/2409.00588)
- DPPO fine-tunes **only the last K′ denoising steps** (the low-noise end): K = 20, K′ = 10 for state tasks; K = 100 with K′ = 5 DDIM steps for pixel tasks; the earlier steps keep frozen θ. "Fine-tuning too few denoising steps (e.g., 3) can lead to subpar asymptotic performance"; "fine-tuning 10 steps leads to the overall best efficiency." Noise floors: σ_min^exp 0.01–0.1 (0.1 for most Robomimic; evaluation 0.001) and σ_min^prob = 0.1 for likelihoods. γ_denoise is 0.99 (0.9 for pixel). — [DPPO](https://arxiv.org/html/2409.00588)
- RL-100 uses a per-denoising-step clipped ratio `Σ_k min(r_k A_t, clip(r_k, 1−ε, 1+ε) A_t)`, with the same environment advantage shared across all K steps, and clips variances to σ ∈ [0.01, 0.8] to keep ratios "well-behaved." — [RL-100](https://arxiv.org/html/2510.14830v1)
- **DiffusionNFT** loss: `E[ r ‖v⁺_θ − v‖² + (1−r) ‖v⁻_θ − v‖² ]`, with `v⁺ = (1−β)v_old + βv_θ` and `v⁻ = (1+β)v_old − βv_θ`. The reward is mapped to `r = 1/2 + 1/2·clip((r_raw − mean_group)/Z_c, −1, 1)`. The optimum is `v* = v_old + (2/β)Δ`. The old policy is an **EMA of the trained policy**, `θ_old ← η_i θ_old + (1−η_i) θ`, with default `η_i = min(0.001·i, 0.5)`. — [DiffusionNFT](https://arxiv.org/html/2509.16117)
- DiffusionNFT trade-offs: "A strictly on-policy scheme (η = 0) yields rapid initial progress but is prone to severe instability"; "η → 1 is robustly stable but suffers from impractically slow convergence." A small η that grows over training works best. β trades stability against speed (β = 1 stable, 0.1 faster). Stability improves with more loss weight at high-noise t, and `w(t) = 1 − t` "lead[s] to collapse". Without the negative term, rewards collapse "almost instantly." — [DiffusionNFT](https://arxiv.org/html/2509.16117)
- WARP uses an EMA of the policy as a **moving** KL anchor, then interpolates toward the initialization (LITI) to recover pre-trained abilities. — [WARP](https://arxiv.org/abs/2406.16768)
- Flow-GRPO holds KL to a **fixed** reference "small and nearly constant during training." — [Flow-GRPO](https://arxiv.org/html/2505.05470)

### Inferences
- **Why per-update limits fail to stop cumulative drift (heuristic).** Suppose each update moves the policy by at most d in a roughly distance-like quantity (√KL, which is locally the Fisher–Rao distance). If update directions are close to uncorrelated (noisy advantages), the distance from the base grows like d√N, so KL-to-base grows like N·δ for a per-update KL budget δ. With δ = 0.01 and 1,000 updates that is about 10 nats, far outside any trust region. If directions are consistent, it grows like N²δ. KL is not a true metric, so this is a scaling argument, not a bound. It matches the project's observation that T success falls roughly in proportion to the number of updates.
- **DiffusionNFT's η limits only the change between updates.** Its EMA "old" velocity is both the data-collection policy and the centre of the implicit positive/negative policies. A larger η slows drift per update but still follows the trained policy, so it does not tie the policy to the base. To anchor, add the fixed-base term from Q1, or replace v_old in a third loss term with v_base.
- **Mapping to the project.**
  1. Keep per-update control (DPPO-style clip, or a small DiffusionNFT β and η schedule) for stability.
  2. Add a fixed-base anchor with a target-KL controller for retention.
  3. Use structural limits that cap total change: LoRA rank 8 rather than the full denoiser (LoRA forgets less), and optionally freeze the highest-noise steps to the base, where the project's drift concentrates. DPPO's choice to adapt only the last (low-noise) K′ steps is the robot precedent.
  4. Before deployment, scale the LoRA delta (WiSE-FT/LITI). Constraints of the second and third kind bound **cumulative** drift; the first bounds only per-update drift.
- **DSRL-style steering is the strongest structural guarantee:** the base weights never change, and the steering policy can be initialized to N(0, I), giving exactly the base behaviour. It does not fit the project's current denoiser fine-tuning plan, but it makes a natural safe baseline or comparison.

### Gaps
- No paper found that measures cumulative KL-to-base over DPPO/DiffusionNFT training, or relates it to retention of the original task, for robot diffusion policies.
- DPPO's clip schedule direction is internally inconsistent in the text. Code was not checked.

---

## Q3. Conservative and safe policy improvement theory (CPI, SPIBB, HCPI) and practical robot-learning analogues: accept an update only if a lower confidence bound beats the base

### Takeaway
Safe-policy-improvement theory gives a direct template:
1. Split the data.
2. Search for and select **one** candidate on the training split.
3. Run **one** pre-planned safety test on a held-out split.
4. Return the candidate only if a 1−δ lower confidence bound on its performance beats the base's level; otherwise return the base ("No Solution Found").

HCPI does exactly this. SPIBB falls back to the baseline wherever data are thin, and CPI moves to a mixture with the baseline instead of jumping. RL-100 is a robot analogue (OPE-gated acceptance with keep-previous fallback). In this project, true simulator rollouts replace off-policy estimates, so the gate can use exact binomial / McNemar bounds.

### Cited Findings
- **HCPI** (Thomas, Theocharous & Ghavamzadeh, ICML 2015) is a batch RL algorithm "that provides probabilistic guarantees." Its POLICYIMPROVEMENT takes trajectories D, a performance lower bound ρ₋ and a confidence δ, and "outputs either a new policy or NO SOLUTION FOUND (NSF)." — [Thomas et al. 2015, HCPI](https://proceedings.mlr.press/v37/thomas15.pdf)
- HCPI on multiple comparisons: testing two candidates on the same data means "its error rate can be as large as 2δ." HCPI avoids this "by setting aside data that is only used for a single safety test." In practice it places "1/5 of the trajectories in the training set and the remainder in the test set." Algorithm 4: `π_c ← GetCandidatePolicy(D_train, δ, ρ₋, |D_test|)`; if `ρ₋^†(π_c, D_test, δ, |D_test|) ≥ ρ₋`, return π_c; otherwise return NSF. — [HCPI](https://proceedings.mlr.press/v37/thomas15.pdf)
- HCPI candidate selection predicts the lower bound **as if computed from m = |D_test| samples**, which counters overfitting. Without this, with few trajectories, selection "finds a policy that it predicts will perform exceptionally well and which will easily pass the safety test, but actually fails" it. Bounds: concentration inequality (safe), one-sided Student t and BCa bootstrap ("semi-safe"; error may exceed δ). Experiments set ρ₋ to an empirical estimate of the initial policy's performance with δ = 0.05. On Mountain Car, the safe CI version needed 900 trajectories (400 with k-fold candidate selection), and all methods reached an optimal policy with 5,000. — [HCPI](https://proceedings.mlr.press/v37/thomas15.pdf)
- HCPI's incremental version (DAEDALUS2) repeats improve-and-test over time. Re-using test data across iterations makes it "semi-safe," and the authors argue the bias is small. — [HCPI](https://proceedings.mlr.press/v37/thomas15.pdf)
- **SPIBB** "bootstraps the trained policy with the baseline when the uncertainty is high" (state–action pairs with too few samples), and the trained policy "is guaranteed to perform at least as well as the baseline policy used to collect the data." ICML 2019 long oral. — [Laroche, Trichelair & Tachet des Combes, arXiv 1712.06924](https://arxiv.org/abs/1712.06924)
- **CPI** uses a mixture update with an explicit improvement lower bound (formula in Q2). — [TRPO restating Kakade & Langford](https://arxiv.org/pdf/1502.05477)
- **RL-100 gate:** a candidate replaces the behaviour policy only if `Ĵ(π) − Ĵ(π_i) ≥ δ`, with adaptive threshold `δ = 0.05·|Ĵ(π_i)|`, using a learned-model/Q estimator (AM-Q) without new interaction. "Otherwise, we reject and keep the behavior policy unchanged." It claims "conservative, monotonic behavior-policy improvement," with no proof. The gate applies only to the offline stage. — [RL-100](https://arxiv.org/html/2510.14830v1)

### Inferences
- **Project gate (HCPI translated to simulator rollouts).** Data roles: development scenes for training diagnostics; validation scenes for choosing a few candidates (checkpoint, LoRA scale s, λ); test scenes used once.
  - **Accept** a candidate only if, on the test scenes and paired against the base on the same scenes:
    - (i) a one-sided superiority test on each target shape passes, or on the pooled target shapes, if that endpoint is fixed in advance;
    - (ii) a **non-inferiority** test on T (and on shapes the base already solves) passes, with a pre-set margin;
    - (iii) a Holm/Bonferroni correction is applied across the shapes and candidates tested.
  - **Otherwise ship the base** (NSF).
  - Sizes are in Q4. Because outcomes come from the real simulator rather than importance sampling, exact binomial bounds give the "safe" (not semi-safe) version.
- **HCPI's "inflated bound" trick, applied to selection.** When ranking candidates on validation scenes, rank by the lower confidence bound computed with the test-set size m, not by the raw success rate. This prefers candidates likely to pass the one-shot test and limits the winner's curse (Q5).
- **Safe default when shapes conflict.** CPI- and SPIBB-style fallbacks map naturally onto choosing the LoRA scale s < 1, or a per-shape fallback. But a per-shape switch would break "one shared policy," so prefer a global s.
- Offline estimates (world-model rewards) must never stand in for the gate. RL-100's OPE gate is a weaker substitute that is justified only when real interaction is expensive, which the simulator here is not.

### Gaps
- No robot-learning paper found that uses a formal high-confidence safety test, HCPI style, to accept a fine-tuned **diffusion** policy on simulator rollouts. RL-100's gate is an offline-model estimate without a confidence bound.
- The SPIBB abstract gives no δ/ζ form of the guarantee or N∧ threshold; the full paper was not fetched.

---

## Q4. Statistical acceptance testing for binary success rates: required episodes for given improvements, paired designs (McNemar/exact binomial), Wilson intervals, sequential tests (SPRT/STEP), multiple-comparison corrections, and robot-evaluation best practices

### Takeaway
**Twenty scenes per shape cannot detect 10-point changes.**
- The 95% interval on 11/20 is [0.34, 0.74].
- A one-sided Fisher test at α = 0.05 has only 8–9% power for 0.55 ↔ 0.45 at n = 20 per arm [computed].

**Required sizes [computed]:**
- Detecting **Y 0.05 → 0.30** needs about 28 (one-sided) to 36 (two-sided) episodes per arm unpaired, or about 27–30 paired scenes, for 80% power.
- Detecting a **10-point T drop** (0.55 → 0.45) needs about 300 episodes per arm unpaired, and 90–310 paired scenes depending on how often base and candidate disagree on the same scene.
- A T **non-inferiority** test with a 10-point margin needs about 307 per arm.

Pairing on the same scenes helps **only** if outcomes are correlated across policies. Chaotic divergence weakens that correlation, so measure the disagreement rate with a base-vs-base (A/A) run before sizing.

Robot-evaluation practice (TRI LBM, STEP, Kress-Gazit et al.):
- 50 real / 200 sim rollouts per condition;
- blinded and randomized order, matched initial conditions or seeds;
- report counts, not only rates;
- Bonferroni across pairwise comparisons;
- sequential tests only if designed as sequential.

The evaluation budget is affordable at about 1,200 episodes per hour.

### Cited Findings
**Robot-learning evaluation practice**
- TRI's LBM study evaluates "50 rollouts per task per policy per condition" in the real world and "200 times per task per policy per condition" in simulation, targeting "policy success rates around 50%." Its other practices:
  - uniform prior on the Bernoulli parameter (posterior violins);
  - a sequential test that "sequentially compares paired outcomes of each evaluation trial" with "Type-I error control for binary data";
  - Bonferroni correction to "maintain a global 95% confidence level," shown as compact letter displays;
  - blinded evaluators, randomized policy order, and the same seed for all policies in sim.

  The study admits Type-I error "is not globally controlled across tasks" in multi-task plots. — [TRI LBM team 2025, arXiv 2507.05331](https://arxiv.org/html/2507.05331)
- **STEP** (RSS 2025) is a sequential test for comparing two Bernoulli policies. Its decision regions are computed offline by solving a linear program at each step, controlling worst-case Type-I error over nulls p0 = p1 with a risk budget f(n) = α*/N_max. It requires α* and N_max to be set in advance, and it "abstains" if evidence stays low. It saves "up to 32%" of trials compared with baselines, and over 160 rollouts in a multitask comparison. — [Snyder et al. 2025, arXiv 2503.10966](https://arxiv.org/html/2503.10966)
- STEP stopping times (Table I) with α* = 0.05: 0.56 vs 0.92 → about 19–23 trials; 0.28 vs 0.80 → 8–13. With α* = 0.01: 0.084 vs 0.386 → 36; 0.400 vs 0.564 → 131; 0.000 vs 0.030 → 225. For 59% vs 68% the authors "would require N_max = 500 trials." — [STEP](https://arxiv.org/html/2503.10966)
- STEP on misuse: running Barnard's batch test repeatedly "violates Type-1 Error control." Appending trials to a batch test and re-running it "would constitute p-hacking," and a non-significant result does not justify accepting the null. Rule of thumb: for N ≥ 36 and p̂ ∈ [0.25, 0.75], a 95% CI is about p̂ ± 1/√N. — [STEP](https://arxiv.org/html/2503.10966)
- Kress-Gazit et al. (TRI best practices):
  - point estimates alone "can be misleading"; use frequentist intervals or Bayesian posteriors;
  - report the number of evaluations and the initial conditions;
  - run blinded, interleaved A/B evaluation within one session;
  - match initial conditions;
  - describe failure modes.

  Example: 15/18 vs 11/17 still leaves a 0.11 posterior probability that the "worse" policy is better. — [Kress-Gazit et al. 2024, arXiv 2409.09491](https://arxiv.org/html/2409.09491v2)
- Vincent et al. bound the whole **distribution** of robot performance (CDF bounds via stochastic ordering) at a user-set confidence, with open-source code. — [Vincent et al. 2024, arXiv 2405.05439](https://arxiv.org/abs/2405.05439)
- rliable (NeurIPS 2021):
  - use stratified-bootstrap CIs (resample runs within each task);
  - use the IQM (25% trimmed mean);
  - use the probability of improvement `P(X > Y) = (1/M) Σ_m P(X_m > Y_m)`;
  - percentile CIs are fine for median/IQM at N = 10 runs, but single-task means need about 20–30 runs for nominal coverage;
  - give CIs on the **difference** between algorithms.

  — [Agarwal et al. 2021, arXiv 2108.13264](https://arxiv.org/html/2108.13264)

**Statistical formulas**
- **Connor (1987)** paired sample size, in number of pairs: `n = (z_{1−α/2} √p_d + z_{1−β} √(p_d − δ²))² / δ²`, with discordance p_d = p₁₂ + p₂₁ and δ = p₂₁ − p₁₂ (Biometrics 43:207–211). The one-sided form uses z_{1−α}. SAS PROC POWER offers Connor, Miettinen and exact methods. — [MetricGate summary](https://metricgate.com/docs/sample-size-mcnemar-paired); [SAS PROC POWER](https://support.sas.com/documentation/cdl/en/statug/65328/HTML/default/statug_power_details55.htm)
- **Wald SPRT:** accumulate the log-likelihood ratio S_i. Accept H1 when S_i ≥ b ≈ log((1−β)/α); accept H0 when S_i ≤ a ≈ log(β/(1−α)); otherwise continue (Wald 1945, Ann. Math. Stat. 16:117–186). — [Wikipedia: SPRT](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test)

**[computed] 95% intervals.** Wilson: `(p̂ + z²/2n ± z√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)`; Clopper–Pearson exact by binomial-tail inversion.

| observed | Wilson 95% | Clopper–Pearson 95% |
|---|---|---|
| 11/20 (T base) | [0.342, 0.742] | [0.315, 0.769] |
| 1/20 (Y base) | [0.009, 0.236] | [0.001, 0.249] |
| 0/20 | [0, 0.161] | [0, 0.168] (one-sided 95% upper 0.139) |
| 55/100 | [0.452, 0.644] | [0.447, 0.650] |
| 5/100 | [0.022, 0.112] | [0.016, 0.113] |
| 30/100 | [0.219, 0.396] | [0.212, 0.400] |
| 110/200 | [0.481, 0.617] | [0.478, 0.620] |

**[computed] Unpaired episodes per arm, 80% power.** Normal approximation with pooled variance: `n = (z_{1−α}√(2p̄q̄) + z_{1−β}√(p₀q₀ + p₁q₁))² / (p₁ − p₀)²`.

| change | one-sided α = .05 | one-sided, 90% power | two-sided α = .05 |
|---|---|---|---|
| 0.05 → 0.30 (Y target) | 28 | 38 | 36 |
| 0.05 → 0.20 | 60 | 82 | 76 |
| 0.05 → 0.15 | 111 | 153 | 141 |
| 0.20 → 0.40 | 64 | 89 | 82 |
| 0.55 → 0.65 (+10, T) | 296 | 410 | 376 |
| 0.55 → 0.45 (−10, T) | 309 | 427 | 392 |
| 0.55 → 0.35 (−20, T) | 76 | 105 | 96 |

**[computed] Exact one-sided Fisher test power at α = .05** (exact enumeration over the conditional hypergeometric). Same n per arm:

| change | n = 20 | 40 | 60 | 100 | 200 | 300 | 400 |
|---|---|---|---|---|---|---|---|
| 0.05 → 0.30 | 0.54 | 0.89 | 0.98 | 1.00 | | | |
| 0.55 → 0.65 | 0.09 | 0.16 | 0.26 | 0.36 | 0.61 | 0.79 | 0.88 |
| 0.55 → 0.45 | 0.08 | 0.16 | 0.26 | 0.36 | 0.60 | 0.76 | 0.86 |
| 0.55 → 0.35 | 0.23 | 0.46 | 0.68 | 0.86 | 0.99 | | |

The actual size at n = 100, p = 0.55 vs 0.55, is 0.038 (Fisher is conservative).

**[computed] Paired design (same scenes): exact one-sided McNemar power** (conditional binomial test on disagreeing pairs) and Connor n. Here p_d = P(base and candidate disagree on a scene).

| scenario | p_d | Connor pairs (80%) | exact power at 20 / 40 / 60 / 100 / 200 pairs |
|---|---|---|---|
| Y 0.05 → 0.30, outcomes independent | 0.32 | 30 | 0.49 / 0.88 / 0.98 / 1.00 / 1.00 |
| Y 0.05 → 0.30, candidate never loses a base success | 0.25 | – | 0.59 / 0.98 / 1.00 / 1.00 / 1.00 |
| T 0.55 → 0.45, independent | 0.505 | 311 | 0.09 / 0.17 / 0.24 / 0.36 / 0.60 |
| T 0.55 → 0.45, p_d = 0.30 | 0.30 | 184 | 0.09 / 0.22 / 0.33 / 0.51 / 0.80 |
| T 0.55 → 0.45, p_d = 0.15 | 0.15 | 91 | 0.06 / 0.31 / 0.52 / 0.80 / 0.98 |
| T 0.55 → 0.65, p_d = 0.20 | 0.20 | 122 | 0.08 / 0.26 / 0.43 / 0.68 / 0.93 |

**[computed] Non-inferiority per arm** (unpaired; true p_C = p_B; one-sided α = .05; 80% power). Formula: `n = (z_{.95} + z_{.80})² · 2p(1−p) / margin²`.

| base p | margin | n per arm |
|---|---|---|
| 0.55 | 0.10 | 307 |
| 0.55 | 0.15 | 137 |
| 0.55 | 0.20 | 77 |

**[computed] Wald SPRT, single sample, for H0: p = 0.05 vs H1: p = 0.30** (α = .05, β = .2):
- Boundaries: stop for H1 at LLR ≥ 2.77 and for H0 at LLR ≤ −1.56. Each success adds +1.79; each failure adds −0.31.
- Average sample size: about 6–8 episodes.
- Simulated with truncation at 100 episodes: P(declare improvement) = 0.03 at true p = 0.05, 0.15 at p = 0.10, and 0.83 at p = 0.30.

**[computed] Bonferroni cost.** Testing K candidates at α/K multiplies the required n by `((z_{1−α/K} + z_{.8})/(z_{.95} + z_{.8}))²`: 1.62× for K = 5, 1.89× for K = 10, 2.15× for K = 20, 2.50× for K = 50.

### Inferences
- **Budget conversion.** About 100 episodes per 5 minutes means roughly 1,200 episodes per GPU-hour, or about 3 s per episode. A final confirmatory test with 200 fresh scenes × 5 shapes × 2 policies (base and candidate) is 2,000 episodes, about 1.7 h, which is roughly 3% of the 60 h budget. A 300-scene non-inferiority test on T for 3 candidates is about 1,200 episodes, about 1 h. A properly powered test is affordable; the 20-scene dev set should only screen candidates.
- **What to estimate given the chaos.** Treat each scene as having a latent success probability q_s, averaged over tiny numerical differences. The target quantity is E_s[q_s] over the scene distribution. With a fixed episode budget, one rollout on each of many distinct scenes estimates this better than several rollouts on fewer scenes: the variance is inflated by the design effect 1 + (m−1)·ICC for m rollouts per scene. This is a standard cluster-sampling result; no source was fetched for it.
- **Pairing and common random numbers.** Use the same scene seed and the same initial-noise seed for base and candidate. This keeps any correlation from scene difficulty, although trajectories diverge after 30–150 controls. Before sizing the test:
  - run the base against itself on about 100 scenes with different policy-noise seeds;
  - the fraction of disagreeing scenes estimates the null disagreement rate p_d;
  - if p_d is near 2p(1−p) ≈ 0.5 for T, pairing buys nothing and the unpaired sizes apply;
  - if many scenes are deterministic (always succeed or always fail), p_d is small and paired tests need far fewer scenes (e.g., 91 vs 311 pairs for a 10-point T drop).
- **Test choice.**
  - Paired design: exact conditional McNemar (one-sided binomial on disagreeing pairs).
  - Unpaired: Fisher (conservative) or Boschloo/Barnard (more powerful).
  - Intervals: Wilson or Clopper–Pearson for per-shape rates, plus a CI on the paired difference.
  - Sequential: use STEP-style or a properly planned SPRT on **disagreeing pairs** (H0: P(candidate wins | disagreement) = 0.5). A one-sample SPRT against "p0 = 0.05" is invalid because the base rate itself is uncertain ([0.009, 0.236] from 1/20).
- **Endpoints.** Fix in advance:
  - primary: superiority on the pooled new shapes (Y, D, narrow T, L), which needs about 4× fewer scenes per shape if the effect is similar;
  - secondary: per-shape superiority;
  - guard: T non-inferiority with margin 0.10 or 0.15, which is about 300 or 140 per arm unpaired.

  Holm correction across endpoints.

### Gaps
- No source found that gives sample sizes or pairing efficiency specifically for chaotic simulators with near-duplicate rollouts. The project's own A/A test is needed.
- The Brown, Cai & DasGupta (2001) recommendation of Wilson/Jeffreys over Wald intervals is standard but could not be fetched (the page was empty).
- STEP's exact boundary tables for small N_max were not extracted. Using it requires running its released code.

---

## Q5. Checkpoint selection bias: selecting the best checkpoint on development scenes inflates its score; how to keep a fresh test set, and how big it must be

### Takeaway
Picking the best of K checkpoints on 20 dev scenes inflates the score substantially even when **no checkpoint is truly better** [computed]:
- true 0.55: the expected maximum is 0.68 for K = 5, 0.72 for K = 10, 0.79 for K = 50;
- on 100 dev episodes the inflation is +0.06 to +0.11;
- a truly 0.05 policy on 20 scenes typically appears at 0.11–0.18.

So dev and validation numbers must never be reported as final. Use a never-touched test set, sized for the post-selection hypothesis test (Q4). Test only a pre-chosen few candidates (ideally one, as in HCPI), with Bonferroni/Holm if more than one is tested.

### Cited Findings
- rliable: taking the maximum over evaluations or configurations "introduces positive bias." CURL's max-over-10-evaluations protocol, applied to its DER baseline, gave scores far above CURL's reported ones; much of SUNRISE's reported gain came from a max over 8 configurations. "The maximum of random variables is a biased estimate of their true maximum." — [Agarwal et al. 2021](https://arxiv.org/html/2108.13264)
- HCPI: using the same data to select and to test lets the error rate reach 2δ with two candidates, and more with more candidates. The fix is a held-out set used for a single safety test. Without that, few-data selection "finds a policy that it predicts will perform exceptionally well … but actually fails" the safety test. — [HCPI](https://proceedings.mlr.press/v37/thomas15.pdf)
- Cawley & Talbot: the variance of model-selection criteria lets selection over-fit. The resulting bias can be "surprisingly large," comparable to differences between learning algorithms, and common evaluation practices that reuse selection data are unreliable. — [Cawley & Talbot 2010, JMLR 11](https://www.jmlr.org/papers/v11/cawley10a.html)
- "Reusing a holdout set adaptively multiple times can easily lead to overfitting to the holdout set itself." Dwork et al. give mechanisms for reusing a holdout safely under adaptive queries. — [Dwork et al. 2015, arXiv 1506.02629](https://arxiv.org/abs/1506.02629)
- STEP: appending trials and re-testing is p-hacking unless the procedure was designed as sequential. — [STEP](https://arxiv.org/html/2503.10966)
- TRI LBM applies Bonferroni across pairwise policy comparisons for a global 95% level. — [TRI LBM](https://arxiv.org/html/2507.05331)

**[computed] Winner's curse:** expected maximum (and 95th percentile) of observed success over K checkpoints with **identical** true success, Monte Carlo.

| true p | dev n | K = 5 | K = 10 | K = 20 | K = 50 |
|---|---|---|---|---|---|
| 0.55 | 20 | 0.678 (+0.13), p95 0.80 | 0.718 (+0.17), p95 0.85 | 0.753 (+0.20) | 0.792 (+0.24) |
| 0.55 | 100 | 0.607 (+0.06), p95 0.66 | 0.627 (+0.08) | 0.642 (+0.09) | 0.660 (+0.11), p95 0.70 |
| 0.05 | 20 | 0.108 (+0.06), p95 0.20 | 0.133 (+0.08) | 0.156 (+0.11) | 0.184 (+0.13), p95 0.25 |
| 0.05 | 100 | 0.076 (+0.03) | 0.086 (+0.04) | 0.094 (+0.04) | 0.104 (+0.05), p95 0.13 |

### Inferences
- **Three disjoint scene pools, by seed range or hash, recorded in the repo:**
  1. **Dev:** the existing 20 per shape. Reused freely for debugging and for retention monitoring (Q6). Never quoted as a result.
  2. **Validation:** about 100 fresh scenes per shape. Used to rank checkpoints, λ and the LoRA scale s, ranking by a lower confidence bound (HCPI trick). Report it with the caveat that it was used for selection.
  3. **Test:** generated once, never inspected during development. At least about 200–300 scenes per shape, depending on the A/A disagreement rate and the T non-inferiority margin (Q4). Used once for one to three pre-registered candidates plus the base, with Holm correction.
- **Size of the test set.** It is set by the hardest pre-registered endpoint, which is usually T non-inferiority:
  - margin 0.10 → about 300 scenes unpaired, or about 90–300 paired depending on p_d;
  - margin 0.15 → about 140;
  - target-shape superiority for gains like 0.05 → 0.30 needs only about 30–40 scenes.

  At about 3 s per episode, 300 scenes × 5 shapes × 2 policies is about 2.5 h.
- If the test set is ever used to make a decision (e.g., a candidate fails and the team iterates), it becomes a validation set. Generate a new test set for the next confirmatory claim.
- Selection inflation for low-success shapes (Y) is as large in relative terms as for T. A Y "improvement" from 1/20 to 3/20 on dev is fully explained by selection noise across 10–20 checkpoints.

### Gaps
- No robot-learning paper found that quantifies checkpoint-selection inflation for success rates. The numbers above are from simulation under the null.

---

## Q6. Early stopping and rollback rules during training: stop or revert when retention metrics fall beyond noise

### Takeaway
Monitor two things:
1. **A cheap, low-noise drift signal:** anchor loss / KL to base on fixed T and target states. RL's Razor shows KL-to-base predicts forgetting, and it needs no rollouts.
2. **A periodic rollout check of retention** with thresholds fixed in advance.

With 20 T episodes, a 5% false-alarm rule ("alarm if ≤ 6/20 successes") catches only collapses to ≤ 0.25 reliably. Use 40–60 T episodes per check to catch drops to 0.35 with 80–93% probability [computed]. On an alarm:
- roll back to the last accepted checkpoint;
- raise λ or lower the step size or learning rate;
- do not keep training from the degraded weights.

Existing recipes (DPPO, Ziegler, RL-100) give only per-update KL stops, adaptive β, or keep-previous gates.

### Cited Findings
- DPPO stops each iteration's batch updates when KL(π_θ‖π_θold) reaches 1 (never triggered in practice). — [DPPO](https://arxiv.org/html/2409.00588)
- Ziegler et al. use an adaptive β controller to hit a target KL (formula in Q1). — [Ziegler et al. 2019](https://arxiv.org/pdf/1909.08593)
- Flow-GRPO: a KL term "is not empirically equivalent to early stopping." With KL, high reward is reached while quality is kept, but training is slower. — [Flow-GRPO](https://arxiv.org/html/2505.05470)
- RL-100 rejects a candidate that fails its gate and "keep[s] the behavior policy unchanged." It describes no online-stage rollback. — [RL-100](https://arxiv.org/html/2510.14830v1)
- DiffusionNFT: on-policy updates (η = 0) make "rapid initial progress but [are] prone to severe instability." A small-then-growing η stabilizes training. — [DiffusionNFT](https://arxiv.org/html/2509.16117)
- Wołczyk et al. decay the kickstarting coefficient (0.99998 per step). BC regularization without decay kept performance in "Far" states that vanilla fine-tuning forgot. — [Wołczyk et al.](https://arxiv.org/html/2402.02868)
- RL's Razor: forward KL to base predicts forgetting (R² 0.96 toy, 0.71 LLM), better than weight-change metrics (0.34–0.58). — [RL's Razor](https://arxiv.org/html/2509.04259)
- STEP: repeated batch testing inflates Type-I error, so a valid sequential procedure must be planned in advance (α*, N_max, risk budget). — [STEP](https://arxiv.org/html/2503.10966)

**[computed] Single-check retention alarms for base T p = 0.55** (exact binomial; alarm threshold chosen so the false-alarm rate is ≤ 5% per check):

| T episodes per check | alarm if successes ≤ | false alarm | P(alarm) if true T = 0.45 | 0.35 | 0.25 | 0.15 |
|---|---|---|---|---|---|---|
| 20 | 6 (0.30) | 0.021 | 0.13 | 0.42 | 0.79 | 0.98 |
| 40 | 16 (0.40) | 0.041 | 0.32 | 0.80 | 0.99 | 1.00 |
| 60 | 26 (0.43) | 0.046 | 0.45 | 0.93 | 1.00 | 1.00 |
| 100 | 46 (0.46) | 0.044 | 0.62 | 0.99 | 1.00 | 1.00 |

### Inferences
- **Proposed rule set (inference, not taken from a paper).**
  1. **Every update:** log anchor loss (KL-to-base proxy) per denoising-step bucket on fixed T and target states. Drive λ with a target-KL controller (Ziegler). If the anchor loss exceeds, say, 2× target for M consecutive logs, stop and roll back. This needs no rollouts, so it is low-noise and cheap.
  2. **Every N iterations:** run 40–60 T episodes on fixed monitor scenes, separate from the test scenes, with fixed seeds for common random numbers. Alarm if T successes are ≤ the table threshold.
  3. **Avoid false alarms from repeated checks** with an α-spending schedule, by requiring two consecutive alarms, or with a one-sided CUSUM on per-episode outcomes. With 20 checks at 5% each, a no-change run alarms at least once with probability up to 1 − 0.95²⁰ ≈ 64% if checks were independent.
  4. **On alarm:** revert to the last checkpoint that passed, raise λ (or lower the LR / DiffusionNFT β, or raise η), and resume.
  5. **Every checkpoint that passes** is a candidate for the validation ranking (Q5).
- Use the T-success decline already measured in fast runs (0.55 → 0.05) to calibrate the threshold: such a collapse is caught at any n ≥ 20. The hard case is a slow 10-point slide, which needs about 300 episodes to confirm. That is why the anchor-loss signal, not rollouts, should be the early warning.
- DPPO-style per-batch KL stops (relative to π_old) remain useful against single bad updates, but do nothing for slow cumulative drift (Q2).

### Gaps
- No published early-stopping or rollback rule for retention in RL fine-tuning of robot diffusion policies was found. The rule set above is a synthesis.
- How well anchor loss predicts T success for this policy is untested. A calibration sweep is needed: perturb the LoRA scale or train briefly without the anchor, then plot T success against anchor loss.
