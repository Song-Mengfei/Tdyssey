# Offline C training diagnosis

Direct saved evidence:

- 2048 controls, 260 plans, 18 fresh batches and 52 actor/critic passes.
- L: 932 controls; 247 contact controls; 0 successes; raw reward sum 302.406800175; training reward sum -1.88496361621; shaping sum -1.88496361621; maximum coverage 0.486981121585.
- T: 1116 controls; 272 contact controls; 1 successes; raw reward sum 190.74539094; training reward sum 0.0281903166987; shaping sum -0.971809683301; maximum coverage 0.958975632042.
- Negative mixed-buffer critic explained variance in 28/52 passes; final value 0.929035597. Maximum post-pass KL 0.000161718749, clip fraction 0.001875.
- Actor pre-clip gradient norm exceeds configured limit in 43 passes. Every pass has positive actor/critic weight change; 18 batch frozen-fingerprint checks passed.

Reward and return semantics (source plus pure arithmetic):

- Optimized reward is first-success bonus + beta*(gamma*next_potential-current_coverage). The original upstream reward is retained for reporting and is not added to this objective. Success means coverage strictly >.95; its next potential is zero and its value bootstrap is disabled.
- Below success, zero coverage with no progress earns zero; flat positive coverage earns -(1-gamma)*beta*coverage. Small positive changes can still yield a negative local shaping reward. Timeout and buffer cutoffs retain the actual final coverage and final-observation value bootstrap.
- For m actual controls, block reward sums gamma**i*r[i], bootstrap uses gamma**m, and GAE continuation uses gamma**m*lambda once per plan. Success clears bootstrap/continuation; timeout/cutoff clear continuation only. Separate fragments prevent GAE crossing shape, episode or buffer boundaries.
- Potential shaping telescopes for a genuinely terminal trajectory. Negative local rewards or a negative successful-batch sum alone do not establish an incorrect ideal objective. Finite bootstrapping, sparse success and exploration remain possible practical causes.

Specific stationary-data observations:

- L batches [12]: 44 controls, zero recorded contacts/coverage change, constant coverage 0.394710322961, training reward -0.173672542103 (flat-coverage formula -0.173672542103).
- L batches [15, 16, 17]: 172 controls, zero recorded contacts/coverage change, constant coverage 0.372334574407, training reward -0.64041546798 (flat-coverage formula -0.64041546798).
- Batch 3 T contains bonus 1; its undiscounted training sum is -0.0736889281364, with shaping sum -1.07368892814. A negative batch sum does not mean the success bonus was absent.

Interpretation and hypotheses:

- High late mixed explained variance can reflect fitting stationary shaped values and between-shape offsets. Per-shape EV, advantage signs and the causal effect of those offsets are unobserved, so this remains a hypothesis.
- A near-zero actor loss at the fresh first pass is expected with identity likelihood ratios and mean-zero normalized advantages. Nonzero gradients and changed weights show that it does not mean no update. Small sampled conditional-transition KL is not a guarantee of task retention.
- Success is sparse and exposure is uneven. Training scenes change over time, so batch reward/coverage changes do not isolate policy learning.

Priority single-factor follow-up proposal (requires a new authorized budget):

- Compare fresh-base training with 2,048 versus 4,096 physical controls, keeping gamma, beta, lambda, sampler floor, learning rates, passes per buffer, frozen subset and the ordered scene schedule fixed. This tests whether doubling exposure is sufficient; it does not presume undertraining caused the observed result. Use the same fixed paired development cases to compare outcomes. The current v1 configuration cannot run 4,096 controls: a future configuration/protocol version and explicit authorization are required. No additional run is authorized or performed by this analysis.
- The configured gamma=.99 gives gamma**300=.0490409. Reward horizon, finite bootstrapping and exploration remain unresolved hypotheses; discount arithmetic alone identifies no cause and is not an alternative experiment recommendation.

Limits:

- Returns, raw/normalized advantages, old values and critic diagnostics are mixed T/L buffer statistics. Per-shape versions cannot be reconstructed from these scalar logs.
- Critic MSE and explained variance evaluate the same fixed buffer used for fitting, after each pass; they are not held-out estimates or task success rates.
- Each fresh batch is counted once. Repeated passes are correlated optimizer diagnostics, not additional controls or independent trajectories.
- Coverage deltas omit jumps across reset. Aggregate maxima and contact counts do not reconstruct trajectories or prove useful directed contacts.
- Three equal batch-index stages have different control counts, scene exposure and boundary-drain schedules; changes are descriptive, not causal learning curves.
- Per-step parameter L2 changes are update magnitudes. Their sum is not final net movement from the base; tensor weights are not loaded by this analyzer.
- This analysis uses prior checkpoint JSON metadata rather than re-verifying checkpoint tensor bytes. No model, simulator, sampler or optimizer is run.
