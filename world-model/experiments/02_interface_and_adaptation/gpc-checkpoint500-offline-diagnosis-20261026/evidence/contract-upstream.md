# Static GPC contract evidence

Scope: cached upstream `a96146e568826ef46134f73c22fdf59a675e7264` (also read from its `.git/HEAD`) and local source only. No Python execution/import, weights, held data, array computation or measurements were used. Read validation instructions (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/AGENTS.md`) and upstream-parent instructions (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/AGENTS.md`).

**No definite local formula, frame/action-index, parameter-selection or sampler-contract bug is established.** The loss matches the first-target (`t=5`) subcase of native phase-two supervision, not its full autoregressive training procedure. Source agreement does not establish scientific adequacy or explain the measured outcome by itself.

## Noise, conditioning and objective

Native independently draws conditioning then target sigma, each `clip(exp(-1.2+1.2*Z), .002,20)`. Draw order is conditioning sigma/offset/pixel noise, then target sigma/offset/pixel noise. For flattened history `H[B,12,96,96]` and target `x[B,3,96,96]`:

`H' = H + .1*Z_h[B,12,1,1] + sigma_cond*epsilon_h`

`x' = x + .1*Z_x[B,3,1,1] + sigma*epsilon_x`.

Offsets are per flattened channel, spatially shared, not multiplied by sigma. Native sigma/noise (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/diffusion/denoiser.py:57`), native order (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/diffusion/denoiser.py:118`), local NoisePlan (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:127`).

Set `s=sqrt(sigma²+.1²)`. Both use `c_in=1/sqrt(s²+.5²)`, `c_skip=.5²/(s²+.5²)`, `c_out=s*sqrt(c_skip)`, `c_noise=log(s)/4`, and `c_noise_cond=log(sigma_cond)/4` or0 for `None`. The inner network receives `H'/.5`, `c_in*x'`, four normalized actions and both noise embeddings. Both optimize `mean((f_theta-(x-c_skip*x')/c_out)²)`: equivalently unquantized reconstruction error weighted by `1/c_out²`, not clean-image/state/contact/ranking loss. Native conditioners (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/diffusion/denoiser.py:71`), native loss (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/diffusion/denoiser.py:134`), local loss (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:156`).

## Frame/action indexes and training length

`a_k` is the absolute workspace control taking `I_k` to `I_(k+1)`, not displacement or a96-pixel coordinate. Both use fixed bounds0/511 and `2*a/511-1` once. Native fixed statistics (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/train.py:63`), normalization (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/utils.py:97`).

| Path | Input images | Actions (half-open slices) | Target |
|---|---|---|---|
| Local training anchor `k` | `I[k-3],I[k-2],I[k-1],I[k]` | `a[k-3:k+1]` | `I[k+1]` |
| AR H1 anchor `t` | `I[t-3],I[t-2],I[t-1],I[t]` | `a[t-3:t+1]` | `I[t+1]` |
| AR H2 | `I[t-2],I[t-1],I[t],P1` | `a[t-2:t+2]` | `I[t+2]` |
| AR H4 | `I[t],P1,P2,P3` | `a[t:t+4]` | `I[t+4]` |
| TF H2/H4 | Same rolling indexes, earlier `Pj` replaced by real `I[t+j]` | Same as AR | Same horizon |

The fifth training image is target only. AR never inserts true future RGB; TF uses real future only through H-1, not the target H. Frozen AR records RNG entry/exit; adapted AR/TF restore corresponding entries and check exits. Local dataset (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation_data.py:237`), loss slicing (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:168`), actual rollout (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:377`).

Native config has16 images, four-frame model history, batch8 and300 epochs. At `i=0..11`, it uses `all_obs[:,i:i+4]`, `act[:,i:i+4]`, target `all_obs[:,i+4]`, then writes its no-grad quantized prediction into that image slot. Later histories contain predictions; average loss covers12 terms. Local t=5 retains only i=0; its unused tail cannot provide that exposure. `obs_horizon=2` is dataset padding, not the denoiser's history length. Native configuration (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/configs/train_phase_two_config.yml:6`), model configuration (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/train.py:115`), native rolling loop (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/diffusion/denoiser.py:104`).

## Images, official sampler and parameters

Native conversion intends float32 RGB[0,1], with ImageNet normalization commented out; historical stored dtype/domain membership is not proved by source alone. Local real images/targets get fixed T1 once; raw AR bypasses T1. RGB-MAE clipping is metric-only. Native transform (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/utils.py:193`), local real-input transform (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation_data.py:238`), raw feedback (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:419`).

Official inference uses three order1 Euler stages, rho7 schedule5 to.002 then0, no churn, `s_cond=0`. Initial `x=randn` is not multiplied by sigma_max. Each denoised estimate clamps/quantizes to256 levels spanning[-1,1]. These choices are inherited, not newly introduced local scaling bugs. Local constructs the native class and feeds raw outputs back, as upstream evaluation does. Sampler/Euler (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/diffusion_sampler.py:30`), schedule (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/diffusion_sampler.py:67`), quantization (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/denoiser.py:85`), upstream AR (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/gpc_rank_evaluation/eval_baseline.py:350`), local construction/wrapper (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:285`).

Native AdamW explicitly sets lr1e-4 and optimizes the denoiser. Local optimizes named `inner_model.*` parameters with lr1e-4, additionally clipping norm at1. Fourier frequencies are registered buffers, so this selection does not unfreeze them. Native starts phase-two from phase-one; local adapts phase-two with500 batch1 updates. Training uses train mode/differentiable model output; loss evaluation uses eval/no-grad; forecasting freezes parameters and uses native no-grad sampling. These are declared differences, not evidence of sampler-based training. Native optimizer (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/train.py:163`), Fourier buffers (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/world_model_train_phase_two/diffusion/blocks.py:78`), local selection/optimizer (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:194`), local update/clip (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:312`), local loss-eval mode (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:183`).

## Attribution limits

Removing11 generated-history terms is a definite, intentional exposure difference;
it cannot alone explain H1 degradation, whose history is real. Native training
also uses sampled noisy-history conditioning while fixed inference uses none
and embedding0. This inherited difference has no measured causal contribution
here and does not justify silently changing the sampler. The final diagnosis
report ranks at most two explanations for the observed measurement phenotype;
these contract facts are not additional established causes.

No production/docs file, threshold or execution authority is changed by this note.
