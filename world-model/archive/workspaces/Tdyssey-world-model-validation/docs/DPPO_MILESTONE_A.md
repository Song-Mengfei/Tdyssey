# Direct-simulator DPPO milestone A

Implemented 2026-10-23 from fetched `origin/main`
`41d99968dc64244f2381a148e74723261db92328`, on `codex/dppo-milestone-a`.
This supplies sampling, likelihoods, rewards and returns, without an online
trainer, critic, real actor optimizer update or learning curves. Simulator reward
labels are privileged supervision; the initial policy retains expert T pretraining.

## Actor API and probability semantics

`tdyssey/policy_rl.py` wraps one loaded `OfficialDiffusionPolicy`. Shared
`observation_tensors(history)` preserves two 96x96 RGB frames, pusher positions,
original normalizers and the official 84x84 center crop. `DenoisingPolicy.train()`
keeps the upstream network in eval; eval still permits selected weight gradients.
No supervised `compute_loss` or third-party source modification is involved.

```python
from tdyssey.official import OfficialDiffusionPolicy
from tdyssey.policy_rl import DenoisingPolicy, SamplerConfig, transition_ppo_loss

official = OfficialDiffusionPolicy(source_root, checkpoint, device="cuda")
actor = DenoisingPolicy(official, subset="output", config=SamplerConfig(std_floor=0.01))
observations = official.observation_tensors(history)
record = actor.sample(observations, generator=generator, actor_version="ema-initial")
steps = [49]  # Recompute bounded subsets, not 100 retained real-network graphs.
new = actor.recompute_log_probs(record, step_indices=steps)
loss = transition_ppo_loss(new, record.old_log_prob[:, steps], advantages)
```

The wrapper shares the official network; the original object is not an independent
frozen teacher after wrapping. `ParameterSelection` reports actual names/counts:

| Subset | Exact prefixes | Parameters / tensors |
| --- | --- | --- |
| `output` | `model.final_conv.` | 1,313,282 / 6 |
| `last_up_and_output` | `model.up_modules.1.`, `model.final_conv.` | 13,127,170 / 30 |

Everything else, including encoder and normalizer, freezes. The denoiser has
251,511,938 parameters. Weight subsets are separate from denoising-time subsets:
one network participates at all 100 times, without a GPU old-actor copy.

`SampledChain` retains detached exact conditioning, normalized chain
`[B,101,16,2]`, actual timesteps 99..0, old means `[B,100,16,2]`, stds `[100]`,
old log probabilities `[B,100]`, sampler version/floor, scheduler config,
caller actor version, denormalized targets `[B,16,2]` and actions `[B,8,2]`.
Execution uses indices 1:9. Recomputing probabilities detaches both endpoints,
encodes under `no_grad`, then differentiates a fresh conditional mean only.
Caller actor versions are provenance labels, not weight hashes.

Each training transition is Gaussian with the locked diffusers 0.31.0 DDPM
epsilon mean, coefficients and predicted-x0 clipping. Its std is
`max(sqrt(posterior_variance), std_floor)` at every time, including t=0.
The normalized-action floor .01 is an explicit untuned proposal. Latents and
noise receive no extra clamp. Physical controller bounds on pixel targets are
a separate execution mapping, outside this Gaussian density.

`mode="baseline"` delegates draws to the actual scheduler, preserving no noise
and no RNG draw at t=0. Its `old_log_prob=None` reflects that deterministic final
transition; baseline records cannot feed training ratios. Future experiments
must separate the sampler change from learning.

`gaussian_log_prob` sums all 16x2 coordinates. `transition_ppo_loss` forms a ratio
per transition and averages sample/time entries, never multiplying 100 ratios.
This joint-density objective differs from the public DPPO implementation's
aggregation/clipping heuristics; it is not a line-by-line reproduction.
Denoising weights are `discount**t` (default 1), separate from physical gamma.
For unequal microbatches accumulate each subset's `reduction="sum"` divided by
the complete batch transition count. Ratio/surrogate overflow is rejected instead
of silently changing densities. No optimizer or regularizer is supplied.

## Physical rewards and returns

`tdyssey/policy_returns.py` is torch-independent. `training_reward` retains raw
current/next coverage and upstream reward, and computes first-success bonus 1 plus
`beta*(gamma*next_potential-current_coverage)`. Success is strictly coverage>.95;
true success sets next potential to zero. Negative shaping remains. The caller
maintains `success_already_seen` per episode and normally stops at first success.
Defaults gamma=.99, beta=1 are proposed settings, not tuned results.

An `ActionBlock` contains rewards for an actually executed nonempty prefix,
unique episode ID, start value and **pre-reset final observation** value.
`plan_gae` processes one chronological rollout stream; call separately for each
parallel environment. It rejects invalid episode joins and requires explicit
tail handling:

| Boundary | Bootstrap b | GAE continuation c |
| --- | --- | --- |
| Normal plan | 1 | 1 |
| True success | 0 | 0 |
| Episode timeout/reset | 1 | 0 |
| Buffer tail | 1 | 0 |

For m executed controls, `R=sum_i gamma**i*r_i`,
`delta=R+gamma**m*b*V(final_observation)-V(start)` and
`A=delta+gamma**m*lambda*c*A_next`. Lambda=.95 applies once per plan boundary,
not per control. Denoising iterations never count as physical interactions.
Numeric helpers cannot verify which image the critic saw; milestone B must
retain/value final observations before reset.

## Reproduction and results

Use an existing interpreter/assets; no sync, installation or download is needed.
PowerShell from the isolated checkout, substituting existing asset paths:

```powershell
$python = 'path/to/existing/.venv/Scripts/python.exe'
New-Item -ItemType Directory -Force artifacts/test-temp | Out-Null
$env:TEMP = (Resolve-Path artifacts/test-temp).Path
$env:TMP = $env:TEMP
$env:SDL_VIDEODRIVER = 'dummy'
$env:SDL_AUDIODRIVER = 'dummy'
$env:PYGAME_HIDE_SUPPORT_PROMPT = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:HF_HOME = Join-Path (Get-Location) 'artifacts/hf-cache'
$env:HF_HUB_OFFLINE = '1'
& $python -S -B scripts/check_repo.py
& $python -S -B -m unittest discover -s tests -p 'test_*.py'
$env:TDYSSEY_OFFICIAL_ROOT = 'path/to/existing/third_party/diffusion_policy'
& $python -B -m unittest discover -s tests -p 'test_*.py'
$env:TDYSSEY_OFFICIAL_CHECKPOINT = 'path/to/existing/checkpoints/official-dp/ema_policy.pt'
# Omit this override for the manifest-pinned compact file. For a separately
# recorded local artifact, supply its known digest; see the limitation below.
$env:TDYSSEY_OFFICIAL_CHECKPOINT_SHA256 = 'recorded-local-sha256'
& $python -B -m unittest discover -s tests -p test_official_runtime.py -v
$env:TDYSSEY_RL_CHECKPOINT_CHECK = '1'
& $python -B -m unittest discover -s tests -p test_policy_rl_checkpoint_runtime.py -v
```

Observed Windows AMD64 / Python 3.12.3 / torch 2.5.1+cu124 / diffusers 0.31.0 /
RTX 3060 Ti, four CPU threads:

| Check | Result |
| --- | --- |
| Math/tiny-model likelihood and gradients | 12 passed |
| Independent reward/block/GAE examples | 9 passed |
| No-site-packages discovery (`-S`) | 71 passed / 58 skipped, 129 total |
| Installed discovery, official source and no checkpoint flags | 100 passed / 29 skipped, 129 total; 5.616s |
| Existing official runtime, local checkpoint | 3 passed / 0 skipped; 15.444s |
| Bounded real-network RL check | 1 passed / 0 skipped; 20.117s |
| Static repo / offline lock / Windows doctor | passed; 119 locked entries; doctor tests 3 passed |

Checks compare all 100 DDPM means/stds to the installed scheduler, joint densities
to `Normal` and hand arithmetic, PPO positive/negative clip and zero advantage
to controlled toy gradients. Fixed-zero-mean t=0 tests use 8,192 residuals
(seed 718, mean tolerance .04, variance tolerance .05) and detect latent clamping.
Real-network baseline actions/chain/next RNG match unchanged official inference
bitwise. Initial same-device FP32 old/new log-prob maximum absolute error is
**0** (test tolerance 1e-4 for 32-coordinate accumulation). Both real subsets have
finite gradients, frozen weights have no gradients, repeated encoding preserves
RNG/fixed crop, and unequal sample/time microbatch gradients match the full batch.
Peak CUDA allocated memory was **1,109.4 MiB**, not a trainer memory estimate.

The first installed-suite attempt hit four `os.replace` errors in the sandbox's
default Windows temporary directory; `artifacts/test-temp` passed without changing
production IO. Existing simulator/collection regression tests generated temporary
fixtures; no production training dataset or on-policy DPPO rollout batch was
collected. GPC/world-model assets and Linux runtime were not enabled;
skips are not passes. No real actor optimizer update or learning experiment ran.

The existing 1,095,708,090-byte compact file has SHA-256
`37d661ce6a487905fa4ff81ab7d65bf1b56b1c06adf480a961f65338fa0e38a0`, different
from the manifest compact digest. Config/source-provenance fields match and the
complete state loads strictly. Released-EMA tensor equality was not independently
established; we cannot conclude that reserialization alone caused the difference.
Default loading keeps its original digest gate; `checkpoint_sha256=` explicitly
pins the local test artifact. Parity compares two paths using that same artifact.
All actual parameter names/counts and machine-specific evidence are saved in
ignored `artifacts/milestone-a/validation.json`, not a shared artifact.

Windows carryover is limited to AMD64 wheels/markers, Linux-only egl-probe,
UTF-8/platform diagnostics, command-scoped Git ownership and published-JSON
newline protection. Original package versions stay pinned; Windows resolution
adds colorama 0.4.6. `uv pip check` retains the documented robomimic missing-egl-probe
metadata exception. No reranking/action-search/contact-sequence code was imported.
See [environment details](ENVIRONMENT.md).

Milestone B needs online collection preserving final observations, critic,
on-policy buffer, weighted minibatch optimizer steps, reloadable trained
checkpoints and separately budgeted physical learning/evaluation curves.
