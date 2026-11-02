# Static foundation contract audit

This is a source audit, not an experiment or a quality result. No Python,
NumPy computation, Torch/SDK import, weight read, image decode, simulator or
native/resource probe was executed by this agent. The paused conditioning
changes were preserved. New tests are written but unexecuted.

## Finding

No definite manual sampler, selected-weight loading, action-normalization or
history-index discrepancy against pinned GPC was found. In particular, the
unscaled initial Gaussian and `sigma_cond=None` are the actual upstream default;
their unusual appearance does not establish a local implementation error.

- The local native factory returns the upstream `DiffusionSampler` with manifest
  settings: three steps, min .002, max5, rho7, Euler order1, churn0, conditioning
  noise0. It does not implement its own denoising formula. See
  make_sampler (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:285`),
  NativeSampler.sample (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_adaptation.py:490`),
  and upstream sample (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/diffusion_sampler.py:30`).
- Upstream initializes `x=randn`, passes clean flattened history with
  `sigma_cond=None` when s_cond0, and updates `x += ((x-denoised)/sigma_hat) *
  (next_sigma-sigma_hat)`. The native denoiser rescales history by sigma_data .5
  and quantizes its wrapped estimate to256 levels in [-1,1]; the local wrapper
  returns raw values, including negative values. These are inherited contracts,
  not permission to rescale predictions or change the sampler.
- The manual predictor applies fixed T1 once to each of four real observations.
  H1 uses the three executed actions plus the proposed absolute pixel target;
  H2..H4 shift that window and append the same proposed target. Actions use
  float32 `a/511*2-1`; generated raw frames enter later histories without T1,
  clipping or real future substitution. This matches the rank evaluator's
  append-action / last-four-images / last-four-actions sequence and its dynamics
  action range[0,511]. See
  manual predictor (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/manual_world_model.py`),
  action_windows (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_world_model.py:55`),
  upstream action statistics (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/gpc_rank_evaluation/eval_baseline.py:92`),
  upstream conditioning (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey/third_party/gpc_code/gpc_rank_evaluation/eval_baseline.py:350`).
- Loading independently verifies the frozen/adapted files and source500 metadata,
  extracts only model state for construction, checks keys/shapes/dtypes/finite
  tensors, then uses strict state loading and eval/no-grad mode. Adapted optimizer,
  RNG and counts are not restored. Closing the previous model precedes loading
  the next role. See
  model-only loader (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_evaluation_assets.py:52`)
  and factory (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/tdyssey/gpc_supervised_adaptation.py:224`).

One separate accounting limitation is definite: the GUI's accepted-forecast
counter is updated in `ManualSession.accept`, after sampling and final-frame
measurement return. A failed decoder, partial rollout or stale-result rejection
can therefore leave actual model work absent from that counter. Physical controls
are likewise counted after `env.step` returns. This is not evidence explaining
image degradation. A bounded diagnostic must retain pre-call reservations and
returned-call counts rather than treating GUI accepted-result counters as spent
work.

## Bounded real-case proposal; not execution authority

Use the existing single-process external wrapper, with no new Job/heartbeat
system. At most two serial runs (before/after), each300s including setup/export/
cleanup, aggregate600s, two CPU/library threads. Preserve original resource
floors: startup RAM4GiB/commit6GiB/GPU4GiB/disk2GiB; runtime RAM1.5GiB/commit3GiB/
GPU1.5GiB, process WS2GiB/private3GiB and CUDA allocated512MiB/reserved1GiB.
Any read/IO/resource/cap error stops, with no retry or replacement run.

Freeze eight cases per run: two shapes x two weight roles x horizons1/4. Group
all frozen cases before adapted cases so there are exactly two serial loads per
run. Use the constructor's initial reset as case1, then seven resets, and count
every reset's three real initialization holds. With full horizons this gives20
forecasts and44 controls per run (24 holds+20 compared controls), not merely20
controls. Record every returned prediction H1..H4, every actual observation and
early physical termination. A reset includes its one source settle substep;
restore itself does not advance physics. Aggregate successful planned work is40
forecasts/120 Euler,88 controls/880 control substeps,16 reset/settle substeps,
within the authorized ceilings96/288/96/960/16. Preserve reserved, returned and
completed values separately on any partial failure. Targets, seeds, scene
snapshots and source/checkpoint identities must be frozen before the first call;
GUI process exit must be verified by the owner. No after-run is justified solely
by an unverified learning hypothesis.

## Untested checks supplied

`tests/test_gpc_foundation_contract.py` exercises the actual manual prediction,
manual loading and NativeSampler forwarding helpers using tiny synthetic NumPy
arrays and ordinary Torch markers. It checks raw feedback, exact action windows,
one transform per real frame, selected-role state and serial release/eval mode.
It deliberately makes no native parity, CUDA RNG or learned accuracy claim.
