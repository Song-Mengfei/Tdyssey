# Static train-to-manual contract review

Scope: source and existing JSON/text only. No Python, tests, arrays, weights,
model, simulator, native/resource query, or image decoding was run by this audit.
The starting identity is `start.json` (HEAD `cbb87ef20dbfe7781762c36290a865df500b2d55`,
twelve already dirty files). Manual source was being repaired concurrently;
the line references below identify the unchanged inference/session bodies as
read during this review, not a claim that today's file is the historical log's
exact source. The frozen-start manual module/script hashes are respectively
`5bd2b013828460d315a2a6698363c2cae7c2881e60e43b1c392b7c7ccfe848e9`
and `01eff46eca41231a42fca807eced9e7ea47b2eaa4ef9591b915ada84e24cfcd9`.

## Findings that the source/log actually establish

No definite local off-by-one, action-coordinate/scaling, repeated-T1, wrong-role
checkpoint, or train/eval-mode bug was found in the inspected inference path.
This is a static finding, not evidence that physical prediction is reliable.

1. The cited target `(410.6666666666667, 240.0)` is **adapted T / H4 / seed5478**:
   `events.jsonl:198`; `:199` names the selected checkpoint SHA
   `0bfd1a944c8fb01555a7ad2d01aab6f526a4805e2bb5239a96703d356c5741d3`;
   `:200` records four executed controls. It is not a frozen-model request.
2. Frozen L was selected at `events.jsonl:352` after adapted L execution at
   `:351`, without a reset. `ManualSession.configure` changes the role and
   invalidates previews, but retains physical state and histories
   (`tdyssey/manual_world_model.py:234-243`). Thus those adjacent role segments
   are not a matched-input checkpoint comparison. Frozen T has its own fresh
   reset at log `:371`, followed by H4 request `:372` and execution `:374`.
3. Prediction requests are not the historical physical action stream. For
   example log `:462/464/466/468/470` each has a following `predicted` event but
   no `executed` event before the next request; request `:472` executes at
   `:474`. Replaying every prediction would produce a different trajectory.
   The read-time script auto-executes at `scripts/manual_world_model.py:171-178`;
   that current behavior must not be projected backward onto this old log.

## Concrete time and representation contract

Let `o_t` be the real observation before control `a_t`, and `o_(t+1)` its result.
The collector saves the initial observation, then records `(before,a_t,after)`
and the new observation (`tdyssey/official_collection.py:124-144`). Dataset
anchor `t` supplies `[o_(t-3),o_(t-2),o_(t-1),o_t,o_(t+1)]` and
`[a_(t-3),a_(t-2),a_(t-1),a_t]`
(`tdyssey/gpc_adaptation_data.py:237-243`). The loss uses the first four images
as history and the fifth as its sole target, without another action conversion
(`tdyssey/gpc_supervised_adaptation.py:109-124,168-175`). The fixed schedule
selects dataset indices, not a shifted time axis
(`tdyssey/gpc_training_schedule.py:238-243`;
`tdyssey/gpc_adaptation.py:307-319`).

The manual session begins with one observation and three actual hold controls,
then maintains four real observations/three intervening actions
(`tdyssey/manual_world_model.py:224-231,252-263,277-284`). H1 appends the chosen
absolute target as the fourth action. H4 shifts both four-frame context and
four-action window, repeating that target, while real execution likewise holds
the same target for four controls (`:286-307,369-384`). A preview alone does not
step that session. This matches the above time indices; four identical initial
frames with invented actions are not used.

Controls are absolute positions in the 512-unit world, not velocities or
96-pixel displacements. The official PD controller uses `(action-agent.position)`
for ten 0.01-second physics substeps per 0.1-second control
(`../Tdyssey/third_party/diffusion_policy/diffusion_policy/env/pusht/pusht_env.py:41-46,109-122`).
The dataset checks `control_dt=.1` (`gpc_adaptation_data.py:214`), and both manual
and dataset apply `a/511*2-1` once (`manual_world_model.py:376-377`;
`gpc_adaptation_data.py:240-243`). Upstream phase-two action bounds are likewise
0 and511 (`../Tdyssey/third_party/gpc_code/world_model_train_phase_two/train.py:63`).
The click scales panel coordinates to this world directly
(`scripts/manual_world_model.py:29-33`); source rendering uses downward-positive
y and HWC RGB followed by CHW float32/255, without an RGB/BGR swap
(`.../pymunk_override.py:53`; `.../pusht_env.py:216-219`;
`.../pusht_image_env.py:37-44`). The T origin/center-of-mass distinction does not
change this pusher action coordinate system (`tdyssey/envs/official.py:105-118`).

Dataset T1 acts once on all five real images; manual T1 acts once on each real
history frame. AR predictions bypass T1 (`gpc_adaptation_data.py:238-239`;
`manual_world_model.py:369,382-384`; fixed rule
`tdyssey/wm_compat_transform.py:58-60`). Showing the current real frame through
T1 does not mutate the stored real history. Display clipping is separate
(`scripts/manual_world_model.py:201-205,223-225`). The manual descriptive MAE is
unclipped, whereas the earlier evaluation RGB-MAE clips raw predictions
(`manual_world_model.py:307`; `tdyssey/gpc_adaptation.py:419`); these two reported
MAEs are therefore not interchangeable, and neither changes sampling.

## Objective, checkpoint and feedback limits

The implementation trains all `inner_model.*` parameters, in train mode, on
one next-frame residual MSE (`gpc_supervised_adaptation.py:29,194-210,268-274`;
`gpc_adaptation.py:312`). Its single-target loss matches the native first
iteration (`../Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/denoiser.py:117-141`).
It does not train a multi-step self-generated-feedback objective. Clean local
training images are [0,1]; upstream phase-two preprocessing also specifies
[0,1] (`.../world_model_train_phase_two/utils.py:196-209`).

The pinned native wrapper clamps/quantizes to [-1,1] internally
(`.../gpc_rank_evaluation/diffusion/denoiser.py:85-96`), and the native sampler
returns its final Euler state (`.../diffusion_sampler.py:50-64`). Manual H4 feeds
that raw returned RGB forward; original upstream rollout does too
(`.../gpc_rank_evaluation/eval_baseline.py:352-364`). The resulting difference
between real [0,1] histories and possible signed generated histories is a
provable inherited convention, not proof of a newly introduced local bug or
the cause of this target's failure. Training history noise and clean native
inference conditioning differ as well (`gpc_supervised_adaptation.py:133-137,169-172`;
native `diffusion_sampler.py:45-50`, `denoiser.py:77`). No causal conclusion or
alternative conditioning experiment follows from that source observation.

Manual role loading resolves the fixed initial 35,043,916-byte/`4084...1892`
file versus selected 105,134,565-byte/`0bfd...41d3` container; only model state is
extracted, source logical update500 is checked, optimizer/RNG is not restored
(`tdyssey/gpc_evaluation_assets.py:14-34,70-95`). Strict keys/shapes/dtypes and
eval/frozen mode are applied by the factory and predictor
(`gpc_supervised_adaptation.py:260-268`; `manual_world_model.py:348-357`). No
manual model-only loading or role swap defect is established here. I did not
open either weight file or independently verify its content in this task.

## Reconstructable sequence and missing exact-state evidence

For the adapted target at log198, replay the **executed** prefix after reset182:

| Request / execute lines | Target, held for four controls |
|---|---|
|183 /185|242.66666666666666,120|
|186 /188|145.33333333333334,80|
|189 /191|368,84|
|192 /194|410.6666666666667,241.33333333333334|
|195 /197|209.33333333333334,253.33333333333334|
|198 /200|410.6666666666667,240|

The source reset uses seed5478, the fixed T body pose `(256,240,0)`, pusher
`(256,405,0)`, and three hold controls (`manual_world_model.py:180-194,224-231`).
This defines an intended source replay, not a verified bit-identical restoration
of the historical request. The log does not seal initial/before/after full
Snapshots, the four real input frames, input/action hashes, per-request model
input RNG states, request IDs, or the source/dependency identities that produced
those historical inputs. Its request events contain role/shape/horizon/target/
seed, and execution events contain aggregate results
(`scripts/manual_world_model.py:121-128`). The manual output directory has only
`events.jsonl`; no captured request-state/array bundle is present. A later
deterministic reconstruction must be labelled as such. New capture helpers
currently being added cannot retroactively fill these missing historical bytes.

The exact failing historical state, physical contact geometry/velocity, raw
input arrays, raw prediction appearance and numerical cause remain unverified
by this static review. No new model or physical action was taken to fill them.
