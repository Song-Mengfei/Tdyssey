# Official RGB to GPC: static compatibility audit

Base `994c6ff`, 2026-10-24. **This audit reads source, configuration, asset
metadata and existing reports only.** No environment, render, image decoder,
model forward, sampler, training, package installation or asset download was run.
Runtime code, weights and old ledgers are unchanged. The companion
[bounded proposal](GPC_INPUT_COMPATIBILITY_PROPOSAL.md) is not execution authority.

The current wrapper matches the documented tensor, coordinate and action-window
interfaces. There are demonstrated appearance and evaluation-path differences,
but the exact checkpoint's training-domain membership is unknown. Neither colors
nor the H1 failure pattern identify a unique cause or a required training method.

## Evidence versions and interpretation

Project file/line references below are at `994c6ff`. Upstream references are
immutable Git objects, read locally without fetching:

- **O**: official `real-stanford/diffusion_policy`, commit
  `5ba07ac6661db573af695b419a7947ecb704690f`, directory
  `diffusion_policy/env/pusht/`.
- **G**: `han20192019/gpc_code`, commit
  `a96146e568826ef46134f73c22fdf59a675e7264`.
- **R**: this repository. The versioned manifests are
  upstream.json (external historical reference: `../manifests/upstream.json`) and
  gpc_world_model.json (external historical reference: `../manifests/gpc_world_model.json`).

**Matched** means the checked code paths implement the same convention, not that
the released training data were correct. **Different** means an observed source
or artifact difference, not an established cause of model failure. **Unknown**
means the necessary checkpoint-bound provenance or experiment is absent.

## Complete interface contract

| Item | Checked contract, status and limitation | File/line evidence |
| --- | --- | --- |
| Channel order and image size | **Matched:** RGB, 96x96, CHW for the predictor; pygame RGB is transposed to HWC and resized with OpenCV. No BGR conversion or image-axis flip is indicated. Training also resizes via torchvision; exact resize parity of the original stored training images is not established. | O `pusht_env.py:191-219`, `pusht_image_env.py:13-26,37-44`; G `gpc_rank_evaluation/pusht_env.py:592-635`; G `world_model_train_phase_two/utils.py:193-209`; R `tdyssey/gpc_world_model.py:103-111` |
| Dtype/range | **Matched:** true inputs are float32 RGB [0,1]. Training reads `data/img` in [0,255], converts to uint8, resizes, then scales to float32. ImageNet normalization is commented out in this training path. Environment `pretrained=True` normalization is an optional path, not the dynamics default. | [O image observation:37-44](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/env/pusht/pusht_image_env.py#L37-L44); G `world_model_train_phase_two/utils.py:130-138,193-209`; G `gpc_rank_evaluation/pusht_env.py:1229,1259-1274`; `eval_baseline.py:235`; R `tdyssey/gpc_world_model.py:103-111` |
| Background, walls and draw order | White background and LightGray walls. Both renderer paths draw the goal before the object/pusher; visible target pixels may be occluded. Palette conversion cannot recover an invisible target by recoloring. | [O renderer:191-205](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/env/pusht/pusht_env.py#L191-L205), `:296-320`; G `gpc_rank_evaluation/pusht_env.py:594-621` |
| Object/goal palette | **Different:** official LightSlateGray object/LightGreen goal, versus the domain18 evaluation configuration's red `(165,28,48)`/black goal. Object border/fill colors are gray `(119,136,153)/(143,163,184)` versus red `(165,28,48)/(198,34,58)`. This is an evaluation-source comparison, not proof of exact training membership. | O `pusht_env.py:304-308,342-359`; [G domain18 YAML:1-12](https://github.com/han20192019/gpc_code/blob/a96146e568826ef46134f73c22fdf59a675e7264/gpc_rank_evaluation/domains_yaml/push_t.yml#L1-L12); R `tdyssey/appearance.py:3-5` |
| Stroke/pusher appearance | Shared 2-world-pixel object outline and 1.2x interior brightening. Shared RoyalBlue radius-15 pusher, with brightened radius-11 interior. These facts do not establish identical antialiasing after resizing. | O `pymunk_override.py:139-140,203-210,245-247`; O `pusht_env.py:305,323-328`; G `gpc_rank_evaluation/pusht_env.py:138-139,202-209,778,801` |
| Action-marker path | **Different:** this project's official `obs['image']` is marker-free. GPC rank evaluation feeds `env.render()` into dynamics history; that cache has a red action cross. Official image observations are converted before its cache marker is drawn. **Unknown:** whether the exact checkpoint's training images contained markers. Do not add an action marker to the proposed RGB-only transform. | O `pusht_image_env.py:19,37-56,60-66`; G `gpc_rank_evaluation/pusht_env.py:1288-1307`; [G rank evaluator:403-405](https://github.com/han20192019/gpc_code/blob/a96146e568826ef46134f73c22fdf59a675e7264/gpc_rank_evaluation/eval_baseline.py#L403-L405), `:336,353-364`; R `scripts/validate_frozen_wm.py:274,289` |
| Coordinates and orientation | **Matched convention:** world extent 512, x right, y down, positive angle clockwise on screen. The HWC transpose is not a physical flip. Existing RGB must not be translated/rotated to reconcile shape-local origins. | O `pymunk_override.py:53,232-235`; G `gpc_rank_evaluation/pusht_env.py:35,67-70`; R `tdyssey/geometry.py:40-43`, `tdyssey/envs/official.py:16-27` |
| T/L geometry and physical domain | T silhouettes share a 120x120 extent and 30-wide strokes, but official T local coordinates/COM differ from native GPC by local `(0,45)`. Official mass/inertia are 1/3000; native GPC 2/5700. Official L is height120, footwidth90, thickness30. These are physical/shape-domain differences; a palette map cannot remove them. | O `pusht_env.py:342-362` and legacy setup; G `gpc_rank_evaluation/pusht_env.py:855-880`; R `tdyssey/envs/official.py:12-27,80-109`; R `docs/WORLD_MODEL.md:223-233` |
| Goal convention | **Different:** official fixed T target `[256,256,pi/4]`; L goal center is aligned by the documented COM rule. GPC evaluation has configurable/randomized native goals. Goal pose is not a predictor input; visible goal RGB is part of the image. Fixed goal geometry enters the decoder only after pose fitting. | O `pusht_env.py:307-308`; G `gpc_rank_evaluation/pusht_env.py:305-325,723`; R `tdyssey/envs/official.py:103-109`; R `tdyssey/wm_image_measurement_v2.py:268-290` |
| History length | **Matched:** four RGB observations, not the policy's two-frame observation history. The training config's `obs_horizon=2` controls sequence/padding construction; the denoiser explicitly sets four conditioning frames and four actions. | [G model construction:115-125](https://github.com/han20192019/gpc_code/blob/a96146e568826ef46134f73c22fdf59a675e7264/world_model_train_phase_two/train.py#L115-L125); G `gpc_rank_evaluation/diffusion/inner_model.py:35-45,62`; R `tdyssey/gpc_world_model.py:106-115` |
| Frame/action alignment | **Matched checked paths:** four observations bracket three already executed controls; append the proposed next control to obtain the fourth action. The recorder stores the initial image, then each post-control image. No static off-by-one error was found. Training data's original timestamp quality remains unknown. | R `scripts/validate_frozen_wm.py:274-290`; R `tdyssey/gpc_world_model.py:55-64,114-127`; G `gpc_rank_evaluation/diffusion/denoiser.py:118-136` |
| Action meaning and scale | **Matched:** absolute 512-workspace pixel position target, not displacement, actual pusher position, or a 96-pixel image coordinate. Current `a/511*2-1` matches training fixed min0/max511. Do not use the official policy normalizer here. | O `pusht_env.py:109-122`; R `tdyssey/frozen_wm_protocol.py:58-63`; R `tdyssey/gpc_world_model.py:104-111,120-122`; G `world_model_train_phase_two/train.py:63`, `utils.py:97-106` |
| Time | **Matched implementation:** 10Hz control/100Hz physics, ten substeps per control, PD gains100/20. This is not proof of the exact training dataset's collection cadence. | O `pusht_env.py:29,43-46,109-122`; G `gpc_rank_evaluation/pusht_env.py:232,245-250`; R `manifests/gpc_world_model.json:88-89` |
| Sampler and feedback | Same frozen three-step order1 Euler sampler, sigma5 to .002 then0, no churn/conditional-input noise. Initial `x=randn`, without multiplication by sigma5, is actual upstream behavior; its adequacy is untested here. Raw generated RGB is appended directly to AR history. Internal `obs/sigma_data` and clamp/quantization do not imply external [-1,1] inputs. | [G sampler:29-67](https://github.com/han20192019/gpc_code/blob/a96146e568826ef46134f73c22fdf59a675e7264/gpc_rank_evaluation/diffusion/diffusion_sampler.py#L29-L67); G `diffusion/denoiser.py:80-89`; `eval_baseline.py:353-364`; R `tdyssey/gpc_world_model.py:118-127`; R `manifests/gpc_world_model.json:63-70` |

The final sampler reference to `diffusion/denoiser.py` is under
`gpc_rank_evaluation/`. A future investigation of initial-noise amplitude would
be a different factor with a separate protocol; it is not silently bundled with
palette conversion or an external image-range change.

For temporal clarity, define current history as
`[o(t-3), o(t-2), o(t-1), o(t)]`, past controls
`[a(t-3), a(t-2), a(t-1)]`, and candidate `a(t)`.
Each `a(k)` takes `o(k)` to `o(k+1)` over one 0.1s control. H1 predicts `o(t+1)`.
The checked recorder follows this convention. The first future object motion is
zero in the saved branches, so their H1 failures cannot be explained simply by
a large true first-step object displacement.

## Which checkpoint and which training domain?

The exact asset is HF `han2019/gpc_checkpoints`, immutable revision
`c16a467ec176af19bba5136564a81d8d61d72af9`, path
`world_model_checkpoint/phase_two_checkpoint/denoiser.pth`, 35,043,916 bytes,
SHA-256 `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
See R `manifests/gpc_world_model.json:27-34`. The wrapper validates actual
size/hash before loading and loads the complete state strictly
(R `tdyssey/gpc_world_model.py:74-89`). The v1 diagnostic report's
`binding.checkpoint_sha256` records this same asset.

The upstream-tracked csgo file has the same size but hash
`15126113b7490fa976290c62ad4c7a92734bc898664522706c67a4aa6bff47b2`.
Its pinned Git blob is `7b0d656212dd80a8cb9274c3115b95cff4176bf7`. The previous
tensor audit found 296/298 entries different; this audit did not reload tensors.
Both on-disk hashes still match the prior audit. Filename, size and network
architecture cannot establish interchangeable weights, training stage or domain.
The csgo asset's exact training provenance remains unknown.

Training-source evidence is narrower than a checkpoint-bound training record:

- G `world_model_train_phase_one/configs/train_phase_one_config.yml:1,7` and
  `world_model_train_phase_two/configs/train_phase_two_config.yml:1,7` point to
  `../world_model_data/dataset_domain/all_data` and `num_train_demos:500`.
  The training script iterates every `.zarr` in that
  directory and concatenates the datasets, not just domain18
  (G `world_model_train_phase_two/train.py:63-90`). The dataset takes at most the
  first 500 episodes per file (`utils.py:124-140`).
- Domain18 red T/black target is an **evaluation configuration**. The actual
  files present during checkpoint training, source renderer, shape/palette
  membership, seeds, selected epoch and complete logs are unavailable here.
  The weights-only asset does not carry that provenance.
- Consequently, neither "trained only on red T/domain18" nor "official gray/
  green or L was absent from training" is established. Source release descriptions
  of training stages and data schemas do not close this gap. No dataset chunks
  were downloaded to infer a distribution from colors.

## Historical source-palette probe: same weights, privileged rendering

The published R
`palette-T-summary.json:139-143` (external historical reference: `media/world-model-2026-10-19/palette-T-summary.json`)
records checkpoint `4084ed...`, with exact source hashes for the probe,
`appearance.py` and `gpc_world_model.py`. All three hashes match local historical
commit `41aade4`. That historical wrapper checks the actual asset's size/hash
before load (`tdyssey/gpc_world_model.py:72-74` at that commit). The published
`pretrained-T-summary.json:40-47` also records the exact HF asset. This is strong
evidence that the old probes used the **same phase-two weights** as the current
frozen diagnostic, not the different csgo weights. The complete old raw reports
are not present at the checked local paths; the auditable chain is the published
summary plus matching historical source, not a newly reproduced experiment.

The intervention reads `env.block.shapes`, changes object/goal drawing colors,
calls `env._render_frame`, and restores colors/screen in `finally`
(R [`tdyssey/appearance.py:8-30`](../../../../experiments/03_native_rgb_supervised/native-rgb-supervised-20261029/source/r2c1/source-snapshot/repo/tdyssey/appearance.py#L8-L30)). It needs the
simulator's shapes and renderer. The probe verifies physical state, render cache
and original policy observations are preserved (`scripts/probe_gpc_appearance.py:
103-120`). Official true futures are collected once and shared across views/
noise; `model.predict` receives image histories, actions and seed (`:185-219`).
No direct future truth pose/mask input to prediction was found. Nevertheless,
this is **privileged auxiliary rendering**, not an RGB-only input adapter.

Its six development scenes support appearance sensitivity: matched official
contact object-mask IoU .73144 to .86106, target recall .03314 to .95403. These
are image proxies, not rigid-state/reward validation. Left-push object IoU becomes
worse (.451-.456 versus .887-.904), with missed true rotation about -9.84 degrees.
See R [`docs/WORLD_MODEL.md:304-341`](WORLD_MODEL.md#matched-palette-experiment)
and published palette summary. It did not establish a uniform motion benefit,
strict success precision, a deployable color adapter, or L transfer.

## What H1 failure and later recovery do, and do not, show

At the frozen v2 measurement thresholds, object validity is L H1 0/141 available,
T H1 0/48; H4 L 4/141, T 34/48. Planned denominators are L144/T48, including
three missing L rows at each horizon. Pusher validity remains much higher.
The v2 soft full-shape metric also rejects six T H1 cases accepted by v1.
See [the measured results](WM_INTERFACE_V2_RESULTS.md#generated-images-and-attribution).

Several explanations remain compatible with these facts:

| Hypothesis | Current evidence | What remains unresolved |
| --- | --- | --- |
| Real-input visual mismatch; later history moves toward generated style | Official input differs from domain18 evaluation appearance; raw AR gradually replaces four real frames; old privileged palette probe had mixed benefit. | Whether one RGB-only transform restores early shape under the exact weights. No causal conclusion follows from red predictions alone. |
| AR error/feedback distribution | H4 is conditioned on three generated frames and one original-history frame; only after appending its output are all four frames generated. | Direction of causality: later recovery may be style stabilization, not better action dynamics. Matched-noise teacher forcing is needed to isolate conditioning. |
| Geometry/shape identity failure | Some saved L H1 forecasts are blobs or lose an arm. T also fails H1, so L-only geometry transfer is insufficient as an explanation. | How much is genuinely generated geometry versus the image estimator's response to blur/appearance. A forced rigid candidate is not a valid state. |
| Measurement failure | V1 binary processing lost real RGB information; v2 real RGB is measurable but generated styles are not independently calibrated. | Full pose ambiguity, valid-state coverage error and style-dependent rejection remain imperfectly characterized. |
| Action/sampler/training mismatch | Checked action windows/range and timing agree with upstream; initial sampler noise amplitude is upstream behavior. | Original dataset timing, training-domain membership, and whether any remaining source convention matters causally. |

Thus the next work is a bounded factor-isolation proposal, not an instruction to
recolor inputs as a proven fix or to begin fine-tuning. The proposal states which
conditional outcomes would justify a limited compatibility claim, a measurement
revision, or a separately reviewed weight-adaptation study.
