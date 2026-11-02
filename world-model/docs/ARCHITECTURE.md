# Model and protocol map

## Native visual predictor

Four RGB history frames and their aligned continuous actions condition the original GPC denoising network. The target is a 96×96 next RGB frame. Training uses one-step supervised examples; recursive evaluation inserts predicted images into subsequent histories. H1 training with four gradient-accumulation microbatches is not a four-step loss.

For the recent original-physics family, images are float RGB in [0,1], actions are `a/255.5 - 1`, and raw inference uses the recorded three-denoising-step sampler. Single-denoising reconstruction used by auxiliary losses is distinct from a full autoregressive sampling graph.

## Protocols that must remain distinct

| Protocol | Location / reference | Interpretation |
| --- | --- | --- |
| Official PushT / Diffusion Policy | `tdyssey/official.py`, `tdyssey/envs/official.py`, official manifests | Primary pretrained policy baseline |
| GPC reference environment/model | `tdyssey/gpc_world_model.py`, pinned GPC source | Separate geometry/data/reference setup |
| Early native RGB finite-force controller | `tdyssey/native_rgb_env.py` and early configurations | Material controller change; not upstream physics parity |
| Original kinematic/contact studies | `tdyssey/native_rgb_original_physics.py`, strict-original scripts and run sources | Later original-physics experiments; consult per-run frozen settings |
| Structured/custom visual models | `tdyssey/models/`, `tdyssey/world_model_training.py`, archived variants | Different model/interface; no native GPC result attributed to them |
| Direct-simulator DPPO | Archived milestone profiles and category 11 | Related policy experiments; not WM-only training |

State, contact and letter metadata used for sampling/audits are distinct from RGB/action model inputs. Any privileged renderer/pose supervision in early studies must remain disclosed. O/C/B were seen shapes in the recent fitting screens; these runs do not establish unseen-letter transfer.
