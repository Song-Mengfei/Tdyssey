# Official original-colour GPC alignment

The official T code path is now connected to the teammate implementation at
`df99f804cb999f851b0f77d6981e3dcdb6733df2`, read from the existing remote-review
checkout. It uses official gray-object/green-goal RGB directly, with no T1,
recoloring, inverse transform or background subtraction. This is the GPC
denoiser/world model, not a new architecture or DP-policy update. No actual
model, weight tensor, SDK/CUDA, prediction, training or physical run was started.

## Entrypoints

Source-only help and asset inspection:

```bash
python3 -B scripts/visualize_world_model.py --help
python3 -B scripts/visualize_world_model.py --check-assets
python3 -B scripts/train_official_contact.py --help
```

The asset check currently reports missing and exits2. A real official viewer
requires an explicit trained file, its current SHA256 and the adjacent
`denoiser.json` (or `--sidecar`). With those available, use an existing environment:

```bash
sh start_official_demo.sh --gpc-root /src/gpc --source-root /src/diffusion_policy \
  --checkpoint /models/official-T/denoiser.pt --checkpoint-sha256 CURRENT_SHA256 \
  --checkpoint-source "official T run and revision"
```

Windows uses the same arguments with `start_official_demo.cmd`; select an active
environment or set `TDYSSEY_PYTHON` explicitly. There is no sibling Windows-venv
fallback in this new entry. Linux also accepts `TDYSSEY_PYTHON`,
`TDYSSEY_GPC_ROOT`, `TDYSSEY_OFFICIAL_SOURCE` and `TDYSSEY_WM_OUTPUT`; source,
checkpoint, sidecar and output paths have explicit CLI options.

Optional `--initial-checkpoint` enables the released initial control and C
comparison. Initial-only use requires explicit `--checkpoint-kind initial`; it
is labelled released initial, never teammate-trained. An unspecified optional
role is disabled, not searched or silently substituted. Known legacy T1
checkpoint500 and the initial hash are rejected when labelled official-trained.
The English two panels retain click→predict→execute once, C without another
push, S complete cases and recording. Official mode is H1-only; legacy
`start_manual_demo.cmd` retains H1/H4 and T1. L in the new viewer is explicitly
exploratory with T-trained weights, not a new L training route.

Training is implemented but default CLI execution is a read-only parameter plan:

```bash
python3 -B scripts/train_official_contact.py --datasets /data/official-T-contact \
  --output artifacts/official-T-new --gpc-root /src/gpc \
  --checkpoint /models/released/denoiser.pth --max-updates 8 --budget-seconds 120
```

Do not add `--execute` without a separately authorized training run. The lower
training APIs likewise require explicit execution. The plan does not certify
asset availability. No missing asset was replaced by training or new collection.

## Reuse and explicit differences

| Component | Reuse |
| --- | --- |
| `official_world_model.py` | Shared remote action normalization, strict native architecture/loader and H1 prediction; one native sampler path |
| `gpc_denoiser_training.py` | Remote raw five-frame/four-action loader and unchanged single-step weighted denoiser loss; thin delegates to the shared helpers |
| `region_weight.py`, `motion_replay.py` | Byte-identical remote modules |
| `official_contact_trainer.py` | Remote global train-pool/chunk and frozen useful/miss validation flow; T/contact admission and explicit execution added |

The same official raw contract serves loader/viewer: CHW96 float32[0,1], four
history frames and four absolute pixel targets, `a/511*2-1`, original3-Euler
sampler and default sigma_cond=None. The teammate H1 helper clips its returned
native image to[0,1]; this is retained exactly and disclosed in new case metadata,
not advertised as the unclipped native Euler tensor. No experimentB conditioning
or new sampling algorithm is used. Existing legacy raw feedback remains intact.

Replay keeps about60% moved windows. Case-split foreground loss uses motion/T/
pusher labels for useful windows and still T/pusher for misses; nonempty mask
weights are foreground1/table0 (the remote empty-mask uniform fallback is also
preserved). Chunk defaults remain64 train windows,8 updates and frozen64
validation windows, stratified useful/miss when labels exist. Privileged geometry
and contact labels affect loss/scoring, not the model's RGB/action inputs. Table
weight0 does not constrain background appearance, so this alignment does not
prove background pollution is fixed. The remote legacy region-weight parameter8
is retained; it does not replace the active foreground1/table0 loss.

Local full-state/history repair is preserved: reset records three hold controls
to create four real observations; the remote simple viewer instead pads missing
history without stepping. Once four real frames are present, window/action
alignment matches. Local scene, geometry and control timing are unchanged. New
v2 cases declare `official-original-color`, `history_model`/`actual_model`, no T1
identity and the clipped-output convention. Legacy v1 cases still load unchanged;
domains enter request identity and cannot cross predictors. Raw viewer mode does
not apply the legacy palette decoder; predicted pose/coverage are unmeasured.
Physical truth and descriptive pixel MAE remain separately labelled.

## Missing assets and Linux scope

Eight documented/default candidates were checked at exact paths in this checkout,
the original checkout and remote-review. None has weights or adjacent sidecars:
hour `model`/`model-30m`, contact-train, dated and undated case-loss/from-k8, and
official-T-init. `asset-audit.json` lists all24 checks. The remote sidecar stores
updates and initial SHA, not the current trained-file SHA; no published unique
trained-weight SHA or download address identifies the teammate video. Existing
stage-A raw T/L manifests have frozen splits but are `object-directed`; they are
not relabelled contact-balanced. The documented contact-batch root is absent.
Those archives and all old data, weights, claims and reviews remain untouched.

Linux x86_64/Python3.12.3 and the existing lock remain the reference. Portable
entrypoints/configurable paths are ready, but WSL listing returned
`Wsl/E_ACCESSDENIED`; no usable distro/host or GPU compatibility is established.
No distribution/VM, SSH session, installation or migration was started, and no
Windows environment was copied. Legacy Win32 Job/PSAPI/PowerShell supervision
stays historical: `wm_compat_guard.WindowsWorker`, `policy_memory` system/process
queries, `gpc_adaptation_guard.supervise`, entry Job checks and PowerShell launch
wrappers need separate Linux handling if reused. The raw viewer does not call
that host monitor; its host-memory
fields are explicitly unverified/null. CUDA checks and session time remain, but
Linux process/resource supervision and matched-checkpoint numerical/manual
acceptance are still future verification, not a ported guardian claim.

Three bounded engineering commands passed54/55/55 tests in10.9034131s total;
observed working-set peak was91435008B. They use arrays/spies/fake environments.
They cover remote helper/loss/replay/chunk parity, RGB/action/time contracts,
T1/legacy-decoder traps, once-only execution, C and S roundtrip. Source-only CLI
probes also pass; synthetic chunk output markers are not real optimizer updates.
There were0 actual source numeric frames, no held/dev/B2 payload reads and no
weight-tensor/model/SDK/physics work. Evidence is in
`artifacts/official-raw-alignment-20261026/`; final source validation and delivery
receipts record exact costs. Matching real official weights are needed before
claiming reproduced predictions or improvement.
