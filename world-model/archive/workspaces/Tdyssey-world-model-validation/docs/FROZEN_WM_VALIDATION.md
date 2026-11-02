# Frozen world-model diagnostic result, 2026-10-24

The manifest-exact GPC asset is verified, but **no action-scoring gate passed**.
All 16 anchors abstained from complete three-noise candidate selection. This is
a geometry-informed development diagnostic, not policy success/generalization
evaluation. Policy/world-model optimizer updates and training were zero.

Base `e0842d7`; implementation `11b7cd9`; same-attempt recovery `52e82b6`.
See [frozen protocol](FROZEN_WM_PROTOCOL_V1.md) and
[persistence recovery](FROZEN_WM_RECOVERY.md). No palette variant was added.

## Asset audit

The original upstream-tracked `gpc_opt_evaluation/csgo/denoiser.pth` is intact:
SHA-256 `15126113b7490fa976290c62ad4c7a92734bc898664522706c67a4aa6bff47b2`.
Its Git HEAD/index blob and working file agree. Both checkpoints have 298 keys,
but 296 tensors differ: this is **different weights**, not serialization alone.

Acquired only the pinned HF phase-two checkpoint, 35,043,916 bytes, SHA-256
`4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`, revision
`c16a467ec176af19bba5136564a81d8d61d72af9`. The initial DNS failure transferred
zero bytes and remains recorded. Original weights and manifest were preserved.
All 20 GPC source hashes and seven published evidence files passed. The exact
old custom best.pt/catalog was not found in the audited workspace, so that
comparison is unavailable; no replacement was initialized. No dataset/environment
was downloaded or installed.

## Calibration and success precision

The decoder was frozen before any real render. Inputs are RGB, known L/T
geometry, fixed renderer/color priors and the fixed goal. Future true pose,
contact and ROI never repair a predicted score.

| Independent calibration | L | T |
| --- | ---: | ---: |
| Measurable | 12/12 | 12/12 |
| Object position P90, pixels | 1.039 | 1.060 |
| Angle P90, degrees | 0.354 | 0.668 |
| Pusher position P90, pixels | 0.821 | 1.175 |
| Coverage MAE | 0.00836 | 0.00570 |
| Continuous-error gate | pass | pass |

Continuous calibration does **not** establish strict-success precision. Of eight
states near 0.95, all four true successes were false negatives. There were no
predicted positives, so precision is undefined. Two T states with true coverage
0.961905/0.923810 decoded to the same pose/coverage 0.939254. This 96-pixel
interface cannot yet support precise termination.

## Model results and denominators

Sixty-four slots were attempted: 63 complete branches plus permanent L01/hold
failure before its reset call. It was not retried or replaced. No physical
branch succeeded under strict coverage >0.95; each complete branch used all
four future controls. Forecasts use raw autoregressive RGB; only measurement/
preview clips. Three common-noise repeats reuse the same physical branch.

| H4 availability | L | T |
| --- | ---: | ---: |
| Raw image decodes | 4/141 | 34/48 |
| Scores after checking preceding predicted stopping history | 0/141 | 6/48 |
| Complete three-noise candidate scores | 0/48 planned | 0/16 planned |
| Anchor selection abstentions | 12/12 | 4/4 |

H1 validity was 0/141 L and 6/48 T. A missing earlier decode cannot secretly
establish whether predicted success already occurred. Thus some usable H4 raw
poses still yield no complete sequence score. No true stopping time is supplied
to prediction. Complete-candidate coverage MAE and observed selected regret
remain unavailable. Conservative pair accuracy zero/regret upper bound one are
penalties for abstention, not measured quality of selected actions.

| Engineering screen | L | T |
| --- | --- | --- |
| Noncontact drift | fail: incomplete measurements | fail: incomplete; observed subset 2.683 px >2 |
| Contact motion | fail: 0/9 complete measurements | fail: only 2/3 complete measurements |
| Rotation >5 degrees | fail: 0/3 complete measurements | insufficient: only one rotating anchor |
| Endpoint coverage | insufficient: missing physical branch and complete scores | fail: no complete scores |
| Ranking | insufficient: 4/6 distinguishable anchors, 12 pairs, 3 unknown pairs | fail: 2 anchors/6 pairs, all abstain |
| Direct contact probability | unavailable: no GPC head | unavailable: no GPC head |

L contact coverage is nine anchors/nine branches, and rotation coverage three/
three: both meet their minima. The original runtime summary wrongly made the
one globally missing physical branch override these counts. Offline summary
v1.2 corrects the classification to measurement failure while retaining that
missing branch separately; it cannot create a pass. It also computes the
privileged last-true-pose persistence reference independently of model decoding.
The raw runtime report and its source binding remain byte-identical.

The image proxies are mixed. Contact object-mask IoU improves over persistence:
L 0.510/0.352, T 0.757/0.376. Noncontact IoU degrades: L 0.490/1.000,
T 0.817/1.000. H4 RGB MSE is L/T 0.00677/0.00692 versus persistence
0.00142/0.00148, including the gray-to-red/goal appearance change. These proxies
do not establish coverage or reliable motion. One T14/toward branch reproduces
rotation: H4 angle MAE 0.814 degrees versus zero-rotation error 11.455 degrees.
One physical rotating branch cannot pass the T rotation sample minimum.

On 11 complete L and four complete T anchors, hold endpoint means are
0.42309/0.47851, uniform-candidate expectations 0.39173/0.44470 and physical
best-of-four 0.42935/0.48858. Hold regrets are 0.00626/0.01007; uniform regrets
0.03762/0.04388. These are offline references, not model selection success or
full-reset policy results. The incomplete L anchor is explicitly excluded from
these baseline point estimates, with its missing denominator retained elsewhere.

## Cost, Windows failure and preservation

| Resource | Actual / conservative spent | Cap |
| --- | ---: | ---: |
| Calibration / branch slots | 24 / 64 | 24 / 64 |
| Resets / restores | 87 actual, 88 charged / 87 | 88 / 88 |
| Reset settling substeps | 87 actual, 88 charged | 88 |
| History + future controls | 189 + 252 = 441 | 448 |
| Control physics substeps | 4410 | 4480 |
| GPC images / Euler steps | 756 / 2268 | 768 / 2304 |
| Custom images / optimizer updates | 0 / 0 | 192 / 0 |
| Wall, including repair downtime | 951.88 seconds | 1800 |
| Runtime results / telemetry including ledger | 11,005,844 / 1,991,724 bytes at guard closeout | 256 / 64 MiB |

Worker 22072 first aborted on Windows ledger replacement at journal event 228.
Its journal-ahead reset/settle reservation remains charged. Recovery worker 3516
reused all 24 calibration arrays/four L00 branches and ran only 59 unused slots,
under the same claim/deadline. It exited zero without a guard stop; the first
report, monitor, ledger backup and journal prefix remain inspectable. Both
workers used actual-PID/nonce/Job Object monitoring. Total actual physics is
4410 control substeps plus 87 reset settles; no hidden physical retry.

Observed RAM/commit/GPU minima: 3.030/10.211/6.281 GiB. Worker peak working set:
1.210 GiB; process commit/pagefile peak: 2.341 GiB. CUDA allocation/reservation:
110,273,536/136,314,880 bytes. The 8,729,827-parameter network retained its full
fingerprint, file hash, eval/freeze state and Python/NumPy/CPU/CUDA RNG; network
destruction passed. Polling cannot guarantee instantaneous resource floors.

All 28 independent reconciliation checks passed. The 150 protected B/C artifacts
(97,364,187 bytes), both checkpoint files, original calibration/physical arrays
and aborted evidence are unchanged. All 1789 journal events/893 operations
reconcile, with no pending operation or refunded cost. No message was sent to
the separate offline-analysis chat and no file there was changed.

## Review files and next step

Local ignored evidence directories:

- `artifacts/frozen-wm-assets/`: asset/tensor/source audits, initial network
  failure, calibration precision review, independent reconciliation.
- `artifacts/frozen-wm-validation-v1-20261024/`: raw report/NPZ, per-branch JSON,
  immutable first abort and 19 scientific preview/error PNGs.
- `artifacts/frozen-wm-ledger-v1/`: spent account, journal and recovery backup.
- `artifacts/frozen-wm-review-v1.2-20261024/`: Chinese review, offline recalculated
  summary, 768 raw rows/256 physical-horizon rows and all stratified CSV tables.

Raw report SHA-256:
`c7b1e3257468874940ad074d3ffc00c3d3d88f9f6de16531aba6d2c3cc6e3795`.

Offline export performs no physics, rendering, decoding or forecasting:

```powershell
& $python -S -B scripts/export_frozen_wm_results.py `
  --run-dir artifacts/frozen-wm-validation-v1-20261024 `
  --ledger-dir artifacts/frozen-wm-ledger-v1 `
  --output-dir artifacts/a-fresh-review-directory --recompute-summary
```

Final focused tests: 39 passed. Full source-only discovery: 217 passed/160
skipped (377 total), using workspace TEMP on Windows. An earlier sandbox TEMP
attempt exposed two existing atomic-replace permission errors; the workspace
TEMP rerun passed. Ordinary tests added no physical experiment.

Next repair the generated-image measurement interface and near-0.95 termination,
then inspect deformation/noncontact drift/contact/rotation under a new frozen
protocol to determine whether official-domain dynamics adaptation is needed.
GPC has no direct reward/contact head; the old custom reward is a different
clipped target. This result supports neither imagined-policy training nor an
automatic adaptation run, extra physical samples or a longer horizon.
