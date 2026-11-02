# Fixed raw T/L training pass

The separately authorized `gpc-raw-tl-fulltrain-01` run starts from the released
GPC phase-two weights and uses the original grey-object/green-goal RGB domain.
The fixed configuration is `configs/gpc_raw_tl_fulltrain.json`. No collection,
policy training, foreground-weighted loss, motion replay or sigma search belongs
to this run. Existing failed/rejected T1 experiments remain unchanged.

The existing object-directed training data supplies 24 trajectories/scenes,
12 per shape and 97 legal five-image/four-action windows per trajectory. The
2,328 unique windows are used once, T/L alternating. The first 500 entries are
identical to the previously frozen schedule; all unused windows follow in
ascending-anchor scene round robin. This increases window exposure, not scene
diversity, and the source data is not described as contact-balanced.

The loss, history noise, offset noise, parameter selection, batch size and
optimizer match pilot03: native phase-two one-step supervised denoising,
`inner_model.*`, batch 1, AdamW learning rate 1e-4, betas .9/.999, epsilon 1e-8,
weight decay .01, gradient norm cap 1 and seed 5478. Training action arithmetic
retains pilot03 float32 normalization. Original float64 absolute actions are
retained separately for the aligned raw prediction helper. All five images are
uint8 RGB divided by 255 without T1, recolouring or background subtraction.
The dataset keeps one episode cache and does not preload the float dataset.

Only model state is exported at updates 500 and 2,328, with current file hashes,
raw input contract and source/data/list identities in sidecars. Final means
fixed update 2,328, never the best development score or an incomplete 500 export.
The training model and optimizer are released before serial reloads of Initial,
raw500 and raw2328 through the aligned official raw loader.

Before training, 16 development H1 windows are fixed from metadata, eight per
shape, aiming at four contact and four noncontact windows in each shape. Bucket
shortages are disclosed without held-data replacement. Each shape contributes
only one independent development scene. No independent-validation/B2/held or
model-test payload is read. Each role receives exactly 16 fixed-noise loss
examples and 16 predictions, seed 5478 and three native Euler steps with default
`sigma_cond=None`. Raw MAE and identical-noise losses are diagnostic only;
legacy palette decoding, pose and coverage measurements are not applied.

The existing shared model lock, append journal and Windows Job guardian enforce
one active model, four loads, 2,328 updates, 48 forecasts/144 Euler steps and 48
loss examples. The total wall/monotonic budget is 1,800 seconds including startup,
export and cleanup; training including its initial load ends within 1,200 seconds.
All original resource floors, CUDA peaks and independent five-second heartbeat /
allocator freshness requirements remain. The first real error stops the fixed
identity, with no retry, resume, CPU fallback or replacement account.

Preparation permits three synthetic CPU commands, at most 60 seconds each and
180 seconds total, two CPU cores, 1 GiB working set and 32 MiB new evidence.
No real neural or physical preflight is used. The 1 GiB run output cap includes
model files, raw predictions and temporary output; metadata is capped at 16 MiB.
The existing 64 MiB telemetry allocation is unchanged.

After a completed export, `start_raw_tl_demo.cmd` loads the actual raw2328 file
and its hashes by default. Argument `500` explicitly chooses the intermediate
model. The button says **Raw T+L**; S/V retain case capture/video. The current
launcher omits Initial and disables comparison shortcuts. Missing or invalid assets raise an error. Preparation
and this training run do not open a GUI automatically.

Three bounded engineering commands passed35,88 and95 synthetic checks in
14.889,18.888 and20.881 seconds (54.658s total; peak working set101,388,288B).
The final command also passed source/message checks. Planning froze all2,328 unique windows (T/L1,164 each)
and the16 development windows with4 contact/4 noncontact per shape and one
development scene per shape. Integrated/source verification is complete before
the single real run. No new real update or prediction is claimed by preparation.

## Completed run

The sole real run on clean source `ecb6241fb45da3acbc2c91cb4fb5eda8b79b05e9`
completed all2,328 updates/examples and both fixed exports. Guarded duration was
590.078 seconds; the external launch receipt records590.660 seconds. The actual
training loop took567.422 seconds; startup, loads, exports,48 forecasts and48
loss examples share the original total deadline. All four serial models loaded
successfully. No optimizer/RNG/account restoration, retry, extra forward,
collection, physics or policy training was performed. Job active processes0,
lock released, first error null and cleanup errors empty; no Python remained.

| Shape | Role | Windows | Raw RGB MAE | Identical-noise loss |
| --- | --- | ---: | ---: | ---: |
| T | raw500 | 8 | .02303410 | .01941239 |
| T | raw2328 | 8 | .01395406 | .00555694 |
| L | raw500 | 8 | .02266191 | .02097856 |
| L | raw2328 | 8 | .01376052 | .00695624 |
| T+L | raw500 | 16 | .02284801 | .02019548 |
| T+L | raw2328 | 16 | .01385729 | .00625659 |

Raw2328 improves MAE and identical-noise loss in16/16 fixed windows relative to
raw500. Mean MAE falls39.35% and loss69.02%. Both models use the same original
grey/green domain. The completed run also retained Initial as a legacy-domain
reference; its red/black pretraining creates a colour-domain mismatch on raw
grey/green inputs. After the user's correction, that reference is excluded from
the primary comparison and current demo. Improvements against it must not be
presented as isolated dynamics gains. The grey object and
green goal reappear in the raw previews, with remaining edge/position differences.
All16 panels were inspected, including the first contact case per shape; none
were removed. This small development check has one scene per shape, overlapping
windows and one forecast seed. It does not establish H4 accuracy, physical
dynamics/policy acceptance, or resolution of every duplicate/missing-part case
in the user's video. No legacy palette decoder, pose or coverage score was used.

Artifacts are in `artifacts/gpc-raw-tl-fulltrain-20261026/`:

- `raw-tl-step-2328.pt`:35,018,654B, SHA256
  `79c1fe8b9eaa44b80f8c90ae27f3280b84a01456aa3efd6ebc75dda253558c05`.
- `raw-tl-step-0500.pt`:35,018,654B, SHA256
  `723fbe8ffff1e4446d27d1dd10d5b101a9d5751cdd75ec0cade4e6fbc6972d82`.
- Matching JSON sidecars and `raw-tl-final.json` retain actual current hashes,
  raw input contract and fixed source/data/list identities.
- `diagnostic.json`, `training-metrics.json`, `completion.json`,64 raw arrays,
  `raw-diagnostic-panels/` (16 panels and overview), guardian and append ledger.

The104 run files total91,447,457B. Observed process peak working set was
1,546,186,752B, private2,873,384,960B; CUDA peak allocated510,962,176B
(487.292MiB) and reserved578,813,952B. Startup available RAM was5,121,777,664B;
the observed runtime minimum was3,273,383,936B. All fixed limits held.
Preparation and external receipts are in
`artifacts/gpc-raw-tl-fulltrain-preparation-20261026/`. Weights/raw data remain
local ignored artifacts. The launcher defaults to this fixed2328 checkpoint,
not a score-selected model, and the GUI was not opened during this run.

The post-run Initial-removal edit changes the launcher/buttons only, with no
new training, forecasts, GUI or fourth engineering command. The95-check result
above belongs to the frozen experiment source before that presentation edit.
Original raw outputs and operation costs remain preserved for provenance.
