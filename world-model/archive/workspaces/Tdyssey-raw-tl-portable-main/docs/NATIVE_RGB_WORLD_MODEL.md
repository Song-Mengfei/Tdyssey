# Original GPC RGB experiment

Use the complete released GPC, four raw RGB96 history frames and four aligned
absolute actions (three executed, one proposed), directly predicting RGB96. All
8,729,827 parameters participate in the original phase-two H1 residual-MSE
objective. The native three-Euler sampler and its quantization are retained.
No motion head, geometry loss, letter ID, template transport or output repair is
part of inference. Shape registration is confined to offline measurement.

## Fixed data and physics

Train letters: **A B C E F H I J L M O P S T U V X Y**.
Final unseen letters: **D G K N Q R W Z**. Released pretraining already saw T.
Use only the original published initialization with SHA256
`4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
Earlier T/L or all-letter adapted weights are ineligible initialization. Whole
letters are held out; seen validation/final tests have independent trajectories
and scenes. Exact geometry unions were checked under 0/90/180/270-degree
rotation before training, with no identical cross-split shapes.

Observations have white background, grey object and blue pusher, with no green
goal. A red target cross belongs only to the viewer surface. Control remains
absolute XY in [0,511], 10Hz controls and 100Hz physics, with near/mid/far targets.
All new native RGB interfaces use `native_rgb_env.make_env` and the same control
step. Historical viewers retain their historical environment and are not the
new protocol's viewer.

The upstream kinematic pusher can force its way into an object against a wall.
The new `finite-force-pd-v1` is an explicit simulator change: dynamic pusher
mass1, infinite rotational moment, force limit10000, pre-step speed limit200,
PD gains100/20, solver50 iterations and collision slop0.1. Post-step velocity
can change through forces/impulses. Object collision geometry, mass/inertia and
walls are retained. This does not establish parity with original-controller
physics. Control-protocol SHA256 is
`fd358dba8f26a616ec6c55b3f6d532681bf5036e2cf79fe5a467001e3a77ae96`.
Goal reward/termination do not affect the new collection.

States/actions/contact records are compact NPZ; RGB is rendered on demand with
a bounded uint8 LRU. Simulator labels guide collection, physical audit and
offline evaluation only. The denoiser receives only RGB and normalized actions.

## Coverage and finite budgets

Contour bins are fixed at8 world-pixel arc length before collection, including
hole rings, corners and concavities. Count actual collision points mapped back
to the object's contour, with independent trajectory IDs and actual contact
angle bins. Requested targets/directions remain separately labelled. Collection
prioritizes low-contact reachable bins, different target distances/durations and
four-wall/four-corner scenes; training cycles letters then independent
trajectories so long or easy trajectories do not dominate.

Closed holes are separately classified. A normal staging point must be connected
to legal exterior free space and its staging-to-contact-center segment must be
collision free. Failed conservative staging tests are **unconfirmed**, not proof
of global impossibility. Reports retain total-contour, confirmed-reachable and
unconfirmed coverage. Post-collection reachability review is versioned and does
not alter states, actions or the original reports.

The local34,560-control allowance includes every probe attempt:1,416 probe
controls plus33,144 collected controls in518 trajectories (train26,232,
seen validation6,912). No collection allowance remains. Probe data from the
upstream failure comparison are excluded from training. The corrected native
dataset has zero invalid transitions and passed a hash-bound physical audit;
any invalid retained transition would fail the whole collection gate.

Local optimizer attempts and time share one persistent ledger, including
engineering/resource probes and tiny fits: at most5,000 attempts or5,400s.
Microbatch starts1; real batch1/2/4 full-GPC backward probes selected4, with
effective batch16. Engineering charged4 updates and passed all-parameter changes
and bit-exact next-step recovery of model/Adam/scaler/scheduler/RNG/order. Measured
microbatch4 peak reserved memory was2,055,208,960 bytes; this is a new native-GPC
measurement. Earlier structured-model memory results are inapplicable.
Tiny fit and independent validation results are recorded in the run's REPORT;
an implementation check is not evidence of generalization.

All new persistent local artifacts, including checkpoints/reports/upload ZIP,
share a3GiB ceiling. No persistent float-frame dataset is created. Best/last
checkpoints rotate, retain complete CPU training state and source/data/config
identities, and require an exact resume. Interrupted update reservations remain
charged. A STOP file inside the experiment root safely stops the loop.

Metadata replacement retries transient Windows permission failures for at most2s.
The first main run stopped after876 logged updates when its877th optimizer step
could not durably update the ledger. Original weights/logs and the complete750
checkpoint are preserved. Recovery changed only this I/O function: all other
trainer AST and13 source hashes match; model/Adam/scaler/scheduler/RNG/order,
progress and config fingerprints are identical across the explicit metadata-only
checkpoint migration. All spent attempts/time remain charged. Logical model
progress after replay must not be confused with aggregate optimizer attempts.

## Measured local outcome (2026-10-29)

This bounded pilot failed the dynamics/shape scale-up gate. The complete last
checkpoint is logical step1668; best denoising validation selected step1500
(loss0.00112789794). The aggregate ledger closed at1,899 charged optimizer attempts,
1,898 durably recorded completions and5,352.516 seconds, including engineering,
tiny fit and the Windows I/O interruption/replay. All296 parameter tensors changed.
The126 logged replay steps matched loss, gradient norm, LR and sample sequence
bit for bit; another applied but unlogged interrupted step remains charged.

Fixed seen validation used72 windows,432 RGB rows and432 offline physical rows.
Physical motion error (world pixels; conditional on measurable outputs) was
hold1.217 vs fine-tuned2.671 atH1, and hold4.279 vs fine-tuned5.764 atH4.
All contact/rotation/boundary paired physical groups were worse than hold.
Fine-tuned prediction scoring failed1/72 atH1 (empty pusher mask); physical scoring
also retained1/72 unavailable atH4 (shape fit). Shape deformation failed2/35
assessableH1 and10/37 assessableH4 cases. Frozen GPC had49/72 and71/72 failed RGB
rows respectively, including raw-range and missing-mask failures, retained without
clipping repair. Low reconstruction loss or apparent angle gains did not establish
physical prediction accuracy. No extra data, training or final-eight predictions
were used to rescue the result.

The from-scratch delivery is for reproducibility and review. `full` must refuse
this failed local report; preparing the package does not qualify server scale-up.
The server CUDA environment and real server throughput remain untested locally.

## Local commands

Use an explicitly supplied existing Python environment; the launcher installs
nothing. Supply pinned source roots and the original released checkpoint:

```powershell
.\run_native_rgb_local.ps1 -Python /path/to/existing/python.exe `
  -PhysicsRoot /path/to/diffusion_policy -GpcRoot /path/to/gpc_code `
  -InitCheckpoint /path/to/original/denoiser.pth -Mode preflight
```

`-Mode pilot` chains audit, full-network engineering, fixed16-example tiny fit,
mixed training, paired seen validation and report. Existing data are audited and
reused; the launcher does not recollect them. Use the run's `pilot-effective.json`
for subsequent stages. A finite time/update stop is preserved, not silently
restarted. `-Mode resume -ResumeCheckpoint /path/to/last.pt` reuses the same
ledger, data, configuration and schedule; it cannot enlarge a spent allowance.

Individual stages are exposed through `scripts/native_rgb_world_model.py`:
`preflight`, `probe`, `collect`, `audit`, `engineering`, `microfit`, `train`,
`evaluate`, `report`, `freeze`. They require explicit `--physics-root`,
`--gpc-root`, `--init-checkpoint`, `--config` and `--output`; shared server data
use `--data`. A failed data/engineering/tiny-fit gate stops advancement.

`scripts/native_rgb_viewer.py --help` describes the new click viewer. Its default
is planning only. `--execute` loads explicitly supplied trained weights; reset
records three real holds, and a click predicts H1 then executes one real control.
The local collection allowance is spent, so no real viewer controls are part of
this recorded local validation. Viewer checks currently use mocks only.

## Paired validation and final-test protection

Compare frozen released GPC, the fine-tuned native model and copying the last
frame. Score raw output, feeding generated RGB back for H4/H8. Report per-letter,
contact/rotation/boundary/stationary groups, failure denominators, P95, position
and pusher errors in512 coordinates, raw/full-truth IoU and shape errors.
Object position is the visible-mask centroid; angular registration is an offline
apparent-angle metric and is ambiguous for symmetric/occluded objects. Pusher
occlusion can change a visible centroid even when the physical object is still.
These measurements are labelled separately from privileged physical motion and
must not alone establish accurate physical pose under occlusion.

The additional offline instrument aligns the true full object mask to each
unchanged predicted RGB while accounting for the predicted blue-pusher occlusion.
It reuses bounded raw CPU copies from the same sampler calls; fitted masks and
poses never enter generation. Fixed known-pose renderer calibration covered108
cases across all18 seen letters (zero physical controls/model calls), with maximum
measurement error2.47 world pixels/1degree against predeclared6px/6degree limits.
Physical-motion qualification keeps unmeasurable rows in the original denominator
and requires improvement beyond twice the finite calibration error bound. Small
apparent gains alone do not establish physical dynamics. Raw RGB/shape metrics
and these more conservative physical measurements are both reported.

Tolerances are calibrated using true rendered seen-letter images at fixed
angles/subpixel positions, independently of model output, and bound to renderer
source SHA256. Raw shape errors remain continuous; occluded/cropped topology is
unavailable and is never counted as a successful shape check. Static image/action
counterfactuals test conditioning, not physical contact success.

The directional motion target is at least20% improvement over hold in contact,
rotation and boundary groups at both H1 and H4; shape failures are separately reported. A copy
baseline cannot qualify merely through low RGB loss. Failure stops scale-up;
no explicit-motion fallback or extra collection allowance is introduced.

Local code blocks both final test splits. On the server, first freeze the selected
checkpoint, data/config/source and seen-renderer calibration identity. Final
seen/unseen evaluation refuses a different identity and refuses overwriting an
inspected result. Seen validation may be rerun for a new checkpoint in a separate
hash-labelled directory, retaining the original frozen tolerances and reports.

## User-run server workflow

No remote job is started by local tooling. Upload the generated from-scratch ZIP,
extract into a fresh directory and run manually on Linux x86_64/Python3.12:

```bash
bash run_native_rgb_server.sh prepare --setup
bash run_native_rgb_server.sh preflight
bash run_native_rgb_server.sh full
```

`prepare --setup` creates an independent server environment using paired
Torch2.10.0/torchvision0.25.0 CUDA13.0 wheels from the
[official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).
Core dependencies are pinned; transitive resolution is captured with actual
`pip freeze --all` after setup rather than claimed as a prevalidated full hash
lock. Real driver/kernel, source, checkpoint and device checks are mandatory.
Local CUDA12.4 success does not verify this server environment.

`full` rejects a failed local physical/scientific report, with no bypass flag.
Requested server budget is593,920 controls:18x400x64 train,18x40x64 seen validation,
18x40x64 seen final test and8x80x64 unseen final test. Probe controls consume this
same ledger; actual valid windows/controls can be lower and are recorded.
Physical collection uses bounded independent processes and is audited before
training. Every seed has its own optimizer, ledger and checkpoints; common data
are read-only. Each seed is limited to50,000 attempts/24h, with three target seeds.
Server sampling declares18/16px legal staging distances and at most2 repeated
misses per approach before rotating approach/pose; misses remain unknown, never
automatically labelled unreachable. These sampling choices change no geometry
or controller and do not retroactively change the local corpus or its denominator.
Eight validation checkpoints without an improvement of at least1e-6 trigger
early stopping; a50s checkpoint reserve protects the aggregate time ceiling.
An engineering or scientific gate failure stops pending jobs rather than blindly
starting the remaining seeds; missing seeds remain explicit in the job report.

The server resource probe tests batch8/16/32/64 with the full native residual
forward/backward, Adam memory reserved and zero optimizer updates. At most one
two-process throughput comparison is allowed. Two simultaneous seeds are chosen
only when measured end-to-end throughput and memory support them; otherwise use
one. CPU process/deadline tests do not prove server CUDA throughput.

After validation and checkpoint selection, manually run final tests, for example:

```bash
bash run_native_rgb_server.sh evaluate --seed 5478 --split seen_test
bash run_native_rgb_server.sh evaluate --seed 5478 --split unseen_test
```

The first final evaluation creates an immutable model/protocol seal before
reading final images. `resume --seed 5478 --resume /path/to/complete/last.pt` is
available before freezing. No single-GPU DDP speed-up is assumed.

## Delivery contents

The upload inventory lists exact absolute local sources, relative server targets,
bytes and SHA256 for every archived item, including original initialization,
small hash-verified upstream source exports, native scripts/configs and curated
local evidence. It excludes environments, Git history, huge images, local data
and trained checkpoints from the default from-scratch package. The archive is
read back and every entry verified after writing. Complete local best/last
checkpoints remain available in the experiment directory.
