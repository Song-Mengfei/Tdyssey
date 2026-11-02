# Frozen world-model diagnostic, protocol v1

Frozen before any calibration render, main physical branch or model forecast on
2026-10-24. Base source: `e0842d7fa9f0ed04c0b5ce827b1b3a6a444c58e3`.
This is a geometry-informed development diagnostic, not reset-distribution success,
an untouched model test, a DPPO return estimate, or authorization to train.

## Coordinates and fixed scenes

Use unchanged official legacy PushT physics, 512-pixel screen coordinates (y down),
96x96 RGB observations, original gray object/green goal, no palette variant.
Angles are radians internally; positive rotation is clockwise on screen.
L origin is its area centroid; T origin is the original upstream origin with
center-of-gravity offset (0,45). Every branch starts with zero velocities/forces.
The fixed official goal is (256,256,45 degrees) for T; for L its center is
(256-45/sqrt(2),256+45/sqrt(2)), angle 45 degrees.
All pusher coordinates/directions below are object-local, transformed by the
declared initial object pose. The pusher radius is 15 pixels. Seed is 700000+i
for zero-based anchor index i. These seeds are diagnostic identities, not stored
training/evaluation scene or model-test indices.

| IDs | Shape/family | Object origin | Initial angle | Local pusher | Local toward |
| --- | --- | --- | --- | --- | --- |
| L00,L01,L02 | L, noncontact approach | (240,245) | -15,0,15 degrees | (-125,0) | (1,0) |
| L03,L04,L05 | L, central translation | (240,245) | -15,0,15 degrees | (-50,0) | (1,0) |
| L06 | L, eccentric clockwise | (240,245) | 0 degrees | (-50,-50) | (1,0) |
| L07 | L, eccentric counterclockwise | (240,245) | 0 degrees | (-50,30) | (1,0) |
| L08 | L, eccentric counterclockwise | (240,245) | 15 degrees | (20,65) | (0,-1) |
| L09,L10,L11 | L, goal correction/overpush | goal center + R(angle)*(-3,0) | 44,45,46 degrees | (-50,0) | (1,0) |
| T12 | T, noncontact approach | (250,210) | 0 degrees | (-145,15) | (1,0) |
| T13 | T, central translation | (250,210) | 0 degrees | (0,140) | (0,-1) |
| T14 | T, eccentric contact | (250,210) | 0 degrees | (-35,-20) | (0,1) |
| T15 | T, goal correction/overpush | goal origin + R(45deg)*(-3,0) | 45 degrees | (0,140) | (0,-1) |

Each of the four candidates uses a **fresh environment**, one reset (one settle),
one full restore of the initial state, then three hold controls at its initial
pusher location. Record all four real history RGB frames, actual pusher states,
controls and solver contacts. Require zero history contacts and geometry-valid
initial state. Compare exact history images and full history-end snapshots among
candidates. No established-contact snapshot restoration. Bad attempts consume
budget and stay in the denominator; no replacement anchors or physical retries.
Targets are controls, never substituted for measured pusher position.

Candidate order and tie preference: hold, toward, side, away. For horizon h=1..4,
targets are p0, p0+12h*d, p0+12h*(-dy,dx), p0-12h*d, respectively. No action is
chosen using physical future outcomes. Success is strictly raw coverage >0.95;
stop physical controls immediately, preserve the terminal image/state/score as
absorbing for remaining horizons. Report that absorption explicitly. History
success invalidates the branch and is recorded, without further controls.
Raw GPC generation retains all four autoregressive controls. For scores only,
apply the same strict stop rule to decoded predicted coverage: its first >0.95
state is absorbing, based only on predictions. A preceding invalid decode makes
later scores unavailable. Keep raw images/decodes. Post-physical-stop RGB errors
are termination-interface diagnostics, not pure dynamics errors; no true stop
time is supplied to predicted scoring.

## Measurement and forecast contract

First use exactly 24 independent render-only calibration states: 12 L then 12 T.
For k=0..7, angle=30k degrees, origin=(170+80*(k mod 3),
170+60*(floor(k/3) mod 3)); T y origin subtracts 45. Pusher=(60,60),
For k=8..11, angle=45 degrees and origin=goal origin + R(45deg)*(dx,0),
with dx=-4,-2,2,4 respectively. Pusher is pose-transformed (-50,0) for L
and (0,140) for T. These states straddle the success threshold by geometry.
Fixed goals as above. One reset/restore/render per state, no controller steps.
The image decoder configuration and tests are frozen before these renders.
No decoder tuning on calibration results or predicted images. It takes only RGB,
shape geometry and fixed goal. Truth geometry is only an error label; no future
truth pose, contact or ROI may repair missing predictions or guide decoding.
Known L/T geometry and fixed object color palette are disclosed diagnostic priors.
Malformed shape, occlusion or missing object/pusher is an abstention.

Calibration gate, separately by shape: measurable >=95%, position and angle P90
<=5 pixels/5 degrees, pusher position P90 <=5 pixels, coverage MAE <=0.03
(missing values also counted explicitly).
Near-goal labels within 0.05 of 0.95 are reported separately. Calibration does
not establish success-classification precision. If the gate fails, predicted
coverage/ranking is unavailable; RGB and appearance proxies remain diagnostic.
No truth substitution. Geometry proximity is only a contact proxy, not contact.

GPC uses the manifest-exact frozen checkpoint, four real RGB [0,1] frames,
three executed history controls and four future targets, a/511*2-1, three Euler
steps. Seeds 5478,5479,5480 are common across candidates within each anchor.
Autoregressive history uses raw generated RGB; clipping occurs only in metrics
and previews with clipped fraction logged. No direct contact/reward/pose head.
If the exact historical custom model is found, evaluate only L, independently
from image decoding; otherwise record unavailable, never initialize a replacement.

## Scores, denominators and engineering gates

Primary score is H4 endpoint raw geometric coverage (not max-ever clipped reward).
Report H1/H4, retain H2/H3 raw records. Persistence copies last observed RGB/pose;
constant contact probability is the empirical marginal descriptive baseline and
has no prospective selection role. Uniform-random candidate expectation is the
mean of the four physical endpoint coverages. Constant reward selects hold under
fixed tie order. Predicted score averages the three fixed noises; any invalid
noise makes that candidate invalid. Invalid candidates cause selection abstention,
and invalid eligible pairs count as incorrect for the conservative gate.
Ties: select the first candidate within 1e-6 of the maximum; pair score differences
<=1e-6 tie. Actual pair eligibility requires >=0.05.
For selection abstentions report rate and regret bound [0,1], never drop the row.

Engineering screens, independently for L and T and stratified by contact:
H4 noncontact mean position drift <=2 pixels; contact motion error better than
persistence; for true rotations >5 degrees, direction accuracy >=80%, angle MAE
<=5 degrees and better than zero rotation; endpoint coverage MAE <=0.05;
eligible pair accuracy >=75%, selected endpoint regret <0.03 and less than hold
and uniform-random expectation. Contact heads, if available, require better Brier
than constant and report recall. GPC contact-head gate is unavailable.
Require >=6 L contact anchors and >=8 contact branches; >=3 rotating anchors and
>=3 rotating branches; >=6 distinguishable anchors and >=12 eligible pairs.
For T require >=2 contact anchors, >=2 contact branches, >=2 rotating anchors/
branches and >=2 distinguishable anchors with >=4 pairs. Otherwise label the
corresponding screen coverage insufficient, never automatically add samples.
Noise is repeated inference on a physical branch, not an independent sample.
Summaries weight anchors equally, with branches/noises nested within anchors.
Report noise variability, near-goal errors, optimism and missing denominators.

## Durable budget and resource limits

One independent diagnostic attempt ledger binds protocol/config/source/checkpoint
hashes. Reserve attempts before operations, fsync journals, never refund failures
or reset accounts under new output names. Preserve raw completed/aborted records.
No training or evaluation account is initialized, resumed or written.

| Resource | Hard cap |
| --- | --- |
| Main branches | 64 |
| Calibration render states | 24 |
| Resets / restores / reset settling substeps | 88 / 88 / 88 |
| Actual controls / control physics substeps | 448 / 4480 |
| GPC forecast images / Euler steps | 768 / 2304 |
| Custom forecast images (only exact old model) | 192 |
| Runtime wall / results and raw images / telemetry | 1800 seconds / 256 MiB / 64 MiB |
| All policy/world-model optimizer updates | 0 |

No physical debug/preflight outside these counters. Synthetic tests use no real
simulator. One complete model resident; release before another. Startup checks
RAM/commit/GPU/disk; actual Windows worker PID and kill-on-close Job Object,
sampled runtime guards reuse existing proven machinery. Startup floors 4/6/4/2
GiB (RAM/commit/GPU/disk); runtime 1/2/1 GiB (RAM/commit/GPU). No user apps closed.
Checkpoint file hash and full network fingerprint must agree before/after.
No policy loading, dataset reading, 8-step expansion, training or adaptation.
