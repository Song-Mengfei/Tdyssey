# R1: small-motion diagnosis from existing records

The small H1 translation is real in the saved physical states, not just a low moving percentage or a body-origin/unit mistake. Across all 24,156 legal TRAIN H1 windows, area-centroid translation has median0.4446, P95 0.9313 and maximum0.9623 at96px resolution. There are zero examples at or above1,2or3 image pixels. Reweighting70% moving cannot create missing amplitude support.

## Existing movement, with denominators kept separate

| Population | N | Translation median / P95 / max (96px) | Rotation median / P95 / max (degrees) | >=1px translations |
|---|---:|---|---|---:|
| H1(0.1s) | 24156 | 0.4446 / 0.9313 / 0.9623 | 0.7648 / 3.9281 / 5.6372 | 0 |
| H4endpoint(0.4s) | 22968 | 1.8722 / 3.7083 / 3.7884 | 3.4563 / 14.9516 / 21.4483 | 14171 |

H1 moving14692 (60.82%), near-static8447 (34.97%) and intermediate1017 (4.21%) are the unchanged original definitions. Moving median translation is0.7985px. The H1 decomposition is3645 translation-threshold-only,1841 rotation-threshold-only,9206 mixed,1017 intermediate and8447 near-static. These use the original2worldpx/1degree gates: threshold-only does not mean mathematically zero motion in the other component.

All18 microfit windows enter solely through rotation>=3degrees. Their translation is0.4817–0.8041px, rotation3.0521–4.6706degrees, and none reaches1px translation or5degrees rotation. H1 has only20 windows>=5degrees and zero>=10degrees.

| Letter | H1 median / P95 / max px | H1 rotation P95 deg | H4 median / P95 / max px |
|---|---|---:|---|
| A | 0.5366 / 0.9328 / 0.9374 | 4.1026 | 2.2229 / 3.7108 / 3.7496 |
| B | 0.5844 / 0.9339 / 0.9623 | 3.2516 | 2.4844 / 3.7162 / 3.7500 |
| C | 0.3801 / 0.9278 / 0.9512 | 3.4863 | 1.5699 / 3.6991 / 3.7505 |
| E | 0.5003 / 0.9340 / 0.9489 | 3.5718 | 1.9328 / 3.7260 / 3.7584 |
| F | 0.4158 / 0.9308 / 0.9401 | 4.2045 | 1.6728 / 3.7117 / 3.7501 |
| H | 0.4424 / 0.9126 / 0.9377 | 3.9474 | 1.7839 / 3.6299 / 3.7505 |
| I | 0.3764 / 0.9285 / 0.9375 | 3.5792 | 1.6922 / 3.6966 / 3.7498 |
| J | 0.5534 / 0.9355 / 0.9374 | 3.8056 | 2.2779 / 3.7331 / 3.7497 |
| L | 0.3956 / 0.9273 / 0.9373 | 4.3025 | 1.6553 / 3.6802 / 3.7490 |
| M | 0.3445 / 0.9234 / 0.9371 | 3.8211 | 1.5174 / 3.6538 / 3.7484 |
| O | 0.4201 / 0.9321 / 0.9452 | 2.5519 | 1.9825 / 3.7167 / 3.7533 |
| P | 0.3922 / 0.9241 / 0.9378 | 4.4558 | 1.6279 / 3.6731 / 3.7505 |
| S | 0.5611 / 0.9329 / 0.9499 | 3.9400 | 2.0227 / 3.7234 / 3.7622 |
| T | 0.5274 / 0.9272 / 0.9393 | 3.4647 | 2.1025 / 3.6782 / 3.7559 |
| U | 0.5923 / 0.9350 / 0.9528 | 3.8509 | 2.3487 / 3.7268 / 3.7513 |
| V | 0.3823 / 0.9303 / 0.9375 | 4.4265 | 1.5908 / 3.7184 / 3.7499 |
| X | 0.4961 / 0.9286 / 0.9528 | 3.9092 | 1.9156 / 3.6931 / 3.7500 |
| Y | 0.3711 / 0.9332 / 0.9530 | 4.4767 | 1.5463 / 3.6945 / 3.7884 |

Full continuous distributions by all18letters x4strata, actual contact, measured boundary/corner/gap, action phase/intent, and motion decomposition are in motion-amplitude-statistics.json. H4 strata/boundary/phase refer to the measured anchor; H4 contact means any contact during its four future controls and its motion decomposition uses endpoint displacement/angle. These are not initial-location coverage claims.

## Contact and controller evidence

H1 contact has16883 windows, median0.7438px and max0.9623px; noncontact has7273, median0 and max0.0520px. Observed near-static contact has1176 windows. Only540 meet the existing likely-wall-blocked proxy (median0.00122px); no object-wall arbiter impulse was recorded, so they cannot be relabelled as causally confirmed wall-blocked. Real stalled contacts and that wall proxy stay separate.

| H1 action phase | N | Median / max translation px | Reconstructed first-force saturation count |
|---|---:|---|---:|
| approach | 396 | 0.2123 / 0.2682 | 0 |
| blocked_escape_recontact | 501 | 0.0000 / 0.0022 | 17 |
| blocked_escape_retreat | 129 | 0.0001 / 0.0323 | 1 |
| contact_recovery_route | 824 | 0.0000 / 0.9328 | 152 |
| push | 13011 | 0.8256 / 0.9623 | 12403 |
| push_again | 245 | 0.6628 / 0.9489 | 191 |
| recontact | 4488 | 0.0000 / 0.9226 | 155 |
| retreat | 1570 | 0.0000 / 0.0890 | 262 |
| sliding | 2992 | 0.3427 / 0.8204 | 27 |

The25,344 raw control targets are far from the pusher: median122.0958worldpx, P90 224.0000. The H1-eligible subset has a different denominator and median145.7221. Targets are absolute0..511 coordinates, with requested distances32/96/224; source clipping is to arena coordinates, not an8px displacement cap. Target distance is not object movement.

Measured control-boundary speeds peak at100.0000worldpx/s for the pusher and50.0098 for the physical object CoG. The200worldpx/s pre-step speed cap is not approached at these recorded boundaries; this does not prove every unlogged substep stays below it. All25,740 boundary state rows have force/torque zero because they are sampled after integration.

Source-formula reconstruction for the first substep only gives force saturation at10000 in13208/25344 controls (52.11%), and12775/16883 contact H1 controls (75.67%);17/18 microfit windows are saturated. This is reconstructed from float32 targets and boundary velocities, not a measured10-substep force trace. Nine subsequent substeps and collision/bias impulses were not logged.

## Confirmed source mechanism and conditional explanation

The local controller applies bounded PDforce, then a pre-step speed cap, then ten0.01s physics steps: native_rgb_env.py (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_env.py:188`) lines19–25,100,196–209. Both pusher and object have mass1; non-T moments are geometry-based, while T retains legacy moment3000 and physical CoG(0,45). The inherited upstream setup sets gravity0 and space.damping=0 (pusht_env.py291), while PD already uses kp100/kv20.

The installed engine identifies as Pymunk6.9.0/Chipmunk7.0.3-dfc2fb8. Its [step integration](https://github.com/vibloteket/munk2d/blob/dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970/src/cpSpaceStep.c) uses damping**dt; [body integration](https://github.com/vibloteket/munk2d/blob/dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970/src/cpBody.c) damps old velocity before adding force/mass*dt, resets forces, and includes positional bias separately. Thus zero means old velocity is erased at every dynamic-body integration, not no damping. The original kinematic pusher skips dynamic velocity integration; the new dynamic pusher inherits this damping setting.

With zero damping, bounded force10000, mass1 and dt0.01, an isolated pusher receives at most100worldpx/s of fresh force velocity. An ideal equal-mass, normal, nonrotating, unconstrained contact references roughly50worldpx/s, or5worldpx/.9375imagepx per0.1s. This is only a conditional scale explanation. It is not a global bound: B/train-2b-163:15 actually moves5.132211worldpx, above5, with endpoint object speed49.9991. Cached/contact impulses, unlogged transient velocities and positional bias prevent turning boundary speeds into a universal displacement bound. A controlled ablation is still needed to separate causes.

Friction needs care: native_env102/official_env99/upstream326,365 assign body.friction=1. The contact coefficient API is Shape.friction (installed shapes.py207–220), and those source paths do not call that setter. The saved data do not record actual shape material values. Do not declare high friction or effective coefficient1 as a proven cause; log shape friction/elasticity in any approved R2.

## Is1/2/3px H1 reachable by actions alone?

| H1 target | Required mean centroid speed (worldpx/s) | Existing supporting examples | Conclusion |
|---|---:|---:|---|
|1px|53.3333|0|Not demonstrated; no global impossibility proof|
|2px|106.6667|0|Not demonstrated; force-saturated targets alone do not increase force magnitude|
|3px|160.0000|0|Not demonstrated; needs an approved paired test, not renamed thresholds|

Continuous verified pushing and eccentric contact can improve the fraction of time spent in productive contact. They cannot be promised to produce the requested2–4px or4–8px translation, or5–15degree rotation, under this unchanged protocol. Current saturated pushes already approach the measured plateau. The source/evidence rules out a simple claim that making the target farther necessarily solves it.

The current R2 request contains only P0D0 (unchanged v1) and P1D1 (a separately versioned damping1 arm); both retain force_limit10000 and pre-step speed_limit200, with every other physical/control parameter held fixed. Retained momentum plus the existing PDvelocity feedback may increase movement; feasibility and the2px substep penetration gate remain untested. Force20000 and force40000 are NOT requested in this R2. They remain later, separately reviewed conditional references only (ideal equal-mass references1.875px and3.75px). Actual commanded force also depends on kp*target_error-kv*velocity and the0..511 target-coordinate range, so raising a cap does not guarantee either that force or those amplitudes. Any later change needs its own protocol hash, independent directory and approval; old data must not be mixed. No parameter was changed here.

Changing the prediction interval to0.4s would use existing larger H4 movement, but it changes task timing/action semantics. It cannot be presented as H1 by skipping frames or replaying faster. It is not executed or selected here.

## Alignment and remaining uncertainty

The audit uses area centroid p+R*c_area, not body origin and not physical CoG. T has local area centroid(0,40.7142857) versus physical CoG(0,45). Independent recording-alignment-audit.json corroborates the indices, actions,96/512scale and timing; its cached-mask comparison has up to0.253184px raster/occlusion centroid discrepancy. Subpixel geometric translation is not a claim of zero pixel change: rotation and antialiasing can change edges. No saved-state, normalization or frame-index mistake explains the observed H1 amplitude gap in the checked records.

The amplitude gap and zero-damping velocity-memory removal are confirmed. Their quantitative interaction with the force cap, contact geometry and solver remains unresolved; no single sole cause has been proven. Corners/holes remain rounded in the existing250-step outputs, and DEV is worse than hold. More training is not guaranteed to repair shape. Neither the narrow microfit pass nor70% resampling solves the amplitude or shape acceptance question.

R2 must first demonstrate actual amplitudes, genuine contact/boundary behavior, all executed-slot denominators and normal1x continuous videos under parent approval. No new physical controls, environments, images, models, optimizers, protocol changes or training were executed. Existing39680 supervised/74240 total controls remain unchanged.
