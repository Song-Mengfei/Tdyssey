# Major experiment lineage

This reading guide summarizes the saved retrospective ledger and later completed screens. It is not a new benchmark. Counts are per branch, not additive training of a single model. A failed implementation, a failed report and a failed scientific gate are different outcomes.

| Ledger / run | Main question or change | Updates | Recorded outcome / limit | Method family |
| --- | --- | ---: | --- | --- |
| RIGID | Retired structured representation | 261, historical-only | Original assets missing; not independently reproduced | Retired representation |
| NATIVE1668 | Full native RGB residual MSE | 1,668 | Did not beat copy-last; best1500/last1668 weights deleted | Native RGB pipeline |
| STAGE3 | Initial micro-fit entry | 0 | Baseline numeric guard stopped before formal fit | Native RGB pipeline |
| MICRO129 | Fit 18 fixed cases | 129 | Fixed-case fit gate failed | Native RGB pipeline |
| MICRO250 | Continue the same 18 cases | +121 | Train gate passed; development/contact failed; not the later 6,000-update lineage | Native RGB pipeline |
| R2C1_FAIL | New motion-sampling launch | 0 | Startup log-name conflict; no training/evaluation | Native RGB pipeline |
| R2C300 | 80% contact/motion exposure | 300 | Motion H4 IoU 0.531987; static/shape failures persist | Native RGB pipeline |
| R2C_SAMPLE | Denoising steps 3 / 8 / 16 | 0 | Motion H4 IoU 0.531987 / 0.504268 / 0.489620 | Native RGB pipeline |
| R2C6000 | Extend the same training lineage | +5,700 | Motion H4 IoU reaches 0.635105; geometry still not rigid | Native RGB pipeline |
| R3A | Recontact data and clean/weak-history preparation | 0 | 912 successful trajectories, 23,984 selected targets; collection/preparation result | Native RGB pipeline |
| R3B_FAIL | Independent worker launch | 0 | Child interpreter failed; formal cursor remained 0 | Native RGB pipeline |
| R3B7499 | First pass over selected new data | 1,499 | Some H4 gains; motion gates did not pass | Native RGB pipeline |
| Original / strict-original contact studies | Restore original kinematic physics and inspect collection/contact events | Consult per-run evidence | Separate from finite-force and damping-modified data | Original-physics contact/data |
| Direct-target / accelerated collection | Fixed action/event and data-selection studies | Consult per-run evidence | No silent mixture with earlier physics | Original-physics contact/data |
| AR1_E1 | Lower-LR original-physics continuation | 1,934 | 48-window H16 IoU about 0.497→0.597; strong O/C still fail | Original-physics contact/data |
| OC_E2 | Second pass, same continuation recipe | +1,934 | 48-window H16 IoU 0.6306; strong O/C 0.4039 | Original-physics contact/data |
| OC_E3 | Third pass, same recipe | +1,934 | 48-window H16 IoU 0.6226; strong O/C regress | Original-physics contact/data |
| OC_DENOISE16 | More inference denoising on strong O/C | 0 | H16 IoU 0.3472→0.2839 | Original-physics contact/data |
| A | True history, native residual MSE | 1,892 | 48-window H16 IoU 0.641489; O rotation/corners and B holes remain faulty | History and region weighting |
| B | Generated history, detached, native residual MSE | 1,892 | 48-window H16 IoU 0.686780; motion H1/shape failures remain | History and region weighting |
| C | True history, region/detail residual weighting | 1,892 | 48-window H16 IoU 0.640077; motion gate not complete; O/B geometry fails | History and region weighting |
| D | Mixed true/generated history plus C objective | 1,892 | 48-window H16 IoU 0.671314; complete support not obtained | History and region weighting |
| CD_DIAGNOSIS | Same 16 O/B cases with real history | 0 | 768 frames; no checked interface mismatch, but AR hole loss persists | History and region weighting |
| SAVED_IDENTITY_AUDIT | Check saved output/checkpoint identity | 0 | Actual updates/new outputs confirmed; improvement not established | Saved-result identity |
| HMF3000 | Higher motion exposure, native objective | 3,000 | Training gains; complete fit fails, C validation and old failures regress | High-motion fitting |
| HMF500 | Earlier checkpoint of the HMF lineage, selected for current app use | 500 | Same 24 O/C/B windows: AR4 macro IoU 0.798170 versus 0.780247 at 3,000; not a complete physical gate pass | High-motion fitting |
| HMF inference diagnostics | History/noise/action sensitivity | 0 | Response diagnostics, not physical counterfactual correctness | High-motion fitting |
| LOSS-AR4 original | Native vs differentiable four-step objective | 0 | Instrumentation preflight failed; retain 33 spent forwards | Rollout objective |
| LOSS-AR4 R1 | Repaired preflight and matched-data A/B comparison | 256 per arm | Training/evaluations complete; Windows filename report failure retained; no complete directional support | Rollout objective |
| LOSS-LIT screen / continuation | BASE, nominal Min-SNR, GDL, FFL | 128 per arm | IO recovery retained; no complete directional support | Noise/gradient/frequency objectives |
| LOSS-DEFINITION | G_ONLY, NATIVE_G, NATIVE_T | 128 per arm | Only NATIVE_G/C short-time subcondition passes; no O/C/B winner | Geometry/transport objectives |
| Earlier frozen-WM / compatibility / supervised pilots | Data, appearance, conditioning, SDK, gradient and checkpoint checks | Consult dated reports | Failed/stopped attempts and constraints retained, not substituted for native baselines | Compatibility and supervised pilots |
| Official policy / simulator search / DPPO | Related policy baseline and direct-simulator learning | Consult profile reports | Separate policy and simulator studies; not proof of WM-assisted improvement | Policy/simulator baseline |


## Source and reproduction

Current data/training/evaluation modules live in `tdyssey/native_rgb_*` and the
explicit driver is `scripts/native_rgb_world_model.py`. That driver retains the
early finite-force protocol; it is not a reproduction of later original-physics
HMF training. The app explicitly uses `native_rgb_original_physics.py`.

The recorded alternatives are callable modules in `tdyssey/wm_objectives/`:
native residual MSE, detached generated histories, region/detail weighting,
AR4 STE, effective-noise Min-SNR/GDL/FFL, and occupancy Dice/spatial W1.
`configs/wm_experiments.json` records the completed attempt settings and limits.
These are method components, not an automatic rerun of all historical jobs.
Private trajectories, full parent state, calibration and RNG/split bindings
are required for matched reproduction. Frozen run launchers, old app versions,
source snapshots, per-frame outputs and local resource receipts are not shipped.
Original local research files remain separate. Unknown measurements remain
unknown; partial image gains are not a rigidity or policy-improvement claim.
