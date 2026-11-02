# Major experiment lineage

This reading guide summarizes the saved retrospective ledger and later completed screens. It is not a new benchmark. Counts are per branch, not additive training of a single model. A failed implementation, a failed report and a failed scientific gate are different outcomes.

| Ledger / run | Main question or change | Updates | Recorded outcome / limit | Archive category |
| --- | --- | ---: | --- | --- |
| RIGID | Retired structured representation | 261, historical-only | Original assets missing; not independently reproduced | 03, retrospective ledger in 10 |
| NATIVE1668 | Full native RGB residual MSE | 1,668 | Did not beat copy-last; best1500/last1668 weights deleted | 03 |
| STAGE3 | Initial micro-fit entry | 0 | Baseline numeric guard stopped before formal fit | 03 |
| MICRO129 | Fit 18 fixed cases | 129 | Fixed-case fit gate failed | 03 |
| MICRO250 | Continue the same 18 cases | +121 | Train gate passed; development/contact failed; not the later 6,000-update lineage | 03 |
| R2C1_FAIL | New motion-sampling launch | 0 | Startup log-name conflict; no training/evaluation | 03 |
| R2C300 | 80% contact/motion exposure | 300 | Motion H4 IoU 0.531987; static/shape failures persist | 03 |
| R2C_SAMPLE | Denoising steps 3 / 8 / 16 | 0 | Motion H4 IoU 0.531987 / 0.504268 / 0.489620 | 03 |
| R2C6000 | Extend the same training lineage | +5,700 | Motion H4 IoU reaches 0.635105; geometry still not rigid | 03 |
| R3A | Recontact data and clean/weak-history preparation | 0 | 912 successful trajectories, 23,984 selected targets; collection/preparation result | 03 |
| R3B_FAIL | Independent worker launch | 0 | Child interpreter failed; formal cursor remained 0 | 03 |
| R3B7499 | First pass over selected new data | 1,499 | Some H4 gains; motion gates did not pass | 03 |
| Original / strict-original contact studies | Restore original kinematic physics and inspect collection/contact events | Consult per-run evidence | Separate from finite-force and damping-modified data | 04 |
| Direct-target / accelerated collection | Fixed action/event and data-selection studies | Consult per-run evidence | No silent mixture with earlier physics | 04 |
| AR1_E1 | Lower-LR original-physics continuation | 1,934 | All48 H16 IoU about 0.497→0.597; strong O/C still fail | 04 |
| OC_E2 | Second pass, same continuation recipe | +1,934 | All48 H16 IoU 0.6306; strong O/C 0.4039 | 04 |
| OC_E3 | Third pass, same recipe | +1,934 | All48 H16 IoU 0.6226; strong O/C regress | 04 |
| OC_DENOISE16 | More inference denoising on strong O/C | 0 | H16 IoU 0.3472→0.2839 | 04 |
| A | True history, native residual MSE | 1,892 | Formal48 H16 IoU 0.641489; O rotation/corners and B holes remain faulty | 05 |
| B | Generated history, detached, native residual MSE | 1,892 | Formal48 H16 IoU 0.686780; motion H1/shape failures remain | 05 |
| C | True history, region/detail residual weighting | 1,892 | Formal48 H16 IoU 0.640077; motion gate not complete; O/B geometry fails | 05 |
| D | Mixed true/generated history plus C objective | 1,892 | Formal48 H16 IoU 0.671314; complete support not obtained | 05 |
| CD_DIAGNOSIS | Same16 O/B cases with real history | 0 | 768 frames; no checked interface mismatch, but AR hole loss persists | 05 |
| SAVED_IDENTITY_AUDIT | Check saved output/checkpoint identity | 0 | Actual updates/new outputs confirmed; improvement not established | 10 |
| HMF3000 | Higher motion exposure, native objective | 3,000 | Training gains; complete fit fails, C validation and old failures regress | 06 |
| HMF inference diagnostics | History/noise/action sensitivity | 0 | Response diagnostics, not physical counterfactual correctness | 06 |
| LOSS-AR4 original | Native vs differentiable four-step objective | 0 | Instrumentation preflight failed; retain 33 spent forwards | 07 |
| LOSS-AR4 R1 | Repaired preflight and matched-data A/B comparison | 256 per arm | Training/evaluations complete; Windows filename report failure retained; no complete directional support | 07 |
| LOSS-LIT screen / continuation | BASE, nominal Min-SNR, GDL, FFL | 128 per arm | IO recovery retained; no complete directional support | 08 |
| LOSS-DEFINITION | G_ONLY, NATIVE_G, NATIVE_T | 128 per arm | Only NATIVE_G/C short-time subcondition passes; no O/C/B winner | 09 |
| Earlier frozen-WM / compatibility / supervised pilots | Data, appearance, conditioning, SDK, gradient and checkpoint checks | Consult dated reports | Failed/stopped attempts and constraints retained, not substituted for native baselines | 02 |
| Official policy / simulator search / DPPO | Related policy baseline and direct-simulator learning | Consult profile reports | Separate policy and simulator studies; not proof of WM-assisted improvement | 01, 11 |

Categories link from the [experiment index](../experiments/README.md). The machine-readable catalog includes every inventoried scientific run/review/preparation directory, including supporting engineering, while nested native stages retain their original relative layout and source indices. Exact data units, physics families, costs, lineage and missing-evidence qualifications remain in the original ledger/evidence files.
