# R3A: real independent trajectories and clean/weak conditioning

The new library contains 912 successful independent trajectories, each with 48 controls: 128 TRAIN and 24 new seen-validation trajectories per letter (T/L/O/C/X/B). Scenes, families, seeds, splits and up to two replacement candidates per slot were frozen before physics collection. No held-out shapes, dynamics lookahead, model construction, checkpoint loading, GPU inference or optimizer work were used.

The selected physical velocity half-life remains 0.1 s (Space.damping = 0.0009765625), with 10 Hz controls, 100 Hz physics, original masses and inertias, finite-force PD control, the original speed cap and walls. The successful trajectories contain 43,776 controls; failed attempts, partial trajectories and reservations incur additional costs.

| Item | Actual |
|---|---:|
| Successful trajectories | 912 |
| Actual attempts, including retained failures | 918 |
| Charged controls, including failures/reservations | 43891 |
| Completed controls, including failed partials | 43886 |
| Actual known physics substeps | 438889 |
| Failed attempts retained | 6 (2 engineering, 4 physical) |
| Maximum penetration among successful trajectories, world pixels | 1.996963 |
| Legal TRAIN H1 windows | 34560 |
| Selected distinct TRAIN H1 windows | 23984 |
| Complete legal TRAIN obvious-motion fraction | 34.699% |
| Selected distinct obvious-motion fraction | 50.000% |

Each selected window appears once in the complete CPU audit epoch; actual optimizer training draws remain zero. There is no repeated quota for the rare category with both translation and rotation. The full legal and valid library remains referenced separately from the selected TRAIN pool. The obvious-motion rule is unchanged: area-centroid translation of at least 2 pixels in the 96-pixel image OR an absolute signed rotation of at least 5 degrees.

The full legal TRAIN library has 11,992 obvious-motion windows among 34,560 (34.699%). The selected pool keeps all of those and 11,992 distinct other windows, giving 50% motion by selection rather than by changing the recorded library. Small movements, static intervals, release phases and wall stalls remain real negative examples. The category other means below both fixed thresholds; it does not mean every such window is completely motionless.

## Independent real movement support

At least one obvious-motion H1 window is supported by 735/768 TRAIN trajectories and 131/144 new seen-validation trajectories. Conversely, 33 TRAIN and 13 validation trajectories have no H1 window crossing either threshold; those trajectories remain in the library. The rare TRAIN both category has 861 different windows from 256 trajectories. X seen-validation has only 2 both windows from 1 trajectory, so its joint translation-and-rotation support is weak and must not be described as broad independent coverage. See data-interface/support-exposure-summary.json.

The table below combines TRAIN and new seen validation. Separate counts for each split are retained in report/physics-audit.json.

| Letter | Trajectories with H1 translation >= 2 pixels | With H1 rotation >= +5 degrees | With H1 rotation <= -5 degrees |
|---|---:|---:|---:|
| T | 97 | 58 | 65 |
| L | 109 | 73 | 79 |
| O | 133 | 64 | 44 |
| C | 115 | 77 | 54 |
| X | 93 | 73 | 59 |
| B | 127 | 74 | 58 |

The complete H1/H4 distributions (zero fraction, P10, median, P90, P95 and maximum), per-trajectory and scenario measurements, 48-control time curves, actual contact runs and wall contacts are recorded in report/physics-audit.json and report/trajectory-audit.csv. These files also report post-contact local and world arcs, yaw, lever arms and normal/tangent projections of the executed target commands. Coverage gaps are disclosed: an unobserved arc or hole is neither covered nor proven unreachable. A finite sample does not establish coverage of every combination.

All five exact-signature duplicate checks found zero groups. One near-initial-scene pair remains within TRAIN (L013 and L121); there are zero declared cross-split near pairs. Both trajectories have independent frozen seeds and distinct state, action and RGB signatures. The similarity is disclosed rather than treated as extra independent coverage.

Observed TRAIN H1 obvious-motion fractions are 54.59% for control anchors 3–21, 0.65% for anchors 22–29 and 28.83% for anchors 30–47. These are actual time bins; release/recontact scenario timings vary. Later pushes retain substantial stalls and are less motion-rich than the first segment. See report/time-bin-motion-summary.json and the complete control-time curves.

Observed exterior-ring arc counts are T55/60, L49/53, O45/56, C59/75, X48/84 and B49/60. Closed interior rings were not initial-contact targets in this exterior-approach collector: O0/37 and B0/19+0/19 interior arcs were observed. Those holes and missing exterior arcs are explicit coverage gaps, with no claim of tested physical impossibility or full Cartesian coverage. Actual local/world contact points, pose-bin diversity and lever/action distributions remain in the audit.

## Inputs and measured CPU contracts

All 1,325 original saved R2B frames render with identical pixels, covering all six letters, holes, rotations, contact and poses near walls. Complete saved float64 states with 15 components supply privileged rendering, motion-based sample selection and audit metadata. They are not network features. The training batch contains five RGB frames (four history frames and the H1 target) and four aligned executed actions, normalized once. Only the four past frames condition history; the fifth frame supplies the native noised target and residual objective. The target is never included in the history conditioner. No persistent RGB bank or library-wide float preload is created. Two workers reuse their renderers, with a bounded 16 MiB RGB cache and 8 MiB state cache per worker and at most two batches in flight. Actual throughput on the new library is recorded in data-interface/two-worker-throughput/throughput.json. The old-library benchmark is historical performance evidence only.

The native FP32 residual MSE and target-noise distribution remain unchanged. The implemented conditioning strategy assigns 50% of microbatches to clean history (sigma_cond=None); the remaining microbatches use per-sample log-uniform history noise in [0.002, 0.05], with zero history offset. Branch, target-noise and weak-history RNG streams are independent and resumable. Seven focused conditioning contracts and the data, sampler, geometry and I/O contracts passed on CPU. These checks do not constitute GPU training or demonstrate prediction quality.

The actual new-library 512-sample, two-worker benchmark measured 30.453 samples/s including startup and 74.020 samples/s after the first batch. The first batch took 10.112 seconds from startup. Each worker reused at most six renderers within its cache limits; maximum in-flight batches were two. Whole-process RSS and a full epoch wall time were not measured. The 512 CPU benchmark samples do not represent model training.

Root inspected all 35 representative strips, the six last actual failed states and all three scientific figures. Before encoding, 1,715 representative frames and 116 failed-prefix control observations matched their original collection RGB hashes exactly. MP4 files are compressed previews; encoded MP4 pixels are not asserted lossless or used as training targets. Original PNG strips and on-demand RGB retain their original pixels. See report/qa/VISUAL_REVIEW.md.

## Failures, storage and next training proposal

Both engineering interruptions remain in the six failed attempts. First, a ledger write failed with WinError 5 before the next engine step, after T completed 48 controls and L000a0 completed 18. An independent audit established that the extra reservation never entered env.step. A versioned I/O-only repair added bounded retries when atomically replacing the same file. Successful T was never recollected; failed L000a0 was retained and its previously frozen a1 was used. The one reserved but unstarted control remains charged.

During the later user-requested handoff to an independent background process, X074a0 stopped at post-save ledger persistence with WinError 5 after exactly one completed control and ten known substeps. The old owner exited. Independent reconciliation retained its partial data and all costs, confirmed zero unknown engine advances and no new unstarted reservation, and preserved all 587 successes already collected. Version 3 changed ownership and continuation selection only; it used the frozen X074a1 instead of restoring or replaying the partial physical state. The independent worker completed the remaining 325 successes, with worker and collector exit codes both zero. It saved local evidence and made no offline chat callback. See background/handoff-receipt.json, background/handoff-independent-review.json and background/run/completion.json.

The other four failures exceeded the fixed two-world-pixel penetration guard and remain physical failures. Their prefixes, failed substeps, costs and videos are retained. The maximum penetration in the summary table covers successful trajectories only; it must not be read as the maximum across failed attempts. No successful trajectory was recollected and no charged work was refunded.

The planning chat separately deleted fourteen earlier failed or redundant binaries under direct human cleanup authorization. The exact deletion receipt, hashes and post-cleanup baseline are retained. Historical audits still describe their original state and were not rewritten. All formal models and original datasets remain.

The R3A stage and its sources occupy 90,091,295 bytes of the 96 MiB allowance. Current shared storage is 1,514,733,777 bytes. Reserving 144 MiB for R3B and 256 MiB for atomic writes leaves 1,287,061,295 bytes below the 3 GiB shared cap. See final-storage.json. Elapsed time is recorded as a statistic; there is no human-imposed model-time cap.

The CPU-validated R3B proposal binds the new dataset, selected pool and cursor identity to the preserved full checkpoint at update 6000. Migration of the dataset and noise streams is explicit; optimizer and learning-rate choices remain subject to parent review. The proposal schedules eight complete epochs over the new library, with observations based on actual exposure. It does not use a 300-update trial as a verdict. Short batches receive weights based on their actual sample counts; AMP skips and successful updates are counted separately. No R3B model work has run, and the new data and conditioning have not yet been shown to improve model predictions.

The bound plan has 1,499 batches of 16 per epoch, 11,992 planned attempts and 191,872 new-library exposures over eight epochs. The lineage would reach 17,992 successful updates only if every new attempt succeeds; AMP skips are separate. The 16-family validation proposal contains six natural, six motion and four near-static cases; the static letters are B/C/T/T, with no claim of static coverage for every letter. GPU training integration and actual optimizer/scaler restoration are still pending R3B work, not CPU-validated accomplishments.

## Review files

- report/videos/index.html — recorded 49-frame clips spanning 0.0–4.8 s at 10 fps. These include the strongest available per-letter examples of translation, each rotation sign and boundary contact, plus fixed-seed random and least-motion trajectories. Unsupported categories are identified.
- report/failure-videos/index.html — retained short failed attempts, no padding or physics replay.
- report/physics-audit.json and report/trajectory-audit.csv — complete distributions and coverage of the recorded data.
- report/motion-distributions.png, report/control-time-curves.png and report/independent-motion-support.png — complete-library scientific figures, including stalls and separate TRAIN/validation support.
- data-interface/interface-gate.json and data-interface/window-catalogue.npz — full legal and selected counts, unique exposures in the complete CPU audit epoch and throughput with two workers.
- data-interface/support-exposure-summary.json — full versus selected motion fractions, independent trajectory support, zero-motion-threshold trajectories and rare-category gaps.
- conditioning-contracts.json; collector-v2-contracts.json; numeric-integrity-review.json; key-interface-review.json — focused CPU and runtime evidence.
- background/handoff-receipt.json; background/handoff-independent-review.json; background/run/completion.json — both preserved engineering interruptions and the completed independent background continuation.
- configs/native_rgb_r3b_proposal.json and r3b-proposal-check.json — the proposal bound to the actual new library and its CPU-only validation.

After sending the authorized callback and saving its receipt, STOP waiting_parent. Review of the recorded movement and videos must precede R3B training.
