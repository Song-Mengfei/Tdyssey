# Frozen input comparison v3: complete, with residual L rollout failure

The one separately authorized comparison on clean source
`ff1c5b43a641bbfd6ff4b99b6f08e8b67ca0e66a` completed M, T and L. M and the
predeclared T gate passed. One frozen model produced all 2268 available forecasts
with 6804 Euler calls; 36 originally missing calls remain missing. No error,
repair, retry, resume, extra probe, budget extension, physics or training occurred.
Native ownership, final journal/content audit and empty-Job cleanup passed.
The [fixed v3 account](GPC_INPUT_COMPATIBILITY_ATTEMPT_V3.md) is now spent.

T1 restores early object/full-shape measurability on these saved contexts: AR-H1
goes from 0/48 to 48/48 on T and 0/141 to 141/141 on L. B passes the T shape and
conditional limited-error screens at H1/H2/H4. L subsequently degrades to
113/141 at H2 and 78/141 at H4, below the fixed 127/141 shape requirement; H4
pusher P90 is 14.383 world pixels. Completing L is therefore not a passing L
compatibility result. These are reused development scenes, not independent
generalization or validated dynamics/termination evidence.

## Authority, identity and timing

The user explicitly authorized one original-protocol run after incremental
engineering verification, with unchanged budgets/gates, stop on error and no
automatic retry. The scientific proposal remains SHA-256
`2bfff44111d41f5c54a626b717ec2ea35f4bee5051b9fbeb19926f8fa4dbacf2`;
T1 remains `800751b7a3e32df4fee5660d1ea22c7ee0fb45b51a7b41632d420189238a7838`.
The exact 35,043,916-byte local checkpoint remains
`4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
Original saved inputs, upstream source, decoder/config, cases, seeds, sampler,
missing branch, gates and resource/storage limits were unchanged.

| Item | Actual value |
| --- | --- |
| Attempt / authority | `wm-input-compat-eval-v3` / `wm-input-compat-eval-v3-20261025-user-authorized` |
| Registry | `artifacts/wm-input-compat-eval-v3-registry/` |
| Output / ledger | `artifacts/wm-input-compat-eval-v3-20261025/` / `artifacts/wm-input-compat-eval-v3-ledger-20261025/` |
| Launch seal | 2026-10-25 16:42:50.154 SGT |
| Final guardian report timestamp | 2026-10-25 17:12:11.067 SGT |
| Seal-to-report wall time | 1760.913 seconds, below the original 1800 seconds |
| Recorded pre-final-audit elapsed | 1737.594 seconds |
| Guardian / launcher / worker PID | 1352 / 10672 / 13640 |
| Nonce | `0261db33f9b14d4590c3a778f1549ce2` |

The final content audit accounts for approximately 23.319 seconds between the
recorded elapsed value and final report timestamp. It is included in the
seal-to-report figure above, not a granted extension. The original deadline
started before setup/binding/M; the worker received only its rounded-down
remaining time. There was exactly one launch and one model load, with two
compute threads and a native kill-on-close 3 GiB Job.

The necessary 106 fake/native-process checks passed before the identity commit;
no additional real transform, decoder or model probe ran before launch. Read-only
preflight verified 279 bound asset/inventory paths, the 20 pinned upstream files,
clean exact HEAD, absent new roots/global lock and sufficient native resources.
The native guardian and worker repeated their own startup/ownership/binding checks.

## Stages, denominators and measurement gate

| Scope | Planned | Available / completed | Original missing |
| --- | ---: | ---: | ---: |
| M, both arms | 1072 | 1056 | 16 |
| T AR | 384 | 384 | 0 |
| T TF | 192 | 192 | 0 |
| L AR | 1152 | 1128 | 24 |
| L TF | 576 | 564 | 12 |
| All forecasts | 2304 | 2268 | 36 |

M has 528 available rows per arm: 388 L and 140 T. All object, shape, pusher and
coverage fields are locally valid and all truth labels are present. The four
arm/shape cells satisfy the original position/angle/coverage P90 bounds and zero
interval false positives. Values below are P90, in world pixels, wrapped degrees
and absolute coverage fraction; pusher error is reported, not an added M gate.

| Arm / shape | Position | Angle | Pusher | Coverage |
| --- | ---: | ---: | ---: | ---: |
| A / L | 1.1342 | .3288 | 1.3714 | .011119 |
| B / L | 1.0169 | .3311 | 1.3709 | .012037 |
| A / T | 1.2478 | .0524 | 1.2957 | .025219 |
| B / T | 1.2982 | .0491 | 1.2957 | .026451 |

Each arm has 490 interval non-success decisions, 38 unknowns and zero interval
positives. Precision is undefined, not 100%; all true-positive calibration cases
remain abstentions. There are no B-only interval positives. Passing M does not
validate generated-image interval precision or the +/- .03 development assumption.

## AR shape and conditional error screens

The T gate uses AR only: B has 48/48 joint object/full-shape validity and 48/48
pusher validity separately at H1/H2/H4, with no disallowed contact-stratum
regression. A has 0/48, 24/48 and 34/48 joint validity, respectively. TF's 192 T
rows are excluded from this gate.

| Shape / horizon | B joint validity | Fixed required | Shape screen | Limited-error screen | Pusher P90 |
| --- | ---: | ---: | --- | --- | ---: |
| T / H1 | 48/48 | 44 | pass | pass | 2.2497 |
| T / H2 | 48/48 | 44 | pass | pass | 2.8386 |
| T / H4 | 48/48 | 44 | pass | pass | 2.3874 |
| L / H1 | 141/141 | 127 | pass | pass | 1.7440 |
| L / H2 | 113/141 | 127 | fail | fail | 3.5214 |
| L / H4 | 78/141 | 127 | fail | fail | 14.3833 |

Each L horizon retains all 144 planned rows, including three missing noise rows
from L01/hold. Pusher P90 uses all 141 available L / 48 T rows. L position, angle
and coverage P90 use only 141, 113 and 78 measurable rows at H1/H2/H4; their
respective coverage errors are .047965, .048362 and .048600. The small errors on
the surviving subset do not rescue the failed full-denominator shape screens.
L-H4 additionally fails the <=5-pixel pusher P90 condition. Cross-arm pose error
at H1 is undefined because A has no measurable object/full-shape rows.

A's L joint counts are 0/141, 3/141 and 4/141 at H1/H2/H4. The L-H2/H4 common
measurable pose subsets contain only three and four rows, each from one anchor,
so their conditional errors cannot support a broad cross-arm accuracy claim.
T's corresponding common subsets are 24/48 and 34/48; position P90 A-to-B is
5.211-to-1.233px at H2 and 3.418-to-.630px at H4 on those same measurable subsets.

At L-H4, the contact-by-H4 stratum has B joint validity 0/27 rows (nine physical
branches/nine anchors), compared with 78/114 noncontact rows. The >5-degree
rotation stratum has 0/9 (three branches/three anchors). These are descriptive
saved-scene strata; rejection does not by itself distinguish generation error
from generated-style measurement inadequacy or establish a general causal
contact-dynamics conclusion.

## Teacher-forced context and terminal limitations

TF supplies saved real intermediate observations under the original conditioning
factor. It is excluded from AR gates and ranking. B's L joint validity is
118/141 at H2 and 103/141 at H4, versus AR's 113/141 and 78/141. Both TF cells
still fall below the original 127/141 shape requirement. Its L-H4 pusher P90 is
1.3151px on all 141 rows, versus 14.3833px for AR on the same available rows.
The TF L-H2 pusher P90 is 1.9536px on the A/B-common 140/141 subset; AR's 3.5214px
uses 141/141, so these H2 values are not a strict equal-subset comparison.
The H4 improvement under real context implicates generated-feedback/distribution
effects, while residual TF shape rejection prevents attributing everything to
compounding or validating the frozen model as a reliable dynamics model.

All 63 available physical outcomes at each primary horizon have true coverage
below strict >.95 success (maximum .942857). Alongside M's zero interval-positive
predictions, this prevents estimating successful terminal precision/recall from
these data. Unknowns and rejected generated measurements remain abstentions,
and no point-coverage threshold replaces the three-state terminal rule.

The stopping-consistent A/B common ranking subset is 0/16 planned anchors at
every horizon, so its comparison is undefined. Raw endpoint ranking has common
subsets 0/16 at H1, 1/16 at H2 (T13) and 2/16 at H4 (T12/T13), with zero common
L anchors. A, B and hold have identical physical reward .323729 and zero regret
on those tiny eligible T subsets; raw H3 similarly has two common T anchors.
There is no demonstrated action-selection advantage. Full physical hold/uniform
baselines retain their separate denominators in `ranking.json`; no model subset
is compared against a full-population baseline, and TF is never ranked as AR.

## Execution evidence, resources and cleanup

`completion.json` is authoritative for final complete status, counts and stages.
`summary.json` was written before `budget.finish()` and may retain the earlier
active budget view; it is not a second or failed run. The durable chain contains
17,288 events / 7,121,623 bytes and 6373 completed outputs, ending with the unique
`output_complete` for `results/completion.json`. Its SHA-256 is
`c2fdaf2be86110139edce0fe72ff611dcd5b524831236e14690529dbda0a847c`;
tip `0e4faee52ea2d58e082f24c305bafde8e8dd3c18d61a7b364357c8b8f0db9fd9`.
Guardian reason is null; worker and launcher exited 0. Final audit passed with
no pending result publication, followed by prepaid terminal report/log writes.
All 2268 call reservations have completions and paired RNG checks match; no
pending output/call or temporary remains. Thirty independent native/ledger
reconciliation checks passed without repeating full payload hashes.

The quiescent audit before those final guardian writes measured:

| Category | Charged bytes | Physical bytes | Original cap |
| --- | ---: | ---: | ---: |
| Results | 266,079,232 | 259,599,582 | 268,435,456 |
| Telemetry | 14,848,291 | 14,586,665 | 16,777,216 |

Charges retain unused maximum bounds and prepaid slots/scratch. Post-run
telemetry is 14,586,782 bytes: the 117-byte difference is exactly the final
`exited` monitor event. Prepaid stop/report slots do not add physical growth.
Final charged telemetry is 14,848,408 bytes; charged results stay 266,079,232.
The largest measured counter-update body was 1.1037 ms, excluding acquisition
and setup, not an upper bound on every OS lock wait. The 0.5-second acquisition
timeout and all sampling/cap settings remained unchanged.

Native sampled minima were 2,923,810,816 bytes available RAM,
12,635,152,384 commit headroom, 6,224,347,136 GPU free and 14,475,526,144 disk free.
Sampled process peaks were 1,358,249,984 working-set and 2,220,331,008 private
bytes, below the original 2/3 GiB bounds. Sampled current CUDA allocated/reserved
maxima were 95,563,264 / 136,314,880 bytes. The backend's tracked lifetime
allocated peak was 109,831,168 bytes (104.743 MiB), with 136,314,880 bytes (130 MiB)
reserved; these differ from maxima of sampled current allocation. All 7960 native
resource samples respected the original thresholds. Seven bounded live-status
unavailability samples used only the existing last-snapshot freshness rule;
the final complete status was freshly read after exit. No stop was triggered.
The retained handles verified the owned Job empty before close and the global
model lock was released. Numeric PIDs can be reused after exit; a later process
with the same number is not evidence that the original instance survived.

Worker output retains a nonfatal library cache-directory warning, followed by
the complete status. The run used the bound local checkpoint; no cache repair,
environment change or extra execution followed that warning.

## Preservation, evidence and limits

All 895 protected files / 204,135,829 bytes across 31 roots are unchanged, with
zero additions, missing files, changes or membership differences. This includes
all 806 earlier files, all 89 incremental-engineering evidence files, all four
old claims, both failed formal attempts and 150 B/C files. The before inventory
and baseline-seal hashes are unchanged.

The three v3 formal roots contain 6386 files / 274,186,364 bytes, including two
new claims totaling 2338 bytes. All six old/new claims total 7031 bytes and remain
spent. The 2268 raw NPY files are each 110,720 bytes: 251,112,960 serialized bytes
for the fixed 250,822,656-byte float32 CHW payload, plus 128 bytes per header.
Byte/stat inventory did not decode those arrays or regenerate images.

Read-only follow-up metadata is separate in
`artifacts/wm-input-compat-eval-v3-audit-20261025/`: `preflight.json`,
`prelaunch-focused-tests.json`, `launcher.stdout.json`, `final-v3-inventory.json`,
`after.json`, `reconciliation.json` and `guardian-ledger-reconciliation.json`.
The latter passed all 30 checks, SHA-256
`66a5d1c6684ef4f746dfce10b0e606692c4251216c1b410567efe267f3544a68`.
Inventory SHA-256 is
`808dff4a03949df7c218b4e368012c81caec97cf3207ebed74d9d93205d54b90`;
preservation reconciliation SHA-256 is
`0dc6ab7b7ec5d75d84162a189a1da3e216f91e52127ccce7532551aa2044c353`.
Audit metadata is outside the guardian policy roots and is not mislabeled as
formal telemetry. Original formal outputs were never edited by analysis.
The compact numeric-only follow-up is `numeric-analysis.json` (41,184 bytes),
SHA-256 `52ec5722ae818d303a2e6235357809328d46ac7f57bcd33f4ee34530040e861e`.

The early appearance/input mismatch is materially reduced on these reused saved
contexts. L's AR deterioration and later pusher error remain unresolved. M's
real-image pass cannot establish measurement accuracy on rejected generated
frames, and small conditional pose errors cannot establish accurate contact
dynamics. No success-precision or generalization claim follows; neither this
single T1 comparison nor a failed later horizon proves fine-tuning necessary
or sufficient. No physical operation, policy sample or optimizer update ran.
Any temporal/adaptation follow-up requires a separate proposal and authority.
