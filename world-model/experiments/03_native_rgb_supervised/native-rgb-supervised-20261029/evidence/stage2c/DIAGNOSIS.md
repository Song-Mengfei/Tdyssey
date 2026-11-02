# Stage2C independent seen-letter validation diagnosis

162/162 fixed seen_val trajectories and10368/10368 controls are present. The independently bound2px physical audit passed with0 invalid transitions and103680 recorded100Hz substeps. All raw data remain unchanged. H1=9882/9882, H4=9396/9396, with complete64-control H4 anchor range3..60 and all7 history/future transitions validated.

Raw controls:6090 moving,3570 near-static,708 intermediate. H1 candidates:6022 moving(60.94%),3408 near-static(34.49%),452 intermediate. All18 letters meet the40%/45% diagnostic target on this independently sampled validation set, but this is a distribution description, not a criterion for deleting frames or resampling. Original2A M failure remains preserved.

| Frozen panel | Unique windows | Independent trajectories | H1 moving/static/intermediate | H4 endpoint moving | H4 path any moving |
|---|---:|---:|---|---:|---:|
| combined | 319 | 162 | {'intermediate': 30, 'moving': 190, 'near_static': 99} | 229 | 228 |
| development | 36 | 32 | {'intermediate': 1, 'moving': 31, 'near_static': 4} | 32 | 32 |
| extra | 157 | 157 | {'intermediate': 18, 'moving': 90, 'near_static': 49} | 116 | 115 |
| natural | 162 | 162 | {'intermediate': 12, 'moving': 100, 'near_static': 50} | 113 | 113 |

Natural162 includes one independently hashed H4-legal anchor per trajectory. Anchor min3/max60/mean30.0679; position bands3–16/17–30/31–44/45–60 contain42/49/30/41 windows. This is the predeclared balanced-scene sampling distribution, not real-world operation frequency.

Extra157/162 requested slots:54 allocated moving,18 rotation,18 boundary-moving,13 blocked-proxy,18 near-static-contact,18 near-static-no-contact,18 intermediate. Exactly5 blocked-proxy slots(C/F/M/S/T) have no remaining support after natural selection and stay empty. No category substitution, new seed, replacement trajectory or physical resampling occurred. All selected windows retain crossing conditions.

Combined319 windows produce548 conditional memberships; development is an overlay of36 selected windows, yielding355 panel memberships. The actual future inference universe remains319 unique(trajectory,anchor) pairs. Development has36 windows but32 independent trajectories;18 come from natural and18 from extra H1-moving truth, including rotation/boundary selections as allowed.

Each of157 trajectories has one natural plus one extra selected anchor;5 have only natural. Of157 same-trajectory window pairs,35 overlap H4 states,31 overlap H4 transitions,17 overlap future transitions and19 overlap H1 states. Development has4 same-trajectory pairs,2 with H4 state/transition overlap and0 with future/H1-state overlap. These are correlated windows, not319 independent physical trajectories.

H1/H4 labels are distinct: combinedH1 moving190 versusH4 endpoint moving229/path-any-step moving228.13 H1-near-static windows contain later moving within their4-step path;39 H1-nonmoving windows have a moving4-step endpoint. No H1-static label is presented as proof of four-step stillness. Actual boundary uses min(start,end) wall gap<=32 including negative values; exact gap bands and four signed gaps are retained in the index and raw statistics.

Original frozen examples remain T-moving, M-requested-blocked-with-explicit-moving-fallback, P-rotation, V-near-static-contact:4 rows×8 consecutive saved frames and2 GIFs. M has no supported blocked-proxy example and is not called blocked. PNG was visually inspected: white background, gray object, blue pusher, no goal/red training marker.

The parent-authorized separate supplement selects frozen extra E window seen-val-2c-093:31 by minimum(existing selection_hash,trajectory_id,anchor), among14 actual proxy-labeled extra windows. Saved frames28..35,1 row/1 GIF; H1 translation0.00878068px,rotation0,actual minimum gap−0.09880035px,contact=true; H4 endpoint translation0.03475079px. It is a wall-blocking proxy, without measured wall force/arbiter evidence. Original4-case files, M-missing disclosure and all indices are untouched.

Operational disclosure: the frozen report CLI requires a fresh directory while root coverage already owned report/coverage. It ran once unchanged into report-main-staging(exit0), then only its16 owned files were moved to report. Both resolved parents were verified strictly inside stage2c, no target was overwritten, all16 hashes remained identical, and original generated JSON/CSV contained no stale absolute staging paths. Actual command and relocation provenance are intentionally preserved in report-execution.json/report-relocation.json.

Separate coverage incident, disclosed by root: the original offline coverage guard omitted source-snapshot/ in a strict snapshot-name comparison and rejected before coverage output. Root preserved the original error log and created a separately hashed offline-repair script/diff, retaining all hash/protocol/counter checks. This was an offline report guard error, not a physical collection failure; all37 frozen sources and the index policy remain unchanged.

All37 frozen source/config snapshots were checked after report rendering and the supplement. Strict index bindings retain original manifest, physics-audit,source-identity and pre-step policy hashes. No model/denoiser, optimizer, training iterator or training-index draw ran. Saved-state rendering and evaluation-index instantiation performed0 additional physics controls. Generic70/10/10/10 feasibility fields in the frozen statistics helper are hypothetical diagnostics; actual evaluation panels are the319 windows above.

Primary evidence: report/seen-val-report.json;report/h1/real-transition-statistics.csv;report/indices/indices.json;report/indices/INDEX_REPORT.md;report/examples/;report/supplementary-blocked/;report-command.log;report-execution.json;report-relocation.json.
