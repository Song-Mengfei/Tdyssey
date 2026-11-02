# D teacher matrix v1 — final

All 8 predeclared runs; native success requires actual coverage >95% and immediate stop. Scope: privileged known-shape simulator action search; development, not held-out generalization.

| Scene | Search | Outcome | Controls | Peak coverage | Final coverage | Peak IoU | Final IoU | Position px | Angle deg | Contact controls | Replay | Forecast | Planning s | Parity S/I/C | Audit | Video |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 100000 | 42 | control_failure | 300 | 61.26% | 61.26% | 44.16% | 44.16% | 1.27 | 100.09 | 28 | 185616 | 96608 | 585.281 | passed/passed/passed | passed (external historical reference: `../videos-v1/scene-100000-search-42/00-scene-100000-search-42/audit.json`) | video (external historical reference: `../videos-v1/scene-100000-search-42/00-scene-100000-search-42/actual-controls-paused.mp4`) |
| 100000 | 43 | timeout | 280 | 61.78% | 61.78% | 44.70% | 44.70% | 3.65 | 101.46 | 37 | 173671 | 96036 | 592.811 | passed/passed/passed | passed (external historical reference: `../videos-v1/scene-100000-search-43/00-scene-100000-search-43/audit.json`) | video (external historical reference: `../videos-v1/scene-100000-search-43/00-scene-100000-search-43/actual-controls-paused.mp4`) |
| 100100 | 42 | control_failure | 300 | 61.55% | 61.55% | 44.46% | 44.46% | 7.19 | 60.61 | 44 | 164600 | 69364 | 477.092 | passed/passed/passed | passed (external historical reference: `../videos-v1/scene-100100-search-42/00-scene-100100-search-42/audit.json`) | video (external historical reference: `../videos-v1/scene-100100-search-42/00-scene-100100-search-42/actual-controls-paused.mp4`) |
| 100100 | 43 | control_failure | 300 | 63.58% | 63.58% | 46.60% | 46.60% | 5.24 | 47.66 | 60 | 164640 | 67700 | 473.076 | passed/passed/passed | passed (external historical reference: `../videos-v1/scene-100100-search-43/00-scene-100100-search-43/audit.json`) | video (external historical reference: `../videos-v1/scene-100100-search-43/00-scene-100100-search-43/actual-controls-paused.mp4`) |
| 100200 | 42 | timeout | 248 | 66.00% | 66.00% | 49.25% | 49.25% | 2.92 | 143.90 | 40 | 142407 | 78036 | 593.887 | passed/passed/passed | passed (external historical reference: `../videos-v1/scene-100200-search-42/00-scene-100200-search-42/audit.json`) | video (external historical reference: `../videos-v1/scene-100200-search-42/00-scene-100200-search-42/actual-controls-paused.mp4`) |
| 100200 | 43 | timeout | 264 | 59.11% | 59.11% | 41.95% | 41.95% | 3.05 | 70.64 | 29 | 152849 | 75312 | 593.494 | passed/passed/passed | passed (external historical reference: `../videos-v1/scene-100200-search-43/00-scene-100200-search-43/audit.json`) | video (external historical reference: `../videos-v1/scene-100200-search-43/00-scene-100200-search-43/actual-controls-paused.mp4`) |
| 100300 | 42 | timeout | 256 | 20.35% | 20.35% | 11.33% | 11.33% | 47.63 | 109.04 | 18 | 151165 | 94048 | 593.999 | passed/passed/passed | passed (external historical reference: `../videos-v1/scene-100300-search-42/00-scene-100300-search-42/audit.json`) | video (external historical reference: `../videos-v1/scene-100300-search-42/00-scene-100300-search-42/actual-controls-paused.mp4`) |
| 100300 | 43 | timeout | 272 | 63.56% | 63.56% | 46.59% | 46.59% | 2.08 | 105.72 | 41 | 161675 | 95572 | 593.954 | passed/passed/passed | passed (external historical reference: `../videos-v1/scene-100300-search-43/00-scene-100300-search-43/audit.json`) | video (external historical reference: `../videos-v1/scene-100300-search-43/00-scene-100300-search-43/actual-controls-paused.mp4`) |

Outcomes: `{"control_failure": 3, "native_success": 0, "protocol_error": 0, "queued": 0, "running": 0, "timeout": 5}`.

Total persisted costs: 2,220 actual controls, 9,040 branch restores, 1,296,623 replay controls, 672,676 forecast controls; 4503.594 summed planning seconds. Final costs.

Deadline attempts: 5; their full work: `{"restores": 92, "replay_steps": 23303, "candidate_steps": 3516}`; discarded partial candidates (subset): `{"restores": 5, "replay_steps": 719, "candidate_steps": 0}`.

Frozen source copy: passed. S/I/C = source provenance / exact initial snapshot / declared evaluator configuration. Independent replay audit is separate from these static checks.

Position/angle errors use the last actually executed D snapshot (body origin = mass center; symmetry one). Angle is the absolute wrapped goal-minus-object angle, in degrees. Running rows describe the latest saved actual state.

Independent replay costs so far (separate): 2,220 controls, 8 reset substeps, 63.250 summed seconds.

Costs include failed and timed-out runs. Per-run evaluation/process time sums are not concurrent elapsed wall time. Preview captures and independent replay costs remain separate. Running reports omit work in the currently executing planning call.
