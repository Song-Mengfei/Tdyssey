# D teacher matrix v2 — final

All 8 predeclared runs; native success requires actual coverage >95% and immediate stop. Scope: privileged known-shape simulator action search; development, not held-out generalization.

| Scene | Search | Outcome | Controls | Peak coverage | Final coverage | Peak IoU | Final IoU | Position px | Angle deg | Contact controls | Replay | Forecast | Planning s | Parity S/I/C | Audit | Video |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 100000 | 42 | timeout | 288 | 63.37% | 63.37% | 46.38% | 46.38% | 1.25 | 104.86 | 35 | 161393 | 89774 | 591.956 | passed/passed/passed | passed (external historical reference: `../videos-v2/scene-100000-search-42/00-scene-100000-search-42/audit.json`) | video (external historical reference: `../videos-v2/scene-100000-search-42/00-scene-100000-search-42/actual-controls-paused.mp4`) |
| 100000 | 43 | control_failure | 300 | 60.53% | 60.53% | 43.40% | 43.40% | 1.15 | 98.23 | 40 | 166120 | 89233 | 548.529 | passed/passed/passed | passed (external historical reference: `../videos-v2/scene-100000-search-43/00-scene-100000-search-43/audit.json`) | video (external historical reference: `../videos-v2/scene-100000-search-43/00-scene-100000-search-43/actual-controls-paused.mp4`) |
| 100100 | 42 | control_failure | 300 | 61.87% | 61.87% | 44.79% | 44.79% | 4.96 | 61.12 | 50 | 161512 | 83781 | 558.342 | passed/passed/passed | passed (external historical reference: `../videos-v2/scene-100100-search-42/00-scene-100100-search-42/audit.json`) | video (external historical reference: `../videos-v2/scene-100100-search-42/00-scene-100100-search-42/actual-controls-paused.mp4`) |
| 100100 | 43 | control_failure | 300 | 56.07% | 56.07% | 38.96% | 38.96% | 2.47 | 89.86 | 47 | 163376 | 79411 | 447.441 | passed/passed/passed | passed (external historical reference: `../videos-v2/scene-100100-search-43/00-scene-100100-search-43/audit.json`) | video (external historical reference: `../videos-v2/scene-100100-search-43/00-scene-100100-search-43/actual-controls-paused.mp4`) |
| 100200 | 42 | timeout | 272 | 63.22% | 63.22% | 46.22% | 46.22% | 1.15 | 103.66 | 31 | 150383 | 81312 | 593.110 | passed/passed/passed | passed (external historical reference: `../videos-v2/scene-100200-search-42/00-scene-100200-search-42/audit.json`) | video (external historical reference: `../videos-v2/scene-100200-search-42/00-scene-100200-search-42/actual-controls-paused.mp4`) |
| 100200 | 43 | timeout | 264 | 63.09% | 63.09% | 46.08% | 46.08% | 0.18 | 107.24 | 36 | 144526 | 80286 | 593.092 | passed/passed/passed | passed (external historical reference: `../videos-v2/scene-100200-search-43/00-scene-100200-search-43/audit.json`) | video (external historical reference: `../videos-v2/scene-100200-search-43/00-scene-100200-search-43/actual-controls-paused.mp4`) |
| 100300 | 42 | timeout | 256 | 63.61% | 63.30% | 46.64% | 46.30% | 1.21 | 128.18 | 29 | 137165 | 86273 | 593.749 | passed/passed/passed | passed (external historical reference: `../videos-v2/scene-100300-search-42/00-scene-100300-search-42/audit.json`) | video (external historical reference: `../videos-v2/scene-100300-search-42/00-scene-100300-search-42/actual-controls-paused.mp4`) |
| 100300 | 43 | timeout | 256 | 62.65% | 62.65% | 45.62% | 45.61% | 3.27 | 117.42 | 29 | 141818 | 87640 | 593.872 | passed/passed/passed | passed (external historical reference: `../videos-v2/scene-100300-search-43/00-scene-100300-search-43/audit.json`) | video (external historical reference: `../videos-v2/scene-100300-search-43/00-scene-100300-search-43/actual-controls-paused.mp4`) |

Outcomes: `{"control_failure": 3, "native_success": 0, "protocol_error": 0, "queued": 0, "running": 0, "timeout": 5}`.

Total persisted costs: 2,236 actual controls, 9,053 branch restores, 1,226,293 replay controls, 677,710 forecast controls; 4520.091 summed planning seconds. Final costs.

Deadline attempts: 5; their full work: `{"restores": 59, "replay_steps": 14869, "candidate_steps": 1909}`; discarded partial candidates (subset): `{"restores": 5, "replay_steps": 597, "candidate_steps": 0}`.

Frozen source copy: passed. S/I/C = source provenance / exact initial snapshot / declared evaluator configuration. Independent replay audit is separate from these static checks.

Position/angle errors use the last actually executed D snapshot (body origin = mass center; symmetry one). Angle is the absolute wrapped goal-minus-object angle, in degrees. Running rows describe the latest saved actual state.

Independent replay costs so far (separate): 2,236 controls, 8 reset substeps, 63.357 summed seconds.

Costs include failed and timed-out runs. Per-run evaluation/process time sums are not concurrent elapsed wall time. Preview captures and independent replay costs remain separate. Running reports omit work in the currently executing planning call.
