# r3b-playground-diagnostics-20261030

Category: Retrospective analysis, metrics and independent reviews.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **completion_not_established**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 15 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 32 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [finalize_state.py](source/finalize_state.py)
- [background/launcher.py](source/background/launcher.py)
- [background/pipeline.py](source/background/pipeline.py)
- [background/start-ui.ps1](source/background/start-ui.ps1)
- [browser-validation/decode_saved_video.py](source/browser-validation/decode_saved_video.py)
- [runs/original-20261030-v1/report-before-manual/native_rgb_playground_report-before-timing.py](source/runs/original-20261030-v1/report-before-manual/native_rgb_playground_report-before-timing.py)
- [source-before-event-lock/app.js](../../04_physics_and_contact/original-physics-contact-pilot-20261030/source/ui-source-before/app.js)
- [source-before-event-lock/server.py](source/source-before-event-lock/server.py)
- [source-before-ui/Start.cmd](../../../archive/app-before-hmf-20261103/Start.cmd)
- [source-before-ui/app.js](source/source-before-ui/app.js)
- [source-before-ui/server.py](source/source-before-ui/server.py)
- [source-before-ui/start.py](source/source-before-ui/start.py)
- [source-before-ui/style.css](source/source-before-ui/style.css)
- [source-before-ui/test_playground.py](source/source-before-ui/test_playground.py)
- [source-before-video-tail/app.js](source/source-before-video-tail/app.js)

## Selected evidence files

- [REPORT.md](evidence/REPORT.md)
- [completion.json](evidence/completion.json)
- [original-physics-protocol.json](evidence/original-physics-protocol.json)
- [plan-approved.json](evidence/plan-approved.json)
- [plan-proposed-before-protocol-binding.json](evidence/plan-proposed-before-protocol-binding.json)
- [plan-proposed.json](evidence/plan-proposed.json)
- [report-CPU-contracts.json](evidence/report-CPU-contracts.json)
- background/launch-spec.json (local machine receipt omitted from publication)
- [background/runtime-config.json](evidence/background/runtime-config.json)
- [physics-alignment/INDEX.md](evidence/physics-alignment/INDEX.md)
- [physics-alignment/summary.json](evidence/physics-alignment/summary.json)
- [recordings/recording-046f19d4a24d47a4a77ad4ebef3c4f48/raw/manifest.json](evidence/recordings/recording-046f19d4a24d47a4a77ad4ebef3c4f48/raw/manifest.json)
- [recordings/recording-07c96e4799284ecaa39af409b80a2f73/raw/manifest.json](evidence/recordings/recording-07c96e4799284ecaa39af409b80a2f73/raw/manifest.json)
- [recordings/recording-26b20033303d4c57ba40dadc0386cdb0/raw/manifest.json](evidence/recordings/recording-26b20033303d4c57ba40dadc0386cdb0/raw/manifest.json)
- [recordings/recording-511172d7beb447c7b28f78c3291d3228/raw/manifest.json](evidence/recordings/recording-511172d7beb447c7b28f78c3291d3228/raw/manifest.json)
- [recordings/recording-5cf6347fdb354c7594b2ea10a4ddecba/raw/manifest.json](evidence/recordings/recording-5cf6347fdb354c7594b2ea10a4ddecba/raw/manifest.json)
- [recordings/recording-61d76a7b4b954b19a83438f676fc01b2/raw/manifest.json](evidence/recordings/recording-61d76a7b4b954b19a83438f676fc01b2/raw/manifest.json)
- [recordings/recording-7917cd3c57594ca1b4c64401251b5e97/raw/manifest.json](evidence/recordings/recording-7917cd3c57594ca1b4c64401251b5e97/raw/manifest.json)
- [recordings/recording-8cdf3d041ce143a289f29865f73f9396/raw/manifest.json](evidence/recordings/recording-8cdf3d041ce143a289f29865f73f9396/raw/manifest.json)
- [recordings/recording-a2ca971487854cbab25129242ab0e277/raw/manifest.json](evidence/recordings/recording-a2ca971487854cbab25129242ab0e277/raw/manifest.json)
- [recordings/recording-b1c2689362b345baa268d0689cbefeaf/raw/manifest.json](evidence/recordings/recording-b1c2689362b345baa268d0689cbefeaf/raw/manifest.json)
- [recordings/recording-e6142e17d4e34e0f97ef6c44e487907e/raw/manifest.json](evidence/recordings/recording-e6142e17d4e34e0f97ef6c44e487907e/raw/manifest.json)
- [runs/original-20261030-v1/REPORT.md](evidence/runs/original-20261030-v1/REPORT.md)
- [runs/original-20261030-v1/completion.json](evidence/runs/original-20261030-v1/completion.json)
- [runs/original-20261030-v1/pipeline-completion.json](evidence/runs/original-20261030-v1/pipeline-completion.json)
- [runs/original-20261030-v1/report/step-summary.csv](evidence/runs/original-20261030-v1/report/step-summary.csv)
- [runs/original-20261030-v1/report-before-manual/REPORT.md](evidence/runs/original-20261030-v1/report-before-manual/REPORT.md)
- [runs/original-20261030-v1/report-before-manual/report-CPU-contracts-before-timing.json](evidence/runs/original-20261030-v1/report-before-manual/report-CPU-contracts-before-timing.json)
- [runs/original-20261030-v1/review/OC-fixed-diagnostic-summary.json](evidence/runs/original-20261030-v1/review/OC-fixed-diagnostic-summary.json)
- [runs/original-20261030-v1/review/VISUAL-REVIEW.md](evidence/runs/original-20261030-v1/review/VISUAL-REVIEW.md)
- [runs/original-20261030-v1/review/report-source-revision.json](evidence/runs/original-20261030-v1/review/report-source-revision.json)
- [runs/original-20261030-v1/review/report-timing-regression.json](evidence/runs/original-20261030-v1/review/report-timing-regression.json)
- [user-references/manifest.json](evidence/user-references/manifest.json)
