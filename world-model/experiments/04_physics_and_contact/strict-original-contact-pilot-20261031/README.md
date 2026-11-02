# strict-original-contact-pilot-20261031

Category: Controller variants, strict original physics and contact data.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 22 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 24 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [freeze_phase.py](source/freeze_phase.py)
- [prepare_ui_restore.py](source/prepare_ui_restore.py)
- [background/phase.py](source/background/phase.py)
- [collection/revisions/calibration-v1-report-frozen/executed-report.py](source/collection/revisions/calibration-v1-report-frozen/executed-report.py)
- [ui-next/Start.cmd](../../../archive/app-before-hmf-20261103/Start.cmd)
- [ui-next/app.js](../../../archive/app-before-hmf-20261103/app.js)
- [ui-next/capture.py](../../../archive/app-before-hmf-20261103/capture.py)
- [ui-next/server.py](../../../archive/app-before-hmf-20261103/server.py)
- [ui-next/start.py](../../../archive/app-before-hmf-20261103/start.py)
- [ui-next/style.css](../../../apps/r3b-playground/style.css)
- [ui-next/test_playground.py](../../../archive/app-before-hmf-20261103/test_playground.py)
- [ui-source-before/Start.cmd](../../../archive/app-before-hmf-20261103/Start.cmd)
- [ui-source-before/app.js](../original-physics-contact-pilot-20261030/source/ui-next/app.js)
- [ui-source-before/capture.py](../original-physics-contact-pilot-20261030/source/ui-next/capture.py)
- [ui-source-before/server.py](../original-physics-contact-pilot-20261030/source/ui-next/server.py)
- [ui-source-before/start.py](../../../archive/app-before-hmf-20261103/start.py)
- [ui-source-before/style.css](../../../apps/r3b-playground/style.css)
- [ui-source-before/test_playground.py](../original-physics-contact-pilot-20261030/source/ui-next/test_playground.py)
- [validation/background-audit/run_audit.py](source/validation/background-audit/run_audit.py)
- [validation/background-audit/test_phase.py](source/validation/background-audit/test_phase.py)
- [validation/executed-background-v1/freeze_phase.py](source/validation/executed-background-v1/freeze_phase.py)
- [validation/executed-background-v1/phase.py](source/validation/executed-background-v1/phase.py)

## Selected evidence files

- [CALIBRATION_V1_REVIEW.json](evidence/CALIBRATION_V1_REVIEW.json)
- [CALIBRATION_V2_REVIEW.json](evidence/CALIBRATION_V2_REVIEW.json)
- [CALIBRATION_V2_REVIEW.md](evidence/CALIBRATION_V2_REVIEW.md)
- [ui-staging-manifest.json](evidence/ui-staging-manifest.json)
- [background/calibration-v1/completion.json](evidence/background/calibration-v1/completion.json)
- [background/calibration-v1/runtime-config.json](evidence/background/calibration-v1/runtime-config.json)
- [background/calibration-v2/completion.json](evidence/background/calibration-v2/completion.json)
- [background/calibration-v2/runtime-config.json](evidence/background/calibration-v2/runtime-config.json)
- [collection/REPORT.md](evidence/collection/REPORT.md)
- [collection/candidate-manifest.json](evidence/collection/candidate-manifest.json)
- [collection/report/summary.json](evidence/collection/report/summary.json)
- [collection/revisions/calibration-v1-report-frozen/REPORT.md](evidence/collection/revisions/calibration-v1-report-frozen/REPORT.md)
- [collection/revisions/calibration-v1-report-frozen/candidate-manifest.json](evidence/collection/revisions/calibration-v1-report-frozen/candidate-manifest.json)
- [collection/runs/calibration-v1/completion.json](evidence/collection/runs/calibration-v1/completion.json)
- [collection/runs/calibration-v1/ledger.json](evidence/collection/runs/calibration-v1/ledger.json)
- [collection/runs/calibration-v2/completion.json](evidence/collection/runs/calibration-v2/completion.json)
- [collection/runs/calibration-v2/ledger.json](evidence/collection/runs/calibration-v2/ledger.json)
- [ui-next/README.txt](evidence/ui-next/README.txt)
- [validation/calibration-v1-request-diagnosis.json](evidence/validation/calibration-v1-request-diagnosis.json)
- [validation/policy-static-review-v1.json](evidence/validation/policy-static-review-v1.json)
- [validation/report-readonly-audit.json](evidence/validation/report-readonly-audit.json)
- [validation/strict-report-readiness.json](evidence/validation/strict-report-readiness.json)
- [validation/strict-report-video-selection-correction.json](evidence/validation/strict-report-video-selection-correction.json)
- [validation/strict-ui-original-v1/summary.json](evidence/validation/strict-ui-original-v1/summary.json)
