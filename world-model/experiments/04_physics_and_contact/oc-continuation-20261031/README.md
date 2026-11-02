# oc-continuation-20261031

Category: Supporting runtime and source checks.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 9 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 19 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [common.py](../../05_history_and_region_losses/generated-history-ab-20261101/source/common.py)
- [diagnostics.py](source/diagnostics.py)
- [diagnostics_tests.py](source/diagnostics_tests.py)
- [engine.py](source/engine.py)
- [engine_tests.py](source/engine_tests.py)
- [prepare.py](source/prepare.py)
- [report.py](source/report.py)
- [report_tests.py](source/report_tests.py)
- [worker.py](source/worker.py)

## Selected evidence files

- [BACKGROUND_RUNNING.md](evidence/BACKGROUND_RUNNING.md)
- [FINAL_REVIEW.md](evidence/FINAL_REVIEW.md)
- [REPORT.md](evidence/REPORT.md)
- [diagnostic-spec.json](evidence/diagnostic-spec.json)
- [loss-spec.json](evidence/loss-spec.json)
- [postrun-diagnostic-review.json](evidence/postrun-diagnostic-review.json)
- [postrun-final-manifest.json](evidence/postrun-final-manifest.json)
- [postrun-formal-review.json](evidence/postrun-formal-review.json)
- [postrun-training-review.json](evidence/postrun-training-review.json)
- [runtime-config.json](evidence/runtime-config.json)
- [background/reporting-process.json](evidence/background/reporting-process.json)
- [background/reporting-result.json](evidence/background/reporting-result.json)
- [background/reporting-storage.json](evidence/background/reporting-storage.json)
- [diagnostics/epoch1/result.json](evidence/diagnostics/epoch1/result.json)
- [diagnostics/epoch2/result.json](evidence/diagnostics/epoch2/result.json)
- [diagnostics/epoch3/result.json](evidence/diagnostics/epoch3/result.json)
- [report/final-manifest.json](evidence/report/final-manifest.json)
- [training/epoch2-result.json](evidence/training/epoch2-result.json)
- [training/epoch3-result.json](evidence/training/epoch3-result.json)
