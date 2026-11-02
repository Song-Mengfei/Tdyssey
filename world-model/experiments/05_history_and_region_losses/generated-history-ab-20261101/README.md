# generated-history-ab-20261101

Category: Generated histories and region/detail loss comparisons.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 13 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 17 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [common.py](source/common.py)
- [core_tests.py](source/core_tests.py)
- [engine.py](source/engine.py)
- [engine_tests.py](source/engine_tests.py)
- [evaluation.py](source/evaluation.py)
- [evaluation_tests.py](source/evaluation_tests.py)
- [pool.py](source/pool.py)
- [prepare.py](source/prepare.py)
- [report.py](source/report.py)
- [report_tests.py](source/report_tests.py)
- [test_pool.py](source/test_pool.py)
- [training_core.py](source/training_core.py)
- [worker.py](source/worker.py)

## Selected evidence files

- [BACKGROUND_RUNNING.md](evidence/BACKGROUND_RUNNING.md)
- [FINAL_REVIEW.md](evidence/FINAL_REVIEW.md)
- [REPORT.md](evidence/REPORT.md)
- [diagnostic-spec.json](evidence/diagnostic-spec.json)
- [formal-spec.json](evidence/formal-spec.json)
- [postrun-final-manifest.json](evidence/postrun-final-manifest.json)
- [runtime-config.json](evidence/runtime-config.json)
- [background/reporting-entry.json](evidence/background/reporting-entry.json)
- [background/reporting-process.json](evidence/background/reporting-process.json)
- [background/reporting-result.json](evidence/background/reporting-result.json)
- [background/reporting-storage.json](evidence/background/reporting-storage.json)
- [evaluation/A/result.json](evidence/evaluation/A/result.json)
- [evaluation/B/result.json](evidence/evaluation/B/result.json)
- [evaluation/baseline/result.json](evidence/evaluation/baseline/result.json)
- [report/final-manifest.json](evidence/report/final-manifest.json)
- [training/A/result.json](evidence/training/A/result.json)
- [training/B/result.json](evidence/training/B/result.json)
