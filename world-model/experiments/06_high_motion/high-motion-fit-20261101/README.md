# high-motion-fit-20261101

Category: High-motion fitting and inference sensitivity.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 13 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 23 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [common.py](../../05_history_and_region_losses/generated-history-ab-20261101/source/common.py)
- [data.py](source/data.py)
- [engine.py](source/engine.py)
- [engine_tests.py](source/engine_tests.py)
- [eval_tests.py](source/eval_tests.py)
- [evaluation.py](source/evaluation.py)
- [prepare.py](source/prepare.py)
- [report.py](source/report.py)
- [selection.py](source/selection.py)
- [selection_tests.py](source/selection_tests.py)
- [training_core.py](source/training_core.py)
- [worker.py](source/worker.py)
- [background/cleanup-completed-registration.ps1](source/background/cleanup-completed-registration.ps1)

## Selected evidence files

- [BACKGROUND_RUNNING.md](evidence/BACKGROUND_RUNNING.md)
- [FINAL_REVIEW.md](evidence/FINAL_REVIEW.md)
- [GT-calibration-correction-receipt.json](evidence/GT-calibration-correction-receipt.json)
- [PLAN.txt](evidence/PLAN.txt)
- [REPORT.md](evidence/REPORT.md)
- [evaluation-spec.json](evidence/evaluation-spec.json)
- [fit-standard.json](evidence/fit-standard.json)
- [runtime-config.json](evidence/runtime-config.json)
- [background/baseline-result.json](evidence/background/baseline-result.json)
- [background/reporting-entry.json](evidence/background/reporting-entry.json)
- [background/reporting-process.json](evidence/background/reporting-process.json)
- [background/reporting-result.json](evidence/background/reporting-result.json)
- [background/reporting-storage.json](evidence/background/reporting-storage.json)
- [evaluation/0/result.json](evidence/background/baseline-result.json)
- [evaluation/1000/result.json](evidence/evaluation/1000/result.json)
- [evaluation/2000/result.json](evidence/evaluation/2000/result.json)
- [evaluation/250/result.json](evidence/evaluation/250/result.json)
- [evaluation/500/result.json](evidence/evaluation/500/result.json)
- [evaluation/final/result.json](evidence/evaluation/final/result.json)
- [report/final-manifest.json](evidence/report/final-manifest.json)
- [selection/gt-manifest.json](evidence/selection/gt-manifest.json)
- [selection/summary.json](evidence/selection/summary.json)
- [training/best-fit-selection.json](evidence/training/best-fit-selection.json)
