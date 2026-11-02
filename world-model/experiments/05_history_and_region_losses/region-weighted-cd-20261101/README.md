# region-weighted-cd-20261101

Category: Generated histories and region/detail loss comparisons.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 33 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 18 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [common.py](../generated-history-ab-20261101/source/common.py)
- [core_tests.py](source/core_tests.py)
- [diagnosis.py](source/diagnosis.py)
- [diagnosis_tests.py](source/diagnosis_tests.py)
- [direction.py](source/direction.py)
- [engine.py](source/engine.py)
- [engine_tests.py](source/engine_tests.py)
- [eval_report_tests.py](source/eval_report_tests.py)
- [evaluation.py](source/evaluation.py)
- [mask_tests.py](source/mask_tests.py)
- [masks.py](source/masks.py)
- [pool.py](source/pool.py)
- [prepare.py](source/prepare.py)
- [report.py](source/report.py)
- [training_core.py](source/training_core.py)
- [worker.py](source/worker.py)
- [background/cleanup-completed-registration.ps1](source/background/cleanup-completed-registration.ps1)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/common.py](../generated-history-ab-20261101/source/common.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/core_tests.py](source/core_tests.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/diagnosis.py](source/diagnosis.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/diagnosis_tests.py](source/diagnosis_tests.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/direction.py](source/direction.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/engine.py](source/engine.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/engine_tests.py](source/engine_tests.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/eval_report_tests.py](source/prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/eval_report_tests.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/evaluation.py](source/evaluation.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/mask_tests.py](source/mask_tests.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/masks.py](source/masks.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/pool.py](source/pool.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/prepare.py](source/prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/prepare.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/report.py](source/prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/report.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/training_core.py](source/training_core.py)
- [prelaunch-freezes/RWCD-20261101-9693416a3df94f9a84d4f3d7ab5722e6/worker.py](source/worker.py)

## Selected evidence files

- [BACKGROUND_RUNNING.md](evidence/BACKGROUND_RUNNING.md)
- [FINAL_REVIEW.md](evidence/FINAL_REVIEW.md)
- [REPORT.md](evidence/REPORT.md)
- [diagnosis-spec.json](evidence/diagnosis-spec.json)
- [diagnostic-spec.json](../generated-history-ab-20261101/evidence/diagnostic-spec.json)
- [direction-protocol.json](evidence/direction-protocol.json)
- [direction-truth-calibration.json](evidence/direction-truth-calibration.json)
- [formal-spec.json](../generated-history-ab-20261101/evidence/formal-spec.json)
- [loss-spec.json](evidence/loss-spec.json)
- [runtime-config.json](evidence/runtime-config.json)
- [background/reporting-entry.json](evidence/background/reporting-entry.json)
- [background/reporting-process.json](evidence/background/reporting-process.json)
- [background/reporting-result.json](evidence/background/reporting-result.json)
- [background/reporting-storage.json](evidence/background/reporting-storage.json)
- [diagnosis/REPORT.md](evidence/diagnosis/REPORT.md)
- [report/final-manifest.json](evidence/report/final-manifest.json)
- [training/C/result.json](evidence/training/C/result.json)
- [training/D/result.json](evidence/training/D/result.json)
