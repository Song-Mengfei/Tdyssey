# accelerated-rgb-run-20261031

Category: Controller variants, strict original physics and contact data.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 13 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 16 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [build_final_review.py](source/build_final_review.py)
- [collect.py](source/collect.py)
- [collect_tests.py](source/collect_tests.py)
- [common.py](source/common.py)
- [launch_detached.py](source/launch_detached.py)
- [orchestration_tests.py](source/orchestration_tests.py)
- [prepare.py](source/prepare.py)
- [selection.py](source/selection.py)
- [selection_tests.py](source/selection_tests.py)
- [train_eval.py](source/train_eval.py)
- [train_tests.py](source/train_tests.py)
- [worker.py](source/worker.py)
- [prestart-v1/worker.py](source/prestart-v1/worker.py)

## Selected evidence files

- [BACKGROUND_RUNNING.md](evidence/BACKGROUND_RUNNING.md)
- [FINAL_REVIEW.md](evidence/FINAL_REVIEW.md)
- [REPORT.md](evidence/REPORT.md)
- [evaluation-spec.json](evidence/evaluation-spec.json)
- [postrun-collection-review.json](evidence/postrun-collection-review.json)
- [postrun-final-manifest.json](evidence/postrun-final-manifest.json)
- [postrun-selection-review.json](evidence/postrun-selection-review.json)
- [runtime-config.json](evidence/runtime-config.json)
- [background/scheduled-entry.json](evidence/background/scheduled-entry.json)
- [background/scheduled-launch-failure.json](evidence/background/scheduled-launch-failure.json)
- [background/scheduled-worker-receipt.json](evidence/background/scheduled-worker-receipt.json)
- [background/training_and_evaluating-result.json](evidence/background/training_and_evaluating-result.json)
- [prestart-v1/runtime-config.json](evidence/prestart-v1/runtime-config.json)
- [report/summary.json](evidence/report/summary.json)
- [training/result.json](evidence/training/result.json)
- [training/summary.json](evidence/training/summary.json)
