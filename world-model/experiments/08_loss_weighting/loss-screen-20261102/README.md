# loss-screen-20261102

Category: Native, Min-SNR, GDL and focal-frequency screen.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **failed_or_incomplete**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 22 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 18 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [authorization.py](../../09_loss_definitions/loss-definition-screen-20261102/source/authorization.py)
- [bounded_io.py](../../09_loss_definitions/loss-definition-screen-20261102/source/bounded_io.py)
- [common.py](source/common.py)
- [continuation_checks.py](source/continuation_checks.py)
- [continuation_io.py](../../09_loss_definitions/loss-definition-screen-20261102/source/continuation_io.py)
- [continuation_ledger.py](source/continuation_ledger.py)
- [continuation_postrun_audit.py](source/continuation_postrun_audit.py)
- [continuation_worker.py](source/continuation_worker.py)
- [launch_continuation.ps1](source/launch_continuation.ps1)
- [ledger.py](source/ledger.py)
- [native_forward.py](../../09_loss_definitions/loss-definition-screen-20261102/source/native_forward.py)
- [objectives.py](source/objectives.py)
- [postrun_audit.py](source/postrun_audit.py)
- [preflight.py](source/preflight.py)
- [prepare_continuation.py](source/prepare_continuation.py)
- [prepare_freeze.py](source/prepare_freeze.py)
- [report.py](source/report.py)
- [safe_names.py](../../09_loss_definitions/loss-definition-screen-20261102/source/safe_names.py)
- [scoring_adapter.py](../../09_loss_definitions/loss-definition-screen-20261102/source/scoring_adapter.py)
- [state_restore.py](../../09_loss_definitions/loss-definition-screen-20261102/source/state_restore.py)
- [static_checks.py](source/static_checks.py)
- [worker.py](source/worker.py)

## Selected evidence files

- [FINAL_REVIEW.md](evidence/FINAL_REVIEW.md)
- [REPORT.txt](evidence/REPORT.txt)
- [SOURCE_TEST_SPEC.txt](evidence/SOURCE_TEST_SPEC.txt)
- [completion.json](evidence/completion.json)
- [continuation-audit-result.json](evidence/continuation-audit-result.json)
- [continuation-completion.json](evidence/continuation-completion.json)
- [continuation-config.json](evidence/continuation-config.json)
- [evaluation-spec.json](../../09_loss_definitions/loss-definition-screen-20261102/evidence/evaluation-spec.json)
- [frozen-config.json](evidence/frozen-config.json)
- [ledger.json](evidence/ledger.json)
- [owner-final-review-summary.json](evidence/owner-final-review-summary.json)
- [schedule.json](../../09_loss_definitions/loss-definition-screen-20261102/evidence/schedule.json)
- [report/final-manifest.json](evidence/report/final-manifest.json)
- [report/summary.json](evidence/report/summary.json)
- [training/BASE-result.json](evidence/training/BASE-result.json)
- [training/FFL-result.json](evidence/training/FFL-result.json)
- [training/GDL-result.json](evidence/training/GDL-result.json)
- [training/SNR-result.json](evidence/training/SNR-result.json)
