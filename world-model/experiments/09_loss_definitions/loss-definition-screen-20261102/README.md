# loss-definition-screen-20261102

Category: Occupancy geometry, spatial transport and diagnostics.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **completed**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 18 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 17 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [authorization.py](source/authorization.py)
- [bounded_io.py](source/bounded_io.py)
- [common.py](source/common.py)
- [continuation_io.py](source/continuation_io.py)
- [diagnostics.py](source/diagnostics.py)
- [ledger.py](source/ledger.py)
- [native_forward.py](source/native_forward.py)
- [objectives.py](source/objectives.py)
- [owner_review.py](source/owner_review.py)
- [postrun_audit.py](source/postrun_audit.py)
- [preflight.py](source/preflight.py)
- [prepare.py](source/prepare.py)
- [report.py](source/report.py)
- [run_once.py](source/run_once.py)
- [safe_names.py](source/safe_names.py)
- [scoring_adapter.py](source/scoring_adapter.py)
- [state_restore.py](source/state_restore.py)
- [worker.py](source/worker.py)

## Selected evidence files

- [FINAL_REVIEW.md](evidence/FINAL_REVIEW.md)
- [REPORT.txt](evidence/REPORT.txt)
- [SOURCE_TEST_SPEC.txt](evidence/SOURCE_TEST_SPEC.txt)
- [audit-result.json](evidence/audit-result.json)
- [coefficients.json](evidence/coefficients.json)
- [completion.json](evidence/completion.json)
- [evaluation-spec.json](evidence/evaluation-spec.json)
- [frozen-config.json](evidence/frozen-config.json)
- [owner-review-summary.json](evidence/owner-review-summary.json)
- [owner-review-verification.json](evidence/owner-review-verification.json)
- [schedule.json](evidence/schedule.json)
- [diagnosis/summary.json](evidence/diagnosis/summary.json)
- [report/final-manifest.json](evidence/report/final-manifest.json)
- [report/summary.json](evidence/report/summary.json)
- [training/G_ONLY-result.json](evidence/training/G_ONLY-result.json)
- [training/NATIVE_G-result.json](evidence/training/NATIVE_G-result.json)
- [training/NATIVE_T-result.json](evidence/training/NATIVE_T-result.json)
