# direct-target-four-case-diagnostic-20261031

Category: Controller variants, strict original physics and contact data.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **completion_not_established**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 7 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 7 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [audit_results.py](source/audit_results.py)
- [observer.py](source/observer.py)
- [prepare.py](source/prepare.py)
- [report.py](source/report.py)
- [run_diagnostic.py](source/run_diagnostic.py)
- [test_report_offline.py](source/test_report_offline.py)
- [tests_offline.py](source/tests_offline.py)

## Selected evidence files

- [PARENT_RESULTS_CHECK.json](evidence/PARENT_RESULTS_CHECK.json)
- [REPORT.md](evidence/REPORT.md)
- [completion.json](evidence/completion.json)
- [ledger.json](evidence/ledger.json)
- [runtime-config.json](evidence/runtime-config.json)
- [report/manifest.json](evidence/report/manifest.json)
- [report/summary.json](evidence/report/summary.json)
