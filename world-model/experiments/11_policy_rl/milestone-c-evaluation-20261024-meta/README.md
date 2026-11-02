# milestone-c-evaluation-20261024-meta

Category: Related direct-simulator DPPO experiments.
Local source profile: `Tdyssey-dppo-milestone-a`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 3 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 6 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [audit_resources.py](source/audit_resources.py)
- [export_comparison_tables.py](source/export_comparison_tables.py)
- [reconcile_paired_evaluation.py](source/reconcile_paired_evaluation.py)

## Selected evidence files

- [CONTINUATION.md](evidence/CONTINUATION.md)
- [interpretation-review.json](evidence/interpretation-review.json)
- resources-review-final.json (local machine receipt omitted from publication)
- [monitor/monitor-report.json](evidence/monitor/monitor-report.json)
- [tables/manifest.json](evidence/tables/manifest.json)
- [tables/shape_summary.csv](evidence/tables/shape_summary.csv)
- [tables/summary.md](evidence/tables/summary.md)
