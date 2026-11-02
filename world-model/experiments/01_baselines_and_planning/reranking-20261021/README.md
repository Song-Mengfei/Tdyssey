# reranking-20261021

Category: Policy baselines, simulator search and planning.
Local source profile: `Tdyssey`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 9 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 2 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [audit_reports.py](source/audit_reports.py)
- [d-push-diagnostic.py](source/d-push-diagnostic.py)
- [export_paused_video.py](source/export_paused_video.py)
- [ranged_download.py](source/ranged_download.py)
- [run-comparison.ps1](source/run-comparison.ps1)
- [search_step176_macro_expand.py](source/search_step176_macro_expand.py)
- [search_step176_macro_probe.py](source/search_step176_macro_probe.py)
- [search_step176_macro_refine.py](source/search_step176_macro_refine.py)
- [show_replay.py](source/show_replay.py)

## Selected evidence files

- [compact-inspection.json](evidence/compact-inspection.json)
- [review-summary.json](evidence/review-summary.json)
