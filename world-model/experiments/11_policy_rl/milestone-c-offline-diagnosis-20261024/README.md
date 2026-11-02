# milestone-c-offline-diagnosis-20261024

Category: Related direct-simulator DPPO experiments.
Local source profile: `Tdyssey-dppo-milestone-a`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 2 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 15 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [audit_inputs.py](source/audit_inputs.py)
- [build_review.py](source/build_review.py)

## Selected evidence files

- [replays/manifest.json](evidence/replays/manifest.json)
- [replays/visual-review.json](evidence/replays/visual-review.json)
- [replays/reviewed/manifest.json](evidence/replays/reviewed/manifest.json)
- [review-final/all_case_tables.md](evidence/review-final/all_case_tables.md)
- [review-final/manifest.json](evidence/review-final/manifest.json)
- [traces/findings.md](evidence/traces/findings.md)
- [traces-final/manifest.json](evidence/traces-final/manifest.json)
- [training/batch_metrics.csv](evidence/training/batch_metrics.csv)
- [training/notes.md](evidence/training/notes.md)
- [training/pass_metrics.csv](evidence/training/pass_metrics.csv)
- [training/shape_batch_metrics.csv](evidence/training/shape_batch_metrics.csv)
- [training-final/batch_metrics.csv](evidence/training/batch_metrics.csv)
- [training-final/notes.md](evidence/training-final/notes.md)
- [training-final/pass_metrics.csv](evidence/training/pass_metrics.csv)
- [training-final/shape_batch_metrics.csv](evidence/training/shape_batch_metrics.csv)
