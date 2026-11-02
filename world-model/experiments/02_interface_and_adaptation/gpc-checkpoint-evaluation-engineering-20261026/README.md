# gpc-checkpoint-evaluation-engineering-20261026

Category: Frozen WM, input compatibility and early adaptation.
Local source profile: `Tdyssey-world-model-validation`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 3 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 8 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [native-journal.py](source/native-journal.py)
- [run-native.ps1](source/run-native.ps1)
- [run-synthetic.py](source/run-synthetic.py)

## Selected evidence files

- [model-only-loader-plan.json](evidence/model-only-loader-plan.json)
- [native-plan.json](evidence/native-plan.json)
- [summary.json](evidence/summary.json)
- [native-journal/frozen-plan.json](evidence/native-plan.json)
- [native-journal/parent-report.json](evidence/native-journal/parent-report.json)
- [native-journal/worker-result.json](evidence/native-journal/worker-result.json)
- [native-journal/normal/case-result.json](evidence/native-journal/normal/case-result.json)
- [native-journal/partial/case-result.json](evidence/native-journal/partial/case-result.json)
