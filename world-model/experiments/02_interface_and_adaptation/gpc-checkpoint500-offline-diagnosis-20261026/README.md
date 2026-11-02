# gpc-checkpoint500-offline-diagnosis-20261026

Category: Frozen WM, input compatibility and early adaptation.
Local source profile: `Tdyssey-world-model-validation`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 1 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 5 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [command-1-wrapper.ps1](source/command-1-wrapper.ps1)

## Selected evidence files

- [analysis-plan.json](evidence/analysis-plan.json)
- [contract-data.md](evidence/contract-data.md)
- [contract-upstream.md](evidence/contract-upstream.md)
- [partial-result.json](evidence/partial-result.json)
- [visual-review.md](evidence/visual-review.md)
