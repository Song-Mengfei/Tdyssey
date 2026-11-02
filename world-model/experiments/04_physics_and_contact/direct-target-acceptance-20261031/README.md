# direct-target-acceptance-20261031

Category: Controller variants, strict original physics and contact data.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 1 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 13 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [prepare.py](source/prepare.py)

## Selected evidence files

- [ACCEPTANCE_REVIEW.json](evidence/ACCEPTANCE_REVIEW.json)
- [ACCEPTANCE_REVIEW.md](evidence/ACCEPTANCE_REVIEW.md)
- [IMPLEMENTATION_REVIEW.txt](evidence/IMPLEMENTATION_REVIEW.txt)
- [REPORT.md](evidence/REPORT.md)
- [runtime-config.json](evidence/runtime-config.json)
- [background/completion.json](evidence/background/completion.json)
- [background/reporting_saved_only-process.json](evidence/background/reporting_saved_only-process.json)
- [background/failed-venv-broker/completion.json](evidence/background/failed-venv-broker/completion.json)
- [background/failed-venv-broker/manifest.json](evidence/background/failed-venv-broker/manifest.json)
- [collection/completion.json](evidence/collection/completion.json)
- [collection/ledger.json](evidence/collection/ledger.json)
- [report/final-manifest.json](evidence/report/final-manifest.json)
- [report/raw-manifest.csv](evidence/report/raw-manifest.csv)
