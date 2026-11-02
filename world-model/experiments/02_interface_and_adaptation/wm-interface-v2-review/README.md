# wm-interface-v2-review

Category: Frozen WM, input compatibility and early adaptation.
Local source profile: `Tdyssey-world-model-validation`. Recorded status: **analysis_or_preparation; not a training result**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 4 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 5 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [correct_reporting.py](source/correct_reporting.py)
- [evidence.py](source/evidence.py)
- [finalize.py](source/finalize.py)
- [summarize.py](source/summarize.py)

## Selected evidence files

- [REVIEW_ZH.md](evidence/REVIEW_ZH.md)
- [output-manifest.json](evidence/output-manifest.json)
- [reporting-corrections.json](evidence/reporting-corrections.json)
- [review-metrics.json](evidence/review-metrics.json)
- [termination-confusion-reviewed.csv](evidence/termination-confusion-reviewed.csv)
