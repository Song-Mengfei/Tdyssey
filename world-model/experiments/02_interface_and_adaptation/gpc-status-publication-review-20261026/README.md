# gpc-status-publication-review-20261026

Category: Frozen WM, input compatibility and early adaptation.
Local source profile: `Tdyssey-world-model-validation`. Recorded status: **analysis_or_preparation; not a training result**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 7 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 23 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [native-status-fixture.py](source/native-status-fixture.py)
- [run-native.ps1](source/run-native.ps1)
- [run-synthetic-final.py](source/run-synthetic-final.py)
- [run-synthetic.py](source/run-synthetic.py)
- [summarize-evidence.py](source/summarize-evidence.py)
- [wm_compat_guard.original.py](source/wm_compat_guard.original.py)
- [wm_compat_guard.rejected-candidate.py](source/wm_compat_guard.rejected-candidate.py)

## Selected evidence files

- [plan-fixed.json](evidence/plan-fixed.json)
- [plan.json](evidence/plan.json)
- [summary.json](evidence/summary.json)
- [native-fixed/frozen-plan.json](evidence/plan-fixed.json)
- [native-fixed/parent-report.json](evidence/native-fixed/parent-report.json)
- [native-fixed/worker-result.json](evidence/native-fixed/worker-result.json)
- [native-fixed/A/case-result.json](evidence/native-fixed/A/case-result.json)
- [native-fixed/B/case-result.json](evidence/native-fixed/B/case-result.json)
- [native-fixed/C/case-result.json](evidence/native-fixed/C/case-result.json)
- [native-fixed/D/case-result.json](evidence/native-fixed/D/case-result.json)
- [native-fixed/E/case-result.json](evidence/native-fixed/E/case-result.json)
- [native-fixed/F/case-result.json](evidence/native-fixed/F/case-result.json)
- [native-fixed/G/case-result.json](evidence/native-fixed/G/case-result.json)
- [native-original/frozen-plan.json](evidence/plan.json)
- [native-original/parent-report.json](evidence/native-original/parent-report.json)
- [native-original/worker-result.json](evidence/native-original/worker-result.json)
- [native-original/A/case-result.json](evidence/native-original/A/case-result.json)
- [native-original/B/case-result.json](evidence/native-original/B/case-result.json)
- [native-original/C/case-result.json](evidence/native-original/C/case-result.json)
- [native-original/D/case-result.json](evidence/native-original/D/case-result.json)
- [native-original/E/case-result.json](evidence/native-original/E/case-result.json)
- [native-original/F/case-result.json](evidence/native-original/F/case-result.json)
- [native-original/G/case-result.json](evidence/native-original/G/case-result.json)
