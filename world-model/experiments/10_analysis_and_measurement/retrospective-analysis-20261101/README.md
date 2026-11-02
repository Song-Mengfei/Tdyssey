# retrospective-analysis-20261101

Category: Retrospective analysis, metrics and independent reviews.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 17 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 24 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [audit_saved.py](source/audit_saved.py)
- [build_ledgers.py](source/build_ledgers.py)
- [render_report.py](source/render_report.py)
- [hmf-final-review-20261102/analyze_final.py](source/hmf-final-review-20261102/analyze_final.py)
- [hmf-final-review-20261102/update_reports.py](source/hmf-final-review-20261102/update_reports.py)
- [inference-diagnostics-review-20261102/analyze_saved.py](source/inference-diagnostics-review-20261102/analyze_saved.py)
- [inference-diagnostics-review-20261102/make_report.py](source/inference-diagnostics-review-20261102/make_report.py)
- [loss-definition-review-20261102/analyze_results.py](source/loss-definition-review-20261102/analyze_results.py)
- [loss-definition-review-20261102/build_review.py](source/loss-definition-review-20261102/build_review.py)
- [loss-definition-review-20261102/finish_review.py](source/loss-definition-review-20261102/finish_review.py)
- [loss-literature-review-20261102/analyze_screen.py](source/loss-literature-review-20261102/analyze_screen.py)
- [loss-literature-review-20261102/build_review.py](source/loss-literature-review-20261102/build_review.py)
- [loss-literature-review-20261102/plot_weight_profile.py](source/loss-literature-review-20261102/plot_weight_profile.py)
- [loss-literature-review-20261102/render_screen_results.py](source/loss-literature-review-20261102/render_screen_results.py)
- [loss-rollout-preflight-review-20261102/review_saved.py](source/loss-rollout-preflight-review-20261102/review_saved.py)
- [loss-rollout-r1-review-20261102/analyze_saved.py](source/loss-rollout-r1-review-20261102/analyze_saved.py)
- [loss-rollout-r1-review-20261102/make_report.py](source/loss-rollout-r1-review-20261102/make_report.py)

## Selected evidence files

- [DATA_LEDGER.json](evidence/DATA_LEDGER.json)
- [EVIDENCE_MANIFEST.json](evidence/EVIDENCE_MANIFEST.json)
- [EXPERIMENT_LEDGER.json](evidence/EXPERIMENT_LEDGER.json)
- [REPORT.txt](evidence/REPORT.txt)
- [REPORT_snapshot_20261101.txt](evidence/REPORT_snapshot_20261101.txt)
- [SOURCES.json](evidence/SOURCES.json)
- [hmf-final-review-20261102/REPORT.txt](../../06_high_motion/hmf-inference-diagnostics-20261102/evidence/SOURCE_ANALYSIS_REPORT.txt)
- [inference-diagnostics-review-20261102/REPORT.txt](evidence/inference-diagnostics-review-20261102/REPORT.txt)
- [loss-definition-review-20261102/BIBLIOGRAPHY.json](evidence/loss-definition-review-20261102/BIBLIOGRAPHY.json)
- [loss-definition-review-20261102/PRE_EXECUTION_SOURCE_REVIEW.json](evidence/loss-definition-review-20261102/PRE_EXECUTION_SOURCE_REVIEW.json)
- [loss-definition-review-20261102/RESULTS.txt](evidence/loss-definition-review-20261102/RESULTS.txt)
- [loss-definition-review-20261102/REVIEW.txt](evidence/loss-definition-review-20261102/REVIEW.txt)
- [loss-definition-review-20261102/TEST_SPEC.txt](../../09_loss_definitions/loss-definition-screen-20261102/evidence/SOURCE_TEST_SPEC.txt)
- [loss-literature-review-20261102/BIBLIOGRAPHY.json](evidence/loss-literature-review-20261102/BIBLIOGRAPHY.json)
- [loss-literature-review-20261102/FROZEN_SOURCE_REVIEW.json](evidence/loss-literature-review-20261102/FROZEN_SOURCE_REVIEW.json)
- [loss-literature-review-20261102/PRE_FREEZE_SOURCE_REVIEW.json](evidence/loss-literature-review-20261102/PRE_FREEZE_SOURCE_REVIEW.json)
- [loss-literature-review-20261102/REVIEW.txt](evidence/loss-literature-review-20261102/REVIEW.txt)
- [loss-literature-review-20261102/REVIEW_STATE.json](evidence/loss-literature-review-20261102/REVIEW_STATE.json)
- [loss-literature-review-20261102/SCREEN_RESULTS.txt](evidence/loss-literature-review-20261102/SCREEN_RESULTS.txt)
- [loss-literature-review-20261102/TEST_RESULTS.json](evidence/loss-literature-review-20261102/TEST_RESULTS.json)
- [loss-literature-review-20261102/TEST_SPEC.txt](../../08_loss_weighting/loss-screen-20261102/evidence/SOURCE_TEST_SPEC.txt)
- [loss-rollout-preflight-review-20261102/REPORT.txt](evidence/loss-rollout-preflight-review-20261102/REPORT.txt)
- [loss-rollout-r1-review-20261102/REPORT.txt](evidence/loss-rollout-r1-review-20261102/REPORT.txt)
- [loss-rollout-r1-review-20261102/SAVED_PANELS_MANIFEST.json](evidence/loss-rollout-r1-review-20261102/SAVED_PANELS_MANIFEST.json)
