# loss-rollout-ab-20261102

Category: Differentiable four-step rollout loss and failed preflight.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **failed_or_incomplete**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 6 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 8 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [common.py](source/common.py)
- [prepare.py](source/prepare.py)
- [report.py](source/report.py)
- [review_failure.py](source/review_failure.py)
- [rollout_loss.py](source/rollout_loss.py)
- [worker.py](source/worker.py)

## Selected evidence files

- [PLAN.txt](evidence/PLAN.txt)
- [REPORT.txt](evidence/REPORT.txt)
- [completion.json](evidence/completion.json)
- [config.json](evidence/config.json)
- [evaluation-spec.json](../../09_loss_definitions/loss-definition-screen-20261102/evidence/evaluation-spec.json)
- [repair-review.json](evidence/repair-review.json)
- [review-storage.json](evidence/review-storage.json)
- [schedule.json](evidence/schedule.json)
