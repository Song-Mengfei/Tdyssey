# loss-rollout-ab-20261102-r1

Category: Differentiable four-step rollout loss and failed preflight.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **failed_or_incomplete**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 6 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 10 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [audit_saved.py](source/audit_saved.py)
- [common.py](source/common.py)
- [prepare_revision.py](source/prepare_revision.py)
- [report.py](../loss-rollout-ab-20261102/source/report.py)
- [rollout_loss.py](../loss-rollout-ab-20261102/source/rollout_loss.py)
- [worker.py](source/worker.py)

## Selected evidence files

- [OWNER_FAILED_REVIEW.txt](evidence/OWNER_FAILED_REVIEW.txt)
- [PLAN.txt](../loss-rollout-ab-20261102/evidence/PLAN.txt)
- [completion.json](evidence/completion.json)
- [config.json](evidence/config.json)
- [evaluation-spec.json](../../09_loss_definitions/loss-definition-screen-20261102/evidence/evaluation-spec.json)
- [owner-terminal-saved-review.json](evidence/owner-terminal-saved-review.json)
- [schedule.json](../loss-rollout-ab-20261102/evidence/schedule.json)
- [unchanged-protocol-proof.json](evidence/unchanged-protocol-proof.json)
- [training/A-result.json](evidence/training/A-result.json)
- [training/B-result.json](evidence/training/B-result.json)
