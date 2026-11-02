# gpc-foundation-repair-20261026

Category: Frozen WM, input compatibility and early adaptation.
Local source profile: `Tdyssey-world-model-validation`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 18 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 11 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [check-source.ps1](source/check-source.ps1)
- [check-source.py](source/check-source.py)
- [review-video.ps1](source/review-video.ps1)
- [review-video.py](source/review-video.py)
- [run-cpu-check.ps1](source/run-cpu-check.ps1)
- [run-offline.ps1](source/run-offline.ps1)
- [run-real.executed.ps1](source/run-real.executed.ps1)
- [run-real.ps1](source/run-real.ps1)
- [run-real.py](source/run-real.py)
- [verify-offline.py](source/verify-offline.py)
- [baseline/start_manual_demo.cmd](source/baseline/start_manual_demo.cmd)
- [baseline/scripts/manual_world_model.py](source/baseline/scripts/manual_world_model.py)
- [baseline/tdyssey/gpc_adaptation_guard.py](../gpc-conditioning-contract-preparation-20261026/source/baseline/tdyssey/gpc_adaptation_guard.py)
- [baseline/tdyssey/gpc_checkpoint_evaluation.py](../gpc-conditioning-contract-preparation-20261026/source/baseline/tdyssey/gpc_checkpoint_evaluation.py)
- [baseline/tdyssey/gpc_conditioning_contract.py](../gpc-conditioning-contract-preparation-20261026/source/baseline/tdyssey/gpc_conditioning_contract.py)
- [baseline/tdyssey/manual_world_model.py](source/baseline/tdyssey/manual_world_model.py)
- [baseline/tests/test_gpc_conditioning_contract.py](../gpc-conditioning-contract-preparation-20261026/source/baseline/tests/test_gpc_conditioning_contract.py)
- [baseline/tests/test_manual_world_model.py](source/baseline/tests/test_manual_world_model.py)

## Selected evidence files

- [contract-audit.md](evidence/contract-audit.md)
- [cpu-plan.json](evidence/cpu-plan.json)
- [diagnostic-summary.json](evidence/diagnostic-summary.json)
- [real-plan.json](evidence/real-plan.json)
- [source-contract-audit.md](evidence/source-contract-audit.md)
- [static-case-selection.json](evidence/static-case-selection.json)
- [baseline/README.md](evidence/baseline/README.md)
- [baseline/docs/DECISIONS.md](evidence/baseline/docs/DECISIONS.md)
- [baseline/docs/MANUAL_WORLD_MODEL.md](evidence/baseline/docs/MANUAL_WORLD_MODEL.md)
- [baseline/docs/STATUS.md](evidence/baseline/docs/STATUS.md)
- [real-01/summary.json](evidence/real-01/summary.json)
