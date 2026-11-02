# milestone-c-load-memory-20261024

Category: Related direct-simulator DPPO experiments.
Local source profile: `Tdyssey-dppo-milestone-a`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 12 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 6 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [check_worker_control.py](source/check_worker_control.py)
- [reconcile_c_run.py](source/reconcile_c_run.py)
- [verify_checkpoint_files.py](source/verify_checkpoint_files.py)
- [worker-control-check/complete/target.py](source/worker-control-check/complete/target.py)
- [worker-control-check/job_close/target.py](source/worker-control-check/job_close/target.py)
- [worker-control-check/terminate/target.py](source/worker-control-check/job_close/target.py)
- [worker-control-check-2/complete/target.py](source/worker-control-check/complete/target.py)
- [worker-control-check-2/job_close/target.py](source/worker-control-check/job_close/target.py)
- [worker-control-check-2/terminate/target.py](source/worker-control-check/job_close/target.py)
- [worker-control-check-3/complete/target.py](source/worker-control-check/complete/target.py)
- [worker-control-check-3/job_close/target.py](source/worker-control-check/job_close/target.py)
- [worker-control-check-3/terminate/target.py](source/worker-control-check/job_close/target.py)

## Selected evidence files

- [first-batch-review.json](evidence/first-batch-review.json)
- [ppo-review-final.json](evidence/ppo-review-final.json)
- resources-initial.json (local machine receipt omitted from publication)
- resources-review-final.json (local machine receipt omitted from publication)
- [load/monitor-report.json](evidence/load/monitor-report.json)
- [load/report.json](evidence/load/report.json)
- [train/monitor-report.json](evidence/train/monitor-report.json)
- [worker-control-check-3/report.json](evidence/worker-control-check-3/report.json)
