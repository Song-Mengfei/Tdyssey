# d-sequence-20261022-r2

Category: Policy baselines, simulator search and planning.
Local source profile: `Tdyssey`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 14 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 7 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [audit_and_video.py](../d-sequence-20261022/source/audit_and_video.py)
- [run_pilot.py](source/run_pilot.py)
- [summarize.py](source/summarize.py)
- [source/pyproject.toml](../../../archive/workspaces/Tdyssey-world-model-validation/pyproject.toml)
- [source/scripts/evaluate_contact_sequence.py](../d-sequence-20261022/source/source/scripts/evaluate_contact_sequence.py)
- [source/tdyssey/contact_geometry.py](../d-sequence-20261022/source/source/tdyssey/contact_geometry.py)
- [source/tdyssey/evaluation.py](../../03_native_rgb_supervised/native-rgb-supervised-20261029/source/r2c1/source-snapshot/repo/tdyssey/evaluation.py)
- [source/tdyssey/geometry.py](../../03_native_rgb_supervised/native-rgb-20261028/source/collection-source/tdyssey/geometry.py)
- [source/tdyssey/official.py](../d-sequence-20261022/source/source/tdyssey/official.py)
- [source/tdyssey/simulator_planning.py](../d-sequence-20261022/source/source/tdyssey/simulator_planning.py)
- [source/tdyssey/simulator_search.py](../d-sequence-20261022/source/source/tdyssey/simulator_search.py)
- [source/tdyssey/simulator_sequence.py](source/source/tdyssey/simulator_sequence.py)
- [source/tdyssey/state.py](../../03_native_rgb_supervised/native-rgb-20261028/source/collection-source/tdyssey/state.py)
- [source/tdyssey/envs/official.py](../../03_native_rgb_supervised/native-rgb-20261028/source/collection-source/tdyssey/envs/official.py)

## Selected evidence files

- [protocol.json](evidence/protocol.json)
- [summary.csv](evidence/summary.csv)
- [summary.json](evidence/summary.json)
- [summary.md](evidence/summary.md)
- [videos/scene-100000/summary.json](evidence/videos/scene-100000/summary.json)
- [videos/scene-100100/summary.json](evidence/videos/scene-100100/summary.json)
- [videos/scene-100200/summary.json](evidence/videos/scene-100200/summary.json)
