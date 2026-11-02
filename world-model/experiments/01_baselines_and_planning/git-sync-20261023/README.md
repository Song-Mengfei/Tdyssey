# git-sync-20261023

Category: Policy baselines, simulator search and planning.
Local source profile: `Tdyssey`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 9 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 9 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [files/scripts/evaluate_contact_sequence.py](../d-sequence-20261022-r4/source/source/scripts/evaluate_contact_sequence.py)
- [files/scripts/evaluate_simulator_search.py](../d-teacher-20261021/source/matrix-v1/source/scripts/evaluate_simulator_search.py)
- [files/tdyssey/contact_geometry.py](../d-sequence-20261022/source/source/tdyssey/contact_geometry.py)
- [files/tdyssey/simulator_search.py](../d-sequence-20261022/source/source/tdyssey/simulator_search.py)
- [files/tdyssey/simulator_sequence.py](../d-sequence-20261022-r4-final/source/source/tdyssey/simulator_sequence.py)
- [files/tests/test_simulator_search.py](../d-sequence-20261022-r4/source/source/tests/test_simulator_search.py)
- [files/tests/test_simulator_search_runtime.py](source/files/tests/test_simulator_search_runtime.py)
- [files/tests/test_simulator_sequence.py](../d-sequence-20261022-r4-final/source/source/tests/test_simulator_sequence.py)
- [files/tests/test_simulator_sequence_runtime.py](../d-sequence-20261022-r4/source/source/tests/test_simulator_sequence_runtime.py)

## Selected evidence files

- [manifest.json](evidence/manifest.json)
- [files/docs/CONTACT_SEQUENCE_SEARCH.md](evidence/files/docs/CONTACT_SEQUENCE_SEARCH.md)
- [files/docs/DECISIONS.md](evidence/files/docs/DECISIONS.md)
- [files/docs/SIMULATOR_SEARCH.md](evidence/files/docs/SIMULATOR_SEARCH.md)
- [files/docs/STATUS.md](evidence/files/docs/STATUS.md)
- [files/docs/VALIDATION.md](evidence/files/docs/VALIDATION.md)
- [restored-docs/DECISIONS.md](evidence/restored-docs/DECISIONS.md)
- [restored-docs/STATUS.md](evidence/restored-docs/STATUS.md)
- [restored-docs/VALIDATION.md](evidence/restored-docs/VALIDATION.md)
