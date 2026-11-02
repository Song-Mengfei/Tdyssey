# d-teacher-20261021

Category: Policy baselines, simulator search and planning.
Local source profile: `Tdyssey`. Recorded status: **historical_evidence; consult supplied report**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 33 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 28 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [audit_aggregates.py](source/audit_aggregates.py)
- [audit_and_video.py](../d-sequence-20261022/source/audit_and_video.py)
- [build_gallery.py](source/build_gallery.py)
- [capture_scene_inventory.py](source/capture_scene_inventory.py)
- [evaluate_v2.py](source/evaluate_v2.py)
- [render_initial_scenes.py](source/render_initial_scenes.py)
- [run_matrix.py](source/run_matrix.py)
- [run_matrix_v2.py](source/run_matrix_v2.py)
- [summarize_matrix.py](source/summarize_matrix.py)
- [matrix-v1/source/pyproject.toml](../../../archive/workspaces/Tdyssey-world-model-validation/pyproject.toml)
- [matrix-v1/source/scripts/evaluate_simulator_search.py](source/matrix-v1/source/scripts/evaluate_simulator_search.py)
- [matrix-v1/source/tdyssey/evaluation.py](../../03_native_rgb_supervised/native-rgb-supervised-20261029/source/r2c1/source-snapshot/repo/tdyssey/evaluation.py)
- [matrix-v1/source/tdyssey/geometry.py](../../03_native_rgb_supervised/native-rgb-20261028/source/collection-source/tdyssey/geometry.py)
- [matrix-v1/source/tdyssey/official.py](../d-sequence-20261022/source/source/tdyssey/official.py)
- [matrix-v1/source/tdyssey/simulator_planning.py](../d-sequence-20261022/source/source/tdyssey/simulator_planning.py)
- [matrix-v1/source/tdyssey/simulator_search.py](../d-sequence-20261022/source/source/tdyssey/simulator_search.py)
- [matrix-v1/source/tdyssey/state.py](../../03_native_rgb_supervised/native-rgb-20261028/source/collection-source/tdyssey/state.py)
- [matrix-v1/source/tdyssey/envs/official.py](../../03_native_rgb_supervised/native-rgb-20261028/source/collection-source/tdyssey/envs/official.py)
- [matrix-v2/source/pyproject.toml](../../../archive/workspaces/Tdyssey-world-model-validation/pyproject.toml)
- [matrix-v2/source/artifacts/d-teacher-20261021/evaluate_v2.py](source/evaluate_v2.py)
- [matrix-v2/source/artifacts/d-teacher-20261021/v2-draft/simulator_search_v2.py](source/matrix-v2/source/artifacts/d-teacher-20261021/v2-draft/simulator_search_v2.py)
- [matrix-v2/source/scripts/evaluate_simulator_search.py](source/matrix-v1/source/scripts/evaluate_simulator_search.py)
- [matrix-v2/source/tdyssey/evaluation.py](../../03_native_rgb_supervised/native-rgb-supervised-20261029/source/r2c1/source-snapshot/repo/tdyssey/evaluation.py)
- [matrix-v2/source/tdyssey/geometry.py](../../03_native_rgb_supervised/native-rgb-20261028/source/collection-source/tdyssey/geometry.py)
- [matrix-v2/source/tdyssey/official.py](../d-sequence-20261022/source/source/tdyssey/official.py)
- [matrix-v2/source/tdyssey/simulator_planning.py](../d-sequence-20261022/source/source/tdyssey/simulator_planning.py)
- [matrix-v2/source/tdyssey/simulator_search.py](../d-sequence-20261022/source/source/tdyssey/simulator_search.py)
- [matrix-v2/source/tdyssey/state.py](../../03_native_rgb_supervised/native-rgb-20261028/source/collection-source/tdyssey/state.py)
- [matrix-v2/source/tdyssey/envs/official.py](../../03_native_rgb_supervised/native-rgb-20261028/source/collection-source/tdyssey/envs/official.py)
- [v2-draft/composite_runtime_check.py](source/v2-draft/composite_runtime_check.py)
- [v2-draft/simulator_search_v2.py](source/matrix-v2/source/artifacts/d-teacher-20261021/v2-draft/simulator_search_v2.py)
- [v2-draft/test_simulator_search_v2.py](source/v2-draft/test_simulator_search_v2.py)
- [v2-draft/tiny_runtime_check.py](source/v2-draft/tiny_runtime_check.py)

## Selected evidence files

- [initial-scenes-preview.json](evidence/initial-scenes-preview.json)
- [helper-validation-r2/summary.json](evidence/helper-validation-r2/summary.json)
- [matrix-v1/protocol.json](evidence/matrix-v1/protocol.json)
- [matrix-v1/summary.csv](evidence/matrix-v1/summary.csv)
- [matrix-v1/summary.json](evidence/matrix-v1/summary.json)
- [matrix-v1/summary.md](evidence/matrix-v1/summary.md)
- [matrix-v2/protocol.json](evidence/matrix-v2/protocol.json)
- [matrix-v2/summary.csv](evidence/matrix-v2/summary.csv)
- [matrix-v2/summary.json](evidence/matrix-v2/summary.json)
- [matrix-v2/summary.md](evidence/matrix-v2/summary.md)
- [v2-draft/PROPOSAL.md](evidence/v2-draft/PROPOSAL.md)
- [v2-launcher-smoke/report.json](evidence/v2-launcher-smoke/report.json)
- [videos-v1/scene-100000-search-42/summary.json](evidence/videos-v1/scene-100000-search-42/summary.json)
- [videos-v1/scene-100000-search-43/summary.json](evidence/videos-v1/scene-100000-search-43/summary.json)
- [videos-v1/scene-100100-search-42/summary.json](evidence/videos-v1/scene-100100-search-42/summary.json)
- [videos-v1/scene-100100-search-43/summary.json](evidence/videos-v1/scene-100100-search-43/summary.json)
- [videos-v1/scene-100200-search-42/summary.json](evidence/videos-v1/scene-100200-search-42/summary.json)
- [videos-v1/scene-100200-search-43/summary.json](evidence/videos-v1/scene-100200-search-43/summary.json)
- [videos-v1/scene-100300-search-42/summary.json](evidence/videos-v1/scene-100300-search-42/summary.json)
- [videos-v1/scene-100300-search-43/summary.json](evidence/videos-v1/scene-100300-search-43/summary.json)
- [videos-v2/scene-100000-search-42/summary.json](evidence/videos-v2/scene-100000-search-42/summary.json)
- [videos-v2/scene-100000-search-43/summary.json](evidence/videos-v2/scene-100000-search-43/summary.json)
- [videos-v2/scene-100100-search-42/summary.json](evidence/videos-v2/scene-100100-search-42/summary.json)
- [videos-v2/scene-100100-search-43/summary.json](evidence/videos-v2/scene-100100-search-43/summary.json)
- [videos-v2/scene-100200-search-42/summary.json](evidence/videos-v2/scene-100200-search-42/summary.json)
- [videos-v2/scene-100200-search-43/summary.json](evidence/videos-v2/scene-100200-search-43/summary.json)
- [videos-v2/scene-100300-search-42/summary.json](evidence/videos-v2/scene-100300-search-42/summary.json)
- [videos-v2/scene-100300-search-43/summary.json](evidence/videos-v2/scene-100300-search-43/summary.json)
