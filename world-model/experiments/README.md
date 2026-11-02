# Experiment archive

All available local scientific source families are indexed here. Failed/preparation-only runs are retained. Repeated files have one canonical copy; SOURCE_INDEX.json records every original occurrence. Historical sources are not all drop-in replacements for the current core.

A directory with zero curated run-source/evidence entries is a registration of a known historical attempt, registry or preparation directory. Its implementation may live in the source profile, while its raw outputs are external or absent. Such a directory does not establish completion or a reproduced result. Use the [major experiment matrix](../docs/EXPERIMENT_MATRIX.md) for the scientific lineage rather than interpreting directory counts as independent experiments.

- [Policy baselines, simulator search and planning](01_baselines_and_planning/README.md)
- [Frozen WM, input compatibility and early adaptation](02_interface_and_adaptation/README.md)
- [Native RGB training, staged fitting and sampling](03_native_rgb_supervised/README.md)
- [Controller variants, strict original physics and contact data](04_physics_and_contact/README.md)
- [Generated histories and region/detail loss comparisons](05_history_and_region_losses/README.md)
- [High-motion fitting and inference sensitivity](06_high_motion/README.md)
- [Differentiable four-step rollout loss and failed preflight](07_rollout_loss/README.md)
- [Native, Min-SNR, GDL and focal-frequency screen](08_loss_weighting/README.md)
- [Occupancy geometry, spatial transport and diagnostics](09_loss_definitions/README.md)
- [Retrospective analysis, metrics and independent reviews](10_analysis_and_measurement/README.md)
- [Related direct-simulator DPPO experiments](11_policy_rl/README.md)
- [Supporting runtime and source checks](12_engineering/README.md)

See [the complete machine-readable catalog](../manifests/experiments.json), [source coverage](../manifests/source-map.json) and [reproduction limits](../docs/REPRODUCIBILITY.md).
