# Contributing

Use an isolated branch and preserve source/experiment provenance. Run `python -B scripts/check_repo.py` before proposing source changes. Runtime changes need the existing relevant focused checks in the correct profile; do not equate syntax checks with a validated GPU run.

Keep original physics, split, preprocessing, action convention, precision, sampler and acceptance thresholds fixed for a claimed matched comparison. Label any intentional change and include a matched baseline. Preserve failures, unknown measurements, compute/data costs and seed limitations. Do not silently merge different archived protocols.

Keep raw assets, credentials, local environments and generated outputs outside Git. Use manifests and bounded reviewed figures/reports. New expensive experiments and publication are separate owner-approved actions. Use a concrete pull-request description explaining behavior, validation and limitations.

Historical source maps may reference a canonical file in the current core. Before changing that file, preserve its old version in the archive and redirect the historical mappings/indices to that preserved copy. Update the relevant public hashes; do not rewrite an old experimental identity to describe new code.
