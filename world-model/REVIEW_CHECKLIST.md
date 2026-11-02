# Owner review before GitHub publication

This curated source bundle is maintained under `world-model/` in Tdyssey. Its integration is reviewed through a pull request. No data/weight release or new training is included.

The app's 3 November update is separate from the original source-only assembly: the owner selected HMF500 for current1/4-step use, external asset pointers are ignored, and the previous R3B source is archived. Review the [app guide](apps/r3b-playground/README.md) and [saved-only selection evidence](docs/APP_MODEL_COMPARISON.json). GPU startup checks are separate from the source-only release check.

## Included

- Core WM/environment/data/training/evaluation/inspection code, versioned configurations and existing tests.
- Scientific source from all six available workspaces and saved source freezes, with original/public SHA-256, deduplication and explicit upstream references.
- Twelve experiment categories, including failed preflights, incomplete/report-failed runs, history/region objectives, high-motion fitting, AR4 loss, Min-SNR/GDL/FFL and G/T definitions.
- Bounded methods/configuration/result evidence, the literature bibliography, recorded negative results and a curated validation figure.
- Per-profile environment manifests/locks, pinned external repositories, asset identities and reconstruction/verification utilities.
- README in English/Chinese, installation/reproduction/architecture/loss/results guides, contribution rules, CI and PR/issue templates.

## Intentionally external

- Virtual environments, dependency/download caches and nested Git histories.
- Private datasets, pretrained/trained weights, optimizer state, raw frame/trajectory arrays and full videos.
- Chat transcripts, task/heartbeat/callback receipts, local launch registration and deployment coordination.
- Large raw/per-step measurement output; its original paths/hashes are indexed where inventoried. Missing/deleted historical assets remain explicitly qualified.

## Owner decisions still needed

- [x] Integrate the curated bundle under `world-model/` in the existing Tdyssey repository.
- [ ] Confirm author names, attribution and rights to publish the combined first-party/adapted code.
- [ ] Choose the project-code license; no MIT/Apache grant has been invented.
- [ ] Decide whether private datasets/checkpoints will remain private or receive a separately reviewed download location.
- [ ] Review the experiment taxonomy and scientific conclusions, including failed/unknown results.
- [x] Owner authorized a new branch and pull request to the existing Tdyssey remote; merging remains a separate review action.

## Verification

Run `python -B tools/check_upload_privacy.py --scope git` before uploading. This checks the exact Git publication candidates. To review actual local files including ignored content, run it without `--scope git`. Runtime logs and local asset configuration are never publication files. Any actively locked logs retained at the owner's request must be deleted after that service exits; do not upload the whole working directory as an unfiltered archive.

See [release validation](docs/RELEASE_VALIDATION.json) for actual checks and counts. Source-only checks do not prove clean-machine simulator/CUDA/training compatibility. Archived workers need matching assets, reconstructed source/evidence layout and regenerated frozen identities before execution.
