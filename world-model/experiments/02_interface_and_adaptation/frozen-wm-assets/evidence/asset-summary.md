# Frozen model asset audit

Final machine-readable evidence: `resolved-v2/asset-audit.json`.
Download attempts and preserved network failure: `download-events.json` and
the initial `asset-audit.json`. Earlier `resolved/asset-audit.json` retains the
first completed tensor comparison. All files are local ignored artifacts.

The exact manifest checkpoint is available at `denoiser.pth`: 35,043,916 bytes,
SHA-256 `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`.
It was fetched from `han2019/gpc_checkpoints` revision
`c16a467ec176af19bba5136564a81d8d61d72af9`, path
`world_model_checkpoint/phase_two_checkpoint/denoiser.pth`. One DNS attempt
failed before transfer; the single-command approved network retry succeeded
with the existing bounded fetcher. No Hugging Face cache was created.

The previous file in the original repository remains untouched:
`third_party/gpc_code/gpc_opt_evaluation/csgo/denoiser.pth`, 35,043,916 bytes,
SHA-256 `15126113b7490fa976290c62ad4c7a92734bc898664522706c67a4aa6bff47b2`.
It is the file tracked by the pinned upstream Git source itself, blob
`7b0d656212dd80a8cb9274c3115b95cff4176bf7`; its current Git blob matches that
tracked blob. Thus the local mismatch is explained by use of a different
upstream-bundled asset, rather than local corruption of the HF phase-two asset.

Both files were loaded safely on CPU with `weights_only=True, mmap=True`.
Their 298 state keys match, but 296 tensor entries differ. Only the two
128-element Fourier-feature buffers match. This is different parameter content,
not merely different serialization. Canonical state fingerprints are:

- Exact HF phase-two: `6b5c3a0ddf11c7812cca3846f891fbb6fed83bc28e120e45cf98c73a787df796`.
- Upstream csgo bundled: `5c3ccc0240b2ce4460843383c1bf0f2e8db21e31193bcd3a0927cf0dabf9a160`.

Each has 8,730,083 state elements, comprising 8,729,827 parameters and 256
registered buffer elements. File SHA-256 values before/after inspection match.
The GPC source is commit `a96146e568826ef46134f73c22fdf59a675e7264` and all 20
manifest-pinned source file hashes pass.

The curated world-model publication snapshot passes all seven byte/hash checks:
772,463 bytes against its 1 MiB cap. A read-only filename inventory of the
`cs5478` project tree, including ignored artifact directories, found no old
custom `best.pt`, world-model checkpoint, or catalog JSON. The historical
custom-L comparison is unavailable in this searched scope. Catalog or dataset
contents were never opened; held-out model-test directories were excluded.

Verification: three focused asset tests pass, including canonical equality
across different serializations and rejection of changed tensors/keys. No
physics, resets, model inference, training, environment changes, or messages to
the separate DPPO analysis chat were performed by this audit. The source and original asset are
read-only dependencies.
