# Installation

## Source-only review

```bash
python -B scripts/check_repo.py
```

No installation, checkpoint loading or network is required. Existing model/runtime tests are not part of this command.

## Recorded dependency profiles

The current `pyproject.toml`, `uv.lock` and `.python-version` are copied from the main local source profile. That manifest targets Linux x86_64 / Python 3.12, with mutually exclusive CPU and CUDA 12.4 Torch profiles. Historical Windows work used a different existing environment; do not infer Windows lock support from the current Linux-only lock. Each older profile retains its own manifest and lock under `archive/workspaces/`.

With installation and GPU compute explicitly intended on a new machine:

```bash
uv sync --locked --extra cu124 --extra gpc
```

Use the `uv` version recorded in `pyproject.toml`. This command installs packages; it was not run while preparing this release. Optional official-policy work uses the `dp` extra. The original lock is retained, not re-resolved during packaging. Supplementary server requirements remain in `requirements/`; they are historical profiles, not an extra unconditional installation step.

## Upstream source repositories

```bash
python tools/fetch_sources.py --name gpc --destination third_party/gpc_code
python tools/fetch_sources.py --name diffusion_policy --destination third_party/diffusion_policy
```

These are explicit network commands and stop if the destination exists. They checkout the full SHAs in `manifests/upstream.json`. DPPO and DIWA are implementation/research references and are not needed for native GPC inference. They can be fetched separately using `--name dppo` or `--name diwa`; they are not represented as experimentally validated replacements.

## Assets and explicit runtime stages

Read [assets](ASSETS.md). Existing checkpoint retrieval scripts are in `scripts/fetch_gpc_world_model.py` and `scripts/fetch_official_assets.py`; inspect their `--help` before downloading. The locally trained E2 parent and project datasets have no public download in this release.

```bash
python -B scripts/native_rgb_world_model.py --help
python -B tools/verify_asset.py --help
```

`native_rgb_world_model.py` exposes separate stages. Its original default controller is **finite-force**, not the strict-original protocol used by later O/C/B studies. Select the matching source profile and configuration; do not use that old default as a substitute for the later original-physics baseline. [Reproduction](REPRODUCIBILITY.md) explains historical worker dependencies and path normalization.
