# Reproduction and archive contract

## What was verified in this preparation

This is source organization and saved-result curation. No dependency installation, CUDA/model execution, training, data collection or simulation was performed. The release check verifies source syntax, structured metadata, mapped-file hashes, documentation links and prohibited file patterns. It cannot establish scientific correctness or clean-machine training compatibility.

## Two source layers

The root `tdyssey/`, `scripts/`, `configs/`, `tests/` and `apps/` retain the main local checkout's working-file implementation. Older workspaces contain different modules and versions, including direct-simulator DPPO and compatibility diagnostics. Their unique files are stored under `archive/workspaces/`; repeated files point at a canonical copy in `manifests/source-map.json`. Each profile preserves its own dependency metadata. No conflicting versions were silently merged.

Per-run source indices retain the relative path within the original workspace, including nested source freezes. Exact unmodified upstream duplicates are represented by an upstream path/reference instead of vendored source. Local scheduler/deployment/coordination glue is excluded and recorded by disposition. Failed scientific preflights, objective code, metrics, tests and scientific configuration remain in scope.

## Restore an inspectable first-party layout

```bash
python -B tools/materialize_sources.py --profile Tdyssey-raw-tl-portable-main --destination ../wm-source-review --include-experiments
```

The destination must not exist. This copies available first-party source only and runs no code. Normalized historical strings remain tokens unless `--local-workspace /path/to/private-assets-workspace` is supplied. Existing assets and original workspaces are never overwritten. This tool does not restore raw evidence, external upstream duplicates, archived launch permissions or a complete binary execution environment.

## Requirements for a new scientific execution

1. Select the actual protocol, source profile, dataset manifest/split and full parent-state lineage; do not replace an original-physics baseline with an earlier finite-force default.
2. Obtain the pinned external repositories and matching private assets. Verify checkpoint/data hashes against the supplied manifests.
3. Place or configure required sibling run sources and evidence in the historical `artifacts/<run>` layout. Archive workers often load earlier helpers, selected clips, scoring calibration and cached baseline outputs dynamically.
4. Regenerate absolute path bindings, source hashes, launch identity and bounded output configuration for the new machine. Publication normalization deliberately makes old bindings invalid; archived records are not fresh authorization.
5. Run the existing relevant preflight and focused correctness tests before an explicitly budgeted experiment. Retain seeds, precision, sampler, actions, split, gates and all failed attempts. Packaging has not performed this runtime validation.

## External and missing material

`manifests/external-evidence.json` records inventoried text evidence left external because it is raw/large/runtime-local. It is not an inventory of every byte array on disk. Datasets, checkpoints and frame archives are intentionally not included. Small retained configurations may still reference such assets; those references are not promised public downloads.

The earlier retired structured-model attempt and deleted native checkpoints are described by saved ledgers with explicit historical-only/missing qualifications. No missing experimental source or weight is invented. Six available workspaces plus their saved source freezes were inventoried; this is not an export of every historical Git object or chat message.
