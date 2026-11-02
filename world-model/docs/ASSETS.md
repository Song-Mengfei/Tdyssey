# External assets

The Git repository contains source, bounded configuration/report evidence and one curated saved-result figure. It excludes datasets, trained weights, optimizer state, raw prediction arrays, videos, environments and download caches.

| Asset | Identity / availability |
| --- | --- |
| Published GPC phase-two denoiser | Pinned HF revision, file size and SHA-256 in `manifests/gpc_world_model.json`; public upstream, not bundled |
| Official pretrained policy | `manifests/official_dp.json`; separate from GPC weights |
| GPC reference policy/data schema | `manifests/gpc_policy.json` and `manifests/upstream.json` |
| E2 full local training state | SHA-256 `c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187`; 107,687,711 bytes; private local file, no public hosting supplied |
| Project data and selected clips | Categorized evidence manifests, schedules and retrospective DATA_LEDGER; separate physics/splits/exposure histories |
| Later checkpoints/raw frames | Referenced by per-run result metadata; stay outside Git and are not all guaranteed still present |

```bash
python -B tools/verify_asset.py --id E2-full-training-state --path /path/to/epoch2.pt
```

That optional check hashes the supplied file without loading Torch/pickle or modifying it. Dataset/weight publication, storage hosting and access terms require a separate owner decision. The source release does not silently transfer or relicense them.
