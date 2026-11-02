# Asset cleanup inventory and execution boundary

Human authorization is in PARENT_HANDOFF.txt (external historical reference: `../PARENT_HANDOFF.txt`). The inventory agent only read files and wrote its own manifests/scripts. The root owner performed the actual literal PowerShell deletions after path, ancestor-reparse, size and SHA checks.

## Executed cleanup

Seven retired checkpoints, **596,831,938 logical bytes (569.18 MiB)**, were actually removed. See [per-file proposal](delete-plan-checkpoints.json) and actual deletion receipt (external historical reference: `checkpoint-deletion-receipt.json`), which contains absolute paths, bytes, SHA256, category, reason, timestamps and confirmed absence.

- Native20261028 stopped1668-update `best.pt` and `last.pt`: different hashes and different historical selection roles, both retired; they are **not duplicate files**.
- Stage3B129 and Stage3C250 complete microfit checkpoints: the later lineage restarted from the published GPC initialization rather than continuing these rejected development models.
- R2C1R1 complete300 checkpoint: exactly continued by R2C3 to the protected6000 state.
- R2C3 model-only1000/3000 milestones: superseded for current use by protected6000/7499; their fixed raw predictions, paired metrics and O/C/B/X focus galleries remain.

The old checkpoints cannot now be loaded again. Historical reports and manifests retain their original conclusions and old filenames/hashes; they are provenance records, **not a claim that every original large artifact still exists**. The generic legacy `scripts/native_rgb_world_model.py` default `training/best.pt` is intentionally no longer available for that retired experiment. A future invocation must explicitly select an appropriate available model; no silent model substitution is allowed.

## Retained assets and unresolved dependency

The complete current7499 and6000 checkpoints remain at their original locations and have matching hashes after deletion. The published GPC initialization, unique official policy, source/environment/third-party trees, teammate original `Tdyssey/tdyssey_pushing_data`, all user videos/raw actions and B penetration evidence are outside the explicit deletion set. Current saved session/export/recentclip, all fixed O/C/B/X inputs/predictions and validation/test assets remain. Post-cleanup verification (external historical reference: `retained-assets-verification.json`) records actual checks and their limits.

No old training data was deleted. R3A has774 old TRAIN NPZ files totaling58,388,513 bytes (sidecars another1,314,026 bytes). The full old loader and trajectory audit use the original full manifest. Selecting only the roughly43 visual/physics example inputs would not preserve the full original loader/audit contract. Root explicitly chose to keep these approximately59 MB rather than damage that evidence after reclaiming569 MiB from weights. This retained set is **not described as a minimal extracted subset**, and no full backup was copied. Retention index (external historical reference: `minimal-fixed-input-retention.json`), [empty old-data deletion plan](delete-plan-old-train.json).

All144 R3A seen-validation families and their sidecars (11,014,405 bytes) remain. Sealed test payloads were not decoded by this inventory; deletion never targets data files. No source, `.git`, working tree, environment, third-party source or other author's original data was a cleanup candidate. No ZIP or duplicate checkpoint-delivery package was present in the main experiment root.

## Storage accounting

Initial read-only scan:14,013 files, main experiment1,643,990,879 bytes; shared deduplicated1,752,289,650 bytes, disk free160,865,648,640 bytes. This scan preceded live-session preservation, the inventory's own output and concurrent pilot files.

Immediately around actual deletion, root measured main artifacts1,659,438,029→1,062,612,120 bytes; filesystem free160,844,292,096→161,439,395,840 bytes. Exact logical removal is596,831,938 bytes. The directory and disk-free differences need not equal it because other authorized diagnostics were writing concurrently. Latest shared snapshot, remaining256 MiB atomic reserve and hash checks are in retained-assets-verification.json. The snapshot precedes its own output and later pilot files; the root can rerun `verify_retained_assets.py` at final delivery.

## Checks and reproducibility

- Inventory scans skipped Windows reparse/junction paths and found none in the scanned experiment roots.
- Root resolved each actual checkpoint and checked every ancestor before any deletion; all7 size/SHA checks passed, then all7 were confirmed absent.
- The verification script reads hashes only: zero model loads, GPU/model calls, physics controls or optimizer updates.
- All saved NPZ from the initial main-experiment inventory are rehashed without decoding arrays; current weights, original GPC, user B video copies and all user recordings remain readable. The unique official policy is fully checksummed but not loaded; no prior expected digest was supplied in this child task, so it is not falsely marked as matched to one.
- Other checkout artifact directories are inventoried read-only and are not deletion candidates. The two user Downloads/homework PDFs observed outside experiment roots are unrelated and untouched.

Inventory source: all initial file rows (external historical reference: `inventory-base.json`), compact final inventory (external historical reference: `inventory.json`). No broad directory deletion, `git clean`, shell chaining across deletion shells or full backup was used.
