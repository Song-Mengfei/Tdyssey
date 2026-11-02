# Stage2A independent post-run review

Integrity/scope audit passed: **True**. The one fixed formal TRAIN batch contains162unique requests/scenes/seeds,18letters x9spaces,10,368saved controls and103,680saved100Hz substeps. No invalid transitions were found; H1/H4 valid candidate totals are9882/9396.

All22 source rows match both their snapshots and current files. All201protected stage1/1B files and original best/last weights match. Combined persistent footprint at this audit is1,538,934,276bytes below3GiB.

All162 per-slot prefix count inputs and fixed oldTRAIN counts were independently reconstructed from saved actual collision points. Pure geometric plan_scene replays reproduce every selected initial arc. No environment was constructed for replay, and no physics step was taken. The final reconstructed bin contact-control counts match saved coverage.

Total supervised controls are14,336; all-native historical plus supervised are48,896. Old native costs remain1,899charged/1,898completed optimizer attempts and5,352.516seconds. This stage has0model calls,0optimizer attempts and0server jobs.

Independent H1 motion-target failures: **M**. These quality results remain separate from physical validity; no replacement, extra controls or next stage is authorized.

Reviewer executed0 environment initializations, physical controls, model calls or optimizer updates. Root must save/report this batch and stop for parent review.

Actual initial gaps were independently recomputed from saved polygon poses: center18 at>64; each of8boundary classes has6 each in0–8/8–32/32–64, with both requested walls checked in every corner. All initial dynamics are zero. Combined18CPU prechecks are bound to the same source/config/request/dedup hashes.
