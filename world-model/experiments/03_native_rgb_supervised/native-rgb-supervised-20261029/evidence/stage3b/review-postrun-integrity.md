# Independent stage 3B post-run integrity review

Integrity and protocol checks pass. The single corrective run completed 129 optimizer attempts and 129 updates, with 2064 draws drawn only from the fixed 18 TRAIN windows. All 296 first-gradient records are finite and positive; all 296 before/after parameter hashes changed consistently across both cohorts. This audit cross-checks the root's in-budget GPU evidence without loading a model.

3B charged 416.281 seconds; original stage 3 plus 3B charged 444.625 seconds, below the conserved 600-second ceiling. The old 28.344-second failed stage remains unchanged. Last update completion was 390.797 seconds, before the 391.656 cutoff. Global limits remain 3000 attempts/5400 seconds; unused resources do not authorize another run.

All 108 native RGB rows are finite and preserved. Frozen TRAIN5/18 and DEV13/36 retain out-of-range failure statuses; fitted18+36 are in-range. Per-row ranges, raw statuses, all14 paired batch members/seeds/shapes, and combined last/true/hold/raw references match. Repeated old/new first4 baseline outputs are byte-identical, maximum absolute difference0. No clipping, mapping, recoloring or threshold change.

All64 frozen current files/snapshots,152 original stage3 files and381 protected older data/index/init/old-weight files retain bytes/SHA256. The sole complete last.pt matches the root's in-budget CPU deserialization evidence (129updates,296optimizer states,T_max3000,cursor2064); independent ZIP CRC and required-key-string inspection passes. Auditor did not perform a Torch load or semantic restore.

Independent CPU recomputation of the same frozen gray-mask IoU, fixed-object-ROI RGB MAE, blue-mask error, raw contour distance and failure flags agrees for all162 method rows. TRAIN fitted gray1-IoU is0.2280759621 versus hold0.2325831710:1.93789% reduction and9/18wins, below the predeclared20%+12/18 target. No TRAIN fitted hard generation failures occurred. This finite microfit has not demonstrated the required motion fit; it is not a general rejection of the GPC route.

Development remains independent31moving/4near-static/1intermediate. Moving fitted mean gray error0.2872416049 is worse than hold0.2237496229, with4/31wins and1hard failure. Static hold error is0 while fitted mean error is0.1741893150, indicating drift. Whole DEV fitted error0.2743519650 versus hold0.1960324595 is reported separately from TRAIN fit.

Shape cannot be reported as passed: TRAIN has0assessable and18unknown; DEV has2failures among3assessable plus33unknown. All18/36 rows,3TRAIN+6DEV gallery pages and162 CSV rows remain in frozen order. No missing/failed subset was removed.

Fixed-noise denoising loss fell from0.0342575386 to0.0024270865. That auxiliary loss result is separate from the unsuccessful raw-motion diagnostic. Report chain correctness, limited TRAIN motion fit, independent DEV degradation and shape uncertainty separately.

Reviewer performed zero model/Torch loads, GPU calls, optimizer updates, physics steps and training-iterator calls. No H4/full319/unseen evaluation, physical collection, server work, packaging or later stage was performed. Return evidence to the parent and stop.
