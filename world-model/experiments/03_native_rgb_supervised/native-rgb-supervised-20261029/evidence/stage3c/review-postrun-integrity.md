# Independent stage 3C post-run review

Integrity and protocol audit passed: all 59 checks are true. Actual six-component restoration (model, optimizer, scaler, scheduler, RNG and WindowOrder) preserved exact state at update 129, followed by 121 completed attempts/updates to update 250. New draws are 1936, cumulative draws 4000, using only the same 18 TRAIN windows. All 296 parameter tensor hashes changed from the restored 129 model. Inherited first-gradient norms remain explicitly 3B evidence; they are not relabelled as new 3C measurements.

The continuation charged 423.859 model-work seconds, including construction/load/restore, training, final native inference, complete saving and CPU deserialization. Global new-round cost is 250 attempts/updates and 868.484 seconds, preserving the earlier 444.625-second history. Limits remain local121/600 and global3000/5400. Remaining2750 attempts/4531.516 seconds do not authorize another execution.

All 96 frozen source/input files and snapshots and all 742 protected older files match their bytes/SHA. Final native inference generated exactly 54 H1 images in the original 14 batches once; original frozen and update129 outputs were referenced, never predicted again. All final raw arrays, range/status rows, batch seeds, truths, masks, original baseline references and final parameter digests match. No clipping, remapping, recoloring or threshold change occurred.

The single complete final last.pt is 105,290,926 bytes, SHA256 7a3c3fed60f444536afe8999d6f6346d1ce41258f8df7d87dc78f18bcb4ba28f. Root's actual CPU checkpoint verification occurred within the charged clock; the independent auditor used only ZIP CRC/opcode-string/hash checks and did not load Torch or restore a model. Peak memory includes restoration/boundary and training/native counters. Existing old/new/source/cache files plus 256 MiB atomic reserve remain below 3 GiB.

Independent same-protocol gray-mask recomputation agrees with all 216 four-method rows. TRAIN gray error1-IoU is0.1851714990 versus hold0.2325831710:20.3848248% reduction and15/18 strict wins, with0 hard generation failures. The registered20%+12/18 motion microfit diagnostic passes, narrowly. All18 windows also improve over update129 (mean error0.2280759621). This establishes only the specified small-sample diagnostic, not generalization or shape acceptance.

Independent DEV36 remains worse than hold: final error0.2594243475 versus0.1960324595,6/36 wins and2 hard generation failures. It improves from update129 error0.2743519650. The moving31 subgroup remains worse than hold (0.2696184638 versus0.2237496229); static4 retain drift (final0.1696457812 versus hold0). The intermediate1 remains separate.

Shape is not passed: TRAIN has0 assessable/18 unknown; DEV has1 failure among3 assessable plus33 unknown. Original18/36 denominators,31/4/1 DEV strata, all18 comparison images, and all216 four-method CSV rows remain intact. Loss declined from0.0024270865 at129 to0.0019337337 at250 as an auxiliary measure only.

Reviewer performed zero Torch/model loads, GPU calls, optimizer updates, physics steps or training-iterator calls. No H4/full319/unseen evaluation, new physical collection, server work, packaging or later stage occurred. Return the evidence to the parent and stop.
