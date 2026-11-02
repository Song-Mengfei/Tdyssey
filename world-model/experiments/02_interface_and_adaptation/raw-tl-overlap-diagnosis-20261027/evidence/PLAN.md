# Frozen offline command plan

Two commands only, each external wall limit60s, aggregate120s, CPU affinity2
cores, working set1GiB. No Torch/SDK/CUDA/model weights/forward/backward,
optimizer, physics, collection, H4 or GUI operation. No retries after failure.
Installed NumPy/Pillow and existing pure geometry/source validators only.
Output plus temporary files stays under16MiB; script payload limit15MiB leaves
1MiB for command receipts/stdout. No production/model/loss changes.

1. `case-and-coverage`: read the two named case JSON/NPZ, sibling NPY/PNG;
   verify declared hashes and deduplicate by request/input/prediction/truth/model
   identities. One log snapshot at its initial byte length, maximum64KiB; never
   poll. Show four raw histories, truth and the exact prediction, plus separate
   diagnostic truth-mask/crop panels. Read the2328 frozen schedule and metadata,
   then only24 train record JSONs, no train image arrays or development/held
   payloads. Compute exact polygon coverage with existing official geometry.
2. `train-frames-and-checks`: only after1 succeeds, inspect the frozen selection
   of at most8 source-train windows, five frames each (maximum40 decoded frames).
   Each shape uses one first trajectory/anchor in fixed bins: exactly0,
   positive<.5, .5<=coverage<.8, coverage>=.8. Empty bins stay empty. Selection
   precedes image reads and never uses prediction quality. Stream `images.npy`
   within NPZ: compressed seeking can inflate/discard prefix bytes, explicitly
   reported; only selected five frames become arrays, never a full episode.
   Validate source hashes, show all selected histories/targets, check source and
   prepared commit message in this same bounded command.

Truth masks use the existing pixel-centre mapping `(j+.5)*512/96-.5`, known
official shape and recorded object/goal/pusher state. Exclude one96-pixel edge
margin and dilated radius15 pusher. Overlap=eroded object & eroded goal;
nonoverlap=eroded object outside dilated goal. Boundary/empty denominators remain
explicit. RGB evidence uses nearest fixed source palette prototypes, with exact
ties separate and continuous palette distances also reported. No fit or tuned
threshold, no predicted pose/coverage claim, no recolouring/repair. Truth ROIs
are offline labels only; geometric displacement can also reduce gray retention.

Training coverage diagnostic bins are not changed success thresholds. Original
success remains strictly>0.95; counts>=.8/.9/.95 and their scene counts are
separately disclosed. The two saves may be one case and are never two independent
scenes. Current user-driven GUI activity is not part of this diagnosis's costs.
