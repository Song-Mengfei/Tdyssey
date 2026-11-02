# Independent stage3 post-run review

Integrity and protocol audit: true. Experiment outcome: pretraining inference-range abort, before any loss, gradient, training draw, optimizer attempt, or update.

The unique closed model-work stage charged 28.3440000000046 seconds,0attempts/0updates. New aggregate remains3000attempts/5400seconds; historical1899attempts/5352.516seconds is separate. Remaining new budget is 5371.655999999995 seconds, without authorization to rerun.

The first frozen TRAIN batch generated4of18 rows. A/train-2b-018:11 retained6 negative RGB values, minimum -0.0196077823638916, maximum1. This is not rounding epsilon. The other3 rows are finite/in-range; unchanged offline score additionally marks E/train-2a-021:28 empty-object and empty-pusher. Here empty-object means no gray-object mask under the frozen palette, not a blank RGB image: the first gallery visibly contains red objects/fragments in frozen B/C/E, unlike gray truths. No recoloring or threshold change is applied. All other14 TRAIN baseline rows, all36 DEV baseline rows, and all18+36 fitted rows are missing and retained.

All18/36 denominator rows, fixed orders,54raw comparison references,3TRAIN+6DEV gallery pages, statuses, and score hashes match. Fitted18/18 and36/36 are missing-data failure penalties with0shape-assessable outputs; their means are not measured trained-model quality. TRAIN motion target is unproven and cannot pass. This does not establish microfit failure, generalization failure, or rejection of the GPC route.

All56 frozen source/input/protocol files and snapshots and381 protected C/index/joint/released-init/old-weight files remain unchanged. Native raw arrays are not clipped or repaired.

The sole t0complete last.pt is35,096,361bytes, SHA256 eb7b8dfc17b1ae157ccc5f477cdcacc400891f1db54709c4ac428234d090c7cc. Independent ZIP CRC and pickle-string checks find all10required state-key names. No Torch loading/model restoration was performed; this is an archive-structure check, not a semantic restore or updated-optimizer test.

Combined old/new/native source/cache files total 1853694044 bytes; plus256MiB atomic reserve remains below3GiB. Physical controls remain39680supervised/74240historical-inclusive, with0new steps.

Reviewer:0model loads/inferences,0Torch loads,0optimizer updates,0env steps,0training-iterator calls. Stop and return the evidence to the parent; no altered condition or second run is authorized.
