# Stage3: stopped before training at the first native baseline batch

**No microfit was performed.** The single authorized run stopped after28.344 seconds and0 optimizer attempts/updates when the entry rejected one finite, out-of-range original GPC RGB output. This is an initialization/baseline attempt with missing training and development measurements; it is not a200-step fitting failure or evidence against GPC capability.

The native sampler returned its first4 TRAIN images normally. The entry range guard found6 negative values in A (`train-2b-018:11`), minimum−0.0196077823638916 and maximum1.0. This is larger than a small rounding discrepancy. The raw array/status remains unchanged; no clipping, shape repair, threshold change or model rerun was performed.

|Measurement|Planned windows|Actually generated|Missing|
|---|---:|---:|---:|
|Original frozen TRAIN|18|4|14|
|Original frozen development|36|0|36|
|Fitted TRAIN|18|0|18|
|Fitted development|36|0|36|

The other3 first-batch images pass finite/range checks only. B and C have gray-object IoU approximately0.003215/0.029126 under the pre-frozen gray palette. E has no detectable gray-object or blue-pusher mask. B/C/E show red foreground in the unmodified RGB; an empty gray mask does not mean the entire image is blank. These are4 partial original-baseline outputs, not fitted-model or generalization results.

All18/36 original denominators remain in per-window scores (external historical reference: `score/per-window.csv`), the raw TRAIN array (external historical reference: `predictions/train18.npz`), development array (external historical reference: `predictions/dev36.npz`) and9 gallery pages. Missing frames retain NaN/status records. The fitted TRAIN mean object error1.0 and0/18 wins are the preregistered missingness penalties; no fitted frame was generated. The hold mean object error is0.2325831710. The20%/12-of18 diagnostic is not demonstrated, and penalty means must not be read as observed fitted performance.

|Question|Evidence from this run|
|---|---|
|Initialization and input/native forward connection|Original published SHA and20 source files verified; full8,729,827 count/all-trainable assertions passed; first native4-image batch returned. Original RGB/action timing and batch-paired noise were frozen.|
|Backward/optimizer training chain|Not exercised:0 attempts,0 draws,0 losses,0 gradient measurements,0 scheduler/optimizer steps. CPU tests do not replace actual training evidence.|
|Motion fitted on TRAIN18|Unmeasured; training never began.|
|Independent DEV36|Both original/fitted model inference unmeasured; no generalization conclusion.|
|Fitted shape quality|Unavailable. Fitted18 shape assessments are unknown, not shape passes. Partial original red appearance/gray-mask failures are shown.|

Before model work, one visible-motion+contact H4-legal window per seen training letter was frozen by the parent-specified SHA256 rule/seed2026102904. Every selected TRAIN target differs from its last observation. The original stage2C development36 and14 batch plans were retained, with same ordered batch members/shapes/seeds across planned frozen/fitted phases. No validation or old probe entered training. Frozen windows (external historical reference: `frozen-windows.json`), batch plan (external historical reference: `paired-batches.json`).

The only checkpoint is run/last.pt (external historical reference: `run/last.pt`), a35,096,361-byte complete t0 pre-baseline archive. It contains model, empty-initial optimizer state, scaler, scheduler, RNG, WindowOrder, progress, config, identity and ledger snapshot. SHA is eb7b8dfc17b1ae157ccc5f477cdcacc400891f1db54709c4ac428234d090c7cc. ZIP CRC and required pickle-key strings were checked without loading or restoring a model after the stop. Actual tensor-by-tensor restoration and optimizer gradient reachability were not tested in this run. No best.pt was created; old experiments/checkpoints remain unchanged. Archive audit (external historical reference: `checkpoint-archive-audit.json`).

The single model-work clock includes CUDA query, original loading, initialization, t0 atomic save, first native batch, failed callback cleanup and process cleanup:28.344 seconds. Ledger is closed,0 reserved/completed attempts. Full new-round ledger caps remain3000 attempts/5400 seconds; numeric remaining budget is3000/5371.656, which is not permission to retry. Old native1899 attempts/5352.516 seconds are separate and retained. Actual peak GPU memory was not written before the early error, so no historical peak is substituted.

The update cap200, latest new-update cutoff420 seconds and tail reserve≥180 seconds were installed but never reached. Operation guards and conservative time reserves do not promise hard real-time preemption of an individual CUDA or atomic-save operation; actual closed elapsed time is below600 seconds. Combined old/new/source/cache accounting and atomic temporary reserve remain below3GiB.

Model/loss/optimizer/sampler mathematics remained original; the small entry/callback/accounting changes were frozen before model work.34 focused CPU tests,18-seen renderer-only palette checks and independent input/source audits passed. Independent final integrity review passed23 checks; these checks establish provenance/accounting, not microfit success.56 frozen files and381 protected data/index/weight files are unchanged.

A reviewable unapplied follow-up diff (external historical reference: `PROPOSED_NOT_APPLIED-finite-range-policy.diff`) would retain finite out-of-range outputs as scored generation failures while allowing the remaining baseline/fit to proceed; nonfinite/structural/OOM failures would still stop. It changes neither RGB values nor score penalties, but it changes this entry abort policy and requires a new parent release. It has not been applied or run.

Pure offline scoring took6.235 seconds outside the model clock. [Diagnosis](DIAGNOSIS.md), [score report](score/SCORE_REPORT.md), first TRAIN gallery (external historical reference: `score/gallery-train/train-01.png`), all scored files (external historical reference: `score/files.json`), [independent review](review-postrun.md), and [budget review](review-budget-postrun.md) retain all failures and missingness. Physical controls remain39,680 supervised/74,240 all-native; this stage adds0. No full319/H4/unseen evaluation, server run or packaging occurred.

The actual failure/result callback to the parent chat succeeded; receipt (external historical reference: `callback-result.json`) is saved and state is waiting_parent. No second model execution is authorized.
