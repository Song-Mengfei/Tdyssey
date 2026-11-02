# Read-only review of saved R3B metrics

No model, GPU, physics, renderer, new scoring or protocol changes. All values are read or grouped from the saved metric rows.

|H|motion stratum|N|copy ROI|t0 ROI|epoch1 ROI|t0/e1 gray IoU|t0/e1 P/F/U|
|---|---|---|---|---|---|---|---|
|1|translation_only|22|0.115331|0.057351|0.026835|0.766009/0.878489|[0, 0, 22]/[1, 0, 21]|
|1|rotation_only|19|0.083295|0.053861|0.032939|0.782334/0.860288|[0, 0, 19]/[0, 0, 19]|
|1|both|3|0.114217|0.054710|0.032880|0.792026/0.864926|[0, 0, 3]/[0, 0, 3]|
|1|other|100|0.012356|0.034146|0.012010|0.867300/0.958249|[59, 9, 32]/[61, 0, 39]|
|4|translation_only|14|0.189709|0.089950|0.065087|0.602202/0.694079|[2, 2, 10]/[1, 2, 11]|
|4|rotation_only|2|0.091541|0.138953|0.078210|0.464679/0.684897|[0, 1, 1]/[0, 1, 1]|
|4|both|50|0.185296|0.096296|0.081829|0.569124/0.618266|[4, 2, 44]/[4, 5, 41]|
|4|other|78|0.011084|0.060959|0.019201|0.770451/0.928544|[44, 8, 26]/[46, 2, 30]|

## Material limits

- Shape fail includes hard_generation_failure such as empty blue pusher, so aggregate fail reduction is not gray-object-only shape improvement.
- Natural obvious-motion H4 shape P/F/U6/5/55 to5/8/53: visible motion shape failures increase despite lower image errors.
- H4 nearstatic copy-last remains better: objectROI0.000435 vs epoch1 0.013422; drift0 vs0.154136px.
- Raw corner count differences are thresholded raster diagnostics; occluded and unavailable counts remain; mismatch cannot establish rigidity.
- Physical pose means use different available subsets unless common paired table is used.
- Old4 are familiar regression cases,not independent validation;oldB H4 angle165deg to167deg remains poor/shapeunknown.
