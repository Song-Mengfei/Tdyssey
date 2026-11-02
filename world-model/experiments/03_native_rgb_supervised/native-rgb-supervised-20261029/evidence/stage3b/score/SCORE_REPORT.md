# Stage3 raw native RGB evidence

TRAIN18 dynamic fitting diagnostic passed: False.
This is a bounded18-sample microfit diagnostic, not generalization or full acceptance. All18/36 failure rows remain in primary means.

| Cohort | Method | N | Mean gray1-IoU | Gray foreground RGB MAE | Raw contour(imagepx) | Hard failures | Legacy shape failures/assessable;unknown |
|---|---|---:|---:|---:|---:|---:|---|
| train | hold | 18 | 0.23258317101512865 | 0.0485253625003286 | 0.42706694380189414 | 0 | 0/0; unknown18 |
| train | frozen | 18 | 0.9906202462099761 | 0.4412381729612275 | 70.3374913630675 | 11 | 0/0; unknown18 |
| train | fitted | 18 | 0.22807596210836506 | 0.05208212736707432 | 0.45635414144882375 | 0 | 0/0; unknown18 |
| development | hold | 36 | 0.1960324594762658 | 0.04162342769536733 | 0.34741450102387506 | 0 | 0/3; unknown33 |
| development | frozen | 36 | 0.9944729063530855 | 0.49629508862525173 | 74.11487887914569 | 26 | 1/1; unknown35 |
| development | fitted | 36 | 0.27435196499414033 | 0.06308500054966887 | 0.5463561963056285 | 1 | 2/3; unknown33 |

TRAIN target details: {'denominator': 18, 'method': 'fitted', 'baseline': 'hold', 'mean_object_error': 0.22807596210836506, 'baseline_mean_object_error': 0.23258317101512865, 'relative_reduction': 0.01937891244276834, 'strict_wins': 9, 'ties': 0, 'method_hard_failures': 0, 'baseline_hard_failures': 0, 'required_relative_reduction': 0.2, 'required_strict_wins': 12, 'required_denominator': 18, 'protocol_valid': True, 'truth_valid': True, 'hold_valid': True, 'passed': False, 'qualification': '18-window microfit diagnostic only. Shape failures/unknown and development evidence remain separate.'}
Object/pusher metrics are separate. Gray RGB ROI is fixed from last/true masks, excluding fixed blue-pusher regions; no prediction-dependent ROI. Actual RGB arrays remain unchanged.
Development overall,31moving/4near-static/1intermediate,letters and actual boundary groups are in JSON. Moving uses pairedhold/frozen comparisons; static uses drift/shape, without requiring a win over zero-error hold.
All windows are shown in frozen supplied order: TRAIN3 pages/development6 pages,5 columns last/true/hold/frozen/fitted. INVALID placeholders indicate unrenderable raw arrays, never repaired predictions.
Legacy alignment is auxiliary visible-mask registration:10degree global,1degree local andhalf-imagepixel translation grid. This grid does not establish pose accuracy. Occluded/clipped shape measurements remain unknown, not passed.
Original NPZ source paths/hashes, per-window failures, visible components/holes/area/fragments, legacydeformation assessability and optionalunoccluded truth-mask diagnostics are retained in JSON/CSV.
