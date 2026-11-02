# Image-interface v2 development diagnosis

All reused calibration, branch and model frames were inspected development data. This report does not establish independent validation, generated-image uncertainty calibration, success precision or policy improvement.

Expected image rows: 1304; available: 1284. RGB hash caches reduce repeated measurement without removing any row.

Decoder inputs contain only saved RGB, known shape and fixed goal. Simulator truth supplies posthoc error labels. Missing object, shape, pusher and score eligibility are independent fields.

## Observed image errors

| Source | Shape | Version | H | Metric | Anchor mean | Measured / expected rows |
| --- | --- | --- | --- | --- | --- | --- |
| calibration | L | v1 | unavailable | position_error_pixels | 0.676392 | 12 / 12 |
| calibration | L | v1 | unavailable | angle_error_degrees | 0.222018 | 12 / 12 |
| calibration | L | v1 | unavailable | pusher_error_pixels | 0.516134 | 12 / 12 |
| calibration | L | v1 | unavailable | coverage_absolute_error | 0.00835995 | 12 / 12 |
| calibration | L | v1 | unavailable | rotation_direction_correct | unavailable | 0 / 12 |
| calibration | L | v2 | unavailable | position_error_pixels | 0.561982 | 12 / 12 |
| calibration | L | v2 | unavailable | angle_error_degrees | 0.229659 | 12 / 12 |
| calibration | L | v2 | unavailable | pusher_error_pixels | 0.455164 | 12 / 12 |
| calibration | L | v2 | unavailable | coverage_absolute_error | 0.0069884 | 12 / 12 |
| calibration | L | v2 | unavailable | rotation_direction_correct | unavailable | 0 / 12 |
| calibration | T | v1 | unavailable | position_error_pixels | 0.642192 | 12 / 12 |
| calibration | T | v1 | unavailable | angle_error_degrees | 0.219161 | 12 / 12 |
| calibration | T | v1 | unavailable | pusher_error_pixels | 0.591854 | 12 / 12 |
| calibration | T | v1 | unavailable | coverage_absolute_error | 0.00569933 | 12 / 12 |
| calibration | T | v1 | unavailable | rotation_direction_correct | unavailable | 0 / 12 |
| calibration | T | v2 | unavailable | position_error_pixels | 0.485949 | 12 / 12 |
| calibration | T | v2 | unavailable | angle_error_degrees | 0.0923751 | 12 / 12 |
| calibration | T | v2 | unavailable | pusher_error_pixels | 0.43953 | 12 / 12 |
| calibration | T | v2 | unavailable | coverage_absolute_error | 0.00465137 | 12 / 12 |
| calibration | T | v2 | unavailable | rotation_direction_correct | unavailable | 0 / 12 |
| forecast | L | v1 | 1 | position_error_pixels | unavailable | 0 / 144 |
| forecast | L | v1 | 1 | angle_error_degrees | unavailable | 0 / 144 |
| forecast | L | v1 | 1 | pusher_error_pixels | unavailable | 0 / 144 |
| forecast | L | v1 | 1 | coverage_absolute_error | unavailable | 0 / 144 |
| forecast | L | v1 | 1 | rotation_direction_correct | unavailable | 0 / 144 |
| forecast | L | v1 | 4 | position_error_pixels | 1.4892 | 4 / 144 |
| forecast | L | v1 | 4 | angle_error_degrees | 1.3956 | 4 / 144 |
| forecast | L | v1 | 4 | pusher_error_pixels | 1.33051 | 4 / 144 |
| forecast | L | v1 | 4 | coverage_absolute_error | 0.00313676 | 4 / 144 |
| forecast | L | v1 | 4 | rotation_direction_correct | 0 | 4 / 144 |
| forecast | L | v2 | 1 | position_error_pixels | unavailable | 0 / 144 |
| forecast | L | v2 | 1 | angle_error_degrees | unavailable | 0 / 144 |
| forecast | L | v2 | 1 | pusher_error_pixels | 1.25794 | 139 / 144 |
| forecast | L | v2 | 1 | coverage_absolute_error | unavailable | 0 / 144 |
| forecast | L | v2 | 1 | rotation_direction_correct | unavailable | 0 / 144 |
| forecast | L | v2 | 4 | position_error_pixels | 1.57662 | 4 / 144 |
| forecast | L | v2 | 4 | angle_error_degrees | 1.20938 | 4 / 144 |
| forecast | L | v2 | 4 | pusher_error_pixels | 1.06362 | 141 / 144 |
| forecast | L | v2 | 4 | coverage_absolute_error | 0.00262696 | 4 / 144 |
| forecast | L | v2 | 4 | rotation_direction_correct | 0 | 4 / 144 |
| forecast | T | v1 | 1 | position_error_pixels | 4.45863 | 6 / 48 |
| forecast | T | v1 | 1 | angle_error_degrees | 2.15924 | 6 / 48 |
| forecast | T | v1 | 1 | pusher_error_pixels | 2.53096 | 6 / 48 |
| forecast | T | v1 | 1 | coverage_absolute_error | 0.0149543 | 6 / 48 |
| forecast | T | v1 | 1 | rotation_direction_correct | 0 | 6 / 48 |
| forecast | T | v1 | 4 | position_error_pixels | 2.52667 | 34 / 48 |
| forecast | T | v1 | 4 | angle_error_degrees | 0.970562 | 34 / 48 |
| forecast | T | v1 | 4 | pusher_error_pixels | 1.96684 | 34 / 48 |
| forecast | T | v1 | 4 | coverage_absolute_error | 0.00894387 | 34 / 48 |
| forecast | T | v1 | 4 | rotation_direction_correct | 0.0833333 | 34 / 48 |
| forecast | T | v2 | 1 | position_error_pixels | unavailable | 0 / 48 |
| forecast | T | v2 | 1 | angle_error_degrees | unavailable | 0 / 48 |
| forecast | T | v2 | 1 | pusher_error_pixels | 1.60226 | 48 / 48 |
| forecast | T | v2 | 1 | coverage_absolute_error | unavailable | 0 / 48 |
| forecast | T | v2 | 1 | rotation_direction_correct | unavailable | 0 / 48 |
| forecast | T | v2 | 4 | position_error_pixels | 2.40048 | 34 / 48 |
| forecast | T | v2 | 4 | angle_error_degrees | 0.824589 | 34 / 48 |
| forecast | T | v2 | 4 | pusher_error_pixels | 1.88801 | 48 / 48 |
| forecast | T | v2 | 4 | coverage_absolute_error | 0.00906518 | 34 / 48 |
| forecast | T | v2 | 4 | rotation_direction_correct | 0.0833333 | 34 / 48 |

Means use the measurable subset. Their denominators and coverage are reported; a small recovered subset cannot pass a validation gate.

## Near-threshold real calibration classification

| Shape | Version | Decision | TP | TN | FP | FN | Unknown | Precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L | v1 | nominal_point_estimate | 0 | 2 | 0 | 2 | 0 | unavailable |
| L | v1 | declared_interval | 0 | 2 | 0 | 2 | 0 | unavailable |
| L | v2 | nominal_point_estimate | 1 | 2 | 0 | 1 | 0 | 1 |
| L | v2 | declared_interval | 0 | 1 | 0 | 0 | 3 | unavailable |
| T | v1 | nominal_point_estimate | 0 | 2 | 0 | 2 | 0 | unavailable |
| T | v1 | declared_interval | 0 | 2 | 0 | 2 | 0 | unavailable |
| T | v2 | nominal_point_estimate | 1 | 2 | 0 | 1 | 0 | 1 |
| T | v2 | declared_interval | 0 | 1 | 0 | 0 | 3 | unavailable |

No predicted-positive cases means precision is unavailable. An absolute-error development interval is an assumption and has no probability/confidence interpretation on model predictions.

## Raw endpoint and stopping-consistent rankings

| Shape | Version | H | Mode | Complete predictions / anchors | Scored / abstained pairs | Observed pair accuracy | Conservative pair credit | Observed regret | Conservative upper | Hold | Uniform |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L | v1 | 1 | raw_endpoint | 0 / 12 | 0 / 0 | unavailable | unavailable | unavailable | 1 | 0 | 0 |
| L | v1 | 1 | stopping_consistent | 0 / 12 | 0 / 0 | unavailable | unavailable | unavailable | 1 | 0 | 0 |
| L | v1 | 4 | raw_endpoint | 0 / 12 | 0 / 12 | unavailable | 0 | unavailable | 1 | 0.00625834 | 0.0376216 |
| L | v1 | 4 | stopping_consistent | 0 / 12 | 0 / 12 | unavailable | 0 | unavailable | 1 | 0.00625834 | 0.0376216 |
| L | v2 | 1 | raw_endpoint | 0 / 12 | 0 / 0 | unavailable | unavailable | unavailable | 1 | 0 | 0 |
| L | v2 | 1 | stopping_consistent | 0 / 12 | 0 / 0 | unavailable | unavailable | unavailable | 1 | 0 | 0 |
| L | v2 | 4 | raw_endpoint | 0 / 12 | 0 / 12 | unavailable | 0 | unavailable | 1 | 0.00625834 | 0.0376216 |
| L | v2 | 4 | stopping_consistent | 0 / 12 | 0 / 12 | unavailable | 0 | unavailable | 1 | 0.00625834 | 0.0376216 |
| T | v1 | 1 | raw_endpoint | 0 / 4 | 0 / 0 | unavailable | unavailable | unavailable | 1 | 0 | 0 |
| T | v1 | 1 | stopping_consistent | 0 / 4 | 0 / 0 | unavailable | unavailable | unavailable | 1 | 0 | 0 |
| T | v1 | 4 | raw_endpoint | 2 / 4 | 3 / 3 | 1 | 0.5 | 0 | 0.5 | 0.0100699 | 0.0438809 |
| T | v1 | 4 | stopping_consistent | 0 / 4 | 0 / 6 | unavailable | 0 | unavailable | 1 | 0.0100699 | 0.0438809 |
| T | v2 | 1 | raw_endpoint | 0 / 4 | 0 / 0 | unavailable | unavailable | unavailable | 1 | 0 | 0 |
| T | v2 | 1 | stopping_consistent | 0 / 4 | 0 / 0 | unavailable | unavailable | unavailable | 1 | 0 | 0 |
| T | v2 | 4 | raw_endpoint | 2 / 4 | 3 / 3 | 1 | 0.5 | 0 | 0.5 | 0.0100699 | 0.0438809 |
| T | v2 | 4 | stopping_consistent | 0 / 4 | 0 / 6 | unavailable | 0 | unavailable | 1 | 0.0100699 | 0.0438809 |

Observed pair accuracy is unavailable when no eligible pair is scored. Abstained pairs are counted separately; zero credit for those pairs belongs only to the explicitly labeled conservative engineering bound.

Raw endpoint ranks are diagnostic, with physical-absorption flags preserved. Stopping-consistent scoring requires a complete unambiguous predicted stopping prefix; true futures never repair it. Hold/uniform and privileged observed-pose persistence retain their independent physical denominators.

## Posthoc case evidence

| Category | Matching image rows | Examples retained |
| --- | --- | --- |
| real_image_interface_bias_reduced | 502 | 6 |
| near_threshold_declared_interval_ambiguity | 38 | 6 |
| model_shape_or_image_ambiguity | 652 | 6 |
| measurable_model_state_error | 10 | 6 |
| near_threshold_missing_measurement | 165 | 6 |

Cases are the first six matches in fixed inventory order, selected only after decoding for explanation. They are not new scene selection, test examples or evidence of a validation pass.

No additional reset, restore, render, controller step, world-model/policy inference or optimizer update occurred.
