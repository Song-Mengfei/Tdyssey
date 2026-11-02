# Frozen world-model diagnostic tables

Saved run status: `complete_with_aborted_branches`.

The fixed export contains 768 anchor/candidate/noise/horizon rows; 756 have saved forecast steps and 24 have calibrated scores. Missing rows remain in the CSV.

Noise repeats are nested inside physical branches; summaries weight anchors equally. Contact strata mean any real solver contact by H4. Image motion after real success absorption is excluded by the saved summary. Missing values are unavailable, never zero.

## Calibration

| Shape | Valid / total | Position P90 px | Pusher P90 px | Angle P90 deg | Coverage MAE | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| L | 12 / 12 | 1.03878 | 0.820683 | 0.354409 | 0.00835995 | pass |
| T | 12 / 12 | 1.06024 | 1.17483 | 0.667852 | 0.00569933 | pass |

Calibration of continuous errors does not establish precise strict-success classification at coverage >0.95. Inspect calibration.csv for crossings and near-threshold errors.

## H4 engineering gates

| Shape | Gate | Status | Reason |
| --- | --- | --- | --- |
| L | noncontact_drift | fail | invalid predictions retained in denominator |
| L | contact_motion | fail | invalid predictions retained in denominator |
| L | rotation | fail | invalid predictions retained in denominator |
| L | coverage | coverage insufficient | predeclared physical sample minima not met |
| L | ranking | coverage insufficient | predeclared physical sample minima not met |
| L | contact_probability | unavailable | GPC has no direct contact-probability head; geometry proximity is a proxy |
| T | noncontact_drift | fail | invalid predictions retained in denominator |
| T | contact_motion | fail | invalid predictions retained in denominator |
| T | rotation | coverage insufficient | predeclared physical sample minima not met |
| T | coverage | fail | invalid predictions retained in denominator |
| T | ranking | fail |  |
| T | contact_probability | unavailable | GPC has no direct contact-probability head; geometry proximity is a proxy |

## Coverage error

| Shape | Horizon | Stratum | Anchor mean MAE | Valid / expected branches |
| --- | --- | --- | --- | --- |
| L | H1 | observed | unavailable | 0 / 48 |
| L | H1 | persistence_error | 0 | 47 / 48 |
| L | H1 | near_goal | unavailable | 0 / 12 |
| L | H4 | observed | unavailable | 0 / 48 |
| L | H4 | persistence_error | 0.031618 | 47 / 48 |
| L | H4 | near_goal | unavailable | 0 / 9 |
| T | H1 | observed | unavailable | 0 / 16 |
| T | H1 | persistence_error | 0 | 16 / 16 |
| T | H1 | near_goal | unavailable | 0 / 4 |
| T | H4 | observed | unavailable | 0 / 16 |
| T | H4 | persistence_error | 0.0388459 | 16 / 16 |
| T | H4 | near_goal | unavailable | 0 / 3 |

## H4 contact and rotation

| Shape | Stratum | Metric | Anchor mean | Valid / expected branches |
| --- | --- | --- | --- | --- |
| L | contact | position_error | unavailable | 0 / 9 |
| L | contact | angle_error | unavailable | 0 / 9 |
| L | contact | position_drift | unavailable | 0 / 9 |
| L | contact | persistence_position_error | 23.6431 | 9 / 9 |
| L | contact | persistence_angle_error | 4.74504 | 9 / 9 |
| L | contact | rotation_direction | unavailable | 0 / 9 |
| L | noncontact | position_error | 0.596004 | 1 / 38 |
| L | noncontact | angle_error | 0.828049 | 1 / 38 |
| L | noncontact | position_drift | 0.596004 | 1 / 38 |
| L | noncontact | persistence_position_error | 0 | 38 / 38 |
| L | noncontact | persistence_angle_error | 0 | 38 / 38 |
| L | noncontact | rotation_direction | 0 | 1 / 38 |
| L | rotation_gt_5deg | position_error | unavailable | 0 / 3 |
| L | rotation_gt_5deg | angle_error | unavailable | 0 / 3 |
| L | rotation_gt_5deg | position_drift | unavailable | 0 / 3 |
| L | rotation_gt_5deg | persistence_position_error | 18.3615 | 3 / 3 |
| L | rotation_gt_5deg | persistence_angle_error | 14.2351 | 3 / 3 |
| L | rotation_gt_5deg | rotation_direction | unavailable | 0 / 3 |
| T | contact | position_error | 2.77674 | 2 / 3 |
| T | contact | angle_error | 0.628145 | 2 / 3 |
| T | contact | position_drift | 25.2881 | 2 / 3 |
| T | contact | persistence_position_error | 24.8531 | 3 / 3 |
| T | contact | persistence_angle_error | 3.81829 | 3 / 3 |
| T | contact | rotation_direction | 0.5 | 2 / 3 |
| T | noncontact | position_error | 2.68267 | 9 / 13 |
| T | noncontact | angle_error | 1.13718 | 9 / 13 |
| T | noncontact | position_drift | 2.68267 | 9 / 13 |
| T | noncontact | persistence_position_error | 0 | 13 / 13 |
| T | noncontact | persistence_angle_error | 0 | 13 / 13 |
| T | noncontact | rotation_direction | 0 | 9 / 13 |
| T | rotation_gt_5deg | position_error | 4.0725 | 1 / 1 |
| T | rotation_gt_5deg | angle_error | 0.813665 | 1 / 1 |
| T | rotation_gt_5deg | position_drift | 24.8404 | 1 / 1 |
| T | rotation_gt_5deg | persistence_position_error | 21.9914 | 1 / 1 |
| T | rotation_gt_5deg | persistence_angle_error | 11.4549 | 1 / 1 |
| T | rotation_gt_5deg | rotation_direction | 1 | 1 / 1 |

## H4 ranking

| Shape | Eligible pairs | Anchor pair accuracy | Conservative regret | Hold regret | Uniform regret | Abstentions |
| --- | --- | --- | --- | --- | --- | --- |
| L | 12 | 0 | 1 | 0.00625834 | 0.0376216 | 12 |
| T | 6 | 0 | 1 | 0.0100699 | 0.0438809 | 4 |

## Windows monitor phases

| Phase | Worker PID | Status | Seconds | Min RAM GiB | Min commit GiB | Min GPU GiB | Peak working set GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| telemetry | 22072 | stopped | 20.469 | 3.90782 | 11.5304 | 6.56348 | 0.436192 |
| telemetry/recovery-1 | 3516 | exited | 395.391 | 3.03026 | 10.2111 | 6.28125 | 1.20988 |

Monitor minima/peaks are sampled observations. Initial and recovery phases remain separate; their elapsed sum is not a new physics or inference budget.

## Costs

| Resource | Reserved | Completed | Aborted | Limit |
| --- | --- | --- | --- | --- |
| branches | 64 | 64 | 0 | 64 |
| calibration | 24 | 24 | 0 | 24 |
| controls | 441 | 441 | 0 | 448 |
| custom_images | 0 | 0 | 0 | 192 |
| euler_steps | 2268 | 2268 | 0 | 2304 |
| forecast_images | 756 | 756 | 0 | 768 |
| optimizer_updates | 0 | 0 | 0 | 0 |
| output_bytes | 0 | 0 | 0 | 268435456 |
| reset | 88 | 87 | 1 | 88 |
| restore | 87 | 87 | 0 | 88 |
| settle | 88 | 87 | 1 | 88 |
| substeps | 4410 | 4410 | 0 | 4480 |
| telemetry_bytes | 0 | 0 | 0 | 67108864 |
| worker_report_wall_seconds | unavailable | 950.704 | unavailable | unavailable |
| ledger_elapsed_seconds | unavailable | 950.858 | unavailable | 1800 |
| guarded_phase_seconds_sum | unavailable | 415.86 | unavailable | unavailable |
| policy_optimizer_updates | unavailable | 0 | unavailable | unavailable |
| world_model_optimizer_updates | unavailable | 0 | unavailable | unavailable |
| peak_cuda_allocated_bytes | unavailable | 110273536 | unavailable | unavailable |
| peak_cuda_reserved_bytes | unavailable | 136314880 | unavailable | unavailable |

Exports read saved reports only. No rendering, decoder invocation, simulation, forecasting, optimizer update or scene selection occurs.
