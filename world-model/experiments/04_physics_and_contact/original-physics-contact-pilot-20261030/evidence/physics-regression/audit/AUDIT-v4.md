# Independent final v6 physical regression audit

Passed: **True**. Read-only saved-array audit; new model/GPU/physics/optimizer work0.
120 raw files:96 candidate and24 original ordinary controls. Cost:{'attempted_controls': 2880, 'completed_controls': 2880, 'microsteps': 466560}. Every control is actual.1s, candidate microstep.0005s.

Maximum candidate OW/AO/AW overlap worldpx:[0.1949161316387915, 0.051680431869401744, 0.0]. Intervention time:26.205%.

| Letter | Cases | Max OW/AO/AW worldpx | Intervention time | Max H1 imagepx / degrees | last6 controls all no AO |
| --- | --- | --- | --- | --- | --- |
|T|16|0.060382/0.051680/0.000000|26.01%|8.107/25.525|16/16|
|L|16|0.153993/0.050292/0.000000|24.71%|3.244/14.016|15/16|
|O|16|0.107920/0.050152/0.000000|27.72%|3.962/11.295|16/16|
|C|16|0.194916/0.050652/0.000000|24.49%|4.819/11.402|16/16|
|X|16|0.132395/0.050675/0.000000|27.66%|6.399/22.151|15/16|
|B|16|0.104199/0.050736/0.000000|26.64%|4.542/16.877|16/16|

## Actual separation and original comparison

{"last6_verified_no_AO_cases": 94, "cases_with_AO_during_last6": [{"letter": "L", "scenario": "long_fast", "last6_hold_controls": 6, "all6_no_AO": false, "AO_seconds": 0.5995000000000001, "OW_seconds": 0.0, "actual_path_world": 13.870234882348806, "control_chord_path_world": 13.866378367926224, "max_control_speed": 52.18186196272281, "max_control_abs_omega": 0.7876320672592502}, {"letter": "X", "scenario": "long_fast", "last6_hold_controls": 6, "all6_no_AO": false, "AO_seconds": 0.6000000000000002, "OW_seconds": 0.0, "actual_path_world": 56.733274561044055, "control_chord_path_world": 56.639078851443756, "max_control_speed": 107.92240107668961, "max_control_abs_omega": 2.2159499961466693}], "max_actual_path_world_px_among_verified_released": 4.867094719429588e-05, "max_end_speed_among_verified_released": 0.0, "max_end_abs_omega_among_verified_released": 0.0, "statement": "No-AO is verified from actual post-solve contacts, not the planned withdraw label. OW contact correction can leave tiny positional residuals despite zero physical end velocity."}

{"pairs": 24, "max_over_all_control_centroid_distance_image_px": 2.0960212311955058, "final_centroid_distance_image_px": {"N": 24, "P10": 0.0050667737961317995, "P50": 0.010700061602971545, "P90": 0.1108955570455799, "P95": 0.1362586264645769, "max": 0.23637313233288926}, "final_angle_difference_deg": {"N": 24, "P10": 0.0, "P50": 1.547738094095372e-12, "P90": 0.5553956220395487, "P95": 0.6948133310442479, "max": 0.7587947814483762}, "statement": "Identical initial RGB/full public state and requested actions, but new contact/integration solver differs. Numerical equality to original is not claimed."}

Full source bindings, true-geometry signed gaps, contact seconds, all exceptions and three preserved failed versions are in AUDIT-v4.json.

## Scope limits

- This is a bounded regression of six seen letters, not a global guarantee for every arbitrary kinematic command.
- Pinned upstream is unchanged; fixed2000Hz/solver50/contact compression.05 and command constraints intentionally differ from original100Hz/solver10 execution.
- Snapshot excludes native pending position bias/contact caches; midcontact replay is not established.
- Runtime and checks here concern physics only. Actual playground integration gate remains the owner responsibility before collection.
