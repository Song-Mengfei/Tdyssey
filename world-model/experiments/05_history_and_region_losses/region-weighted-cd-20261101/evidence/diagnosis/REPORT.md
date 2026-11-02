# Same-case O/B diagnosis

768 independent true-history predicted targets completed: original16cases ×16targets ×3 checkpoints. Original AR16 reused. Zero optimizer updates and physical steps.

Interface contract passed: no demonstrated XY/sign/time/normalization/input-label corruption. A wrong model rotation is not evidence of an interface fault.

RH direction compares each previous REAL frame with its independent prediction. AR local and cumulative directions are separate. Old formal angular unknowns remain unchanged.

Historical AR epsilon tensors were not archived. New actual B4 epsilon and slots are hashed and checked against independent same-seed/B4 reconstruction; original sample order/action hashes match old A/B runtime records.

Initial-target output parity with original AR (independent evidence; numerical discrepancy is not silently called an action fault):

- baseline: 16/16 bitwise equal; max absolute difference 0.0; nonidentical IDs []
- A: 16/16 bitwise equal; max absolute difference 0.0; nonidentical IDs []
- B: 16/16 bitwise equal; max absolute difference 0.0; nonidentical IDs []

Independent visible support (unknown never passes):

- baseline_AR16: {"O_display_known_correct_clockwise": false, "O_display_classification": "opposite", "O_display_mode": "autoregressive", "B_display_H16_two_visible_holes": true, "B_other7_no_hole_mismatch_or_unknown": true, "B_visible_hole_support_passed": true, "rigid_shape_or_exact_physical_angle_proven": false}
- A_AR16: {"O_display_known_correct_clockwise": false, "O_display_classification": "no_resolved_rotation", "O_display_mode": "autoregressive", "B_display_H16_two_visible_holes": false, "B_other7_no_hole_mismatch_or_unknown": true, "B_visible_hole_support_passed": false, "rigid_shape_or_exact_physical_angle_proven": false}
- B_AR16: {"O_display_known_correct_clockwise": false, "O_display_classification": "opposite", "O_display_mode": "autoregressive", "B_display_H16_two_visible_holes": false, "B_other7_no_hole_mismatch_or_unknown": true, "B_visible_hole_support_passed": false, "rigid_shape_or_exact_physical_angle_proven": false}
- baseline_RH1: {"O_display_known_correct_clockwise": false, "O_display_classification": "unknown", "O_display_mode": "real_history", "B_display_H16_two_visible_holes": true, "B_other7_no_hole_mismatch_or_unknown": true, "B_visible_hole_support_passed": true, "rigid_shape_or_exact_physical_angle_proven": false}
- A_RH1: {"O_display_known_correct_clockwise": false, "O_display_classification": "unknown", "O_display_mode": "real_history", "B_display_H16_two_visible_holes": true, "B_other7_no_hole_mismatch_or_unknown": true, "B_visible_hole_support_passed": true, "rigid_shape_or_exact_physical_angle_proven": false}
- B_RH1: {"O_display_known_correct_clockwise": false, "O_display_classification": "unknown", "O_display_mode": "real_history", "B_display_H16_two_visible_holes": true, "B_other7_no_hole_mismatch_or_unknown": true, "B_visible_hole_support_passed": true, "rigid_shape_or_exact_physical_angle_proven": false}

All case counts/directions, local/cumulative tables, holes and uncertainty: __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\direction-and-holes.json

Saved comparisons (RH is not a physical rollout):

- AC-B-06-08:AC-B-06-08:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-06-08-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-06-08-a3-10fps.mp4
- AC-B-05-00:AC-B-05-00:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-05-00-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-05-00-a3-10fps.mp4
- AC-B-07-24:AC-B-07-24:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-07-24-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-07-24-a3-10fps.mp4
- AC-B-02-08:AC-B-02-08:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-02-08-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-02-08-a3-10fps.mp4
- AC-B-01-08:AC-B-01-08:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-01-08-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-01-08-a3-10fps.mp4
- AC-B-04-00:AC-B-04-00:a19:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-04-00-a19-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-04-00-a19-10fps.mp4
- AC-B-03-16:AC-B-03-16:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-03-16-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-03-16-a3-10fps.mp4
- AC-B-00-08:AC-B-00-08:a19:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-00-08-a19-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-B-00-08-a19-10fps.mp4
- AC-O-06-08:AC-O-06-08:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-06-08-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-06-08-a3-10fps.mp4
- AC-O-05-24:AC-O-05-24:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-05-24-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-05-24-a3-10fps.mp4
- AC-O-07-24:AC-O-07-24:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-07-24-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-07-24-a3-10fps.mp4
- AC-O-02-00:AC-O-02-00:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-02-00-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-02-00-a3-10fps.mp4
- AC-O-01-08:AC-O-01-08:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-01-08-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-01-08-a3-10fps.mp4
- AC-O-04-16:AC-O-04-16:a19:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-04-16-a19-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-04-16-a19-10fps.mp4
- AC-O-03-08:AC-O-03-08:a3:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-03-08-a3-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-03-08-a3-10fps.mp4
- AC-O-00-16:AC-O-00-16:a19:H16 | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-00-16-a19-H1-H4-H8-H16.png | __LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\region-weighted-cd-20261101\diagnosis\media\AC-O-00-16-a19-10fps.mp4
