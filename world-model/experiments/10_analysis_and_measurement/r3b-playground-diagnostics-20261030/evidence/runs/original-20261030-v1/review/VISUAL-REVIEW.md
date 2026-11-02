# Actual saved-frame visual review

Reviewed all24 main3step B/C/O groups (four fixed anchors in each of two scenes), plus X030/036/039/038 and four B/C/O closeups. Only these viewed cells are annotated. New original-kinematic trajectories are cross-physics previews of a model trained under historicaldynamic0.1s; none is an exact replay of the old screenshots or proof of a specific training-distribution cause.

The first identical A/B output is already wrong in examples: B-inner anchor15 clip001 loses the middle bridge and merges two holes into one ring at+0.1s; C-inner anchor15 clip011 and C-outer anchor7 clip019 round at+0.1s; O-inner anchor7 clip024 and O-outer anchor7 clip029 round at+0.1s even though actual first-step body motion is only.13/.24imagepixels. Thus four-step recursion is not necessary for those errors.

Later divergence is also visible: B-inner anchor7 clip004 becomes one-hole O in B atstep2, while A retains more B structure. C-inner anchor15 clip011 has first-step roundness in both; A recovers straighter geometry at3/4 with true-history correction, while B remains round. C-outer anchor11 clip016 and X-outer anchor11 clip036 acquire clear B-only gray debris atstep3. O-outer anchor3 clip028 loses blue in B at3/4 while A retains it. This supports prediction-history accumulation in those fixed clips, not a universal mechanism or a causal proof about network training.

Gray fragments are not exclusively a recursive error: C-inner anchor7 clip014 has an Astep3 gray dot; C-outer anchor7 clip019 has Astep2 gray debris; X-outer anchor7 clip039 has Astep2/3 debris as well as B2-4. O-outer anchor11 clip026 B3/4 has bluish pusher ghosts, not confidently gray debris, and is deliberately not counted as a manually confirmed gray fragment.

Six old screenshot classes: B hole collapse and blue loss both occur, but in different main clips/steps; the full combined oldB screenshot is not claimed as a same-frame exact reproduction. X fragments/blue-offset examples036B3 or039B2; C roundness/relative-position error011B1, C fragment016B3 or019B2; O roundness024B1/029B1; O abnormal contour plusblue disappearance028B3. All are same-class cross-physics evidence only. Old screenshot actions/history/RNG are missing.

The automatic palette/occlusion labels do not override visible observations: clip001A1/B1 andclip004B2 are visibly topology-damaged despite old unknown labels. Tiny gray specks/hole-count changes next to blue remain qualified; blue overlay can open an apparent O boundary. A missing blue pusher and a visibly destroyed B middle bridge remain separately reportable. No listed manual defect is not a rigidity pass.

Initial generated REPORT/summary/reference JSON are retained in report-before-manual. manual-observations.json contains each viewed cell and source-image hashes; rebuilding the saved-array report is allowed without new prediction, physics or source changes.
