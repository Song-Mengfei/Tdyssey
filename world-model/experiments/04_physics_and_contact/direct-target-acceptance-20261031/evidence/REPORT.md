# Direct-target acceptance: saved ground truth

Run: `DTA-20261031-7b4e4819a73b41a78dba24947fd6579b`. Planned 120 attempts / 96 independent scene families; attempted 120, raw saved 120, complete 46, invalid terminated 74.

Actual 3446 controls / 3372 legal controls / 33992 native substeps; 339.920 simulated seconds (not CPU seconds). Model calls=0, optimizer updates=0; this report executes no physics.

These are true observations, not model predictions. No collection expansion or training is released by this report.

## Motion at identical H16-capable anchors

Each cell is P10 / P50 / P90. Centroid displacement uses actual geometry area centroid, not body origin. Rotation is raw unwrapped absolute net angle; symmetry-equivalent and cumulative angles are in summary.json and CSV. Overlapping windows are not independent.

|Letter|H|Independent families|Legal paired windows|Net translation imagepx|Net rotation deg|Obvious translation / rotation|
|---|---:|---:|---:|---|---|---|
|T|1|9|460|0.000 / 0.000 / 0.697|0.000 / 0.000 / 1.308|19 / 16|
|T|4|9|460|0.000 / 0.000 / 2.211|0.000 / 0.000 / 4.886|53 / 46|
|T|8|9|460|0.000 / 0.000 / 3.334|0.000 / 0.000 / 6.366|38 / 39|
|T|16|9|460|0.000 / 0.000 / 3.538|0.000 / 0.000 / 8.238|25 / 14|
|L|1|7|277|0.000 / 0.000 / 0.148|0.000 / 0.000 / 1.644|1 / 13|
|L|4|7|277|0.000 / 0.000 / 0.475|0.000 / 0.000 / 4.947|2 / 28|
|L|8|7|277|0.000 / 0.000 / 0.661|0.000 / 0.000 / 6.289|0 / 25|
|L|16|7|277|0.000 / 0.000 / 0.664|0.000 / 0.000 / 6.327|0 / 16|
|O|1|7|322|0.000 / 0.000 / 0.356|0.000 / 0.000 / 0.846|4 / 12|
|O|4|7|322|0.000 / 0.000 / 1.405|0.000 / 0.000 / 3.202|21 / 29|
|O|8|7|322|0.000 / 0.000 / 2.680|0.000 / 0.000 / 5.596|18 / 25|
|O|16|7|322|0.000 / 0.000 / 4.130|0.000 / 0.000 / 7.162|9 / 19|
|C|1|7|368|0.000 / 0.000 / 0.776|0.000 / 0.000 / 1.293|27 / 14|
|C|4|7|368|0.000 / 0.000 / 3.029|0.000 / 0.000 / 4.722|55 / 36|
|C|8|7|368|0.000 / 0.000 / 5.787|0.000 / 0.000 / 8.985|48 / 33|
|C|16|7|368|0.000 / 0.000 / 6.926|0.000 / 0.000 / 14.029|35 / 20|
|X|1|7|322|0.000 / 0.000 / 0.292|0.000 / 0.000 / 2.053|1 / 22|
|X|4|7|322|0.000 / 0.000 / 1.107|0.000 / 0.000 / 7.689|17 / 38|
|X|8|7|322|0.000 / 0.000 / 1.706|0.000 / 0.000 / 11.400|11 / 34|
|X|16|7|322|0.000 / 0.000 / 1.714|0.000 / 0.000 / 11.739|4 / 22|
|B|1|8|368|0.000 / 0.000 / 0.462|0.000 / 0.000 / 1.702|5 / 19|
|B|4|8|368|0.000 / 0.000 / 1.776|0.000 / 0.000 / 6.599|30 / 47|
|B|8|8|368|0.000 / 0.000 / 3.303|0.000 / 0.000 / 11.599|23 / 41|
|B|16|8|368|0.000 / 0.000 / 4.738|0.000 / 0.000 / 13.524|10 / 28|

Own-horizon denominators (including valid short prefixes) and per-letter/per-scenario distributions are in summary.json; paired-common-H16.csv uses the same anchor set across all four horizons.

## Contact and complete denominators

|Letter|Attempted / planned|Complete|AO positive time / all executed|AO positive fraction|OW pair seconds|Contact-and-push control seconds|
|---|---|---:|---|---:|---:|---:|
|T|20 / 20|10|30.790 / 70.310|0.438|0.780|10.400|
|L|20 / 20|6|10.660 / 48.050|0.222|1.090|6.300|
|O|20 / 20|7|20.330 / 52.480|0.387|0.080|8.000|
|C|20 / 20|8|29.900 / 57.450|0.520|0.130|8.700|
|X|20 / 20|7|14.580 / 52.190|0.279|0.030|5.800|
|B|20 / 20|8|26.060 / 59.440|0.438|1.270|9.000|

AO contact ratio counts actual post-solve positive impulse microstep time / all executed simulation time. Attempts with no executed time stay in attempt counts, not an invented time denominator. Missing contact telemetry makes the full ratio unavailable, with exact missing seconds disclosed; observed contact is never passed off as complete. Contact-and-push is a separate control diagnostic: contact AND (>=0.1 imagepx translation OR >=0.5deg rotation). It does not establish sufficient motion. Static, blocked/small contact, empty motion, searching, invalid and wall-only categories retain all actual controls.

Display thresholds, translation / rotation separately: H1 1px / 3deg; H4 2px / 5deg; H8 4px / 10deg; H16 8px / 20deg. These select media, not data admission. Missing examples remain unavailable.

## Coverage and missing evidence

- T: unobserved exterior arcs [12, 13, 14, 15]; missing actual wall sides ['left', 'top', 'bottom']; missing actual corners ['left-top', 'right-top', 'left-bottom', 'right-bottom']; actual continuous recontact attempts 0.
- L: unobserved exterior arcs [2, 3, 4, 5, 12]; missing actual wall sides ['left', 'right']; missing actual corners ['left-top', 'right-top', 'left-bottom', 'right-bottom']; actual continuous recontact attempts 0.
- O: unobserved exterior arcs [3, 8, 12]; missing actual wall sides ['left', 'top', 'bottom']; missing actual corners ['left-top', 'right-top', 'left-bottom', 'right-bottom']; actual continuous recontact attempts 0.
- C: unobserved exterior arcs [3, 4, 5, 9, 10, 13, 14, 15]; missing actual wall sides ['left', 'top']; missing actual corners ['left-top', 'right-top', 'left-bottom']; actual continuous recontact attempts 0.
- X: unobserved exterior arcs [0, 2, 4, 5, 10, 13, 14]; missing actual wall sides ['left', 'top', 'bottom']; missing actual corners ['left-top', 'right-top', 'left-bottom', 'right-bottom']; actual continuous recontact attempts 1.
- B: unobserved exterior arcs [4, 6, 14, 15]; missing actual wall sides ['left', 'right', 'top']; missing actual corners ['left-top', 'right-top', 'left-bottom', 'right-bottom']; actual continuous recontact attempts 0.

All16 exterior bins stay visible, with unknown reachability retained. O/B closed holes are not required externally accessible targets. A boundary-intended scene is not labelled actual wall contact without telemetry. Corner coverage requires both actual wall sides during one microstep.

Unavailable qualifying media:
- T recontact: No complete legal qualifying saved window; not replaced or relabelled.
- L recontact: No complete legal qualifying saved window; not replaced or relabelled.
- O recontact: No complete legal qualifying saved window; not replaced or relabelled.
- C recontact: No complete legal qualifying saved window; not replaced or relabelled.
- X boundary: No complete legal qualifying saved window; not replaced or relabelled.
- B recontact: No complete legal qualifying saved window; not replaced or relabelled.

## Media, identity and limits

Open report/index.html for full-attempt real10Hz videos, actual contact heatmaps and same-anchor t/t+1/t+4/t+8/t+16 truth boards. Every saved frame appears once; an invalid final partial is explicitly timestamped and held for one display frame. Display overlays never enter raw RGB.

Full attempts: report/attempt-index.csv. Raw identity and request=passed=frozen-plan checks: report/raw-manifest.csv. Complete controls/contacts/windows: report/*.csv. All outcomes, branch equality and actual outcome separation, per-scene statistics, action distances/changes and actual pusher motion are separate in report/summary.json.

Original targets use [0,512]. Existing model uses [0,511] and a/511 normalization; no model interface was changed, no512->511 target clipping applied. Later compatibility work requires approval.

Sampling uses privileged geometry/state only for planning and these true-data audits. No learned-model result can be inferred. Subsequent formal collection and training require user review.
