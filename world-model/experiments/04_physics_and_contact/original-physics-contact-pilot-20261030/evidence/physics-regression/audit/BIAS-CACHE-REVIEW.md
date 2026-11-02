# Remaining corner failures: saved-state/source audit

No new physics, model, GPU or optimizer execution. This review reads existing v2/v3 regression arrays (candidate protocols v4/v5), installed Pymunk source and the matching upstream Chipmunk revision. It does not establish a passing final protocol or release collection.

## Main finding

The large v5 failures are actual velocity/cached-impulse instability, not several pixels of hidden position-bias correction. A variable-dt warm-start residual is identified exactly in O/corner_tl. Hidden position bias exists and is absent from state15/current-twist lookahead, but its measured contribution is much smaller in these failures.

### Fixed-dt v4 C/corner_tr

- At zero-based microstep308, post velocity becomes(-943.9950,458.2666)worldpx/s and omega13.5991rad/s while dt stays.001s. The AO contact manifold changes from two arbiters to one. Native wall/object impulses have been very large and mostly cancelling in the preceding steps.
- Step309 integrates that real velocity and rotation before the next solve. AO overlap rises to1.648956worldpx; the final solve leaves near-zero physical velocity. Reporting only that final velocity would miss the cause.
- Across the saved trajectory, the largest consumed hidden-bias translation is.009517worldpx and angular increment.006772degrees. At failure these are small alongside the real velocity-driven movement. Bias alone does not explain the failure.

### Variable-dt v5 O/corner_tl

- Step547 has very large balancing AO/OW impulses. The total previous OW impulse applied to the mass1 object is exactly(20168.184078265935,57960.15898348031).
- Step548 increases dt from.000125 to.00025 (ratio2). There is no current AO post event; all five current OW final impulse vectors are zero. Nevertheless post velocity is exactly(20168.184078265935,57960.15898348031), omega869.05777rad/s.
- This velocity equals `(dt_ratio-1)*previous_OW_impulse_sum/mass` with numerical error0. The next.000125s step then reaches OW1.749353/AO7.571486worldpx overlap. Largest consumed hidden-bias translation over this case is only.0001724worldpx.
- Source inspection supports the mechanism: cached impulses are applied with the dt ratio, while the contact accumulator used by later impulse clamping retains its old value. The observed zero-current-accumulator residual agrees exactly with the difference above. This is stronger than merely observing that failures coincide with dt changes, though this audit is not a new intervention experiment.

The other nine v5 failed cases likewise have large real post velocities (about5,072–35,278worldpx/s, O/corner_tl61,369), whereas maximum consumed-bias translations are.000125–.001663worldpx. Full figures and tail records are in BIAS-v4-v5.json (external historical reference: `BIAS-v4-v5.json`).

## Hidden position bias and query cache

Installed Pymunk6.9.0 names Chipmunk revision `dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970`. Native position integration consumes both physical and bias velocities, then clears the bias fields. The public velocity-at-world-point excludes bias. Position integration precedes collision detection and velocity solving. Thus a budget using only public body velocity/angular velocity omits a real numerical term. [Matching cpBody source](https://raw.githubusercontent.com/viblo/Chipmunk2D/dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970/src/cpBody.c), [cpSpaceStep source](https://raw.githubusercontent.com/viblo/Chipmunk2D/dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970/src/cpSpaceStep.c), [cpArbiter source](https://raw.githubusercontent.com/viblo/Chipmunk2D/dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970/src/cpArbiter.c).

Consumed bias can be inferred retrospectively from saved CoG translation/dt minus pre physical velocity, and angle change/dt minus pre omega. The audit explicitly transforms T's nonzero(0,45)CoG; it does not treat the body origin as CoG. This inference measures bias already consumed during the step, not an actual read of next pending solver bias.

For the selected failing C/v4 and O/v5 files, all recorded component-query distances and surface points (55 and3 respectively) match the saved pre-pose component geometry within4.7e-14worldpx. None of those surface witnesses lies on a hidden union seam. Native space stepping refreshes shape caches after position integration; no body poses are changed between that refresh and the next saved query in the reviewed wrapper. There is no positive evidence here that stale shape-query transforms or an internal seam caused these particular failures. This limited sample cannot establish that every unrecorded query/contact transition is benign. Exact vector/query checks (external historical reference: `CACHE-v4-v5.json`).

## Minimal advice

1. Prefer a fixed microstep for this engine/protocol, avoiding the demonstrated cached-impulse dt-rescaling residual. Do not infer that making dt smaller adaptively is always safer.
2. Keep intended contact compression below the native numerical collision slop, so ordinary permitted AO compression does not continually generate extra positional-bias correction; retain the physical contact solver and zero damping. The proposed fixed.0005s/compression.05 revision is consistent with this direction, but two settings change together, so any improvement must not be attributed separately without an ablation.
3. Preserve full requested/executed command and every guard failure. Keep the1worldpx validity limit and real movement checks. A passing B replay or seven-case probe alone is insufficient; use the full frozen six-letter regression plus later UI gate.
4. State the pending-bias/cache limitation explicitly. Do not manually clear object velocity/bias, write object poses, widen tolerances or silently rebuild cached scenes to conceal failure.

All original arrays and reports are unchanged. New audit scripts only read compressed saved arrays and source. Primary-source facts above and inference from the actual arrays are kept separate.
