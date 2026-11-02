# Targeted next-collection proposal — not executed

The library contains11,983 contact+obvious TRAIN targets across735 independent
trajectories and11,787 with both valid nonzero impulse and>=5contact substeps.
Do not discard it wholesale or replace its real negatives with duplicated good
windows. Withhold the claim that all acquisition is contact-rich:40.96% of TRAIN
intervals are moving-pusher/no-contact, and late-half contact+obvious is21.60%.
R3B remains paused for the planning chat's decision.

1. Before collecting more, amend the controller proposal so phases follow actual
   contact observations. Use current/past post_solve AO flags and valid impulse
   presence for contact acquisition/loss; keep planned phase/mode separately.
   A route towards an outdated staging point must not silently consume the rest
   of the push phase. Recompute feasible routes from current geometry, cap a
   recovery attempt, log failure/censoring and move to a new independent family.
   Diagnose wall-contact+nearstatic with tangential or alternate exterior contact
   choices. Preserve those failures as negatives rather than deleting them.
2. Keep all physics unchanged:half-life0.1s,10Hz control/100Hz physics,force PD,
   speed/force caps,mass/inertia,walls and penetration guard. No teleporting,
   midtrajectory resets,future-physics lookahead or reduced motion thresholds.
   Report both any-contact and sustained/impulse-supported movement; contact
   alone cannot establish useful motion.
3. Suggested first pilot, pending release:at most48 fresh TRAIN families,8 per
   letter,with a2,304charged-control cap including failures. Per letter,allocate
   four recovery cases,two boundary rotation cases and two low-support exterior
   arc cases; distribute initial yaws/regions and retain independent seeds.
   This is a targeted pilot,not an unbiased replacement for the existing library.
   Prioritize X corner negative rotation (79controls/19families versus positive
   261/29),its known exterior-arc gaps,and T's low actual contact duty44.70%.
   O/B closed inner rings need a geometry-feasibility study; do not call them
   reachable or unreachable from zero coverage,or force the pusher through walls.
4. Freeze proposed acceptance targets before running:contact+obvious>=40% of all
   pilot controls,moving-pusher/no-contact<=25%,observed recontact median<=0.5s,
   P90<=1.0s,and right-censored recovery<=5%. These are proposed engineering
   targets,not retrospectively relaxed gates. Always include all attempted
   controls and failures; report per-letter/family support so pooled means cannot
   hide weak shapes. Compare to existing matched strata descriptively; targeted
   nonrandom changes do not establish a causal controller improvement.
5. Freeze current TRAIN/validation identities and original natural-validation
   sampling. Do not filter or reassign validation using these motion/contact
   outcomes. Keep natural validation separate from targeted diagnostics and keep
   all eight held-out letters sealed. Any training release should use paired
   original6000-versus-new results on the same natural/stress windows and copylast
   baseline; this audit itself makes no model-improvement claim.

No new collection or model work is authorized by this proposal. Send this audit
and proposal to the planning chat under the standing callback authorization,
preserve its actual receipt,then stop in waiting_parent without polling.
