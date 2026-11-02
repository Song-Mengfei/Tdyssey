# CONTACT-PILOT2: one versioned repair and frozen60case verification

Released by planning chat__LOCAL_CHAT__ on2026-10-30 after
critical-review.json independently confirmed CP1's wall failure and integration
defects. Preserve CP1 source/rules/data/reports. R3B,model/checkpoint/GPU/server
work remain paused. Do not merge this pilot into any TRAIN pool.

First repair the identified mechanisms using real saved-state regressions:

* Commit target plus verified route as one transaction,retaining the old-contact
  departure bridge and target identity. Revalidate the next segment against
  current geometry every control and explicitly replan invalid paths. T03state10
  must issue an executable exit/route rather than select then hold.
* Keep raw point/normal/arc observations separate from executable contact targets.
  Verify point+normal+arc together against the current complete union exterior
  and staging/entry geometry. Do not independently clip point then attach an
  incompatible normal. Retain a still-legal old target or reacquire on invalid
  observation. Regress B04states8/15/31/39 with previous valid targets;this fixes
  observed legality mismatches,not a claimed uniquely isolated dynamic cause.
* Before every follow/entry command,check conservative wall drive/braking risk
  from current clearance,boundary-point normal velocity including rotation,the
  next0.1s command,known damping,and actual bounded pusher action. React before
  three static wall contacts,cover X03states0..5 before5is already too close and
  rotating corners. Log inputs,trigger,original and final action. No future
  physics rollout or mathematical safety claim;never change dynamics,caps,state
  or the2pxguard to meet a target.
* Keep initial_requested_arc immutable and separate from tracked target. Actual
  requested hits require real postsolve arc evidence,not24px/45degree tolerance
  or min(arc_ids). Diagnose all six old weak targets using recorded first-contact
  evidence and current geometry;log incidental hits,retarget/fallback causes and
  unconfirmed feasibility. Do not claim unknown equals unreachable. Closed O/B
  inner holes are outside this stage.

Tests use those real frozen states through production paths,including bridge
commit,B04invalid observation,early linear/rotational wall risk,requested versus
incidental contact,fixed withdrawal and geometry/bounds. Fix failing static
regressions before physics. Freeze rules,source hashes and regression evidence,
then freeze the physical case plan. No post-physics tuning/retry loop.

Physical queue:all48CP1initial states/targets/tasks repeat once,plus12 genuinely
new TRAIN families(sixletters x recovery/nearwall challenge + weakexterior).
The48repeat scenes are not new independent support;original24A retain R3A/CP1/CP2
comparison,original24B retain CP1/CP2. Check new12 exact/near against R3A/CP1/val
and withinnew,using geometry/coverage only and sealed heldouts. Order:X03first,
thenT03,B04,sixCP1weakcases,deduplicated,then remaining fixed cases. Everycase
max48controls,oneattempt;total2,880charged including failed/partial/unknown.
Any>2pxguard,unknown advance or engineering failure saves partial/costs and
immediately stops the entire remaining queue. No replacement or same-version retry.
Other complete but poor-quality dynamics remain and the frozen queue continues.

Unchanged physical protocol:0.1svelocityhalf-life,10Hz/100Hz,PD,force/speedcaps,
mass/inertia,walls,renderer and2pxguard. No teleport,midtrajectory resets,new
model interfaces or futurephysics. Preserve withdrawal/recontact tasks. Recovery
success requires a verified complete no-AO control during withdrawal followed by
real contact after fixed recontact start;never separated is not a zero-second
success. Notreached and right-censored remain separate.

Evaluation retains CONTACT_AUDIT six contact-by-motion classes and impulse
presence. Give allcharged,completeinterval and planned48slot worst-case bounds;
late,wallstall,plannedpush_noAO,longesthold and unconfirmedroute. Frozen targets:
contact+obvious>=40%,empty<=25%,observed recoverymedian<=.5/P90<=1s,unrecovered<=5%,
newguardfailures0. Give eachcase/letter/version,including B04regression and normal
T040pushing. Report sixold+sixnew weak arcs individually:actual requested hit,
sustained impulse and motion under that contact;zero iszero. New12family results
are separate and targeted/small,not natural-population guarantees.

Stage persistent increment<=16MiB;shared<=3GiB with256MiBatomic reserve. Estimate
necessary bytes before physics;avoid duplicate old media and largeRGB caches or
deletions. Use fixed X03/T03/B04version comparisons,failed partial endstates,
one-page findings,regression/source evidence,all cost/storage data. Reuse old
artifacts rather than copying the library. Actual callback to parent with saved
receipt and memo,then waiting_parent STOP,no polling,no expansion,no TRAIN merge
or R3B resume. Future model work and collector improvements remain separate goals.
