# CONTACT-PILOT1: cold diagnosis of the guard failure and remaining stalls

This reads saved arrays, telemetry and current-pose geometry only. It performs
**0 new physics/model calls**, changes no controller/rules, and proposes no retry.
Exact file hashes and array indices are in diagnosis-evidence.json (external historical reference: `diagnosis-evidence.json`).
The frozen source SHA remains8353a21c9d720af15097908ad4529449ee8e4f4b437799a43fb8ad0ed3b37aa6.

## X-03: object-wall guard failure, not agent-object penetration

CP1-A-X-03 (external historical reference: `trajectories/CP1-A-X-03.json`) ends after5 complete controls plus8 known
substeps of control5:6charged,58known substeps,zero unknown. Zero-based global
substep57 (control5/local7), interval[.57,.58)s, records penetrations
**[2.6419284,0.1977386,0] worldpx**. The source `native_rgb_env.py` function
`penetrations()` lines176–185 returns **(object-wall,agent-object,agent-wall)**.
This order differs from contact_flags(AO,OW,AW). Column0, therefore, identifies
object penetration of the top wall. Independent saved-geometry top gap is
-2.6419285worldpx, matching the float32 recorded value. The last OW postsolve
points have outward normal(0,-1), including world point(319.8824,4.3581), and
saved distance-2.6419284. Repeated event impulses are not summed.

The action for the failed control is(378.4242,29.4394),62.432worldpx from the
current pusher. Telemetry says **contact_follow**, planned push, established
contact=true,target_revision=0,no replan,no wall-stall trigger. All six commands
use arc49/localnormal approximately(1,0); localpoints move only from
(10,1.1429) through(10,-.7200),(10,.0038) to(10,.3595). This was not a recorded
large point/normal jump or wall-escape retarget. Every issued geometry check
passed, which proves current command entry checks, not future object-wall safety.

At control5 start, the top gap is6.0340worldpx and object velocity is
(-10.2113,-150.8073)worldpx/s. An elementary gap/normal-speed estimate is about
.040s to the top wall. The first OW postsolve event appears at global53,
interval.53,.54)s. The action remains unchanged through all eight partial
substeps; the guard stops the run at global57 before a complete control can be
observed. Object angular velocity rises to2.143rad/s at substep53 and2.740rad/s
at56; the final collision leaves almost zero normal velocity while geometry
already exceeds the guard. The control's maximum measured PD force norm is
3221.46 (below10000),maximum capped-entry pusher speed164.12 (below200),and last
force(-104.093,-2985.390). There is no evidence of an altered cap or physics.

The frozen controller reacts to three completed AO+OW+nearstatic controls. That
condition had never accumulated here: the object was moving towards its first
wall impact within the failing control. Current contact-follow uses48worldpx
normal advance plus bounded current-velocity lead; it does not perform a
near-wall braking/drive-room check on every established-follow command. This is
an identified coverage gap in its safety mechanism, consistent with this trace.
The old matched trajectory first records OW at global64 and has maximum wall
penetration.152545; at old control5 its top gap is19.3606 and vertical velocity
-119.1045,versus6.0340/-150.8073 in the new run. Several controller mechanisms
changed together,so these paired observations do not isolate feedforward,
contact-point tracking or another individual change as the sole cause.

## T and B wall stalls: letter totals versus the named cases

Wall stall below means any AO+any OW+nearstatic within the same0.1s control.

| Matched case | Old → new wall-stall controls | Old → new contact+obvious |
| --- | --- | --- |
| T-03 |32 →41 |1 →2 |
| T-04 |0 →5 |16 →13 |
| T letter,all4cases |32 →46 |37 →69 |
| B-01 |0 →2 |15 →11 |
| B-02 |0 →0 |8 →12 |
| B-03 |29 →34 |5 →5 |
| B-04 |0 →0 |33 →14 |
| B letter,all4cases |29 →36 |61 →42 |

**T-03 selects a new target but loses the usable departure bridge.** At
control10 it replaces the tracked cap-side point with arc23/localpoint(-52,30),
normal(0,1). The new stage exists,but pusher-to-new-stage free_route isNone,
monotone departure to it isfalse,and new-center distance31.470worldpx exceeds
the contact-shell entry allowance24. The issued command becomes a safe hold.
At exactly the same saved state,a departure to the *old* stage is confirmed,
and old-stage→new-stage is a confirmed free segment. Evidence records those
coordinates and predicates.

The code's candidate search uses route(),whose departure bridge depends on
self.contact. It can confirm a candidate using the old contact's bridge,but
then assigns self.contact to the new point before command routing. No accepted
bridge/path is retained; route() now tries the new point as its own departure
bridge. The recorded control10selected→hold and these static predicates expose
that integration failure without replaying dynamics. Later searches repeatedly
report no_confirmed_alternative. New T-03 has34unconfirmed_hold,7contact_follow,
4withdraw and3initial_acquisition controls. Contact remains persistent but
mostly static. During its4 planned withdrawal controls it never loses AO
contact; the reported recontact delay0 at26 is therefore not evidence of a
successful separation and reacquisition.

**B-03 never finds a finite alternative and continues following the blocked
contact.** It stays at target_revision0. Searches at controls12,16,20,31,35,39,43,
47 say no_confirmed_alternative;37controls are contact_follow. At saved control12
both left and top gaps are slightly negative(-.1743,-.1278). Of59alternative
exterior arcs,the translation-role filter accepts0. Generic fallback accepts23
geometric candidates,but none has the required16worldpx of drive room in its
proposed direction. These are limitations of the frozen finite point/normal
search,not proof that no physical tangential or rotational escape exists.
Because unsuccessful search retains established following,the same blocked
contact continues to receive drive commands. Fixed withdrawal is attempted and
achieves3no-AO controls; return is faster(.26s versus.62s),but this returns to a
still-blocked situation rather than removing the stall.

## B's contact+obvious regression is concentrated in B-04

The four paired deltas are -4,+4,0,-19controls; B-04 accounts for the net
61→42drop. B-04 actual any-contact controls fall38→30 and100Hz contact duty
74.17%→47.29%. It is **not a new nearstatic failure**: nearstatic falls5→2,
while small-motion controls rise10→32; contact+obvious falls33→14. Median
absolute rotation falls4.881→1.920degrees and median translation2.029→.940px.

[New B-04 (external historical reference: `trajectories/CP1-A-B-04.json`) has16contact_follow,14recovery_entry,
8unconfirmed_hold,4recovery_route,4withdraw and2initial_acquisition commands.
At controls8 and15,recorded contact tracking uses localpoint(51.6667,-8.5714)
with oblique normals(.8049,.5934) and(.8229,.5681); `_stage` at each exact saved
pose returnsNone. The preceding target normal at those same poses does admit
an exterior stage. Controls31 and39 repeat the rejection/hold pattern. After
two unconfirmed commands,the policy selects arc42/point(51.6667,-20),normal(1,0)
at controls10,17,33,41,then repeatedly reacquires and loses the geometry-supported
following state. This shows an incompatibility between the observed point/normal
tracking representation and the conservative full-union staging check in these
states. Point clipping and per-arbiter normal conversion are possible contributors,
but the compact record lacks unique arbiter/component identity; this diagnosis
does not identify one as the isolated cause or project a repaired outcome.

## Implication and limits

The pilot demonstrates useful improvements in some matched recovery cases,but
its wall-escape execution remains incomplete,one first wall impact violates the
guard,and a strong rotation case regresses through repeated hold/reacquisition.
Persistent contact or zero measured recontact delay cannot replace useful motion
and successful withdrawal. These traces support the existing decision **not to
expand collection or resume model work**. Any future repair needs a separately
released version and fresh frozen verification; this stage remains unchanged.
