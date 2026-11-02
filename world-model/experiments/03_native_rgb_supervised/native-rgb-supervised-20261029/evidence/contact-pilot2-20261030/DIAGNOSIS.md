# CONTACT-PILOT2: actual command-path diagnosis

This is a cold read of the60 completed trajectories and gzip decision telemetry.
It runs0new physics/model calls and changes no controller or rules. Compact
evidence, source hashes and exact case/telemetry references are in
diagnosis-evidence.json (external historical reference: `diagnosis-evidence.json`). The version remains frozen.

The actual pass achieved guard0/unknown0, but310/2880contact+obvious and
1984/2880moving-pusher/no-AO. This does not support promotion. Safety mitigation,
strict acquisition and route cycling overlap; there is no single-change ablation
that would identify one unique cause of the deterioration.

## Where the2880 control intervals went

The following production command states are mutually exclusive and sum to2880.
Outcomes are measured in the same actual interval after that command.

| Command state | Controls | Contact substeps | Contact+obvious | Empty travel |
| --- | ---: | ---: | ---: | ---: |
| Normal-only acquisition |630 |2093 |1 |297 |
| Contact follow |337 |3169 |286 |10 |
| Wall-risk mitigation,free route |390 |58 |16 |304 |
| Transactional recovery route |1167 |264 |2 |1053 |
| Fixed withdrawal |356 |24 |5 |320 |

Only337controls(11.70%)were established contact-follow commands; they account
for286of310obvious co-occurrences. Normal-only entry630(21.88%)and recovery
routing1167(40.52%)dominate the pass without producing comparable motion. The
630normal-only intervals comprise291nearstatic,338small and1obvious intervals.
This is observed under the declared thresholds,not evidence to relax them.

Raw modes are967contact intents,1896free routes and17holds. All17holds occur in
the fixed withdrawal command state. The main inefficiency is active empty travel,
not a controller that simply holds every scene. Of1984empty intervals,320are
withdrawal and1664are outside withdrawal. Planned push itself has1373empty out
of2075controls; recontact230/269,approach61/180. The tasks were not deleted.

## Risk mitigation is real,and extends into route travel

There are390risk cancellations(13.54%of commands),389outward18px retreats and
one negative-tangent action.225cancel direct entry/follow candidates;165occur
while a route transaction is active. Independent current-geometry checks of
those165original bounded route segments find every one free in the fixed object
geometry. A fixed-pose free segment is not a guarantee against a moving object
during the following0.1s; these checks therefore do not prove the cancellations
were unnecessary. They do show that the conservative heuristic also interrupts
currently verified route progress,and is not confined to sustained pushing.

Repeated mitigation produces92committed near-wall reacquisitions and28searches
with no confirmed alternative. One additional target commit follows incidental
requested-arc contact. In total121replan commands produce93target revisions.
No route_no_progress or unconfirmed_route replan trigger occurs. Local progress
along the issued action can be positive during both an outward retreat and the
later return to staging; it is not equivalent to completing acquisition.

479commands also carry a state-risk flag,which overlaps cancellation.1376commands
begin before stable initial-requested acquisition.562commands carry the latest
observed-pair rejection label:134normal/arc mismatch and428no legal union staging.
These are carried command flags,not562independent bad physical contacts. No issued
executable target is labelled invalid,and no current-geometry-route-invalid
reason appears. The pair validator avoids submitting those rejected pairs,but
these counts alone do not quantify its isolated effect on efficiency.

## The four named same-scene checks

### X03: protection intervenes early,but acquisition is largely lost

`CP2-R-A-X-03` completes48controls,with0obvious intervals and42empty intervals.
It initially acquires arc49:the requested-stable flag is true beforecontrol3.
The first actual risk intervention iscontrol4,earlier than CP1's failingcontrol5:
topgap23.3138worldpx,maximum closing boundary speed130.1164px/s,and projected
remaining room-5.7064px. Candidate(377.9800,48.4449)is replaced by outward action
(383.4129,126.2641). The next command retreats again,thencontrol6commits arc24.

The48commands are5normal-only,1follow,16risk retreats,22routes and4withdrawals.
Five target changes repeatedly select arcs24/23 after the original acquisition.
The record supports that the risk gate actually operated before the old failure;
it does not establish that this one gate alone explains guard avoidance or all
lost motion. The old CP1 run was truncated at6charged controls,so it is not a
48-control efficiency baseline. The original R3A counterpart had3obvious/12empty.

### T03: the static bridge regression is not the actual trajectory

`CP2-R-A-T-03` has0obvious/38empty:10normal-only,22routes,12risk retreats and
4withdrawals. Stable initial-requested acquisition never completes. Its first
risk cancellation iscontrol12,with bottomgap21.9877worldpx and projected room
-5.3090px; its first target commit iscontrol21.

The recorded CP1 state10 regression still demonstrates that committing a new
target retains its old departure bridge and emits a non-hold command. CP2's
physical trajectory diverged before that state. In the entire actual60case pass,
the1332active route packets are1214free routes and118direct departures; none is
`retained_departure_bridge`. Thus this actual pass cannot be presented as a
same-state physical validation of the bridge fix. CP2 replaces CP1's repeated
holds with movement,but here that movement is mostly empty route/retreat travel.

### B04: deterioration exists with no risk cancellation or pair rejection

This is matched `CP2-R-A-B-04`(original R3A B018),not fresh weak B04. Obvious
intervals are33in R3A,14in CP1,and0in CP2; empty intervals are10,14,33.
CP2has14normal-only commands,30routes and4withdrawals,with0risk cancellations,
0replan commands,0target revisions and0carried observed-pair rejection flags.
Consequently this regression cannot be attributed to wall gating alone.

Its initial requested arc44has42actual positive-impulse substeps. Control0has4
requested substeps including the terminal one,so fails the5-substep threshold.
Control1has6requested substeps,but the terminal point is on neighbouring arc43.
Controls11,21,33,43repeat5or6requested substeps without a terminal requested
point. Strict requested-stable acquisition therefore never completes.

Atcontrol2the actual lateral error to the requested normal line is5.2564px,
above the frozen2px alignment threshold,although the production normal-entry
prefix remains geometrically verified. The command routes back towards staging.
Atcontrol8error is2.1711(route); at9it is1.4619(normal entry). Actual contact
returns at10/11,andcontrol12lateral error5.5424starts the route cycle again.
The same pattern recurs after withdrawal. This is direct evidence of the strict
arc-establishment/alignment interaction and repeated staging travel; it is not
a counterfactual proof that changing either threshold alone would be sufficient.
No incompatible executable pair is submitted on this actual path,so the four
CP1 illegal-pair static regressions should not be claimed to have occurred again.

### T040: a previously improved scene retains some motion,but regresses

`CP2-R-A-T-01`(T040)has11obvious/30empty,versus CP1's23/11 and R3A's6/40.
Its10normal-only,9follow,4risk retreats,17routes and8withdrawals still include
normal pushing. First cancellation occurscontrol5: leftgap23.3495px,closing
boundary speed154.5721px/s,projected room1.3014px. Control7commits alternate arc22;
the immutable request is arc38,while the tracked executable point had already
moved through arcs37/36.
Follow resumes at16and35; cancellation39and later route travel consume the tail.
This shows the safety mechanism is not a blanket initial freeze,but the combined
controller loses much of the useful continuation that CP1 had obtained.

## Requested arcs were hit;stable useful pushing was mostly not established

The12weak cases have actual exact-ID contact and positive impulse evidence,as
reported from raw arrays. Only old weakT,new weakT,and new weakB ever expose a
stable-requested-before-action flag; the other9do not. Five old weak non-T cases
each spend14commands in normal-only acquisition and30in routes. Four of these
have neither risk cancellations nor carried pair-rejection flags; old weakC
carries rejected raw pairs but still has0risk commands. Hence zero useful motion
on those cases cannot be collapsed into a single near-wall explanation.

For example old weakL's arc28has36positive substeps:3terminal hits atcontrol1,
then6nonterminal hits at2,and similar later cycles. Old weakC arc73has32positive
substeps with the same incomplete/terminal pattern. Old weakT arc1instead has
10positive requested substeps atcontrol2including terminal,and enters follow.
Raw exact hit and the narrower controller establishment condition are distinct.
The reported requested-contact obvious motion remains only1old-T interval and
0for the other11. No neighbour arc is substituted as requested success.

The data identify concrete costs and interactions under this one frozen version.
They do not isolate a unique cause by ablation or authorize another tuning pass.
Preserve the version and evidence,do not promote or merge it,and keep the
original useful R3A model-validation question separate from collector changes.
