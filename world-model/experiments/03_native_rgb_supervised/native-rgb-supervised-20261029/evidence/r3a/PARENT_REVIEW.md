# R3A direct implementation instruction — 2026-10-30

Source: supervising chat “制定 Tdyssey 世界模型实施计划”,
__LOCAL_CHAT__. Human correction:6000 updates repeatedly
used the same small41-trajectory/861-window library; independent scenes and
clearer history conditioning must now be substantially improved together.
Implement and actually collect,not just another plan. No repeated user approval
for ordinary local implementation/collection. R3B GPU training waits for the
retained human requirement of actual movement/video review by the parent.
At R3A completion actively send reports/data summaries/local paths to that chat
under ../standing-callback-authorization.json,save real receipt,then STOP
waiting_parent. No parent polling. No model-time cap; elapsed time is statistics.

## Verified diagnosis and retained constraints

Old TRAIN is6seen lettersT/L/O/C/X/B,41 trajectories(X6,others7),24controls each,
984controls/98.4simulated seconds,861 overlapping H1windows.96000draws exposed
only those861,window min/median/max21/38/1280. Gray target area mean2.61735%.
Original history noise is exp(N(-1.2,1.2))clipped.002..20 plus.1offset on every
training history. Inference s_cond0 uses sigma_cond=None,c_noise_cond0,which
equals log(1)/4 numerically; clean must truly useNone,not sigma1 noise. This
condition mismatch is a factor to investigate,not a proven unique rounding cause.
Some6000 C TRAIN H1 windows haveIoU≈.908 and clear turns,so no claim the network
cannot draw corners. Old full6000 checkpoint105331438B,SHA
c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148
is preserved. Earlier experiment data/weights/failures/parent galleries stay.

Parallel CPU work:A independent-scene collection and state→RGB on-demand data;
B clean/weak conditioning with focused CPU behaviour checks. No GPU training
or inference inR3A,no servers/downloads/new environments/no8heldoutDGKNQRWZ.

## Actual new data and split

- Six seen letters only. Perletter128 complete TRAIN+24 new seen-validation
  trajectories =768TRAIN+144val=912 successful independent48-control trajectories.
  Expected no-failure43776controls/437760physics substeps. H1TRAIN legal upper
  bound768*(48-3)=34560 beforequality,not guaranteed selected/quality-passing.
- Freeze independent scene/family/seed and split before collection. No copied
  files,repeated seed/base scenes,rotated/mirrored derived train/val or trajectory
  chunking as independent scenes. Check exact state/action/image summaries and
  near initial scenes,including disclosure of any crosssplit similarity.
- Fixed selected0.1s half-life/protocolSHA
  58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6.
  Same mass/inertia,finite-force PD,pusher speed200,10Hzcontrol/100Hzphysics,
  walls/solver/2worldpx penetration guard. No changed damping/inertia/teleports.
- Extend current-state-only R2B contact-following to48 controls:sustained pushing,
  unload/withdraw/recontact. Controller may use current privileged geometry/state
  but no future truth/dynamics search;model inputs remain onlyimages/actions.
  Save truly executed absolute actions and actual contact,not planned flags.
- At least12initial-yaw bins;center,fouredges,fourcorners with within-bin spatial
  variation. Rotate reachable arcs/corners/concavities,near-COM and±lever,
  grazing/wall/static/small/release cases. Report actualcontact arc/localpoint,
  lever sign,normal/tangent commands and realwall contacts,not only plannedlabels.
  Multiple orientations/actions perreachable arc;unreachable holes/concavities
  and missing combinations explicit,no full Cartesian coverage claim.
- H1 obvious motion stays translation>=2image96px OR abs rotation>=5degrees.
  Fill actualtranslation/positive/negative rotation gaps with effective segments,
  not lowerthreshold or repeat a few windows. Selected TRAIN pool target at least
  abouthalf distinct obvious-motion windows. Keep full legal/valid library and
  separately disclose selectedpool;retain static/small/release/wallstall negatives.
- Report perletter/pertrajectory/scenario H1/H4 translation and signedrotation,
  zero fraction/P10/median/P90/P95/max,contactrun lengths,time displacement,
  penetration/earlystop/failure and actualarc/boundarycoverage. Need many distinct
  trajectories supporting translation and both rotation signs,not one extreme.
- Failed slots remain;no recollection of a completed success. Each slot at most
  two replacement candidates,with reasons/attempts/allactualcontrols counted.
  Widespread engineeringfailure stops for repair/report,not blind batchretries.
  Do not silently shrink back to tens of trajectories or rename deficits done.

## Compact storage,rendering and loader must be implemented

- Save actual complete15state at everycontrol,executed actions and necessary
  substep/contact/force/penetration evidence as compact numeric arrays;do not
  save only seeds and replayphysics. Avoid perstep repeatedJSON.
- Large RGB is not persisted. Render saved states at access using the same96RGB
  white/gray/blue specification. Small representativeframes/videos/consistency
  examples only. State/letter/masks are renderer/audit metadata,never modelinputs.
  Loader returns4historyRGB+aligned4actions+H1targetRGB,normalized once.
- First prove offline state-render against saved R2B original RGB across6letters,
  rotations/contact/wall/holes,pixels/shape/timing. No state evolution rerun or
  replacementappearance. Diagnose mismatch rather than silently changingGT.
- Bounded CPU multiprocessing allowed,loader first2workers with small bounded
  RAM cache/prefetch and measured throughput;no fullfloat-library preload or
  environment/window re-creation perbatch. GPU modelwork reserved forR3B.
- Fresh scan before actualcollection. R3A allnewdata/code/audit/representative
  videos target<=96MiB;reserve144MiB forR3B plus256MiBatomic under3GiB shared.
  Count every directory;no deleting oldfiles/hidingnewfiles/reducingdata. If not
  possible,report exactbyte deficit and concrete storage options before launch.
- Default trajectory/letter/amplitude-stratified sampling without replacement
  per-epoch;avoid neighbouring overlapping windows inonebatch. No rare both-axis
  category oversampling. Log unique trajectories/windows/repetitions/maxmedian
  exposures separately from completelegalpool and selectedpool size.

## Clear/weak history first version

Same original full-parameter GPC,target-frame sigma+offset distribution and
FP32 residualMSE;no foreground/frequency/geometry loss,motion/pose heads,warp,
maskoutput,shape repair,color change or resolutionchange.

Per-microbatch independent50/50 branch:clean histories unmodified,no pixel/offset
noise,sigma_cond=None using actual inference encoding;weak histories use
log-uniform sigma_cond[.002,.05],historyoffset0,real matching sigma encoding.
Microbatch shares branch to avoidNone/tensor mixing.50/50 is an experiment
setting,not an optimum. Target sigma expN(-1.2,1.2)clip.002..20,offset.1 and
epsilon remain original. Branch/targetnoise/weaknoise use explicit independent
serializable RNGs. Focused CPU checks for actualNone branch,weak encoding,target
noise semantics,FP32loss,normalization and exact RNG resume. No future target as
condition. Preserve oldsource/config/results;new strategy has distinct identity.

R3B adapts the delivered6000 fullcheckpoint to newmain data,not old861 stream.
Explicit data identity/cursor migration;old6000 lineage and newdata exposures/
stageupdates counted separately. Proposal must state optimizer/scaler/LR restore
or reset choices exactly for parent finalization,not pretend same-distribution
continuation. Schedule observations from actual newunique pool/completeepochs,
not a mechanical300-step trial. Newdata+clearhistory joint improvement does not
prove onefactor uniquely. Serverfull training is always manually run by human.

## Delivery

Actualmanifest/split/counts/bytes/integrity/amplitude/contact/boundarycoverage
with deficits,loader+noiseimplementation/contracts/RNGrestore/throughput,
resourceplan and executableR3B proposal/config referencing actualnewlibrary.
At least perletter real48-step10Hz videos for visibletranslation,±rotation and
boundary cases,plus frozen-seed random samples and worst/failure examples.
No sped-up playback hidingstall/inertia,no syntheticphysicsresimulation.
Actively callback thenstop. Parent first reviews actualmovement/videos and
directly releasesR3B,without the human relaying approval again.
