# R3B: approved full first epoch and complete independent background pipeline

Received2026-10-30 through planning chat__LOCAL_CHAT__,
quoting the human's explicit approvals:

> 可以，就按这个方案先训完整一轮，重点看O和C的棱角和运动预测。让他完成后主动回传给你，你把结果给我看，等我审批再继续。

> 还是一样让训练在后台跑，负责人启动了之后就可以停止他那个对话了。

This releases only R3B's full first epoch and paired evaluation. It lifts the
current stage's waiting_user_approval gate;all later epochs,collection/sampler
changes and server work still need a new explicit human approval. Retain the
standing proactive stage callback consent. Do not ask again for approved GPU,
background execution or callback permission.

## Frozen training

Use only accepted R3A:768TRAIN+144seenval,T/L/O/C/X/B;manifest
0e91e1412f937c7d8f39473243a27a3720317a07474e730a510e628d040c0a19.
No CP1/CP2 data or collector replacement. DGKNQRWZ remain sealed. SelectedTRAIN
23,984 distinct windows and exact first1499batches x16,orderSHA
800655f0dc54234f92c9784dc4b0ff94e44910d488d39f32439ff625a2ee2335.
The old8epoch proposal is provenance,not permission to continue past epoch1.

Starting complete checkpoint is ../r2c3/run/last.pt,105331438B,SHA
c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148.
Restore model/AdamW/scaler;holdLR1e-5,no oldcosine rebound or silentreset. Newdata
cursor0 and three conditioning RNG migrations are explicit,with oldlineage6000
separate from stageattempt/success/skip/draw and separately billed engineering.
Frozen RNG seeds:branch1409496638345316890,target5508129404114374297,
weak7192873019884664397;branchCPU,target/weakinputdevice.

Original native GPC image+action->image fullparameter training,micro4x4,
effective16,AMP. Clean Bernoulli50% per microbatch truly sigma_cond=None;weak50%
per-sample loguniform[.002,.05],historyoffset0. Original targetnoise andFP32
residualMSE unchanged;no pose/motionhead,warp,geometryrepair,maskinput,edgeloss,
newresolution orrecoloring. TwoCPUloaderworkers,boundedcache/prefetch;no fullRGB
library or dependency install. AMPskip consumes the batch once and remains billed,
with no redraw. Exactly1499formalattempts ends this run;time is statistics,no cap.

Perform only necessary actual GPU restore/save-resume/accumulation/parameter-update
integration checks beforeformalupdates. Record engineering exposures/replays and
fees separately;restore6000/cursor0/RNGs afterwards. Originalcheckpoint readonly.
One rolling complete R3B last.pt,every100successfulupdates andepochboundary,
containsmodel/Adam/scaler/fullnewcursor/global+threeconditioningRNGstates.
Exceptions save exact state where possible and exit;no automatic restart,blind
epochreplay or experiment changes. Explicit validcheckpoint resume cannot repeat
already-consumed data;unrecoverable durable-progress gaps must be disclosed.

## Paired evaluation and report inside the worker

Before first update freeze144natural seenvalwindows,one seededuniformlegalH4
anchor per family,24perletter,no motionfilter;evaluate H1/H4 at the sameanchor.
Keep old16stressdiagnostics separate;deduplicate identical calls only,not their
statistical populations. Old6000 t0 precedesupdates;epoch1 uses identical windows,
seeds,native3steps andcopylast. H4feedback israwprediction only,nofutureGT.
Persist deduplicated rawfloat32,no extra float64 copies.

Report allnatural/perletter/motiongroups/nearstaticdrift,strictshape fail/unknown
full denominators. Focus O/Ccorners,motion,holes/breakage/additions/contours,
grayobjectIoU andmeasurablecentroid/rotation;bluepusher separately. Occlusion or
unmeasurable pose cannot count as a pass. No registration,smoothing,edgefixes or
maskinputs. Use integernearest O/Ccloseups.

Old familiar regression cases remain separate,not independentnewvalidation:
O R2B-new-25-O-left-bottom:12;B R2B-new-48-B-top:5;
C R2B-new-28-C-left:6;X R2B-new-38-X-right:13.
Reuse existing6000raw with exact originalseed/native3step/batchcontext;runepoch1
samecontexts and showGT/copylast/6000/epoch1 H1/H4. Do not shrink context to a
single case and claim equivalence. One findings page andO/Cfigs must say whether
the model improves overcopylast andsame-window6000,and disclose remaining issues.

## Independent background and completion handoff

One hidden worker coverspreflight/realGPUchecks,t0,all1499formalbatches,epoch1eval,
figures/reports,terminalmanifest andcompletion. Save status/15sheartbeat/logs/
exceptions/counters so the chat can end. Reuse the previously approved R3A
existing Explorer desktop broker and directbasePython/venv startup method.
Check no duplicate,actualworker WindowsJob=false,launcher exits,and genuine
heartbeat/taskprogress advances after launcherexit. A PID alone is not proof.
If automatic approval review denies a launch,report the actual action/reason;
do not bypass it. No autostart,system sleep/security changes or offline MCP.
Computer poweredon and notsleeping is necessary.

After actual independence proof,send a concise handoff to parent withPID,cwd,
phase/log/status/evidence/reportpaths;save actualreceipt,then end the turn rather
than busywait. Complete callbacks require app/tools available:Python continues
and saves locally afterCodexclose,but cannot callMCPoffline. Official thread
heartbeat completion check is permitted if no supported local completion event
exists. It reads completion/status quietly,does no training/repair/restarts,
and uponterminal result wakes thisowner toreadrealoutputs,sendparentcallback,
savereceipt,thenpauseitself andSTOP waiting_user_approval. Parentdoesnotpoll.
If monitoringunavailable disclose localcompletion/next-open-callback limitation.

## Resource scope

R3B persistentstagecap320MiB=335544320B;shared3GiB=3221225472B plus256MiBatomic
reserve. Rebaseline afterCP2;record actualdiskfree. The old144MiB internaldraft
cap cannot silently reduce the approved144naturalvalidation orselectedpool.
Count code,reports,notification andcallbackreceipts. No checkpoint/librarycopies
or largeRGBcache. Temporaryatomic swap is covered explicitly byreserve. No
serverexecution;local3060Ti firstepoch only. Finalreport is not later approval.
