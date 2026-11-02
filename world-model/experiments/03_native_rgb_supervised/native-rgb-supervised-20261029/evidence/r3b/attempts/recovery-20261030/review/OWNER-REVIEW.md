# R3B first-epoch completion: owner read-only review

Windows venv redirector/duplicated-handle failure was confirmed and repaired by
direct base-Python multiprocessing with the same venv. The authorized1499batches
and23984distinct windows completed successfully. Average image prediction improved;
motion-conditioned shape reliability and O/C corners remain unresolved.

Finished2026-10-30 18:19:56 Asia/Singapore. This review only reads saved results,
hashes and figures;no model,GPU,physics,new rendering,scoring,repair or restart.

## What is verified

Completion,durable progress,first-epoch result,pipeline,checkpoint metadata and
1499actual optimizer-event records agree:1499successes,0skips,0abandoned,
23984unique targets,exact frozen order,epoch_complete=true,lineage7499. The
checkpoint is105438958bytes,SHA256
23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7.
Static checkpoint component inspection used no weight loading. Original6000,
R3Amanifest and exact144natural/16stress/old4 spec stayed unchanged. All456final
manifest bindings and161frozen inputs match. No epoch2,heldout access,new physics,
CP1/CP2 merge or training-protocol change occurred.

The terminal status/heartbeat formal summary remains1475 because its progress
cache updates every25batches. This is a reporting limitation,not an early stop:
all authoritative records and independently reconstructed actual draws show1499.
Original output files are preserved;future readers must use completion/progress.

## Natural validation (144windows; error lower is better)

|H|copy-last ROI|6000 ROI|epoch1 ROI|copy/6000/epoch1 gray IoU|
|---|---:|---:|---:|---|
|1|.0396|.0407|.0175|.835/.839/.931|
|4|.0901|.0771|.0462|.638/.680/.795|

NaturalH4 ROI improves about40% against6000 and49% againstcopy-last in the mean.
It wins125/144against6000 but75/144againstcopy-last;H1wins66/144againstcopy-last.
Lower mean error does not mean every window improves. NearstaticH4 N52 still
favorscopy-last:.000435vs.013422epoch1ROI,drift0vs.154136px,despite a large
improvement over6000(.053965ROI/1.012787pxdrift).

On the66obvious-motion H4windows,ROI improves .096242to.078168,but strict
pass/fail/unknown changes6/5/55to5/8/53. OverallH4 is50/13/81to51/10/83;
83unknown/144 is not a pass. Existing hard-generation failure also includes an
empty blue pusher:6000has7empty-pusher cases pernatural horizon,epoch1has0.
Therefore the overall shape-fail decrease cannot be read as gray-object-only
shape improvement. Frozen scores were not changed.

Common-measurable pairedH4 pose comparison uses96/144windows:centroid error
.881223to.429145px,angle10.09375to2.08333degrees. The separate101/144and122/144
available-case means have different support and are not a matched comparison.
NaturalH4 blue-pusher ROI is separately .169644/.052070/.034846(copy/6000/e1).

## O/C and fixed visual review

|Natural H4|N|copy/6000/epoch1 ROI|6000/e1 gray IoU|6000/e1 shape P/F/U|
|---|---:|---|---|---|
|O|24|.0950/.0949/.0575|.615/.762|9/2/13 to8/2/14|
|C|24|.0905/.0912/.0519|.591/.736|5/5/14 to7/3/14|

O's two original failures are empty-pusher cases;its two epoch1failures are
gray-object deformation. O hole-count differences2to0/fragments0to0 do not
prove rigidity. C retains gray-shape failures and has one extra-fragment case
aftertraining. Raster corner-count differences are diagnostics,not a rigid-body
proof;occluded/unmeasurable cases keep their full denominator.

The owner visually reviewed all8H4panels(naturalO/C four,oldO/B/C/X four),oldO/C
H4integer-nearest closeups,andoldO/C H1panels. Static naturalO/C examples are
sharper and closer;movingO still rounds edges/hole boundaries,including the oldO
case labeledpass. OldC motion/orientation and corner placement improve,but its
H4shape still fails and the contour remains rounded/distorted. OldB orientation
remains poor(shapeunknown,angle165to167degrees). OldX improves. Do not promote
a local shape pass as evidence that corner fidelity or rigidity is solved.

Old4 are familiar regression cases,not new independent validation:meanH4ROI
.2042/.0949/.0519,IoU.193/.551/.753(copy/6000/e1),P/F/U0/3/1to2/1/1.
Stress16 remain separate:H4ROI.1102/.0759/.0513,grayIoU.554/.666/.752.
Full six-letter/H1/H4/four-motion strata and common-measurable comparisons are
in metrics-readonly-summary.json and METRICS-REVIEW.md;these aggregate existing
saved rows only and do not introduce another evaluation run.

## Costs, storage and stop boundary

Retain old failedmodel33.047s;separateCPUdiagnostics9.876s. Newmodel pipeline
records1456.531s:restore17.641,postGPUloader4.532,engineering6.157,t0149.641,
formaltraining1205.891,e1evaluation72.172plusrecordingoverhead. Engineering1update/
16draws is discarded;formal1499updates/23984draws remains separate. Workerwall
1469.752s includes nonmodelreporting,so do not add it again to modeltime. Global
charged/completed updates7750,ledgerclosed,nopending,no time hardcap.

Read-only artifact audit counts allR3B147186678B,shared1694369059B,under320MiB/
3GiB with256MiBatomicreserve;review/callback metadata add only small overhead.
No raw artifacts or historical failures were deleted.

After the actual authorized parent callback and saved receipt,finish the
completion check and STOP waiting_user_approval. No further epoch,data,
sampling,model,server,repair or restart is released by this report.

Evidence:../REPORT.md,../report/index.html,final-artifact-audit.json,
metrics-readonly-summary.json,../background/completion.json,
../run/first-epoch-result.json,../run/progress.json and../run/last.pt.
