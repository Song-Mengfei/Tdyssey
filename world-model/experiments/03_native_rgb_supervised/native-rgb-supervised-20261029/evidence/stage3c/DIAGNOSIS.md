# Stage3C diagnosis: registered18-window fit target reached; validation/shape limits remain

One authorized exact continuation completed129→250:121 new updates/1936 new draws,4000 cumulative draws over the same18 unique TRAIN windows. Original optimizer/scaler/scheduler/RNG/order/progress restoration and the complete final checkpoint were verified inside the counted model run by root; CPU scoring did not load weights or Torch.3C model time423.859s; cumulative new-round model time868.484s/250attempts. The old3+3B444.625s/129 history remains unchanged.

## Registered TRAIN18 result

Hold raw gray error(1−IoU)=0.23258317101512865; update129=0.22807596210836506; update250=0.18517149901054739. Final improvement versus hold is20.3848248339%,with15/18 strict wins and0 finalTRAIN hard failures. The unchanged>=20%+>=12/18 diagnostic therefore passes. This is a fitting result on18 exposed windows,not formal TRAIN or generalization acceptance.

All18 final windows improve over their own update129 rawgray error(18.8115% mean reduction versus129). Three final windows still do not beat hold: J train-2b-061:8,P train-2b-011:18,Y train-2b-071:5. Full hold/frozen/129/final216-method-row table for18TRAIN+36DEV is in comparison129-final/all54-method-comparison.csv.

TRAIN raw contour error(imagepx):hold0.427067→1290.456354→2500.343421. Fixed gray-object RGB MAE:hold0.0485254→1290.0520821→2500.0427277. Pusher error remains separate:hold0.393613→1290.296002→2500.253980; pushercentroid error1.075708→0.753388→0.616375imagepx. Gray-object fit cannot be credited solely to blue-pusher improvement.

Fixed denoising loss fell0.0342575386(original)→0.0024270865(129)→0.0019337337(250). These training numbers demonstrate optimization but do not substitute for native RGB results. Existing visible-mask centroid auxiliaries are retained in DIAGNOSIS.json/CSV; they are influenced by shape/occlusion and do not prove action causality.

## DEV36 and actual boundary/static groups

DEV final mean gray error0.259424348 versus1290.274351965(5.441% better,23/36 wins and1tie),but hold0.196032459 remains better by the original reference. Final wins over hold are6/36. For31moving windows,final0.269618464 versus hold0.223749623 is20.5001% worse,with6/31 wins. The1intermediate window is0.302521008 versus hold0.120930233.

For4near-static windows,hold error0; final grayerror0.169645781 versus1290.174189315. Final mean centroiddrift0.610816imagepx versus1290.582984:some contour improvement coexists with increased drift. No requirement to beat zero-error hold is imposed and no static success is inferred. Original per-letter,actualboundary<=32(includingnegative) andgap-band groups remain in score-report.json.

All54 final RGB arrays are finite/in-range. DEV F seen-val-2c-112:29 andJ seen-val-2c-097:38 have empty blue-pusher masks and remain hard failures. Gray masks are present and independently scored,while pusher-position penalties remain; the reportedDEV pusherposition mean8.25954imagepx includes2 unavailable-centroid diagonal penalties,not an observed8px drift of every pusher. The129 point had onlyJ empty-pusher; this deterioration is retained.

## Shape, full coverage of examples and failures

Legacy deformation:TRAIN0/18assessable and18unknown;DEV1failure outof3assessable,33unknown. The remaining assessable failure isT seen-val-2c-157:57; O seen-val-2c-082:58 changes fromlegacyfail at129tolegacy-pass at250 under the unchanged calibrated rule. This narrow assessment is not a claim that corners/grooves are restored.

All18 TRAIN comparisons were visually inspected on3fixed pages with true/hold(last)/frozen/129/final columns. Gray appearance and some contour/position details improve,but rounded corners,hole boundaries andnarrow-groove smoothing remain visible in multipleletters. Visible holes/components/relativearea andfulltruth-mask auxiliaries are allretained. Contact/clippingunknown is never counted as shape-pass,and no alignment or template repairs the predictedRGB.

Original finalscore galleries retain every18TRAIN/36DEVwindow in3+6pages; the supplementary129→250 comparison has3additionalall18pages,without outcome ranking. All originaltruth/hold/frozen/windowID arrays are byte-identical to3B in both cohorts(16checks). The original frozen and129 predictions were directly reused,not rerun.

Frozen baseline finiteOOR counts remain5TRAIN/13DEV,allgeneratedraw,notmissing. Original failurepenalties andfull18/36denominators remain unchanged. Both129and250 numericarrays areinrange; finalDEV empty-pusher failures areseparate. range-accounting.json decomposes failurepenalty sums versusactualrange-valid error sums only; it is not another score or a finite-only acceptance.

## Timing and scope

CPU scoring ran exactlyonce for11.480739 measuredwallseconds; frozencomparison toolonce for0.9521395seconds(total12.4328785 measuredCLIwall),includingPython/stdouttee. Timestamps/exits are in score-cpu-timing.json/comparison-cpu-timing.json. This is separate from the model ledger. Source/calibration/protocol and old3/3B files were hashchecked unchanged; noTorch/model/weights/GPU orphysics operations occurred in these CPU tools.

Conclusion: the original18-window motion-fit diagnostic is reached after250 cumulativeupdates. Generalization is not demonstrated:DEVmoving stillloses tohold,staticdrift andshape/pusher deficiencies remain. Formal data training,H4/full319/unseen evaluation andanyadditionalmodelstart remain unapproved; no continuation is inferred from unusedbudget.

Evidence: score/score-report.json;score/per-window.csv;comparison129-final/comparison.json;comparison129-final/all54-method-comparison.csv;comparison129-final/train129-final-01.png through03;DIAGNOSIS.json;range-accounting.json;score-cpu-timing.json;comparison-cpu-timing.json;model-work.json;run/last.pt.
