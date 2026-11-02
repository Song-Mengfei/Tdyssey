# Stage3B diagnosis: training worked, registered motion-fit target not met

One authorized corrective run completed129 optimizer updates/2064 draws on18 uniqueTRAIN windows,114–115 exposures/window. All296 trainable parameter tensors had finite positive first gradients and all296 changed. The model has8,729,827 parameters; the complete final checkpoint and optimizer/RNG/order/scheduler review passed. Fixed denoising loss decreased0.0342575386→0.0024270865, but loss reduction is not native prediction acceptance.

3B model work416.281s plus the preserved old28.344s totals444.625s,129 attempts,within the original combined600s/200 caps. Stop reason is stage_gradient_cutoff_with_reserved_tail. Old native1899attempts/5352.516s remain separate historical charges. No unused budget authorizes another run.

## TRAIN18: actual native motion-fit diagnostic

Fitted mean unaligned gray-object error(1−IoU)=0.22807596210836506 versus hold0.23258317101512865:1.937891% relative reduction and9/18 strict wins. Registered requirements were>=20% and>=12/18. The diagnostic is NOT passed; this conclusion uses all18 and unchanged thresholds. All18 fitted RGBs are finite/in-range with nonempty gray and blue masks,so this is an actual bounded fitting result rather than missingness penalties.

The fitted gray foreground RGB MAE0.0520821 is worse than hold0.0485254; raw unaligned contour error0.456354imagepx is worse than hold0.427067. Pusher mask error improves0.393613→0.296002 and pusher position error1.075708→0.753388imagepx, but these improvements cannot substitute for gray-object motion fitting. Fitted object centroid change fromlast averages0.507171imagepx; that alone does not prove action use or copying.

All18TRAIN fitted legacy deformation assessments are unknown0assessable/18unknown due the retained contact/clipping ambiguity,not shape-pass. Visible topology/holes/area fields remain in the per-window CSV. The displayed raw outputs learn gray appearance but visibly round/smooth several rectilinear strokes and hole boundaries; no output is repaired or replaced by registration.

## Independent DEV36

Overall fitted gray error0.274351965 versus hold0.196032459,wins4/36. For31moving windows,fitted0.287241605 versus hold0.223749623,wins4/31:28.376% worse than hold. The one intermediate window has fitted0.275423729 versus hold0.120930233. These are development-validation results,not unbiased final testing.

For4near-static windows,hold gray error0; fitted error0.174189315 andmean centroid drift0.582984imagepx. No requirement to beat zero-error hold is imposed; this quantifies drift. Of3assessableDEV shapes,2 fail the unchanged legacy deformation rule(O seen-val-2c-082:58 andT seen-val-2c-157:57),1passes;33/36remainunknown. Unknown does not count as a pass.

DEV J seen-val-2c-097:38 has an empty blue-pusher mask and remains a hard generation failure despite valid finite RGB and measurable gray mask(IoU0.6117647). The gray measurement is retained; the separate pusher penalty remains. All other fitted DEV windows have gray/blue masks. Grouped actualboundary<=32 includes signednegative gaps:14boundary windows fittederror0.2618203 versus hold0.1850526,wins3;22others0.2823267 versus0.2030196,wins1.

## Complete baseline, range failures and denominator accounting

All54frozen and54fitted raw outputs were generated. Frozen finite out-of-range statuses are5/18TRAIN and13/36DEV; these are GENERATED raw arrays,not missing data. They remain unchanged,withmin/max/negative/>1counts inrange-accounting.json,and receive the original hard penalties. All54 fitted arrays are finite and in-range;1DEVempty-pusher failure is additionallyretained.

Frozen mean grayerrors0.990620246TRAIN/0.994472906DEV include those range penalties and empty-mask outcomes. The large fitted improvement relative to frozen is appearance/domain adaptation plus the registered penalty accounting; it does not satisfy the independent hold-based motion criterion. range-accounting.json separates status-failure penalty sums from actual range-valid gray-error sums without introducing a finite-subset selection or a new acceptance metric.

The repeated first frozen batch is exactly the original4windows,seed3335533039. Every rawarray byteSHA is identical,maxabs difference0,status identical. The6copied index/truth/batch files,published checkpoint identity andcommonfrozen sources match. No preferred baseline was selected. The original stage3 abort,arrays,score and28.344s remain preserved.

## Evidence and limits

All18TRAIN and36DEV windows appear in frozen order on3+6gallerypages. Allmethod rows,rawsourceNPZs,shape/pusher metrics andfailurestatuses are retained. Only the fixed original score9a165.../protocol353ba.../calibrationcd2ed...were used. Registration remains auxiliary visible-mask scoring with its disclosed10degree/global,1degree/local andhalf-imagepixel raster grid,not pose accuracy or prediction repair.

CPU baseline comparison and scoring loaded no model,Torch or weights and made0GPU/model/physics calls. Exact totaloffline scoring CLI walltime was not instrumented: initialtoolwait10.0060182s yielded while itran; finalpollcompleted. score-execution.json discloses this andrecords completion timestamps rather than inventing an exactduration or rerunning. Model-clock accounting remains separately measured444.625combinedseconds.

Conclusion: backward/optimizer/checkpoint chain and gray appearance adaptation are evidenced. The18-sample motion-fit target was not reached,DEVmoving doesnot beathold,andstaticdrift/shape deficienciesremain. This one boundedrun doesnot establish generalization anddoesnot disprove theGPCroute. No furthermodelstart,scoring-policy change,newphysics orlaterstagewas performed orauthorized.

Evidence: baseline-prefix-comparison.json;range-accounting.json;score/score-report.json;score/per-window.csv;score/gallery-train/;score/gallery-development/;model-work.json;run/last.pt;score-command.log;score-execution.json.
