# Stage3 diagnosis: strict range abort before training

The single authorized run stopped during the first frozen TRAIN batch of4 native outputs, before entering denoising-loss measurement, gradient computation or optimizer attempts. Strict original-weight initialization and generation of these first4 raw outputs were exercised; an optimizer/gradient training chain and actual motion fitting were not exercised.

Root model-work accounting is28.3440000000046 seconds,0 optimizer attempts and0 completed updates. The first A output(train-2b-018:11) has6 negative channel values,minimum−0.0196077823638916 andmaximum1.0. Its raw values are retained unchanged. The predeclared strictfinite[0,1] guard stopped the run. No clipping, epsilon adjustment, second run, reinitialization or extra prediction occurred.

Of18 TRAIN frozen windows,4 raw outputs exist:1 range-invalid and3 range-valid.14 were not attempted. No development frozen outputs and no fitted TRAIN/development outputs were generated. All18/36 composed rows retain those status labels and NaN missingness. The initialized checkpoint_t0 and original model-work/range-error evidence remain preserved by root; it is not a trained final checkpoint.

## Frozen scoring results and their limits

CPU scoring completed once(exit0), keeping all18 TRAIN and36 development denominators. TRAIN hold mean raw gray1−IoU is0.23258317101512865. Fitted mean is the registered missingness penalty1.0 across18/18 failures,0/18 wins; therefore the predeclared dynamic fit target is not demonstrated. The reported relative reduction−3.2995372177420084 is arithmetic on missingness penalties, not evidence that training made the model worse.

Frozen TRAIN all18 mean gray error0.9982032417957869 contains15 inference-status failures(1 invalid+14 unattempted) and1 additional empty-object/pusher failure. It is not a complete18-window measured baseline. The3 range-valid early raw outputs are shown without recoloring: B gray-maskIoU0.0032154341(one tiny gray pixel), C IoU0.0291262136(3 gray components,2 extra fragments), E empty gray/blue masks. The gallery visibly contains red objectlike output rather than the required gray appearance. These are pretraining observations from3 valid outputs, not fitted-model or generalization results.

All fitted TRAIN shape measurements are unknown18/18; fitted development unknown36/36. Frozen TRAIN legacy shape assessability is0/18,unknown18/18. Empty/range/missing failures and contact/clipping ambiguity remain explicit; unknown is not shape-pass. Raw components/holes/area/fragment fields and the old deformation_failure/assessable counts remain in score-report.json/per-window.csv.

Development retains the fixed36 windows and its31moving/4near-static/1intermediate,letter and actual-boundary groups. Hold references are measurable(mean gray error0.1960324594762658); frozen/fitted model quality, static drift and generalization are unmeasured because there are0 model outputs. No finite-subset comparison or development success claim is made.

## Complete evidence and stop scope

All18 TRAIN and36 development rows are retained in their frozen order. TRAIN has3 gallery pages and development6,each showing last/true/hold/frozen/fitted. Every pagewindow ID matches the frozen metadata and composed raw NPZ IDs. The invalid A RGB and missing rows use clearly labeled display placeholders; original numeric arrays are retained and never clipped. The first TRAIN anddevelopment pages were visually inspected.

All56 frozen source/config/sample files were hash-verified after scoring, and both composed raw NPZ hashes match their scoring inputs. Palette calibration, score thresholds and source are unchanged. Pure CPU scoring took6.2352516 observed wall seconds and is separate from the closed28.344-second model-work ledger. Scoring/diagnosis loaded no weights or model, made0 GPU/model calls,0 optimizer attempts and0 physical controls.

Conclusion: initialization and first-batch native generation are evidenced; strict range safety stopped the stage. Training, loss/gradient verification and motion fitting did not occur. The18-sample fitting objective and independent36-window prediction quality remain untested. This is not a failure after200 attempted fitting steps and does not disprove the GPC route. No retry or later stage is authorized by unused budget.

Evidence: score/score-report.json;score/per-window.csv;score/gallery-train/;score/gallery-development/;predictions/train18.npz;predictions/dev36.npz;native-output-range-diagnosis.json;model-work.json;model-work-error.json;score-command.log;score-execution.json.
