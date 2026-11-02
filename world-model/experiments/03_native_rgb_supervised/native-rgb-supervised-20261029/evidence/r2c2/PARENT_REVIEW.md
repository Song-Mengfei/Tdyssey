# R2C2 parent instruction and acceptance — 2026-10-29

Received from supervising chat “制定 Tdyssey 世界模型实施计划”, thread
__LOCAL_CHAT__. The directly human-authorized standing
callback in ../standing-callback-authorization.json remains effective.
Implement, execute, report, actually callback, save receipt, then STOP
waiting_parent. No polling and no automatic subsequent training.

## Human time-policy correction is still binding

The human never set a model-work time budget. record_only_v2 continues; elapsed
time is only measured and charged. Do not restore1200/900/300/5400-second hard
cutoffs, invent “remaining model time”, ask for extra time, shift clocks or pause
timing. Settings8 and16 define this experiment, not a time budget or human total
training limit. The prior300 updates were one diagnostic batch, not sufficient
training or completion of the overall objective. Storage, numerical/device,
identity, one-use and experiment-scope safeguards still apply.

## Accepted evidence and unresolved failures

The parent independently audited ../r2c1r1/parent-review/independent-audit.json:
72 raw NPZ files/288 frames/23 source trajectories,144 independent copy/trained
IoU+centroid rows with zero difference and correct checkpoint SHA. It viewed
val_motion/val_natural H4 second pages. Six-seen-letter motion-learning signal
is accepted (motion H1/H4 each12/12 beat copy-last), not shape/static fidelity:
nearstatic4/4 regressions,O/B rounding,C/X stroke changes,H4 shape12failures/
24unknown. Unknown never means pass. Three historical publication JSON CRLF/LF
hash differences are known; do not change old published files or claim all-green.

## Authorized R2C2 experiment, in order

1. New artifacts/native-rgb-supervised-20261029/r2c2; retain every older artifact,
   checkpoint and failed record. Freeze config and input/source identity before
   model execution. Only checkpoint ../r2c1r1/run/last.pt,105300718 bytes,SHA256
   3c397035857fe98b896ea5083b52f32c129f1787c1be6884d4c740a9e58e5be0,
   300 updates. Load trusted local state into the same original GPC,eval mode.
   No training loop,optimizer update or duplicate complete checkpoint. Compare
   model parameters after loading and after inference to establish no changes.
2. Sole variable DiffusionSamplerConfig.num_steps_denoising: reuse saved R1
   trained3 raw+scores; newly evaluate8 and16. sigma_min.002,sigma_max5,rho7,
   order1,s_churn0,s_cond0 unchanged. Preserve native precision,normalization,
   denoiser/sampler quantization and behaviour. No added postprocessing,shape
   projection,clipping repair,geometry/pose modules,heads or losses. Do not assume
   more denoising will solve rounding or drift.
3. Same frozen R2C0 36 windows,three groups of12,same order,batch4,paired per-batch/
   step seeds as R1. Reuse checked4RGB-history/4actions/four-step raw-feedback
   interface. Future truth/state/labels must never enter prediction. Each new
   setting produces144 float32 rawH1-H4 frames via36 sampler batch calls,72H1/H4
   endpoint scores; both settings total288 frames/72calls/144scores. Record
   actual denoiser-forward counts,durations and CUDA peaks,never fill expected
   counts as observed values.
4. Preserve all finite out-of-range raw/intermediate outputs and score failures
   under the existing protocol,without best-seed selection. Real nonfinite,
   structural,OOM/runtime exceptions preserve traces/state and stop model work
   with no automatic retries. Otherwise finish both settings and full report,
   irrespective of elapsed time.
5. Existing fixed metrics,cohorts separately:copy_last/trained3/trained8/trained16,
   H1/H4 gray IoU,gray ROI RGB MAE,contour,centroid error and drift fromLAST;
   pusher separately. val_natural8nonobvious and4strictnearstatic auxiliaries
   separately,per-window outcomes. Shape failure/unknown,holes/fragments;
   occluded unknown remains unknown. Evidence only six seen letters/current
   windows,natural/motion same trajectories overlap,not independent generalization.
6. Full readable truth/copy/3/8/16 pages for all36windows/H1+H4,objects large
   enough and labels separate. Do not regenerate red-T original-baseline gallery.
   Existing labelled display encoding only; raw scoring/feedback unchanged.
   Explain effects on corners,static changes,motion accuracy honestly,including
   no improvement. Do not automatically adjust training sampling or continue.
7. Minimal engineering supporting this variable; reuse existing contracts and
   targeted checks. Total experiment<=3GiB including256MiB atomic reserve;
   new stage persistent<=128MiB,no deletion,no duplicate weights;reference old
   files. Same3060Ti/batch4. No hard time budget. Cold identity/data/complete-result
   checks and necessary readable-page QA; no gratuitous old test repetition.
8. Actually callback concise result,time statistics,paths,per-cohort conclusions,
   failures/unknowns and evidence-based next-step suggestion. Parent arranges
   subsequent scope. No collection,physics rollout,half-life revalidation,heldout
   eight letters,server or automatic training. Full server training is always
   manually executed by the user.
