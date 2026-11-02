# Human correction about training duration and R2C3 preparation

Received via parent chat__LOCAL_CHAT__ on2026-10-29.
Direct human words: “300步是不是太少了？你让他查清楚棱角磨圆到底是训练不够，还是去噪采样的问题，别只训几百步就下结论。”

300 updates prove a working training chain and early learning only. They do not
establish convergence,capacity limits,hopeless shape preservation or uselessness
of more training. No human model-time limit; keep record_only_v2. The objective
stays native GPC image+action to image,without motion modules or geometric losses.

Continue R2C2 without redoing completed work. Its scope is sampler effects at
the fixed300-update checkpoint; it cannot alone decide whether training is
insufficient. Complete the actual callback,then wait for parent direction.

With this callback prepare,do not execute,R2C3 config and byte budget:
- Restore the actual300-update full R1 checkpoint including optimizer/scaler/
  scheduler/RNG/PreparedInputs order. Do not reinitialize. Count this lineage's
  updates,excluding250 from a different old experiment.
- Observe cumulative1000,3000,6000 complete updates,not another300-step stopping
  batch and not human total allowances. Parent assesses whether10000+ is needed
  if learning still improves. No convergence/failure claim merely on reaching6000.
- Same six seen letters,41train trajectories,80/20 sampling,split,batch16/micro4,
  native loss,optimizer,network,color,resolution and saved data. One-variable
  reasoning requires no simultaneous sampling/data/model change.
- Restore cosineT_max3000,eta_min1e-5 to3000; thereafter hold eta_min,never allow
  unbounded CosineAnnealingLR rebound or reset warmup. State exactlr and step
  semantics in the proposal. Replace any global3000-attempt proposal guard with
  explicit stage scope; it is not a human lifetime limit.
- At each training observation evaluate fixed native3 and16 on the same36/batch4/
  pairedseed/four-step raw-feedback matrix. Reuse R2C2's300-point3/16. Eight is
  an intermediate reference only. No choosing a prettier matrix afterward.
- Fixed-window loss/update/AMPskip/exposure/lr logging; avoid frequent complete
  evaluation. Preserve per-group/per-window curves of raw grayIoU,contour/shape,
  RGB,centroid/staticdrift,corner/hole images,failures/unknown and overfitting.
- Horizontal sampler gains support a sampler factor; longitudinal gains at fixed
  sampler support insufficient training at300; both can matter. Neither improving
  only limits this data/training scheme; do not assert a unique cause. Small or
  ambiguous differences mean insufficient evidence and a minimal further check.
- Total3GiB with256MiB atomic reserve,all previously delivered weights/data kept.
  No original-data copies/downloads/environment rebuild. Candidate one full
  resumable last checkpoint atomically updated,1000/3000 model-only milestones+
  progress/identity,6000 full point. Lossless raw prediction storage/reference
  galleries. Show exact planned bytes and actual headroom; report shortage rather
  than deleting old artifacts or shortening training.
- Eight unseen letters stay sealed,0newphysics/no half-life comparisons/no servers.
  Full server training is always manually executed by the user. Parent acceptance
  of R2C2 precedes R2C3 execution; ordinary continuation needs no repeated human
  permission request once the parent supplies its stage instruction.
