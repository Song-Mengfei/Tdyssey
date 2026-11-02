Tdyssey R3B original-protocol playground

The active environment executes the pinned PushTEnv.step itself:100Hz,dt0.01,
10 solver iterations, original PD kp100/kv20 at100Hz,10Hz actions/images,
KINEMATIC pusher,DYNAMIC object,Space.damping0. Original geometry,mass,inertia,
friction,walls and collision parameters are unchanged. No extra microsteps,
velocity projection,speed disk,wall budget,position repair or velocity clearing.
The original environment can still penetrate; this restoration is not a fix.

R3B lineage7499 remains trained on historical dynamic-pusher/.1s half-life
physics. Predictions remain cross-physics preview; no model adaptation or update.
H1 uses4 real frames;H4 feeds raw predictions only within its4-control batch.
Requested absolute targets are the actual actions to the original controller.

Recording and clip behavior remains bounded and diagnostic-only:64MiBWebM,
128MiBstreaming raw,32MiB/16batch ring; shared3GiB with256MiBatomic reserve.
Snapshot restore preserves public body/controller state and model histories,
but does not contain native solver caches/hidden position-bias velocities and
does not claim arbitrary mid-contact exact physics replay. Nonoriginal protocol
and incomplete controller metadata are rejected before reset/body writes.

Old modified-physics data remain diagnostic,training_eligible=false. The new
strict-original collection has an independent output root and eligibility gate.
Training,architecture,loss,denoising and server experiments remain unapproved.
