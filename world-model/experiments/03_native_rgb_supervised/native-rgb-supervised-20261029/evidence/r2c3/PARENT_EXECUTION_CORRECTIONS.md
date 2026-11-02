# Confirmed C3 execution and engineering corrections — 2026-10-29

The supervising parent independently re-aggregated all288 C2 endpoint CSV rows
and accepted the same C2 result and its interpretation boundary. It confirmed
actual C3 execution directly under the human's latest instructions,without
waiting for another permission message. This note supersedes older proposal
prepare-only/wait,fixed-loss300 and stop-on-every-AMPskip provisions.

The entire learning curve1000/3000/6000 is one authorized stage. At1000 and3000,
save/evaluate and actively send a short progress plus metric paths,then continue
the fixed plan without pausing for permission. At6000 send full results and stop
for curve review before10000+. No new300 inference or fixed-loss.

Restore the actual300 model/optimizer/scaler/scheduler/RNG/data-order state.
Use lineage-successful updates,exclude the unrelated historical250. Replace all
oldstage300/global3000/stream4800 guards explicitly. Cosine through3000 then
hold eta_min1e-5 is accepted; no rebound or warmup reset.

Recoverable GradScaler overflow: lower the scale and continue. Every attempted
draw/time is recorded,but the skipped optimizer step adds no successful update
and does not advance the scheduler. Nonfinite loss/model parameters,or persistent
overflow preventing useful updates,are real numerical instability and stop with
safe checkpoint and evidence. The implementation uses a recorded16-consecutive-
overflow numerical guard,reset on a successful update; no elapsed-time cutoff.
No pretending a skipped update was completed. Old stop-on-any-skip language is
historical and does not enter the actual runner.

5700new complete/91200newdraw/96000lineage draw are expectations only if no skip.
The publicly versioned stream extends by the actual16 consumed draws for each
recorded skip, preserving RNG/queue progression and every480-draw80/20 cycle.
Do not rewind or impose96000 as a limit before6000 successful model updates.

One full last checkpoint must also be saved periodically at complete optimizer
boundaries during long segments,not only at1000/3000/6000. Save every100
successful updates plus registered observations; use atomic replace,not extra
full copies or extra evaluations. Keep1000/3000 lightweight model checkpoints,
final6000full and all prior weights. Stage400MiB/shared3GiB/atomic256MiB guards
remain. Do only necessary targeted restore/order/LR/AMP/RNG/reference checks,
then train; no large redundant audits delaying learning.

Same frozen data/80-20/network/native loss/batch and3/16window matrix; unknown
shape is not pass. Eight unseen letters sealed,no new physics/data,half-life
comparisons,servers or added motion/geometry modules. Time is statistics only.
