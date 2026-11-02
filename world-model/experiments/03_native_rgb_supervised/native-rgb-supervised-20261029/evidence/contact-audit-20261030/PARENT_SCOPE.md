# Contact and effective-motion audit: frozen scope

Received through the supervising planning chat on 2026-10-30. The human asked:

> 那你让他统计一下整个训练集真正接触并推动物体的比例，有多少时间是在空跑和找接触点，别又采了一堆基本不动的数据，光改标注没用。

This supersedes the earlier R3B first-epoch release. R3B is paused before model
loading, Torch/GPU execution, forwards or optimizer updates. CPU drafts are
preserved. This stage reads recorded arrays only: no new collection, physics
replay, checkpoint loading or model evaluation. Stage completion requires an
actual callback to the planning chat, its saved receipt, then waiting_parent.
Training or collection must not resume automatically.

## Distinct denominators

* All successful TRAIN acquisition: 768 trajectories, 36,864 controls, 368,640
  substeps and 3,686.4 simulated seconds. Include the first three controls,
  approach, withdrawal, seeking and stalls.
* Legal TRAIN H1: 34,560 targets at controls 3..47. Only four-frame history
  excludes controls 0,1,2 (2,304 total).
* Selected TRAIN: 23,984 distinct target controls, each drawn once in the frozen
  first epoch of 1,499 batches of 16. The aligned four-action input prefixes
  produce 95,936 repeated control exposures; those are not independent physical
  time. No actual new model training draws have occurred.
* Seen validation: 144 trajectories, 6,912 controls, 69,120 substeps, 691.2s and
  6,480 legal H1 targets. Keep these out of TRAIN denominators.
* Six failed attempts: retain separately all partial physical time, contact,
  motion and charged controls, including the unstarted reservation.

## Actual contact and motion

Use recorded agent-object post_solve bit 2 of contact_flags[:,0]. Report 100Hz
contact duty/seconds, contact runs, 10Hz any-contact, terminal contact and the
0 / 1-4 / 5-9 / 10 contacting-substep bins separately. Object-wall and agent-wall
are separate measurements. Planned phase and controller mode are not contact.

Per-substep impulse presence requires a valid post_solve AO impulse vector with
norm above epsilon. Primary epsilon is 1e-8; sensitivity is [0,1e-8,1e-6,1e-4].
CompactAttempt copies total event impulse into each point row and stores no
unique arbiter/event ID. Never sum those rows into physical impulse or force.
Flagged contact without a valid impulse observation is unknown, not zero.

Per-control translation uses the transformed area centroid, not body origin.
Rotation is unwrapped signed angular displacement, without symmetry folding.
Nearstatic means translation <0.1 image96px AND abs rotation <0.1 degree.
Obvious motion means translation >=2 image96px OR abs rotation >=5 degrees.
Small motion is the remainder. Sensitivity >=0.5px OR >=1degree does not replace
the main thresholds. Cross any actual contact with the three motion classes
into six mutually exclusive outcomes summing exactly to each denominator.
Contact and motion co-occurrence is not a causal estimate of pusher contribution.

## Required breakdowns and evidence

Report contact+nearstatic with/without wall; planned push without contact;
free_route/hold within planned push; no-contact nearstatic and residual motion.
Keep phase/mode/region/walls as strata rather than overlapping categories added
to 100%. Show time spent on approach, withdrawal and recontact, signed/absolute
motion distributions, zero fractions and impulse sensitivity.

Stratify TRAIN and validation separately by T/L/O/C/X/B, initial region/space,
scenario, first/last 24 controls and control index. Emphasize X and boundaries.
Measure first actual contact after planned recontact starts; use the start of
the contacting physics interval, show observed delay median/P90/max and the
right-censored no-recovery share. Static cases without recontact are N/A.

For selected targets and frozen batches, count real contact+obvious support,
independent trajectories, nearstatic/small support, top-trajectory and actual
arc concentration. Report batch contact/push P10/median/P90/worst and repeated
four-action prefix exposure weights. Explain how much of the 50% motion target
pool is no-contact residual movement.

Choose fixed-seed random and measured examples: no-contact/nearstatic, slow
recovery, wall stall, translation and both rotation signs. Reuse original media
where possible; new timelines/annotations must not alter original RGB, PNGs or
MP4s. State frame j>0 follows control j-1, with terminal flag at substep10*j-1;
t0 has no preceding interval. B014 must expose its long recovery, rather than
using its exhibition role 'translation' to imply continuous pushing.

Deliver one clear findings page, machine-readable tables and a targeted sampling
proposal. Do not quietly lower thresholds, change physics or repeat a few good
windows. If coverage is inadequate, recommend revisiting the data release or a
targeted supplement; neither new collection nor training is authorized here.

## Accounting

New audit increment cap: 16 MiB; shared cap: 3 GiB; atomic reserve: 256 MiB.
The audit cap is separate from the old R3A 96 MiB increment and R3B draft cap.
All new presentation copies and source files count. The standing callback
authorization is ../standing-callback-authorization.json. Actual callback target:
__LOCAL_CHAT__ on local.
