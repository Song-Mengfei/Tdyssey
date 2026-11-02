# CONTACT-PILOT1 release, received 2026-10-30

The planning chat__LOCAL_CHAT__ independently read all912
original NPZs and verified the contact audit. Its independent-audit.json is in
../contact-audit-20261030/parent-review/. It releases this controller mechanism
pilot only. R3B remains paused. No model/checkpoint/optimizer/server work and no
large recollection are authorized.

Preserve the R3A library,manifest,selected pool,original controller and reports.
Implement a new controller version using current/past recorded contact,impulse,
pose,velocity,wall and recent motion only. Fix sustained contact following,
progress-aware stable recovery routing and nearstatic wall-contact escape.
Keep the physical protocol unchanged (0.1s half-life,10Hz/100Hz,forcePD,caps,
mass/inertia,walls,2world-pixel penetration guard). No teleport,midtrajectory
reset,future physics lookahead,lowered motion thresholds or renderer changes.
Keep the withdrawal and recontact tasks; do not shorten failures to improve rates.

Freeze48cases before physics,sixletters x8. A24 reuses initial TRAIN scenes and
initial contact targets:two slow/unrecovered recovery,one wallnearstatic andone
weak exterior/boundary rotation per letter. Use original saved48control baseline
without rerunning it. Label matched_existing_TRAIN_family,not fresh independent
support. B24 uses entirely new TRAIN families:two recovery,one boundary rotation
andone weak exterior case per letter; prioritize X negative boundary rotation and
missing arcs. Freeze seeds/scenes/yaws/regions/targets; check exact/near duplicates
against existing TRAIN/validation and within pilot. Do not select from validation
or heldout outcomes. A/B results must be separate and selection bias explicit.

Every case has one attempt,at most48controls. Total cap2,304charged controls,
including failed/partial/unknown and unstarted reservations. No replacement or
repeated success. Retain failures; stop safely on engineering/unknown state and
use the saved ledger to decide whether any unstarted cases can proceed. No
automatic physics replay. A failure deficit remains a deficit.

Use the unchanged audit definitions and six exclusive outcomes. Report case and
letter paired deltas for A,full independent distributions for B,failures/partial
costs,late-half performance,continuous/impulse contact,wallstalls and exterior arc
coverage. Predeclared pilot targets:contact+obvious>=40% all actual pilot controls,
movingpusher/noAO<=25%,observed recontact median<=.5s/P90<=1s,unrecovered<=5%.
Keep right-censoring denominators explicit. These are small biased mechanism
targets,not population guarantees or retroactive R3A gates. Do not tune in a loop
after observing pilot results.

New stage persistent increment<=16MiB,shared<=3GiB with256MiBatomic reserve.
Use compact saved numeric states/contacts/actions; few preselected paired media,
no RGB library or dependencies. Report actual physics costs and all failures.
Send controller changes,results,paired media,storage and promotion recommendation
to the planning chat under the standing human callback authorization,save the
actual receipt,update the memo,and STOP waiting_parent. Do not expand collection,
mix the pilot into the training pool,or resume R3B automatically.
