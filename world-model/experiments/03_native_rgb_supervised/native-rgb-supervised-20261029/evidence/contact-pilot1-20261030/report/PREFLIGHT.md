# CONTACT-PILOT1 reporting rules, frozen before execution

Use CONTACT_AUDIT motion thresholds and actual post_solve agent-object flags.
Primary impulse evidence is finite valid impulse norm >1e-8, counted at most
once per physical substep. Never sum duplicated point/event impulses.

A_matched and B_fresh are separate populations. Existing A baseline trajectories
are read, never rerun. Per-case and per-letter old/new changes are mandatory;
B_fresh has no paired baseline and is the only new independent-family support.
The fixed 48-control cases retain approach, withdrawal and recontact tasks.

Recovery timing starts at the case's frozen task_recontact_start_control (26 for
all nonstatic tasks), independent of dynamic controller-state labels. Report
first observed contact interval start, observed-only delay quantiles, right
censoring duration, failures before the task and withdrawal with no-contact
evidence. Contact at task start without observed release is disclosed separately.

Six mutually exclusive contact-by-motion outcomes use complete actual control
intervals. Partial/unknown physical time and unstarted charged reservations are
additional categories, never promoted to full motion-labelled controls. Report
the six counts against both complete-control and charged-control denominators,
with unclassified charged controls making up the charged total. Engineering
contact+obvious uses its counted numerator / all charged controls; empty-travel
reports measured complete-control rate and a pessimistic upper rate counting
unclassified charges as empty. Failures may not improve the gate by truncation.

Frozen engineering targets: contact+obvious >=40%; moving-pusher/noAO <=25%;
observed recovery median <=0.5s and P90 <=1s; unrecovered <=5%. Targets apply to
this small mechanism pilot, not retrospectively to R3A. Always show A/B, letters,
failure rates, late-half decline, sustained contact/valid impulse and exterior
arcs. Passing these means neither population-wide guarantees nor model quality.

Predeclared paired media: CP1-A-T-01 (first T slow/unrecovered recovery role) and
CP1-A-X-04 (first X weak-boundary role), chosen from the frozen case plan before
new physical results. Show full same-scene old/new 10Hz saved-state sequences,
up to the actual failure endpoint; original RGB hashes must match before labels.
No result-based replacement. Include all-case lightweight timelines and all
failure metadata. Reuse old clips when available. If media would exceed2MiB,
use pair strips and timelines first; retain the fixed selection and disclose
omitted videos. Annotations remain outside RGB and never enter training inputs.

All outputs are saved-array CPU work only. Renderer setup may create a display
environment but must never step physics. No models, checkpoints, training draws,
new dependencies, edits to old data/media or automatic follow-up collection.
