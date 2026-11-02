# Policy-free simulator action search

This is a separate privileged control experiment. It searches pusher targets
using the known D collision geometry, full physical state and original simulator.
It does not need a T-policy sample, a checkpoint, a learned world model or training.
The existing frozen-policy reranker and baseline remain separate.

The newer two-contact backend and its bounded development-state rescue checks
are documented in [CONTACT_SEQUENCE_SEARCH.md](CONTACT_SEQUENCE_SEARCH.md).
The original generic-search negative results below remain unchanged.

The user authorized this extension on 2026-10-21 after the frozen T policy failed
to propose useful contact routes. A hand-built diagnostic route improved overlap
from the recorded control-176 state, but was not a learned policy or an optimizer.
The new search varies and refines routes instead of replaying that fixed answer.

## Search protocol

Search over geometric approach, contact and push parameters. Diverse contact
sides allow translation and off-center rotation; continuous refinements vary
contact offset, direction, distance and timing. The optimizer evaluates effects
with actual simulator dynamics. Geometry supplies a proposal prior and does not
replace physical evaluation. Report that prior explicitly.

Every hypothetical rollout reconstructs contact history from the accepted initial
snapshot plus **all actually executed controls**. A fresh restore of a moving
current snapshot is insufficient because internal contact caches are absent.
The live simulator stays unchanged during search. Selected predictions must
match subsequent real controls within 1e-7; a mismatch aborts the run.

Forecasts include approach, pushing and physical braking. Hold targets use the
attained pusher position, not zero coordinates or `step(None)`. Each candidate
receives the same remaining-task-clipped budget. Native coverage greater than
0.95 remains absorbing success. Peak overlap alone is not completion.

Replan after a bounded prefix, with the unfinished previous plan retained as a
candidate so useful detours are not discarded before contact. Search has explicit
per-call population/generation/evaluation limits, a local random seed and an
overall wall-time cap. Count real controls, reconstruction work and hypothetical
forecasts separately, including discarded or interrupted work.

## Bounded Windows commands

Use the existing environment and pinned official source. No additional install,
model download or GPU is required. The first experiment resumes the recorded
problem state. The second starts at the original matched scene and never invokes
the T policy. A successful resumed episode does not establish success from reset.

```powershell
.\.venv\Scripts\python.exe -B scripts/check_repo.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_search.py' -v
$env:TDYSSEY_OFFICIAL_ROOT = 'third_party/diffusion_policy'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_simulator_search_runtime.py' -v
.\.venv\Scripts\python.exe -B scripts/evaluate_simulator_search.py --resume-report artifacts/td-baseline-300-20261021/report.json --resume-step 176 --steps 300 --seconds 900 --horizon 80 --output artifacts/d-search-resume
.\.venv\Scripts\python.exe -B scripts/evaluate_simulator_search.py --seed 100000 --search-seed 42 --steps 300 --seconds 900 --horizon 80 --video --output artifacts/d-search-initial
```

Use a new output directory each time. `--steps 300` is the **total episode**
budget, including controls imported by resume. Reconstruction costs are not new
live controls. `--display --video` shows execution and records wall-time planning
pauses; playback duration differs from 10 Hz simulated task time. Raw reports,
videos, optimizer traces and any prototype failures belong in ignored artifacts.

## Qualifying the D teacher

Use several initial states, not just a rescue from a favorable recorded contact.
Freeze the scene requests and search seeds before inspecting performance. Keep
all control-budget failures, wall-time truncations and protocol errors in the
denominator. A different random seed or contact side alone is not evidence of
multiple successful strategies: each route must actually reach native success,
and its executed contact sequence must differ meaningfully from other successes.

The 2026-10-21 development pilot uses scene requests 100000, 100100, 100200 and
100300 with search seeds 42 and 43. Each run has at most 300 actual controls,
600 wall seconds, 40 proposals per decision and an 80-control forecast budget
including four settle controls. At most eight actual controls execute before
replanning. V1 and the isolated V2 contact-search revision use identical initial
snapshots and limits; V2 changes several search mechanisms together, so their
comparison does not isolate any single mechanism. See the full outcomes and
teacher-readiness conclusion in [VALIDATION.md](VALIDATION.md).

Local frozen-source copies, exact commands and all reports are under
`artifacts/d-teacher-20261021/matrix-v1/` and `matrix-v2/`. Every report is
independently replayed from reset and checked against the recorded actual states,
rewards, contacts and selected predictions. `gallery.html` in that artifact root
links every case and its checked video. Playback repeats each actual control for
0.3 seconds with no interpolation; measured search latency is recorded separately.
These artifacts are diagnostic evidence, not a student training dataset, and
remain out of Git.

## Relationship to later policy fine-tuning

Action search changes targets, not network weights. The Pymunk controller and
contact solver are not a PyTorch computation graph, so their reward cannot be
backpropagated directly into the diffusion network through this implementation.

After search reliably succeeds, a separate experiment can distill the successful
controller into the T policy. Render actual observation histories, use actual
executed action sequences as teacher targets, retain the original normalizer and
timing protocol, and train with the official diffusion denoising `compute_loss`.
Do not label every proposed/optimistic candidate as a successful teacher action.
This would be D-specific supervision and policy adaptation, not zero-shot
generalization. Reward-based policy optimization is another, more complex route.

Preserve the original checkpoint and T evaluations, split scenes by whole
episodes before optimization/training, and reserve independent D evaluation
scenes. The already-inspected scenes are development examples. Reusing them as
training data removes their status as independent evaluations. A small update or
training-loss reduction alone is not policy improvement; rerun the simulator and
report T retention as well as D performance. No fine-tuning is part of this
untrained search implementation.
