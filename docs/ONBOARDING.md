# Coauthor onboarding

## Five-minute orientation

1. Read `README.md`, `docs/PROJECT.md`, and `docs/STATUS.md`.
2. The project asks whether a T-trained pushing policy can acquire useful behavior
   for new object shapes using non-expert data and imagined training.
3. First code exists for geometry, the GPC shape adapter, interaction collection,
   state/data records, and evaluation. Read `docs/IMPLEMENTATION.md`,
   `docs/COLLECTION.md`, and `docs/VALIDATION.md`. The CPU/GPC
   simulator checks pass; policy/model training and experiment results remain
   pending. A negative outcome is acceptable for the course if the work and
   analysis are substantial.
4. Run `python3 -B scripts/check_repo.py` using an existing Python 3.11+ installation.
5. Start Codex from the repository and have it summarize the active constraints,
   completed work, and next milestone before assigning a substantial task.

## Record before training

| Item | Current value |
| --- | --- |
| Course deadline / rubric | Not yet provided |
| Team | Three authors; assign names and branches together |
| Training machine / OS / GPU | Not yet selected |
| Data and artifact location | Choose a configurable path on that machine |
| Allowed disk / compute budget | Set before downloads or long jobs |
| Development profile | CPU/GPC for small simulator checks; CUDA/GPC for GPU work |

## Suggested work split

| Workstream | Deliverable | Shared interface to agree on |
| --- | --- | --- |
| Baseline and simulator | T reproduction, shape registry, contact/replay validation | Action units, reset state, observation/reward API |
| Data and world model | Collectors, splits, dynamics and reward validation | Trajectory schema, histories, transition timing |
| Policy and evaluation | Direct/imagined optimization, metrics, comparison report | Action chunks, likelihoods, budgets, initial-state sets |

For three authors, propose one person per workstream. These are not assigned owners
yet: agree on names and boundaries together. Schedule short reviews of how the
parts connect. Use one branch/checkout per author. Agree on file ownership before
simultaneous edits; pull and inspect shared status before changing a common contract.

Put shared setup and later changes through branches and pull requests so both
other authors can review them before they become the default on `main`.
Follow `docs/GIT.md` for structured commit titles, useful change/check information,
and the shared PR template. Each author may enable the commit editor template in
their own checkout; no package installation is needed.

## First implementation task

Run the selected profile's doctor, then the opt-in real simulator checks described
in `docs/ENVIRONMENT.md`. Keep reports and absolute machine paths in ignored
`artifacts/` directories. On the training machine, verify required assets one by one
and bound evaluation episodes before running the T policy baseline.
Record local patches and whether the checkpoint actually loads. Do not start a
full training run as an installation check.

Without a training machine, useful work includes specifying shape geometry and
symmetry rules, implementing small pure-Python data contracts, designing contact
coverage diagnostics, and preparing simulator adapters for later focused testing.
Clearly label unexecuted code and provisional interfaces.

## Handoff template

```text
Task / owner / branch:
Implemented behavior:
Verification command and result:
Data / model / code versions:
Artifact location (if any):
Failures or untested behavior:
Next concrete task:
```

For an actual experiment use `manifests/experiment.example.json` as the provenance
starting point. Fill actual values; do not treat the example as a completed run.
