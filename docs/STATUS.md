# Shared project status

Updated: 2026-10-13. Stage: first code implementation; no training experiments completed.

Review branch: `chore/shared-project-setup`. Per the owner's request, the shared
setup and first code milestone are reviewed together in draft PR #1. Neither
change is merged into main.

## Completed

- Reviewed the original plan and public GPC/DiWA/DPPO resources.
- Recorded that this is an effort-graded course project; failure of ambitious
  performance goals is acceptable with substantive work and analysis.
- Team size is three authors. Shared changes use a separate branch and review;
  communicate in plain English and avoid unnecessary jargon.
- Added shared dependency specification, Codex guidance, onboarding, provenance
  records, example experiment configuration, and dependency-free repository checks.
- Added shared Conventional Commit rules, commit/PR templates, and a dependency-free
  PR-format check for consistent messages across the three authors and their agents.
- Implemented T/I/L geometry, coverage/IoU and symmetry-aware pose metrics, sustained
  success tracking, full body/controller records, trajectory JSON and whole-record
  splits, plus an adapter extending GPC through its existing shape-building hook.
- T body creation, physics, controller, images, and upstream reward/step results
  remain delegated to GPC. I/L construction attaches components to one body.
- Added a small geometry preview command and a later-machine GPC interface check;
  the latter does not download assets, load a policy, or train anything.
- Read-only laptop audit: RTX 4060 Laptop GPU, approximately 8 GB VRAM, 16 GB RAM,
  and approximately 119 GiB free at inspection time.
- Found CUDA-enabled PyTorch in another project's `realsense/.venv-ml`; a tiny GPU
  calculation passed there. That environment was not changed and is not Tdyssey's
  locked candidate environment.

## Verification status

- uv 0.13.0 resolved 86 package entries and passed an offline lock consistency
  check. Gym metadata was parsed from a checksum-verified source archive without
  executing it. Resolver inspection found zero temporary virtual environments,
  zero build-cache files, and no project `.venv`. Temporary tool/cache were removed.
- Added IPython to the GPC optional requirements after inspecting its upstream
  import. Updated resolution contains 101 entries and passed offline consistency;
  again no environment/package builds occurred and temporary tooling was removed.
- `python3 -B scripts/check_repo.py` passed configuration, lock-pin, provenance,
  Python syntax, and local documentation-link checks. `scripts/doctor.py --help`
  passed without importing third-party packages. Git whitespace checks passed.
- Six standard-library tests passed for valid/invalid message titles, commit body
  spacing, empty template sections, complete descriptions, and duplicate sections.
- All 32 tests passed: geometry and metrics, state/data integrity, message checks,
  and six construction/API tests using explicit adapter doubles. The doubles do not
  simulate physical motion. The CLI help and generated SVG XML also passed checks.
- No Tdyssey environment, CUDA profile, GPC checkpoint, original dataset schema,
  real simulator behavior, or policy/model training has been runtime-validated.
- Upstream revisions are recorded, but code has not been cloned into this repo.

## Next task

M0: on a user-authorized training machine, verify the documented profile, record
hardware and package versions, and reproduce a short T baseline at pinned GPC
revision. Compare the original and adapted T environment from the same states/actions,
then validate I/L collisions, rendering, reset penetration, and matched-state replay.
The interface smoke command is in `docs/IMPLEMENTATION.md`; it is not a policy
baseline. After these checks, implement non-expert collection and inspect WM
throughput before choosing visual route A or state route B. On this laptop,
continue source-only code work without claiming simulator/policy results.

## Open questions

- Course deadline, rubric details, and team members/task owners.
- Coauthor operating system, GPU, and where training/storage will live.
- Whether a newer CUDA/PyTorch profile is needed for the selected machine.
- Runtime compatibility of the candidate lock with GPC and actual checkpoints.

## Handoff convention

Replace stale next-task entries when work advances. Record the task owner/branch,
changed behavior, exact verification commands, artifact locations, and remaining
issues. Keep failed experiments visible. Commit this status with the related work;
private chat history is not a handoff.
