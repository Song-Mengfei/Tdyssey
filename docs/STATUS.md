# Shared project status

Updated: 2026-10-13. Stage: source-only scaffold; no experiments completed.

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
- `python3 -B scripts/check_repo.py` passed configuration, lock-pin, provenance,
  Python syntax, and local documentation-link checks. `scripts/doctor.py --help`
  passed without importing third-party packages. Git whitespace checks passed.
- Six standard-library tests passed for valid/invalid message titles, commit body
  spacing, empty template sections, complete descriptions, and duplicate sections.
- No Tdyssey environment, CUDA profile, GPC checkpoint, dataset schema, simulator,
  training loop, or task-performance result has been runtime-validated.
- Upstream revisions are recorded, but code has not been cloned into this repo.

## Next task

M0: on a user-authorized training machine, verify the documented profile, record
hardware and package versions, and reproduce a short T baseline at pinned GPC
revision. Then measure world-model throughput and choose visual route A or state
route B. On this laptop, continue source-only implementation/design work.

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
