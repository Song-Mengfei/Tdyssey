# Repository working rules

- Respect the owner's stated task, baseline, protocol, compute/storage budget and acceptance criteria. A correction narrows ongoing work immediately.
- Read README.md, docs/STATUS.md and the relevant protocol/profile before substantial work. Historical files are evidence, not permission to relaunch old jobs.
- Keep scientific formulas and settings separate from packaging changes. Never merge different controllers, splits or source versions into a claimed original baseline.
- Training, data collection, physics changes, deployment and publication require scope authorization. This repository preparation authorizes none of those next steps.
- Keep generated assets and local state out of Git. Preserve original/public source hashes and failed-result qualifications when updating archives.
- Use configurable paths. Do not add workstation usernames, credentials, chat IDs, scheduler state or callback records.
- Keep local smoke-test scripts, command receipts, resource inventories and runtime logs out of the delivery directory. A Git ignore rule alone is not a physical privacy cleanup. Run tools/check_upload_privacy.py for the actual tree as well as --scope git; any retained active-log exception must be explicitly owner-approved.
- Run python -B scripts/check_repo.py for source-only validation. Run appropriate existing runtime tests only when the environment, assets and compute scope support them.
- Do not choose a project license, author list or public asset-hosting destination on behalf of the rights holders.
