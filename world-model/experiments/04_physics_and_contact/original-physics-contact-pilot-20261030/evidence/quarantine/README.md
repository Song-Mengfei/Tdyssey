# Modified-physics data exclusion

The human correction in `../STRICT_ORIGINAL_CORRECTION-20261031.txt` revokes the
v1-v6 collection protocol and its calibration/formal releases. The entire legacy
pilot root is now diagnostic only, including its original comparison baselines.
The files remain at their existing paths; no large bank was copied, relabelled,
re-rendered or deleted. `strict-original-exclusion-manifest.json` records actual
root counts, protocol evidence, frozen cost references and `training_eligible=false`.

Calibration-v1 and v2 each actually completed 18 attempts, 1152 controls and
230400 microsteps. Each ledger's 115.2 `actual_physics_seconds` is **simulated
integrated duration**, not CPU/GPU elapsed. Existing completions preserve wall
elapsed; prototype failures and their costs remain. No formal/supplement data
were present at the exclusion snapshot. None of these results releases the new
strict-original gate.

`../OLD_MODIFIED_PHYSICS_REVOKED.json` is checked before old coordinator validation
and the direct legacy collector's first work. Calling either now rejects the old
authorization. The exact previously frozen collector/common bytes were retained
under `revoked-source/`; old runtime configs and raw identities were not updated
to pretend they match the amended executable sources.

The future loader is `tdyssey.native_rgb_data_eligibility.StrictOriginalRGBDataset`.
It requires a separate strict-original membership manifest plus its externally
frozen SHA, verified actual original wrapper/collector/loader/pinned upstream
sources, raw and metadata SHA, the actual construction identity and adjacent legal
10Hz windows. Diagnostic roots, modified protocols, dynamic/.1-half-life data,
copied/renamed unregistered files, gaps, resets and invalid transitions are rejected.
Candidate audit uses `verify_strict_original_dataset(...,
require_training_authorization=False)` and `guard_trajectory(...)`; this does
**not** permit the training loader. Training construction additionally requires
a manifest recording explicit human approval. Current candidates remain pending.
Raw/manifest splits preserve `train_candidate` and `seen_validation` verbatim;
the approved future loader maps these explicitly to logical `train`/`seen_val`
partitions without changing their recorded provenance.

The existing old-protocol R3A/R2C loaders are preserved for historical report and
checkpoint reproducibility. They are not an admission route for the new
strict-original experiment, and no trainer was started or rewired in this task.
The new dataset interface returns only RGB/action model inputs; privileged
physics data remain admission/audit material.

`guard-check.json` and `eligibility-tests.log` contain actual saved-only checks;
all fixtures are synthetic, with zero physics steps, model calls or updates.
