## Summary

Make the existing raw RGB T/L world-model predictor, manual viewer and supervised
fine-tuning path usable with explicit asset and source paths instead of local
experiment directories. A click predicts one control step and executes that same
control once, displaying actual and predicted results side by side.

## Changes

- Add model-only checkpoint and sidecar checks, plus the native three-Euler H1 predictor.
- Add the compact English T/L viewer with reset, matched input/action/prediction captures (S), and recording (V).
- Add whole-episode train-split windows, deterministic balanced T/L ordering, the existing phase-two one-step residual loss, and model-only exports. Training uses batch size 1 and AdamW with learning rate 1e-4; it defaults to a read-only plan.
- Keep a dedicated bounded Windows training worker; unsupported Linux training fails before execution.
- Document explicit asset paths and viewer/trainer commands in the README.

The diff contains only eight necessary code/README files. No tests, checker changes,
experiment reports, data, weights, media, logs, third-party source, or dependency
changes are included. The branch is
based directly on merged `main` and does not include PR #6 or #7.

## Validation

- Existing source-only suite: `python -X utf8 -S -B -m unittest discover -s tests -p 'test_*.py'`: 104 tests, 59 passed and 45 skipped for unavailable dependencies. Existing tests and CI are unchanged from main.
- Viewer/trainer `--help`, read-only `--check-assets` and training plan checks passed. Asset inspection only hashes the checkpoint and reads its sidecar; no tensor loading.
- The restored main `scripts/check_repo.py` reports three existing published JSON size/hash mismatches in this Windows CRLF checkout. Published evidence and the checker are unchanged from main.
- Commit/PR message checks and `git diff --check` passed.
- No real model, inference, training, CUDA, physics, collection or GUI run was performed for this packaging change.

## Limitations

The extracted Windows/CUDA integration and Linux GUI have not been exercised.
Actual training currently requires Windows; Linux training remains unavailable.
Prediction is H1 only and can still lose grey object parts where they overlap the
green goal. This is a code handoff, with no model-quality or policy-improvement
claim. Checkpoints, sidecars, datasets and pinned upstream sources must be supplied
separately.

## Related issues

None.
