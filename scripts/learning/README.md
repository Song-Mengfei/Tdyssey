# Learning experiments

Standalone research scripts behind [the learning findings](../../docs/LEARNING_FINDINGS.md).
Run them from the repository root with the CUDA profile (`.venv/bin/python -B
scripts/learning/<script> --help`). They import each other as siblings and find the
repository root themselves (`REPO` overrides it). Outputs go wherever `--out` points;
keep them outside Git.

| Script | Purpose | Status |
| --- | --- | --- |
| `base_rollouts.py` | Frozen-base (or `--student`) evaluation on fixed scene pools; state/action rollouts for world-model data | used for every claim |
| `fast_replay.py` | Exact prefix replay that renders only the last 2 frames (7.6-11x faster, verified identical) | used by the oracle |
| `ddim_eval.py` | Base performance under deterministic DDIM sampling | DDIM-10 keeps performance |
| `score_snr.py` | How much of a training score comes from the candidate's own plan | main diagnosis |
| `degrade_report.py` | Retention weight, clipping and per-timestep drift of long forward runs | diagnosis |
| `value_fn.py`, `q_validate.py` | Privileged value function and its candidate-ranking check | negative |
| `oracle_bon.py`, `oracle_summary.py` | Simulator-oracle best-of-N with the frozen base (upper bound) | small gains |
| `distill.py` | Imitation with the policy's denoising loss plus an anchor to the base | oracle data: within noise |
| `wm_state.py` | Generic MLP state world model | too coarse |
| `wm2.py` | Structured state world model: exact pusher, exact contact geometry, learned object motion | used by the planner |
| `teacher_bptt.py`, `teacher2.py` | SuperMimic-style BPTT teachers through the world models | negative |
| `mpc_teacher.py` | Sampling planner inside `wm2`, evaluated in the real simulator | D 50/50, Y 36/50, L 21/50 validation scenes |
| `mpc_dagger.py` | Planner demonstrations in `distill.py` format, chunk labels, DAgger, self-label retention data | used for rounds 1-3 and DAgger |
| `interpolate.py` | Blend a student's weights with the base (WiSE-FT style) | recovers T on validation |

Scene pools are listed in the findings document. Teachers and the oracle read
privileged simulator state; only distilled image policies are deployable results.
