# Learning experiments

Standalone research scripts behind [the learning findings](../../docs/LEARNING_FINDINGS.md).
Run them from the repository root with the CUDA profile (`.venv/bin/python -B
scripts/learning/<script> --help`). They import each other as siblings and find the
repository root themselves (`REPO` overrides it). Outputs go wherever `--out` points;
keep them outside Git.

The shell scripts that ran each experiment, with their seeds and settings, are in
[`experiments/`](experiments/README.md).

## The method behind the reported policies

| Step | Script | What it does |
| --- | --- | --- |
| World model | `wm2.py` | Structured state world model: exact pusher and contact geometry, learned object motion, trained on machine-generated pushes |
| Teacher | `mpc_teacher.py` | Sampling planner inside `wm2`; reads exact simulator state (training only) and records its intended 16-step plans |
| Training data | `mpc_dagger.py` | Teacher demonstrations and correction rounds: the policy drives, the teacher labels every visited state |
| Camera frames | `image_wm_data/replay_frames.py` | Exact camera images for recorded episodes, made by replaying their physics |
| Camera policy | `bc_train.py`, `policy_builder.py` | Official training recipe on intended-plan labels with an equal share per shape; `policy_builder.py` builds the 192 px variant |

## Testing and reporting

| Script | What it does | Used for |
| --- | --- | --- |
| `base_rollouts.py` | Fresh-scene evaluation of the original policy or a trained one (`--student`); also rollouts for world-model data | every claim |
| `report_pair.py` | Paired comparison on the same scenes by shape group, with 95% intervals | final tests D051, D054, D056 |
| `select_candidates.py` | The fixed rule that picked the final-test candidates | D051 |
| `final_test_report.py` | Paired analysis of the earlier pre-registered tests | D040-D049 |
| `record_letters.py` | One video per letter, all on the same starting scene | figures |
| `diagnose_plateau.py` | Reads stored results to explain why the 26-letter run levelled off | D053 |
| `fast_replay.py` | Exact prefix replay that renders only the last two frames (7.6-11x faster, verified identical) | the simulator oracle |
| `ddim_eval.py` | The original policy with deterministic DDIM sampling | DDIM-10 keeps performance |

## Earlier approaches and diagnostics (mostly negative)

| Script | What it does | Outcome |
| --- | --- | --- |
| `score_snr.py` | How much of a forward-RL training score comes from the candidate's own plan | main reason RL did not learn |
| `degrade_report.py` | Retention weight, clipping and per-timestep drift of long forward-RL runs | diagnosis |
| `value_fn.py`, `q_validate.py` | Privileged value function and its candidate-ranking check | negative |
| `oracle_bon.py`, `oracle_summary.py` | Simulator best-of-N with the frozen original policy (upper bound) | small gains |
| `distill.py` | Imitation with the policy's denoising loss plus an anchor to the original policy | within noise on oracle data; first teacher distillation |
| `interpolate.py` | Weight blends of a student and the original policy (WiSE-FT style) | recovers T on validation; missed the final test on T |
| `wm_state.py` | Generic MLP state world model | too coarse |
| `teacher_bptt.py`, `teacher2.py` | Gradient-based (SuperMimic-style) teachers through the world models | negative; `mpc_teacher.py` reuses `teacher2.py`'s geometry helpers |
| `state_student.py` | Can a policy that reads exact state copy the teacher? | diagnosis |

Scene pools are listed in the findings document. Teachers and the oracle read
privileged simulator state; only distilled image policies are deployable results.
