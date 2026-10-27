# Policy learning experiment log, 2026-10-24 to 2026-10-27

This is the complete record of every policy-learning experiment run since the
training setup began, including failed and abandoned ones. It is meant as source
material: [LEARNING_FINDINGS.md](LEARNING_FINDINGS.md) is the curated summary, and
material for presentations or reports can be selected from either. Tables in
sections 1-9 were generated directly from saved outputs (run logs, result JSON files
and the private run store's per-lineage summaries), not typed by hand. Raw runs,
checkpoints and data stay outside Git in the author's private store.

## How the work ran

- **Machines.** 2026-10-24: two 3-hour NUS SLURM sessions on one H200 NVL (141 GB).
  From 2026-10-25: a 3-day job on one H100 NVL (94 GB), still running when this was
  written. All environments, caches and runs lived in node-local `/tmp`, which is wiped
  when a job ends, because the home quota was nearly full.
- **Infrastructure kept outside Git** (tied to the author's account and cluster):
  SLURM batch scripts that rebuild the environment on a fresh node, a private Hugging
  Face dataset repository used as the run store (pushed every 10 minutes and at job
  end), lineage claims so several sessions can resume different runs, a queue runner
  that keeps a fixed number of GPU jobs running, CUDA MPS to share the GPU between
  processes, and read-only progress viewers that read the store from the login node.
  The repository-side pieces (batched rollouts D033, presets, portable runs) are in Git.
- **Throughput.** Up to 12 concurrent jobs per GPU under MPS. Eight concurrent
  smokes ran 1.4x faster in aggregate than one, 2.4x with MPS; batched lock-step
  rollouts gave about 4x per lineage (details in `STATUS.md`, D033).
- **Idle GPU time.** The GPU sat idle twice because the queue ran out while no one was
  working: about 12 hours overnight (00:30-12:48 on 2026-10-26) and about 2.5 hours
  (16:16-18:47 on 2026-10-26), plus shorter gaps. An idle guard now starts more planner
  data collection whenever the GPU has been empty for 3 minutes. Overnight on
  2026-10-26/27 a chained job never started (its wait matched its own launcher), so
  the guard's data collection ran for about 12 hours instead of DAgger iterations 3-4.
- **Scene pools** (disjoint seed ranges, `base + 20*i`): development 100000,
  validation 400000, monitor test 300000, value/world-model data 600000, oracle data
  700000, teacher data 800000, final test 900000 (1000 scenes, reserved).

## Timeline

| When | What | Outcome |
| --- | --- | --- |
| 10-24 | Setup on SLURM `/tmp`, private run store, PR #5 trainers | doctor, preflight and smokes passed |
| 10-24 | 8 x 20k-control T/L pilots (NFT/AdvantageFlow, LoRA/dense) | no measurable change (section 1) |
| 10-24 | Batched lock-step rollouts, MPS, near-miss presets (D033, D034) | about 4x faster per lineage |
| 10-24 | 6 T/L and 2 near-miss lineages, about 1M controls each | policy barely moved (lr 1e-5) |
| 10-25 | Step-size sweep, then long LoRA and dense runs | no learning; dense runs degraded T |
| 10-25 | Score reliability, value function, A/A noise, DDIM checks | signal mostly noise (D035, D036) |
| 10-25 | Literature review (`research/`) | recommended an oracle and distillation, then critics/steering |
| 10-25 | Oracle best-of-N rounds and DAgger rounds | small gains; imitation within noise (D037) |
| 10-25 | 1000-scene final-test base evaluation | T 44.4%, near-T 41.9%, Y 9.4%, D 0.3%, L 0.1% |
| 10-25/26 | Generic and structured state world models | structured model about 0.2 px over 8 controls (D038) |
| 10-26 | BPTT teachers (SuperMimic-style) | failed to push |
| 10-26 | Sampling planner, step-by-step fixes | D 50/50, Y 36/50, L 21/50 validation scenes (D039) |
| 10-26 | Planner distillation rounds 1-2 | students worse than the base |
| 10-26 | Round 3-4: train the image encoder too, 60 epochs | first D/L successes from images; T 10-15 points low |
| 10-26/27 | DAgger iterations 1-2 (iterations 3-4 lost to a chaining bug) | Y up to 16/100, L up to 6/100; T still low |
| 10-26/27 night | Idle guard: 70 clean rounds of planner data; 7 rounds quarantined (final-test seeds) | 31,500 clean episodes |
| 10-27 | Weight blending of students with the base (WiSE-FT style) | T recovered on validation |
| 10-27 | Pre-registered 1000-scene final test of one blend (D040) | Y +2.9, D +4.6 (significant), T -4.8: not passed |
| 10-27 | Part-swap diagnosis and encoder-heavy blends | T loss comes from the student's action network |
| 10-27 | Secondary final test of an encoder blend | Y +8.7, D +1.1 (significant), T -2.5 [-6.2, +1.2]: not passed |
| 10-27 | D041 confirmatory test of the 70% encoder blend, 2000 fresh scenes | Y +8.0 confirmed; T -3.1 [-5.9, -0.4]: not passed |
| 10-27 | Diffusion-policy samples as planner candidates; LoRA students; encoder-only students | no improvement over existing results |
| 10-27 | D042 test of 30% and 50% encoder blends, 2000 fresh scenes, 97.5% intervals | both passed; 50%: Y 8.6% -> 14.8%, D 0.3% -> 1.1% |
| 10-27 | Narrated explainer video (29 min) and a 3:41 milestone summary (outside Git) | done |
| 10-28 | D043 replication of the 50% encoder blend on a third fresh pool | passed again: Y 9.8% -> 14.1%, D 0.1% -> 0.9% |

Notes on specific rows below: the value-function D row averages the 3 of 6 contexts
in which candidates differed (in the others every candidate failed). Planner run
`diagW` replaced every candidate with push candidates, including the warm start, so
the plan changed every control; it shows that plan consistency matters, not that two
iterations are too few. Runs with only 2-12 scenes were diagnostics, not estimates.
`extended-s1` was the sequential version of the 1M-control T/L runs; it recorded no
update before the batched group replaced it, and its failure cause was not kept.
The value function's R^2 of 0.64 on held-out scenes comes from its training output,
which is not kept on the current node.

## 1. 2026-10-24, H200 sessions: forward-process RL pilots and 1M-control runs

Source: per-lineage summaries in the private run store. Each evaluation is success rate / mean final coverage on
3-5 development scenes per shape (100000 pool), so single changes of 0.2-0.4 are within noise.

| Group | Lineage | Updates | Training controls | Update 0 (success/coverage) | Last update |
| --- | --- | --- | --- | --- | --- |
| pilot-s1 | advantage_flow-action_network-seed42 | 8 | 19,971 | T 0.67/0.80 L 0.00/0.24 | 8: T 0.33/0.69 L 0.00/0.38 |
| pilot-s1 | advantage_flow-lora-seed42 | 8 | 19,877 | T 0.67/0.80 L 0.00/0.24 | 8: T 1.00/0.95 L 0.00/0.24 |
| pilot-s1 | advantage_flow-lora-seed43 | 8 | 19,385 | T 0.67/0.93 L 0.00/0.33 | 8: T 0.67/0.81 L 0.00/0.28 |
| pilot-s1 | advantage_flow-lora-seed44 | 8 | 19,815 | T 0.33/0.62 L 0.00/0.26 | 8: T 0.67/0.80 L 0.00/0.35 |
| pilot-s1 | nft-action_network-seed42 | 8 | 19,680 | T 0.67/0.80 L 0.00/0.24 | 8: T 0.67/0.77 L 0.00/0.25 |
| pilot-s1 | nft-lora-seed42 | 8 | 19,863 | T 0.67/0.80 L 0.00/0.24 | 8: T 0.67/0.92 L 0.00/0.24 |
| pilot-s1 | nft-lora-seed43 | 8 | 19,952 | T 0.67/0.93 L 0.00/0.33 | 8: T 0.67/0.86 L 0.00/0.22 |
| pilot-s1 | nft-lora-seed44 | 8 | 19,874 | T 0.33/0.62 L 0.00/0.26 | 8: T 0.67/0.78 L 0.00/0.39 |
| extended-s1 | advantage_flow-lora-seed42 | 0 | 0 | failed |  |
| extended-s1 | nft-lora-seed42 | 0 | 0 | failed |  |
| extended-batched-s1 | advantage_flow-lora-seed42 | 157 | 999,063 | T 0.40/0.74 L 0.00/0.28 | 157: T 0.60/0.69 L 0.00/0.26 |
| extended-batched-s1 | advantage_flow-lora-seed43 | 163 | 999,521 | T 0.40/0.72 L 0.00/0.34 | 163: T 0.80/0.90 L 0.00/0.36 |
| extended-batched-s1 | advantage_flow-lora-seed44 | 158 | 999,972 | T 0.80/0.85 L 0.00/0.38 | 158: T 0.40/0.88 L 0.00/0.38 |
| extended-batched-s1 | nft-lora-seed42 | 157 | 999,849 | T 0.40/0.74 L 0.00/0.28 | 157: T 0.40/0.59 L 0.00/0.30 |
| extended-batched-s1 | nft-lora-seed43 | 159 | 999,689 | T 0.40/0.72 L 0.00/0.34 | 159: T 0.80/0.88 L 0.00/0.36 |
| extended-batched-s1 | nft-lora-seed44 | 158 | 999,493 | T 0.80/0.85 L 0.00/0.38 | 158: T 0.60/0.80 L 0.00/0.36 |
| nearmiss-batched-s1 | advantage_flow-lora-seed42 | 157 | 999,786 | T 0.40/0.72 T_narrow 0.80/0.89 Y 0.00/0.67 D 0.00/0.34 L 0.00/0.32 | 157: T 0.80/0.81 T_narrow 0.40/0.83 Y 0.20/0.65 D 0.00/0.40 L 0.00/0.33 |
| nearmiss-batched-s1 | nft-lora-seed42 | 158 | 999,760 | T 0.40/0.72 T_narrow 0.80/0.89 Y 0.00/0.67 D 0.00/0.34 L 0.00/0.32 | 158: T 0.60/0.82 T_narrow 0.40/0.80 Y 0.20/0.68 D 0.00/0.29 L 0.00/0.29 |

## 2. 2026-10-25, H100: step-size sweep and long runs (stopped)

Step-size sweep (`lrsweep-s2`, LoRA unless named dense, about 120k controls each). Drift = fixed-probe change of
the T denoiser output relative to the denoiser's own error; values far below 1 mean the policy barely moved.

| Run | Updates | Controls | T drift ratio | Evaluation (success/coverage) |
| --- | --- | --- | --- | --- |
| advantage_flow-lr1e-3-seed42 | 20 | 123,565 | 2.9e-02 | 10: T 0.50/0.71 T_narrow 0.35/0.68 Y 0.20/0.61 D 0.00/0.29 L 0.00/0.34 |
| advantage_flow-lr1e-4-seed42 | 18 | 111,168 | 1.1e-04 | 10: T 0.60/0.77 T_narrow 0.55/0.76 Y 0.05/0.49 D 0.00/0.36 L 0.00/0.34 |
| advantage_flow-lr3e-4-seed42 | 20 | 122,499 | 3.0e-03 | 10: T 0.50/0.76 T_narrow 0.45/0.68 Y 0.35/0.72 D 0.00/0.28 L 0.00/0.33 |
| nft-lr1e-3-seed42 | 20 | 122,771 | 1.5e-01 | 10: T 0.55/0.80 T_narrow 0.45/0.74 Y 0.05/0.60 D 0.00/0.30 L 0.00/0.32 |
| nft-lr1e-4-seed42 | 20 | 122,432 | 1.8e-04 | 10: T 0.60/0.81 T_narrow 0.35/0.76 Y 0.05/0.60 D 0.00/0.31 L 0.00/0.32 |
| nft-lr1e-5-seed42 | 18 | 111,857 | 2.9e-06 | 10: T 0.60/0.76 T_narrow 0.45/0.75 Y 0.05/0.60 D 0.00/0.28 L 0.00/0.33 |
| nft-lr3e-4-seed42 | 20 | 122,804 | 1.4e-02 | 10: T 0.60/0.74 T_narrow 0.50/0.75 Y 0.10/0.61 D 0.00/0.33 L 0.00/0.29 |
| nft-lr3e-5-seed42 | 19 | 116,884 | 1.0e-05 | 10: T 0.55/0.82 T_narrow 0.40/0.73 Y 0.00/0.63 D 0.00/0.28 L 0.00/0.31 |

Long runs (`long-s2`, up to about 1M controls each):

| Run | Updates | Controls | T drift ratio | Update 0 | Last evaluation |
| --- | --- | --- | --- | --- | --- |
| advantage_flow-dense-lr1e-4-seed42 | 27 | 168,449 | 6.5e-01 | T 0.55/0.78 T_narrow 0.55/0.70 Y 0.05/0.55 D 0.00/0.27 L 0.00/0.34 | 25: T 0.15/0.55 T_narrow 0.15/0.59 Y 0.00/0.40 D 0.00/0.20 L 0.00/0.21 |
| advantage_flow-dense-lr1e-5-seed42 | 159 | 1,030,243 | 3.2e-01 | T 0.55/0.78 T_narrow 0.55/0.70 Y 0.05/0.55 D 0.00/0.27 L 0.00/0.34 | 150: T 0.30/0.72 T_narrow 0.10/0.64 Y 0.00/0.52 D 0.00/0.33 L 0.00/0.37 |
| advantage_flow-dense-lr3e-5-seed42 | 162 | 1,044,923 | 1.9e+01 | T 0.55/0.78 T_narrow 0.55/0.70 Y 0.05/0.55 D 0.00/0.27 L 0.00/0.34 | 150: T 0.00/0.31 T_narrow 0.00/0.28 Y 0.00/0.17 D 0.00/0.18 L 0.00/0.15 |
| advantage_flow-dense-lr3e-6-seed42 | 132 | 844,569 | 8.5e-03 | T 0.55/0.78 T_narrow 0.55/0.70 Y 0.05/0.55 D 0.00/0.27 L 0.00/0.34 | 125: T 0.60/0.78 T_narrow 0.30/0.71 Y 0.15/0.69 D 0.00/0.32 L 0.00/0.32 |
| advantage_flow-lr3e-4-seed42 | 152 | 972,584 | 6.8e-02 | T 0.55/0.78 T_narrow 0.55/0.70 Y 0.05/0.55 D 0.00/0.27 L 0.00/0.34 | 150: T 0.40/0.67 T_narrow 0.40/0.76 Y 0.05/0.61 D 0.00/0.39 L 0.00/0.35 |
| advantage_flow-lr3e-4-seed43 | 150 | 952,039 | 7.6e-02 | T 0.55/0.78 T_narrow 0.65/0.75 Y 0.15/0.58 D 0.00/0.37 L 0.00/0.31 | 150: T 0.55/0.74 T_narrow 0.45/0.78 Y 0.05/0.56 D 0.00/0.40 L 0.00/0.33 |
| advantage_flow-lr3e-4-seed44 | 0 |  |  | no evaluation recorded |  |
| nft-dense-lr1e-4-seed42 | 26 | 164,978 | 1.5e+00 | T 0.55/0.78 T_narrow 0.55/0.70 Y 0.05/0.55 D 0.00/0.27 L 0.00/0.34 | 25: T 0.00/0.24 T_narrow 0.00/0.17 Y 0.00/0.13 D 0.00/0.18 L 0.00/0.17 |
| nft-dense-lr1e-5-seed42 | 158 | 1,013,441 | 1.1e-01 | T 0.55/0.78 T_narrow 0.55/0.70 Y 0.05/0.55 D 0.00/0.27 L 0.00/0.34 | 150: T 0.30/0.70 T_narrow 0.40/0.74 Y 0.00/0.62 D 0.00/0.36 L 0.00/0.38 |
| nft-dense-lr3e-5-seed42 | 161 | 1,031,407 | 9.0e-01 | T 0.55/0.78 T_narrow 0.55/0.70 Y 0.05/0.55 D 0.00/0.27 L 0.00/0.34 | 150: T 0.30/0.53 T_narrow 0.20/0.56 Y 0.05/0.41 D 0.00/0.20 L 0.00/0.19 |
| nft-dense-lr3e-6-seed42 | 133 | 847,705 | 1.0e-02 | T 0.55/0.78 T_narrow 0.55/0.70 Y 0.05/0.55 D 0.00/0.27 L 0.00/0.34 | 125: T 0.60/0.80 T_narrow 0.55/0.81 Y 0.10/0.61 D 0.00/0.34 L 0.00/0.35 |
| nft-lr3e-4-seed42 | 0 |  |  | no evaluation recorded |  |

## 3. 2026-10-25, diagnostics of the training signal

| Shape | Contexts | Plan share of score variance | Split-half reliability |
| --- | --- | --- | --- |
| T | 6 | 0.33 | 0.27 |
| T_narrow | 6 | 0.17 | 0.18 |
| Y | 6 | 0.16 | 0.14 |
| D | 6 | 0.36 | 0.29 |

Privileged value function as a candidate ranker (`q_validate.py`): mean correlation with the true full-episode
return across candidates, per context.

| Shape | Contexts | Trainer's 32-control score | Value-function score | One full-episode draw |
| --- | --- | --- | --- | --- |
| D | 6 | -0.11 | 0.12 | 0.27 |
| T | 6 | 0.06 | 0.04 | 0.43 |
| T_narrow | 6 | 0.15 | 0.19 | 0.47 |
| Y | 6 | 0.24 | -0.08 | 0.51 |

Frozen base on the 100-scene monitor test (300000 pool): evaluation noise and DDIM sampling.

| Policy | T | T_narrow | Y | D | L |
| --- | --- | --- | --- | --- | --- |
| base | 38 (0.65) | 38 (0.66) | 7 (0.49) | 0 (0.32) | 0 (0.30) |
| base_noise43 | 40 (0.66) | 33 (0.63) | 11 (0.55) | 0 (0.31) | 0 (0.27) |
| ddim10 | 35 (0.68) | 34 (0.58) | 6 (0.51) | 0 (0.25) | 0 (0.26) |
| ddim32 | 28 (0.60) | 32 (0.63) | 6 (0.51) | 0 (0.32) | 0 (0.28) |

## 4. 2026-10-25, simulator-oracle best-of-N rounds (privileged upper bound)

| Round | Shape | Candidates N | Look-ahead | Plain policy, same scenes | Oracle choice |
| --- | --- | --- | --- | --- | --- |
| oracle | T | 16 | 16 | 6/10 (0.78) | 7/10 (0.80) |
| oracle | T_narrow | 16 | 16 | 10/20 (0.74) | 13/20 (0.77) |
| oracle | Y | 16 | 16 | 3/20 (0.55) | 8/20 (0.77) |
| oracle | D | 8 | 16 | 0/5 (0.17) | 0/5 (0.44) |
| oracle | L | 8 | 16 | 0/5 (0.31) | 0/5 (0.43) |
| oracle2 | T | 16 | 24 | not run | 7/12 (0.78) |
| oracle2 | T_narrow | 16 | 24 | not run | 4/12 (0.71) |
| oracle2 | Y | 16 | 24 | not run | 10/18 (0.81) |
| oracle2 | D | 16 | 24 | not run | 0/12 (0.60) |
| oracle2 | L | 16 | 24 | not run | 1/6 (0.51) |
| oracle3 | T | 16 | 24 | not run | 9/12 (0.83) |
| oracle3 | T_narrow | 16 | 24 | not run | 19/30 (0.87) |
| oracle3 | Y | 16 | 24 | not run | 27/42 (0.88) |
| oracle3 | D | 16 | 24 | not run | 0/24 (0.58) |
| oracle3 | L | 16 | 24 | not run | 0/12 (0.36) |
| dagger1 | T | 16 | 24 | not run | 4/6 (0.76) |
| dagger1 | T_narrow | 16 | 24 | not run | 2/6 (0.56) |
| dagger1 | Y | 16 | 24 | not run | 8/12 (0.85) |
| dagger1 | D | 16 | 24 | not run | 0/6 (0.56) |
| dagger1c | T | 16 | 24 | not run | 4/6 (0.87) |
| dagger1c | T_narrow | 16 | 24 | not run | 5/6 (0.91) |
| dagger1c | Y | 16 | 24 | not run | 10/18 (0.82) |
| dagger1c | D | 16 | 24 | not run | 0/6 (0.61) |

`dagger1`/`dagger1c` sampled candidates from a distilled student instead of the base (DAgger data).

## 5. Distillation runs (validation pool 400000)

| Run | Data | Setting | Records | Checkpoint | Validation success (final coverage) |
| --- | --- | --- | --- | --- | --- |
| dense-lr1e-5-a1 | oracle | action_network, lr 1e-05, anchor 1.0 | 1779 | base | T 20/50 (0.70) T_narrow 17/50 (0.71) Y 4/50 (0.55) D 0/50 (0.29) L 0/50 (0.33) |
|  |  |  |  | epoch 5 | T 15/50 (0.60) T_narrow 20/50 (0.69) Y 1/50 (0.49) D 0/50 (0.31) L 0/50 (0.32) |
|  |  |  |  | epoch 10 | T 21/50 (0.65) T_narrow 16/50 (0.69) Y 2/50 (0.57) D 0/50 (0.32) L 0/50 (0.33) |
|  |  |  |  | epoch 15 | T 21/50 (0.66) T_narrow 19/50 (0.69) Y 5/50 (0.58) D 0/50 (0.30) L 0/50 (0.31) |
|  |  |  |  | epoch 20 | T 18/50 (0.68) T_narrow 23/50 (0.70) Y 8/50 (0.60) D 0/50 (0.31) L 0/50 (0.32) |
| lora-lr3e-4-a1 | oracle | lora, lr 0.0003, anchor 1.0 | 1779 | base | T 20/50 (0.70) T_narrow 17/50 (0.71) Y 4/50 (0.55) D 0/50 (0.29) L 0/50 (0.33) |
|  |  |  |  | epoch 5 | T 21/50 (0.69) T_narrow 19/50 (0.72) Y 5/50 (0.55) D 0/50 (0.32) L 0/50 (0.33) |
|  |  |  |  | epoch 10 | T 22/50 (0.69) T_narrow 23/50 (0.67) Y 4/50 (0.58) D 0/50 (0.34) L 0/50 (0.35) |
|  |  |  |  | epoch 15 | T 20/50 (0.68) T_narrow 28/50 (0.76) Y 7/50 (0.61) D 0/50 (0.31) L 0/50 (0.31) |
|  |  |  |  | epoch 20 | T 17/50 (0.65) T_narrow 22/50 (0.75) Y 2/50 (0.58) D 0/50 (0.35) L 0/50 (0.33) |
| dense-lr1e-5-r2 | oracle, oracle2 | action_network, lr 1e-05, anchor 1.0 | 3679 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 43/100 (0.69) T_narrow 43/100 (0.70) Y 11/100 (0.54) D 0/100 (0.33) L 0/100 (0.33) |
|  |  |  |  | epoch 20 | T 44/100 (0.70) T_narrow 44/100 (0.72) Y 8/100 (0.59) D 0/100 (0.30) L 0/100 (0.33) |
|  |  |  |  | epoch 30 | T 43/100 (0.69) T_narrow 51/100 (0.75) Y 7/100 (0.61) D 0/100 (0.31) L 0/100 (0.31) |
|  |  |  |  | epoch 40 | T 40/100 (0.70) T_narrow 40/100 (0.69) Y 12/100 (0.63) D 0/100 (0.35) L 0/100 (0.32) |
| lora-lr3e-4-r2 | oracle, oracle2 | lora, lr 0.0003, anchor 1.0 | 3679 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 41/100 (0.70) T_narrow 37/100 (0.70) Y 7/100 (0.57) D 0/100 (0.33) L 0/100 (0.33) |
|  |  |  |  | epoch 20 | T 40/100 (0.63) T_narrow 39/100 (0.70) Y 12/100 (0.60) D 0/100 (0.32) L 0/100 (0.32) |
|  |  |  |  | epoch 30 | T 43/100 (0.70) T_narrow 43/100 (0.71) Y 13/100 (0.57) D 0/100 (0.30) L 0/100 (0.33) |
|  |  |  |  | epoch 40 | T 43/100 (0.68) T_narrow 39/100 (0.70) Y 4/100 (0.55) D 0/100 (0.34) L 0/100 (0.32) |
| dense-lr1e-5-soft-r2 | oracle, oracle2 | action_network, lr 1e-05, anchor 1.0, softmax | 3679 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 45/100 (0.68) T_narrow 40/100 (0.64) Y 12/100 (0.57) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 20 | T 40/100 (0.68) T_narrow 40/100 (0.68) Y 13/100 (0.56) D 0/100 (0.32) L 0/100 (0.32) |
|  |  |  |  | epoch 30 | T 47/100 (0.73) T_narrow 40/100 (0.69) Y 16/100 (0.60) D 0/100 (0.32) L 0/100 (0.31) |
|  |  |  |  | epoch 40 | T 47/100 (0.69) T_narrow 37/100 (0.68) Y 13/100 (0.59) D 0/100 (0.31) L 0/100 (0.33) |
| lora-lr3e-4-soft-r2 | oracle, oracle2 | lora, lr 0.0003, anchor 1.0, softmax | 3679 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 44/100 (0.70) T_narrow 45/100 (0.71) Y 11/100 (0.58) D 0/100 (0.32) L 0/100 (0.33) |
|  |  |  |  | epoch 20 | T 37/100 (0.62) T_narrow 40/100 (0.69) Y 10/100 (0.59) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 30 | T 46/100 (0.69) T_narrow 40/100 (0.72) Y 10/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 40 | T 43/100 (0.70) T_narrow 34/100 (0.68) Y 11/100 (0.57) D 0/100 (0.31) L 0/100 (0.33) |
| dense-lr1e-5-soft-r3 | oracle, oracle2, oracle3 | action_network, lr 1e-05, anchor 1.0, softmax | 6372 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 44/100 (0.68) T_narrow 37/100 (0.67) Y 5/100 (0.53) D 0/100 (0.33) L 0/100 (0.33) |
|  |  |  |  | epoch 20 | T 45/100 (0.68) T_narrow 41/100 (0.67) Y 8/100 (0.54) D 1/100 (0.34) L 0/100 (0.30) |
|  |  |  |  | epoch 30 | T 36/100 (0.68) T_narrow 43/100 (0.68) Y 10/100 (0.58) D 0/100 (0.33) L 0/100 (0.31) |
|  |  |  |  | epoch 40 | T 45/100 (0.70) T_narrow 44/100 (0.69) Y 11/100 (0.61) D 0/100 (0.32) L 0/100 (0.32) |
| dense-lr1e-5-r3 | oracle, oracle2, oracle3 | action_network, lr 1e-05, anchor 1.0 | 6977 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 39/100 (0.65) T_narrow 48/100 (0.73) Y 9/100 (0.61) D 0/100 (0.35) L 0/100 (0.34) |
|  |  |  |  | epoch 20 | T 39/100 (0.68) T_narrow 45/100 (0.71) Y 7/100 (0.54) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 30 | T 49/100 (0.72) T_narrow 42/100 (0.70) Y 6/100 (0.56) D 0/100 (0.31) L 0/100 (0.33) |
|  |  |  |  | epoch 40 | T 37/100 (0.67) T_narrow 43/100 (0.71) Y 12/100 (0.59) D 0/100 (0.33) L 0/100 (0.33) |
| dense-lr1e-5-r4 | oracle, oracle2, oracle3, dagger1, dagger1c | action_network, lr 1e-05, anchor 1.0 | 8932 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 45/100 (0.68) T_narrow 39/100 (0.72) Y 14/100 (0.62) D 0/100 (0.33) L 0/100 (0.31) |
|  |  |  |  | epoch 20 | T 43/100 (0.70) T_narrow 39/100 (0.70) Y 7/100 (0.61) D 0/100 (0.31) L 0/100 (0.33) |
|  |  |  |  | epoch 30 | T 33/100 (0.65) T_narrow 39/100 (0.67) Y 9/100 (0.55) D 0/100 (0.33) L 0/100 (0.32) |
|  |  |  |  | epoch 40 | T 53/100 (0.74) T_narrow 37/100 (0.70) Y 6/100 (0.57) D 0/100 (0.34) L 0/100 (0.32) |
| dense-lr1e-5-soft-r4 | oracle, oracle2, oracle3, dagger1, dagger1c | action_network, lr 1e-05, anchor 1.0, softmax | 8932 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 41/100 (0.69) T_narrow 47/100 (0.73) Y 14/100 (0.61) D 0/100 (0.29) L 0/100 (0.31) |
|  |  |  |  | epoch 20 | T 46/100 (0.72) T_narrow 47/100 (0.69) Y 12/100 (0.61) D 0/100 (0.31) L 0/100 (0.32) |
|  |  |  |  | epoch 30 | T 46/100 (0.66) T_narrow 43/100 (0.70) Y 10/100 (0.60) D 0/100 (0.33) L 0/100 (0.31) |
|  |  |  |  | epoch 40 | T 40/100 (0.68) T_narrow 41/100 (0.73) Y 11/100 (0.59) D 1/100 (0.31) L 0/100 (0.33) |
| mpc-r1-lr3e-5 | r1, retain-T | action_network, lr 3e-05, anchor 1.0 on ['T', 'T_narrow'] / other 0.0 | 14566 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 15/100 (0.47) T_narrow 17/100 (0.48) Y 3/100 (0.43) D 0/100 (0.12) L 0/100 (0.21) |
|  |  |  |  | epoch 20 | T 21/100 (0.46) T_narrow 14/100 (0.40) Y 1/100 (0.35) D 0/100 (0.08) L 0/100 (0.15) |
|  |  |  |  | epoch 30 | T 20/100 (0.45) T_narrow 15/100 (0.49) Y 2/100 (0.39) D 0/100 (0.13) L 0/100 (0.15) |
| mpc-r1-lr1e-4 | r1, retain-T | action_network, lr 0.0001, anchor 1.0 on ['T', 'T_narrow'] / other 0.0 | 14566 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 18/100 (0.49) T_narrow 14/100 (0.48) Y 1/100 (0.37) D 0/100 (0.20) L 0/100 (0.20) |
|  |  |  |  | epoch 20 | T 18/100 (0.51) T_narrow 12/100 (0.51) Y 5/100 (0.47) D 0/100 (0.12) L 0/100 (0.22) |
|  |  |  |  | epoch 30 | T 17/100 (0.45) T_narrow 14/100 (0.42) Y 3/100 (0.45) D 1/100 (0.15) L 0/100 (0.21) |
| mpc-r1-lr3e-5-ao0.1 | r1, retain-T | action_network, lr 3e-05, anchor 1.0 on ['T', 'T_narrow'] / other 0.1 | 14566 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 20/100 (0.54) T_narrow 17/100 (0.56) Y 3/100 (0.37) D 0/100 (0.14) L 0/100 (0.24) |
|  |  |  |  | epoch 20 | T 25/100 (0.51) T_narrow 13/100 (0.50) Y 7/100 (0.47) D 0/100 (0.09) L 0/100 (0.20) |
|  |  |  |  | epoch 30 | T 22/100 (0.48) T_narrow 21/100 (0.54) Y 1/100 (0.44) D 1/100 (0.12) L 0/100 (0.18) |
| mpc-r2-lr3e-5 | r2, retain-T | action_network, lr 3e-05, anchor 1.0 on ['T', 'T_narrow'] / other 0.0 | 13355 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 18/100 (0.51) T_narrow 23/100 (0.50) Y 6/100 (0.41) D 0/100 (0.24) L 0/100 (0.24) |
|  |  |  |  | epoch 20 | T 17/100 (0.46) T_narrow 15/100 (0.45) Y 0/100 (0.36) D 0/100 (0.25) L 0/100 (0.19) |
|  |  |  |  | epoch 30 | T 24/100 (0.55) T_narrow 13/100 (0.49) Y 5/100 (0.43) D 0/100 (0.24) L 0/100 (0.22) |
| mpc-r2-lr3e-5-a3 | r2, retain-T | action_network, lr 3e-05, anchor 3.0 on ['T', 'T_narrow'] / other 0.3 | 13355 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 31/100 (0.59) T_narrow 33/100 (0.65) Y 8/100 (0.50) D 0/100 (0.23) L 0/100 (0.28) |
|  |  |  |  | epoch 20 | T 28/100 (0.51) T_narrow 18/100 (0.52) Y 2/100 (0.42) D 0/100 (0.19) L 0/100 (0.23) |
|  |  |  |  | epoch 30 | T 26/100 (0.56) T_narrow 22/100 (0.54) Y 8/100 (0.48) D 0/100 (0.25) L 0/100 (0.24) |
| mpc-r2-lr1e-5 | r2, retain-T | action_network, lr 1e-05, anchor 1.0 on ['T', 'T_narrow'] / other 0.1 | 13355 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 29/100 (0.54) T_narrow 16/100 (0.49) Y 1/100 (0.44) D 0/100 (0.19) L 0/100 (0.25) |
|  |  |  |  | epoch 20 | T 22/100 (0.46) T_narrow 21/100 (0.51) Y 0/100 (0.34) D 1/100 (0.17) L 0/100 (0.20) |
|  |  |  |  | epoch 30 | T 21/100 (0.56) T_narrow 15/100 (0.54) Y 3/100 (0.40) D 0/100 (0.26) L 0/100 (0.24) |
| mpc-r3-frozen-lr3e-5-a3 | r2, r2b, retain-T | action_network, lr 3e-05, anchor 3.0 on ['T', 'T_narrow'] / other 0.3 | 23608 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 22/100 (0.54) T_narrow 21/100 (0.51) Y 4/100 (0.41) D 0/100 (0.29) L 0/100 (0.27) |
|  |  |  |  | epoch 20 | T 9/100 (0.40) T_narrow 10/100 (0.39) Y 4/100 (0.36) D 0/100 (0.22) L 0/100 (0.18) |
|  |  |  |  | epoch 30 | T 24/100 (0.59) T_narrow 26/100 (0.53) Y 4/100 (0.49) D 0/100 (0.28) L 0/100 (0.25) |
| mpc-r3-enc-lr1e-5-a3 | r2, r2b, retain-T | action_network, lr 1e-05, anchor 3.0 on ['T', 'T_narrow'] / other 0.3, encoder trained | 23608 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 18/100 (0.48) T_narrow 22/100 (0.58) Y 3/100 (0.43) D 0/100 (0.22) L 0/100 (0.24) |
|  |  |  |  | epoch 20 | T 20/100 (0.42) T_narrow 21/100 (0.52) Y 5/100 (0.35) D 0/100 (0.17) L 0/100 (0.19) |
|  |  |  |  | epoch 30 | T 37/100 (0.62) T_narrow 28/100 (0.63) Y 4/100 (0.46) D 0/100 (0.26) L 0/100 (0.24) |
| mpc-r3-enc-lr3e-5 | r2, r2b, retain-T | action_network, lr 3e-05, anchor 1.0 on ['T', 'T_narrow'] / other 0.1, encoder trained | 23608 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 17/100 (0.54) T_narrow 13/100 (0.54) Y 2/100 (0.47) D 1/100 (0.25) L 0/100 (0.24) |
|  |  |  |  | epoch 20 | T 34/100 (0.63) T_narrow 19/100 (0.57) Y 2/100 (0.46) D 0/100 (0.25) L 0/100 (0.21) |
|  |  |  |  | epoch 30 | T 29/100 (0.62) T_narrow 22/100 (0.59) Y 8/100 (0.51) D 0/100 (0.38) L 1/100 (0.24) |
| mpc-r4-enc-lr3e-5-e60 | r2, r2b, r2c, retain-T | action_network, lr 3e-05, anchor 1.0 on ['T', 'T_narrow'] / other 0.1, encoder trained | 33849 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 15 | T 30/100 (0.65) T_narrow 24/100 (0.57) Y 4/100 (0.49) D 2/100 (0.36) L 1/100 (0.28) |
|  |  |  |  | epoch 30 | T 31/100 (0.67) T_narrow 35/100 (0.67) Y 5/100 (0.51) D 2/100 (0.47) L 2/100 (0.33) |
|  |  |  |  | epoch 45 | T 39/100 (0.68) T_narrow 25/100 (0.66) Y 9/100 (0.57) D 6/100 (0.50) L 4/100 (0.37) |
|  |  |  |  | epoch 60 | T 36/100 (0.69) T_narrow 32/100 (0.73) Y 14/100 (0.60) D 6/100 (0.55) L 5/100 (0.50) |
| mpc-dagger-it1-enc | r2, r2b, r2c, dagger-it1, retain-T | action_network, lr 3e-05, anchor 3.0 on ['T', 'T_narrow'] / other 0.3, encoder trained | 48449 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 23/100 (0.50) T_narrow 18/100 (0.53) Y 4/100 (0.41) D 1/100 (0.32) L 0/100 (0.19) |
|  |  |  |  | epoch 20 | T 24/100 (0.54) T_narrow 30/100 (0.63) Y 6/100 (0.48) D 0/100 (0.40) L 2/100 (0.32) |
|  |  |  |  | epoch 30 | T 27/100 (0.60) T_narrow 26/100 (0.65) Y 8/100 (0.48) D 4/100 (0.45) L 4/100 (0.36) |
|  |  |  |  | epoch 40 | T 29/100 (0.65) T_narrow 29/100 (0.66) Y 10/100 (0.50) D 6/100 (0.55) L 3/100 (0.35) |
| mpc-r4-enc-lr3e-5-a3-e60 | r2, r2b, r2c, retain-T | action_network, lr 3e-05, anchor 3.0 on ['T', 'T_narrow'] / other 0.3, encoder trained | 33849 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 15 | T 26/100 (0.60) T_narrow 23/100 (0.62) Y 7/100 (0.53) D 0/100 (0.31) L 0/100 (0.24) |
|  |  |  |  | epoch 30 | T 34/100 (0.66) T_narrow 27/100 (0.64) Y 2/100 (0.51) D 1/100 (0.39) L 0/100 (0.22) |
|  |  |  |  | epoch 45 | T 38/100 (0.71) T_narrow 34/100 (0.68) Y 11/100 (0.57) D 1/100 (0.45) L 3/100 (0.37) |
|  |  |  |  | epoch 60 | T 30/100 (0.68) T_narrow 32/100 (0.71) Y 11/100 (0.55) D 0/100 (0.50) L 6/100 (0.41) |
| mpc-dagger-it2-enc | r2, r2b, r2c, dagger-it1, dagger-it2, retain-T | action_network, lr 3e-05, anchor 3.0 on ['T', 'T_narrow'] / other 0.3, encoder trained | 62259 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 14/100 (0.50) T_narrow 25/100 (0.56) Y 7/100 (0.45) D 1/100 (0.39) L 0/100 (0.23) |
|  |  |  |  | epoch 20 | T 37/100 (0.68) T_narrow 30/100 (0.68) Y 9/100 (0.54) D 5/100 (0.49) L 1/100 (0.34) |
|  |  |  |  | epoch 30 | T 31/100 (0.62) T_narrow 30/100 (0.67) Y 7/100 (0.58) D 2/100 (0.46) L 1/100 (0.38) |
|  |  |  |  | epoch 40 | T 37/100 (0.65) T_narrow 31/100 (0.68) Y 15/100 (0.60) D 2/100 (0.48) L 6/100 (0.46) |
| mpc-big-lr3e-5 | r2, r2b, r2c, dagger-it1, dagger-it2, guard-10050, guard-10200, guard-10350, guard-1050, guard-10500, guard-10650, guard-10800, guard-10950, guard-11100, guard-11250, retain-T, retain-self | action_network, lr 3e-05, anchor 3.0 on ['T', 'T_narrow'] / other 0.3, encoder trained | 178498 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 8 | T 32/100 (0.64) T_narrow 38/100 (0.67) Y 4/100 (0.46) D 2/100 (0.40) L 2/100 (0.31) |
|  |  |  |  | epoch 16 | T 37/100 (0.68) T_narrow 31/100 (0.67) Y 15/100 (0.53) D 4/100 (0.43) L 2/100 (0.34) |
| mpc-big-lr1e-4 | r2, r2b, r2c, dagger-it1, dagger-it2, guard-10050, guard-10200, guard-10350, guard-1050, guard-10500, guard-10650, guard-10800, guard-10950, guard-11100, guard-11250, retain-T, retain-self | action_network, lr 0.0001, anchor 3.0 on ['T', 'T_narrow'] / other 0.3, encoder trained | 178498 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 8 | T 22/100 (0.59) T_narrow 18/100 (0.56) Y 2/100 (0.35) D 0/100 (0.39) L 1/100 (0.33) |
|  |  |  |  | epoch 16 | T 27/100 (0.61) T_narrow 22/100 (0.55) Y 14/100 (0.52) D 4/100 (0.43) L 5/100 (0.44) |
| mpc-enconly-lr3e-5 | r2, r2b, r2c, dagger-it1, dagger-it2, retain-T, retain-self | action_network, lr 3e-05, anchor 3.0 on ['T', 'T_narrow'] / other 0.3, encoder trained | 73892 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 37/100 (0.64) T_narrow 27/100 (0.65) Y 0/100 (0.02) D 0/100 (0.00) L 0/100 (0.01) |
|  |  |  |  | epoch 20 | T 43/100 (0.69) T_narrow 38/100 (0.70) Y 0/100 (0.00) D 0/100 (0.01) L 0/100 (0.00) |
| mpc-enconly-lr1e-4 | r2, r2b, r2c, dagger-it1, dagger-it2, retain-T, retain-self | action_network, lr 0.0001, anchor 3.0 on ['T', 'T_narrow'] / other 0.3, encoder trained | 73892 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 50/100 (0.70) T_narrow 42/100 (0.67) Y 0/100 (0.01) D 1/100 (0.03) L 0/100 (0.02) |
|  |  |  |  | epoch 20 | T 41/100 (0.71) T_narrow 35/100 (0.67) Y 0/100 (0.01) D 0/100 (0.05) L 0/100 (0.01) |
| mpc-lora-enc1e-4 | r2, r2b, r2c, dagger-it1, dagger-it2, retain-T, retain-self | lora, lr 0.0003, anchor 3.0 on ['T', 'T_narrow'] / other 0.3, encoder trained | 73892 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 31/100 (0.65) T_narrow 28/100 (0.62) Y 2/100 (0.31) D 1/100 (0.34) L 1/100 (0.18) |
|  |  |  |  | epoch 20 | T 42/100 (0.67) T_narrow 35/100 (0.68) Y 2/100 (0.31) D 3/100 (0.36) L 1/100 (0.25) |
| mpc-lora-enc3e-5 | r2, r2b, r2c, dagger-it1, dagger-it2, retain-T, retain-self | lora, lr 0.0003, anchor 3.0 on ['T', 'T_narrow'] / other 0.3, encoder trained | 73892 | base | T 35/100 (0.71) T_narrow 37/100 (0.69) Y 9/100 (0.56) D 0/100 (0.30) L 0/100 (0.32) |
|  |  |  |  | epoch 10 | T 43/100 (0.70) T_narrow 38/100 (0.68) Y 11/100 (0.47) D 2/100 (0.28) L 1/100 (0.17) |
|  |  |  |  | epoch 20 | T 42/100 (0.67) T_narrow 34/100 (0.65) Y 2/100 (0.30) D 0/100 (0.36) L 0/100 (0.21) |

Students on the monitor test (300000 pool, 100 scenes per shape):

| Student | T | T_narrow | Y | D | L |
| --- | --- | --- | --- | --- | --- |
| dense-soft-r2-e30 | 37 (0.66) | 40 (0.69) | 9 (0.53) | 1 (0.32) | 0 (0.30) |
| dense-chosen-r2-e30 | 40 (0.69) | 39 (0.65) | 13 (0.59) | 0 (0.34) | 0 (0.30) |
| dense-chosen-r4-e40 | 45 (0.69) | 34 (0.64) | 4 (0.59) | 0 (0.31) | 0 (0.32) |
| dense-soft-r3-e40 | 42 (0.68) | 39 (0.66) | 10 (0.59) | 0 (0.33) | 0 (0.30) |
| dense-chosen-r3-e40 | 49 (0.74) | 31 (0.62) | 10 (0.58) | 0 (0.32) | 0 (0.31) |
| dense-soft-r4-e40 | 40 (0.67) | 34 (0.62) | 9 (0.57) | 0 (0.32) | 0 (0.30) |
| mpc-r1-lr1e-4-last | 13 (0.41) | 15 (0.45) | 5 (0.41) | 0 (0.19) | 1 (0.21) |
| mpc-r1-lr3e-5-last | 22 (0.46) | 17 (0.45) | 1 (0.40) | 0 (0.12) | 0 (0.18) |
| mpc-r3-enc-lr1e-5-a3-last | 26 (0.53) | 23 (0.55) | 5 (0.44) | 1 (0.25) | 0 (0.23) |
| mpc-r3-enc-lr3e-5-last | 24 (0.56) | 24 (0.62) | 6 (0.46) | 1 (0.36) | 0 (0.25) |
| mpc-r3-frozen-lr3e-5-a3-last | 25 (0.53) | 27 (0.53) | 2 (0.50) | 0 (0.28) | 0 (0.24) |
| mpc-r4-enc-lr3e-5-e60-last | 24 (0.64) | 39 (0.71) | 10 (0.62) | 5 (0.54) | 5 (0.51) |
| mpc-dagger-it1-enc-last | 27 (0.56) | 21 (0.59) | 10 (0.52) | 2 (0.51) | 4 (0.35) |
| mpc-r4-enc-lr3e-5-a3-e60-last | 31 (0.66) | 29 (0.65) | 6 (0.47) | 1 (0.47) | 2 (0.37) |
| mpc-dagger-it2-enc-last | 29 (0.65) | 27 (0.64) | 16 (0.60) | 0 (0.44) | 6 (0.45) |
| blend-r4e60-s0.7 | 47 (0.73) | 38 (0.71) | 13 (0.66) | 3 (0.56) | 0 (0.37) |
| blend-r4e60-s0.5 | 36 (0.71) | 42 (0.72) | 14 (0.69) | 1 (0.51) | 0 (0.37) |
| blend-dag2-s0.5 | 43 (0.73) | 31 (0.75) | 17 (0.71) | 2 (0.45) | 0 (0.37) |
| mpc-big-lr3e-5-last | 35 (0.61) | 31 (0.65) | 8 (0.49) | 10 (0.49) | 1 (0.36) |
| mpc-big-lr1e-4-last | 25 (0.56) | 26 (0.61) | 8 (0.48) | 1 (0.40) | 4 (0.41) |
| blend-dag2-unet0.3-enc1.0 | 43 (0.74) | 39 (0.69) | 14 (0.65) | 0 (0.48) | 0 (0.35) |
| blend-big4-unet0-enc0.7 | 39 (0.68) | 33 (0.64) | 16 (0.61) | 1 (0.43) | 0 (0.33) |
| blend-r4e60-unet0.3-enc0.7 | 39 (0.72) | 35 (0.69) | 11 (0.67) | 2 (0.50) | 0 (0.35) |
| mpc-enconly-lr1e-4-last | 44 (0.70) | 33 (0.66) | 0 (0.03) | 0 (0.05) | 0 (0.00) |
| mpc-enconly-lr3e-5-last | 39 (0.66) | 30 (0.61) | 0 (0.01) | 1 (0.04) | 1 (0.01) |
| blend-big4-unet0-enc0.3 | 41 (0.66) | 41 (0.67) | 13 (0.56) | 0 (0.41) | 0 (0.32) |
| blend-big4-unet0-enc0.5 | 39 (0.66) | 43 (0.68) | 16 (0.61) | 1 (0.41) | 0 (0.34) |
| blend-big4-unet0-enc1.0 | 40 (0.68) | 39 (0.65) | 10 (0.62) | 0 (0.42) | 0 (0.31) |
| blend-r4e60-unet0-enc0.7 | 44 (0.72) | 29 (0.65) | 10 (0.55) | 1 (0.39) | 0 (0.33) |
| mpc-lora-enc1e-4-last | 39 (0.65) | 32 (0.63) | 6 (0.34) | 2 (0.38) | 2 (0.25) |
| mpc-lora-enc3e-5-last | 44 (0.69) | 32 (0.63) | 4 (0.31) | 2 (0.35) | 1 (0.25) |

## 6. 2026-10-25/26, privileged state world models and BPTT teachers

| Model | Steps | 8-control contact error (median) | 32-control contact error (median) |
| --- | --- | --- | --- |
| wm-state/smoke | 4000 | 3.40 px / 5.03 deg | 7.51 px / 11.56 deg |
| wm-state/v1 | 30000 | 2.32 px / 1.62 deg | 4.21 px / 3.12 deg |
| wm2/smoke | 6000 | T 0.45 px / 0.20 deg; T_narrow 0.51 px / 0.21 deg; Y 0.58 px / 0.28 deg; D 0.59 px / 0.32 deg; L 0.54 px / 0.27 deg | T 0.84 px / 0.39 deg; T_narrow 1.01 px / 0.43 deg; Y 0.98 px / 0.50 deg; D 1.02 px / 0.53 deg; L 0.88 px / 0.48 deg |
| wm2/v1 | 30000 | T 0.23 px / 0.12 deg; T_narrow 0.22 px / 0.11 deg; Y 0.24 px / 0.13 deg; D 0.32 px / 0.21 deg; L 0.23 px / 0.13 deg | T 0.42 px / 0.21 deg; T_narrow 0.47 px / 0.23 deg; Y 0.40 px / 0.23 deg; D 0.56 px / 0.35 deg; L 0.39 px / 0.25 deg |

| Teacher run | Last iteration | Real-simulator result (success/scenes, final coverage) |
| --- | --- | --- |
| teacher-v1 | 2000 | T 0/20 0.089  T_narrow 0/20 0.039  Y 0/20 0.050  D 0/20 0.070  L 0/20 0.051 |
| t2-dist1-h64-nocritic | 250 | T 0/20 0.090  T_narrow 0/20 0.090  Y 0/20 0.089  D 0/20 0.058  L 0/20 0.036 |
| t2-dist1 | 500 | T 0/20 0.114  T_narrow 0/20 0.118  Y 0/20 0.099  D 0/20 0.109  L 0/20 0.100 |
| t2-dist2-b2 | 500 | T 0/20 0.071  T_narrow 0/20 0.065  Y 0/20 0.041  D 0/20 0.039  L 0/20 0.058 |

## 7. 2026-10-26, sampling planner variants (validation pool 400000, chronological)

Settings show only what differs from the script defaults (CEM, 256 samples, horizon 32, success bonus 10,
pusher-gap penalty 0.5, contact margin 3 px, corner pose error, elite-mean execution).

| Run | Shape | Success | Final coverage | Median final pose error | Settings |
| --- | --- | --- | --- | --- | --- |
| smoke-T.json | T | 0/4 | 0.578 |  | defaults |
| v1/T.json | T | 0/20 | 0.701 |  | defaults |
| v1/L.json | L | 0/20 | 0.807 |  | defaults |
| v1/T_narrow.json | T_narrow | 1/20 | 0.568 |  | defaults |
| v1/Y.json | Y | 2/20 | 0.792 |  | defaults |
| v1/D.json | D | 4/20 | 0.789 |  | defaults |
| v4-nobonus-exec8/L.json | L | 1/20 | 0.726 |  | bonus 0.0, execute 8 |
| v4-nobonus-exec8/T.json | T | 0/20 | 0.319 |  | bonus 0.0, execute 8 |
| v3-nobonus-s20/L.json | L | 0/20 | 0.547 |  | samples 512, elites 32, sigma 20.0, bonus 0.0 |
| v2-nobonus/Y.json | Y | 0/20 | 0.774 |  | bonus 0.0 |
| v2-nobonus/L.json | L | 0/20 | 0.769 |  | bonus 0.0 |
| v2-nobonus/T.json | T | 0/20 | 0.711 |  | bonus 0.0 |
| v3-nobonus-s20/T.json | T | 0/20 | 0.471 |  | samples 512, elites 32, sigma 20.0, bonus 0.0 |
| diag-T.json | T | 0/12 | 0.652 | 4.9 px / 3.3 deg | bonus 0.0 |
| diagD-noapproach-nobonus-T.json | T | 1/12 | 0.590 | 12.7 px / 8.2 deg | bonus 0.0, approach 0.0 |
| diagA-noapproach-T.json | T | 1/12 | 0.374 | 59.8 px / 31.5 deg | approach 0.0 |
| diagC-search-T.json | T | 1/12 | 0.534 | 16.0 px / 5.9 deg | samples 1024, elites 64, iters 6 |
| diagB-noapproach-h64-T.json | T | 0/12 | 0.720 | 10.1 px / 6.0 deg | horizon 64, knots 10, approach 0.0 |
| diagE-ms-T.json | T | 0/12 | 0.169 | 78.4 px / 41.3 deg | scales [1.0, 0.3, 0.1] |
| diagF-ms-nobonus-T.json | T | 0/12 | 0.168 | 78.4 px / 41.3 deg | scales [1.0, 0.3, 0.1], bonus 0.0 |
| diagH-ms-512-T.json | T | 0/12 | 0.177 | 85.8 px / 36.8 deg | samples 512, elites 32, scales [1.0, 0.3, 0.1] |
| diagG-ms4-it6-T.json | T | 0/12 | 0.084 | 88.9 px / 42.7 deg | iters 6, scales [1.0, 0.3, 0.1, 0.03] |
| diagL-near10-T.json | T | 0/12 | 0.615 | 8.9 px / 5.7 deg | near 10.0, min_scale 0.05 |
| diagI-near20-T.json | T | 0/12 | 0.615 | 8.8 px / 5.8 deg | near 20.0 |
| diagJ-near40-T.json | T | 0/12 | 0.606 | 10.2 px / 2.7 deg | near 40.0, min_scale 0.05 |
| diagK-near20-nobonus-T.json | T | 0/12 | 0.654 | 4.6 px / 3.1 deg | near 20.0, bonus 0.0 |
| diagM-pred-T.json | T | 0/12 | 0.615 | 8.8 px / 5.8 deg | near 20.0 |
| diagP-m05-near20-T.json | T | 0/12 | 0.667 | 4.2 px / 3.0 deg | near 20.0, gate_margin 0.5 |
| diagO-m0-near20-T.json | T | 0/12 | 0.685 | 5.0 px / 3.4 deg | near 20.0, gate_margin 0.0 |
| diagN-m0-T.json | T | 0/12 | 0.696 | 5.1 px / 3.1 deg | gate_margin 0.0 |
| diagQ-push-best-m0-T.json | T | 0/12 | 0.841 | 3.9 px / 2.8 deg | gate_margin 0.0, pushes 128, select best |
| diagS-push-best-T.json | T | 0/12 | 0.843 | 3.9 px / 2.1 deg | pushes 128, select best |
| diagR-push-mean-T.json | T | 0/12 | 0.750 | 5.7 px / 3.6 deg | pushes 128 |
| diagT-push256-best-m0-T.json | T | 0/12 | 0.847 | 3.7 px / 2.9 deg | samples 512, elites 32, gate_margin 0.0, pushes 256, select best |
| diagW-h48-p256-it2-T.json | T | 0/12 | 0.122 | 104.0 px / 92.7 deg | iters 2, horizon 48, knots 8, gate_margin 0.0, pushes 256, select best |
| diagX-h48-g1-T.json | T | 0/12 | 0.837 | 4.4 px / 2.5 deg | horizon 48, knots 8, gamma 1.0, gate_margin 0.0, pushes 128, select best |
| diagU-h48-T.json | T | 0/12 | 0.842 | 4.4 px / 2.6 deg | horizon 48, knots 8, gate_margin 0.0, pushes 128, select best |
| trace-T.json | T | 0/2 | 0.903 | 4.5 px / 2.8 deg | horizon 48, knots 8, gate_margin 0.0, pushes 128, select best |
| diagY-h48-commit-T.json | T | 0/12 | 0.843 | 4.1 px / 2.7 deg | horizon 48, knots 8, gate_margin 0.0, pushes 128, select best, commit 0.02 |
| z3-T.json | T | 1/12 | 0.927 | 3.4 px / 2.3 deg | horizon 48, knots 8, approach 0.1, tight 2.0, gate_margin 0.0, pushes 128, select best |
| z2-T.json | T | 3/12 | 0.931 | 3.2 px / 2.2 deg | horizon 48, knots 8, approach 0.0, tight 2.0, gate_margin 0.0, pushes 128, select best |
| z4-T.json | T | 0/12 | 0.921 | 3.6 px / 2.4 deg | horizon 48, knots 8, approach 0.0, tight 2.0, gate_margin 0.0, pushes 128, select best, commit 0.02 |
| z1-T.json | T | 0/12 | 0.906 | 4.3 px / 3.1 deg | horizon 48, knots 8, approach 0.0, gate_margin 0.0, pushes 128, select best |
| trace2-T.json | T | 2/3 | 0.947 | 1.9 px / 1.0 deg | horizon 48, knots 8, approach 0.0, tight 2.0, gate_margin 0.0, pushes 128, select best |
| z5-T.json | T | 0/12 | 0.924 | 3.4 px / 2.5 deg | horizon 48, knots 8, approach 0.0, tight 2.0, push_min 0.3, gate_margin 0.0, pushes 128, select best |
| z8-T.json | T | 0/12 | 0.913 | 3.8 px / 2.4 deg | horizon 48, knots 8, approach 0.0, tight 2.0, tight_scale 1.0, push_min 0.3, gate_margin 0.0, pushes 128, select best |
| z7-T.json | T | 0/12 | 0.913 | 3.8 px / 2.6 deg | samples 512, elites 32, horizon 48, knots 8, approach 0.0, tight 2.0, tight_scale 1.0, push_min 0.3, gate_margin 0.0, pushes 256, select best |
| z6-T.json | T | 0/12 | 0.913 | 3.8 px / 2.4 deg | horizon 48, knots 8, approach 0.0, tight 2.0, tight_scale 1.0, push_min 0.3, gate_margin 0.0, pushes 128, select best |
| a1-T.json | T | 2/12 | 0.934 | 0.6 px / 3.1 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| a2-T.json | T | 1/12 | 0.931 | 1.4 px / 3.0 deg | horizon 48, knots 8, approach 0.0, tight 2.0, push_min 0.3, kp area, gate_margin 0.0, pushes 128, select best |
| a3-T.json | T | 2/12 | 0.927 | 1.6 px / 2.4 deg | horizon 48, knots 8, approach 0.0, tight 4.0, tight_scale 1.5, push_min 0.3, kp area, gate_margin 0.0, pushes 128, select best |
| a1b-T.json | T | 1/12 | 0.932 | 1.9 px / 2.4 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v5/D.json | D | 20/20 | 0.967 | 0.9 px / 0.6 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v5/Y.json | Y | 16/20 | 0.956 | 1.0 px / 0.9 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| a5-T.json | T | 2/24 | 0.918 | 1.5 px / 2.7 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best, commit 0.01 |
| v5/T_narrow.json | T_narrow | 3/20 | 0.937 | 1.7 px / 2.2 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v5/L.json | L | 7/20 | 0.943 | 2.1 px / 1.2 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| a4-T.json | T | 3/24 | 0.919 | 1.2 px / 3.0 deg | horizon 64, knots 10, approach 0.0, tight 2.0, transit 35.0, kp area, gate_margin 0.0, pushes 128, select best |
| v5-50/D.json | D | 50/50 | 0.967 | 0.9 px / 0.4 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v5-50/L.json | L | 21/50 | 0.943 | 2.2 px / 0.9 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v5-50/T_narrow.json | T_narrow | 12/50 | 0.938 | 1.7 px / 2.2 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v5-50/Y.json | Y | 36/50 | 0.954 | 1.0 px / 1.0 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v5-50/T.json | T | 5/50 | 0.932 | 1.2 px / 2.9 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v6-fix/Y.json | Y | 18/20 | 0.955 | 0.9 px / 1.0 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v6-fix/T.json | T | 1/24 | 0.932 | 1.3 px / 2.8 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v6-fix/D.json | D | 19/20 | 0.965 | 0.9 px / 0.5 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v6-fix/T_narrow.json | T_narrow | 4/20 | 0.937 | 1.6 px / 2.1 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v6-fix/L.json | L | 6/20 | 0.941 | 2.2 px / 1.5 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| v7-exec8/D.json | D | 20/20 | 0.962 | 0.9 px / 0.5 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, pushes 128, select best |
| v7-exec8-commit/D.json | D | 20/20 | 0.962 | 0.9 px / 0.5 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, pushes 128, select best, commit 0.02 |
| v7-exec8-commit/Y.json | Y | 14/20 | 0.953 | 1.0 px / 0.9 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, pushes 128, select best, commit 0.02 |
| v7-exec8-commit/L.json | L | 7/20 | 0.942 | 2.2 px / 1.0 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, pushes 128, select best, commit 0.02 |
| v7-exec8/L.json | L | 7/20 | 0.945 | 1.9 px / 1.8 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, pushes 128, select best |
| v7-exec8/Y.json | Y | 14/20 | 0.954 | 1.1 px / 0.8 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, pushes 128, select best |
| smoke-dponly.json | Y | 0/2 | 0.008 | 87.3 px / 66.5 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| smoke-hybrid.json | Y | 2/2 | 0.963 | 0.6 px / 1.0 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| dp-only/T_narrow.json | T_narrow | 0/20 | 0.312 | 68.0 px / 57.7 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-only/T.json | T | 0/20 | 0.346 | 68.0 px / 61.3 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-only/L.json | L | 0/20 | 0.252 | 62.4 px / 102.8 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-only/Y.json | Y | 0/20 | 0.259 | 76.1 px / 62.0 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-only/D.json | D | 0/20 | 0.361 | 33.1 px / 51.7 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-first/T.json | T | 10/20 | 0.731 | 6.0 px / 1.7 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-nowarm-h15/T.json | T | 11/20 | 0.804 | 2.1 px / 1.5 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-warm-h15/T.json | T | 0/20 | 0.192 | 95.4 px / 62.0 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-nowarm-h48/T.json | T | 8/20 | 0.821 | 5.4 px / 2.2 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-hybrid/T.json | T | 0/20 | 0.929 | 1.3 px / 3.0 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| dp-hybrid/D.json | D | 18/20 | 0.939 | 0.9 px / 0.6 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| dp-nowarm-h15/L.json | L | 0/20 | 0.379 | 18.2 px / 167.9 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-nowarm-h15/D.json | D | 1/20 | 0.530 | 14.7 px / 33.8 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-hybrid/T_narrow.json | T_narrow | 3/20 | 0.936 | 1.6 px / 2.4 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| dp-hybrid/Y.json | Y | 15/20 | 0.957 | 1.0 px / 0.7 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| dp-hybrid/L.json | L | 4/20 | 0.937 | 2.5 px / 0.4 deg | horizon 48, knots 8, approach 0.0, tight 2.0, kp area, gate_margin 0.0, pushes 128, select best |
| dp-nowarm-h15/T_narrow.json | T_narrow | 9/20 | 0.812 | 2.6 px / 1.8 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-nowarm-h15/Y.json | Y | 7/20 | 0.799 | 1.7 px / 2.5 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-blend-nowarm-h15/T.json | T | 11/20 | 0.850 | 2.0 px / 1.5 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-blend-nowarm-h15/T_narrow.json | T_narrow | 8/20 | 0.765 | 4.7 px / 10.2 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-blend-nowarm-h15/Y.json | Y | 8/20 | 0.796 | 1.9 px / 0.7 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-blend-nowarm-h15/D.json | D | 5/20 | 0.703 | 4.0 px / 9.4 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |
| dp-blend-nowarm-h15/L.json | L | 0/20 | 0.404 | 19.1 px / 154.6 deg | horizon 15, knots 8, approach 0.0, tight 2.0, kp area, execute 8, gate_margin 0.0, select best |

## 8. Planner demonstrations for distillation (teacher-data pool 800000)

| Round | Shape | Episode success | Records | Labels | Teacher share (beta) |
| --- | --- | --- | --- | --- | --- |
| r1 | D | 144/150 | 2239 | planner, every control | beta 1.0 |
| r1 | L | 40/100 | 5834 | planner, every control | beta 1.0 |
| r1 | Y | 74/100 | 3448 | planner, every control | beta 1.0 |
| r2 | D | 135/150 | 2097 | planner, 8-control chunks | beta 1.0 |
| r2 | L | 44/150 | 4810 | planner, 8-control chunks | beta 1.0 |
| r2 | Y | 96/150 | 3403 | planner, 8-control chunks | beta 1.0 |
| r2b | D | 138/150 | 2039 | planner, 8-control chunks | beta 1.0 |
| r2b | L | 36/150 | 4874 | planner, 8-control chunks | beta 1.0 |
| r2b | Y | 97/150 | 3340 | planner, 8-control chunks | beta 1.0 |
| r2c | D | 133/150 | 2149 | planner, 8-control chunks | beta 1.0 |
| r2c | L | 33/150 | 4930 | planner, 8-control chunks | beta 1.0 |
| r2c | Y | 104/150 | 3162 | planner, 8-control chunks | beta 1.0 |
| dagger-it1 | D | 74/150 | 4552 | planner, 8-control chunks | beta 0.3 |
| dagger-it1 | L | 26/150 | 5359 | planner, 8-control chunks | beta 0.3 |
| dagger-it1 | Y | 66/150 | 4689 | planner, 8-control chunks | beta 0.3 |
| dagger-it2 | D | 88/150 | 4255 | planner, 8-control chunks | beta 0.3 |
| dagger-it2 | L | 47/150 | 5017 | planner, 8-control chunks | beta 0.3 |
| dagger-it2 | Y | 80/150 | 4538 | planner, 8-control chunks | beta 0.3 |
| guard (overnight idle-guard rounds) (71 rounds) | D | 9857/10650 | 141544 | planner, 8-control chunks | beta 1.0 |
| guard (overnight idle-guard rounds) (71 rounds) | L | 2499/10650 | 349426 | planner, 8-control chunks | beta 1.0 |
| guard (overnight idle-guard rounds) (71 rounds) | Y | 6553/10650 | 244447 | planner, 8-control chunks | beta 1.0 |
| retain-self | T | 71/180 | 5803 | policy's own plans | beta 0.0 |
| retain-self | T_narrow | 83/180 | 5830 | policy's own plans | beta 0.0 |

Beta is the share of decisions the planner executes; below 1 the image student drives the rest (DAgger). Seven
overnight guard rounds that used final-test scenes were quarantined (moved out of the data folder) before any
training; they are not counted above.

## 8b. Weight blends of students with the base (validation pool 400000, 100 scenes per shape)

| Blend (student, share s) | T | T_narrow | Y | D | L |
| --- | --- | --- | --- | --- | --- |
| blend-r4e60-s0.3 | 45 (0.73) | 41 (0.75) | 15 (0.62) | 1 (0.39) | 0 (0.33) |
| blend-r4e60-s0.5 | 42 (0.74) | 41 (0.76) | 17 (0.65) | 0 (0.50) | 0 (0.36) |
| blend-dag2-s0.3 | 42 (0.73) | 35 (0.74) | 10 (0.62) | 1 (0.39) | 0 (0.35) |
| blend-r4e60-s0.7 | 39 (0.71) | 42 (0.73) | 12 (0.72) | 4 (0.58) | 0 (0.40) |
| blend-dag2-s0.5 | 45 (0.76) | 40 (0.74) | 15 (0.69) | 0 (0.47) | 0 (0.35) |
| blend-dag2-s0.7 | 39 (0.76) | 44 (0.75) | 15 (0.70) | 1 (0.52) | 0 (0.35) |
| blend-r4e60-unet0-enc0.7 | 51 (0.72) | 45 (0.70) | 12 (0.59) | 2 (0.39) | 0 (0.34) |
| blend-r4e60-unet0.7-enc0 | 33 (0.69) | 40 (0.77) | 7 (0.64) | 0 (0.43) | 0 (0.34) |
| blend-r4e60-unet0.7-enc0.3 | 38 (0.71) | 32 (0.72) | 11 (0.63) | 0 (0.48) | 0 (0.34) |
| blend-r4e60-unet1-enc0 | 30 (0.66) | 23 (0.64) | 2 (0.52) | 1 (0.41) | 0 (0.31) |
| blend-mpc-big-lr3e-5-s0.5 | 46 (0.74) | 33 (0.68) | 16 (0.66) | 1 (0.41) | 0 (0.32) |
| blend-mpc-big-lr3e-5-s0.7 | 41 (0.72) | 32 (0.68) | 12 (0.65) | 2 (0.49) | 1 (0.36) |
| blend-mpc-big-lr1e-4-s0.5 | 41 (0.71) | 36 (0.69) | 6 (0.62) | 1 (0.49) | 0 (0.37) |
| blend-mpc-big-lr1e-4-s0.7 | 40 (0.66) | 31 (0.65) | 10 (0.61) | 6 (0.52) | 0 (0.36) |
| blend-r4e60-unet0-enc1.0 | 42 (0.68) | 46 (0.72) | 8 (0.54) | 0 (0.40) | 0 (0.33) |
| blend-r4e60-unet0-enc0.5 | 47 (0.71) | 41 (0.72) | 9 (0.55) | 0 (0.36) | 0 (0.34) |
| blend-r4e60-unet0.3-enc0.7 | 44 (0.75) | 46 (0.77) | 15 (0.68) | 1 (0.47) | 0 (0.33) |
| blend-r4e60-unet0.3-enc1.0 | 46 (0.74) | 46 (0.74) | 10 (0.68) | 1 (0.48) | 1 (0.35) |
| blend-dag2-unet0-enc0.7 | 50 (0.75) | 41 (0.73) | 7 (0.57) | 0 (0.41) | 0 (0.33) |
| blend-dag2-unet0.3-enc1.0 | 48 (0.76) | 51 (0.76) | 18 (0.67) | 3 (0.49) | 0 (0.34) |
| blend-big3-unet0-enc0.7 | 46 (0.70) | 46 (0.74) | 6 (0.60) | 1 (0.38) | 0 (0.32) |
| blend-big4-unet0-enc0.7 | 42 (0.73) | 41 (0.68) | 20 (0.59) | 1 (0.43) | 0 (0.32) |

## 9. Frozen base on the 1000-scene final test pool (900000)

| Shape | Scenes | Successes | Rate | Mean final coverage |
| --- | --- | --- | --- | --- |
| T | 1000 | 444 | 0.444 | 0.692 |
| T_narrow | 1000 | 419 | 0.419 | 0.696 |
| Y | 1000 | 94 | 0.094 | 0.578 |
| D | 1000 | 3 | 0.003 | 0.314 |
| L | 1000 | 1 | 0.001 | 0.317 |

Primary pre-registered final test (D040) of `blend-r4e60-s0.7`, paired by scene with the base:

| Shape | Scenes | Base | Candidate | Difference [95% interval] | Coverage base -> candidate |
| --- | --- | --- | --- | --- | --- |
| T | 1000 | 44.4% | 39.6% | -4.8 [-8.7, -0.9] | 0.692 -> 0.730 |
| T_narrow | 1000 | 41.9% | 41.0% | -0.9 [-4.7, +2.9] | 0.696 -> 0.761 |
| Y | 1000 | 9.4% | 12.3% | +2.9 [+0.2, +5.6] | 0.578 -> 0.688 |
| D | 1000 | 0.3% | 4.9% | +4.6 [+3.2, +6.0] | 0.314 -> 0.581 |
| L | 1000 | 0.1% | 0.2% | +0.1 [-0.2, +0.4] | 0.317 -> 0.398 |

Pass conditions: 1_T_noninferior no, 2_T_narrow_noninferior yes, 3_new_letter_better yes, 4_no_new_letter_worse yes, complete yes. Overall: not passed.

Secondary final test (look 2 at the final pool) of `blend-big4-unet0-enc0.7`, paired by scene with the base:

| Shape | Scenes | Base | Candidate | Difference [95% interval] | Coverage base -> candidate |
| --- | --- | --- | --- | --- | --- |
| T | 1000 | 44.4% | 41.9% | -2.5 [-6.2, +1.2] | 0.692 -> 0.701 |
| T_narrow | 1000 | 41.9% | 39.9% | -2.0 [-5.8, +1.8] | 0.696 -> 0.709 |
| Y | 1000 | 9.4% | 18.1% | +8.7 [+5.7, +11.7] | 0.578 -> 0.635 |
| D | 1000 | 0.3% | 1.4% | +1.1 [+0.3, +1.9] | 0.314 -> 0.432 |
| L | 1000 | 0.1% | 0.1% | +0.0 [-0.3, +0.3] | 0.317 -> 0.334 |

Pass conditions: 1_T_noninferior no, 2_T_narrow_noninferior no, 3_new_letter_better yes, 4_no_new_letter_worse yes, complete yes. Overall: not passed.

D041 confirmatory test on a fresh pool (2000 scenes per letter) of `blend-big4-unet0-enc0.7`, paired by scene with the base:

| Shape | Scenes | Base | Candidate | Difference [95% interval] | Coverage base -> candidate |
| --- | --- | --- | --- | --- | --- |
| T | 2000 | 46.0% | 42.8% | -3.1 [-5.9, -0.4] | 0.714 -> 0.719 |
| T_narrow | 2000 | 42.3% | 42.8% | +0.4 [-2.2, +3.1] | 0.698 -> 0.718 |
| Y | 2000 | 8.8% | 16.8% | +8.0 [+6.0, +9.9] | 0.561 -> 0.640 |
| D | 2000 | 0.1% | 0.9% | +0.8 [+0.3, +1.3] | 0.318 -> 0.439 |
| L | 2000 | 0.0% | 0.0% | +0.0 [+0.0, +0.0] | 0.319 -> 0.330 |

Pass conditions: 1_T_noninferior no, 2_T_narrow_noninferior yes, 3_new_letter_better yes, 4_no_new_letter_worse yes, complete yes. Overall: not passed.

D042 test on a second fresh pool (2000 scenes per letter, 97.5% intervals, two candidates) of `blend-big4-unet0-enc0.3`, paired by scene with the base:

| Shape | Scenes | Base | Candidate | Difference [95% interval] | Coverage base -> candidate |
| --- | --- | --- | --- | --- | --- |
| T | 2000 | 43.6% | 46.7% | +3.0 [+0.1, +6.0] | 0.700 -> 0.716 |
| T_narrow | 2000 | 40.4% | 43.0% | +2.6 [-0.3, +5.6] | 0.691 -> 0.703 |
| Y | 2000 | 8.6% | 11.8% | +3.2 [+1.1, +5.3] | 0.552 -> 0.590 |
| D | 2000 | 0.3% | 0.3% | +0.0 [-0.4, +0.4] | 0.315 -> 0.377 |
| L | 2000 | 0.0% | 0.0% | +0.0 [+0.0, +0.0] | 0.319 -> 0.334 |

Pass conditions: 1_T_noninferior yes, 2_T_narrow_noninferior yes, 3_new_letter_better yes, 4_no_new_letter_worse yes, complete yes. Overall: PASS.

D042 test on a second fresh pool (2000 scenes per letter, 97.5% intervals, two candidates) of `blend-big4-unet0-enc0.5`, paired by scene with the base:

| Shape | Scenes | Base | Candidate | Difference [95% interval] | Coverage base -> candidate |
| --- | --- | --- | --- | --- | --- |
| T | 2000 | 43.6% | 45.4% | +1.8 [-1.3, +4.8] | 0.700 -> 0.714 |
| T_narrow | 2000 | 40.4% | 41.8% | +1.4 [-1.6, +4.3] | 0.691 -> 0.706 |
| Y | 2000 | 8.6% | 14.8% | +6.2 [+4.0, +8.4] | 0.552 -> 0.608 |
| D | 2000 | 0.3% | 1.1% | +0.9 [+0.2, +1.5] | 0.315 -> 0.414 |
| L | 2000 | 0.0% | 0.1% | +0.1 [-0.1, +0.3] | 0.319 -> 0.339 |

Pass conditions: 1_T_noninferior yes, 2_T_narrow_noninferior yes, 3_new_letter_better yes, 4_no_new_letter_worse yes, complete yes. Overall: PASS.

D043 replication on a third fresh pool (2000 scenes per letter, 95% intervals) of `blend-big4-unet0-enc0.5`, paired by scene with the base:

| Shape | Scenes | Base | Candidate | Difference [95% interval] | Coverage base -> candidate |
| --- | --- | --- | --- | --- | --- |
| T | 2000 | 43.4% | 45.1% | +1.8 [-0.9, +4.4] | 0.699 -> 0.710 |
| T_narrow | 2000 | 40.0% | 40.9% | +0.9 [-1.7, +3.6] | 0.690 -> 0.705 |
| Y | 2000 | 9.8% | 14.1% | +4.3 [+2.4, +6.3] | 0.572 -> 0.608 |
| D | 2000 | 0.1% | 0.9% | +0.9 [+0.4, +1.3] | 0.306 -> 0.416 |
| L | 2000 | 0.1% | 0.0% | -0.1 [-0.1, +0.0] | 0.318 -> 0.331 |

Pass conditions: 1_T_noninferior yes, 2_T_narrow_noninferior yes, 3_new_letter_better yes, 4_no_new_letter_worse yes, complete yes. Overall: PASS.

## 10. Running or pending when this was written

- All 26 letters plus near-T, base vs the 50% encoder blend (100 fresh scenes each, seeds
  5100000 + 20*i): a descriptive check of letters never used in adaptation.
- Possible follow-ups: a DAgger round driven by the passing blend, and L, which no image
  policy has improved yet.
