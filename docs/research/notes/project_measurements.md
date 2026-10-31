# Project measurements (this project's own experiments, 2026-10-24/25)

Source: runs on one H100 NVL (SLURM job 1009007) and one H200 (2026-10-24); raw files in the private
the author's private run store (runs/, logs/). These are first-hand results, not literature.

## Base policy performance

| Evaluation | T | T_narrow | Y | D | L |
| --- | --- | --- | --- | --- | --- |
| 20 development scenes (100000-100019), noise seed 42 | 11/20 | 11/20 | 1/20 | 0/20 | 0/20 |
| Same 20 scenes, noise seed 43 | 11/20 | 13/20 | 3/20 | 0/20 | 0/20 |
| **100 fresh TEST scenes (300000+20i), DDPM-100, noise 42** | **38/100 [0.29, 0.48]** | **38/100 [0.29, 0.48]** | **7/100 [0.03, 0.14]** | **0/100** | **0/100** |
| Same test scenes, DDIM eta=0, 10 steps | 35/100 | 34/100 | 6/100 | 0 | 0 |
| Same test scenes, DDIM eta=0, 32 steps | 28/100 | 32/100 | 6/100 | 0 | 0 |
| 1000 value-data episodes (200 scenes per shape, 600000+20i) | 40% | 38% | 6.5% | 0% | 0% |

Mean final coverage on the test set: T 0.65, T_narrow 0.66, Y 0.49, D 0.32, L 0.30.
DDPM-100 vs DDIM-32 on the same test scenes, T: 24 scenes succeed only under DDPM, 14 only under DDIM:
scene-level agreement between two similar policies is low (chaos), so pairing gives limited power.

## Long fine-tuning runs (near-miss preset: train T, D, T_narrow, Y; eval + L; 20 dev scenes, eval every 25 updates)

Drift = fixed-probe MSE from the initial policy / denoiser error. Ordered by drift at update ~70-98:

| Run | Drift | T success at latest eval | T_narrow |
| --- | --- | --- | --- |
| dense AF/NFT lr 3e-6 | 0.007 | 0.50-0.55 (u50) | 0.25-0.40 |
| LoRA AF lr 3e-4 (2 seeds) | 0.04 | 0.55 / 0.70 (u75) | 0.45 |
| dense NFT lr 1e-5 | 0.09 | 0.45 (u75) | 0.20 |
| dense AF lr 1e-5 | 0.15 | 0.20 (u75) | 0.15 |
| dense NFT/AF lr 3e-5 | 0.7-2.2 | 0.05-0.15 (u75) | 0.15-0.20 |
| dense NFT/AF lr 1e-4 | 0.6-1.5 by u25 | 0.00-0.15 (collapsed; stopped) | 0.00-0.15 |

No run improved Y, D (or untrained L) beyond the base-vs-base noise band. Yesterday's eight 1M-control
runs at lr 1e-5 LoRA moved the policy only ~1e-6 drift and changed nothing.
Weighted T-retention is ~1e-4 of the main loss. 12-39% of candidate groups fully tied.

## Score reliability (frozen base, 8 candidates x 8 independent follow-up draws, 6 contexts per shape)

Raw between-candidate share of 32-control score variance: T 0.33, T_narrow 0.17, Y 0.16, D 0.36; split-half r
0.14-0.29. Bias-corrected by reviewers: T ~0.25, T_narrow ~0.08, Y ~0.06, D ~0.29.

## Learned state-value function (privileged state, MC targets of the trainer's shaped return)

Trained on 800 of the 1000 value-data episodes; validation by scene: R^2 0.64 overall; correlation with
single-rollout targets T 0.75, T_narrow 0.73, Y 0.76, D 0.87, L 0.93.
Candidate-ranking test (G=8 plans, each followed to episode end 8 times; 'truth' = mean of 8 full-episode
returns; partial results, 3-5 contexts per shape): per-context correlation with truth swings -0.9..+0.9 for
both the current 32-control single-draw score and the value-based score (own 8 controls + gamma^8 V);
the value score is not consistently better. In D contexts all candidates often end identical (no contact,
spread 0.000). Within-candidate draw std exceeds the between-candidate truth spread in most contexts.

## Simulator-oracle best-of-N (upper bound for reweighting/steering), smoke test only

One Y scene (700000): base policy final coverage 0.671 (no success); oracle choosing the best of 4 base
samples per decision with 8-control lookahead (exact full-prefix replay): final coverage 0.909 (no success).
Cost ~6.5 min per oracle episode at N=4 under load. Full test (N=16, lookahead 24, ~10 scenes x Y/D/L/T)
pending.

## Cost facts

~900 simulator controls/s aggregate on one GPU with 8 jobs (8 is the throughput optimum; 16 is 22% worse).
Prefix replay is 81-85% of counted training controls. One 100-episode batched evaluation ~5 min;
a 500-episode test evaluation ~10-14 min under load.
