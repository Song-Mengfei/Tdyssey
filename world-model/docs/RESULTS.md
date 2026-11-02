# Recorded results and limits

This page summarizes existing saved evidence. Packaging did not rerun models or independently reproduce experiments on a clean machine. Completion, implementation correctness and scientific acceptance are separate claims.

## Recent loss-definition screen

Common parent: full E2 model/Adam/scaler/RNG state. Three arms, each 128 attempts, 128 updates and 2,048 native H1 exposures; AMP skips 0. Fixed batches/noise/actions/precision and original evaluation. Cached E2 and BASE128 were reused; they were not retrained as part of the screen. Validation has been reused across experiments, so this is an exploratory comparison.

Raw autoregressive H16 gray IoU on the original seen validation split, eight cases per letter (higher is better):

| Arm | O | C | B |
| --- | ---: | ---: | ---: |
| E2 | 0.456442 | 0.253279 | 0.514797 |
| Native BASE128 | 0.466823 | 0.249992 | 0.523932 |
| G_ONLY | 0.466318 | 0.204634 | 0.506821 |
| NATIVE_G | 0.458797 | 0.244821 | 0.522678 |
| NATIVE_T | 0.461089 | 0.249656 | 0.525916 |

No arm passed the complete O/C/B directional criteria or the original absolute acceptance gates. Only NATIVE_G/C passed the declared short-horizon per-letter condition: AR4 IoU 0.651441→0.659662 and position error 0.454861→0.386860 px. Its C H16 still regressed. NATIVE_G introduced an O extra-hole error; NATIVE_T's small B IoU gain coincided with worse centroid position error. Shape/rotation qualification remained largely unknown, which is not a pass.

T discriminated disjoint near/far shifts that Dice scored equally. Near-zero foreground mass also produced very large transport gradients in diagnostics. These controlled image responses establish properties of a definition, not correct learned action dynamics. No winner is justified from this single-seed short screen; untrained alternatives are not ruled out.

The saved independent analysis checked 6,240 score rows, 1,485 groups / 4,455 group means, 384 RNG/branch pairings, fixed coefficients and 16 frozen Python source identities. The owner audit hashed raw/checkpoint assets. Packaging carries those reports; it does not claim to have repeated the full raw-asset audit.

## Other attempts retained

| Attempt | Outcome / qualification |
| --- | --- |
| High-motion fit, 3,000 updates | Training/RH gains; full joint fit did not pass, C validation and old failures regressed |
| A/B/C/D history/region comparisons | Local aggregate AR gains without complete rigidity/shape/rotation validation |
| Four-step rollout comparison | First preflight failed before training; repaired R1 trained, then Windows report filename failure remained in terminal status; no complete support |
| Min-SNR / GDL / FFL, 128 updates/arm | Mixed effects; GDL improved B H16 but worsened C; no complete directional pass; recovery/replayed cost provenance retained |
| Early native / staged pilots | Distinct controllers, samplers and failures; cannot add their results into one homogeneous model trajectory |
| Older structured attempt / deleted weights | Historical-only evidence is explicitly qualified; missing files are not represented as reproduced |

See [the full experiment archive](../experiments/README.md), per-run evidence indices and retrospective ledgers for exact original costs, statuses and interfaces. Do not sum independent branches as if they were one training run. No policy improvement, unseen-letter generalization or reliable contact physics follows from these image metrics alone.
