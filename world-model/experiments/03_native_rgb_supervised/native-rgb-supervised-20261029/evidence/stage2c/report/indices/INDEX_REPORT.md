# Frozen seen-letter evaluation indices

Policy seed=2026102903; policySHA256 `f55fad565e6cc013e165529f81f84cf1ef46a67985b2d3f1bdae391545f6c6a1`.
Natural is uniform pseudorandom per-trajectory H4-legal anchor ranking, under balanced scene sampling rather than real-world operation frequencies.

| Panel | Unique windows | Independent trajectories | H1 motion | H4 endpoint moving | H4 path any moving |
|---|---:|---:|---|---:|---:|
| natural | 162 | 162 | {'moving': 100, 'near_static': 50, 'intermediate': 12} | 113 | 113 |
| extra | 157 | 157 | {'near_static': 49, 'intermediate': 18, 'moving': 90} | 116 | 115 |
| combined | 319 | 162 | {'moving': 190, 'near_static': 99, 'intermediate': 30} | 229 | 228 |
| development | 36 | 32 | {'moving': 31, 'intermediate': 1, 'near_static': 4} | 32 | 32 |

Missing requested slots: 5; exact reasons in indices.json. No category replacements.
H1 conditional labels cross categories. Each (trajectory,anchor) is inferred once despite multiple labels/panels. H4 endpoint and four-step path labels remain separate.
Distinct windows can share states/transitions. Exact correlated pairs and per-trajectory support are in temporal_overlap; no independent-window claim.
Complete64 controls: anchors3..60; every history/future transition a−3..a+3 must be physically valid. No padding or cross-trajectory history.
Development is a subset of these seen-letter development-validation indices, not an unbiased final test. No models, optimizers or training sampling iterators were run.

Index sourceSHA256 `787fe2a3f19162e622b363942072b5b45f0f24d23009aa6de4b819c4c6beae15`.
