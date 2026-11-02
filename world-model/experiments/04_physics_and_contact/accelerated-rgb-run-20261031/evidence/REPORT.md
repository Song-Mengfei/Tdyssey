# One new-data RGB epoch and paired predictions

Full epoch: True; unique windows 30935; optimizer successes 1934; AMP skips 0.
Independent seen-validation families: 48. Horizons1/4/8/16 share anchors/actions/noise across before/after. Copy-last uses the real anchor.

New original-physics data, six exposed development letters only; no unseen-letter or policy improvement claim

Visible gray/blue masks only; occlusion/viewport ambiguity retained. Hole/corner counts are diagnostics, not proof of rigid shapes. No privileged state enters predictions.

Only PNG/video display clips/quantizes; lossless saved rawfloat32 feeds recursion and scoring without repair.

| Method | H | Gray IoU | Gray centroid error(px) | Blue centroid error(px) | Missing blue / denominator |
|---|---:|---:|---:|---:|---:|
| before | 1 | 0.9635 | 0.1208 | 0.2351 | 0/48 |
| before | 4 | 0.7638 | 0.8535 | 0.8985 | 0/48 |
| before | 8 | 0.6273 | 1.7420 | 1.2626 | 0/48 |
| before | 16 | 0.4968 | 3.3690 | 1.6468 | 0/48 |
| after | 1 | 0.9771 | 0.0875 | 0.1876 | 0/48 |
| after | 4 | 0.8240 | 0.5673 | 0.3467 | 0/48 |
| after | 8 | 0.7137 | 1.1009 | 0.3478 | 0/48 |
| after | 16 | 0.5972 | 2.1323 | 0.3956 | 0/48 |
| copy_last | 1 | 0.9860 | 0.0661 | 0.3272 | 0/48 |
| copy_last | 4 | 0.7483 | 1.1256 | 2.7990 | 0/48 |
| copy_last | 8 | 0.5453 | 3.1322 | 6.7995 | 0/48 |
| copy_last | 16 | 0.4423 | 6.2682 | 14.3706 | 0/48 |

T comparison (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\accelerated-rgb-run-20261031\report\T-H1-H4-H8-H16.png`)

L comparison (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\accelerated-rgb-run-20261031\report\L-H1-H4-H8-H16.png`)

O comparison (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\accelerated-rgb-run-20261031\report\O-H1-H4-H8-H16.png`)

C comparison (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\accelerated-rgb-run-20261031\report\C-H1-H4-H8-H16.png`)

X comparison (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\accelerated-rgb-run-20261031\report\X-H1-H4-H8-H16.png`)

B comparison (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\accelerated-rgb-run-20261031\report\B-H1-H4-H8-H16.png`)
