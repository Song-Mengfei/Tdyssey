# O/C continuation: new-data epochs 1 / 2 / 3

Additional exposures: 61870; successful updates: 3868; AMP skips: 0.
Epoch1 formal predictions were reused from AR1. Epoch2/3 use the same cases/actions/noise and formal three-denoise sampler.

## Formal48: all six seen letters

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 48 | 0.977 (48) | 0.087 (48) | 3.812 (48) | 0.038 (48) | 0.000 (48) | 0 | 41 |
| 1 | 4 | 48 | 0.824 (48) | 0.567 (48) | 27.958 (48) | 0.334 (48) | 0.000 (48) | 0 | 39 |
| 1 | 8 | 48 | 0.714 (48) | 1.101 (48) | 23.688 (48) | 0.629 (48) | 0.000 (48) | 0 | 43 |
| 1 | 16 | 48 | 0.597 (48) | 2.132 (48) | 46.604 (48) | 0.998 (48) | 0.000 (48) | 0 | 37 |
| 2 | 1 | 48 | 0.981 (48) | 0.089 (48) | 3.708 (48) | 0.031 (48) | 0.000 (48) | 0 | 42 |
| 2 | 4 | 48 | 0.850 (48) | 0.438 (48) | 20.604 (48) | 0.283 (48) | 0.000 (48) | 0 | 39 |
| 2 | 8 | 48 | 0.751 (48) | 0.855 (48) | 18.875 (48) | 0.529 (48) | 0.000 (48) | 0 | 43 |
| 2 | 16 | 48 | 0.631 (48) | 1.819 (48) | 43.521 (48) | 0.918 (48) | 0.000 (48) | 0 | 36 |
| 3 | 1 | 48 | 0.982 (48) | 0.069 (48) | 3.729 (48) | 0.031 (48) | 0.000 (48) | 0 | 42 |
| 3 | 4 | 48 | 0.856 (48) | 0.407 (48) | 12.938 (48) | 0.271 (48) | 0.000 (48) | 0 | 38 |
| 3 | 8 | 48 | 0.754 (48) | 0.821 (48) | 29.229 (48) | 0.517 (48) | 0.000 (48) | 0 | 43 |
| 3 | 16 | 48 | 0.623 (48) | 1.768 (48) | 26.604 (48) | 0.904 (48) | 0.042 (48) | 0 | 37 |
| copy_last | 1 | 48 | 0.986 (48) | 0.066 (48) | 3.771 (48) | 0.025 (48) | 0.000 (48) | 0 | 39 |
| copy_last | 4 | 48 | 0.748 (48) | 1.126 (48) | 24.750 (48) | 0.550 (48) | 0.000 (48) | 0 | 42 |
| copy_last | 8 | 48 | 0.545 (48) | 3.132 (48) | 40.417 (48) | 1.419 (48) | 0.000 (48) | 0 | 43 |
| copy_last | 16 | 48 | 0.442 (48) | 6.268 (48) | 45.271 (48) | 2.896 (48) | 0.000 (48) | 0 | 42 |

## Per-letter H16, including regressions


### T

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 8 | 0.738 (8) | 1.171 (8) | 4.625 (8) | 0.706 (8) | 0.000 (8) | 0 | 7 |
| 2 | 16 | 8 | 0.754 (8) | 1.213 (8) | 5.125 (8) | 0.696 (8) | 0.000 (8) | 0 | 6 |
| 3 | 16 | 8 | 0.774 (8) | 0.864 (8) | 4.500 (8) | 0.591 (8) | 0.000 (8) | 0 | 6 |

### L

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 8 | 0.722 (8) | 1.222 (8) | 3.000 (8) | 0.697 (8) | 0.000 (8) | 0 | 4 |
| 2 | 16 | 8 | 0.696 (8) | 1.280 (8) | 2.625 (8) | 0.796 (8) | 0.000 (8) | 0 | 4 |
| 3 | 16 | 8 | 0.749 (8) | 0.910 (8) | 2.375 (8) | 0.621 (8) | 0.000 (8) | 0 | 5 |

### O

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 8 | 0.572 (8) | 2.056 (8) | 113.875 (8) | 0.805 (8) | 0.000 (8) | 0 | 7 |
| 2 | 16 | 8 | 0.641 (8) | 1.659 (8) | 80.125 (8) | 0.677 (8) | 0.000 (8) | 0 | 7 |
| 3 | 16 | 8 | 0.624 (8) | 1.812 (8) | 59.125 (8) | 0.746 (8) | 0.000 (8) | 0 | 7 |

### C

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 8 | 0.558 (8) | 2.576 (8) | 16.625 (8) | 1.453 (8) | 0.000 (8) | 0 | 4 |
| 2 | 16 | 8 | 0.586 (8) | 2.061 (8) | 13.875 (8) | 1.355 (8) | 0.000 (8) | 0 | 4 |
| 3 | 16 | 8 | 0.525 (8) | 2.262 (8) | 17.125 (8) | 1.462 (8) | 0.000 (8) | 0 | 4 |

### X

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 8 | 0.485 (8) | 2.834 (8) | 72.000 (8) | 1.286 (8) | 0.000 (8) | 0 | 7 |
| 2 | 16 | 8 | 0.533 (8) | 2.221 (8) | 113.125 (8) | 1.069 (8) | 0.000 (8) | 0 | 7 |
| 3 | 16 | 8 | 0.533 (8) | 2.045 (8) | 71.750 (8) | 1.012 (8) | 0.000 (8) | 0 | 7 |

### B

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 8 | 0.509 (8) | 2.934 (8) | 69.500 (8) | 1.043 (8) | 0.000 (8) | 0 | 8 |
| 2 | 16 | 8 | 0.574 (8) | 2.480 (8) | 46.250 (8) | 0.917 (8) | 0.000 (8) | 0 | 8 |
| 3 | 16 | 8 | 0.531 (8) | 2.714 (8) | 4.750 (8) | 0.995 (8) | 0.250 (8) | 0 | 8 |

## Fixed strong O/C subset (eight original seen-val cases)

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8 | 0.987 (8) | 0.060 (8) | 0.000 (8) | 0.021 (8) | 0.000 (8) | 0 | 6 |
| 1 | 4 | 8 | 0.669 (8) | 0.888 (8) | 27.750 (8) | 0.598 (8) | 0.000 (8) | 0 | 7 |
| 1 | 8 | 8 | 0.479 (8) | 1.922 (8) | 18.750 (8) | 1.358 (8) | 0.000 (8) | 0 | 8 |
| 1 | 16 | 8 | 0.347 (8) | 3.829 (8) | 65.000 (8) | 1.876 (8) | 0.000 (8) | 0 | 6 |
| 2 | 1 | 8 | 0.986 (8) | 0.066 (8) | 0.000 (8) | 0.021 (8) | 0.000 (8) | 0 | 7 |
| 2 | 4 | 8 | 0.729 (8) | 0.612 (8) | 26.875 (8) | 0.498 (8) | 0.000 (8) | 0 | 7 |
| 2 | 8 | 8 | 0.559 (8) | 1.317 (8) | 33.000 (8) | 1.028 (8) | 0.000 (8) | 0 | 8 |
| 2 | 16 | 8 | 0.404 (8) | 3.124 (8) | 49.500 (8) | 1.725 (8) | 0.000 (8) | 0 | 6 |
| 3 | 1 | 8 | 0.985 (8) | 0.061 (8) | 0.000 (8) | 0.023 (8) | 0.000 (8) | 0 | 7 |
| 3 | 4 | 8 | 0.724 (8) | 0.587 (8) | 5.000 (8) | 0.504 (8) | 0.000 (8) | 0 | 7 |
| 3 | 8 | 8 | 0.513 (8) | 1.570 (8) | 35.375 (8) | 1.137 (8) | 0.000 (8) | 0 | 8 |
| 3 | 16 | 8 | 0.332 (8) | 3.576 (8) | 32.875 (8) | 1.896 (8) | 0.000 (8) | 0 | 6 |

## Movement diagnostics: real history versus AR16

Each table keeps train and seen-val separate. Real-history t+16 is a separate one-step prediction using its own true preceding history; it is not a successful closed-loop H16.


### train / real_history

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 16 | 0.797 (16) | 0.569 (16) | 45.250 (16) | 0.373 (16) | 0.000 (16) | 0 | 16 |
| 1 | 4 | 16 | 0.907 (16) | 0.235 (16) | 11.688 (16) | 0.144 (16) | 0.000 (16) | 0 | 16 |
| 1 | 8 | 16 | 0.940 (16) | 0.201 (16) | 0.000 (16) | 0.095 (16) | 0.000 (16) | 0 | 11 |
| 1 | 16 | 16 | 0.978 (16) | 0.095 (16) | 0.125 (16) | 0.035 (16) | 0.000 (16) | 0 | 7 |
| 3 | 1 | 16 | 0.824 (16) | 0.509 (16) | 34.500 (16) | 0.326 (16) | 0.000 (16) | 0 | 16 |
| 3 | 4 | 16 | 0.925 (16) | 0.188 (16) | 11.312 (16) | 0.118 (16) | 0.000 (16) | 0 | 16 |
| 3 | 8 | 16 | 0.947 (16) | 0.211 (16) | 0.000 (16) | 0.085 (16) | 0.000 (16) | 0 | 10 |
| 3 | 16 | 16 | 0.981 (16) | 0.105 (16) | 0.000 (16) | 0.031 (16) | 0.000 (16) | 0 | 7 |

### train / autoregressive

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 16 | 0.797 (16) | 0.569 (16) | 45.250 (16) | 0.373 (16) | 0.000 (16) | 0 | 16 |
| 1 | 4 | 16 | 0.603 (16) | 1.641 (16) | 57.312 (16) | 0.726 (16) | 0.000 (16) | 0 | 16 |
| 1 | 8 | 16 | 0.455 (16) | 2.983 (16) | 87.375 (16) | 1.174 (16) | 0.000 (16) | 0 | 12 |
| 1 | 16 | 16 | 0.393 (16) | 4.335 (16) | 52.375 (16) | 1.806 (16) | 0.000 (16) | 0 | 9 |
| 3 | 1 | 16 | 0.824 (16) | 0.509 (16) | 34.500 (16) | 0.326 (16) | 0.000 (16) | 0 | 16 |
| 3 | 4 | 16 | 0.638 (16) | 1.275 (16) | 67.750 (16) | 0.659 (16) | 0.000 (16) | 0 | 16 |
| 3 | 8 | 16 | 0.496 (16) | 2.450 (16) | 78.375 (16) | 0.993 (16) | 0.000 (16) | 0 | 14 |
| 3 | 16 | 16 | 0.425 (16) | 3.356 (16) | 41.062 (16) | 1.368 (16) | 0.000 (16) | 0 | 9 |

### seen_val / real_history

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 16 | 0.899 (16) | 0.286 (16) | 11.875 (16) | 0.160 (16) | 0.000 (16) | 0 | 16 |
| 1 | 4 | 16 | 0.935 (16) | 0.167 (16) | 0.125 (16) | 0.096 (16) | 0.000 (16) | 0 | 13 |
| 1 | 8 | 16 | 0.957 (16) | 0.089 (16) | 0.000 (16) | 0.063 (16) | 0.000 (16) | 0 | 12 |
| 1 | 16 | 16 | 0.969 (16) | 0.109 (16) | 0.000 (16) | 0.050 (16) | 0.000 (16) | 0 | 9 |
| 3 | 1 | 16 | 0.911 (16) | 0.263 (16) | 11.562 (16) | 0.140 (16) | 0.000 (16) | 0 | 16 |
| 3 | 4 | 16 | 0.950 (16) | 0.160 (16) | 0.250 (16) | 0.071 (16) | 0.000 (16) | 0 | 13 |
| 3 | 8 | 16 | 0.961 (16) | 0.085 (16) | 0.125 (16) | 0.059 (16) | 0.000 (16) | 0 | 12 |
| 3 | 16 | 16 | 0.976 (16) | 0.079 (16) | 0.000 (16) | 0.037 (16) | 0.000 (16) | 0 | 9 |

### seen_val / autoregressive

| Epoch | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 16 | 0.899 (16) | 0.286 (16) | 11.875 (16) | 0.160 (16) | 0.000 (16) | 0 | 16 |
| 1 | 4 | 16 | 0.736 (16) | 0.736 (16) | 35.688 (16) | 0.449 (16) | 0.000 (16) | 0 | 15 |
| 1 | 8 | 16 | 0.660 (16) | 1.116 (16) | 27.000 (16) | 0.592 (16) | 0.000 (16) | 0 | 12 |
| 1 | 16 | 16 | 0.570 (16) | 1.711 (16) | 50.688 (16) | 0.803 (16) | 0.000 (16) | 0 | 11 |
| 3 | 1 | 16 | 0.911 (16) | 0.263 (16) | 11.562 (16) | 0.140 (16) | 0.000 (16) | 0 | 16 |
| 3 | 4 | 16 | 0.767 (16) | 0.675 (16) | 35.750 (16) | 0.390 (16) | 0.000 (16) | 0 | 14 |
| 3 | 8 | 16 | 0.697 (16) | 0.840 (16) | 24.438 (16) | 0.516 (16) | 0.000 (16) | 0 | 12 |
| 3 | 16 | 16 | 0.626 (16) | 1.193 (16) | 56.688 (16) | 0.654 (16) | 0.000 (16) | 0 | 9 |

## Epoch1 inference-only: three versus sixteen denoising steps

The original three-step outputs are reused. This changes inference, not trained weights. Original per-case noise is injected; batch-one versus original batch-four rounding remains a computational caveat.

| Checkpoint / sampler | H | n | Gray IoU | Position px | Angle error deg (known n) | Contour px | Hole count diff | Blue missing | Shape unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| epoch1/denoise3 | 1 | 8 | 0.987 (8) | 0.060 (8) | 0.000 (8) | 0.021 (8) | 0.000 (8) | 0 | 6 |
| epoch1/denoise3 | 4 | 8 | 0.669 (8) | 0.888 (8) | 27.750 (8) | 0.598 (8) | 0.000 (8) | 0 | 7 |
| epoch1/denoise3 | 8 | 8 | 0.479 (8) | 1.922 (8) | 18.750 (8) | 1.358 (8) | 0.000 (8) | 0 | 8 |
| epoch1/denoise3 | 16 | 8 | 0.347 (8) | 3.829 (8) | 65.000 (8) | 1.876 (8) | 0.000 (8) | 0 | 6 |
| epoch1/denoise16 | 1 | 8 | 0.987 (8) | 0.045 (8) | 0.000 (8) | 0.019 (8) | 0.000 (8) | 0 | 6 |
| epoch1/denoise16 | 4 | 8 | 0.637 (8) | 0.996 (8) | 25.250 (8) | 0.660 (8) | 0.000 (8) | 0 | 7 |
| epoch1/denoise16 | 8 | 8 | 0.415 (8) | 2.479 (8) | 68.125 (8) | 1.592 (8) | 0.000 (8) | 0 | 8 |
| epoch1/denoise16 | 16 | 8 | 0.284 (8) | 4.676 (8) | 44.000 (8) | 2.235 (8) | 0.000 (8) | 0 | 6 |

## Fixed samples and fixed diffusion-noise residual loss

Train and seen-val fixed-sample measurements are saved separately. These objective values use matched cases/noise; ordinary random-batch loss alone is not evidence of convergence.

| Epoch | Split / letter | Fixed samples | Residual MSE |
|---|---|---:|---:|
| 1 | train | 16 | 0.00139060 |
| 1 | seen_val | 16 | 0.00076707 |
| 1 | train:O | 8 | 0.00130024 |
| 1 | train:C | 8 | 0.00148097 |
| 1 | seen_val:O | 8 | 0.00077337 |
| 1 | seen_val:C | 8 | 0.00076077 |
| 2 | train | 16 | 0.00133907 |
| 2 | seen_val | 16 | 0.00065084 |
| 2 | train:O | 8 | 0.00128164 |
| 2 | train:C | 8 | 0.00139650 |
| 2 | seen_val:O | 8 | 0.00068316 |
| 2 | seen_val:C | 8 | 0.00061852 |
| 3 | train | 16 | 0.00128330 |
| 3 | seen_val | 16 | 0.00062763 |
| 3 | train:O | 8 | 0.00120612 |
| 3 | train:C | 8 | 0.00136048 |
| 3 | seen_val:O | 8 | 0.00068058 |
| 3 | seen_val:C | 8 | 0.00057468 |
- Epoch 1: fixed-loss machine result (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\diagnostics\epoch1\fixed-loss.json`)
- Epoch 2: fixed-loss machine result (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\diagnostics\epoch2\fixed-loss.json`)
- Epoch 3: fixed-loss machine result (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\diagnostics\epoch3\fixed-loss.json`)

## Interpretation limits

- Only two additional full epochs were approved. No new physical sampling, architecture/loss/LR change, server or fourth epoch.
- Formal48 uses the unchanged seen-val cases, three-denoise sampler and recursive predictions. All six letters, including B regressions, remain in the tables.
- The O/C movement diagnostic deliberately favors recorded movement/contact. Training and seen-val are separate; this is not a natural-distribution or unseen-letter test.
- Real-history predictions each receive four prior true frames for that target time. Their t+16 image is a single-step diagnostic, never a sixteen-step closed-loop prediction.
- AR16 receives true history only at its initial anchor, then raw predictions, with no future truth feedback or shape repair.
- Position/IoU, angular error, raw contour and visible hole counts are distinct observations. Occlusion/symmetry/deformation can make angle or shape unknown; unknown is not a pass.
- Raw contour error mixes position, rotation and boundary shape. Corner counts or lower contour error alone cannot establish rigid-corner or hole preservation.
- Absolute angular alignment error alone cannot establish underrotation. A signed, assessable motion comparison is required; unavailable underrotation remains unknown.
- Three-vs-sixteen denoising is an inference-only comparison at the same epoch1 checkpoint on eight fixed O/C strong cases. It cannot be counted as training improvement or a new default.
- The sixteen-denoise diagnostic injects each case's original batch-four initial noise into a batch-one call. Batch-one versus batch-four floating-point kernel rounding may also differ; this is not a perfectly isolated causal estimate of denoising-step count.
- Display PNG/video clips and quantizes only for viewing. Saved raw floats and all existing machine metrics remain unchanged. No shape or penetration correction is claimed.
- These finite comparisons can support candidate causes; they do not prove model-capacity failure, convergence or a universal cure for autoregressive drift.

## Saved comparisons

- seen_val T: formal48_same_case (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\formal-T-epoch1-2-3.png`)
- seen_val L: formal48_same_case (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\formal-L-epoch1-2-3.png`)
- seen_val O: formal48_same_case (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\formal-O-epoch1-2-3.png`)
- seen_val C: formal48_same_case (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\formal-C-epoch1-2-3.png`)
- seen_val X: formal48_same_case (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\formal-X-epoch1-2-3.png`)
- seen_val B: formal48_same_case (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\formal-B-epoch1-2-3.png`)
- train O: motion_biased_diagnostic (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\diagnostic-train-O-true-vs-ar.png`)
- train C: motion_biased_diagnostic (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\diagnostic-train-C-true-vs-ar.png`)
- seen_val O: motion_biased_diagnostic (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\diagnostic-seen_val-O-true-vs-ar.png`)
- seen_val C: motion_biased_diagnostic (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\diagnostic-seen_val-C-true-vs-ar.png`)
- seen_val O: epoch1_inference_only (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\inference-only-O-epoch1-denoise3-vs16.png`)
- seen_val C: epoch1_inference_only (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\inference-only-C-epoch1-denoise3-vs16.png`)

Complete grouped machine report (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\oc-continuation-20261031\report\summary.json`)
