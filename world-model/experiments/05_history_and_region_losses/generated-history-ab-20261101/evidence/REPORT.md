# Four-step generated-history A/B: saved paired results

Common start: OC new-data epoch2. A uses true history for all four targets; B feeds back its current detached predictions for the next losses. The primary contrast is B versus A.

Supervised targets total: 60544; optimizer batches total: 3784; physical steps: 0.
Supervision is matched. B has additional generation computation; compare actual timing/accounting in summary.json.

## Formal48: all six letters

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 48 | 0.981 / 0.991 / 0.951 | 0.089 / 0.061 / 0.155 | 0.031 / 0.016 / 0.084 | 42 | 48 | 0 |
| baseline | 4 | 48 | 0.850 / 0.874 / 0.715 | 0.438 / 0.326 / 0.899 | 0.283 / 0.246 / 0.537 | 39 | 48 | 0 |
| baseline | 8 | 48 | 0.751 / 0.747 / 0.514 | 0.855 / 0.670 / 1.822 | 0.529 / 0.489 / 0.967 | 43 | 48 | 0 |
| baseline | 16 | 48 | 0.631 / 0.658 / 0.322 | 1.819 / 1.310 / 4.337 | 0.918 / 0.843 / 1.738 | 36 | 48 | 0 |
| A | 1 | 48 | 0.983 / 0.991 / 0.948 | 0.079 / 0.058 / 0.153 | 0.028 / 0.017 / 0.086 | 42 | 48 | 0 |
| A | 4 | 48 | 0.863 / 0.894 / 0.727 | 0.397 / 0.281 / 0.747 | 0.256 / 0.181 / 0.475 | 39 | 48 | 0 |
| A | 8 | 48 | 0.764 / 0.787 / 0.529 | 0.817 / 0.639 / 1.827 | 0.502 / 0.353 / 1.041 | 43 | 48 | 0 |
| A | 16 | 48 | 0.641 / 0.708 / 0.292 | 1.708 / 0.875 / 3.908 | 0.869 / 0.650 / 1.723 | 37 | 48 | 0 |
| B | 1 | 48 | 0.986 / 0.992 / 0.963 | 0.068 / 0.051 / 0.112 | 0.023 / 0.012 / 0.058 | 42 | 48 | 0 |
| B | 4 | 48 | 0.894 / 0.914 / 0.778 | 0.299 / 0.182 / 0.596 | 0.199 / 0.162 / 0.417 | 39 | 48 | 0 |
| B | 8 | 48 | 0.803 / 0.815 / 0.622 | 0.666 / 0.385 / 1.554 | 0.420 / 0.306 / 0.883 | 43 | 48 | 0 |
| B | 16 | 48 | 0.687 / 0.752 / 0.350 | 1.397 / 0.784 / 3.705 | 0.746 / 0.521 / 1.481 | 37 | 48 | 0 |
| copy_last_reference | 1 | 48 | 0.986 / 1.000 / 0.956 | 0.066 / 0.000 / 0.179 | 0.025 / 0.000 / 0.077 | 39 | 48 | 0 |
| copy_last_reference | 4 | 48 | 0.748 / 0.805 / 0.429 | 1.126 / 0.721 / 2.918 | 0.550 / 0.414 / 1.290 | 42 | 48 | 0 |
| copy_last_reference | 8 | 48 | 0.545 / 0.479 / 0.153 | 3.132 / 2.567 / 7.759 | 1.419 / 1.121 / 3.265 | 43 | 48 | 0 |
| copy_last_reference | 16 | 48 | 0.442 / 0.297 / 0.100 | 6.268 / 4.624 / 15.887 | 2.896 / 1.874 / 7.211 | 42 | 48 | 0 |

## Primary paired comparison: B versus A

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 48 | 26 / 14 / 8 / 0 | 20 / 13 / 15 / 0 | 23 / 16 / 9 / 0 | 9 |
| B_vs_A | 4 | 48 | 37 / 4 / 7 / 0 | 34 / 3 / 11 / 0 | 35 / 6 / 7 / 0 | 10 |
| B_vs_A | 8 | 48 | 38 / 2 / 8 / 0 | 31 / 1 / 16 / 0 | 37 / 2 / 9 / 0 | 11 |
| B_vs_A | 16 | 48 | 37 / 1 / 10 / 0 | 31 / 1 / 16 / 0 | 37 / 1 / 10 / 0 | 14 |

## O_strong4

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 4 | 0.988 / 0.989 / 0.981 | 0.048 / 0.048 / 0.055 | 0.018 / 0.018 / 0.028 | 4 | 4 | 0 |
| baseline | 4 | 4 | 0.776 / 0.777 / 0.691 | 0.779 / 0.824 / 1.173 | 0.419 / 0.418 / 0.590 | 4 | 4 | 0 |
| baseline | 8 | 4 | 0.639 / 0.630 / 0.533 | 1.538 / 1.599 / 2.252 | 0.720 / 0.755 / 0.945 | 4 | 4 | 0 |
| baseline | 16 | 4 | 0.532 / 0.500 / 0.380 | 2.386 / 2.020 / 3.833 | 0.901 / 0.904 / 1.279 | 3 | 4 | 0 |
| A | 1 | 4 | 0.988 / 0.989 / 0.981 | 0.048 / 0.048 / 0.055 | 0.018 / 0.018 / 0.028 | 4 | 4 | 0 |
| A | 4 | 4 | 0.777 / 0.788 / 0.668 | 0.737 / 0.624 / 1.258 | 0.412 / 0.391 / 0.625 | 4 | 4 | 0 |
| A | 8 | 4 | 0.618 / 0.607 / 0.452 | 1.702 / 1.518 / 3.030 | 0.762 / 0.806 / 1.087 | 4 | 4 | 0 |
| A | 16 | 4 | 0.494 / 0.442 / 0.315 | 2.917 / 2.901 / 4.655 | 0.997 / 1.058 / 1.444 | 3 | 4 | 0 |
| B | 1 | 4 | 0.986 / 0.991 / 0.973 | 0.040 / 0.045 / 0.048 | 0.022 / 0.015 / 0.043 | 4 | 4 | 0 |
| B | 4 | 4 | 0.806 / 0.846 / 0.690 | 0.519 / 0.337 / 1.073 | 0.363 / 0.285 / 0.595 | 4 | 4 | 0 |
| B | 8 | 4 | 0.634 / 0.638 / 0.435 | 1.582 / 1.283 / 3.076 | 0.735 / 0.750 / 1.152 | 4 | 4 | 0 |
| B | 16 | 4 | 0.496 / 0.489 / 0.270 | 2.892 / 2.833 / 4.756 | 1.025 / 1.025 / 1.553 | 3 | 4 | 0 |
| copy_last_reference | 1 | 4 | 0.993 / 0.998 / 0.983 | 0.022 / 0.020 / 0.045 | 0.009 / 0.002 / 0.023 | 3 | 4 | 0 |
| copy_last_reference | 4 | 4 | 0.615 / 0.618 / 0.397 | 2.004 / 1.863 / 3.601 | 0.709 / 0.686 / 1.113 | 4 | 4 | 0 |
| copy_last_reference | 8 | 4 | 0.337 / 0.270 / 0.141 | 5.433 / 5.721 / 8.821 | 1.959 / 1.916 / 3.136 | 4 | 4 | 0 |
| copy_last_reference | 16 | 4 | 0.211 / 0.132 / 0.104 | 9.846 / 9.496 / 15.876 | 3.992 / 3.517 / 6.884 | 3 | 4 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 4 | 1 / 2 / 1 / 0 | 1 / 2 / 1 / 0 | 1 / 2 / 1 / 0 | 0 |
| B_vs_A | 4 | 4 | 3 / 0 / 1 / 0 | 4 / 0 / 0 / 0 | 3 / 0 / 1 / 0 | 1 |
| B_vs_A | 8 | 4 | 3 / 0 / 1 / 0 | 2 / 0 / 2 / 0 | 3 / 0 / 1 / 0 | 1 |
| B_vs_A | 16 | 4 | 2 / 0 / 2 / 0 | 2 / 0 / 2 / 0 | 2 / 0 / 2 / 0 | 0 |

## C_strong4

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 4 | 0.984 / 0.995 / 0.965 | 0.084 / 0.066 / 0.129 | 0.025 / 0.007 / 0.058 | 3 | 4 | 0 |
| baseline | 4 | 4 | 0.682 / 0.635 / 0.490 | 0.445 / 0.519 / 0.651 | 0.577 / 0.631 / 0.958 | 3 | 4 | 0 |
| baseline | 8 | 4 | 0.479 / 0.487 / 0.251 | 1.096 / 0.849 / 1.745 | 1.336 / 1.144 / 2.288 | 4 | 4 | 0 |
| baseline | 16 | 4 | 0.276 / 0.236 / 0.133 | 3.863 / 3.880 / 5.052 | 2.549 / 2.635 / 3.864 | 3 | 4 | 0 |
| A | 1 | 4 | 0.984 / 0.995 / 0.965 | 0.084 / 0.066 / 0.129 | 0.025 / 0.007 / 0.058 | 3 | 4 | 0 |
| A | 4 | 4 | 0.695 / 0.678 / 0.485 | 0.470 / 0.567 / 0.659 | 0.559 / 0.553 / 0.988 | 3 | 4 | 0 |
| A | 8 | 4 | 0.457 / 0.506 / 0.205 | 1.394 / 0.885 / 2.513 | 1.453 / 1.023 / 2.650 | 4 | 4 | 0 |
| A | 16 | 4 | 0.208 / 0.209 / 0.099 | 4.225 / 4.146 / 5.630 | 2.704 / 2.769 / 3.883 | 3 | 4 | 0 |
| B | 1 | 4 | 0.989 / 0.995 / 0.975 | 0.103 / 0.066 / 0.216 | 0.017 / 0.006 / 0.041 | 3 | 4 | 0 |
| B | 4 | 4 | 0.778 / 0.802 / 0.577 | 0.472 / 0.272 / 1.045 | 0.379 / 0.300 / 0.761 | 3 | 4 | 0 |
| B | 8 | 4 | 0.548 / 0.631 / 0.249 | 1.415 / 0.864 / 2.783 | 1.222 / 0.638 / 2.546 | 4 | 4 | 0 |
| B | 16 | 4 | 0.289 / 0.243 / 0.087 | 3.597 / 3.426 / 5.435 | 2.316 / 2.401 / 3.680 | 3 | 4 | 0 |
| copy_last_reference | 1 | 4 | 0.987 / 0.995 / 0.968 | 0.057 / 0.067 / 0.089 | 0.022 / 0.008 / 0.053 | 3 | 4 | 0 |
| copy_last_reference | 4 | 4 | 0.489 / 0.398 / 0.222 | 1.817 / 1.907 / 3.027 | 1.179 / 1.033 / 2.212 | 3 | 4 | 0 |
| copy_last_reference | 8 | 4 | 0.140 / 0.118 / 0.103 | 6.039 / 5.849 / 8.712 | 3.420 / 3.561 / 4.390 | 4 | 4 | 0 |
| copy_last_reference | 16 | 4 | 0.117 / 0.121 / 0.108 | 12.303 / 11.668 / 14.199 | 5.544 / 5.335 / 6.350 | 4 | 4 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 4 | 2 / 2 / 0 / 0 | 1 / 2 / 1 / 0 | 2 / 2 / 0 / 0 | 1 |
| B_vs_A | 4 | 4 | 3 / 1 / 0 / 0 | 2 / 1 / 1 / 0 | 3 / 1 / 0 / 0 | 1 |
| B_vs_A | 8 | 4 | 3 / 0 / 1 / 0 | 1 / 0 / 3 / 0 | 3 / 0 / 1 / 0 | 2 |
| B_vs_A | 16 | 4 | 3 / 0 / 1 / 0 | 3 / 0 / 1 / 0 | 4 / 0 / 0 / 0 | 2 |

## B

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 8 | 0.985 / 0.992 / 0.959 | 0.060 / 0.049 / 0.125 | 0.026 / 0.014 / 0.067 | 8 | 8 | 0 |
| baseline | 4 | 8 | 0.847 / 0.844 / 0.771 | 0.467 / 0.343 / 0.908 | 0.314 / 0.300 / 0.511 | 8 | 8 | 0 |
| baseline | 8 | 8 | 0.743 / 0.704 / 0.640 | 0.951 / 0.985 / 1.612 | 0.530 / 0.593 / 0.770 | 8 | 8 | 0 |
| baseline | 16 | 8 | 0.574 / 0.480 / 0.345 | 2.480 / 2.551 / 4.737 | 0.917 / 0.984 / 1.625 | 8 | 8 | 0 |
| A | 1 | 8 | 0.983 / 0.992 / 0.955 | 0.059 / 0.040 / 0.121 | 0.031 / 0.017 / 0.077 | 8 | 8 | 0 |
| A | 4 | 8 | 0.859 / 0.869 / 0.773 | 0.418 / 0.311 / 0.876 | 0.288 / 0.248 / 0.485 | 8 | 8 | 0 |
| A | 8 | 8 | 0.746 / 0.707 / 0.618 | 1.016 / 1.008 / 1.827 | 0.518 / 0.602 / 0.832 | 8 | 8 | 0 |
| A | 16 | 8 | 0.570 / 0.572 / 0.264 | 2.558 / 1.975 / 5.038 | 0.943 / 0.830 / 1.701 | 8 | 8 | 0 |
| B | 1 | 8 | 0.986 / 0.991 / 0.969 | 0.057 / 0.045 / 0.107 | 0.027 / 0.018 / 0.054 | 8 | 8 | 0 |
| B | 4 | 8 | 0.873 / 0.859 / 0.790 | 0.373 / 0.316 / 0.684 | 0.270 / 0.271 / 0.493 | 8 | 8 | 0 |
| B | 8 | 8 | 0.768 / 0.720 / 0.637 | 0.853 / 0.948 / 1.603 | 0.473 / 0.547 / 0.814 | 8 | 8 | 0 |
| B | 16 | 8 | 0.625 / 0.646 / 0.364 | 2.178 / 1.404 / 4.479 | 0.871 / 0.692 / 1.632 | 8 | 8 | 0 |
| copy_last_reference | 1 | 8 | 0.973 / 0.998 / 0.919 | 0.156 / 0.022 / 0.449 | 0.056 / 0.001 / 0.166 | 7 | 8 | 0 |
| copy_last_reference | 4 | 8 | 0.754 / 0.821 / 0.488 | 1.338 / 0.665 / 3.041 | 0.513 / 0.349 / 1.096 | 7 | 8 | 0 |
| copy_last_reference | 8 | 8 | 0.582 / 0.566 / 0.270 | 3.643 / 2.075 / 8.448 | 1.184 / 0.822 / 2.545 | 7 | 8 | 0 |
| copy_last_reference | 16 | 8 | 0.416 / 0.323 / 0.072 | 7.383 / 4.808 / 17.630 | 3.122 / 1.571 / 8.083 | 7 | 8 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 8 | 4 / 3 / 1 / 0 | 2 / 3 / 3 / 0 | 4 / 3 / 1 / 0 | 2 |
| B_vs_A | 4 | 8 | 6 / 0 / 2 / 0 | 4 / 0 / 4 / 0 | 5 / 1 / 2 / 0 | 2 |
| B_vs_A | 8 | 8 | 6 / 0 / 2 / 0 | 5 / 0 / 3 / 0 | 6 / 0 / 2 / 0 | 3 |
| B_vs_A | 16 | 8 | 7 / 0 / 1 / 0 | 5 / 0 / 3 / 0 | 7 / 0 / 1 / 0 | 2 |

## T

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 8 | 0.987 / 0.992 / 0.975 | 0.075 / 0.077 / 0.113 | 0.028 / 0.019 / 0.056 | 7 | 8 | 0 |
| baseline | 4 | 8 | 0.932 / 0.927 / 0.885 | 0.249 / 0.268 / 0.477 | 0.151 / 0.154 / 0.242 | 6 | 8 | 0 |
| baseline | 8 | 8 | 0.843 / 0.880 / 0.714 | 0.854 / 0.514 / 1.665 | 0.394 / 0.278 / 0.736 | 7 | 8 | 0 |
| baseline | 16 | 8 | 0.754 / 0.811 / 0.531 | 1.213 / 0.634 / 2.546 | 0.696 / 0.469 / 1.423 | 6 | 8 | 0 |
| A | 1 | 8 | 0.991 / 0.994 / 0.983 | 0.051 / 0.037 / 0.110 | 0.019 / 0.014 / 0.036 | 7 | 8 | 0 |
| A | 4 | 8 | 0.939 / 0.937 / 0.900 | 0.254 / 0.231 / 0.447 | 0.135 / 0.129 / 0.224 | 6 | 8 | 0 |
| A | 8 | 8 | 0.873 / 0.925 / 0.760 | 0.646 / 0.270 / 1.364 | 0.329 / 0.155 / 0.666 | 7 | 8 | 0 |
| A | 16 | 8 | 0.788 / 0.824 / 0.630 | 0.877 / 0.559 / 1.757 | 0.558 / 0.444 / 1.024 | 6 | 8 | 0 |
| B | 1 | 8 | 0.990 / 0.994 / 0.975 | 0.050 / 0.049 / 0.080 | 0.023 / 0.014 / 0.057 | 7 | 8 | 0 |
| B | 4 | 8 | 0.956 / 0.973 / 0.898 | 0.160 / 0.084 / 0.377 | 0.100 / 0.071 / 0.217 | 6 | 8 | 0 |
| B | 8 | 8 | 0.913 / 0.947 / 0.828 | 0.434 / 0.279 / 0.931 | 0.224 / 0.112 / 0.469 | 7 | 8 | 0 |
| B | 16 | 8 | 0.842 / 0.906 / 0.708 | 0.730 / 0.379 / 1.479 | 0.400 / 0.206 / 0.765 | 6 | 8 | 0 |
| copy_last_reference | 1 | 8 | 0.997 / 1.000 / 0.993 | 0.034 / 0.000 / 0.090 | 0.005 / 0.000 / 0.014 | 7 | 8 | 0 |
| copy_last_reference | 4 | 8 | 0.778 / 0.792 / 0.525 | 0.990 / 0.662 / 2.179 | 0.619 / 0.557 / 1.390 | 7 | 8 | 0 |
| copy_last_reference | 8 | 8 | 0.618 / 0.580 / 0.267 | 2.206 / 2.357 / 4.257 | 1.317 / 1.221 / 2.709 | 7 | 8 | 0 |
| copy_last_reference | 16 | 8 | 0.493 / 0.305 / 0.147 | 4.647 / 4.102 / 11.380 | 2.399 / 2.542 / 4.894 | 7 | 8 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 8 | 2 / 4 / 2 / 0 | 2 / 3 / 3 / 0 | 2 / 4 / 2 / 0 | 0 |
| B_vs_A | 4 | 8 | 6 / 0 / 2 / 0 | 7 / 0 / 1 / 0 | 6 / 0 / 2 / 0 | 1 |
| B_vs_A | 8 | 8 | 6 / 0 / 2 / 0 | 6 / 0 / 2 / 0 | 6 / 0 / 2 / 0 | 0 |
| B_vs_A | 16 | 8 | 5 / 1 / 2 / 0 | 4 / 1 / 3 / 0 | 6 / 1 / 1 / 0 | 3 |

## L

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 8 | 0.983 / 0.993 / 0.965 | 0.079 / 0.058 / 0.151 | 0.035 / 0.012 / 0.076 | 7 | 8 | 0 |
| baseline | 4 | 8 | 0.902 / 0.922 / 0.812 | 0.281 / 0.160 / 0.593 | 0.214 / 0.155 / 0.461 | 6 | 8 | 0 |
| baseline | 8 | 8 | 0.841 / 0.820 / 0.732 | 0.541 / 0.432 / 1.142 | 0.361 / 0.370 / 0.670 | 7 | 8 | 0 |
| baseline | 16 | 8 | 0.696 / 0.643 / 0.526 | 1.280 / 1.481 / 2.359 | 0.796 / 0.913 / 1.308 | 4 | 8 | 0 |
| A | 1 | 8 | 0.985 / 0.993 / 0.970 | 0.065 / 0.055 / 0.109 | 0.028 / 0.012 / 0.058 | 7 | 8 | 0 |
| A | 4 | 8 | 0.922 / 0.934 / 0.843 | 0.226 / 0.203 / 0.475 | 0.165 / 0.130 / 0.367 | 6 | 8 | 0 |
| A | 8 | 8 | 0.872 / 0.881 / 0.769 | 0.456 / 0.347 / 0.985 | 0.283 / 0.227 / 0.583 | 7 | 8 | 0 |
| A | 16 | 8 | 0.762 / 0.755 / 0.570 | 0.855 / 0.644 / 2.093 | 0.591 / 0.589 / 1.119 | 4 | 8 | 0 |
| B | 1 | 8 | 0.989 / 0.993 / 0.973 | 0.051 / 0.044 / 0.112 | 0.022 / 0.012 / 0.054 | 7 | 8 | 0 |
| B | 4 | 8 | 0.937 / 0.957 / 0.853 | 0.174 / 0.120 / 0.354 | 0.132 / 0.080 / 0.313 | 6 | 8 | 0 |
| B | 8 | 8 | 0.889 / 0.908 / 0.785 | 0.454 / 0.281 / 1.189 | 0.245 / 0.172 / 0.553 | 7 | 8 | 0 |
| B | 16 | 8 | 0.811 / 0.833 / 0.601 | 0.547 / 0.217 / 1.283 | 0.469 / 0.354 / 1.066 | 5 | 8 | 0 |
| copy_last_reference | 1 | 8 | 0.992 / 0.998 / 0.981 | 0.040 / 0.007 / 0.100 | 0.017 / 0.002 / 0.040 | 7 | 8 | 0 |
| copy_last_reference | 4 | 8 | 0.771 / 0.854 / 0.497 | 1.053 / 0.520 / 2.918 | 0.571 / 0.347 / 1.313 | 7 | 8 | 0 |
| copy_last_reference | 8 | 8 | 0.590 / 0.581 / 0.199 | 2.969 / 1.997 / 7.896 | 1.657 / 1.178 / 3.565 | 7 | 8 | 0 |
| copy_last_reference | 16 | 8 | 0.512 / 0.422 / 0.095 | 5.383 / 2.628 / 15.081 | 3.074 / 1.701 / 7.818 | 7 | 8 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 8 | 4 / 2 / 2 / 0 | 3 / 2 / 3 / 0 | 3 / 3 / 2 / 0 | 1 |
| B_vs_A | 4 | 8 | 5 / 2 / 1 / 0 | 5 / 1 / 2 / 0 | 4 / 3 / 1 / 0 | 1 |
| B_vs_A | 8 | 8 | 5 / 2 / 1 / 0 | 4 / 1 / 3 / 0 | 5 / 2 / 1 / 0 | 2 |
| B_vs_A | 16 | 8 | 6 / 0 / 2 / 0 | 6 / 0 / 2 / 0 | 5 / 0 / 3 / 0 | 3 |

## X

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 8 | 0.977 / 0.991 / 0.954 | 0.101 / 0.050 / 0.208 | 0.033 / 0.015 / 0.066 | 7 | 8 | 0 |
| baseline | 4 | 8 | 0.788 / 0.777 / 0.715 | 0.659 / 0.725 / 0.858 | 0.366 / 0.381 / 0.524 | 7 | 8 | 0 |
| baseline | 8 | 8 | 0.666 / 0.653 / 0.497 | 0.914 / 1.122 / 1.293 | 0.602 / 0.602 / 0.936 | 8 | 8 | 0 |
| baseline | 16 | 8 | 0.533 / 0.449 / 0.322 | 2.221 / 2.520 / 3.772 | 1.069 / 1.102 / 1.806 | 7 | 8 | 0 |
| A | 1 | 8 | 0.982 / 0.991 / 0.963 | 0.100 / 0.046 / 0.207 | 0.025 / 0.012 / 0.046 | 7 | 8 | 0 |
| A | 4 | 8 | 0.812 / 0.789 / 0.751 | 0.572 / 0.591 / 0.769 | 0.317 / 0.370 / 0.434 | 7 | 8 | 0 |
| A | 8 | 8 | 0.689 / 0.716 / 0.527 | 0.750 / 0.728 / 1.213 | 0.547 / 0.476 / 0.874 | 8 | 8 | 0 |
| A | 16 | 8 | 0.548 / 0.511 / 0.316 | 1.919 / 1.887 / 3.663 | 0.984 / 0.939 / 1.725 | 7 | 8 | 0 |
| B | 1 | 8 | 0.987 / 0.989 / 0.971 | 0.085 / 0.051 / 0.173 | 0.016 / 0.013 / 0.035 | 7 | 8 | 0 |
| B | 4 | 8 | 0.863 / 0.854 / 0.785 | 0.360 / 0.364 / 0.493 | 0.224 / 0.241 / 0.354 | 7 | 8 | 0 |
| B | 8 | 8 | 0.752 / 0.749 / 0.626 | 0.474 / 0.497 / 0.784 | 0.437 / 0.449 / 0.720 | 8 | 8 | 0 |
| B | 16 | 8 | 0.573 / 0.581 / 0.391 | 1.343 / 1.213 / 2.493 | 0.853 / 0.839 / 1.279 | 7 | 8 | 0 |
| copy_last_reference | 1 | 8 | 0.993 / 1.000 / 0.981 | 0.036 / 0.000 / 0.089 | 0.010 / 0.000 / 0.027 | 7 | 8 | 0 |
| copy_last_reference | 4 | 8 | 0.730 / 0.743 / 0.494 | 1.085 / 0.838 / 2.451 | 0.475 / 0.464 / 0.861 | 8 | 8 | 0 |
| copy_last_reference | 8 | 8 | 0.449 / 0.379 / 0.180 | 3.278 / 2.688 / 6.973 | 1.301 / 1.121 / 2.446 | 8 | 8 | 0 |
| copy_last_reference | 16 | 8 | 0.329 / 0.210 / 0.081 | 7.408 / 5.479 / 16.481 | 3.382 / 2.872 / 7.501 | 8 | 8 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 8 | 5 / 1 / 2 / 0 | 5 / 1 / 2 / 0 | 4 / 2 / 2 / 0 | 2 |
| B_vs_A | 4 | 8 | 7 / 0 / 1 / 0 | 7 / 0 / 1 / 0 | 7 / 0 / 1 / 0 | 2 |
| B_vs_A | 8 | 8 | 7 / 0 / 1 / 0 | 7 / 0 / 1 / 0 | 6 / 0 / 2 / 0 | 1 |
| B_vs_A | 16 | 8 | 6 / 0 / 2 / 0 | 6 / 0 / 2 / 0 | 5 / 0 / 3 / 0 | 1 |

## frozen_bitstatic

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 6 | 0.994 / 0.994 / 0.992 | 0.050 / 0.057 / 0.066 | 0.011 / 0.009 / 0.020 | 5 | 6 | 0 |
| baseline | 4 | 6 | 0.981 / 0.980 / 0.972 | 0.064 / 0.052 / 0.115 | 0.036 / 0.037 / 0.056 | 5 | 6 | 0 |
| baseline | 8 | 6 | 0.971 / 0.973 / 0.955 | 0.104 / 0.100 / 0.170 | 0.053 / 0.049 / 0.088 | 5 | 6 | 0 |
| baseline | 16 | 6 | 0.947 / 0.951 / 0.926 | 0.135 / 0.122 / 0.174 | 0.099 / 0.100 / 0.133 | 5 | 6 | 0 |
| A | 1 | 6 | 0.993 / 0.994 / 0.989 | 0.058 / 0.047 / 0.100 | 0.012 / 0.010 / 0.021 | 5 | 6 | 0 |
| A | 4 | 6 | 0.983 / 0.985 / 0.971 | 0.066 / 0.059 / 0.132 | 0.033 / 0.030 / 0.061 | 5 | 6 | 0 |
| A | 8 | 6 | 0.971 / 0.968 / 0.957 | 0.127 / 0.119 / 0.195 | 0.058 / 0.051 / 0.097 | 5 | 6 | 0 |
| A | 16 | 6 | 0.949 / 0.951 / 0.926 | 0.141 / 0.154 / 0.177 | 0.097 / 0.099 / 0.135 | 5 | 6 | 0 |
| B | 1 | 6 | 0.995 / 0.995 / 0.992 | 0.042 / 0.040 / 0.066 | 0.008 / 0.008 / 0.014 | 5 | 6 | 0 |
| B | 4 | 6 | 0.985 / 0.987 / 0.978 | 0.064 / 0.065 / 0.091 | 0.030 / 0.028 / 0.045 | 5 | 6 | 0 |
| B | 8 | 6 | 0.979 / 0.978 / 0.970 | 0.097 / 0.102 / 0.161 | 0.039 / 0.038 / 0.061 | 5 | 6 | 0 |
| B | 16 | 6 | 0.962 / 0.965 / 0.942 | 0.121 / 0.137 / 0.176 | 0.070 / 0.069 / 0.107 | 5 | 6 | 0 |
| copy_last_reference | 1 | 6 | 1.000 / 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 4 | 6 | 0 |
| copy_last_reference | 4 | 6 | 1.000 / 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 4 | 6 | 0 |
| copy_last_reference | 8 | 6 | 1.000 / 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 4 | 6 | 0 |
| copy_last_reference | 16 | 6 | 1.000 / 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 4 | 6 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 6 | 3 / 2 / 1 / 0 | 3 / 2 / 1 / 0 | 3 / 2 / 1 / 0 | 0 |
| B_vs_A | 4 | 6 | 3 / 1 / 2 / 0 | 3 / 1 / 2 / 0 | 3 / 1 / 2 / 0 | 2 |
| B_vs_A | 8 | 6 | 5 / 1 / 0 / 0 | 5 / 0 / 1 / 0 | 6 / 0 / 0 / 0 | 1 |
| B_vs_A | 16 | 6 | 5 / 1 / 0 / 0 | 4 / 1 / 1 / 0 | 5 / 1 / 0 / 0 | 1 |

## Motion diagnostic32: train / real_history

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 16 | 0.813 / 0.871 / 0.670 | 0.498 / 0.363 / 1.113 | 0.349 / 0.221 / 0.642 | 16 | 16 | 0 |
| baseline | 4 | 16 | 0.919 / 0.917 / 0.902 | 0.204 / 0.200 / 0.302 | 0.131 / 0.120 / 0.178 | 16 | 16 | 0 |
| baseline | 8 | 16 | 0.947 / 0.953 / 0.892 | 0.169 / 0.123 / 0.331 | 0.087 / 0.087 / 0.178 | 11 | 16 | 0 |
| baseline | 16 | 16 | 0.984 / 0.992 / 0.945 | 0.054 / 0.049 / 0.102 | 0.027 / 0.018 / 0.085 | 7 | 16 | 0 |
| A | 1 | 16 | 0.826 / 0.868 / 0.703 | 0.478 / 0.362 / 1.000 | 0.321 / 0.215 / 0.574 | 16 | 16 | 0 |
| A | 4 | 16 | 0.926 / 0.926 / 0.899 | 0.207 / 0.185 / 0.326 | 0.118 / 0.118 / 0.167 | 16 | 16 | 0 |
| A | 8 | 16 | 0.950 / 0.964 / 0.898 | 0.190 / 0.143 / 0.343 | 0.083 / 0.068 / 0.160 | 10 | 16 | 0 |
| A | 16 | 16 | 0.982 / 0.990 / 0.943 | 0.075 / 0.079 / 0.142 | 0.031 / 0.023 / 0.097 | 7 | 16 | 0 |
| B | 1 | 16 | 0.814 / 0.848 / 0.687 | 0.567 / 0.354 / 1.162 | 0.342 / 0.255 / 0.610 | 16 | 16 | 0 |
| B | 4 | 16 | 0.907 / 0.898 / 0.878 | 0.267 / 0.246 / 0.429 | 0.147 / 0.146 / 0.217 | 16 | 16 | 0 |
| B | 8 | 16 | 0.938 / 0.944 / 0.887 | 0.188 / 0.171 / 0.364 | 0.101 / 0.081 / 0.172 | 10 | 16 | 0 |
| B | 16 | 16 | 0.981 / 0.991 / 0.947 | 0.073 / 0.055 / 0.170 | 0.031 / 0.016 / 0.087 | 7 | 16 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 16 | 6 / 0 / 10 / 0 | 4 / 0 / 12 / 0 | 4 / 0 / 12 / 0 | 6 |
| B_vs_A | 4 | 16 | 3 / 1 / 12 / 0 | 4 / 0 / 12 / 0 | 2 / 1 / 13 / 0 | 3 |
| B_vs_A | 8 | 16 | 2 / 1 / 13 / 0 | 9 / 0 / 7 / 0 | 3 / 1 / 12 / 0 | 9 |
| B_vs_A | 16 | 16 | 5 / 9 / 2 / 0 | 8 / 6 / 2 / 0 | 7 / 7 / 2 / 0 | 0 |

## Motion diagnostic32: train / autoregressive

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 16 | 0.813 / 0.871 / 0.670 | 0.498 / 0.363 / 1.113 | 0.349 / 0.221 / 0.642 | 16 | 16 | 0 |
| baseline | 4 | 16 | 0.607 / 0.610 / 0.473 | 1.529 / 1.487 / 2.296 | 0.724 / 0.701 / 0.968 | 16 | 16 | 0 |
| baseline | 8 | 16 | 0.447 / 0.451 / 0.240 | 2.895 / 2.514 / 5.250 | 1.187 / 0.974 / 1.781 | 13 | 16 | 0 |
| baseline | 16 | 16 | 0.391 / 0.411 / 0.105 | 3.895 / 2.609 / 8.571 | 1.625 / 1.260 / 3.359 | 11 | 16 | 0 |
| A | 1 | 16 | 0.826 / 0.868 / 0.703 | 0.478 / 0.362 / 1.000 | 0.321 / 0.215 / 0.574 | 16 | 16 | 0 |
| A | 4 | 16 | 0.636 / 0.609 / 0.523 | 1.272 / 1.229 / 1.944 | 0.664 / 0.696 / 0.896 | 16 | 16 | 0 |
| A | 8 | 16 | 0.481 / 0.496 / 0.241 | 2.624 / 2.059 / 5.335 | 1.105 / 0.912 / 1.918 | 12 | 16 | 0 |
| A | 16 | 16 | 0.412 / 0.362 / 0.099 | 3.749 / 2.756 / 9.112 | 1.629 / 1.227 / 3.833 | 7 | 16 | 0 |
| B | 1 | 16 | 0.814 / 0.848 / 0.687 | 0.567 / 0.354 / 1.162 | 0.342 / 0.255 / 0.610 | 16 | 16 | 0 |
| B | 4 | 16 | 0.629 / 0.618 / 0.502 | 1.410 / 1.252 / 2.273 | 0.687 / 0.690 / 0.910 | 16 | 16 | 0 |
| B | 8 | 16 | 0.501 / 0.522 / 0.248 | 2.470 / 1.892 / 5.408 | 1.049 / 0.896 / 1.792 | 12 | 16 | 0 |
| B | 16 | 16 | 0.452 / 0.470 / 0.110 | 3.473 / 2.284 / 9.197 | 1.584 / 0.957 / 3.950 | 8 | 16 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 16 | 6 / 0 / 10 / 0 | 4 / 0 / 12 / 0 | 4 / 0 / 12 / 0 | 6 |
| B_vs_A | 4 | 16 | 9 / 0 / 7 / 0 | 8 / 0 / 8 / 0 | 7 / 0 / 9 / 0 | 4 |
| B_vs_A | 8 | 16 | 8 / 0 / 8 / 0 | 9 / 0 / 7 / 0 | 8 / 0 / 8 / 0 | 4 |
| B_vs_A | 16 | 16 | 11 / 0 / 5 / 0 | 10 / 0 / 6 / 0 | 9 / 0 / 7 / 0 | 3 |

## Motion diagnostic32: seen_val / real_history

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 16 | 0.908 / 0.918 / 0.853 | 0.253 / 0.206 / 0.478 | 0.146 / 0.114 / 0.239 | 16 | 16 | 0 |
| baseline | 4 | 16 | 0.944 / 0.945 / 0.909 | 0.153 / 0.154 / 0.249 | 0.081 / 0.082 / 0.123 | 13 | 16 | 0 |
| baseline | 8 | 16 | 0.962 / 0.961 / 0.924 | 0.106 / 0.095 / 0.193 | 0.057 / 0.065 / 0.105 | 12 | 16 | 0 |
| baseline | 16 | 16 | 0.978 / 0.991 / 0.935 | 0.087 / 0.065 / 0.197 | 0.035 / 0.011 / 0.100 | 9 | 16 | 0 |
| A | 1 | 16 | 0.913 / 0.918 / 0.851 | 0.270 / 0.240 / 0.543 | 0.136 / 0.106 / 0.243 | 16 | 16 | 0 |
| A | 4 | 16 | 0.952 / 0.959 / 0.915 | 0.158 / 0.142 / 0.242 | 0.069 / 0.068 / 0.114 | 13 | 16 | 0 |
| A | 8 | 16 | 0.963 / 0.962 / 0.923 | 0.093 / 0.073 / 0.210 | 0.057 / 0.059 / 0.107 | 12 | 16 | 0 |
| A | 16 | 16 | 0.978 / 0.990 / 0.943 | 0.067 / 0.050 / 0.157 | 0.036 / 0.013 / 0.088 | 9 | 16 | 0 |
| B | 1 | 16 | 0.893 / 0.901 / 0.835 | 0.311 / 0.243 / 0.635 | 0.168 / 0.147 / 0.261 | 16 | 16 | 0 |
| B | 4 | 16 | 0.939 / 0.948 / 0.906 | 0.142 / 0.121 / 0.235 | 0.089 / 0.073 / 0.140 | 13 | 16 | 0 |
| B | 8 | 16 | 0.958 / 0.957 / 0.922 | 0.096 / 0.099 / 0.163 | 0.061 / 0.067 / 0.097 | 12 | 16 | 0 |
| B | 16 | 16 | 0.971 / 0.980 / 0.928 | 0.087 / 0.070 / 0.197 | 0.045 / 0.038 / 0.100 | 9 | 16 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 16 | 1 / 0 / 15 / 0 | 4 / 0 / 12 / 0 | 2 / 0 / 14 / 0 | 6 |
| B_vs_A | 4 | 16 | 5 / 1 / 10 / 0 | 8 / 0 / 8 / 0 | 6 / 1 / 9 / 0 | 6 |
| B_vs_A | 8 | 16 | 5 / 2 / 9 / 0 | 7 / 1 / 8 / 0 | 6 / 2 / 8 / 0 | 3 |
| B_vs_A | 16 | 16 | 5 / 3 / 8 / 0 | 5 / 3 / 8 / 0 | 5 / 3 / 8 / 0 | 2 |

## Motion diagnostic32: seen_val / autoregressive

| Arm | H | n | IoU mean / median / P10 | Position mean / median / P90 px | Contour mean / median / P90 px | Shape unknown | Angle unknown | Blue missing |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 16 | 0.908 / 0.918 / 0.853 | 0.253 / 0.206 / 0.478 | 0.146 / 0.114 / 0.239 | 16 | 16 | 0 |
| baseline | 4 | 16 | 0.757 / 0.779 / 0.668 | 0.720 / 0.611 / 1.233 | 0.421 / 0.337 / 0.640 | 14 | 16 | 0 |
| baseline | 8 | 16 | 0.674 / 0.700 / 0.521 | 1.141 / 1.024 / 2.075 | 0.582 / 0.509 / 0.879 | 12 | 16 | 0 |
| baseline | 16 | 16 | 0.590 / 0.577 / 0.441 | 1.593 / 1.346 / 2.664 | 0.790 / 0.827 / 1.081 | 11 | 16 | 0 |
| A | 1 | 16 | 0.913 / 0.918 / 0.851 | 0.270 / 0.240 / 0.543 | 0.136 / 0.106 / 0.243 | 16 | 16 | 0 |
| A | 4 | 16 | 0.781 / 0.797 / 0.707 | 0.676 / 0.551 / 0.990 | 0.365 / 0.311 / 0.501 | 14 | 16 | 0 |
| A | 8 | 16 | 0.715 / 0.712 / 0.633 | 0.862 / 0.741 / 1.362 | 0.489 / 0.473 / 0.648 | 12 | 16 | 0 |
| A | 16 | 16 | 0.647 / 0.657 / 0.456 | 1.139 / 0.916 / 2.233 | 0.624 / 0.625 / 0.906 | 11 | 16 | 0 |
| B | 1 | 16 | 0.893 / 0.901 / 0.835 | 0.311 / 0.243 / 0.635 | 0.168 / 0.147 / 0.261 | 16 | 16 | 0 |
| B | 4 | 16 | 0.777 / 0.795 / 0.640 | 0.698 / 0.610 / 1.470 | 0.367 / 0.285 / 0.637 | 14 | 16 | 0 |
| B | 8 | 16 | 0.727 / 0.715 / 0.617 | 0.912 / 0.730 / 1.541 | 0.457 / 0.501 / 0.655 | 12 | 16 | 0 |
| B | 16 | 16 | 0.668 / 0.667 / 0.511 | 1.319 / 1.152 / 2.135 | 0.585 / 0.609 / 0.848 | 11 | 16 | 0 |

| Comparison | H | n | IoU better / tied / worse / unknown | Position better / tied / worse / unknown | Contour better / tied / worse / unknown | Mixed image directions |
|---|---|---|---|---|---|---|
| B_vs_A | 1 | 16 | 1 / 0 / 15 / 0 | 4 / 0 / 12 / 0 | 2 / 0 / 14 / 0 | 6 |
| B_vs_A | 4 | 16 | 6 / 0 / 10 / 0 | 6 / 0 / 10 / 0 | 7 / 0 / 9 / 0 | 5 |
| B_vs_A | 8 | 16 | 8 / 0 / 8 / 0 | 5 / 0 / 11 / 0 | 8 / 0 / 8 / 0 | 3 |
| B_vs_A | 16 | 16 | 9 / 0 / 7 / 0 | 5 / 0 / 11 / 0 | 9 / 0 / 7 / 0 | 4 |

## Interpretation limits

- The primary method comparison is B versus A. Beating only the shared epoch2 start cannot establish generated-history benefit.
- A/B each receive the same30272 supervised targets,7568 segments and1892 optimizer batches. B additionally generates22704 online historical images (68112 image-level denoiser forwards at3steps); this is not equal compute.
- Only one A/B run is approved. The shared epoch2 start was selected using development evidence; these are exposed seen letters and seen-val families, not an untouched final test.
- Copy-last reference scores are reused where actual saved scores exist; absent horizons are not invented. They are an additional retention/reference baseline, never a replacement for the trained A/B contrast.
- All image statistics are from raw saved outputs. Display clipping/quantization is viewing only and is never fed back or used to repair shape.
- Raw IoU, contour and centroid errors mix pose and shape. Hole counts describe visible masks; zero visible-hole difference cannot establish hole preservation under occlusion.
- Numeric angular registration is not angle assessability. Angle statistics and paired angle changes require an explicit angle_assessable=true; all other numeric angles remain an unvalidated diagnostic and unknown for conclusions.
- Signed underrotation is not inferred from absolute visible-mask angular registration. The dedicated underrotation measurement remains unknown unless explicitly declared assessable.
- Shape-aligned IoU is summarized only for explicitly assessable cases. Occlusion, clipping, symmetry and deformation unknowns do not count as passing.
- Real-history images at t+16 are separate one-step predictions with four true prior frames for that target. They are not sixteen-step closed-loop forecasts. AR16 starts from four true frames then feeds back raw predictions.
- The32movement cases are deliberately motion-biased and preserve separate train/seen-val groups; late target frames can have smaller motion than the chosen initial transition.
- Each per-case metric is classified by its signed change with1e-8 numerical tie tolerance. These signs are not a statistical significance or physical-validity test. Mixed image direction uses only IoU/centroid/contour, not an overall model-success label.
- Four-step generated-history training still retains one real frame at its fourth loss. Its truncated no-grad generated history is not full trajectory backpropagation and does not guarantee H16.
- No new physical data, new loss, architecture, server execution, second A/B pass or unapproved extra training is included.

## Same-case raw comparisons

- formal48 seen_val T autoregressive (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-T-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-T-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-T-10fps.mp4`)
- formal48 seen_val L autoregressive (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-L-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-L-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-L-10fps.mp4`)
- formal48 seen_val O autoregressive (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-O-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-O-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-O-10fps.mp4`)
- formal48 seen_val C autoregressive (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-C-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-C-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-C-10fps.mp4`)
- formal48 seen_val X autoregressive (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-X-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-X-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-X-10fps.mp4`)
- formal48 seen_val B autoregressive (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-B-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-B-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\formal-B-10fps.mp4`)
- diagnostic32 train O real_history (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-O-real_history-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-O-real_history-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-O-real_history-10fps.mp4`)
- diagnostic32 train O autoregressive (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-O-autoregressive-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-O-autoregressive-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-O-autoregressive-10fps.mp4`)
- diagnostic32 train C real_history (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-C-real_history-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-C-real_history-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-C-real_history-10fps.mp4`)
- diagnostic32 train C autoregressive (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-C-autoregressive-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-C-autoregressive-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-train-C-autoregressive-10fps.mp4`)
- diagnostic32 seen_val O real_history (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-O-real_history-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-O-real_history-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-O-real_history-10fps.mp4`)
- diagnostic32 seen_val O autoregressive (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-O-autoregressive-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-O-autoregressive-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-O-autoregressive-10fps.mp4`)
- diagnostic32 seen_val C real_history (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-C-real_history-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-C-real_history-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-C-real_history-10fps.mp4`)
- diagnostic32 seen_val C autoregressive (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-C-autoregressive-H1-H4-H8-H16.png`) · all16 (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-C-autoregressive-all-H1-H16.png`) · 10fps video (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\diagnostic-seen_val-C-autoregressive-10fps.mp4`)

All1..16 grouped metrics and paired counts (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\summary.json`) · Every paired case/metric (external historical reference: `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\generated-history-ab-20261101\report\case-comparisons.json`)
