# Loss definitions and experiment lineage

## Native residual MSE

For target RGB `y`, noisy target `z`, network output `F` and diffusion preconditioning:

```text
z = y + 0.1 * channel_offset + sigma * epsilon
residual_target = (y - c_skip * z) / c_out
L_native = mean_BCHW((F - residual_target)^2)
D = c_skip * z + c_out * F
```

The residual is a diffusion-preconditioning residual, not temporal motion. Equivalently the reconstructed-image error is weighted by `1/c_out^2`. For the recorded effective sigma, the weight is `4 + 1/(sigma_draw^2 + 0.01)`. Later experiments preserve the original native branch unless explicitly testing replacement.

## Comparison families

| Family | Actual change | Where |
| --- | --- | --- |
| A/B history | True history vs detached generated history; native H1 MSE | Category 05, generated-history runs |
| C/D region/detail | Foreground/region/detail weighting, with history variants | Category 05, region-weighted runs |
| High-motion fitting | Training exposure/sampling and duration, native objective retained | Category 06; not a loss-only comparison |
| AR4 auxiliary | Native loss plus four-step RGB reconstruction through actual sampler forward and declared STE derivative | Category 07; failed original preflight and repaired R1 retained |
| Nominal Min-SNR | Effective-noise per-sample weighting of native residual MSE | Category 08; not a new spatial definition |
| GDL | Adjacent image-gradient-magnitude error on reconstructed image | Category 08; not signed contour-distance loss |
| FFL | Detached focal weighting of complex orthonormal FFT error | Category 08; not a Fourier shape-descriptor model |
| G_ONLY / NATIVE_G | Squared-denominator Dice on RGB-derived gray/blue occupancy; pure replacement vs native anchor | Category 09 |
| NATIVE_T | Native plus fixed 16-direction spatial W1 and relative area error | Category 09; finite-direction W1 adaptation, not full 2-D OT/Sinkhorn |

## Tested new definitions

Palette unmixing uses known white/gray/blue RGB vectors, a fixed least-squares inverse and a clamp to [0,1]. No new segmentation network, hidden silhouette, privileged future pose or output repair is added. Equal mask values can correspond to different RGB values, and strict clamp saturation has zero gradient.

```text
G = mean_classes[1 - (2 sum(p*g) + eps) / (sum(p^2) + sum(g^2) + eps)]
T = mean_classes[mean_16_directions(W1(projected normalized spatial mass))
                 + abs(sum(p)-sum(g))/(sum(g)+eps)]
eps = 1e-6
```

The new definitions act on continuous single-denoising reconstruction `D`, not a full sampled rollout. Fixed coefficients were calibrated from the first 16 training examples' complete parameter-gradient norms: `cG=0.023378236655937423`, `cT=0.00003850309057337518`. All three arms restored the same full E2 state and trained for 128 updates. A small scalar coefficient is not evidence of a negligible gradient.

The literature archive distinguishes abstract screening, method inspection and deep equation/code checks. Boundary distance, Hausdorff, topology, rigid alignment, learned flow/features and value-aware alternatives are covered, but only G/T were trained in the final screen. Diagnostic boundary/clDice/moment computations are not trained-model results. See [results](RESULTS.md) and the original categorized review for limits.
