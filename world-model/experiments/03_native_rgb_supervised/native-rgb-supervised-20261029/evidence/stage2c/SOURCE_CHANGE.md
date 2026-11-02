# Stage2C source scope

The stage2B sampler remains exactly
8f84baa2a7a6f4a89ffb2191d09e4bf86acef211b25e6b80918169ea86fd8cd8.
Initialization ranking, complete1B ContinuousContactSampler, limits, phases,
distances, side signs, geometry and physical protocol do not change.

The common collector accepts an explicit seen_val split and names its counter
evidence seen_val_contact_controls_before. A separate2C runner starts both hit
and initial-attempt counters at zero, with a separate frozen oldTRAIN prior.
Its162 requests, seeds, gap quotas and execution order are frozen; every actual
control is charged before execution. A first exception stops the whole batch;
no replacement or extra probe is available. Training data stays immutable.

The NativeRGBDataset adapter dispatches only the new reference-manifest kind,
resolves original files within explicit cohort roots and verifies source NPZ,
metadata, collector snapshots, physical audits, protocol and split identities.
Legacy manifests retain their existing path. Rendering, getitem timing and
normalization remain byte-identical:5 saved CHW frames and4 executed actions,
with uint8 LRU and float windows on demand. Model/training code is unchanged.

The prephysics index policy is configs/native_rgb_stage2c_indices.json,
seed2026102903 and SHA
f55fad565e6cc013e165529f81f84cf1ef46a67985b2d3f1bdae391545f6c6a1.
It freezes natural/extra/development selection before any validation step.
Production indices require the same frozen policy, collector source and passed
physical audit. No model output or training iterator participates.

The new source snapshot contains37 files/inputs. Forty-six focused CPU/mock
checks cover counters, failed-cost accounting, split/hash protection, old data
compatibility, actual timing, H4 boundaries and deterministic index selection.
Pure geometric preflight does not execute a physical control. Old2A/2B files
are protected separately by review-protected-train-files.json; the old source
snapshots remain their original identity despite the allowed newer adapter.

Postcollection erratum: the offline coverage script's exact prior snapshot-name
comparison omitted source-snapshot/, so it rejected a correctly hash-bound prior.
No physics was running. The frozen37 source files remain unchanged. A separately
versioned offline-repair copy fixes that path plus repository discovery for its
artifact location; it retains the original SHA comparison and every counter/
coverage calculation. The old failure log, two-line diff, old/new source hashes
and successful0-control replay are preserved. Independent review checks the
repair rather than concealing it or rewriting prephysics provenance.
