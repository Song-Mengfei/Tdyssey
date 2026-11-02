# Staged lossless microstate storage helper

Status: implemented and tested, **not integrated into the running collector**.
No collector, report, physics or UI file was changed by this task. No simulator,
model, GPU or optimizer was imported/executed. Only saved T data were read.

## Interface

`tdyssey.original_contact_pilot_storage` supplies:

- `pack_telemetry(telemetry, decisions)` → storage-only NumPy arrays suitable for
  merging into the existing `np.savez_compressed` payload.
- `unpack_telemetry(arrays)` → original `telemetry` and `decisions`, accepting both
  the legacy zlib archive and V2. It verifies the original canonical JSON SHA.
- `load_attempt_archive(path)` → the existing public result shape: primary arrays
  plus telemetry/decisions, with all storage-only keys removed. Always uses
  `allow_pickle=False`.
- `TELEMETRY_STORAGE_KEYS` → the complete legacy/V2 storage-key list.

The only prospective collector integration is to replace its two zlib storage
fields with `**pack_telemetry(...)` and make `load_attempt` delegate to
`load_attempt_archive(path)`. Do this only after the current phase finishes and
record the new source/format binding. The helper does not change storage guards,
resource allocation, trajectories, policy actions or physical execution.

## Exactness contract

The legacy sanitizer runs first. Only a complete microstep triple with all three
15-element finite **exact Python float** lists is extracted. Integer/mixed/bool,
missing, malformed and sanitized nonfinite rows stay in JSON. Each distinct
float64 vector is keyed by its exact bytes, retaining negative zero. Identical
vectors may share a reference, but reconstruction creates independent lists.

Both vector/reference arrays use column-major `.npy` order to improve deflate
compression. Logical shapes/dtypes and numeric values are unchanged. No rounding,
state deletion, frame deletion, reduced precision or geometry approximation is
used. Other contacts, impulses, interventions and controller/decision fields
remain unchanged JSON.

The compact JSON hash is checked, reference locations/types/shapes are checked,
and then reconstructed canonical JSON must match the **original payload SHA**.
This also detects corruption of extracted float values. Legacy archives still
check their original payload hash directly. No pickle/object arrays are accepted.

## Actual saved-T comparison

Source: completed `OPC-calibration-v1-calibration-T-000/raw.npz`; original file SHA
before/after is unchanged. This is one existing trajectory, not a future replay.

| Format | Actual bytes | Relative to legacy |
| --- | ---: | ---: |
| Existing JSON/zlib NPZ | 2,673,917 | 100% |
| V2 row-major experiment | 2,545,761 | 95.21% |
| Final V2 column-major layout | 2,436,299 | 91.11% |

The final sample saves **237,618 bytes, 8.89%**. This is a modest saving: it does
**not** make a 360-trajectory batch fit the previous512MiB operational allowance.
Use the actual18-calibration distribution and existing shared3GiB/256MiB reserve
to decide resource allocation. Do not extrapolate a single T sample into a
guaranteed dataset size. Keeping this helper staged and using a sufficient
explicit resource allocation is reasonable if integration complexity or CPU cost
outweighs the saving for this pilot.

All64controls/12,800microsteps were retained. All38,400 state fields were compared
bitwise; primary RGB/actions/control-valid/state arrays retain exact dtype, shape
and bytes. Original and reconstructed canonical payload SHA both equal
`21ad33632312daedb6ef77066ece678220ab9c17e0830f7108a9b74d7c7a20aa`.

Six pure tests pass, covering float extremes/subnormals/signed zero, integer/type
preservation, missing/ragged/nonfinite rows, legacy loading, `allow_pickle=False`,
independent aliases, reserved-looking user data and multiple corruption modes.
The final helper was repacked in memory from the compared column-major archive;
all storage-array dtypes/shapes/bytes matched exactly.

Evidence: `comparison.json`, `layout-comparison.json`, `final-check.json`,
`tests.txt` and the two small comparison NPZs. The row-major experiment/source
snapshot is preserved, not overwritten. Root owns any later cleanup of these
comparison copies; this task deleted no project asset.

Measured CPU-only timings: initial whole T comparison13.532s (legacy decode.782s,
pack3.473s, reconstruct2.495s), column-major layout comparison3.109s, final helper
check5.718s including six tests. Reconstructing and verifying canonical JSON costs
CPU; no speedup claim is made.

Final helper SHA256:
`be65023bf655e57d2cb30ebb6a4f0bc67d5bcdc116f1528069da9fbcdd6b2218`.
Test SHA256:
`014776915069e26e68fdf67f523eb50515918a0d326c7b20a1f644d5654369a3`.


## Not adopted; alternate archive copies removed

The owner chose to keep the established raw format:8.89% storage savings do not justify extra encode/decode CPU and a format migration. The two alternate comparison NPZ copies were safely removed after exact equality checks; original calibration raw remains. Their historical hashes/results and reproducible helper/tests remain; see decision-and-deletion-summary.json and actual deletion receipt. No claim that removed comparison files still exist.
