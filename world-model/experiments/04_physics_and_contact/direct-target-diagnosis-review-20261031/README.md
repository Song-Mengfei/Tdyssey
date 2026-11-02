# direct-target-diagnosis-review-20261031

Category: Controller variants, strict original physics and contact data.
Local source profile: `Tdyssey-raw-tl-portable-main`. Recorded status: **analysis_or_preparation; not a training result**.

This directory is a historical source/evidence archive. Packaging did not run models, training, collection, physics, a server, or a GUI. Completion is separate from scientific acceptance. See the cited reports for failed gates and unknown measurements.

## Contents

- [Source index](SOURCE_INDEX.json): 4 source entries, including deduplicated references.
- [Evidence index](EVIDENCE_INDEX.json): 9 selected configurations/reports/measurements.

## Reproduction status

Historical worker files depend on the original `artifacts/<run>` layout, pinned source files, datasets, checkpoint lineage and frozen runtime evidence. They are preserved for inspection, not advertised as a one-command reproduction. Paths were normalized for publication; original and public hashes are both recorded. Source-hash bindings must be regenerated for a new execution. No archived approval or execution claim is valid authorization.

Use `tools/materialize_sources.py` to reconstruct the first-party source layout in a new external directory. Supply the private assets listed in the evidence and [asset guide](../../../docs/ASSETS.md). Consult [reproduction](../../../docs/REPRODUCIBILITY.md) before selecting the appropriate protocol.

## Selected source files

- [contact/analyze_saved_contact.py](source/contact/analyze_saved_contact.py)
- [contact/write_contact_review.py](source/contact/write_contact_review.py)
- [exposure/analyze_exposure.py](source/exposure/analyze_exposure.py)
- [geometry/analyze_geometry.py](source/geometry/analyze_geometry.py)

## Selected evidence files

- [CHANGES.review.json](evidence/CHANGES.review.json)
- [OUTPUT_MANIFEST.json](evidence/OUTPUT_MANIFEST.json)
- [SOURCE_MANIFEST.json](evidence/SOURCE_MANIFEST.json)
- [contact/INPUT_MANIFEST.json](evidence/contact/INPUT_MANIFEST.json)
- [contact/MINIMAL_VERIFICATION.review.json](evidence/contact/MINIMAL_VERIFICATION.review.json)
- [contact/OUTPUT_MANIFEST.json](evidence/contact/OUTPUT_MANIFEST.json)
- [exposure/EXPOSURE_SUMMARY.json](evidence/exposure/EXPOSURE_SUMMARY.json)
- [geometry/input-manifest.json](evidence/geometry/input-manifest.json)
- [geometry/output-manifest.json](evidence/geometry/output-manifest.json)
