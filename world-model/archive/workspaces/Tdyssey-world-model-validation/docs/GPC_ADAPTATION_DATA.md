# GPC adaptation data: preparation and isolation

This is data preparation and implementation evidence, based on clean
`7a807b3447fe0e3511b499a189f753d1d4831cc3`. It authorizes no collection or real
GPC/policy execution. No new model load, forward, prediction, training, physics,
reset, control, policy sampling, installation, download or permanent claim was
performed by this data work. Tiny NumPy fixtures are synthetic, not model output.

The current data manifest is deliberately **not ready**. The requested local
projects have no eligible complete official T/L collector catalogs or episode
payload filenames in the inspected `artifacts`, `examples` and `data` roots.
Missing training data and independent validation block a real adaptation start.
This is a scoped local inventory, not a claim about every machine or cache.

Local ignored evidence under
`artifacts/wm-l-adaptation-preparation-20261025/`:

- `data-audit.json`
  records metadata/source identities, availability, historical use and gaps.
- `split-proposal.json`
  is a valid preparation manifest with empty `catalogs` and `episodes`, fixed T1,
  and 316 reviewed full-state hashes. Its four readiness gaps remain explicit.
- `collection-plan.json`
  is metadata only; it creates no collection directory or episode.
- `before.json`
  seals 7,460 existing files / 547,169,548 bytes across 36 roots, including the
  preceding diagnosis and all six spent formal claims. Closing `after.json` and
  `reconciliation.json` confirm all 7,460 files, both checkpoints and six claims
  unchanged, with no added/missing/changed members in the protected roots.

## Existing sources and their limits

The historical official L pilot is documented as eight complete episodes, 800
controls and 354 contact controls, with source whole-episode membership 6/1/1.
Its published custom-CNN report records six training episodes / 600 controls /
275 contact controls, and one validation episode / 100 controls / 40 contact
controls. The historical catalog identity is
`d3b4b49bad025e4d235844857fecd8ad5d1963f00cd352416c9505bfc4bf42c6`.
The catalog and episode payloads were not found locally in this inventory, so
these are documentary counts, not a fresh verification of the original data.

That L validation episode selected the custom-CNN checkpoint and was used in
published diagnostics. It cannot become independent validation by changing its
name. The old report's `model_test_used=false` is retained as historical metadata;
this preparation did not open a held model-test record or image archive. Training
translation/rotation coverage is unknown without eligible episode records. It
must not be inferred from the CNN's validation prediction errors.

The released official T expert-policy checkpoint is not a local supervised
dataset. No eligible local official T collector catalog or expert dataset
metadata was found. A checkpoint's presence does not supply its training episodes.

The frozen GPC-comparison fixtures contain **official** RGB and geometry, but are
short, repeatedly reviewed diagnostic branches. Their available physical futures
are 16 T branches from four anchors / 64 controls / nine contact controls, and 47
L branches from 12 anchors / 188 controls / 27 contact controls. Candidate branches
share conditioning histories. These are not complete collector episodes or an
independent validation/test set. Native GPC-domain fixtures, simulator-search and
DPPO/policy evaluation artifacts also retain their original purposes; matching
shape names or control rates does not turn them into official collector data.
Frozen GPC regression results and custom-CNN training results remain distinct.

No real train/development episode arrays were loaded for this inventory because
none were found with eligible local source manifests/catalogs. Contact,
translation, rotation and near-success coverage for a new adaptation dataset
therefore remain gaps. Missing coverage is unknown, not evidence of zero motion.

## Data API

Implementation: [gpc_adaptation_data.py](../tdyssey/gpc_adaptation_data.py).
The module imports no Torch and does not construct an environment or model.

```python
read_adaptation_manifest(path, *, require_ready=False) -> dict
readiness(normalized_manifest) -> dict
GPCAdaptationDataset(manifest_path, split="train", *, require_ready=False)
dataset.shape_indices(shape) -> tuple[int, ...]  # shape is "T" or "L"
fit_train_stats(dataset) -> dict
```

`read_adaptation_manifest` reads the declared adaptation manifest, SHA-bound
source catalogs and their source manifests. Episode payloads receive only path,
existence and size checks. Catalogs must identify the official
`official-pusht-capitals-v1` family, official collector/geometry, and 100 Hz
simulation / 10 Hz control. Every source episode must be declared, including the
inaccessible held split. Source manifest identities, complete status, lengths,
whole-episode splits and scene hashes remain bound; there is no implicit discovery,
reshuffle, window-level split or padding.

The adaptation manifest uses:

```json
{
  "schema_version": 1,
  "kind": "tdyssey.gpc_adaptation_data.v1",
  "environment_family": "official-pusht-capitals-v1",
  "t1_spec_sha256": "800751b7a3e32df4fee5660d1ea22c7ee0fb45b51a7b41632d420189238a7838",
  "catalogs": [{"path": "relative/source.catalog.json", "sha256": "<SHA256>"}],
  "episodes": [{"trajectory_id": "<source ID>", "role": "train", "prior_uses": []}],
  "reviewed_scene_sha256": [],
  "previously_used_episode_ids": []
}
```

Catalog paths are explicit and relative to the adaptation manifest. Episode
declarations must exhaust the bound catalogs. Source episode IDs, split membership
and scene hashes come from those catalogs; the adaptation manifest does not
invent replacement episodes. Prior-use declarations require reviewed provenance.
An empty list by itself cannot establish that an old episode was untouched.

| Original source split | Permitted adaptation role | Loading in preparation |
| --- | --- | --- |
| `train` | `train` | Allowed |
| `validation` | `development` | Allowed for development, excluded from independent comparison |
| `validation` | `independent_validation` | Allowed only with independent, unreviewed provenance |
| `model_test` | `held_model_test` | Manifest/catalog identity and stat only; payload loading rejected |

The dataset's `split` argument selects the **exact adaptation role**:
`train`, `development` or `independent_validation`. Ambiguous `validation` and
both held-role spellings are rejected before payload loading. Development and
independent validation have separate episode lists and indices; they are never
combined for a final comparison.

Duplicate episode identities, scenes crossing source splits, and duplicate image
payload hashes crossing splits are rejected. A scene marked reviewed, previously
used or development cannot reappear through an alias as independent validation.
Held source membership cannot be reassigned. Dataset loading rechecks frozen
manifest/catalog identities, including when an episode is cached, and verifies
actual loaded scene, collector, geometry, timing, contact count and reset
provenance. It rejects public-row reassignment before opening a payload.

Readiness requires both T and L training episodes **and** genuinely independent
validation episodes for both shapes, with present complete episode files and
nonempty four-control windows. Development is optional and cannot fill an
independent-validation gap. `require_ready=True` raises on any gap. The empty
local proposal consequently cannot permit real startup.

## Exact window and range contract

For an episode with `N` controls, there are `N+1` observations. Anchors are
`a=3..N-1`; the loader returns no incomplete window or episode-crossing sample:

| Sample key | Value |
| --- | --- |
| `image` | float32 `[5,3,96,96]`: real observations `o[a-3:a+2]`; first four condition the model, fifth is next-frame RGB target |
| `action` | float32 `[4,2]`: executed absolute targets `action[a-3:a+1]`, already normalized by `a/511*2-1` |
| `episode_id`, `shape`, `anchor` | Frozen source identity, T/L shape and local anchor |
| `split`, `role` | Exact adaptation role selected by the dataset |
| `source_split` | Original `train` or `validation` membership |
| `t1_spec_sha256` | The fixed T1 identity above |

All five **saved real** frames receive the same fixed T1 map once per sample,
starting from stored uint8 RGB converted to float32 `[0,1]`. The target is in the
same domain as the conditioning frames. There is no generated frame, external
RGB `2*x-1`, image clipping, second action normalization or learned input scale.
Only RGB and aligned actions enter the intended GPC loss; physical state/contact
information remains provenance or descriptive supervision.

`shape_indices("T")` and `shape_indices("L")` expose stable indices for an
explicit alternating batch-size-one T:L=1:1 sampler. This avoids silently changing
shape weighting when episode/window counts differ. `fit_train_stats` accepts a
training dataset only and counts unique controls once, rather than counting
overlapping windows repeatedly. Its action/contact/translation/rotation summaries
are descriptive; they do not fit or replace GPC's fixed input normalization.

## Fixed future collection plan and unmet preconditions

The plan declares T-A/T-B/L-A/L-B master seeds 410000/410001/410002/410003. Each
run has eight episodes of at most 100 controls, source 6/1/1 splits, 32 MiB artifact
cap, 120 seconds and at most 20 reset candidates per episode. A-run validation
is development; B-run validation is independent validation, subject to actual
scene/provenance checks. Totals are 24 train, two development, two independent
validation and four held model-test episodes.

Aggregate maxima are 3,200 controls, 640 reset candidates, 480 collection seconds
and 128 MiB raw collection artifacts. Full-length uint8 RGB payload alone would
be 89,358,336 bytes; 128 MiB is the artifact cap, not that payload size. The
source writer limits published contents; temporary manifest/file accounting and
aggregate-cap enforcement still need review before any future collection.
Original success can end an episode before 100 controls; the plan never pads or
repeats controls to reach a count.

Actual reset seeds are derived from master seed, collector version, shape,
episode index and candidate index. The plan's 640 candidates are unique and
disjoint from the known excluded seed set, including 100000..100002. This does
not prove disjoint physical scenes or coverage. Actual episode IDs also depend
on accepted reset, geometry/physics identities, action seed and collector settings.
Source splitting hash-sorts those actual IDs; exact episode IDs or split indices
cannot be assigned now without inventing collection outcomes.

Two source-level gaps block using the unchanged collector workflow as-is:

1. The current collector checks unsafe/near-goal states and excluded integer
   seeds, but does not reject the 316 reviewed full-state hashes. A future
   pre-control exclusion gate must fit the same fixed reset-candidate budget.
   Cross-split or repeated scenes found at freeze stop the plan; they do not
   trigger reassignment or recollection.
2. `collect_official_play.py` calls `validate_official_dataset` after collection,
   and `training_data.write_catalog` also loads every episode. Both currently
   open held model-test payloads. Future collection/catalog freezing needs
   reviewed metadata/in-memory identity capture and held manifest/stat-only
   handling, or separate explicit integrity-read authority. Neither path was
   invoked by this preparation.

Contact, translation, contact rotation and near-success coverage are not
guaranteed by the object-directed exploration prior. The collector initially
rejects coverage above .90. Poor coverage must remain a reported gap: no extra
run, added reset budget, adaptive seed search, held-test inspection or automatic
recollection is authorized by this plan. Actual collection requires separate
human authority and reviewed implementation of these gates.

## Focused verification

[Eleven data checks](../tests/test_gpc_adaptation_data.py) passed with small
synthetic NumPy episodes, one-thread numerical settings and Torch imports blocked
in the test harness. They cover exact history/action/target alignment, T1 on both
conditioning and target, stable shape indices, unique-control training statistics,
frozen manifest drift including cached loads, duplicate scene and membership
rejection, prior-use/held reassignment rejection, explicit development versus
independent-validation loaders, and missing-data readiness failure.

A payload-open trap rejects either held synthetic episode JSON or NPZ during
metadata reading and training-window access. The held dataset constructor also
fails before loading. No existing held model-test payload was opened, hashed,
decoded or used for statistics/tuning. These checks establish data plumbing and
isolation behavior, not GPC adaptation quality, independent coverage or a learned
model result.
