# Checkpoint500 offline JSON repair and re-export

Main review independently reproduced the NumPy ROI integer serialization defect
in the [stopped diagnosis](GPC_CHECKPOINT500_OFFLINE_DIAGNOSIS.md), then explicitly
authorized this separate bounded task from clean`639995a`. The old failure,
receipts, budget, sample manifest and32 PNGs remain unchanged.

The only diagnostic change is four explicit`int(...)` conversions in
`pixel_diagnostics()['roi']`. Numerical algorithms, input arrays, colors, drawing,
production T1/measurement/model/training/sampling code and scientific gates are
unchanged. One regression exercises actual diagnostics and nested serialization
with interior/empty/edge-clamped ROIs and strict NaN rejection. Its sealed
original-helper comparison must reproduce old failure and retain all other
metrics, image views and raw bytes. The existing ten checks run once with it.

Input is the same252831B `fixed-samples.json`, SHA256
`193096b5f23df1bec4e8f2ed73df5d15fbc4f2ef9c3edc571ad9701eb61a7ed7`.
There is no repeat freeze or3996-row statistics pass. The unchanged eight
branches supply48 predictions/64 real frames; eight train windows supply40
frames. AR H4's16 unavailable input hashes stay unknown.

The fixed wrapper is`scripts/run_gpc_offline_reexport_command.ps1`: command1 is
the11 synthetic checks; command2 directly calls the existing pixel helper CLI
into`artifacts/gpc-checkpoint500-offline-reexport-20261026`. Its same timed
command verifies parsed durable counts,48 action/32 real-input checks and all32
PNG hashes against the old partial output. A mismatch or first error stops;
there is no third command or retry. The original fixed-output CLI is not relaxed.

Each command has60s external timeout, total120s, two CPU cores/library threads2,
one actual installed base CPython process,1GiB working set and32MiB new evidence.
No installs, weights/held contents, Torch/model/SDK/CUDA/neural/policy/training/
new prediction/physics/collection operations are authorized or needed. The
original helper snapshot comes from the sealed639995a Git blob, not a weight.

## Completed execution

Execution used clean`f0ed9c2d5acfbb77e8a75fbfbfda955130e948c1`, frozen in
`preexecution-binding.json`. Both fixed commands completed, no failure/retry or
third numerical command. Command1 passed11 tests/0skips in.796s, including the
old TypeError/new success comparison and unchanged-other-values checks.

| Command | External seconds | Observed actual-process peak |
| --- | ---: | ---: |
| 1: synthetic regression plus old ten checks | 1.3513544 | 89968640B |
| 2: fixed export and strict byte/JSON checks | 4.7375704 | 101486592B |

Total6.0889248s fits the new120s budget. The old failed task's8.2616727s and
three spent commands remain separate and unchanged. No real model, Torch,
weights, CUDA, neural/optimizer/policy/physics/collection work ran.

`pixels/pixels.json` is532601B, SHA256
`421ced6c340784e4c9492fc2876d1cecc049ef2b497c938d8c6e6c6aaa1b5c77`.
It parses fully, using the unchanged strict finite-or-null serializer. Durable
counts are48 prediction frames,64 real case frames,40 train frames and80 small
state rows. All48 action hashes and32 reconstructible H1/TF4 input hashes match;
the16 AR4 inputs remain explicitly unknown. All eight train state alignments
complete. All32 new PNG hashes are identical to the old partial PNGs, as recorded
in`export-verification.json`. No image content or rendering rule changed.

## Fixed-panel numerical evidence

These are existing scalar fields from the new report, not a new score, threshold
or full-grid reanalysis. Each H1 row below is one fixed physical branch at one
noise seed; four are regression and four independent. All48 selected prediction
records have one thresholded seed component and component-mass replay delta0
against their original measurements, with no missing replay values.

| H1 branch | Adapted outside-ROI alpha fraction | Adapted component fraction | Adapted ROI/full IoU | Frozen ROI IoU |
| --- | ---: | ---: | ---: | ---: |
| T12/away | .600680 | .361117 | .840658/.340534 | .945407 |
| T13/toward | .602807 | .360364 | .839275/.338141 | .946458 |
| IT00/away | .598337 | .359523 | .846501/.343610 | .949646 |
| IT01/toward | .603661 | .353979 | .834904/.335972 | .929832 |
| L00/away | .633509 | .319559 | .811968/.301760 | .907986 |
| L03/toward | .631000 | .324363 | .815012/.305110 | .904994 |
| IL00/away | .635949 | .323800 | .812743/.300088 | .912411 |
| IL01/toward | .638013 | .323451 | .837371/.307218 | .899336 |

Frozen outside-ROI alpha is0 or floating-point residue up to about2.64e-16 in
these H1 records, with component fractions.999811..1. Adapted has roughly60-64%
of its continuous chromatic evidence outside the image-derived ROI, despite one
seed component. This quantifies the retained pink-background/weak-alpha visual
observation, not literal disconnected object pieces. Outside-ROI alpha fraction
and component fraction are not complements: the component numerator uses a
dilated largest seed, whereas ROI is a padded bounding box.

The saved original ROI IoU also falls relative to frozen. The actual fixed gate
uses full-image IoU, not this ROI diagnostic alone; no hypothetical background
correction or re-fit was performed. Thus color response contributes to rejection,
while local shape/area evidence and physical accuracy remain unresolved. All
eight adapted H1 original states are invalid, and coverage remains null. The
unchanged full evaluation still rejects checkpoint500; no gate is reopened,
candidate pose promoted, or imagined-DPPO acceptance inferred.

The serializer defect is now closed without changing pixels or metrics. Main
review can use the complete existing-input evidence to decide the smallest
scientific follow-up; this task authorizes no new training, model test, data,
color correction or threshold adjustment. Old failed reports are not rewritten.
