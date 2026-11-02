# Static review of already-exported PNGs

This note was written after command3 failed. Only its existing derived PNGs,
text and source were viewed. No NPY/NPZ reopen, metric reconstruction, model,
new numerical command or fourth command occurred. All32 retained panels were
visually reviewed by root or its static reviewer; this is descriptive evidence.

All eight comparison panels show recognizable red T/L objects, blue pushers
and black T1 goals. Adapted background has a pale pink cast, including ARH1 and
TFH4, whereas real T1 targets and frozen backgrounds appear near-white. Local
contour/position differences are visible, especially later contact horizons.
Recognizability is not fixed-gate validity or a physical accuracy assessment.

All eight segmentation panels' adapted rows show a main connected silhouette
and seed-count caption1. Continuous object-alpha panels retain weak background
response outside the yellow ROI. This distinguishes weak diffuse alpha from
multiple disconnected thresholded pieces. Rounded H1 component-fraction captions:

| Branch | Adapted caption | Frozen caption |
| --- | ---: | ---: |
| T12/away | .3611 | 1.0000 |
| T13/toward | .3604 | 1.0000 |
| IT00/away | .3595 | 1.0000 |
| IT01/toward | .3540 | .9998 |
| L00/away | .3196 | not transcribed |
| L03/toward | .3244 | not transcribed |
| IL00/away | .3238 | 1.0000 |
| IL01/toward | .3235 | not transcribed |

These are manually read rounded captions, not a replacement pixels.json or
newly computed numerical estimates. Old original JSON values remain authoritative.
Some frozen L later-horizon panels retain rigid_shape_mismatch: the comparison
does not assert that frozen geometry always passes or that background alone
accounts for every validity failure.

The eight history panels label four real observations -3..0 and actualH1..H4;
contact/noncontact futures remain visible. The eight train panels show five
selected source indexes with the fifth labelled TARGET, raw gray/green and
T1 red/black rows. Blue pushers, spatial layouts and near-white backgrounds are
preserved visually. Train06/07 L include changing/contacting and partially
cropped targets; this observation does not estimate prevalence across500windows
or prove a cause of checkpoint failure.

Full-precision alpha/ROI/MAE deltas were only in process memory and were lost at
serialization. Their values are not recovered. ARH4 intermediate conditioning
remains unknown; existing truth was never substituted for excluded predictions.
