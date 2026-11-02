# Stage2A formal TRAIN contact coverage

H=actual hit; R=frozen historical TRAIN canonical normal reachability; E=union over recorded legal initial poses; K=R∪E∪H.
K is a global across observed poses reachable lower bound, not all physically reachable segments. U=all−K−closed holes remains unconfirmed.
Historical TRAIN has 22/23 independent trajectories per letter; new first batch has 9 planned. These are different collection budgets and scenes.

| Letter | H/all | H∩R/R | H∩E/E | H/K | U | Closed | Contact traj. | Contact controls | Planned arc hit | Old TRAIN H/all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 42/92 | 41/52 | 41/52 | 42/53 | 15 | 24 | 9/9 | 426/576 | 9/9 | 57/92 |
| B | 46/98 | 42/45 | 42/45 | 46/49 | 11 | 38 | 9/9 | 382/576 | 9/9 | 49/98 |
| C | 48/75 | 44/51 | 44/51 | 48/55 | 20 | 0 | 9/9 | 382/576 | 9/9 | 57/75 |
| E | 65/90 | 62/74 | 62/74 | 65/77 | 13 | 0 | 9/9 | 413/576 | 9/9 | 79/90 |
| F | 60/70 | 57/58 | 57/58 | 60/61 | 9 | 0 | 9/9 | 422/576 | 9/9 | 62/70 |
| H | 51/81 | 48/65 | 48/65 | 51/68 | 13 | 0 | 9/9 | 431/576 | 9/9 | 70/81 |
| I | 48/75 | 45/59 | 45/59 | 48/62 | 13 | 0 | 9/9 | 384/576 | 9/9 | 65/75 |
| J | 42/74 | 38/47 | 38/47 | 42/51 | 23 | 0 | 9/9 | 378/576 | 9/9 | 55/74 |
| L | 38/53 | 37/49 | 37/49 | 38/50 | 3 | 0 | 9/9 | 414/576 | 9/9 | 49/53 |
| M | 59/85 | 55/60 | 55/60 | 59/64 | 21 | 0 | 9/9 | 354/576 | 9/9 | 64/85 |
| O | 34/93 | 32/40 | 32/40 | 34/42 | 14 | 37 | 9/9 | 379/576 | 5/9 | 44/93 |
| P | 35/75 | 31/44 | 31/44 | 35/48 | 8 | 19 | 9/9 | 393/576 | 3/9 | 46/75 |
| S | 42/85 | 38/53 | 38/53 | 42/57 | 28 | 0 | 9/9 | 411/576 | 9/9 | 62/85 |
| T | 50/60 | 48/52 | 48/52 | 50/54 | 6 | 0 | 9/9 | 433/576 | 9/9 | 55/60 |
| U | 47/81 | 45/58 | 45/58 | 47/60 | 21 | 0 | 9/9 | 410/576 | 9/9 | 63/81 |
| V | 41/77 | 35/43 | 35/43 | 41/49 | 28 | 0 | 9/9 | 382/576 | 9/9 | 50/77 |
| X | 41/84 | 34/39 | 34/39 | 41/46 | 38 | 0 | 9/9 | 384/576 | 9/9 | 48/84 |
| Y | 38/60 | 33/36 | 33/36 | 38/41 | 19 | 0 | 9/9 | 409/576 | 9/9 | 40/60 |

Evidence consistency risks: 0. Zero-contact trajectories: [].

`stage2a-reachability-summary.json` contains every unhit/known/unknown/closed bin ID and overlapping feature counts.
`target-contact-comparison.csv` compares the planned initial contact arc to actual collisions, not pusher click arrival.
`initial-scene-gaps.csv` lists all four true initial wall gaps. `contact-controls.csv` joins actual collision bins to continuous area-centroid dx/dy, rotation, motion class, requested approach and four before/after wall gaps.
`contact-cube-summary.csv` gives distinct controls and independent trajectory support per arc × requested approach × motion class × actual minimum-gap band. Actual contact-normal angle counts are separately reported; requested direction is not actual collision direction.
No wall-arbiter points were retained. Bounding-gap proximity does not establish an actual wall-collision count.
Repeated touches do not enlarge H. A control can hit multiple bins, so contact-control/bin incidences exceed distinct contact controls. Historical priors are not added to actual counts.

Postprocessing source SHA256: `fb8f5495708f853e7a8a6c0a530b86171d7da6a1719ff0a490af516fc44d697b`.
This report performed zero physics controls, environment construction, renderer/model calls or data/controller edits.
