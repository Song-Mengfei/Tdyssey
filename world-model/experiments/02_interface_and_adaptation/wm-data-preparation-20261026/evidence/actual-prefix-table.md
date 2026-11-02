# Actual A: full training corpus versus fixed 500-window prefix

The prefix selects data for the frozen 500-update schedule. No optimizer update, model forward, image-archive load or simulation occurred in this audit.

| Shape / population | Controls/windows | Episodes/scenes | Contact | Rotation >5 deg | Rotation P90 deg | Translation P90 px | Max coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T full train controls | 1200 | 12/12 | 577/1200 (48.08%) | 9/1200 | 1.6785 | 4.8291 | 0.4942 |
| T eligible train windows | 1164 | 12/12 | 577/1164 (49.57%) | 9/1164 | 1.7361 | 4.8308 | 0.4942 |
| T fixed prefix | 250 | 3/3 | 148/250 (59.20%) | 0/250 | 0.7306 | 4.7146 | 0.0699 |
| L full train controls | 1200 | 12/12 | 703/1200 (58.58%) | 18/1200 | 2.0614 | 4.8665 | 0.3626 |
| L eligible train windows | 1164 | 12/12 | 703/1164 (60.40%) | 18/1164 | 2.0908 | 4.8695 | 0.3626 |
| L fixed prefix | 250 | 3/3 | 109/250 (43.60%) | 1/250 | 1.9571 | 4.8376 | 0.2489 |

250 T and250 L targets are500 unique windows, with no repeats, from3episodes/scenes per shape. L contact falls703/1164eligiblewindows (60.40%) to109/250prefix (43.60%); >5-degree target rotations fall18 to1. T has0 >5-degree rotations in its prefix. No train/prefix target has near-success [.9,.95] or strict success >.95.

Independent validation:97distinctwindows per shape, each from1episode/1scene. This meets8windows/shape, not8independent scenes. Four model-test episodes remain held. The held-access proof trapped Path.open/read_bytes and inode aliases:0 attempts across8held payload paths; repeated offline audit matched exactly.

A-data window support is ready. Global adaptation/benchmark readiness is not established: B stopped before its first physical call; benchmark=None, separation/termination support unavailable. No retry, repair or relaunch.

Contact/motion labels describe the executed target control. Full controls include the first3controls that cannot be targets with4-frame history. No data order or coverage threshold was tuned.

Audit: actual-prefix-audit.json, 257795bytes, SHA256 `73b6db2463c239409e3014390de3d5a5ebe317b6c7391f1635876807c6a177e9`.
