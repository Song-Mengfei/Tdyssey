# Frozen schedule audit

No regeneration, search, overlap optimization, re-selection or training was performed.

| Shape / order | Targets | Scenes | Contacts | Turns >5 | CW / CCW >5 |
|---|---:|---:|---:|---:|---:|
| T full train | 1164 | 12 | 577 | 9 | 9 / 0 |
| T old prefix | 250 | 3 | 148 | 0 | 0 / 0 |
| T new schedule | 250 | 12 | 136 | 9 | 9 / 0 |
| L full train | 1164 | 12 | 703 | 18 | 8 / 10 |
| L old prefix | 250 | 3 | 109 | 1 | 0 / 1 |
| L new schedule | 250 | 12 | 138 | 18 | 8 / 10 |

Each shape:1250 full-five-frame occurrences;1000 context/action occurrences. Unique identities are (trajectory,index).

| Shape/order | Unique 5frames | Unique contexts/actions | Same-scene unordered pairs | Pairs delta<=4 | Pairs delta<=3 |
|---|---:|---:|---:|---:|---:|
| T old | 262 | 259 | 10852 | 970 | 732 |
| T new | 809 | 707 | 2480 | 230 | 175 |
| L old | 262 | 259 | 10852 | 970 | 732 |
| L new | 811 | 712 | 2480 | 220 | 164 |

500 unique windows, no repeats;12episodes/scenes per shape,20or21 windows/scene. All9T/18L >5-degree targets are included. T CCW>5 and near/strict-success targets remain absent. Privileged train labels alter the sampling distribution; this does not create missing data.

Same-scene occurrence pairs overlap full5frames at |delta|<=4 and context/actions at<=3. Overlap is descriptive, not a selection criterion or independence claim.

All16 nontraining payloads,32image archives andheldpath/inode aliases were trapped:0attempts. Exactly24train JSON records read;0dev/independent/held JSON. Independent validation97windows/shape is metadata-only (1scene/shape).

Schedule SHA `f7217a2d09393798529b00e72fcd971e8d7be11cab7a9c769704a9b30ddd004a`; order SHA `54edfd533df01652b3208996f80bf88e9f19f9806bb01ddcde93adbcaafc170d`.
