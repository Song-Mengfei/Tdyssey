# First physical regression: saved-only audit

120 raw files checked:96 candidate,24 original. Candidate gate FAILED: 28/96 cases failed. No new model/GPU/physics or optimizer work.

Cost reconstructed: {'attempted_controls': 2305, 'completed_controls': 2277, 'microsteps': 177150, 'invalid_partial_controls': 28}. Matches actual aggregate ledger: True.
Raw hashes mismatched: 0; exact geometric penetration recomputation mismatched: 0.

| Letter | Pass/16 | Actual attempted/completed controls | Intervention time | Max OW/AO/AW world px | max valid H1 px / deg |
| --- | --- | --- | --- | --- | --- |
|T|12|302/298|12.66%|1.8243/1.5819/0.0000|15.998/49.384|
|L|12|303/299|16.76%|2.0845/0.7910/0.0000|10.199/33.862|
|O|10|260/254|10.86%|1.5908/1.1584/0.0000|8.146/27.704|
|C|9|242/235|7.63%|2.2305/0.7118/0.0000|7.339/18.760|
|X|13|321/318|28.81%|1.2857/0.7646/0.0000|12.243/40.801|
|B|12|301/297|19.60%|1.5466/0.7972/0.0000|8.142/30.901|

## Corrections and identity limits

- Original per-case completed_controls includes invalid attempted control; use valid_completed_controls here.
- Original per-case intervened_microsteps uses truthiness of a nonempty dict; use delta_norm>1e-8 and actual dt here.
- Original illegal_microsteps field omits pusher_region. This audit recomputes solid/closed-hole region from saved geometry and post-state.
- Original cumulative translation sums control-boundary chords; this audit separately supplies actual microstep path.
- This v1 attempt ran v3 physics protocol d106d5c10f40a341121fd54531ace02066ce493dd4ba6a30913bd2982afa2ae0. Its raw identity still says adaptive/variable although every candidate microstep is .001s. Metadata-only revision changes identity and snapshot protocol binding, not motion.
- started.json binds the regression runner SHA, not the runtime physics source SHA. The reported e6ad source and its preserved snapshot are supported by parent execution history and metadata-revision; do not claim started.json contains that binding.
- Report-only audit does not establish a passing future candidate or authorize collection/training.
