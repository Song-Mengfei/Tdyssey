# One real checkpoint500 paired evaluation

The sole authorized evaluation completed, but checkpoint500 is **not accepted**
under the unchanged scientific gates. Adapted object/shape and coverage validity
are0/1998, including AR H1 and every TF horizon. Independent loss increased
4.235x. T preservation fails; L object-motion/coverage errors and usable ranking
cannot be established through the fixed interface. This is no imagined-DPPO or
physical-policy success. No retraining, collection, fallback or second run followed.

## Identity and complete costs

Execution used clean `042f7186e6eb5b3f5b2e170f50b80e4ce8aa41b9`, unchanged
`configs/gpc_checkpoint500_evaluation.json`, SHA256
`cd586ed72153ad5d246b4109d18b8ee1faca5dfd7312c721c2c470ca0369e6e2`,
and authorization `20261026-user-autonomous-gpc-checkpoint500-evaluation-01`.
The permanent spent output is `artifacts/gpc-checkpoint500-evaluation-20261026`.
Assets SHA is `808d380635f3c5f73d2aa010f38524479468b345d2d8e21e9f7100b55e773c52`;
GPC source remains `a96146e568826ef46134f73c22fdf59a675e7264`. Initial and selected
checkpoint500 hashes/bytes are the fixed [engineering identities](GPC_CHECKPOINT500_EVALUATION.md).

The guarded run lasted1783.203s, from2026-10-26 07:09:30.983 to07:39:14.178 SGT;
the external receipt is1783.911828s, exit0. No earlier SDK warmup or free forward
ran. Only read-only identity/process/old-evidence checks preceded execution; fresh
native resource checks were performed inside the existing guardian.

| Account operation | Reserved | Completed |
| --- | ---: | ---: |
| Model loads | 2 | 2 |
| Loss batches/examples | 32 | 32 |
| Forecast images | 3996 | 3996 |
| Euler calls | 11988 | 11988 |
| New updates/training examples | 0 | 0 |

Both frozen and adapted roles completed16 loss examples and1998 predictions.
The8064-event hash chain is complete, no operation is pending, maximum active
role is1, and release order is frozen/adapted. Source logical500 remains distinct
from new updates0; no optimizer, old account, selected cursor or training RNG was
restored and no checkpoint was selected again. Native model-only loading, strict
state checks and both roles' real inference completed within the frozen caps.
No separate post-run model load or parameter introspection was performed.

Review evidence is under `artifacts/gpc-checkpoint500-evaluation-review-20261026/`:
`prelaunch.json`, `guardian-audit.json`, `output-integrity.json`,
`output-inventory.json`, `scientific-review.json` and `delivery.json`. The source
account's old333 forecasts/999 executed Euler remain spent and were not spliced
into this run. Across the three pilots and this evaluation:5 model loads,
502 reserved/501 durable updates,4329 actual forecasts/12987 Euler versus
4328/12984 durable completions, and2177.688 guarded seconds. The old SDK-only
3.860s is separate. These historical pending charges are not refunded.

## Paired inputs and output integrity

All3996 saved NPY files independently passed stdlib header/shape/dtype/byte-length
and raw-array hash checks: float32 CHW3x96x96,441925632 payload bytes and442437120
NPY file bytes. Each matches its saved measurement and the frozen111-branch grid:
63 regression+48 B2, three declared noises, AR H1-H4 and TF H2/H4, both roles.
The raw hashes bind the runtime pre-export finite checks; no new decoder or tensor
operation was run offline. Raw ranges are frozen[-.0352941,1] and
adapted[-.0431372,1], not a numerical-NaN failure.

All1998 cross-weight pairs have identical RNG entry/exit, actions, physical truth,
latest state, contact label, shape and stratum. The999 AR-H1/TF-H2/TF-H4 real-input
hashes match. AR-H2..4 inputs differ in all999 pairs, as expected when each model
feeds back its own raw prediction. The1332 within-role AR/TF horizon pairs also
match noise entry/exit and actions. All16 loss example/noise pairs match. No pair,
prediction or invalid measurement was dropped.

## Scientific result

Loss is descriptive: eight neighboring windows from one independent scene per
shape, not eight independent scenes. Adapted loss is higher on14/16 matched pairs.

| Independent loss | Frozen | Adapted |
| --- | ---: | ---: |
| All16 mean | .00295458 | .01251361 |
| T8 mean | .00216643 | .01281465 |
| L8 mean | .00374273 | .01221258 |

Across all horizons/strata, object/shape and coverage validity fall from1650/1998
to0/1998. Adapted reasons are `fragmented_object`1982 and
`object_area_mismatch`16. These are fixed interface diagnostics, not proof of a
localized dynamics defect or blank RGB images. Adapted pusher validity remains
1856/1998 versus1998/1998 frozen. Invalid candidate poses are never promoted into
accepted state estimates; adapted object position/angle/coverage error values
are undefined, not zero.

T retention fails for AR, absolute AR and paired ranking; TF separately fails.
Frozen T AR object validity is120/120 per horizon across the pooled strata,
whereas adapted is0/120. Failure already occurs at H1 and with real TF history,
so it cannot be described solely as long autoregressive accumulation.

| L object/coverage validity | Frozen | Adapted |
| --- | ---: | ---: |
| Regression AR H1 | 141/141 | 0/141 |
| Independent AR H1 | 57/72 | 0/72 |
| Regression AR H4 | 78/141 | 0/141 |
| Independent AR H4 | 43/72 | 0/72 |
| Regression TF H4 | 103/141 | 0/141 |
| Independent TF H4 | 48/72 | 0/72 |

Conditional L pusher P90 pixels at AR H4 are14.3833→32.8326 in regression and
2.0913→31.2929 in independent data. Adapted AR H4 pusher counts are128/141 and
63/72. Its TF H4 P90 is8.2184 and7.6125, with121/141 and61/72 valid. These use
different measurable subsets, so the AR/TF contrast is descriptive rather than
a paired causal improvement claim. TF does not rescue object validity.

Contact and rotation decisions remain insufficient/unmeasurable. Independent
physical contact support is4<8 and large-rotation support2<4 branches per shape;
adapted valid object-error rows are0. Regression L has9 contact branches, which
meets the branch floor, but0 measurable contact errors; its null result is not
explained by claiming9<8. GPC supplies no direct contact head.

H4 has no common comparable ranking anchors/pairs in any shape/stratum. Adapted
abstains on27/27 truth-separable pairs across the four groups. Scored accuracy,
common-subset regret, hold and uniform baselines remain null; these are not27
scored wrong answers. T ranking retention fails; L ranking remains insufficient.
H1/H2 have no .05-separable truth pairs. Three noises do not add physical anchors.

All adapted AR-H4 termination decisions are unknown333/333 and precision is null;
certification remains false. The independent data does contain one actual-success
branch/anchor per shape. Frozen T's three false-negative rows repeat the same one
physical branch; `positive_anchors=0` refers to predicted true-positive anchors,
not absence of actual positives. Zero predicted positives is not perfect precision.

## Guardian, capacity and preservation

The real append journal contains5800 checked complete frames/6012519B, max
payload983B, no partial tail and maximum publication gap1.125s. There are15
legitimate unavailable-CUDA setup records and5785 fresh records. Parent maximum
observed heartbeat/CUDA age is1.05946/1.06246s. The first-error slot is `{}`.
Final worker21872/launcher24688 exit0, guard27756; Job active0, lock released,
no cleanup errors. Old1175's OS cause remains unknown; no status-replacement
error occurred on the new append path during this run.

Final fresh cumulative CUDA allocated peak is110494720B=105.376MiB; reserved
peak136314880B=130MiB. The sampled current allocated maximum103305216B is a
different quantity. Sampled private maximum2354749440B=2.193GiB and native
working-set peak1422323712B=1.325GiB fit the unchanged3/2GiB limits. Free RAM,
commit, GPU and disk minima are6221774848/15186317312/7050625024/28462907392B.
Two releases have model_instances0/ledger-active0, with nonzero terminal allocator
residuals: allocated8519680B/reserved20971520B. This records serial lifecycle,
not zero retained CUDA tensors or a diagnosis of a leak; process/Job exit is clean.
An initial offline zero-allocator assumption was corrected; its draft is retained.

There are8006 formal files/472896760B. Actual writer metadata is9741246B<16MiB;
derived prepaid output charge468955582B<1GiB. Actual telemetry20718394B fits its
original64MiB partition: journal6012519, ledger2105180, monitor12462358, stdout216,
fixed138121B. Monitor includes138 final-event bytes beyond the report's earlier
counter snapshot. Every original resource/time/quota rule remained unchanged.

Closing audit preserves all9245 old files/779569307B in66 roots plus standalone
SDK claim, all16 old claim documents/20174B, six old identities and execution
config. Eight held payloads received stat checks only, no opens/hashes. Old
claims/failed temporaries, weights, configs01-03 and500list are unchanged. This
new attempt is spent, with no retry or balance reuse.

The fixed T near-positive band still lacks support (.981947>.98), training lacks
near-success/T>5deg CCW examples, and H4/independent-scene support stays sparse.
Those gaps do not make this checkpoint pass the failed validity/T-retention gates.
The smallest possible follow-up is a separately authorized read-only audit of
existing H1/TF outputs and train/eval visual/coordinate handling before proposing
any new training, data or threshold. No follow-up execution is authorized here.
