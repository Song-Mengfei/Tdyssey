# Frozen train schedule and one new B-only attempt

This stage starts from reviewed clean `fb1bb20`. Existing A data and stopped B
remain immutable. No GPC/policy load, model forward, training, A/v3 rerun,
installation, download, push or PR merge is authorized. After engineering
checks and freezing the train list, one new B2 may collect the original
12 anchors / 48 branches, then returns for review.

## Train-only rule

The rule precedes list generation in
`artifacts/wm-sampling-b-only-20261026/sampling-rule-before-generation.json`.
Use only the 24 existing train trajectories: 12 scenes per shape, each with
97 legal windows. No development/independent-validation/held payload or B outcome
determines selection. Alternate T/L for exactly 500 exposures, 250 each. Sort
each shape's scenes by trajectory ID and use round-robin scene order: the first
ten get 21 exposures and the last two get 20. In each scene take rotation >5deg
targets first up to quota, then alternate contact-only and other targets,
skipping/draining empty buckets. Each bucket uses one fixed SHA order of rule
version, seed5478, shape, episode ID and anchor. Repeat only if a scene has fewer
legal windows than its quota; repeats remain explicit. No redraw/search occurs.

Target anchors `a=3..N-1` use observations/actions `a-3..a` and target observation
`a+1`. Every ordered dataset index, trajectory, scene and anchor is recorded.
Manifest SHA, ordered identity, rule identity and cumulative cursor bind the
existing budget/checkpoint contract. Training consumes this list, never silently
the old prefix; a changed list/cursor cannot resume. Model, T1, objective,
optimizer, 500 updates, checkpoint500 and evaluation gates stay fixed.

Equal scene exposure replaces old prefix weighting. Rotation is intentionally
oversampled and remaining contact/noncontact fill balanced where available.
This is privileged label selection, not a natural data distribution. Report
contacts, signed/absolute rotation, coverage, repeats and overlapping histories.
No near-success target is invented. Independent loss validation remains one
trajectory/scene and 97 neighboring windows per shape, not 97 independent scenes.

## Separate B2 account

New account `B2`, run ID `B`, uses `artifacts/gpc-data-stage-b2-20261026/` and
`artifacts/gpc-data-stage-20261026-registry/B2.json`. It cannot refund/reset old
A/B. Only the exact new registry member is authorized; the shared native slot
still prevents concurrent collection. Original geometry/actions/seeds remain in
`artifacts/wm-data-preparation-20261026/independent-benchmark-plan.json`, SHA
`829eeb5c911025618c670fcb8538befc14e07e1988c24c802a006b1e2a2159cf`.
The threshold is checked after the precharged reset as fixed in `fb1bb20`.

Caps remain 256 controls, 256 resets, 64 restores, 512 renders, 2,816 substeps
(2,560 control +256 reset), 600 seconds and 64 MiB including temporary,
publication and guardian costs. No unaccounted smoke reset occurs. Strict
success >.95 ends physical futures; absorbing labels add no controls. Any error
stops B2 without retry, refill, replacement seed or extra example.

Startup binds clean source, frozen train list/config, old A/B stop records and
claims, 779 known state/109 seed exclusions, and every A source role's scene/seed
metadata, including held. Held payloads stay stat-only. Existing RAM/commit/disk/
process floors, native ownership and 3GiB Job stay fixed; no GPU/model query is
needed. Collection memory observations do not establish training feasibility.

Actual branches, common history/actions, contact/translation/CW/CCW, physical
candidate separation, near-success signs and absorption are audited afterward.
Geometry category intent is insufficient; missing support stays a gap. Even a
complete B2 requires further review before GPC adaptation or imagined DPPO.

## Actual results

The one B2 collection completed on clean `1ecf3da6d8f6da58021a73c8b117faeb86839081`.
The train list was generated exactly once before B2, SHA256
`f7217a2d09393798529b00e72fcd971e8d7be11cab7a9c769704a9b30ddd004a`;
order SHA `54edfd533df01652b3208996f80bf88e9f19f9806bb01ddcde93adbcaafc170d`,
rule SHA `909d35e2816e317ec0766291632b91cef5e0bc1545bc7da176572176637a84bc`.
The config SHA is `1c5683baa16daee11ca419772532cdda95595275cca794b977261188d99abe2c`.
Only asset paths/identities and the authorized training order changed from
`fb1bb20`; model, objective, optimizer, T1, budgets and evaluation gates did not.

| Shape/order | Legal/exposed windows | Scenes | Contact targets | Turns >5deg | CW/CCW >5deg |
| --- | ---: | ---: | ---: | ---: | ---: |
| T full train | 1,164 | 12 | 577 | 9 | 9/0 |
| T old prefix | 250 | 3 | 148 | 0 | 0/0 |
| T frozen list | 250 | 12 | 136 | 9 | 9/0 |
| L full train | 1,164 | 12 | 703 | 18 | 8/10 |
| L old prefix | 250 | 3 | 109 | 1 | 0/1 |
| L frozen list | 250 | 12 | 138 | 18 | 8/10 |

All 500 selected target windows are unique; scene quotas are20/21. Rotation
values here describe one target control, not four-control benchmark endpoints.
T has no >5deg CCW training target and neither shape has a near-success/success
training target. Sampling cannot create those missing examples. Independent
loss validation remains one scene/97 overlapping windows per shape.

For T/L respectively, selected mean absolute target turns are .858/.998deg
(old prefix .217/.500), P90 2.426/2.278deg and maxima17.004/13.096deg. Mean
translations are1.580/2.160px, P90 4.881/4.839px. Maximum coverage is
.494209/.361918; near-success and strict-success target counts both remain zero.
These summaries describe the intentionally changed training exposure only.

Per shape the list contains 1,250 five-frame and 1,000 four-action occurrences.
Unique five-frame identities increase from262 to809(T)/811(L); unique context/
action identities increase from259 to707/712. Same-scene unordered sample pairs
decrease from10,852 to2,480. Full-five-frame overlapping pairs are230(T)/220(L),
versus970 in the old prefix; context/action-overlap pairs are175/164, versus732.
Overlap remains present and is not independent replication. No overlap search
or optimization was performed. Generation and audit trapped all16 nontrain
payload paths and all32 image archives (40-path union, including eight held
paths), with zero attempts; each read exactly24 source-train JSON records.

| B2 cost | Actual | Original cap | Unused |
| --- | ---: | ---: | ---: |
| Controls | 224 | 256 | 32 |
| Resets | 12 | 256 | 244 |
| Restores | 60 | 64 | 4 |
| RGB renders | 296 | 512 | 216 |
| Control physics substeps | 2,240 | 2,560 | 320 |
| Reset physics substeps | 12 | 256 | 244 |
| Total substeps | 2,252 | 2,816 | 564 |
| Wall seconds | 5.297 | 600 | 594.703 |
| Charged bytes | 5,613,159 | 67,108,864 | 61,495,705 |

The new B2 claim SHA is
`0568138910141d5d5ca38cd898b0bcc948ef3cd67be0cc00f274fec197c3a089`.
It neither reuses nor replenishes the old B claim
`8e4714a522f6af37205d5449b08d4eb833d3b60b0f415af17525a94112e306de`,
which remains stopped with zero physical operations. B2 has36 history controls
and188 executed future controls, with four absorbing frames after two early
successes. There was no error, retry, refill, seed replacement, A/v3 rerun,
model load, GPU query or training. Unused one-shot capacity is not reusable.
The owned Job was empty after cleanup; its lock was released without errors.

B2 contains48 cases/12 anchors with no overlap against the effective811 known
state/141 seed exclusions. No-contact, translation and both rotation directions
have physical support for each shape: translation is29.293px; endpoint T turns
are+13.879/-13.879deg and L turns+18.914/-16.464deg. These are geometric/physics
labels, not a measured model result or broad generalization claim.

Physical coverage separation uses the unchanged `abs(diff)+1e-12 >= .05` rule.
H1/H2 have zero separable pairs for both shapes. H3 has3pairs/oneanchor per
shape. H4 has3pairs/oneanchor onT (`IT01`) and6pairs/twoanchors onL
(`IL01`,`IL02`). Keep the complete48-case denominator; uninformative controls
are not ranking successes. Pairs within an anchor are not independent scenes.

Each shape has one strict-success branch and23 negatives. Toward at IT05/IL05
terminates after its second future control and contributes two absorbing frames
each. T ends at coverage .981947 and L at .978938. The T positive exceeds the
predeclared near-positive upper bound .98, so IT05 does not satisfy `(.95,.98]`.
Eleven of twelve category intents are supported; the remaining one stays a gap.
Strict positive/negative labels exist, but generated-image termination precision
has not been tested or certified. No extra sample or relaxed bound follows.

The metadata planner reports complete assets; this does not remove coverage
limitations or authorize training. The new benchmark metadata SHA is
`dde6b5b8c31c8aac17f341d08f8f12eef78fe0e8ca4ebfe88e289c6289e759ea`.
Main review must consider the narrow T near-success gap, H1/H2 ranking support,
limited informative anchors, absent train near-success targets and one-scene
loss validation before any next bounded adaptation.

Native collection startup passed with8,244,809,728 bytes free RAM and
15,984,685,056 bytes commit headroom. Observed worker peak working set was
77,451,264 bytes (73.86MiB); sampled private-memory peak85,471,232 bytes
(81.51MiB), under the unchanged3GiB Job. Minimum free RAM was7,790,399,488
bytes. These CPU collection measurements do not validate model/GPU training.

The engineering gate passed64 aggregated stdlib/NumPy/fake checks and10 final
marker checks, including immutable consumption order, spent-cursor recovery and
T1/window boundaries. Real Torch/neural/simulator imports were blocked for the
aggregate; marker counters are not executed neural optimization. Source/message
and whitespace checks passed. Evidence is under
`artifacts/wm-sampling-b-only-20261026/`: frozen rule/list, generation/access
proofs, schedule comparison/audit, configuration difference proof, native
preflight, B2 budget/support audits and preservation reconciliation. New B2
raw assets/guardian live only in their declared new root. No push or PR merge.

Independent contract verification opened each of the48 new B2 NPZs once and
verified SHA/size, array format, shared four-frame history/three past actions,
latest full state, fixed actions/seeds/geometry and absorbed RGB/state repetition.
No old/A/held image archive was opened. Closing preservation passed for all
7,626 old files /600,533,534 bytes/46 roots:7,618 fresh content hashes and eight
held files checked only through sealed manifest/stat identities. All eight old
claims (six model/two data,9,249bytes) remain unchanged. The sole allowed member
addition is B2.json (1,353bytes); no old file was changed/removed and the native
lock is absent. Schedule and configuration retain their pre-B2 hashes.
Independent replay also verified all842 B2 journal events,296 reserved/completed
physical calls,123 immutable publications, exact5,613,159-byte charges and no
pending operation/write. The original A/B accounts were not changed or reused.
