# Stage2B remaining formal TRAIN result

The fixed234-slot batch completed14,976 controls with0invalid transitions. New2B and the merged396-trajectory TRAIN both meet the per-letter H1 moving>=40% and near-static<=45% targets for all18 letters. Old2A remains unchanged and its M failure remains reported. These are sampling targets; no model prediction or optimizer update was performed.

| Cohort | Trajectories | Real controls | Valid H1 | Valid H4 | H1 letter targets |
|---|---:|---:|---:|---:|---|
|2A|162|10368|9882|9396|17/18; M failed|
|2B|234|14976|14274|13572|18/18|
|merged|396|25344|24156|22968|18/18|

Every merged letter has22 independent trajectories,1408 controls,1342 H1 and1276 H4 candidates. H4 uses real0.4-second endpoint/path statistics, separately from the H1 gate. No static frame was removed to meet targets.

| Letter | Moving /1342 | Moving % | Near-static /1342 | Near-static % | Passed |
|---|---:|---:|---:|---:|---|
|A|888|66.17|405|30.18|yes|
|B|813|60.58|490|36.51|yes|
|C|759|56.56|532|39.64|yes|
|E|811|60.43|457|34.05|yes|
|F|808|60.21|460|34.28|yes|
|H|816|60.80|483|35.99|yes|
|I|798|59.46|486|36.21|yes|
|J|873|65.05|418|31.15|yes|
|L|798|59.46|477|35.54|yes|
|M|745|55.51|540|40.24|yes|
|O|760|56.63|517|38.52|yes|
|P|795|59.24|511|38.08|yes|
|S|859|64.01|424|31.59|yes|
|T|879|65.50|405|30.18|yes|
|U|906|67.51|381|28.39|yes|
|V|793|59.09|501|37.33|yes|
|X|783|58.35|493|36.74|yes|
|Y|808|60.21|467|34.80|yes|

M retains material local weaknesses. Left-top H1 was7 moving/52 static of61 in2A, then97/20 of122 in2B. Left-bottom was2/57 of61, then75/45 of122; merged left-bottom still has102/183=55.74% static. New right-top has1/61 moving and58/61 static; merged right-top88/122=72.13% static. Aggregate M is745/1342 moving55.51%,540/1342 static40.24%. These different seeds/gaps/initial arcs do not isolate a causal selector effect. See [diagnosis](DIAGNOSIS.md).

Actual contour hits increased from827/1408 in2A to1023/1408 merged (+196 distinct hit arcs). Canonical reachable R=925 with923 hit; confirmed reachable lower bound K=1025, unconfirmed U=265, closed holes C=118. O55/P55 remain known but unhit: old independent planned attempts4/6, new attempts0/0, actual hits0/0. O/P gained10/13 other newly hit arcs. All396 trajectories made actual contact. Every234 count/ranking prefix replay matches, with0 saved-coverage differences and0 closed-hole conflicts. See [coverage](report/coverage/COVERAGE_DIAGNOSIS.md).

The merged requested spatial classes have44 trajectories each. Per letter,19 or20 of24 initial boundary-space/gap combinations are supported;80 cells remain missing across18 letters. Both wall gaps at corners were verified and center has no requested gap tier. See the [exact missing combinations](gap-quota-audit.md).

| Proposed stratum | Unique valid H1 | Independent trajectories | Draws per unique support at70/10/10/10 |
|---|---:|---:|---:|
|moving|14692|395|1.151|
|intermediate|1017|368|2.375|
|near_static_contact|1176|357|2.054|
|near_static_no_contact|7271|386|0.332|

The24150-draw exact-ratio example is analysis only: replacement is needed; no training index was drawn/materialized. Global strata have all9 requested spaces, but per-letter support remains uneven: X static-contact has30 unique windows/17 trajectories (~4.47 draws per unique for its proposed quota), P intermediate36 (~3.72), and P bottom static-contact has0 windows. Repeats are not new physical evidence.

The2A originals remain417-file hash matched. The [reference-only TRAIN view](report/merged-train-manifest.json) retains162 v1 and234 v2 selector entries and original NPZ/metadata hashes, with0 data copies. Its base_root is the supervised experiment folder. `read_merged_records(view_path, supervised_root)` in the new report script verifies every source and loads records without constructing a training dataset.

Validation:15 exact-launch focused checks plus8 report checks passed;234 pure geometric initializations passed;149760100Hz substeps passed the unchanged2px threshold (maximum1.05175px);234 full ranking/count-prefix replays and149 cold report cross-checks matched. The source-only repository checker retains its3 preexisting published-media JSON hash/size failures, recorded in check-repo.log.

Resources:2B collection480.062s,14,976 controls. New supervised total29,312 controls; old+new native63,872. Historical1899 optimizer attempts/5352.516 training seconds remain recorded; this supervised round adds0 optimizer/model/index/validation/unseen/server/package work. Old weights/reports remain retained and the shared artifact limit is3GiB.

Frozen sampling version: motion-room-planned-attempts-v2. Only initialization ranking changes; the continuous1B controller/limits and finite-force-pd-v1 protocol are unchanged. See [source change](SOURCE_CHANGE.md), [commands](COMMANDS.md), physical audit (external historical reference: `physics-audit.json`), [independent review](review-postrun.md), [full report](report/stage2b-report.md), and eight saved-frame examples (external historical reference: `report/examples/examples-1.png`).

The actual callback to the parent chat succeeded; receipt (external historical reference: `callback-result.json`) is saved and state is waiting_parent. Stage2C validation and stages3–6 require the next parent release.
