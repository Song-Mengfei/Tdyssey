# Stage2C independent validation and pre-model indices

All162 independent seen_val trajectories completed10,368 controls. The103,680 substeps pass the unchanged2px physical gate with0invalid transitions. TRAIN396 remains immutable and all558 joint records are split/seed/full-scene/hash isolated. No model, denoiser, optimizer or training iterator ran.

| Scope | Trajectories | Real controls | Valid H1 | Valid H4 |
|---|---:|---:|---:|---:|
|seen_val|162|10368|9882|9396|
|Frozen TRAIN|396|25344|24156|22968|

Every validation letter has9 scenes,576 controls,549 H1 and522 H4 candidates. All frames remain retained. Moving>=40% and near-static<=45% are distribution diagnostics, not validation acceptance/resampling gates.

|Letter|H1 moving /549|Moving %|H1 near-static /549|Near-static %|
|---|---:|---:|---:|---:|
|A|348|63.39|177|32.24|
|B|331|60.29|196|35.70|
|C|332|60.47|198|36.07|
|E|321|58.47|201|36.61|
|F|324|59.02|203|36.98|
|H|355|64.66|163|29.69|
|I|325|59.20|197|35.88|
|J|335|61.02|174|31.69|
|L|370|67.40|155|28.23|
|M|307|55.92|216|39.34|
|O|313|57.01|209|38.07|
|P|371|67.58|157|28.60|
|S|331|60.29|194|35.34|
|T|383|69.76|140|25.50|
|U|358|65.21|154|28.05|
|V|299|54.46|234|42.62|
|X|290|52.82|237|43.17|
|Y|329|59.93|203|36.98|

Raw controls:6090 moving/3570 near-static/708 intermediate. Valid H1:6022/3408/452. All18 diagnostic targets happen to be satisfied; no seed, scene, frame or definition was changed to obtain that result. Nine initial spatial classes each have18 independent trajectories, with every boundary class having six4px/six18px/six48px gaps and both corner gaps matching. Full signed/actual gaps and contact×motion tables are retained in truth statistics (external historical reference: `report/h1/motion-statistics.json`).

|Panel|Unique windows|Independent trajectories|H1 moving/static/intermediate|H4 endpoint moving|H4 path any moving|
|---|---:|---:|---|---:|---:|
|natural|162|162|100/50/12|113|113|
|extra|157|157|90/49/18|116|115|
|combined|319|162|190/99/30|229|228|
|development|36|32|31/4/1|32|32|

Natural anchors range3–60, mean30.07; bands3–16/17–30/31–44/45–60 have42/49/30/41 windows. They are uniform hash-ranked within each trajectory's H4-legal anchors under the balanced scene distribution, not real-world operation frequencies. Policy seed2026102903 and all selection rules were frozen before the first physical step. Indices and exact hashes (external historical reference: `report/indices/indices.json`).

Exactly five extra blocked-proxy slots lack support: C/F/M/S/T. They remain empty, with no physical replacement or category substitution. Every selected window retains all crossing H1 conditions; actual inference IDs are unique. The36-window development subset uses32 independent trajectories. Combined319 windows have157 same-trajectory pairs:35 share H4 states,31 H4 transitions,17 future transitions,19 H1 states. Development has4 same-trajectory pairs,2 H4 overlaps and0 H1 overlaps. These are correlated windows, not319 independent trials.

H1 and H4 truth labels remain separate:99 combined windows are H1 near-static,84 have a near-static H4 endpoint, and13 H1-static windows contain a later moving step. Future H4 prediction must feed back predicted RGB; no future truth image is to be substituted as prediction feedback. This dataset remains seen-letter development validation, not an unbiased final test.

Validation actual coverage is809/1408 arcs; canonical R925 has740 hit, confirmed lower bound K994, unknown U296, closed holes C118. All162 trajectories made contact (7240 real contact controls). Every own actual/planned prefix matches from a zero baseline, with0 saved-coverage differences or closed-hole conflicts. Validation evidence never updates TRAIN. [Coverage](report/coverage/coverage-summary.md).

The [joint reference directory](joint-data/manifest.json) points to original2A/2B TRAIN and2C seen_val NPZ/metadata, with0 data copies. Physical audits, protocols, collector snapshots, every source hash and whole-trajectory/seed/full-scene separation are enforced. NativeRGBDataset keeps the original5 CHW RGB frames/4 actions, uint8 LRU and float-on-demand interface. Four predeclared direct items validate s[a-3:a+1],u[a-3:a],u[a],s[a+1], imagesfloat32[0,1] and actions[-1,1]. Both split batches are[2,5,3,96,96]/[2,4,2], cache≤1MiB and torch was not imported. [CPU evidence](dataset-audit/dataset-validation.md).

Validation checks:46 focused CPU/mock tests and162 pure-geometry initializations passed;37 frozen source/input hashes and1027 original2A/2B files remain intact; independent raw/prefix/selection/index/reference audits pass. The source-only checker still reports the same3 preexisting published-media JSON hash/size problems.

One postcollection offline coverage invocation rejected its valid prior because its strict snapshot-name comparison omitted source-snapshot/. Original source/log remain preserved. A separate v1_1 artifact copy fixes only that exact path and root discovery for its location; its successful replay preserves all SHA/counter guards. It adds0 controls and changes none of the37 frozen files. Repair record (external historical reference: `offline-repair/repair-record.json`) and two-line diff (external historical reference: `offline-repair/source.diff`). The main report ran once in a fresh staging directory, then16 owned outputs were relocated with identical hashes to coexist with coverage; no stale paths remain.

The four configured true-frame examples are here (external historical reference: `report/examples/saved-continuous-examples.png`). M has no blocked-proxy support, so its explicitly marked fallback remains moving. A separate E blocked-proxy sequence (external historical reference: `report/supplementary-blocked/saved-continuous-examples.png`) selects an existing frozen extra window by its original hash; it changes no indices and does not claim M support or measured wall force. All frames were rendered from saved states, with0 physics steps.

Resources: collection357.656s/10,368 controls. Supervised round39,680 controls; all-native74,240. Remaining320 controls are not automatically authorized. Historical1899 optimizer attempts/5352.516 training seconds remain recorded; this round adds0 model/optimizer/training-index/unseen/server/package work. All old data/weights remain retained within the shared3GiB artifact limit.

See [diagnosis](DIAGNOSIS.md), [source scope](SOURCE_CHANGE.md), [commands](COMMANDS.md), [independent final review](review-postrun.md), and [full validation report](report/SEEN_VAL_REPORT.md). The actual parent callback succeeded; receipt (external historical reference: `callback-result.json`) is saved and state is waiting_parent. Stage3 microfit/training requires a new explicit parent instruction.
