"""Cold2C result/callback assembly only; no physics, model or messaging."""
import json
from datetime import datetime,timezone
from pathlib import Path
import sys

c=Path(__file__).resolve().parent;parent=c.parent;repo=c.parents[2]
sys.path.insert(0,str(repo))
from tdyssey.native_rgb_training import atomic_json,sha256
from scripts.native_rgb_supervised_stage1 import verify_source,storage_check

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

verify_source(c)
report=read(c/'report/seen-val-report.json');indices=read(c/'report/indices/indices.json')
physics=read(c/'physics-audit.json');review=read(c/'review-postrun.json');raw_review=read(c/'review-raw.json')
coverage=read(c/'report/coverage/coverage-summary.json');dataset=read(c/'dataset-audit/dataset-validation.json')
ledger=read(c/'resource-ledger.json');manifest=read(c/'data/manifest.json');cfg=read(repo/'configs/native_rgb_supervised_stage2c.json')
joint=read(c/'joint-data/manifest.json');repair=read(c/'offline-repair/repair-record.json')
assert physics['passed'] and review['audit_passed'] and raw_review['audit_passed'] and dataset['passed'] and coverage['passed']
assert ledger['charged_controls']==ledger['completed_controls']==10368 and len(manifest['entries'])==162 and manifest['failure'] is None
assert len(indices['natural'])==162 and len(indices['extra'])==157 and len(indices['combined'])==319 and len(indices['development'])==36
assert len(indices['missing_slots'])==5 and len(joint['entries'])==558
assert sha256(repo/cfg['index_spec'])==cfg['index_spec_sha256']==indices['policy_sha256']
assert repair['replay_passed'] and not repair['frozen_sources_modified'] and repair['actual_physics_controls']==0

evidence_files=['PARENT_REVIEW.md','source-identity.json','frozen-requests.json','dedup-reference.json','focused-checks.json',
    'review-prelaunch.json','review-protected-train-files.json','review-raw.json','review-postrun.json','review-offline-repair.json',
    'resource-ledger.json','data/manifest.json','physics-audit.json','counter-baseline.json','DIAGNOSIS.md',
    'report/seen-val-report.json','report/indices/indices.json','report/indices/INDEX_REPORT.md',
    'report/coverage/coverage-summary.json','joint-data/manifest.json','joint-data/physics-audit.json',
    'dataset-audit/dataset-validation.json','dataset-audit/cpu-samples.json','report-relocation.json',
    'report/examples/examples.json','report/supplementary-blocked/selection-from-frozen-index.json',
    'coverage-command.log','coverage-repair-command.log','offline-repair/repair-record.json','offline-repair/source.diff','check-repo.log']
summaries=indices['panel_summaries'];per_letter=report['per_letter']
coverage_totals={key:sum(d['summary'][key] for d in coverage['per_letter'].values()) for key in
    ['all_bins','actual_hit_bins','canonical_known_reachable_bins','canonical_hit_bins','reachable_lower_bound_bins',
     'unconfirmed_bins','proven_closed_hole_bins','independent_contact_trajectories','actual_contact_controls']}
natural_anchors=[int(x['anchor']) for x in indices['natural']]
anchor_bands={f'{low}-{high}':sum(low<=a<=high for a in natural_anchors) for low,high in [(3,16),(17,30),(31,44),(45,60)]}
result=dict(kind='native-rgb-stage2c-final-result.v1',stage='2C',split='seen_val',completed_at_utc=datetime.now(timezone.utc).isoformat(),
    collection_complete=True,physics_gate_passed=True,split_isolation_passed=True,dataset_CPU_passed=True,independent_audit_passed=True,
    actual_controls=10368,charged_controls=10368,execution_slots=162,invalid_transitions=0,physics_substeps=103680,
    collection_wall_seconds=ledger['collection_wall_seconds'],raw_motion=report['raw']['class_counts'],
    valid_h1=9882,valid_h4=9396,h1_motion=report['valid_h1']['class_counts'],
    per_letter_distribution_diagnostic={l:d['distribution_target_diagnostic'] for l,d in per_letter.items()},
    diagnostic_targets_used_to_accept_or_resample_validation=False,all_original_transitions_retained=True,
    panel_summaries=summaries,missing_index_slots=indices['missing_slots'],index_policy_sha256=indices['policy_sha256'],
    index_sha256=sha256(c/'report/indices/indices.json'),index_algorithm_seed=indices['algorithm_seed'],
    natural_anchor=dict(min=min(natural_anchors),max=max(natural_anchors),mean=sum(natural_anchors)/len(natural_anchors),bands=anchor_bands),
    temporal_overlap_counts={name:value['counts'] for name,value in indices['temporal_overlap'].items()},
    coverage=coverage_totals,coverage_counter_scope='zeroed seen_val only; frozen oldTRAIN prior separate; no TRAIN update',
    joint_manifest_sha256=sha256(c/'joint-data/manifest.json'),joint_entries=558,train_trajectories=396,seen_val_trajectories=162,npz_copies=0,
    CPU_direct_samples=4,CPU_batch_shapes={'train':{'image':[2,5,3,96,96],'action':[2,4,2]},'seen_val':{'image':[2,5,3,96,96],'action':[2,4,2]}},
    model_calls=0,optimizer_updates=0,training_iterator_calls=0,training_index_draws=0,unseen_predictions=0,server_jobs=0,packages_created=0,
    supervised_round_controls=39680,all_native_controls=74240,remaining_controls_not_automatic=320,
    historical_native_optimizer_attempts=1899,historical_native_training_seconds=5352.516,
    offline_postprocessing_errors=1,offline_error_repair='Preserved frozen source; separate strict snapshot path-prefix repair,0 extra controls',
    remaining_limits=['five unsupported blocked-proxy extra slots','window temporal correlation','validation is development data, not finaltest',
        'unknown/unhit contour arcs','oldTRAIN M/O-P55/gap defects retained','preexisting3published-media hash failures',
        'no model performance measured'],
    next_action='Actualparentcallbackthenwaiting_parent;stage3/modelworkrequiresnextrelease',callback_required=True,
    callback_tracking='callback-result.json and ../stage-state.json',
    evidence={name:dict(path=str((c/name).resolve()),sha256=sha256(c/name)) for name in evidence_files})

lines=['# Stage2C independent validation and pre-model indices','',
    'All162 independent seen_val trajectories completed10,368 controls. The103,680 substeps pass the unchanged2px physical gate with0invalid transitions. TRAIN396 remains immutable and all558 joint records are split/seed/full-scene/hash isolated. No model, denoiser, optimizer or training iterator ran.','',
    '| Scope | Trajectories | Real controls | Valid H1 | Valid H4 |','|---|---:|---:|---:|---:|',
    '|seen_val|162|10368|9882|9396|','|Frozen TRAIN|396|25344|24156|22968|','',
    'Every validation letter has9 scenes,576 controls,549 H1 and522 H4 candidates. All frames remain retained. Moving>=40% and near-static<=45% are distribution diagnostics, not validation acceptance/resampling gates.','',
    '|Letter|H1 moving /549|Moving %|H1 near-static /549|Near-static %|','|---|---:|---:|---:|---:|']
for letter,d in per_letter.items():
    gate=d['distribution_target_diagnostic'];n=d['valid_h1']['class_counts']
    lines.append(f"|{letter}|{n['moving']}|{100*gate['moving_fraction']:.2f}|{n['near_static']}|{100*gate['near_static_fraction']:.2f}|")
lines+=['',
    'Raw controls:6090 moving/3570 near-static/708 intermediate. Valid H1:6022/3408/452. All18 diagnostic targets happen to be satisfied; no seed, scene, frame or definition was changed to obtain that result. Nine initial spatial classes each have18 independent trajectories, with every boundary class having six4px/six18px/six48px gaps and both corner gaps matching. Full signed/actual gaps and contact×motion tables are retained in [truth statistics](report/h1/motion-statistics.json).','',
    '|Panel|Unique windows|Independent trajectories|H1 moving/static/intermediate|H4 endpoint moving|H4 path any moving|',
    '|---|---:|---:|---|---:|---:|']
for name in ['natural','extra','combined','development']:
    d=summaries[name];n=d['h1_motion'];lines.append(f"|{name}|{d['unique_windows']}|{d['independent_trajectories']}|{n.get('moving',0)}/{n.get('near_static',0)}/{n.get('intermediate',0)}|{d['h4_endpoint_moving']}|{d['h4_path_any_step_moving']}|")
lines+=['',
    f"Natural anchors range3–60, mean{sum(natural_anchors)/162:.2f}; bands3–16/17–30/31–44/45–60 have42/49/30/41 windows. They are uniform hash-ranked within each trajectory's H4-legal anchors under the balanced scene distribution, not real-world operation frequencies. Policy seed2026102903 and all selection rules were frozen before the first physical step. [Indices and exact hashes](report/indices/indices.json).",'',
    'Exactly five extra blocked-proxy slots lack support: C/F/M/S/T. They remain empty, with no physical replacement or category substitution. Every selected window retains all crossing H1 conditions; actual inference IDs are unique. The36-window development subset uses32 independent trajectories. Combined319 windows have157 same-trajectory pairs:35 share H4 states,31 H4 transitions,17 future transitions,19 H1 states. Development has4 same-trajectory pairs,2 H4 overlaps and0 H1 overlaps. These are correlated windows, not319 independent trials.','',
    'H1 and H4 truth labels remain separate:99 combined windows are H1 near-static,84 have a near-static H4 endpoint, and13 H1-static windows contain a later moving step. Future H4 prediction must feed back predicted RGB; no future truth image is to be substituted as prediction feedback. This dataset remains seen-letter development validation, not an unbiased final test.','',
    'Validation actual coverage is809/1408 arcs; canonical R925 has740 hit, confirmed lower bound K994, unknown U296, closed holes C118. All162 trajectories made contact (7240 real contact controls). Every own actual/planned prefix matches from a zero baseline, with0 saved-coverage differences or closed-hole conflicts. Validation evidence never updates TRAIN. [Coverage](report/coverage/coverage-summary.md).','',
    'The [joint reference directory](joint-data/manifest.json) points to original2A/2B TRAIN and2C seen_val NPZ/metadata, with0 data copies. Physical audits, protocols, collector snapshots, every source hash and whole-trajectory/seed/full-scene separation are enforced. NativeRGBDataset keeps the original5 CHW RGB frames/4 actions, uint8 LRU and float-on-demand interface. Four predeclared direct items validate s[a-3:a+1],u[a-3:a],u[a],s[a+1], imagesfloat32[0,1] and actions[-1,1]. Both split batches are[2,5,3,96,96]/[2,4,2], cache≤1MiB and torch was not imported. [CPU evidence](dataset-audit/dataset-validation.md).','',
    'Validation checks:46 focused CPU/mock tests and162 pure-geometry initializations passed;37 frozen source/input hashes and1027 original2A/2B files remain intact; independent raw/prefix/selection/index/reference audits pass. The source-only checker still reports the same3 preexisting published-media JSON hash/size problems.','',
    'One postcollection offline coverage invocation rejected its valid prior because its strict snapshot-name comparison omitted source-snapshot/. Original source/log remain preserved. A separate v1_1 artifact copy fixes only that exact path and root discovery for its location; its successful replay preserves all SHA/counter guards. It adds0 controls and changes none of the37 frozen files. [Repair record](offline-repair/repair-record.json) and [two-line diff](offline-repair/source.diff). The main report ran once in a fresh staging directory, then16 owned outputs were relocated with identical hashes to coexist with coverage; no stale paths remain.','',
    'The four configured true-frame examples are [here](report/examples/saved-continuous-examples.png). M has no blocked-proxy support, so its explicitly marked fallback remains moving. A separate [E blocked-proxy sequence](report/supplementary-blocked/saved-continuous-examples.png) selects an existing frozen extra window by its original hash; it changes no indices and does not claim M support or measured wall force. All frames were rendered from saved states, with0 physics steps.','',
    'Resources: collection357.656s/10,368 controls. Supervised round39,680 controls; all-native74,240. Remaining320 controls are not automatically authorized. Historical1899 optimizer attempts/5352.516 training seconds remain recorded; this round adds0 model/optimizer/training-index/unseen/server/package work. All old data/weights remain retained within the shared3GiB artifact limit.','',
    'See [diagnosis](DIAGNOSIS.md), [source scope](SOURCE_CHANGE.md), [commands](COMMANDS.md), [independent final review](review-postrun.md), and [full validation report](report/SEEN_VAL_REPORT.md). Finish with the actual parent callback and waiting_parent. Stage3 microfit/training requires a new explicit parent instruction.']
(c/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
result['footprint_measured_before_result_write']=storage_check(cfg,parent);atomic_json(c/'stage2c-result.json',result)
state=read(parent/'stage-state.json');state.update(status='ready_to_report',stage2c_complete=True,
    next_action='Actuallysend2Cfullresults/oneofflineguardrepairtoparent;recordreceiptwaiting_parentthenSTOP.No3/model.',
    stage2c_result_file='stage2c/stage2c-result.json',stage2c_result_sha256=sha256(c/'stage2c-result.json'),parent_review_received=False)
state['stage2c_index_summary']={name:dict(unique=d['unique_windows'],independent_trajectories=d['independent_trajectories']) for name,d in summaries.items()}
state['stage2c_index_sha256']=sha256(c/'report/indices/indices.json');state['stage2c_joint_manifest_sha256']=sha256(c/'joint-data/manifest.json')
atomic_json(parent/'stage-state.json',state);footprint=storage_check(cfg,parent)
state['resources']['combined_persistent_bytes_at_report']=footprint['combined_bytes'];state['resources']['stage2c_final_storage_before_callback']=footprint;atomic_json(parent/'stage-state.json',state)

body=['[阶段2C回传] 独立seen_val采集、CPU引用适配和预冻结评测索引已完成，请父审查。本实施回传成功即记录waiting_parent并停止；阶段3/任何模型工作尚未执行。','',
    '实际162/162固定槽×64=10368 charged=actual controls，357.656s，103680个100Hz子步、0invalid/0采集失败/0替代/0重放。最大子步穿透1.051749px，原2px阈值未动。18字母各9空间/576controls/549H1/522H4；总体9882H1、9396H4。每边界空间4/18/48各6条，corner双墙间隙匹配，center无requestedgap。970历史seed/988旧fullscenehash排除，TRAIN396原样保留。','',
    '原始真值分布：raw moving6090/static3570/I708；H1 moving6022/static3408/I452。40%/45%在seen_val仅作分布诊断，没有删帧/换seed/再采。逐字母H1全分母549：','',
    '|字母|moving/549|moving%|near_static/549|static%|','|---|---:|---:|---:|---:|']
for letter,d in per_letter.items():
    gate=d['distribution_target_diagnostic'];n=d['valid_h1']['class_counts'];body.append(f"|{letter}|{n['moving']}|{100*gate['moving_fraction']:.2f}|{n['near_static']}|{100*gate['near_static_fraction']:.2f}|")
body+=['',
    '覆盖/隔离：动态actual-hit/planned从零、只累计真实seen_val，旧TRAIN prior单列固定hash。162真实start及计数/完整候选排序回放均匹配，0savedcoverage差异、0closedconflict。seen_val实际H809/1408；R925其中hit740；K994下界、U296未确认、C118封闭洞。162/162真实接触，7240contactcontrols；角度/运动/实际空间/双墙signedgap/接触×运动全表保留。验证没有反馈修改TRAIN，旧M角落/O-P55/80缺gapcells仍保留。','',
    '索引规则和seed2026102903在采前冻结，policySHAf55fad565e6cc013e165529f81f84cf1ef46a67985b2d3f1bdae391545f6c6a1。9396个H4合法候选anchors3..60，全部7历史+未来转移有效；自然每轨uniform hash选1，共162，anchor均值30.07，四段3–16/17–30/31–44/45–60=42/49/30/41，未固定末尾。额外按rare-once/leasttraj/hash：157；5个blockedproxy缺槽C/F/M/S/T留空，不替换。combined319唯一窗口，dev36/32独立轨迹，各letter1natural+1额外truthmoving。','',
    '面板真实H1 M/S/I：natural100/50/12；extra90/49/18；combined190/99/30；dev31/4/1。combined条件标签rotation138、boundary-moving74、blockedproxy17、SC35、SN64、I30，标签交叉但每window只推理1次。H4endpointmoving229/path-any-moving228，与H1分开；13个H1static窗在4步路径内后来moving，39个H1nonmoving有moving的H4endpoint。未来必须预测RGB自回归反馈，本阶段0预测。','',
    '相关性：combined162独立轨迹/319窗，同轨pair157，其中H4state35/H4transition31/future17/H1state19重叠；dev36窗仅32独立轨迹，4pair含2H4重叠/0H1重叠。完整seen_val是开发验证，不称无偏最终测试或现实频率自然分布。','',
    'CPU数据入口：joint-data零复制引用162v1+234v2 TRAIN及162seen_val，558全轨迹/seed/fullscene独立，NPZ/meta/collector快照/物理audit/协议hash全部强验；TRAIN396/seen_val162/unseen0。NativeRGBDataset必要reference分支，原取帧/渲染/归一化/LRU字节不变，training模块未改。实际4预定direct items(2A000@3、2B000@60、val000@3、val161@60)检查history s[a-3:a+1]、历史u[a-3:a]、当前u[a]、target s[a+1]；两个split批次各image[2,5,3,96,96]float32[0,1]、actions[2,4,2][-1,1]。uint8LRU≤1MiB，torch_imported=False，data_identity纯数据函数接受joint目录，0训练iterator。','',
    '检查：46CPU/mock+162纯几何预检通过；37冻结源/1027旧TRAIN文件原SHA不变，独立raw+natural/extra/dev全hash选择+标签/重叠+558引用+4CPU样本审计通过。固定2B sampler8f84和1B class/physicsfd358完全未改；0模型/denoiser/权重加载/优化器/训练索引消费/未见预测/服务器/ZIP。check_repo仍仅3个既有published-media JSON size/hash问题。','',
    '必须披露的后处理错误：原离线coverage guard比较snapshot名漏了source-snapshot/前缀，拒绝了正确SHA的prior；不是物理采集异常，错误日志保留。冻结37源未改，另存v1_1离线副本只修strict路径+异地root发现、保留SHAguard，独立AST/diff已审；一次成功回放0新增控制，162prefix/coverage全匹配。主report先跑fresh staging后16文件同SHA迁入report（原coverage已建父目录），路径/命令/迁移记录保留、无stale路径。','',
    '4主真值例/8连续帧+2GIF已完成。M blockedproxy无支持，主例明确fallbackmoving；另从既冻结extra14个真proxy窗按既定hash最小补E:left-top seen-val-2c-093:31，frames28..35，1行/1GIF，不改319评测索引、不假称M被挡或测得墙力。', '',
    f"资源：监督轮39680controls，旧34560+新=74240，余320不自动花。旧1899optimizer attempts/5352.516训练秒保留，新轮0。回传前native全部持久产物{footprint['combined_bytes']}B={footprint['combined_bytes']/1024**3:.4f}GiB<3GiB；C盘可用{footprint['disk_free_bytes']/1024**3:.2f}GiB，无下载/安装/新ZIP/删旧权重。",'',
    '实际工作目录__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main；SDL dummy、CUDA空、HFoffline、MPLAgg。实际分步命令（失败与成功日志均保留）：',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c.py prepare --physics-root ../Tdyssey/third_party/diffusion_policy',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c.py freeze --physics-root ../Tdyssey/third_party/diffusion_policy',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c.py collect --physics-root ../Tdyssey/third_party/diffusion_policy --execute',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c.py audit --physics-root ../Tdyssey/third_party/diffusion_policy',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c_report.py --data artifacts/native-rgb-supervised-20261029/stage2c/data --output artifacts/native-rgb-supervised-20261029/stage2c/report-main-staging --physics-root ../Tdyssey/third_party/diffusion_policy --examples',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2c_coverage.py --stage artifacts/native-rgb-supervised-20261029/stage2c（strictsnapshot路径guard拒绝；原失败log保存）',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B artifacts/native-rgb-supervised-20261029/stage2c/offline-repair/native_rgb_stage2c_coverage_v1_1.py --stage artifacts/native-rgb-supervised-20261029/stage2c（成功，0新增控制）',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B artifacts/native-rgb-supervised-20261029/stage2c/dataset-audit/run-cold-dataset-audit.py --execute-after-root-cold-confirmation',
    '46项检查完整命令见focused-checks.json/log；独立复算命令/报告命令/迁移/SHA记录见COMMANDS.md及各execution JSON。','',
    '绝对证据路径：']
for name in ['REPORT.md','stage2c-result.json','DIAGNOSIS.md','data/manifest.json','resource-ledger.json','physics-audit.json',
    'source-identity.json','review-postrun.json','report/seen-val-report.json','report/indices/indices.json','report/indices/INDEX_REPORT.md',
    'joint-data/manifest.json','dataset-audit/dataset-validation.json','report/coverage/coverage-summary.json',
    'offline-repair/repair-record.json','report/examples/saved-continuous-examples.png','report/supplementary-blocked/saved-continuous-examples.png','COMMANDS.md']:
    body.append(str((c/name).resolve()))
body+=['','请审查数据/索引、5缺槽/时间相关性及离线guard修复披露，再明确下一阶段指令。当前仅2C完成，不自行进入阶段3微拟合。']
(c/'callback-stage2c.txt').write_text('\n'.join(body)+'\n',encoding='utf-8')
print(json.dumps(dict(status='ready_to_report',result_sha256=sha256(c/'stage2c-result.json'),index_sha256=sha256(c/'report/indices/indices.json'),joint_sha256=sha256(c/'joint-data/manifest.json'),combined_bytes=footprint['combined_bytes'],callback_chars=len('\n'.join(body)))))
