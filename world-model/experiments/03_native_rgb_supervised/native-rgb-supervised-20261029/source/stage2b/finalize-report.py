"""Assemble completed2B evidence; this never executes physics/model or messages."""
import json
from datetime import datetime,timezone
from pathlib import Path
import sys

b=Path(__file__).resolve().parent;parent=b.parent;repo=b.parents[2]
sys.path.insert(0,str(repo))
from tdyssey.native_rgb_training import atomic_json,sha256
from scripts.native_rgb_supervised_stage1 import storage_check,verify_source

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

verify_source(b)
report=read(b/'report/stage2b-report.json');audit=read(b/'physics-audit.json');review=read(b/'review-postrun.json')
coverage=read(b/'report/coverage/coverage-summary.json');ledger=read(b/'resource-ledger.json');gap=read(b/'gap-quota-audit.json')
view=read(b/'report/merged-train-manifest.json');manifest=read(b/'data/manifest.json');cfg=read(repo/'configs/native_rgb_supervised_stage2b.json')
assert audit['passed'] and review['audit_passed'] and gap['passed'] and manifest['failure'] is None
assert ledger['charged_controls']==ledger['completed_controls']==14976 and len(manifest['entries'])==234
assert not coverage['saved_coverage_differences'] and not coverage['reachability_conflicts']
assert coverage['attempt_prefix_audit']['mismatching_trajectories']==0
assert report['cohorts']['2B']['all_h1_letter_gates_passed'] and report['cohorts']['merged']['all_h1_letter_gates_passed']
assert len(view['entries'])==396 and sum(x['controls'] for x in view['entries'])==25344

core_files=['PARENT_REVIEW.md','frozen-requests.json','dedup-reference.json','source-identity.json',
    'focused-checks.json','review-prelaunch.json','preserved2A-files.json','data/manifest.json','resource-ledger.json',
    'physics-audit.json','review-postrun.json','DIAGNOSIS.md','gap-quota-audit.json',
    'report/stage2b-report.json','report/merged-train-manifest.json','report/coverage/coverage-summary.json',
    'report/coverage/O-P-arc55-focus.json','report/coverage/COVERAGE_DIAGNOSIS.md','report/examples/examples.json',
    'report/examples/examples-1.png','check-repo.log']
result=dict(kind='native-rgb-stage2b-final-result.v1',stage='2B',completed_at_utc=datetime.now(timezone.utc).isoformat(),
    collection_complete=True,physics_gate_passed=True,quality_gate_passed_merged=True,
    independent_audit_passed=True,charged_controls=14976,actual_controls=14976,executed_slots=234,
    collection_wall_seconds=ledger['collection_wall_seconds'],invalid_transitions=0,physics_substeps=149760,
    cohorts={name:dict(trajectories=c['recorded_slots'],controls=c['actual_controls'],h1=c['actual_h1'],
        h4=c['h4']['valid_candidates'],all_h1_letter_gates_passed=c['all_h1_letter_gates_passed']) for name,c in report['cohorts'].items()},
    merged_per_letter_h1={letter:c['h1_gate'] for letter,c in report['cohorts']['merged']['per_letter'].items()},
    selector_versions=view['selector_versions'],protocol_sha256=manifest['protocol_sha256'],
    model_inference_calls=0,optimizer_updates=0,training_index_draws=0,validation_controls=0,
    unseen_predictions=0,server_jobs=0,packages_created=0,
    supervised_round_controls=29312,all_native_controls=63872,
    historical_native_optimizer_attempts=1899,historical_native_training_seconds=5352.516,
    formal_train_controls=25344,formal_train_trajectories=396,merged_data_copies=0,
    residual_limitations=['Mleft-bottom/right-toplocalnear-staticfailures','O55/P55remainunhit',
        '265unconfirmedcontourarcs','80missinginitialboundary-gapcells','stratumrepeat/concentrationlimits',
        '2Avs2Bscene/gap/seedchangespreventisolatedcausalattribution','known3publishedmediahashcheckfailures'],
    next_phase='Waitingforparentreview;2Cvalidationand3-6arenotauthorized',callback_required=True,
    callback_tracking='callback-result.json and ../stage-state.json',
    evidence={name:dict(path=str((b/name).resolve()),sha256=sha256(b/name)) for name in core_files})

lines=['# Stage2B remaining formal TRAIN result','',
    'The fixed234-slot batch completed14,976 controls with0invalid transitions. New2B and the merged396-trajectory TRAIN both meet the per-letter H1 moving>=40% and near-static<=45% targets for all18 letters. Old2A remains unchanged and its M failure remains reported. These are sampling targets; no model prediction or optimizer update was performed.','',
    '| Cohort | Trajectories | Real controls | Valid H1 | Valid H4 | H1 letter targets |',
    '|---|---:|---:|---:|---:|---|']
for name,c in result['cohorts'].items():lines.append(f"|{name}|{c['trajectories']}|{c['controls']}|{c['h1']}|{c['h4']}|{'18/18' if c['all_h1_letter_gates_passed'] else '17/18; M failed'}|")
lines+=['','Every merged letter has22 independent trajectories,1408 controls,1342 H1 and1276 H4 candidates. H4 uses real0.4-second endpoint/path statistics, separately from the H1 gate. No static frame was removed to meet targets.','',
    '| Letter | Moving /1342 | Moving % | Near-static /1342 | Near-static % | Passed |',
    '|---|---:|---:|---:|---:|---|']
for letter,c in report['cohorts']['merged']['per_letter'].items():
    h=c['h1_gate'];n=c['h1']['class_counts'];lines.append(f"|{letter}|{n['moving']}|{100*h['moving_fraction']:.2f}|{n['near_static']}|{100*h['near_static_fraction']:.2f}|yes|")
lines+=['',
    'M retains material local weaknesses. Left-top H1 was7 moving/52 static of61 in2A, then97/20 of122 in2B. Left-bottom was2/57 of61, then75/45 of122; merged left-bottom still has102/183=55.74% static. New right-top has1/61 moving and58/61 static; merged right-top88/122=72.13% static. Aggregate M is745/1342 moving55.51%,540/1342 static40.24%. These different seeds/gaps/initial arcs do not isolate a causal selector effect. See [diagnosis](DIAGNOSIS.md).','',
    'Actual contour hits increased from827/1408 in2A to1023/1408 merged (+196 distinct hit arcs). Canonical reachable R=925 with923 hit; confirmed reachable lower bound K=1025, unconfirmed U=265, closed holes C=118. O55/P55 remain known but unhit: old independent planned attempts4/6, new attempts0/0, actual hits0/0. O/P gained10/13 other newly hit arcs. All396 trajectories made actual contact. Every234 count/ranking prefix replay matches, with0 saved-coverage differences and0 closed-hole conflicts. See [coverage](report/coverage/COVERAGE_DIAGNOSIS.md).','',
    'The merged requested spatial classes have44 trajectories each. Per letter,19 or20 of24 initial boundary-space/gap combinations are supported;80 cells remain missing across18 letters. Both wall gaps at corners were verified and center has no requested gap tier. See the [exact missing combinations](gap-quota-audit.md).','',
    '| Proposed stratum | Unique valid H1 | Independent trajectories | Draws per unique support at70/10/10/10 |',
    '|---|---:|---:|---:|']
support=report['cohorts']['merged']['strata_support'];ratios=support['feasibility_analysis_only']['required_draws_per_unique_support']
for name in ['moving','intermediate','near_static_contact','near_static_no_contact']:
    c=support['strata'][name];lines.append(f"|{name}|{c['unique_windows']}|{c['independent_trajectories']}|{ratios[name]:.3f}|")
lines+=['',
    'The24150-draw exact-ratio example is analysis only: replacement is needed; no training index was drawn/materialized. Global strata have all9 requested spaces, but per-letter support remains uneven: X static-contact has30 unique windows/17 trajectories (~4.47 draws per unique for its proposed quota), P intermediate36 (~3.72), and P bottom static-contact has0 windows. Repeats are not new physical evidence.','',
    'The2A originals remain417-file hash matched. The [reference-only TRAIN view](report/merged-train-manifest.json) retains162 v1 and234 v2 selector entries and original NPZ/metadata hashes, with0 data copies. Its base_root is the supervised experiment folder. `read_merged_records(view_path, supervised_root)` in the new report script verifies every source and loads records without constructing a training dataset.','',
    'Validation:15 exact-launch focused checks plus8 report checks passed;234 pure geometric initializations passed;149760100Hz substeps passed the unchanged2px threshold (maximum1.05175px);234 full ranking/count-prefix replays and149 cold report cross-checks matched. The source-only repository checker retains its3 preexisting published-media JSON hash/size failures, recorded in check-repo.log.','',
    'Resources:2B collection480.062s,14,976 controls. New supervised total29,312 controls; old+new native63,872. Historical1899 optimizer attempts/5352.516 training seconds remain recorded; this supervised round adds0 optimizer/model/index/validation/unseen/server/package work. Old weights/reports remain retained and the shared artifact limit is3GiB.','',
    'Frozen sampling version: motion-room-planned-attempts-v2. Only initialization ranking changes; the continuous1B controller/limits and finite-force-pd-v1 protocol are unchanged. See [source change](SOURCE_CHANGE.md), [commands](COMMANDS.md), [physical audit](physics-audit.json), [independent review](review-postrun.md), [full report](report/stage2b-report.md), and [eight saved-frame examples](report/examples/examples-1.png).','',
    'This stage ends with an actual callback to the parent chat, followed by waiting_parent. Stage2C validation and stages3–6 require the next parent release.']
(b/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
result['footprint_measured_before_result_write']=storage_check(cfg,parent)
atomic_json(b/'stage2b-result.json',result)
state=read(parent/'stage-state.json');state['status']='ready_to_report';state['next_action']='Actuallysend2Bevidencetoparent,recordcallbackreceipt/waiting_parentthenSTOP;no2C/model.'
state['stage2b_result_file']='stage2b/stage2b-result.json';state['stage2b_result_sha256']=sha256(b/'stage2b-result.json')
state['stage2b_gate']={letter:c['h1_gate'] for letter,c in report['cohorts']['merged']['per_letter'].items()}
state['stage2b_complete']=True;state['parent_review_received']=False
atomic_json(parent/'stage-state.json',state)
footprint=storage_check(cfg,parent)
state['resources']['combined_persistent_bytes_at_report']=footprint['combined_bytes'];state['resources']['stage2b_final_storage_before_callback']=footprint
atomic_json(parent/'stage-state.json',state)

body=['[阶段2B回传] 已完成唯一放行的剩余正式TRAIN采集与审计；请审查后给下一条阶段指令。本实施回传成功后写waiting_parent并停止，2C验证和3–6仍未执行。','',
    '234/234新独立轨迹×64=14976 charged=actual controls，149760个100Hz子步，0invalid/0失败/0替代/0重放。最大子步穿透1.051749px，2px阈值未动。采集480.062s。没有model/optimizer/index/seen_val/unseen/server/ZIP工作。','',
    '三组原始分母：2A162条/10368controls/9882H1/9396H4（M近静止253/549=46.08%失败原样保留）；2B234条/14976/14274/13572（18/18通过）；合并396条/25344/24156/22968（18/18通过）。每字母22条、1408controls、1342H1、1276H4。以下是合并原始有效H1，全量无删静止：','',
    '|字母|moving/1342|moving%|near_static/1342|static%|40/45门槛|','|---|---:|---:|---:|---:|---|']
for letter,c in report['cohorts']['merged']['per_letter'].items():
    h=c['h1_gate'];n=c['h1']['class_counts'];body.append(f"|{letter}|{n['moving']}|{100*h['moving_fraction']:.2f}|{n['near_static']}|{100*h['near_static_fraction']:.2f}|通过|")
body+=['',
    'M局部未达仍明确保留：旧左上7M/52S/61→新97M/20S/122；旧左下2M/57S/61→新75M/45S/122，合并左下静止102/183=55.74%；新右上1M/58S/61、合并右上88/122=72.13%静止。M总体745M/540S/1342=55.51%/40.24%。不同seed/gap/初始arc会混杂比较，不能归因为单一排序改动。','',
    '覆盖：实际H2A827/1408→merged1023/1408，新增196实际命中弧段；R925中hit923，K已证实下界1025，U265未确认，C118封闭洞。O55旧planned4、P55旧planned6仍均0hit；新2B对55均0新增planned且0自然hit。O/P新增其它实际弧段10/13，未宣称55已修复/不可达。396/396轨迹真实接触；234逐轨actual/planned计数前缀与完整排序/choice复现一致，0savedcoverage差异、0封闭孔冲突。','',
    '四层合并H1unique/独立轨迹：moving14692/395、intermediate1017/368、static-contact1176/357、static-no-contact7271/386。70/10/10/10假设24150 draws需有放回，各层draws/unique约1.151/2.375/2.054/.332；0实际索引。X static-contact仅30窗口/17轨迹、约4.47倍重复；P intermediate36约3.72倍，Pbottom static-contact0窗口。详细逐字母/空间支持保持。','',
    '空间：九类合并各44轨迹，父配额/+1+2间隙轮转全符；8字母仅19/24、10字母20/24个边界空间×gapcells，合计80缺组合明确列出，corner双墙真实gap逐项匹配，center无requestedgap标签。','',
    '来源/可用TRAIN：仅允许3个初始化helper更新v2，完整ContinuousContactSampler/RECOVERY_LIMITS/PHASES/side_sign/动作距离与物理fd358未动。samplerSHA8f84baa2a7a6f4a89ffb2191d09e4bf86acef211b25e6b80918169ea86fd8cd8；collector e18d06c841f6299f3ceb607d887f4e5e24d05e7f43dfdcd71f923a1d01d0cc31。32冻结来源均匹配，417旧2A+201早期关键证据/bestlast完整。merged view引用原NPZ/meta/hash、162v1+234v2、0复制，严格TRAIN，无probe/oldnative/validation混入。','',
    '验证：15 exactlaunch CPU+8 report检查通过，234纯几何预检通过；独立最终审计通过并149项三组冷报告交叉全match。check_repo仍只有3项已知旧publishedmedia JSON size/hash错误，没有扩展修复。8预定新样例/连续8帧+1PNG4GIF完成并目检。','',
    f"资源：本监督轮29312controls，历史native34560+新=63872；旧1899optimizer attempts/5352.516训练秒保持，监督轮0。回传前全部native持久产物{footprint['combined_bytes']}B={footprint['combined_bytes']/1024**3:.4f}GiB<3GiB，C盘可用{footprint['disk_free_bytes']/1024**3:.2f}GiB；无下载/安装/删除旧权重。",'',
    '实际命令（全部复用现有Python，SDL dummy/CUDA空/HF offline/MPL Agg）：',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b.py prepare --physics-root ../Tdyssey/third_party/diffusion_policy',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b.py freeze --physics-root ../Tdyssey/third_party/diffusion_policy',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b.py collect --physics-root ../Tdyssey/third_party/diffusion_policy --execute',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b.py audit --physics-root ../Tdyssey/third_party/diffusion_policy',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b_report.py --root artifacts/native-rgb-supervised-20261029 --requests artifacts/native-rgb-supervised-20261029/stage2b/frozen-requests.json --output artifacts/native-rgb-supervised-20261029/stage2b/report --physics-root ../Tdyssey/third_party/diffusion_policy --examples',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage2b_coverage.py --stage artifacts/native-rgb-supervised-20261029/stage2b',
    '精确分步命令/检查命令与日志见COMMANDS.md、focused-checks.log、collect.log、physics-audit.log、coverage-command.log和报告execution JSON。','',
    '绝对证据路径：']
for name in ['REPORT.md','stage2b-result.json','DIAGNOSIS.md','report/stage2b-report.json','report/merged-train-manifest.json',
    'physics-audit.json','review-postrun.json','report/coverage/COVERAGE_DIAGNOSIS.md','report/coverage/O-P-arc55-focus.md',
    'gap-quota-audit.md','report/examples/examples-1.png','source-identity.json','resource-ledger.json','COMMANDS.md']:
    body.append(str((b/name).resolve()))
body+=['','请审查M局部困难/55未命中/未知和缺gapcells/重复支持，并给下一阶段明确指令。当前仅完成2B，等待父审查。']
(b/'callback-stage2b.txt').write_text('\n'.join(body)+'\n',encoding='utf-8')
print(json.dumps(dict(status='ready_to_report',result_sha256=sha256(b/'stage2b-result.json'),merged_manifest_sha256=sha256(b/'report/merged-train-manifest.json'),combined_bytes=footprint['combined_bytes'],callback_chars=len('\n'.join(body)))))
