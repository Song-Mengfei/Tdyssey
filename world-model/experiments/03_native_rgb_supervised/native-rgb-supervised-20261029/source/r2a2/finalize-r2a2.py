"""Cold R2A2 synthesis only. Run once before callback; no physical/model calls."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r2a2_run import verify_sources,storage_check
R=Path(__file__).resolve().parent
E=R.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
verify_sources()
j=read(R/'report/r2a2-report.json')
t=read(R/'telemetry-audit/statistics.json')
m=read(R/'media/manifest.json')
mc=read(R/'media/post-generation-audit.json')
integrity=read(R/'root-integrity-review.json')
ledger=read(R/'resource-ledger.json')
assert integrity['passed'] and t['passed'] and mc['all_passed']
assert j['qualified_four_arm_control_comparisons']==256 and not j['data_errors'] and not j['telemetry_contact_count_mismatches']
assert ledger['closed'] and ledger['charged_controls']==ledger['completed_controls']==512 and ledger['executed_substeps']==5120 and not ledger['failures']
arms=['P0-frozen','P1-damping1-candidate','D-half-life-0.1s','D-half-life-0.3s']
labels=['P0：D0','P1：D1','半衰期0.1s','半衰期0.3s']
def triple(summary,key):return '/'.join(f'{summary[key][k]:.4f}' for k in ['median','p95','max'])
lines=['# R2A2：有限阻尼对照完成，等待父验收','',
    '**两个有限阻尼候选均在无接触、无外力的记录中实现预定线速度和角速度衰减，同时保留部分明显接触推动/旋转。两者在幅度、接触持续性、自由运动和覆盖之间存在取舍；本阶段不选择最终协议，不宣布正式采样通过，训练继续暂停。**','',
    '[全部32组四臂关联视频](media/index.html) · [四臂幅度/分组/覆盖](report/DIAGNOSIS.md) · [完整统计与原始索引](report/r2a2-report.json) · [独立衰减审计](telemetry-audit/REPORT.md)','',
    '## 执行和配对','',
    '仅新增父批准的64槽/512控制/5120子步，全部完成，0物理失败/非有限/物理IO异常/partial。复用原16family、32场景/动作变体，**新增独立family=0**。每槽仅初始reset，未调整初态速度、动作、接触跟随、力限、摩擦、几何或时间。先T-center A0新两臂16控制gate，经独立冷态核对后一次执行剩余496；gate含在512内，没有预热、额外调试控制、重放、补槽或参数搜索。旧P0/P1只读复算，没有重新运行或重编码。','',
    '32组新两臂实际state15/初始RGB、非D物理参数和全部8个float64绝对动作，均逐值匹配旧P0/P1；匹配的是初态与控制目标序列，不是用中途reset强迫不同动力学的后续状态一致。[完整冷态检查](root-integrity-review.json)、[真实首gate](root-gate-review.json)。','',
    '|臂|space.damping（每秒系数）|每0.01s保留|每0.1s保留|协议SHA256|','|---|---:|---:|---:|---|']
for arm,label in zip(arms,labels):
    a=j['by_arm'][arm];d=a['requested_damping'];q=t['expected'][arm]
    lines.append(f"|{label}|{d!r}|{q['single_substep_retention']:.10f}|{q['full_control_retention']:.10f}|{','.join(a['actual_protocol_sha256s'])}|")
lines += ['',
    '仅调用原引擎Space.damping，线/角速度均由原更新衰减，没有自定义velocity回调或直接写物体速度。候选版本/目录/SHA独立，P0/P1原定义与旧TRAIN未改。实际所有agent/object/wall碰撞Shape摩擦/弹性仍0/0；mass、惯量/CoG、PD100/20、force10000、pusher步前cap200、10Hz控制/100Hz物理、walls/solver和2worldpx阈值均不变。D影响两个动态body，PD实际force与碰撞响应会随状态改变；不是保持施力序列相同的“只拖慢物体”实验。候选不是已标定的真实桌面材料。','',
    '## 自由衰减是否真实发生','',
    '纯自由物体仅选每个相关子步均无AO/物体墙接触、物体force/torque=0且gravity=0的连续片段；任一碰撞/外力/缺测即断段，未跨碰撞套指数公式。带PD力但无agent接触时检验完整更新式D**dt×v_entry+F×dt/m，不能叫纯自由半衰期。','',
    '|候选|完整无接触10子步控制数|非零入口线/角样本数|线速度保留率范围|角速度保留率范围|','|---|---:|---|---|---|']
for arm,label in zip(arms[2:],labels[2:]):
    v=t['pure_object_retention'][arm]['complete_controls'];lin=v['linear_retention'];ang=v['angular_signed_retention']
    lines.append(f"|{label}|{v['records']}|{lin['finite']}/{ang['finite']}|{lin['min']:.15f}–{lin['max']:.15f}|{ang['min']:.15f}–{ang['max']:.15f}|")
lines += ['',
    '有限候选符合指定0.5/约0.793700526的单控制保留率，观测推回半衰期约0.1/0.3秒，浮点误差范围内。P0的零入口控制、各候选exact-zero/near-zero入口单列，不做0/0比率；ratio字段的undefined不是轨迹测量丢失。P1自由速度保持、P0清除速度记忆也只用旧日志核验。无相关body接触的线/角完整更新残差检查全部通过。**位移不是速度**：面积质心轨迹还涉及旋转和原引擎位置/bias更新，未把图像位移比率当半衰期证明。[逐子步/完整控制/脱离后分段及分类](telemetry-audit/statistics.json)。','',
    '## raw、真正H1和H4分别比较','',
    '四臂各256raw/160H1/64H4全部有效；新增两臂只有512raw/320H1/128H4，另512raw为旧数据离线引用，不重复计新控制。8控制/9帧仅有5个四帧历史H1、2个H4。面积质心=origin+R×面积局部质心，比例96/512；T面积质心与物理CoG分开。H1=.1s，H4=.4s端点，窗口重叠不冒充独立场景。','',
    '|总体/时长|臂|N|平移中位/P95/max（图像px）|绝对角中位/P95/max（度）|','|---|---|---:|---|---|']
for pop,label in [('all_controls','raw .1s'),('h1_windows','H1 .1s'),('h4_windows','H4 .4s')]:
    for arm,name in zip(arms,labels):
        s=j['by_arm'][arm]['populations'][pop]
        lines.append(f"|{label}|{name}|{s['observed_complete']}|{triple(s,'translation_image96_px')}|{triple(s,'rotation_deg')}|")
lines += ['',
    'H1的>=2px/4px/5°数量：P0为0/0/0，P1为77/31/57，0.1s为33/3/27，0.3s为53/6/35（各160分母）；全部H1无>=8px或>=15°。降低滑行速度并不保证每一类明显运动仍有充分支持。','',
    '## 接触时长、冲量、自由运动：都保留，不以一个比例过关','',
    '每控制AO命中子步0..10完整直方图如下。>=8/10是父在本阶段预先规定的较持续接触**描述分组**，不等于8个连续子步，不等于主动推动成功，也不单独决定验收。它对旧P0/P1是本阶段统一复算，不倒写成R2A旧验收指标。','',
    '|臂|AO命中子步0,1,2,3,4,5,6,7,8,9,10（raw256）|','|---|---|']
for arm,name in zip(arms,labels):
    h=j['by_arm'][arm]['ao_contact_substeps_histogram']
    assert h['unknown_observed']==0 and sum(h['counts_0_to_10'].values())==256
    lines.append(f"|{name}|"+','.join(str(h['counts_0_to_10'][str(i)]) for i in range(11))+'|')
lines += ['',
    '下表H1按实际观测分组。不同组可能重叠；N是条件组自身分母，不能拿筛后均值代替raw总体。非零冲量只计post_solve、避免pre/post双计；任一次AO接触或非零冲量均不能独自证明整步运动由当前主动推力发起。','',
    '|臂|H1组|N|平移中位/P95/max|>=2px|>=5°|','|---|---|---:|---|---:|---:|']
groups=[('all','全体'),('any_agent_object','任一AO'),('ao_at_least_80pct_future_substeps','>=8/10'),('nonzero_ao_impulse','非零AO冲量'),('full_no_agent_object_no_object_wall','全步无AO/物体墙')]
for arm,name in zip(arms,labels):
    for group,label in groups:
        s=j['by_arm'][arm]['conditional_groups']['h1_windows'][group]
        lines.append(f"|{name}|{label}|{s['observed_complete']}|{triple(s,'translation_image96_px')}|{s['translation_image96_px']['counts_gte']['2']}|{s['rotation_deg']['counts_gte']['5']}|")
lines += ['',
    '0.1s的33个H1>=2px均有AO，但其中只有26个有非零AO冲量；0.3s的53个中43个有AO、35个有非零冲量。较持续组中>=2px为0.1s23/48、0.3s10/21；>=5°为15/48、7/21。严格无AO/无物体墙H1分别63和70个，0.1s无>=2px或>=5°，0.3s仍有7/10个。这些都是有效自由动力学响应，完整保留，不改标签/判坏或删除，也不混称持续接触推动。','',
    'H4辅助累计>=32/40与“每个未来控制均>=8/10”分别：P0 5/1、P1 1/1、0.1s2/0、0.3s1/1（各64分母）。32/40不是父原话预注册条件，不保证四步各>=8或连续32子步；每步各>=8也不证明连续。固定8步脚本包含第6号撤回动作，两个H4合法anchor3/4都跨撤回，因此这些计数亦受原动作时序影响，不能单独作为协议性能否决/通过标准。本阶段没有改动作。','',
    '## 联合目标、字母/弧段/力臂和边界缺口','',
    '|新候选H1组|N|2–4px且角<1°|4–8px且角<1°|5–15°且平移<1px|混合>=2px且>=5°|','|---|---:|---:|---:|---:|---:|']
keys=['translation_2_to_4_angle_lt1','translation_4_to_8_angle_lt1','rotation_5_to_15_translation_lt1','translation_ge2_rotation_ge5']
for arm,name in zip(arms[2:],labels[2:]):
    for group,label in [groups[0],groups[2],groups[3]]:
        s=j['by_arm'][arm]['conditional_groups']['h1_windows'][group]
        lines.append(f"|{name}/{label}|{s['observed_complete']}|"+'|'.join(str(s['joint_targets'].get(k,0)) for k in keys)+'|')
lines += ['',
    '两新候选C/X的H1>=2px均0，B在0.1s为0、0.3s仅1；4–8px只T/O；T在所有四臂H1均无>=5°。不能用全局均值/比例掩盖这些缺层。16family每字母仅2–4，实际每arc独立family最多1–2，技术重复/100Hz点数不增加独立性。每目标的family/实际arc/有符号力臂及双墙gap逐项支持见四臂JSON的by_arm.actual_contact_boundary_coverage；C实际法向力臂仍只有positive。','',
    '接触弧段来自实际点，不用plannedarc；签名力臂为实际点相对物理CoG的cross(r,单位向内normal)，有数值近零阈值声明，不把点样本当独立事件。四墙四角初始槽均执行，但实际沿边/离边/碰撞/静止组合按signed gap/真实arbiter分组。C两候选无along/leaving，X的0.1s无leaving、0.3s有4raw控；初始location不等于实际过程覆盖。墙接触静止与仅几何proxy分开，未作单独无墙反事实墙因果结论。[详细实际诊断](report/DIAGNOSIS.md)。','',
    '## 速度、冲量、穿透与有限时长范围','',
    '|臂|force裁剪子步|pusher步前speed裁剪|AO段/丢失/重接触|子步后pusher峰速|物理CoG峰速|最大穿透|','|---|---:|---:|---|---:|---:|---:|']
for arm,name in zip(arms,labels):
    c=t['counts'][arm];v=t['force_cap_velocity_statistics'][arm]
    lines.append(f"|{name}|{c['force_clipped']}|{c['speed_clipped']}|{c['AO_contact_episodes']}/{c['AO_lost_contact_transitions']}/{c['AO_recontacts_after_observed_episode']}|{v['agent_speed_post']['max']:.4f}|{v['object_COM_speed_post']['max']:.4f}|{v['max_penetration']['max']:.6f}|")
lines += ['',
    '速度单位worldpx/s，穿透worldpx。200仍仅是pusher步前cap，不是物体/子步后全局界。接触段/重接触为实际100Hz事件序列，事件碎片增加不能冒称有目的恢复成功；每段非独立family。新两臂整体max子步穿透[object-wall,object-pusher,pusher-wall]=[1.787012,1.315380,.425866]，均<=原2阈值。只支持已运行每槽.8s范围，不外推24步稳定性。','',
    '## 原始视频与可复核性','',
    '新576个原96RGB/PNG、64个FFV1无损视频、32对新候选H264审阅视频全部保留。每槽8控制/9帧、10fps/1×、物理.8s；编码.9s中的末帧.1s显示停留单列。64个新原视频逐像素解码=记录RGB/帧SHA；32新pair帧数/fps齐。6×nearest只显示放大；点坐标、实际arc、gap/穿透/动作/每.1s位移角度和曲线在原96区域外，负gap未隐藏。首T/首corner X state4已视觉核查。真实partial/失败0，未冒称实际测试过失败兼容性。','',
    'HTML按family/action引用旧P0/P1和新两候选视频，可按相同帧时刻比较；旧99个输入/source/96视频前后SHA未变，0旧重编码。32组同步markup/路径/帧对齐已核，浏览器按钮同步运行未实际测试，不将静态检查冒称runtime。原PNG/MKV是无损依据，H264仅显示编码，无插帧/改速/剪静止/修物体/拼轨/额外物理渲染。[全部媒体](media/index.html)、[manifest](media/manifest.json)、[编码/链接/旧哈希核查](media/post-generation-audit.json)。','',
    '## 离线修复、资源和停止边界','',
    '新reporter首次严格family全字段比较遇到唯一phase标签R2A→R2A2差异；原错误完整保留。独立offline-repair副本只验证该精确阶段标签变化后逐值比较其余字段，ROOT按副本路径修正；原冻结source/请求/度量均未改，修复副本一次离线运行成功，0物理重跑。[错误](report/offline-error-first-run.json)、[精确修复记录](offline-repair/repair-record.json)。','',
    '既有venv使用原Pymunk6.9.0/Chipmunk7.0.3-dfc2fb8；10纯mock/source预检未运行真实环境/控制，报告7合成例亦非物理验证。实际控制只有以下两个一次性入口，预算在每控前持久预扣，失败/partial不退；独占claim防并发/重启重放。','',
    '```powershell',
    "& '../Tdyssey/.venv/Scripts/python.exe' -X utf8 -B scripts/native_rgb_r2a2_run.py prepare",
    "& '../Tdyssey/.venv/Scripts/python.exe' -X utf8 -B scripts/native_rgb_r2a2_run.py gate",
    '# root实际gate核查后仅一次：',
    "& '../Tdyssey/.venv/Scripts/python.exe' -X utf8 -B scripts/native_rgb_r2a2_run.py remaining",
    '```','',
    '新512/512账本已关闭，链接40192/74752→40704监督控制/75264历史总控。旧320/R2B/未分配额度均未动，原模型ledger逐字节未变closed250尝试/868.484000000004秒，旧native1899/5352.516单列不改。**本阶段0模型加载/原生推理/GPU/优化器/训练/服务器/安装/下载/上传包**；8保留字母封存，旧数据/权重/失败和父审产物全部保留，未改模型架构/学习目标。几何状态仅采样/审计使用，没有新模型输出，所以世界模型运动预测/刚性保形问题仍未验证。','',
    '新候选额外产物<=128MiB，新旧native/源/cache/既有外部资产/父审文件+256MiB原子预留<=3GiB，执行前及逐槽检查；[最终资源metadata快照](storage-and-resource-audit.json)另预留1MiB收尾report/callback，不是全电脑存储统计。','',
    '人类持续授权阶段报告/实验摘要/本机路径主动回传同父对话，见[持续授权](../standing-callback-authorization.json)。R1早先拒绝和后续成功历史、R2A成功receipt均保留。R2A2 callback待实际执行；成功保存receipt、waiting_parent并停止。未经父再次明确放行，不选胜出协议、不做接触跟随/补R2B缺口、不跑R2B/正式重采样/训练。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
storage=storage_check('R2A2_final_report_before_callback',1024**2)
storage.update(kind='R2A2-final-resource-and-storage-audit.v1',qualification='Metadata snapshot before this small audit/result/callback; 1MiB explicit headroom reserved in both limits.',
    model_ledger_byte_identical=(E/'native-rgb-cost-ledger.json').read_bytes()==(R/'prior-model-ledger.json').read_bytes(),
    model_ledger_sha256=sha(E/'native-rgb-cost-ledger.json'),original_source_snapshots_unchanged=True,
    new_charged_controls=512,new_completed_controls=512,new_substeps=5120,
    cumulative_supervised_controls=40704,cumulative_all_native_controls=75264,
    old_baseline_controls_reexecuted=0,old_videos_reencoded=0,new_independent_families=0,
    new_model_calls=0,new_GPU_work=0,new_optimizer_attempts=0,old_evidence_deleted=False,
    R2B_executed=False,formal_collection_executed=False,training_executed=False)
assert storage['model_ledger_byte_identical']
write(R/'storage-and-resource-audit.json',storage)
names=['REPORT.md','PARENT_REVIEW.md','frozen-config.json','frozen-requests.json','source-identity.json',
    'runtime-identity.json','preflight-checks.json','resource-ledger.json','index.json','root-gate-review.json',
    'root-integrity-review.json','static-inheritance-review.json','frozen-baseline-references.json',
    'storage-and-resource-audit.json','report/r2a2-report.json','report/DIAGNOSIS.md','report/actual-summary.json',
    'report/offline-error-first-run.json','offline-repair/repair-record.json',
    'offline-repair/native_rgb_r2a2_report_v1a.py','telemetry-audit/statistics.json',
    'telemetry-audit/REPORT.md','telemetry-audit/helper.py','media/index.html','media/manifest.json',
    'media/post-generation-audit.json','gate-command-result.json','remaining-command-result.json','finalize-r2a2.py']
result=dict(kind='native-rgb-R2A2-finite-damping-result.v1',stage='R2A2',created_at_utc=datetime.now(timezone.utc).isoformat(),
    authorized_stage_evidence_complete=True,physical_collection_complete=True,callback_sent=False,
    outcome='native_finite_linear_and_angular_decay_verified_with_retained_contact_motion_and_remaining_support_tradeoffs_no_protocol_selected',
    new_slots=64,new_charged_controls=512,new_completed_controls=512,new_substeps=5120,
    failures=0,raw_frames=576,actual_four_arm_initial_actions_nonD_match_pairs=32,
    reused_independent_families=16,new_independent_families=0,
    new_raw=512,new_H1=320,new_H4=128,old_controls_offline_reused=512,old_controls_reexecuted=0,
    old_videos_reencoded=0,new_lossless_videos=64,new_pair_videos=32,
    four_arm_population_summaries={arm:j['by_arm'][arm]['populations'] for arm in arms},
    four_arm_H1_groups={arm:j['by_arm'][arm]['conditional_groups']['h1_windows'] for arm in arms},
    contact_descriptors='H1>=8/10 predeclared for this stage; H4>=32/40 auxiliary aggregate and everyfuturecontrol>=8 separate; none a sole pass/continuous-contact proof or retroactive old criterion',
    pure_decay_verified=True,ratio_zero_entry_is_undefined_not_missing_trajectory=True,
    baseline_and_original_source_unchanged=True,model_ledger_byte_identical=True,
    new_model_loads=0,new_native_inference=0,new_GPU_work=0,new_optimizer_attempts=0,
    new_unseen_access=0,new_server_jobs=0,new_installations=0,new_upload_packages=0,
    cumulative_supervised_controls=40704,cumulative_all_native_controls=75264,
    final_protocol_selected=None,formal_sampling_accepted=False,WM_prediction_or_shape_improvement_claimed=False,
    R2B_executed=False,formal_collection_executed=False,training_executed=False,
    old_unused320_spent=False,old_evidence_deleted=False,
    offline_phase_label_compatibility_repair_preserved=True,
    max_substep_penetrations=integrity['max_substep_penetrations'],storage_snapshot=storage,
    evidence={name:dict(path=str((R/name).resolve()),sha256=sha(R/name)) for name in names},
    next_action='Actual standing-authorized parent callback, preserve receipt, waiting_parent STOP. No winner/protocol/physics/R2B/formal/model/training advance.')
write(R/'r2a2-result.json',result)
s=read(E/'stage-state.json');assert s['current_stage']=='R2A2' and s['r2a2_collection_closed']
s.update(status='reporting_parent',r2a2_complete=True,r2a2_result_file='r2a2/r2a2-result.json',
    r2a2_result_sha256=sha(R/'r2a2-result.json'),r2a2_execution_authorized=False,
    parent_review_received=False,next_action=result['next_action'])
write(E/'stage-state.json',s)
print(json.dumps(dict(stage='R2A2',evidence_complete=True,new_controls=512,new_families=0,
    result_sha256=sha(R/'r2a2-result.json'),storage_bytes=storage['combined_bytes'],new_candidate_bytes=storage['r2a2_artifact_bytes']),ensure_ascii=False))
