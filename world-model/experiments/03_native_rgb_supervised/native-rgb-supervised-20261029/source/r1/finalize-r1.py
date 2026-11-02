"""Write R1 CPU diagnosis/design evidence and callback; no runtime imports."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def put(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

stats=read(HERE/'motion-amplitude-statistics.json')
source=read(HERE/'motion-amplitude-source-evidence.json')
proposal=read(HERE/'r2-probe-requests.json')
budget=read(HERE/'r2-probe-budget.json')
review=read(HERE/'root-proposal-review.json')
resources=read(HERE/'storage-and-resource-audit.json')
ledger=read(BASE/'native-rgb-cost-ledger.json')
assert ledger==read(HERE/'prior-ledger.json') and ledger['active_stage'] is None
assert stats['H1']['n']==24156 and stats['H4']['n']==22968
assert stats['H1']['translation_image96_px']['at_least']['1']==0
assert resources['passed'] and resources['global_model_ledger_byte_identical']
assert budget['total_planned_cap']==1808 and all(r['execution_enabled'] is False for r in proposal['requests'])
assert review['only_proposed_parameter_difference']=={'space_damping':[0.,1.]}
now=datetime.now(timezone.utc).isoformat()
files=['REPORT.md','PARENT_REVIEW.md','motion-amplitude-audit.py','motion-amplitude-statistics.json',
       'motion-amplitude-diagnosis.md','motion-amplitude-source-evidence.json',
       'recording-alignment-audit.md','recording-alignment-audit.json','new-sampling-design.md',
       'r2-probe-requests.json','r2-probe-requests.csv','r2-probe-budget.json','root-proposal-review.json',
       'video-acceptance.md','video-acceptance.json','storage-and-resource-audit.json',
       'root-4a-recoverable-boundary.json','../stage4a/io-paused-state.json',
       '../stage4a/validation-suspended-boundary.json']
if (HERE/'r2-proposal-generation-error.json').exists():files.append('r2-proposal-generation-error.json')
result=dict(kind='native-rgb-R1-read-only-diagnosis-and-sampling-design-result.v1',stage='R1',created_at_utc=now,
    authorized_R1_scope_complete=True,outcome='subpixel_H1_support_gap_confirmed_force_damping_scale_explained_R2_causal_probe_proposed_not_executed',
    H1=stats['H1'],H4_endpoint=stats['H4'],H1_strata_counts=stats['H1_strata_counts'],
    H1_motion_decomposition=stats['H1_motion_decomposition'],microfit18=stats['microfit18_summary'],
    source_parameters=source['parameters'],causal_contribution_not_uniquely_verified=True,
    conditional5worldpx_scale_is_not_global_bound=True,existing_B_5point132211_worldpx_counterexample=True,
    action_only_1_2_3_imagepx_reachability='0 supporting H1 examples; no global impossibility proof; no safe guarantee',
    all70_initial_state_specs_unique=True,R2_proposal=budget,R2A_executed=False,R2B_executed=False,
    R2B_protocol_unselected=True,R2B_static_missing_goals=review['static_missing_families'],
    protocol_changes_applied=False,proposed_protocol_only_difference=review['only_proposed_parameter_difference'],
    future_order=['R1 parent review','explicit R2A bounded probe authorization','R2A actual amplitude/video/quality review',
                  'separate R2B approval','new sampling acceptance','new formal TRAIN/independent validation freeze','separate training authorization'],
    training_suspended=True,stage4A_incomplete_and_preserved=True,stage4B_execution_authorized=False,
    R1_new_physical_controls=0,R1_model_or_weights_loads=0,R1_native_inference_calls=0,R1_GPU_work=0,
    R1_optimizer_updates=0,R1_simulation_renders=0,R1_new_sampling_video_frames=0,R1_new_unit_tests=0,
    supervised_physical_controls=39680,all_native_physical_controls=74240,
    historical_native_optimizer_attempts=1899,historical_native_model_seconds=5352.516,
    new_round_optimizer_attempts=250,new_round_model_seconds=868.484000000004,
    global_model_ledger_closed_and_unchanged=True,old_unused320_not_spent=True,
    shape_rounding_not_guaranteed_to_resolve_with_more_training=True,
    no_new_modules_heads_or_learning_objective=True,no_new_unseen_access=True,no_server=True,no_packaging=True,
    storage=dict(bytes=resources['bytes'],atomic_reserved_bytes=268435456,projected_bytes=resources['projected_bytes'],
        remaining_after_atomic_bytes=resources['remaining_after_atomic_reserve_bytes'],
        proposed_R2_artifact_cap_bytes=134217728,projected_if_R2_artifact_cap_used=resources['projected_if_R2_artifact_cap_used'],
        combined_limit_bytes=3221225472,passed=True),
    evidence={name:dict(path=str((HERE/name).resolve()),sha256=sha(HERE/name)) for name in files},
    next_action='Actual parent callback, save receipt, waiting_parent and STOP. No R2 sampling or training.')
put(HERE/'r1-result.json',result)
s=read(BASE/'stage-state.json');assert s['current_stage']=='R1' and not s['callback']['sent']
s.update(status='ready_to_report',r1_complete=True,r1_result_file='r1/r1-result.json',
    r1_result_sha256=sha(HERE/'r1-result.json'),r1_outcome=result['outcome'],
    phase4a_status='suspended_by_direct_human_instruction',phase4a_complete=False,
    phase4b_execution_authorized=False,r2_execution_authorized=False,future_stages_authorized=False,
    next_action='Actual R1 parent callback thenwaiting_parent STOP; R2 and training remain unauthorized.')
s['resources'].update(r1_new_physical_controls=0,r1_model_loads=0,r1_native_inference_calls=0,r1_optimizer_updates=0,
    r1_GPU_work=0,r1_simulation_renders=0,r1_new_video_frames=0,r1_persistent_bytes_at_report=resources['bytes'])
s['callback'].update(message_file='r1/callback-r1.txt',result_file='r1/callback-result.json',sent=False)
put(BASE/'stage-state.json',s)
body=f'''[R1位移根因与新采样设计回传] 直接人类“先别训、先验实际幅度和视频”已落实：4A保存未完成边界并暂停，4B/R2均未授权。本次仅只读诊断+可审请求表，0新增物理/模型权重加载/原生推理/GPU/优化器/物体渲染/采样视频。完成本回传即waiting_parent并结束，不执行探针/训练。

确认幅度缺口：正式TRAIN396轨迹25344控制，H1有效24156，面积质心位移median .444615684/P95 .931297667/max .962289607图像px(96/512,H1=.1s)，>=1/2/3px全0。H4端点22968窗(.4s)median1.872150001/P953.708300826/max3.788435137。H1转角median .7648/P953.9281/max5.6372deg；>=3deg3572、>=5deg20。moving14692(60.82%)/nearstatic8447(34.97%)/I1017保持旧定义，不叫明显大平移。18微拟合窗全靠>=3deg旋转入选，平移.4817–.8041px、旋转3.0521–4.6706deg，无>=1px平移。70%重抽样不会补出缺失幅度。

分解/记录：H1旧门槛仅过平移3645、仅过旋转1841、两者9206，其余8447nearstatic/1017I；阈值only不等于另一分量零。contact16883的median .7438px，no-contact7273median0/max .0520；push13011median .8256/max .9623仍小。完整18×4层、actual contact/boundary/corner/signedgaps/phase分布都保存。540仅wallblocked_proxy，缺object-wall arbiter/impulse不宣称真正墙因果。54原冻结窗/50轨迹对齐核查无缩放/压缩/时间错位，bodyorigin/面积质心/物理CoG分开，65states/64actions/10substeps正确；target-distance用已存float32动作重算距precastplanning最大差1.36e-5worldpx（不是原actionXY逐坐标核查）。T面积质心local40.7143 vs物理CoG45，旧统计实际用area而非origin；cachedmask与几何差max .2532imagepx如实列。

源因证据：动态pusher/object mass1、kp100/kv20/Fmax10000/**pre-step agent cap200**、100Hz×10=.1s，继承space.damping0。匹配安装Pymunk6.9.0/Chipmunk7.0.3-dfc2fb8官方C源码的velocity更新乘damping**dt再加F/m*dt，D0逐子步抹旧速度，不是无阻尼；原kinematicpusher跳过此动态积分，新dynamicpusher继承该配置。源/版本/URL/具体行已给。自由力速度项<=100world/s，理想等质量近非弹性正面接触约50world/s→5world/.9375px是条件量级，不是全局硬界/唯一原因；B train-2b-163:15有5.132211world实测反例。未记录的中间速度/bias/冲量仍需探针分离。

raw target-distance median122.0958/P90224，H1子集median145.7221，分母不混。边界pushermax100/object物理CoGmax50.0098world/s；记录force/torque25740行全0是积分后清零，不是没施力。只读重构首substep力cap13208/25344=52.11%、contactH1 12775/16883≈75.67%、微拟合17/18；非全10子步真实力迹。Body.friction=1不等于已证Shape摩擦1，材料值未知需获批后日志。不能保证单纯拉远target、提高200cap或多训达到1/2/3px，也不声称已证明全局不可能。

方案：优先P0下真实持续接触/偏心/法向斜切向/合法重接触；最小机制P1**仅damping0→1**、其它F10000/pre200/mass/inertia/kp/kv/dt/碰撞规则与2px阈值全固定，独立新协议/目录、未实现未执行。force20000/40000、speed或horizon改变均不在本R2请求。P1能否安全达到2–4px/5–15deg待证，4–8挑战缺口不强凑；改变.4s任务不能跳帧伪装.1s。

新条件目标配额25%平移2–4/10%较大4–8挑战/20%旋转5–15/15%混合/10%小运动/10%无接触/10%真受阻或明确proxy负例，仅待审目标不强筛raw。未来18letter建议至少12独立families/4真实弧段，仍不是充分训练证明。实际8world弧段命中、独立轨迹、normal/oblique/tangent/forcearm/失联/滑动/再接触分列；同点多帧不增覆盖，孔/窄槽/未确认可达不假成功。按过程四墙四角signed/doublegaps与进入沿墙退出受阻，初始location不作覆盖。新增模型模块/几何loss均无。

具体待审R2：先16families×2protocol×2action×8=64slots/512新控，32对protocol输入state/actions逐值一致；完成gate必回父审。后仅父接受的**单一协议**做6代表T/L/O/C/X/B×9空间×24=54slots/1296控，B绝不自动选P1。总70不同计划initialstates/118计划slots/1808<=2000；余192不分配/replay0/旧320不支。4个B静态缺口(T右上/L右下/X右下力臂、X左上方向)保留计划分母且不派发，不换seed。JSON含源NPZ/metaSHA、initial origin/物理CoG/areaCOM/yaw、合法arc/staging/4或18或48gap、zero v/F/reset参数、gate8个absoluteactions、pairSHA；全部execution_enabled=false，只是proposal-spec hash非实际新协议身份。任一substep>2px/nonfinite/physicsIO异常停整个phase，失败/半步/重放计费，不补槽。512超过旧余320必须新R2预算明确授权；若全1808后来获批完成监督41488/历史76048，目前仍39680/74240。

视频门槛：每槽全部连续96px原图+明示6x显示放大，10Hz/1x真实时间，N控N+1帧、末帧停留单列，关联rawstate/action/SHA；逐步面积位移/转角/contact/arc/4signedwall+双墙gap/2px质量曲线；真实平移/旋转/混合/边界入沿出/受阻失败/负例全槽分母，不剪最佳、不插帧重绘改速。全部原始control与H1/H4可训练历史窗口分母另列。R1没有新视频，后续R2只拍真实physics无模型预测；未来model主图中文3列，原错误颜色GPC独立baseline页，不混成训练数据。

资源：globalmodelledger逐字节未变closed250/868.484000000004，旧native1899/5352.516另列；物理39680/74240不变，8未见仍封存。确认physics/geometry/modelmath8源未改，4A暂停源码保留。新旧native目前{resources['bytes']}B+256MiB原子预留={resources['projected_bytes']}B<3GiB，预留后余{resources['remaining_after_atomic_reserve_bytes']}B；再用拟议R2证据128MiB仍约{resources['projected_if_R2_artifact_cap_used']}B<3GiB。旧weights/data/failure不删，无安装、服务器、ZIP。CSV初写重复字段错误已另存并只从既有JSON修报告，不重采场景或改参数。

绝对证据目录：{HERE}
主文件REPORT.md、r1-result.json；motion-amplitude-statistics.json/diagnosis.md/source-evidence.json；recording-alignment-audit.md/json；new-sampling-design.md；r2-probe-requests.json/csv、r2-probe-budget.json、root-proposal-review.json；video-acceptance.md/json；storage-and-resource-audit.json；4A三份暂停boundary。全部具体源码行/确切engine primary links在证据内。

请审核已确认配置/实际支持缺口与未证假说，并决定是否单独放行R2A512及其仅D1候选、记录实现与视频规格；R2B/新正式TRAIN/验证/训练均继续待后续验收，不在本次执行。小样本轮廓改善不保证圆化能靠多训解决。回传成功后本实施停止。
'''
(HERE/'callback-r1.txt').write_text(body,encoding='utf-8')
s['callback']['message_sha256']=sha(HERE/'callback-r1.txt');put(BASE/'stage-state.json',s)
print(json.dumps(dict(ready_to_report=True,result_sha256=s['r1_result_sha256'],callback_characters=len(body),R2_executed=False,training_authorized=False)))
