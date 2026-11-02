"""Assemble closed R2B evidence only; no physical/model execution or messaging."""
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r2b_run import OUT,EXPERIMENT,read,sha,utc,atomic_json,verify_sources,storage_check

verify_sources()
d=read(OUT/'report/r2b-report.json'); integrity=read(OUT/'root-integrity-review.json')
media=read(OUT/'media/manifest.json'); ledger=read(OUT/'resource-ledger.json')
policyreview=read(OUT/'independent-policy-review.json'); mediaquality=read(OUT/'media/root-media-quality.json')
assert policyreview['passed'] and not policyreview['anomalies'] and mediaquality['all_passed']
assert not d['data_errors'] and integrity['passed'] and ledger['closed']
assert media['planned_slots']==54 and media['saved_slots']==53 and media['complete_controls']==1272
assert media['index_sha256']==sha(OUT/'index.json') and d['index_sha256']==sha(OUT/'index.json')
assert sha(EXPERIMENT/'native-rgb-cost-ledger.json')=='2729b06c8e565618624c65e1835701f2a37eabbcc9da0722c5bccad8bd7c40e6'
model=read(EXPERIMENT/'native-rgb-cost-ledger.json')
requests=read(OUT/'frozen-requests.json')['requests']
gaps=[{k:r[k] for k in ['request_id','requested_gap_world_px','selected_gap_world_px']} for r in requests if r.get('gap_request_met') is False]
labels={'all_controls':'原始相邻控制','h1_windows':'完整历史 H1','h4_windows':'完整历史 H4','push_h4_windows':'未来4控全在计划push的H4','push_full_history_h4_windows':'历史3控+未来4控全在计划push的H4'}
names=list(labels)
def fmt(x):return '不可评估' if x is None else f'{x:.6g}'
def table(rows):return '\n'.join('|'+ '|'.join(map(str,r))+'|' for r in rows)
rows=[['统计集合','计划','实际/有效','平移中位/P95/max (96图像px)','≥2px','旋转≥5°'],['---','---:','---:','---','---:','---:']]
for name in names:
 s=d['summaries'][name];a=s['translation_image96_px']; rows.append([labels[name],s['planned'],f"{s['observed_complete']}/{s['valid_windows']}",'/'.join(fmt(a[k]) for k in ['median','p95','max']),a['counts_gte']['2'],s['rotation_deg']['counts_gte']['5']])
focus=[['字母','H1分母','平移≥2px','其中2–4px且角<1°','正旋转≥5°','负旋转≤−5°'],['---','---:','---:','---:','---:','---:']]
for letter in ['T','L','O','C','X','B']:
 f=d['focus_support'][letter]; j=d['by_letter'][letter]['h1_windows']['valid_joint_targets']
 focus.append([letter,f['valid_h1'],f['translation_ge2'],j.get('translation_2_to_4_angle_lt1',0),f['positive_rotation_ge5'],f['negative_rotation_le_minus5']])
jointkeys=['translation_2_to_4','translation_4_to_8','rotation_5_to_15','translation_2_to_4_angle_lt1','translation_4_to_8_angle_lt1','rotation_5_to_15_translation_lt1','translation_ge2_rotation_ge5']
joint=[['联合门槛']+[labels[k] for k in names],['---']+['---:']*len(names)]
for key in jointkeys:joint.append([key]+[d['summaries'][k]['valid_joint_targets'].get(key,0) for k in names])
coast=d['groups']['h1_windows']['actual_contact_groups']['no_AO_no_OW_coast']
release=d['release_summary'];hist=integrity['agent_object_contact_substeps_histogram']
body=f'''# R2B 已选0.1秒半衰期：接触跟随重采样结果

本阶段完成53条新轨迹、1272次完整控制（12720物理子步），没有物理质量失败。54计划槽中的X左上静态不可派发，保留缺口且未替换。新数据已出现C/X/B平移及T正反旋转；总体H1仍有小运动、H1≥3图像px为0，不能据此宣称正式训练集或模型通过。物理采集已关闭，完成实际主动回传后等待父审。

## 固定配置与执行边界

用户已直接选定“0.1秒半衰期（已选）”，D精确0.0009765625，协议SHA256 `58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6`。本阶段没有开展新的D比较、阻尼验证或搜索，之后所有新TRAIN/验证/测试生成默认此配置。历史candidate技术名/哈希保留，旧D0/D1及历史候选数据没有改标签。唯一新生成默认入口见 `selected-default-routing.md`、`configs/native_rgb_default.json`及selected factory；历史直接make_env CLI不作为新生成默认路线，8保留字母仍封存。

54冻结新请求=6个既有TRAIN代表×9空间，每槽24控制封顶。首6固定T正旋转/C平移/X平移/B平移/L负旋转/O边界共144控制，经时序、记录、规则、质量审查后继续其余冻结槽；不以运动阈值筛选或换seed。53可派发全完成，X-left-top混合正方向静态受阻24控制未花。授权上限1296不是实际花费；未补24、未动旧余320。只有初始reset，0中途reset/物体状态写入/直接物体施力扭矩。PD100/20、force10000、pusher步前200、物体材料/质量/惯量/形状、10Hz控制/100Hz物理及walls/solver全部保持选定协议。

## 幅度与完整窗口

平移为实际面积质心的96图像像素尺度；旋转为原始未缠绕角差绝对值。raw相邻控制没有历史要求，24控制仅给21个完整历史H1、18个完整历史H4。窗口可以重叠，计数不是独立样本数。完整分位数、逐family/真实arc/真实有符号力臂/方向/边界gap/接触与滑行分层在 `report/r2b-report.json` 和对应CSV。

{table(rows)}

push未来H4使用anchor4..10，每槽7个（计划378/实际371），仅保证未来4控在计划push，历史可能含approach。全历史push H4使用anchor7..10，每槽4个（计划216/实际212），3过去+4未来控全在计划push。二者均不自动代表持续实际接触：实际AO/冲量和真实执行分支另分层。所有集合的缺数正好来自唯一静态不可派发槽。

{table(focus)}

C/X/B的25/33/31个H1≥2px确为实测，其中角<1°的2–4px支持为15/18/14。T正/负≥5°为30/15。6个字母均观测到正负法向力臂；这不保证每个预期触点/过程或两个平移弧段都达标，也不把计划标签当实际覆盖。逐触点、family与边界过程证据须父审。

全部174个H1≥2px均有实际AO、非零AO冲量及≥8/10子步接触。C/X/B这些平移窗分别来自4/4/5个独立family，T正/负旋转分别来自4/2个family；重叠窗口仍不能当作同数独立轨迹。C负力臂H1有34窗/4family，正力臂32窗/6family。完整窗/真实触点来源详见offline-repair/repair-record.json。

{table(joint)}

H1最大平移2.42424px，≥3px与4–8px组仍为空，≥2px且≥5°仅2窗。全历史push H4虽有更大位移，但角<1°的2–4px及4–8px联合目标均0。总体平均不能代替这些缺口，当前没有正式采样/训练放行或模型保形结论。

## 实际接触、边界与释放

1272控制的AO命中子步0..10直方图依次为 `{hist}`；574控制有AO、698无AO，>=8/10为488。101个实际接触episode中位0.28s、最长1.27s，100次失联转移和48次观测重接触全部保留。H1中完全无AO与物体墙接触的滑行窗455；其全部原始数据及分布保留，滑行是有效动力学。接触点/真实弧段、pre_solve物理CoG法向有符号力臂、post_solve冲量以及时间与四墙signed gap保留在100Hz遥测，pre/post事件没有重复算冲量。

独立审查进一步显示触点跟随仍有缺口：574个AO控制中246个包含各自计划arc，328个只命中其它arc；逐字母“包含计划arc/只有其它arc”为T26/81、C33/40、X28/49、B42/55、L44/68、O73/35。601次contact-intent、601次free-route、70次hold中，85次接触意图未形成AO。固定局部目标/法线的计算一致，不等于实际持续命中目标arc；这些偏移/失联原样保留，不能称所有触点目标已覆盖。无AO/OW的455个H1中≥2px及≥5°均0。

H1边界分类（几何趋势，不冒充墙接触成功）：`{json.dumps(d['summaries']['h1_windows']['boundary_process_counts'],ensure_ascii=False)}`。初始4/18/48worldpx分层实际几何明确有3项偏差：L右上4→48、X右上4→18、B右上18→48，gap_request_met=false保留。旧R1四个静态问题中T右上改为新translation_2，L右下/X右下改新wall_along后可派发；这是新角色的几何可行性，不能称旧lever/direction或旧动态目标已覆盖。X左上仍未解决、未派发。

释放观察固定state14速度为10%/5%参考，不重设参考到脱离速度。53条都有14→20撤回阶段额外运动：平移中位{fmt(release['extra_translation_image96_px']['median'])}px、P95{fmt(release['extra_translation_image96_px']['p95'])}、max{fmt(release['extra_translation_image96_px']['max'])}；额外角位移中位{fmt(release['extra_absolute_rotation_deg']['median'])}°、max{fmt(release['extra_absolute_rotation_deg']['max'])}°。该总量含期间接触/碰撞，不能全称自由滑行。

首自由段实测线速度10%/5%首次下穿各40条，中位时间0.34s/0.44s；角速度相同各40条。其余11条无自由段、1条被物体墙碰撞删失、1条脱离时已低于阈值但下穿未观测；另1计划静态槽缺测。所有自由段、真正脱离区间、阈值下穿0.01s区间和删失原因逐条保留，未用半衰期公式填时间。这是本批动作行为展示。

## 冻结规则与审查

0..3 approach，4..13 push，14..19 withdraw，20..23 bounded samearc recontact。每0.1s由当前位姿更新固定局部触点/法线；法向内推进48worldpx，approach/recontact24，平移切向比例绝对≤0.5、mixed±0.35；相对当前推杆目标≤64、相邻目标变化≤96worldpx。失联2控后最多4控同arc恢复、无法几何证明则hold+miss，恢复占24控不延期。平移切向是所需目标方向的力矩消除意图；在Shape摩擦0的实际法向碰撞中不保证物理扭矩抵消。

root已逐条纯StateView重放1272个决策，动作/阶段/当前state SHA/世界触点法线与冻结规则一致，无未来动力学前瞻；完整12720子步状态链、动作绑定、1325原帧哈希、全部选定非状态物理参数身份通过。最大子步穿透（AO/OW/AW）`{integrity['max_penetrations_world_px']}`worldpx，均≤2；nonfinite/IO/未知推进/失败0。本次冷态核验没有环境构造或物理推进，任何小运动/无接触/墙受阻均未剔除。

另一独立审查完成46936项代数/source/log/hash检查0异常，实际接触local→world最大误差1.14e-13，文件independent-policy-review.json/.md。它与root有状态规则重放互补，没有新增物理或模型调用。

冻结请求SHA `eb4521bfcb321628c33831f1bebb7f6e2d839c3cead660fe9e26ebacbd1bcd31`；规则canonical SHA `8b62864db8fc018ef0c546805aae44bf6034b1ce2334bcd92b8f2f060de57fb3`；规则+source bundle SHA `4ba928ca5cb21ab9259acc8b42e482f6a1236ecef3dd9408f7447b0d0929864d`。原源码/快照/运行时/父审/旧ledger全部不变。报告器曾在14..19阶段名称把withdraw误写为withdraw_decay，318个合法日志标签因此拒绝；原失败输出保留。仅独立 `offline-repair/native_rgb_r2b_report_v1_1.py` 修正标签及新位置ROOT定位，AST/diff见repair-record，冻结原报告器未改，0补采，最终data_errors=0。

## 全量媒体、费用与后续

`media/index.html`保留54槽含静态缺口；全部53条都有25PNG原始96帧、FFV1无损96连续视频及中文单配置960×1120审阅片。10fps，24控制=2.4s物理跨度，25帧=2.5s显示，末帧0.1s停留单列；6×nearest仅显示放大，没有重绘物理或修图。没有重编码旧D对照视频。媒体manifest绑定实际动态动作与decision。

媒体脚本FFV1全解码之外，独立OpenCV/FFMPEG再次核对53个原视频1325帧与NPZ RGB逐像素完全一致；全部H264审阅片25帧/10fps。已视检T推送、C左上撤回、X右上重接触图，中文标注、实际AO与意图分列、负gap/曲线均可读。所有54槽均在索引，没有只交精选动作；media/root-media-quality.json保存完整QA。

本阶段新增模型加载/推理/optimizer/GPU/训练/未见数据/服务器/安装下载/上传包均0。全局closed模型ledger SHA `2729b06c8e565618624c65e1835701f2a37eabbcc9da0722c5bccad8bd7c40e6`逐字节不变，250attempt/868.484000000004秒；历史另一native实验1899attempt/5352.516秒单列不冲销。新物理1272charged=1272completed；累计监督41976、历史native76536。授权1296剩余24不自动再用。

存储逐槽及最终同时核：R2B≤128MiB，共享旧/新native、source、cache、父审全部+256MiB原子预留+1MiB收尾预算≤3GiB；确数见storage-final-evidence.json。没有删除旧数据/模型/失败记录。22项前置纯CPU测试通过；后续仅因离线标签修复执行相关纯测试/AST和数据复算。

当前只交父审实际幅度、接触/边界缺层及全部视频。0.1s已由用户选定，无需再次确认；正式18字母采集、模型训练和后续阶段仍未授权。回传完成后保存真实receipt并waiting_parent停止，不轮询父或自动开下一阶段。
'''
(OUT/'REPORT.md').write_text(body,encoding='utf-8')
evidence_names=['REPORT.md','report/r2b-report.json','report/REPORT.md','index.json','resource-ledger.json','root-integrity-review.json','independent-policy-review.json','independent-policy-review.md','source-identity.json','frozen-config.json','frozen-requests.json','frozen-rules.json','media/manifest.json','media/index.html','media/root-media-quality.json','offline-repair/repair-record.json','preflight-checks.json','root-first-six-review.json']
result=dict(kind='R2B-closed-feedback-resampling-result.v1',at_utc=utc(),outcome='completed53_of54_one_static_blocked_pending_parent_acceptance',
    selected_physics=d['selected_physics'],collection_closed=True,execution_authorized=False,
    planned_slots=54,completed_slots=53,static_blocked_slots=1,charged_controls=1272,completed_controls=1272,
    recorded_substeps=12720,raw_frames=1325,all_controls_cumulative=dict(supervised=41976,historical_native=76536),
    summaries=d['summaries'],focus_support=d['focus_support'],release_summary=release,actual_contact_summary=d['actual_contact_summary'],
    static_gap_changes=gaps,media_manifest_sha256=sha(OUT/'media/manifest.json'),
    independent_policy_review_sha256=sha(OUT/'independent-policy-review.json'),media_quality_sha256=sha(OUT/'media/root-media-quality.json'),
    new_model_loads=0,new_model_inference_calls=0,new_optimizer_attempts=0,new_GPU_work=0,new_training_seconds=0,
    prior_closed_model_ledger_sha256=sha(EXPERIMENT/'native-rgb-cost-ledger.json'),
    prior_model_charged_attempts=model['charged_updates'],prior_model_charged_seconds=model['charged_seconds'],
    callback_sent=False,parent_review_received=False,formal_sampling_accepted=False,training_authorized=False,
    evidence=[dict(file=n,absolute_path=str((OUT/n).resolve()),sha256=sha(OUT/n)) for n in evidence_names if (OUT/n).exists()])
atomic_json(OUT/'r2b-result.json',result)
atomic_json(OUT/'storage-final-evidence.json',storage_check('final-evidence-before-callback',1024**2))
state=read(EXPERIMENT/'stage-state.json');state.update(status='R2B_evidence_ready_for_callback',r2b_complete=True,
    r2b_execution_authorized=False,r2b_physics_authorization_spent=True,
    r2b_result_file='r2b/r2b-result.json',r2b_result_sha256=sha(OUT/'r2b-result.json'),
    parent_review_received=False,next_action='Send authorized R2B evidence to parent, save actual receipt, waiting_parent and STOP')
atomic_json(EXPERIMENT/'stage-state.json',state)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),result_sha256=sha(OUT/'r2b-result.json'),data_errors=len(d['data_errors']),storage=read(OUT/'storage-final-evidence.json'))))
