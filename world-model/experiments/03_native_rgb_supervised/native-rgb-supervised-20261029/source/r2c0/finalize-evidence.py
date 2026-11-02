"""Freeze R2C0 CPU-only deliverables; no simulator, model or messaging calls."""
import sys,json,shutil,ast
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r2b_run import EXPERIMENT,read,sha,atomic_json,utc,storage_check,verify_sources
OUT=EXPERIMENT/'r2c0';D=OUT/'data';E=OUT/'evaluation'
verify_sources()
for row in read(OUT/'input-identities.json')['files']:assert sha(row['path'])==row['sha256'],row['path']
assert read(OUT/'root-data-interface-audit.json')['passed'] and read(OUT/'root-joint-interface-audit.json')['passed']
manifest=read(D/'manifest.json');split=read(D/'split-plan.json');audit=read(D/'draw-audit.json');indices=read(E/'indices.json');evalaudit=read(E/'preparation-audit.json');proposal=read(ROOT/'configs/native_rgb_r2c0_proposed.json')
assert proposal['is_proposal'] and proposal['execution_authorized'] is False and proposal['execute'] is False and proposal['actual_model_entry_ready'] is False
assert {k:len(v) for k,v in indices['cohorts'].items()}==dict(train_visual=12,val_motion=12,val_natural=12)
assert not indices['missing_support'] and evalaudit['model_calls']==evalaudit['environment_constructions']==0
assert sha(E/'indices.json')==evalaudit['indices_sha256'] and sha(E/'protocol.json')==evalaudit['protocol_sha256']
joint=read(OUT/'root-joint-interface-audit.json');assert joint['eval_indices_sha256']==sha(E/'indices.json') and joint['eval_protocol_sha256']==sha(E/'protocol.json')
assert read(D/'cpu-checks.json')['focused_tests_passed']
assert read(E/'cpu-checks.json')['tests_passed']==8
assert proposal['evaluation']['indices_sha256']==sha(E/'indices.json')
assert proposal['evaluation']['protocol_sha256']==sha(E/'protocol.json')
assert proposal['evaluation']['cohort_identity_file_sha256']==sha(E/'preparation-audit.json')
assert read(OUT/'interface-joint-cpu-check.json')['failed']==0
for reference in proposal['data']['references'].values():assert sha(ROOT/reference['path'])==reference['sha256']
newfiles=sorted({p for pattern in ['tdyssey/native_rgb_r2c0*.py','scripts/native_rgb_r2c0*.py','tests/test_native_rgb_r2c0*.py','configs/native_rgb_r2c0*.json'] for p in ROOT.glob(pattern)})
extra=['tdyssey/native_rgb_training.py','tdyssey/native_rgb_evaluation.py','tdyssey/raw_world_model_training.py','tdyssey/gpc_world_model.py','tdyssey/native_rgb_run_control.py','scripts/native_rgb_stage3_score.py','configs/native_rgb_stage3_score.json','configs/native_rgb_default.json','configs/native_rgb_selected_physics.json','artifacts/native-rgb-20261028/frozen-renderer-calibration.json']
sourcefiles=newfiles+[ROOT/n for n in extra]+sorted(OUT.glob('*.py'))
assert not (OUT/'source-identity.json').exists(), 'Freeze source once after all owners stop'
source=[]
for p in sourcefiles:
 assert p.is_file(),p
 if p.suffix=='.py':
  tree=ast.parse(p.read_text(encoding='utf-8'));compile(tree,str(p),'exec')
  if p in newfiles:
   assert not any(isinstance(n,ast.Attribute) and n.attr in ['__globals__','FunctionType'] for n in ast.walk(tree))
 target=OUT/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 source.append(dict(path=str(p),relative=p.relative_to(ROOT).as_posix(),sha256=sha(p),snapshot=target.relative_to(OUT).as_posix()))
atomic_json(OUT/'source-identity.json',dict(kind='R2C0-final-source-and-protocol-freeze.v1',at_utc=utc(),files=source,new_model_calls=0))
first=audit['first_cycle'];full=audit['full_proposal'];rest=read(OUT/'supplemental-draw-support.json')
def table(rows):return '\n'.join('|'+ '|'.join(map(str,r))+'|' for r in rows)
splitrows=[['字母','TRAIN轨迹','seen-val平移family','seen-val旋转family','TRAIN明显/其余合法窗'],['---','---:','---','---','---:']]
for r in split['selections']:
 l=r['letter'];support=audit['source_support'][l]
 splitrows.append([l,6 if l=='X' else 7,r['translation_family'],r['rotation_family'],f"{support['moving']['unique_windows']}/{support['other']['unique_windows']}"])
evalrows=[['冻结组','N','H1平移≥2px','H1旋转≥5°','H4平移median/max','H4绝对角median/max'],['---','---:','---:','---:','---','---']]
for group,rows in indices['cohorts'].items():
 tx=[r['h4_translation_image96_px'] for r in rows];ang=[abs(r['h4_signed_rotation_deg']) for r in rows]
 evalrows.append([group,len(rows),sum(r['translation'] for r in rows),sum(r['rotation'] for r in rows),f'{np.median(tx):.6g}/{max(tx):.6g}',f'{np.median(ang):.6g}/{max(ang):.6g}'])
text=f'''# R2C0 已完成：冻结小规模学习诊断的数据与接口

已完成R2B53条引用、TRAIN41/seen-val12切分、80/20固定抽样序列、H1/H4评估索引、真实输入预览及纯CPU接口。**0新控制、0模型加载/推理、0GPU/训练。300次尝试/1200秒只是拟下一步上限，当前未获执行许可。** R2B仅通过小规模WM学习诊断的数据准入，完整触点/方向/边界覆盖仍未通过。

## 数据与切分

源仅R2B已验收53×24控制/25帧，引用原NPZ/metadata/遥测哈希，不复制整套RGB，不混旧D0/D1/R2A/R2A2。选定0.1s协议SHA `58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6`，无阻尼比较。X-left-top静态blocked仍在R2B覆盖报告，不是训练样本。6字母都是既有TRAIN18成员，8保留字母继续封存。

seed5478、domain-separated SHA排序，从满足约束的不同family对中确定val一平移/一旋转；TRAIN保留平移、旋转及可用正负方向和完整H4可视支持。root独立重算选择与可行候选一致。完整轨迹/family分离，不让同轨迹/重叠窗跨TRAIN与val。

{table(splitrows)}

TRAIN H1/H4=861/738，seen-val=252/216，合计1113/954。H1 anchor3..23、H4 anchor3..20，历史3个转移+4帧正确对齐。53条全状态/动作/RGB及初始状态完全重复0，跨轨迹完全相同原RGB帧或state帧0。另明确“同letter、推杆距离≤15worldpx、物体origin≤12worldpx、wrapped角≤5°”的近重复规则，初始及任意保存帧对均0；不因此声称无几何相似性或无偏抽样。所有6letter为seen，val轨迹刻意含运动，本地诊断不能冒充未见字母/无偏自然测试。

## 真正进入训练draw的分布

本次明显运动严格定义为H1面积质心平移≥2个96图像px **OR** abs角≥5°；旧2worldpx定义不用于本池。源TRAIN283明显+578其余窗全部保留。每480周期每letter80次=64明显+16其余，总384/96；先字母、letter/group内独立family均衡，再互斥translation_only/rotation_only/both类别及window队列。重叠仅归both一次；源both仅C的2窗/1family，其它字母此联合类缺层保留。

首480实际category=169平移only+211旋转only+4both+96other，故173次≥2px、215次≥5°、明显union384；覆盖41family、372个unique窗，108次重复曝光，单窗max7。平移median/P95/max={first['translation_image96_px']['median']:.6g}/{first['translation_image96_px']['p95']:.6g}/{first['translation_image96_px']['max']:.6g}px；abs角median/P95/max={first['abs_rotation_deg']['median']:.6g}/{first['abs_rotation_deg']['p95']:.6g}/{first['abs_rotation_deg']['max']:.6g}°。这是OR定义，不要求两个边际中位数分别超过门槛。

拟4800暴露=1676平移only+2121旋转only+43both+960other；覆盖861个unique TRAIN窗/41family，3939次重复曝光，单窗max64，不是新增4800独立数据。每letter640明显+160other。10周期使用延续RNG/独立队列，不机械重复首周期；冻结480/4800 indices、state480/state4800及逐16draw共300batch/逐cycle幅度、phase、contact、实际letter:arc、正负lever、family和重复次数见data/draw-audit.json。

other支持不伪造静止：严格零平移且零角的合法H1为0。辅助near-static仅审计定义(<.1图像px且<.5°)，TRAIN中接触23窗、无AO322窗，较小运动233窗；release242、实际AO loss43、wallstall22（辅助标签重叠）。首480确抽到nearstatic接触3/无AO46、小运动47、release35、AO loss9、wallstall3；逐字母真实arc/独立family及缺类详见supplemental-draw-support.json。它们不改变冻结80/20策略或源数据。

## 冻结评估与真实输入预览

{table(evalrows)}

TRAIN可视12=每letter平移/旋转各1，均来自首480真实暴露且H4合法；没有按模型输出挑图。val natural每轨迹hash取1个H4合法anchor，未按运动筛anchor（但val轨迹自身预选含运动）；12中4明显、8其余。val motion每轨迹1个明显H1且H4合法窗，共12，与natural按trajectory+anchor去重。两val组仍可能共享历史/未来帧与轨迹，重叠披露于preparation-audit.json；分别评分，不合成24条独立轨迹。

中文原RGB预览来自保存96像素：4历史帧、H1/H4 truth、图外绝对动作数字/箭头、真实幅度、分类、split、family与首周期曝光。所有输入保持白/灰/蓝，root检查1325原帧绿色和红色像素0；标注不进入网络。训练12主可视、自然val和motion val预览位于evaluation/truth-*.png；另有首480 other训练示例页（仅展示，不增评估cohort）。没有生成任何模型预测或环境重渲染。

## 可复用接口与CPU证据

`R2C0Dataset`直接读源，五帧=[4历史输入+H1 target]，四动作=[3历史+当前]；`PreparedInputs`输出的model_inputs仅image/action，不传state、pose、letter、moving标签、触点或模板。float32 RGB原uint8/255，动作原式/511*2-1归一化；全部1113窗与原帧/动作bitwise一致。单轨迹cache≤1MiB，没有RGB磁盘副本。

`R2C0WindowOrder`有显式池身份、RNG、队列、游标；完整4800流和state480后的4320步逐项与冻结序列一致。PreparedInputs支持数据层midmicro精确恢复及校验，**这不表示未保存梯度的真实优化器可中途恢复**；未来训练完整checkpoint边界须另保存model/optimizer/scheduler/scaler/RNG、draw状态和预算。

`rollout_raw`仅接真实初始history4与7动作审计前缀；每次predict callback只收history4/actions4。H1复用H4第一步，此后只滑入模型生成的原RGB。动作在此接口恰归一一次，不能把normalized动作交旧absolute-normalizing wrapper，也不能逆变换后再归一化。真实模型bridge尚需下一阶段显式接入，当前没有借旧FunctionType/private globals覆盖。

root联合实际数据+mock审查中，修改未来truth不改变callback输入或输出，false执行guard阻止任何callback；有限OOR原值精确保留/回灌，非finite/wrong-shape硬停并保留原失败与缺测分母。15项聚焦unit tests、数据真实加载、接口合成/实际联测及root独立对齐审查均通过，未导入Torch或构造环境/模型。源码与依赖闭包见source-identity.json/source-snapshot。

后续同窗比较copy-last、原始GPC、训练后模型，按TRAIN展示/val natural/val motion分开报H1/H4、静止/小运动漂移、灰物体前景RGB/IoU/轮廓、蓝推杆指标及硬失败。保持原始数值，有限越界评分硬失败不裁色；遮挡或viewport不完整保留shape unknown，assessable_no_detected_failure不能写成刚性保形通过。灰物体改善不能被蓝推杆改善替代，训练改善不等于未见泛化。评分桥已用既有固定calibration做纯数组检查，无阈值再拟合/渲染。

## 拟下一步配置与当前限制

`configs/native_rgb_r2c0_proposed.json`：原始本地GPC phase-two denoiser SHA `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`（35,043,916B），只做流式哈希未反序列化；不接旧250更新权重。原8,729,827参数image/action→image架构、全参数训练、原residual objective，无motion/pose head、几何loss或颜色/形状修复。沿用已实际跑过的native3060Ti设置：effective batch16、micro4×4、AMP fp16/float32 loss、AdamW lr1e-4/betas(.9,.999)/weight_decay.01、gradclip1、GradScaler128、原cosine3000 horizon/eta_min1e-5；来源和原文件SHA见source-settings-audit.json。

拟上限300 optimizer attempts与1200秒总模型计费先到停；包括model/load、原始baseline H1/H4、训练、最终H1/H4、save/必要reload、失败/清理。最晚900秒停止新尝试并留300秒尾部；实际所需headroom增加只允许更早停，不保证满300。OOM/失败预扣不退、不自动试microbatch/重跑。原全局ledger仍closed250/868.484000000004秒，余2750attempt/4531.516秒；拟300/1200只是余量内提案，未reserve或reset。历史另一native1899/5352.516单列不冲销。

必要数据/抽样/评估数组接口已准备并CPU联测；**真实模型执行入口仍未接入/未获准启动**。旧train的18TRAIN/36dev或旧formal36/250/500硬限制不直接复用，本阶段未调用它；下一阶段需显式新runner、already-normalized predictor bridge和model/time/storage/checkpoint guard。没有用临时私有全局覆盖来假装已可训练。

本阶段0新physics/env/model/GPU/optimizer/training/扩采/服务器/安装下载/上传包；累计物理41976监督/76536历史，旧余24/320不动。原R2B及父审2254个输入身份逐项不变，旧账本与权重/失败记录保留。新增产物连新源码≤32MiB，共享所有旧新source/cache/父审+256MiB原子预留≤3GiB，确数见storage-final.json。完成后实际主动回传并保存receipt，waiting_parent停止，等下一阶段明确指令。
'''
(OUT/'REPORT.md').write_text(text,encoding='utf-8')
keys=['data/manifest.json','data/split-plan.json','data/pools.json','data/draws480.json','data/draws4800.json','data/draw-audit.json','data/cpu-checks.json','data/cpu-loader-check.json','evaluation/indices.json','evaluation/protocol.json','evaluation/preparation-audit.json','evaluation/cpu-checks.json','evaluation/train-other-preview.json','evaluation/index.html','source-identity.json','source-settings-audit.json','interface-cpu-check.json','interface-joint-cpu-check.json','proposed-training-protocol.md','root-data-interface-audit.json','root-joint-interface-audit.json','root-preview-review.json','source-duplicate-audit.json','supplemental-draw-support.json','input-identities.json','REPORT.md']
keys += [p.relative_to(OUT).as_posix() for p in sorted(E.glob('*.png'))]
result=dict(kind='R2C0-CPU-preparation-result.v1',at_utc=utc(),preparation_complete=True,R2B_diagnostic_data_admission=True,formal_full_coverage_dataset_accepted=False,counts=manifest['counts'],cohorts={k:len(v) for k,v in indices['cohorts'].items()},first480={k:first[k] for k in ['draws','category','unique_windows','maximum_window_exposures','independent_families']},proposed4800={k:full[k] for k in ['draws','category','unique_windows','maximum_window_exposures','independent_families']},proposed_attempts=300,proposed_seconds=1200,execution_authorized=False,training_authorized=False,actual_model_entry_ready=False,physics_controls=0,model_loads=0,model_calls=0,GPU_work=0,optimizer_attempts=0,model_seconds=0,physics_cumulative=dict(supervised=41976,historical_native=76536),prior_model_ledger_sha256=sha(EXPERIMENT/'native-rgb-cost-ledger.json'),callback_sent=False,parent_review_received=False,evidence=[dict(file=k,path=str(OUT/k),sha256=sha(OUT/k)) for k in keys],proposal_path=str(ROOT/'configs/native_rgb_r2c0_proposed.json'),proposal_sha256=sha(ROOT/'configs/native_rgb_r2c0_proposed.json'))
atomic_json(OUT/'r2c0-result.json',result)
storage=storage_check('R2C0-complete-before-callback',1024**2);size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+sum(p.stat().st_size for p in newfiles)
storage.update(r2c0_including_new_source_bytes=size,r2c0_limit_bytes=32*1024**2);storage['passed']=storage['passed'] and size+1024**2<=32*1024**2;assert storage['passed'];atomic_json(OUT/'storage-final.json',storage)
s=read(EXPERIMENT/'stage-state.json');s.update(status='R2C0_evidence_ready_for_callback',r2c0_complete=True,r2c0_preparation_authorized=False,r2c0_preparation_spent=True,r2c0_training_authorized=False,r2c0_physics_authorized=False,r2c0_result_file='r2c0/r2c0-result.json',r2c0_result_sha256=sha(OUT/'r2c0-result.json'),parent_review_received=False,next_action='Actual authorized R2C0 callback, receipt, waiting_parent STOP;300/1200 remains proposal, no model/physics/training')
s['resources'].update(r2c0_new_controls=0,r2c0_model_calls=0,r2c0_model_loads=0,r2c0_GPU_work=0,r2c0_optimizer_attempts=0,r2c0_model_seconds=0)
atomic_json(EXPERIMENT/'stage-state.json',s)
print(json.dumps(dict(preparation_complete=True,counts=manifest['counts'],cohorts=result['cohorts'],result_sha256=sha(OUT/'r2c0-result.json'),storage=storage)))
