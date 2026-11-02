"""Closed-run aggregation only. No model, scoring, rendering or weight loading."""
import sys,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c1r1_training import sha,atomic_json
from tdyssey.native_rgb_r2c1r1_budget import storage_for_config
EXP=ROOT/'artifacts/native-rgb-supervised-20261029';OUT=EXP/'r2c1r1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
w=read(OUT/'model-work.json');t=w['training_result'];g=read(EXP/'native-rgb-cost-ledger.json');old=read(OUT/'prior-model-ledger.json')
assert w['closed'] and w['status']=='complete' and w['failure']is None and not t['cleanup_errors']
assert t['progress']['attempts']==t['progress']['updates']==300 and t['progress']['samples']==4800 and t['reload_verified']
assert t['native_sampler_batch_completed']==72 and t['training_forward_completed_calls']==1200 and t['parameter_tensors_changed']==296
assert g['active_stage']is None and g['charged_updates']==g['completed_updates']==550 and g['max_seconds']is None
assert g['stages'][:-1]==old['stages'] and abs(g['charged_seconds']-old['charged_seconds']-486.375)<1e-8
assert read(OUT/'cold-integrity-review.json')['passed'] and read(OUT/'actual-draw-audit.json')['passed']
r=read(OUT/'report/r2c1-report.json');assert r['endpoint_metrics_recomputed_cold']is False and r['frozen_windows']==36
assert (OUT/'report/index.html').exists(),'Wait for final readable full gallery'
assert read(OUT/'report/page-QA.json')['finalized'] is True
for row in read(OUT/'source-identity.json')['files']:assert sha(row['source'])==sha(OUT/row['snapshot'])==row['sha256']
events=[readline for line in (OUT/'run/events.jsonl').read_text(encoding='utf-8').splitlines() if (readline:=json.loads(line))['kind']=='update_complete']
draw=read(OUT/'actual-draw-audit.json')['all']
methods=['copy_last','original_GPC','trained']
def num(x):return 'NA'if x is None else f'{x:.6g}'
def triple(a,key):return '/'.join(num(a['methods'][m]['gray_object'][key]['mean'])for m in methods)
table=['|组|H|N|灰物体IoU C/O/T|灰RGB MAE C/O/T|轮廓误差px C/O/T|训练vs复制误差降低|逐窗赢/输|','|---|---:|---:|---|---|---|---:|---|']
groups={}
for cohort in ['train_visual','val_natural','val_motion']:
 groups[cohort]={}
 for h in ['H1','H4']:
  a=r['cohorts'][cohort][h]['all'];c=a['trained_vs_copy_last'];groups[cohort][h]=dict(methods=a['methods'],trained_vs_copy_last={k:v for k,v in c.items()if k!='windows'})
  table.append('|'+ '|'.join([cohort,h,'12',triple(a,'raw_mask_iou'),triple(a,'object_foreground_rgb_mae'),triple(a,'raw_contour_error_image_px'),f"{100*c['relative_gray_error_reduction']:.2f}%",str(c['endpoint_outcomes'])])+'|')
near={h:r['cohorts']['val_natural'][h]['H1_nearstatic_aux']for h in ['H1','H4']}
small={h:r['cohorts']['val_natural'][h]['H1_small_or_nearstatic']for h in ['H1','H4']}
shape=['|组|H1 trained形状结果|H4 trained形状结果|','|---|---|---|']
for c in ['train_visual','val_natural','val_motion']:shape.append(f"|{c}|{r['cohorts'][c]['H1']['all']['methods']['trained']['shape_outcomes']}|{r['cohorts'][c]['H4']['all']['methods']['trained']['shape_outcomes']}|")
ck=t['checkpoint'];assert sha(ck['path'])==ck['sha256'] and len(list((OUT/'run').glob('*.pt')))==1
text=f'''# R2C1R1：明显运动预测改善，静止漂移和形状问题仍在

已完成本批300次尝试/300次完整参数更新、4800次冻结TRAIN抽样、完整检查点保存与重载核验，以及36固定窗原始GPC/训练后H1/H4和copy-last评分。**12个明显运动seen-val窗口，训练后H1/H4均12/12胜copy-last；H4灰物体IoU从0.258134提高到0.531987。自然验证组有6/12退步，近静止4窗全部退步；不能称保形或未见泛化通过。** 本批结束后不自动续训，交父审。

## 计时纠正先于实际模型启动

用户未设置模型工作时间预算。父此前提出的1200/1172.641秒、900/872.641截断、300秒尾部及全局5400秒拒绝均已撤销；r1模型启动前改为明确record_only_v2，非巨大伪上限/改起时/暂停统计。旧时间帽预备源码和授权JSON另存历史，新配置time字段为null；完整迁移来源见PARENT_TIME_CORRECTION.md、time-policy-v2.json和共享user-model-time-policy.json。7项CPU合同真实验证临时v1→v2账本保留历史，跨900/1200/5400/6000不拒本批操作，300次尝试边界及存储/数值/设备/身份/一次性保护仍有效。

本次486.375秒、全局1382.218秒均为统计，不是剩余额度或自动停训条件。先前失败27.359秒及更早868.484秒全留；global550attempt/550complete、active_stage=null，旧4个历史stage原样，仅追加record-only新stage与显式policy migration。300是本批诊断规模，不是用户指定的总训练上限，也不意味着无限自动续训。停止原因为本批300次完成，未因耗时提前停止。

## 真正进入模型的数据和训练

只用R2C0冻结的R2B53条、0.1秒已选物理协议58563255...20f81f6所派生TRAIN41/seen-val12；原RGB/动作/metadata引用及SHA不变，无旧D0/D1/R2A/R2A2混入，8保留字母未访问。实际4800 draw逐项等冻结序列：3840明显+960其它（80/20），translation_only1676/rotation_only2121/both43/other960。861个unique窗、41独立family、单窗max64曝光；重复曝光不是新增独立数据。每480周期各字母80次=64明显+16其它，10周期延续RNG/队列，逐batch/letter/family/phase/contact/实际arc/±lever审计见actual-draw-audit.json。

原始GPC denoiser4084...1892严格初始化，fresh optimizer/scaler/scheduler/RNG/cursor0；没有用旧250或失败t0权重。8,729,827参数全部trainable，初始298 state tensors完整摘要且全finite，296参数梯度均存在、首轮全非零，最终296参数SHA全改变。原image/action→image、H1 phase-two residual objective、effective16/micro4×4、AMPfp16/FP32loss、AdamW1e-4、clip1、GradScaler128、cosineTmax3000与原3步sampler完全保持。无pose/motion头、几何loss、LoRA或输出修形改色。

网络只收保存的RGB和一次归一化动作。4history+当前动作对齐，4800源索引/72批原始采样/288帧raw与四步反馈、pairedseed5478、endpoint和计费评分输入hash全核对；H4只回灌预测原RGB，没有未来truth/state/pose/模板或重新渲染。1200次micro forward/backward，72 native batch calls（原模型36、训练后36）；H1复用H4第一步。baseline保持/恢复训练RNG和模式，原/终同batch/IDs/actions/history/seed。

loss首个0.0503326，末个0.00289595；首25均值0.0125520，末25均值0.00228131。这仅是训练objective统计，下面同窗灰物体结果才支持运动改善。

## 全分母同窗比较

C=复制上一帧，O=原始T任务GPC，T=训练后。每行固定12窗，没有删除失败或只挑有效子集；IoU越高越好、RGB MAE/轮廓误差越低越好。误差降低按(1−IoU)均值相对copy-last计算，逐窗原值/胜负与蓝推杆指标在report JSON/CSV，蓝推杆不替代灰物体结论。

{chr(10).join(table)}

明显运动val H1灰物体误差下降45.06%，H4下降36.91%，两者12/12赢且四步输出有效；TRAIN展示窗同样12/12赢。自然val平均改善H1 15.35%/H4 20.58%，但两者仅6/12赢、另6/12输，不能仅用平均值掩盖退步。

自然val中8个H1未达“平移≥2图像px OR abs角≥5°”窗，H1误差从0.08522升到0.13829、H4从0.23065升到0.25473，6/8退步。辅助近静止4窗（<.1图像px且<.5°，不是精确静止）全部退步：H1 error0.006768→0.120171，H4 0.007562→0.235675；训练后H4质心相对最后输入平均漂移0.665119图像px、max1.62873px。原样保留，没有用改门槛或额外采样补足。

## 输出安全与形状分开判断

训练后72endpoint和144四步raw均无数值/范围/空palette硬失败，中间步失败0；这不等于形状刚性。原始GPC144 raw中78有限越界，仍原值存储、反馈并继续训练；原端点34有限越界，另有错误palette等硬失败，H1/H2/H3/H4硬失败分别22/27/34/34。原H4路径失败35而endpoint34，说明中途失败没有被最后一帧恢复掩盖。全部原raw/范围/status/评分保留，未裁剪评估值或修色使baseline合格。

{chr(10).join(shape)}

视觉可见部分H4的O/B更圆，C/X笔画或肢段改变，自然近静止灰物体有漂移。unknown包含接触遮挡/边界等不可判断情形，不能当pass；assessable_no_detected_failure也不等于已证明刚性。明显运动预测学到一部分，但当前形状与静止保持仍不通过。

三个cohort分别评分：TRAINvisual用于展示，seen-val natural每轨迹hash选anchor、但轨迹本身刻意含运动，seen-val motion按明显运动筛选。后两组同12条seen轨迹且部分历史/未来重叠，不能说24条独立验证，也不是8未见字母泛化或无偏自然测试；本轮单seed/小数据/300步不能给出所有形状可靠结论。

## 检查点、原始预测与完整展示

唯一R1完整payload：`{ck['path']}`，updates300，{ck['bytes']:,}B，SHA `{ck['sha256']}`。含model/optimizer/scaler/scheduler/所有CPU+CUDA+Python+NumPy+noise RNG/PreparedInputs队列游标/配置源数据评估SHA/费用链接；只在完整optimizer边界保存。实际单次重载严格load通过，运行中比较model named_parameters摘要、optimizer/scaler/scheduler/RNG/prepared和progress一致；state_dict buffers严格load但未另做独立前后digest，不夸大该验证范围。旧C1失败t0 b044...、claim、源码/快照/费用及49MB失败报告全保留。

run/evaluation保留全部72批/288原帧（包括H2/H3）、H1/H4endpoint、范围/硬失败及已计时评分。端点216行全部完成，cold报告不重评分，不读checkpoint、不调用模型。report/index.html给全36窗H1/H4主72行与独立“原始T任务GPC”72行、24页；图像显示编码与原float评估分开。原frozen报告页标题拥挤，独立布局helper仅font/header/DPI调整，可读新24页与原24页均保留；raw/scoring/source数学没有改变，不按好坏选图。

## 审计、资源和结束

独立11177项检查0异常：128 active+128snapshot、2594旧保护文件、4800draw、四步反馈与原始动作/pairedseed/endpoint/canonical评分输入hash一致，旧C1 t0不变。runtime-review24项通过，9个method×cohort评分JSON均在模型时钟阶段完成，GPU清理后才结束统计；冷态没有新推理/重载/评分。CUDA allocator peak reserved2,025,848,832B<6GiB，0physics/env/render/unseen/server/install/download/uploadpack。物理累计41976/76536及旧余24/320不动。原GPC库有HFcache写权限提示但本地权重严格加载成功、无下载。

R1新增产物≤256MiB；共享旧新native/source/cache/父审与旧49MB失败+256MiB原子预留≤3GiB，具体storage-final.json。可复现实验命令为 `python -X utf8 -B scripts/native_rgb_r2c1r1_run.py --config configs/native_rgb_r2c1r1.json --execute`；该stage once-claim已用，不直接重复、不中途删claim或自动续训。

本批完整结束。按用户长期授权主动回传报告/实际抽样/原预测图库/检查点与policy迁移，保存真实receipt后waiting_parent STOP；后续规模根据父验收和结果决定，不以父曾设诊断时长冒充用户预算，也不反复要求补时间。
'''
(OUT/'REPORT.md').write_text(text,encoding='utf-8')
files=['REPORT.md','model-work.json','source-identity.json','input-identities.json','time-policy-v2.json','PARENT_TIME_CORRECTION.md','actual-draw-audit.json','cold-integrity-review.json','runtime-review.json','record-only-cpu-contracts.json','run/initial-tensors.json','run/events.jsonl','run/training-result.json','report/r2c1-report.json','report/index.html','report/endpoint-scores.csv','report/all-raw-steps.csv','report/layout-repair-record.json','report/page-QA.json','report/SCIENCE_SUMMARY.md','report/SCIENCE_SUMMARY.json']
result=dict(kind='R2C1R1-complete-seen-motion-diagnostic-result.v2',status='complete_pending_parent_review',attempts=300,completed_updates=300,actual_draws=draw,parameter_count=8729827,parameter_tensors_changed=296,training_forward_calls=1200,native_sampler_batch_calls=72,raw_prediction_frames=288,endpoint_rows=216,endpoint_rows_missing=0,learning_result='seen-motion foreground improves overcopy-last;near-staticdrift_and_shape_failures_remain',gray_motion_improved=True,shape_accepted=False,near_static_accepted=False,unseen_generalization_claimed=False,cohort_summary=groups,time_policy='record_only_v2',user_set_model_time_budget=False,model_time_hard_limit=None,current_model_seconds_statistics=486.375,historical_total_model_seconds_statistics=g['charged_seconds'],historical_total_attempts=550,previous_failed_seconds_preserved=27.35899999999674,checkpoint=ck,checkpoint_reload_verified=True,new_physics_controls=0,automatic_training_continuation=False,execution_authorized=False,callback_sent=False,parent_review_received=False,evidence=[dict(file=f,path=str(OUT/f),sha256=sha(OUT/f))for f in files])
atomic_json(OUT/'r2c1r1-result.json',result)
a=storage_for_config(read(ROOT/'configs/native_rgb_r2c1r1.json')).check(force=True);atomic_json(OUT/'storage-final.json',a)
s=read(EXP/'stage-state.json');s.update(status='R2C1R1_complete_evidence_ready_for_callback',r2c1r1_complete=True,r2c1r1_training_completed=True,r2c1r1_evaluation_completed=True,r2c1r1_model_work_closed=True,r2c1r1_execution_authorized=False,training_authorized=False,r2c1r1_gray_motion_improved=True,r2c1r1_shape_accepted=False,r2c1r1_near_static_accepted=False,r2c1r1_result_file='r2c1r1/r2c1r1-result.json',r2c1r1_result_sha256=sha(OUT/'r2c1r1-result.json'),next_action='Actual authorized result callback thenreceipt/waiting_parentSTOP; noautomatictraining. Modeltime statistics only, no oldtimebudget.')
atomic_json(EXP/'stage-state.json',s)
print(json.dumps(dict(attempts=300,completed=300,draws=4800,time_policy='record_only_v2',current_seconds_statistics=486.375,global_seconds_statistics=g['charged_seconds'],motion_val_H4_IoU=[.25813392380176253,.53198667750081],stage_bytes=a['r2c1_artifact_bytes'],shared_bytes=a['bytes'],report=str(OUT/'REPORT.md'))))
