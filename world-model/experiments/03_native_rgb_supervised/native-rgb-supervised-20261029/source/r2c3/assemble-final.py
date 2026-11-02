"""Only aggregate a closed, audited6000-point run; no model or re-scoring."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c3_continuation import storage_for_config
OUT=Path(__file__).resolve().parent;EXP=OUT.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
w=read(OUT/'model-work.json');t=w['training_result'];p=t['progress'];a=p['r2c3']
assert w['closed'] and w['status']=='complete' and w['failure'] is None and not t['cleanup_errors']
assert p['updates']==6000 and a['completed_updates']==5700
assert a['attempts']==5700+a['AMP_skips'] and a['actual_draws']==16*a['attempts']
assert p['actual_draws_total']==p['samples']==4800+a['actual_draws']
assert t['final_checkpoint_reload_verified'] and t['optimizer_step_applications_observed']==5700
assert t['native_sampler_batch_completed']==t['native_sampler_batch_attempts']==216
assert t['denoiser_forward_completed']==t['denoiser_forward_attempts']==2052 and t['fixed_loss_forward_calls']==27
assert t['training_forward_completed_calls']==t['training_forward_calls']==4*a['attempts']
audit=read(OUT/'cold-integrity-review.json');assert audit['passed']
r=read(OUT/'report/r2c3-learning-curve.json');assert r['endpoint_rows']==648 and r['missing_endpoint_metrics']==0 and r['endpoint_metrics_recomputed_cold']is False
assert read(OUT/'report/page-QA.json')['finalized']is True
visual=read(OUT/'report/focus-visual-review.json');assert visual['reviewed']is True
config=read(ROOT/'configs/native_rgb_r2c3.json')
for row in read(config['source_identity'])['files']:assert sha(row['source'])==sha(OUT/row['snapshot'])==row['sha256']
g=read(EXP/'native-rgb-cost-ledger.json');old=read(OUT/'prior-model-ledger.json')
assert g['active_stage']is None and g['max_seconds']is None and g['max_updates']is None
assert g['completed_updates']==6250 and g['charged_updates']==550+a['attempts'] and g['stages'][:-1]==old['stages']
ck=t['checkpoint'];assert sha(ck['path'])==ck['sha256']
assert len(list((OUT/'run').rglob('*.pt')))==3
table=['|组|去噪次数|H4 IoU：300→1000→3000→6000|','|---|---:|---|']
for cohort in ('train_visual','val_motion','val_natural'):
    for steps in (3,16):
        values=[r['cohorts'][cohort]['H4']['all']['methods'][f'u{u}_s{steps}']['gray_object']['raw_mask_iou']['mean']for u in (300,1000,3000,6000)]
        table.append(f'|{cohort}|{steps}|'+ ' → '.join(f'{v:.6f}'for v in values)+'|')
seconds=w['budget']['elapsed_model_work_seconds']
files=['model-work.json','source-identity-v2.json','input-identities-v2.json','cpu-contracts.json','engineering-review.json','cold-integrity-review.json','point1000-review.json','point3000-review.json','USER_CONTINUATION_AUTHORIZATION.md','PARENT_EXECUTION_CORRECTIONS.md','FINAL_FOCUS_DELIVERY.md','run/resume-verification.json','run/training-result.json','report/r2c3-learning-curve.json','report/REPORT.md','report/RUNTIME.md','report/SCIENCE_SUMMARY.md','report/SCIENCE_SUMMARY.json','report/page-QA.json','report/focus-visual-review.json','report/index.html','report/focus-gallery/index.html','callback-point1000-result.json','callback-point3000-result.json']
result=dict(kind='R2C3-complete-actual-learning-curve-result.v1',status='complete_pending_final_callback',lineage_updates=6000,
    resumed_from_lineage_updates=300,new_complete_updates=5700,new_attempts=a['attempts'],AMP_skips=a['AMP_skips'],
    new_actual_draws=a['actual_draws'],total_lineage_actual_draws=p['actual_draws_total'],historical_total_complete_updates=6250,
    training_forward_calls=t['training_forward_completed_calls'],new_sampler_batch_calls=216,actual_sampling_denoiser_forwards=2052,
    fixed_loss_forward_calls=27,new_raw_float32_frames=864,new_endpoint_rows=432,reused_endpoint_rows=216,full_report_endpoint_rows=648,
    final_checkpoint=ck,final_checkpoint_actual_reload_verified=True,parameter_tensors_changed_from300=t['parameter_tensors_changed_from_300'],
    full_resume_verification=read(OUT/'run/resume-verification.json')['components'],time_policy='record_only_v2',human_model_time_budget=False,
    model_time_limit=None,current_model_seconds_statistics=seconds,historical_total_model_seconds_statistics=g['charged_seconds'],
    no_new300_evaluation=True,fixed_loss_points=[1000,3000,6000],same_noise_plan_verified_not_actual_noise_tensor_bytes=True,
    no_new_physics=True,no_unseen_access=True,no_server=True,shape_accepted=False,convergence_claimed=False,
    capacity_limit_claimed=False,unseen_generalization_claimed=False,automatic_further_training=False,
    focus_visual_review=visual,callback_sent=False,parent_review_received=False,
    evidence=[dict(file=f,path=str(OUT/f),sha256=sha(OUT/f))for f in files])
save(OUT/'r2c3-result.json',result)
text=f'''# R2C3：真实完整恢复至6000更新的训练长度学习曲线

已从实际300更新checkpoint恢复model/optimizer/scaler/scheduler/全部RNG与PreparedInputs顺序，完成新增{a['attempts']}次尝试、5700次真正optimizer更新、{a['AMP_skips']}次AMPskip、{a['actual_draws']}次新draw；本谱系6000更新、累计{p['actual_draws_total']}draw。旧另一谱系250只留历史账本，不计这条训练曲线。无300新推理/固定loss，复用全部旧输出；不是再训300就下结论。

{chr(10).join(table)}

全部同窗逐组、逐窗横向3/16和纵向训练长度比较，非明显8窗、近静止4窗、轮廓/形状/孔洞碎片、RGB、质心、漂移与蓝推杆单列见[完整科学报告](report/REPORT.md)、[实际曲线](report/index.html)、[科学摘要](report/SCIENCE_SUMMARY.md)。TRAIN增益和验证变慢/停滞必须并列；不能用训练loss、均值或条件可判定集合代替所有窗结论。unknown只是既定遮挡/近邻代理或边界门控，不能当pass，也不能据此遮掩可见畸变。1000的24unknown均为1px灰mask膨胀与蓝mask相交代理，非24个真实遮挡证明。

## 用户指定的O/C棱角对照

[固定4窗主图及O/C细节](report/focus-gallery/index.html)严格四列：真实未来/训练1000/训练3000/训练6000，全部3次去噪、同动作/配对seed，H1(.1s)与H4(.4s)分开。O/B/C/X固定窗口，未挑好看样本。O/C同row所有列使用同坐标裁切、同整数nearest，不旋转平移对齐、不锐化平滑修形，不保存重复大数组。具体外直边/转角/O孔/C开口和笔画目视判断在report/focus-visual-review.json；IoU提高或阈值未触发不能代替棱角。旧parent-view-1000/3000均保留并计入stage。

## 实际工程与资源事实

原GPC图像+动作→图像、原phase-two residual loss、全部参数、固定6seen/41TRAIN+12seen-val、80/20、batch16/micro4与物理0.1s协议不变。无新数据/physics/env/阻尼验证/未见8字母/服务器/下载/几何或运动模块。旧stream4800、stage300和global3000计划守卫显式迁移；所有slot/queue RNG连续，不重复前4800。实际draw索引全存并冷重建核对。正常GradScaler溢出降scale继续，skip消费draw/费用而不计complete或LR；本次实际skip数见上。学习率原cos至3000后保持1e-5，原scheduler.last_epoch停3000、真实complete计数独立保存，没有反弹或warmup重置。

新6组生成评测共216 sampler批、2052实际sampling denoiser forward、864原始float32四步帧、432新端点评分；300与copy复用216行，合计648端点。固定noise计划loss只在1000/3000/6000，27批；核对seed/IDs/shape与generator前后hash，未独立逐字节核对未保存的sigma/epsilon tensors。所有新评分在实际model clock内完成，冷报告0模型/0load/0评分。

每100success保存同一个完整last，1000/3000另留model-only，6000完整点实际重载核验。最终checkpoint `{ck['path']}`，{ck['bytes']}B，SHA `{ck['sha256']}`。只有2轻量+1完整新权重，无重复t0；旧权重/数据/失败/费用全保留。初始完整恢复7项实际核验全true，最终参数相对300变化{t['parameter_tensors_changed_from_300']}个张量，完整reload=true。13必要CPU合同通过，141源码/快照与3222旧保护文件及raw/评分对应独立冷审通过，详cold-integrity-review.json；旧通用仓库CRLF/LF三差异保持已知，未声称全绿。

本次实际全model work {seconds:.6f}秒、全局{g['charged_seconds']:.6f}秒仅统计，用户未设模型时间上限，所有旧硬截止不再生效。stage<=400MiB/shared<=3GiB含256MiBatomic，最终storage-final.json；所有旧交付、两套prelaunch source freeze历史、父图都保留。

6000是当前学习曲线观察点，不是收敛、能力上限、保形无望或用户总训练额度。固定采样下训练更久的收益支持300不足是因素；同权重采样效应及其交互独立呈现，不强行二选一。是否继续10000+依实际曲线交父审，不自动执行。

1000/3000已实际主动回传且不中断训练。现在按长期直接人类授权主动回传完整6000结果、绝对图片路径与图页，保存真实receipt后waiting_parent STOP；不轮询、不自动更多模型工作。
'''
(OUT/'REPORT.md').write_text(text,encoding='utf-8')
result['evidence'].append(dict(file='REPORT.md',path=str(OUT/'REPORT.md'),sha256=sha(OUT/'REPORT.md')));save(OUT/'r2c3-result.json',result)
save(OUT/'storage-final.json',storage_for_config(config).check(force=True))
s=read(EXP/'stage-state.json');s.update(status='R2C3_complete_ready_for_actual_final_callback',r2c3_complete=True,r2c3_model_work_closed=True,
    r2c3_execution_authorized=False,r2c3_training_authorized=False,training_authorized=False,future_stages_authorized=False,
    r2c3_result_file='r2c3/r2c3-result.json',r2c3_result_sha256=sha(OUT/'r2c3-result.json'),next_action='Actual final6000callback with focus images+curves,thenreceipt/waiting_parentSTOP. Noauto10000+.')
save(EXP/'stage-state.json',s)
print(json.dumps(dict(complete=True,lineage_updates=6000,new_updates=5700,attempts=a['attempts'],skips=a['AMP_skips'],draws=a['actual_draws'],current_seconds_statistics=seconds,global_seconds_statistics=g['charged_seconds'],checkpoint=ck)))
