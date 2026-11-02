"""Aggregate closed R2C2 evidence; never load a model, checkpoint or re-score."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c2_inference import storage_for_config
EXP=ROOT/'artifacts/native-rgb-supervised-20261029';OUT=EXP/'r2c2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
w=read(OUT/'model-work.json');i=w['inference_result'];r=read(OUT/'report/r2c2-report.json')
assert w['closed'] and w['status']=='complete' and w['failure'] is None and not i['cleanup_errors']
assert i['parameters_unchanged'] and i['all_state_tensors_unchanged'] and i['model_checkpoint_loads']==1
assert i['optimizer_attempts']==i['completed_updates']==i['new_checkpoints_saved']==0
assert i['denoiser_forward_attempts']==i['denoiser_forward_completed']==864
assert r['endpoint_rows']==288 and r['missing_endpoint_metrics']==0 and r['endpoint_metrics_recomputed_cold'] is False
assert read(OUT/'report/page-QA.json')['finalized'] is True
audit=read(OUT/'cold-integrity-review.json');assert audit['passed']
for row in read(OUT/'source-identity.json')['files']:assert sha(row['source'])==sha(OUT/row['snapshot'])==row['sha256']
g=read(EXP/'native-rgb-cost-ledger.json');old=read(OUT/'prior-model-ledger.json')
assert g['active_stage'] is None and g['charged_updates']==g['completed_updates']==550 and g['max_seconds'] is None
assert g['stages'][:-1]==old['stages']
for n in ('8','16'):
    q=i['settings'][n]
    assert (q['sampler_batch_attempts'],q['sampler_batch_completed'],q['raw_frames_saved'],q['endpoint_rows_scored'])==(36,36,144,72)
assert not list(OUT.rglob('*.pt')),'No duplicate weights in C2'
seconds=w['budget']['elapsed_model_work_seconds']
groups={c:{h:r['cohorts'][c][h]['all']for h in ('H1','H4')}for c in ('train_visual','val_natural','val_motion')}
motion=[groups['val_motion']['H4']['methods'][m]['gray_object']['raw_mask_iou']['mean']for m in ('trained3','trained8','trained16')]
near=[r['cohorts']['val_natural']['H4']['H1_nearstatic_aux']['methods'][m]['gray_object']['raw_object_error']['mean']for m in ('trained3','trained8','trained16')]
files=['PARENT_REVIEW.md','PARENT_TRAINING_LENGTH_NOTE.md','model-work.json','source-identity.json','input-identities.json','cpu-contracts.json','engineering-review.json','cold-integrity-review.json','report/r2c2-report.json','report/REPORT.md','report/index.html','report/page-QA.json','R2C3_PROPOSAL.md']
if (OUT/'report/RUNTIME.md').exists():files.append('report/RUNTIME.md')
result=dict(kind='R2C2-fixed300-checkpoint-native-denoising-result.v1',status='complete_pending_parent_review',
    settings=[3,8,16],reused_settings=[3],new_settings=[8,16],checkpoint_lineage_updates=300,
    checkpoint_path=str(EXP/'r2c1r1/run/last.pt'),checkpoint_sha256=i['checkpoint_sha256_after'],checkpoint_loads=1,
    parameters_unchanged=True,all298_state_tensors_unchanged=True,new_optimizer_attempts=0,new_completed_updates=0,new_checkpoints=0,
    new_sampler_batch_calls=sum(q['sampler_batch_completed']for q in i['settings'].values()),
    new_raw_float32_frames=sum(q['raw_frames_saved']for q in i['settings'].values()),
    new_endpoint_rows=sum(q['endpoint_rows_scored']for q in i['settings'].values()),
    actual_denoiser_forward_attempts=i['denoiser_forward_attempts'],actual_denoiser_forward_completed=i['denoiser_forward_completed'],
    time_policy='record_only_v2',user_model_time_budget=False,model_time_hard_limit=None,model_seconds_statistics=seconds,
    historical_total_model_seconds_statistics=g['charged_seconds'],historical_attempts=550,historical_completed_updates=550,
    cohort_summary=groups,shape_accepted=False,training_sufficiency_decided=False,convergence_claimed=False,
    conclusion_scope='Sampler effects at fixed300-update weights only; training-length sufficiency requires planned longitudinal R2C3 matrix.',
    R2C3_proposal_file=str(OUT/'R2C3_PROPOSAL.md'),R2C3_execution_authorized=False,
    no_new_physics=True,no_unseen_access=True,no_server=True,automatic_training_continuation=False,
    callback_sent=False,parent_review_received=False,
    evidence=[dict(file=f,path=str(OUT/f),sha256=sha(OUT/f))for f in files])
save(OUT/'r2c2-result.json',result)
summary=f'''# R2C2：固定300更新权重的原生采样次数诊断

本阶段只比较同一个可信300更新检查点的3/8/16次去噪。3次及copy-last完全引用R1，新增8/16各36次sampler批调用、144原始float32四步帧及72个H1/H4端点评分。合计72批、288帧、144新端点评分；实际inner_model forward共{i['denoiser_forward_completed']}次，初始和结束全部296参数/298state张量摘要相同。无训练、参数更新或重复权重。

主要结果：3/8/16次去噪的motion H4灰IoU依次{motion[0]:.6f}/{motion[1]:.6f}/{motion[2]:.6f}；nearstatic4窗H4灰error依次{near[0]:.6f}/{near[1]:.6f}/{near[2]:.6f}。H4形状failure/unknown分别12/24、13/23、12/24，更多步未修复O/B磨圆、C/X笔画变化。局部指标改善仍保留，例如8次motion H4 ROI MAE略好，8/16的nearstatic H1质心漂移低于3次，但未转为H4整体保形或静止保持通过。

完整逐组/逐窗/非明显运动8窗/近静止4窗/孔洞碎片/形状failure和unknown结果见[固定口径报告](report/REPORT.md)及[全部24页同窗图库](report/index.html)。真实调用、耗时和显存见model-work.json及report/RUNTIME.md，raw和运行内评分在run/evaluation；冷态未补评分。source-identity.json保留133当前源码及快照，input-identities.json绑定2932旧保护文件，独立审计见cold-integrity-review.json。所有旧产物和原105300718字节检查点保留，SHA为{i['checkpoint_sha256_after']}。

**300更新仅说明初期学习信号。本实验只能检验固定300权重上的采样因素，不证明训练充分、收敛、能力上限或保形无望，也不能断言继续训练无用。** 六seen字母/当前36窗，natural和motion同轨迹重叠；无未见泛化或独立测试声明。若同训练点采样改善支持采样是因素；训练长度是否为因素须看固定采样的纵向曲线，二者可以共同作用。

本次实际模型耗时{seconds:.6f}秒、全局{g['charged_seconds']:.6f}秒均仅统计，record_only_v2，没有人类时间预算或旧1200/900/300/5400停止规则。旧550次训练完成和全部先前失败费用不动；本阶段新增训练0。物理41976/76536不动，0新数据/physics/env/阻尼比较/未见八字母/服务器/安装下载。旧发布JSON三项CRLF/LF差异已知且未修改，不声称通用检查全绿。

下一阶段只准备[R2C3完整提案](R2C3_PROPOSAL.md)和项目configs/native_rgb_r2c3_proposed.json：从本谱系300完整状态恢复至1000/3000/6000观察点，横向3/16、纵向训练长度交叉，cos到3000后hold eta_min，扩展抽样顺序及旧attempt守卫须明确版本化，不冒充人类总限额。空间先按最坏字节预算，不删除交付权重、不复制数据。父验收R2C2后下达执行，本阶段不自动续训或跳转。服务器完整训练始终由用户手动执行。

阶段文件<=128MiB/shared<=3GiB含256MiB原子余量；最终storage-final.json。完成后按直接人类长期授权实际回传父对话__LOCAL_CHAT__，保存真实receipt后waiting_parent STOP，不轮询。
'''
(OUT/'REPORT.md').write_text(summary,encoding='utf-8')
result['evidence'].append(dict(file='REPORT.md',path=str(OUT/'REPORT.md'),sha256=sha(OUT/'REPORT.md')))
save(OUT/'r2c2-result.json',result)
save(OUT/'storage-final.json',storage_for_config(read(ROOT/'configs/native_rgb_r2c2.json')).check(force=True))
s=read(EXP/'stage-state.json');s.update(status='R2C2_complete_evidence_ready_for_callback',r2c2_complete=True,r2c2_model_work_closed=True,r2c2_execution_authorized=False,r2c2_training_authorized=False,training_authorized=False,future_stages_authorized=False,r2c3_execution_authorized=False,r2c2_result_file='r2c2/r2c2-result.json',r2c2_result_sha256=sha(OUT/'r2c2-result.json'),next_action='Actual authorized callback including R2C3 proposal,thenreceipt/waiting_parent STOP; noauto training,modeltime statistics only.')
save(EXP/'stage-state.json',s)
print(json.dumps(dict(complete=True,new_sampler_batches=72,raw_frames=288,new_endpoints=144,actual_forwards=864,model_seconds_statistics=seconds,global_seconds_statistics=g['charged_seconds'],report=str(OUT/'REPORT.md'))))
