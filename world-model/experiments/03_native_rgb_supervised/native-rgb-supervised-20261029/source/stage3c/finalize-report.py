"""CPU-only final evidence/state writer, without model imports."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parent.parent


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def put(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


w=read(HERE/'model-work.json');r=w['training_result'];p=r['progress'];ledger=read(BASE/'native-rgb-cost-ledger.json')
score=read(HERE/'score/score-report.json');comparison=read(HERE/'comparison129-final/comparison.json')
integrity=read(HERE/'review-postrun-integrity.json');review=read(HERE/'review-postrun-budget.json')
restore=read(HERE/'restore-review.json');now=datetime.now(timezone.utc).isoformat()
assert w['failure'] is None and r['status']=='complete' and w['budget']['closed'] and ledger['active_stage'] is None
assert ledger==w['budget']['ledger'] and ledger['charged_updates']==ledger['completed_updates']==p['updates']==250
assert integrity['integrity_checks_passed'] and integrity['finalized'] and not integrity['pending']
assert review['all_budget_and_integrity_checks_passed'] and restore['all_protected_state_equal']
assert score['train']['dynamic_fit_target']['passed'] and comparison['inference_calls_for_old_baseline_or129']==0

storage=read(HERE/'final-storage.json');seen=set();total=0
for root in storage['roots']:
    path=Path(root);assert path.exists()
    for f in path.rglob('*') if path.is_dir() else [path]:
        if f.is_file() and str(f.resolve()).casefold() not in seen:
            seen.add(str(f.resolve()).casefold());total+=f.stat().st_size
snapshot=dict(kind='stage3c-storage-at-callback.v1',checked_at_utc=now,root_list_source='final-storage.json',
    files=len(seen),bytes=total,reserved_atomic_bytes=268435456,projected_bytes=total+268435456,
    max_bytes=3221225472,passed=total+268435456<=3221225472,
    qualification='Metadata snapshot before final result/callback small files; no deletion or cleanup.')
assert snapshot['passed'];put(HERE/'storage-at-callback.json',snapshot)

names=['REPORT.md','DIAGNOSIS.md','range-accounting.json','model-work.json','run/result.json','run/last.pt',
       'restore-review.json','run/history.jsonl','run/sampling-history.jsonl','source-identity.json',
       'continuation-plan.json','entry-from-stage3b.diff','continuation-cpu-checks.json','root-entry-cpu-check.json',
       'predictions/fitted-train18.npz','predictions/fitted-dev36.npz','score/score-report.json','score/per-window.csv',
       'score-cpu-timing.json','comparison129-final/comparison.json','comparison129-final/all54-method-comparison.csv',
       'comparison129-final/gallery-pages.json','review-postrun-budget.json','review-postrun-integrity.json',
       'root-visual-review.json','final-storage.json','storage-at-callback.json']
evidence={name:dict(path=str(HERE/name),sha256=sha(HERE/name)) for name in names}
result=dict(kind='native-rgb-stage3c-exact-continuation-result.v1',stage='3C',created_at_utc=now,
    outcome='exact_resume_verified_train18_registered_microfit_passed_dev_and_shape_not_accepted',
    one_actual_execution=True,execution_cycle_finished=True,failure=None,training_chain_verified=True,
    start_updates=129,final_updates=250,local_optimizer_attempts=121,local_optimizer_updates=121,
    local_draws=1936,cumulative_draws=4000,unique_training_windows=18,selected_window_exposures=p['selected_window_exposures'],
    native_parameter_count=r['native_parameter_count'],parameter_tensors_changed_since129=integrity['parameter_tensors_changed_since129'],
    inherited_first_gradient_norms_scope='stage3B; not new3C per-parameter measurements',
    prior_fixed_train_loss=0.0024270865445335708,final_fixed_train_loss=p['fixed_train_loss_after'],
    denoising_validation=p['validation'],final_learning_rate=9.84666621830081e-05,
    restored_components=['model','optimizer','scaler','scheduler','rng','order'],actual_state_restore_verified=True,
    parent_checkpoint_sha256=w['parent_checkpoint_sha256'],checkpoint=r['last'],checkpoint_semantic_review=w['checkpoint_review'],
    new_native_prediction_frames=54,new_native_sampler_batch_calls=14,rebaseline_calls=0,repredict129_calls=0,
    raw_arrays_unmodified=True,train_dynamic_fit_target=score['train']['dynamic_fit_target'],
    four_method_summary=comparison['summary'],shape_interpretation='Rounded contours persist; TRAIN18unknown; DEV1fail/3assessable+33unknown. Unknown is not a pass.',
    model_work_seconds=w['budget']['elapsed_model_work_seconds'],prior_3_plus3b_model_seconds=444.6250000000073,
    aggregate_new_round_model_seconds=ledger['charged_seconds'],aggregate_new_round_attempts=ledger['charged_updates'],
    global_ledger_closed=True,overrun_seconds=w['budget']['overrun_seconds'],
    peak_cuda_allocated_bytes=w['peak_cuda_allocated'],peak_cuda_reserved_bytes=w['peak_cuda_reserved'],
    remaining_new_round_attempts=3000-ledger['charged_updates'],remaining_new_round_seconds=5400-ledger['charged_seconds'],
    remaining_is_not_spending_authorization=True,historical_native_attempts=1899,historical_native_model_seconds=5352.516,
    physical_controls=0,supervised_physical_controls=39680,all_native_physical_controls=74240,
    no_formal_train=True,no_H4=True,no_full319=True,no_unseen=True,no_server=True,no_packaging=True,
    integrity_audit_passed=True,budget_audit_passed=True,storage_at_callback=snapshot,evidence=evidence,
    next_action='Actual parent callback then waiting_parent and stop, even with TRAIN microfit target passed.')
put(HERE/'stage3c-result.json',result)
s=read(BASE/'stage-state.json');s.update(status='ready_to_report',stage3c_result_file='stage3c/stage3c-result.json',
    stage3c_result_sha256=sha(HERE/'stage3c-result.json'),stage3c_dynamic_fit_demonstrated=True,
    stage3c_independent_dev_accepted=False,stage3c_shape_accepted=False,stage3c_outcome=result['outcome'],
    future_stages_authorized=False,next_action='Actual3C parent callback, save receipt, then waiting_parent and stop.')
s['resources'].update(optimizer_attempts=250,stage3c_completed_optimizer_updates=121,stage3c_cumulative_updates=250,
    stage3c_new_draws=1936,stage3c_cumulative_draws=4000,stage3c_model_work_seconds=result['model_work_seconds'],
    aggregate_model_work_seconds=ledger['charged_seconds'],stage3c_native_prediction_frames=54,
    stage3c_native_prediction_batch_calls=14,supervised_native_prediction_batch_calls=43,model_inference_calls=43,
    combined_persistent_bytes_at_report=total)
s['callback'].update(message_file='stage3c/callback-stage3c.txt',result_file='stage3c/callback-result.json',sent=False)
put(BASE/'stage-state.json',s)
callback=f'''[阶段3C完成回传] 唯一精确续行已完成：129→250，新增121尝试=121完成更新，1936新draws/累计4000，同18窗口。TRAIN预注册微拟合门槛达到；DEV与刚性形状仍未验收。请父审查再决定下一步；本实施实际回传成功后waiting_parent并结束，不自行正式TRAIN或额外续训。

主结论：TRAIN final灰1−IoU=.18517149901054739，hold=.23258317101512865，改善20.3848248%，15/18wins、0hardfail，原20%+12/18标准未改。129为.2280759621；到250全部18/18都改善，均值下降18.81148%。TRAIN contour .456354(129)→.343421imagepx、前景RGB .052082→.042728，亦优于hold。不是仅以loss宣布通过，仍只是同18样本小诊断。

DEV36 final=.259424347516095，129=.274351965，下降5.4410%但hold=.196032459仍更好；final6/36胜hold，2hardfail。moving31 final=.269618464 vs hold=.223749623，差20.5001%、6/31wins；static4 drift error=.169645781/centroid .610816310imagepx（129质心.582984），不要求胜过零。DEV F seen-val-2c-112:29、J seen-val-2c-097:38推杆mask空，gray仍可测，各自pusher惩罚保留。TRAIN legacy0assessable/18unknown；DEV1fail/3assessable+33unknown；unknown不算通过。全18对照图已检查，棱角/洞/窄槽仍有圆化，不能称保形/泛化已过。

真实恢复：父last SHA e94f9349e1affb8372aacc4613bd57ade496704a50b63d257f77cb4c5ad3626a。同一计费运行内构造原8,729,827模型并载完整129，model/optimizer/scaler/scheduler/RNG/WindowOrder6状态type-shape-dtype-bytes digest全部相同；129/2064cursor/296optimizer states/Tmax3000/currentLR核验，参数hash全对3B129实际fitted。新目录先同名保存完整129边界再130→250，130首采样序列精确续接；父129完整保留。原trainer字节/严格通用restore未改，仅FunctionType同bytecode的3个私有globals做窄验证/真实source/verified-state-saveguard。14明确metadata delta、39语义字段不变；恢复前失败禁止写fresh0。phase/旧stop标签迁移有记录，原init/first-gradient字段保留且明确是3B历史证据；121新step都记录实际globalnorm/LR、原all-gradient检查，296参数自129全变化。

fixednoise TRAIN loss .00242708654→.00193373367，DEV denoise .00265394265→.00238773488；lr129 9.959002382e-5→250 9.846666218e-5。末新更新local392.031s<420，最终14批54H1只做一次，不重测原baseline/129；全部新raw finite[0,1]，原finite-OOR政策/失败惩罚/完整18+36分母不变，无clip/映射/校准/阈值/头/loss改动。

最终last105290926B SHA7a3c3fed60f444536afe8999d6f6346d1ce41258f8df7d87dc78f18bcb4ba28f，10完整字段/currentidentity/250/cursor4000/296states/Tmax3000 CPU语义检查.140s，于423.843s内完成，非额外resume实验。3C clock423.85899999999674s/121尝试；原3+3B444.6250000000073s/129费用及200/600授权原样保留，新的独立121/600授权单列。全轮868.484000000004s/250，ledgerclosed、overrun0；余额2750/4531.516仅数值无花费授权。旧native1899/5352.516另列不重置。加载/训练/loss/finalnative/save/CPU语义/cleanup全计费；CPUscore一次11.480739s、comparison .9521395s另列。峰allocated1793521152B/reserved2059403264B取restore与training/native最大。

资源/完整性：96frozen+snapshot和742旧保护全部原SHA；33预算checks及最终完整性审查通过，5mock+2entry检查均无真实GPU工作。storage当前{total}B+268435456预留={total+268435456}B<3221225472；不复制完整物理数据、不删旧权重、不下载/安装。物理新增0，监督39680/全native74240不变。无H4/全319/未见8/正式TRAIN/服务器/ZIP。

实际命令（cwd={REPO}）：
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage3c_run.py --execute（唯一模型执行，session25229 exit0）
原scripts/native_rgb_stage3_score.py score --train-npz stage3c/predictions/train18.npz --dev-npz stage3c/predictions/dev36.npz --train-windows stage3c/train18-windows.json --dev-windows stage3c/dev36-windows.json --output stage3c/score（实际完整路径与命令见REPORT和score timing，CPU一次exit0）
stage3c/cpu-score-comparison.py finish --stage stage3c（CPU exit0，原指标四方法216行+all18三页）

绝对证据目录：{HERE}
关键证据：REPORT.md、stage3c-result.json、DIAGNOSIS.md、model-work.json、restore-review.json、run/last.pt、run/history.jsonl、run/sampling-history.jsonl、source-identity.json、continuation-plan.json、entry-from-stage3b.diff、review-postrun-budget.json/md、review-postrun-integrity.json/md、score/score-report.json、score/per-window.csv、comparison129-final/comparison.json、all54-method-comparison.csv、train129-final-01/02/03.png、storage-at-callback.json、score-cpu-timing.json。旧129/frozen均引用stage3b且SHA不变。

请分别审查TRAIN微拟合通过、DEV仍差于hold、2推杆失败及形状缺陷，再明确是否需要下一步诊断或正式训练。本次授权只执行这一轮，回传后立即停止。
'''
(HERE/'callback-stage3c.txt').write_text(callback,encoding='utf-8');s['callback']['message_sha256']=sha(HERE/'callback-stage3c.txt');put(BASE/'stage-state.json',s)
print(json.dumps(dict(ready_to_report=True,result_sha256=s['stage3c_result_sha256'],storage_bytes=total,callback_characters=len(callback))))
