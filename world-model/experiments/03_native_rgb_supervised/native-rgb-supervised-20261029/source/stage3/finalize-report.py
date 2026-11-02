"""Report the single pretraining range abort; never load or rerun a model."""
import json
from datetime import datetime,timezone
from pathlib import Path
import sys

p=Path(__file__).resolve().parent;parent=p.parent;repo=p.parents[2]
sys.path.insert(0,str(repo))
from tdyssey.native_rgb_training import atomic_json,sha256
from tdyssey.native_rgb_stage_budget import StorageBudget,native_storage_paths
from scripts.native_rgb_stage3_run import verify_frozen

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

verify_frozen(p);work=read(p/'model-work.json');ledger=work['budget']['ledger'];score=read(p/'score/score-report.json')
review=read(p/'review-postrun.json');ranges=read(p/'native-output-range-diagnosis.json');archive=read(p/'checkpoint-archive-audit.json')
cfg=read(repo/'configs/native_rgb_supervised_stage3.json');windows=read(p/'frozen-windows.json');target=score['train']['dynamic_fit_target']
assert review['integrity_and_protocol_audit_passed'] and ledger['active_stage'] is None
assert ledger['charged_updates']==ledger['completed_updates']==0 and ledger['charged_seconds']<=600
assert work['failure']=='FloatingPointError: Invalid native RGB retained without repair'
assert ranges['rows'][0]['below_zero']==6 and not target['passed']
assert archive['zip_CRC_passed'] and archive['all_required_keys_present'] and archive['torch_load_calls']==0
assert score['train']['denominator']==18 and score['development']['denominator']==36
storage=StorageBudget(native_storage_paths(repo,parent,extra_paths=[cfg['gpc_source_root'],cfg['physics_source_root'],cfg['init_checkpoint']]),minimum_free_bytes=512*1024**2)
footprint=storage.check(force=True)
evidence=['PARENT_REVIEW.md','source-identity.json','frozen-windows.json','train18-windows.json','dev36-windows.json','paired-batches.json',
    'original-assets-CPU-identity.json','calibration-check/calibration-check.json','focused-checks.json','review-prelaunch.json',
    'review-prepared-inputs.json','review-source-proof.json','review-postrun.json','review-budget-postrun.json','model-work.json',
    'model-work-error.json','model-work-command.log','run/error.json','run/last.pt','native-output-range-diagnosis.json',
    'checkpoint-archive-audit.json','predictions/frozen-train18.npz','predictions/frozen-train18.json',
    'predictions/train18.npz','predictions/dev36.npz','score/score-report.json','score/per-window.csv','score/files.json',
    'DIAGNOSIS.md','score-execution.json','PROPOSED_NOT_APPLIED-finite-range-policy.diff','proposed-followup-policy.json']
result=dict(kind='native-rgb-stage3-bounded-attempt-result.v1',stage='3',created_at_utc=datetime.now(timezone.utc).isoformat(),
    outcome='stopped_before_training_by_entry_finite_RGB_range_guard',execution_cycle_finished=True,microfit_completed=False,
    error=work['failure'],native_sampler_returned_first_batch=True,trigger_component='stage3 entry output-range guard, not optimizer/backward',
    optimizer_attempts=0,optimizer_updates=0,training_draws=0,unique_training_windows_selected=18,development_windows_frozen=36,
    model_work_seconds=ledger['charged_seconds'],model_work_clock_closed=True,model_work_overrun_seconds=work['budget']['overrun_seconds'],
    original_initialization_sha256='4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892',
    original_parameter_count_runtime_assertion_passed=8729827,all_requires_grad_runtime_assertion_passed=True,
    gradient_reachability_measured=False,parameter_change_measurement_completed=False,loss_before=None,loss_after=None,peak_CUDA_memory_recorded=False,
    frozen_prediction_frames_generated=4,frozen_prediction_batch_calls=1,planned_frozen_frames=54,fitted_prediction_frames_generated=0,
    first_native_batch=ranges['rows'][:4],actual_range_failure_values=6,minimum_native_RGB=ranges['rows'][0]['min'],
    missing_frozen_train_windows=14,missing_frozen_dev_windows=36,missing_fitted_train_windows=18,missing_fitted_dev_windows=36,
    train_dynamic_fit_target=target,target_interpretation='Not demonstrated: no optimizer work or fitted predictions. Penalty means are missingness accounting, not measured fitted performance.',
    shape_interpretation='Fitted shape unavailable. Three range-valid original baseline frames show red output; empty_object refers to gray-palette mask, not necessarily a blank RGB image.',
    checkpoint=dict(**archive,updates=0,role='complete t0 pre-baseline archive',semantic_model_restore_tested_in_this_run=False),
    integrity_and_protocol_audit_passed=True,physical_controls=0,supervised_physical_controls=39680,all_native_physical_controls=74240,
    historical_native_optimizer_attempts=1899,historical_native_charged_seconds=5352.516,
    aggregate_new_round_attempts=0,aggregate_new_round_model_seconds=ledger['charged_seconds'],
    remaining_new_round_attempts=3000,remaining_new_round_seconds=5400-ledger['charged_seconds'],remaining_is_not_retry_authorization=True,
    no_unseen=True,no_H4=True,no_full319=True,no_server=True,no_packaging=True,one_actual_execution=True,raw_outputs_unchanged=True,
    proposed_followup_applied=False,next_action='Actualparentcallbackthenwaiting_parent;no model retry or changed threshold without new release',
    storage_before_final_report=footprint,
    evidence={name:dict(path=str((p/name).resolve()),sha256=sha256(p/name)) for name in evidence})
lines=['# Stage3: stopped before training at the first native baseline batch','',
    '**No microfit was performed.** The single authorized run stopped after28.344 seconds and0 optimizer attempts/updates when the entry rejected one finite, out-of-range original GPC RGB output. This is an initialization/baseline attempt with missing training and development measurements; it is not a200-step fitting failure or evidence against GPC capability.','',
    'The native sampler returned its first4 TRAIN images normally. The entry range guard found6 negative values in A (`train-2b-018:11`), minimum−0.0196077823638916 and maximum1.0. This is larger than a small rounding discrepancy. The raw array/status remains unchanged; no clipping, shape repair, threshold change or model rerun was performed.','',
    '|Measurement|Planned windows|Actually generated|Missing|','|---|---:|---:|---:|',
    '|Original frozen TRAIN|18|4|14|','|Original frozen development|36|0|36|','|Fitted TRAIN|18|0|18|','|Fitted development|36|0|36|','',
    'The other3 first-batch images pass finite/range checks only. B and C have gray-object IoU approximately0.003215/0.029126 under the pre-frozen gray palette. E has no detectable gray-object or blue-pusher mask. B/C/E show red foreground in the unmodified RGB; an empty gray mask does not mean the entire image is blank. These are4 partial original-baseline outputs, not fitted-model or generalization results.','',
    'All18/36 original denominators remain in [per-window scores](score/per-window.csv), the raw [TRAIN array](predictions/train18.npz), [development array](predictions/dev36.npz) and9 gallery pages. Missing frames retain NaN/status records. The fitted TRAIN mean object error1.0 and0/18 wins are the preregistered missingness penalties; no fitted frame was generated. The hold mean object error is0.2325831710. The20%/12-of18 diagnostic is not demonstrated, and penalty means must not be read as observed fitted performance.','',
    '|Question|Evidence from this run|','|---|---|',
    '|Initialization and input/native forward connection|Original published SHA and20 source files verified; full8,729,827 count/all-trainable assertions passed; first native4-image batch returned. Original RGB/action timing and batch-paired noise were frozen.|',
    '|Backward/optimizer training chain|Not exercised:0 attempts,0 draws,0 losses,0 gradient measurements,0 scheduler/optimizer steps. CPU tests do not replace actual training evidence.|',
    '|Motion fitted on TRAIN18|Unmeasured; training never began.|','|Independent DEV36|Both original/fitted model inference unmeasured; no generalization conclusion.|',
    '|Fitted shape quality|Unavailable. Fitted18 shape assessments are unknown, not shape passes. Partial original red appearance/gray-mask failures are shown.|','',
    'Before model work, one visible-motion+contact H4-legal window per seen training letter was frozen by the parent-specified SHA256 rule/seed2026102904. Every selected TRAIN target differs from its last observation. The original stage2C development36 and14 batch plans were retained, with same ordered batch members/shapes/seeds across planned frozen/fitted phases. No validation or old probe entered training. [Frozen windows](frozen-windows.json), [batch plan](paired-batches.json).','',
    'The only checkpoint is [run/last.pt](run/last.pt), a35,096,361-byte complete t0 pre-baseline archive. It contains model, empty-initial optimizer state, scaler, scheduler, RNG, WindowOrder, progress, config, identity and ledger snapshot. SHA is '+archive['sha256']+'. ZIP CRC and required pickle-key strings were checked without loading or restoring a model after the stop. Actual tensor-by-tensor restoration and optimizer gradient reachability were not tested in this run. No best.pt was created; old experiments/checkpoints remain unchanged. [Archive audit](checkpoint-archive-audit.json).','',
    'The single model-work clock includes CUDA query, original loading, initialization, t0 atomic save, first native batch, failed callback cleanup and process cleanup:28.344 seconds. Ledger is closed,0 reserved/completed attempts. Full new-round ledger caps remain3000 attempts/5400 seconds; numeric remaining budget is3000/5371.656, which is not permission to retry. Old native1899 attempts/5352.516 seconds are separate and retained. Actual peak GPU memory was not written before the early error, so no historical peak is substituted.','',
    'The update cap200, latest new-update cutoff420 seconds and tail reserve≥180 seconds were installed but never reached. Operation guards and conservative time reserves do not promise hard real-time preemption of an individual CUDA or atomic-save operation; actual closed elapsed time is below600 seconds. Combined old/new/source/cache accounting and atomic temporary reserve remain below3GiB.','',
    'Model/loss/optimizer/sampler mathematics remained original; the small entry/callback/accounting changes were frozen before model work.34 focused CPU tests,18-seen renderer-only palette checks and independent input/source audits passed. Independent final integrity review passed23 checks; these checks establish provenance/accounting, not microfit success.56 frozen files and381 protected data/index/weight files are unchanged.','',
    'A reviewable [unapplied follow-up diff](PROPOSED_NOT_APPLIED-finite-range-policy.diff) would retain finite out-of-range outputs as scored generation failures while allowing the remaining baseline/fit to proceed; nonfinite/structural/OOM failures would still stop. It changes neither RGB values nor score penalties, but it changes this entry abort policy and requires a new parent release. It has not been applied or run.','',
    'Pure offline scoring took6.235 seconds outside the model clock. [Diagnosis](DIAGNOSIS.md), [score report](score/SCORE_REPORT.md), [first TRAIN gallery](score/gallery-train/train-01.png), [all scored files](score/files.json), [independent review](review-postrun.md), and [budget review](review-budget-postrun.md) retain all failures and missingness. Physical controls remain39,680 supervised/74,240 all-native; this stage adds0. No full319/H4/unseen evaluation, server run or packaging occurred.','',
    'The immediate next action is an actual failure/result callback to the parent chat, followed by waiting_parent. No second model execution is authorized.']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');atomic_json(p/'stage3-result.json',result)
state=read(parent/'stage-state.json');state.update(status='ready_to_report',stage3_execution_cycle_complete=True,stage3_training_completed=False,
    stage3_outcome=result['outcome'],stage3_result_file='stage3/stage3-result.json',stage3_result_sha256=sha256(p/'stage3-result.json'),
    next_action='Actualparentfailurecallbackwithrawranges/missingness/t0checkpoint/accounting;recordreceiptthenSTOP.No retry.',parent_review_received=False)
state['resources'].update(stage3_optimizer_attempts=0,stage3_completed_optimizer_updates=0,stage3_model_work_seconds=ledger['charged_seconds'],
    stage3_native_prediction_frames=4,stage3_model_inference_calls=1,optimizer_attempts=0,model_inference_calls=1)
state['stage3_remaining_new_round_budget']=dict(attempts=3000,seconds=5400-ledger['charged_seconds'],not_authorization_to_retry=True)
atomic_json(parent/'stage-state.json',state)
body=['[阶段3异常/结果回传] 唯一实际尝试已在训练前停止，未重跑。28.344秒计费、0优化器尝试/0更新，原始输出与t0完整检查点已保留；请父审查后另行决定后续口径。本实施本次回传后waiting_parent并结束。','',
    '关键事实：首次frozen TRAIN批次4张原生sampler正常返回；入口预注册的strict RGB[0,1]guard在A(train-2b-018:11)发现6个负值，min=-0.0196077823638916、max=1，抛FloatingPointError。不是epsilon舍入，也不是optimizer/backward或OOM。冻结入口把这类finite范围失败提升为abort，导致没有启动微拟合；我们未在看过结果后改阈值/clip/填补/再次运行。','',
    '实际覆盖：frozen TRAIN4/18生成、14未尝试；frozen DEV0/36；fitted TRAIN0/18、DEV0/36。全部18/36原分母、缺测NaN/status和rawphase数组保留。B/C/E的其余3张只是range-valid；冻结评分B/C灰物体IoU约.003215/.029126，E无gray/blue mask。RGB中有红色前景，empty_object指冻结灰palette检测为空，不等于整张白图。不能从3张finite样本均值声称模型或拟合合格。','',
    '目标结论：TRAIN20%误差下降+12/18胜hold未证明，因为0训练和0fitted预测。score fittedmeanerror=1、0/18wins是预注册缺测惩罚，并非实测拟合性能；holdmeanerror=.23258317101512865。全18拟合shapeassessable0/unknown18；DEV所有模型推理未测，31moving/4static/1I分组及全部54行/9图库页仍在。不能称“200步未拟合成功”、泛化失败或GPC路线无效。','',
    '链路分开报告：strict原发布权重SHA4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892、20源文件、8,729,827全requires_grad断言/首4原生forward有实际证据；AMP后向、所有梯度可达、参数变化、optimizer/scheduler更新、loss前后均未执行/未验证，不以34CPU测试替代。实际峰值显存未在早停前落盘，不拿旧1.93GiB冒充本次。','',
    '预注册与输入均完整：18TRAIN各字母minSHA可见运动+contact窗口、a3..60/7valid、像素位移>=1或角>=3°，seed2026102904；支持35–321/letter，18独立轨迹。DEV原36顺序/来源不变；TRAIN4,4,4,4,2和DEV9×4/配对seed固定。history=item.image[:4]、last=image[3]、true=image[4]；sampler用原绝对动作arrays[a-3:a+1]后由原wrapper归一化，训练batch仅image/action。原loss/AdamW/AMP/clip/scheduler数学未动，T_max3000；last_only无best/linspace混入。','',
    '账本/检查点：唯一new-round stage已closed，0reserved/completed，28.344s覆盖加载、CUDA、t0保存、首推理、异常/cleanup。new aggregate3000/5400仍保持，余额3000/5371.656仅数值余额不是重试授权；旧1899/5352.516独立不重置。仅last.pt=t0完整点，35,096,361B SHA'+archive['sha256']+'，10字段模型/optimizer/scaler/scheduler/RNG/WindowOrder/progress/config/identity/ledger齐，ZIP CRC/key-string审计通过；没有stop后torch.load或语义restore，不冒称本轮做过实际恢复验证。','',
    f"资源/完整性：56冻结文件/381保护文件/旧weights全部原SHA，独立23完整性检查通过；CPUscore6.235s单列。全old/new/native source/cache当前{footprint['bytes']}B，加256MiB原子预留projected={footprint['projected_bytes']}B<3GiB。新物理0、监督39680/总74240不变。无H4/全319/未见/服务器/ZIP。",'',
    '本轮没有应用后续修改。已给出unapplied小diff供你具体审查：仅让finite out-of-range baseline/fitted RGB继续作为生成失败保留并完整评分，而不abort；nonfinite/结构/OOM仍stop，绝不clip或改变失败惩罚/训练loss/样本。若你认为应这样继续，需要新的明确阶段放行，保留本次28.344秒费用；本实施不会自行重跑。','',
    '实际命令（cwd=__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main；复用现有Python，无下载/安装）：',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage3_prepare.py --physics-root ../Tdyssey/third_party/diffusion_policy',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage3_run.py --execute（唯一模型执行，exit1，GPU0/SDLdummy/HFoffline）',
    '../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage3_score.py score --train-npz artifacts/native-rgb-supervised-20261029/stage3/predictions/train18.npz --dev-npz artifacts/native-rgb-supervised-20261029/stage3/predictions/dev36.npz --train-windows artifacts/native-rgb-supervised-20261029/stage3/train18-windows.json --dev-windows artifacts/native-rgb-supervised-20261029/stage3/dev36-windows.json --output artifacts/native-rgb-supervised-20261029/stage3/score（纯CPU，exit0）','',
    '绝对证据路径：']
for name in ['REPORT.md','stage3-result.json','DIAGNOSIS.md','model-work.json','run/error.json','run/last.pt','native-output-range-diagnosis.json',
    'checkpoint-archive-audit.json','frozen-windows.json','paired-batches.json','predictions/frozen-train18.npz','predictions/train18.npz','predictions/dev36.npz',
    'score/score-report.json','score/per-window.csv','score/gallery-train/train-01.png','review-postrun.json','review-budget-postrun.json',
    'source-identity.json','focused-checks.json','PROPOSED_NOT_APPLIED-finite-range-policy.diff','proposed-followup-policy.json']:
    body.append(str((p/name).resolve()))
body+=['','请审查入口abort政策及这次早停的未完成项，再明确下一步。本次只归档回传，停止等待，不进入正式训练。']
(p/'callback-stage3.txt').write_text('\n'.join(body)+'\n',encoding='utf-8')
print(json.dumps(dict(status='ready_to_report',outcome=result['outcome'],model_work_seconds=ledger['charged_seconds'],attempts=0,result_sha256=sha256(p/'stage3-result.json'),checkpoint_sha256=archive['sha256'],storage_bytes=footprint['bytes'],callback_chars=len('\n'.join(body)))))
