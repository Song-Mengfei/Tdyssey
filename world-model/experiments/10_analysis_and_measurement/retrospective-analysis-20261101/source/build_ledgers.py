from pathlib import Path
import json,datetime
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'artifacts'
def save(n,j):(O/n).write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
sources={
'E01':R/'AGENTS.md','E02':R/'docs/PROJECT.md','E03':R/'docs/NATIVE_RGB_WORLD_MODEL.md','E04':R/'docs/WORLD_MODEL.md',
'E05':O/'HUMAN_HISTORY_EXTENDED.json','E06':A/'native-rgb-20261028/REPORT.md','E07':A/'native-rgb-supervised-20261029/REPORT.md',
'E08':A/'native-rgb-supervised-20261029/stage3b/REPORT.md','E09':A/'native-rgb-supervised-20261029/stage3c/REPORT.md','E10':A/'native-rgb-supervised-20261029/r1/REPORT.md',
'E11':A/'native-rgb-supervised-20261029/r2a/REPORT.md','E12':A/'native-rgb-supervised-20261029/r2a2/REPORT.md','E13':A/'native-rgb-supervised-20261029/r2b/REPORT.md',
'E14':A/'native-rgb-supervised-20261029/r2c0/REPORT.md','E15':A/'native-rgb-supervised-20261029/r2c1/REPORT.md','E16':A/'native-rgb-supervised-20261029/r2c1r1/REPORT.md',
'E17':A/'native-rgb-supervised-20261029/r2c2/REPORT.md','E18':A/'native-rgb-supervised-20261029/r2c3/REPORT.md','E19':A/'native-rgb-supervised-20261029/r3a/REPORT.md',
'E20':A/'native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/review/OWNER-REVIEW.md','E21':A/'native-rgb-supervised-20261029/r3b/FAILURE-REPORT.md',
'E22':A/'original-physics-contact-pilot-20261030/STRICT_ORIGINAL_CORRECTION-20261031.txt','E23':A/'original-physics-contact-pilot-20261030/summary.json',
'E24':A/'strict-original-contact-pilot-20261031/CALIBRATION_V2_REVIEW.md','E25':A/'direct-target-acceptance-20261031/ACCEPTANCE_REVIEW.md',
'E26':A/'direct-target-diagnosis-review-20261031/DIAGNOSIS_AND_REVISION.txt','E27':A/'direct-target-four-case-diagnostic-20261031/REPORT.md',
'E28':A/'accelerated-rgb-run-20261031/FINAL_REVIEW.md','E29':A/'accelerated-rgb-run-20261031/selection/summary.json','E30':A/'accelerated-rgb-run-20261031/training/source-recipe.json',
'E31':A/'oc-continuation-20261031/FINAL_REVIEW.md','E32':A/'generated-history-ab-20261101/FINAL_REVIEW.md','E33':A/'generated-history-ab-20261101/training_core.py',
'E34':A/'region-weighted-cd-20261101/FINAL_REVIEW.md','E35':A/'region-weighted-cd-20261101/training_core.py','E36':A/'region-weighted-cd-20261101/masks.py',
'E37':A/'region-weighted-cd-20261101/diagnosis/REPORT.md','E38':A/'generated-history-next-step-review-20261101/O_DIRECTION_ANALYSIS.json',
'E39':A/'saved-evaluation-recheck-20261101/REPORT.txt','E40':A/'high-motion-fit-20261101/PLAN.txt','E41':A/'high-motion-fit-20261101/selection/summary.json',
'E42':A/'high-motion-fit-20261101/fit-standard.json','E43':O/'CURRENT_SNAPSHOT.json','E44':O/'INDEPENDENT_DATA_AUDIT.json',
'E45':O/'INDEPENDENT_METRIC_AUDIT.json','E46':O/'INDEPENDENT_RAW_AUDIT.json','E47':O/'DISK_INVENTORY.json','E48':O/'DELETED_CHECKPOINTS.json',
'E49':R.parent/'tmp/pushing-data-review-20261030/audit.json','E50':R.parent/'tmp/pushing-data-review-20261030/physics-contact-audit/summary.json',
'E51':R.parent/'tmp/pushing-data-review-20261030/physics-contact-audit/final-candidate-summary.json','E52':R.parent/'tmp/pushing-data-review-20261030/physics-contact-audit/audit-receipt.json',
'E53':R.parent/'Tdyssey/tdyssey_pushing_data/tdyssey_pushing_data/README.md','E54':R/'tdyssey/gpc_world_model.py','E55':R/'tdyssey/native_rgb_r3a_conditioning.py',
'E56':R.parent/'Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/denoiser.py','E57':R.parent/'Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/inner_model.py',
'E58':R.parent/'Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/diffusion_sampler.py','E59':R.parent/'Tdyssey/third_party/gpc_code/world_model_train_phase_two/utils.py',
'E60':R/'tdyssey/direct_target_collection.py','E61':A/'accelerated-rgb-run-20261031/train_eval.py','E62':A/'accelerated-rgb-run-20261031/selection.py',
'E63':A/'pipeline-config-review-20261031/parts/TRAINING_REVIEW.txt','E64':A/'high-motion-fit-20261101/GT-calibration-correction-receipt.json',
'E65':A/'original-physics-contact-pilot-20261030/cleanup/checkpoint-deletion-receipt.json','E66':A/'high-motion-fit-proposal-20261101/USER_AUTHORIZATION.json',
'E67':R/'tdyssey/native_rgb_evaluation.py','E68':R/'tdyssey/native_rgb_original_physics.py',
'E69':A/'native-rgb-supervised-20261029/contact-pilot1-20261030/REPORT.md','E70':A/'native-rgb-supervised-20261029/contact-pilot2-20261030/REPORT.md',
'E71':A/'strict-original-contact-pilot-20261031/CALIBRATION_V1_REVIEW.json',
'E72':A/'high-motion-fit-20261101/FINAL_REVIEW.md',
'E73':O/'hmf-final-review-20261102/REPORT.txt',
'E74':O/'hmf-final-review-20261102/CURVES.json',
'E75':O/'hmf-final-review-20261102/FINAL_STATE.json',
'E76':O/'hmf-final-review-20261102/POSE_EXPOSURES.json',
'E77':O/'hmf-final-review-20261102/SAVED_ONLY_AUDIT.json',
'E78':A/'high-motion-fit-20261101/postrun-fit-diagnostic-audit.json',
'E79':O/'hmf-final-review-20261102/O_SAVED_AXIS_CHECK.json',
}
save('SOURCES.json',{k:dict(path=str(v),exists=v.exists())for k,v in sources.items()})
data=[]
def d(id,name,letters,domain,counts,use,evidence,note=''):
 data.append(dict(id=id,name=name,letters=letters,physics_domain=domain,counts=counts,actual_model_use=use,evidence=evidence,note=note))
d('D00','已取消结构化刚体实验','原26字母范围；实际细分未能复核','旧结构化/非当前RGB路线',{'early_stops_recorded_in_chat':200},'已删除；只剩历史聊天信息',['E05'],'261更新/保留250等仅历史消息，原REPORT路径已缺失，不冒充本次独立复核。')
d('D01','native RGB首轮','ABCEFHIJLMOPSTUVXY','finite-force dynamic pusher; damping0',{'trajectories':518,'controls':33144,'train_controls':26232,'seenval_controls':6912,'probe_controls_separate':1416},'native1668/best1500；不属于后续6000谱系',['E06'])
d('D02','supervised阶段1','TLOCXB','finite-force dynamic/damping0',{'trajectories':54,'controls':3456,'H1':3294},'采样探针，未混入正式TRAIN396',['E07'])
d('D03','阶段1B重接触配对','TLOCXB代表；X重点','同D02',{'runs':8,'new_families':2,'paired_old_families':6,'controls':512,'H1':488},'采样对照；未混入正式TRAIN396',['E07'],'六配对重放不是六个新家族。')
d('D04','阶段2A+2B正式训练','ABCEFHIJLMOPSTUVXY','同D02',{'train_families':396,'controls':25344,'unique_H1':24156,'H4':22968,'stage2a_runs':162,'stage2b_runs':234},'250微拟合仅取18窗；stage4a只准备未正式训练',['E10','E08','E09'])
d('D05','阶段2C seen-val','ABCEFHIJLMOPSTUVXY','同D02',{'families':162,'controls':10368,'H1':9882,'H4':9396,'selected_diagnostic_windows':319,'DEV_windows':36,'DEV_families':32},'DEV36评过；319未作为全部已评测计数',['E09','E10'])
d('D06','R2A/R2A2阻尼配对','TLOCXB代表','dynamic pusher D0/D1/0.1s/0.3s half-life',{'independent_initial_families':16,'arms':4,'actual_runs':128,'controls':1024,'H1':640,'H4':256},'只做物理/采样诊断，无训练',['E11','E12'],'R2A2引用旧两臂，不将旧视频再计成新物理。')
d('D07','R2B小库','TLOCXB','dynamic pusher/0.1s half-life',{'planned':54,'actual_trajectories':53,'controls':1272,'train_families':41,'seenval_families':12,'train_unique_H1':861,'seenval_H1':252},'R2C1R1的300及R2C3至6000；反复曝光96000次',['E13','E14','E16','E18'])
d('D08','R3A独立新库','TLOCXB','dynamic pusher/0.1s half-life',{'attempts':918,'success':912,'failures_retained':6,'train_families':768,'seenval_families':144,'successful_controls':43776,'completed_controls_including_partials':43886,'charged_controls':43891,'known_substeps':438889,'legal_train_H1':34560,'selected_unique_H1':23984},'R3B从6000继续1499更新到7499；不是AR1的新物理数据',['E19','E20'])
d('D09','CONTACT-PILOT1/2','TLOCXB','旧dynamic/half-life，安全/采样改动各自独立',{'CP1_runs':48,'CP1_complete':47,'CP1_completed_controls':2261,'CP1_charged':2262,'CP2_runs':60,'CP2_new_families':12,'CP2_controls':2880},'未并入R3B/AR1',['E69','E70'],'CP2含48个旧场景重放；不当60新家族。')
d('D10','v1-v6物理原型与回归','TLOCXB','改积分/求解/速度约束；非严格原协议',{'regression_v1_to4_runs_each':96,'regression_completed_controls':[2277,2859,2686,2880],'calibration_v2_controls':1152,'calibration_v2_substeps':230400},'全部诊断隔离，未进当前训练',['E22','E23'],'回归有共享场景与版本差异，不相加声称独立新数据。')
d('D11','严格原物理cal1/cal2','TLOCXB','100Hz/10/KINEMATIC/damping0',{'attempts':36,'controls':2304,'substeps':23040,'per_phase_attempts':18},'采样质量未通过；未训练',['E24','E71'])
d('D12','直接目标120验收','TLOCXB','原物理/绝对0..512；旧>1worldpx早停观察器',{'attempts':120,'families':96,'complete':46,'early_stop':74,'actual_controls':3446,'legal_controls':3372,'substeps':33992,'H1':3012,'H4':2808,'H8':2566,'H16':2117,'H16_families':45},'诊断/未并入AR1训练',['E25','E26'])
d('D13','四案例超过1后的延续','O/T/L/C预选','原物理；仅诊断观察器改变',{'cases':4,'actual_controls':33,'max_controls':35,'substeps':321},'training_eligible=false',['E27'],'证明存在不同恢复过程，不能据4例估计74例误拦率。')
d('D14','AR1原物理六字母新库','TLOCXB','100Hz/10/KINEMATIC/damping0/原PD，RGB10Hz',{'attempts':1536,'controls':45969,'substeps':459213,'clean_prefix_controls':41882,'selected_train_families':1339,'unique_train_H1':30935,'candidate_val_H16_windows':285,'candidate_val_families':163,'formal_val_families':48,'H1_allclean':37274,'H4_allclean':32682,'H8_allclean':26663,'H16_allclean':16187},'AR1/E1、OC E2/E3；后续ABCD和HMF均取其子集',['E28','E29','E30','E44'])
d('D15','ABCD四步段训练视图','TLOCXB','完全引用D14',{'segments':7568,'families':1313,'unique_targets':30272,'excluded_short_tail_H1':663,'new_physical_controls':0},'A/B/C/D各独立一遍30272目标',['E32','E34','E44'],'这是同一批数据的四次独立实验，不是四套新数据。')
d('D16','HMF高运动集中拟合视图','OCB','完全引用D14',{'train_families':48,'seenval_families':24,'old_failure_families':3,'distinct_all_families':75,'unique_train_targets':768,'strong_unique_targets':444,'max_planned_exposures':48000},'进行中；截至23:00快照64更新/1024曝光',['E40','E41','E43','E44'],'训练集拟合诊断；强曝光计划89.15%，非泛化实验。')
d('D17','队友原包','T/T_narrow/Y/D/L；候选仅T/L/Y','原official KINEMATIC/damping0；接触标签原含非AO',{'episodes_all':2500,'transitions_all':681260,'TLY_episodes':1500,'TLY_transitions':416000,'TLY_actual_AO_controls':116951,'provided_positive_without_AO':13993,'screened_H1_candidates':18610,'H4_context_pass':16782,'candidate_trajectories':1379,'candidate_scene_families':498},'本次查到的谱系未并入；仅接入审查',['E49','E50','E51','E52','E53'],'scene seed跨字母共享，按scene而非frame拆分；历史审查发生过CPU回放，本复盘没有回放。')
storage={
 'D00':'Retired/deleted structured assets; original interface not reconstructed.',
 'D01':'Compact state/action/contact NPZ; RGB via bound renderer on demand.',
 'D07':'Stored states/actions/RGB uint8 with partial-attempt arrays.',
 'D08':'State/action/substep telemetry NPZ and RGB frame hashes; RGB rendered on demand with bound original renderer.',
 'D14':'Stored raw RGB uint8 CHW96, states15, requested/passed actions, control validity and compressed telemetry.',
 'D15':'Existing uint8 mmap [7568,8,3,96,96] plus normalized float32 actions [7568,7,2]; references D14.',
 'D16':'References 75 original D14 NPZs and selected indices; no copied data library.',
 'D17':'Frames NPZ NHWC96 uint8 aligned with states9/actions/contact/mask files.'}
for row in data:
 row['storage_and_image_provenance']=storage.get(row['id'],'Version-specific raw/metadata/telemetry retained; see linked report. Diagnostic views are not automatically training datasets.')
 row['model_interface']=('Retired structured path, not the current RGB-only interface.'if row['id']=='D00'else
     'When admitted to native GPC: four history RGB CHW96 float32/255 plus four aligned absolute XY actions float32/255.5-1, next RGB target. State/contact/letter labels used only for selection/audit, not network inputs. Diagnostic-only data rows are not admitted by this statement.')
terminal_path=O/'hmf-final-review-20261102/FINAL_STATE.json'
terminal=json.loads(terminal_path.read_text(encoding='utf-8'))if terminal_path.exists()else None
if terminal:
 row=next(x for x in data if x['id']=='D16')
 row['actual_model_use']='HMF completed3000successfulupdates/48000exposures/AMP0; fulljointfit notpassed. Closed2026-11-01 23:53:45SG, waiting_user_review.'
 row['counts'].update(actual_exposures=48000,actual_strong_exposures=42790,repeated_exposures=47232,successful_updates=3000)
 row['evidence']+=['E72','E75','E77']
save('DATA_LEDGER.json',dict(schema='retrospective-data-ledger.v1',units='controls at10Hz unless stated; pixels image96 unless world explicitly; overlapping windows are not trajectories',rows=data))

ex=[]
def e(id,parent,data,change,updates,exposures,seconds,result,evidence,**other):
 ex.append(dict(id=id,parent=parent,data_ids=data,controlled_change=change,successful_updates=updates,supervised_exposures=exposures,wall_or_model_seconds=seconds,result=result,evidence=evidence,**other))
e('RIGID','旧结构化起点',['D00'],'结构化路线，已撤销',261,None,None,'未通过，数字仅历史消息',['E05'],verification='historical_chat_only')
e('NATIVE1668','published GPC',['D01'],'全参数native RGB/残差MSE',1668,None,5352.516,'未胜copy-last；best1500/last1668已删除',['E06'],optimizer_attempts_including_replay=1899)
e('STAGE3','published GPC',['D04'],'微拟合入口',0,0,28.344,'首baseline负值守卫中断，不能说200步拟合失败',['E08'])
e('MICRO129','published GPC',['D04'],'18窗微拟合',129,2064,416.281,'未通过18窗拟合门槛',['E08'])
e('MICRO250','MICRO129 complete state',['D04'],'同18窗继续121更新',121,1936,423.859,'train18旧门槛通过，DEV/刚性失败；独立于6000线',['E09'])
e('R2C1_FAIL','published GPC',['D07'],'新入口日志参数冲突',0,0,27.359,'保存t0后停；无训练/推理',['E15'])
e('R2C300','published GPC fresh optimizer',['D07'],'80%明显运动曝光',300,4800,486.375,'motionH4 IoU .531987，近静止漂移/形状失败',['E16'],AMP_skips=0)
e('R2C_SAMPLE','fixed R2C300',['D07'],'denoise3/8/16',0,0,86.907,'motionH4 IoU .531987/.504268/.489620',['E17'])
e('R2C6000','R2C300 complete state',['D07'],'延长训练至6000',5700,91200,6876.985,'motionH4 .531987→.635105；棱角仍未恢复',['E18'],AMP_skips=0,lineage_end=6000)
e('R3A','无模型',['D08'],'独立采集；50/50 clean-weak接口准备',0,0,None,'912成功；选23984目标',['E19'])
e('R3B_FAIL','6000 complete state',['D08'],'后台加载器启动失败',0,0,33.047,'正式cursor0，未训练；后经授权修复',['E21'])
e('R3B7499','6000 complete state',['D08'],'新数据+cleanweak；一遍23984目标',1499,23984,1469.752,'H4均值改善，运动保形未通过',['E20'],AMP_skips=0,lineage_end=7499,formal_training_loop_seconds=1205.891)
e('AR1_E1','7499 full model/Adam/scaler/RNG',['D14'],'跨物理域新数据一遍，LR1e-5',1934,30935,6452.18,'全48 H16 IoU .497→.597；强OC失败',['E28','E30'],AMP_skips=0,lineage_end=9433,collection_wall_seconds=1625.66,training_side_seconds_including_baseline=4326.94)
e('OC_E2','AR1_E1 full state',['D14'],'同数据再完整一遍',1934,30935,15146,'全48 H16 IoU .6306；强OC .4039',['E31'],AMP_skips=0,lineage_end=11367)
e('OC_E3','OC_E2 full state',['D14'],'同数据再完整一遍',1934,30935,13859.9,'全48 H16 IoU .6226；强OC .3316，退步',['E31'],AMP_skips=0,lineage_end=13301)
e('OC_DENOISE16','fixed AR1_E1',['D14'],'仅强OC8例denoise3→16',0,0,None,'H16 .3472→.2839；有限配对限制',['E31'])
for arm in 'ABCD':
 change={'A':'4步分组但全部真实历史，原残差MSE','B':'h2起自身生成history/detach，原残差MSE','C':'全部真实历史；.50全图+.25区域+.25边缘残差','D':'同C loss，每batch2真2生成，62.5%目标全真上下文'}[arm]
 sec={'A':1291.207,'B':1593.091,'C':1178.230,'D':1895.643}[arm]
 result={'A':'正式48 H16 .641489；O不转/短暂反向，B丢孔','B':'正式48 H16 .686780，但运动H1/落点变差','C':'运动门槛三项过，O方向/B孔未过；总体.640077','D':'运动门槛三项失败，O/B失败；总体.671314'}[arm]
 e(arm,'OC_E2 full model/Adam/scaler/RNG',['D15'],change,1892,30272,sec,result,['E32'if arm in 'AB'else'E34'],AMP_skips=0,lineage_end=13259,time_scope='training loop, not full experiment wall')
e('CD_DIAGNOSIS','E2/A/B fixed',['D14'],'同16个O/B案例真实历史单步768帧',0,0,240.078,'未发现核查范围内接口错位；B RH保两孔、AR丢孔；O单步方向多unknown',['E37'],predicted_frames=768)
e('SAVED_IDENTITY_AUDIT','A/B/C/D',['D15'],'CPU保存结果与checkpoint身份核验',0,0,None,'确实更新/新输出，不能证明改善',['E39'])
snap=json.loads((O/'CURRENT_SNAPSHOT.json').read_text());pr=snap['files']['training/progress.json']
if terminal:
 e('HMF3000_COMPLETE','OC_E2 full state',['D16'],'单分支高运动曝光，沿用A配方；样本范围/曝光比例/训练量同时改变',3000,48000,terminal['timings']['training_loop_wall_seconds'],'单步与训练AR明显改善，完整联合拟合未通过；验证C及旧3例退步',['E40','E72','E73','E74','E75'],AMP_skips=0,lineage_end=14367,time_scope='training loop; worker total4123.711s',completed_singapore='2026-11-01 23:53:45',historical_snapshot_retained=str(O/'CURRENT_SNAPSHOT.json'),automatic_next_experiment=False)
else:
 e('HMF_IN_PROGRESS','OC_E2 full state',['D16'],'单分支高运动曝光，沿用A配方',pr['successful_updates'],pr['actual_exposures'],pr['timings']['training_loop_wall_seconds'],'未完成，不能提前判定',['E40','E43'],AMP_skips=pr['AMP_skips'],snapshot_singapore=snap['read_at_singapore'],latest_full_checkpoint_successful_updates=snap['files']['training/latest.json']['successful_updates'])
save('EXPERIMENT_LEDGER.json',dict(schema='retrospective-experiment-ledger.v1',caveat='Do not sum branches as one model; missing cost is null, not zero. Time scopes differ.',rows=ex))
print('sources',len(sources),'data_rows',len(data),'experiment_rows',len(ex))
