"""Read-only closed-run audit: hashes and saved arrays, no model or scoring."""
from pathlib import Path
import json,hashlib,sys,math
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.native_rgb_r2c0_evaluation import load_eval_cohort
from tdyssey.native_rgb_r2c0_evaluation import paired_seed,normalize_actions
from scripts.native_rgb_r2c1_report import metrics_input_identity

EXP=ROOT/'artifacts/native-rgb-supervised-20261029';OUT=EXP/'r2c2';R1=EXP/'r2c1r1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrsha(a):return hashlib.sha256(np.asarray(a).tobytes(order='C')).hexdigest()
checks=0;issues=[]
def check(v,label):
    global checks
    checks+=1
    if not v:issues.append(label)

config=read(ROOT/'configs/native_rgb_r2c2.json');work=read(OUT/'model-work.json')
assert work['closed'] is True,'Refuse hot audit'
check(work['status']=='complete' and work['failure'] is None,'run-complete-no-failure')
result=work['inference_result'];check(not result['cleanup_errors'],'cleanup')
check(result['optimizer_attempts']==result['completed_updates']==result['training_forward_calls']==result['new_checkpoints_saved']==0,'no-training-or-duplicate-checkpoint')
check(result['model_checkpoint_loads']==1,'one-checkpoint-load')
check(not list(OUT.rglob('*.pt')),'no-new-pt')
checkpoint=Path(config['checkpoint'])
check(checkpoint.stat().st_size==105300718 and sha(checkpoint)=='3c397035857fe98b896ea5083b52f32c129f1787c1be6884d4c740a9e58e5be0','unchanged-trusted-checkpoint')
for identity,base in [(read(R1/'source-identity.json'),R1),(read(OUT/'source-identity.json'),OUT)]:
    for row in identity['files']:
        check(sha(row['source'])==row['sha256'],f"active-source:{row['source']}")
        check(sha(base/row['snapshot'])==row['sha256'],f"snapshot:{row['snapshot']}")
oldprotected=read(OUT/'input-identities.json')['files']
for row in oldprotected:
    p=Path(row['path']);check(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],f'old-protected:{p}')
for p,digest in config['input_hashes'].items():check(sha(p)==digest,f'config-input:{p}')

prior=read(OUT/'prior-model-ledger.json');ledger=read(config['ledger_path']);stage=ledger['stages'][-1]
check(ledger['active_stage'] is None,'ledger-closed')
check(ledger['stages'][:-1]==prior['stages'],'prior-stage-history-preserved')
check(ledger.get('policy_migrations')==prior.get('policy_migrations'),'no-new-policy-migration')
check(ledger['charged_updates']==ledger['completed_updates']==550,'optimizer-cost-unchanged550')
check(stage['reserved_updates']==stage['completed_updates']==0,'stage-zero-updates')
check(ledger['max_seconds'] is None and stage['time_policy']=='record_only_v2','record-only-no-cap')
check(abs(ledger['charged_seconds']-prior['charged_seconds']-work['budget']['elapsed_model_work_seconds'])<1e-8,'actual-elapsed-accounting')

initial=read(OUT/'run/initial-tensors.json');final=read(OUT/'run/final-tensors.json')
initialparams=read(OUT/'run/initial-parameter-hashes.json');finalparams=read(OUT/'run/final-parameter-hashes.json')
check(len(initialparams)==len(finalparams)==296 and initialparams==finalparams==read(R1/'run/training-result.json')['final_parameter_sha256'],'296-parameter-digests-unchanged-and-R1-matched')
initialstate={r['name']:(r['shape'],r['dtype'],r['sha256'])for r in initial['state_tensors']}
finalstate={r['name']:(r['shape'],r['dtype'],r['sha256'])for r in final['state_tensors']}
check(len(initialstate)==len(finalstate)==298 and initialstate==finalstate,'298-state-tensors-unchanged')
check(initial['trainable_parameter_count']==final['trainable_parameter_count']==0,'all-parameters-frozen')

inventory=work['evaluation_inventory']
check(len(inventory)==90,'90-new-prediction-and-metric-files')
for row in inventory:
    p=OUT/'run/evaluation'/row['file'];check(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],f'inventory:{p.name}')
totals=dict(raw_files=0,raw_float32_frames=0,endpoint_rows=0,raw_step_score_rows=0,sampler_batches=0,denoiser_forwards=0)
cohort_reports={}
for setting in (8,16):
    stats=result['settings'][str(setting)];check(stats['sampler_batch_attempts']==stats['sampler_batch_completed']==36,f'{setting}:36-sampler-batches')
    check(stats['raw_frames_saved']==144 and stats['endpoint_rows_scored']==72,f'{setting}:saved-counts')
    check(stats['denoiser_forward_attempts']==stats['denoiser_forward_completed']==36*setting,f'{setting}:observed-denoiser-count')
    check(len(stats['calls'])==36 and sum(c['denoiser_forward_completed'] for c in stats['calls'])==stats['denoiser_forward_completed'],f'{setting}:call-counts-sum')
    check(all(c['batch_size']==4 and c['denoiser_forward_attempts']==c['denoiser_forward_completed']==setting and math.isfinite(c['seconds']) and c['seconds']>=0 for c in stats['calls']),f'{setting}:per-call-counts-timing')
    check(abs(sum(c['seconds']for c in stats['calls'])-stats['sampler_seconds'])<1e-8,f'{setting}:timing-sum')
    check(stats['peak_cuda_reserved']<=config['max_cuda_bytes'],f'{setting}:GPU-reserved')
    call_index=0
    for cohort in ('train_visual','val_natural','val_motion'):
        prepared=load_eval_cohort(config['data_manifest'],config['evaluation_indices'],cohort)
        folder=OUT/'run/evaluation'/f'steps{setting}';stem=folder/f'cohort-{cohort}'
        side=read(stem.with_suffix('.json'));oldside=read(R1/'run/evaluation/final'/f'cohort-{cohort}.json')
        check(side['cohort_identity']==prepared['identity'] and side['window_ids']==prepared['window_ids']==oldside['window_ids'],f'{setting}/{cohort}:IDs-identity')
        check(side['evaluation_complete'] and not side['hard_stop'],f'{setting}/{cohort}:complete')
        check(side['checkpoint_sha256']==config['checkpoint_sha256'],f'{setting}/{cohort}:checkpoint-bound')
        check(side['sampler']==dict(config['sampler'],num_steps_denoising=setting),f'{setting}/{cohort}:only-sampler-step-variable')
        with np.load(stem.with_suffix('.npz'),allow_pickle=False) as z:
            saved={k:z[k].copy() for k in z.files}
        check(saved['window_ids'].tolist()==prepared['window_ids'],f'{setting}/{cohort}:endpoint-IDs')
        raw_steps=[[None]*4 for _ in range(12)]
        refs={(r['batch_index'],r['step']):r for r in side['raw_step_files']}
        records={(r['batch_index'],r['step']):r for r in side['step_records']}
        oldrefs={(r['batch_index'],r['step']):r for r in oldside['raw_step_files']}
        check(len(refs)==len(records)==12,f'{setting}/{cohort}:all-four-steps')
        for batch in range(3):
            start=batch*4;ids=prepared['window_ids'][start:start+4]
            history=prepared['histories'][start:start+4].copy()
            for step in range(4):
                ref=refs[batch,step];record=records[batch,step];p=folder/ref['file']
                check(p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],f'{setting}/{cohort}/{batch}/{step}:raw-file')
                with np.load(p,allow_pickle=False) as z:raw=z['raw'].copy();rawids=z['window_ids'].tolist()
                check(raw.dtype==np.float32 and raw.shape==(4,3,96,96) and rawids==ids,f'{setting}/{cohort}/{batch}/{step}:raw-shape-dtype-IDs')
                check(np.isfinite(raw).all(),f'{setting}/{cohort}/{batch}/{step}:finite')
                seed=paired_seed(5478,cohort,batch,ids,step)
                check(ref['seed']==record['seed']==oldrefs[batch,step]['seed']==seed==stats['calls'][call_index]['seed'],f'{setting}/{cohort}/{batch}/{step}:paired-seed')
                call_index+=1
                action=normalize_actions(prepared['absolute_actions'][start:start+4,step:step+4])
                check(ref['normalized_actions_sha256']==record['normalized_actions_sha256']==arrsha(action)==oldrefs[batch,step]['normalized_actions_sha256'],f'{setting}/{cohort}/{batch}/{step}:normalized-actions')
                check(ref['input_history_sha256']==record['input_history_sha256']==arrsha(history),f'{setting}/{cohort}/{batch}/{step}:raw-feedback-history')
                if step==0:check(arrsha(history)==oldrefs[batch,step]['input_history_sha256'],f'{setting}/{cohort}/{batch}:same-real-initial-history')
                check(record['raw_sha256']==arrsha(raw),f'{setting}/{cohort}/{batch}/{step}:raw-array-hash')
                if step in (0,3):check(np.array_equal(saved['h1'if step==0 else'h4'][start:start+4],raw),f'{setting}/{cohort}/{batch}/{step}:endpoint-value-equality')
                for i in range(4):raw_steps[start+i][step]=raw[i]
                history=np.concatenate((history[:,1:],raw[:,None]),axis=1)
                totals['raw_files']+=1;totals['raw_float32_frames']+=len(raw)
        metrics=read(stem.with_name(stem.name+'.metrics.json'))
        check(metrics['evaluation_complete'] and metrics['scoring_complete'] and metrics['requires_runner_active_budget'],f'{setting}/{cohort}:score-complete')
        gen=dict(window_ids=prepared['window_ids'],h1=saved['h1'],h4=saved['h4'],status_h1=saved['status_h1'].tolist(),status_h4=saved['status_h4'].tolist())
        check(metrics['input_identity']==metrics_input_identity(prepared,gen,raw_steps),f'{setting}/{cohort}:scoring-input-hashes')
        check(len(metrics['rows'])==24 and len(metrics['raw_step_rows'])==48,f'{setting}/{cohort}:full-score-denominators')
        mtime=stem.with_name(stem.name+'.metrics.json').stat().st_mtime
        check(stage['started_unix']<=mtime<=stage['finished_unix'],f'{setting}/{cohort}:scores-written-during-clock')
        totals['endpoint_rows']+=len(metrics['rows']);totals['raw_step_score_rows']+=len(metrics['raw_step_rows'])
        cohort_reports[f'{setting}/{cohort}']=dict(rows=len(metrics['rows']),raw_step_rows=len(metrics['raw_step_rows']))
    totals['sampler_batches']+=stats['sampler_batch_completed'];totals['denoiser_forwards']+=stats['denoiser_forward_completed']
check(totals==dict(raw_files=72,raw_float32_frames=288,endpoint_rows=144,raw_step_score_rows=288,sampler_batches=72,denoiser_forwards=864),'complete-total-workload')
check(result['denoiser_forward_attempts']==result['denoiser_forward_completed']==864,'overall-observed-counter')
events=[json.loads(line)for line in(OUT/'run/events.jsonl').read_text(encoding='utf-8').splitlines()]
check(len([r for r in events if r['kind']=='native_call_start'])==72,'72-actual-call-start-events')
check(len([r for r in events if r['kind']=='setting_complete'])==2,'two-complete-settings-events')
check(not [r for r in events if r['kind'] in ('fatal_stop','sampler_exception','predictor_exception')],'no-runtime-error-events')
report=dict(kind='R2C2-closed-readonly-integrity-review.v1',passed=not issues,checks=checks,issues=issues,
    checked_at_utc=datetime.now(timezone.utc).isoformat(),totals=totals,cohorts=cohort_reports,
    old_protected_files=len(oldprotected),old_active_source_files=128,new_stage_source_files=len(read(OUT/'source-identity.json')['files']),
    checkpoint_loaded_by_auditor=False,model_calls=0,new_scores=0,physics_controls=0,
    time_policy='record_only_v2',current_model_seconds_statistics=work['budget']['elapsed_model_work_seconds'])
(OUT/'cold-integrity-review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 2)
