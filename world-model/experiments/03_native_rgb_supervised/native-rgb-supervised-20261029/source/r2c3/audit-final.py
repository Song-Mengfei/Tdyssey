"""Closed C3 facts, hashes and saved-array identities only; never model/scoring."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,sys,math

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
EXP=ROOT/'artifacts/native-rgb-supervised-20261029';OUT=EXP/'r2c3'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
issues=[];checks=0
def check(v,label):
    global checks
    checks+=1
    if not v:issues.append(label)

# This gate precedes any checkpoint hashing or whole-run reading.
work=read(OUT/'model-work.json')
assert work['closed'] is True,'Refuse active training stage'
assert work['budget']['closed'] is True,'Refuse unclosed model accounting'

import numpy as np
from tdyssey.native_rgb_r2c3_continuation import R2C3WindowOrder
from tdyssey.native_rgb_r2c3_evaluation_io import load_referenced_predictions
from tdyssey.native_rgb_r2c0_evaluation import normalize_actions,paired_seed
from scripts.native_rgb_r2c0_evaluation import load_eval_cohort
from scripts.native_rgb_r2c1_report import metrics_input_identity

config=read(ROOT/'configs/native_rgb_r2c3.json');result=work['training_result'];progress=result['progress'];local=progress['r2c3']
check(work['status']=='complete' and work['failure'] is None and not result['cleanup_errors'],'complete-no-runtime-or-cleanup-failure')
check(progress['updates']==6000 and local['completed_updates']==5700,'6000-lineage-5700-new-complete')
attempts=local['attempts'];skips=local['AMP_skips'];draws=local['actual_draws']
check(attempts==5700+skips,'attempts-complete-plus-normal-skips')
check(draws==local['attempted_draws']==16*attempts==91200+16*skips,'actual-new-draws-include-skips')
check(progress['samples']==progress['actual_draws_total']==96000+16*skips,'actual-total-draws-include-skips')
check(progress['attempts']==300+attempts,'lineage-attempts-preserved')
check(result['optimizer_step_applications_observed']==5700,'5700-public-post-hook-real-optimizer-applications')
check(result['resume_checkpoint_loads']==1 and result['final_checkpoint_reload_verified'],'full-restore-and-real-final-reload-reported')
restore=read(OUT/'run/resume-verification.json')
check(restore['passed'] and all(restore['components'].values()),'every-initial-restore-component-runtime-digest')

source=read(config['source_identity']);protected=read(config['input_identity'])['files']
check(len(source['files'])==141 and len(protected)==3222,'active-v2-source141-protected3222')
for row in source['files']:
    check(sha(row['source'])==row['sha256'],'active-source:'+row['source'])
    check(sha(OUT/row['snapshot'])==row['sha256'],'snapshot:'+row['snapshot'])
for row in protected:
    p=Path(row['path']);check(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'old-protected:'+str(p))
for p,digest in config['input_hashes'].items():check(sha(p)==digest,'config-input:'+p)

ledger=read(config['ledger_path']);prior=read(config['prior_ledger_copy']);stage=ledger['stages'][-1]
check(ledger['active_stage'] is None and ledger['stages'][:-1]==prior['stages'],'closed-ledger-history-prefix')
check(ledger.get('policy_migrations')==prior.get('policy_migrations'),'time-policy-migration-history-unchanged')
check(ledger['max_updates'] is None and ledger['max_seconds'] is None,'no-old-global-attempt-or-time-guard')
check(stage['reserved_updates']==attempts and stage['completed_updates']==5700 and stage['skipped_updates']==skips,'durable-stage-attempt-complete-skip-accounting')
check(ledger['charged_updates']==550+attempts and ledger['completed_updates']==6250,'all-historical-attempts-and-completions-preserved')
check(abs(ledger['charged_seconds']-prior['charged_seconds']-work['budget']['elapsed_model_work_seconds'])<1e-8,'all-actual-time-recorded-no-refund')
check(work['budget']['time_policy']=='record_only_v2' and work['budget']['stage_max_model_work_seconds'] is None,'record-only-no-time-ceiling')

events=[json.loads(s)for s in(OUT/'run/events.jsonl').read_text(encoding='utf-8').splitlines()]
reserved=[e for e in events if e['kind']=='attempt_reserved_before_draw'];completed=[e for e in events if e['kind']=='update_complete']
skip_events=[e for e in events if e['kind']=='AMP_skipped_update'];micro=[e for e in events if e['kind']=='microbatch_backward_complete']
check(len(reserved)==attempts and len(completed)==5700 and len(skip_events)==skips,'real-event-attempt-update-skip-counts')
check(len(micro)==result['training_forward_calls']==result['training_forward_completed_calls']==result['training_backward_completed_microbatches']==4*attempts,'actual-four-micro-forwards-backwards-each-attempt')
check([e['update']for e in completed]==list(range(301,6001)),'complete-update-sequence301-to6000')
check(not[e for e in events if e['kind'] in('fatal_stop','predictor_exception')],'no-fatal-or-predictor-errors')
for row in completed:
    u=row['update'];expected=1e-5+9e-5*(1+math.cos(math.pi*min(u-1,3000)/3000))/2
    check(math.isclose(row['optimizer_lr_used'],expected,rel_tol=1e-7,abs_tol=1e-12),'lr-used:'+str(u))
    check(row['scheduler_last_epoch']==min(u,3000),'scheduler-last-epoch:'+str(u))
    if u>=3000:check(row['lr_after_scheduler_or_hold']==1e-5,'eta-min-hold:'+str(u))
for row in skip_events:
    check(row['optimizer_applied'] is False and row['scheduler_advanced'] is False and row['scale_after']<row['scale_before'],'real-AMP-skip-scale-and-state')

with np.load(OUT/'run/actual-new-draw-indices.npz',allow_pickle=False)as z:actual=z['indices'].copy()
check(actual.dtype==np.int32 and actual.shape==(draws,) and actual.min()>=0 and actual.max()<861,'actual-new-TRAIN-index-array')
pool=read(EXP/'r2c0/data/pools.json');order=R2C3WindowOrder(pool);order.load_state_dict(read(EXP/'r2c0/data/order-state4800.json'))
skip_attempts={e['attempt']for e in skip_events};expected_stream=[]
for i,row in enumerate(reserved,1):
    selected=order.take(16);expected_stream.extend(selected)
    check(row['attempt']==i and row['lineage_attempt']==300+i and row['indices']==selected,'actual-continuing-TRAIN-draw:'+str(i))
    check(row['window_ids']==[pool['windows'][j]['window_id']for j in selected],'actual-window-IDs:'+str(i))
    if i in skip_attempts:order.extend_after_amp_skip()
check(np.array_equal(actual,np.asarray(expected_stream,dtype=np.int32)),'all-actual-draws-equal-unbroken-restored-stream')
check(order.cursor==order.max_draws==96000+16*skips and order.skip_extension_count==skips,'final-public-stream-scope-and-skip-extensions')

saved=[e for e in events if e['kind']=='checkpoint'];periodic=[e for e in saved if e['label']=='periodic_complete_boundary']
check([e['updates']for e in periodic]==list(range(400,6001,100)),'all57-periodic-success-boundary-full-saves')
check(len(saved)==57,'no-duplicate-normal-full-checkpoint-saves')
checkpoint=result['checkpoint'];p=Path(checkpoint['path'])
check(checkpoint['updates']==6000 and p.stat().st_size==checkpoint['bytes'] and sha(p)==checkpoint['sha256'],'closed-final6000-full-checkpoint-SHA')
check(checkpoint['bytes']<=112*1024**2,'bounded-full-payload')
ptfiles=list((OUT/'run').rglob('*.pt'))
check(len(ptfiles)==3 and sorted(p.name for p in ptfiles)==['last.pt','model-001000.pt','model-003000.pt'],'one-full-two-model-only-no-duplicates')
for m in result['milestones']:
    p=Path(m['path']);check(m['updates']in(1000,3000) and p.stat().st_size==m['bytes'] and sha(p)==m['sha256'],'stable-light-model-SHA:'+str(m['updates']))
initial=read(OUT/'run/initial-parameter-hashes.json');final=result['final_parameter_sha256'];changed=sum(initial[n]!=final[n]for n in initial)
check(len(initial)==len(final)==296 and changed==result['parameter_tensors_changed_from_300']==296,'all296-parameter-tensor-digests-changed')

evalroot=OUT/'run/evaluation';inventory=work['evaluation_inventory'];check(len(inventory)==252,'252-final-bound-evaluation-files')
for row in inventory:
    p=evalroot/row['file'];check(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'eval-inventory:'+row['file'])
prepared={c:load_eval_cohort(config['data_manifest'],config['evaluation_indices'],c)for c in('train_visual','val_natural','val_motion')}
totals=Counter();shape={};noise_reference=None
for point in(1000,3000,6000):
    obs=read(OUT/f'run/milestones/observation-{point}.json')
    check(obs['closed'] and obs['lineage_update']==point and obs['continuing_without_waiting']==(point<6000),'immutable-point-closure:'+str(point))
    check(len(obs['evaluation_inventory'])==84,'84-files-at-point:'+str(point))
    loss=read(OUT/f'run/fixed-loss-{point}.json');check(len(loss['rows'])==9,'nine-fixed-loss-batches:'+str(point))
    noise=[(r['cohort'],r['batch'],r['window_ids'],r['seed'],r['noise_generator_before_sha256'],r['noise_generator_after_sha256'])for r in loss['rows']]
    if noise_reference is None:noise_reference=noise
    else:check(noise==noise_reference,'same-recorded-noise-seed-and-generator-plan:'+str(point))
    check(all(math.isfinite(r['loss'])for r in loss['rows']),'finite-fixed-loss:'+str(point))
    totals['fixed_loss_batches']+=len(loss['rows'])
    for setting in(3,16):
        stats=obs['settings'][str(setting)];folder=evalroot/f'update{point}'/f'steps{setting}'
        check(stats['sampler_batch_attempts']==stats['sampler_batch_completed']==len(stats['calls'])==36,f'{point}/{setting}:36-observed-sampler-batches')
        check(stats['denoiser_forward_attempts']==stats['denoiser_forward_completed']==36*setting,f'{point}/{setting}:actual-denoiser-forwards')
        check(sum(c['denoiser_forward_completed']for c in stats['calls'])==36*setting,f'{point}/{setting}:actual-count-sum')
        totals['sampler_batches']+=stats['sampler_batch_completed'];totals['sampling_denoiser_forwards']+=stats['denoiser_forward_completed']
        for cohort,a in prepared.items():
            stem=folder/f'cohort-{cohort}';side=read(stem.with_suffix('.json'));metrics=read(stem.with_name(stem.name+'.metrics.json'))
            r,steps,_=load_referenced_predictions(folder,cohort,a)
            check(side['checkpoint_sha256']==obs['full_payload_at_observation']['sha256'],f'{point}/{setting}/{cohort}:correct-observation-weights')
            check(side['evaluation_complete'] and not side['hard_stop'] and metrics['evaluation_complete'] and metrics['scoring_complete'],f'{point}/{setting}/{cohort}:complete')
            check(metrics['input_identity']==metrics_input_identity(a,r,steps),f'{point}/{setting}/{cohort}:exact-saved-scoring-input-identity')
            check(len(metrics['rows'])==24 and len(metrics['raw_step_rows'])==48,f'{point}/{setting}/{cohort}:full-denominators')
            check(stage['started_unix']<=stem.with_name(stem.name+'.metrics.json').stat().st_mtime<=stage['finished_unix'],f'{point}/{setting}/{cohort}:scores-inside-model-clock')
            refs={(x['batch_index'],x['step']):x for x in side['raw_step_files']};records={(x['batch_index'],x['step']):x for x in side['step_records']}
            check(len(refs)==len(records)==12,f'{point}/{setting}/{cohort}:all-raw-batch-steps')
            for batch in range(3):
                ids=a['window_ids'][batch*4:batch*4+4];history=a['histories'][batch*4:batch*4+4].copy()
                for step in range(4):
                    ref=refs[batch,step];rec=records[batch,step];raw=np.stack([steps[batch*4+i][step]for i in range(4)])
                    check(raw.dtype==np.float32 and raw.shape==(4,3,96,96) and np.isfinite(raw).all(),f'{point}/{setting}/{cohort}/{batch}/{step}:raw-float32-finite')
                    expected_seed=paired_seed(5478,cohort,batch,ids,step)
                    check(ref['seed']==rec['seed']==expected_seed,f'{point}/{setting}/{cohort}/{batch}/{step}:paired-seed')
                    action=normalize_actions(a['absolute_actions'][batch*4:batch*4+4,step:step+4])
                    check(ref['normalized_actions_sha256']==rec['normalized_actions_sha256']==hashlib.sha256(action.tobytes()).hexdigest(),f'{point}/{setting}/{cohort}/{batch}/{step}:actual-actions')
                    check(ref['input_history_sha256']==rec['input_history_sha256']==hashlib.sha256(history.tobytes()).hexdigest(),f'{point}/{setting}/{cohort}/{batch}/{step}:raw-feedback-history')
                    check(rec['raw_sha256']==hashlib.sha256(raw.tobytes()).hexdigest(),f'{point}/{setting}/{cohort}/{batch}/{step}:actual-raw-digest')
                    history=np.concatenate((history[:,1:],raw[:,None]),axis=1);totals['raw_files']+=1;totals['raw_float32_frames']+=4
            for horizon in(1,4):
                rows=[x for x in metrics['rows']if x['horizon']==horizon]
                check(len(rows)==12,f'{point}/{setting}/{cohort}/H{horizon}:twelve-window-shape-denominator')
                if cohort=='val_natural':
                    check(sum(not x['H1_obvious_motion']for x in rows)==8 and sum(x['H1_nearstatic_aux']for x in rows)==4,f'{point}/{setting}/H{horizon}:natural8-nearstatic4')
                shape[f'{point}/{setting}/{cohort}/H{horizon}']=dict(denominator=12,outcomes=dict(Counter(x['shape_outcome']for x in rows)),endpoint_hard_failures=sum(x['endpoint_hard_generation_failure']is True for x in rows),path_hard_failures=sum(x['any_path_step_hard_failure']is True for x in rows))
            totals['endpoint_rows']+=len(metrics['rows']);totals['raw_step_score_rows']+=len(metrics['raw_step_rows'])
check(dict(totals)==dict(fixed_loss_batches=27,sampler_batches=216,sampling_denoiser_forwards=2052,raw_files=216,raw_float32_frames=864,endpoint_rows=432,raw_step_score_rows=864),'complete-new-evaluation-workload')
check(result['fixed_loss_forward_calls']==27 and result['native_sampler_batch_attempts']==result['native_sampler_batch_completed']==216 and result['denoiser_forward_attempts']==result['denoiser_forward_completed']==2052,'runtime-evaluation-counters')
check('torch'not in sys.modules,'audit-never-imported-Torch')
report=dict(kind='R2C3-final-closed-readonly-integrity-review.v1',passed=not issues,checks=checks,issues=issues,checked_at_utc=datetime.now(timezone.utc).isoformat(),frozen_source_files=141,protected_old_files=3222,model_work_closed=True,final_lineage_updates=6000,new_completed_updates=5700,new_attempts=attempts,AMP_skips=skips,new_actual_draws=draws,total_actual_draws=progress['actual_draws_total'],totals=dict(totals),parameter_tensors_changed_from300=changed,checkpoint=checkpoint,periodic_full_saves=len(periodic),shape_denominators=shape,time_policy='record_only_v2',current_model_seconds_statistics=work['budget']['elapsed_model_work_seconds'],global_model_seconds_statistics=ledger['charged_seconds'],fixed_noise_verification_scope='Same per-batch IDs/seeds and recorded generator before/after hashes at1000/3000/6000, fixed code/shapes; actual noise tensor bytes were not separately recorded or independently recomputed.',checkpoint_loads=0,new_model_calls=0,new_scores=0,physics_controls=0)
(OUT/'cold-integrity-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
raise SystemExit(0 if report['passed']else 2)
