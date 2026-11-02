"""One corrective stage3B GPC run with conserved stage3 budget."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_training import atomic_json,sha256
def authorize(output):
    parent=Path(output).resolve().parent
    (parent/'SUPERVISION.md').read_text(encoding='utf-8')
    state=json.loads((parent/'stage-state.json').read_text())
    if state['current_stage']!='3B' or state['authorized_stages']!=['3B'] or state['status']!='in_progress' or state['callback']['sent']:
        raise RuntimeError('Only one newly released3B correction may run')
    return state


def validate_prior_ledger(value,prior):
    if value!=prior or value['active_stage'] is not None or value['charged_updates']!=0 or value['completed_updates']!=0:
        raise RuntimeError('The original closed stage3 ledger may not be reset, replaced or replayed')
    if value['charged_seconds']!=28.3440000000046 or len(value['stages'])!=1:
        raise RuntimeError('Exactly the preserved28.3440000000046s stage3 cost is required')
    old=value['stages'][0]
    if old['phase']!='stage3-visible-microfit' or old['reserved_updates']!=0 or old['completed_updates']!=0 or not old['status'].startswith('failed:FloatingPointError: Invalid native RGB'):
        raise RuntimeError('The prior failed stage3 identity differs')
    return dict(max_attempts=200-value['charged_updates'],max_seconds=600-value['charged_seconds'],update_cutoff_seconds=420-value['charged_seconds'])


def verify_frozen(output):
    output=Path(output);identity=json.loads((output/'source-identity.json').read_text())
    for row in identity['files']:
        if sha256(row['source'])!=row['sha256'] or sha256(output/row['snapshot'])!=row['sha256']:
            raise ValueError('Stage3 frozen source/input changed: '+row['source'])
    return identity


def _parameter_digest(model):
    from tdyssey.native_rgb_training import tensor_hashes
    values=tensor_hashes(model)
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest(),values


def prediction_hook(output,phase,truth,plans):
    def callback(context):
        import numpy as np
        from tdyssey.native_rgb_evaluation import native_sampler_batch_predictor
        model=context['model'];check=context['check'];budget=context['budget']
        if sum(p.numel() for p in model.parameters())!=8729827 or any(not p.requires_grad for p in model.parameters()):
            raise ValueError('Expected every original8,729,827 GPC parameter trainable')
        digest,param_hashes=_parameter_digest(model)
        predictor=native_sampler_batch_predictor(model,context['source'])
        phase_started=time.monotonic();summaries={}
        for cohort in ['train18','dev36']:
            source=truth[cohort];n=len(source['window_ids'])
            result=np.full((n,3,96,96),np.nan,dtype=np.float32);statuses=np.full(n,'not_attempted',dtype='<U512');records=[];failure=None
            path=Path(output)/'predictions'/f'{phase}-{cohort}.npz';path.parent.mkdir(exist_ok=True)
            try:
                for batch in plans['cohorts'][cohort]:
                    check()
                    budget.require_operation(30. if phase=='frozen' else max(3.,float(budget.baseline_seconds or 0)/14*1.5),f'{phase} native H1 batch')
                    positions=batch['positions'];began=time.monotonic()
                    if [str(source['window_ids'][i]) for i in positions]!=batch['window_ids']:raise ValueError('Paired sampler window order changed')
                    try:
                        values=predictor(source['histories'][positions],source['actions_absolute'][positions],batch_seed=batch['batch_seed'])
                    except BaseException as error:
                        statuses[positions]=type(error).__name__+': '+str(error);raise
                    if values.shape!=(len(positions),3,96,96):
                        statuses[positions]='wrong_native_output_shape';raise ValueError('Wrong unmodified native sampler shape')
                    result[positions]=values
                    bad=False
                    for i,value in zip(positions,values):
                        if not np.isfinite(value).all():statuses[i]='nonfinite_native_RGB';bad=True
                        elif value.min()<0 or value.max()>1:statuses[i]='out_of_range_native_RGB'
                        else:statuses[i]='ok'
                    records.append(dict(batch,phase=phase,wall_seconds=time.monotonic()-began,model_work_elapsed=budget.elapsed))
                    print(json.dumps(dict(event='native_H1_batch',phase=phase,cohort=cohort,batch=batch['batch_index'],windows=len(positions),model_work_seconds=budget.elapsed)),flush=True)
                    check()
                    if bad:raise FloatingPointError('Invalid native RGB retained without repair')
            except BaseException as error:
                failure=type(error).__name__+': '+str(error)
                raise
            finally:
                with path.open('xb') as stream:np.savez_compressed(stream,prediction=result,status=statuses,window_ids=source['window_ids'])
                atomic_json(path.with_suffix('.json'),dict(phase=phase,cohort=cohort,prediction_sha256=sha256(path),statuses=dict(Counter(statuses.tolist())),
                    batch_records=records,failure=failure,unmodified_native_RGB=True,postprocessing=False,horizon=1,
                    parameter_digest=digest,parameter_sha256=param_hashes,updates=context['progress']['updates'],
                    range_rows=[dict(index=i,window_id=str(source['window_ids'][i]),status=str(statuses[i]),
                        finite_values=int(np.isfinite(v).sum()),min=float(v[np.isfinite(v)].min()) if np.isfinite(v).any() else None,
                        max=float(v[np.isfinite(v)].max()) if np.isfinite(v).any() else None,
                        negative_values=int((v<0).sum()),above_one_values=int((v>1).sum())) for i,v in enumerate(result)],
                    paired_batches_sha256=sha256(Path(output)/'paired-batches.json'),same_batch_seed_is_not_sequential_per_window_noise=True))
            summaries[cohort]=dict(windows=n,ok=int((statuses=='ok').sum()),sha256=sha256(path),parameter_digest=digest,batches=len(records))
        return dict(cohorts=summaries,wall_seconds=time.monotonic()-phase_started,model_parameters=8729827,all_requires_grad=True,
            native_sampler_calls=sum(len(v) for v in plans['cohorts'].values()),native_prediction_frames=54,extra_heads=False,geometric_loss=False)
    return callback


def compose_comparisons(output):
    """Pure CPU copies of original arrays; never infer missing predictions."""
    import numpy as np
    output=Path(output);result={};(output/'predictions').mkdir(exist_ok=True)
    for cohort in ['train18','dev36']:
        with np.load(output/'truth'/f'{cohort}.npz',allow_pickle=False) as file:truth={k:file[k] for k in file.files}
        n=len(truth['last']);arrays={k:truth[k] for k in ['last','true','hold','last_full_mask','true_full_mask']}
        arrays['window_ids']=truth['window_ids']
        for phase,key in [('frozen','frozen'),('fitted','fitted')]:
            path=output/'predictions'/f'{phase}-{cohort}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as file:
                    if not np.array_equal(file['window_ids'],truth['window_ids']):raise ValueError('Prediction/truth IDs differ')
                    arrays[key]=file['prediction'];arrays[key+'_status']=file['status']
            else:arrays[key]=np.full_like(truth['last'],np.nan);arrays[key+'_status']=np.full(n,'not_measured_budget_or_failure',dtype='<U512')
        path=output/'predictions'/f'{cohort}.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        result[cohort]=dict(file=str(path),sha256=sha256(path),windows=n)
    atomic_json(output/'comparison-arrays.json',result);return result


def run(output,config):
    import numpy as np
    output=Path(output).resolve();authorize(output);verify_frozen(output)
    cfg=json.loads(Path(config).read_text());windows=json.loads((output/'frozen-windows.json').read_text());plans=json.loads((output/'paired-batches.json').read_text())
    if (cfg['max_updates']!=3000 or cfg['max_train_seconds']!=5400 or cfg['stage_max_updates']!=200
        or cfg['stage_max_model_work_seconds']!=600-cfg['prior_charged_model_seconds'] or cfg['microbatch']!=4 or cfg['effective_batch']!=16
        or cfg['checkpoint_policy']!='last_only' or cfg['train_indices']!=[r['dataset_index'] for r in windows['train']]
        or cfg['validation_indices']!=[r['dataset_index'] for r in windows['dev']]):raise ValueError('Exact stage3 protocol/caps/indices required')
    if (output/'MODEL_WORK_STARTED.json').exists():raise FileExistsError('Stage3B permits one corrective model-work execution; no automatic replay')
    truth={}
    for name in ['train18','dev36']:
        with np.load(output/'truth'/f'{name}.npz',allow_pickle=False) as file:truth[name]={k:file[k] for k in file.files}
        if not np.array_equal(truth[name]['last'],truth[name]['histories'][:,3]) or not np.isfinite(truth[name]['actions_absolute']).all():raise ValueError('Absolute actions/last image changed')
    from tdyssey.native_rgb_stage_budget import ModelWorkBudget,StorageBudget,native_storage_paths
    from tdyssey.native_rgb_training import CostLedger,train,load_checkpoint_payload
    model_config=dict(cfg)
    for key in ['artifact_root','ledger_path','joint_data','output']:
        model_config[key]=str((ROOT/cfg[key]).resolve())
    ledger=CostLedger(model_config['ledger_path'],3000,5400.,legacy_cost=dict(historical_native_attempts=1899,historical_native_model_seconds=5352.516,included_in_new_round_budget=False))
    previous=ledger.snapshot()
    prior=json.loads((output/'prior-ledger.json').read_text())
    if sha256(output/'prior-ledger.json')!=cfg['prior_ledger_sha256']:raise ValueError('Frozen prior ledger changed')
    residual=validate_prior_ledger(previous,prior)
    old_work=json.loads(Path(cfg['prior_model_work_file']).read_text())
    if old_work['budget']['ledger']!=previous:raise ValueError('Prior model-work and actual global ledger do not match')
    if cfg['stage_max_updates']!=residual['max_attempts'] or cfg['stage_max_model_work_seconds']!=residual['max_seconds'] or cfg['no_new_update_after_seconds']!=residual['update_cutoff_seconds']:
        raise ValueError('3+3B200/600/420 residual budget must be conserved exactly')
    paths=native_storage_paths(ROOT,output.parent,extra_paths=[cfg['gpc_source_root'],cfg['physics_source_root'],cfg['init_checkpoint']])
    storage=StorageBudget(paths,max_bytes=3*1024**3,atomic_reserve_bytes=256*1024**2,minimum_free_bytes=512*1024**2)
    before_storage=storage.check(force=True)
    atomic_json(output/'MODEL_WORK_STARTED.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha256(config),
        source_identity_sha256=sha256(output/'source-identity.json'),initial_storage=before_storage,no_automatic_retry=True))
    raw_stage=ledger.start('stage3b-visible-microfit');budget=ModelWorkBudget(raw_stage,storage=storage,**residual)
    result=None;failure=None;checkpoint_review=None
    try:
        budget.require_operation(60.,'original full GPC model initialization')
        import torch
        if not torch.cuda.is_available() or '3060 Ti' not in torch.cuda.get_device_name(0):raise RuntimeError('This release is the existing3060Ti run; no device substitution')
        atomic_json(output/'timed-device.json',dict(name=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
            torch=torch.__version__,cuda=torch.version.cuda,model_work_elapsed=budget.elapsed,device_query_inside_model_clock=True))
        result=train(model_config,model_config['output'],model_config['joint_data'],cfg['gpc_source_root'],cfg['init_checkpoint'],
            physics_source_root=cfg['physics_source_root'],execute=True,phase='stage3b-visible-microfit',updates=residual['max_attempts'],
            indices=cfg['train_indices'],validation_indices=cfg['validation_indices'],checkpoint_policy='last_only',work_budget=budget,
            before_updates_callback=prediction_hook(output,'frozen',truth,plans),after_updates_callback=prediction_hook(output,'fitted',truth,plans))
        path=Path(model_config['output'])/'last.pt'
        if path.exists():
            budget.require_operation(15.,'complete CPU checkpoint verification')
            began=time.monotonic();payload=load_checkpoint_payload(path)
            required={'model','optimizer','scaler','scheduler','rng','order','progress','config','identity','ledger'}
            if not required.issubset(payload):raise ValueError('Incomplete last checkpoint')
            if payload['scheduler']['T_max']!=3000 or payload['config']['validation_indices']!=cfg['validation_indices']:
                raise ValueError('Checkpoint scheduler/explicit validation protocol differs')
            checkpoint_review=dict(passed=True,required_fields=sorted(required),checkpoint_sha256=sha256(path),
                updates=payload['progress']['updates'],optimizer_state_tensors=len(payload['optimizer']['state']),
                scheduler_T_max=payload['scheduler']['T_max'],order_cursor=payload['order']['cursor'],
                verification_wall_seconds=time.monotonic()-began,model_work_elapsed=budget.elapsed,CPU_torch_load_counted=True)
            del payload;budget.heartbeat()
        else:
            raise RuntimeError('No usable complete last checkpoint; retain missing measurements and report failure')
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
        atomic_json(output/'model-work-error.json',dict(error=failure,model_work_elapsed=budget.elapsed,stage_budget=budget.report(),no_retry=True))
    finally:
        budget.finish('failed:'+failure if failure else 'complete' if result and result.get('status')=='complete' else 'bounded_stop')
        atomic_json(output/'model-work.json',dict(failure=failure,training_result=result,checkpoint_review=checkpoint_review,
            budget=budget.report(),physical_controls=0,no_unseen=True,no_H4=True,no_full319=True,one_run_only=True))
    compose_comparisons(output)
    return dict(failure=failure,status=None if result is None else result['status'],budget=budget.report(),checkpoint_review=checkpoint_review)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/native-rgb-supervised-20261029/stage3b')
    p.add_argument('--config',type=Path,default=ROOT/'configs/native_rgb_supervised_stage3b.json');a=p.parse_args()
    if not a.execute:raise SystemExit('Explicit --execute required; this entry never performs a smoke run')
    result=run(a.output,a.config);print(json.dumps(dict(failure=result['failure'],status=result['status'],model_work_seconds=result['budget']['elapsed_model_work_seconds'],attempts=result['budget']['ledger']['charged_updates'])));raise SystemExit(1 if result['failure'] else 0)
