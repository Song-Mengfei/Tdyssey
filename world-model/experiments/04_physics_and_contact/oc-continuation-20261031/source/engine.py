"""Two complete epochs from the exact AR1 full boundary; no import-time Torch.

The detached worker owns execution. Existing AR1 loaders, objective, model,
optimizer-step observer and formal evaluator are reused without editing them.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np

REPO=Path(__file__).resolve().parents[2]
AR1_ROOT=REPO/'artifacts/accelerated-rgb-run-20261031'
PARENT_SHA='492e9badaa3e622254c829e1ea44cad6ad8c4c4636feb15c7764ec07e3745f11'
PLAN_SHA='e2a84aca0c38a24f5bfd4a72a0c0b68b906e429f32b2be501b9dac4fe023c3f0'

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load_ar1():
    spec=importlib.util.spec_from_file_location('oc_engine_frozen_ar1_train_eval',AR1_ROOT/'train_eval.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def validate_parent(payload, train):
    required={'model','optimizer','scaler','global_rng','conditioning_rngs','order','progress','identity','legacy_scheduler_provenance'}
    if payload.get('kind')!='accelerated_native_rgb_complete.v1' or not required.issubset(payload): raise ValueError('Exact full AR1 checkpoint required; no model-only or7499 fallback')
    p=payload['progress']
    if (p.get('lineage_updates')!=9433 or not p.get('epoch_complete') or not p.get('complete_optimizer_boundary')
            or p.get('pending_attempt') is not None or p.get('unique_samples_exposed')!=30935 or p.get('batches_consumed')!=1934):
        raise ValueError('AR1 first-new-data-epoch full9433boundary required')
    if payload['order']['cursor']!=1934 or payload['order']['sample_ids']!=[r['sample_id'] for r in train]: raise ValueError('Frozen AR1 sample pool/order differs')
    if len(train)!=30935 or len({r['sample_id'] for r in train})!=30935 or len({r['family_id'] for r in train})!=1339: raise ValueError('Approved30935unique samples/1339families required')
    if any(abs(g['lr']-1e-5)>1e-14 for g in payload['optimizer']['param_groups']): raise ValueError('Held parent1e-5LR differs')
    return True

def freeze_orders(root, config, train, ar1):
    from common import atomic_json
    seeds=config.get('epoch_order_seeds',{'2':20261033,'3':20261034}); orders={}; records={}
    for epoch in (2,3):
        seed=int(seeds[str(epoch)]); batches=ar1.epoch_batches(train,seed)
        flat=[i for b in batches for i in b]
        if len(batches)!=1934 or sorted(flat)!=list(range(30935)) or [len(x) for x in batches[:-1]]!=[16]*1933 or len(batches[-1])!=7: raise ValueError('One complete unique30935window epoch required')
        orders[epoch]=batches; records[str(epoch)]=dict(seed=seed,batches=batches,order_sample_id_sha256=digest([train[i]['sample_id'] for i in flat]),unique_samples=30935,optimizer_batches=1934)
    if records['2']['order_sample_id_sha256']==records['3']['order_sample_id_sha256']: raise ValueError('Each additional epoch must be independently reordered')
    path=Path(root)/'epoch-orders.json'
    if path.exists():
        if read(path)!=records: raise ValueError('Frozen additional epoch orders differ')
    else: atomic_json(path,records)
    return orders,records

def new_progress(epoch,lineage,stage):
    return dict(epoch=epoch,epoch_complete=False,unique_samples_exposed=0,successful_updates=0,AMP_skips=0,batches_consumed=0,
        planned_samples=30935,planned_batches=1934,lineage_updates=lineage,complete_optimizer_boundary=True,pending_attempt=None,
        model_forward_attempts=0,model_forward_completed=0,backward_microbatches=0,clean_microbatches=0,weak_microbatches=0,
        consecutive_AMP_overflows=0,additional_stage=deepcopy(stage))

def run(root,config):
    """Only the authorized detached worker calls this, once; no auto restart."""
    import torch
    from common import atomic_json,check_storage,replace_atomic
    from tdyssey.gpc_world_model import load_world_model_source,world_model_spec
    from tdyssey.raw_world_model_training import OPTIMIZER,make_generator
    from tdyssey.native_rgb_training import _cpu_tree,capture_rng,restore_rng
    from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs,conditioning_residual_loss,POLICY_SHA
    from tdyssey.native_rgb_r2c3_training import scaled_optimizer_step
    from tdyssey.native_rgb_r2c1r1_training import _tree_digest
    import diagnostics
    root=Path(root).resolve(); output=root/'training'; output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'execution-claim.json',dict(pid=os.getpid(),started_unix=time.time(),epochs=[2,3],automatic_retry=False),exclusive=True)
    if config.get('epochs',[2,3])!=[2,3] or int(config.get('planned_samples_per_epoch',30935))!=30935: raise ValueError('Only additional new-data epochs2and3 released')
    ar1=load_ar1(); data_root=Path(config.get('dataset_root',AR1_ROOT)).resolve()
    train=read(data_root/'selection/train-index.json'); val=read(data_root/'selection/seen-val-index.json')
    ar1.validate_indices(train,val); selection=read(data_root/'selection/summary.json')
    for name,expected in selection['index_sha256'].items():
        if sha(data_root/'selection'/name)!=expected: raise ValueError('Immutable AR1 selected index changed')
    spec=read(data_root/'evaluation-spec.json'); old_formal=read(data_root/'evaluation/after/metrics.json')
    if len(spec['rows'])!=48 or not old_formal['complete'] or old_formal['planned_windows']!=48 or spec['sampler']!=ar1.SAMPLER: raise ValueError('Exact completed original48family formal3step evaluation required')
    # Pure saved-data freeze precedes any native model construction/GPU call.
    diagnostic_spec=diagnostics.freeze(root,config)
    orders,order_records=freeze_orders(root,config,train,ar1)
    path=Path(config.get('starting_checkpoint',data_root/'training/last.pt')).resolve()
    expected=config.get('starting_checkpoint_sha256',PARENT_SHA)
    if expected!=PARENT_SHA or sha(path)!=PARENT_SHA: raise ValueError('Exact AR1 full9433checkpoint hash required')
    torch.set_num_threads(2); parent=torch.load(path,map_location='cpu',weights_only=False)
    validate_parent(parent,train)
    if parent['identity']['train_index_sha256']!=sha(data_root/'selection/train-index.json'): raise ValueError('Checkpoint and current frozen data differ')
    if parent['conditioning_rngs']['policy_sha256']!=POLICY_SHA: raise ValueError('Conditioning policy changed')
    if not torch.cuda.is_available(): raise RuntimeError('Authorized local CUDA unavailable; no CPU/fresh-weight fallback')
    torch.cuda.reset_peak_memory_stats(); check_storage(root,config,full=True)
    source=load_world_model_source(config.get('source_root',str(REPO.parent/'Tdyssey/third_party/gpc_code')))
    inner=deepcopy(world_model_spec()['model']); outer={k:inner.pop(k) for k in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
    with torch.random.fork_rng(devices=[]): model=source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**inner),**outer)).float()
    model.to('cuda').requires_grad_(True); model.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.)); model.train()
    optimizer=torch.optim.AdamW(model.parameters(),**dict(OPTIMIZER,lr=1e-5)); scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=128.)
    model.load_state_dict(parent['model'],strict=True); optimizer.load_state_dict(parent['optimizer']); scaler.load_state_dict(parent['scaler'])
    generator=make_generator(5478,next(model.parameters()).device); restore_rng(parent['global_rng'],generator)
    rngs=make_conditioning_rngs(**parent['conditioning_rngs']['seeds'],device=next(model.parameters()).device); rngs.load_state_dict(parent['conditioning_rngs'])
    td=lambda value:_tree_digest(value,torch)
    restored=dict(model=td(model.state_dict())==td(parent['model']),Adam=td(optimizer.state_dict())==td(parent['optimizer']),
        scaler=td(scaler.state_dict())==td(parent['scaler']),global_rng=td(capture_rng(generator))==td(parent['global_rng']),
        conditioning_rngs=td(rngs.state_dict())==td(parent['conditioning_rngs']),complete_parent_boundary=True)
    restoration=dict(passed=all(restored.values()),components=restored,parent_checkpoint=dict(path=str(path),sha256=PARENT_SHA,bytes=path.stat().st_size),
        starting_lineage_updates=9433,stale_sidecar_ignored=True,new_order_for_each_epoch=True,optimizer_reset=False,noise_streams_reset=False)
    atomic_json(output/'restore-proof.json',restoration)
    if not restoration['passed']: raise RuntimeError('Full model/Adam/scaler/RNG restore differs; stop before optimizer/inference')
    store=ar1.RawStore(); applied=[0]; hook=optimizer.register_step_post_hook(lambda *args:applied.__setitem__(0,applied[0]+1))
    stage=dict(unique_samples_exposed=0,batches_consumed=0,successful_updates=0,AMP_skips=0)
    progress=new_progress(2,9433,stage); ctx=SimpleNamespace(model=model,source=source,store=store,generator=generator,
        conditioning_rngs=rngs,config=config,progress=progress,ar1=ar1,device=next(model.parameters()).device)
    rng_proofs=[]; diag_results={}; epoch_results=[]; checkpoint=None; clean=True; began=time.monotonic()
    identity=dict(kind='oc-two-epoch-same-data-continuation.v1',starting_checkpoint_sha256=PARENT_SHA,starting_lineage_updates=9433,
        train_index_sha256=sha(data_root/'selection/train-index.json'),seen_val_index_sha256=sha(data_root/'selection/seen-val-index.json'),
        formal_spec_sha256=sha(data_root/'evaluation-spec.json'),epoch_orders_sha256=sha(root/'epoch-orders.json'),
        additional_epochs=[2,3],unique_samples_per_epoch=30935,additional_exposures=61870,planned_optimizer_batches=3868,
        microbatch=4,effective_batch=16,final_short_batch=7,learning_rate=1e-5,conditioning_policy_sha256=POLICY_SHA,
        sampler=ar1.SAMPLER,physics_steps=0,old_training_recipe_unchanged=True)
    atomic_json(output/'identity.json',identity)
    def protected(label,callback):
        global_state=capture_rng(generator); condition_state=rngs.state_dict(); mode=model.training
        before=dict(global_rng=td(global_state),conditioning_rngs=td(condition_state)); start=time.monotonic(); application_before=applied[0]
        old_progress=deepcopy(progress); result=None; error=None
        try: result=callback(); return result
        except BaseException as exc: error=repr(exc); raise
        finally:
            restore_rng(global_state,generator); rngs.load_state_dict(condition_state); model.train(mode)
            after=dict(global_rng=td(capture_rng(generator)),conditioning_rngs=td(rngs.state_dict()))
            proof=dict(label=label,before=before,after=after,matches=before==after,model_mode_restored=model.training==mode,
                optimizer_applications=applied[0]-application_before,training_progress_unchanged=progress==old_progress,
                wall_seconds=time.monotonic()-start,error=error)
            rng_proofs.append(proof); atomic_json(output/'rng-preservation.json',rng_proofs)
            if not proof['matches'] or not proof['model_mode_restored'] or proof['optimizer_applications'] or not proof['training_progress_unchanged']:
                raise RuntimeError('Evaluation changed training RNG/mode/optimizer/progress')
    def save(reason,retain_epoch=None):
        nonlocal checkpoint
        check_storage(root,config,reserve_bytes=path.stat().st_size+8*1024**2)
        if not clean or not progress['complete_optimizer_boundary']: raise RuntimeError('Refusing checkpoint from uncertain optimizer application')
        payload=dict(kind='oc_native_rgb_complete.v1',model=_cpu_tree(model.state_dict()),optimizer=_cpu_tree(optimizer.state_dict()),
            scaler=_cpu_tree(scaler.state_dict()),global_rng=capture_rng(generator),conditioning_rngs=rngs.state_dict(),
            order=dict(epoch=progress['epoch'],batches=orders[progress['epoch']],cursor=progress['batches_consumed'],sample_ids=[r['sample_id'] for r in train]),
            progress=deepcopy(progress),identity=identity,legacy_scheduler_provenance=deepcopy(parent['legacy_scheduler_provenance']),
            migration_provenance=dict(parent_identity=parent['identity'],parent_progress=parent['progress'],parent_order=parent['order'],parent_global_rng=parent['global_rng']),reason=reason)
        last=output/'last.pt'; temp=last.with_name('last.pt.tmp')
        try:
            torch.save(payload,temp)
            with temp.open('rb+') as f: os.fsync(f.fileno())
            replace_atomic(temp,last)
        finally: temp.unlink(missing_ok=True)
        saved=last
        if retain_epoch is not None:
            saved=output/f'epoch{retain_epoch}.pt'
            if saved.exists(): raise FileExistsError('Completed epoch checkpoint already exists; no overwrite/replay')
            # Rolling last is replaced atomically later; its completed inode stays
            # immutable through this local hardlink. Fallback is an atomic copy.
            try: os.link(last,saved)
            except OSError:
                tmp=saved.with_suffix('.pt.tmp'); shutil.copyfile(last,tmp)
                with tmp.open('rb+') as f: os.fsync(f.fileno())
                replace_atomic(tmp,saved)
        checkpoint=dict(path=str(saved),last_path=str(last),sha256=sha(saved),bytes=saved.stat().st_size,
            epoch=progress['epoch'],consumed=progress['unique_samples_exposed'],lineage_updates=progress['lineage_updates'])
        return checkpoint
    try:
        # Epoch1 formal48 already exists. Only the separately approved diagnosis
        # and8-case16-denoising supplement run here before any new update.
        diag_results['1']=protected('epoch1_diagnostics',lambda:diagnostics.run_epoch(root,config,ctx,1))
        for epoch in (2,3):
            progress=new_progress(epoch,progress['lineage_updates'],stage); ctx.progress=progress
            atomic_json(output/'progress.json',progress); epoch_start=time.monotonic()
            for bi,indices in enumerate(orders[epoch]):
                if bi%25==0: check_storage(root,config)
                arrays=store.batch([train[i] for i in indices]); n=len(indices); optimizer.zero_grad(set_to_none=True)
                clean=False; progress['complete_optimizer_boundary']=False
                progress['pending_attempt']=dict(epoch=epoch,batch_index=bi,sample_ids=[train[i]['sample_id'] for i in indices],optimizer_application='not_started')
                progress['batches_consumed']=bi+1; progress['unique_samples_exposed']+=n; stage['batches_consumed']+=1; stage['unique_samples_exposed']+=n
                progress['additional_stage']=deepcopy(stage); atomic_json(output/'progress.json',progress)
                loss_total=0.; details=[]
                for start in range(0,n,4):
                    size=min(4,n-start); batch={k:torch.as_tensor(v[start:start+size],device='cuda') for k,v in arrays.items()}
                    progress['model_forward_attempts']+=1
                    with torch.autocast('cuda',dtype=torch.float16,enabled=True): loss,noise=conditioning_residual_loss(model,batch,rngs=rngs)
                    progress['model_forward_completed']+=1; scaler.scale(loss*(size/n)).backward(); loss_total+=float(loss.detach())*(size/n)
                    progress['backward_microbatches']+=1; progress[noise['history_branch']+'_microbatches']+=1; details.append(noise)
                scaler.unscale_(optimizer); params=list(model.parameters())
                if any(p.grad is None for p in params): raise FloatingPointError('Missing full-network gradient')
                if any(not bool(torch.isfinite(p).all()) for p in params): raise FloatingPointError('Nonfinite model weight')
                finite=all(bool(torch.isfinite(p.grad).all()) for p in params)
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)) if finite else None
                progress['pending_attempt']['optimizer_application']='started'; atomic_json(output/'progress.json',progress)
                outcome=scaled_optimizer_step(scaler,optimizer,application_count=lambda:applied[0])
                if outcome['applied']:
                    progress['successful_updates']+=1; stage['successful_updates']+=1; progress['lineage_updates']+=1; consecutive=0
                else:
                    progress['AMP_skips']+=1; stage['AMP_skips']+=1; consecutive=progress['consecutive_AMP_overflows']+1
                progress['consecutive_AMP_overflows']=consecutive; optimizer.zero_grad(set_to_none=True)
                if any(not bool(torch.isfinite(p).all()) for p in params): raise FloatingPointError('Nonfinite parameter after optimizer')
                clean=True; progress['complete_optimizer_boundary']=True; progress['pending_attempt']=None
                progress.update(additional_stage=deepcopy(stage),wall_seconds=time.monotonic()-epoch_start,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                atomic_json(output/'progress.json',progress)
                with (output/'losses.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(epoch=epoch,batch=bi,loss=loss_total,gradient_norm=norm,draws=n,**outcome,branches=[d['history_branch'] for d in details]))+'\n')
                if progress['peak_cuda_reserved_bytes']>int(config.get('max_cuda_bytes',6*1024**3)): raise RuntimeError('Existing6GiB CUDA envelope exceeded')
                if consecutive>=16: raise FloatingPointError('16consecutiveAMPoverflows; noautomaticretry/newdraw')
                if outcome['applied'] and stage['successful_updates']%100==0: save('existing100successboundary')
            if progress['unique_samples_exposed']!=30935 or progress['batches_consumed']!=1934: raise RuntimeError('Incomplete unique epoch; no padding/retry')
            progress['epoch_complete']=True; atomic_json(output/'progress.json',progress)
            checkpoint=save(f'new_data_epoch{epoch}_complete',retain_epoch=epoch)
            formal=protected(f'epoch{epoch}_formal48',lambda:ar1.evaluate(root,config,model,source,spec,f'epoch{epoch}',store))
            diag_results[str(epoch)]=protected(f'epoch{epoch}_diagnostics',lambda:diagnostics.run_epoch(root,config,ctx,epoch))
            result=dict(epoch=epoch,progress=deepcopy(progress),checkpoint=checkpoint,formal_metrics=str(root/f'evaluation/epoch{epoch}/metrics.json'),
                diagnostics=diag_results[str(epoch)],formal_wall_seconds=formal['wall_seconds'])
            epoch_results.append(result); atomic_json(output/f'epoch{epoch}-result.json',result)
        if stage['unique_samples_exposed']!=61870 or stage['batches_consumed']!=3868: raise RuntimeError('Exactly two additional epochs required')
        summary=dict(complete=True,starting_checkpoint=restoration['parent_checkpoint'],restoration_proof=str(output/'restore-proof.json'),
            epochs=epoch_results,formal48={'1':str(data_root/'evaluation/after/metrics.json'),'2':str(root/'evaluation/epoch2/metrics.json'),'3':str(root/'evaluation/epoch3/metrics.json')},
            diagnostics=diag_results,diagnostic_spec=diagnostic_spec,actual_additional_exposures=stage['unique_samples_exposed'],attempted_batches=stage['batches_consumed'],
            successful_updates=stage['successful_updates'],AMP_skips=stage['AMP_skips'],final_lineage_updates=progress['lineage_updates'],
            rng_preservation=str(output/'rng-preservation.json'),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            wall_seconds=time.monotonic()-began,physics_steps=0,new_samples=0,fourth_epoch_authorized=False)
        atomic_json(output/'result.json',summary); return summary
    except BaseException as error:
        if clean:
            try: checkpoint=save('failure_complete_optimizer_boundary')
            except BaseException as saving: checkpoint=dict(saved=False,error=repr(saving))
        atomic_json(output/'failure.json',dict(error_type=type(error).__name__,error=str(error),progress=progress,checkpoint=checkpoint,
            optimizer_state_uncertain=not clean,automatic_restart=False,wall_seconds=time.monotonic()-began))
        raise
    finally:
        hook.remove(); optimizer.zero_grad(set_to_none=True); store.cache.clear()
        del parent,model,optimizer,scaler,rngs,generator; torch.cuda.empty_cache()
