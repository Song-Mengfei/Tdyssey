"""User-authorized recovery: reuse saved work, finish same four-arm protocol."""
import sys,os,time,json,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(REPO))
from continuation_io import install
from common import read,write,sha,load,utc,output_io,save_raw,HMF
from authorization import issue_permit
from continuation_ledger import ResumeLedger

def resume_snr_evaluation(permit,model,source,ledger,torch):
    import numpy as np
    from copy import deepcopy
    from types import SimpleNamespace
    from scoring_adapter import score_native_batch,truth_series
    ev=load('_loss_resume_hmf_eval',HMF/'evaluation.py');old=ev.helper().old();store=old._ar1().RawStore(max_bytes=48*1024**2)
    spec=read(ROOT/'evaluation-spec.json');std=read(HMF/'fit-standard.json')
    features={x['sample_id']:x['truth_features'] for x in read(HMF/'GT-measurement-calibration.json')['rows']}
    reference=read(permit.config['E2_metrics']);reference_evidence={(e['mode'],e['batch_id']):e for e in reference['evidence']}
    reused={(x['mode'],x['batch_id']):x for x in read(ROOT/'continuation-config.json')['reused_batches']}
    rows=[];raws=[];evidence=[];provenance=[];costs=[];started=time.perf_counter();model.eval()
    with ledger.phase('eval/SNR'):
        for mode in ('real_history','autoregressive'):
            for batch in spec['batches']:
                rr=batch['rows'];bid=batch['batch_id'];seeds=batch['seeds'];inputs=old.prepare_inputs(store,rr,mode)
                ledger.active=dict(stage='evaluation',arm='SNR',mode=mode,batch=bid);ledger.persist();reuse=reused.get((mode,bid))
                if reuse is not None:
                    values=[];records=[]
                    for slot,row in enumerate(rr):
                        rec=next(x for x in reuse['files'] if x['slot']==slot)
                        if sha(rec['path'])!=rec['sha256']:raise RuntimeError('Previously saved SNR raw changed')
                        with np.load(rec['path'],allow_pickle=False) as z:value=z['raw'].copy()
                        if value.dtype!=np.float32 or value.shape!=(16,3,96,96) or not np.isfinite(value).all():raise RuntimeError('Invalid existing SNR raw')
                        values.append(value);records.append(dict(path=rec['path'],bytes=rec['bytes'],sha256=rec['sha256']))
                    prediction=np.stack(values);noise=deepcopy(reference_evidence[mode,bid]['noise'])
                    # Old frozen capture_noise checked each actual tensor before
                    # the archive could persist. Its in-memory receipt was lost;
                    # independently reconstruct hashes, without any model call.
                    for seed,n in zip(seeds,noise):
                        expected=torch.randn(len(rr),3,96,96,device='cuda',dtype=torch.float32,generator=torch.Generator(device='cuda').manual_seed(int(seed)))
                        a=expected.cpu().contiguous().numpy()
                        if hashlib.sha256(a.tobytes()).hexdigest()!=n['actual_initial_epsilon_sha256']:raise RuntimeError('Reused noise independent replay mismatch')
                    provenance.append(dict(mode=mode,batch_id=bid,reused_saved_archives=True,new_model_calls=0,
                        original_actual_observation_receipt_was_not_durable=True,original_frozen_capture_checked_actual_before_archive=True,
                        canonical_noise_hashes_independently_replayed=True,not_a_new_native_noise_observation=True))
                else:
                    prediction,noise=ev.helper().native_rollout(SimpleNamespace(model=model,source=source),inputs,seeds);records=[]
                    for slot,row in enumerate(rr):records.append(save_raw(ROOT/f'evaluation/SNR/{mode}-b{bid:02d}-s{slot}.npz',prediction[slot]))
                    provenance.append(dict(mode=mode,batch_id=bid,reused_saved_archives=False,new_native_noise_observed=True))
                evidence.append(dict(mode=mode,batch_id=bid,inputs=old._inputs_evidence(rr,inputs,seeds),noise=noise));truths=[]
                for slot,(row,rec) in enumerate(zip(rr,records)):
                    raws.append(dict(**rec,sample_id=row['sample_id'],letter=row['letter'],selection_role=row['selection_role'],mode=mode,batch_id=bid,slot=slot))
                    truths.append(truth_series(row,store.arrays(row),features[row['sample_id']]))
                gpu=torch.as_tensor(prediction,device='cuda',dtype=torch.float32)
                scored=score_native_batch(permit,'SNR',batch,mode,gpu,prediction,truths);rows.extend(scored['rows']);costs.append(scored['scoring'])
                del gpu,prediction;ledger.active=None;ledger.persist()
    if evidence!=reference['evidence']:raise RuntimeError('SNR canonical eval pairing mismatch')
    copies=read(permit.config['copy_reference']);support={role:ev.assess(rows,copies,std,role) for role in ('train','seen_val','original_failure')}
    result=dict(label='SNR',rows=rows,raw_files=raws,evidence=evidence,noise_provenance=provenance,support=support,predictions=len(raws)*16,
        wall_seconds=time.perf_counter()-started,scoring_cost=costs,scope='Same original train12/seen-val24/old3;8 archives reused, remainder resumed; no seed/batch changes')
    write(ROOT/'evaluation/SNR/metrics.json',result);store.cache.clear();output_io().guard(refresh=True)
    return dict(metrics=str(ROOT/'evaluation/SNR/metrics.json'),wall_seconds=result['wall_seconds'],reused_archives=8)

def main():
    io=install();cfg=read(ROOT/'continuation-config.json');release=read(ROOT/'continuation-release.json')
    digest=sha(ROOT/'continuation-config.json')
    if release['config_sha256']!=digest or not release['resume_authorized'] or '--config-sha' not in sys.argv or sys.argv[sys.argv.index('--config-sha')+1]!=digest:
        raise RuntimeError('Continuation release/launch identity mismatch')
    if (ROOT/'continuation-execution-claim.json').exists():raise RuntimeError('Continuation already claimed; no duplicate')
    permit=issue_permit();original=read(ROOT/'completion.json');ledger=ResumeLedger(io,original['ledger'])
    started=time.perf_counter();failure=None;results={};CPU=None
    original_worker=load('_loss_resume_original_worker',ROOT/'worker.py')
    sys.stdout=original_worker.BoundedLog(ROOT/'continuation-stdout.log');sys.stderr=original_worker.BoundedLog(ROOT/'continuation-stderr.log')
    try:
        limits=load('_loss_resume_cpu_limits',permit.config['cpu_limits_path']);CPU=limits.constrain_cpu()
        import torch
        CPU.update(limits.torch_limits());ledger.extra['CPU']=CPU;ledger.extra['started_perf']=started
        ledger.extra['original_active_worker_wall_seconds']=original['wall_seconds'];ledger.extra['continuation_user_authorized']=True
        io.write_json('continuation-execution-claim.json',dict(run_id=cfg['run_id'],config_sha256=digest,pid=os.getpid(),CPU=CPU,at_utc=utc()),exclusive=True)
        for b in cfg['bindings']:
            if sha(b['path'])!=b['sha256']:raise RuntimeError('Continuation binding changed: '+b['path'])
        for b in permit.config['bindings']:
            if sha(b['path'])!=b['sha256']:raise RuntimeError('Original scientific/frozen source changed: '+b['path'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        permit.scoring_parity_passed=read(ROOT/'scoring-parity.json')['passed']
        if not permit.scoring_parity_passed:raise RuntimeError('Original scoring parity was not passed')
        from state_restore import load_parent,restore_full
        parent=load_parent(permit)
        from tdyssey.gpc_world_model import load_world_model_source,world_model_spec
        from copy import deepcopy
        from types import SimpleNamespace
        source=load_world_model_source(permit.config['source_root']);inner=deepcopy(world_model_spec()['model']);outer={k:inner.pop(k) for k in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
        ledger.extra['model_constructions_attempted']+=1;ledger.persist()
        with torch.random.fork_rng(devices=[]):model=source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**inner),**outer)).float().cuda()
        ledger.extra['model_constructions_completed']+=1;model.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.))
        hooks=[model.inner_model.register_forward_pre_hook(ledger.before_forward),model.inner_model.register_forward_hook(ledger.after_forward)]
        checkpoint=read(ROOT/'training/SNR-128.json')
        if sha(checkpoint['path'])!=checkpoint['sha256']:raise RuntimeError('SNR final weights changed')
        saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
        ledger.extra['continuation_SNR_final_model_only_loads']=1;ledger.persist();torch.cuda.reset_peak_memory_stats()
        results['SNR_evaluation']=resume_snr_evaluation(permit,model,source,ledger,torch)
        snr=read(ROOT/'training/SNR-result.json');snr['evaluation']=results['SNR_evaluation'];write(ROOT/'training/SNR-result.json',snr)
        native=load('_loss_resume_native',HMF/'training_core.py');data=load('_loss_resume_data',HMF/'data.py');dataset=data.FitDataset(HMF)
        base_log=read(ROOT/'training/BASE-log.json')
        for arm in ('GDL','FFL'):
            io.guard(refresh=True);opt,scaler,generator,rngs=restore_full(permit,model,parent,arm,io);ledger.extra['full_arm_restores']+=1;ledger.persist()
            results[arm]=original_worker.train(permit,model,parent,dataset,native,arm,opt,scaler,generator,rngs,ledger,torch)
            log=read(ROOT/f'training/{arm}-log.json')
            if any(a['branches']!=b['branches'] or a['conditioning_rng_hashes']!=b['conditioning_rng_hashes'] for a,b in zip(base_log,log)):raise RuntimeError('Continuation native pairing diverged')
            del opt,scaler,generator,rngs
            results[arm]['evaluation']=original_worker.evaluate(permit,model,source,arm,ledger,torch);write(ROOT/f'training/{arm}-result.json',results[arm])
        write(ROOT/'pairing-proof.json',dict(passed=True,arms=permit.config['arms'],same_native_order=True,same_native_rng_per_step=True,full_arm_restores=4,updates_each=128,
            resumed_by_direct_user=True,precrash_SNR_reused_archives=8,noise_receipt_provenance='See evaluation/SNR/metrics.json.noise_provenance'))
        for h in hooks:h.remove()
        results['peak_cuda']=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        del model,parent;dataset.close();torch.cuda.empty_cache()
        from report import generate
        results['report']=generate()
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at_utc=utc(),phase=ledger.active)
        write(ROOT/'continuation-failure.json',failure,emergency=True)
    final=dict(run_id=cfg['run_id'],config_sha256=digest,original_config_sha256=permit.config_sha256,complete=failure is None,failed=failure is not None,failure=failure,
        results=results,ledger=ledger.snapshot(),CPU=CPU,own_CPU_seconds=dict(user=os.times().user,system=os.times().system),continuation_wall_seconds=time.perf_counter()-started,
        original_worker_wall_seconds=original['wall_seconds'],combined_active_worker_wall_seconds=original['wall_seconds']+time.perf_counter()-started,
        io_replace_retries=io.io_retry_count,original_failure_retained=True,new_physical_steps=0,new_data=0,callback_sent=False,state='waiting_callback',storage=io.guard(final_metadata=True,refresh=True),finished_utc=utc())
    write(ROOT/'continuation-completion.json',final,emergency=True);write(ROOT/'continuation-storage-final.json',io.guard(final_metadata=True,refresh=True),emergency=True)
    return int(failure is not None)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        install();write(ROOT/'continuation-bootstrap-failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at_utc=utc()),emergency=True);raise

