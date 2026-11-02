"""One authorized three-arm definition test; no extra model work on IO retries."""
import os,sys,time,io,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(REPO))
from common import read,sha,write,write_bytes,output_io,load,utc,HMF,save_raw
from authorization import issue_permit
class BoundedLog(io.TextIOBase):
    def __init__(self,path):self.path=path
    @property
    def encoding(self):return 'utf-8'
    def write(self,value):
        b=str(value).encode('utf-8');old=self.path.read_bytes() if self.path.exists() else b''
        if len(old)+len(b)>65536:raise RuntimeError('Worker text log cap')
        write_bytes(self.path,old+b);return len(value)
    def flush(self):pass
def saved_scoring_parity(permit,torch):
    import numpy as np
    from scoring_adapter import load_module
    m=load_module('_definition_original_hid_metrics',permit.config['hid_metrics_path']);samples=[]
    for item in read(ROOT/'scoring-parity-inputs.json'):
        with np.load(item['prediction_path'],allow_pickle=False) as z:prediction=z['raw'][item['horizon']-1].copy()
        with np.load(item['truth_path'],allow_pickle=False) as z:
            a=item['anchor'];truth=z['rgb'][a+item['horizon']].copy();last=z['rgb'][a].copy()
        samples.append(dict(sample_id=item['sample_id'],letter=item['letter'],prediction=prediction,truth=truth,last=last))
    proof=m.check_saved_samples(samples,torch=torch,device=torch.device('cuda'));write(ROOT/'scoring-parity.json',proof)
    if not proof['passed']:raise RuntimeError('Original metric/qualification parity failed')
    permit.scoring_parity_passed=True;return proof
def evaluate(permit,model,source,label,ledger,torch):
    import numpy as np
    from types import SimpleNamespace
    from scoring_adapter import score_native_batch,truth_series
    from objectives import masks
    ev=load('_definition_hmf_eval',HMF/'evaluation.py');old=ev.helper().old();store=old._ar1().RawStore(max_bytes=48*1024**2)
    spec=read(ROOT/'evaluation-spec.json');std=read(HMF/'fit-standard.json');features={x['sample_id']:x['truth_features'] for x in read(HMF/'GT-measurement-calibration.json')['rows']}
    rows=[];raws=[];evidence=[];costs=[];rgb_risks=[];started=time.perf_counter();model.eval()
    with ledger.phase('eval/'+label):
        for mode in ['real_history','autoregressive']:
            for batch in spec['batches']:
                rr=batch['rows'];bid=batch['batch_id'];seeds=batch['seeds'];inputs=old.prepare_inputs(store,rr,mode)
                ledger.active=dict(stage='evaluation',arm=label,mode=mode,batch=bid);ledger.persist()
                prediction,noise=ev.helper().native_rollout(SimpleNamespace(model=model,source=source),inputs,seeds)
                evidence.append(dict(mode=mode,batch_id=bid,inputs=old._inputs_evidence(rr,inputs,seeds),noise=noise))
                truths=[]
                for slot,row in enumerate(rr):
                    record=save_raw(ROOT/f'evaluation/{label}/{mode}-b{bid:02d}-s{slot}.npz',prediction[slot]);raws.append(dict(**record,sample_id=row['sample_id'],letter=row['letter'],selection_role=row['selection_role'],mode=mode,batch_id=bid,slot=slot))
                    truths.append(truth_series(row,store.arrays(row),features[row['sample_id']]))
                gpu=torch.as_tensor(prediction,device='cuda',dtype=torch.float32)
                scored=score_native_batch(permit,label,batch,mode,gpu,prediction,truths);rows.extend(scored['rows']);costs.append(scored['scoring'])
                with torch.no_grad():
                    flat=gpu.reshape(-1,3,96,96);gt=torch.as_tensor(np.stack([t['targets'] for t in truths]),device='cuda',dtype=torch.float32).reshape(-1,3,96,96)/255
                    p,u=masks(flat,True);g=masks(gt);background=(g.sum(1,keepdim=True)<.001).float();bgerror=((flat-gt).square()*background).sum((1,2,3))/(background.sum((1,2,3))*3).clamp_min(1)
                    ar=p.sum((-2,-1));ga=g.sum((-2,-1))
                    for slot,row in enumerate(rr):
                        for h in range(16):
                            i=slot*16+h;rgb_risks.append(dict(sample_id=row['sample_id'],letter=row['letter'],role=row['selection_role'],mode=mode,horizon=h+1,
                                RGB_outside_01_fraction=float(((flat[i]<0)|(flat[i]>1)).float().mean()),background_MSE=float(bgerror[i]),u_saturated_fraction=float(((u[i]<=0)|(u[i]>=1)).float().mean()),u_strict_outside_fraction=float(((u[i]<0)|(u[i]>1)).float().mean()),u_exact_boundary_fraction=float(((u[i]==0)|(u[i]==1)).float().mean()),soft_mask_areas=ar[i].cpu().tolist(),GT_soft_mask_areas=ga[i].cpu().tolist(),
                                gray_empty=bool(ar[i,0]<1),blue_empty=bool(ar[i,1]<1)))
                del gpu,prediction,flat,gt,p,u,g;ledger.active=None;ledger.persist()
    copies=read(permit.config['copy_reference']);support={role:ev.assess(rows,copies,std,role) for role in ['train','seen_val','original_failure']}
    parent=read(permit.config['E2_metrics'])
    if evidence!=parent['evidence']:raise RuntimeError('Original eval batch/slot/inputs/actual epsilon changed')
    value=dict(label=label,rows=rows,raw_files=raws,evidence=evidence,support=support,predictions=len(raws)*16,wall_seconds=time.perf_counter()-started,scoring_cost=costs,
        auxiliary_RGB_mask_diagnostics=rgb_risks,scope='Original train12/seen-val24/old3 RH independent single-step and unmodified raw AR16; no future GT feedback in AR.')
    write(ROOT/f'evaluation/{label}/metrics.json',value);store.cache.clear();output_io().guard(refresh=True)
    return dict(metrics=str(ROOT/f'evaluation/{label}/metrics.json'),wall_seconds=value['wall_seconds'])
def train(permit,model,dataset,native,arm,opt,scaler,rngs,ledger,coefficients,torch):
    from native_forward import forward_bundle
    from objectives import loss_for
    from tdyssey.native_rgb_r2c3_training import scaled_optimizer_step
    schedule=read(ROOT/'schedule.json');log=[];seconds=0.;applied=[0];handle=opt.register_step_post_hook(lambda *x:applied.__setitem__(0,applied[0]+1))
    reference=read(permit.config['BASE_log'])
    for i,indices in enumerate(schedule['native_h1_batches']):
        if i>=128:raise RuntimeError('Attempt cap')
        output_io().guard();ledger.attempts[arm]+=1;ledger.active=dict(stage='training',arm=arm,attempt=i+1);ledger.persist();start=time.perf_counter()
        model.train();opt.zero_grad(set_to_none=True);batch=dataset.model_batch(indices);images=torch.as_tensor(batch['image'],device='cuda');actions=torch.as_tensor(batch['action'],device='cuda');branches=[];micro=[];finite=[]
        with ledger.phase('train/'+arm):
            for j in range(4):
                x=images[j*4:j*4+4];a=actions[j*4:j*4+4];noise=native.draw_noise(rngs,4,model.device)
                with torch.autocast('cuda',dtype=torch.float16):bundle=forward_bundle(permit,model,x[:,:4],x[:,4],a,noise)
                loss,parts=loss_for(arm,bundle,coefficients);finite.append(bundle['finite']&torch.isfinite(loss));ledger.exposures[arm]+=4
                ledger.grad_begin(arm);scaler.scale(loss*.25).backward();ledger.grad_end(arm)
                p=parts['p'];u=parts['u'];micro.append(dict(native=float(parts['native'].detach()),G=float(parts['G'].detach()),T=float(parts['T'].detach()),W1=float(parts['W1'].detach()),relative_area=float(parts['area'].detach()),total=float(loss.detach()),
                    u_saturated_fraction=float(((u<=0)|(u>=1)).float().mean()),u_strict_outside_fraction=float(((u<0)|(u>1)).float().mean()),u_exact_boundary_fraction=float(((u==0)|(u==1)).float().mean()),soft_mask_areas=p.sum((-2,-1)).detach().cpu().tolist()))
                branches.append(noise.branch);del bundle,loss,parts,p,u
        if not bool(torch.stack(finite).all()):raise FloatingPointError('Nonfinite loss/native contract')
        hashes=rngs.hashes()
        if branches!=reference[i]['branches'] or hashes!=reference[i]['conditioning_rng_hashes']:raise RuntimeError('Training native conditioning noise/order pairing changed')
        scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite parameter gradient')
        update=scaled_optimizer_step(scaler,opt,application_count=lambda:applied[0]);ledger.updates[arm]+=update['applied']
        if update['applied']!=1:raise FloatingPointError('AMP skip; no replacement or hidden update')
        torch.cuda.synchronize();seconds+=time.perf_counter()-start
        log.append(dict(attempt=i+1,updates=ledger.updates[arm],microbatch_losses=micro,gradient_norm_before_clip=float(norm),branches=branches,conditioning_rng_hashes=hashes,scaler=update))
        ledger.active=None;ledger.persist();write(ROOT/f'training/{arm}-log.json',log)
        if i+1 in [32,128]:write(ROOT/f'training/{arm}-throughput-{i+1}.json',dict(arm=arm,at_utc=utc(),updates=i+1,training_seconds=seconds,updates_per_second=(i+1)/seconds,peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),measured_only_no_intermediate_callback=True))
    handle.remove();opt.zero_grad(set_to_none=True)
    from tdyssey.native_rgb_training import _cpu_tree
    buffer=io.BytesIO();torch.save(dict(kind='loss_definition_weights.v1',model=_cpu_tree(model.state_dict()),arm=arm,attempts=128,updates=128,parent_sha256=permit.config['parent_sha256'],config_sha256=permit.config_sha256,coefficients_sha256=sha(ROOT/'coefficients.json'),lineage_updates=11495),buffer)
    checkpoint=output_io().write_bytes(f'training/{arm}-128.pt',buffer.getvalue(),exclusive=True);write(ROOT/f'training/{arm}-128.json',checkpoint)
    result=dict(arm=arm,attempts=128,updates=ledger.updates[arm],native_exposures=ledger.exposures[arm],training_seconds=seconds,final_rng_hashes=hashes,checkpoint=checkpoint)
    write(ROOT/f'training/{arm}-result.json',result);return result
def main():
    permit=issue_permit();c=permit.config;out=output_io();started=time.perf_counter()
    if '--config-sha' not in sys.argv or sys.argv[sys.argv.index('--config-sha')+1]!=permit.config_sha256:raise RuntimeError('Launch identity mismatch')
    if (ROOT/'execution-claim.json').exists():raise RuntimeError('Already claimed; duplicate execution prohibited')
    sys.stdout=BoundedLog(ROOT/'stdout.log');sys.stderr=BoundedLog(ROOT/'stderr.log')
    from ledger import Ledger
    ledger=Ledger(out,c['arms']);results={};failure=None;CPU=None
    try:
        limits=load('_definition_cpu_limits',c['cpu_limits_path']);CPU=limits.constrain_cpu()
        import torch
        CPU.update(limits.torch_limits());ledger.extra.update(CPU=CPU,started_perf=started,model_constructions_attempted=0,model_constructions_completed=0,full_arm_restores=0)
        out.write_json('execution-claim.json',dict(run_id=c['run_id'],config_sha256=permit.config_sha256,pid=os.getpid(),CPU=CPU,at_utc=utc()),exclusive=True)
        for b in c['bindings']:
            if sha(b['path'])!=b['sha256']:raise RuntimeError('Bound scientific source changed: '+b['path'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
        out.guard(refresh=True);results['scoring_parity']=saved_scoring_parity(permit,torch)
        from diagnostics import run as diagnose
        results['diagnosis']=diagnose()
        from state_restore import load_parent,restore_full
        parent=load_parent(permit)
        from tdyssey.gpc_world_model import load_world_model_source,world_model_spec
        from types import SimpleNamespace
        from copy import deepcopy
        source=load_world_model_source(c['source_root']);inner=deepcopy(world_model_spec()['model']);outer={k:inner.pop(k) for k in ['sigma_data','sigma_offset_noise','noise_previous_obs']}
        ledger.extra['model_constructions_attempted']=1;ledger.persist()
        with torch.random.fork_rng(devices=[]):model=source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**inner),**outer)).float().cuda()
        ledger.extra['model_constructions_completed']=1;model.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.))
        hooks=[model.inner_model.register_forward_pre_hook(ledger.before_forward),model.inner_model.register_forward_hook(ledger.after_forward)]
        native=load('_definition_native',HMF/'training_core.py');data=load('_definition_data',HMF/'data.py');dataset=data.FitDataset(HMF)
        torch.cuda.reset_peak_memory_stats();model.load_state_dict(parent['model'],strict=True);model.requires_grad_(True);ledger.extra['preflight_model_only_loads']=1;ledger.persist()
        from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs
        pre_rngs=make_conditioning_rngs(**parent['conditioning_rngs']['seeds'],device=model.device);pre_rngs.load_state_dict(parent['conditioning_rngs'])
        from preflight import run
        results['preflight']=run(permit,model,native,dataset,pre_rngs,ledger);del pre_rngs
        coefficients=read(ROOT/'coefficients.json');usable=results['preflight']['usable_arms']
        skipped={a:'Specified G/T calibration gradient zero/nonfinite; no epsilon substitute, no formula change.' for a in c['arms'] if a not in usable};results['skipped_arms']=skipped
        for arm in usable:
            out.guard(refresh=True);opt,scaler,generator,rngs=restore_full(permit,model,parent,arm,out);ledger.extra['full_arm_restores']+=1;ledger.persist()
            results[arm]=train(permit,model,dataset,native,arm,opt,scaler,rngs,ledger,coefficients,torch)
            del opt,scaler,generator,rngs
            results[arm]['evaluation']=evaluate(permit,model,source,arm,ledger,torch);write(ROOT/f'training/{arm}-result.json',results[arm])
        write(ROOT/'pairing-proof.json',dict(passed=True,executed_arms=usable,skipped=skipped,same_native_order=True,same_native_RNG_as_BASE_per_step=True,full_arm_restores=len(usable),updates_each=128))
        for h in hooks:h.remove()
        results['peak_cuda']=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved());del model,parent;dataset.close();torch.cuda.empty_cache()
        write(ROOT/'model-stages-complete.json',dict(run_id=c['run_id'],config_sha256=permit.config_sha256,all_arms_complete=len(usable)==3,executed_arms=usable,skipped=skipped,ledger=ledger.snapshot(),results=results,at_utc=utc()))
        from report import generate
        results['report']=generate()
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at_utc=utc(),phase=ledger.active);write(ROOT/'failure.json',failure,emergency=True)
    complete=failure is None
    value=dict(run_id=c['run_id'],config_sha256=permit.config_sha256,complete=complete,failed=not complete,failure=failure,results=results,ledger=ledger.snapshot(),CPU=CPU,
        own_CPU_seconds=dict(user=os.times().user,system=os.times().system),wall_seconds=time.perf_counter()-started,physical_steps=0,new_data=0,callback_sent=False,state='waiting_final_audit_and_callback',
        io_replace_retries=out.io_retry_count,storage=out.guard(final_metadata=True,refresh=True),finished_utc=utc())
    write(ROOT/'completion.json',value,emergency=True);return int(not complete)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        write(ROOT/'bootstrap-failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at_utc=utc()),emergency=True);raise
