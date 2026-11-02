"""One authorized, fail-closed paired loss experiment; no automatic resume."""
import sys, os, time, traceback, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT.parents[1]))
from common import *
limits=load('_loss_cpu_limits',ROOT.parent/'hmf-inference-diagnostics-20261102/resource_limits.py')
CPU=limits.constrain_cpu()
import numpy as np
import torch
CPU.update(limits.torch_limits())
from contextlib import contextmanager
from copy import deepcopy
from collections import Counter,defaultdict
from types import SimpleNamespace
from rollout_loss import rollout,objective

class Ledger:
    def __init__(self,c):self.c=c;self.scope='bootstrap';self.counts=defaultdict(Counter);self.active=None;self.attempts=Counter();self.updates=Counter();self.backward=Counter();self.exposures=Counter();self.started=time.perf_counter()
    @contextmanager
    def phase(self,key):
        old=self.scope;self.scope=key
        try:yield
        finally:self.scope=old
    def pre(self,module,args):
        n=len(args[0]);total=sum(v['images_attempted'] for v in self.counts.values());batch=sum(v['batches_attempted'] for v in self.counts.values())
        cap=self.c['caps'];section=self.scope.split('/')[0]
        section_images=sum(v['images_attempted'] for k,v in self.counts.items() if k.split('/')[0]==section)
        section_cap=cap['preflight_image_forwards'] if section=='preflight' else cap['eval_image_forwards'] if section=='eval' else cap['train_image_forwards']
        if total+n>cap['all_image_forwards'] or batch+1>cap['all_batch_forwards'] or section_images+n>section_cap:raise RuntimeError('Forward budget exhausted')
        self.counts[self.scope]['images_attempted']+=n;self.counts[self.scope]['batches_attempted']+=1
    def post(self,module,args,output):
        self.counts[self.scope]['images_completed']+=len(args[0]);self.counts[self.scope]['batches_completed']+=1
    def snapshot(self):return dict(counts={k:dict(v) for k,v in self.counts.items()},attempts=dict(self.attempts),successful_updates=dict(self.updates),backward_calls=dict(self.backward),target_exposures=dict(self.exposures),active=self.active,wall_seconds=time.perf_counter()-self.started)
    def persist(self,phase,**extra):write(ROOT/'status.json',dict(phase=phase,pid=os.getpid(),at_utc=utc(),ledger=self.snapshot(),**extra))

def digest_tree(x):
    from tdyssey.native_rgb_r2c1r1_training import _tree_digest
    return _tree_digest(x,torch)

def create_model(c):
    from tdyssey.gpc_world_model import load_world_model_source,world_model_spec
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no fallback')
    source=load_world_model_source(c['source_root']);inner=deepcopy(world_model_spec()['model']);outer={k:inner.pop(k) for k in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
    with torch.random.fork_rng(devices=[]):model=source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**inner),**outer)).float().cuda()
    model.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.))
    return model,source

def restore(model,parent,label):
    from tdyssey.raw_world_model_training import OPTIMIZER,make_generator
    from tdyssey.native_rgb_training import restore_rng,capture_rng
    from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs
    # load_state_dict may share GPU storage if passed CUDA tensors; parent stays CPU.
    model.load_state_dict(parent['model'],strict=True);model.requires_grad_(True)
    opt=torch.optim.AdamW(model.parameters(),**dict(OPTIMIZER,lr=1e-5));opt.load_state_dict(deepcopy(parent['optimizer']))
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=128);scaler.load_state_dict(deepcopy(parent['scaler']))
    generator=make_generator(5478,model.device);restore_rng(parent['global_rng'],generator)
    rngs=make_conditioning_rngs(**parent['conditioning_rngs']['seeds'],device=model.device);rngs.load_state_dict(parent['conditioning_rngs'])
    proof=dict(model=digest_tree(model.state_dict())==digest_tree(parent['model']),optimizer=digest_tree(opt.state_dict())==digest_tree(parent['optimizer']),scaler=digest_tree(scaler.state_dict())==digest_tree(parent['scaler']),global_rng=digest_tree(capture_rng(generator))==digest_tree(parent['global_rng']),conditioning_rngs=digest_tree(rngs.state_dict())==digest_tree(parent['conditioning_rngs']))
    write(ROOT/f'restore-{label}.json',proof)
    if not all(proof.values()) or any(g['lr']!=1e-5 for g in opt.param_groups):raise RuntimeError('Complete E2 restore mismatch')
    return opt,scaler,generator,rngs

def aux_inputs(dataset,block):
    d=dataset.cache[block['segment_id']];p=block['phase']
    image=torch.as_tensor(d['rgb'][p:p+8].copy(),device='cuda',dtype=torch.float32)[None]/255
    act=torch.as_tensor(d['action'][p:p+7].copy(),device='cuda',dtype=torch.float32)[None]
    assert image.shape==(1,8,3,96,96) and act.shape==(1,7,2)
    return image[:,:4],image[:,4:],act

def preflight(model,source,dataset,schedule,ledger):
    model.eval();model.zero_grad(set_to_none=True);history,targets,actions=aux_inputs(dataset,schedule['aux_blocks'][schedule['aux_block_indices'][0]])
    sampler=source.DiffusionSampler(model,source.DiffusionSamplerConfig(**read(ROOT/'config.json')['sampler']))
    eps=[];ref=[];h=history
    with ledger.phase('preflight/native'),torch.no_grad(),torch.autocast('cuda',enabled=False):
        for k in range(4):
            seed=2026110290+k
            eps.append(torch.randn(1,3,96,96,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed)))
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.cuda.manual_seed(seed);p,_=sampler.sample(h,actions[:,k:k+4].contiguous())
            ref.append(p);h=torch.cat((h[:,1:],p[:,None]),dim=1)
    with ledger.phase('preflight/surrogate'),torch.autocast('cuda',enabled=False):
        out=rollout(model,history,actions,eps,sampler.sigmas,recompute_context=lambda:ledger.phase('preflight/recompute'))
        parity=[dict(step=k+1,exact=torch.equal(a,b),max_abs=float((a-b).abs().max())) for k,(a,b) in enumerate(zip(out,ref))]
        write(ROOT/'preflight-forward.json',parity)
        if not all(x['exact'] for x in parity):raise RuntimeError('Native/surrogate forward mismatch; no training')
        ledger.backward['preflight_gradient_attempted']+=1
        g=torch.autograd.grad((out[-1]-targets[:,-1]).square().mean(),out[0],retain_graph=True)[0]
        ledger.backward['preflight_gradient_completed']+=1
        grad_to_first=float(torch.linalg.vector_norm(g));finite=bool(torch.isfinite(g).all())
        if not finite or grad_to_first<=0:raise RuntimeError('No finite H4->predictedH1 gradient')
        loss=objective(out,targets);ledger.backward['preflight_backward_attempted']+=1;loss.backward();ledger.backward['preflight_backward_completed']+=1
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None]));good=bool(torch.isfinite(norm)) and float(norm)>0
    result=dict(passed=good,forward_parity=parity,H4_gradient_to_predicted_H1=grad_to_first,parameter_gradient_norm=float(norm),aux_objective=float(loss.detach()),parameter_count=sum(p.numel() for p in model.parameters()),optimizer_updates=0,gradient_is_quantization_STE=True,ledger=ledger.snapshot())
    write(ROOT/'preflight.json',result)
    model.zero_grad(set_to_none=True)
    if not good:raise RuntimeError('Invalid auxiliary gradient')
    return result

def evaluate(model,source,label,ledger):
    ev=load('_loss_hmf_eval',HMF/'evaluation.py');old=ev.helper().old();store=old._ar1().RawStore(max_bytes=48*1024**2)
    spec=read(ROOT/'evaluation-spec.json');std=read(HMF/'fit-standard.json');cal=read(ev.CAL)
    truth={x['sample_id']:x['truth_features'] for x in read(HMF/'GT-measurement-calibration.json')['rows']}
    rows=[];copies=[];raws=[];evidence=[];beg=time.perf_counter();model.eval()
    # Inputs and inference RNG are independent of training; actual epsilon hashes retained.
    with ledger.phase('eval/'+label):
        for mode in ('real_history','autoregressive'):
            for batch in spec['batches']:
                rr=batch['rows'];bid=batch['batch_id'];seeds=batch['seeds'];inputs=old.prepare_inputs(store,rr,mode)
                ledger.active=dict(stage='evaluation',label=label,mode=mode,batch=bid,predictions_reserved=len(rr)*16);ledger.persist('evaluation')
                prediction,noise=ev.helper().native_rollout(SimpleNamespace(model=model,source=source),inputs,seeds)
                evidence.append(dict(mode=mode,batch_id=bid,inputs=old._inputs_evidence(rr,inputs,seeds),noise=noise))
                for slot,row in enumerate(rr):
                    record=save_raw(ROOT/f'evaluation/{label}/{mode}-b{bid:02d}-s{slot}.npz',prediction[slot]);raws.append(dict(**record,sample_id=row['sample_id'],letter=row['letter'],selection_role=row['selection_role'],mode=mode,batch_id=bid,slot=slot))
                    d=store.arrays(row);a=int(row['anchor']);gt=d['rgb'][a+1:a+17];pf=[ev.features(im,cal) for im in prediction[slot]]
                    for k in range(16):rows.append(ev._scored(row,prediction[slot,k],gt[k],d['rgb'][a],pf,truth[row['sample_id']],k+1,mode,cal,std))
                    if label=='E2':
                        rawcopy=d['rgb'][a:a+16] if mode=='real_history' else np.repeat(d['rgb'][a:a+1],16,axis=0)
                        cf=[ev.features(im,cal) for im in rawcopy]
                        for k in range(16):copies.append(ev._scored(row,rawcopy[k],gt[k],d['rgb'][a],cf,truth[row['sample_id']],k+1,mode,cal,std))
                ledger.active=None;ledger.persist('evaluation')
    if label=='E2':write(ROOT/'copy-reference.json',copies)
    else:copies=read(ROOT/'copy-reference.json')
    support={role:ev.assess(rows,copies,std,role) for role in ('train','seen_val','original_failure')}
    result=dict(label=label,rows=rows,raw_files=raws,evidence=evidence,support=support,scope='fixed train12/seen-val24/old3, not full train48',wall_seconds=time.perf_counter()-beg,predictions=len(raws)*16)
    write(ROOT/f'evaluation/{label}/metrics.json',result);store.cache.clear();guard(full=True)
    return dict(path=str(ROOT/f'evaluation/{label}/metrics.json'),wall_seconds=result['wall_seconds'])

def save_checkpoint(model,opt,scaler,generator,rngs,parent,arm,step,ledger,full):
    from tdyssey.native_rgb_training import _cpu_tree,capture_rng
    path=checked(ROOT/f'training/{arm}-{step}.pt');guard(128*1024**2 if full else 48*1024**2,full=True)
    if path.exists():raise FileExistsError(path)
    payload=dict(kind='loss_ar4_full.v1' if full else 'loss_ar4_weights.v1',model=_cpu_tree(model.state_dict()),arm=arm,attempt=step,updates=ledger.updates[arm],config_sha256=sha(ROOT/'config.json'),parent_sha256=read(ROOT/'config.json')['parent_sha256'],lineage_updates=11367+ledger.updates[arm])
    if full:payload.update(optimizer=_cpu_tree(opt.state_dict()),scaler=_cpu_tree(scaler.state_dict()),global_rng=capture_rng(generator),conditioning_rngs=rngs.state_dict(),schedule_cursor=step,ledger=ledger.snapshot(),legacy_scheduler_provenance=parent['legacy_scheduler_provenance'])
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
    rec=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,full_state=full);write(path.with_suffix('.json'),rec);return rec

def train_arm(arm,model,source,parent,dataset,schedule,c,ledger):
    from tdyssey.native_rgb_training import capture_rng,restore_rng
    from tdyssey.native_rgb_r2c3_training import scaled_optimizer_step
    native=load('_loss_native_core',HMF/'training_core.py')
    opt,scaler,generator,rngs=restore(model,parent,arm)
    aux_rng=torch.Generator(device='cuda').manual_seed(schedule['aux_seed']);sampler=source.DiffusionSampler(model,source.DiffusionSamplerConfig(**c['sampler']))
    applied=[0]
    def hook(*args):applied[0]+=1
    handle=opt.register_step_post_hook(hook);log=[];evaluations={};checkpoints=[];training_seconds=0.
    for i,indices in enumerate(schedule['native_h1_batches']):
        step=i+1;guard();ledger.attempts[arm]+=1;ledger.active=dict(stage='training',arm=arm,attempt=step,indices=indices,aux_block=schedule['aux_block_indices'][i] if arm=='B' else None)
        ledger.persist('training');t0=time.perf_counter();model.train();opt.zero_grad(set_to_none=True)
        batch=dataset.model_batch(indices);images=torch.as_tensor(batch['image'],device='cuda');actions=torch.as_tensor(batch['action'],device='cuda')
        def observe(event,micro,n):
            if event=='loss_complete':ledger.exposures[arm+'_native']+=n
            if event=='backward_attempt':ledger.backward[arm+'_native_attempted']+=1
            if event=='backward_complete':ledger.backward[arm+'_native_completed']+=1
        with ledger.phase('train/'+arm+'/native'):result=native.microbatch_backward(model,images,actions,rngs,scaler,observer=observe)
        if not bool(result['forward_finite']):raise FloatingPointError('Native loss nonfinite/range contract failed')
        aux_loss=None
        if arm=='B':
            model.eval();block=schedule['aux_blocks'][schedule['aux_block_indices'][i]];history,targets,acts=aux_inputs(dataset,block)
            eps=[torch.randn(1,3,96,96,device='cuda',generator=aux_rng) for _ in range(4)]
            with ledger.phase('train/B/aux'),torch.autocast('cuda',enabled=False):
                out=rollout(model,history,acts,eps,sampler.sigmas,recompute_context=lambda:ledger.phase('train/B/aux_recompute'))
                extra=objective(out,targets)
                if not bool(torch.isfinite(extra)):raise FloatingPointError('Auxiliary loss nonfinite')
                ledger.exposures['B_aux']+=4;ledger.backward['B_aux_attempted']+=1;scaler.scale(extra).backward();ledger.backward['B_aux_completed']+=1
                aux_loss=float(extra.detach())
            del out,extra,history,targets,acts,eps
        scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient; no retry or optimizer step')
        application=scaled_optimizer_step(scaler,opt,application_count=lambda:applied[0])
        ledger.updates[arm]+=application['applied']
        if application['applied']!=1:raise FloatingPointError('AMP skip; bounded comparison stops, no replacement')
        torch.cuda.synchronize();elapsed=time.perf_counter()-t0;training_seconds+=elapsed
        row=dict(attempt=step,updates=ledger.updates[arm],native_loss=float(result['loss']),aux_loss=aux_loss,gradient_norm_before_clip=float(norm),branches=result['branches'],seconds=elapsed,scaler=application,conditioning_rng_hashes=rngs.hashes())
        log.append(row);ledger.active=None;write(ROOT/f'training/{arm}-log.json',log)
        if step in (16,128,256):
            write(ROOT/f'training/{arm}-throughput-{step}.json',dict(arm=arm,updates=ledger.updates[arm],train_seconds=training_seconds,updates_per_second=step/training_seconds,remaining_training_seconds_estimate=(256-step)*training_seconds/step,includes_extra_B_compute=arm=='B',paired_batch_unchanged=True))
        ledger.persist('training',arm=arm,step=step,train_seconds=training_seconds)
        if step in c['milestones']:
            checkpoints.append(save_checkpoint(model,opt,scaler,generator,rngs,parent,arm,step,ledger,step==256))
            saved=capture_rng(generator);noise=deepcopy(rngs.state_dict());mode=model.training
            evaluations[str(step)]=evaluate(model,source,f'{arm}{step}',ledger)
            restore_rng(saved,generator);rngs.load_state_dict(noise);model.train(mode)
            if digest_tree(capture_rng(generator))!=digest_tree(saved) or digest_tree(rngs.state_dict())!=digest_tree(noise):raise RuntimeError('Evaluation changed training RNG')
    handle.remove();result=dict(arm=arm,attempts=ledger.attempts[arm],updates=ledger.updates[arm],training_seconds=training_seconds,evaluations=evaluations,checkpoints=checkpoints,final_rng_hashes=rngs.hashes())
    write(ROOT/f'training/{arm}-result.json',result);opt.zero_grad(set_to_none=True);return result

def main():
    c=read(ROOT/'config.json');expected=read(ROOT/'freeze.json')['config_sha256']
    if sha(ROOT/'config.json')!=expected:raise ValueError('Frozen config changed')
    if (ROOT/'execution-claim.json').exists():raise RuntimeError('Already claimed: no automatic rerun')
    for binding in c['bindings']:
        if sha(binding['path'])!=binding['sha256']:raise ValueError('Frozen source changed: '+binding['path'])
    write(ROOT/'execution-claim.json',dict(pid=os.getpid(),at_utc=utc(),config_sha256=expected,CPU=CPU))
    ledger=Ledger(c);results={};failure=None;started=time.perf_counter()
    try:
        guard(full=True)
        parent=torch.load(c['parent'],map_location='cpu',weights_only=False)
        if parent['kind']!='oc_native_rgb_complete.v1' or parent['progress']['lineage_updates']!=11367 or not parent['progress']['epoch_complete']:raise ValueError('Invalid E2 boundary')
        data=load('_loss_data',HMF/'data.py');dataset=data.FitDataset(HMF);schedule=read(ROOT/'schedule.json')
        model,source=create_model(c);model.load_state_dict(parent['model'],strict=True)
        h1=model.inner_model.register_forward_pre_hook(ledger.pre);h2=model.inner_model.register_forward_hook(ledger.post)
        torch.cuda.reset_peak_memory_stats();ledger.persist('preflight');results['preflight']=preflight(model,source,dataset,schedule,ledger)
        model.load_state_dict(parent['model'],strict=True);results['E2']=evaluate(model,source,'E2',ledger)
        for arm in ('A','B'):results[arm]=train_arm(arm,model,source,parent,dataset,schedule,c,ledger)
        if read(ROOT/'training/A-result.json')['final_rng_hashes']!=read(ROOT/'training/B-result.json')['final_rng_hashes']:raise RuntimeError('Paired native noise streams diverged')
        # Native noise branches/state were recorded at every identical schedule step.
        la,lb=read(ROOT/'training/A-log.json'),read(ROOT/'training/B-log.json')
        if any(a['conditioning_rng_hashes']!=b['conditioning_rng_hashes'] or a['branches']!=b['branches'] for a,b in zip(la,lb)):raise RuntimeError('Native noise pairing failed')
        write(ROOT/'pairing-proof.json',dict(passed=True,paired_attempts=256,same_native_order=True,same_conditioning_rng_after_every_step=True,same_initial_full_state=True,total_compute_equal=False,total_exposures_equal=False))
        h1.remove();h2.remove();results['peak_cuda']=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        del model,parent;dataset.close();torch.cuda.empty_cache()
        ledger.persist('report')
        from report import generate
        results['report']=generate()
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at_utc=utc())
        write(ROOT/'failure.json',failure,emergency=True)
    try:storage=guard(full=True,emergency=True)
    except Exception as exc:storage=dict(passed=False,error=str(exc));failure=failure or dict(message=str(exc))
    final=dict(complete=failure is None,failed=failure is not None,failure=failure,results=results,ledger=ledger.snapshot(),CPU=CPU,own_CPU_seconds=dict(user=os.times().user,system=os.times().system),wall_seconds=time.perf_counter()-started,storage=storage,physical_steps=0,callback_sent=False,state='waiting_callback',finished_utc=utc())
    write(ROOT/'completion.json',final,emergency=True)
    # Completion file itself was budgeted; final measured root includes its growth.
    write(ROOT/'storage-final.json',guard(full=True,emergency=True),emergency=True)
    print(json.dumps(dict(complete=final['complete'],failure=failure,wall_seconds=final['wall_seconds'],ledger=final['ledger'])))
    return int(final['failed'])
if __name__=='__main__':raise SystemExit(main())
