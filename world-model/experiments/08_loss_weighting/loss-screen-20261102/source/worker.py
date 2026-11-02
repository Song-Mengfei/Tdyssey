"""One released four-arm screen, one numerical process, no automatic reruns."""
import sys,os,time,json,traceback,io
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(REPO))
from authorization import issue_permit
from common import read,sha,load,utc,write,write_bytes,output_io,save_raw,HMF

class BoundedLog(io.TextIOBase):
    def __init__(self,path):self.path=path
    @property
    def encoding(self):return 'utf-8'
    def write(self,value):
        data=str(value).encode('utf-8');old=self.path.read_bytes() if self.path.exists() else b''
        if len(old)+len(data)>65536:raise RuntimeError('Worker text log cap reached')
        write_bytes(self.path,old+data);return len(value)
    def flush(self):pass

def saved_scoring_parity(permit,torch):
    import numpy as np
    from scoring_adapter import load_module
    m=load_module('_loss_screen_original_hid_metrics',permit.config['hid_metrics_path'])
    samples=[]
    for item in read(ROOT/'scoring-parity-inputs.json'):
        with np.load(item['prediction_path'],allow_pickle=False) as z:prediction=z['raw'][item['horizon']-1].copy()
        with np.load(item['truth_path'],allow_pickle=False) as z:
            a=item['anchor'];truth=z['rgb'][a+item['horizon']].copy();last=z['rgb'][a].copy()
        samples.append(dict(sample_id=item['sample_id'],letter=item['letter'],prediction=prediction,truth=truth,last=last))
    proof=m.check_saved_samples(samples,torch=torch,device=torch.device('cuda'));write(ROOT/'scoring-parity.json',proof)
    if not proof['passed']:raise RuntimeError('Original scoring field/qualification parity failed')
    permit.scoring_parity_passed=True;return proof

def evaluate(permit,model,source,label,ledger,torch):
    import numpy as np
    from types import SimpleNamespace
    from scoring_adapter import score_native_batch,truth_series
    ev=load('_loss_screen_hmf_eval',HMF/'evaluation.py');old=ev.helper().old();store=old._ar1().RawStore(max_bytes=48*1024**2)
    spec=read(ROOT/'evaluation-spec.json');std=read(HMF/'fit-standard.json')
    features={x['sample_id']:x['truth_features'] for x in read(HMF/'GT-measurement-calibration.json')['rows']}
    rows=[];raws=[];evidence=[];costs=[];started=time.perf_counter();model.eval()
    with ledger.phase('eval/'+label):
        for mode in ('real_history','autoregressive'):
            for batch in spec['batches']:
                rr=batch['rows'];bid=batch['batch_id'];seeds=batch['seeds'];inputs=old.prepare_inputs(store,rr,mode)
                ledger.active=dict(stage='evaluation',arm=label,mode=mode,batch=bid);ledger.persist()
                prediction,noise=ev.helper().native_rollout(SimpleNamespace(model=model,source=source),inputs,seeds)
                evidence.append(dict(mode=mode,batch_id=bid,inputs=old._inputs_evidence(rr,inputs,seeds),noise=noise))
                truths=[]
                for slot,row in enumerate(rr):
                    record=save_raw(ROOT/f'evaluation/{label}/{mode}-b{bid:02d}-s{slot}.npz',prediction[slot])
                    raws.append(dict(**record,sample_id=row['sample_id'],letter=row['letter'],selection_role=row['selection_role'],mode=mode,batch_id=bid,slot=slot))
                    truths.append(truth_series(row,store.arrays(row),features[row['sample_id']]))
                gpu=torch.as_tensor(prediction,device='cuda',dtype=torch.float32)
                scored=score_native_batch(permit,label,batch,mode,gpu,prediction,truths);rows.extend(scored['rows']);costs.append(scored['scoring'])
                del gpu,prediction;ledger.active=None;ledger.persist()
    copies=read(permit.config['copy_reference'])
    support={role:ev.assess(rows,copies,std,role) for role in ('train','seen_val','original_failure')}
    parent=read(permit.config['E2_metrics'])
    if evidence!=parent['evidence']:raise RuntimeError('Original eval batch/slot/inputs/actual epsilon changed')
    value=dict(label=label,rows=rows,raw_files=raws,evidence=evidence,support=support,
               predictions=len(raws)*16,wall_seconds=time.perf_counter()-started,scoring_cost=costs,scope='original train12/seen-val24/old3, not full48')
    write(ROOT/f'evaluation/{label}/metrics.json',value);store.cache.clear();output_io().guard(refresh=True)
    if label=='BASE':
        previous=read(permit.config['R1_A128_metrics']);key=lambda x:(x['sample_id'],x['input_mode'],x['horizon'])
        lookup={key(x):x for x in previous['rows']};diff=[]
        for row in rows:
            oldrow=lookup[key(row)]
            for field in ('raw_mask_iou','object_position_error_image_px','raw_contour_error_image_px','pusher_position_error_image_px'):
                a=oldrow['metrics'].get(field);b=row['metrics'].get(field)
                if a!=b:diff.append(dict(key=key(row),metric=field,R1=a,new=b,abs_difference=None if a is None or b is None else abs(a-b)))
            for field in ('visible_topology_status','visible_shape_status'):
                if oldrow[field]!=row[field]:diff.append(dict(key=key(row),metric=field,R1=oldrow[field],new=row[field]))
        raw_equal=all(a['sha256']==b['sha256'] for a,b in zip(raws,previous['raw_files']))
        write(ROOT/'BASE-vs-R1-A128.json',dict(raw_archive_hashes_equal=raw_equal,primary_exact_equal=not diff,
              differences=diff,old_baseline_not_substituted=True,scope='All actual differences reported, no silent reference replacement'))
    return dict(metrics=str(ROOT/f'evaluation/{label}/metrics.json'),wall_seconds=value['wall_seconds'])

def train(permit,model,parent,dataset,native,arm,opt,scaler,generator,rngs,ledger,torch):
    from native_forward import forward_bundle
    from objectives import loss_for
    from tdyssey.native_rgb_r2c3_training import scaled_optimizer_step
    schedule=read(ROOT/'schedule.json');log=[];seconds=0.;applied=[0]
    handle=opt.register_step_post_hook(lambda *x:applied.__setitem__(0,applied[0]+1))
    for i,indices in enumerate(schedule['native_h1_batches']):
        if i>=128:raise RuntimeError('Training schedule exceeded128')
        output_io().guard();ledger.attempts[arm]+=1
        if ledger.attempts[arm]>128:raise RuntimeError('Attempt cap reached')
        ledger.active=dict(stage='training',arm=arm,attempt=i+1);ledger.persist();started=time.perf_counter()
        model.train();opt.zero_grad(set_to_none=True)
        batch=dataset.model_batch(indices);images=torch.as_tensor(batch['image'],device='cuda');actions=torch.as_tensor(batch['action'],device='cuda')
        branches=[];losses=[];extras=[];flags=[]
        with ledger.phase('train/'+arm):
            if arm=='BASE':
                def observe(event,micro,n):
                    if event=='loss_complete':ledger.exposures[arm]+=n
                    elif event=='backward_attempt':ledger.grad_begin(arm)
                    elif event=='backward_complete':ledger.grad_end(arm)
                result=native.microbatch_backward(model,images,actions,rngs,scaler,observer=observe)
                finite=result['forward_finite'];loss=float(result['loss']);branches=result['branches']
            else:
                for j in range(4):
                    x=images[j*4:j*4+4];a=actions[j*4:j*4+4];noise=native.draw_noise(rngs,4,model.device)
                    with torch.autocast('cuda',dtype=torch.float16):bundle=forward_bundle(permit,model,x[:,:4],x[:,4],a,noise)
                    value,parts=loss_for(arm,bundle);flags.append(bundle['finite']&torch.isfinite(value));ledger.exposures[arm]+=4
                    ledger.grad_begin(arm);scaler.scale(value*.25).backward();ledger.grad_end(arm)
                    losses.append(value.detach());branches.append(noise.branch)
                    if parts.get('extra') is not None:extras.append(parts['extra'])
                    del bundle,value,parts
                finite=torch.stack(flags).all();loss=float(torch.stack(losses).mean())
        if not bool(finite):raise FloatingPointError('Nonfinite native/candidate loss or input contract')
        scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient')
        update=scaled_optimizer_step(scaler,opt,application_count=lambda:applied[0]);ledger.updates[arm]+=update['applied']
        if update['applied']!=1:raise FloatingPointError('AMP skip; no replacement attempt')
        torch.cuda.synchronize();seconds+=time.perf_counter()-started
        log.append(dict(attempt=i+1,updates=ledger.updates[arm],loss=loss,extra=None if not extras else float(torch.stack(extras).mean()),
            gradient_norm_before_clip=float(norm),branches=branches,conditioning_rng_hashes=rngs.hashes(),scaler=update))
        ledger.active=None;ledger.persist();write(ROOT/f'training/{arm}-log.json',log)
        if i+1 in (16,128):
            write(ROOT/f'training/{arm}-throughput-{i+1}.json',dict(arm=arm,at_utc=utc(),updates=i+1,train_seconds=seconds,updates_per_second=(i+1)/seconds,
                remaining_arm_train_seconds_estimate=(128-i-1)*seconds/(i+1),peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),
                includes_formal_first16=True,CPU=ledger.extra['CPU'],wall_since_worker_start=time.perf_counter()-ledger.extra['started_perf']))
    handle.remove();opt.zero_grad(set_to_none=True)
    from tdyssey.native_rgb_training import _cpu_tree
    buffer=io.BytesIO();torch.save(dict(kind='loss_lit_screen_weights.v1',model=_cpu_tree(model.state_dict()),arm=arm,attempts=128,updates=128,
        parent_sha256=permit.config['parent_sha256'],config_sha256=permit.config_sha256,lineage_updates=11495),buffer)
    checkpoint=output_io().write_bytes(f'training/{arm}-128.pt',buffer.getvalue(),exclusive=True)
    write(ROOT/f'training/{arm}-128.json',checkpoint)
    result=dict(arm=arm,attempts=128,updates=ledger.updates[arm],native_exposures=ledger.exposures[arm],training_seconds=seconds,
                final_rng_hashes=rngs.hashes(),checkpoint=checkpoint)
    write(ROOT/f'training/{arm}-result.json',result);return result

def main():
    permit=issue_permit();c=permit.config;io_out=output_io();started=time.perf_counter()
    if '--config-sha' not in sys.argv or sys.argv[sys.argv.index('--config-sha')+1]!=permit.config_sha256:raise RuntimeError('Launch config identity mismatch')
    if (ROOT/'execution-claim.json').exists():raise RuntimeError('Already claimed; no duplicate worker/retry')
    sys.stdout=BoundedLog(ROOT/'stdout.log');sys.stderr=BoundedLog(ROOT/'stderr.log')
    from ledger import Ledger
    phases={**{'preflight/'+a:128 for a in ('REFERENCE','BASE','SNR','GDL','FFL')},**{'train/'+a:2048 for a in c['arms']},**{'eval/'+a:3744 for a in c['arms']}}
    ledger=Ledger(io_out,phases,23296,category_caps=dict(preflight=128,train=8192,eval=14976))
    failure=None;results={};CPU=None
    try:
        limits=load('_loss_screen_cpu_limits',c['cpu_limits_path']);CPU=limits.constrain_cpu()
        import torch
        CPU.update(limits.torch_limits());ledger.extra.update(CPU=CPU,started_perf=started,model_constructions_attempted=0,model_constructions_completed=0,full_arm_restores=0)
        io_out.write_json('execution-claim.json',dict(run_id=c['run_id'],config_sha256=permit.config_sha256,pid=os.getpid(),CPU=CPU,at_utc=utc()),exclusive=True)
        for b in c['bindings']:
            if sha(b['path'])!=b['sha256']:raise RuntimeError('Frozen reference/source changed: '+b['path'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU model fallback')
        io_out.guard(refresh=True);results['scoring_parity']=saved_scoring_parity(permit,torch)
        from state_restore import load_parent,restore_full
        parent=load_parent(permit)
        from tdyssey.gpc_world_model import load_world_model_source,world_model_spec
        from types import SimpleNamespace
        from copy import deepcopy
        source=load_world_model_source(c['source_root']);inner=deepcopy(world_model_spec()['model']);outer={k:inner.pop(k) for k in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
        ledger.extra['model_constructions_attempted']=1;ledger.persist()
        with torch.random.fork_rng(devices=[]):model=source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**inner),**outer)).float().cuda()
        ledger.extra['model_constructions_completed']=1;model.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.))
        hooks=[model.inner_model.register_forward_pre_hook(ledger.before_forward),model.inner_model.register_forward_hook(ledger.after_forward)]
        native=load('_loss_screen_native',HMF/'training_core.py');data=load('_loss_screen_data',HMF/'data.py');dataset=data.FitDataset(HMF)
        torch.cuda.reset_peak_memory_stats();native_logs={}
        model.load_state_dict(parent['model'],strict=True);model.requires_grad_(True)
        ledger.extra['preflight_model_only_loads']=1;ledger.persist()
        from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs
        pre_rngs=make_conditioning_rngs(**parent['conditioning_rngs']['seeds'],device=model.device);pre_rngs.load_state_dict(parent['conditioning_rngs'])
        from preflight import run
        results['preflight']=run(permit,model,native,dataset,read(ROOT/'schedule.json'),pre_rngs,ledger,io_out)
        del pre_rngs
        for arm in c['arms']:
            io_out.guard(refresh=True);opt,scaler,generator,rngs=restore_full(permit,model,parent,arm,io_out);ledger.extra['full_arm_restores']+=1;ledger.persist()
            results[arm]=train(permit,model,parent,dataset,native,arm,opt,scaler,generator,rngs,ledger,torch)
            native_logs[arm]=read(ROOT/f'training/{arm}-log.json')
            if arm!='BASE' and any(a['branches']!=b['branches'] or a['conditioning_rng_hashes']!=b['conditioning_rng_hashes'] for a,b in zip(native_logs['BASE'],native_logs[arm])):
                raise RuntimeError('Native order/noise pairing diverged')
            del opt,scaler,generator,rngs
            results[arm]['evaluation']=evaluate(permit,model,source,arm,ledger,torch);write(ROOT/f'training/{arm}-result.json',results[arm])
        write(ROOT/'pairing-proof.json',dict(passed=True,arms=c['arms'],same_native_order=True,same_native_rng_per_step=True,full_arm_restores=4,updates_each=128))
        for h in hooks:h.remove()
        results['peak_cuda']=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        del model,parent;dataset.close();torch.cuda.empty_cache()
        from report import generate
        results['report']=generate()
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at_utc=utc(),phase=ledger.active)
        write(ROOT/'failure.json',failure,emergency=True)
    complete=failure is None
    value=dict(run_id=c['run_id'],config_sha256=permit.config_sha256,complete=complete,failed=not complete,failure=failure,results=results,ledger=ledger.snapshot(),
        CPU=CPU,own_CPU_seconds=dict(user=os.times().user,system=os.times().system),wall_seconds=time.perf_counter()-started,physical_steps=0,new_data=0,
        callback_sent=False,state='waiting_callback',storage=io_out.guard(final_metadata=True,refresh=True),finished_utc=utc())
    write(ROOT/'completion.json',value,emergency=True);write(ROOT/'storage-final.json',io_out.guard(final_metadata=True,refresh=True),emergency=True)
    return int(not complete)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        write(ROOT/'bootstrap-failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at_utc=utc()),emergency=True)
        raise

