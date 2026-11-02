"""One matched A32/B32/Aremaining/Bremaining run; no import-time Torch/GPU."""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime
from types import SimpleNamespace
import numpy as np

REPO=Path(__file__).resolve().parents[2]
PARENT=REPO/'artifacts/oc-continuation-20261031/training/epoch2.pt'
PARENT_SHA='c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()

def validate_scope(pool,order,parent):
    rows=pool['segments'];batches=order['batches'];flat=[i for b in batches for i in b]
    if len(rows)!=7568 or len({r['family_id']for r in rows})!=1313 or len(batches)!=1892 or any(len(b)!=4 for b in batches):raise ValueError('Approved7568segments/1313families/1892batches required')
    if sorted(flat)!=list(range(7568)) or [rows[i]['segment_id']for i in flat]!=order['segment_ids']:raise ValueError('Paired segment order repeats/drops/misidentifies a segment')
    targets=[x for r in rows for x in r['supervised_H1_ids']]
    if len(targets)!=30272 or len(set(targets))!=30272 or any(not r['committed_future']or r['split']!='train'for r in rows):raise ValueError('Only complete clean committed train targets once')
    p=parent['progress']
    if parent.get('kind')!='oc_native_rgb_complete.v1' or p['epoch']!=2 or not p['epoch_complete']or p['lineage_updates']!=11367 or p['unique_samples_exposed']!=30935 or p['batches_consumed']!=1934 or not p['complete_optimizer_boundary']or p['pending_attempt']is not None:raise ValueError('Exact completed epoch2/11367full state required')
    if any(abs(g['lr']-1e-5)>1e-14 for g in parent['optimizer']['param_groups']):raise ValueError('Common Adam LR changed')
    return True

def initial_progress(label):
    return dict(arm=label,batches_consumed=0,segments_reserved=0,target_frames_reserved=0,segments_exposed=0,target_frames_exposed=0,successful_updates=0,AMP_skips=0,
        lineage_updates=11367,complete_optimizer_boundary=True,pending_attempt=None,epoch_complete=False,
        supervised_batch_forwards=0,backward_calls=0,consecutive_AMP_overflows=0)

class CudaTimer:
    def __init__(self):self.events=[]
    def start(self):
        import torch
        e=torch.cuda.Event(enable_timing=True);e.record();return e
    def stop(self,key,start):
        import torch
        end=torch.cuda.Event(enable_timing=True);end.record();self.events.append((key,start,end))
    def finish(self):
        import torch
        torch.cuda.synchronize();out=defaultdict(float)
        for key,start,end in self.events:out[key]+=start.elapsed_time(end)/1000
        return dict(out)

class Session:
    def __init__(self,root,c):
        import torch
        from common import atomic_json,check_storage
        from pool import SegmentLoader
        from tdyssey.gpc_world_model import load_world_model_source,world_model_spec
        from tdyssey.raw_world_model_training import OPTIMIZER,make_generator
        from tdyssey.native_rgb_r2c1r1_training import _tree_digest
        self.root=Path(root);self.c=c;self.atomic_json=atomic_json;self.check_storage=check_storage;self.torch=torch
        self.output=self.root/'training';self.output.mkdir(parents=True,exist_ok=True)
        self.pool=read(self.root/'pool-index.json');self.order=read(self.root/'order.json')
        self.parent_path=Path(c.get('starting_checkpoint',PARENT));self.parent_sha=sha(self.parent_path)
        if self.parent_sha!=PARENT_SHA or c.get('starting_checkpoint_sha256',PARENT_SHA)!=PARENT_SHA:raise ValueError('Pinned common epoch2checkpoint differs')
        torch.set_num_threads(2);self.parent=torch.load(self.parent_path,map_location='cpu',weights_only=False)
        validate_scope(self.pool,self.order,self.parent)
        self.loader=SegmentLoader(self.root,c)  # Complete once-only entry hashes, bounded read-only mmap.
        if not torch.cuda.is_available():raise RuntimeError('Authorized local CUDA unavailable; no fallback configuration')
        self.device=torch.device('cuda');torch.cuda.reset_peak_memory_stats();check_storage(self.root,c,full=True)
        self.source=load_world_model_source(c.get('source_root',str(REPO.parent/'Tdyssey/third_party/gpc_code')))
        inner=deepcopy(world_model_spec()['model']);outer={k:inner.pop(k)for k in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
        with torch.random.fork_rng(devices=[]):self.model=self.source.Denoiser(self.source.DenoiserConfig(inner_model=self.source.InnerModelConfig(**inner),**outer)).float()
        self.model.to(self.device).requires_grad_(True);self.model.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.));self.model.train()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),**dict(OPTIMIZER,lr=1e-5));self.scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=128.)
        self.legacy_generator=make_generator(5478,self.device);self.td=lambda x:_tree_digest(x,torch)
        self.applications=0;self.application_hook=self.optimizer.register_step_post_hook(self._applied)
        self.activity='startup';self.forward_counts={};self.forward_hook=self.model.inner_model.register_forward_pre_hook(self._forward)
        self.forward_complete_hook=self.model.inner_model.register_forward_hook(self._forward_complete);self.backward_counts={}
        self.progress=None;self.rngs=None;self.generation_rng=None;self.history_generator=None;self.clean=True;self.arm=None
        self.restore_proofs=[];self.rng_proofs=[];self.checkpoints={};self.timings={x:defaultdict(float)for x in ('A','B')};self.slices={x:[]for x in ('A','B')}
        self.identity=dict(kind='generated-history-four-target-ab.v1',common_checkpoint_sha256=PARENT_SHA,common_lineage_updates=11367,
            pool_index_sha256=sha(self.root/'pool-index.json'),order_sha256=sha(self.root/'order.json'),arms=['A','B'],segments_per_arm=7568,
            supervised_targets_per_arm=30272,batches_per_arm=1892,sequence_microbatch=4,effective_target_batch=16,learning_rate=1e-5,
            generation='Independent pure-noise stream, original3step sampler, eval FP32/no_grad, P1-P3 detached; no future targets in generation API',
            residual='Original phase2FP32 equal pixel MSE, four losses*.25, training AMP, original50/50clean-weak noise',
            real_RGB_conversion='Exact original numpy uint8.astype(float32)/np.float32(255), bounded4x8 batch before H2D',
            physical_steps=0,new_samples=0)
        atomic_json(self.output/'identity.json',self.identity)
    def _applied(self,*args):self.applications+=1
    def _forward(self,module,args):
        kind='supervised'if self.model.training else'generated';row=self.forward_counts.setdefault(self.activity,defaultdict(int))
        row[kind+'_batch_forwards']+=1;row[kind+'_image_forwards']+=len(args[0])
    def _forward_complete(self,module,args,output):
        kind='supervised'if self.model.training else'generated';row=self.forward_counts.setdefault(self.activity,defaultdict(int))
        row[kind+'_completed_batch_forwards']+=1;row[kind+'_completed_image_forwards']+=len(args[0])
    def _observe(self,event,h,batch_size):
        self.progress['pending_attempt'].update(target_h=h+1,stage=event)
        row=self.backward_counts.setdefault(self.arm,defaultdict(int))
        if event=='backward_attempt':row['attempted_batch_backwards']+=1;row['attempted_target_backwards']+=batch_size
        if event=='backward_complete':
            row['completed_batch_backwards']+=1;row['completed_target_backwards']+=batch_size;self.progress['backward_calls']+=1
        if event=='loss_complete':
            self.progress['target_frames_exposed']+=batch_size
            if h==0:self.progress['segments_exposed']+=batch_size
    def restore(self,label,path=None,expected=None):
        from common import atomic_json
        from tdyssey.native_rgb_training import capture_rng,restore_rng
        from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs
        from training_core import NativeHistoryGenerator
        began=time.perf_counter();p=self.parent
        if path is not None:
            if sha(path)!=expected:raise ValueError('Internal32boundary checkpoint differs')
            p=self.torch.load(path,map_location='cpu',weights_only=False)
            if p['kind']!='generated_history_ab_complete.v1'or p['identity']!=self.identity or p['progress']['arm']!=label or not p['progress']['complete_optimizer_boundary']or p['progress']['pending_attempt']is not None:raise ValueError('Wrong/incomplete matched arm boundary')
        self.model.load_state_dict(p['model'],strict=True);self.optimizer.load_state_dict(p['optimizer']);self.scaler.load_state_dict(p['scaler'])
        restore_rng(p['global_rng'],self.legacy_generator)
        self.rngs=make_conditioning_rngs(**p['conditioning_rngs']['seeds'],device=next(self.model.parameters()).device);self.rngs.load_state_dict(p['conditioning_rngs'])
        self.generation_rng=self.torch.Generator(device=self.device).manual_seed(int(self.c.get('generation_seed',202611015478)))
        if path is not None:self.generation_rng.set_state(p['generation_rng'])
        self.history_generator=NativeHistoryGenerator(self.model,self.source,self.generation_rng)
        if path is not None:
            for key,value in p['generation_counts'].items():setattr(self.history_generator,key,value)
        self.progress=initial_progress(label)if path is None else deepcopy(p['progress']);self.arm=label;self.activity=label
        self.optimizer.zero_grad(set_to_none=True);self.model.train();self.clean=True
        proof=dict(label=label,source='common_epoch2'if path is None else str(path),source_sha256=PARENT_SHA if path is None else expected,
            cursor=self.progress['batches_consumed'],components=dict(model=self.td(self.model.state_dict())==self.td(p['model']),
            Adam=self.td(self.optimizer.state_dict())==self.td(p['optimizer']),scaler=self.td(self.scaler.state_dict())==self.td(p['scaler']),
            global_rng=self.td(capture_rng(self.legacy_generator))==self.td(p['global_rng']),conditioning_rngs=self.td(self.rngs.state_dict())==self.td(p['conditioning_rngs'])),
            wall_seconds=time.perf_counter()-began,optimizer_reset=False)
        if path is not None:proof['components']['generation_rng']=self.td(self.generation_rng.get_state())==self.td(p['generation_rng'])
        else:
            expected_generation=self.torch.Generator(device=self.device).manual_seed(int(self.c.get('generation_seed',202611015478)))
            proof['components']['generation_rng']=self.td(self.generation_rng.get_state())==self.td(expected_generation.get_state())
            proof['generation_seed']=int(self.c.get('generation_seed',202611015478));proof['generation_seed_state_sha256']=self.td(self.generation_rng.get_state())
        proof['passed']=all(proof['components'].values());self.restore_proofs.append(proof);atomic_json(self.output/'restore-proofs.json',self.restore_proofs)
        if not proof['passed']:raise RuntimeError('Actual complete arm restore is not exact')
        return proof
    def ctx(self,checkpoint):
        return SimpleNamespace(model=self.model,source=self.source,checkpoint_identity=checkpoint,
            generator=self.legacy_generator,conditioning_rngs=self.rngs,generation_generator=self.generation_rng,progress=self.progress)
    def protected(self,label,callback):
        from tdyssey.native_rgb_training import capture_rng,restore_rng
        state=capture_rng(self.legacy_generator);condition=self.rngs.state_dict();generation=self.generation_rng.get_state();mode=self.model.training
        before=dict(global_rng=self.td(state),supervision=self.td(condition),generation=self.td(generation));apps=self.applications;progress=deepcopy(self.progress)
        activity=self.activity;self.activity=label;began=time.perf_counter();error=None
        try:return callback()
        except BaseException as exc:error=repr(exc);raise
        finally:
            restore_rng(state,self.legacy_generator);self.rngs.load_state_dict(condition);self.generation_rng.set_state(generation);self.model.train(mode);self.activity=activity
            after=dict(global_rng=self.td(capture_rng(self.legacy_generator)),supervision=self.td(self.rngs.state_dict()),generation=self.td(self.generation_rng.get_state()))
            proof=dict(label=label,before=before,after=after,matches=before==after,model_mode_restored=self.model.training==mode,
                optimizer_applications=self.applications-apps,training_progress_unchanged=progress==self.progress,wall_seconds=time.perf_counter()-began,error=error)
            self.rng_proofs.append(proof);self.atomic_json(self.output/'rng-preservation.json',self.rng_proofs)
            if not proof['matches']or not proof['model_mode_restored']or proof['optimizer_applications']or not proof['training_progress_unchanged']:raise RuntimeError('Diagnostic/inference changed training state')
    def gpu_preflight(self):
        from training_core import NativeHistoryGenerator,SAMPLER
        torch=self.torch;old_activity=self.activity;self.activity='gpu_preflight';began=time.perf_counter()
        arrays=self.loader.batch(self.order['batches'][0]);images=torch.as_tensor(arrays['image'].astype(np.float32)/np.float32(255),device=self.device)
        actions=torch.as_tensor(arrays['action'],device=self.device);seed=2026110171;mode=self.model.training
        device=next(self.model.parameters()).device;index=device.index if device.index is not None else torch.cuda.current_device()
        self.model.eval();native=self.source.DiffusionSampler(self.model,self.source.DiffusionSamplerConfig(**SAMPLER))
        with torch.random.fork_rng(devices=[index]),torch.no_grad(),torch.autocast('cuda',enabled=False):
            torch.cuda.default_generators[index].manual_seed(seed);expected,_=native.sample(images[:,:4],actions[:,:4])
        independent=torch.Generator(device=device).manual_seed(seed);g=NativeHistoryGenerator(self.model,self.source,independent)
        actual=g.generate(images[:,:4],actions[:,:4]);equal=bool(torch.equal(actual,expected));finite=bool(torch.isfinite(actual).all())
        self.model.train(mode);self.activity=old_activity;torch.cuda.synchronize()
        proof=dict(passed=equal and finite,native_raw_exact=equal,finite=finite,actual_model_calls=dict(self.forward_counts['gpu_preflight']),
            engineering_sampler_batch_calls=2,engineering_generated_images=8,optimizer_updates=0,physical_steps=0,wall_seconds=time.perf_counter()-began,
            label_isolation='SeparateCPUtest changes futurelabels and obtains identicalP1P2P3; productiongenerate API accepts history/actions only')
        self.atomic_json(self.output/'gpu-preflight.json',proof)
        if not proof['passed']:raise RuntimeError('Actual native sampler equivalence preflight failed')
        return proof
    def save(self,name):
        from common import replace_atomic
        from tdyssey.native_rgb_training import _cpu_tree,capture_rng
        if not self.clean or not self.progress['complete_optimizer_boundary']:raise RuntimeError('No checkpoint from incomplete/uncertain optimizer state')
        began=time.perf_counter();self.check_storage(self.root,self.c,reserve_bytes=128*1024**2)
        folder=self.output/self.arm;folder.mkdir(exist_ok=True);path=folder/name;temp=path.with_name(path.name+'.tmp')
        payload=dict(kind='generated_history_ab_complete.v1',identity=self.identity,model=_cpu_tree(self.model.state_dict()),optimizer=_cpu_tree(self.optimizer.state_dict()),
            scaler=_cpu_tree(self.scaler.state_dict()),global_rng=capture_rng(self.legacy_generator),conditioning_rngs=self.rngs.state_dict(),
            generation_rng=self.generation_rng.get_state().cpu(),generation_counts={k:getattr(self.history_generator,k)for k in ('batch_calls','generated_frames','denoiser_batch_forwards')},
            order=dict(pool_index_sha256=self.identity['pool_index_sha256'],order_sha256=self.identity['order_sha256'],cursor=self.progress['batches_consumed']),
            progress=deepcopy(self.progress),legacy_scheduler_provenance=deepcopy(self.parent['legacy_scheduler_provenance']),
            parent_checkpoint=dict(path=str(self.parent_path),sha256=PARENT_SHA),actual_forward_counts=deepcopy(self.forward_counts),actual_backward_counts=deepcopy(self.backward_counts))
        try:
            self.torch.save(payload,temp)
            with temp.open('rb+')as f:os.fsync(f.fileno())
            replace_atomic(temp,path)
        finally:temp.unlink(missing_ok=True)
        receipt=dict(label=self.arm,path=str(path),sha256=sha(path),bytes=path.stat().st_size,cursor=self.progress['batches_consumed'],
            target_frames=self.progress['target_frames_exposed'],lineage_updates=self.progress['lineage_updates'],wall_seconds=time.perf_counter()-began)
        self.timings[self.arm]['checkpoint_seconds']+=receipt['wall_seconds'];self.checkpoints.setdefault(self.arm,{})[name]=receipt
        self.atomic_json(path.with_suffix('.json'),receipt);return receipt
    def train_slice(self,until):
        from training_core import four_step_backward
        from tdyssey.native_rgb_r2c3_training import scaled_optimizer_step
        torch=self.torch;start_cursor=self.progress['batches_consumed'];began=time.perf_counter();arm=self.arm
        if not start_cursor<until<=1892:raise ValueError('No replay or extra batch outside1892')
        batches=self.order['batches'];params=list(self.model.parameters());folder=self.output/arm;folder.mkdir(exist_ok=True)
        with (folder/'losses.jsonl').open('a',encoding='utf-8',buffering=65536)as log,ThreadPoolExecutor(max_workers=1)as prefetch:
            pending=prefetch.submit(self.loader.batch,batches[start_cursor])
            for bi in range(start_cursor,until):
                data_start=time.perf_counter();arrays=pending.result();self.timings[arm]['data_wait_seconds']+=time.perf_counter()-data_start
                if bi+1<until:pending=prefetch.submit(self.loader.batch,batches[bi+1])
                convert_start=time.perf_counter();real=arrays['image'].astype(np.float32)/np.float32(255)
                self.timings[arm]['CPU_normalization_seconds']+=time.perf_counter()-convert_start
                timer=CudaTimer();stamp=timer.start();images=torch.as_tensor(real,device=self.device);actions=torch.as_tensor(arrays['action'],device=self.device);timer.stop('H2D_seconds',stamp)
                self.optimizer.zero_grad(set_to_none=True);self.clean=False;self.progress['complete_optimizer_boundary']=False
                self.progress['pending_attempt']=dict(batch=bi,optimizer_application='not_started');self.progress['batches_consumed']=bi+1
                self.progress['segments_reserved']+=4;self.progress['target_frames_reserved']+=16
                before_counts=dict(self.forward_counts.get(arm,{}));out=four_step_backward(self.model,images,actions,self.rngs,self.history_generator,self.scaler,arm,timer=timer,observer=self._observe)
                stamp=timer.start();self.scaler.unscale_(self.optimizer)
                if any(p.grad is None for p in params):raise FloatingPointError('Missing full-network gradient')
                # All GPU finite reductions are submitted before one host read.
                flags=torch.stack([out['forward_finite'],torch.stack([torch.isfinite(p).all()for p in params]).all(),torch.stack([torch.isfinite(p.grad).all()for p in params]).all()]).cpu().tolist()
                if not flags[0]or not flags[1]:raise FloatingPointError('Nonfinite forward/loss/history/parameter beforeAdam')
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)if flags[2]else None
                self.progress['pending_attempt']['optimizer_application']='started'
                outcome=scaled_optimizer_step(self.scaler,self.optimizer,application_count=lambda:self.applications)
                if outcome['applied']:
                    self.progress['successful_updates']+=1;self.progress['lineage_updates']+=1;self.progress['consecutive_AMP_overflows']=0
                else:self.progress['AMP_skips']+=1;self.progress['consecutive_AMP_overflows']+=1
                if not bool(torch.stack([torch.isfinite(p).all()for p in params]).all()):raise FloatingPointError('Nonfinite parameter afterAdam')
                self.optimizer.zero_grad(set_to_none=True);timer.stop('optimizer_and_checks_seconds',stamp);times=timer.finish()
                for key,value in times.items():self.timings[arm][key]+=value
                self.progress['supervised_batch_forwards']+=4
                self.clean=True;self.progress['complete_optimizer_boundary']=True;self.progress['pending_attempt']=None
                counts=self.forward_counts[arm]
                if counts['supervised_batch_forwards']-before_counts.get('supervised_batch_forwards',0)!=4 or counts.get('generated_batch_forwards',0)-before_counts.get('generated_batch_forwards',0)!=(9 if arm=='B'else 0):raise RuntimeError('Observed native model calls differ from approved4targets/3Ps')
                record=dict(batch=bi,segments=4,target_frames=16,loss=float(out['loss']),gradient_norm=None if norm is None else float(norm),branches=out['branches'],**outcome,timing=times)
                log.write(json.dumps(record,allow_nan=False)+'\n')
                if (bi+1)%32==0:
                    log.flush();self.atomic_json(folder/'progress.json',dict(self.progress,timings=dict(self.timings[arm]),actual_model_calls=dict(counts)))
                if self.progress['consecutive_AMP_overflows']>=16:raise FloatingPointError('16consecutive realAMPoverflows; no extra draws')
                if torch.cuda.max_memory_reserved()>int(self.c.get('max_cuda_bytes',6*1024**3)):raise RuntimeError('Authorized6GiB CUDA envelope exceeded')
                if outcome['applied']and self.progress['successful_updates']%100==0:
                    log.flush();os.fsync(log.fileno());self.save('last.pt')
                del out,images,actions
            log.flush();os.fsync(log.fileno())
        wall=time.perf_counter()-began;self.timings[arm]['loop_wall_seconds']+=wall
        self.progress['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();self.progress['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
        record=dict(start_cursor=start_cursor,end_cursor=until,batches=until-start_cursor,wall_seconds=wall);self.slices[arm].append(record)
        self.atomic_json(folder/'progress.json',dict(self.progress,timings=dict(self.timings[arm]),actual_model_calls=dict(self.forward_counts[arm])))
        return record
    def close(self):
        self.application_hook.remove();self.forward_hook.remove();self.forward_complete_hook.remove();self.optimizer.zero_grad(set_to_none=True);self.loader.close()

def throughput_summary(a,b,startup,preflight,*,root=None):
    arms={}
    for label,result in [('A',a),('B',b)]:
        p=result['progress'];t=result['timings'];wall=t['loop_wall_seconds'];success=p['successful_updates']
        remaining=1892-p['batches_consumed'];save=t.get('checkpoint_seconds',0)
        arms[label]=dict(measured_optimizer_batches=p['batches_consumed'],successful_updates=success,AMP_skips=p['AMP_skips'],
            measured_target_frames=p['target_frames_exposed'],loop_wall_seconds=wall,successful_updates_per_second=success/wall,
            optimizer_batches_per_second=p['batches_consumed']/wall,target_frames_per_second=p['target_frames_exposed']/wall,
            seconds_per_optimizer_batch=wall/p['batches_consumed'],remaining_optimizer_batches=remaining,
            estimated_remaining_training_seconds=remaining*wall/p['batches_consumed'],timings=t,
            boundary_checkpoint_save_seconds=save,peak_cuda_allocated_bytes=p['peak_cuda_allocated_bytes'],peak_cuda_reserved_bytes=p['peak_cuda_reserved_bytes'])
    #1152native batch4 sampler calls:2newformal48*16 and3diag32*2modes*16.
    per_call=b['timings']['generation_seconds']/96
    gpu_nominal=per_call*1152
    historical_root=REPO/'artifacts/oc-continuation-20261031'
    historical_formal=read(historical_root/'training/epoch2-result.json')['formal_wall_seconds']
    reporting_started=read(historical_root/'background/reporting-process.json')['at_utc']
    reporting_finished=read(historical_root/'background/completion.json')['finished_at_utc']
    historical_report=(datetime.fromisoformat(reporting_finished)-datetime.fromisoformat(reporting_started)).total_seconds()
    #Old formal384scores includes GPU and CPU. This is a deliberately broad,
    #explicitly mixed-wall proxy, NOT a newly measured CPU throughput claim.
    mixed_score_proxy=historical_formal/384*5184
    cpu_media_range=[.25*mixed_score_proxy+historical_report,2*mixed_score_proxy+3*historical_report]
    evaluation=dict(remaining_sampler_batch_calls=1152,remaining_predicted_images=4608,remaining_native_denoiser_batch_forwards=3456,
        measured_B32_sampler_batch_calls=96,measured_seconds_per_sampler_call=per_call,
        estimated_native_GPU_stream_seconds=gpu_nominal,estimated_GPU_stream_range_seconds=[.5*gpu_nominal,2*gpu_nominal],
        new_saved_image_scores=5184,historical_formal48_mixed_wall_seconds=historical_formal,historical_formal_score_rows=384,
        historical_report_wall_seconds=historical_report,estimated_CPU_scoring_media_overhead_range_seconds=cpu_media_range,
        estimated_evaluation_report_total_range_seconds=[.5*gpu_nominal+cpu_media_range[0],2*gpu_nominal+cpu_media_range[1]],
        qualification='GPUstream extrapolated from actualB32generation96calls; inference_mode/batching overhead can differ. CPU/media range uses oldmixedGPU+CPU wall/score as a conservative proxy with.25x..2x andreport1x..3x; pureCPU split is unavailable, not measured in thisrun. Range is provisional, not a deadline.')
    receipts={}
    if root is not None:
        root=Path(root);receipts['cache_build_wall_seconds']=read(root/'pool-index.json')['summary']['build_wall_seconds']
        receipts['CPU_restore_preflight']=read(root/'RESTORE_PREFLIGHT.json')
        if (root/'background/training-entry.json').exists():receipts['worker_before_engine_validation']=read(root/'background/training-entry.json')
    train_eta=sum(x['estimated_remaining_training_seconds']for x in arms.values());save_eta=sum(x['boundary_checkpoint_save_seconds']*20 for x in arms.values())
    total_range=[.7*train_eta+save_eta+evaluation['estimated_evaluation_report_total_range_seconds'][0],1.5*train_eta+save_eta+evaluation['estimated_evaluation_report_total_range_seconds'][1]]
    return dict(arms=arms,estimated_remaining_training_seconds=train_eta,
        estimated_future_checkpoint_seconds=save_eta,estimated_remaining_total_range_seconds=total_range,
        startup_before_first_update_seconds=startup,gpu_preflight=preflight,source_receipts=receipts,evaluation_and_reporting_extra=evaluation,
        uncertainty='32-batch extrapolation; cache/warmup/thermal/competing load and checkpoints can change throughput. No runtime cap; same approvedrun continues without approval pause.')

def run(root,config):
    from common import atomic_json,check_storage
    import evaluation
    root=Path(root).resolve();began=time.perf_counter();(root/'training').mkdir(exist_ok=True)
    atomic_json(root/'training/execution-claim.json',dict(pid=os.getpid(),run_id=config['run_id'],schedule=['A32','B32','Aremaining','Bremaining'],automatic_retry=False),exclusive=True)
    if hasattr(evaluation,'freeze'):evaluation.freeze(root,config)
    session=None;arms={};evaluations={};prefix={};supervision={};preflight=None
    try:
        session=Session(root,config);session.restore('A')
        preflight=session.protected('gpu_preflight',session.gpu_preflight);startup=time.perf_counter()-began
        for label in ('A','B'):
            if label=='B':session.restore('B')
            session.torch.cuda.reset_peak_memory_stats();session.train_slice(32);checkpoint=session.save('after32.pt')
            prefix[label]=dict(progress=deepcopy(session.progress),checkpoint=checkpoint,timings=dict(session.timings[label]))
            supervision[label]=session.rngs.state_dict()['state_sha256']
        if supervision['A']!=supervision['B']:raise RuntimeError('A/B first32 supervised noise streams are not paired')
        throughput=throughput_summary(prefix['A'],prefix['B'],startup,preflight,root=root)
        throughput.update(run_id=config['run_id'],config_sha256=sha(root/'runtime-config.json'),worker_pid=os.getpid(),continues_automatically=True,
            measured_at_unix=time.time(),paired_supervision_after32=True,prefix_checkpoints={k:v['checkpoint']for k,v in prefix.items()})
        atomic_json(root/'throughput.json',throughput);atomic_json(root/'training/throughput.json',throughput)
        # The owner wakeup can report this file; execution never waits for MCP.
        session.restore('A');baseline_identity=dict(label='baseline',path=str(session.parent_path),sha256=PARENT_SHA)
        evaluations['baseline']=session.protected('evaluation_baseline',lambda:evaluation.run_baseline(root,config,session.ctx(baseline_identity)))
        final_supervision={}
        for label in ('A','B'):
            cp=prefix[label]['checkpoint'];session.restore(label,cp['path'],cp['sha256']);session.train_slice(1892)
            p=session.progress
            if p['batches_consumed']!=1892 or p['segments_exposed']!=7568 or p['target_frames_exposed']!=30272:raise RuntimeError('Full matched pass incomplete')
            p['epoch_complete']=True;checkpoint=session.save('final.pt');final_supervision[label]=session.rngs.state_dict()['state_sha256']
            session.atomic_json(root/f'training/{label}/progress.json',dict(p,timings=dict(session.timings[label]),actual_model_calls=dict(session.forward_counts[label])))
            evaluations[label]=session.protected('evaluation_'+label,lambda:evaluation.run_arm(root,config,session.ctx(dict(label=label,path=checkpoint['path'],sha256=checkpoint['sha256'])),label))
            arms[label]=dict(progress=deepcopy(p),checkpoint=checkpoint,timings=dict(session.timings[label]),slices=session.slices[label],actual_model_calls=dict(session.forward_counts[label]),actual_backward_calls=dict(session.backward_counts[label]),
                generation_counts={k:getattr(session.history_generator,k)for k in ('batch_calls','generated_frames','denoiser_batch_forwards')},supervision_rng_final=final_supervision[label])
            atomic_json(root/f'training/{label}/result.json',arms[label])
        if final_supervision['A']!=final_supervision['B']:raise RuntimeError('A/B complete supervised-noise streams differ')
        if arms['A']['generation_counts']['batch_calls']!=0 or arms['B']['generation_counts']!=dict(batch_calls=5676,generated_frames=22704,denoiser_batch_forwards=17028):raise RuntimeError('Actual online generated history count differs')
        result=dict(complete=True,start=dict(path=str(session.parent_path),sha256=PARENT_SHA,lineage_updates=11367),arms=arms,evaluations=evaluations,
            throughput=str(root/'throughput.json'),restore_proofs=str(root/'training/restore-proofs.json'),rng_preservation=str(root/'training/rng-preservation.json'),
            paired_supervised_noise_after32=True,paired_supervised_noise_final=True,supervised_targets_total=60544,optimizer_batches_total=3784,
            actual_successful_updates=sum(x['progress']['successful_updates']for x in arms.values()),AMP_skips=sum(x['progress']['AMP_skips']for x in arms.values()),
            actual_model_calls=session.forward_counts,actual_backward_calls=session.backward_counts,generation_counts={'B':arms['B']['generation_counts']},startup_seconds=startup,gpu_preflight=preflight,
            wall_seconds=time.perf_counter()-began,physical_steps=0,new_samples=0,automatic_next_experiment=False)
        atomic_json(root/'training/result.json',result);check_storage(root,config,full=True);return result
    except BaseException as exc:
        failure=dict(error_type=type(exc).__name__,message=str(exc),automatic_retry=False,wall_seconds=time.perf_counter()-began,
            current_arm=None if session is None else session.arm,progress=None if session is None else session.progress,
            current_optimizer_boundary=None if session is None else session.clean,completed_arms=arms,completed_evaluations=evaluations,prefix_checkpoints=prefix,
            actual_model_calls=None if session is None else session.forward_counts,actual_backward_calls=None if session is None else session.backward_counts,
            generation_counts=None if session is None or session.history_generator is None else {k:getattr(session.history_generator,k)for k in ('batch_calls','generated_frames','denoiser_batch_forwards')},
            timings=None if session is None else {k:dict(v)for k,v in session.timings.items()},
            pending_stage=None if session is None or session.progress is None else session.progress['pending_attempt'],
            checkpoint_cursors=None if session is None else {k:max((v['cursor']for v in values.values()),default=0)for k,values in session.checkpoints.items()},
            partial_semantics='batches_consumed is a reservationcursor beforefourtargetwork; segments/target_frames_reserved are plannedwithinthatcursor. segments/target_frames_exposed count completedloss computations. Forwardhooks retain attempted/completedmodelcalls; backwardobserverretains attempted/completedbackwards. Pendingh/stage may be incomplete; no automaticreplay. Timingdict excludes unfinishedGPUeventinterval.')
        atomic_json(root/'training/failure.json',failure);raise
    finally:
        if session is not None:session.close()
