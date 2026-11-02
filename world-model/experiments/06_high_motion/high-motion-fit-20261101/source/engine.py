"""One bounded, deliberately repeated high-motion training-fit line.

Original RGB/actions/native residual/Adam/noise. No collection, generated
training history, region loss or import-time Torch/model/GPU operation.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np

REPO=Path(__file__).resolve().parents[2]
PARENT=REPO/'artifacts/oc-continuation-20261031/training/epoch2.pt'
PARENT_SHA='c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187'
MILESTONES=(250,500,1000,2000)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def initial_progress():
    return dict(attempted_batches=0,successful_updates=0,AMP_skips=0,reserved_exposures=0,actual_exposures=0,
        completed_backwards=0,supervised_forwards=0,lineage_updates=11367,complete_optimizer_boundary=True,
        pending_attempt=None,consecutive_AMP_overflows=0,finished=False,stop_reason=None,evaluated_milestones=[])

def budget_allows(p,c):
    return (p['successful_updates']<int(c.get('max_successful_updates',3000))and p['attempted_batches']<int(c.get('max_attempts',3000))
        and p['reserved_exposures']+16<=int(c.get('max_supervised_exposures',48000)))

class StopRule:
    """Only two full48 passes >=250 successful updates apart can stop early."""
    def __init__(self):self.checks=[]
    def consider(self,updates,result):
        unknown=result.get('full_train_unknown_count',result.get('full_train_unknown_conditions',0))
        if isinstance(unknown,(list,dict)):unknown=len(unknown)
        passed=result.get('full_train_all_passed')is True and unknown==0
        previous=[x for x in self.checks if x['passed']and updates-x['successful_updates']>=250]
        row=dict(successful_updates=int(updates),passed=passed,unknown_count=unknown,metrics=result.get('metrics'))
        self.checks.append(row);return bool(passed and previous)

class CudaTimer:
    def __init__(self):self.events=[]
    def start(self):
        import torch
        x=torch.cuda.Event(enable_timing=True);x.record();return x
    def stop(self,key,start):
        import torch
        end=torch.cuda.Event(enable_timing=True);end.record();self.events.append((key,start,end))
    def finish(self):
        import torch
        torch.cuda.synchronize();out=defaultdict(float)
        for key,a,b in self.events:out[key]+=a.elapsed_time(b)/1000
        return dict(out)

class Session:
    def __init__(self,root,c):
        import torch
        from common import atomic_json,check_storage
        from data import FitDataset,FrozenSampler
        from tdyssey.gpc_world_model import load_world_model_source,world_model_spec
        from tdyssey.raw_world_model_training import OPTIMIZER,make_generator
        from tdyssey.native_rgb_training import capture_rng,restore_rng
        from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs,POLICY_SHA
        from tdyssey.native_rgb_r2c1r1_training import _tree_digest
        self.root=Path(root);self.c=c;self.torch=torch;self.atomic_json=atomic_json;self.check_storage=check_storage
        self.output=self.root/'training';self.output.mkdir(exist_ok=True);self.parent_path=Path(c.get('starting_checkpoint',PARENT))
        if sha(self.parent_path)!=PARENT_SHA or c.get('starting_checkpoint_sha256',PARENT_SHA)!=PARENT_SHA:raise ValueError('Exact common epoch2full checkpoint required')
        torch.set_num_threads(2);parent=torch.load(self.parent_path,map_location='cpu',weights_only=False)
        required={'model','optimizer','scaler','global_rng','conditioning_rngs','progress','legacy_scheduler_provenance','order'}
        if parent.get('kind')!='oc_native_rgb_complete.v1'or not required.issubset(parent):raise ValueError('Complete epoch2 parent payload required')
        p=parent['progress']
        if p['epoch']!=2 or not p['epoch_complete']or p['lineage_updates']!=11367 or not p['complete_optimizer_boundary']or p['pending_attempt']is not None:raise ValueError('Parent is not the completed11367boundary')
        self.parent=parent;self.dataset=FitDataset(self.root,c);self.sampler=FrozenSampler(self.root,c)
        self.rows=self.dataset.rows
        if len(self.rows)!=768 or len({x['family_id']for x in self.rows})!=48 or {x['letter']for x in self.rows}!={'O','C','B'}:raise ValueError('Only frozen48trainingfamilies/768targets O/C/B permitted')
        self.row_ids=[x.get('target_id',x.get('window_id'))for x in self.rows]
        if len(set(self.row_ids))!=768 or any(x is None for x in self.row_ids):raise ValueError('Unique frozen target identifiers required')
        if not torch.cuda.is_available():raise RuntimeError('Authorized localCUDA unavailable; no other device/initialization fallback')
        check_storage(self.root,c,full=True);self.device=torch.device('cuda');torch.cuda.reset_peak_memory_stats()
        self.source=load_world_model_source(c.get('source_root',str(REPO.parent/'Tdyssey/third_party/gpc_code')))
        inner=deepcopy(world_model_spec()['model']);outer={k:inner.pop(k)for k in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
        with torch.random.fork_rng(devices=[]):self.model=self.source.Denoiser(self.source.DenoiserConfig(inner_model=self.source.InnerModelConfig(**inner),**outer)).float()
        self.model.to(self.device).requires_grad_(True);self.model.setup_training(SimpleNamespace(loc=-1.2,scale=1.2,sigma_min=.002,sigma_max=20.));self.model.train()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),**dict(OPTIMIZER,lr=1e-5));self.scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=128.)
        self.model.load_state_dict(parent['model'],strict=True);self.optimizer.load_state_dict(parent['optimizer']);self.scaler.load_state_dict(parent['scaler'])
        if any(abs(g['lr']-1e-5)>1e-14 for g in self.optimizer.param_groups):raise ValueError('Parent Adam heldLR differs')
        self.generator=make_generator(5478,next(self.model.parameters()).device);restore_rng(parent['global_rng'],self.generator)
        self.rngs=make_conditioning_rngs(**parent['conditioning_rngs']['seeds'],device=next(self.model.parameters()).device);self.rngs.load_state_dict(parent['conditioning_rngs'])
        if parent['conditioning_rngs']['policy_sha256']!=POLICY_SHA:raise ValueError('Original50/50cleanweak policy required')
        self.td=lambda v:_tree_digest(v,torch)
        proof=dict(parent_path=str(self.parent_path),parent_sha256=PARENT_SHA,lineage_updates=11367,
            components=dict(model=self.td(self.model.state_dict())==self.td(parent['model']),Adam=self.td(self.optimizer.state_dict())==self.td(parent['optimizer']),
            scaler=self.td(self.scaler.state_dict())==self.td(parent['scaler']),global_rng=self.td(capture_rng(self.generator))==self.td(parent['global_rng']),
            conditioning_rngs=self.td(self.rngs.state_dict())==self.td(parent['conditioning_rngs'])),new_data_sampler=self.sampler.state_dict(),
            old_order_provenance_only=True,optimizer_reset=False,noise_stream_reset=False)
        proof['passed']=all(proof['components'].values());atomic_json(self.output/'restore-proof.json',proof)
        if not proof['passed']:raise RuntimeError('Actual full restore differs before any model work')
        self.progress=initial_progress();self.windows=Counter();self.families=Counter();self.letters=Counter();self.balanced_draws=Counter()
        self.first_complete_cover=None;self.timings=defaultdict(float);self.forward_counts={};self.backward_counts=Counter();self.activity='training';self.application_count=0
        self.pending_metadata=[];self.clean=True;self.rng_proofs=[];self.last_full=None;self.best_full=None;self.best_score=None;self.best_label=None
        self.optimizer_hook=self.optimizer.register_step_post_hook(self._applied)
        self.prehook=self.model.inner_model.register_forward_pre_hook(self._forward)
        self.posthook=self.model.inner_model.register_forward_hook(self._forward_complete)
        selection_path=Path(c.get('selection_index',self.root/'selection/index.json'));order_path=Path(c.get('order_path',self.root/'selection/order.json'))
        self.identity=dict(kind='high-motion-native-real-history-fit.v1',parent_checkpoint_sha256=PARENT_SHA,selection_index_sha256=sha(selection_path),
            order_sha256=sha(order_path),max_successful_updates=3000,max_attempts=3000,max_supervised_exposures=48000,unique_train_targets=768,
            families=48,letters=['O','C','B'],microbatch=4,effective_batch=16,learning_rate=1e-5,conditioning_policy_sha256=POLICY_SHA,
            objective='OriginalFP32nativephase2single-stepresidualMSE,allrealhistory,unchanged50/50cleanweakpolicy',
            no_generated_training_history=True,no_region_loss=True,data_policy='Frozen12strong+4complete targets, intentionallyrepeatedsmalltrainingfitset; notnewdata or generalization')
        atomic_json(self.output/'identity.json',self.identity);self.log=(self.output/'losses.jsonl').open('a',encoding='utf-8',buffering=65536)
        self.reservations=(self.output/'reservations.jsonl').open('a',encoding='utf-8',buffering=1)
    def _applied(self,*args):self.application_count+=1
    def _forward(self,module,args):
        row=self.forward_counts.setdefault(self.activity,defaultdict(int));row['batch_forward_attempts']+=1;row['image_forward_attempts']+=len(args[0])
    def _forward_complete(self,module,args,output):
        row=self.forward_counts.setdefault(self.activity,defaultdict(int));row['batch_forward_completed']+=1;row['image_forward_completed']+=len(args[0])
    def _observe(self,event,micro,n):
        self.progress['pending_attempt'].update(microbatch=micro,stage=event)
        if event=='loss_complete':
            self.progress['actual_exposures']+=n
            for row in self.pending_metadata[micro*4:micro*4+n]:
                wid=row.get('target_id',row.get('window_id'));self.windows[wid]+=1;self.families[row['family_id']]+=1;self.letters[row['letter']]+=1
            if self.first_complete_cover is None and len(self.windows)==768:self.first_complete_cover=dict(attempt=self.progress['attempted_batches'],exposures=self.progress['actual_exposures'])
        if event=='backward_attempt':self.backward_counts['attempted_microbatches']+=1
        if event=='backward_complete':self.backward_counts['completed_microbatches']+=1;self.backward_counts['completed_target_frames']+=n;self.progress['completed_backwards']+=1
    def exposure_summary(self):
        return dict(unique_targets=len(self.windows),actual_exposures=self.progress['actual_exposures'],repeated_exposures=self.progress['actual_exposures']-len(self.windows),
            unique_families=len(self.families),target_counts=dict(self.windows),family_counts=dict(self.families),letter_counts=dict(self.letters),
            first_complete_unique_cover=self.first_complete_cover,minimum_per_target=min(self.windows.values(),default=0),maximum_per_target=max(self.windows.values(),default=0))
    def ctx(self,checkpoint):
        return SimpleNamespace(model=self.model,source=self.source,checkpoint_identity=checkpoint,successful_updates=self.progress['successful_updates'],
            exposure_count=self.progress['actual_exposures'],attempted_batches=self.progress['attempted_batches'])
    def protected(self,label,callback):
        from tdyssey.native_rgb_training import capture_rng,restore_rng
        state=capture_rng(self.generator);noise=self.rngs.state_dict();mode=self.model.training;sampler=deepcopy(self.sampler.state_dict());progress=deepcopy(self.progress)
        counter=deepcopy(self.windows);applied=self.application_count;activity=self.activity;self.activity=label;started=time.perf_counter();error=None
        before=dict(global_rng=self.td(state),conditioning_rngs=self.td(noise),sampler=self.td(sampler))
        try:return callback()
        except BaseException as exc:error=repr(exc);raise
        finally:
            restore_rng(state,self.generator);self.rngs.load_state_dict(noise);self.model.train(mode);self.activity=activity
            after=dict(global_rng=self.td(capture_rng(self.generator)),conditioning_rngs=self.td(self.rngs.state_dict()),sampler=self.td(self.sampler.state_dict()))
            proof=dict(label=label,before=before,after=after,matches=before==after,mode_restored=self.model.training==mode,
                optimizer_applications=self.application_count-applied,progress_unchanged=progress==self.progress,counter_unchanged=counter==self.windows,
                wall_seconds=time.perf_counter()-started,error=error)
            self.rng_proofs.append(proof);self.atomic_json(self.output/'rng-preservation.json',self.rng_proofs)
            if not proof['matches']or not proof['mode_restored']or proof['optimizer_applications']or not proof['progress_unchanged']or not proof['counter_unchanged']:raise RuntimeError('Evaluation altered training order/noise/mode/progress')
    def light_weights(self,label):
        from common import replace_atomic
        from tdyssey.native_rgb_training import _cpu_tree
        started=time.perf_counter();folder=self.output/'weights';folder.mkdir(exist_ok=True);path=folder/f'{label}.pt';temp=path.with_name(path.name+'.tmp')
        if path.exists():raise FileExistsError('No repeated checkpoint milestone')
        self.check_storage(self.root,self.c,reserve_bytes=48*1024**2)
        try:
            self.torch.save(dict(kind='high_motion_fit_weights.v1',model=_cpu_tree(self.model.state_dict()),identity=self.identity,progress=deepcopy(self.progress)),temp)
            with temp.open('rb+')as f:os.fsync(f.fileno())
            replace_atomic(temp,path)
        finally:temp.unlink(missing_ok=True)
        receipt=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,label=label,successful_updates=self.progress['successful_updates'],actual_exposures=self.progress['actual_exposures'])
        self.timings['lightweight_save_seconds']+=time.perf_counter()-started;return receipt
    def save_full(self,reason):
        from common import replace_atomic
        from tdyssey.native_rgb_training import _cpu_tree,capture_rng
        if not self.clean or not self.progress['complete_optimizer_boundary']or self.progress['pending_attempt']is not None:raise RuntimeError('No full checkpoint from uncertain optimizer/data reservation state')
        started=time.perf_counter();self.check_storage(self.root,self.c,reserve_bytes=128*1024**2)
        self.log.flush();os.fsync(self.log.fileno());self.reservations.flush();os.fsync(self.reservations.fileno())
        path=self.output/'latest.pt';temp=path.with_name('latest.pt.tmp')
        payload=dict(kind='high_motion_fit_full.v1',model=_cpu_tree(self.model.state_dict()),optimizer=_cpu_tree(self.optimizer.state_dict()),scaler=_cpu_tree(self.scaler.state_dict()),
            global_rng=capture_rng(self.generator),conditioning_rngs=self.rngs.state_dict(),sampler=deepcopy(self.sampler.state_dict()),
            progress=deepcopy(self.progress),exposures=self.exposure_summary(),identity=self.identity,actual_model_calls=deepcopy(self.forward_counts),
            actual_backward_calls=dict(self.backward_counts),legacy_scheduler_provenance=deepcopy(self.parent['legacy_scheduler_provenance']),
            parent_checkpoint=dict(path=str(self.parent_path),sha256=PARENT_SHA),parent_order_provenance=deepcopy(self.parent['order']),reason=reason)
        try:
            self.torch.save(payload,temp)
            with temp.open('rb+')as f:os.fsync(f.fileno())
            replace_atomic(temp,path)
        finally:temp.unlink(missing_ok=True)
        receipt=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,successful_updates=self.progress['successful_updates'],attempted_batches=self.progress['attempted_batches'],
            actual_exposures=self.progress['actual_exposures'],sampler_cursor=self.sampler.state_dict()['cursor'],lineage_updates=self.progress['lineage_updates'])
        self.last_full=receipt;self.timings['full_checkpoint_seconds']+=time.perf_counter()-started;self.atomic_json(self.output/'latest.json',receipt);return receipt
    def alias_full(self,name,source):
        from common import replace_atomic
        path=self.output/name;temp=path.with_name(path.name+'.link.tmp');temp.unlink(missing_ok=True)
        self.check_storage(self.root,self.c,reserve_bytes=source['bytes'])
        try:
            try:os.link(source['path'],temp)
            except OSError:
                import shutil
                shutil.copyfile(source['path'],temp)
            replace_atomic(temp,path)
        finally:temp.unlink(missing_ok=True)
        receipt=dict(source,path=str(path));self.atomic_json(path.with_suffix('.json'),receipt);return receipt
    def maybe_best(self,evaluated,label,scope):
        score=evaluated.get('panel_fit_score',evaluated.get('fit_score'))
        if score is None or not math.isfinite(float(score)):return
        if self.best_score is None or float(score)<self.best_score:
            self.best_score=float(score);self.best_label=label;latest=self.save_full('best_fixed_panel_fit')
            self.best_full=self.alias_full('best-fit.pt',latest)
            self.atomic_json(self.output/'best-fit-selection.json',dict(label=label,score=self.best_score,comparison_set='same frozen12trainingpanel',scope=scope,checkpoint=self.best_full,
                qualification='Training-fit checkpointselection, not held-outscene performance or allcriteria pass'))
    def train_attempt(self):
        from training_core import microbatch_backward
        from tdyssey.native_rgb_r2c3_training import scaled_optimizer_step
        if not budget_allows(self.progress,self.c):raise RuntimeError('Training budget exhausted; no furtherdraw')
        torch=self.torch;began=time.perf_counter();started=time.perf_counter();indices=self.sampler.next_batch();meta=self.dataset.metadata(indices)
        if len(indices)!=16 or len(meta)!=16 or any(not bool(x.get('strong',x.get('obvious',False)))for x in meta[:12]):raise ValueError('Frozen12strong/4full draw contract failed')
        self.progress['attempted_batches']+=1;self.progress['reserved_exposures']+=16
        self.progress['pending_attempt']=dict(batch=self.progress['attempted_batches']-1,indices=list(map(int,indices)),target_ids=[x.get('target_id',x.get('window_id'))for x in meta],stage='data',optimizer_application='not_started')
        self.reservations.write(json.dumps(dict(batch=self.progress['attempted_batches']-1,indices=list(map(int,indices)),sampler_cursor=self.sampler.state_dict()['cursor']))+'\n')
        self.balanced_draws.update(x['letter']for x in meta)
        arrays=self.dataset.model_batch(indices);self.timings['data_wait_seconds']+=time.perf_counter()-started
        if (np.asarray(arrays['image']).shape!=(16,5,3,96,96)or arrays['image'].dtype!=np.float32 or arrays['action'].dtype!=np.float32):raise ValueError('Exact original normalizedrealRGB/action loader contract required')
        if self.progress['attempted_batches']%3==0:
            if self.balanced_draws!=Counter({'O':16,'C':16,'B':16}):raise ValueError('Frozen three-batch letter exposure ratio differs')
            self.balanced_draws.clear()
        self.pending_metadata=meta;self.clean=False;self.progress['complete_optimizer_boundary']=False;timer=CudaTimer();stamp=timer.start()
        images=torch.as_tensor(arrays['image'],device=self.device);actions=torch.as_tensor(arrays['action'],device=self.device);timer.stop('H2D_seconds',stamp)
        self.optimizer.zero_grad(set_to_none=True);out=microbatch_backward(self.model,images,actions,self.rngs,self.scaler,timer=timer,observer=self._observe)
        stamp=timer.start();self.scaler.unscale_(self.optimizer);params=list(self.model.parameters())
        if any(p.grad is None for p in params):raise FloatingPointError('Missing native full-network gradient')
        finite=torch.stack([out['forward_finite'],torch.stack([torch.isfinite(p).all()for p in params]).all(),torch.stack([torch.isfinite(p.grad).all()for p in params]).all()]).cpu().tolist()
        if not finite[0]or not finite[1]:raise FloatingPointError('Nonfinite nativeforward/loss/parameters')
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)if finite[2]else None
        self.progress['pending_attempt'].update(stage='optimizer',optimizer_application='started')
        outcome=scaled_optimizer_step(self.scaler,self.optimizer,application_count=lambda:self.application_count)
        if outcome['applied']:
            self.progress['successful_updates']+=1;self.progress['lineage_updates']+=1;self.progress['consecutive_AMP_overflows']=0
        else:self.progress['AMP_skips']+=1;self.progress['consecutive_AMP_overflows']+=1
        if not bool(torch.stack([torch.isfinite(p).all()for p in params]).all()):raise FloatingPointError('Nonfinite parameter afterAdam')
        self.optimizer.zero_grad(set_to_none=True);timer.stop('optimizer_checks_seconds',stamp)
        for key,value in timer.finish().items():self.timings[key]+=value
        self.progress['supervised_forwards']+=4;self.progress['complete_optimizer_boundary']=True;self.progress['pending_attempt']=None;self.clean=True
        wall=time.perf_counter()-began;self.timings['training_loop_wall_seconds']+=wall
        self.progress['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();self.progress['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
        self.log.write(json.dumps(dict(attempt=self.progress['attempted_batches']-1,indices=list(map(int,indices)),target_ids=[x.get('target_id',x.get('window_id'))for x in meta],
            strong_targets=sum(bool(x.get('strong',x.get('obvious',False)))for x in meta),loss=float(out['loss']),gradient_norm=None if norm is None else float(norm),branches=out['branches'],**outcome,wall_seconds=wall))+'\n')
        if self.progress['attempted_batches']%32==0:
            self.log.flush();self.persist_progress()
        if self.progress['consecutive_AMP_overflows']>=16:raise FloatingPointError('16consecutiveAMPoverflows; no automaticretries')
        if self.progress['peak_cuda_reserved_bytes']>int(self.c.get('max_cuda_bytes',6*1024**3)):raise RuntimeError('AuthorizedCUDAresourceenvelopeexceeded')
        return outcome
    def persist_progress(self):
        self.atomic_json(self.output/'progress.json',dict(self.progress,unique_targets=len(self.windows),repeated_exposures=self.progress['actual_exposures']-len(self.windows),
            unique_families=len(self.families),timings=dict(self.timings),actual_model_calls=deepcopy(self.forward_counts),actual_backward_calls=dict(self.backward_counts)))
    def close(self):
        self.log.close();self.reservations.close();self.optimizer_hook.remove();self.prehook.remove();self.posthook.remove();self.optimizer.zero_grad(set_to_none=True);self.dataset.close()

def throughput_record(root,c,session,baseline,startup):
    p=session.progress;t=dict(session.timings);wall=t['training_loop_wall_seconds'];left_attempts=min(int(c.get('max_attempts',3000))-p['attempted_batches'],(int(c.get('max_supervised_exposures',48000))-p['reserved_exposures'])//16)
    baseline_frames=baseline.get('actual_predicted_frames',3168);evalwall=float(baseline.get('wall_seconds',0))
    per_frame=evalwall/baseline_frames if baseline_frames else None;base_remaining_frames=4*384+3168;optional_frames=4*1536
    eval_est=per_frame*base_remaining_frames if per_frame is not None else None;train_est=left_attempts*wall/p['attempted_batches']
    return dict(run_id=c['run_id'],config_sha256=sha(Path(root)/'runtime-config.json'),worker_pid=os.getpid(),measured_at_unix=time.time(),
        successful_updates=p['successful_updates'],attempted_batches=p['attempted_batches'],AMP_skips=p['AMP_skips'],actual_supervised_exposures=p['actual_exposures'],unique_targets=len(session.windows),
        repeated_exposures=p['actual_exposures']-len(session.windows),updates_per_second=p['successful_updates']/wall,target_frames_per_second=p['actual_exposures']/wall,
        seconds_per_attempt=wall/p['attempted_batches'],training_only_wall_seconds=wall,timings=t,startup_including_baseline_seconds=startup,
        baseline_evaluation_wall_seconds=evalwall,baseline_predicted_frames=baseline_frames,measured_evaluation_frames_per_second=baseline_frames/evalwall if evalwall else None,
        remaining_attempt_budget=left_attempts,remaining_success_upper_bound=min(3000-p['successful_updates'],left_attempts),estimated_remaining_training_seconds=train_est,
        estimated_base_remaining_evaluation_seconds=eval_est,base_remaining_evaluation_predicted_frames=base_remaining_frames,
        optional_full48_gate_predicted_frames_upper_bound=optional_frames,optional_full48_gate_seconds_upper_proxy=per_frame*optional_frames if per_frame else None,
        estimated_base_remaining_total_range_seconds=None if eval_est is None else [.7*train_est+.5*eval_est,1.5*train_est+2*eval_est+120],
        peak_cuda_allocated_bytes=p['peak_cuda_allocated_bytes'],peak_cuda_reserved_bytes=p['peak_cuda_reserved_bytes'],continues_automatically=True,
        uncertainty='Measuredfirst32successes; AMPskipattempts/exposuresincluded. TrainETAusesattemptbudget,notextra draws toforce3000successes. EvaluationETAusesactualfullbaselineframes/wall; panels/final/noiseextra can differ, optionalfull48gates andreports/checkpointsaddcost. Nofixedtimedeadline; intentionalrepeatfitnotnewdata.')

def run(root,c):
    from common import atomic_json,check_storage
    import evaluation
    root=Path(root).resolve();(root/'training').mkdir(exist_ok=True);began=time.perf_counter();session=None;results={};stop=StopRule();weights={};throughput=None
    atomic_json(root/'training/execution-claim.json',dict(run_id=c['run_id'],pid=os.getpid(),single_line=True,automatic_retry=False),exclusive=True)
    try:
        if hasattr(evaluation,'freeze'):evaluation.freeze(root,c)
        session=Session(root,c);weights['0']=session.light_weights('0');initial=session.save_full('initial_complete_fit_state')
        baseline=session.protected('evaluation_0',lambda:evaluation.run_checkpoint(root,c,session.ctx(dict(path=str(session.parent_path),sha256=PARENT_SHA,label='0')),'0',scope='full',extra_seeds=True))
        results['0']=baseline;atomic_json(root/'background/baseline-result.json',baseline);stop.consider(0,baseline);session.maybe_best(baseline,'0','full')
        startup=time.perf_counter()-began
        while budget_allows(session.progress,c):
            outcome=session.train_attempt();p=session.progress;updates=p['successful_updates']
            if throughput is None and updates>=32:
                recovery=session.save_full('first32successes');throughput=throughput_record(root,c,session,baseline,startup);throughput['recovery_checkpoint']=recovery
                atomic_json(root/'throughput.json',throughput);atomic_json(root/'training/throughput.json',throughput)
            if outcome['applied']and updates%100==0:session.save_full('100success_boundary')
            if updates in MILESTONES and updates not in p['evaluated_milestones']:
                label=str(updates);weights[label]=session.light_weights(label)
                panel=session.protected('evaluation_'+label,lambda:evaluation.run_checkpoint(root,c,session.ctx(dict(weights[label],label=label)),label,scope='panel',extra_seeds=False))
                results[label]=panel;p['evaluated_milestones'].append(updates);session.maybe_best(panel,label,'panel')
                if panel.get('panel_all_passed')is True:
                    full_label=label+'-train_full'
                    full=session.protected('evaluation_'+full_label,lambda:evaluation.run_checkpoint(root,c,session.ctx(dict(weights[label],label=full_label)),full_label,scope='train_full',extra_seeds=False))
                    results[full_label]=full;early=stop.consider(updates,full)
                    atomic_json(root/'training/full-fit-checks.json',stop.checks)
                    if early:p['stop_reason']='two_full48_criteria_passes';break
                session.persist_progress()
        p=session.progress
        if p['stop_reason']is None:p['stop_reason']='successful_update_limit'if p['successful_updates']>=3000 else'exposure_or_attempt_limit'
        final_label=str(p['successful_updates'])
        if final_label not in weights:weights[final_label]=session.light_weights(final_label)
        weights['final']=weights[final_label]
        final=session.protected('evaluation_final',lambda:evaluation.run_checkpoint(root,c,session.ctx(dict(weights['final'],label='final')),'final',scope='full',extra_seeds=True))
        results['final']=final;session.maybe_best(final,'final','full');p['finished']=True;session.persist_progress()
        latest=session.save_full('final_complete_boundary');final_checkpoint=session.alias_full('final.pt',latest)
        if session.best_full is None:session.best_full=session.alias_full('best-fit.pt',latest)
        result=dict(complete=True,stop_reason=p['stop_reason'],progress=deepcopy(p),exposures=session.exposure_summary(),evaluations=results,lightweight_checkpoints=weights,
            latest=latest,best_fit=session.best_full,best_fit_score=session.best_score,best_fit_label=session.best_label,final=final_checkpoint,
            throughput=None if throughput is None else str(root/'throughput.json'),full_fit_checks=stop.checks,timings=dict(session.timings),
            actual_model_calls=session.forward_counts,actual_backward_calls=dict(session.backward_counts),restore_proof=str(root/'training/restore-proof.json'),
            RNG_proofs=str(root/'training/rng-preservation.json'),rng_preservation=str(root/'training/rng-preservation.json'),wall_seconds=time.perf_counter()-began,physical_steps=0,new_data=0,
            generated_training_history_frames=0,region_weighted_loss=False,training_is_deliberate_repeated_fit=True,automatic_next_experiment=False)
        atomic_json(root/'training/result.json',result);check_storage(root,c,full=True);return result
    except BaseException as exc:
        atomic_json(root/'training/failure.json',dict(type=type(exc).__name__,message=str(exc),wall_seconds=time.perf_counter()-began,
            progress=None if session is None else session.progress,exposures=None if session is None else session.exposure_summary(),
            sampler_state=None if session is None else session.sampler.state_dict(),latest_complete_checkpoint=None if session is None else session.last_full,
            actual_model_calls=None if session is None else session.forward_counts,actual_backward_calls=None if session is None else dict(session.backward_counts),
            current_optimizer_state_uncertain=None if session is None else not session.clean,completed_evaluations=results,automatic_retry=False,
            partial_semantics='Reserved16draws countagainst48000; actualexposurescountcompletedlossmicrobatches. Neverclaimstalecheckpointexactresumeifreservationcursoradvancedbeyondit; noauto replay.'))
        raise
    finally:
        if session is not None:session.close()
