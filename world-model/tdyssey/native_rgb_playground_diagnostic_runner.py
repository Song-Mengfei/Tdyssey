"""Finite, replayable image-only R3B playground diagnostics; no training.

The truth path is generated once. Corrected H1 and raw-feedback H4 use the same
recorded actions/seeds; future truth is never provided to the open-loop branch.
Importing this file does not load Torch or construct a simulator.
"""
from __future__ import annotations
from collections import deque
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/run/last.pt'
CHECKPOINT_SHA='23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7'
TRAINING_PHYSICS_SHA='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6'
EVALUATION_PHYSICS_SHA='d83b4c544b1e5193c92e2b882d0904995b1d49e174adaf0e8c5b284031835e9b'
SAMPLER=dict(sigma_min=.002,sigma_max=5,rho=7,order=1,s_churn=0,s_cond=0)
LETTERS=tuple('TLOCXB')

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def safe(v):
    if isinstance(v,np.ndarray):return safe(v.tolist())
    if isinstance(v,np.generic):return safe(v.item())
    if isinstance(v,dict):return{str(k):safe(x)for k,x in v.items()}
    if isinstance(v,(list,tuple,deque)):return[safe(x)for x in v]
    if isinstance(v,Path):return str(v)
    if isinstance(v,float)and not math.isfinite(v):return 'positive_infinity'if v>0 else'negative_infinity'if v<0 else'NaN'
    return v
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n')as stream:
        json.dump(safe(value),stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,p)
def npz(p,**arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('wb')as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,p)


def _world_geometry(letter,center,angle):
    from .envs.official import official_shape
    from shapely.geometry import box
    from shapely.ops import unary_union
    from shapely import affinity
    shape=official_shape(letter)
    return affinity.translate(affinity.rotate(unary_union([box(r.x0,r.y0,r.x1,r.y1)for r in shape.components]),
        angle,origin=(0,0),use_radians=True),*center)


def make_plan():
    """Geometry-only fixed proposal: no renderer, physics, model or seed search."""
    from shapely.geometry import Point,Polygon,LineString
    cases=[]
    for letter in('B','C','O','X'):
        for variant in('outer','inner_corner'):
            angle=0. if variant=='outer'or letter in('B','C','O')else math.pi/4
            center=np.asarray([256.,256.]);g=_world_geometry(letter,center,angle);x0,y0,x1,y1=g.bounds
            if variant=='outer':
                yy=246.;cut=g.intersection(LineString([(0,yy),(511,yy)]));left=cut.bounds[0]
                p=np.asarray([left-23.,yy])
                phases=[('contact_oblique_push',[center[0]+105,center[1]+18]),
                    ('withdraw',[x0-85,yy-20]),('recontact',[center[0]+108,center[1]-18]),
                    ('turn_contact',[center[0]+45,center[1]+100])]
            else:
                holes=list(getattr(g,'interiors',()))
                if holes:
                    # B upper hole and O's central hole; true initial clearance is
                    # validated before the sole environment initialization.
                    hole=min((Polygon(r)for r in holes),key=lambda h:h.centroid.y)
                    point=hole.centroid;p=np.asarray([point.x,point.y])
                    phases=[('inner_wall_contact',[p[0]+105,p[1]-8]),('withdraw_in_cavity',[p[0],p[1]+4]),
                        ('inner_recontact',[p[0]-105,p[1]+12]),('inner_corner_turn',[p[0]+45,p[1]+90])]
                elif letter=='C':
                    p=np.asarray([x1-24.,(y0+y1)/2])
                    phases=[('inner_wall_contact',[x0-60,p[1]-18]),('withdraw',[x1+75,p[1]]),
                        ('inner_recontact',[x0-60,p[1]+12]),('inner_corner_turn',[x1-10,y1+70])]
                else:
                    yy=y0+20;cut=g.intersection(LineString([(0,yy),(511,yy)]));p=np.asarray([cut.bounds[0]-24.,yy])
                    phases=[('corner_contact',[center[0]+80,center[1]+25]),('withdraw',[x0-70,y0-25]),
                        ('corner_recontact',[center[0]+105,center[1]-5]),('corner_rotation',[center[0]+10,center[1]+105])]
            if g.distance(Point(*p))<15-1e-8:raise ValueError('Fixed initial pusher intersects material: '+letter+variant)
            targets=[p.tolist()]*3+[list(map(float,target))for _,target in phases for _ in range(4)]
            if any(not(0<=q<=511)for target in targets for q in target):raise ValueError('Fixed target outside image world')
            case_id=f'{letter}-{variant}'
            cases.append(dict(case_id=case_id,letter=letter,variant=variant,center=center.tolist(),angle=angle,
                pusher=p.tolist(),initial_material_clearance=float(g.distance(Point(*p))),
                warmup_controls=3,diagnostic_controls=16,actions_absolute=targets,anchors=[3,7,11,15],
                phases=[dict(name='history_warmup',start_control=0,end_control=3,target=p.tolist())]+
                    [dict(name=name,start_control=3+4*i,end_control=7+4*i,target=list(map(float,target)))for i,(name,target)in enumerate(phases)],
                representative_anchor=7,planned_mechanisms=['contact','withdrawal','recontact','offcenter_rotation'],
                preview_source='programmatic_fixed_targets_no_live_mouse',screenshot_status='same_failure_class_attempt_not_exact_user_history'))
    return dict(kind='R3B-playground-fixed-diagnostic-plan.v1',is_proposal=True,execution_authorized=False,
        training_authorized=False,checkpoint_path=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,lineage_updates=7499,
        checkpoint_training_physics_sha256=TRAINING_PHYSICS_SHA,physics_profile='original-kinematic',
        cross_physics_preview=True,physics_protocol_sha256=EVALUATION_PHYSICS_SHA,
        cases=cases,planned_cases=8,planned_controls=152,planned_full_H4_clips=32,
        sampler_main=dict(num_steps_denoising=3,**SAMPLER),representative_steps=[8,16],seed_base=5478,
        seed_rule='5478 + diagnostic_control_index_after_three_warmup_controls + step_offset; same as UI steps, paired A/B and sampler counts',
        extra_seeds=[],planned_sampler_calls=320,planned_sampler_denoiser_forwards=1536,
        first_frame_check='independent_model_calls_identical_inputs_actions_seed',
        model_features=['RGB_history','normalized_executed_actions'],no_future_truth_in_H4=True,
        failure_selection='predeclared diagnostic cases; no population failure-rate claim; no replacement or adaptive search')


def body_state(body):
    return dict(position=list(body.position),angle=float(body.angle),velocity=list(body.velocity),
        angular_velocity=float(body.angular_velocity),force=list(body.force),torque=float(body.torque),
        center_of_gravity=list(body.center_of_gravity),mass=float(body.mass),moment=float(body.moment),
        body_type=int(body.body_type),is_sleeping=bool(body.is_sleeping))


def full_state(env):
    base=getattr(env,'_base',env)
    return safe(dict(state15=np.asarray(env.state(),np.float64),agent=body_state(base.agent),object=body_state(base.block),
        controller=dict(control_step=int(base.control_step),latest_action=None if base.latest_action is None else list(base.latest_action),
            kp=float(base.k_p),kv=float(base.k_v),controller=base.controller,
            action=None if getattr(env,'_action',None)is None else list(env._action),
            in_step=bool(getattr(env,'_in_step',False)),terminal=bool(getattr(env,'_terminal',False))),
        space=dict(damping=float(base.space.damping),iterations=int(base.space.iterations),
            gravity=list(base.space.gravity),collision_slop=float(base.space.collision_slop)),
        replay_convention='Exact diagnostic prediction replay uses saved RGB/actions/raw predictions. Public complete body/controller states and all initial-scene+executed-prefix/substep records are retained; private Chipmunk contact-cache is not serialized or claimed resumable from arbitrary intermediate state.'))


def generate_truth(case,config,output,status):
    """One real trajectory only. Both prediction methods later share its labels."""
    from .native_rgb_original_physics import make_original_env
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    complete=output/'complete.json'
    if complete.exists():
        prior=read(complete)
        if prior.get('case')!=case:raise ValueError('Saved truth case differs')
        return prior
    started=output/'started.json'
    if started.exists():raise RuntimeError('Incomplete truth attempt exists; refusing automatic recollection')
    write(started,dict(at_utc=utc(),case=case));env=None;frames=[];states=[];full=[];infos=[];failure=None;attempts=0
    began=time.perf_counter()
    try:
        env=make_original_env(config['physics_source_root'],case['letter'])
        first=env.reset_scene(center=case['center'],angle=case['angle'],pusher=case['pusher'])
        identity=env.initial_identity()
        if identity['protocol_sha256']!=config['physics_protocol_sha256']:raise RuntimeError('Actual original physics hash differs')
        if np.max(env.penetrations())>2:raise RuntimeError('Predeclared initial scene is invalid; do not replace it')
        frames.append(np.asarray(first,np.uint8).copy());states.append(np.asarray(env.state(),np.float64).copy());full.append(full_state(env))
        for index,target in enumerate(case['actions_absolute']):
            attempts+=1;status('truth_control_attempt',dict(case_id=case['case_id'],control=index,target=target))
            try:rgb,info=env.step(np.asarray(target,np.float64))
            except BaseException as exc:
                failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),
                    attempted_control=index,partial_info=safe(getattr(exc,'info',{})),partial_public_state=full_state(env))
                # A diagnosed physical guard preserves its partial evidence and
                # this case's missing labels. Other failures stop the whole run.
                if type(exc).__name__not in('R2AStepAborted','OriginalStepAborted'):raise
                break
            frames.append(np.asarray(rgb,np.uint8).copy());states.append(np.asarray(env.state(),np.float64).copy())
            full.append(full_state(env));infos.append(safe(info))
            write(output/'progress.json',dict(case_id=case['case_id'],completed_controls=len(infos),attempted_controls=attempts))
    except BaseException as exc:
        if failure is None:failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),
            attempted_controls=attempts,completed_controls=len(infos))
        raise
    finally:
        if env is not None:env.close()
        if frames:npz(output/'truth.npz',truth_rgb=np.stack(frames),states15=np.stack(states),
            actions_absolute=np.asarray(case['actions_absolute'][:len(infos)],np.float64))
        value=dict(kind='R3B-playground-recorded-truth.v1',case=case,physics_profile=config['physics_profile'],
            physics_protocol_sha256=config['physics_protocol_sha256'],initial_identity=locals().get('identity'),
            public_full_states=full,control_telemetry=infos,completed_controls=len(infos),attempted_controls=attempts,
            planned_controls=len(case['actions_absolute']),failure=failure,wall_seconds=time.perf_counter()-began,
            truth_archive=str(output/'truth.npz'),optimizer_updates=0,complete_trajectory=len(infos)==19)
        write(output/'case.json',value)
        if failure is None or type(failure).__name__=='dict'and failure['type']in('R2AStepAborted','OriginalStepAborted'):
            write(complete,value)
    return value


class RawPredictionFailure(RuntimeError):
    def __init__(self,message,partial,records):
        super().__init__(message);self.partial=partial;self.records=records


def paired_rollout(sample,truth,past_actions,future_actions,seed_steps,*,on_call=None):
    """The only network boundary; callback tests can prove lack of GT leakage."""
    truth=np.asarray(truth,np.float32);past=np.asarray(past_actions,np.float64);future=np.asarray(future_actions,np.float64)
    if truth.shape!=(8,3,96,96)or past.shape!=(3,2)or future.shape!=(4,2):raise ValueError('Four histories and four truth labels/actions required')
    outputs={};records={}
    all_actions=np.concatenate([past,future])
    for branch in('A','B'):
        history=truth[:4].copy();predictions=[];steps=[]
        for j in range(4):
            if branch=='A':history=truth[j:j+4].copy()
            action=(all_actions[j:j+4].astype(np.float32)/np.float32(511)*np.float32(2)-np.float32(1))
            record=dict(branch=branch,step=j+1,seed=int(seed_steps[j]),history_sha256=array_sha(history),
                normalized_actions_sha256=array_sha(action),action_normalization='/511*2-1',
                input_dtype=str(history.dtype),future_truth_in_model_input=False,
                conditioning='true_history_every_step'if branch=='A'else'initial_true_history_then_raw_prediction_only')
            if on_call:on_call('started',record)
            began=time.perf_counter();raw=np.asarray(sample(history[None].copy(),action[None].copy(),seed=int(seed_steps[j])))[0]
            finite_values=raw[np.isfinite(raw)]
            record.update(seconds=time.perf_counter()-began,raw_shape=list(raw.shape),raw_dtype=str(raw.dtype),
                raw_sha256=array_sha(raw),finite=bool(np.isfinite(raw).all()),
                raw_min=float(finite_values.min())if finite_values.size else None,raw_max=float(finite_values.max())if finite_values.size else None)
            if raw.shape!=(3,96,96)or raw.dtype!=np.float32:
                raise RawPredictionFailure('Raw native sampler shape/dtype changed',dict(**outputs,**{branch:predictions+[raw.copy()]}),dict(**records,**{branch:steps+[record]}))
            predictions.append(raw.copy());steps.append(record)
            if on_call:on_call('completed',record)
            if not record['finite']:
                raise RawPredictionFailure('Nonfinite raw prediction; no repair or retry',dict(**outputs,**{branch:predictions}),dict(**records,**{branch:steps}))
            if branch=='B':history=np.concatenate([history[1:],raw[None]],axis=0)
        outputs[branch]=np.stack(predictions);records[branch]=steps
    first_equal=bool(np.array_equal(outputs['A'][0],outputs['B'][0]))
    if not first_equal or records['A'][0]['history_sha256']!=records['B'][0]['history_sha256']or records['A'][0]['normalized_actions_sha256']!=records['B'][0]['normalized_actions_sha256']:
        raise RuntimeError('Independent paired first-step interface failed')
    return outputs,records


def motion_contact(case,anchor,states,infos,actions):
    from .envs.official import official_shape
    def center(state):
        x,y=official_shape(case['letter']).centroid;c,s=math.cos(state[4]),math.sin(state[4])
        return state[2:4]+np.asarray([c*x-s*y,s*x+c*y])
    result=[]
    for j in range(4):
        before,after=states[anchor+j],states[anchor+j+1];info=infos[anchor+j]
        result.append(dict(step=j+1,absolute_control_step=anchor+j+1,ui_control_step=anchor+j-2,
            translation_image96_px=float(np.linalg.norm(center(after)-center(before))*96/512),
            signed_rotation_deg=float(np.degrees(after[4]-before[4])),
            contact_substeps=int(info.get('agent_object_contact_substeps',0)),
            positive_impulse_substeps=int(info.get('agent_object_positive_impulse_substeps',0)),
            contact_arc_ids=info.get('actual_contact_arc_ids',[]),completed_substeps=int(info.get('substep_count',len(info.get('substeps',[])))),
            actual_target=np.asarray(actions[anchor+j]).tolist(),mouse_preview_target=None,
            preview_source='programmatic_case_no_live_mouse'))
    return result


class FrozenPredictor:
    def __init__(self,config):
        import torch
        from .gpc_world_model import load_world_model_source,world_model_spec
        from .native_rgb_r2c1r1_training import torch_native_sample,_tree_digest
        if sha(config['checkpoint_path'])!=CHECKPOINT_SHA:raise RuntimeError('Wrong checkpoint; expected immutable7499')
        if not torch.cuda.is_available():raise RuntimeError('Authorized local inference requires existing CUDA')
        torch.set_num_threads(2);self.torch=torch;self.samplers={};self.calls=0;self.forwards=0;self.seconds=0.;self.config=config
        self.source=load_world_model_source(config['model_source_root']);inner=dict(world_model_spec()['model'])
        denoiser={k:inner.pop(k)for k in('sigma_data','sigma_offset_noise','noise_previous_obs')}
        self.model=self.source.Denoiser(self.source.DenoiserConfig(inner_model=self.source.InnerModelConfig(**inner),**denoiser)).float()
        payload=torch.load(config['checkpoint_path'],map_location='cpu',weights_only=False)
        if payload['progress']['lineage_updates']!=7499:raise RuntimeError('Wrong model lineage')
        self.model.load_state_dict(payload['model'],strict=True);del payload
        self.model.to('cuda').eval().requires_grad_(False)
        self.before_digest=_tree_digest(self.model.state_dict(),torch);self.sample_factory=torch_native_sample;self.digest=_tree_digest
        self.actual_forward_attempts=0;self.actual_forward_completed=0
        self.handles=[self.model.inner_model.register_forward_pre_hook(lambda *_:setattr(self,'actual_forward_attempts',self.actual_forward_attempts+1)),
            self.model.inner_model.register_forward_hook(lambda *_:setattr(self,'actual_forward_completed',self.actual_forward_completed+1))]
    def sampler(self,steps):
        if steps not in(3,8,16):raise ValueError('Only fixed3/8/16 sampler comparison authorized')
        if steps not in self.samplers:self.samplers[steps]=self.sample_factory(self.model,self.source,dict(num_steps_denoising=steps,**SAMPLER))
        def call(history,actions,*,seed):
            began=time.perf_counter();self.calls+=1
            try:return self.samplers[steps](history,actions,seed=seed)
            finally:self.seconds+=time.perf_counter()-began
        return call
    def proof(self):
        after=self.digest(self.model.state_dict(),self.torch);checkpoint=sha(self.config['checkpoint_path'])
        return dict(checkpoint_sha256_before=CHECKPOINT_SHA,checkpoint_sha256_after=checkpoint,
            checkpoint_unchanged=checkpoint==CHECKPOINT_SHA,model_state_sha256_before=self.before_digest,
            model_state_sha256_after=after,model_weights_unchanged=self.before_digest==after,
            model_frozen=not self.model.training and all(not p.requires_grad for p in self.model.parameters()),
            optimizer_updates=0,optimizer_constructed=False,sampler_calls=self.calls,
            denoiser_forward_attempts=self.actual_forward_attempts,denoiser_forward_completed=self.actual_forward_completed,
            sampler_seconds=self.seconds,peak_cuda_allocated_bytes=self.torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=self.torch.cuda.max_memory_reserved(),device=self.torch.cuda.get_device_name(0))
    def close(self):
        for handle in self.handles:handle.remove()
        self.samplers.clear();self.model=None;self.torch.cuda.empty_cache()


class Status:
    def __init__(self,output,run_id):
        self.output=Path(output);self.lock=threading.RLock();self.stop=threading.Event();self.start=time.perf_counter()
        self.value=dict(run_id=run_id,pid=os.getpid(),started_at_utc=utc(),closed=False,phase='startup',progress_updates=0,heartbeat_sequence=0,optimizer_updates=0,training_authorized=False)
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def flush(self):
        self.value.update(at_utc=utc(),wall_seconds=time.perf_counter()-self.start);write(self.output/'status.json',self.value)
    def begin(self):self.flush();self.thread.start()
    def loop(self):
        while not self.stop.wait(15):
            with self.lock:self.value['heartbeat_sequence']+=1;self.flush()
    def __call__(self,event,details):
        with self.lock:
            self.value.update(last_event=event,last_details=safe(details),progress_updates=self.value['progress_updates']+1);self.flush()
            with(self.output/'events.jsonl').open('a',encoding='utf-8')as s:s.write(json.dumps(safe(dict(at_utc=utc(),event=event,**details)),allow_nan=False)+'\n');s.flush()
    def phase(self,name):self.value['phase']=name;self('phase_changed',dict(phase=name))
    def finish(self,success):
        self.stop.set();self.thread.join(timeout=2)
        with self.lock:self.value.update(closed=True,phase='complete'if success else'failed');self.flush()


def preflight(config_path):
    config=read(config_path);plan=read(config['plan_path'])
    if not(config.get('execution_authorized')is True and config.get('training_authorized')is False):raise PermissionError('Explicit inference-only release required')
    if config.get('physics_profile')!='original-kinematic'or config.get('physics_protocol_sha256')!=EVALUATION_PHYSICS_SHA:
        raise PermissionError('New direct-human original-physics override required')
    if plan['cases']!=make_plan()['cases']:raise ValueError('Fixed diagnostic scene/action plan changed')
    if sha(config['checkpoint_path'])!=CHECKPOINT_SHA:raise ValueError('Wrong immutable7499 checkpoint')
    if plan.get('extra_seeds')or config.get('extra_seeds'):raise ValueError('No extra seed search authorized in the fixed pass')
    if not config.get('input_bindings'):raise ValueError('Freeze diagnostic source/input identity before execution')
    for b in config['input_bindings']:
        if sha(b['path'])!=b['sha256']:raise ValueError('Bound diagnostic source changed: '+b['path'])
    output=Path(config['output']).resolve();allowed=ROOT/'artifacts/r3b-playground-diagnostics-20261030'
    if not output.is_relative_to(allowed)or output==allowed:raise ValueError('Use a new run subdirectory under diagnostic stage')
    if not config.get('shared_storage_roots'):raise ValueError('Explicit shared experiment storage roots required')
    if type(config.get('resume_authorized',False))is not bool:raise ValueError('resume_authorized must be a boolean')
    config['output']=str(output);return config,plan


class StorageGuard:
    def __init__(self,config):
        self.config=config;self.stage=ROOT/'artifacts/r3b-playground-diagnostics-20261030';files={}
        roots=[Path(p)for p in config['shared_storage_roots']]+[Path(p)for p in config.get('new_stage_source_files',[])]+[self.stage]
        for root in roots:
            for p in([root]if root.is_file()else root.rglob('*')if root.is_dir()else[]):
                if p.is_file():files[str(p.resolve()).lower()]=p
        shared=sum(p.stat().st_size for p in files.values());inside=sum(p.stat().st_size for p in self.stage.rglob('*')if p.is_file())
        self.external=shared-inside;self.initial=dict(shared_bytes=shared,diagnostic_stage_bytes=inside)
        self.check()
    def check(self):
        import shutil
        inside=sum(p.stat().st_size for p in self.stage.rglob('*')if p.is_file());reserve=256*1024**2
        free=shutil.disk_usage(self.stage).free;shared=self.external+inside;cap=int(self.config.get('shared_cap_bytes',3*1024**3))
        result=dict(shared_bytes_estimate=shared,diagnostic_stage_bytes=inside,shared_cap_bytes=cap,
            atomic_reserve_bytes=reserve,filesystem_free_bytes=free)
        if cap>3*1024**3:raise ValueError('Do not enlarge the human shared3GiB cap')
        result['passed']=shared+reserve<=cap and free>=reserve and inside<=int(self.config.get('diagnostic_stage_cap_bytes',cap-reserve))
        if not result['passed']:raise RuntimeError('Diagnostic storage/atomic budget exhausted: '+json.dumps(result))
        return result


def execute(config_path,*,resume=False):
    config,plan=preflight(config_path);output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    if(output/'completion.json').exists():raise RuntimeError('Closed run exists; no automatic replay')
    if(output/'once-claim.json').exists()and not resume:raise RuntimeError('Prior run started; explicit resume cannot replay partial groups')
    if resume and config.get('resume_authorized')is not True:raise PermissionError('Explicit authorized resume required; partial groups never replay')
    owner_path=output/'worker-owner.lock';owner_fd=os.open(owner_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(owner_fd,str(os.getpid()).encode());os.fsync(owner_fd)
    status=Status(output,config['run_id']);predictor=None;failure=None;began=time.perf_counter();truth_records=[];groups=[]
    new_groups=[];storage=None
    try:
        if not resume:
            with(output/'once-claim.json').open('x',encoding='utf-8')as s:json.dump(dict(pid=os.getpid(),at_utc=utc(),run_id=config['run_id']),s)
        from .native_rgb_r3b_loader_diagnostics import actual_pe,job
        write(output/'startup-proof.json',dict(run_id=config['run_id'],pid=os.getpid(),parent_pid=os.getppid(),actual_process_executable=actual_pe(),
            executable=sys.executable,prefix=sys.prefix,working_directory=str(Path.cwd().resolve()),job=job(),torch_imported='torch'in sys.modules))
        if config.get('require_detached_job',True)and job()['is_in_job']is not False:raise RuntimeError('Batch diagnostic worker is not independent')
        status.begin();status.phase('verify_inference_lease')
        # Prevent simultaneous GPU use with the user's live playground.
        import urllib.request
        try:
            with urllib.request.urlopen('http://127.0.0.1:8767/api/health',timeout=1)as response:
                if json.loads(response.read()).get('ready'):raise RuntimeError('Live playground still owns GPU; preserve/stop it before batch diagnostics')
        except (ConnectionError,OSError):pass
        storage=StorageGuard(config);write(output/'storage-preflight.json',storage.initial)
        status.phase('truth_trajectories')
        for case in plan['cases']:
            truth_records.append(generate_truth(case,config,output/'cases'/case['case_id'],status))
        status.phase('frozen_model_restore');predictor=FrozenPredictor(config)
        for case_record in truth_records:
            case=case_record['case'];archive=Path(case_record['truth_archive'])
            with np.load(archive,allow_pickle=False)as ar:
                rgb=ar['truth_rgb'];states=ar['states15'];actions=ar['actions_absolute']
            for anchor in case['anchors']:
                clip_id=f"{case['case_id']}-a{anchor}"
                if anchor+4>=len(rgb):
                    groups.append(dict(clip_id=clip_id,status='unavailable_true_future',completed_controls=case_record['completed_controls']));continue
                seed_steps=np.arange(4,dtype=np.int64)+int(plan['seed_base'])+anchor-3
                steps_list=[3]+([8,16]if anchor==case['representative_anchor']and case['variant']=='inner_corner'else[])
                for denoise_steps in steps_list:
                    folder=output/'clips'/clip_id/f'{denoise_steps}-s{seed_steps[0]}'
                    if(folder/'metadata.json').exists():
                        old=read(folder/'metadata.json')
                        if old.get('complete')and sha(folder/'clip.npz')==old['archive_sha256']:groups.append(old);continue
                        raise RuntimeError('Incomplete prediction group exists; refusing repeat of ambiguous calls')
                    folder.mkdir(parents=True,exist_ok=True)
                    if(folder/'started.json').exists():raise RuntimeError('Interrupted prediction group; no blind retry')
                    write(folder/'started.json',dict(clip_id=clip_id,sampler_steps=denoise_steps,seed_steps=seed_steps))
                    status.phase('paired_inference');context=dict(clip_id=clip_id,sampler_steps=denoise_steps)
                    def on_call(event,record):status('sampler_call_'+event,dict(**context,**record))
                    truth8=rgb[anchor-3:anchor+5].astype(np.float32)/np.float32(255)
                    try:
                        raw,records=paired_rollout(predictor.sampler(denoise_steps),truth8,actions[anchor-3:anchor],actions[anchor:anchor+4],seed_steps,on_call=on_call)
                    except RawPredictionFailure as exc:
                        arrays={}
                        for branch,values in exc.partial.items():
                            for number,value in enumerate(values):arrays[f'{branch}_{number+1}']=np.asarray(value)
                        npz(folder/'raw-failure.npz',**arrays)
                        write(folder/'failed-metadata.json',dict(clip_id=clip_id,sampler_steps=denoise_steps,
                            reason=str(exc),records=exc.records,raw_archive=str(folder/'raw-failure.npz'),complete=False))
                        raise
                    npz(folder/'clip.npz',truth_rgb=truth8[3:],initial_history=truth8[:4],actions_absolute=actions[anchor-3:anchor+4],
                        predictions_A=raw['A'],predictions_B=raw['B'],states15=states[anchor:anchor+5],seed_steps=seed_steps)
                    metadata=dict(kind='R3B-playground-paired-frame-clip.v1',complete=True,clip_id=clip_id,case_id=case['case_id'],
                        letter=case['letter'],variant=case['variant'],anchor=anchor,run_id=config['run_id'],
                        source_case=str(output/'cases'/case['case_id']/'case.json'),raw_archive=str(folder/'clip.npz'),archive_sha256=sha(folder/'clip.npz'),
                        sampler=dict(num_steps_denoising=denoise_steps,**SAMPLER),seed_steps=seed_steps,
                        checkpoint_sha256=CHECKPOINT_SHA,checkpoint_training_physics_sha256=TRAINING_PHYSICS_SHA,
                        physics_profile=config['physics_profile'],physics_protocol_sha256=config['physics_protocol_sha256'],cross_physics_preview=True,
                        first_frame_check_mode='independently_checked',first_output_equal=True,step_records_A=records['A'],step_records_B=records['B'],
                        branch_A='Four single-step predictions, each corrected with true recorded history',branch_B='TrueH4: initial true history only, then rawfloat32 predictions',
                        motion_contact=motion_contact(case,anchor,states,case_record['control_telemetry'],actions),
                        screenshot_status='same_failure_class_attempt_not_exact_user_history',optimizer_updates=0)
                    write(folder/'metadata.json',metadata);groups.append(metadata);new_groups.append(metadata);write(output/'group-index.json',groups)
                    storage.check()
                    status('prediction_group_complete',dict(clip_id=clip_id,sampler_steps=denoise_steps,groups_completed=sum(g.get('complete',False)for g in groups)))
        status.phase('immutable_model_verification');proof=predictor.proof();write(output/'model-proof.json',proof)
        if not(proof['checkpoint_unchanged']and proof['model_weights_unchanged']and proof['model_frozen']):raise RuntimeError('Frozen model identity changed')
        proof.update(prior_completed_group_count=len(groups)-len(new_groups),current_process_group_count=len(new_groups),
            total_completed_group_sampler_calls=sum(8 for g in groups if g.get('complete')))
        write(output/'model-proof.json',proof)
        expected_calls=sum(8 for g in new_groups if g.get('complete'));expected_forwards=sum(g['sampler']['num_steps_denoising']*8 for g in new_groups)
        if proof['sampler_calls']!=expected_calls or proof['denoiser_forward_attempts']!=expected_forwards or proof['denoiser_forward_completed']!=expected_forwards:
            raise RuntimeError('Actual native sampler/forward ledger disagrees with completed group calls')
        write(output/'run-index.json',dict(kind='R3B-playground-bounded-diagnostic-run.v1',run_id=config['run_id'],groups=groups,
            cases=[dict(case_id=r['case']['case_id'],path=str(output/'cases'/r['case']['case_id']/'case.json'),completed_controls=r['completed_controls'],failure=r['failure'])for r in truth_records],
            planned_cases=8,planned_main_clips=32,model_proof=str(output/'model-proof.json'),weights_frozen=True,optimizer_updates=0,
            physics_profile=config['physics_profile'],physics_protocol_sha256=config['physics_protocol_sha256'],cross_physics_preview=True))
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),at_utc=utc())
        write(output/'failure.json',failure)
        if predictor is not None:
            try:write(output/'model-proof.json',predictor.proof())
            except BaseException as verifying:write(output/'model-proof-failed.json',dict(reason=repr(verifying)))
    finally:
        if predictor is not None:
            try:predictor.close()
            except BaseException as exc:
                if failure is None:failure=dict(type=type(exc).__name__,reason=str(exc),phase='close_frozen_predictor',traceback=traceback.format_exc())
        if status.thread.is_alive():status.finish(failure is None)
        if storage is not None:
            try:write(output/'storage-final.json',storage.check())
            except BaseException as exc:
                if failure is None:failure=dict(type=type(exc).__name__,reason=str(exc),phase='final_storage_check')
        completion=dict(kind='R3B-playground-diagnostic-inference-completion.v1',run_id=config['run_id'],completed=failure is None,
            inference_complete=failure is None,report_complete=False,closed=True,failure=failure,finished_at_utc=utc(),
            wall_seconds=time.perf_counter()-began,optimizer_updates=0,training_authorized=False,
            checkpoint_sha256_after=sha(config['checkpoint_path']),checkpoint_unchanged=sha(config['checkpoint_path'])==CHECKPOINT_SHA,
            run_index=str(output/'run-index.json'),model_proof=str(output/'model-proof.json'),
            after_inference='Read-only report generation and actual parent callback, then STOP waiting_user_approval; no training.')
        write(output/'completion.json',completion)
        os.close(owner_fd)
        if owner_path.exists()and owner_path.read_text(encoding='ascii')==str(os.getpid()):owner_path.unlink()
    return 0 if failure is None else 2
