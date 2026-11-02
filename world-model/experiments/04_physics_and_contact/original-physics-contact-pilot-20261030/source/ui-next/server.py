"""Local, inference-only R3B playground. No optimizer, dataset write or training.

The fixed checkpoint and exact historical physics are explicit. Ground-truth
images never enter an open-loop forecast after its first conditioning window.
"""
from pathlib import Path
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import argparse,base64,hashlib,json,math,threading,time,uuid,sys,mimetypes
from collections import deque
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
import numpy as np

APP=Path(__file__).resolve().parent
REPO=APP.parents[1]
WORKSPACE=REPO.parent
sys.path.insert(0,str(REPO))
GPC=WORKSPACE/'Tdyssey'/'third_party'/'gpc_code'
PHYSICS=WORKSPACE/'Tdyssey'/'third_party'/'diffusion_policy'
RUN=REPO/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030'
CHECKPOINT=RUN/'run/last.pt'
CHECKPOINT_SHA='23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7'
MODEL_TRAINING_PHYSICS_SHA='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6'
LETTERS=('T','L','O','C','X','B')
SAMPLER=dict(num_steps_denoising=3,sigma_min=.002,sigma_max=5,rho=7,order=1,s_churn=0,s_cond=0)
RUNTIME=REPO/'artifacts/r3b-playground'
DIAGNOSTICS=REPO/'artifacts/r3b-playground-diagnostics-20261030'
from capture import ClipRing,RecordingWriter,physics_snapshot,restore_physics,persist_clip,session_export,unpack,safe,json_bytes,storage_check,MAX_VIDEO_BYTES,digest,utc

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def png(rgb):
    import cv2
    a=np.asarray(rgb)
    if a.shape!=(3,96,96):raise ValueError('Unexpected display image shape')
    if not np.isfinite(a).all():raise ValueError('Nonfinite image cannot be displayed')
    # Display conversion only. Raw float prediction is retained for autoregression.
    a=np.rint(np.clip(a,0,1)*255).astype(np.uint8)
    ok,b=cv2.imencode('.png',np.moveaxis(a,0,-1)[...,::-1]);assert ok
    return 'data:image/png;base64,'+base64.b64encode(b).decode('ascii')

class Predictor:
    def __init__(self):
        import torch
        from tdyssey.gpc_world_model import load_world_model_source,world_model_spec
        from tdyssey.native_rgb_r2c1r1_training import torch_native_sample
        if sha(CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('R3B checkpoint hash differs; refusing another model')
        if not torch.cuda.is_available():raise RuntimeError('The trained model requires the local CUDA GPU')
        torch.set_num_threads(2)
        source=load_world_model_source(GPC);config=dict(world_model_spec()['model'])
        denoiser={k:config.pop(k) for k in ('sigma_data','sigma_offset_noise','noise_previous_obs')}
        self.model=source.Denoiser(source.DenoiserConfig(inner_model=source.InnerModelConfig(**config),**denoiser)).float()
        # This is the user's local checkpoint; verify immutable SHA before pickle loading.
        payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        if payload['progress']['lineage_updates']!=7499 or not payload['progress']['epoch_complete']:
            raise RuntimeError('Expected completed R3B epoch1, lineage 7499')
        self.model.load_state_dict(payload['model'],strict=True);del payload
        self.model.to('cuda').eval().requires_grad_(False)
        self.sample=torch_native_sample(self.model,source,SAMPLER)
        self.calls=0;self.seconds=0.;self.device=torch.cuda.get_device_name(0)
    def __call__(self,history,actions,seed):
        began=time.perf_counter()
        out=self.sample(np.asarray(history,np.float32)[None],np.asarray(actions,np.float32)[None]/511*2-1,seed=seed)[0]
        self.calls+=1;self.seconds+=time.perf_counter()-began
        if out.shape!=(3,96,96) or out.dtype!=np.float32 or not np.isfinite(out).all():
            raise RuntimeError('The model returned invalid raw RGB')
        return out

class Playground:
    def __init__(self,predictor):
        self.predict=predictor;self.env=None;self.revision=0;self.session='';self.total_controls=0
        self.capture=ClipRing();self.events=deque(maxlen=256);self.event_lock=threading.Lock();self.preview_lock=threading.Lock();self.preview=None
        self.recordings={};self.recording_writers={};self.recording_id=None;self.recording_stop_reason=None;self.last_saved_clip=None
        self.reset('O','center')
    def reset(self,letter,scene):
        from tdyssey.native_rgb_contact_safe_physics import make_contact_safe_env,PROTOCOL
        from tdyssey.envs.official import official_shape
        from shapely.geometry import box,LineString,Point
        from shapely.ops import unary_union
        from shapely import affinity
        if letter not in LETTERS:raise ValueError('Only the six trained letters T/L/O/C/X/B are available')
        if scene not in ('center','left','right','rotated'):raise ValueError('Unknown scene')
        center=np.array((90 if scene=='left' else 422 if scene=='right' else 256,256),float)
        angle=math.pi/4 if scene=='rotated' else 0.
        cog=np.array((0.,45.) if letter=='T' else (0.,0.));c,s=math.cos(angle),math.sin(angle)
        origin=center-np.array([c*cog[0]-s*cog[1],s*cog[0]+c*cog[1]])
        shape=official_shape(letter);geom=unary_union([box(r.x0,r.y0,r.x1,r.y1) for r in shape.components])
        geom=affinity.translate(affinity.rotate(geom,angle,origin=(0,0),use_radians=True),*origin)
        yy=(geom.bounds[1]+geom.bounds[3])/2
        cut=geom.intersection(LineString([(0,yy),(512,yy)]))
        x0,y0,x1,y1=cut.bounds;right=scene=='left' or x0-23<24
        px=x1+23 if right else x0-23
        # A scanline can pass through a concavity (notably B). Place the whole
        # circle outside the union, not merely outside one horizontal segment.
        while geom.distance(Point(px,yy))<18:px+=2 if right else -2
        pusher=(px,yy)
        newenv=make_contact_safe_env(PHYSICS,letter)
        im=newenv.reset_scene(center=center,angle=angle,pusher=pusher)
        identity=newenv.initial_identity()
        if identity['physics']['agent']['body_type']!=1 or identity['physics']['space']['damping']!=0 or identity['physics']['space']['iterations']!=PROTOCOL['solver_iterations']:
            raise RuntimeError('Expected versioned contact-safe kinematic pusher/damping/iterations')
        self.physics_sha=identity['protocol_sha256']
        if np.max(newenv.penetrations())>2:raise RuntimeError('Invalid initial scene')
        history=deque([im.astype(np.float32)/255],maxlen=4);past=deque(maxlen=3)
        for _ in range(3):
            im,info=newenv.step(np.asarray(pusher,dtype=float));history.append(im.astype(np.float32)/255);past.append(np.array(pusher,dtype=float))
        if self.env is not None:self.env.close()
        self.env=newenv;self.history=history;self.past=past;self.letter=letter;self.scene=scene
        self.session=uuid.uuid4().hex;self.revision+=1;self.steps=0;self.terminal=False
        self.latest_prediction=None;self.last_frames=[];self.target=list(pusher);self.info=None
        self.metrics=dict(translation_px=0.,rotation_deg=0.,contact_substeps=0,physics_substeps=0,contact_seconds=0.,intervention_ratio=0.,latency_ms=0.,horizon=0,raw_out_of_range=False)
        self.set_preview(self.session,pusher,None);self.event('reset',letter=letter,scene=scene)
        self.message='接触约束修复版已就绪：运动学推杆、阻尼0；当前模型仍是旧物理训练的跨物理预览。'
        return self.snapshot()
    def event(self,kind,**value):
        item=dict(kind=kind,at_utc=utc(),session=self.session,ui_step=getattr(self,'steps',0),recording_id=self.recording_id,**safe(value))
        with self.event_lock:self.events.append(item)
    def event_snapshot(self):
        with self.event_lock:return list(self.events)
    def set_preview(self,session,target,client_time_ms):
        a=np.asarray(target,dtype=float)
        if session!=self.session:raise ValueError('Preview belongs to a previous session')
        if a.shape!=(2,) or not np.isfinite(a).all() or (a<0).any() or (a>511).any():raise ValueError('Invalid preview target')
        with self.preview_lock:self.preview=dict(target=a.tolist(),server_received_at_utc=utc(),client_time_ms=client_time_ms)
    def preview_snapshot(self):
        with self.preview_lock:return None if self.preview is None else dict(self.preview)
    def session_record(self):
        return dict(kind='R3B-playground-public-session.v2',session=self.session,revision=self.revision,letter=self.letter,scene=self.scene,
            ui_steps=self.steps,total_controls=self.total_controls,terminal=self.terminal,target=self.target,metrics=self.metrics,
            message=self.message,checkpoint_sha256=CHECKPOINT_SHA,model_training_physics_sha256=MODEL_TRAINING_PHYSICS_SHA,
            environment_physics_sha256=self.physics_sha,sampler=SAMPLER,physical_state=physics_snapshot(self.env),
            preview_target=self.preview_snapshot(),ring=self.capture.stats(),saved_at_utc=utc())
    def export_session(self):
        arrays=dict(session_history=np.stack(self.history),session_past=np.stack(self.past),
            session_latest_prediction=np.empty((0,),np.float32) if self.latest_prediction is None else self.latest_prediction)
        return session_export(self.session_record(),arrays)
    def restore_session(self,value):
        if value.get('kind')!='R3B-playground-session-export.v1':raise ValueError('Expected complete session export')
        data=base64.b64decode(value['archive_base64'],validate=True)
        if digest(data)!=value['sha256']:raise ValueError('Session archive hash differs')
        metadata,arrays=unpack(data)
        if metadata['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Saved model differs')
        if metadata['environment_physics_sha256']!=self.physics_sha:raise ValueError('Saved environment differs; current session was preserved')
        saved_ticks=metadata.get('physical_state',{}).get('controller',{}).get('integration_microstep_ticks')
        allowed_ticks=self.env.initial_identity()['protocol']['integration_step_tick_choices']
        if type(saved_ticks)is not int or saved_ticks not in allowed_ticks:raise ValueError('Saved integration-controller state is incomplete; current session was preserved')
        self.reset(metadata['letter'],metadata['scene'])
        if metadata['environment_physics_sha256']!=self.physics_sha:raise ValueError('Saved environment differs; do not cross physics on restore')
        restore_physics(self.env,metadata['physical_state'])
        history=arrays['session_history'];past=arrays['session_past'];prediction=arrays['session_latest_prediction']
        if history.shape!=(4,3,96,96) or history.dtype!=np.float32 or past.shape!=(3,2):raise ValueError('Saved model history shape differs')
        self.history=deque((v.copy()for v in history),maxlen=4);self.past=deque((v.copy()for v in past),maxlen=3)
        self.steps=int(metadata['ui_steps']);self.total_controls=int(metadata['total_controls']);self.terminal=bool(metadata['terminal'])
        self.target=metadata['target'];self.metrics=metadata['metrics'];self.latest_prediction=None if not prediction.size else prediction.copy()
        self.last_frames=[];self.message='已恢复公开身体、控制器和模型历史；隐藏位置偏置速度与碰撞缓存未序列化，不承诺后续接触物理精确重放。'
        self.event('session_restored',source_session=metadata['session']);return self.snapshot()
    def save_clip(self,recording_id=None):
        first=None
        if recording_id is not None:
            if recording_id not in self.recordings:raise ValueError('Unknown recording')
            first=self.recordings[recording_id]['first_sequence']
        data,metadata=self.capture.pack(self.session_record(),recording_id,first,self.event_snapshot(),
            dict(history_float32=np.stack(self.history),past_actions_absolute=np.stack(self.past),
                latest_prediction_float32=np.empty((0,),np.float32) if self.latest_prediction is None else self.latest_prediction))
        result=persist_clip(REPO,data,metadata);self.last_saved_clip=result
        if recording_id is not None:self.recordings[recording_id]['clip']=result
        return result
    def start_recording(self):
        if self.recording_id is not None:raise ValueError('A recording is already active')
        identity='recording-'+uuid.uuid4().hex
        writer=RecordingWriter(REPO,identity,self.export_session());self.recording_writers[identity]=writer
        self.recordings[identity]=dict(recording_id=identity,first_sequence=self.capture.sequence+1,started_at_utc=utc(),
            initial_session=self.session,initial_ui_step=self.steps,checkpoint_sha256=CHECKPOINT_SHA,
            model_training_physics_sha256=MODEL_TRAINING_PHYSICS_SHA,environment_physics_sha256=self.physics_sha,
            video_content='Only actual browser-visible paired canvases, labels and target overlays; no desktop/audio capture and no invented H4 intermediate frames.')
        self.recording_id=identity;self.recording_stop_reason=None;self.event('recording_started');return dict(recording_id=identity,session=self.session,maximum_video_bytes=MAX_VIDEO_BYTES,raw_ring=self.capture.stats(),raw_streaming=True)
    def stop_recording(self,identity):
        if identity!=self.recording_id:raise ValueError('No matching active recording')
        self.event('recording_stopped');raw_stream=self.recording_writers[identity].finish();clip=self.save_clip(identity)
        self.recordings[identity].update(stopped_at_utc=utc(),raw_stream=raw_stream);self.recording_id=None
        return dict(recording_id=identity,clip=clip,raw_stream=raw_stream,session=self.session)
    def save_video(self,identity,data,client_metadata):
        if identity not in self.recordings:raise ValueError('Unknown recording ID')
        if not 512<=len(data)<=MAX_VIDEO_BYTES or not data.startswith(b'\x1aE\xdf\xa3'):raise ValueError('Expected nonempty bounded WebM recording')
        resource=storage_check(REPO,len(data)+8192)
        output=DIAGNOSTICS/'recordings';output.mkdir(parents=True,exist_ok=True);path=output/(identity+'.webm')
        if path.exists():raise ValueError('Recording already saved; refusing overwrite')
        temporary=path.with_suffix('.webm.tmp')
        with temporary.open('xb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        metadata=dict(self.recordings[identity]);metadata.update(bytes=len(data),sha256=digest(data),video_path=str(path),
            client=safe(client_metadata),saved_at_utc=utc(),optimizer_updates=0,storage=resource)
        meta=path.with_suffix('.json');meta.write_bytes(json_bytes(metadata))
        return dict(saved=True,recording_id=identity,path=str(path),metadata_path=str(meta),bytes=len(data),sha256=digest(data),
            video_url='/diagnostics-root/recordings/'+path.name,metadata_url='/diagnostics-root/recordings/'+meta.name,
            clip=self.recordings[identity].get('clip'),raw_stream=self.recordings[identity].get('raw_stream'))
    def centroid(self,state):
        from tdyssey.envs.official import official_shape
        x,y=official_shape(self.letter).centroid;c,s=np.cos(state[4]),np.sin(state[4])
        return state[2:4]+[c*x-s*y,s*x+c*y]
    def step(self,session,revision,target,horizon,preview_target=None,client_time_ms=None):
        from tdyssey.native_rgb_r2a_env import R2AStepAborted
        if session!=self.session or revision!=self.revision:raise ValueError('界面状态已更新，请刷新后重试')
        if self.terminal:raise ValueError('本场景已暂停，请重置场景')
        a=np.asarray(target,dtype=float)
        if a.shape!=(2,) or not np.isfinite(a).all() or (a<0).any() or (a>511).any():raise ValueError('目标必须位于画布内')
        if type(horizon) is not int or horizon not in (1,4):raise ValueError('Horizon must be 1 or 4')
        if self.recording_id is not None and not self.recording_writers[self.recording_id].can_accept(horizon):
            self.recording_stop_reason='原始录像记录达到资源额度，已在下一批执行前停止，保存已录内容。';self.message=self.recording_stop_reason
            return self.snapshot()
        self.set_preview(session,a if preview_target is None else preview_target,client_time_ms)
        batch_id=uuid.uuid4().hex;ui_start=self.steps;before_snapshot=physics_snapshot(self.env);preview_at_request=self.preview_snapshot()
        start=time.perf_counter();before=self.env.state().copy();forecast=deque((v.copy() for v in self.history),maxlen=4)
        forecast_actions=deque((v.copy() for v in self.past),maxlen=3)
        predictions=[];input_histories=[];input_actions=[]
        # Predict the entire open-loop horizon BEFORE observing any future truth.
        for j in range(horizon):
            history_input=np.stack(forecast);action_input=np.stack([*forecast_actions,a]);input_histories.append(history_input.copy());input_actions.append(action_input.copy())
            raw=self.predict(history_input,action_input,5478+self.steps+j)
            predictions.append(raw.copy());forecast.append(raw.copy());forecast_actions.append(a.copy())
        inference_ms=(time.perf_counter()-start)*1000;frames=[];completed=0;truth_raw=[];physical_states=[before_snapshot];preview_each=[];infos=[];abort=None
        for j,raw in enumerate(predictions):
            try:im,info=self.env.step(a)
            except R2AStepAborted as exc:
                abort=dict(type=type(exc).__name__,reason=str(exc),info=safe(exc.info),partial_physical_state=physics_snapshot(self.env))
                self.terminal=True;self.message='物理环境触发穿透保护，已停在最后一个完整对比步；请重置场景。'
                break
            truth=im.astype(np.float32)/255;self.history.append(truth);self.past.append(a.copy())
            truth_raw.append(im.copy());physical_states.append(physics_snapshot(self.env));preview_each.append(self.preview_snapshot());infos.append(safe(info))
            self.latest_prediction=raw;self.info=info;self.steps+=1;self.total_controls+=1;completed+=1
            frames.append(dict(truth=png(truth),prediction=png(raw),step=self.steps,offset=j+1,time_seconds=round(self.steps*.1,1),requested_target=a.tolist(),preview_target=self.preview_snapshot(),seed=5478+ui_start+j,batch_id=batch_id,environment_control_step=int(self.env.control_step)))
        if completed:
            # For an aborted control, keep metrics tied to the last completed frame.
            end=self.env.state().copy() if not self.terminal else before
            self.metrics=dict(translation_px=float(np.linalg.norm(self.centroid(end)-self.centroid(before))*96/512) if not self.terminal else None,
                rotation_deg=float(np.degrees(end[4]-before[4])) if not self.terminal else None,
                contact_substeps=int(self.info['agent_object_contact_substeps']),physics_substeps=int(self.info['physics_substeps']),contact_seconds=float(self.info['agent_object_contact_seconds']),intervention_ratio=float(self.info['intervention_time_ratio']),max_penetration=max(self.info['penetration']),latency_ms=inference_ms,
                horizon=horizon,completed=completed,raw_out_of_range=bool(np.any(self.latest_prediction<0) or np.any(self.latest_prediction>1)))
        self.last_frames=frames;self.target=a.tolist();self.revision+=1
        self.capture.append(dict(batch_id=batch_id,session=self.session,recording_id=self.recording_id,letter=self.letter,scene=self.scene,
            requested_at_utc=utc(),ui_step_before=ui_start,environment_control_step_before=before_snapshot['controller']['control_step'],
            requested_horizon=horizon,generated_predictions=len(predictions),completed_controls=completed,seed_rule='5478+UIstep+offset',
            seeds=[5478+ui_start+j for j in range(horizon)],requested_target=a.tolist(),preview_at_request=preview_at_request,preview_at_completed_controls=preview_each,
            checkpoint_sha256=CHECKPOINT_SHA,model_training_physics_sha256=MODEL_TRAINING_PHYSICS_SHA,environment_physics_sha256=self.physics_sha,sampler=SAMPLER,
            action_contract='requested absolute controller target retained; actual endpoints/velocities and unilateral contact interventions in control_infos',
            normalized_actions='float32 absolute/511*2-1',RGB_input='float32 [0,1] real histories; H4 raw float32 predictions fed back unchanged',
            physical_states=physical_states,control_infos=infos,abort=abort),dict(input_histories_float32=np.stack(input_histories).astype(np.float32),
            input_actions_absolute=np.stack(input_actions),input_actions_normalized=np.stack(input_actions).astype(np.float32)/511*2-1,
            raw_predictions_float32=np.stack(predictions),true_frames_uint8=np.stack(truth_raw) if truth_raw else np.empty((0,3,96,96),np.uint8)))
        if self.recording_id is not None:
            try:self.recording_writers[self.recording_id].append(self.capture.records[-1])
            except Exception as exc:
                self.recording_stop_reason='原始记录写入失败，当前视频/内存片段会保留，但完整性不可声明：'+str(exc)
                self.message=self.recording_stop_reason
        if not self.terminal:self.message='下一步使用真实历史帧对比。' if horizon==1 else '这 4 步使用模型自己的预测连续递推，没有喂入未来真实图像。'
        return self.snapshot()
    def snapshot(self):
        return dict(session=self.session,revision=self.revision,letter=self.letter,scene=self.scene,steps=self.steps,time_seconds=round(self.steps*.1,1),terminal=self.terminal,
            truth=png(self.history[-1]),prediction=None if self.latest_prediction is None else png(self.latest_prediction),
            history=[png(v) for v in self.history],frames=self.last_frames,target=self.target,metrics=self.metrics,message=self.message,
            model='R3B · 7,499 步 · 旧物理训练检查点',physics='环境：运动学推杆 / 阻尼0 / 接触约束修复版；跨物理预览',letters=list(LETTERS),checkpoint_sha256=CHECKPOINT_SHA,physics_sha256=self.physics_sha,model_training_physics_sha256=MODEL_TRAINING_PHYSICS_SHA,
            capture=self.capture.stats(),recording_id=self.recording_id,recording_stop_reason=self.recording_stop_reason,preview_target=self.preview_snapshot(),actual_batch_target=self.target,command_target=self.target,action_contract='requested target; actual pusher may be constrained by contact',report_available=(REPO/'artifacts/original-physics-contact-pilot-20261030/index.html').is_file())

def serve(port):
    RUNTIME.mkdir(parents=True,exist_ok=True);started=time.time();predictor=Predictor();app=Playground(predictor);lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,obj,status=200):
            body=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/api/state':
                with lock:result=app.snapshot()
                self.respond(result)
                return
            if path=='/api/health':self.respond(dict(service='tdyssey-r3b-playground-v2',ready=True,pid=os.getpid(),calls=predictor.calls,device=predictor.device,checkpoint_sha256=CHECKPOINT_SHA,physics_sha256=app.physics_sha,model_training_physics_sha256=MODEL_TRAINING_PHYSICS_SHA));return
            if path=='/api/session/export':
                with lock:result=app.export_session()
                self.respond(result)
                return
            if path=='/api/reports':self.respond(dict(available=(REPO/'artifacts/original-physics-contact-pilot-20261030/index.html').is_file(),url='/contact-pilot/index.html'));return
            for prefix,root in (('/contact-pilot/',REPO/'artifacts/original-physics-contact-pilot-20261030'),('/captures/',DIAGNOSTICS/'captures'),('/diagnostics/',DIAGNOSTICS/'report'),('/diagnostics-root/',DIAGNOSTICS)):
                if path.startswith(prefix):
                    file=(root/path[len(prefix):]).resolve()
                    if not file.is_relative_to(root.resolve()) or not file.is_file() or file.suffix.lower()not in('.html','.png','.mp4','.webm','.json','.csv','.npz','.md','.gz','.txt'):
                        self.respond({'error':'Not found'},404);return
                    body=file.read_bytes();mime=mimetypes.guess_type(str(file))[0]or'application/octet-stream'
                    self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            files={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
            if path not in files:self.respond({'error':'Not found'},404);return
            file,mime=files[path];body=(APP/file).read_bytes();self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            expected=f'http://127.0.0.1:{self.server.server_port}'
            if self.headers.get('Origin') not in (None,expected):self.respond({'error':'Origin rejected'},403);return
            path=urlparse(self.path).path;content_type=self.headers.get('Content-Type','').split(';')[0]
            if path=='/api/video/save':
                if content_type!='video/webm':self.respond({'error':'WebM required'},415);return
                if not lock.acquire(blocking=False):self.respond({'error':'正在计算，请稍后保存；本地视频仍保留'},409);return
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 512<=size<=MAX_VIDEO_BYTES:raise ValueError('Video exceeds recording resource limit')
                    identity=self.headers.get('X-Recording-Id','')
                    result=app.save_video(identity,self.rfile.read(size),app.recordings.get(identity,{}).get('client',{}));self.respond(result)
                except (ValueError,KeyError)as exc:self.respond({'error':str(exc)},400)
                except Exception as exc:self.respond({'error':f'{type(exc).__name__}: {exc}'},500)
                finally:lock.release()
                return
            if content_type!='application/json':self.respond({'error':'JSON required'},415);return
            try:
                size=int(self.headers.get('Content-Length','0'));limit=8*1024**2 if path=='/api/session/restore'else 512*1024 if path=='/api/recording/stop'else 65536
                if not 0<size<limit:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(size))
                if path=='/api/preview':
                    app.set_preview(data['session'],data['target'],data.get('client_time_ms'));self.respond({'received':True});return
                if path=='/api/event':
                    app.event(data['kind'],client_time_ms=data.get('client_time_ms'),detail=data.get('detail'));self.respond({'received':True});return
            except (KeyError,TypeError,ValueError)as exc:self.respond({'error':str(exc)},400);return
            if not lock.acquire(blocking=False):self.respond({'error':'正在计算，请稍后再试'},409);return
            try:
                if path=='/api/reset':result=app.reset(data['letter'],data['scene'])
                elif path=='/api/step':result=app.step(data['session'],data['revision'],data['target'],data.get('horizon',1),data.get('preview_target'),data.get('client_time_ms'))
                elif path=='/api/save-clip':result=app.save_clip(data.get('recording_id'))
                elif path=='/api/session/restore':result=app.restore_session(data['export'])
                elif path=='/api/recording/start':result=app.start_recording()
                elif path=='/api/recording/stop':
                    identity=data['recording_id']
                    if identity in app.recordings:app.recordings[identity]['client']=data.get('metadata',{})
                    result=app.stop_recording(identity)
                elif path=='/api/shutdown':
                    self.respond({'closed':True});threading.Thread(target=self.server.shutdown,daemon=True).start();return
                else:self.respond({'error':'Not found'},404);return
                self.respond(result)
            except (KeyError,TypeError,ValueError) as exc:self.respond({'error':str(exc)},400)
            except Exception as exc:
                if path in('/api/step','/api/reset','/api/session/restore'):app.terminal=True
                self.respond({'error':f'{type(exc).__name__}: {exc}'},500)
            finally:lock.release()
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    ready=dict(url=f'http://127.0.0.1:{server.server_port}',pid=os.getpid(),ready=True,checkpoint_sha256=CHECKPOINT_SHA,device=predictor.device,physics_sha256=app.physics_sha,model_training_physics_sha256=MODEL_TRAINING_PHYSICS_SHA)
    (RUNTIME/'server.json').write_text(json.dumps(ready,indent=2),encoding='utf-8');print(json.dumps(ready),flush=True)
    try:server.serve_forever(poll_interval=.2)
    finally:
        server.server_close();app.env.close()
        (RUNTIME/'last-session.json').write_text(json.dumps(dict(inference_calls=predictor.calls,inference_seconds=predictor.seconds,user_controls=app.total_controls,wall_seconds=time.time()-started,optimizer_updates=0,checkpoint_sha256=CHECKPOINT_SHA)),encoding='utf-8')
        (RUNTIME/'server.json').write_text(json.dumps(dict(ready,ready=False)),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8767);args=ap.parse_args();serve(args.port)
