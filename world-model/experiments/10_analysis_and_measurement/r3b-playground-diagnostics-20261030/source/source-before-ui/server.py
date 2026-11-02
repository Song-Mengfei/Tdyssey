"""Local, inference-only R3B playground. No optimizer, dataset write or training.

The fixed checkpoint and exact historical physics are explicit. Ground-truth
images never enter an open-loop forecast after its first conditioning window.
"""
from pathlib import Path
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import argparse,base64,hashlib,json,math,threading,time,uuid,sys
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
PHYSICS_SHA='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6'
LETTERS=('T','L','O','C','X','B')
SAMPLER=dict(num_steps_denoising=3,sigma_min=.002,sigma_max=5,rho=7,order=1,s_churn=0,s_cond=0)
RUNTIME=REPO/'artifacts/r3b-playground'

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
        self.reset('O','center')
    def reset(self,letter,scene):
        from tdyssey.native_rgb_r2a2_env import make_r2a2_env
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
        newenv=make_r2a2_env(PHYSICS,letter,'D-half-life-0.1s')
        im=newenv.reset_scene(center=center,angle=angle,pusher=pusher)
        identity=newenv.initial_identity()
        if identity['protocol_sha256']!=PHYSICS_SHA:raise RuntimeError('Historical R3B physics identity changed')
        if np.max(newenv.penetrations())>2:raise RuntimeError('Invalid initial scene')
        history=deque([im.astype(np.float32)/255],maxlen=4);past=deque(maxlen=3)
        for _ in range(3):
            im,info=newenv.step(np.asarray(pusher,dtype=float));history.append(im.astype(np.float32)/255);past.append(np.array(pusher,dtype=float))
        if self.env is not None:self.env.close()
        self.env=newenv;self.history=history;self.past=past;self.letter=letter;self.scene=scene
        self.session=uuid.uuid4().hex;self.revision+=1;self.steps=0;self.terminal=False
        self.latest_prediction=None;self.last_frames=[];self.target=list(pusher);self.info=None
        self.metrics=dict(translation_px=0.,rotation_deg=0.,contact_substeps=0,latency_ms=0.,horizon=0,raw_out_of_range=False)
        self.message='按住左侧画面并移动鼠标，蓝色推杆会朝红十字移动。'
        return self.snapshot()
    def centroid(self,state):
        from tdyssey.envs.official import official_shape
        x,y=official_shape(self.letter).centroid;c,s=np.cos(state[4]),np.sin(state[4])
        return state[2:4]+[c*x-s*y,s*x+c*y]
    def step(self,session,revision,target,horizon):
        from tdyssey.native_rgb_r2a_env import R2AStepAborted
        if session!=self.session or revision!=self.revision:raise ValueError('界面状态已更新，请刷新后重试')
        if self.terminal:raise ValueError('本场景已暂停，请重置场景')
        a=np.asarray(target,dtype=float)
        if a.shape!=(2,) or not np.isfinite(a).all() or (a<0).any() or (a>511).any():raise ValueError('目标必须位于画布内')
        if type(horizon) is not int or horizon not in (1,4):raise ValueError('Horizon must be 1 or 4')
        start=time.perf_counter();before=self.env.state().copy();forecast=deque((v.copy() for v in self.history),maxlen=4)
        forecast_actions=deque((v.copy() for v in self.past),maxlen=3)
        predictions=[]
        # Predict the entire open-loop horizon BEFORE observing any future truth.
        for j in range(horizon):
            raw=self.predict(np.stack(forecast),np.stack([*forecast_actions,a]),5478+self.steps+j)
            predictions.append(raw.copy());forecast.append(raw.copy());forecast_actions.append(a.copy())
        inference_ms=(time.perf_counter()-start)*1000;frames=[];completed=0
        for j,raw in enumerate(predictions):
            try:im,info=self.env.step(a)
            except R2AStepAborted:
                self.terminal=True;self.message='物理环境触发穿透保护，已停在最后一个完整对比步；请重置场景。'
                break
            truth=im.astype(np.float32)/255;self.history.append(truth);self.past.append(a.copy())
            self.latest_prediction=raw;self.info=info;self.steps+=1;self.total_controls+=1;completed+=1
            frames.append(dict(truth=png(truth),prediction=png(raw),step=self.steps,offset=j+1))
        if completed:
            # For an aborted control, keep metrics tied to the last completed frame.
            end=self.env.state().copy() if not self.terminal else before
            self.metrics=dict(translation_px=float(np.linalg.norm(self.centroid(end)-self.centroid(before))*96/512) if not self.terminal else None,
                rotation_deg=float(np.degrees(end[4]-before[4])) if not self.terminal else None,
                contact_substeps=int(self.info['agent_object_contact_substeps']),latency_ms=inference_ms,
                horizon=horizon,completed=completed,raw_out_of_range=bool(np.any(self.latest_prediction<0) or np.any(self.latest_prediction>1)))
        self.last_frames=frames;self.target=a.tolist();self.revision+=1
        if not self.terminal:self.message='下一步使用真实历史帧对比。' if horizon==1 else '这 4 步使用模型自己的预测连续递推，没有喂入未来真实图像。'
        return self.snapshot()
    def snapshot(self):
        return dict(session=self.session,revision=self.revision,letter=self.letter,scene=self.scene,steps=self.steps,time_seconds=round(self.steps*.1,1),terminal=self.terminal,
            truth=png(self.history[-1]),prediction=None if self.latest_prediction is None else png(self.latest_prediction),
            history=[png(v) for v in self.history],frames=self.last_frames,target=self.target,metrics=self.metrics,message=self.message,
            model='R3B · 7,499 步 · 完整一轮',physics='训练时环境：动态推杆 · 0.1 秒半衰期',letters=list(LETTERS),checkpoint_sha256=CHECKPOINT_SHA,physics_sha256=PHYSICS_SHA)

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
                with lock:self.respond(app.snapshot())
                return
            if path=='/api/health':self.respond(dict(service='tdyssey-r3b-playground-v1',ready=True,pid=os.getpid(),calls=predictor.calls,device=predictor.device,checkpoint_sha256=CHECKPOINT_SHA));return
            files={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
            if path not in files:self.respond({'error':'Not found'},404);return
            file,mime=files[path];body=(APP/file).read_bytes();self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_POST(self):
            expected=f'http://127.0.0.1:{self.server.server_port}'
            if self.headers.get('Origin') not in (None,expected):self.respond({'error':'Origin rejected'},403);return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':self.respond({'error':'JSON required'},415);return
            if not lock.acquire(blocking=False):self.respond({'error':'正在计算，请稍后再试'},409);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<4096:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(size));path=urlparse(self.path).path
                if path=='/api/reset':result=app.reset(data['letter'],data['scene'])
                elif path=='/api/step':result=app.step(data['session'],data['revision'],data['target'],data.get('horizon',1))
                elif path=='/api/shutdown':
                    self.respond({'closed':True});threading.Thread(target=self.server.shutdown,daemon=True).start();return
                else:self.respond({'error':'Not found'},404);return
                self.respond(result)
            except (KeyError,TypeError,ValueError) as e:self.respond({'error':str(e)},400)
            except Exception as e:
                app.terminal=True
                self.respond({'error':f'{type(e).__name__}: {e}'},500)
            finally:lock.release()
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    ready=dict(url=f'http://127.0.0.1:{server.server_port}',pid=os.getpid(),ready=True,checkpoint_sha256=CHECKPOINT_SHA,device=predictor.device)
    (RUNTIME/'server.json').write_text(json.dumps(ready,indent=2),encoding='utf-8');print(json.dumps(ready),flush=True)
    try:server.serve_forever(poll_interval=.2)
    finally:
        server.server_close();app.env.close()
        (RUNTIME/'last-session.json').write_text(json.dumps(dict(inference_calls=predictor.calls,inference_seconds=predictor.seconds,user_controls=app.total_controls,wall_seconds=time.time()-started,optimizer_updates=0,checkpoint_sha256=CHECKPOINT_SHA)),encoding='utf-8')
        (RUNTIME/'server.json').write_text(json.dumps(dict(ready,ready=False)),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8767);args=ap.parse_args();serve(args.port)
