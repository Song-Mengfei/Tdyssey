from pathlib import Path
import cv2,gzip,hashlib,json,subprocess,sys,time
import numpy as np
import imageio_ffmpeg
repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_original_physics import make_original_env
d=Path(__file__).resolve().parent/'B-corner-paired-final';d.mkdir(exist_ok=True)
results=[]
for name in ['B-corner_tl','B-corner_tr']:
 counterpart=repo/'artifacts/original-physics-contact-pilot-20261030/physics-regression/v4'/f'{name}-candidate.npz'
 with np.load(counterpart)as z:
  spec=json.loads(z['spec_json'].item());new_rgb=z['rgb'].copy();new_states=z['states'].copy();new_rows=json.loads(z['substep_metadata_json'].item());new_controls=json.loads(z['control_metadata_json'].item())
 start=time.perf_counter();env=make_original_env(repo.parent/'Tdyssey/third_party/diffusion_policy','B')
 rgb=[env.reset_scene(center=spec['center'],angle=spec['angle'],pusher=spec['pusher'])];states=[env.state().copy()];records=[];failure=None
 for i,action in enumerate(spec['actions']):
  try:frame,info=env.step(action);rgb.append(frame);states.append(env.state().copy())
  except Exception as exc:failure={'control':i+1,'error':repr(exc)};info=getattr(exc,'info',{});records.append(info);break
  records.append(info)
 rows=[dict(control=i+1,**r)for i,c in enumerate(records)for r in c['substeps']]
 summary=dict(name=name,original_attempted=len(records),original_completed=sum(c['valid']for c in records),original_warmup_controls=0,original_native_substeps=len(rows),wall_seconds=time.perf_counter()-start,original_failure=failure,
  original_max_penetration=np.max([r['penetrations']for r in rows],axis=0).tolist(),v6_max_penetration=np.max([r['penetrations']for r in new_rows],axis=0).tolist(),
  v6_raw_reused=str(counterpart),v6_raw_sha256=hashlib.sha256(counterpart.read_bytes()).hexdigest(),new_v6_physics_controls=0,model_calls=0,optimizer_updates=0,
  initial_state15_match=bool(np.array_equal(states[0],new_states[0])),requested_actions_match=True,interpretation='Same initial state/requested actions, different actual trajectories; no identical midcontact-state claim.')
 np.savez_compressed(d/f'{name}-original.npz',rgb=np.stack(rgb),states=np.stack(states),actions=np.asarray(spec['actions']),substep_pre=np.asarray([r['pre_state']for r in rows]),substep_post=np.asarray([r['post_state']for r in rows]),spec_json=json.dumps(spec),summary_json=json.dumps(summary))
 with gzip.open(d/f'{name}-original-substeps.json.gz','wt',encoding='utf-8')as f:json.dump(records,f,default=lambda x:x.tolist()if hasattr(x,'tolist')else str(x))
 (d/f'{name}.json').write_text(json.dumps(summary,indent=2)+'\n')
 states=np.asarray(states)
 def new_pen(index):
  if index==0:return[0,0,0]
  return np.max([r['penetrations']for r in new_rows if r['control']==index],axis=0)
 # Actual RGB only; no physics or model call for counterpart/video creation.
 video=d/f'{name}-before-after-10Hz.mp4'
 cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','800x560','-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 for index in range(min(len(rgb),len(new_rgb))):
  canvas=np.full((560,800,3),248,np.uint8)
  for branch,(frames,ss)in enumerate([(rgb,states),(new_rgb,new_states)]):
   x=8+branch*400;canvas[90:474,x:x+384]=cv2.resize(np.moveaxis(frames[index],0,-1),(384,384),interpolation=cv2.INTER_NEAREST)
   title='BEFORE original kinematic'if branch==0 else'AFTER contact-safe v6'
   cv2.putText(canvas,title,(x,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(25,25,25),1,cv2.LINE_AA)
   cv2.putText(canvas,f'{name} c{index} t={index/10:.1f}s',(x,50),cv2.FONT_HERSHEY_SIMPLEX,.52,(25,25,25),1,cv2.LINE_AA)
   if index:
    pen=records[index-1]['penetration']if branch==0 else new_pen(index)
    cv2.putText(canvas,f'OW {pen[0]:.3f} AO {pen[1]:.3f} worldpx',(x,75),cv2.FONT_HERSHEY_SIMPLEX,.44,(25,25,25),1,cv2.LINE_AA)
    for h,y in[(1,505),(4,535)]:
     if index>=h:
      move=float(np.linalg.norm(ss[index,2:4]-ss[index-h,2:4])*96/512);rot=float(abs(ss[index,4]-ss[index-h,4])*180/np.pi)
      cv2.putText(canvas,f'H{h} {h/10:.1f}s: {move:.2f}px / {rot:.1f}deg',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(25,25,25),1,cv2.LINE_AA)
  proc.stdin.write(canvas.tobytes())
 proc.stdin.close();error=proc.stderr.read();assert proc.wait()==0,error[-1000:]
 cap=cv2.VideoCapture(str(video));summary['video_frames']=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));summary['video_fps']=cap.get(cv2.CAP_PROP_FPS);cap.release()
 summary['video_sha256']=hashlib.sha256(video.read_bytes()).hexdigest();summary['video_path']=str(video)
 peak=1+int(np.argmax([c['penetration'][1]for c in records]));indices=sorted(set([2,peak,min(24,len(rgb)-1)]))
 canvas=np.full((len(indices)*325+10,800,3),248,np.uint8)
 for j,index in enumerate(indices):
  for branch,frames in enumerate([rgb,new_rgb]):
   x=20+branch*400;y=35+j*325;canvas[y:y+288,x:x+288]=cv2.resize(np.moveaxis(frames[index],0,-1),(288,288),interpolation=cv2.INTER_NEAREST)
   cv2.putText(canvas,f'{"BEFORE"if branch==0 else"V6"} c{index} t={index/10:.1f}s',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,.52,(25,25,25),1,cv2.LINE_AA)
 cv2.imwrite(str(d/f'{name}-comparison.png'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
 (d/f'{name}.json').write_text(json.dumps(summary,indent=2)+'\n');results.append(summary);print(json.dumps(summary),flush=True)
(d/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
