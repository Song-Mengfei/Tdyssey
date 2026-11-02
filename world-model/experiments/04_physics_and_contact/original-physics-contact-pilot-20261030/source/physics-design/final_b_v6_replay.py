"""Final candidate5 replay; reuses saved exact original baseline, no old physics rerun."""
from pathlib import Path
import gzip,hashlib,json,sys,time,subprocess
import numpy as np
import cv2
repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_contact_safe_physics import make_contact_safe_env,PROTOCOL_SHA256
d=Path(__file__).resolve().parent;name='B-v6-replay-final'
with np.load(d/'B-original-replay-final.npz')as z:
 actions=z['actions'].copy();old_rgb=z['rgb'].copy();old_states=z['states'].copy()
old_records=json.load(gzip.open(d/'B-original-replay-final-substeps.json.gz','rt'))['records']
start=time.perf_counter();env=make_contact_safe_env(repo.parent/'Tdyssey/third_party/diffusion_policy','B')
env.reset_scene(center=(256,256),angle=0,pusher=(184.66666666666666,256))
warmup=[]
for _ in range(3):_,info=env.step((184.66666666666666,256));warmup.append(info)
rgb=[env.render_clean()];states=[env.state().copy()];records=[];failure=None
for i,action in enumerate(actions):
 try:frame,info=env.step(action);rgb.append(frame);states.append(env.state().copy())
 except Exception as exc:failure={'control':i+1,'error':repr(exc)};info=getattr(exc,'info',{});records.append(info);break
 records.append(info)
rows=[dict(control=i+1,**row)for i,c in enumerate(records)for row in c['substeps']]
summary=dict(name=name,attempted=len(records),completed=sum(c['valid']for c in records),warmup_controls=3,warmup_microsteps=sum(len(c['substeps'])for c in warmup),known_microsteps=len(rows),failure=failure,wall_seconds=time.perf_counter()-start,
 max_penetration=np.max([r['penetrations']for r in rows],axis=0).tolist(),protocol_sha256=PROTOCOL_SHA256,source_sha256=hashlib.sha256((repo/'tdyssey/native_rgb_contact_safe_physics.py').read_bytes()).hexdigest(),model_calls=0,optimizer_updates=0,
 intervention_seconds=sum(c['intervention_seconds']for c in records),physics_seconds=sum(r['dt']for r in rows),illegal_center_microsteps=sum(r['pusher_region']!='outside'for r in rows),
 old_baseline_reused='B-original-replay-final.npz/.json/-substeps.json.gz; exact recorded prefix proof already saved; no repeated original physics')
np.savez_compressed(d/f'{name}.npz',rgb=np.stack(rgb),states=np.stack(states),actions=actions,substep_pre=np.asarray([r['pre_state']for r in rows]),substep_post=np.asarray([r['post_state']for r in rows]),substep_dt=np.asarray([r['dt']for r in rows]),summary_json=json.dumps(summary))
with gzip.open(d/f'{name}-substeps.json.gz','wt',encoding='utf-8')as f:json.dump(dict(warmup=warmup,records=records),f)
(d/f'{name}.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if failure:raise RuntimeError('v6 final B replay failed; evidence preserved')

import imageio_ffmpeg
video=d/'B-v6-before-after-10Hz.mp4'
cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','800x508','-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
for index in range(len(rgb)):
 canvas=np.full((508,800,3),248,np.uint8)
 for branch,(frames,info_rows) in enumerate([(old_rgb,old_records),(rgb,records)]):
  x=8+branch*400;canvas[80:464,x:x+384]=cv2.resize(np.moveaxis(frames[index],0,-1),(384,384),interpolation=cv2.INTER_NEAREST)
  title='BEFORE original kinematic'if branch==0 else'AFTER contact-safe v6'
  cv2.putText(canvas,title,(x,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(25,25,25),1,cv2.LINE_AA)
  cv2.putText(canvas,f'B | control{index} | t={index*.1:.1f}s',(x,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(25,25,25),1,cv2.LINE_AA)
  if index:
   pen=info_rows[index-1]['penetration'];cv2.putText(canvas,f'OW {pen[0]:.3f}  AO {pen[1]:.3f} world px',(x,485),cv2.FONT_HERSHEY_SIMPLEX,.48,(25,25,25),1,cv2.LINE_AA)
 proc.stdin.write(canvas.tobytes())
proc.stdin.close();stderr=proc.stderr.read();code=proc.wait();assert code==0,stderr[-2000:]
cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
canvas=np.full((648,800,3),248,np.uint8)
for branch,frames in enumerate([old_rgb,rgb]):
 for j,index in enumerate([31,32]):
  x=20+branch*400;y=40+j*300;canvas[y:y+288,x:x+288]=cv2.resize(np.moveaxis(frames[index],0,-1),(288,288),interpolation=cv2.INTER_NEAREST)
  cv2.putText(canvas,f'{"BEFORE"if branch==0 else"AFTER v6"} | control{index} t={index/10:.1f}s',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,.5,(25,25,25),1,cv2.LINE_AA)
cv2.imwrite(str(d/'B-v6-before-after-control32.png'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
proof=dict(path=str(video),sha256=hashlib.sha256(video.read_bytes()).hexdigest(),frames=count,fps=fps,physical_interval_seconds=.1,interpolated=False,old_baseline_reused=True,model_calls=0,optimizer_updates=0)
(d/'B-v6-video-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
