from pathlib import Path
import gzip,hashlib,json,sys,time,subprocess
import numpy as np
import cv2
repo=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(repo))
from tdyssey.native_rgb_original_physics import make_original_env
from tdyssey.native_rgb_contact_safe_physics import make_contact_safe_env,PROTOCOL_SHA256
out=Path(__file__).resolve().parent
raw=repo/'artifacts/r3b-playground-diagnostics-20261030/recordings/recording-7917cd3c57594ca1b4c64401251b5e97/raw'
manifest=json.loads((raw/'manifest.json').read_text())
entries=[]
for row in manifest['batches']:
 with np.load(raw/row['file']) as z:meta=json.loads(z['metadata_json'].item())
 entries.append(meta)
results=[];data=[]
for name,factory in [('B-original-replay-final',make_original_env),('B-v4-replay-final',make_contact_safe_env)]:
 start=time.perf_counter();env=factory(repo.parent/'Tdyssey/third_party/diffusion_policy','B')
 env.reset_scene(center=(256,256),angle=0,pusher=(184.66666666666666,256))
 warmup=[]
 for _ in range(3):_,info=env.step((184.66666666666666,256));warmup.append(info)
 rgb=[env.render_clean()];states=[env.state().copy()];records=[];failure=None;control_errors=[];substep_errors=[]
 for i,meta in enumerate(entries):
  try:frame,info=env.step(meta['executed_target']);rgb.append(frame);states.append(env.state().copy())
  except Exception as exc:failure={'control':i+1,'error':repr(exc)};info=getattr(exc,'info',{});records.append(info);break
  records.append(info)
  control_errors.append(float(np.max(np.abs(env.state()-np.asarray(meta['physical_states'][-1]['state15'])))))
  if name.startswith('B-original'):
   substep_errors.extend(float(np.max(np.abs(np.asarray(a['post_state'])-np.asarray(b['post_state']))))for a,b in zip(info['substeps'],meta['control_infos'][0]['substeps']))
 rows=[dict(control=i+1,**row)for i,c in enumerate(records)for row in c['substeps']]
 summary=dict(name=name,attempted=len(records),completed=sum(r['valid']for r in records),warmup_controls=3,warmup_microsteps=sum(len(r['substeps'])for r in warmup),known_microsteps=len(rows),failure=failure,wall_seconds=time.perf_counter()-start,
  max_penetration=np.max([r['penetrations']for r in rows],axis=0).tolist(),control_state15_max_diff_from_recorded=max(control_errors,default=None),
  original_substep_state15_max_diff_from_recorded=max(substep_errors,default=None),first_control_diff_gt_1e_8=next((i+1 for i,e in enumerate(control_errors)if e>1e-8),None),
  protocol_sha256=info['protocol_sha256'],model_calls=0,optimizer_updates=0,
  intervention_seconds=sum(r.get('intervention_seconds',0)for r in records),physics_seconds=sum(r['dt']for r in rows),
  interpretation='Complete initial construction plus three actual hold controls then37recorded actions; no midcontact restore; solver cache not serialized.')
 np.savez_compressed(out/f'{name}.npz',rgb=np.stack(rgb),states=np.stack(states),actions=np.asarray([m['executed_target']for m in entries]),
  substep_pre=np.asarray([r['pre_state']for r in rows]),substep_post=np.asarray([r['post_state']for r in rows]),substep_dt=np.asarray([r['dt']for r in rows]),summary_json=json.dumps(summary))
 with gzip.open(out/f'{name}-substeps.json.gz','wt',encoding='utf-8') as f:json.dump(dict(warmup=warmup,records=records),f,default=lambda x:x.tolist()if hasattr(x,'tolist')else str(x))
 (out/f'{name}.json').write_text(json.dumps(summary,indent=2)+'\n')
 results.append(summary);data.append((rgb,states,records));print(json.dumps(summary),flush=True)
(out/'final-B-paired-summary.json').write_text(json.dumps(results,indent=2)+'\n')

# Real10Hz paired simulator movie, only actual saved control-boundary frames.
import imageio_ffmpeg
video=out/'B-before-after-10Hz.mp4'
cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','800x508','-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
for index in range(min(len(d[0])for d in data)):
 canvas=np.full((508,800,3),248,np.uint8)
 for branch,(frames,states,records) in enumerate(data):
  x=8+branch*400;img=cv2.resize(np.moveaxis(frames[index],0,-1),(384,384),interpolation=cv2.INTER_NEAREST)
  canvas[80:464,x:x+384]=img
  title='BEFORE: original100Hz'if branch==0 else'AFTER: contact-safe v4'
  cv2.putText(canvas,title,(x,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(25,25,25),1,cv2.LINE_AA)
  label=f'B | control{index} | t={index*.1:.1f}s'
  cv2.putText(canvas,label,(x,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(25,25,25),1,cv2.LINE_AA)
  if index:
   pen=records[index-1]['penetration'];label=f'OW {pen[0]:.3f}  AO {pen[1]:.3f} world px'
   cv2.putText(canvas,label,(x,485),cv2.FONT_HERSHEY_SIMPLEX,.48,(25,25,25),1,cv2.LINE_AA)
 proc.stdin.write(canvas.tobytes())
proc.stdin.close();stderr=proc.stderr.read();code=proc.wait()
assert code==0,stderr[-2000:]
cap=cv2.VideoCapture(str(video));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
proof=dict(path=str(video),sha256=hashlib.sha256(video.read_bytes()).hexdigest(),frames=frames,fps=fps,physical_interval_seconds=.1,interpolated=False,trained=False)
(out/'B-video-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
