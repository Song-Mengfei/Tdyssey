from pathlib import Path
import gzip,json,sys,time,hashlib
import numpy as np
repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_contact_safe_physics import make_contact_safe_env,PROTOCOL_SHA256
d=Path(__file__).resolve().parent/'v6-sevencase';d.mkdir(exist_ok=True)
results=[]
for name in ['C-corner_tr','T-corner_tl','L-corner_br','O-corner_tl','O-corner_bl','C-free_left','C-eccentric_left']:
 with np.load(repo/'artifacts/original-physics-contact-pilot-20261030/physics-regression/v1'/f'{name}-candidate.npz')as z:spec=json.loads(z['spec_json'].item())
 start=time.perf_counter();env=make_contact_safe_env(repo.parent/'Tdyssey/third_party/diffusion_policy',spec['letter'])
 rgb=[env.reset_scene(center=spec['center'],angle=spec['angle'],pusher=spec['pusher'])];states=[env.state().copy()];records=[];failure=None
 for i,action in enumerate(spec['actions']):
  try:frame,info=env.step(action);rgb.append(frame);states.append(env.state().copy())
  except Exception as exc:failure={'control':i+1,'error':repr(exc)};info=getattr(exc,'info',{});records.append(info);break
  records.append(info)
 rows=[r for c in records for r in c['substeps']];states=np.stack(states)
 summary=dict(name=name,attempted=len(records),completed=sum(c['valid']for c in records),failure=failure,known_substeps=len(rows),wall_seconds=time.perf_counter()-start,
  max_penetration=np.max([r['penetrations']for r in rows],axis=0).tolist(),physics_seconds=sum(r['dt']for r in rows),intervention_seconds=sum(c['intervention_seconds']for c in records),
  cumulative_origin_translation_world=float(np.linalg.norm(np.diff(states[:,2:4],axis=0),axis=1).sum()),cumulative_rotation_deg=float(np.abs(np.diff(states[:,4])).sum()*180/np.pi),
  final_state15=env.state().tolist(),protocol_sha256=PROTOCOL_SHA256,source_sha256=hashlib.sha256((repo/'tdyssey/native_rgb_contact_safe_physics.py').read_bytes()).hexdigest(),model_calls=0,optimizer_updates=0)
 np.savez_compressed(d/f'{name}.npz',rgb=np.stack(rgb),states=states,spec_json=json.dumps(spec),summary_json=json.dumps(summary))
 with gzip.open(d/f'{name}-substeps.json.gz','wt',encoding='utf-8')as f:json.dump(records,f)
 (d/f'{name}.json').write_text(json.dumps(summary,indent=2)+'\n');results.append(summary);print(json.dumps(summary),flush=True)
(d/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
