from pathlib import Path
import gzip,json,sys,time,hashlib
import numpy as np
repo=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(repo))
from tdyssey.native_rgb_contact_safe_physics import make_contact_safe_env,PROTOCOL,PROTOCOL_SHA256
out=Path(__file__).resolve().parent/'v4-fourcase'
out.mkdir(exist_ok=True)
results=[]
for name in ['T-wall_top','T-corner_tl','O-wall_left','C-corner_br']:
 with np.load(repo/'artifacts/original-physics-contact-pilot-20261030/physics-regression/v1'/f'{name}-candidate.npz') as z:spec=json.loads(z['spec_json'].item())
 start=time.perf_counter();env=make_contact_safe_env(repo.parent/'Tdyssey/third_party/diffusion_policy',spec['letter'])
 rgb=[env.reset_scene(center=spec['center'],angle=spec['angle'],pusher=spec['pusher'])];states=[env.state().copy()];records=[];failure=None
 for i,action in enumerate(spec['actions']):
  try:frame,info=env.step(action);rgb.append(frame);states.append(env.state().copy())
  except Exception as exc:failure={'control':i+1,'error':repr(exc)};info=getattr(exc,'info',{});records.append(info);break
  records.append(info)
 rows=[r for c in records for r in c['substeps']]
 summary=dict(name=name,completed=sum(c['valid']for c in records),attempted=len(records),failure=failure,known_microsteps=len(rows),wall_seconds=time.perf_counter()-start,
  max_penetration=np.max([r['penetrations']for r in rows],axis=0).tolist(),intervention_seconds=sum(c['intervention_seconds']for c in records),physics_seconds=sum(r['dt']for r in rows),
  final_state15=env.state().tolist(),protocol_sha256=PROTOCOL_SHA256,source_sha256=hashlib.sha256((repo/'tdyssey/native_rgb_contact_safe_physics.py').read_bytes()).hexdigest(),model_calls=0,optimizer_updates=0)
 np.savez_compressed(out/f'{name}.npz',rgb=np.stack(rgb),states=np.stack(states),spec_json=json.dumps(spec),summary_json=json.dumps(summary))
 with gzip.open(out/f'{name}-substeps.json.gz','wt',encoding='utf-8') as f:json.dump(records,f)
 results.append(summary);print(json.dumps(summary),flush=True)
(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
