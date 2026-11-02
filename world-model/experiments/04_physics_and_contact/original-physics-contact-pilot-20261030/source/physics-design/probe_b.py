from pathlib import Path
import gzip
import json
import time
import sys
import numpy as np

root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
from tdyssey.native_rgb_original_physics import make_original_env
from tdyssey.native_rgb_contact_safe_physics import make_contact_safe_env,PROTOCOL,PROTOCOL_SHA256
source=root.parent/'Tdyssey/third_party/diffusion_policy'
out=Path(__file__).resolve().parent
raw=root/'artifacts/r3b-playground-diagnostics-20261030/recordings/recording-7917cd3c57594ca1b4c64401251b5e97/raw'
manifest=json.loads((raw/'manifest.json').read_text())
with np.load(raw/'initial-session.npz') as z:initial=json.loads(z['metadata_json'].item())
entries=[]
for row in manifest['batches']:
 with np.load(raw/row['file']) as z:
  metadata=json.loads(z['metadata_json'].item())
 entries.append((metadata['executed_target'], metadata['physical_states'][-1]['state15']))
results=[]
for name,factory in [('candidate',make_contact_safe_env)]:
 started=time.perf_counter();env=factory(source,'B')
 env.reset_scene(center=(256,256),angle=0,pusher=(184.66666666666666,256))
 records=[];attempted=0;warmup_micro=0
 for _ in range(3):
  _,info=env.step((184.66666666666666,256));warmup_micro+=len(info['substeps'])
 maxerr=0;failure=None
 for index,(action,reference) in enumerate(entries):
  attempted+=1
  try:_,info=env.step(action)
  except Exception as exc:
   info=getattr(exc,'info',{});failure=dict(control=index+1,error=repr(exc));records.append(info);break
  records.append(info)
  maxerr=max(maxerr,float(np.max(np.abs(env.state()-np.asarray(reference)))))
 data=dict(name=name,completed=sum(x['valid']for x in records),attempted=attempted,warmup_controls=3,warmup_microsteps=warmup_micro,records=records,failure=failure,
  max_state15_diff_from_recorded=maxerr,final_state15=env.state().tolist(),wall_seconds=time.perf_counter()-started,
  known_microsteps=sum(len(x['substeps'])for x in records),max_penetration=np.max([row['penetrations']for info in records for row in info['substeps']],axis=0).tolist(),
  intervention_time_ratio=sum(x.get('intervention_seconds',0)for x in records)/sum(sum(r['dt']for r in x['substeps'])for x in records),
  protocol=PROTOCOL if name=='candidate'else None,protocol_sha256=PROTOCOL_SHA256 if name=='candidate'else None,
  no_model_calls=True,optimizer_updates=0)
 with gzip.open(out/f'B-{name}-probe.json.gz','wt',encoding='utf-8') as f:json.dump(data,f,default=lambda x:x.tolist() if hasattr(x,'tolist')else str(x))
 small={k:v for k,v in data.items()if k not in('records','protocol')};results.append(small);print(json.dumps(small),flush=True)
(out/'B-probe-summary.json').write_text(json.dumps(results,indent=2)+'\n')
