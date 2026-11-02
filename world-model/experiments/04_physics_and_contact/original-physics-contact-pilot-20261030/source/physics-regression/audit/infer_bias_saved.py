"""Infer consumed native position bias from recorded pre/post poses, no physics."""
import json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
OUT=Path(__file__).resolve().parent
REG=OUT.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def center(st,cog):
 c,s=np.cos(st[:,4]),np.sin(st[:,4])
 return st[:,2:4]+np.column_stack((c*cog[0]-s*cog[1],s*cog[0]+c*cog[1]))
results=[]
for version in('v2','v3'):
 summary=json.loads((REG/version/'summary.json').read_text(encoding='utf-8'))
 failures=[r for r in summary['candidates']if not r['geometry_safe']]
 for item in failures:
  path=REG/version/item['file']
  with np.load(path,allow_pickle=False)as data:
   pre=data['substep_pre'].copy();post=data['substep_post'].copy()
   rows=json.loads(str(data['substep_metadata_json']))
   identity=json.loads(str(data['identity_json']))
  cog=np.asarray(identity['physics']['object']['center_of_gravity'])
  dt=np.asarray([r['dt']for r in rows],float)
  consumed_v_bias=(center(post,cog)-center(pre,cog))/dt[:,None]-pre[:,7:9]
  consumed_w_bias=(post[:,4]-pre[:,4])/dt-pre[:,9]
  bias_delta=consumed_v_bias*dt[:,None]
  post_speed=np.linalg.norm(post[:,7:9],axis=1);pre_speed=np.linalg.norm(pre[:,7:9],axis=1)
  dt_ratio=np.concatenate(([1.],dt[1:]/dt[:-1]))
  peak=int(np.argmax(post_speed));bias_peak=int(np.argmax(np.linalg.norm(bias_delta,axis=1)))
  angular_peak=int(np.argmax(np.abs(consumed_w_bias*dt)))
  indices=sorted(set(range(max(0,len(rows)-9),len(rows)))|{peak,bias_peak,angular_peak})
  evidence=[]
  for i in indices:
   row=rows[i];events=row.get('contacts',[])
   evidence.append(dict(index=i,control=row.get('control'),controller_index=row.get('controller_index'),dt=float(dt[i]),dt_ratio=float(dt_ratio[i]),
    pre_object_velocity=pre[i,7:9].tolist(),post_object_velocity=post[i,7:9].tolist(),pre_omega=float(pre[i,9]),post_omega=float(post[i,9]),
    consumed_v_bias=consumed_v_bias[i].tolist(),consumed_w_bias=float(consumed_w_bias[i]),
    bias_translation_delta_world=bias_delta[i].tolist(),bias_angle_delta_deg=float(np.degrees(consumed_w_bias[i]*dt[i])),
    actual_CoG_translation_world=(center(post[i:i+1],cog)-center(pre[i:i+1],cog))[0].tolist(),
    executed_pusher_velocity=row.get('executed_velocity'),unconstrained_pusher_velocity=row.get('unconstrained_velocity'),
    penetration=row.get('penetrations'),pusher_region=row.get('pusher_region'),
    events_by_phase_kind=dict(Counter(e.get('phase','')+':'+e.get('kind','')for e in events)),
    post_solve_impulse_norms=[float(np.linalg.norm(e.get('total_impulse',[0,0])))for e in events if e.get('phase')=='post_solve'],
    intervention=row.get('intervention')))
  no_events_jump=[i for i,row in enumerate(rows)if not any(e.get('phase')=='post_solve'for e in row.get('contacts',[]))and post_speed[i]>pre_speed[i]+100]
  results.append(dict(version=version,file=item['file'],sha256=sha(path),microsteps=len(rows),
   protocol_sha256=identity['protocol_sha256'],max_penetration=item['max_penetrations_wall_agentobject_agentwall'],
   peak_object_speed=float(post_speed.max()),peak_abs_omega=float(np.abs(post[:,9]).max()),
   peak_consumed_bias_translation_delta_world=float(np.linalg.norm(bias_delta,axis=1).max()),
   peak_consumed_bias_angle_delta_deg=float(np.degrees(np.abs(consumed_w_bias*dt)).max()),
   max_dt_ratio=float(dt_ratio.max()),min_dt_ratio=float(dt_ratio.min()),
   no_post_event_velocity_jump_indices=no_events_jump,
   no_post_event_velocity_jump_at_dt_growth=sum(dt_ratio[i]>1 for i in no_events_jump),
   evidence=evidence))
result=dict(kind='saved-only-native-position-bias-inference.v1',new_physics_controls=0,new_model_calls=0,GPU_calls=0,optimizer_updates=0,
 method='Native position integration source uses CoG delta=(pre_v+v_bias)*dt and angle delta=(pre_omega+w_bias)*dt. Infer the bias consumed in this step from actual saved poses. This is not a direct read of next pending bias. T nonzero CoG is transformed explicitly.',
 cases=results,
 sources=[
  dict(local='../Tdyssey/.venv/Lib/site-packages/pymunk/_version.py',version='6.9.0',chipmunk_revision='dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970'),
  dict(url='https://raw.githubusercontent.com/viblo/Chipmunk2D/dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970/src/cpBody.c',positions_lines='500-510',velocity_at_world_point_lines='559-564'),
  dict(url='https://raw.githubusercontent.com/viblo/Chipmunk2D/dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970/src/cpSpaceStep.c',integration_order_lines='338-399'),
  dict(url='https://raw.githubusercontent.com/viblo/Chipmunk2D/dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970/src/cpArbiter.c',bias_and_cache_lines='394-469')])
(OUT/'BIAS-v4-v5.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
compact=[{k:v for k,v in row.items()if k not in('evidence','protocol_sha256','sha256')}for row in results]
print(json.dumps(compact,indent=2))
