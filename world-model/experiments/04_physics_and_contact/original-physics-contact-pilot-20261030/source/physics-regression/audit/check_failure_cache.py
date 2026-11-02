"""Compare saved queries/cache impulses to saved geometry, no physics."""
import json,math,hashlib
from pathlib import Path
import numpy as np
from shapely.geometry import Point,Polygon
from shapely.ops import unary_union
OUT=Path(__file__).resolve().parent
REG=OUT.parent
result=[]
for version,name in [('v2','C-corner_tr'),('v3','O-corner_tl')]:
 path=REG/version/(name+'-candidate.npz')
 with np.load(path,allow_pickle=False)as z:
  pre=z['substep_pre'];post=z['substep_post'];meta=json.loads(str(z['substep_metadata_json']));ident=json.loads(str(z['identity_json']))
 parts=[Polygon(s['geometry']['vertices'])for s in ident['physics']['shapes']if s['role']=='object']
 union=unary_union(parts);query_errors=[];point_errors=[];internal=[]
 for i,row in enumerate(meta):
  st=pre[i];c,s=np.cos(st[4]),np.sin(st[4]);R=np.asarray([[c,-s],[s,c]])
  local=(st[:2]-st[2:4])@R
  q=Point(*local)
  distances=np.asarray([-part.boundary.distance(q)if part.contains(q)else part.distance(q)for part in parts])
  details=row.get('intervention',{}).get('object',[])
  if isinstance(details,dict):details=[details]
  for one in details:
   if 'shape_index'not in one:continue
   gap=one['signed_gap'];err=float(np.min(np.abs(distances-(gap+15))))
   query_errors.append(err)
   point=(np.asarray(one['point_world'])-st[2:4])@R
   p=Point(*point);perr=min(part.boundary.distance(p)for part in parts);point_errors.append(perr)
   if union.boundary.distance(p)>1e-6 and union.contains(p):
    internal.append(dict(index=i,query_gap=gap,union_boundary_distance=float(union.boundary.distance(p))))
 ratios=[1.]+[meta[i]['dt']/meta[i-1]['dt']for i in range(1,len(meta))]
 biggest=int(np.argmax(np.linalg.norm(post[:,7:9],axis=1)))
 prior=meta[biggest-1]if biggest else None;current=meta[biggest]
 wall_impulse=np.zeros(2)
 prior_event_rows=[]
 for e in prior.get('contacts',[])if prior else[]:
  if e.get('phase')!='post_solve'or e.get('kind')!='object_wall':continue
  impulse=np.asarray(e['total_impulse']);object_impulse=impulse if e['shapes_order'][0]=='object'else-impulse
  wall_impulse+=object_impulse
  prior_event_rows.append(dict(roles=e['shapes_order'],reported_impulse=e['total_impulse'],object_impulse=object_impulse.tolist()))
 expected=(ratios[biggest]-1)*wall_impulse/ident['physics']['object']['mass']
 observed=post[biggest,7:9]
 row=dict(version=version,file=path.name,query_records=len(query_errors),max_saved_component_query_distance_error_world=max(query_errors,default=0),
  max_saved_component_surface_point_error_world=max(point_errors,default=0),
  hidden_union_seam_query_count=len(internal),hidden_union_seam_examples=internal[:5],
  peak_velocity_step=biggest,dt_ratio=ratios[biggest],prior_wall_impulses=prior_event_rows,
  prior_wall_impulse_sum=wall_impulse.tolist(),residual_velocity_from_scaled_warmstart_if_current_accumulators_zero=expected.tolist(),
  actual_post_velocity=observed.tolist(),residual_formula_max_abs_error=float(np.max(np.abs(expected-observed))),
  current_posts=[dict(kind=e['kind'],impulse=e['total_impulse'])for e in current.get('contacts',[])if e.get('phase')=='post_solve'])
 result.append(row)
(OUT/'CACHE-v4-v5.json').write_text(json.dumps(dict(new_physics_controls=0,new_model_calls=0,findings=result),indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:v for k,v in row.items()if k not in('prior_wall_impulses','current_posts')}for row in result],indent=2))
