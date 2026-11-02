"""Fixed CPU physics regression. Never imports/calls a model or optimizer."""
from __future__ import annotations
import argparse,datetime,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from shapely import affinity
from shapely.geometry import Point,Polygon,LineString
from tdyssey.native_rgb_env import shape_geometry

LETTERS=tuple('TLOCXB')
SCENARIOS=('free_left','free_right','free_top','free_bottom','eccentric_left','eccentric_top','concave_approach',
 'wall_left','wall_right','wall_top','wall_bottom','corner_tl','corner_tr','corner_bl','corner_br','long_fast')
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scene(letter,name):
 angle=.31 if name.startswith('eccentric')else .17 if name.startswith('corner')else .0
 cog=np.array((0.,45.)if letter=='T'else(0.,0.))
 geom=affinity.rotate(affinity.translate(shape_geometry(letter),*-cog),angle,origin=(0,0),use_radians=True)
 x0,y0,x1,y1=geom.bounds;center=np.array([256.,256.])
 for side in ('left','right','top','bottom'):
  if name=='wall_'+side:
   if side=='left':center[0]=7-x0+3
   if side=='right':center[0]=504-x1-3
   if side=='top':center[1]=7-y0+3
   if side=='bottom':center[1]=504-y1-3
 if name.startswith('corner'):
  code=name[-2:];center[0]=7-x0+3 if code[1]=='l'else 504-x1-3
  center[1]=7-y0+3 if code[0]=='t'else 504-y1-3
 geom=affinity.translate(geom,*center);x0,y0,x1,y1=geom.bounds
 side=name.split('_')[-1]if not name.startswith('wall')else {'left':'right','right':'left','top':'bottom','bottom':'top'}[name.split('_')[-1]]
 if name=='concave_approach':side='right'
 if name=='long_fast':side='left'
 off=.24 if name.startswith('eccentric')else 0
 if name.startswith('corner'):
  code=name[-2:];p=np.array([x1+23 if code[1]=='l'else x0-23,y1+23 if code[0]=='t'else y0-23])
  target=np.array([0 if code[1]=='l'else 511,0 if code[0]=='t'else 511],float)
 else:
  horizontal=side in('left','right')
  at=(y0+y1)/2+off*(y1-y0)if horizontal else(x0+x1)/2+off*(x1-x0)
  line=LineString([(-1000,at),(1500,at)])if horizontal else LineString([(at,-1000),(at,1500)])
  cut=geom.intersection(line)
  if cut.is_empty:raise RuntimeError('Frozen scanline misses shape')
  a,b,c,d=cut.bounds
  p=np.array([a-17 if side=='left'else c+17,at])if horizontal else np.array([at,b-17 if side=='top'else d+17])
  outward=np.array({'left':(-1,0),'right':(1,0),'top':(0,-1),'bottom':(0,1)}[side],float)
  for _ in range(120):
   if geom.distance(Point(*p))>=17-1e-9:break
   p+=outward
  target=p-outward*75
  if name.startswith('eccentric'):target+=np.array([0,15]if horizontal else[15,0])
  if name.startswith('wall'):target=p-outward*130
  if name=='concave_approach':target=np.array([x0-20,(y0+y1)/2])
  if name=='long_fast':target=np.array([511.,0.])
 if not (p>=22).all()or not(p<=489).all()or geom.distance(Point(*p))<15:raise RuntimeError((letter,name,'invalid initial pusher',p.tolist()))
 actions=np.asarray([p]*2+[np.clip(target,0,511)]*10+[p]*12,np.float64)
 return dict(letter=letter,name=name,center=center.tolist(),angle=angle,pusher=p.tolist(),actions=actions.tolist(),
  phases=['hold']*2+['push']*10+['withdraw']*6+['post_withdraw_hold']*6,initial_outside_sealed_holes=True)
def illegal(env):
 geom=env.world_geometry();p=Point(*env.agent.position)
 parts=[geom]if geom.geom_type=='Polygon'else list(geom.geoms)
 return dict(center_in_solid=bool(geom.contains(p)),center_in_hole=any(Polygon(ring).contains(p)for part in parts for ring in part.interiors))
def run_case(spec,kind,source,out,cost):
 if kind=='candidate':
  from tdyssey.native_rgb_contact_safe_physics import make_contact_safe_env
  env=make_contact_safe_env(source,spec['letter'])
 else:
  from tdyssey.native_rgb_original_physics import make_original_env
  env=make_original_env(source,spec['letter'])
 begin=time.perf_counter();frames=[];states=[];infos=[];substeps=[];error=None
 try:
  frames.append(env.reset_scene(center=spec['center'],angle=spec['angle'],pusher=spec['pusher']))
  identity=env.initial_identity();states.append(env.state().copy());assert not any(illegal(env).values())
  for control,action in enumerate(spec['actions']):
   cost['attempted_controls']+=1
   try:
    im,info=env.step(action);cost['completed_controls']+=1
    if not info.get('valid',False):raise RuntimeError('Control returned invalid')
   except BaseException as exc:
    error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),control=control+1)
    info=getattr(exc,'info',{'valid':False,'substeps':env.last_substeps});im=env.render_clean()
   rows=info.get('substeps',env.last_substeps);cost['microsteps']+=len(rows)
   substeps.extend([dict(control=control+1,**row)for row in rows])
   infos.append(dict(control=control+1,phase=spec['phases'][control],illegal=illegal(env),**{k:v for k,v in info.items()if k!='substeps'}))
   frames.append(im);states.append(env.state().copy())
   if error:break
  arr=np.asarray(states);allpen=np.asarray([r.get('penetrations',[0,0,0])for r in substeps],float).reshape(-1,3)
  maxpen=allpen.max(axis=0).tolist()if len(allpen)else[0,0,0]
  centers=[]
  from tdyssey.envs.official import official_shape
  local=np.asarray(official_shape(spec['letter']).centroid)
  for st in arr:
   c,s=np.cos(st[4]),np.sin(st[4]);centers.append(st[2:4]+[c*local[0]-s*local[1],s*local[0]+c*local[1]])
  centers=np.asarray(centers);distance=np.linalg.norm(np.diff(centers,axis=0),axis=1)
  angular=np.abs(np.degrees(np.diff(arr[:,4])))
  intervention=sum(float(r.get('intervention',{}).get('delta_norm',0))>1e-8 for r in substeps)
  microbad=sum(any(row.get('illegal',{}).values())or row.get('center_in_solid',False)or row.get('center_in_hole',False)or row.get('pusher_region','')=='solid'or row.get('pusher_region','').startswith('hole-')for row in substeps)
  controlbad=sum(any(row['illegal'].values())for row in infos)
  safe=error is None and np.isfinite(arr).all()and max(maxpen)<=1.0+1e-9 and not(controlbad or microbad)
  dest=out/f'{spec["letter"]}-{spec["name"]}-{kind}.npz'
  submeta=[{k:v for k,v in r.items()if k not in('pre_state','post_state','integrator_pre_state')}for r in substeps]
  np.savez_compressed(dest,rgb=np.asarray(frames,np.uint8),states=arr,actions=np.asarray(spec['actions']),
   substep_pre=np.asarray([r['pre_state']for r in substeps]),substep_post=np.asarray([r['post_state']for r in substeps]),
   substep_metadata_json=np.asarray(json.dumps(submeta,separators=(',',':'))),
   control_metadata_json=np.asarray(json.dumps(infos,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else float(v),separators=(',',':'))),
   spec_json=np.asarray(json.dumps(spec)),identity_json=np.asarray(json.dumps(identity)))
  summary=dict(letter=spec['letter'],scenario=spec['name'],kind=kind,attempted_controls=len(infos),completed_controls=sum(bool(row.get('valid',False))for row in infos),microsteps=len(substeps),max_penetrations_wall_agentobject_agentwall=maxpen,
   geometry_safe=bool(safe),illegal_control_boundaries=controlbad,illegal_microsteps=microbad,intervened_microsteps=intervention,
   intervention_seconds=sum(r['dt']for r in substeps if float(r.get('intervention',{}).get('delta_norm',0))>1e-8),
   observed_physics_seconds=sum(r['dt']for r in substeps if r.get('executed',False)),
   cumulative_translation_world=float(distance.sum()),cumulative_translation_image=float(distance.sum()*96/512),
   cumulative_rotation_deg=float(angular.sum()),max_control_translation_image=float(distance.max()*96/512)if len(distance)else 0,
   max_control_rotation_deg=float(angular.max())if len(angular)else 0,
   final_object_speed=float(np.linalg.norm(arr[-1,7:9])),final_omega=float(arr[-1,9]),
   after_withdraw_last6_translation_world=float(distance[-6:].sum())if len(distance)>=24 else None,
   file=dest.name,sha256=sha(dest),error=error,wall_seconds=time.perf_counter()-begin)
  write(dest.with_suffix('.json'),summary);return summary,arr
 finally:env.close()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 plan=[scene(letter,name)for letter in LETTERS for name in SCENARIOS]
 planpath=out/'frozen-plan.json'
 if planpath.exists():assert json.loads(planpath.read_text(encoding='utf-8'))==plan
 else:write(planpath,plan)
 if args.freeze_only:print(json.dumps(dict(cases=len(plan),controls_per_case=24,plan_sha256=sha(planpath))));return
 if(out/'started.json').exists():raise RuntimeError('Existing attempt cannot be overwritten/repeated')
 from tdyssey import native_rgb_contact_safe_physics as candidate_module
 from tdyssey import native_rgb_original_physics as original_module
 began=time.perf_counter()
 for name,path in [('runner-source.py.txt',__file__),('candidate-source.py.txt',candidate_module.__file__),('original-source.py.txt',original_module.__file__)]:
  (out/name).write_bytes(Path(path).read_bytes())
 write(out/'started.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(planpath),source_sha256=sha(__file__),
  candidate_source_sha256=sha(candidate_module.__file__),candidate_protocol_sha256=candidate_module.PROTOCOL_SHA256,
  original_source_sha256=sha(original_module.__file__),original_protocol_sha256=original_module.PROTOCOL_SHA256,training_updates=0))
 rows=[];paired=[];cost=dict(attempted_controls=0,completed_controls=0,microsteps=0,model_calls=0,optimizer_updates=0)
 try:
  for number,spec in enumerate(plan):
   actual,states=run_case(spec,'candidate',args.source,out,cost);rows.append(actual)
   if spec['name'].startswith('free'):
    old,previous=run_case(spec,'original',args.source,out,cost);rows.append(old)
    count=min(len(states),len(previous));diff=states[:count]-previous[:count]
    paired.append(dict(letter=spec['letter'],scenario=spec['name'],common_states=count,max_abs_state_difference=float(np.abs(diff).max()),
     final_body_origin_distance_world=float(np.linalg.norm(diff[-1,2:4])),final_angle_difference_deg=float(np.degrees(diff[-1,4]))))
   write(out/'progress.json',dict(completed_cases=number+1,total_cases=96,cost=cost,last=actual))
   print(f'{number+1}/96 {spec["letter"]} {spec["name"]} safe={actual["geometry_safe"]} pen={max(actual["max_penetrations_wall_agentobject_agentwall"]):.4f}',flush=True)
  candidate=[r for r in rows if r['kind']=='candidate']
  result=dict(completed=True,physics_gate_passed=all(r['geometry_safe']for r in candidate),cases=len(candidate),
   moving_letters=sorted({r['letter']for r in candidate if r['cumulative_translation_image']>=2}),
   rotating_letters=sorted({r['letter']for r in candidate if r['cumulative_rotation_deg']>=5}),
   candidates=candidate,original=[r for r in rows if r['kind']=='original'],paired_ordinary=paired,cost=cost,wall_seconds=time.perf_counter()-began)
  result['physics_gate_passed']=result['physics_gate_passed']and len(result['moving_letters'])==6 and len(result['rotating_letters'])==6
  write(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items()if k not in('candidates','original','paired_ordinary')}))
 except BaseException as exc:write(out/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),cost=cost,completed=rows));raise
if __name__=='__main__':main()
