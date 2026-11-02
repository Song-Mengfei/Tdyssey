"""Final frozen regression saved-array audit; no physics/model imports."""
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

OUT=Path(__file__).resolve().parent;BASE=OUT.parent/'v4';PILOT=OUT.parent.parent;ROOT=PILOT.parent.parent
began=time.perf_counter()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
summary=read(BASE/'summary.json');started=read(BASE/'started.json');plan=read(BASE/'frozen-plan.json')
def shape(identity):
 return unary_union([Polygon(s['geometry']['vertices'])for s in identity['physics']['shapes']if s['role']=='object'])
def measure(geom,st):
 st=np.asarray(st,float);c,s=np.cos(st[:,4]),np.sin(st[:,4]);d=st[:,:2]-st[:,2:4]
 p=shapely.points(np.column_stack((c*d[:,0]+s*d[:,1],-s*d[:,0]+c*d[:,1])))
 solid=shapely.contains(geom,p);bd=shapely.distance(geom.boundary,p)
 signed=np.where(solid,-bd,bd);parts=list(geom.geoms)if hasattr(geom,'geoms')else[geom]
 hole=np.zeros(len(st),bool)
 for part in parts:
  for ring in part.interiors:hole|=shapely.contains(Polygon(ring),p)
 vertices=np.concatenate([np.asarray(part.exterior.coords)for part in parts])
 xx=st[:,2,None]+c[:,None]*vertices[None,:,0]-s[:,None]*vertices[None,:,1]
 yy=st[:,3,None]+s[:,None]*vertices[None,:,0]+c[:,None]*vertices[None,:,1]
 wall=np.min(np.column_stack((xx.min(1)-7,504-xx.max(1),yy.min(1)-7,504-yy.max(1))),1)
 pw=np.min(np.column_stack((st[:,0]-22,489-st[:,0],st[:,1]-22,489-st[:,1])),1)
 gaps=np.column_stack((wall,signed-15,pw));local=(geom.centroid.x,geom.centroid.y)
 center=st[:,2:4]+np.column_stack((c*local[0]-s*local[1],s*local[0]+c*local[1]))
 return dict(gaps=gaps,overlap=np.maximum(0,-gaps),solid=solid,hole=hole,center=center)
def quant(values):
 values=np.asarray(values,float)
 return dict(N=len(values),**{k:float(v)for k,v in zip(('P10','P50','P90','P95','max'),np.quantile(values,[.1,.5,.9,.95,1]))})if len(values)else dict(N=0)

rows=[];errors=[];totals=Counter();pair_cache={};geometry_by_letter={};all_checks=[]
for number,entry in enumerate([*summary['candidates'],*summary['original']]):
 path=BASE/entry['file'];digest=sha(path)
 if digest!=entry['sha256']:errors.append(dict(file=path.name,error='SHA mismatch'))
 with np.load(path,allow_pickle=False)as z:
  st=z['states'].copy();pre=z['substep_pre'].copy();post=z['substep_post'].copy()
  micro=json.loads(str(z['substep_metadata_json']));controls=json.loads(str(z['control_metadata_json']))
  ident=json.loads(str(z['identity_json']));spec=json.loads(str(z['spec_json']));actions=z['actions'].copy();rgb=z['rgb'].copy()
 geom=shape(ident);a=measure(geom,pre);b=measure(geom,post);boundary=measure(geom,st)
 dt=np.asarray([r['dt']for r in micro]);control_numbers=np.asarray([r['control']for r in micro])
 done=np.asarray([bool(r.get('executed'))for r in micro]);constrained=np.asarray([isinstance(r.get('intervention'),dict)and r['intervention'].get('delta_norm',0)>1e-8 for r in micro])
 raw_pen=np.asarray([r.get('penetrations',[0,0,0])for r in micro]);max_geom_error=float(np.abs(raw_pen-b['overlap']).max())
 maxpen=b['overlap'].max(0);physical=float(dt[done].sum())
 finite=bool(np.isfinite(st).all()and np.isfinite(pre).all()and np.isfinite(post).all())
 path_by_micro=np.linalg.norm(b['center']-a['center'],axis=1);angle_by_micro=np.abs(np.degrees(post[:,4]-pre[:,4]))
 net=np.linalg.norm(np.diff(boundary['center'],axis=0),axis=1);turn=np.abs(np.degrees(np.diff(st[:,4])))
 micro_contact={kind:np.asarray([any(e.get('phase')=='post_solve'and e.get('kind')==kind for e in r.get('contacts',[]))for r in micro])for kind in('agent_object','object_wall','agent_wall')}
 ctrl=[];contact_error=0.;intervention_error=0.;velocity_excess=0.;snapshot_bind=True
 for index,row in enumerate(controls,1):
  select=(control_numbers==index)&done;seconds=float(dt[select].sum());ao=float(dt[select&micro_contact['agent_object']].sum());ow=float(dt[select&micro_contact['object_wall']].sum())
  inter=float(dt[select&constrained].sum());intervention_error=max(intervention_error,abs(inter-float(row.get('intervention_seconds',0))))
  if 'agent_object_contact_seconds'in row:contact_error=max(contact_error,abs(ao-row['agent_object_contact_seconds']),abs(ow-row['object_wall_contact_seconds']))
  ctrl.append(dict(control=index,phase=row['phase'],valid=bool(row.get('valid')),seconds=seconds,AO_seconds=ao,OW_seconds=ow,
   intervention_seconds=inter,path_world=float(path_by_micro[select].sum()),path_image=float(path_by_micro[select].sum()*96/512),
   angular_path_deg=float(angle_by_micro[select].sum()),net_translation_world=float(net[index-1]),rotation_deg=float(turn[index-1]),
   final_speed=float(np.linalg.norm(st[index,7:9])),final_abs_omega=float(abs(st[index,9]))))
 if entry['kind']=='candidate':
  for row in micro:
   velocity_excess=max(velocity_excess,float(np.linalg.norm(row['executed_velocity'])-np.linalg.norm(row['unconstrained_velocity'])))
  p=ident['protocol'];p_sha=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
  snapshot_bind=p_sha==started['candidate_protocol_sha256']==ident['protocol_sha256']
  exact_dt=bool((dt==.0005).all())
 else:exact_dt=bool((dt==.01).all());snapshot_bind=ident['protocol_sha256']==started['original_protocol_sha256']
 ordinary_fields={k:ident['physics'][k]for k in('agent','object','shapes')}
 geometry_by_letter.setdefault(entry['letter'],[]).append(ordinary_fields)
 release=[r for r in ctrl if r['phase']=='post_withdraw_hold'];released=bool(len(release)==6 and all(r['AO_seconds']==0 for r in release))
 release_details=dict(last6_hold_controls=len(release),all6_no_AO=released,AO_seconds=sum(r['AO_seconds']for r in release),OW_seconds=sum(r['OW_seconds']for r in release),
  actual_path_world=sum(r['path_world']for r in release),control_chord_path_world=sum(r['net_translation_world']for r in release),
  max_control_speed=max((r['final_speed']for r in release),default=0),max_control_abs_omega=max((r['final_abs_omega']for r in release),default=0))
 checks=dict(raw_hash=digest==entry['sha256'],timeline_shapes=len(st)==25==len(rgb)and len(controls)==24,
  all_controls_valid=all(r['valid']for r in ctrl),every_control_seconds_01=all(abs(r['seconds']-.1)<1e-9 for r in ctrl),
  all_executed=bool(done.all()),microstate_finite=finite,protocol_binding=snapshot_bind,correct_dt=exact_dt,
  true_geometry_matches_saved=max_geom_error<1e-7,no_illegal_center=not bool(b['solid'].any()or b['hole'].any()),
  command_plan=spec==next(p for p in plan if p['letter']==entry['letter']and p['name']==entry['scenario'])and np.array_equal(actions,np.asarray(spec['actions'])),
  contact_seconds_match=contact_error<1e-9,intervention_seconds_match=intervention_error<1e-9,
  pusher_speed_not_amplified=velocity_excess<1e-6)
 if entry['kind']=='candidate':checks['overlap_under_1world']=max(maxpen)<=1+1e-8
 for key,value in checks.items():
  if not value:errors.append(dict(file=path.name,error=key))
 record=dict(file=path.name,letter=entry['letter'],scenario=entry['scenario'],kind=entry['kind'],sha256=digest,checks=checks,
  attempted_controls=24,completed_controls=sum(r['valid']for r in ctrl),microsteps=len(micro),physical_seconds=physical,
  minimum_signed_gaps_OW_AO_AW_world_px=b['gaps'].min(0).tolist(),max_penetrations_OW_AO_AW_world_px=maxpen.tolist(),
  geometry_recompute_max_abs_error=max_geom_error,intervention_seconds=float(dt[constrained&done].sum()),
  intervention_time_ratio=float(dt[constrained&done].sum()/physical),
  infeasible_constraint_microsteps=sum(isinstance(r.get('intervention'),dict)and not r['intervention'].get('feasible',True)for r in micro),
  contact_seconds={kind:float(dt[done&mask].sum())for kind,mask in micro_contact.items()},
  cumulative_actual_translation_world_px=float(path_by_micro[done].sum()),cumulative_actual_translation_image_px=float(path_by_micro[done].sum()*96/512),
  cumulative_actual_rotation_deg=float(angle_by_micro[done].sum()),max_H1_translation_image_px=float(net.max()*96/512),max_H1_rotation_deg=float(turn.max()),
  final_object_speed=float(np.linalg.norm(st[-1,7:9])),final_abs_omega=float(abs(st[-1,9])),
  post_withdraw=release_details,controls=ctrl)
 rows.append(record);pair_cache[(entry['letter'],entry['scenario'],entry['kind'])]=(st,boundary['center'],actions,rgb[0])
 totals.update(attempted_controls=24,completed_controls=sum(r['valid']for r in ctrl),microsteps=len(micro))
 if(number+1)%24==0:print(f'checked {number+1}/120',flush=True)

candidate=[r for r in rows if r['kind']=='candidate'];baseline=[r for r in rows if r['kind']=='original']
paired=[]
for entry in baseline:
 key=(entry['letter'],entry['scenario']);new,ncenter,na,nrgb=pair_cache[(*key,'candidate')];old,ocenter,oa,orgb=pair_cache[(*key,'original')]
 displacement=np.linalg.norm(ncenter-ocenter,axis=1);angle=np.abs(np.degrees(new[:,4]-old[:,4]))
 paired.append(dict(letter=key[0],scenario=key[1],same_full_initial_state=bool(np.array_equal(new[0],old[0])),same_initial_RGB=bool(np.array_equal(nrgb,orgb)),
  same_requested_actions=bool(np.array_equal(na,oa)),centroid_distance_world_px=displacement.tolist(),centroid_distance_image_px=(displacement*96/512).tolist(),
  absolute_angle_difference_deg=angle.tolist(),final_centroid_distance_image_px=float(displacement[-1]*96/512),max_centroid_distance_image_px=float(displacement.max()*96/512),
  final_angle_difference_deg=float(angle[-1]),max_angle_difference_deg=float(angle.max())))
source_checks={
 'frozen_plan_sha256':sha(BASE/'frozen-plan.json')==started['plan_sha256'],
 'runner_snapshot_sha256':sha(BASE/'runner-source.py.txt')==started['source_sha256'],
 'candidate_snapshot_sha256':sha(BASE/'candidate-source.py.txt')==started['candidate_source_sha256'],
 'original_snapshot_sha256':sha(BASE/'original-source.py.txt')==started['original_source_sha256'],
 'current_candidate_source_sha256':sha(ROOT/'tdyssey/native_rgb_contact_safe_physics.py')==started['candidate_source_sha256'],
 'current_original_source_sha256':sha(ROOT/'tdyssey/native_rgb_original_physics.py')==started['original_source_sha256'],
 'geometry_material_body_properties_match_all_cases_and_original':all(all(v==entries[0]for v in entries)for entries in geometry_by_letter.values()),
 'all_ordinary_initial_states_RGB_requested_actions_equal':all(r['same_full_initial_state']and r['same_initial_RGB']and r['same_requested_actions']for r in paired),
 'cost_ledger_match':all(totals[k]==summary['cost'][k]for k in('attempted_controls','completed_controls','microsteps')),
 'no_model_optimizer_work':summary['cost']['model_calls']==0 and summary['cost']['optimizer_updates']==0 and started['training_updates']==0,
}
failed=[]
for version in('v1','v2','v3'):
 old=read(OUT.parent/version/'summary.json');files=[*old['candidates'],*old['original']]
 bad=[r['file']for r in files if not(OUT.parent/version/r['file']).exists()or sha(OUT.parent/version/r['file'])!=r['sha256']]
 failed.append(dict(version=version,recorded_gate_passed=old['physics_gate_passed'],candidate_failures=sum(not r['geometry_safe']for r in old['candidates']),
  raw_files=len(files),all_raw_files_and_hashes_retained=not bad,mismatches=bad,summary_sha256=sha(OUT.parent/version/'summary.json')))
source_checks['three_failed_versions_retained']=all(not r['recorded_gate_passed']and r['all_raw_files_and_hashes_retained']for r in failed)
perletter=[]
for letter in'TLOCXB':
 part=[r for r in candidate if r['letter']==letter];sec=sum(r['physical_seconds']for r in part)
 perletter.append(dict(letter=letter,cases=len(part),max_penetrations_OW_AO_AW_world_px=np.max([r['max_penetrations_OW_AO_AW_world_px']for r in part],axis=0).tolist(),
  intervention_time_ratio=sum(r['intervention_seconds']for r in part)/sec,
  max_H1_translation_image_px=max(r['max_H1_translation_image_px']for r in part),max_H1_rotation_deg=max(r['max_H1_rotation_deg']for r in part),
  actual_released_last6_cases=sum(r['post_withdraw']['all6_no_AO']for r in part),
  max_final_speed=max(r['final_object_speed']for r in part),max_final_abs_omega=max(r['final_abs_omega']for r in part)))
released=[r for r in candidate if r['post_withdraw']['all6_no_AO']];still_contact=[dict(letter=r['letter'],scenario=r['scenario'],**r['post_withdraw'])for r in candidate if not r['post_withdraw']['all6_no_AO']]
passed=not errors and all(source_checks.values())and len(candidate)==96 and len(baseline)==24 and all(r['max_H1_translation_image_px']>=2 and r['max_H1_rotation_deg']>=5 for r in perletter)
result=dict(kind='independent-final-v6-physical-regression-audit.v1',checked_at_utc=datetime.now(timezone.utc).isoformat(),passed=passed,
 physics_gate_passed=passed,training_authorized=False,new_physics_controls=0,new_model_calls=0,GPU_calls=0,optimizer_updates=0,
 actual_protocol_name='contact-constrained-original-kinematic-rgb.v6',protocol_sha256=started['candidate_protocol_sha256'],physics_source_sha256=started['candidate_source_sha256'],
 source_checks=source_checks,raw_files=len(rows),candidate_cases=96,ordinary_original_cases=24,geometry_or_record_errors=errors,
 cost_reconstructed=dict(totals),actual_physical_seconds=sum(r['physical_seconds']for r in rows),
 max_candidate_penetrations_OW_AO_AW_world_px=np.max([r['max_penetrations_OW_AO_AW_world_px']for r in candidate],axis=0).tolist(),
 candidate_intervention_time_ratio=sum(r['intervention_seconds']for r in candidate)/sum(r['physical_seconds']for r in candidate),
 per_letter=perletter,
 post_withdraw_summary=dict(last6_verified_no_AO_cases=len(released),cases_with_AO_during_last6=still_contact,
  max_actual_path_world_px_among_verified_released=max((r['post_withdraw']['actual_path_world']for r in released),default=0),
  max_end_speed_among_verified_released=max((r['post_withdraw']['max_control_speed']for r in released),default=0),
  max_end_abs_omega_among_verified_released=max((r['post_withdraw']['max_control_abs_omega']for r in released),default=0),
  statement='No-AO is verified from actual post-solve contacts, not the planned withdraw label. OW contact correction can leave tiny positional residuals despite zero physical end velocity.'),
 ordinary_pair_summary=dict(pairs=len(paired),max_over_all_control_centroid_distance_image_px=max(r['max_centroid_distance_image_px']for r in paired),
  final_centroid_distance_image_px=quant([r['final_centroid_distance_image_px']for r in paired]),
  final_angle_difference_deg=quant([r['final_angle_difference_deg']for r in paired]),
  statement='Identical initial RGB/full public state and requested actions, but new contact/integration solver differs. Numerical equality to original is not claimed.'),
 ordinary_pairs=paired,previous_failed_versions=failed,cases=rows,
 limitations=['This is a bounded regression of six seen letters, not a global guarantee for every arbitrary kinematic command.',
  'Pinned upstream is unchanged; fixed2000Hz/solver50/contact compression.05 and command constraints intentionally differ from original100Hz/solver10 execution.',
  'Snapshot excludes native pending position bias/contact caches; midcontact replay is not established.',
  'Runtime and checks here concern physics only. Actual playground integration gate remains the owner responsibility before collection.'],
 audit_script_sha256=sha(__file__),audit_CPU_seconds=time.perf_counter()-began)
def numpy_json(value):
 if isinstance(value,np.generic):return value.item()
 raise TypeError(type(value).__name__)
(OUT/'AUDIT-v4.json').write_text(json.dumps(result,indent=2,default=numpy_json)+'\n',encoding='utf-8')
md=['# Independent final v6 physical regression audit','',f'Passed: **{passed}**. Read-only saved-array audit; new model/GPU/physics/optimizer work0.',
 f'120 raw files:96 candidate and24 original ordinary controls. Cost:{dict(totals)}. Every control is actual.1s, candidate microstep.0005s.',
 '',f'Maximum candidate OW/AO/AW overlap worldpx:{result["max_candidate_penetrations_OW_AO_AW_world_px"]}. Intervention time:{result["candidate_intervention_time_ratio"]:.3%}.',
 '', '| Letter | Cases | Max OW/AO/AW worldpx | Intervention time | Max H1 imagepx / degrees | last6 controls all no AO |',
 '| --- | --- | --- | --- | --- | --- |']
for r in perletter:md.append('|'+r['letter']+'|16|'+'/'.join(f'{v:.6f}'for v in r['max_penetrations_OW_AO_AW_world_px'])+f'|{r["intervention_time_ratio"]:.2%}|{r["max_H1_translation_image_px"]:.3f}/{r["max_H1_rotation_deg"]:.3f}|{r["actual_released_last6_cases"]}/16|')
md+=['','## Actual separation and original comparison','',json.dumps(result['post_withdraw_summary'],ensure_ascii=False),
 '',json.dumps(result['ordinary_pair_summary'],ensure_ascii=False),
 '', 'Full source bindings, true-geometry signed gaps, contact seconds, all exceptions and three preserved failed versions are in AUDIT-v4.json.',
 '', '## Scope limits','']+['- '+text for text in result['limitations']]
(OUT/'AUDIT-v4.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps({k:result[k]for k in('passed','source_checks','geometry_or_record_errors','cost_reconstructed','max_candidate_penetrations_OW_AO_AW_world_px','candidate_intervention_time_ratio','post_withdraw_summary','ordinary_pair_summary','audit_CPU_seconds')},indent=2))
assert passed
