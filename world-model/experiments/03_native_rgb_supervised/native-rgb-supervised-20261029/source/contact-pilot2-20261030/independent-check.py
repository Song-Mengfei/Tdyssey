"""Independent CP2 saved-array arithmetic; no collector/report metric imports."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

STAGE=Path(__file__).resolve().parent
ROOT=STAGE.parents[2]
sys.path.insert(0,str(ROOT))
from tdyssey.envs.official import official_shape

CLASSES=tuple(f'{c}_{m}'for c in('no_contact','contact')for m in('nearstatic','small','obvious'))
NUMERIC=('charged_controls','completed_controls','saved_attempted_controls','known_substeps','unknown_advance_substeps',
 'partial_substeps','reserved_not_started_controls','unclassified_charged_controls','any_contact_controls',
 'terminal_contact_controls','contact_substeps','object_wall_substeps','agent_wall_substeps','unknown_impulse_contact_substeps',
 'positive_impulse_substeps','contact_obvious_controls','contact_obvious_impulse_controls','contact_obvious_atleast5_controls',
 'contact_obvious_all10_controls','contact_obvious_impulse_atleast5_controls','nearstatic_controls','moving_pusher_noAO_controls',
 'moving_pusher_noAO_objectnearstatic_controls','route_recovery_noAO_controls','planned_push_controls','planned_push_noAO_controls',
 'wallstall_controls','simultaneous_AO_OW_nearstatic_substeps','late_completed_controls','late_charged_controls',
 'late_contact_obvious_controls','early_completed_controls','early_contact_obvious_controls','withdraw_completed_controls',
 'withdraw_noAO_controls','hold_complete_controls','unconfirmed_route_commands','wall_risk_trigger_commands')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def runs(bits):
 result=[];start=None
 for i,value in enumerate(list(map(bool,bits))+[False]):
  if value and start is None:start=i
  if not value and start is not None:result.append((start,i));start=None
 return result
def center(letter,state):
 rects=official_shape(letter).components
 weights=np.asarray([(r.x1-r.x0)*(r.y1-r.y0)for r in rects])
 xy=np.asarray([[(r.x0+r.x1)/2,(r.y0+r.y1)/2]for r in rects]);local=(xy*weights[:,None]).sum(axis=0)/weights.sum()
 cs,sn=math.cos(float(state[4])),math.sin(float(state[4]))
 return np.asarray([state[2]+cs*local[0]-sn*local[1],state[3]+sn*local[0]+cs*local[1]])

CACHE={}
def load(path,expected=None):
 path=Path(path).resolve()
 if path not in CACHE:
  digest=sha(path)
  with np.load(path,allow_pickle=False)as z:CACHE[path]=(digest,{key:z[key]for key in z.files})
 digest,arrays=CACHE[path]
 if expected is not None:assert digest==expected,('source_SHA',str(path))
 return arrays,digest

def empty():
 return dict(states=np.empty((1,15)),actions=np.empty((0,2)),attempted_actions=np.empty((0,2)),
 substep_status=np.empty(0,np.int8),contact_flags=np.empty((0,3),np.uint8),contact_tags=np.empty((0,5),np.int16),
 contact_values=np.empty((0,11)),contact_arc_offsets=np.asarray([0]),contact_arc_ids=np.empty(0,np.int16),
 control_phase=np.empty(0,np.uint8),control_mode=np.empty(0,np.uint8),rgb_frame_sha256=np.empty((0,32),np.uint8))

def decisions(metadata,root):
 values=metadata.get('decision_telemetry',[])
 if metadata.get('decision_telemetry_file'):
  path=Path(metadata['decision_telemetry_file']);path=path if path.is_absolute()else root/path
  expected=metadata.get('decision_telemetry_sha256',metadata.get('telemetry_sha256'))
  if expected:assert sha(path)==expected,'telemetry_SHA'
  with gzip.open(path,'rt',encoding='utf-8')as f:payload=json.load(f)
  values=payload['decisions']if isinstance(payload,dict)else payload
 return values

def one(case,entry,arrays,version,path=None,metadata=None,source_root=STAGE):
 a=arrays;n=len(a['actions']);m=len(a['substep_status']);charged=int(entry.get('charged_controls',0))
 states=a['states'];assert states.shape==(n+1,15)
 if n:assert np.isfinite(states).all()
 known=a['substep_status']==1;unknown=a['substep_status']==2
 if n:assert known[:n*10].reshape(n,10).all(),'complete_control_has_unknown_physics'
 ao=(a['contact_flags'][:,0]&2)!=0;ow=(a['contact_flags'][:,1]&2)!=0;aw=(a['contact_flags'][:,2]&2)!=0
 counts=ao[:n*10].reshape(n,10).sum(axis=1);contact=counts>0
 wall=ow[:n*10].reshape(n,10).any(axis=1)
 translation=np.asarray([np.linalg.norm(center(case['letter'],states[i+1])-center(case['letter'],states[i]))*96/512 for i in range(n)])
 angle=np.diff(states[:,4])*180/math.pi if n else np.empty(0)
 pusher=np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1)*96/512 if n else np.empty(0)
 near=(translation<.1)&(np.abs(angle)<.1);obvious=(translation>=2)|(np.abs(angle)>=5)
 labels=np.where(near,0,np.where(obvious,2,1));six=dict.fromkeys(CLASSES,0)
 for has,motion in zip(contact,labels):six[('contact'if has else'no_contact')+'_'+('nearstatic','small','obvious')[int(motion)]]+=1
 assert sum(six.values())==n
 point_tags=a['contact_tags'];point_values=a['contact_values'];offsets=a['contact_arc_offsets'];arc_ids=a['contact_arc_ids']
 positive=np.zeros(m,bool);valid=np.zeros(m,bool);target=int(case['initial_requested_arc']['arc_id'])
 hit=np.zeros(m,bool);target_valid=np.zeros(m,bool);target_positive=np.zeros(m,bool)
 for point,(tag,value)in enumerate(zip(point_tags,point_values)):
  si,kind,phase,_,mask=map(int,tag)
  if kind!=0 or phase!=1:continue
  assert 0<=si<m and ao[si],('postsolve_row_flag',case['case_id'],si)
  row_hit=known[si]and target in arc_ids[offsets[point]:offsets[point+1]]
  hit[si]|=row_hit
  row_valid=bool(mask&4)and np.isfinite(value[7:9]).all()
  row_positive=row_valid and math.hypot(float(value[7]),float(value[8]))>1e-8
  valid[si]|=row_valid;positive[si]|=row_positive
  # Impulse and requested arc must belong to this same point row.
  target_valid[si]|=row_hit and row_valid;target_positive[si]|=row_hit and row_positive
 impulse=positive[:n*10].reshape(n,10).any(axis=1)
 phase=a['control_phase'][:n];mode=a['control_mode'][:n]
 withdraw_start=int(case['task_withdraw_start_control']);withdraw_end=int(case['task_withdraw_end_control_exclusive'])
 withdraw=(np.arange(n)>=withdraw_start)&(np.arange(n)<withdraw_end)
 assert np.array_equal(phase==2,withdraw),'planned_withdraw_not_preserved'
 complete_known=known[:n*10].reshape(n,10).all(axis=1)
 release_indices=np.flatnonzero(withdraw&complete_known&~contact);released=bool(len(release_indices))
 start=int(case['task_recontact_start_control'])*10
 after_known=np.flatnonzero(known[start:])+start if m>start else np.empty(0,int)
 reached=bool(len(after_known));after_contact=np.flatnonzero(ao[start:]&known[start:])+start if reached else np.empty(0,int)
 any_first=int(after_contact[0])if len(after_contact)else None
 first=any_first if released else None
 never=reached and not released;censored=reached and released and first is None
 trace=decisions(metadata or{},source_root)
 unconfirmed=np.asarray([str(d.get('controller_state',''))=='unconfirmed_hold'or'unconfirmed_route'in str(d.get('controller_state',''))or int(d.get('unconfirmed_route_controls',0)or 0)>0 for d in trace],bool)
 risk=[d.get('wall_risk',d.get('safety_risk',d.get('risk',{})))for d in trace]
 row=dict(case_id=case['case_id'],source_cp1_case_id=case.get('source_cp1_case_id'),group=case['group'],letter=case['letter'],
 version=version,status=entry['status'],source_file=None if path is None else str(path),
 charged_controls=charged,completed_controls=n,saved_attempted_controls=len(a['attempted_actions']),known_substeps=int(known.sum()),
 unknown_advance_substeps=int(unknown.sum()),partial_substeps=m-n*10,reserved_not_started_controls=int(entry.get('reserved_not_started_controls',0)),
 unclassified_charged_controls=charged-n,any_contact_controls=int(contact.sum()),terminal_contact_controls=int(ao[9:n*10:10].sum()),
 contact_substeps=int(ao.sum()),object_wall_substeps=int(ow.sum()),agent_wall_substeps=int(aw.sum()),
 unknown_impulse_contact_substeps=int((ao&~valid).sum()),positive_impulse_substeps=int((ao&positive).sum()),
 contact_obvious_controls=int((contact&obvious).sum()),contact_obvious_impulse_controls=int((contact&obvious&impulse).sum()),
 contact_obvious_atleast5_controls=int(((counts>=5)&obvious).sum()),contact_obvious_all10_controls=int(((counts==10)&obvious).sum()),
 contact_obvious_impulse_atleast5_controls=int(((counts>=5)&obvious&impulse).sum()),nearstatic_controls=int(near.sum()),
 moving_pusher_noAO_controls=int((~contact&(pusher>=.1)).sum()),moving_pusher_noAO_objectnearstatic_controls=int((~contact&(pusher>=.1)&near).sum()),
 route_recovery_noAO_controls=int((~contact&(mode==1)&np.isin(phase,[0,1,3])).sum()),planned_push_controls=int((phase==1).sum()),
 planned_push_noAO_controls=int(((phase==1)&~contact).sum()),wallstall_controls=int((contact&wall&near).sum()),
 simultaneous_AO_OW_nearstatic_substeps=int((ao[:n*10].reshape(n,10)[near]&ow[:n*10].reshape(n,10)[near]).sum()),
 late_completed_controls=max(n-24,0),late_charged_controls=max(charged-24,0),late_contact_obvious_controls=int((contact&obvious&(np.arange(n)>=24)).sum()),
 early_completed_controls=min(n,24),early_contact_obvious_controls=int((contact&obvious&(np.arange(n)<24)).sum()),
 withdraw_completed_controls=int(withdraw.sum()),withdraw_noAO_controls=int((withdraw&~contact).sum()),hold_complete_controls=int((mode==0).sum()),
 longest_hold_complete_controls=max((b-a for a,b in runs(mode==0)),default=0),unconfirmed_route_commands=int(unconfirmed.sum()),
 longest_unconfirmed_route_commands=max((b-a for a,b in runs(unconfirmed)),default=0),
 wall_risk_trigger_commands=sum(bool(v.get('risky',False))if isinstance(v,dict)else False for v in risk),
 unconfirmed_telemetry_available=bool(trace),six_classes=six,withdrawal_release_observed=released,
 verified_release_control_indices=release_indices.tolist(),prior_AO_before_withdrawal=bool(ao[:withdraw_start*10].any()),
 recontact_reached=reached,recontact_observed=first is not None,recontact_first_substep=first,
 recontact_delay_seconds=None if first is None else(first-start)*.01,recontact_never_separated=bool(never),
 recontact_right_censored=bool(censored),recontact_not_reached=not reached,
 recontact_censor_seconds=(int(after_known[-1])+1-start)*.01 if censored else None,
 physical_guard_failure=entry['status']=='failed_physics')
 target_counts=hit[:n*10].reshape(n,10).sum(axis=1);positive_counts=target_positive[:n*10].reshape(n,10).sum(axis=1)
 target_runs=runs(hit&target_positive)
 arc=dict(case_id=case['case_id'],version=version,group=case['group'],letter=case['letter'],initial_requested_arc_id=target,
 requested_arc_hit_observed=bool(hit.any()),first_requested_arc_substep=int(np.flatnonzero(hit)[0])if hit.any()else None,
 requested_arc_contact_substeps=int(hit.sum()),requested_arc_valid_impulse_substeps=int((hit&target_valid).sum()),
 requested_arc_positive_impulse_substeps=int((hit&target_positive).sum()),requested_arc_unknown_impulse_substeps=int((hit&~target_valid).sum()),
 requested_arc_contact_complete_controls=int((target_counts>0).sum()),requested_arc_positive_impulse_complete_controls=int((positive_counts>0).sum()),
 requested_arc_contact_and_obvious_complete_controls=int(((target_counts>0)&obvious).sum()),
 requested_arc_positive_impulse_and_obvious_complete_controls=int(((positive_counts>0)&obvious).sum()),
 requested_arc_atleast5_positive_substeps_and_obvious_controls=int(((positive_counts>=5)&obvious).sum()),
 requested_arc_max_positive_impulse_run_seconds=max(((b-a)*.01 for a,b in target_runs),default=0.),
 requested_arc_positive_impulse_runs_substeps=[[a,b]for a,b in target_runs],
 hit_control_motion=[dict(control=int(i),contact_substeps=int(target_counts[i]),positive_impulse_substeps=int(positive_counts[i]),
 translation_image96_px=float(translation[i]),signed_rotation_deg=float(angle[i]),motion_class=int(labels[i]))for i in np.flatnonzero(target_counts)])
 return row,arc

def aggregate(rows):
 result={key:sum(int(r[key])for r in rows)for key in NUMERIC}
 result['six_classes']={key:sum(r['six_classes'][key]for r in rows)for key in CLASSES}
 result.update(planned_trajectories=len(rows),attempted_trajectories=sum(r['status']!='not_executed'for r in rows),
  planned_control_slots=len(rows)*48,longest_hold_complete_controls=max((r['longest_hold_complete_controls']for r in rows),default=0),
  longest_unconfirmed_route_commands=max((r['longest_unconfirmed_route_commands']for r in rows),default=0))
 result['recovery']=dict(planned_tasks=len(rows),reached_tasks=sum(r['recontact_reached']for r in rows),
 observed=sum(r['recontact_observed']for r in rows),right_censored=sum(r['recontact_right_censored']for r in rows),
 never_separated=sum(r['recontact_never_separated']for r in rows),not_reached=sum(r['recontact_not_reached']for r in rows),
 verified_withdrawal_release=sum(r['withdrawal_release_observed']for r in rows))
 return result

def main():
 plan=read(STAGE/'plan/cases.json');manifest=read(STAGE/'manifest.json');ledger=read(STAGE/'resource-ledger.json')
 if not manifest['closed']or not ledger['closed']:raise RuntimeError('Wait for explicit closed notification; this audit does not poll')
 assert sha(STAGE/'plan/cases.json')==manifest['plan_sha256']
 entries={e['case_id']:e for e in manifest['entries']};all_rows=[];weak=[];state_checks=[];costs=Counter();mismatches=[]
 for case in plan['cases']:
  versions={}
  for version,ref in(('R3A',case.get('r3a_baseline')),('CP1',case.get('cp1_baseline'))):
   if ref is None:continue
   path=Path(ref['file']);arrays,digest=load(path,ref['sha256']);metadata=read(ref['metadata_file'])
   assert sha(ref['metadata_file'])==ref['metadata_sha256'],'baseline_metadata_SHA'
   entry=dict(status=ref.get('status',metadata['status']),charged_controls=ref.get('charged_controls',metadata['charged_controls']),reserved_not_started_controls=0)
   row,arc=one(case,entry,arrays,version,path,metadata,path.parent.parent);all_rows.append(row);versions[version]=(arrays,arc)
  entry=entries.get(case['case_id']);has_data=entry is not None and bool(entry.get('file'))
  if entry is None:entry=dict(status='not_executed',charged_controls=0,reserved_not_started_controls=0);arrays=empty();metadata={};path=None
  else:
   metadata_path=STAGE/entry['metadata_file'];assert sha(metadata_path)==entry['metadata_sha256'],'new_metadata_SHA';metadata=read(metadata_path)
   if has_data:path=STAGE/entry['file'];arrays,_=load(path,entry['sha256'])
   else:path=None;arrays=empty()
  row,arc=one(case,entry,arrays,'CP2',path,metadata,STAGE);all_rows.append(row);versions['CP2']=(arrays,arc)
  if entry['status']!='not_executed':
   for key in('charged_controls','completed_controls','known_substeps','unknown_advance_substeps','reserved_not_started_controls'):costs[key]+=row[key]
   for key in('completed_controls','known_substeps','unknown_advance_substeps','partial_substeps'):
    if key in entry:assert row[key]==entry[key],(case['case_id'],key,row[key],entry[key])
  if has_data and case.get('cp1_baseline'):
   old=versions['CP1'][0]
   state_checks.append(dict(case_id=case['case_id'],full_state_bitwise_equal=bool(np.array_equal(arrays['states'][0],old['states'][0])),
    original_RGB_hash_equal=bool(np.array_equal(arrays['rgb_frame_sha256'][0],old['rgb_frame_sha256'][0]))))
  is_weak=(case['group']=='CP1_B_repeat'and case['source_cp1_case_id'].endswith('-04'))or(case['group']=='CP2_fresh'and'weak'in case['role'])
  if is_weak:
   for version in('CP1','CP2'):
    if version in versions:weak.append(versions[version][1])
 for key,value in costs.items():assert ledger[key]==value,('ledger',key,ledger[key],value)
 assert costs['known_substeps']+costs['unknown_advance_substeps']==ledger['engine_step_calls']
 assert all(r['full_state_bitwise_equal']and r['original_RGB_hash_equal']for r in state_checks)
 populations={
 'R3A_original_A':[r for r in all_rows if r['version']=='R3A'],
 'CP1_original_A':[r for r in all_rows if r['version']=='CP1'and r['group']=='CP1_A_repeat'],
 'CP2_repeat_A':[r for r in all_rows if r['version']=='CP2'and r['group']=='CP1_A_repeat'],
 'CP1_original_B':[r for r in all_rows if r['version']=='CP1'and r['group']=='CP1_B_repeat'],
 'CP2_repeat_B':[r for r in all_rows if r['version']=='CP2'and r['group']=='CP1_B_repeat'],
 'CP2_new_families':[r for r in all_rows if r['version']=='CP2'and r['group']=='CP2_fresh'],
 'CP2_all_planned':[r for r in all_rows if r['version']=='CP2'],
 'CP2_attempted_prefix':[r for r in all_rows if r['version']=='CP2'and r['status']!='not_executed']}
 groups={key:aggregate(rows)for key,rows in populations.items()}
 summary_path=STAGE/'report/summary.json'
 if summary_path.exists():
  summary=read(summary_path)
  def compare(location,key,ours,theirs):
   if ours!=theirs:
    if isinstance(ours,(float,np.floating))and isinstance(theirs,(float,int))and abs(ours-theirs)<1e-11:return
    mismatches.append(dict(location=location,key=key,raw=ours,reported=theirs))
  for name,ours in groups.items():
   theirs=summary['groups'][name]
   for key,value in ours.items():
    if key=='recovery':
     for k,v in value.items():compare(name,'recovery.'+k,v,theirs[key].get(k))
    else:compare(name,key,value,theirs.get(key))
  theirs_rows={(r['case_id'],r['version']):r for r in summary['all_version_rows']}
  for ours in all_rows:
   theirs=theirs_rows[(ours['case_id'],ours['version'])]
   for key,value in ours.items():
    if key in theirs and key not in('source_file','status','unconfirmed_telemetry_available'):compare(ours['case_id']+'/'+ours['version'],key,value,theirs[key])
  theirs_arcs={(r['case_id'],r['version']):r for r in summary['weak_requested_arc_rows']}
  for ours in weak:
   theirs=theirs_arcs[(ours['case_id'],ours['version'])]
   for key,value in ours.items():
    if key in theirs and key!='hit_control_motion':compare(ours['case_id']+'/'+ours['version'],key,value,theirs[key])
 payload=dict(kind='CONTACT-PILOT2-independent-raw-check.v1',at_utc=datetime.now(timezone.utc).isoformat(),passed=not mismatches,
 mismatches=mismatches,groups=groups,all_version_rows=all_rows,weak_requested_arc_rows=weak,repeat_initial_checks=state_checks,
 ledger_raw_costs=dict(costs),raw_NPZ_read_count=len(CACHE),actual_new_NPZ_read_count=sum(r['version']=='CP2'and r['source_file']is not None for r in all_rows),
 comparison_summary_sha256=sha(summary_path)if summary_path.exists()else None,plan_sha256=sha(STAGE/'plan/cases.json'),
 manifest_sha256=sha(STAGE/'manifest.json'),new_report_scoring_helpers_imported=False,new_physics_controls=0,
 model_calls=0,Torch_imported='torch'in sys.modules,
 definitions=dict(complete_motion='Area-weighted rectangle centroid and signed unfurled angle; near<.1px AND abs<.1deg,obvious>=2px OR abs>=5deg.',
 strict_recovery='Complete10known/noAO interval in fixedwithdraw required; never-separated cannot be0delay success.',
 requested_impulse='Actual requested arcID and finite validmask norm>1e-8 in the same postsolveAO pointrow; existence only,never sums.',
 unexecuted='Zero observed counts/N/A,not fabricated static intervals; plannedslots reported separately.'))
 (STAGE/'independent-check.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps(dict(passed=payload['passed'],mismatches=mismatches,raw_NPZ_reads=len(CACHE),repeat_initial_exact=len(state_checks),
 costs=dict(costs),groups={k:dict(six=v['six_classes'],contact_substeps=v['contact_substeps'],moving_noAO=v['moving_pusher_noAO_controls'],
 late_contact_obvious=v['late_contact_obvious_controls'],recovery=v['recovery'])for k,v in groups.items()},
 weak=[dict(case_id=r['case_id'],version=r['version'],arc=r['initial_requested_arc_id'],contact=r['requested_arc_contact_substeps'],
 positive=r['requested_arc_positive_impulse_substeps'],motion=r['requested_arc_positive_impulse_and_obvious_complete_controls'])for r in weak])))

if __name__=='__main__':main()
