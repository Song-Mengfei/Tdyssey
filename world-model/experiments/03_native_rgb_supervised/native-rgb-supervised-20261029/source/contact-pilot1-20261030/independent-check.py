"""Independent raw pilot arithmetic; no import of collector/report scoring code."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
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

CLASSES=tuple(f'{contact}_{motion}'for contact in('no_contact','contact')for motion in('nearstatic','small','obvious'))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def center(letter,state):
    # Explicit component area weighting, independently of new audit/score helpers.
    rectangles=official_shape(letter).components
    area=sum((r.x1-r.x0)*(r.y1-r.y0)for r in rectangles)
    local=[sum((r.x1-r.x0)*(r.y1-r.y0)*((r.x0+r.x1)/2 if i==0 else(r.y0+r.y1)/2)for r in rectangles)/area for i in(0,1)]
    cs,sn=math.cos(float(state[4])),math.sin(float(state[4]))
    return np.array([state[2]+cs*local[0]-sn*local[1],state[3]+sn*local[0]+cs*local[1]])


def one(path,case,entry,baseline=False):
    with np.load(path,allow_pickle=False)as archive:
        a={key:archive[key]for key in('states','actions','attempted_actions','control_valid','substep_states','substep_status',
            'contact_flags','contact_tags','contact_values','control_substep_offsets','control_phase','control_mode','rgb_frame_sha256')}
    states=a['states'];n=len(states)-1;m=len(a['substep_status']);c=len(a['attempted_actions'])
    assert states.shape==(n+1,15)and len(a['actions'])==n
    assert np.array_equal(a['control_substep_offsets'][:n+1],np.arange(n+1)*10)
    assert np.isfinite(states).all()
    ao=(a['contact_flags'][:,0]&2)!=0;ow=(a['contact_flags'][:,1]&2)!=0;aw=(a['contact_flags'][:,2]&2)!=0
    counts=ao[:n*10].reshape(n,10).sum(axis=1)
    wall=ow[:n*10].reshape(n,10).any(axis=1)
    trans=np.array([float(np.linalg.norm(center(case['letter'],states[i+1])-center(case['letter'],states[i])))*96/512 for i in range(n)])
    angle=np.diff(states[:,4])*180/math.pi
    pusher=np.linalg.norm(np.diff(states[:,:2],axis=0),axis=1)*96/512
    near=(trans<.1)&(np.abs(angle)<.1)
    obvious=(trans>=2)|(np.abs(angle)>=5)
    label=np.where(near,0,np.where(obvious,2,1))
    contact=counts>0;positive=np.zeros(m,bool);known=np.zeros(m,bool)
    # Per-substep existence only. A duplicated event/point impulse cannot add
    # multiple counts or be converted to net physical force by this check.
    for tag,value in zip(a['contact_tags'],a['contact_values']):
        index,kind,phase,_,mask=map(int,tag)
        if kind==0 and phase==1 and mask&4 and np.isfinite(value[7:9]).all():
            known[index]=True
            if math.hypot(float(value[7]),float(value[8]))>1e-8:positive[index]=True
    impulse=positive[:n*10].reshape(n,10).any(axis=1)
    six={name:0 for name in CLASSES}
    for has_contact,motion in zip(contact,label):
        six[('contact'if has_contact else'no_contact')+'_'+('nearstatic','small','obvious')[int(motion)]]+=1
    assert sum(six.values())==n
    phase=a['control_phase'][:n];mode=a['control_mode'][:n]
    charged=48 if baseline else int(entry['charged_controls'])
    result=dict(case_id=case['case_id'],letter=case['letter'],group='A_old_baseline'if baseline else'A_new_controller'if case['group']=='A_matched'else'B_new_fresh',
        source_file=str(path),source_sha256=sha(path),status='completed'if baseline else entry['status'],
        charged_controls=charged,completed_controls=n,saved_attempted_controls=c,
        known_substeps=int(np.count_nonzero(a['substep_status']==1)),unknown_advance_substeps=int(np.count_nonzero(a['substep_status']==2)),
        partial_substeps=m-n*10,reserved_not_started_controls=0 if baseline else int(entry['reserved_not_started_controls']),
        unclassified_charged_controls=charged-n,
        any_contact_controls=int(contact.sum()),terminal_contact_controls=int(ao[9:n*10:10].sum()),
        contact_substeps=int(ao.sum()),object_wall_substeps=int(ow.sum()),agent_wall_substeps=int(aw.sum()),
        positive_impulse_substeps=int((positive&ao).sum()),unknown_impulse_contact_substeps=int((ao&~known).sum()),
        contact_obvious_controls=int((contact&obvious).sum()),contact_obvious_impulse_controls=int((contact&obvious&impulse).sum()),
        contact_obvious_atleast5_controls=int(((counts>=5)&obvious).sum()),contact_obvious_all10_controls=int(((counts==10)&obvious).sum()),
        contact_obvious_impulse_atleast5_controls=int(((counts>=5)&obvious&impulse).sum()),nearstatic_controls=int(near.sum()),
        moving_pusher_noAO_controls=int((~contact&(pusher>=.1)).sum()),
        moving_pusher_noAO_objectnearstatic_controls=int((~contact&(pusher>=.1)&near).sum()),
        route_recovery_noAO_controls=int((~contact&(mode==1)&np.isin(phase,[0,1,3])).sum()),
        planned_push_controls=int((phase==1).sum()),planned_push_noAO_controls=int(((phase==1)&~contact).sum()),
        wallstall_controls=int((contact&wall&near).sum()),
        simultaneous_AO_OW_nearstatic_substeps=int((ao[:n*10].reshape(n,10)[near]&ow[:n*10].reshape(n,10)[near]).sum()),
        late_completed_controls=max(n-24,0),late_charged_controls=max(charged-24,0),late_contact_obvious_controls=int((contact&obvious&(np.arange(n)>=24)).sum()),
        early_completed_controls=min(n,24),early_contact_obvious_controls=int((contact&obvious&(np.arange(n)<24)).sum()),
        withdraw_completed_controls=int((phase==2).sum()),withdraw_noAO_controls=int(((phase==2)&~contact).sum()),six_classes=six)
    start=case['task_recontact_start_control']*10
    reached=m>start
    candidates=np.flatnonzero(ao[start:])if reached else np.array([],int)
    recovered=len(candidates)>0
    result.update(recontact_reached=reached,recontact_observed=recovered,
        recontact_delay_seconds=float(candidates[0]/100)if recovered else None,
        recontact_censor_seconds=(m-start)/100 if reached and not recovered else None,
        recontact_not_reached=not reached)
    if m>n*10 and np.isfinite(a['substep_states'][-1]).all():
        begin,end=states[-1],a['substep_states'][-1]
        result['partial_motion']=dict(translation_image96_px=float(np.linalg.norm(center(case['letter'],end)-center(case['letter'],begin))*96/512),
            signed_rotation_deg=float((end[4]-begin[4])*180/math.pi),
            pusher_translation_image96_px=float(np.linalg.norm(end[:2]-begin[:2])*96/512),
            observed_substeps=m-n*10,not_a_complete_0p1s_control=True)
    if not baseline:
        for key in('completed_controls','known_substeps','unknown_advance_substeps','partial_substeps'):
            assert result[key]==entry[key],(case['case_id'],key,result[key],entry[key])
    return result,a['states'][0],a['rgb_frame_sha256'][0]


def aggregate(rows):
    exclusions={'case_id','letter','group','source_file','source_sha256','status','six_classes','partial_motion',
                'recontact_reached','recontact_observed','recontact_delay_seconds','recontact_censor_seconds','recontact_not_reached'}
    result={key:sum(int(row[key])for row in rows)for key in rows[0]if key not in exclusions}
    result['six_classes']={key:sum(row['six_classes'][key]for row in rows)for key in CLASSES}
    delays=[r['recontact_delay_seconds']for r in rows if r['recontact_observed']]
    result['recovery']=dict(planned_tasks=len(rows),reached_tasks=sum(r['recontact_reached']for r in rows),
        observed=sum(r['recontact_observed']for r in rows),right_censored=sum(r['recontact_reached']and not r['recontact_observed']for r in rows),
        not_reached=sum(r['recontact_not_reached']for r in rows),
        observed_delay_seconds=dict(count=len(delays),median=float(np.median(delays))if delays else None,
            p90=float(np.quantile(delays,.9))if delays else None,maximum=max(delays)if delays else None))
    result['trajectories']=len(rows)
    return result


def main():
    manifest=read(STAGE/'manifest.json');ledger=read(STAGE/'resource-ledger.json');plan=read(STAGE/'plan/cases.json')
    if not manifest['closed']or not ledger['closed']:raise RuntimeError('Independent check waits for explicit closed manifest')
    assert sha(STAGE/'plan/cases.json')==manifest['plan_sha256']
    cases={case['case_id']:case for case in plan['cases']}
    rows=[];statechecks=[];costs=Counter();mismatches=[]
    for entry in manifest['entries']:
        case=cases[entry['case_id']]
        if 'file'not in entry:raise RuntimeError('No saved raw data; record unobserved case explicitly before comparing')
        path=STAGE/entry['file']
        assert sha(path)==entry['sha256'],entry['case_id']+' NPZ SHA'
        assert sha(STAGE/entry['metadata_file'])==entry['metadata_sha256'],entry['case_id']+' metadata SHA'
        new,initial,rgb=one(path,case,entry)
        rows.append(new)
        for key in('charged_controls','completed_controls','known_substeps','unknown_advance_substeps','reserved_not_started_controls'):
            costs[key]+=new[key]
        if case['group']=='A_matched':
            baseline=Path(case['baseline_npz'])
            assert sha(baseline)==case['baseline']['sha256']
            old,oldinitial,oldrgb=one(baseline,case,None,baseline=True)
            rows.append(old)
            statechecks.append(dict(case_id=case['case_id'],full_state_bitwise_equal=bool(np.array_equal(initial,oldinitial)),
                                    original_RGB_hash_equal=bool(np.array_equal(rgb,oldrgb))))
    for key,value in costs.items():assert ledger[key]==value,(key,ledger[key],value)
    assert costs['known_substeps']+costs['unknown_advance_substeps']==ledger['engine_step_calls']
    assert all(r['full_state_bitwise_equal']and r['original_RGB_hash_equal']for r in statechecks)
    groups={group:aggregate([row for row in rows if row['group']==group])for group in('A_old_baseline','A_new_controller','B_new_fresh')}
    groups['all_new_diagnostic_only']=aggregate([row for row in rows if row['group']!='A_old_baseline'])
    reported=STAGE/'report/summary.json'
    if reported.exists():
        summary=read(reported)
        for group,ours in groups.items():
            theirs=summary['groups'][group]
            for key,value in ours.items():
                if key=='recovery':
                    for subkey in('planned_tasks','reached_tasks','observed','right_censored','not_reached'):
                        if value[subkey]!=theirs[key][subkey]:mismatches.append(dict(group=group,key=key+'.'+subkey,raw=value[subkey],reported=theirs[key][subkey]))
                elif value!=theirs.get(key):mismatches.append(dict(group=group,key=key,raw=value,reported=theirs.get(key)))
        attempted={row['case_id']:row for row in summary['attempts']}
        for row in rows:
            if row['group']=='A_old_baseline':continue
            theirs=attempted[row['case_id']]
            for key,value in row.items():
                if key not in('partial_motion','source_file','source_sha256','status','group')and key in theirs and value!=theirs[key]:
                    if isinstance(value,float)and abs(value-theirs[key])<1e-12:continue
                    mismatches.append(dict(case_id=row['case_id'],key=key,raw=value,reported=theirs[key]))
    report=dict(kind='CONTACT-PILOT1-independent-raw-numeric-check.v1',at_utc=datetime.now(timezone.utc).isoformat(),
        passed=not mismatches,mismatches=mismatches,groups=groups,attempts_and_baselines=rows,
        ledger_raw_costs=dict(costs),paired_initial_checks=statechecks,
        physical_failures=[dict(case_id=entry['case_id'],status=entry['status'],charged_controls=entry['charged_controls'],partial_substeps=entry['partial_substeps'])for entry in manifest['entries']if entry['status']!='completed'],
        original_NPZ_read_count=24,new_NPZ_read_count=len(manifest['entries']),
        new_report_scoring_helpers_imported=False,comparison_summary_sha256=sha(reported)if reported.exists()else None,
        plan_sha256=sha(STAGE/'plan/cases.json'),manifest_sha256=sha(STAGE/'manifest.json'),
        six_class_denominator='Observed complete controls only; partial and unstarted charges explicitly unclassified rather than promoted to0.1s targets.',
        contact_substep_denominator='All actually saved known physical intervals including failed partial control.',
        new_physics_controls=0,model_calls=0,Torch_imported='torch'in sys.modules)
    (STAGE/'independent-check.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],mismatches=mismatches,groups={g:dict(six=v['six_classes'],contact_substeps=v['contact_substeps'],charged=v['charged_controls'],moving_noAO=v['moving_pusher_noAO_controls'],late_contact_obvious=v['late_contact_obvious_controls'])for g,v in groups.items()},costs=dict(costs),paired_initial_equal=len(statechecks))))


if __name__=='__main__':main()
