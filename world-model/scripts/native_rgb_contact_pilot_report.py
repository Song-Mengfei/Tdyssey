"""Saved-array CONTACT-PILOT1 comparisons; no collection, model or physics steps."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_contact_audit import actual_signals,motion,classify_motion,boolean_runs
from tdyssey.native_rgb_r3a_collection import segments

STAGE=ROOT/'artifacts/native-rgb-supervised-20261029/contact-pilot1-20261030'
R3A=STAGE.parent/'r3a'
GROUPS=('A_matched','B_fresh')
CLASSES=tuple(f'{contact}_{movement}'for contact in('no_contact','contact')for movement in('nearstatic','small','obvious'))
MEDIA_CASES=('CP1-A-T-01','CP1-A-X-04')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def ratio(n,d):return float(n/d)if d else None
def pct(value):return 'N/A'if value is None else f'{value*100:.2f}%'
def quantile(values):
    if not len(values):return dict(count=0,min=None,p10=None,median=None,p90=None,max=None)
    return dict(count=len(values),**dict(zip(('min','p10','median','p90','max'),map(float,np.quantile(values,[0,.1,.5,.9,1])))))
def resolve(path,base):
    path=Path(path);return path.resolve()if path.is_absolute()else(base/path).resolve()
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |'for row in rows])
def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def load_arrays(path):
    with np.load(path,allow_pickle=False)as z:return{k:z[k]for k in z.files}


def measure(case,entry,arrays,path,*,baseline=False):
    """Complete-control classes plus honest partial/charged denominator scope."""
    n=len(arrays['actions']);attempted=len(arrays['attempted_actions']);m=len(arrays['contact_flags'])
    charged=48 if baseline else int(entry['charged_controls'])
    status=np.asarray(arrays['substep_status']);known=status==1;unknown=status==2
    if arrays['states'].shape!=(n+1,15)or m<n*10 or charged<n:raise ValueError('Invalid saved attempt timeline')
    signal=actual_signals(arrays);t,r,p=motion(arrays['states'],case['letter']);classes,intermediate=classify_motion(t,r)
    ao=signal['ao'][:n*10].reshape(n,10);ow=signal['ow'][:n*10].reshape(n,10);aw=signal['aw'][:n*10].reshape(n,10)
    contact=ao.any(axis=1);counts=ao.sum(axis=1);walls=ow.any(axis=1);near=classes==0;obvious=classes==2
    impulse=signal['nonzero'][:n*10,1].reshape(n,10);positive=impulse.any(axis=1)
    six={name:int(np.count_nonzero((contact==bool(i//3))&(classes==i%3)))for i,name in enumerate(CLASSES)}
    if sum(six.values())!=n:raise AssertionError('Six complete-control classes do not reconcile')
    phase=arrays['control_phase'][:n];mode=arrays['control_mode'][:n];push=phase==1
    late=np.arange(n)>=24;withdraw=phase==2;route=np.isin(phase,[0,1,3])&(mode==1)&~contact
    start=case.get('task_recontact_start_control',None if case.get('scenario')=='static'else 26)
    start_sub=None if start is None else int(start)*10
    reached=start_sub is not None and m>start_sub
    after=np.flatnonzero(signal['ao'][start_sub:]&known[start_sub:])+start_sub if reached else np.empty(0,int)
    first=int(after[0])if len(after)else None
    release=bool(np.any(withdraw&~contact))
    all_arcs=sorted(set(arc for subarcs in signal['arcs']for arc in subarcs))
    outer={int(s['id'])for s in segments(case['letter'])}
    full_wallstall=contact&near&walls
    partial=m-n*10;partial_motion=None
    if partial and np.isfinite(arrays['substep_states'][-1]).all():
        pt,pr,pp=motion(np.stack((arrays['states'][-1],arrays['substep_states'][-1])),case['letter'])
        partial_motion=dict(translation_image96_px=float(pt[0]),signed_rotation_deg=float(pr[0]),pusher_translation_image96_px=float(pp[0]))
    row=dict(case_id=case['case_id'],group=case['group'],letter=case['letter'],role=case.get('role'),scenario=case.get('scenario',case['slot']['scenario']),
        version='old_baseline'if baseline else'new_controller',source_file=str(path),source_sha256=sha(path),
        status='completed'if baseline else entry['status'],charged_controls=charged,completed_controls=n,saved_attempted_controls=attempted,
        known_substeps=int(known.sum()),unknown_advance_substeps=int(unknown.sum()),partial_substeps=partial,
        reserved_not_started_controls=0 if baseline else int(entry.get('reserved_not_started_controls',charged-attempted)),unclassified_charged_controls=charged-n,
        any_contact_controls=int(contact.sum()),terminal_contact_controls=int(ao[:,-1].sum())if n else 0,
        contact_substeps=int((signal['ao']&known).sum()),object_wall_substeps=int((signal['ow']&known).sum()),
        agent_wall_substeps=int((signal['aw']&known).sum()),unknown_impulse_contact_substeps=int((signal['ao']&~signal['impulse_known']&known).sum()),
        positive_impulse_substeps=int((signal['nonzero'][:,1]&known).sum()),
        contact_obvious_controls=int((contact&obvious).sum()),contact_obvious_impulse_controls=int((contact&obvious&positive).sum()),
        contact_obvious_atleast5_controls=int((contact&obvious&(counts>=5)).sum()),
        contact_obvious_all10_controls=int((contact&obvious&(counts==10)).sum()),
        contact_obvious_impulse_atleast5_controls=int((contact&obvious&positive&(counts>=5)).sum()),
        nearstatic_controls=int(near.sum()),moving_pusher_noAO_controls=int((~contact&(p>=.1)).sum()),
        moving_pusher_noAO_objectnearstatic_controls=int((~contact&near&(p>=.1)).sum()),
        route_recovery_noAO_controls=int(route.sum()),planned_push_controls=int(push.sum()),planned_push_noAO_controls=int((push&~contact).sum()),
        wallstall_controls=int(full_wallstall.sum()),wallstall_max_run_controls=max((b-a for a,b in boolean_runs(full_wallstall)),default=0),
        simultaneous_AO_OW_nearstatic_substeps=int((ao[near]&ow[near]).sum()),
        late_completed_controls=int(late.sum()),late_charged_controls=max(charged-24,0),late_contact_obvious_controls=int((late&contact&obvious).sum()),
        early_completed_controls=min(n,24),early_contact_obvious_controls=int((~late&contact&obvious).sum()),
        withdraw_completed_controls=int(withdraw.sum()),withdraw_noAO_controls=int((withdraw&~contact).sum()),
        withdrawal_release_observed=release,recontact_task_start_control=start,recontact_reached=reached,
        recontact_observed=first is not None,recontact_first_substep=first,
        recontact_delay_seconds=None if first is None else(first-start_sub)*.01,
        recontact_censor_seconds=None if not reached or first is not None else(m-start_sub)*.01,
        recontact_not_reached=start is not None and not reached,
        first_contact_without_verified_release=first is not None and not release,
        six_classes=six,actual_outer_arc_ids=sorted(outer&set(all_arcs)),actual_all_arc_ids=all_arcs,
        contact_run_substeps=[b-a for a,b in boolean_runs(signal['ao']&known)],
        positive_impulse_run_substeps=[b-a for a,b in boolean_runs(signal['nonzero'][:,1]&known)],partial_motion=partial_motion)
    row.update(any_contact_fraction_complete=ratio(row['any_contact_controls'],n),contact_substep_duty_known=ratio(row['contact_substeps'],row['known_substeps']),
        contact_obvious_fraction_complete=ratio(row['contact_obvious_controls'],n),contact_obvious_fraction_charged=ratio(row['contact_obvious_controls'],charged),
        nearstatic_fraction_complete=ratio(row['nearstatic_controls'],n),moving_pusher_noAO_fraction_complete=ratio(row['moving_pusher_noAO_controls'],n),
        moving_pusher_noAO_fraction_charged_upper=ratio(row['moving_pusher_noAO_controls']+charged-n,charged),
        late_contact_obvious_fraction_complete=ratio(row['late_contact_obvious_controls'],row['late_completed_controls']),
        late_contact_obvious_fraction_charged=ratio(row['late_contact_obvious_controls'],row['late_charged_controls']))
    controls=[dict(index=i,contact_substeps=int(counts[i]),terminal_contact=bool(ao[i,-1]),motion_class=int(classes[i]),
        translation_image96_px=float(t[i]),signed_rotation_deg=float(r[i]),pusher_translation_image96_px=float(p[i]),
        phase=int(phase[i]),mode=int(mode[i]),object_wall_substeps=int(ow[i].sum()),agent_wall_substeps=int(aw[i].sum()),
        positive_impulse_substeps=int(impulse[i].sum()))for i in range(n)]
    return row,controls


def aggregate(rows):
    numeric=('charged_controls','completed_controls','saved_attempted_controls','known_substeps','unknown_advance_substeps','partial_substeps',
        'reserved_not_started_controls','unclassified_charged_controls','any_contact_controls','terminal_contact_controls','contact_substeps',
        'object_wall_substeps','agent_wall_substeps','unknown_impulse_contact_substeps','positive_impulse_substeps','contact_obvious_controls',
        'contact_obvious_impulse_controls','contact_obvious_atleast5_controls','contact_obvious_all10_controls','contact_obvious_impulse_atleast5_controls',
        'nearstatic_controls','moving_pusher_noAO_controls','moving_pusher_noAO_objectnearstatic_controls','route_recovery_noAO_controls',
        'planned_push_controls','planned_push_noAO_controls','wallstall_controls','simultaneous_AO_OW_nearstatic_substeps',
        'late_completed_controls','late_charged_controls','late_contact_obvious_controls','early_completed_controls','early_contact_obvious_controls',
        'withdraw_completed_controls','withdraw_noAO_controls')
    result={key:sum(r[key]for r in rows)for key in numeric};n=result['completed_controls'];charged=result['charged_controls']
    tasks=[r for r in rows if r['recontact_task_start_control']is not None]
    reached=[r for r in tasks if r['recontact_reached']];observed=[r for r in reached if r['recontact_observed']]
    result.update(trajectories=len(rows),successful_48control_trajectories=sum(r['status']=='completed'and r['completed_controls']==48 for r in rows),
        failed_or_incomplete_trajectories=sum(r['status']!='completed'or r['completed_controls']!=48 for r in rows),
        failure_or_incomplete_fraction=ratio(sum(r['status']!='completed'or r['completed_controls']!=48 for r in rows),len(rows)),
        six_classes={key:sum(r['six_classes'][key]for r in rows)for key in CLASSES},
        any_contact_fraction_complete=ratio(result['any_contact_controls'],n),contact_substep_duty_known=ratio(result['contact_substeps'],result['known_substeps']),
        contact_obvious_fraction_complete=ratio(result['contact_obvious_controls'],n),contact_obvious_fraction_charged=ratio(result['contact_obvious_controls'],charged),
        nearstatic_fraction_complete=ratio(result['nearstatic_controls'],n),moving_pusher_noAO_fraction_complete=ratio(result['moving_pusher_noAO_controls'],n),
        moving_pusher_noAO_fraction_charged_upper=ratio(result['moving_pusher_noAO_controls']+charged-n,charged),
        late_contact_obvious_fraction_complete=ratio(result['late_contact_obvious_controls'],result['late_completed_controls']),
        late_contact_obvious_fraction_charged=ratio(result['late_contact_obvious_controls'],result['late_charged_controls']),
        planned_push_noAO_fraction=ratio(result['planned_push_noAO_controls'],result['planned_push_controls']),
        recovery=dict(planned_tasks=len(tasks),reached_tasks=len(reached),observed=len(observed),
            right_censored=len(reached)-len(observed),not_reached=len(tasks)-len(reached),
            unrecovered_fraction_reached=ratio(len(reached)-len(observed),len(reached)),
            unrecovered_or_not_reached_fraction=ratio(len(tasks)-len(observed),len(tasks)),
            observed_delay_seconds=quantile([r['recontact_delay_seconds']for r in observed]),
            censor_window_seconds=quantile([r['recontact_censor_seconds']for r in reached if not r['recontact_observed']]),
            withdrawal_release_observed=sum(r['withdrawal_release_observed']for r in tasks),
            contact_observed_without_verified_release=sum(r['first_contact_without_verified_release']for r in tasks)),
        contact_run_seconds=quantile([v*.01 for r in rows for v in r['contact_run_substeps']]),
        positive_impulse_run_seconds=quantile([v*.01 for r in rows for v in r['positive_impulse_run_substeps']]),
        case_contact_obvious_fraction_charged=quantile([r['contact_obvious_fraction_charged']for r in rows if r['charged_controls']]),
        case_empty_fraction_complete=quantile([r['moving_pusher_noAO_fraction_complete']for r in rows if r['completed_controls']]))
    result['outer_coverage_by_letter']={letter:dict(actual_arc_ids=sorted(set(a for r in rows if r['letter']==letter for a in r['actual_outer_arc_ids'])),
        possible_arc_count=len(segments(letter)))for letter in'TLOCXB'}
    recovery=result['recovery'];delays=recovery['observed_delay_seconds']
    result['engineering_targets']={
        'contact_obvious_at_least40pct':dict(value=result['contact_obvious_fraction_charged'],passed=charged>0 and result['contact_obvious_fraction_charged']>=.4),
        'moving_pusher_noAO_at_most25pct':dict(value=result['moving_pusher_noAO_fraction_charged_upper'],passed=charged>0 and result['moving_pusher_noAO_fraction_charged_upper']<=.25),
        'observed_recovery_median_at_most0_5s':dict(value=delays['median'],passed=delays['median']is not None and delays['median']<=.5),
        'observed_recovery_p90_at_most1s':dict(value=delays['p90'],passed=delays['p90']is not None and delays['p90']<=1),
        'unrecovered_at_most5pct':dict(value=recovery['unrecovered_or_not_reached_fraction'],passed=bool(tasks)and recovery['unrecovered_or_not_reached_fraction']<=.05)}
    return result


def paired(old,new):
    keys=('contact_obvious_fraction_charged','any_contact_fraction_complete','contact_substep_duty_known','nearstatic_fraction_complete',
        'moving_pusher_noAO_fraction_complete','moving_pusher_noAO_fraction_charged_upper','late_contact_obvious_fraction_charged',
        'wallstall_controls','planned_push_noAO_controls','contact_obvious_impulse_atleast5_controls','withdraw_noAO_controls')
    out=dict(case_id=new['case_id'],letter=new['letter'],role=new['role'],new_status=new['status'],old_baseline=new.get('baseline_trajectory_id'))
    for key in keys:
        out['old_'+key]=old[key];out['new_'+key]=new[key]
        out['delta_'+key]=None if old[key]is None or new[key]is None else new[key]-old[key]
    out.update(old_recontact_observed=old['recontact_observed'],new_recontact_observed=new['recontact_observed'],
        old_recontact_delay_seconds=old['recontact_delay_seconds'],new_recontact_delay_seconds=new['recontact_delay_seconds'],
        recontact_delay_delta_seconds=None if old['recontact_delay_seconds']is None or new['recontact_delay_seconds']is None else new['recontact_delay_seconds']-old['recontact_delay_seconds'])
    return out


def timeline_svg(case,controls,baseline=None):
    colors=('#cbd5e1','#fbbf24','#16a34a');left=148;cell=15;width=left+48*cell+16
    rows=[('New AO /10',controls,'contact_substeps'),('New motion',controls,'motion_class')]
    if baseline is not None:rows.extend([('Old AO /10',baseline,'contact_substeps'),('Old motion',baseline,'motion_class')])
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{52+len(rows)*26}" viewBox="0 0 {width} {52+len(rows)*26}">',
         '<rect width="100%" height="100%" fill="white"/>',f'<text x="8" y="17" font-family="sans-serif" font-size="12">{html.escape(case["case_id"])}: actual recorded controls; gray unobserved</text>']
    for j,(label,data,field)in enumerate(rows):
        y=28+j*26;svg.append(f'<text x="8" y="{y+14}" font-family="sans-serif" font-size="11">{label}</text>')
        for i in range(48):
            value=data[i]if i<len(data)else None
            color='#eee'if value is None else('#16a34a'if value[field]>0 else'#d8dde2')if field=='contact_substeps'else colors[value[field]]
            height=18 if field!='contact_substeps'or value is None else max(1,value[field]*1.8)
            title='unobserved'if value is None else f'control {i}: AO {value["contact_substeps"]}/10, motion={value["motion_class"]}, t={value["translation_image96_px"]:.4g}px, rot={value["signed_rotation_deg"]:.4g}deg'
            svg.append(f'<rect x="{left+i*cell}" y="{y+18-height:.1f}" width="14" height="{height:.1f}" fill="{color}"><title>{html.escape(title)}</title></rect>')
    for i in(0,12,24,36,48):svg.append(f'<text x="{left+i*cell}" y="{48+len(rows)*26}" font-family="sans-serif" font-size="10">{i/10:.1f}s</text>')
    svg.append('</svg>');return''.join(svg)


def scalar_rows(rows):return[{k:v for k,v in row.items()if not isinstance(v,(list,dict))}for row in rows]


def empty_arrays(case):
    """Zero numeric intervals for a failed setup; planned pose is never rendered."""
    return dict(states=np.asarray([case['candidate']['initial_state15']],np.float64),actions=np.empty((0,2)),attempted_actions=np.empty((0,2)),
        substep_states=np.empty((0,15)),substep_status=np.empty(0,np.int8),contact_flags=np.empty((0,3),np.uint8),
        contact_tags=np.empty((0,5),np.int16),contact_values=np.empty((0,11),np.float32),contact_arc_offsets=np.array([0],np.int32),
        contact_arc_ids=np.empty(0,np.int16),control_phase=np.empty(0,np.uint8),control_mode=np.empty(0,np.uint8))


def build(stage,manifest_path,plan_path,source_root,media=True):
    stage=Path(stage).resolve();out=stage/'report';out.mkdir(parents=True,exist_ok=True)
    manifest=read(manifest_path);plan=read(plan_path);cases=plan['cases'];lookup={c['case_id']:c for c in cases}
    if not manifest.get('closed',manifest.get('collection_closed',False)):raise ValueError('Pilot runtime must be closed before reporting')
    if len(cases)!=48 or Counter(c['group']for c in cases)!={'A_matched':24,'B_fresh':24}:raise ValueError('Frozen48case A/B plan required')
    records=[];old_rows=[];pairs=[];timelines=[];rows_by_id={};source_checks=[];telemetry=[]
    for entry in manifest['entries']:
        case=lookup[entry['case_id']];has_data=bool(entry.get('file'));path=resolve(entry['file']if has_data else entry['metadata_file'],stage)
        if entry.get('sha256')and sha(path)!=entry['sha256']:raise ValueError('Pilot source SHA differs')
        arrays=load_arrays(path)if has_data else empty_arrays(case)
        row,controls=measure(case,entry,arrays,path);row['no_saved_raw_data']=not has_data
        records.append(row);rows_by_id[case['case_id']]=(row,controls,arrays if has_data else None)
        if entry.get('metadata_file'):
            metadata=read(resolve(entry['metadata_file'],stage))
            telemetry.extend(dict(case_id=case['case_id'],group=case['group'],letter=case['letter'],**t)for t in metadata.get('decision_telemetry',[]))
        old_controls=None
        if case['group']=='A_matched':
            base=case['baseline'];old_path=resolve(case.get('baseline_npz',base['file']),R3A)
            if sha(old_path)!=base['sha256']:raise ValueError('Frozen old baseline SHA differs')
            old_arrays=load_arrays(old_path)
            if has_data and not np.array_equal(arrays['states'][0],old_arrays['states'][0]):raise ValueError('Paired initial full state differs')
            old_row,old_controls=measure(case,{},old_arrays,old_path,baseline=True);old_rows.append(old_row)
            row['baseline_trajectory_id']=base['trajectory_id'];pairs.append(paired(old_row,row))
            rows_by_id[case['case_id']]+=(old_row,old_controls,old_arrays)
        name=case['case_id']+'-timeline.svg';(out/name).write_text(timeline_svg(case,controls,old_controls),encoding='utf-8')
        timelines.append(dict(case_id=case['case_id'],group=case['group'],file=name))
        source_checks.append(dict(file=str(path),sha256=sha(path),bytes=path.stat().st_size))
    all_new=aggregate(records)
    if all_new['charged_controls']>2304:raise ValueError('Pilot charged budget exceeded')
    groups={'A_old_baseline':aggregate(old_rows),'A_new_controller':aggregate([r for r in records if r['group']=='A_matched']),
            'B_new_fresh':aggregate([r for r in records if r['group']=='B_fresh']),'all_new_diagnostic_only':all_new}
    letters=[]
    for population,rows in(('A_old_baseline',old_rows),('A_new_controller',[r for r in records if r['group']=='A_matched']),
                            ('B_new_fresh',[r for r in records if r['group']=='B_fresh'])):
        for letter in'TLOCXB':letters.append(dict(population=population,letter=letter,**aggregate([r for r in rows if r['letter']==letter])))
    letter_deltas=[]
    for letter in'TLOCXB':
        old=next(r for r in letters if r['population']=='A_old_baseline'and r['letter']==letter)
        new=next(r for r in letters if r['population']=='A_new_controller'and r['letter']==letter)
        item=dict(letter=letter,old_cases=old['trajectories'],new_cases=new['trajectories'])
        for key in('contact_obvious_fraction_charged','contact_substep_duty_known','moving_pusher_noAO_fraction_charged_upper',
                   'late_contact_obvious_fraction_charged','wallstall_controls','planned_push_noAO_controls'):
            item['old_'+key]=old[key];item['new_'+key]=new[key]
            item['delta_'+key]=None if old[key]is None or new[key]is None else new[key]-old[key]
        for version,q in(('old',old),('new',new)):
            for key in('planned_tasks','reached_tasks','observed','right_censored','not_reached','withdrawal_release_observed'):
                item[version+'_recovery_'+key]=q['recovery'][key]
            for key in('median','p90'):item[version+'_observed_recovery_'+key]=q['recovery']['observed_delay_seconds'][key]
        letter_deltas.append(item)
    result=dict(kind='CONTACT-PILOT1-saved-array-report.v1',closed=True,manifest=dict(file=str(manifest_path),sha256=sha(manifest_path)),
        plan=dict(file=str(plan_path),sha256=sha(plan_path)),definitions_source='../contact-audit-20261030/definitions.json',
        groups=groups,letters=letters,paired_letter_deltas=letter_deltas,attempts=records,paired=pairs,planned_cases=48,attempted_cases=len(records),
        planned_unattempted_case_ids=sorted(set(lookup)-set(rows_by_id)),model_calls=0,checkpoint_loads=0,physics_steps_in_report=0,
        interpretation='A is matched difficult existing TRAIN families, not new independent support. B alone is new families. Both are targeted and small, not unbiased general-population estimates.',
        recovery_definition='First actual known postsolveAO interval start at/after frozen taskstart26, not dynamic controller state. Unobserved and not-reached cases stay separate; withdrawal release evidence disclosed.',
        charged_scope='Six classes cover complete controls; unclassified charged controls remain separate. Primary contact-obvious uses all charges; empty charged upper bound counts unclassified charges as empty.',
        sources=source_checks,timelines=timelines)
    csv_write(out/'cases.csv',scalar_rows(records+old_rows));csv_write(out/'paired-deltas.csv',pairs)
    csv_write(out/'letters.csv',scalar_rows(letters))
    csv_write(out/'paired-letter-deltas.csv',letter_deltas)
    six=[]
    for population,group in groups.items():
        for name,count in group['six_classes'].items():six.append(dict(population=population,outcome=name,count=count,
            complete_denominator=group['completed_controls'],fraction_complete=ratio(count,group['completed_controls']),
            charged_denominator=group['charged_controls'],fraction_charged=ratio(count,group['charged_controls'])))
        six.append(dict(population=population,outcome='unclassified_charged',count=group['unclassified_charged_controls'],
            complete_denominator=group['completed_controls'],fraction_complete=None,charged_denominator=group['charged_controls'],
            fraction_charged=ratio(group['unclassified_charged_controls'],group['charged_controls'])))
    csv_write(out/'six-classes.csv',six)
    csv_write(out/'recontact.csv',[{k:v for k,v in r.items()if k in('case_id','group','letter','version','withdrawal_release_observed')or k.startswith('recontact_')or k=='first_contact_without_verified_release'}for r in records+old_rows])
    csv_write(out/'failures.csv',scalar_rows([r for r in records if r['status']!='completed'or r['completed_controls']!=48]))
    csv_write(out/'controller-telemetry.csv',scalar_rows(telemetry))
    media_result=make_media(out,cases,rows_by_id,source_root)if media else dict(enabled=False)
    result['media']=media_result;write(out/'summary.json',result);write_report(out,result)
    if 'torch'in sys.modules:raise RuntimeError('Report unexpectedly imported Torch')
    report_bytes=sum(p.stat().st_size for p in out.rglob('*')if p.is_file())
    write(out/'inventory.json',dict(report_bytes_before_inventory=report_bytes,files=[dict(file=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(out.rglob('*'))if p.is_file()and p.name!='inventory.json'],model_calls=0,physics_steps=0))
    print(json.dumps(dict(closed=True,charged_controls=all_new['charged_controls'],completed_controls=all_new['completed_controls'],failed_cases=all_new['failed_or_incomplete_trajectories'],report_bytes=report_bytes,
        contact_obvious={k:v['contact_obvious_fraction_charged']for k,v in groups.items()},empty_upper={k:v['moving_pusher_noAO_fraction_charged_upper']for k,v in groups.items()})))
    return result


def make_media(out,cases,rows_by_id,source_root):
    """Predeclared old/new videos from hash-verified saved RGB, no dynamics."""
    import cv2
    from tdyssey.native_rgb_r3a_data import RecordedStateRenderer
    renderer=RecordedStateRenderer(source_root,maximum_frame_cache_bytes=0)
    records=[]
    def render_frames(letter,arrays,version):
        frames=[];proof=[]
        for index,state in enumerate(arrays['states']):
            chw=renderer.render(letter,state)
            expected=bytes(arrays['rgb_frame_sha256'][index]).hex();observed=hashlib.sha256(chw.tobytes()).hexdigest()
            if expected!=observed:raise ValueError('Paired saved RGB differs from actual collection hash')
            proof.append(dict(version=version,frame_index=index,time_seconds=index*.1,collection_sha256=expected,render_sha256=observed))
            frames.append(np.moveaxis(chw,0,-1))
        return frames,proof
    try:
        for case_id in MEDIA_CASES:
            if case_id not in rows_by_id or rows_by_id[case_id][2]is None:
                records.append(dict(case_id=case_id,not_attempted=True));continue
            row,controls,new_arrays,old,old_controls,old_arrays=rows_by_id[case_id]
            new_frames,new_proof=render_frames(row['letter'],new_arrays,'new')
            old_frames,old_proof=render_frames(row['letter'],old_arrays,'old')
            if not np.array_equal(new_frames[0],old_frames[0]):raise ValueError('Matched initial RGB differs')
            path=out/(case_id+'-old-new.mp4');writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10.,(576,288))
            if not writer.isOpened():raise RuntimeError('Installed mp4v encoder unavailable')
            for index in range(max(len(old_frames),len(new_frames))):
                canvas=np.full((288,576,3),248,np.uint8)
                cv2.putText(canvas,f'{case_id} | recorded state {index} | t={index/10:.1f}s',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.46,(25,25,25),1,cv2.LINE_AA)
                for label,frames,data,x in(('OLD baseline',old_frames,old_controls,18),('NEW controller',new_frames,controls,306)):
                    cv2.putText(canvas,label,(x,42),cv2.FONT_HERSHEY_SIMPLEX,.46,(25,25,25),1,cv2.LINE_AA)
                    if index<len(frames):canvas[50:242,x:x+192]=cv2.resize(frames[index],(192,192),interpolation=cv2.INTER_NEAREST)
                    else:cv2.putText(canvas,'NO OBSERVATION',(x,148),cv2.FONT_HERSHEY_SIMPLEX,.4,(40,40,40),1,cv2.LINE_AA)
                    if index==0:label='Initial state; preceding contact N/A'
                    elif index-1<len(data):label=f'control {index-1}: AO {data[index-1]["contact_substeps"]}/10; motion {data[index-1]["motion_class"]}'
                    else:label='Trial stopped; no state extension'
                    cv2.putText(canvas,label,(x,262),cv2.FONT_HERSHEY_SIMPLEX,.35,(25,25,25),1,cv2.LINE_AA)
                writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
            writer.release();cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));cap.release()
            if count!=max(len(old_frames),len(new_frames))or abs(fps-10)>1e-8:raise ValueError('Paired video timing differs')
            # Five actual timepoints; unobserved new states are blank, not held.
            strip=np.full((432,1088,3),255,np.uint8)
            for j,index in enumerate((0,12,24,36,48)):
                x=128+j*192
                cv2.putText(strip,f't={index/10:.1f}s',(x+5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1,cv2.LINE_AA)
                for label,frames,y in(('OLD',old_frames,28),('NEW',new_frames,232)):
                    cv2.putText(strip,label,(10,y+70),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
                    if index<len(frames):strip[y:y+192,x:x+192]=cv2.resize(frames[index],(192,192),interpolation=cv2.INTER_NEAREST)
            strip_path=out/(case_id+'-old-new.png')
            cv2.imwrite(str(strip_path),cv2.cvtColor(strip,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,9])
            records.append(dict(case_id=case_id,old_trajectory_id=row['baseline_trajectory_id'],new_status=row['status'],video=path.name,video_sha256=sha(path),
                video_bytes=path.stat().st_size,video_frames=count,video_fps=fps,old_actual_saved_frames=len(old_frames),new_actual_saved_frames=len(new_frames),
                strip=strip_path.name,strip_sha256=sha(strip_path),strip_bytes=strip_path.stat().st_size,
                verified_saved_RGB_frames=old_proof+new_proof,original_RGB_resized_nearest=True,no_unobserved_state_extension=True))
    finally:renderer.close()
    # All timelines are tiny audit layers; neither video nor labels touch source.
    media_files=[p for p in out.iterdir()if p.suffix in('.svg','.mp4','.png')]
    total=sum(p.stat().st_size for p in media_files)
    if total>2*1024**2:raise RuntimeError('Predeclared media exceeded2MiB; preserve evidence and ask root for required bytes, do not change samples')
    result=dict(enabled=True,fixed_case_ids=list(MEDIA_CASES),records=records,renderer_stats=renderer.stats,
        media_bytes=total,media_cap_bytes=2*1024**2,physics_steps=0,model_calls=0,
        source='Saved native96 RGB; labels outside source pixels; MP4 is a lossy viewing copy, PNG is nearest-upscaled original RGB.')
    write(out/'media-provenance.json',result);return result


def write_report(out,result):
    rows=[]
    for name,q in result['groups'].items():
        rec=q['recovery'];delay=rec['observed_delay_seconds']
        rows.append([name,f"{q['trajectories']} / {q['failed_or_incomplete_trajectories']}",f"{q['completed_controls']} / {q['charged_controls']}",
            pct(q['contact_substep_duty_known']),pct(q['contact_obvious_fraction_charged']),pct(q['moving_pusher_noAO_fraction_charged_upper']),
            pct(q['late_contact_obvious_fraction_charged']),f"{delay['median']} / {delay['p90']}",f"{rec['right_censored']} / {rec['reached_tasks']} (+{rec['not_reached']}not reached)"])
    report='# CONTACT-PILOT1 saved-array results\n\n'+table(['Population','Cases / failed','Complete / charged','AO duty','Contact+obvious / charged','Empty upper / charged','Late contact+obvious','Recovery median / P90 s','Censored'],rows)
    report+='\n\nA is difficult matched existing TRAIN scenes; only B contributes new independent families. Do not pool these as48freshfamilies. Six-class CSV has completed and charged denominators, with unclassified charges explicit. Recovery timing uses the frozen task start, observed postsolve flags and interval-start convention; failed/never-reached tasks are not zero-delay recoveries. Contact+motion is co-occurrence, not strict causal force attribution.\n\n'
    report+=table(['Case','Letter','Role','Old contact+obvious','New contact+obvious','Delta','Old empty','New empty','Old/new wallstall controls'],[
        [p['case_id'],p['letter'],p['role'],pct(p['old_contact_obvious_fraction_charged']),pct(p['new_contact_obvious_fraction_charged']),
         pct(p['delta_contact_obvious_fraction_charged']),pct(p['old_moving_pusher_noAO_fraction_charged_upper']),pct(p['new_moving_pusher_noAO_fraction_charged_upper']),
         f"{p['old_wallstall_controls']} / {p['new_wallstall_controls']}"]for p in result['paired']])
    report+='\n\nEngineering targets are frozen>=40% contact+obvious,<=25% empty upper, observed recovery median<=0.5s/P90<=1s and<=5%unrecovered. Targets are mechanism checks, not population guarantees. Per-letter support and failures must be reviewed alongside them.\n\n'
    report+=table(['Population','Letter','Cases / failures','Contact+obvious','Empty upper','Late contact+obvious','Wallstall controls'],[
        [q['population'],q['letter'],f"{q['trajectories']} / {q['failed_or_incomplete_trajectories']}",pct(q['contact_obvious_fraction_charged']),
         pct(q['moving_pusher_noAO_fraction_charged_upper']),pct(q['late_contact_obvious_fraction_charged']),q['wallstall_controls']]for q in result['letters']])
    report+='\n\n'+table(['Population','Letter','Observed / reached','Censored / not reached','Withdrawal released','Observed median / P90 s'],[
        [q['population'],q['letter'],f"{q['recovery']['observed']} / {q['recovery']['reached_tasks']}",
         f"{q['recovery']['right_censored']} / {q['recovery']['not_reached']}",q['recovery']['withdrawal_release_observed'],
         f"{q['recovery']['observed_delay_seconds']['median']} / {q['recovery']['observed_delay_seconds']['p90']}"]for q in result['letters']])
    report+='\n\nMachine-readable files:summary.json,cases.csv,paired-deltas.csv,paired-letter-deltas.csv,letters.csv,six-classes.csv,recontact.csv,failures.csv. No new physics/model calls were made for reporting. Original R3A data/checkpoints/selected training pool remain unchanged.\n'
    (out/'RESULTS.md').write_text(report,encoding='utf-8')
    cards=[]
    for item in result['media'].get('records',[]):
        if item.get('not_attempted'):
            cards.append(f'<section><h2>{item["case_id"]}</h2><p>Fixed media case was not attempted; no replacement selected.</p></section>');continue
        cards.append(f'<section><h2>{item["case_id"]}: fixed same-scene comparison</h2><p>Old baseline {html.escape(item["old_trajectory_id"])}; new status {html.escape(item["new_status"])}. Real saved states at10Hz; no extension after failure.</p><video controls preload="metadata" src="{item["video"]}" style="max-width:100%"></video><br><img src="{item["strip"]}" style="max-width:100%"></section>')
    for item in result['timelines']:
        cards.append(f'<section><h2>{html.escape(item["case_id"])} ({item["group"]})</h2><img src="{item["file"]}" style="max-width:100%"></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>CONTACT-PILOT1 paired results</title><style>body{font:15px system-ui;margin:24px;background:#eee}section{background:white;padding:16px;margin:16px 0}</style><h1>CONTACT-PILOT1 recorded comparisons</h1><p>Actual postsolve AO counts and measured motion. RGB/video comparisons are fixed before execution; no model prediction or new physics.</p>'+''.join(cards),encoding='utf-8')


def self_test():
    arrays=dict(states=np.zeros((49,15),np.float64),actions=np.zeros((48,2)),attempted_actions=np.zeros((48,2)),substep_status=np.ones(480,np.int8),
        contact_flags=np.zeros((480,3),np.uint8),contact_tags=np.array([[391,0,1,0,4]],np.int16),contact_values=np.zeros((1,11),np.float32),
        contact_arc_offsets=np.array([0,0],np.int32),contact_arc_ids=np.empty(0,np.int16),control_phase=np.r_[np.zeros(18,np.uint8),np.full(8,2,np.uint8),np.full(22,3,np.uint8)],
        control_mode=np.ones(48,np.uint8),substep_states=np.zeros((481,15),np.float64))
    arrays['contact_flags'][391,0]=2;arrays['contact_values'][0,7]=1
    case=dict(case_id='fixture',group='A_matched',letter='T',role='fixture',scenario='release_recontact',slot=dict(scenario='release_recontact'),task_recontact_start_control=26)
    # No file hashing or filesystem side effects needed for the numeric fixture.
    path=Path(__file__);row,_=measure(case,dict(charged_controls=48,status='completed'),arrays,path)
    assert row['recontact_delay_seconds']==1.31 and row['withdrawal_release_observed']
    assert row['contact_substeps']==1 and row['terminal_contact_controls']==0 and row['unknown_impulse_contact_substeps']==0
    assert sum(row['six_classes'].values())==48
    q=aggregate([row]);assert q['recovery']['observed']==1 and q['contact_substep_duty_known']==1/480
    assert q['engineering_targets']['observed_recovery_p90_at_most1s']['passed']is False
    partial=dict(arrays);partial.update(states=arrays['states'][:2],actions=arrays['actions'][:1],attempted_actions=arrays['attempted_actions'][:2],
        substep_status=arrays['substep_status'][:13],contact_flags=arrays['contact_flags'][:13],contact_tags=np.empty((0,5),np.int16),
        contact_values=np.empty((0,11),np.float32),contact_arc_offsets=np.array([0],np.int32),control_phase=arrays['control_phase'][:2],
        control_mode=arrays['control_mode'][:2],substep_states=arrays['substep_states'][:14])
    p,_=measure(case,dict(charged_controls=3,status='failed'),partial,path)
    assert p['partial_substeps']==3 and p['unclassified_charged_controls']==2 and p['reserved_not_started_controls']==1
    assert p['moving_pusher_noAO_fraction_charged_upper']==2/3 and p['recontact_not_reached']
    q=aggregate([p]);assert q['failed_or_incomplete_trajectories']==1 and q['recovery']['unrecovered_or_not_reached_fraction']==1
    return dict(passed=True,checks=['actual postsolve and impulse presence','six complete-control classes','any versus terminal contact','fixed-task recovery and censoring','partial/unknown/charged scope','failed-prefix pessimistic empty bound'],model_calls=0,physics_steps=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,default=STAGE)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--source-root',type=Path,default=ROOT.parent/'Tdyssey/third_party/diffusion_policy')
    parser.add_argument('--no-media',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    build(args.stage,args.manifest or args.stage/'manifest.json',args.plan or args.stage/'plan/cases.json',args.source_root,not args.no_media)


if __name__=='__main__':main()
