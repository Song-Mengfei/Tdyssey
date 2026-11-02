"""Cold CONTACT-PILOT2 fixed-queue, strict-recovery and requested-arc report.

No controller, simulator or model is executed. Original CP1 measurement helpers
are imported unchanged; strict recovery, planned slots and queue failures are
computed here without treating unexecuted cases as observed static data.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.native_rgb_contact_pilot_report import (measure as old_measure,empty_arrays,load_arrays,
    read,write,sha,resolve,ratio,pct,quantile,csv_write,scalar_rows,CLASSES,table)
from tdyssey.native_rgb_contact_audit import actual_signals,boolean_runs,motion,classify_motion
from tdyssey.native_rgb_r3a_collection import segments

STAGE=ROOT/'artifacts/native-rgb-supervised-20261029/contact-pilot2-20261030'
GROUPS=('CP1_A_repeat','CP1_B_repeat','CP2_fresh')
FIXED_MEDIA=('CP1-A-X-03','CP1-A-T-03','CP1-A-B-04')
EPSILON=1e-8
NUMERIC=('charged_controls','completed_controls','saved_attempted_controls','known_substeps','unknown_advance_substeps',
    'partial_substeps','reserved_not_started_controls','unclassified_charged_controls','any_contact_controls',
    'terminal_contact_controls','contact_substeps','object_wall_substeps','agent_wall_substeps','unknown_impulse_contact_substeps',
    'positive_impulse_substeps','contact_obvious_controls','contact_obvious_impulse_controls','contact_obvious_atleast5_controls',
    'contact_obvious_all10_controls','contact_obvious_impulse_atleast5_controls','nearstatic_controls','moving_pusher_noAO_controls',
    'moving_pusher_noAO_objectnearstatic_controls','route_recovery_noAO_controls','planned_push_controls','planned_push_noAO_controls',
    'wallstall_controls','simultaneous_AO_OW_nearstatic_substeps','late_completed_controls','late_charged_controls',
    'late_contact_obvious_controls','early_completed_controls','early_contact_obvious_controls','withdraw_completed_controls',
    'withdraw_noAO_controls','hold_complete_controls','unconfirmed_route_commands','wall_risk_trigger_commands')


def json_csv(path,rows):
    """Keep CSV cells scalar; nested evidence remains in JSON/gzip sources."""
    csv_write(path,[{k:(json.dumps(v,separators=(',',':'))if isinstance(v,(list,dict))else v)for k,v in row.items()}for row in rows])


def compress_owned_csv(path,out):
    """Keep only a hash-verified lossless copy of this script's new CSV output."""
    path=Path(path).resolve();out=Path(out).resolve()
    if not path.is_relative_to(out)or path.suffix!='.csv':raise ValueError('Only owned report CSVs may be compressed')
    if not path.exists():return None
    raw=path.read_bytes();compressed=gzip.compress(raw,mtime=0);target=path.with_suffix('.csv.gz')
    target.write_bytes(compressed)
    if gzip.decompress(target.read_bytes())!=raw:raise ValueError('CSV lossless compression check failed')
    proof=dict(file=target.name,uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
        compressed_bytes=len(compressed),compressed_sha256=sha(target))
    path.unlink()
    return proof


def strict_recovery(case,arrays):
    n=len(arrays['actions']);status=np.asarray(arrays['substep_status']);known=status==1
    signals=actual_signals(arrays);ao=signals['ao'];phase=arrays['control_phase'][:n]
    full_known=known[:n*10].reshape(n,10).all(axis=1)if n else np.empty(0,bool)
    contact=ao[:n*10].reshape(n,10).any(axis=1)if n else np.empty(0,bool)
    release_controls=np.flatnonzero((phase==2)&full_known&~contact)
    released=bool(len(release_controls));start=case.get('task_recontact_start_control',26)
    start_sub=None if start is None else int(start)*10
    observed_indices=np.flatnonzero(known[start_sub:])+start_sub if start_sub is not None and len(known)>start_sub else np.empty(0,int)
    reached=bool(len(observed_indices));all_after=np.flatnonzero(ao[start_sub:]&known[start_sub:])+start_sub if reached else np.empty(0,int)
    candidate_first=int(all_after[0])if len(all_after)else None
    first=candidate_first if released else None
    withdraw_start=int(case.get('task_withdraw_start_control',18 if case.get('scenario')=='release_recontact'else 22))
    prior_AO=bool(np.any(ao[:withdraw_start*10]&known[:withdraw_start*10]))
    censor_window=None if not reached else float((int(observed_indices[-1])+1-start_sub)*.01)
    never_separated=bool(reached and not released)
    right_censored=bool(reached and released and first is None)
    early_return=False
    if released and start_sub is not None:
        after_release=(int(release_controls[0])+1)*10
        early_return=bool(np.any(ao[after_release:start_sub]&known[after_release:start_sub]))
    return dict(withdrawal_release_observed=released,verified_release_control_indices=release_controls.tolist(),
        withdrawal_known_full_noAO_controls=len(release_controls),recontact_task_start_control=start,
        recontact_reached=reached,recontact_observed=first is not None,recontact_first_substep=first,
        recontact_delay_seconds=None if first is None else float((first-start_sub)*.01),
        recontact_right_censored=right_censored,recontact_censor_seconds=censor_window if right_censored else None,
        recontact_not_reached=bool(start is not None and not reached),recontact_never_separated=never_separated,
        first_contact_without_verified_release=bool(candidate_first is not None and not released),
        first_any_AO_after_task_substep=candidate_first,contact_returned_during_withdraw=early_return,
        prior_contact_before_withdraw=prior_AO,recovery_without_prior_contact=bool(first is not None and not prior_AO),
        separated_recovery_with_prior_contact=bool(first is not None and prior_AO),
        strict_recovery_definition='At least one complete ten-known-substep no-AO control during fixed withdrawal,then first actual known postsolve AO at/after fixed task26; never-separated is not zero-delay success.')


def requested_arc(case,arrays,*,version,status):
    requested=case.get('initial_requested_arc',{})
    target=int(requested['arc_id']if isinstance(requested,dict)and'arc_id'in requested else
               requested if isinstance(requested,(int,np.integer))else case['candidate']['selected_contact']['arc_id'])
    signals=actual_signals(arrays);m=len(signals['ao']);known=np.asarray(arrays['substep_status'])==1
    hit=np.asarray([target in arcs for arcs in signals['arcs']],bool)&signals['ao']&known
    valid=np.zeros(m,bool);positive=np.zeros(m,bool)
    tags=arrays['contact_tags'];values=arrays['contact_values'];offsets=arrays['contact_arc_offsets'];ids=arrays['contact_arc_ids']
    for row in range(len(tags)):
        si,kind,phase,_,mask=map(int,tags[row])
        if kind!=0 or phase!=1 or not 0<=si<m or not known[si]or target not in ids[offsets[row]:offsets[row+1]]:continue
        vector=values[row,7:9].astype(np.float64)
        if mask&4 and np.isfinite(vector).all():
            valid[si]=True
            if np.linalg.norm(vector)>EPSILON:positive[si]=True
    n=len(arrays['actions']);t,r,_=motion(arrays['states'],case['letter']);classes,_=classify_motion(t,r)
    counts=hit[:n*10].reshape(n,10).sum(axis=1)if n else np.empty(0,int)
    poscounts=positive[:n*10].reshape(n,10).sum(axis=1)if n else np.empty(0,int)
    hit_controls=counts>0;positive_controls=poscounts>0
    first=int(np.flatnonzero(hit)[0])if hit.any()else None
    first_any=int(np.flatnonzero(signals['ao']&known)[0])if(signals['ao']&known).any()else None
    return dict(case_id=case['case_id'],source_cp1_case_id=case.get('source_cp1_case_id'),group=case['group'],letter=case['letter'],
        version=version,status=status,initial_requested_arc_id=target,requested_arc_hit_observed=bool(hit.any()),
        first_requested_arc_substep=first,first_any_AO_substep=first_any,
        first_contact_arc_ids=[]if first_any is None else sorted(signals['arcs'][first_any]),
        requested_arc_contact_substeps=int(hit.sum()),requested_arc_valid_impulse_substeps=int((hit&valid).sum()),
        requested_arc_positive_impulse_substeps=int((hit&positive).sum()),
        requested_arc_unknown_impulse_substeps=int((hit&~valid).sum()),
        requested_arc_contact_complete_controls=int(hit_controls.sum()),
        requested_arc_positive_impulse_complete_controls=int(positive_controls.sum()),
        requested_arc_contact_and_obvious_complete_controls=int((hit_controls&(classes==2)).sum()),
        requested_arc_positive_impulse_and_obvious_complete_controls=int((positive_controls&(classes==2)).sum()),
        requested_arc_atleast5_positive_substeps_and_obvious_controls=int(((poscounts>=5)&(classes==2)).sum()),
        requested_arc_positive_impulse_run_seconds=quantile([(b-a)*.01 for a,b in boolean_runs(hit&positive)]),
        requested_arc_max_positive_impulse_run_seconds=max(((b-a)*.01 for a,b in boolean_runs(hit&positive)),default=0.),
        hit_control_motion=[dict(control=int(i),contact_substeps=int(counts[i]),positive_impulse_substeps=int(poscounts[i]),
            translation_image96_px=float(t[i]),signed_rotation_deg=float(r[i]),motion_class=int(classes[i]))for i in np.flatnonzero(hit_controls)],
        incidental_AO_without_requested_arc_substeps=int((signals['ao']&known&~hit).sum()),
        actual_AO_arc_ids=sorted(set(arc for arcs in signals['arcs']for arc in arcs)),
        interpretation='Exact requested-arc membership of actual postsolve point rows; adjacent or other arcs are not substitutes. Point/event impulses are existence tests,never sums. Motion is co-occurrence in the containing complete control,not isolated causation.',
        no_observed_opportunity=status=='not_executed')


def telemetry(stage,entry,metadata):
    values=metadata.get('decision_telemetry',[]);source=None
    if metadata.get('decision_telemetry_file'):
        source=resolve(metadata['decision_telemetry_file'],stage)
        expected=metadata.get('decision_telemetry_sha256',metadata.get('telemetry_sha256'))
        if expected and sha(source)!=expected:raise ValueError('CP2 telemetry SHA differs')
        with gzip.open(source,'rt',encoding='utf-8')as stream:payload=json.load(stream)
        values=payload['decisions']if isinstance(payload,dict)else payload
    return values,None if source is None else dict(file=str(source),sha256=sha(source))


def measure_case(case,entry,arrays,path,*,version,decisions=(),has_data=True):
    row,controls=old_measure(case,entry,arrays,path,baseline=False)
    row.update(version=version,source_cp1_case_id=case.get('source_cp1_case_id'),family_id=case['family_id'],
        new_independent_family=bool(case.get('new_independent_family',case['group']=='CP2_fresh')),
        planned_control_slots=48,no_saved_raw_data=not has_data)
    if not has_data:row.update(source_file=None,source_sha256=None)
    row.update(strict_recovery(case,arrays))
    modes=arrays['control_mode'][:len(arrays['actions'])]
    row['hold_complete_controls']=int(np.count_nonzero(modes==0))
    row['longest_hold_complete_controls']=max((b-a for a,b in boolean_runs(modes==0)),default=0)
    states=[str(d.get('controller_state',''))for d in decisions]
    unconfirmed=np.asarray([state=='unconfirmed_hold'or'unconfirmed_route'in state or int(d.get('unconfirmed_route_controls',0)or 0)>0
        for state,d in zip(states,decisions)],bool)
    row['unconfirmed_route_commands']=int(unconfirmed.sum())
    row['longest_unconfirmed_route_commands']=max((b-a for a,b in boolean_runs(unconfirmed)),default=0)
    risk_flags=[]
    for decision in decisions:
        risk=decision.get('wall_risk',decision.get('safety_risk',decision.get('risk',{})))
        risk_flags.append(bool(risk.get('risky',False))if isinstance(risk,dict)else False)
    row['wall_risk_trigger_commands']=sum(risk_flags)
    return row,controls


def aggregate(rows):
    out={key:sum(int(r.get(key,0))for r in rows)for key in NUMERIC}
    n=out['completed_controls'];charged=out['charged_controls'];slots=len(rows)*48
    attempted=[r for r in rows if r['status']!='not_executed'];tasks=[r for r in rows if r['recontact_task_start_control']is not None]
    observed=[r for r in tasks if r['recontact_observed']];reached=[r for r in tasks if r['recontact_reached']]
    censored=[r for r in tasks if r['recontact_right_censored']]
    failed=[r for r in attempted if r['status']!='completed'or r['completed_controls']!=48]
    guard=[r for r in attempted if r.get('physical_guard_failure')]
    complete_queue=len(attempted)==len(rows)and bool(rows)
    out.update(planned_trajectories=len(rows),attempted_trajectories=len(attempted),unattempted_trajectories=len(rows)-len(attempted),
        successful_48control_trajectories=sum(r['status']=='completed'and r['completed_controls']==48 for r in attempted),
        failed_or_incomplete_attempted_trajectories=len(failed),failure_fraction_attempted=ratio(len(failed),len(attempted)),
        physical_guard_failures=len(guard),complete_fixed_queue_assessed=complete_queue,
        six_classes={key:sum(r['six_classes'][key]for r in rows)for key in CLASSES},
        planned_control_slots=slots,unexecuted_planned_control_slots=slots-charged,unclassified_planned_slots=slots-n,
        any_contact_fraction_complete=ratio(out['any_contact_controls'],n),contact_substep_duty_known=ratio(out['contact_substeps'],out['known_substeps']),
        contact_obvious_fraction_complete=ratio(out['contact_obvious_controls'],n),contact_obvious_fraction_charged=ratio(out['contact_obvious_controls'],charged),
        contact_obvious_fraction_planned_lower=ratio(out['contact_obvious_controls'],slots),
        moving_pusher_noAO_fraction_complete=ratio(out['moving_pusher_noAO_controls'],n),
        moving_pusher_noAO_fraction_charged_upper=ratio(out['moving_pusher_noAO_controls']+charged-n,charged),
        moving_pusher_noAO_fraction_planned_upper=ratio(out['moving_pusher_noAO_controls']+slots-n,slots),
        nearstatic_fraction_complete=ratio(out['nearstatic_controls'],n),
        late_contact_obvious_fraction_charged=ratio(out['late_contact_obvious_controls'],out['late_charged_controls']),
        late_contact_obvious_fraction_complete=ratio(out['late_contact_obvious_controls'],out['late_completed_controls']),
        late_contact_obvious_fraction_planned_lower=ratio(out['late_contact_obvious_controls'],len(rows)*24),
        planned_push_noAO_fraction=ratio(out['planned_push_noAO_controls'],out['planned_push_controls']),
        longest_hold_complete_controls=max((r['longest_hold_complete_controls']for r in rows),default=0),
        longest_unconfirmed_route_commands=max((r['longest_unconfirmed_route_commands']for r in rows),default=0),
        contact_run_seconds=quantile([v*.01 for r in rows for v in r['contact_run_substeps']]),
        positive_impulse_run_seconds=quantile([v*.01 for r in rows for v in r['positive_impulse_run_substeps']]),
        recovery=dict(planned_tasks=len(tasks),reached_tasks=len(reached),observed=len(observed),right_censored=len(censored),
            never_separated=sum(r['recontact_never_separated']for r in tasks),not_reached=sum(r['recontact_not_reached']for r in tasks),
            verified_withdrawal_release=sum(r['withdrawal_release_observed']for r in tasks),
            observed_with_prior_AO=sum(r['separated_recovery_with_prior_contact']for r in tasks),
            first_acquisition_without_prior_AO=sum(r['recovery_without_prior_contact']for r in tasks),
            observed_delay_with_prior_AO_seconds=quantile([r['recontact_delay_seconds']for r in observed if r['prior_contact_before_withdraw']]),
            unrecovered_fraction_all_planned=ratio(len(tasks)-len(observed),len(tasks)),
            observed_delay_seconds=quantile([r['recontact_delay_seconds']for r in observed]),
            censor_window_seconds=quantile([r['recontact_censor_seconds']for r in censored])),
        outer_coverage_by_letter={letter:dict(actual_arc_ids=sorted(set(a for r in rows if r['letter']==letter for a in r['actual_outer_arc_ids'])),possible_arc_count=len(segments(letter)))for letter in'TLOCXB'})
    recovery=out['recovery'];delay=recovery['observed_delay_seconds']
    values={
        'contact_obvious_at_least40pct':(out['contact_obvious_fraction_charged'],lambda x:x>=.4),
        'moving_pusher_noAO_at_most25pct':(out['moving_pusher_noAO_fraction_charged_upper'],lambda x:x<=.25),
        'observed_recovery_median_at_most0_5s':(delay['median'],lambda x:x<=.5),
        'observed_recovery_p90_at_most1s':(delay['p90'],lambda x:x<=1.),
        'unrecovered_all_planned_at_most5pct':(recovery['unrecovered_fraction_all_planned'],lambda x:x<=.05),
        'physical_guard_failures_zero':(len(guard),lambda x:x==0)}
    out['engineering_targets']={key:dict(value=value,observed_threshold_met=value is not None and bool(predicate(value)),
        complete_queue_assessed=complete_queue,passed=complete_queue and value is not None and bool(predicate(value)))for key,(value,predicate)in values.items()}
    out['all_engineering_targets_passed']=all(q['passed']for q in out['engineering_targets'].values())and out['unknown_advance_substeps']==0
    return out


def comparison(old,new):
    keys=('contact_obvious_fraction_complete','contact_obvious_fraction_charged','contact_obvious_fraction_planned_lower',
        'contact_substep_duty_known','moving_pusher_noAO_fraction_charged_upper','moving_pusher_noAO_fraction_planned_upper',
        'late_contact_obvious_fraction_charged','wallstall_controls','planned_push_noAO_controls','longest_hold_complete_controls',
        'unconfirmed_route_commands','contact_obvious_impulse_atleast5_controls')
    out=dict(case_id=new['case_id'],source_cp1_case_id=new.get('source_cp1_case_id'),group=new['group'],letter=new['letter'],
        baseline_version=old['version'],new_version=new['version'],baseline_status=old['status'],new_status=new['status'],
        comparison_observed=new['status']!='not_executed')
    for key in keys:
        # Case-level slot sensitivity uses its fixed48 slots, including prefixes.
        def value(row):
            if key=='contact_obvious_fraction_planned_lower':return row['contact_obvious_controls']/48
            if key=='moving_pusher_noAO_fraction_planned_upper':return(row['moving_pusher_noAO_controls']+48-row['completed_controls'])/48
            return row.get(key)
        left,right=value(old),value(new);out['old_'+key]=left;out['new_'+key]=right
        out['delta_'+key]=None if left is None or right is None or not out['comparison_observed']else right-left
    for prefix,row in(('old',old),('new',new)):
        out[prefix+'_strict_recontact_observed']=row['recontact_observed']
        out[prefix+'_recontact_delay_seconds']=row['recontact_delay_seconds']
        out[prefix+'_never_separated']=row['recontact_never_separated']
    return out


def compact_telemetry(case,decisions,source):
    rows=[];risks=[]
    for index,decision in enumerate(decisions):
        row=dict(case_id=case['case_id'],source_cp1_case_id=case.get('source_cp1_case_id'),group=case['group'],letter=case['letter'],
            control=int(decision.get('control',index)),controller_state=decision.get('controller_state'),
            replan_reason=decision.get('replan_reason'),target_revision=decision.get('target_revision'),
            requested_arc=decision.get('initial_requested_arc',decision.get('requested_arc_id')),
            controller_requested_initial_established=decision.get('requested_initial_established'),
            fallback_reason=decision.get('requested_fallback_reason',decision.get('requested_arc_fallback_reason',decision.get('fallback_reason'))),
            route_remaining_world_px=decision.get('route_remaining_world_px'),route_eta_controls=decision.get('route_eta_controls'),
            no_progress_controls=decision.get('no_progress_controls'),unconfirmed_route_controls=decision.get('unconfirmed_route_controls'),
            source_telemetry=None if source is None else source['file'])
        rows.append(row)
        for field,value in decision.items():
            if not isinstance(value,dict):continue
            candidates=[(field,value)]if'wall_risks'in value else[(field+':'+key,nested)for key,nested in value.items()if isinstance(nested,dict)and'wall_risks'in nested]
            for label,risk in candidates:
                risks.append(dict(case_id=case['case_id'],control=row['control'],risk_field=label,
                    risky=risk.get('risky'),state_risky=risk.get('state_risky'),pushing_command=risk.get('pushing_command'),
                    hazardous_walls=risk.get('hazardous_walls'),risk_score=risk.get('risk_score'),
                    original_candidate_action=risk.get('original_candidate_action'),final_action=decision.get('final_action',decision.get('action')),
                    risk_contact_source=risk.get('assessment_contact_source',decision.get('risk_contact_source')),
                    risk_contact_pair=risk.get('assessment_contact_pair'),
                    top_gap_world_px=risk.get('wall_risks',{}).get('top',{}).get('gap_world_px'),
                    top_remaining_room_world_px=risk.get('wall_risks',{}).get('top',{}).get('projected_remaining_room_world_px'),
                    source_telemetry=None if source is None else source['file']))
    return rows,risks


def build(stage,manifest_path,plan_path,source_root,*,media=True):
    stage=Path(stage).resolve();out=stage/'report';out.mkdir(parents=True,exist_ok=True)
    manifest=read(manifest_path);plan=read(plan_path);cases=plan['cases']
    if not manifest.get('closed',manifest.get('collection_closed',False)):raise ValueError('CP2 runtime must close before reporting')
    if len(cases)!=60 or Counter(c['group']for c in cases)!=dict(CP1_A_repeat=24,CP1_B_repeat=24,CP2_fresh=12):
        raise ValueError('CP2 fixed60-case catalogue differs')
    entries={row['case_id']:row for row in manifest['entries']}
    if len(entries)!=len(manifest['entries']):raise ValueError('Repeated CP2 execution case')
    arrays_cache={};all_rows=[];new_rows=[];paired=[];timeline=[];media_cases={};weak=[];telemetry_rows=[];risk_rows=[];sources={}
    def arrays_for(path,expected=None):
        path=Path(path).resolve();key=str(path)
        if key not in arrays_cache:
            if expected and sha(path)!=expected:raise ValueError('Frozen saved-source SHA differs: '+key)
            arrays_cache[key]=load_arrays(path);sources[key]=dict(file=key,sha256=sha(path),bytes=path.stat().st_size)
        return arrays_cache[key]
    for case in cases:
        entry=entries.get(case['case_id'])
        version_rows={};version_controls={};version_arrays={}
        for version,key in(('R3A','r3a_baseline'),('CP1','cp1_baseline')):
            reference=case.get(key)
            if not reference:continue
            path=resolve(reference['file'],stage.parent/('r3a'if version=='R3A'else'contact-pilot1-20261030'))
            arrays=arrays_for(path,reference.get('sha256'))
            metadata_path=resolve(reference.get('metadata_file',str(path.with_suffix('.json'))),path.parent)
            if reference.get('metadata_sha256')and sha(metadata_path)!=reference['metadata_sha256']:
                raise ValueError('Frozen baseline metadata SHA differs')
            metadata=read(metadata_path)
            reference_entry=dict(status=reference.get('status',metadata.get('status','completed')),
                charged_controls=int(reference.get('charged_controls',metadata.get('charged_controls',len(arrays['attempted_actions'])))),
                reserved_not_started_controls=int(metadata.get('reserved_not_started_controls',0)))
            decisions=metadata.get('decision_telemetry',[])
            row,controls=measure_case(case,reference_entry,arrays,path,version=version,decisions=decisions)
            row['physical_guard_failure']=reference_entry['status']=='failed_physics'
            row['source_trajectory_id']=reference.get('trajectory_id');all_rows.append(row)
            version_rows[version]=row;version_controls[version]=controls;version_arrays[version]=arrays
        has_data=entry is not None and bool(entry.get('file'))
        if entry is None:
            effective_entry=dict(status='not_executed',charged_controls=0,reserved_not_started_controls=0)
            arrays=empty_arrays(case);path=Path(plan_path);metadata={};decisions=[];telemetry_source=None
        else:
            effective_entry=entry
            path=resolve(entry['file']if has_data else entry['metadata_file'],stage)
            arrays=arrays_for(path,entry.get('sha256'))if has_data else empty_arrays(case)
            metadata_path=resolve(entry['metadata_file'],stage);metadata=read(metadata_path)
            if entry.get('metadata_sha256')and sha(metadata_path)!=entry['metadata_sha256']:raise ValueError('New metadata SHA differs')
            decisions,telemetry_source=telemetry(stage,entry,metadata)
            rows,risk=compact_telemetry(case,decisions,telemetry_source);telemetry_rows.extend(rows);risk_rows.extend(risk)
        row,controls=measure_case(case,effective_entry,arrays,path,version='CP2',decisions=decisions,has_data=has_data)
        row['physical_guard_failure']=effective_entry['status']=='failed_physics'or'penetration exceeds'in str(metadata.get('failure',{}))
        row['source_trajectory_id']=case['case_id'];row['telemetry_source']=telemetry_source
        all_rows.append(row);new_rows.append(row);version_rows['CP2']=row;version_controls['CP2']=controls
        version_arrays['CP2']=arrays if has_data else None
        if has_data and case.get('cp1_baseline'):
            baseline_arrays=version_arrays['CP1']
            if not np.array_equal(arrays['states'][0],baseline_arrays['states'][0]):raise ValueError('Repeat initial full state differs fromCP1')
            if not np.array_equal(arrays['rgb_frame_sha256'][0],baseline_arrays['rgb_frame_sha256'][0]):raise ValueError('Repeat initial original RGB differs')
        for version in('R3A','CP1'):
            if version in version_rows:paired.append(comparison(version_rows[version],row))
        is_weak=(case['group']=='CP1_B_repeat'and str(case.get('source_cp1_case_id','')).startswith('CP1-B-')and str(case.get('source_cp1_case_id')).endswith('-04'))or(case['group']=='CP2_fresh'and'weak'in str(case.get('role','')))
        if is_weak:
            for version in('CP1','CP2'):
                if version in version_arrays:
                    candidate_arrays=version_arrays[version]if version_arrays[version]is not None else empty_arrays(case)
                    weak.append(requested_arc(case,candidate_arrays,version=version,status=version_rows[version]['status']))
        timeline.append(dict(case_id=case['case_id'],source_cp1_case_id=case.get('source_cp1_case_id'),group=case['group'],letter=case['letter'],
            status=row['status'],versions={version:dict(status=version_rows[version]['status'],controls=[[
                control['contact_substeps'],control['motion_class'],control['phase'],control['mode'],
                round(control['translation_image96_px'],6),round(control['signed_rotation_deg'],6)]for control in data])for version,data in version_controls.items()}))
        if case.get('source_cp1_case_id')in FIXED_MEDIA:media_cases[case['source_cp1_case_id']]=(case,version_rows,version_arrays)
    all_new=aggregate(new_rows)
    if all_new['charged_controls']>2880:raise ValueError('CP2 charged budget exceeded')
    populations={
        'R3A_original_A': [r for r in all_rows if r['version']=='R3A'],
        'CP1_original_A': [r for r in all_rows if r['version']=='CP1'and r['group']=='CP1_A_repeat'],
        'CP2_repeat_A': [r for r in new_rows if r['group']=='CP1_A_repeat'],
        'CP1_original_B': [r for r in all_rows if r['version']=='CP1'and r['group']=='CP1_B_repeat'],
        'CP2_repeat_B': [r for r in new_rows if r['group']=='CP1_B_repeat'],
        'CP2_new_families': [r for r in new_rows if r['group']=='CP2_fresh'],
        'CP2_all_planned': new_rows,
        'CP2_attempted_prefix': [r for r in new_rows if r['status']!='not_executed']}
    groups={key:aggregate(rows)for key,rows in populations.items()}
    letters=[dict(population=key,letter=letter,**aggregate([r for r in rows if r['letter']==letter]))
        for key,rows in populations.items()for letter in'TLOCXB']
    letter_comparisons=[]
    for old_key,new_key in(('R3A_original_A','CP2_repeat_A'),('CP1_original_A','CP2_repeat_A'),('CP1_original_B','CP2_repeat_B')):
        for letter in'TLOCXB':
            old=next(q for q in letters if q['population']==old_key and q['letter']==letter)
            new=next(q for q in letters if q['population']==new_key and q['letter']==letter)
            item=dict(letter=letter,baseline_population=old_key,new_population=new_key,new_attempted=new['attempted_trajectories'],new_planned=new['planned_trajectories'])
            for key in('contact_obvious_fraction_charged','contact_obvious_fraction_planned_lower','moving_pusher_noAO_fraction_charged_upper',
                        'moving_pusher_noAO_fraction_planned_upper','late_contact_obvious_fraction_charged','wallstall_controls',
                        'planned_push_noAO_controls','longest_hold_complete_controls','unconfirmed_route_commands'):
                item['old_'+key]=old[key];item['new_'+key]=new[key]
                item['delta_'+key]=None if not new['attempted_trajectories']or old[key]is None or new[key]is None else new[key]-old[key]
            letter_comparisons.append(item)
    result=dict(kind='CONTACT-PILOT2-fixed-queue-saved-array-report.v1',closed=True,
        manifest=dict(file=str(Path(manifest_path).resolve()),sha256=sha(manifest_path)),plan=dict(file=str(Path(plan_path).resolve()),sha256=sha(plan_path)),
        planned_cases=60,attempted_cases=len(entries),unattempted_cases=60-len(entries),groups=groups,letters=letters,
        attempts=new_rows,all_version_rows=all_rows,paired=paired,paired_letter_deltas=letter_comparisons,
        planned_unattempted_case_ids=[r['case_id']for r in new_rows if r['status']=='not_executed'],
        weak_requested_arc_rows=weak,telemetry_index_rows=len(telemetry_rows),risk_index_rows=len(risk_rows),sources=list(sources.values()),
        telemetry_index_file='controller-telemetry-index.csv.gz',risk_index_file='risk-index.csv.gz',
        interpretation='Original48are repeated scenes,not new independent support. Only planned12CP2fresh can add independent families if physically attempted. Fresh cases are targeted and small,not natural-distribution evidence.',
        strict_recovery='Verified complete10known/noAO interval during fixedwithdraw,then first realpostsolve at/after26. Never-separated does not produce zero-delay success. Notreached,right-censored,andnever-separated remain disjoint.',
        denominator_note='Actual completecontrol sixclasses; allcharges include partial/unknown/reserved; planned48slots pessimistically give unobservedslots zero movement and empty credit. Plannedslots are not acquired physicaltime.',
        motion_definitions=dict(nearstatic='area-centroid translation<.1image96px AND abs unwrapped rotation<.1degree',
            obvious='area-centroid translation>=2image96px OR abs unwrapped rotation>=5degrees',small='remaining controls',
            actual_contact='known100Hz agent-object post_solve flags,not planned intent',impulse='per-substep valid point-row presence norm>1e-8,never summed point/event impulses'),
        physical_stop_rule='Anynewphysicalguard/unknownadvance/systemfailurestopsremainingfixedqueue. Unattemptedcases are N/A,not static observations.',
        model_calls=0,checkpoint_loads=0,physics_steps_in_report=0,new_training_pool_members=0)
    case_csv=scalar_rows(all_rows)
    for output,row in zip(case_csv,all_rows):output.update({'six_'+key:value for key,value in row['six_classes'].items()})
    csv_write(out/'cases.csv',case_csv);json_csv(out/'paired-deltas.csv',paired)
    csv_write(out/'letters.csv',scalar_rows(letters));json_csv(out/'paired-letter-deltas.csv',letter_comparisons)
    six=[]
    for population,q in groups.items():
        for key,count in q['six_classes'].items():
            six.append(dict(population=population,outcome=key,count=count,complete_denominator=q['completed_controls'],
                fraction_complete=ratio(count,q['completed_controls']),charged_denominator=q['charged_controls'],
                fraction_charged=ratio(count,q['charged_controls']),planned_slot_denominator=q['planned_control_slots'],fraction_planned_slots=ratio(count,q['planned_control_slots'])))
        six.extend([dict(population=population,outcome='unclassified_charged',count=q['unclassified_charged_controls'],charged_denominator=q['charged_controls'],fraction_charged=ratio(q['unclassified_charged_controls'],q['charged_controls'])),
            dict(population=population,outcome='unclassified_planned_slot',count=q['unclassified_planned_slots'],planned_slot_denominator=q['planned_control_slots'],fraction_planned_slots=ratio(q['unclassified_planned_slots'],q['planned_control_slots']))])
    csv_write(out/'six-classes.csv',six)
    csv_write(out/'recontact.csv',scalar_rows([{k:v for k,v in row.items()if k in('case_id','source_cp1_case_id','group','letter','version','status','withdrawal_release_observed','withdrawal_known_full_noAO_controls','first_contact_without_verified_release','contact_returned_during_withdraw','prior_contact_before_withdraw','recovery_without_prior_contact','separated_recovery_with_prior_contact')or k.startswith('recontact_')}for row in all_rows]))
    json_csv(out/'requested-arcs.csv',[{k:v for k,v in row.items()if k!='hit_control_motion'}for row in weak])
    csv_write(out/'failures.csv',scalar_rows([r for r in new_rows if r['status']not in('completed','not_executed')]))
    csv_write(out/'controller-telemetry-index.csv',telemetry_rows);json_csv(out/'risk-index.csv',risk_rows)
    compression=[compress_owned_csv(out/name,out)for name in('controller-telemetry-index.csv','risk-index.csv')]
    write(out/'csv-compression-proof.json',dict(files=[proof for proof in compression if proof is not None],lossless=True,original_physics_data_untouched=True))
    (out/'timelines.json').write_text(json.dumps(timeline,separators=(',',':'),allow_nan=False),encoding='utf-8')
    result['media']=make_media(out,media_cases,source_root)if media else dict(enabled=False,records=[])
    write(out/'summary.json',result);write_report(out,result,timeline)
    if'torch'in sys.modules:raise RuntimeError('Cold reporter unexpectedly importedTorch')
    files=[p for p in out.rglob('*')if p.is_file()and p.name!='inventory.json']
    write(out/'inventory.json',dict(files=[dict(file=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p))for p in files],
        report_bytes_before_inventory=sum(p.stat().st_size for p in files),model_calls=0,physics_steps=0))
    print(json.dumps(dict(closed=True,planned=60,attempted=len(entries),unattempted=60-len(entries),
        charged_controls=all_new['charged_controls'],known_substeps=all_new['known_substeps'],
        full_queue_assessed=all_new['complete_fixed_queue_assessed'],guard_failures=all_new['physical_guard_failures'],
        motion_fraction_charged=all_new['contact_obvious_fraction_charged'],motion_fraction_planned_lower=all_new['contact_obvious_fraction_planned_lower'],
        report_bytes=sum(p.stat().st_size for p in out.rglob('*')if p.is_file()))))
    return result


def make_media(out,selected,source_root):
    """Fixed real-length version clips; no planned or repeated terminal frames."""
    import cv2
    from tdyssey.native_rgb_r3a_data import RecordedStateRenderer
    renderer=RecordedStateRenderer(source_root,maximum_frame_cache_bytes=0)
    records=[];generated=[];proofs=[]
    def relative(path):return Path(os.path.relpath(Path(path).resolve(),out.resolve())).as_posix()
    try:
        for source_id in FIXED_MEDIA:
            if source_id not in selected:raise ValueError('A fixed media case vanished from the plan')
            case,rows,archives=selected[source_id];record=dict(case_id=case['case_id'],source_cp1_case_id=source_id,versions=[])
            frame_sets={}
            for version in('R3A','CP1','CP2'):
                arrays=archives.get(version);row=rows.get(version)
                if arrays is None or row is None:
                    record['versions'].append(dict(version=version,status='not_executed',frames=0,video=None,known_state_available=False))
                    frame_sets[version]=[];continue
                frames=[];checks=[]
                for index,state in enumerate(arrays['states']):
                    chw=renderer.render(case['letter'],state)
                    expected=bytes(arrays['rgb_frame_sha256'][index]).hex();observed=hashlib.sha256(chw.tobytes()).hexdigest()
                    if expected!=observed:raise ValueError('Saved version RGB does not match collection hash')
                    checks.append(dict(frame=index,time_seconds=index*.1,collection_sha256=expected,render_sha256=observed))
                    frames.append(np.moveaxis(chw,0,-1))
                frame_sets[version]=frames
                source_path=Path(row['source_file']);old_video=source_path.parent.parent/'report/videos'/f'{row["source_trajectory_id"]}.mp4'
                reused=version=='R3A'and old_video.is_file()
                if reused:
                    video=relative(old_video)
                else:
                    path=out/f'{case["case_id"]}-{version}.mp4'
                    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10.,(192,224))
                    if not writer.isOpened():raise RuntimeError('Installed mp4v encoder unavailable')
                    for index,frame in enumerate(frames):
                        canvas=np.full((224,192,3),255,np.uint8)
                        cv2.putText(canvas,f'{version} {case["letter"]} t={index/10:.1f}s',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
                        canvas[32:]=cv2.resize(frame,(192,192),interpolation=cv2.INTER_NEAREST)
                        writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
                    writer.release();generated.append(path);video=path.name
                check=cv2.VideoCapture(str(old_video if reused else out/video));decoded=int(check.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(check.get(cv2.CAP_PROP_FPS));check.release()
                if decoded!=len(frames)or abs(fps-10)>1e-6:raise ValueError('Real-length clip frame count/fps differs')
                item=dict(version=version,status=row['status'],frames=len(frames),fps=10.,last_control_frame_seconds=(len(frames)-1)*.1,
                    video=video,reused_original_video=reused,video_sha256=sha(old_video if reused else out/video),
                    known_state_available=True,verified_frames=checks,no_terminal_frame_extension=True)
                proofs.extend(checks)
                n=row['completed_controls'];known=np.asarray(arrays['substep_status'])==1
                if len(known)>n*10:
                    valid=np.flatnonzero(known)
                    if len(valid)and int(valid[-1])+1>n*10:
                        si=int(valid[-1]);state=arrays['substep_states'][si+1]
                        if np.isfinite(state).all():
                            chw=renderer.render(case['letter'],state);partial=out/f'{case["case_id"]}-{version}-last-known-partial.png'
                            cv2.imwrite(str(partial),cv2.cvtColor(np.moveaxis(chw,0,-1),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,9]);generated.append(partial)
                            item['last_known_partial']=dict(file=partial.name,state_index=si+1,time_seconds=(si+1)*.01,sha256=sha(partial),
                                evidence='Original renderer of known saved partial poststate; aborted partial control returned no collector RGB hash,so no hash equality is claimed for this display.')
                record['versions'].append(item)
            strip=np.full((44+3*224,96+5*192,3),255,np.uint8)
            for column,index in enumerate((0,12,24,36,48)):
                cv2.putText(strip,f't={index/10:.1f}s',(106+column*192,28),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
            for row_index,version in enumerate(('R3A','CP1','CP2')):
                y=44+row_index*224;frames=frame_sets[version]
                cv2.putText(strip,version,(7,y+25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA)
                for column,index in enumerate((0,12,24,36,48)):
                    x=96+column*192
                    if index<len(frames):strip[y:y+192,x:x+192]=cv2.resize(frames[index],(192,192),interpolation=cv2.INTER_NEAREST)
                    else:cv2.putText(strip,'N/A: unobserved',(x+12,y+96),cv2.FONT_HERSHEY_SIMPLEX,.45,(80,80,80),1,cv2.LINE_AA)
            path=out/f'{case["case_id"]}-versions.png';cv2.imwrite(str(path),cv2.cvtColor(strip,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,9]);generated.append(path)
            record['strip']=path.name;record['strip_sha256']=sha(path);records.append(record)
    finally:renderer.close()
    total=sum(p.stat().st_size for p in generated)
    if total>2*1024**2:raise RuntimeError('New fixed media exceeds2MiB; retain evidence and report exact bytes: '+str(total))
    result=dict(kind='CONTACT-PILOT2-fixed-version-media.v1',records=records,media_bytes=total,media_cap_bytes=2097152,
        generated_files=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in generated],
        verified_control_frames=len(proofs),renderer_stats=renderer.stats,physics_steps=0,model_calls=0,
        policy='Every version clip ends at its actual last saved control frame. Unexecuted CP2 is N/A. Known partial-state displays are separate and never encoded as a completed control or invented RGB ground truth.')
    write(out/'media-provenance.json',result);return result


def write_report(out,result,timelines):
    groups=result['groups'];q=groups['CP2_all_planned']
    rows=[]
    for label,value in groups.items():
        recovery=value['recovery'];delay=recovery['observed_delay_seconds']
        rows.append([label,f"{value['attempted_trajectories']}/{value['planned_trajectories']}",
            f"{value['charged_controls']}/{value['completed_controls']}",pct(value['contact_obvious_fraction_charged']),
            pct(value['contact_obvious_fraction_planned_lower']),pct(value['moving_pusher_noAO_fraction_charged_upper']),
            pct(value['moving_pusher_noAO_fraction_planned_upper']),f"{recovery['observed']}/{recovery['planned_tasks']}",
            f"{recovery['right_censored']}/{recovery['never_separated']}/{recovery['not_reached']}",
            f"{delay['median']}/{delay['p90']}"])
    report='# CONTACT-PILOT2 cold report\n\n'
    report+=f"Fixed60cases; attempted{result['attempted_cases']},unattempted{result['unattempted_cases']}. Actual charged{q['charged_controls']},complete{q['completed_controls']},knownsubsteps{q['known_substeps']},unknown{q['unknown_advance_substeps']},partial{q['partial_substeps']},reservation{q['reserved_not_started_controls']}.\n\n"
    report+='Original48are repeat diagnostics,not new families. Only the12fresh planned cases can supply new independent support when actually executed. Allunexecuted entries below are N/A,not static observations. A halted prefix does not establish performance of the fullfixedqueue.\n\n'
    report+='Contact means actual known100Hz agent-object post_solve evidence. Nearstatic means area-centroid translation<0.1image96px AND absolute unwrapped rotation<0.1degree per0.1s control. Obvious motion means translation>=2image96px OR absolute rotation>=5degrees; small motion is the remainder. These unchanged CONTACT_AUDIT thresholds are not reset to make this pilot pass.\n\n'
    report+=table(['Population','Attempted/planned','Charged/full','Contact+obv charged','Contact+obv48slot lower','Emptycharged upper','Empty48slot upper','Strictrecovery observed/tasks','Censored/never-separated/notreached','Observedmedian/P90s'],rows)
    report+='\n\nStrictprotocol recovery requires a complete10known-substep noAO control during fixedwithdraw,then realpostsolve contact at/aftertask26. Never-separated does not become0delay success. Rightcensored,never-separated andnotreached remain separate. Anyprotocol observation without priorAO beforewithdraw is separatelyflagged as firstacquisition,not a proven separation/reacquisition; observed_with_prior_AO and itsdelayquantiles give the conservative sensitivity without changing frozenprimarydefinitions. Unrecovered engineeringgoal uses allplannedtasks conservatively. Plannedslots are hypothetical fixedschedule sensitivity,not additional physicaltime.\n\n'
    report+='Engineeringgoals remain>=40%contact+obvious,<=25%empty,observedrecoverymedian<=.5s/P90<=1s,unrecovered<=5%,and zero newphysicalguardfailures. Allcriteria must be assessed on the completed fixedqueue; thresholdmet on a shortprefix is not a pass.\n\n'
    report+=table(['Population','Letter','Attempted/planned','Contactobv charged','Latecontactobv charged','Wallstall','Push/noAO','Longesthold','Unconfirmedroutecommands'],[
        [row['population'],row['letter'],f"{row['attempted_trajectories']}/{row['planned_trajectories']}",pct(row['contact_obvious_fraction_charged']),
         pct(row['late_contact_obvious_fraction_charged']),row['wallstall_controls'],f"{row['planned_push_noAO_controls']}/{row['planned_push_controls']}",
         row['longest_hold_complete_controls'],row['unconfirmed_route_commands']]for row in result['letters']if row['population']in('CP2_repeat_A','CP2_repeat_B','CP2_new_families')])
    report+='\n\nFixedfault/regression cases retain smallmotion andhold counts besideobviousmotion,so aggregatedimprovement cannot hide B04 or T040 regressions.\n\n'
    report+=table(['Case','Version','Status','Contactobvious','Contactsmall','No-contactsmall','Nearstatic','Holdcomplete'],[
        [row['case_id'],row['version'],row['status'],row['contact_obvious_controls'],row['six_classes']['contact_small'],
         row['six_classes']['no_contact_small'],row['nearstatic_controls'],row['hold_complete_controls']]
        for row in result['all_version_rows']if row.get('source_cp1_case_id')in(*FIXED_MEDIA,'CP1-A-T-01')])
    report+='\n\nRequestedarc evidence uses exact actualpostsolve arc membership; validimpulse requires a pointrow on thatsamearc,norm>1e-8. Adjacent/incidental contact is no substitute; zeroobserved hits are not proof of unreachability. Completecontaining-control motion is a co-occurrence,not isolated causal motion.\n\n'
    report+=table(['Case','Version','Status','Requestedarc','Contactsubsteps','Positiveimpulsesubsteps','Longestpositive run s','Motion+positivecontrols'],[
        [row['case_id'],row['version'],row['status'],row['initial_requested_arc_id'],row['requested_arc_contact_substeps'],
         row['requested_arc_positive_impulse_substeps'],row['requested_arc_max_positive_impulse_run_seconds'],
         row['requested_arc_positive_impulse_and_obvious_complete_controls']]for row in result['weak_requested_arc_rows']])
    report+='\n\nFullcase/version metrics:cases.csv; percase andletter deltas:paired-deltas.csv/paired-letter-deltas.csv; sixclasses and denominator variants:six-classes.csv; strictrecovery:recontact.csv; weakgoals:requested-arcs.csv; completeoriginalrisk evidence remains in hashed*.telemetry.json.gz referenced by lossless controller-telemetry-index.csv.gz/risk-index.csv.gz. Reporting uses0newphysics/modelcalls; originaltrainingpool and checkpointhashes are unchanged by this script.\n'
    (out/'RESULTS.md').write_text(report,encoding='utf-8')
    cards=[]
    for item in result['media'].get('records',[]):
        columns=[]
        for version in item['versions']:
            if not version.get('video'):
                columns.append(f'<div><h3>{version["version"]}</h3><p>N/A: no known executed state.</p></div>');continue
            partial=version.get('last_known_partial')
            extra=''if not partial else f'<p>Lastknownpartialt={partial["time_seconds"]:.2f}s; nocontrollereturnedRGBhash.</p><img src="{partial["file"]}" width="192">'
            columns.append(f'<div><h3>{version["version"]} / {html.escape(version["status"])}</h3><video controls preload="metadata" src="{html.escape(version["video"],quote=True)}" width="192"></video><p>{version["frames"]} actualcontrolframes,noextension</p>{extra}</div>')
        cards.append(f'<section><h2>{item["case_id"]}:fixedsame-initialstatecomparison</h2><div class="versions">'+''.join(columns)+f'</div><img src="{item["strip"]}" style="max-width:100%"></section>')
    data=json.dumps(timelines,separators=(',',':'),allow_nan=False).replace('</','<'+chr(92)+'/')
    script='''<script>const data=DATA;const target=document.getElementById('timeline');for(const c of data){const s=document.createElement('details');const h=document.createElement('summary');h.textContent=c.case_id+' ('+c.group+', '+c.status+')';s.append(h);for(const [v,d]of Object.entries(c.versions)){let row=document.createElement('div');row.className='timeline';let label=document.createElement('span');label.textContent=v+' AO/motion ';row.append(label);for(let i=0;i<48;i++){let b=document.createElement('i');const p=d.controls[i];b.style.background=p?['#cbd5e1','#fbbf24','#16a34a'][p[1]]:'#eee';b.style.borderTop=p?'5px solid rgba(0,90,40,'+(p[0]/10)+')':'5px solid #eee';b.title=p?'control'+i+' AO'+p[0]+'/10,motion'+p[1]+',phase'+p[2]+',mode'+p[3]+',translation'+p[4]+'px,rotation'+p[5]+'deg':'unobserved';row.append(b)}s.append(row)}target.append(s)}</script>'''.replace('DATA',data)
    page='<!doctype html><meta charset="utf-8"><title>CONTACT-PILOT2 recordedversions</title><style>body{font:15px system-ui;background:#eee;margin:24px}section,details{background:white;padding:12px;margin:10px 0}.versions{display:flex;gap:20px;flex-wrap:wrap}.timeline{display:flex;align-items:center;margin:8px 0}.timeline span{width:120px;flex:none}.timeline i{display:inline-block;width:12px;height:18px;margin-right:1px;flex:none}</style><h1>CONTACT-PILOT2:realrecordedversions</h1><p>FixedX03/T03/B04panels.Realpostsolvecontact/motion;missingcontrolsareunobserved,notstatic.No modelprediction ornewphysics forreporting.</p>'+''.join(cards)+'<h2>All60frozencases</h2><p>Openeachcase;topborderopacity=AOsubsteps/10,fill=nearstaticgray/smallyellow/obviousgreen.Hoverforactualvalues.</p><div id="timeline"></div>'+script
    (out/'index.html').write_text(page,encoding='utf-8')


def self_test():
    n=48
    arrays=dict(states=np.zeros((n+1,15),np.float64),actions=np.zeros((n,2)),attempted_actions=np.zeros((n,2)),
        substep_states=np.zeros((n*10+1,15),np.float64),substep_status=np.ones(n*10,np.int8),
        contact_flags=np.zeros((n*10,3),np.uint8),contact_tags=np.empty((0,5),np.int16),
        contact_values=np.empty((0,11),np.float32),contact_arc_offsets=np.array([0],np.int32),contact_arc_ids=np.empty(0,np.int16),
        control_phase=np.r_[np.zeros(3,np.uint8),np.ones(15,np.uint8),np.full(8,2,np.uint8),np.ones(22,np.uint8)],
        control_mode=np.zeros(n,np.uint8))
    case=dict(case_id='fixture',source_cp1_case_id='CP1-A-X-03',group='CP1_A_repeat',letter='X',family_id='fixture',
        role='fixture',scenario='release_recontact',slot=dict(scenario='release_recontact'),task_recontact_start_control=26,
        candidate=dict(initial_state15=np.zeros(15).tolist(),selected_contact=dict(arc_id=7)),initial_requested_arc=dict(arc_id=7))
    # A continuing contact at26 without even one releasedwithdrawcontrol is not0srecovery.
    arrays['contact_flags'][:,0]=2
    r=strict_recovery(case,arrays)
    assert r['recontact_never_separated']and not r['recontact_observed']and r['recontact_delay_seconds']is None
    arrays['contact_flags'][180:190,0]=0
    r=strict_recovery(case,arrays)
    assert r['withdrawal_release_observed']and r['recontact_observed']and r['recontact_delay_seconds']==0
    # Unknown withdrawal interval is not a completeknown noAO release.
    arrays['substep_status'][180:190]=2
    assert not strict_recovery(case,arrays)['withdrawal_release_observed']
    arrays['substep_status'][:]=1;arrays['contact_flags'][:]=0;arrays['contact_flags'][391,0]=2
    r=strict_recovery(case,arrays)
    assert r['recontact_observed']and r['recontact_delay_seconds']==1.31
    assert r['recovery_without_prior_contact']and not r['separated_recovery_with_prior_contact']
    arrays['contact_flags'][10,0]=2
    assert strict_recovery(case,arrays)['separated_recovery_with_prior_contact']
    # Exactrequestedrow impulse:adjacentpositive event cannot certify target7.
    arrays['contact_flags'][390:392,0]=2
    arrays.update(contact_tags=np.array([[390,0,1,0,4],[391,0,1,0,4]],np.int16),contact_values=np.zeros((2,11),np.float32),
        contact_arc_offsets=np.array([0,1,2],np.int32),contact_arc_ids=np.array([8,7],np.int16))
    arrays['contact_values'][0,7]=3
    arc=requested_arc(case,arrays,version='CP2',status='completed')
    assert arc['requested_arc_contact_substeps']==1 and arc['requested_arc_positive_impulse_substeps']==0
    arrays['contact_values'][1,7]=2
    assert requested_arc(case,arrays,version='CP2',status='completed')['requested_arc_positive_impulse_substeps']==1
    # Firstcase earlyabort cannot turn the other59cases intoobservedstatic/success.
    short=dict(arrays)
    for key,length in(('states',2),('actions',1),('attempted_actions',2),('substep_states',14),('substep_status',13),('contact_flags',13),('control_phase',2),('control_mode',2)):
        short[key]=arrays[key][:length]
    short.update(contact_tags=np.empty((0,5),np.int16),contact_values=np.empty((0,11),np.float32),contact_arc_offsets=np.array([0],np.int32),contact_arc_ids=np.empty(0,np.int16))
    first,_=measure_case(case,dict(status='failed_physics',charged_controls=2,reserved_not_started_controls=0),short,Path(__file__),version='CP2')
    first['physical_guard_failure']=True
    empty,_=measure_case(case,dict(status='not_executed',charged_controls=0,reserved_not_started_controls=0),empty_arrays(case),Path(__file__),version='CP2',has_data=False)
    q=aggregate([first]+[empty.copy()for _ in range(59)])
    assert q['planned_control_slots']==2880 and q['charged_controls']==2 and q['completed_controls']==1
    assert q['attempted_trajectories']==1 and q['unattempted_trajectories']==59 and q['physical_guard_failures']==1
    assert q['moving_pusher_noAO_fraction_planned_upper']==2879/2880
    assert not q['complete_fixed_queue_assessed']and not q['all_engineering_targets_passed']
    assert q['recovery']['not_reached']==60 and q['recovery']['observed']==0
    return dict(passed=True,checks=['never-separated is notzero-delay recovery','completeknownwithdrawrelease required',
        'fixedtask391=>1.31andpriorcontact sensitivity','exactrequestedarc impulse notneighbor','earlyabort60case planneddenominators','guard/partial/unexecuted neverpass fullqueue'],
        model_calls=0,physics_steps=0,Torch_imported='torch'in sys.modules)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,default=STAGE)
    parser.add_argument('--plan',type=Path);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--source-root',type=Path,default=ROOT.parent/'Tdyssey/third_party/diffusion_policy')
    parser.add_argument('--no-media',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    build(args.stage,args.manifest or args.stage/'manifest.json',args.plan or args.stage/'plan/cases.json',args.source_root,media=not args.no_media)


if __name__=='__main__':main()
