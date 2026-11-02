"""Saved-only exposure diagnosis. No simulator/model imports or execution.

The exported indices/weights are review-only analysis, never an enabled dataset.
"""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO/'artifacts/direct-target-acceptance-20261031'
HS = (1, 4, 8, 16)
LETTERS = 'TLOCXB'
ORDINARY = {'translation', 'rotation_cw', 'rotation_ccw'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_json(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def save_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v, (list,dict)) else v for k,v in row.items()})


def boolean(v):
    return str(v).lower() == 'true'


def phase(i):
    return 'history_0_2' if i < 3 else 'first_task_3_18' if i < 19 else 'later_19_plus'


def cohort(a):
    if a.get('branch_id',0)>0:
        return 'branch_'+a['branch']
    return 'ordinary_'+a['target_mode'] if a['scenario'] in ORDINARY else a['scenario']+'_'+a['target_mode']


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    config=json.loads((SOURCE/'runtime-config.json').read_text())
    plan=json.loads((SOURCE/'plan.json').read_text())
    attempts={a['attempt_id']:a for a in plan['attempts']}
    families=defaultdict(list)
    for a in attempts.values():
        families[a['family_id']].append(a['attempt_id'])
    assert len(attempts)==120 and len(families)==96
    controls=read_csv(SOURCE/'report/controls.csv')
    windows=read_csv(SOURCE/'report/windows.csv')
    grouped=defaultdict(list)
    for row in controls:
        row['control']=int(row['control']); row['valid']=boolean(row['valid'])
        for k in ('actual_seconds','translation_image_px','rotation_deg','AO_positive_seconds','AO_pair_seconds','OW_pair_seconds','target_change_world_px','pusher_motion_image_px'):
            row[k]=float(row[k]) if row[k] else None
        grouped[row['attempt_id']].append(row)
    assert len(controls)==3446
    assert sum(row['valid'] for row in controls)==3372
    assert all([x['control'] for x in rr]==list(range(len(rr))) for rr in grouped.values())
    control_table=[]
    events=defaultdict(lambda:defaultdict(set))
    # These are audit strata, never replacements for original labels.
    for aid,a in attempts.items():
        rows=sorted(grouped[aid],key=lambda x:x['control'])
        had_motion=had_contact=False
        prev_pair=prev_positive=prev_wall=prev_motion=False
        prev_phase=None
        for row in rows:
            i=row['control']; motion=row['translation_image_px']>=.1 or row['rotation_deg']>=.5
            pair=row['AO_pair_seconds']>0; positive=row['AO_positive_seconds']>0; wall=row['OW_pair_seconds']>0
            if row['valid']:
                if pair != prev_pair:events[aid][i].add('contact_transition')
                if positive and not prev_positive:events[aid][i].add('positive_impulse_onset')
                if wall != prev_wall:events[aid][i].add('wall_transition')
                if motion != prev_motion:events[aid][i].add('motion_transition')
                if prev_phase is not None and row['planned_phase'] != prev_phase and (row['target_change_world_px'] or 0)>0:
                    events[aid][i].add('action_phase_switch')
            if not row['valid']:kind='invalid_terminal'
            elif i<3:kind='real_history_static' if row['category']=='static' else 'real_history_other'
            elif motion:kind='observed_motion_with_AO' if positive else 'observed_motion_without_positive_AO'
            elif row['category']=='contact_blocked_or_small':kind='contact_but_subdiagnostic_motion'
            elif row['category']=='static' and had_motion:kind='observed_static_after_motion'
            elif row['category']=='static' and a['scenario']=='negative':kind='planned_negative_static'
            elif row['category']=='static':kind='static_without_prior_diagnostic_motion'
            elif not pair and had_contact:kind='no_AO_pair_after_prior_positive_contact'
            elif row['category']=='empty_pusher_motion' and a['scenario']=='negative':kind='planned_negative_pusher_motion'
            elif not pair:kind='no_AO_pair_current_push_or_other_intent'
            else:kind='other_observed'
            row.update(period=phase(i),cohort=cohort(a),observed_stratum=kind,motion_diagnostic=motion)
            control_table.append(row)
            had_motion |= motion;had_contact |= positive
            prev_pair,prev_positive,prev_wall,prev_motion,prev_phase=pair,positive,wall,motion,row['planned_phase']
    raw={}
    bindings=[]
    expected_raw_hashes={x['attempt_id']:x['sha256'] for x in read_csv(SOURCE/'report/raw-manifest.csv')}
    for aid in attempts:
        path=SOURCE/'collection/raw'/(aid+'.npz')
        with np.load(path,allow_pickle=False) as z:
            raw[aid]={k:z[k].copy() for k in ('rgb','states','requested_actions','control_valid')}
        assert raw[aid]['rgb'].shape==(len(raw[aid]['requested_actions'])+1,3,96,96)
        assert np.array_equal(raw[aid]['requested_actions'],np.asarray(attempts[aid]['actions'])[:len(raw[aid]['requested_actions'])])
        actual_hash=sha(path);assert actual_hash==expected_raw_hashes[aid]
        bindings.append(dict(attempt_id=aid,path=str(path),sha256=actual_hash))
    indexed=[]
    by_ah=defaultdict(list)
    for w in windows:
        aid=w['attempt_id'];a=attempts[aid];t=int(w['anchor']);h=int(w['horizon']);rr=grouped[aid][t:t+h];z=raw[aid]
        assert t>=3 and t+h<=len(z['control_valid']) and z['control_valid'][t-3:t+h].all()
        near=all(x['translation_image_px']<.1 and x['rotation_deg']<.5 for x in rr)
        rgb_equal=bool(np.all(z['rgb'][t+1:t+h+1]==z['rgb'][t]))
        state_equal=bool(np.all(z['states'][t+1:t+h+1]==z['states'][t]))
        types=sorted({kind for e,kinds in events[aid].items() if t<=e<t+h for kind in kinds})
        row=dict(attempt_id=aid,family_id=a['family_id'],letter=a['letter'],scenario=a['scenario'],cohort=cohort(a),branch_id=a['branch_id'],
                 anchor=t,horizon=h,post19_future_only=t>=19,all_future_steps_below_existing_motion_diagnostic=near,
                 exact_RGB_static=rgb_equal,exact_state_static=state_equal,
                 net_translation_image_px=float(w['translation_image_px']),net_rotation_deg=float(w['rotation_deg']),
                 cumulative_translation_image_px=float(w['cumulative_translation_image_px']),cumulative_rotation_deg=float(w['cumulative_rotation_deg']),
                 constant_future_target=bool(np.all(z['requested_actions'][t:t+h]==z['requested_actions'][t])),
                 has_positive_AO=any(x['AO_positive_seconds']>0 for x in rr),has_AO_pair=any(x['AO_pair_seconds']>0 for x in rr),
                 has_OW_pair=any(x['OW_pair_seconds']>0 for x in rr),event_types=types,
                 has_pusher_motion=any(x['pusher_motion_image_px']>=.1 for x in rr),
                 event_centered=bool(types),candidate_keep=bool(types) or not near or any(x['pusher_motion_image_px']>=.1 for x in rr),
                 candidate_reason='event_in_future' if types else 'observed_motion_in_future' if not near else
                     'pusher_motion_in_future' if any(x['pusher_motion_image_px']>=.1 for x in rr) else '',
                 review_only=True,training_eligible=False)
        indexed.append(row);by_ah[aid,h].append(row)
    # Necessary non-motion support is explicitly retained, rather than deleting
    # all quiet outcomes. Only indices are proposed; raw and labels never change.
    for (aid,h),rows in by_ah.items():
        a=attempts[aid]
        groups={
            'negative_static_or_empty':[w for w in rows if a['scenario']=='negative' and not w['has_AO_pair']],
            'contact_subdiagnostic_hold':[w for w in rows if w['all_future_steps_below_existing_motion_diagnostic'] and w['has_positive_AO']],
            'post_motion_still_response':[w for w in rows if w['all_future_steps_below_existing_motion_diagnostic']
                and any(x['motion_diagnostic'] for x in grouped[aid][:w['anchor']])],
        }
        for reason,pool in groups.items():
            if pool:
                selected=min(pool,key=lambda w:w['anchor'])
                selected['candidate_keep']=True
                selected['candidate_reason'] += ('|' if selected['candidate_reason'] else '')+reason
    priority=('wall_transition','contact_transition','positive_impulse_onset','action_phase_switch','motion_transition')
    for w in indexed:
        if w['event_types']:
            w['event_bucket']=next(k for k in priority if k in w['event_types'])
        elif 'observed_motion_in_future' in w['candidate_reason']:w['event_bucket']='observed_motion_segment'
        elif 'pusher_motion_in_future' in w['candidate_reason']:w['event_bucket']='observed_pusher_only_motion_segment'
        elif 'negative_static_or_empty' in w['candidate_reason']:w['event_bucket']='negative_static_or_empty'
        elif 'contact_subdiagnostic_hold' in w['candidate_reason']:w['event_bucket']='contact_subdiagnostic_hold'
        elif 'post_motion_still_response' in w['candidate_reason']:w['event_bucket']='post_motion_still_response'
        else:w['event_bucket']='not_selected'
    selected=[w for w in indexed if w['candidate_keep']]
    assert len({(w['attempt_id'],w['anchor'],w['horizon']) for w in indexed})==len(indexed)
    assert [sum(w['horizon']==h for w in indexed) for h in HS]==[3012,2808,2566,2117]
    balancing=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for w in selected:balancing[w['horizon']][w['family_id']][w['event_bucket']].append(w)
    for w in indexed:
        aid=w['attempt_id'];h=w['horizon'];fam=w['family_id']
        # Every one of96 planned families gets1/96mass; each planned branch
        # gets equal mass. A branch without windows remains unsupported mass.
        w['equal_family_branch_weight_all']=1/(96*len(families[fam])*len(by_ah[aid,h]))
        w['equal_family_event_weight_candidate']=0.
        if w['candidate_keep']:
            strata=balancing[h][fam];members=strata[w['event_bucket']]
            branches={x['attempt_id'] for x in members};n=sum(x['attempt_id']==aid for x in members)
            w['equal_family_event_weight_candidate']=1/(96*len(strata)*len(branches)*n)
    for h in HS:
        candidate_h=[w for w in selected if w['horizon']==h]
        assert np.isclose(sum(w['equal_family_event_weight_candidate'] for w in candidate_h),len({w['family_id'] for w in candidate_h})/96,rtol=0,atol=1e-12)
    def window_summary(rows):
        total=len(rows)
        allmass=sum(w['equal_family_branch_weight_all'] for w in rows)
        balmass=sum(w['equal_family_event_weight_candidate'] for w in rows)
        counts_by_attempt=Counter(w['attempt_id'] for w in rows)
        pool_weight=lambda w:1/(96*len(families[w['family_id']])*counts_by_attempt[w['attempt_id']])
        poolmass=sum(pool_weight(w) for w in rows)
        result=dict(windows=total,independent_families=len({w['family_id'] for w in rows}),attempts=len({w['attempt_id'] for w in rows}),
            stationary_windows=sum(w['all_future_steps_below_existing_motion_diagnostic'] for w in rows),
            exact_RGB_static_windows=sum(w['exact_RGB_static'] for w in rows),exact_state_static_windows=sum(w['exact_state_static'] for w in rows),
            post19_future_only_windows=sum(w['post19_future_only'] for w in rows),constant_future_target_windows=sum(w['constant_future_target'] for w in rows),
            post19_stationary_windows=sum(w['post19_future_only'] and w['all_future_steps_below_existing_motion_diagnostic'] for w in rows),
            post19_exact_RGB_static_windows=sum(w['post19_future_only'] and w['exact_RGB_static'] for w in rows),
            post19_constant_target_windows=sum(w['post19_future_only'] and w['constant_future_target'] for w in rows),
            object_low_motion_but_pusher_motion_windows=sum(w['all_future_steps_below_existing_motion_diagnostic'] and w['has_pusher_motion'] for w in rows),
            negative_scenario_windows=sum(w['scenario']=='negative' for w in rows),
            withdraw_branch_windows=sum(w['cohort']=='branch_withdraw' for w in rows),
            all_family_branch_observed_mass=allmass,candidate_family_event_observed_mass=balmass,
            within_pool_equal_family_branch_observed_mass=poolmass,
            candidate_bucket_counts=dict(Counter(w['event_bucket'] for w in rows)))
        for name,key in (('stationary','all_future_steps_below_existing_motion_diagnostic'),('exact_RGB_static','exact_RGB_static'),('post19_future_only','post19_future_only')):
            result[name+'_window_fraction']=sum(w[key] for w in rows)/total if total else None
            result[name+'_family_branch_conditional_fraction']=sum(w['equal_family_branch_weight_all']*w[key] for w in rows)/allmass if allmass else None
            result[name+'_family_event_conditional_fraction']=sum(w['equal_family_event_weight_candidate']*w[key] for w in rows)/balmass if balmass else None
            result[name+'_within_pool_family_branch_conditional_fraction']=sum(pool_weight(w)*w[key] for w in rows)/poolmass if poolmass else None
        return result
    comparison={}
    for h in HS:
        allh=[w for w in indexed if w['horizon']==h];eventh=[w for w in allh if w['event_centered']];candidateh=[w for w in allh if w['candidate_keep']]
        comparison[str(h)]={'all_legal':window_summary(allh),'events_only_diagnostic_NOT_recommendation':window_summary(eventh),
            'events_plus_explicit_negative_and_hold':window_summary(candidateh),
            'unsupported_planned_family_branch_mass_all':1-sum(w['equal_family_branch_weight_all'] for w in allh),
            'unsupported_family_mass_candidate':1-len({w['family_id'] for w in candidateh})/96}
    window_groups=[]
    for scope,values,key in [('letter',LETTERS,'letter'),('scenario',sorted({a['scenario'] for a in attempts.values()}),'scenario'),
                              ('cohort',sorted({cohort(a) for a in attempts.values()}),'cohort')]:
        for value in values:
            for h in HS:
                for pool_name,pool in [('all_legal',indexed),('candidate',selected)]:
                    ss=window_summary([w for w in pool if w[key]==value and w['horizon']==h])
                    window_groups.append(dict(scope=scope,value=value,horizon=h,pool=pool_name,**ss))
    period_rows=[]
    for scope,key,values in [('all',None,['all']),('cohort','cohort',sorted({x['cohort'] for x in control_table})),
                            ('letter','letter',LETTERS),('scenario','scenario',sorted({x['scenario'] for x in control_table}))]:
        for value in values:
            for period_name in ('history_0_2','first_task_3_18','later_19_plus'):
                rr=[x for x in control_table if x['period']==period_name and (key is None or x[key]==value)]
                seconds=sum(x['actual_seconds'] for x in rr);static=sum(x['actual_seconds'] for x in rr if x['category']=='static')
                # Equal planned-family, equal planned-branch control-time fractions;
                # missing phase time has explicit unsupported mass, never static.
                fractions=[];missing=0
                def in_scope(aid):
                    return key is None or (cohort(attempts[aid]) if key=='cohort' else attempts[aid][key])==value
                scoped_families={f:[aid for aid in ids if in_scope(aid)] for f,ids in families.items()}
                scoped_families={f:ids for f,ids in scoped_families.items() if ids}
                famids=list(scoped_families)
                for fam,branch_ids in scoped_families.items():
                    for aid in branch_ids:
                        br=[x for x in rr if x['attempt_id']==aid];den=sum(x['actual_seconds'] for x in br)
                        if den:fractions.append((sum(x['actual_seconds'] for x in br if x['category']=='static')/den)/len(branch_ids))
                        else:missing+=1/len(branch_ids)
                cats=Counter()
                for x in rr:cats[x['observed_stratum']]+=x['actual_seconds']
                period_rows.append(dict(scope=scope,value=value,period=period_name,actual_seconds=seconds,static_seconds=static,
                    static_time_weighted_fraction=static/seconds if seconds else None,planned_families=len(famids),
                    observed_families=len({x['family_id'] for x in rr}),unsupported_family_branch_mass=missing/len(famids) if famids else None,
                    equal_family_branch_static_mass=sum(fractions)/len(famids) if famids else None,
                    equal_family_branch_static_fraction_conditional=sum(fractions)/(len(famids)-missing) if len(famids)>missing else None,
                    observed_stratum_seconds=dict(cats)))
    loss=[]
    for h in HS:
        maxima=64-h-2
        for scope,key,values in [('all',None,['all']),('letter','letter',LETTERS),('scenario','scenario',sorted({a['scenario'] for a in attempts.values()}))]:
            for val in values:
                aa=[a for a in attempts.values() if key is None or a[key]==val]
                complete=[a for a in aa if len(raw[a['attempt_id']]['control_valid'])==64 and raw[a['attempt_id']]['control_valid'].all()]
                partial=[a for a in aa if a not in complete]
                fullwindows=sum(len(by_ah[a['attempt_id'],h]) for a in complete);partwindows=sum(len(by_ah[a['attempt_id'],h]) for a in partial)
                loss.append(dict(scope=scope,value=val,horizon=h,planned_attempts=len(aa),independent_families=len({a['family_id'] for a in aa}),
                    theoretical_windows_if_all64legal=len(aa)*maxima,complete_attempts=len(complete),complete_legal_windows=fullwindows,
                    early_terminated_attempts=len(partial),retained_early_prefix_windows=partwindows,
                    missing_due_to_early_termination=len(partial)*maxima-partwindows))
    # Avoid embedding huge per-window arrays in JSON; indices are small CSVs.
    index_keys=('attempt_id','family_id','letter','scenario','cohort','branch_id','anchor','horizon','post19_future_only','all_future_steps_below_existing_motion_diagnostic',
        'exact_RGB_static','exact_state_static','constant_future_target','has_pusher_motion','event_types','event_centered','candidate_keep','candidate_reason','event_bucket',
        'equal_family_branch_weight_all','equal_family_event_weight_candidate','review_only','training_eligible')
    save_csv('REVIEW_ONLY_window_indices.csv',[{k:w[k] for k in index_keys} for w in indexed])
    save_csv('window_exposure_by_group.csv',window_groups)
    save_csv('control_phase_exposure.csv',period_rows)
    save_csv('early_termination_window_loss.csv',loss)
    events_rows=[dict(attempt_id=aid,family_id=attempts[aid]['family_id'],control=i,event_types=sorted(k)) for aid,entries in events.items() for i,k in entries.items()]
    save_csv('observed_event_index.csv',events_rows)
    compact=[];overlap=[]
    for h,v in comparison.items():
        for method,key,weighted in [('all_legal_unit_weight','all_legal',False),
                ('event_motion_negative_candidate_unit_weight','events_plus_explicit_negative_and_hold',False),
                ('candidate_equal_family_event_weight','events_plus_explicit_negative_and_hold',True)]:
            x=v[key];suffix='family_event_conditional_fraction' if weighted else 'window_fraction'
            compact.append(dict(horizon=int(h),method=method,unique_indices=x['windows'],supported_families=x['independent_families'],planned_families=96,
                object_low_motion_fraction=x['stationary_'+suffix],exact_RGB_static_fraction=x['exact_RGB_static_'+suffix],
                within_pool_equal_family_sibling_mean_object_low_motion=x['stationary_within_pool_family_branch_conditional_fraction'],
                within_pool_equal_family_sibling_mean_exact_RGB_static=x['exact_RGB_static_within_pool_family_branch_conditional_fraction'],
                planned_family_sibling_mass_observed=x['within_pool_equal_family_branch_observed_mass'],
                family_event_mass_observed=x['candidate_family_event_observed_mass'] if weighted else None,
                actual_object_low_motion_but_blue_pusher_motion_indices=x['object_low_motion_but_pusher_motion_windows'],
                negative_scenario_indices=x['negative_scenario_windows'],withdraw_branch_indices=x['withdraw_branch_windows']))
        hh=[w for w in selected if w['horizon']==int(h)]
        for reason in sorted({reason for w in hh for reason in w['candidate_reason'].split('|') if reason}):
            rr=[w for w in hh if reason in w['candidate_reason'].split('|')]
            overlap.append(dict(horizon=int(h),dimension='candidate_reason_overlap',label=reason,windows=len(rr),families=len({w['family_id'] for w in rr})))
        for reason in sorted({reason for w in hh for reason in w['event_types']}):
            rr=[w for w in hh if reason in w['event_types']]
            overlap.append(dict(horizon=int(h),dimension='event_type_overlap',label=reason,windows=len(rr),families=len({w['family_id'] for w in rr})))
    save_csv('compact_three_exposure_views.csv',compact)
    save_csv('candidate_reason_overlap.csv',overlap)
    save_json('EXPOSURE_SUMMARY.json',dict(review_only=True,loadable_by_runtime=False,execution_authorized=False,
        new_physics_calls=0,new_model_calls=0,new_optimizer_updates=0,raw_labels_changed=False,source_run=config['run_id'],
        stationarity_definition='Every future control has centroid translation<0.1 imagepx AND abs rotation<0.5deg; separate exact RGB and exact state equality. This is descriptive, not a legal/admission rule.',
        events_definition='Saved AO pair gain/loss, first positive impulse, OW gain/loss, object diagnostic motion on/off, or a changed programmed phase with actual changed requested target. Future truth is privileged offline index selection, never model input.',
        candidate_definition='All legal windows whose future contains a saved event OR diagnostic object motion OR pusher motion>=0.1imagepx (existing empty-pusher-motion diagnostic); add earliest available negative/no-AO, contact-with-subdiagnostic-motion hold, and post-motion still response per attempt/H. Sustained object/blue-pusher motion retained, not only transitions. No failed/invalid transition is admitted.',
        weighting_definition='All-window family/branch: each of96families has1/96mass, each planned branch equal then own windows equal; missing branch support stays unsupported mass. Candidate: supported family1/96, eventbuckets equal, available branches withinbucket equal, then windows equal; absent families explicit unsupported mass. No oversampling/repetition performed.',
        within_pool_family_sibling_definition='For either alllegal or candidatepool separately: family1/96, planned siblings equal, own available indices in thatpool equal. Branches with noindices retain explicitunsupportedmass. Conditional means divide onlyobservedmass; unsupportedfamilies/siblings are never assigned zero-motion or fabricatedindices.',
        comparison=comparison,all_periods=[x for x in period_rows if x['scope']=='all'],
        total_legal_windows=len(indexed),candidate_indices=len(selected),observed_events=len(events_rows),
        mathematical_tail_note='For a complete64control trajectory, H16anchors3..48total46; anchors19..48total30havefutureentirelyaftercontrol19 (30/46=65.217%). Holding a target is not itself motion cessation.',
        post19_definition='post19_future_only means future begins atcontrol19or later. It is a time-indexstratum acrossallscenarios, not an assumption that every scenario holds a target there. Constant-future-target and scenario/cohort counts are separate.',
        early_termination_note='The planned64legal-window ceiling minus observedwindows is unobserved capacity only. It does not establish continued future would remain legal, move, or provide useful data.',
        all_early_termination=[x for x in loss if x['scope']=='all']))
    paths=[SOURCE/'runtime-config.json',SOURCE/'plan.json',SOURCE/'report/controls.csv',SOURCE/'report/windows.csv',REPO/'tdyssey/direct_target_policy.py',Path(__file__)]
    save_json('SOURCE_BINDINGS.json',dict(files=[dict(path=str(p),sha256=sha(p)) for p in paths],raw_records=bindings))
    save_json('CHECKS.json',dict(passed=True,source_run=config['run_id'],source_runtime_config_sha256=sha(SOURCE/'runtime-config.json'),
        raw_records_checked=120,families=96,control_records_checked=3446,legal_control_records=3372,
        raw_actions_equal_frozen_plan=True,indexes_have_true3historical_and_Hfuture_legal_controls=True,
        actual_raw_hashes_equal_original_manifest=True,
        no_duplicate_window_keys=True,own_horizon_counts=[3012,2808,2566,2117],
        candidate_family_event_weights_sum_to_supported_families_divided_by96=True,
        dataset_repacked=False,old_labels_changed=False,enabled_loader_or_config_changed=False,
        new_physics_calls=0,new_model_calls=0,new_optimizer_updates=0))
    os.environ['MPLCONFIGDIR']=str(HERE/'.mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    aps=[x for x in period_rows if x['scope']=='all'];labels=['Real history0–2','First task3–18','Later19+']
    axes[0].bar(labels,[x['actual_seconds'] for x in aps],label='Other executed time',color='#75aadb')
    axes[0].bar(labels,[x['static_seconds'] for x in aps],label='Original static category',color='#bfc2c6')
    axes[0].set_ylabel('Executed simulation seconds');axes[0].legend(fontsize=8)
    x=np.arange(4)
    for shift,name,sourcekey,col in [(-.23,'Alllegal','all_legal','#555555'),(0,'Events+negative/holds','events_plus_explicit_negative_and_hold','#4596bd')]:
        vals=[comparison[str(h)][sourcekey]['stationary_window_fraction']*100 for h in HS]
        axes[1].bar(x+shift,vals,width=.23,label=name,color=col)
    axes[1].bar(x+.23,[comparison[str(h)]['events_plus_explicit_negative_and_hold']['stationary_family_event_conditional_fraction']*100 for h in HS],width=.23,label='Family/event weightedcandidate',color='#b77737')
    axes[1].set_xticks(x,[f'H{h}' for h in HS]);axes[1].set_ylabel('Allfuturesteps subdiagnosticmotion (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Saved-data exposure diagnosis — review only; no physics/model calls')
    fig.tight_layout();fig.savefig(HERE/'exposure_summary.png',dpi=160);plt.close(fig)
    text=['保存数据曝光诊断（B问题附件，2026-10-31）',
        '范围：只读DTA120条保存记录、原controls/windows表、冻结计划与源码；新增物理/模型/训练调用均0。所有原始记录和旧合法/非法标签保留。这里只输出审查索引/权重，未重打包训练集，未修改collector/policy/guard/loader/UI/启用配置。',
        '', '1. 45步保持的代码机制与证据边界',
        'direct_target_policy.py:218先分配64个动作，:220 fixed从3开始固定，:222普通moving用3..18的16步直线，19..63保持终点；分支:338固定其新目标。特定改向/再接触场景另有时间表，不可把普通moving结论套给全部场景。',
        '可证实：实现只给普通moving安排了16步主动目标路径，随后填终点到统一64长度。未发现45步尾长有独立、经验证的停止响应需求或曝光配额。不能仅凭数组推测作者主观动机，也不能因实现未超最大64预算而证明该尾长合理。缺少的审查点是“每个目标时间块教什么、事件结束后保留多少独立监督”，而不是目标坐标是否合法。',
        '', '2. 原类别静止与真实模型图像重复要分开',
        '原static类别排除了正冲量但小运动的contact_blocked_or_small，因此static比例不能代表所有低运动。下列“低运动窗口”要求未来每个控制的物体面积质心平移<0.1图像px且旋转<0.5度（既有诊断门槛）；不等于严格物理静止。另独立统计全部96pxRGB逐比特等于锚点的窗口。来回运动不能因净位移0被误算静止。',
        '初始真实历史0..2：36.00秒，原static36.00秒。首任务3..18：96.91秒，原static21.90秒(22.60%)。19+：207.01秒，原static112.40秒(54.30%)。混合不同幸存/场景，非因果对照。',
        '19+原static112.40秒中：withdraw分支54.00秒、固定静止negative27.00秒，合计81.00秒(72.06%)；这些有动作反事实/负例意义，但45步重复曝光没有因此获得依据。普通moving基线单独在19+执行72.00秒，原static17.40秒(24.17%)；剩余还有大量正冲量接触但低运动，不能仅凭static类别说普通moving尾部大多仍在有效推动。',
        '', '|H|所有合法窗口/家族|逐步低运动窗口|RGB完全重复窗口|全部未来在19+的窗口|其中逐步低运动/RGB重复|',
        '|---|---|---|---|---|---|']
    for h in HS:
        x=comparison[str(h)]['all_legal'];text.append(f'|{h}|{x["windows"]}/{x["independent_families"]} of96|{x["stationary_windows"]} ({x["stationary_window_fraction"]*100:.2f}%)|{x["exact_RGB_static_windows"]} ({x["exact_RGB_static_window_fraction"]*100:.2f}%)|{x["post19_future_only_windows"]}|{x["post19_stationary_windows"]}/{x["post19_exact_RGB_static_windows"]}|')
    text += ['', 'H16共2117窗口中1380个未来完全落在19+（65.19%）；其中1317逐步低运动，1123张未来序列所有RGB均与锚点相同。普通moving基线贡献341个完全重复H16窗口，withdraw460、静止negative276。19+是时间分层，不自动代表全部场景都在保持目标；H16的1380个19+窗口中1320个未来动作保持同目标，另60个有后续事件动作。',
        '', '3. 一版审查索引的三个曝光视图（不是三路训练试验）',
        '保留：未来含AO接触gain/loss、正冲量起始、OWgain/loss、物体运动启停、实际目标变化的计划相位切换；以及持续物体运动或蓝推杆移动>=0.1图像px的合法窗口。另逐尝试/H保留最早合法无AO负例、接触但低运动hold、运动后低运动响应代表；已被事件窗口包含时不复制。所有H仍要求真实3历史控制+H完整未来，不跨旧非法/缺帧/reset，不改动作。未来真值用于离线索引选择是特权信息，不能成为模型输入。',
        '与“只选事件瞬间”不同，此候选保留所有原有低物体运动但蓝推杆实际移动窗口：H1/H4/H8/H16分别345/284/256/200；静止物体不等于无图像动作。',
        '', '|H|所有窗口：低运动% / RGB重复%|候选唯一索引/家族|候选单位窗口：低运动% / RGB重复%|候选家族/事件权重：低运动% / RGB重复%|',
        '|---|---|---|---|---|']
    for h in HS:
        aa=comparison[str(h)]['all_legal'];cc=comparison[str(h)]['events_plus_explicit_negative_and_hold']
        text.append(f'|{h}|{aa["stationary_window_fraction"]*100:.2f} / {aa["exact_RGB_static_window_fraction"]*100:.2f}|{cc["windows"]}/{cc["independent_families"]} of96|{cc["stationary_window_fraction"]*100:.2f} / {cc["exact_RGB_static_window_fraction"]*100:.2f}|{cc["stationary_family_event_conditional_fraction"]*100:.2f} / {cc["exact_RGB_static_family_event_conditional_fraction"]*100:.2f}|')
    text += ['', '候选没有增加独立家族数：H16仍仅45/96家族，缺失51家族，不因索引把数据变大。候选H16保留666唯一索引，其中237逐步低运动、31完全重复RGB；6静止negative家族每家族至少1个代表，另movingnegative合法短前缀、withdraw的蓝推杆响应、受阻/停止响应均保留。不能把较低重复比例说成模型效果提高。',
        '两种家族权重均明确披露：',
        'A) 对每个池单独计算：96个计划家族各1/96，计划同胞分支均分，再在该分支本池现有窗口内均分；无窗口分支为unsupported质量，不伪造静止。所有H16条件低运动均值75.66%、RGB重复56.58%；同一公式用于候选为45.32%、17.58%。H16观测分支质量仅0.392361，缺失0.607639，不将缺失置零后算均值。',
        'B) 候选平衡建议：每个可支持家族先占1/96，再让有支持的事件桶均分，桶内可支持分支均分、分支内唯一索引均分。H16质量和45/96=0.46875；缺失51/96不伪造。低运动条件质量55.48%高于候选未加权35.59%，因为必要静止/小运动家族仍有同等权重，目标不是把运动比例做得最好看。权重和已检查，未重复抽样或执行训练。',
        '事件原因有重叠，不能相加：H16候选事件窗口523、额外持续物体运动21、额外蓝推杆移动116；负例/受阻/停止代表可能已在这些窗口中。candidate_reason_overlap.csv保留全部重叠标签，event_bucket仅是可审的优先归桶，不是已批准最优配额。',
        '', '4. 早停与64长度的上限损失',
        '|H|120条均64合法的数学窗口上限|46完整轨迹贡献|74早停保留|未观测上限差|',
        '|---|---|---|---|---|']
    for x in loss:
        if x['scope']=='all':text.append(f'|{x["horizon"]}|{x["theoretical_windows_if_all64legal"]}|{x["complete_legal_windows"]}|{x["retained_early_prefix_windows"]}|{x["missing_due_to_early_termination"]}|')
    text += ['', '上限差不是如果继续就会合法/运动的预测。索引只能减少已有重复监督，不能补出早停后未观察H16未来，也不能解除1世界px判据的不确定性。',
        '', '5. 对最小修改方案的支持与不能解决的事',
        '建议联合动作时间表与索引：任务最低3真实历史+16未来=19控制/20帧；长样本可按2或3个预先规划16步任务块为候选35/51控制，而不是统一补45步同目标。所有数字仍需审批，不能据本诊断直接裁原数据或新采。19/35/51控制的各H合法窗口上限分别为H1[16,32,48]、H4[13,29,45]、H8[9,25,41]、H16[1,17,33]；不能全改19导致长事件与多锚点丢失。',
        '固定目标直到停止的任务、非接触蓝推杆移动、撤离后物体静止、受阻都保留独立角色；事件后曝光长度/配额应明确，以可审索引或权重控制，不删除旧记录。停止可能要16步未来来观察响应，可在16步任务中安排已知转向/hold/撤离，不能假定目标不变就保证物体静止。',
        '若在块边界读取当前真实状态再定下一块，该块内部未来16目标可anchor_committed；跨反馈边界只能feedback_recorded。不能用裁短、复制历史、补帧或动作限幅凑合法窗口，也不能用新索引偷偷让旧非法转训练有效。',
        '', '交付：EXPOSURE_SUMMARY.json、compact_three_exposure_views.csv、control_phase_exposure.csv、window_exposure_by_group.csv、early_termination_window_loss.csv、observed_event_index.csv、REVIEW_ONLY_window_indices.csv、candidate_reason_overlap.csv、exposure_summary.png、SOURCE_BINDINGS.json、CHECKS.json。全部运行/标签变更关闭，索引仅审查。']
    (HERE/'EXPOSURE_FINDINGS.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(legal_windows=len(indexed),candidate_indices=len(selected),all_physics_model_optimizer_calls=0,
        outputs=str(HERE),output_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())),ensure_ascii=False))


if __name__=='__main__':main()
