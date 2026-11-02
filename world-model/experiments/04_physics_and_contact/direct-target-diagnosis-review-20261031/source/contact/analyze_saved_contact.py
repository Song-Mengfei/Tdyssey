"""Saved-array contact diagnosis only: no simulator/model imports or execution."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[3]
INPUT = REPO/'artifacts/direct-target-acceptance-20261031'
OUTPUT = Path(__file__).resolve().parent
PAIRS = ('OW', 'AO', 'AW')
KIND = {'OW':'object_wall', 'AO':'agent_object', 'AW':'agent_wall'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    (OUTPUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False)+'\n', encoding='utf-8')


def write_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if name.endswith('.gz') else open
    with opener(OUTPUT/name, 'wt', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(list,dict)) else v
                        for k,v in row.items()})


def stats(values):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return dict(N=len(x), **(dict(zip(('min','P10','P50','P90','max'),
                     map(float,np.quantile(x,[0,.1,.5,.9,1])))) if len(x) else
                     dict.fromkeys(('min','P10','P50','P90','max'))))


def rotate(v, angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.asarray((c*v[0]-s*v[1],s*v[0]+c*v[1]))


def contact_normal_velocity(event, pre, post, local_cog):
    """AO contact velocity evaluated at collision geometry's saved post-step pose.

    'Before' is the detection/pre_solve relative velocity reconstructed from
    integrator-pre velocities (already updated by PD) at collision geometry's
    post-integration pose. It is NOT the immediate solver-pre-impulse velocity:
    Chipmunk integrates dynamic velocities after collision detection/pre_solve;
    the frozen damping0 makes the dynamic velocity damping factor zero. The
    physical pusher remains kinematic and its PD velocity is not that dynamic
    update. 'After' uses actual post_solve velocities at that same pose. Neither
    quantity by itself establishes later position recovery.
    """
    if event['kind'] != 'agent_object' or not event.get('points'):
        return None, None
    agent_first=event['shapes_order'][0]=='agent'
    n=np.asarray(event['normal'],float)*(-1 if agent_first else 1)  # object -> agent
    p=np.asarray(event['points'][0]['point_b' if agent_first else 'point_a'],float)
    cog=post[2:4]+rotate(local_cog,post[4]);arm=p-cog
    def project(st):
        object_point_velocity=st[7:9]+st[9]*np.asarray((-arm[1],arm[0]))
        return float(np.dot(st[5:7]-object_point_velocity,n))
    return project(pre),project(post)


def episodes(rows, pair_index, band=.8, recovery=.1):
    """Descriptive excursions touching band then returning <= recovery.

    An excursion begins when depth first exceeds recovery, ends when <=recovery,
    and is near1 iff peak>=band. An unended/truncated excursion is censored.
    These analysis bands are not runtime thresholds and do not change labels.
    """
    out=[];begin=None
    for i,row in enumerate(rows):
        depth=row['depths'][pair_index]
        if begin is None and depth>recovery:
            begin=i
        if begin is not None and depth<=recovery:
            segment=rows[begin:i+1];peak=max(x['depths'][pair_index] for x in segment)
            if peak>=band:
                peak_i=max(range(begin,i+1),key=lambda j:rows[j]['depths'][pair_index])
                out.append(dict(pair=PAIRS[pair_index],begin_micro=begin,peak_micro=peak_i,
                    end_micro=i,peak_depth=peak,recovered_observed=True,
                    all_observed_legal=all(x['valid'] for x in segment),
                    peak_control=rows[peak_i]['control'],peak_substep=rows[peak_i]['substep'],
                    peak_time_seconds=rows[peak_i]['time_seconds'],
                    recovery_time_seconds=row['time_seconds'],
                    elapsed_from_peak_seconds=row['time_seconds']-rows[peak_i]['time_seconds']))
            begin=None
    if begin is not None:
        segment=rows[begin:];peak=max(x['depths'][pair_index] for x in segment)
        if peak>=band:
            peak_i=max(range(begin,len(rows)),key=lambda j:rows[j]['depths'][pair_index])
            out.append(dict(pair=PAIRS[pair_index],begin_micro=begin,peak_micro=peak_i,
                end_micro=None,peak_depth=peak,recovered_observed=False,
                all_observed_legal=all(x['valid'] for x in segment),
                peak_control=rows[peak_i]['control'],peak_substep=rows[peak_i]['substep'],
                peak_time_seconds=rows[peak_i]['time_seconds'],recovery_time_seconds=None,
                elapsed_from_peak_seconds=None))
    return out


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    config=read(INPUT/'runtime-config.json');plan=read(INPUT/'plan.json')
    assert sha(INPUT/'runtime-config.json')=='205d313d2feced75db088c9120ede0f9cd9738a059e1af14d584b07f95be643c'
    manifest=[dict(path=str(INPUT/'runtime-config.json'),sha256=sha(INPUT/'runtime-config.json')),
              dict(path=str(INPUT/'plan.json'),sha256=sha(INPUT/'plan.json'))]
    all_rows=[];first=[];summaries=[];recovery_rows=[];per_attempt={}
    group_values=defaultdict(list)
    for a in plan['attempts']:
        ident=a['attempt_id'];mp=INPUT/'collection/attempts'/f'{ident}.json';rp=INPUT/'collection/raw'/f'{ident}.npz'
        m=read(mp);outcome='complete' if m['complete'] else 'terminated'
        assert sha(rp)==m['raw']['sha256']
        manifest.extend([dict(path=str(mp),sha256=sha(mp)),dict(path=str(rp),sha256=sha(rp))])
        local_cog=np.asarray(m['initial_identity']['physics']['object']['center_of_gravity'],float)
        with np.load(rp,allow_pickle=False) as ar:
            payload=zlib.decompress(ar['telemetry_zlib'].tobytes())
            assert hashlib.sha256(payload).digest()==ar['telemetry_sha256'].tobytes()
            data=json.loads(payload);actions=ar['requested_actions'].copy()
        rows=[];elapsed=0.
        for ci,info in enumerate(data['telemetry']):
            action=actions[ci]
            for si,sub in enumerate(info['substeps']):
                if not sub.get('executed'):continue
                elapsed+=float(sub['dt']);pre=np.asarray(sub['pre_state'],float);post=np.asarray(sub['post_state'],float)
                post_events=[e for e in sub['contacts'] if e['phase']=='post_solve']
                pre_events=[e for e in sub['contacts'] if e['phase']=='pre_solve']
                depths=np.asarray(sub['penetrations'],float)
                row=dict(attempt_id=ident,family_id=a['family_id'],letter=a['letter'],scenario=a['scenario'],
                    mode=a['target_mode'],branch=a['branch'],outcome=outcome,
                    control=ci,substep=si,time_seconds=elapsed,phase=a['planned_phase'][ci],
                    valid=bool(sub['valid']),depths=depths.tolist(),
                    target_x=float(action[0]),target_y=float(action[1]),
                    target_error_world=float(np.linalg.norm(action-pre[:2])),
                    target_error_over_D=float(np.linalg.norm(action-pre[:2])/a['geometry_diameter_world_px']),
                    target_change_world=float(np.linalg.norm(action-actions[ci-1])) if ci else 0.,
                    pusher_substep_world=float(np.linalg.norm(post[:2]-pre[:2])),
                    pusher_substep_over_D=float(np.linalg.norm(post[:2]-pre[:2])/a['geometry_diameter_world_px']),
                    object_origin_substep_world=float(np.linalg.norm(post[2:4]-pre[2:4])),
                    object_rotation_substep_deg=float((post[4]-pre[4])*180/math.pi),
                    region=sub.get('pusher_region'),contact_kinds=sorted({e['kind'] for e in post_events}),
                    native_depths={},native_pre_depths={},pre_event_counts={},post_event_counts={},
                    positive_impulses={},normal_relative_before=None,normal_relative_after=None)
                for pi,pair in enumerate(PAIRS):
                    events=[e for e in post_events if e['kind']==KIND[pair]]
                    pre_pair=[e for e in pre_events if e['kind']==KIND[pair]]
                    native=[float(p['distance']) for e in events for p in e['points']]
                    native_pre=[float(p['distance']) for e in pre_pair for p in e['points']]
                    row['native_depths'][pair]=max(0.,-min(native)) if native else None
                    row['native_pre_depths'][pair]=max(0.,-min(native_pre)) if native_pre else None
                    row['pre_event_counts'][pair]=len(pre_pair);row['post_event_counts'][pair]=len(events)
                    row['positive_impulses'][pair]=bool(any(np.linalg.norm(e.get('total_impulse',[0,0]))>0 for e in events))
                    for scope,include in [('all',True),('native_contact',bool(events)),
                                          ('native_pre_contact',bool(pre_pair)),('positive_depth',depths[pi]>0)]:
                        if include:
                            for dimension,value in [('outcome',outcome),('mode',a['target_mode']),('scenario',a['scenario']),('letter',a['letter'])]:
                                group_values[(dimension,value,pair,scope)].append(row)
                    row['depth_'+pair]=float(depths[pi])
                    row['native_depth_'+pair]=row['native_depths'][pair]
                    row['positive_impulse_'+pair]=row['positive_impulses'][pair]
                ao=[e for e in post_events if e['kind']=='agent_object']
                if ao:
                    event=max(ao,key=lambda e:max((-p['distance'] for p in e['points']),default=0))
                    row['normal_relative_before'],row['normal_relative_after']=contact_normal_velocity(event,pre,post,local_cog)
                rows.append(row);all_rows.append(row)
                if not sub['valid']:
                    pairs=[PAIRS[i] for i,x in enumerate(depths) if x>1.+1e-8]
                    first.append(dict(attempt_id=ident,family_id=a['family_id'],letter=a['letter'],scenario=a['scenario'],
                        mode=a['target_mode'],branch=a['branch'],control=ci,substep=si,time_seconds=elapsed,
                        phase=a['planned_phase'][ci],violation_pairs=pairs,depths=depths.tolist(),max_depth=float(max(depths)),
                        region=sub.get('pusher_region'),native_depths=row['native_depths'],
                        native_pre_depths=row['native_pre_depths'],pre_event_counts=row['pre_event_counts'],
                        post_event_counts=row['post_event_counts'],
                        target=action.tolist(),target_error_world=row['target_error_world'],
                        target_error_over_D=row['target_error_over_D'],target_change_world=row['target_change_world'],
                        target_inside_pusher_center_no_overlap_box=bool((action>=22).all() and (action<=489).all()),
                        pusher_substep_world=row['pusher_substep_world'],pusher_substep_over_D=row['pusher_substep_over_D'],
                        normal_relative_before=row['normal_relative_before'],normal_relative_after=row['normal_relative_after'],
                        contact_kinds=row['contact_kinds'],positive_impulses=row['positive_impulses']))
        per_attempt[ident]=rows
        for pi,pair in enumerate(PAIRS):
            for ep in episodes(rows,pi):
                recovery_rows.append(dict(attempt_id=ident,family_id=a['family_id'],letter=a['letter'],scenario=a['scenario'],outcome=outcome,**ep))
        summaries.append(dict(attempt_id=ident,family_id=a['family_id'],letter=a['letter'],scenario=a['scenario'],
            mode=a['target_mode'],branch=a['branch'],outcome=outcome,controls=m['cost']['actual_controls'],
            legal_controls=m['cost']['actual_valid_controls'],substeps=len(rows),
            maximum_depths={pair:max((x['depths'][i] for x in rows),default=0.) for i,pair in enumerate(PAIRS)},
            maximum_target_error_world=max((x['target_error_world'] for x in rows),default=0.),
            maximum_pusher_substep_world=max((x['pusher_substep_world'] for x in rows),default=0.)))
    grouped=[]
    for (dim,value,pair,scope),rr in sorted(group_values.items()):
        pi=PAIRS.index(pair)
        grouped.append(dict(dimension=dim,value=value,pair=pair,scope=scope,substeps=len(rr),
            families=len({x['family_id'] for x in rr}),attempts=len({x['attempt_id'] for x in rr}),
            custom_depth=stats(x['depths'][pi] for x in rr),native_depth=stats(x['native_depths'][pair] for x in rr),
            native_pre_depth=stats(x['native_pre_depths'][pair] for x in rr),
            target_error_world=stats(x['target_error_world'] for x in rr),
            target_change_world=stats(x['target_change_world'] for x in rr),
            pusher_substep_world=stats(x['pusher_substep_world'] for x in rr),
            pusher_substep_over_D=stats(x['pusher_substep_over_D'] for x in rr)))
    sensitivity=[]
    for band in (.7,.8,.9):
        for low in (.1,.2):
            for pi,pair in enumerate(PAIRS):
                ep=[dict(attempt_id=k,**e) for k,rr in per_attempt.items() for e in episodes(rr,pi,band,low)]
                good=[x for x in ep if x['recovered_observed'] and x['all_observed_legal'] and x['peak_depth']<=1.+1e-8]
                sensitivity.append(dict(near_band_lower=band,recovery_upper=low,pair=pair,excursions=len(ep),
                    recovered_legal_excursions=len(good),recovered_attempts=len({x['attempt_id'] for x in good}),
                    censored_excursions=sum(not x['recovered_observed'] for x in ep)))
    recovered=[x for x in recovery_rows if x['recovered_observed'] and x['all_observed_legal'] and x['peak_depth']<=1.+1e-8]
    aw=[x for x in first if 'AW' in x['violation_pairs']]
    legal_samples=sorted(recovered,key=lambda x:(-x['peak_depth'],x['attempt_id'],x['peak_micro']))[:8]
    closest=sorted(first,key=lambda x:(abs(x['max_depth']-1),x['attempt_id']))[:6]
    facts=dict(kind='saved-contact-diagnosis.v1',run_id=config['run_id'],new_physics_calls=0,model_calls=0,optimizer_updates=0,
        normal_velocity_timing='Before=detection/pre_solve relative velocity reconstructed from PD-updated integrator-pre velocities and collision post-integration pose; not immediate solver-pre-impulse. Dynamic velocity integration occurs after pre_solve and damping0 zeros its damping contribution. After=actual post_solve relative velocity at same collision pose; no later position integration is present in that step.',
        raw_records=120,micro_records=len(all_rows),attempts=summaries,
        first_violation_count=len(first),first_violation_maxdepth=stats(x['max_depth'] for x in first),
        first_violation_pair_counts=dict(Counter(p for x in first for p in x['violation_pairs'])),
        first_violation_mode_counts=dict(Counter(x['mode'] for x in first)),
        first_main_control3_violations=sum(x['control']==3 for x in first),
        all_regions=dict(Counter(x['region'] for x in first)),
        first_violation_native_depth_error_by_pair={pair:stats(abs(x['depths'][pi]-x['native_depths'][pair]) for x in first if pair in x['violation_pairs'] and x['native_depths'][pair] is not None) for pi,pair in enumerate(PAIRS)},
        first_AO_relative_velocity_before=stats(x['normal_relative_before'] for x in first if 'AO' in x['violation_pairs']),
        first_AO_relative_velocity_after=stats(x['normal_relative_after'] for x in first if 'AO' in x['violation_pairs']),
        first_AO_relative_normal_response=dict(
            observations=64,
            before_closing=sum(x['normal_relative_before']<0 for x in first if 'AO' in x['violation_pairs']),
            after_abs_le_1e_6=sum(abs(x['normal_relative_after'])<=1e-6 for x in first if 'AO' in x['violation_pairs']),
            after_closing_below_minus_1e_6=sum(x['normal_relative_after']<-1e-6 for x in first if 'AO' in x['violation_pairs']),
            after_separating_above_1e_6=sum(x['normal_relative_after']>1e-6 for x in first if 'AO' in x['violation_pairs'])),
        observed_recovery_definition='Descriptive depth excursion above0.1, touches[0.8,1], subsequently returns<=0.1 in saved legal rows. Not runtime gate, not proof that any >1 termination would recover.',
        observed_recovered_excursions=len(recovered),observed_recovered_attempts=len({x['attempt_id'] for x in recovered}),
        observed_recovered_families=len({x['family_id'] for x in recovered}),
        recovery_pair_counts=dict(Counter(x['pair'] for x in recovered)),
        representative_recovery_cases=legal_samples,closest_to_threshold_censored=closest,
        AW_terminations=aw,AW_by_scenario=dict(Counter(x['scenario'] for x in aw)),
        AW_negative_corner_count=sum(x['scenario']=='negative' and any(v in (0.,512.) for v in x['target']) for x in aw),
        all_AW_targets_outside_no_overlap_pusher_center_box=all(not x['target_inside_pusher_center_no_overlap_box'] for x in aw),
        groups=grouped,recovery_sensitivity=sensitivity,
        interpretation='All74 >1 observations are right-censored by termination. Observed recovery is only below the gate and cannot establish counterfactual >1 behavior. Different modes/scenes are descriptive, not matched causal contrasts.')
    write_json('CONTACT_DIAGNOSIS.json',facts)
    write_json('INPUT_MANIFEST.json',dict(inputs=manifest,source_bindings=[dict(**b,current_sha256=sha(b['path'])) for b in config['bindings']],script_sha256=sha(__file__),new_physics_calls=0,model_calls=0,optimizer_updates=0))
    write_csv('first-violations.csv',first);write_csv('attempt-depths.csv',summaries)
    write_csv('observed-near1-recovery-episodes.csv',recovery_rows)
    write_csv('micro-contact-diagnostics.csv.gz',all_rows)
    assert sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file())<8*1024**2
    print(json.dumps({k:facts[k] for k in ('micro_records','first_violation_maxdepth','first_violation_pair_counts','AW_by_scenario','AW_negative_corner_count','all_AW_targets_outside_no_overlap_pusher_center_box','observed_recovered_excursions','observed_recovered_attempts','observed_recovered_families','recovery_pair_counts','representative_recovery_cases','first_AO_relative_velocity_before','first_AO_relative_velocity_after')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
