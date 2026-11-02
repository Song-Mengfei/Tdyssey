"""Cold seen_val-only coverage and zero-baseline counter audit; no simulation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.native_rgb_stage2b_coverage import (_blank_coverage,_record_contact,_normalized_counts,
    _coverage_differences,_summarize)
from scripts.native_rgb_stage2a_coverage import LETTERS,_write_json,_write_csv,load_json,sha256,_reach_ids


def build_coverage(stage,output=None):
    import numpy as np
    from tdyssey.native_rgb_motion_statistics import transition_rows
    from tdyssey.native_rgb_data import _write_coverage
    stage=Path(stage).resolve();output=Path(output) if output else stage/'report/coverage'
    if (stage/'COLLECTION_RUNNING.lock').exists():raise RuntimeError('No hot-data review')
    manifest=load_json(stage/'data/manifest.json');identity=load_json(stage/'source-identity.json')
    if manifest['stage']!='2C' or manifest['split']!='seen_val':raise ValueError('Independent seen_val only')
    prior_path=stage/'source-snapshot/frozen-oldTRAIN-prior.json';prior=load_json(prior_path)
    if not any(x['snapshot']=='frozen-oldTRAIN-prior.json' and x['sha256']==sha256(prior_path) for x in identity['files']):
        raise ValueError('OldTRAIN prior must remain frozen and separate')
    coverage=_blank_coverage();planned={l:Counter() for l in LETTERS};trajectories=[];audits=[];seen=set()
    baseline=load_json(stage/'counter-baseline.json')
    if baseline['TRAIN_dynamic_counts_inherited'] or any(v for d in baseline['actual_hit_counts'].values() for v in d.values()) or any(baseline['planned_attempts'].values()):
        raise ValueError('Validation counters must have a zero baseline')
    for entry in manifest['entries']:
        tid=entry['trajectory_id'];letter=entry['shape']
        if entry['split']!='seen_val' or letter not in LETTERS or tid in seen:raise ValueError('Split/identity leakage')
        seen.add(tid);p=stage/'data'/entry['file'];m=stage/'data'/entry['metadata_file']
        if sha256(p)!=entry['sha256'] or sha256(m)!=entry['metadata_sha256']:raise ValueError('Raw identities changed')
        meta=load_json(m);scene=meta['scene'];steps=meta['steps'];inputs=meta['selection_inputs'];rows=coverage[letter]
        actual={str(r['id']):r['contact_control_count'] for r in rows};attempts=_normalized_counts(dict(planned[letter]),rows)
        issues=[]
        if inputs.get('counter_split')!='seen_val' or 'formal_train_contact_controls_before' in inputs:issues.append('counter_split_leak')
        if _normalized_counts(inputs.get('seen_val_contact_controls_before'),rows)!=actual:issues.append('own_actual_prefix')
        if _normalized_counts(inputs.get('planned_initial_attempts_before'),rows)!=attempts:issues.append('own_attempt_prefix')
        if _normalized_counts(scene['selection_evidence']['planned_attempts_before'],rows)!=attempts:issues.append('scene_attempt_prefix')
        if scene['selection_version']!='motion-room-planned-attempts-v2':issues.append('selector_changed')
        with np.load(p,allow_pickle=False) as source:arrays={k:source[k] for k in source.files}
        n=len(arrays['actions']);stats=transition_rows(entry,arrays,steps);arc=int(scene['staging_segment'])
        if n:
            if not inputs['planned_attempt_counted']:issues.append('started_not_counted')
            planned[letter][arc]+=1
            if _normalized_counts(inputs.get('planned_initial_attempts_after_start'),rows)!=_normalized_counts(dict(planned[letter]),rows):issues.append('after_start_count')
        audits.append(dict(trajectory_id=tid,letter=letter,matches=not issues,issues=issues,selected_initial_arc=arc,
            planned_before=attempts,actual_before=actual))
        hit_arcs=set();classes=Counter();contact_classes=Counter();gaps=Counter();contact_gaps=Counter();approaches=Counter()
        offsets=arrays['contact_offsets'];points=arrays['contact_points'];angles=arrays['contact_angles']
        for t,(stat,step) in enumerate(zip(stats,steps)):
            begin,end=int(offsets[t]),int(offsets[t+1]);approach=str(step.get('approach_kind','unknown'))
            hit_arcs|=_record_contact(letter,rows,points[begin:end],angles[begin:end],tid,stat['motion_class']=='moving',approach)
            classes[stat['actual_spatial_class_before']]+=1;gaps[stat['gap_band_before']]+=1
            if stat['contact']:
                contact_classes[stat['actual_spatial_class_before']]+=1;contact_gaps[stat['gap_band_before']]+=1;approaches[approach]+=1
        trajectories.append(dict(batch='2C',letter=letter,trajectory_id=tid,selected_initial_arc=arc,
            planned_initial_arc=arc,planned_initial_arc_actually_hit=arc in hit_arcs,actual_hit_arc_ids=sorted(hit_arcs),
            actual_contact_controls=sum(r['contact'] is True for r in stats),recorded_controls=n,recorded_contact_points=len(points),
            initial_reachable_arc_ids=sorted(_reach_ids(scene['reachability'])),
            actual_control_spatial_class_counts=dict(classes),actual_contact_control_spatial_class_counts=dict(contact_classes),
            actual_control_gap_band_counts=dict(gaps),actual_contact_control_gap_band_counts=dict(contact_gaps),
            requested_approach_contact_control_counts=dict(approaches),requested_spatial_class=entry['spatial_class'],
            requested_gap_world_px=entry['gap_requested_world_px'],initial_wall_gaps=scene['actual_wall_gaps'],
            selection_version=scene['selection_version']))
    saved=load_json(stage/'contact-coverage.json')['segments'];differences=_coverage_differences(coverage,saved,'seen_val only')
    summaries,details,conflicts=_summarize('seen_val',coverage,trajectories,prior)
    report=dict(kind='stage2c-independent-seen-val-coverage.v1',split='seen_val',per_letter=details,
        passed=not differences and not conflicts and all(x['matches'] for x in audits),
        zeroed_baseline_verified=True,TRAIN_counters_inherited=False,TRAIN_modified=False,
        counter_prefix_audit=dict(trajectories=len(audits),mismatching_trajectories=sum(not x['matches'] for x in audits)),
        saved_coverage_differences=differences,reachability_conflicts=conflicts,
        final_planned_attempts={l:dict(v) for l,v in planned.items()},independent_trajectories=len(trajectories),
        frozen_old_TRAIN_prior=dict(path=str(prior_path),sha256=sha256(prior_path)),
        qualification='Real float32 contact-point replay; points/controls are not independent trajectories; unknown arcs remain unknown. Validation evidence never flows into TRAIN.',
        physics_controls=0,model_calls=0,source_sha256=sha256(__file__))
    output.mkdir(parents=True,exist_ok=False);_write_json(output/'coverage-summary.json',report)
    _write_json(output/'counter-prefix-audit.json',audits);_write_csv(output/'coverage-summary.csv',summaries)
    _write_json(output/'trajectory-contacts.json',trajectories)
    _write_coverage(output/'svg',coverage,prior['canonical_reachability'])
    lines=['# Independent seen_val actual coverage','',report['qualification'],'',
        '|Letter|H/all|Canonical R hit/R|K lower bound|Unknown U|Closed C|Independent contact traj/all|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        s=details[row['letter']]['sets'];lines.append(f"|{row['letter']}|{len(s['actual_hit'])}/{row['all_bins']}|{len(set(s['actual_hit'])&set(s['canonical_known_reachable']))}/{len(s['canonical_known_reachable'])}|{len(s['reachable_lower_bound'])}|{len(s['unconfirmed'])}|{len(s['proven_closed_hole'])}|{row['independent_contact_trajectories']}/{row['independent_trajectories']}|")
    lines+=['',f"Counter mismatches: {report['counter_prefix_audit']['mismatching_trajectories']}; saved coverage differences: {len(differences)}; closed-hole conflicts: {len(conflicts)}."]
    (output/'coverage-summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True);p.add_argument('--output');a=p.parse_args()
    r=build_coverage(a.stage,a.output);print(json.dumps(dict(passed=r['passed'],counter_prefix_audit=r['counter_prefix_audit'],saved_differences=len(r['saved_coverage_differences']),conflicts=len(r['reachability_conflicts']))));raise SystemExit(0 if r['passed'] else 1)
