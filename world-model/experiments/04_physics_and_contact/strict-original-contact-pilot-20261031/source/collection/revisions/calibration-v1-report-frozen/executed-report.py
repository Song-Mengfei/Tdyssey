"""Saved-only strict-original motion/contact reports and guarded 10Hz videos.

No environment construction, stepping, model imports, inference or training.
All attempts remain in denominators; legal H1/H4 windows are reported separately.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import zlib

import numpy as np

from .native_rgb_original_physics import PROTOCOL, PROTOCOL_SHA256
from .native_rgb_data_eligibility import (ROOT, guard_trajectory, sha256,
    validate_window, verify_strict_original_dataset)
from .original_contact_pilot_resources import ResourceReservation
from .original_contact_pilot_report import centers, wall_gap, _actual_arc

LETTERS = tuple('TLOCXB')
TRANSLATION_BINS = [0., .5, 1., 2., 4., 8., 16., None]
ROTATION_BINS = [0., 1., 5., 10., 20., 45., 90., None]
TIME_FIELDS = ('AO_positive_seconds', 'AO_pair_seconds', 'OW_pair_seconds',
    'AW_pair_seconds', 'contact_moving_seconds', 'contact_still_seconds',
    'wall_only_seconds', 'wall_without_positive_AO_seconds', 'AO_pair_zero_impulse_seconds',
    'empty_search_seconds', 'noncontact_moving_seconds',
    'noncontact_still_seconds', 'boundary_seconds', 'invalid_substep_seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_attempt(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    payload = zlib.decompress(values.pop('telemetry_zlib').tobytes())
    if hashlib.sha256(payload).digest() != values.pop('telemetry_sha256').tobytes():
        raise ValueError('Actual saved telemetry SHA differs')
    values.update(json.loads(payload))
    return values


def _positive(event):
    return float(np.linalg.norm(event.get('total_impulse', [0., 0.]))) > 0


def micro_metrics(letter, info):
    """Use actual dt and positive post-solve AO impulse, never arbiter counts."""
    rows = [r for r in info.get('substeps', []) if r.get('executed')]
    result = dict(seconds=0., actual_substeps=len(rows), **{key: 0. for key in TIME_FIELDS})
    result.update(max_penetration_world_px_OW_AO_AW=[0., 0., 0.], arc_hits=[])
    for index, row in enumerate(rows):
        dt = float(row['dt'])
        if abs(dt-.01) > 1e-12:
            raise ValueError('Saved strict-original substep is not0.01 seconds')
        before = np.asarray(row.get('pre_state', row.get('integrator_pre_state')), float)
        after = np.asarray(row['post_state'], float)
        distance = float(np.linalg.norm(centers(letter, np.array([before, after]))[1]-
                                        centers(letter, np.array([before, after]))[0]))
        angle = abs(float(after[4]-before[4]))*180/math.pi
        moving = distance*96/512 >= dt or angle >= dt  # >=1 imagepx/s OR1deg/s
        pusher_moving = np.linalg.norm(after[:2]-before[:2])*96/512 >= dt
        events = [e for e in row.get('contacts', []) if e.get('phase') == 'post_solve']
        kinds = {e['kind'] for e in events}
        ao_events = [e for e in events if e['kind'] == 'agent_object' and _positive(e)]
        ao, ao_pair, ow = bool(ao_events), 'agent_object' in kinds, 'object_wall' in kinds
        result['seconds'] += dt
        flags = dict(AO_positive_seconds=ao, AO_pair_seconds='agent_object' in kinds,
            OW_pair_seconds=ow, AW_pair_seconds='agent_wall' in kinds,
            contact_moving_seconds=ao and moving, contact_still_seconds=ao and not moving,
            wall_only_seconds=ow and not ao_pair,
            wall_without_positive_AO_seconds=ow and not ao,
            AO_pair_zero_impulse_seconds=ao_pair and not ao,
            empty_search_seconds=not ao_pair and not ow and pusher_moving,
            noncontact_moving_seconds=not ao_pair and not ow and moving,
            noncontact_still_seconds=not ao_pair and not ow and not moving,
            boundary_seconds=min(wall_gap(letter, np.array([before, after]))) <= 24.,
            invalid_substep_seconds=row.get('valid') is False)
        for key, flag in flags.items():
            result[key] += dt*bool(flag)
        depths = np.asarray(row.get('penetrations', [0., 0., 0.]), float)
        if depths.shape != (3,) or not np.isfinite(depths).all():
            raise ValueError('Saved penetration labels are malformed')
        result['max_penetration_world_px_OW_AO_AW'] = np.maximum(
            result['max_penetration_world_px_OW_AO_AW'], depths).tolist()
        hits = set()
        for event in ao_events:
            object_first = event['shapes_order'][0] == 'object'
            for point in event.get('points', []):
                world = point.get('object_world') or point.get('point_a' if object_first else 'point_b')
                if world is None:
                    continue
                delta = np.asarray(world)-after[2:4]
                c, s = math.cos(after[4]), math.sin(after[4])
                arc = _actual_arc(letter, [c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1]])
                hits.add((-1, -1) if arc is None else (arc['ring'], arc['arc_bin']))
        if ao and not hits:
            hits.add((-1, -1))
        result['arc_hits'].extend(dict(ring=ring, arc_bin=arc, seconds=dt, microstep=index)
                                  for ring, arc in sorted(hits))
    return result


def trajectory_metrics(metadata, data):
    slot = metadata['slot']
    states, valid = data['states'], data['control_valid']
    n = len(data['requested_actions'])
    if (states.shape != (n+1, 15) or data['rgb'].shape != (n+1, 3, 96, 96)
            or len(data['telemetry']) != n or len(data['decisions']) != n):
        raise ValueError('Actual saved trajectory timeline lengths differ')
    positions = centers(slot['letter'], states)
    base = {key:slot[key] for key in ('letter', 'phase', 'split')}
    base.update(trajectory_id=slot['slot_id'],family_id=slot.get('family_id'),
        seed=slot.get('seed'),scenario=slot.get('scenario'),location=slot.get('location'),
        trajectory_complete=bool(metadata.get('complete')))
    controls, arcs, windows = [], [], []
    for index, info in enumerate(data['telemetry']):
        micro = micro_metrics(slot['letter'], info)
        h1 = bool(valid[index])
        if h1:
            try:
                validate_window(data['frame_indices'],valid,index,index+1,
                    control_time_seconds=data['control_time_seconds'],segment_ids=data['segment_ids'])
            except ValueError:
                h1 = False
        distance = float(np.linalg.norm(positions[index+1]-positions[index]))
        angle = float(states[index+1,4]-states[index,4])*180/math.pi
        moving = distance*96/512 >= .1 or abs(angle) >= .1
        ao, ao_pair, ow = micro['AO_positive_seconds'] > 0, micro['AO_pair_seconds'] > 0, micro['OW_pair_seconds'] > 0
        category = ('invalid' if not valid[index] else 'contact_moving' if ao and moving
            else 'contact_still' if ao else 'contact_pair_zero_impulse' if ao_pair else 'wall_only' if ow
            else 'empty_search' if micro['empty_search_seconds'] else
            'free_moving' if moving else 'free_still')
        row = dict(base, control=index, valid=bool(valid[index]),h1_eligible=h1,
            category=category,planned_phase=data['decisions'][index].get('phase'),
            translation_world_px=distance,translation_image_px=distance*96/512,
            signed_rotation_deg=angle,rotation_deg=abs(angle),
            **{key:value for key,value in micro.items() if key!='arc_hits'})
        controls.append(row)
        arcs.extend(dict(base,control=index,control_valid=bool(valid[index]),**hit)
                    for hit in micro['arc_hits'])
    for horizon in (1,4):
        for start in range(n-horizon+1):
            try:
                validate_window(data['frame_indices'],valid,start,start+horizon,
                    control_time_seconds=data['control_time_seconds'],segment_ids=data['segment_ids'])
                legal=True
            except ValueError:
                legal=False
            history_legal=False
            if start>=3 and legal:
                try:
                    validate_window(data['frame_indices'],valid,start-3,start+horizon,
                        control_time_seconds=data['control_time_seconds'],segment_ids=data['segment_ids'])
                    history_legal=True
                except ValueError:
                    pass
            distance=float(np.linalg.norm(positions[start+horizon]-positions[start]))
            angle=float(states[start+horizon,4]-states[start,4])*180/math.pi
            windows.append(dict(base,start_control=start,horizon=horizon,legal=legal,
                seconds=float(data['control_time_seconds'][start+horizon]-data['control_time_seconds'][start]),
                history4_legal=history_legal,pending_training_approval=True,
                candidate=history_legal and slot['phase'] in ('formal','supplement'),
                translation_world_px=distance,translation_image_px=distance*96/512,
                signed_rotation_deg=angle,rotation_deg=abs(angle),
                AO_positive=any(r['AO_positive_seconds']>0 for r in controls[start:start+horizon]),
                boundary=any(r['boundary_seconds']>0 for r in controls[start:start+horizon])))
    return controls,windows,arcs


def distribution(values, edges):
    values=np.asarray(values,float)
    finite=values[np.isfinite(values)]
    quantile_names=('P10','P25','P50','P75','P90','P95','max')
    out=dict(N=len(values),measurable=len(finite),unknown=len(values)-len(finite),
        bins=edges,bin_counts=np.histogram(finite,bins=[np.inf if e is None else e for e in edges])[0].tolist())
    out.update(dict(zip(quantile_names,map(float,np.quantile(finite,[.1,.25,.5,.75,.9,.95,1]))))
        if len(finite) else dict.fromkeys(quantile_names))
    return out


def summarize(controls, windows, metadata):
    groups=[]
    phases=sorted({m['slot']['phase'] for m in metadata})
    for phase in phases:
        for letter in LETTERS:
            attempts=[m for m in metadata if m['slot']['phase']==phase and m['slot']['letter']==letter]
            rows=[r for r in controls if r['phase']==phase and r['letter']==letter]
            if not attempts:
                continue
            observed_seconds=sum(r['seconds'] for r in rows)
            known_controls=Counter(r['trajectory_id'] for r in rows)
            known_substeps=Counter()
            known_seconds=Counter()
            for row in rows:
                known_substeps[row['trajectory_id']]+=row['actual_substeps']
                known_seconds[row['trajectory_id']]+=row['seconds']
            recorded_controls=sum(m.get('reservation',{}).get('actual_controls',known_controls[m['slot']['slot_id']]) for m in attempts)
            recorded_substeps=sum(m.get('reservation',{}).get('actual_substeps',known_substeps[m['slot']['slot_id']]) for m in attempts)
            seconds=sum(m.get('reservation',{}).get('simulated_seconds',known_seconds[m['slot']['slot_id']]) for m in attempts)
            unavailable_controls=max(0,recorded_controls-len(rows))
            legal_seconds=sum(r['seconds'] for r in rows if r['valid'])
            item=dict(phase=phase,letter=letter,attempts=len(attempts),
                complete_attempts=sum(bool(m.get('complete')) for m in attempts),
                failed_or_partial_attempts=sum(not bool(m.get('complete')) for m in attempts),
                actual_controls=recorded_controls,saved_observed_controls=len(rows),
                unavailable_saved_control_metrics=unavailable_controls,valid_controls=sum(r['valid'] for r in rows),
                invalid_controls=sum(not r['valid'] for r in rows),actual_substeps=recorded_substeps,
                saved_observed_substeps=sum(r['actual_substeps'] for r in rows),simulated_seconds=seconds,
                saved_observed_simulated_seconds=observed_seconds,valid_control_seconds=legal_seconds,
                category_control_counts=dict(Counter(r['category'] for r in rows)),
                category_control_fractions_all_attempted_controls={k:v/recorded_controls for k,v in Counter(r['category'] for r in rows).items()} if recorded_controls else {},
                independent_families=len({m['slot'].get('family_id') for m in attempts}),
                independent_scene_seeds=len({m['slot'].get('seed') for m in attempts}),
                incomplete_or_failed_attempt_fraction=sum(not bool(m.get('complete')) for m in attempts)/len(attempts),
                original_invalid_attempt_fraction=sum(bool(m.get('expected_original_invalidity')) or m.get('reservation',{}).get('status')=='invalid_original_state' for m in attempts)/len(attempts),
                invalid_control_fraction=sum(not r['valid'] for r in rows)/recorded_controls if recorded_controls else None,
                unavailable_control_fraction=unavailable_controls/recorded_controls if recorded_controls else None,
                max_penetration_world_px_OW_AO_AW=[max((r['max_penetration_world_px_OW_AO_AW'][j] for r in rows),default=None) for j in range(3)])
            for key in TIME_FIELDS:
                item[key]=sum(r[key] for r in rows)
                item[key+'_fraction_all_actual_time']=item[key]/seconds if seconds else None
                item[key+'_fraction_valid_control_time']=sum(r[key] for r in rows if r['valid'])/legal_seconds if legal_seconds else None
            item['motion']={}
            for split in sorted({m['slot']['split'] for m in attempts}):
                for horizon in (1,4):
                    win=[w for w in windows if w['phase']==phase and w['letter']==letter and w['split']==split and w['horizon']==horizon and w['legal']]
                    item['motion'][f'{split}_H{horizon}']=dict(legal_windows=len(win),
                        windows_with_legal_history4=sum(w['history4_legal'] for w in win),
                        translation_world_px=distribution([w['translation_world_px'] for w in win],[None if x is None else x*512/96 for x in TRANSLATION_BINS]),
                        translation_image_px=distribution([w['translation_image_px'] for w in win],TRANSLATION_BINS),
                        rotation_deg=distribution([w['rotation_deg'] for w in win],ROTATION_BINS),
                        trajectory_H1_rotation_at_least5_count=len({w['trajectory_id'] for w in win if horizon==1 and w['rotation_deg']>=5}))
            groups.append(item)
    formal_trigger={}
    for letter in LETTERS:
        rows=[r for r in controls if r['phase']=='formal' and r['letter']==letter and r['h1_eligible']]
        weak=sum(r['translation_image_px']<.5 for r in rows)
        rotating={r['trajectory_id'] for r in rows if r['rotation_deg']>=5}
        formal_trigger[letter]=dict(valid_formal_H1_controls=len(rows),weak_H1_controls=weak,
            weak_fraction=weak/len(rows) if rows else None,trajectory_H1_rotation_at_least5_count=len(rotating),
            trigger=(weak/len(rows)>.5 or not rotating) if rows else None,
            rule='strictly >50% valid formal H1 translation<0.5imagepx OR no trajectory with valid H1 rotation>=5deg',
            denominator='all valid formal H1 controls including early controls without four-frame history; excludes calibration/supplement',
            automatic_release=False)
    return dict(groups=groups,formal_supplement_trigger=formal_trigger)


def _write_json(lease,path,value):
    lease.write_bytes(path,(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8'))


def _write_csv(lease,path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    memory=io.StringIO(newline='')
    writer=csv.DictWriter(memory,fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key:json.dumps(value,separators=(',',':')) if isinstance(value,(list,dict)) else value for key,value in row.items()})
    lease.write_bytes(path,memory.getvalue().encode('utf8'))


def _source_root(roots, explicit=None):
    candidates=[Path(explicit)] if explicit else [Path(p) for p in roots]
    for root in candidates:
        if all((root/relative).is_file() and sha256(root/relative)==digest
               for relative,digest in PROTOCOL['source_sha256'].items()):
            return root.resolve()
    raise ValueError('Declared storage roots/--source-root do not contain unchanged pinned physics sources')


def make_candidate_manifest(output, metadata_paths, source_root):
    sources=[ROOT/'tdyssey'/name for name in ('native_rgb_original_physics.py',
        'strict_original_contact_pilot.py','native_rgb_data_eligibility.py','strict_original_contact_report.py',
        'original_contact_pilot.py','original_contact_pilot_resources.py','native_rgb_env.py',
        'native_rgb_r2a_env.py','envs/official.py')]
    sources += [ROOT/'manifests/official_dp.json']
    sources += [source_root/relative for relative in PROTOCOL['source_sha256']]
    entries,rejected=[],[]
    for path in metadata_paths:
        meta=read(path);raw=path.parent/(meta.get('raw') or {}).get('file','raw.npz')
        if not raw.is_file() or meta.get('initial_identity') is None or not meta.get('contiguous_frames'):
            rejected.append(dict(metadata_file=str(path.relative_to(output)),reason='No raw/complete initial identity/contiguous rendered timeline',training_eligible=False))
            continue
        slot=meta['slot']
        entries.append(dict(file=str(raw.relative_to(output)),sha256=sha256(raw),
            metadata_file=str(path.relative_to(output)),metadata_sha256=sha256(path),
            trajectory_id=slot['slot_id'],letter=slot['letter'],phase=slot['phase'],split=slot['split'],
            family_id=slot.get('family_id'),seed=slot.get('seed'),complete=bool(meta.get('complete'))))
    return dict(kind='strict-original-contact-reference.v1',root='.',protocol_sha256=PROTOCOL_SHA256,
        control_dt=.1,old_physics_data=False,training_eligible=False,training_authorized=False,
        pending_user_approval=True,human_approval_evidence=None,upstream_source_root=str(source_root),
        source_bindings=[dict(path=str(path.resolve()),sha256=sha256(path)) for path in sources],
        entries=entries,rejected_attempts=rejected,
        validity='Only continuous exact10Hz legal windows; partial invalid terminal transition retained and excluded',
        collection_physics_modified=False,model_calls=0,optimizer_updates=0)


def _video(path,meta,data,lease):
    import cv2
    import imageio_ffmpeg
    limit=4*1024**2
    margin=1024**2
    lease.ensure_capacity(limit+margin)
    old_size=path.stat().st_size if path.exists() else 0
    path.parent.mkdir(parents=True,exist_ok=True)
    size=(576,640)
    process=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','bgr24','-s',f'{size[0]}x{size[1]}','-r','10','-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',
        '-fs',str(limit),str(path)],stdin=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    error=None
    try:
        for index,image in enumerate(data['rgb']):
            canvas=np.full((640,576,3),248,np.uint8)
            picture=cv2.cvtColor(image.transpose(1,2,0),cv2.COLOR_RGB2BGR)
            canvas[146:626,48:528]=cv2.resize(picture,(480,480),interpolation=cv2.INTER_NEAREST)
            lines=[f"ORIGINAL100Hz solver10 | {meta['slot']['letter']} | {meta['slot']['phase']}",
                meta['slot']['slot_id'],
                f"saved t={float(data['control_time_seconds'][index]):.3f}s | frame {index}/{len(data['rgb'])-1}",
                'REAL SAVED RGB 10fps | no interpolation / skipped frames',
                'invalid final partial retained' if index and not data['control_valid'][index-1] else 'original controller; no model prediction']
            for line,text in enumerate(lines):
                cv2.putText(canvas,text,(10,23+25*line),cv2.FONT_HERSHEY_SIMPLEX,.43,(35,45,55),1,cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
    except BaseException as exc:
        error=exc
    finally:
        try:process.stdin.close()
        except BrokenPipeError:pass
        code=process.wait()
        if path.exists():lease.commit_external_write(path,previous_size=old_size)
    if error is not None or code or path.stat().st_size>limit+margin:
        raise RuntimeError('Bounded encoder failed; preserve actual partial video') from error
    capture=cv2.VideoCapture(str(path));frames=0;fps=capture.get(cv2.CAP_PROP_FPS)
    while True:
        ok,_=capture.read()
        if not ok:break
        frames+=1
    capture.release()
    if frames!=len(data['rgb']) or abs(fps-10)>1e-6:
        raise RuntimeError('Video actual full decode/frame rate differs')
    return dict(file=str(path),sha256=sha256(path),bytes=path.stat().st_size,frames=frames,fps=fps,
        actual_simulated_seconds=float(data['control_time_seconds'][-1]),video_seconds=frames/fps,
        complete=bool(meta.get('complete')),final_partial_control=bool(len(data['control_valid']) and not data['control_valid'][-1]),
        timing_note='Every saved actual frame shown once at10fps. Complete controlframes are10Hz; final initial/end hold adds0.1s. '
            'An invalid terminal partial is shown and labelled at its actual timestamp; display holds it one frame, never stretches a legal H1/H4 metric.')


def build_report(output, *, resource_policy, shared_roots, videos=False, source_root=None):
    output=Path(output).resolve();destination=output/'report'
    meta_paths=sorted((output/'trajectories').glob('*/metadata.json'))
    if not meta_paths:
        raise ValueError('No actual saved strict-original attempt metadata')
    upstream=_source_root(shared_roots,source_root)
    metadata=[];controls=[];windows=[];arcs=[];sources={};attempts=[]
    for path in meta_paths:
        meta=read(path);metadata.append(meta)
        raw=path.parent/(meta.get('raw') or {}).get('file','raw.npz')
        row=dict(trajectory_id=meta['slot']['slot_id'],phase=meta['slot']['phase'],letter=meta['slot']['letter'],
            split=meta['slot']['split'],complete=bool(meta.get('complete')),failure=meta.get('failure'),
            raw_present=raw.is_file(),metadata_file=str(path),raw_file=str(raw),
            metadata_sha256=sha256(path),raw_sha256=sha256(raw) if raw.is_file() else None,
            observed_status=meta.get('reservation',{}).get('status'),training_eligible=False)
        attempts.append(row)
        if raw.is_file():
            if sha256(raw)!=meta['raw']['sha256']:
                raise ValueError('Actual raw SHA differs from collector metadata')
            data=load_attempt(raw)
            c,w,a=trajectory_metrics(meta,data)
            controls.extend(c);windows.extend(w);arcs.extend(a)
            sources[meta['slot']['slot_id']]=(meta,raw)
    # Any claim/ledger attempt without metadata remains visible; no quota filling.
    ledger_attempts={}
    ledgers=[]
    for path in sorted((output/'runs').glob('*/ledger.json')):
        ledger=read(path);ledgers.append(dict(path=str(path),sha256=sha256(path),
            actual_controls=ledger.get('actual_controls'),actual_substeps=ledger.get('actual_substeps'),
            simulated_seconds=ledger.get('simulated_seconds'),model_calls=ledger.get('model_calls'),
            optimizer_updates=ledger.get('optimizer_updates')))
        for row in ledger.get('attempts',[]):
            if row['slot_id'] in ledger_attempts:
                raise ValueError('Duplicate actual attempted scene across run ledgers')
            ledger_attempts[row['slot_id']]=row
    known={m['slot']['slot_id'] for m in metadata}
    unreported=[row for key,row in ledger_attempts.items() if key not in known]
    for row in unreported:
        metadata.append(dict(slot=dict(slot_id=row['slot_id'],phase=row['phase'],
            letter=row['letter'],split=row['split']),complete=False,reservation=row,
            missing_saved_metadata=True))
        attempts.append(dict(trajectory_id=row['slot_id'],phase=row['phase'],letter=row['letter'],
            split=row['split'],complete=False,failure='Ledger attempt lacks final saved metadata',
            raw_present=False,metadata_file=None,raw_file=None,observed_status=row.get('status'),
            ledger_actual_controls=row.get('actual_controls'),ledger_actual_substeps=row.get('actual_substeps'),
            training_eligible=False))
    summary=summarize(controls,windows,metadata)
    ledger_controls=sum(row.get('actual_controls',0) for row in ledger_attempts.values())
    ledger_substeps=sum(row.get('actual_substeps',0) for row in ledger_attempts.values())
    ledger_seconds=sum(row.get('simulated_seconds',0.) for row in ledger_attempts.values())
    observed_substeps=sum(r['actual_substeps'] for r in controls)
    observed_seconds=sum(r['seconds'] for r in controls)
    mismatch=bool(ledger_attempts) and (ledger_controls!=len(controls) or ledger_substeps!=observed_substeps or abs(ledger_seconds-observed_seconds)>1e-8)
    summary.update(kind='strict-original-contact-saved-report.v1',protocol_sha256=PROTOCOL_SHA256,
        all_attempts=len(metadata),metadata_attempts=len(meta_paths),
        attempted_without_metadata=unreported,actual_controls=ledger_controls if ledger_attempts else len(controls),
        saved_observed_controls=len(controls),valid_controls=sum(r['valid'] for r in controls),
        actual_substeps=ledger_substeps if ledger_attempts else observed_substeps,
        saved_observed_substeps=observed_substeps,
        simulated_seconds=ledger_seconds if ledger_attempts else observed_seconds,
        saved_observed_simulated_seconds=observed_seconds,
        totals_unresolved=bool(unreported) or mismatch,
        missing_saved_metric_controls=max(0,ledger_controls-len(controls)) if ledger_attempts else 0,
        original_controller=True,physics_hz=100,solver_iterations=10,physics_transition_modified=False,
        model_calls=0,optimizer_updates=0,training_eligible=False,training_authorized=False,pending_user_approval=True,
        old_modified_data_included=False,ledger_evidence=ledgers,
        moving_definition='Positive AO impulse + displacement>=1imagepx/s OR rotation>=1deg/s at actual0.01s microstep; all impulse/time evidence retained',
        contact_definition='AO positive actual post-solve impulse; pair presence and zero-impulse AO are separate. Wall-only means OW pair with no AO pair. OW without positive AO is a separate metric. Simultaneous arcs are nonexclusive time.',
        invalidity_note='Original-environment illegal states are rejected observations, not physical repairs. All attempts and partial costs remain.',
        sample_denominators='All attempts/controls/actual elapsed simulation time are separate from legal H1/H4 windows and eventual approved training windows.')
    arc_cohorts=defaultdict(lambda:dict(seconds=0.,distinct=set()))
    for row in arcs:
        key=(row['phase'],row['letter'],row['split'],row['ring'],row['arc_bin'])
        arc_cohorts[key]['seconds']+=row['seconds']
        arc_cohorts[key]['distinct'].add(row['trajectory_id'])
    coverage=[dict(phase=k[0],letter=k[1],split=k[2],ring=k[3],arc_bin=k[4],
        positive_AO_seconds=v['seconds'],independent_trajectories=len(v['distinct'])) for k,v in sorted(arc_cohorts.items())]
    with ResourceReservation(output,shared_roots,resource_policy,f'strict-original-report:pid{os.getpid()}',role='postprocess') as lease:
        manifest=make_candidate_manifest(output,meta_paths,upstream)
        manifest_path=output/'candidate-manifest.json'
        _write_json(lease,manifest_path,manifest)
        admission=[]
        if manifest['entries']:
            verified=verify_strict_original_dataset(manifest_path,expected_manifest_sha256=sha256(manifest_path),require_training_authorization=False)
            for raw,entry in verified.entries.items():
                meta=read(verified.root/entry['metadata_file']);data=load_attempt(raw)
                guard_trajectory(raw,meta,verified,frame_indices=data['frame_indices'],valid_controls=data['control_valid'],
                    control_time_seconds=data['control_time_seconds'],segment_ids=data['segment_ids'])
                admission.append(dict(trajectory_id=entry['trajectory_id'],protocol_candidate_verified=True,
                    valid_controls=int(data['control_valid'].sum()),training_eligible=False))
        summary['candidate_admission']=dict(manifest=str(manifest_path),manifest_sha256=sha256(manifest_path),
            admitted_attempts=admission,rejected_attempts=manifest['rejected_attempts'],training_authorized=False)
        _write_csv(lease,destination/'attempts.csv',attempts)
        _write_csv(lease,destination/'controls.csv',controls)
        _write_csv(lease,destination/'windows.csv',windows)
        _write_csv(lease,destination/'actual-contact-arcs.csv',arcs)
        _write_csv(lease,destination/'contact-coverage.csv',coverage)
        video_rows=[]
        if videos and sources:
            # Formal examples take priority once they exist; calibration remains
            # separately labelled, never presented as a formal model result.
            preferred='formal' if any(m['slot']['phase']=='formal' for m,_ in sources.values()) else sorted({m['slot']['phase'] for m,_ in sources.values()})[-1]
            pool=[(key,m,p) for key,(m,p) in sources.items() if m['slot']['phase']==preferred]
            complete=[x for x in pool if x[1].get('complete')]
            legal=[w for w in windows if w['legal']]
            def amplitude(key,field):
                vals=[w[field] for w in legal if w['trajectory_id']==key and w['horizon']==1]
                return float(np.median(vals)) if vals else 0.
            choices=[]
            if complete:
                choices.extend([('translation',max(complete,key=lambda x:amplitude(x[0],'translation_image_px'))),
                    ('rotation',max(complete,key=lambda x:amplitude(x[0],'rotation_deg'))),
                    ('random',random.Random(20261031).choice(sorted(complete))),
                    ('weak-motion',min(complete,key=lambda x:amplitude(x[0],'translation_image_px')))])
                boundary=[x for x in complete if any(r['trajectory_id']==x[0] and r['boundary_seconds']>0 for r in controls)]
                if boundary:choices.append(('boundary',boundary[0]))
            failed=[x for x in pool if not x[1].get('complete')]
            if failed:choices.append(('failure',failed[0]))
            for role,(key,meta,raw) in choices:
                video=_video(destination/'videos'/f'{role}.mp4',meta,load_attempt(raw),lease)
                video.update(role=role,trajectory_id=key,phase=meta['slot']['phase'],letter=meta['slot']['letter'])
                video_rows.append(video)
        summary['videos']=video_rows
        _write_json(lease,destination/'summary.json',summary)
        lines=['# Strict original contact sampling report','',
            'Original100Hz / solver10 / KINEMATIC pusher / damping0 / upstream PD100,20. '
            'These are saved physical collection results; no model prediction improvement was measured. '
            'Training is paused and all data remain pending human approval.','',
            f"All attempted scenes: {summary['all_attempts']}; observed controls: {summary['actual_controls']}; "
            f"legal controls: {summary['valid_controls']}; actual substeps: {summary['actual_substeps']}; "
            f"simulated duration: {summary['simulated_seconds']:.4f}s. Simulation duration is not CPU/GPU elapsed.",'',
            '|Phase|Letter|Attempts|Complete|Valid/observed controls|H1 translation P50 imagepx|H4 P50 imagepx|Positive AO moving time|Wall-only time|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for group in summary['groups']:
            h1=[v for k,v in group['motion'].items() if k.endswith('_H1')]
            h4=[v for k,v in group['motion'].items() if k.endswith('_H4')]
            value=lambda vals: ','.join('NA' if v['translation_image_px']['P50'] is None else f"{v['translation_image_px']['P50']:.3f}" for v in vals)
            fraction=lambda key:'NA' if group[key] is None else f"{group[key]:.1%}"
            lines.append(f"|{group['phase']}|{group['letter']}|{group['attempts']}|{group['complete_attempts']}|{group['valid_controls']}/{group['actual_controls']}|{value(h1)}|{value(h4)}|{fraction('contact_moving_seconds_fraction_all_actual_time')}|{fraction('wall_only_seconds_fraction_all_actual_time')}|")
        lines += ['', 'P50 cells preserve each split separately when more than one split exists; full quantiles/bins and denominators are in [summary](report/summary.json).',
            '', 'Original penetration remains a possible environment limitation. An illegal observed substep ends that trajectory; '
            'no pose/velocity correction, hidden interpolation, reset, retry or replacement is performed. '
            'Legal H1/H4 metrics use exactly0.1/0.4s adjacent saved states and never skip frames.',
            '', 'Actual positive-impulse contact arcs are listed separately from planned contacts. '
            'Unassigned arcs remain ring=-1/arc=-1. Arc times are nonexclusive; simultaneous points do not inflate contact seconds.',
            '', 'The formal supplement trigger is diagnostic only and uses every valid formal H1, including early controls without four-frame history. '
            'Calibration and supplement do not enter its denominator. No automatic release is granted.',
            '', '[HTML report](report/index.html) · [all attempts](report/attempts.csv) · [controls](report/controls.csv) · '
            '[H1/H4 windows](report/windows.csv) · [coverage](report/contact-coverage.csv) · '
            '[candidate membership](candidate-manifest.json)', '']
        lease.write_bytes(output/'REPORT.md','\n'.join(lines).encode('utf8'))
        cards=''.join(f'<section><h3>{html.escape(v["role"])} | {v["letter"]} | {html.escape(v["phase"])}</h3>'
            f'<p>{html.escape(v["trajectory_id"])}</p><video controls preload="metadata" width="576" src="videos/{Path(v["file"]).name}"></video></section>' for v in video_rows)
        page='<!doctype html><meta charset="utf-8"><title>Strict original contact report</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}pre{white-space:pre-wrap}video{max-width:100%}section{margin:30px 0}</style>'
        page+='<h1>Strict original sampling — training paused</h1><p>Original100Hz, solver10, kinematic pusher, zero damping. Saved actual physical data; no model prediction claim.</p>'
        page+='<p><a href="../REPORT.md">Full report</a> · <a href="summary.json">Quantiles and denominators</a> · <a href="controls.csv">Controls</a> · <a href="windows.csv">H1/H4</a> · <a href="contact-coverage.csv">Actual contact coverage</a></p>'+cards
        page+='<h2>Per phase and letter</h2><pre>'+html.escape(json.dumps(summary['groups'],indent=2))+'</pre>'
        lease.write_bytes(destination/'index.html',page.encode('utf8'))
        _write_json(lease,destination/'resource-receipt.json',lease.snapshot())
    return summary
