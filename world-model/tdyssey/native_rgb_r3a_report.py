"""Audit real compact R3A physics and render representative saved-state videos.

No model work or physics evolution occurs here. Every video frame is rendered
from a recorded state; geometric labels remain audit-only.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

from .native_rgb_r3a_data import RecordedStateRenderer


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, zero_fraction=None, minimum=None, p10=None, median=None, p90=None, p95=None, maximum=None)
    p = np.percentile(values, [0,10,50,90,95,100])
    return dict(count=len(values), zero_fraction=float(np.mean(np.abs(values)<=1e-9)),
                **dict(zip(('minimum','p10','median','p90','p95','maximum'), map(float,p))))


def runs(flags):
    lengths=[]; n=0
    for yes in list(flags)+[False]:
        if yes:n+=1
        elif n:lengths.append(n);n=0
    return lengths


def area_centers(states,letter):
    from .envs.official import official_shape
    local=np.asarray(official_shape(letter).centroid,dtype=float)
    a=np.asarray(states)[:,4];c,s=np.cos(a),np.sin(a)
    return np.asarray(states)[:,2:4]+np.column_stack((c*local[0]-s*local[1],s*local[0]+c*local[1]))


def motion(states, horizon, letter):
    """H1/H4 anchors follow four history frames: t=3 through final-H."""
    s=np.asarray(states, dtype=float)
    before=s[3:-horizon];after=s[3+horizon:]
    centers=area_centers(s,letter)
    translation=np.linalg.norm(centers[3+horizon:]-centers[3:-horizon],axis=1)*96/512
    # Angular velocity is continuous; no symmetry folding or unsigned rotation.
    rotation=np.rad2deg(after[:,4]-before[:,4])
    return translation, rotation


def motion_summary(translation, rotation):
    translation=np.asarray(translation,dtype=float);rotation=np.asarray(rotation,dtype=float)
    return dict(translation_image96px=summary(translation),signed_rotation_deg=summary(rotation),
        absolute_rotation_deg=summary(np.abs(rotation)),
        obvious_count=int(np.sum((translation>=2)|(np.abs(rotation)>=5))),
        count=len(translation),positive_rotation_ge5=int(np.sum(rotation>=5)),
        negative_rotation_le_minus5=int(np.sum(rotation<=-5)))


def _resolve(base,path):
    path=Path(path)
    return path if path.is_absolute() else base/path


def audit(manifest_path,output):
    """Full distinct-trajectory audit; no sampling or data changes."""
    manifest_path=Path(manifest_path).resolve();base=manifest_path.parent;out=Path(output)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    base=(base/manifest.get('source_root','.')).resolve()
    groups=defaultdict(lambda:defaultdict(list));rows=[];digests=defaultdict(list)
    observed_arcs=defaultdict(Counter);contact_kinds=Counter();initial=[]
    actual_coverage=defaultdict(lambda:dict(trajectories=set(),yaw_bins=set(),
        normal_action=[],tangent_action=[],lever=[],local_points=[],world_points=[]))
    actual_walls=defaultdict(Counter)
    physics_controls=0;substeps=0;global_max_pen=0.;invalid=[];time_curves=defaultdict(list)
    for entry in manifest['entries']:
        file=_resolve(base,entry['file']);meta_path=_resolve(base,entry['metadata_file'])
        if hashlib.sha256(file.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError(f'File hash changed: {file}')
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if hashlib.sha256(meta_path.read_bytes()).hexdigest()!=entry['metadata_sha256']:raise ValueError(f'Metadata hash changed: {meta_path}')
        with np.load(file,allow_pickle=False)as archive:
            # NPZ members are decompressed on every __getitem__. Materialize
            # this small trajectory once before the per-contact audit loop.
            z={name:archive[name]for name in archive.files}
            states=z['states'];actions=z['actions'];valid=z['control_valid']
            pen=z['substep_penetrations'];flags=z['contact_flags'];tags=z['contact_tags'];contacts=z['contact_values']
            if len(actions)!=48 or states.shape!=(49,15) or not bool(np.all(valid)):
                invalid.append(entry['trajectory_id'])
            physics_controls+=len(actions);substeps+=len(z['substep_control_index'])
            maximum_pen=float(np.max(pen)) if pen.size else 0.;global_max_pen=max(global_max_pen,maximum_pen)
            contact_control=np.zeros(len(actions),bool)
            if len(flags):
                sid=z['substep_control_index'].astype(int)
                for ci in range(len(actions)):contact_control[ci]=bool(np.any(flags[sid==ci,0]))
            lengths=runs(contact_control)
            centers=area_centers(states,entry['letter'])
            for ci in range(len(actions)):
                for key in ('all',f"letter:{entry['letter']}"):
                    time_curves[(key,ci)].append((float(np.linalg.norm(centers[ci+1]-centers[ci])*96/512),
                        float(np.rad2deg(states[ci+1,4]-states[ci,4])),float(contact_control[ci]),
                        float(np.max(pen[z['substep_control_index']==ci]))))
            actual_ao=(tags[:,1]==0)&(tags[:,2]==1) if len(tags) else np.zeros(0,bool)
            kinds=Counter(map(int,tags[:,1])) if len(tags) else Counter()
            contact_kinds.update(kinds)
            for i in np.flatnonzero(actual_ao):
                lo,hi=z['contact_arc_offsets'][i:i+2]
                arcs=list(map(int,z['contact_arc_ids'][int(lo):int(hi)]))
                observed_arcs[entry['letter']].update(arcs)
                si=int(tags[i,0]);ci=int(z['substep_control_index'][si])
                source=z['substep_states'][si];post=z['substep_states'][si+1]
                target=actions[ci] if ci<len(actions) else z['attempted_actions'][ci]
                delta=target-source[:2];normal=np.asarray(contacts[i,4:6],float)
                tangent=np.array((-normal[1],normal[0]))
                for arc in arcs:
                    v=actual_coverage[(entry['letter'],arc)]
                    v['trajectories'].add(entry['trajectory_id']);v['yaw_bins'].add(int((post[4]%(2*math.pi))/(2*math.pi/12)))
                    v['normal_action'].append(float(delta@normal));v['tangent_action'].append(float(delta@tangent))
                    v['lever'].append(float(contacts[i,10]));v['local_points'].append(contacts[i,2:4]);v['world_points'].append(contacts[i,:2])
            for i in np.flatnonzero((tags[:,1]==1)&(tags[:,2]==1)) if len(tags) else []:
                p=contacts[i,:2];wall=('left','right','top','bottom')[int(np.argmin((abs(p[0]-7),abs(p[0]-504),abs(p[1]-7),abs(p[1]-504))))]
                actual_walls[entry['letter']][wall]+=1
            row={k:entry.get(k) for k in ('trajectory_id','family_id','slot_id','letter','split','seed')}
            row.update(scenario=meta.get('scenario',meta.get('location',meta.get('initial',{}).get('location','unspecified'))),
                initial_space=meta.get('initial_space',entry.get('initial_space','unspecified')),
                initial_yaw_bin12=int((states[0,4]%(2*math.pi))/(2*math.pi/12)),
                initial_angle_degrees=float(np.rad2deg(states[0,4])),
                initial_area_centroid_x=float(centers[0,0]),initial_area_centroid_y=float(centers[0,1]),
                controls=len(actions),physics_substeps=len(z['substep_control_index']),
                max_penetration_worldpx=maximum_pen,contact_control_count=int(contact_control.sum()),
                max_contact_run_controls=max(lengths,default=0),
                contact_run_count=len(lengths),contact_run_lengths_controls=';'.join(map(str,lengths)),
                agent_object_contacts=kinds[0],object_wall_contacts=kinds[1],agent_wall_contacts=kinds[2],
                total_translation_image96px=float(np.linalg.norm(area_centers(states,entry['letter'])[-1]-area_centers(states,entry['letter'])[0])*96/512),
                total_signed_rotation_deg=float(np.rad2deg(states[-1,4]-states[0,4])))
            for h in (1,4):
                t,r=motion(states,h,entry['letter']);m=motion_summary(t,r)
                row.update({f'H{h}_translation_{stat}':value for stat,value in m['translation_image96px'].items()})
                row.update({f'H{h}_signed_rotation_{stat}':value for stat,value in m['signed_rotation_deg'].items()})
                row.update({f'H{h}_obvious_count':m['obvious_count'],f'H{h}_count':m['count'],
                    f'H{h}_translation_max':float(t.max()) if len(t) else 0.,
                    f'H{h}_positive_rotation_count':m['positive_rotation_ge5'],
                    f'H{h}_negative_rotation_count':m['negative_rotation_le_minus5'],
                    f'H{h}_rotation_min':float(r.min()) if len(r) else 0.,
                    f'H{h}_rotation_max':float(r.max()) if len(r) else 0.})
                for key in ('all',f"split:{entry['split']}",f"letter:{entry['letter']}",
                            f"{entry['split']}:{entry['letter']}",f"scenario:{row['scenario']}"):
                    groups[key][f'H{h}_translation'].extend(map(float,t))
                    groups[key][f'H{h}_rotation'].extend(map(float,r))
            for key,array in (('state',states),('action',actions)):
                digests[key+':'+hashlib.sha256(array.tobytes()).hexdigest()].append(entry['trajectory_id'])
            for key in ('rgb_frame_sha256','frame_sha256','frame_sha256_bytes','rgb_sha256'):
                if key in z:
                    digests['RGB:'+hashlib.sha256(z[key].tobytes()).hexdigest()].append(entry['trajectory_id'])
                    digests['initial_RGB:'+hashlib.sha256(z[key][0].tobytes()).hexdigest()].append(entry['trajectory_id'])
                    break
            digests['initial_state:'+hashlib.sha256(states[0].tobytes()).hexdigest()].append(entry['trajectory_id'])
            initial.append((entry['trajectory_id'],entry['letter'],entry['split'],states[0].copy()))
            rows.append(row)
    # Same declared proximity rule as R2C0, with area centroid, shape symmetry
    # and object-local pusher. This reports near scenes, not independent proof.
    near=[]
    for letter in 'TLOCXB':
        selected=[v for v in initial if v[1]==letter]
        for i,a in enumerate(selected):
            for b in selected[:i]:
                from .envs.official import official_shape
                dp=np.linalg.norm(area_centers(a[3][None,:],letter)[0]-area_centers(b[3][None,:],letter)[0])
                period=2*math.pi/max(1,official_shape(letter).symmetry_order)
                da=abs((a[3][4]-b[3][4]+period/2)%period-period/2)
                def local_pusher(state):
                    q=state[:2]-state[2:4];c,s=math.cos(state[4]),math.sin(state[4])
                    return np.array((c*q[0]+s*q[1],-s*q[0]+c*q[1]))
                agent=np.linalg.norm(local_pusher(a[3])-local_pusher(b[3]))
                if dp<=8. and da<=math.radians(5.) and agent<=16.:
                    near.append(dict(a=a[0],b=b[0],cross_split=a[2]!=b[2],center_worldpx=float(dp),yaw_deg=math.degrees(da),pusher_worldpx=float(agent)))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'trajectory-audit.csv').open('w',newline='',encoding='utf-8')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else [])
        writer.writeheader();writer.writerows(rows)
    coverage=[]
    for (letter,arc),v in sorted(actual_coverage.items()):
        local=np.asarray(v['local_points']);world=np.asarray(v['world_points'])
        coverage.append(dict(letter=letter,arc=arc,post_contact_records=len(v['lever']),
            independent_trajectories=len(v['trajectories']),yaw_bins12=sorted(v['yaw_bins']),
            executed_target_minus_pre_substep_pusher_on_post_normal_worldpx=summary(v['normal_action']),
            executed_target_minus_pre_substep_pusher_on_post_tangent_worldpx=summary(v['tangent_action']),
            actual_signed_normal_lever_worldpx=summary(v['lever']),
            local_contact_point_bounds=[local.min(axis=0).tolist(),local.max(axis=0).tolist()],
            world_contact_point_bounds=[world.min(axis=0).tolist(),world.max(axis=0).tolist()]))
    from .native_rgb_data import contour_segments
    ring_coverage={}
    for letter in 'TLOCXB':
        arcs=contour_segments(letter,8);seen=set(observed_arcs[letter])
        ring_coverage[letter]={str(ring):dict(
            total_arcs=len([a for a in arcs if a['ring']==ring]),
            observed_arcs=sorted(a['id']for a in arcs if a['ring']==ring and a['id']in seen),
            unobserved_arcs=sorted(a['id']for a in arcs if a['ring']==ring and a['id']not in seen),
            collection_planned_ring=ring==0,
            reachability_scope='Exterior-ring approach was attempted; geometric rejection is not proof of physical impossibility.'if ring==0 else'Closed interior ring was not an initial-contact target in this exterior-approach collector; no claim of tested reachability.')
            for ring in sorted({a['ring']for a in arcs})}
    result=dict(kind='R3A-real-physics-amplitude-contact-audit.v1',trajectories=len(rows),
        counts={f'{s}:{l}':sum(r['letter']==l and r['split']==s for r in rows) for s in ('train','seen_val')for l in 'TLOCXB'},
        completed_controls=physics_controls,known_substeps=substeps,max_penetration_worldpx=global_max_pen,
        invalid_successful_entries=invalid,duplicate_state_or_action_sequences={k:v for k,v in digests.items() if len(v)>1},
        near_initial_scenes=near,near_initial_definition='same letter,area center<=8worldpx,symmetry yaw<=5degrees,local pusher<=16worldpx; same rule as R2C0',
        actual_contact_rows_by_kind=dict(contact_kinds),
        actual_post_agent_object_contact_arc_counts={l:dict(v)for l,v in observed_arcs.items()},
        actual_post_object_wall_contacts_by_wall={l:dict(v)for l,v in actual_walls.items()},
        actual_boundary_arc_pose_action_coverage=coverage,
        canonical_ring_arc_coverage=ring_coverage,
        initial_pose_coverage={split:{l:dict(
            yaw_bin12_counts=dict(Counter(r['initial_yaw_bin12']for r in rows if r['letter']==l and r['split']==split)),
            declared_region_counts=dict(Counter(r['initial_space']for r in rows if r['letter']==l and r['split']==split)),
            initial_angle_degrees=summary([r['initial_angle_degrees']for r in rows if r['letter']==l and r['split']==split]))
            for l in 'TLOCXB'}for split in ('train','seen_val')},
        boundary_coverage_scope='Observed post-step contacts only; post-contact yaw/points/normals/lever, executed-target minus pre-substep pusher projected on post normal/tangent. Missing arcs/holes are not claimed unreachable or covered.',
        numerical_zero_absolute_tolerance=1e-9,
        motion={key:{f'H{h}':motion_summary(value[f'H{h}_translation'],value[f'H{h}_rotation'])for h in (1,4)}for key,value in groups.items()},
        independent_support_by_letter={l:dict(translation_H1=sum(r['letter']==l and r['H1_translation_max']>=2 for r in rows),
            positive_H1=sum(r['letter']==l and r['H1_rotation_max']>=5 for r in rows),
            negative_H1=sum(r['letter']==l and r['H1_rotation_min']<=-5 for r in rows))for l in 'TLOCXB'},
        independent_support_by_split_and_letter={split:{l:dict(
            translation_H1=sum(r['letter']==l and r['split']==split and r['H1_translation_max']>=2 for r in rows),
            positive_H1=sum(r['letter']==l and r['split']==split and r['H1_rotation_max']>=5 for r in rows),
            negative_H1=sum(r['letter']==l and r['split']==split and r['H1_rotation_min']<=-5 for r in rows))
            for l in 'TLOCXB'}for split in ('train','seen_val')},
        control_time_curves={key:[dict(control=ci,simulation_start_seconds=ci*.1,
            translation_image96px=summary(np.asarray(time_curves[(key,ci)])[:,0]),
            signed_rotation_deg=summary(np.asarray(time_curves[(key,ci)])[:,1]),
            contact_fraction=float(np.mean(np.asarray(time_curves[(key,ci)])[:,2])),
            penetration_worldpx=summary(np.asarray(time_curves[(key,ci)])[:,3]))for ci in range(48)if (key,ci)in time_curves]
            for key in ['all']+[f'letter:{l}'for l in 'TLOCXB']},
        scope='All successful distinct new trajectories. Failed/partial attempted controls are additional and reported in the collection ledger.',
        physics_replay_controls=0,model_calls=0)
    write_json(out/'physics-audit.json',result)
    return result,rows,manifest


def videos(manifest_path,rows,source_root,output,seed=5478):
    import cv2
    manifest_path=Path(manifest_path).resolve();base=manifest_path.parent;out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));base=(base/manifest.get('source_root','.')).resolve()
    entries={e['trajectory_id']:e for e in manifest['entries']}
    selected=defaultdict(list)
    for letter in 'TLOCXB':
        choices=[r for r in rows if r['letter']==letter]
        if not choices:continue
        criteria=dict(translation=lambda r:r['H1_translation_max'],positive_rotation=lambda r:r['H1_rotation_max'],
            negative_rotation=lambda r:-r['H1_rotation_min'],boundary=lambda r:r['object_wall_contacts'])
        for role,score in criteria.items():
            best=max(choices,key=score);selected[best['trajectory_id']].append(f'{letter}:{role}')
    rng=np.random.default_rng(seed)
    for letter in 'TLOCXB':
        choices=[r for r in rows if r['letter']==letter]
        if choices:selected[choices[int(rng.integers(len(choices)))]['trajectory_id']].append(f'{letter}:fixed_seed_random')
        if choices:selected[min(choices,key=lambda r:r['H1_obvious_count'])['trajectory_id']].append(f'{letter}:least_obvious_motion')
    row_by_id={r['trajectory_id']:r for r in rows}
    renderer=RecordedStateRenderer(source_root,maximum_frame_cache_bytes=0);records=[]
    try:
        for trajectory_id,roles in selected.items():
            entry=entries[trajectory_id]
            with np.load(_resolve(base,entry['file']),allow_pickle=False)as z:
                states=z['states'].copy();frame_hashes=z['rgb_frame_sha256'].copy()
            frames=[np.transpose(renderer.render(entry['letter'],s),(1,2,0))for s in states]
            for frame,expected in zip(frames,frame_hashes):
                if hashlib.sha256(np.moveaxis(frame,-1,0).tobytes()).digest()!=bytes(expected):
                    raise ValueError('Saved-state video RGB differs from actual original collection frame hash')
            name=trajectory_id.replace(':','_').replace('/','_');path=out/(name+'.mp4')
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10.,(384,384))
            if not writer.isOpened():raise RuntimeError('Installed OpenCV mp4v encoder unavailable')
            for frame in frames:
                writer.write(cv2.cvtColor(cv2.resize(frame,(384,384),interpolation=cv2.INTER_NEAREST),cv2.COLOR_RGB2BGR))
            writer.release()
            capture=cv2.VideoCapture(str(path));n=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=capture.get(cv2.CAP_PROP_FPS);capture.release()
            if n!=len(states)or abs(fps-10)>1e-8:raise ValueError('Video frame count/fps differs from real10Hz')
            strip=np.concatenate([cv2.resize(frames[i],(192,192),interpolation=cv2.INTER_NEAREST)for i in (0,12,24,36,48)],axis=1)
            cv2.imwrite(str(out/(name+'.png')),cv2.cvtColor(strip,cv2.COLOR_RGB2BGR))
            record=dict(trajectory_id=trajectory_id,roles=roles,file=path.name,strip=name+'.png',frames=n,fps=fps,
                simulated_first_time_seconds=0.,simulated_last_time_seconds=(n-1)/10.,file_bytes=path.stat().st_size,
                source_file=entry['file'],source_sha256=entry['sha256'])
            record['original_collection_frame_hashes_pixel_exact']=len(frame_hashes)
            row=row_by_id[trajectory_id];checks={}
            for role in roles:
                name=role.split(':',1)[1]
                value_threshold={'translation':(row['H1_translation_max'],2.),
                    'positive_rotation':(row['H1_rotation_max'],5.),
                    'negative_rotation':(-row['H1_rotation_min'],5.),
                    'boundary':(row['object_wall_contacts'],1.)}.get(name)
                if value_threshold:
                    value,threshold=value_threshold
                    checks[name]=dict(value=value,threshold=threshold,supported=value>=threshold)
            record['actual_role_support']=checks
            records.append(record)
    finally:renderer.close()
    write_json(out/'video-provenance.json',dict(kind='R3A-saved-actual-state-real10Hz-videos.v1',random_seed=seed,
        records=records,renderer_stats=renderer.stats,physics_controls=0,model_calls=0,
        display_resize='96px original RGB upscaled nearest4x to384; no new appearance, motion interpolation or speedup'))
    cards=[]
    for record in records:
        label=html.escape(', '.join(record['roles'])+' — '+record['trajectory_id'])
        missing=[name for name,value in record['actual_role_support'].items()if not value['supported']]
        note='' if not missing else '<p>Coverage gap: best available example still below the declared threshold for '+html.escape(', '.join(missing))+'.</p>'
        cards.append(f'<section><h2>{label}</h2>{note}<video controls preload="metadata" width="384" src="{html.escape(record["file"])}"></video><br><img width="960" src="{html.escape(record["strip"])}"></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R3A actual48control10Hz videos</title><style>body{font:16px system-ui;margin:24px;background:#eee}section{background:white;padding:16px;margin:20px 0}h2{font-size:17px}</style><h1>Real saved trajectories,48 controls at10Hz</h1><p>All49 frames0.0–4.8s, unchanged playback10fps. Each role chooses a real maximum from all trajectories; fixed-seed random and least-motion examples disclose stalls. Geometry is collection/audit privileged; all model inputs remain RGB/actions.</p>'+''.join(cards),encoding='utf-8')
    return records


def failure_videos(index_path,source_root,output):
    """Keep a viewable short clip of failed partial attempts; never pad it."""
    import cv2
    index_path=Path(index_path).resolve();base=index_path.parent;out=Path(output)
    index=json.loads(index_path.read_text(encoding='utf-8'));failures=[]
    for slot in index['slots']:
        for attempt in slot['attempts']:
            if attempt.get('status','').startswith('failed'):
                file=base/'trajectories'/(attempt['trajectory_id']+'.npz')
                failures.append((slot,attempt,file))
    out.mkdir(parents=True,exist_ok=True)
    renderer=RecordedStateRenderer(source_root,maximum_frame_cache_bytes=0);records=[]
    try:
        for slot,attempt,file in failures[:6]:
            record=dict(trajectory_id=attempt['trajectory_id'],letter=slot['letter'],
                status=attempt['status'],error=attempt.get('error'),partial=True,source_file=str(file),
                charged_controls=attempt['charged_controls'],completed_controls=attempt['completed_controls'])
            if not file.exists():record['video_unavailable']='No complete saved control state after initialization failure';records.append(record);continue
            with np.load(file,allow_pickle=False)as archive:
                states=archive['states'].copy();substates=archive['substep_states'].copy();frame_hashes=archive['rgb_frame_sha256'].copy()
                available=archive['substep_state_available'].copy()
            if not np.isfinite(states).all():record['video_unavailable']='Nonfinite raw saved states retained without repair';records.append(record);continue
            path=out/(attempt['trajectory_id']+'.mp4')
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10.,(384,384))
            if not writer.isOpened():raise RuntimeError('Installed mp4v encoder unavailable')
            for state,expected in zip(states,frame_hashes):
                frame=np.transpose(renderer.render(slot['letter'],state),(1,2,0))
                if hashlib.sha256(np.moveaxis(frame,-1,0).tobytes()).digest()!=bytes(expected):
                    raise ValueError('Failed-attempt saved control RGB differs from actual original collection hash')
                writer.write(cv2.cvtColor(cv2.resize(frame,(384,384),interpolation=cv2.INTER_NEAREST),cv2.COLOR_RGB2BGR))
            writer.release();capture=cv2.VideoCapture(str(path));n=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=capture.get(cv2.CAP_PROP_FPS);capture.release()
            if n!=len(states)or fps!=10.:raise ValueError('Partial failure video timing differs')
            record.update(file=path.name,frames=n,fps=fps,actual_last_time_seconds=(n-1)*.1,
                source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),no_padding_or_replay=True)
            record['original_collection_frame_hashes_pixel_exact']=len(frame_hashes)
            record['actual_substep_chain_last_time_seconds']=(len(substates)-1)*.01
            record['clip_scope']='Complete recorded control states only; any final partial100Hz advance is separately shown below.'
            if len(substates)>1 and bool(available[-1])and np.isfinite(substates[-1]).all():
                final_rgb=np.transpose(renderer.render(slot['letter'],substates[-1]),(1,2,0))
                final_path=out/(attempt['trajectory_id']+'-last-substep.png')
                cv2.imwrite(str(final_path),cv2.cvtColor(cv2.resize(final_rgb,(384,384),interpolation=cv2.INTER_NEAREST),cv2.COLOR_RGB2BGR))
                record['actual_last_substep_image']=final_path.name
            records.append(record)
    finally:renderer.close()
    write_json(out/'failure-video-provenance.json',dict(kind='R3A-retained-failure-partial-videos.v1',
        all_failed_attempts=len(failures),shown_first_failed_attempts=len(records),records=records,physics_controls=0,model_calls=0))
    cards=['<!doctype html><meta charset="utf-8"><title>R3A actual partial failures</title><h1>Retained partial failures, real10Hz</h1><p>Stopped attempts remain short. No resimulation, padding or repaired pixels.</p>']
    for r in records:
        cards.append('<h2>'+html.escape(r['trajectory_id'])+'</h2><p>'+html.escape(str(r['error']))+'</p>')
        if 'file'in r:cards.append('<video controls width="384" src="'+html.escape(r['file'])+'"></video>')
        if 'actual_last_substep_image'in r:
            cards.append('<p>Last recorded actual physics state at '+str(r['actual_substep_chain_last_time_seconds'])+' seconds, including any final partial control:</p><img width="384" src="'+html.escape(r['actual_last_substep_image'])+'">')
    (out/'index.html').write_text(''.join(cards),encoding='utf-8')
    return records
