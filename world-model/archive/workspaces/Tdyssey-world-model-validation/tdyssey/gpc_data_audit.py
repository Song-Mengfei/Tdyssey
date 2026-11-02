"""Audit data metadata and the exact 500-update prefix; no arrays or models."""

from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

from .gpc_adaptation_data import GPCAdaptationDataset, read_adaptation_manifest
from .training_data import initial_state_hash
from .trajectories import Trajectory


def training_prefix(dataset):
    """Match balanced_indices(dataset, 0, 500), without invoking a trainer."""
    if dataset.split != "train":
        raise ValueError("The fixed prefix is training only")
    buckets = [dataset.shape_indices(shape) for shape in ("T", "L")]
    if not all(buckets):
        raise ValueError("Both T and L require training windows")
    return [buckets[k % 2][(k//2) % len(buckets[k % 2])] for k in range(500)]


def _stats(rows):
    rows = list(rows)
    def distribution(key):
        values = sorted(row[key] for row in rows)
        def quantile(p):
            if not values:
                return None
            a=(len(values)-1)*p; i=int(a)
            return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(a-i)
        return dict(mean=sum(values)/len(values) if values else None,p50=quantile(.5),
                    p90=quantile(.9),maximum=max(values) if values else None)
    return dict(occurrences=len(rows),unique_controls=len({(r['episode_id'],r['control']) for r in rows}),
        unique_episodes=len({r['episode_id'] for r in rows}),unique_scenes=len({r['scene'] for r in rows}),
        contact=sum(r['contact'] for r in rows),translation_pixels=distribution('translation'),
        absolute_rotation_degrees=distribution('rotation'),rotation_over_5_degrees=sum(r['rotation']>5 for r in rows),
        clockwise_over_5_degrees=sum(r.get('signed_rotation',0)>5 for r in rows),
        counterclockwise_over_5_degrees=sum(r.get('signed_rotation',0)<-5 for r in rows),
        coverage=distribution('coverage'),near_success=sum(.9<=r['coverage']<=.95 for r in rows),
        strict_success=sum(r['coverage']>.95 for r in rows),
        coverage_bins={name:sum(test(r['coverage']) for r in rows) for name,test in (
            ('[0,.5)',lambda v:v<.5),('[.5,.8)',lambda v:.5<=v<.8),('[.8,.9)',lambda v:.8<=v<.9),
            ('[.9,.95]',lambda v:.9<=v<=.95),('(.95,1]',lambda v:v>.95))})


def _shapes(rows):
    rows=list(rows)
    return {s:_stats(r for r in rows if r['shape']==s) for s in ('T','L')}


def _held_guard(data):
    paths,identities=set(),set()
    for row in data['episodes']:
        if row['role']=='held_model_test':
            for entry in row['entry']['files']:
                path=Path(row['root'])/entry['name']; paths.add(path.resolve())
                if path.is_file():
                    value=path.stat(); identities.add((value.st_dev,value.st_ino))
    def guard(path):
        path=Path(path)
        if path.resolve() in paths:
            raise ValueError('Held model_test payload access is forbidden')
        if path.is_file():
            value=path.stat()
            if (value.st_dev,value.st_ino) in identities:
                raise ValueError('Held model_test payload alias access is forbidden')
    return guard


def _episode(row, guard):
    if row['role'] not in ('train','development','independent_validation') or row['split']=='model_test':
        raise ValueError('Held model_test records cannot enter statistics')
    entry=next(e for e in row['entry']['files'] if Path(e['name']).suffix=='.json')
    path=Path(row['root'])/entry['name']; guard(path); raw=path.read_bytes()
    if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
        raise ValueError('Episode JSON identity changed')
    record=json.loads(raw); trajectory=Trajectory.from_json(json.dumps(record['trajectory']))
    if (trajectory.trajectory_id!=row['trajectory_id'] or len(trajectory.transitions)!=row['control_steps']
            or initial_state_hash(trajectory)!=row['initial_state_sha256']
            or len(record['step_diagnostics'])!=row['control_steps']
            or not math.isclose(trajectory.control_dt,.1,rel_tol=0,abs_tol=1e-12)):
        raise ValueError('Episode scene/timing metadata differs')
    result=[]
    for i,(transition,diagnostic) in enumerate(zip(trajectory.transitions,record['step_diagnostics'])):
        a,b=transition.before.object_body.pose,transition.after.object_body.pose
        coverage,contacts=diagnostic.get('coverage'),diagnostic.get('n_contacts')
        if (type(coverage) not in (int,float) or not math.isfinite(coverage) or not 0<=coverage<=1
                or type(contacts) is not int or contacts<0):
            raise ValueError('Invalid physical contact/coverage metadata')
        result.append(dict(episode_id=row['trajectory_id'],scene=row['initial_state_sha256'],shape=row['shape'],
            control=i,contact=contacts>0,translation=math.dist(a.position,b.position),
            rotation=abs(math.degrees(math.atan2(math.sin(b.angle-a.angle),math.cos(b.angle-a.angle)))),
            signed_rotation=math.degrees(math.atan2(math.sin(b.angle-a.angle),math.cos(b.angle-a.angle))),coverage=coverage))
    if sum(r['contact'] for r in result)!=row['contact_steps']:
        raise ValueError('Frozen contact total differs')
    return result,dict(path=str(path),bytes=len(raw),sha256=entry['sha256'],role=row['role'])


def audit_benchmark_support(path, *, data):
    """Physical-label separation/termination support; never prediction accuracy."""
    from itertools import combinations
    path=Path(path).resolve()
    guard=_held_guard(data)
    guard(path); raw=path.read_bytes(); value=json.loads(raw); cases=value.get('cases',[])
    forbidden={r['initial_state_sha256'] for r in data['episodes']} if data else set()
    if data:
        forbidden.update(json.loads(Path(data['path']).read_bytes()).get('reviewed_scene_sha256',[]))
    seen=set(); groups={}; scene_anchors={}; overlaps=[]; bad_payloads=[]; positive=[]; negative=[]; near=[]
    for case in cases:
        shape,anchor,candidate=case.get('shape'),case.get('anchor'),case.get('candidate')
        key=(shape,anchor,candidate); scene=case.get('initial_state_sha256'); truth=case.get('truth')
        if (shape not in ('T','L') or candidate not in ('hold','toward','side','away') or key in seen
                or not isinstance(scene,str) or len(scene)!=64 or any(c not in '0123456789abcdef' for c in scene)
                or case.get('role')!='independent' or case.get('prior_uses')!=[] or not isinstance(truth,list) or len(truth)!=4):
            raise ValueError('Invalid benchmark identity/independence/physical future')
        seen.add(key); scores=[row.get('coverage') for row in truth]
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in scores):
            raise ValueError('Invalid benchmark physical coverage')
        group_key=(shape,anchor)
        if scene in forbidden: overlaps.append(list(key))
        if scene in scene_anchors and scene_anchors[scene]!=group_key:
            raise ValueError('Distinct benchmark anchors share a scene')
        scene_anchors[scene]=group_key
        group=groups.setdefault(group_key,dict(scene=scene,candidates={}))
        if group['scene']!=scene: raise ValueError('Candidates do not share their anchor scene')
        group['candidates'][candidate]=scores
        payload=case.get('payload',{}); source=path.parent/payload.get('path','<missing>'); guard(source)
        if source.is_symlink() or not source.is_file() or source.stat().st_size!=payload.get('bytes'):
            bad_payloads.append(list(key))
        (positive if any(v>.95 for v in scores) else negative).append(list(key))
        if any(.9<=v<=.95 for v in scores): near.append(list(key))
    expected={(s,f'I{s}{i:02d}',c) for s in ('T','L') for i in range(6) for c in ('hold','toward','side','away')}
    separation={}
    for shape in ('T','L'):
        horizons={}
        for h in range(4):
            informative=[]; pairs=0; complete=0
            for (s,anchor),group in sorted(groups.items()):
                values=group['candidates']
                if s!=shape or set(values)!={'hold','toward','side','away'}: continue
                complete+=1
                count=sum(abs(values[a][h]-values[b][h])+1e-12>=.05 for a,b in combinations(sorted(values),2))
                pairs+=count
                if count: informative.append(anchor)
            horizons[str(h+1)]=dict(complete_anchors=complete,informative_anchors=informative,separable_pairs=pairs,
                status='supported' if pairs else 'insufficient')
        separation[shape]=horizons
    termination_by_shape={s:dict(positive_cases=sum(k[0]==s for k in positive),negative_cases=sum(k[0]==s for k in negative),
        sufficient=any(k[0]==s for k in positive) and any(k[0]==s for k in negative)) for s in ('T','L')}
    return dict(kind='gpc_benchmark_metadata_support.v1',path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
        cases=len(cases),expected_cases=48,grid_complete=seen==expected,
        missing_cases=[list(k) for k in sorted(expected-seen)],extra_cases=[list(k) for k in sorted(seen-expected)],
        data_or_reviewed_scene_overlaps=overlaps,payload_stat_failures=bad_payloads,
        metadata_and_stat_ready=seen==expected and not overlaps and not bad_payloads,
        pair_separation=.05,pair_tolerance=1e-12,pair_rule='abs(diff)+1e-12 >= .05',physical_endpoint_separation=separation,
        ranking_support_by_shape={s:any(r['separable_pairs'] for r in separation[s].values()) for s in ('T','L')},
        termination=dict(strict_success_above=.95,positive_cases=len(positive),negative_cases=len(negative),near_success_cases=len(near),
            positive_case_keys=positive,negative_case_keys=negative,by_shape=termination_by_shape,
            sufficient=all(r['sufficient'] for r in termination_by_shape.values()),prediction_certified=False),
        image_archives_opened=0,held_payloads_opened=0,
        scope='Physical endpoint labels only. All-uninformative support is insufficient, not success. No certification or extra collection.')


def audit_data_manifest(path, *, benchmark_path=None):
    """Compare full allowed corpus with weighted/unique actual prefix targets.

    Only declared non-held episode JSON is read. NPZ files receive stat through
    the existing manifest API; no image/archive is opened or hashed.
    """
    data=read_adaptation_manifest(path); guard=_held_guard(data)
    dataset=GPCAdaptationDataset(path,'train'); cache={}; identities={}; roles={}
    for role in ('train','development','independent_validation'):
        episodes=[r for r in data['episodes'] if r['role']==role]
        for row in episodes:
            cache[row['trajectory_id']],identity=_episode(row,guard); identities[identity['path']]=identity
        controls=[r for e in episodes for r in cache[e['trajectory_id']]]
        windows=[r for e in episodes for r in cache[e['trajectory_id']][3:]]
        roles[role]=dict(episodes=len(episodes),unique_scenes=len({e['initial_state_sha256'] for e in episodes}),
            windows=sum(e['windows'] for e in episodes),controls=_shapes(controls),window_targets=_shapes(windows))
    try: indices=training_prefix(dataset)
    except ValueError as error: prefix=dict(available=False,reason=str(error),updates=0,indices=[],rows=[])
    else:
        rows=[]; selected=[]
        for ordinal,index in enumerate(indices):
            n=bisect_right(dataset.offsets,index)-1; episode=dataset.episodes[n]
            anchor=3+index-dataset.offsets[n]; selected.append(cache[episode['trajectory_id']][anchor])
            rows.append(dict(update=ordinal+1,index=index,episode_id=episode['trajectory_id'],shape=episode['shape'],
                initial_state_sha256=episode['initial_state_sha256'],anchor=anchor,
                conditioning_observations=[anchor-3,anchor],aligned_controls=[anchor-3,anchor],target_observation=anchor+1))
        unique={(r['episode_id'],r['control']):r for r in selected}
        prefix=dict(available=True,updates=500,indices=indices,rows=rows,weighted_target_statistics=_shapes(selected),
            unique_window_statistics=_shapes(unique.values()),unique_windows=len(unique),repeated_window_occurrences=500-len(unique),
            selected_episode_ids=sorted({r['episode_id'] for r in selected}),selected_scene_sha256=sorted({r['scene'] for r in selected}))
    independent={}
    for shape in ('T','L'):
        rows=[r for r in data['episodes'] if r['shape']==shape and r['role']=='independent_validation' and r['payloads_present']]
        count=sum(r['windows'] for r in rows)
        independent[shape]=dict(episodes=len(rows),unique_scenes=len({r['initial_state_sha256'] for r in rows}),
            windows=count,required_distinct_windows=8,sufficient=count>=8)
    return dict(kind='gpc_collected_data_prefix_audit.v1',manifest=dict(path=data['path'],sha256=data['sha256']),
        dataset_readiness=data['readiness'],full_corpus_by_role=roles,actual_500_prefix=prefix,
        independent_validation_support=independent,
        data_support_ready=data['readiness']['ready'] and prefix['available'] and all(r['sufficient'] for r in independent.values()),
        benchmark=audit_benchmark_support(benchmark_path,data=data) if benchmark_path else dict(available=False,reason='Benchmark metadata not supplied'),
        permitted_record_identities=identities,held_model_test=dict(episodes=sum(r['role']=='held_model_test' for r in data['episodes']),payloads_opened=0,payloads_hashed=0,statistics=False),
        definitions=dict(prefix='Exactly balanced_indices(train,0,500): T/L alternate; first250 per shape cycle if short; no shuffle.',
            target='Contact/motion/coverage at the executed target control anchor; four actions/observations are context.',
            near_success='.9 <= coverage <= .95; success strictly > .95',rotation='absolute wrapped delta degrees',quantiles='linear interpolation (n-1)*p'),
        operations=dict(Torch_imports=0,image_archive_opens=0,held_payload_opens=0,model_operations=0,simulation=0,training=0),
        scope='Coverage is descriptive, not threshold tuning, model validation or authority for extra collection.')


def _window_overlap(rows):
    from itertools import combinations
    result={}
    for shape in ('T','L'):
        selected=[r for r in rows if r['shape']==shape]
        frames={(r['episode_id'],i) for r in selected for i in range(r['control']-3,r['control']+2)}
        contexts={(r['episode_id'],i) for r in selected for i in range(r['control']-3,r['control']+1)}
        groups={}
        for row in selected: groups.setdefault(row['scene'],[]).append(row['control'])
        pairs=[abs(a-b) for anchors in groups.values() for a,b in combinations(anchors,2)]
        result[shape]=dict(windows=len(selected),five_frame_occurrences=5*len(selected),unique_five_frames=len(frames),
            context_frame_occurrences=4*len(selected),unique_context_frames=len(contexts),
            action_occurrences=4*len(selected),unique_actions=len(contexts),same_scene_unordered_pairs=len(pairs),
            five_frame_overlapping_pairs=sum(d<=4 for d in pairs),context_or_action_overlapping_pairs=sum(d<=3 for d in pairs))
    return result


def audit_training_schedule(data_path, schedule_path):
    """Read the one generated order; never regenerate, search or load images."""
    data=read_adaptation_manifest(data_path); guard=_held_guard(data)
    schedule_path=Path(schedule_path).resolve(); guard(schedule_path)
    raw=schedule_path.read_bytes(); schedule=json.loads(raw)
    entries=schedule.get('entries')
    if (schedule.get('kind')!='tdyssey.gpc_training_schedule.v1'
            or schedule.get('source_manifest_sha256')!=data['sha256']
            or not isinstance(entries,list) or len(entries)!=500):
        raise ValueError('Schedule identity/source/fixed500 count differs')
    encoded=json.dumps(entries,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest()!=schedule.get('order_sha256'):
        raise ValueError('Frozen schedule order hash differs')
    dataset=GPCAdaptationDataset(data_path,'train'); cache={}; selected=[]; quota={}; identities={}
    for ordinal,entry in enumerate(entries):
        index=entry.get('dataset_index')
        if type(index) is not int or not 0<=index<dataset.offsets[-1]:
            raise ValueError('Schedule index is outside source training windows')
        n=bisect_right(dataset.offsets,index)-1; row=dataset.episodes[n]
        anchor=3+index-dataset.offsets[n]
        if (entry.get('cursor')!=ordinal or entry.get('step')!=ordinal+1
                or entry.get('shape')!=('T','L')[ordinal%2] or entry['shape']!=row['shape']
                or entry.get('trajectory_id')!=row['trajectory_id'] or entry.get('anchor')!=anchor
                or entry.get('initial_state_sha256')!=row['initial_state_sha256'] or row['role']!='train'):
            raise ValueError('Schedule entry does not identify its exact source-train window')
        identifier=row['trajectory_id']
        if identifier not in cache:
            cache[identifier],identity=_episode(row,guard); identities[identity['path']]=identity
        target=cache[identifier][anchor]; signed=target['signed_rotation']; contact=target['contact']
        stratum=('contact_rotation' if contact else 'rotation_only') if abs(signed)>5 else ('contact_only' if contact else 'other')
        if (entry.get('contact') is not contact or entry.get('stratum')!=stratum
                or type(entry.get('rotation_degrees')) not in (int,float)
                or abs(entry['rotation_degrees']-signed)>1e-9):
            raise ValueError('Schedule privileged labels differ from the source target control')
        selected.append(target); quota[identifier]=quota.get(identifier,0)+1
    unique={(r['episode_id'],r['control']):r for r in selected}
    for row in dataset.episodes:
        identifier=row['trajectory_id']
        if identifier not in cache:
            cache[identifier],identity=_episode(row,guard); identities[identity['path']]=identity
    full_controls=[r for row in dataset.episodes for r in cache[row['trajectory_id']]]
    full_windows=[r for row in dataset.episodes for r in cache[row['trajectory_id']][3:]]
    old_indices=training_prefix(dataset); old_selected=[]
    for index in old_indices:
        n=bisect_right(dataset.offsets,index)-1; row=dataset.episodes[n]
        old_selected.append(cache[row['trajectory_id']][3+index-dataset.offsets[n]])
    old_unique={(r['episode_id'],r['control']):r for r in old_selected}
    old_prefix=dict(updates=500,indices=old_indices,weighted_target_statistics=_shapes(old_selected),
        unique_window_statistics=_shapes(old_unique.values()),unique_windows=len(old_unique),repeated_window_occurrences=500-len(old_unique))
    independent={}
    for shape in ('T','L'):
        val=[r for r in data['episodes'] if r['role']=='independent_validation' and r['shape']==shape and r['payloads_present']]
        count=sum(r['windows'] for r in val)
        independent[shape]=dict(episodes=len(val),unique_scenes=len({r['initial_state_sha256'] for r in val}),windows=count,required_distinct_windows=8,sufficient=count>=8)
    by_shape={s:dict(windows=sum(r['shape']==s for r in selected),
        episodes=len({r['episode_id'] for r in selected if r['shape']==s}),
        scenes=len({r['scene'] for r in selected if r['shape']==s})) for s in ('T','L')}
    return dict(kind='gpc_frozen_training_schedule_audit.v1',
        schedule=dict(path=str(schedule_path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
            order_sha256=schedule['order_sha256'],rule_sha256=schedule.get('rule_sha256'),source_manifest_sha256=data['sha256']),
        schedule_regenerated=False,updates=500,by_shape=by_shape,per_episode_windows=quota,
        weighted_target_statistics=_shapes(selected),unique_window_statistics=_shapes(unique.values()),
        unique_windows=len(unique),repeated_window_occurrences=500-len(unique),
        old500_prefix=old_prefix,full_training_corpus=dict(episodes=len(dataset.episodes),controls=_shapes(full_controls),window_targets=_shapes(full_windows)),
        overlap=dict(new_schedule=_window_overlap(selected),old500_prefix=_window_overlap(old_selected),
            definition='Occurrence identities are (trajectory,index). Unordered same-scene pairs overlap full five frames at |anchor delta|<=4 and context/actions at <=3. Duplicate window occurrences remain counted.'),
        independent_validation_support=independent,
        permitted_schedule_record_identities=identities,
        definitions=dict(direction='Positive wrapped source angle is clockwise in the y-down image convention; negative is counterclockwise.',
            target='Executed control at anchor; strictly >5 degrees, unchanged from the frozen sampling rule.',
            scope='Read entries already generated once. This audit does not reselect, regenerate or train.'),
        held_model_test=dict(episodes=sum(r['role']=='held_model_test' for r in data['episodes']),payloads_opened=0,payloads_hashed=0),
        operations=dict(model_operations=0,simulation=0,training=0,image_archive_opens=0,held_payload_opens=0,nontraining_record_opens=0))
