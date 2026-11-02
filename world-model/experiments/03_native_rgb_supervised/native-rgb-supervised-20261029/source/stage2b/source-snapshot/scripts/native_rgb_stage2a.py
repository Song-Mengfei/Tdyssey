"""Parent-released first formal TRAIN batch; no validation/model/training stages."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_training import atomic_json,sha256
from scripts.native_rgb_supervised_stage1 import StageLedger,storage_check,contacted_bin_ids,verify_source

LETTERS=tuple('ABCEFHIJLMOPSTUVXY')
LOCATIONS=('center','left','right','top','bottom','left-top','right-top','left-bottom','right-bottom')


def diagnostic_json(value):
    """Encode failed nonfinite diagnostics as null; raw NPZ values are untouched."""
    if isinstance(value,float):return value if math.isfinite(value) else None
    if isinstance(value,dict):return {k:diagnostic_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [diagnostic_json(v) for v in value]
    if hasattr(value,'tolist'):return diagnostic_json(value.tolist())
    return value


class TrainingLedger(StageLedger):
    def begin_trajectory(self,trajectory_id,maximum_controls=64):
        if maximum_controls!=64 or len(self.value['trajectories'])>=162:
            raise RuntimeError('Exactly the released162 slots, at most64 controls each; no replacement')
        if any(row['status']=='running' for row in self.value['trajectories']):
            raise RuntimeError('An unfinished slot may not be silently retried')
        row=dict(trajectory_id=trajectory_id,maximum_controls=64,charged_controls=0,
            completed_controls=0,status='running',pending_control=None)
        self.value['trajectories'].append(row);self.save()
        return len(self.value['trajectories'])-1


def authorize(output):
    parent=Path(output).resolve().parent
    (parent/'SUPERVISION.md').read_text(encoding='utf-8')
    state=json.loads((parent/'stage-state.json').read_text())
    if state['current_stage']!='2A' or state['authorized_stages']!=['2A'] or state['status']!='in_progress' or state['callback']['sent']:
        raise RuntimeError('Only currently released2A is allowed; read supervision')
    return parent,state


def load_config(path):
    cfg=json.loads(Path(path).read_text())
    from tdyssey.native_rgb_env import PROTOCOL_SHA256
    if (tuple(cfg['letters'])!=LETTERS or tuple(cfg['locations'])!=LOCATIONS or cfg['split']!='train'
        or cfg['max_execution_slots']!=162 or cfg['max_attempted_controls']!=10368 or cfg['controls_per_trajectory']!=64
        or cfg['model_calls_allowed']!=0 or cfg['optimizer_updates_allowed']!=0 or cfg['validation_collection_allowed']!=0
        or cfg['environment']['protocol_sha256']!=PROTOCOL_SHA256):
        raise ValueError('The exact released2A scope/protocol cannot be enlarged')
    return cfg


def prepare(output,config):
    import numpy as np
    output=Path(output);parent,_=authorize(output);cfg=load_config(config)
    if (output/'frozen-requests.json').exists():raise FileExistsError('Preserve the first frozen162 requests')
    hashes=set();seeds=set();sources=[]
    for key in ['old_native_manifest','stage1_manifest','stage1b_manifest']:
        path=ROOT/cfg['dedup'][key];m=json.loads(path.read_text())
        for e in m['entries']:
            hashes.add(e['initial_scene_hash']);seeds.add(int(e['seed']))
        sources.append(dict(path=str(path.resolve()),sha256=sha256(path),use='initial_state_hash and seed only'))
    # Include raw old-native probes that were never accepted into its train data.
    for path in sorted((ROOT/cfg['old_artifact_root']).rglob('*.npz')):
        with np.load(path,allow_pickle=False) as arrays:
            if 'states' not in arrays or not len(arrays['states']):continue
            initial=np.asarray(arrays['states'][0],dtype=np.float64)
            hashes.add(hashlib.sha256(initial.tobytes()).hexdigest())
        sources.append(dict(path=str(path.resolve()),sha256=sha256(path),use='states[0] identity only; no motion/image/validation feedback'))
    requests=[]
    for location_index,location in enumerate(LOCATIONS):
        for letter_index,letter in enumerate(LETTERS):
            seed=int(cfg['seed_base'])+location_index*1000+letter_index
            if seed in seeds:raise ValueError('Predeclared seed collides with historical identity; do not silently replace')
            seeds.add(seed);tier=(location_index-1+letter_index)%3
            requests.append(dict(request_id=f'train-2a-{len(requests):03d}',letter=letter,spatial_class=location,
                location=location,seed=seed,split='train',controls=64,
                gap_band=None if location=='center' else cfg['sampling']['wall_gap_bands_world_px'][tier],
                gap_world_px=None if location=='center' else cfg['sampling']['wall_gap_representatives_world_px'][tier],
                sampling=cfg['sampling']))
    assert len(requests)==162 and len({x['seed'] for x in requests})==162
    atomic_json(output/'dedup-reference.json',dict(kind='stage2a-identity-only-dedup.v1',sources=sources,
        initial_state_hashes=sorted(hashes),historical_seeds=sorted(seeds-set(x['seed'] for x in requests)),
        validation_use='identity-only exclusion; not selection or performance feedback'))
    result=dict(kind='stage2a-frozen-formal-train-requests.v1',requests=requests,split='train',
        maximum_slots=162,maximum_controls=10368,previous_supervised_controls=3968,
        planned_supervised_controls=14336,historical_plus_planned_native_controls=48896,
        no_validation_or_probe_records_in_formal_train=True,examples=cfg['examples'],
        config_sha256=sha256(config),dedup_reference_sha256=sha256(output/'dedup-reference.json'))
    atomic_json(output/'frozen-requests.json',result)
    return result


def freeze(output,config,physics_root):
    output=Path(output)
    files=['scripts/native_rgb_stage2a.py','scripts/native_rgb_stage2a_report.py','scripts/native_rgb_supervised_stage1.py',
        'tdyssey/native_rgb_supervised_sampling.py','tdyssey/native_rgb_motion_statistics.py','tdyssey/native_rgb_env.py',
        'tdyssey/native_rgb_data.py','tdyssey/native_rgb_training.py','tdyssey/envs/official.py','tdyssey/geometry.py',
        'tdyssey/letters.py','tdyssey/collection.py','tdyssey/state.py','tdyssey/official.py','tdyssey/source_export.py','manifests/official_dp.json']
    pairs=[(ROOT/p,p) for p in files]
    pairs.extend([(Path(config),'frozen-config.json'),(output/'frozen-requests.json','frozen-requests.json'),
        (output/'dedup-reference.json','dedup-reference.json')])
    cfg=load_config(config)
    pairs.append((ROOT/cfg['sampling']['prior_coverage'],'historical-TRAIN-contact-coverage.json'))
    pairs.extend((Path(physics_root)/p,'official-source/'+p) for p in ['diffusion_policy/env/pusht/pusht_env.py','diffusion_policy/env/pusht/pymunk_override.py'])
    rows=[]
    for source,name in pairs:
        target=output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(source.read_bytes())
        rows.append(dict(source=str(source.resolve()),snapshot=target.relative_to(output).as_posix(),sha256=sha256(target),bytes=target.stat().st_size))
    atomic_json(output/'source-identity.json',dict(kind='stage2a-source-before-actual-controls.v1',files=rows,
        recovery_logic='unchanged1B; only18letter entry and deterministic side_sign extension',validation_feedback=False))


def run_slot(output,request,physics_root,ledger,coverage,counts,historical,initial_hashes,planned_attempts=None):
    import numpy as np
    from tdyssey.native_rgb_env import make_env
    from tdyssey.native_rgb_data import _contact_coverage
    from tdyssey.native_rgb_supervised_sampling import initialize_scene,ContinuousContactSampler
    data=Path(output)/'data';tid=request['request_id'];letter=request['letter']
    attempt=ledger.begin_trajectory(tid,64);env=None;scene=None;error=None;exception_state=None
    states=[];actions=[];contacts=[];valid=[];penetrations=[];substeps=[];points=[];offsets=[0];angles=[];steps=[]
    step_entered=False;initial_hash=None;initial_unique=False
    selection_inputs=dict(new_2a_contact_controls_before={int(k):int(v) for k,v in counts.items()},
        historical_TRAIN_independent_trajectories={int(k):int(v) for k,v in historical.items()},
        validation_feedback=False,previous_probes_added_to_counts=False)
    if planned_attempts is not None:
        selection_inputs.pop('new_2a_contact_controls_before')
        selection_inputs.update(formal_train_contact_controls_before={int(k):int(v) for k,v in counts.items()},
            planned_initial_attempts_before={int(k):int(v) for k,v in planned_attempts.items()},
            planned_attempt_counted=False)
    try:
        env=make_env(physics_root,letter)
        scene=initialize_scene(env,request,coverage_counts=counts,historical_counts=historical,planned_attempts=planned_attempts)
        initial=env.state().copy();states=[initial];penetrations=[env.penetrations().copy()]
        initial_hash=hashlib.sha256(initial.tobytes()).hexdigest();initial_unique=initial_hash not in initial_hashes
        if env.control_step!=0 or not scene['legal'] or np.max(penetrations[0])>.1 or not initial_unique:
            raise RuntimeError('Illegal/duplicate initialization or hidden control; no replacement slot')
        initial_hashes.add(initial_hash)
        policy=ContinuousContactSampler(env,coverage,counts,scene)
        for t in range(64):
            step_entered=False
            if env.control_step!=t or not np.array_equal(env.state(),states[-1]):raise RuntimeError('Uncharged state mutation')
            action,metadata=policy.next_action(t)
            if not np.array_equal(env.state(),states[-1]):raise RuntimeError('Planning mutated state')
            action=np.asarray(action,dtype=np.float64)
            if action.shape!=(2,) or not np.isfinite(action).all() or (action<0).any() or (action>511).any():raise ValueError('Absolute action contract')
            before=env.state().copy();ledger.charge(attempt);step_entered=True
            try:
                _,info=env.step(action)
            finally:
                if t==0 and planned_attempts is not None:
                    arc=int(scene['staging_segment'])
                    planned_attempts[arc]=planned_attempts.get(arc,0)+1
                    selection_inputs['planned_attempt_counted']=True
                    selection_inputs['planned_initial_attempts_after_start']=dict(planned_attempts)
                    ledger.value['trajectories'][attempt]['initial_arc_attempt_counted']=arc
                    ledger.save()
            after=env.state().copy();sub=np.asarray(env.substep_penetrations,dtype=np.float64)
            if sub.shape!=(10,3) or info['physics_substeps']!=10 or env.control_step!=t+1:raise RuntimeError('100Hz timeline contract')
            states.append(after);actions.append(action);contacts.append(info['contact']);valid.append(info['valid'])
            penetrations.append(info['penetration']);substeps.append(sub);points.extend(info['contact_points'].tolist());offsets.append(len(points));angles.extend(info['contact_angles'].tolist())
            steps.append(dict(metadata,step=t));ledger.complete(attempt,10)
            observed=policy.observe(before,action,after,info);steps[-1]['observed']=observed
            if env.control_step!=t+1 or not np.array_equal(env.state(),after):raise RuntimeError('Observe mutated physical state')
            _contact_coverage(letter,coverage,info['contact_points'],tid,moving=observed['moving'],
                direction=str(metadata.get('approach_kind','unknown')),angles=info['contact_angles'])
            for index in contacted_bin_ids(letter,coverage,info['contact_points']):
                row=coverage[index];row['contact_control_count']+=1;row['moving_contact_control_count']+=int(observed['moving'])
                row['trajectory_contact_controls'][tid]=row['trajectory_contact_controls'].get(tid,0)+1
                counts[index]=row['contact_control_count']
            if not info['valid']:raise RuntimeError('Invalid physical transition at unchanged2px threshold')
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        if env is not None:
            pending=ledger.value['trajectories'][attempt]['pending_control']
            partial=len(env.substep_penetrations) if step_entered and pending is not None else 0
            ledger.value['recorded_partial_substeps_lower_bound']+=partial
            exception_state=dict(state=env.state().tolist(),step_entered=step_entered,pending_control=pending,
                recorded_partial_substeps_lower_bound=partial,exact_partial_execution_unconfirmed=bool(step_entered and pending is not None))
    if states:
        relative=f'trajectories/{tid}-{letter}.npz';path=data/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream,states=np.asarray(states,dtype=np.float64),actions=np.asarray(actions,dtype=np.float32).reshape(-1,2),
                contact=np.asarray(contacts,dtype=np.int16),valid_transition=np.asarray(valid,dtype=bool),
                penetrations=np.asarray(penetrations,dtype=np.float64).reshape(-1,3),substep_penetrations=np.asarray(substeps,dtype=np.float64).reshape(-1,10,3),
                contact_points=np.asarray(points,dtype=np.float32).reshape(-1,2),contact_offsets=np.asarray(offsets,dtype=np.int32),contact_angles=np.asarray(angles,dtype=np.float32))
        meta=path.with_suffix('.json');atomic_json(meta,diagnostic_json(dict(request=request,scene=scene,selection_inputs=selection_inputs,steps=steps,error=error,
            exception_state=exception_state,initial_state_unique=initial_unique,hidden_physics_controls=0,
            nonfinite_failure_json_encoding='null; raw NPZ arrays retained unchanged')))
        entry=dict(request_id=tid,trajectory_id=tid,shape=letter,letter=letter,split='train',stage=request.get('stage','2A'),
            location=request['spatial_class'],spatial_class=request['spatial_class'],seed=request['seed'],
            file=relative,sha256=sha256(path),metadata_file=meta.relative_to(data).as_posix(),metadata_sha256=sha256(meta),
            initial_scene_hash=initial_hash,initial_scene_unique=initial_unique,initial_scene=states[0].tolist(),
            selected_initial_arc=scene.get('staging_segment') if scene else None,
            selection_version=scene.get('selection_version','motion-room-actual-hit-prior-v1') if scene else None,
            gap_requested_world_px=request['gap_world_px'],gap_band_requested=request['gap_band'],
            controls=len(actions),invalid_controls=sum(not v for v in valid),error=error,
            parent_release_required_before_training=True)
    else:entry=None
    ledger.finish(attempt,'failed' if error else 'completed',saved_controls=len(actions),error=error)
    return diagnostic_json(entry),error


def collect(output,config,physics_root):
    output=Path(output).resolve();authorize(output)
    lock=output/'COLLECTION_RUNNING.lock'
    with lock.open('x') as stream:stream.write('2A')
    try:return _collect_locked(output,config,physics_root)
    finally:lock.unlink(missing_ok=True)


def _collect_locked(output,config,physics_root):
    from tdyssey.native_rgb_env import CONTROLLER,PROTOCOL_SHA256,STATE_COLUMNS
    from tdyssey.native_rgb_data import contour_segments,_reachable_segments,_write_coverage
    from tdyssey.native_rgb_env import make_env
    output=Path(output).resolve();parent,state=authorize(output);cfg=load_config(config)
    if (output/'data/manifest.json').exists():raise FileExistsError('Inspected batch cannot be replayed/replaced')
    plan=json.loads((output/'frozen-requests.json').read_text());dedup=json.loads((output/'dedup-reference.json').read_text())
    if len(plan['requests'])!=162 or plan['config_sha256']!=sha256(config) or plan['dedup_reference_sha256']!=sha256(output/'dedup-reference.json'):
        raise RuntimeError('Frozen162 requests/config identity changed')
    seen=set(dedup['initial_state_hashes']);storage_check(cfg,parent)
    prior=json.loads((ROOT/cfg['sampling']['prior_coverage']).read_text())['segments']
    coverage={l:contour_segments(l,8) for l in LETTERS}
    for rows in coverage.values():
        for row in rows:row.update(contact_control_count=0,moving_contact_control_count=0,trajectory_contact_controls={})
    counts={l:{row['id']:0 for row in coverage[l]} for l in LETTERS}
    historical={l:{row['id']:len(row['trajectory_ids']) for row in prior[l]} for l in LETTERS}
    entries=[];failed_requests=[];failure=None;started=time.monotonic();ledger=None
    try:
        ledger=TrainingLedger(output,cap=10368)
        if ledger.value['trajectories'] or ledger.value['charged_controls']:raise RuntimeError('No replay or failure replacement')
        ledger.value.update(stage='2A',split='train',maximum_execution_slots=162,previous_supervised_controls=3968,
            historical_native_controls=34560,historical_native_optimizer_attempts=1899,historical_native_training_seconds=5352.516)
        ledger.save();freeze(output,config,physics_root)
        for request in plan['requests']:
            verify_source(output);storage_check(cfg,parent)
            entry,error=run_slot(output,request,physics_root,ledger,coverage[request['letter']],counts[request['letter']],historical[request['letter']],seen)
            if entry is not None:entries.append(entry)
            if error:failed_requests.append(dict(request=request,error=error));failure=error
            atomic_json(output/'collection-progress.json',dict(executed_slots=len(ledger.value['trajectories']),planned_slots=162,
                charged_controls=ledger.value['charged_controls'],last_request=request['request_id'],failure=failure))
            print(json.dumps(dict(slot=len(ledger.value['trajectories']),letter=request['letter'],location=request['spatial_class'],
                charged_controls=ledger.value['charged_controls'],error=error)),flush=True)
            if failure:break
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        try:final_storage=storage_check(cfg,parent)
        except BaseException as error:
            final_storage=dict(error=type(error).__name__+': '+str(error))
            failure=failure or 'Final resource check: '+str(error)
        data=output/'data';data.mkdir(exist_ok=True)
        used=ledger.value['charged_controls'] if ledger is not None else 0
        actual=ledger.value['completed_controls'] if ledger is not None else 0
        atomic_json(data/'manifest.json',dict(kind='native-rgb-supervised-formal-train-batch.v1',stage='2A',split='train',
            entries=entries,failure=failure,failed_requests=failed_requests,planned_slots=162,
            unexecuted_request_ids=[q['request_id'] for q in plan['requests'][len(ledger.value['trajectories']) if ledger else 0:]],
            charged_controls=used,actual_controls=actual,saved_controls=sum(e['controls'] for e in entries),
            controller=CONTROLLER,protocol_sha256=PROTOCOL_SHA256,state_columns=list(STATE_COLUMNS),source_root=str(Path(physics_root).resolve()),
            frozen_requests_sha256=sha256(output/'frozen-requests.json'),source_identity_sha256=sha256(output/'source-identity.json') if (output/'source-identity.json').exists() else None,
            no_probe_or_validation_entries=True,parent_release_required_before_training=True,model_calls=0,optimizer_updates=0))
        atomic_json(output/'contact-coverage.json',dict(segments=coverage,scope='2A actual TRAIN collisions only; no prior contacts added'))
        if ledger is not None:
            ledger.value.update(collection_wall_seconds=time.monotonic()-started,supervised_round_controls=3968+used,
                historical_plus_new_native_controls=38528+used,physics_substeps_lower_bound=ledger.value['physics_substeps']+ledger.value['recorded_partial_substeps_lower_bound'],
                charged_physics_substep_allowance=used*10,failure=failure,storage_after_collection=final_storage)
            ledger.save()
    return dict(failure=failure,charged_controls=used,actual_controls=actual,executed_slots=len(ledger.value['trajectories']) if ledger else 0)


def audit(output,expected_slots=162,max_controls=10368,stage='2A'):
    import numpy as np
    from tdyssey.native_rgb_env import make_env,PROTOCOL_SHA256
    output=Path(output);manifest=json.loads((output/'data/manifest.json').read_text());plan=json.loads((output/'frozen-requests.json').read_text())
    verify_source(output);dedup=json.loads((output/'dedup-reference.json').read_text());seen=set(dedup['initial_state_hashes']);seeds=set(dedup['historical_seeds'])
    expected={q['request_id']:q for q in plan['requests']};issues=[];rows=[];maxima=np.zeros(3);invalid=0;h1=0;h4=0;envs={}
    for e in manifest['entries']:
        row_issues=[];q=expected.get(e['request_id']);path=output/'data'/e['file']
        if q is None or (e['shape'],e['location'],e['seed'],e['split'])!=(q['letter'],q['spatial_class'],q['seed'],'train'):row_issues.append('request_or_split_binding')
        if sha256(path)!=e['sha256'] or sha256(output/'data'/e['metadata_file'])!=e['metadata_sha256']:row_issues.append('file_identity')
        with np.load(path,allow_pickle=False) as a:
            n=len(a['actions']);s=a['states'];sub=a['substep_penetrations'];pen=a['penetrations']
            digest=hashlib.sha256(s[0].tobytes()).hexdigest()
            if digest in seen or e['seed'] in seeds or digest!=e['initial_scene_hash'] or not e['initial_scene_unique']:row_issues.append('scene_or_seed_duplicate')
            seen.add(digest);seeds.add(e['seed'])
            if s.shape!=(n+1,15) or sub.shape!=(n,10,3):row_issues.append('alignment')
            if not np.isfinite(s).all() or not np.isfinite(sub).all() or not np.isfinite(a['actions']).all():row_issues.append('nonfinite')
            if a['actions'].shape!=(n,2) or (a['actions']<0).any() or (a['actions']>511).any():row_issues.append('action_contract')
            if n and not np.array_equal(sub.max(axis=1),pen[1:]):row_issues.append('substep_max')
            computed=sub.max(axis=(1,2))<=2 if n else np.ones(0,dtype=bool)
            if not np.array_equal(computed,a['valid_transition']):row_issues.append('valid_flags')
            bad=int((~computed).sum());invalid+=bad
            h1+=sum(computed[t-3:t+1].all() for t in range(3,n))
            h4+=sum(computed[t-3:t+4].all() for t in range(3,n-3))
            if n:maxima=np.maximum(maxima,sub.max(axis=(0,1)))
            env=envs.get(e['shape'])
            if env is None:env=make_env(manifest['source_root'],e['shape']);env.reset_scene();envs[e['shape']]=env
            for i,state in enumerate(s):
                env.set_render_state(state)
                if (env.penetrations()>pen[i]+1e-3).any():row_issues.append('endpoint_penetration_underreported')
            if a['contact_offsets'].shape!=(n+1,) or a['contact_offsets'][-1]!=len(a['contact_points']) or len(a['contact_angles'])!=len(a['contact_points']):row_issues.append('contact_alignment')
        if e['error']:row_issues.append('failed_slot')
        rows.append(dict(trajectory_id=e['trajectory_id'],controls=n,invalid_controls=bad,issues=sorted(set(row_issues))))
        issues.extend(row_issues)
    if manifest['failure']:issues.append('collection_stopped_on_error')
    if manifest['protocol_sha256']!=PROTOCOL_SHA256:issues.append('protocol_identity')
    if manifest['frozen_requests_sha256']!=sha256(output/'frozen-requests.json'):issues.append('frozen_requests_identity')
    if len(rows)!=expected_slots:issues.append('incomplete_released_batch')
    if invalid:issues.append('invalid_actual_transition')
    ledger=json.loads((output/'resource-ledger.json').read_text())
    if ledger['charged_controls']>max_controls or len(ledger['trajectories'])>expected_slots:issues.append('budget_exceeded')
    report=dict(kind=f'stage{stage.lower()}-fixed-physical-data-audit.v1',passed=not issues,physics_collection_gate_passed=not issues,
        issues=sorted(set(issues)),invalid_controls=invalid,max_substep_penetration_world_px=maxima.tolist(),
        controls=sum(x['controls'] for x in rows),audited_substeps=sum(x['controls'] for x in rows)*10,
        valid_h1_candidates=int(h1),valid_h4_candidates=int(h4),invalid_candidates_allowed_into_train=0,
        threshold_world_px=2,threshold_relaxed=False,trajectories=rows,manifest_sha256=sha256(output/'data/manifest.json'),
        no_training_consumption=True,extra_physics_controls=0,model_calls=0)
    atomic_json(output/'physics-audit.json',report);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','collect','audit'])
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/native-rgb-supervised-20261029/stage2a')
    p.add_argument('--config',type=Path,default=ROOT/'configs/native_rgb_supervised_stage2a.json')
    p.add_argument('--physics-root',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv);authorize(a.output)
    if a.stage=='prepare':result=prepare(a.output,a.config)
    elif a.stage=='audit':result=audit(a.output)
    else:
        if not a.execute:raise RuntimeError('Real TRAIN collection requires --execute')
        result=collect(a.output,a.config,a.physics_root)
    print(json.dumps({k:v for k,v in result.items() if k in ['failure','passed','charged_controls','actual_controls','executed_slots','issues']},indent=2))
    return 1 if result.get('failure') or result.get('passed') is False else 0


if __name__=='__main__':raise SystemExit(main())
