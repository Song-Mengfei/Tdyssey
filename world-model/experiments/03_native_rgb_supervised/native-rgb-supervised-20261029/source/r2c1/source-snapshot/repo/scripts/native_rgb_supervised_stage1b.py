"""Exactly eight predeclared recovery probes; stage2 and training remain blocked."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_training import atomic_json,sha256
from scripts.native_rgb_supervised_stage1 import StageLedger,contacted_bin_ids,storage_check,verify_source,audit_probe


def authorize(output):
    root=Path(output).resolve().parent
    (root/'SUPERVISION.md').read_text(encoding='utf-8')
    state=json.loads((root/'stage-state.json').read_text(encoding='utf-8'))
    if state['current_stage']!='1B' or state['authorized_stages']!=['1B'] or state['callback']['sent'] or state['status']!='in_progress':
        raise RuntimeError('Only the released1B scope may execute; read supervision')
    return root,state


def prepare(output):
    import numpy as np
    from tdyssey.native_rgb_supervised_sampling import plan_scene
    output=Path(output);root,state=authorize(output)
    target=output/'frozen-slots.json'
    if target.exists():raise FileExistsError('Never replace a frozen eight-slot list')
    manifest=json.loads((root/'probe/manifest.json').read_text())
    coverage=json.loads((root/'contact-coverage.json').read_text())['segments']
    prior=json.loads((root/'source-snapshot/historical-contact-coverage.json').read_text())['segments']
    entries=manifest['entries'];order={e['trajectory_id']:i for i,e in enumerate(entries)}
    wanted=[('X','center'),('X','left-bottom'),('X','right-top'),('B','left-top'),('O','left-top'),('T','center')]
    slots=[];old_hashes=set()
    for e in entries:
        old_hashes.add(e['initial_scene_hash'])
    for letter,location in wanted:
        e=next(e for e in entries if e['shape']==letter and e['location']==location)
        assert sha256(root/'probe'/e['file'])==e['sha256']
        assert sha256(root/'probe'/e['metadata_file'])==e['metadata_sha256']
        metadata=json.loads((root/'probe'/e['metadata_file']).read_text())
        with np.load(root/'probe'/e['file'],allow_pickle=False) as a:initial=a['states'][0].copy()
        assert hashlib.sha256(initial.tobytes()).hexdigest()==e['initial_scene_hash']
        counts={row['id']:sum(c for tid,c in row['trajectory_contact_controls'].items() if order[tid]<order[e['trajectory_id']]) for row in coverage[letter]}
        slots.append(dict(letter=letter,spatial_class=location,role='paired',baseline_trajectory_id=e['trajectory_id'],
            seed=e['seed'],scene=metadata['scene'],initial_state=initial.tolist(),initial_state_sha256=e['initial_scene_hash'],
            initial_coverage_counts=counts,historical_counts={row['id']:len(row['trajectory_ids']) for row in prior[letter]}))
    for location,seed,gap in [('center',2026102991,None),('right',2026103991,18.)]:
        counts={row['id']:row['contact_control_count'] for row in coverage['X']}
        historical={row['id']:len(row['trajectory_ids']) for row in prior['X']}
        sampling=manifest['config']['sampling']
        scene=plan_scene('X',location,seed,gap_world_px=gap,gap_band=None if gap is None else [8,32],
            sampling=sampling,coverage_counts=counts,historical_counts=historical)
        initial=np.asarray([*scene['pusher_start'],*scene['object_origin'],scene['object_angle'],*([0.]*10)],dtype=np.float64)
        digest=hashlib.sha256(initial.tobytes()).hexdigest()
        assert digest not in old_hashes and seed not in [e['seed'] for e in entries]
        old_hashes.add(digest)
        slots.append(dict(letter='X',spatial_class=location,role='new',baseline_trajectory_id=None,seed=seed,
            scene=scene,initial_state=initial.tolist(),initial_state_sha256=digest,
            initial_coverage_counts=counts,historical_counts=historical))
    for i,slot in enumerate(slots):
        slot['slot_id']=f'stage1b-{i:02d}'
        slot['new_trajectory_id']=f"stage1b-{i:02d}-{slot['letter']}-{slot['spatial_class']}-{slot['role']}"
        slot['maximum_controls']=64
    result=dict(kind='native-rgb-supervised-1b-frozen-slots.v1',slots=slots,previous_controls=3456,
        new_max_controls=512,combined_cap=4000,combined_planned_controls=3968,
        maximum_execution_slots=62,paired_scenes_are_replays_not_new_independent_scenes=True,
        baseline_manifest_sha256=sha256(root/'probe/manifest.json'))
    atomic_json(target,result)
    return result


def freeze(output,physics_root):
    output=Path(output);paths=['scripts/native_rgb_supervised_stage1b.py','scripts/native_rgb_supervised_stage1.py',
        'scripts/native_rgb_supervised_recontact_report.py','tdyssey/native_rgb_supervised_sampling.py',
        'tdyssey/native_rgb_motion_statistics.py','tdyssey/native_rgb_env.py','tdyssey/native_rgb_data.py',
        'tdyssey/native_rgb_training.py','tdyssey/envs/official.py','tdyssey/geometry.py','tdyssey/letters.py','manifests/official_dp.json']
    rows=[]
    pairs=[(ROOT/p,p) for p in paths]+[(output/'frozen-slots.json','frozen-slots.json'),(output/'baseline-preservation.json','baseline-preservation.json')]
    pairs.extend((Path(physics_root)/p,'official-source/'+p) for p in ['diffusion_policy/env/pusht/pusht_env.py','diffusion_policy/env/pusht/pymunk_override.py'])
    for source,name in pairs:
        dest=output/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write(source.read_bytes())
        rows.append(dict(source=str(source.resolve()),snapshot=dest.relative_to(output).as_posix(),sha256=sha256(dest),bytes=dest.stat().st_size))
    atomic_json(output/'source-identity.json',dict(kind='stage1b-source-before-real-controls.v1',files=rows))


def collect(output,physics_root):
    import numpy as np
    from tdyssey.native_rgb_env import make_env,CONTROLLER,PROTOCOL_SHA256,STATE_COLUMNS
    from tdyssey.native_rgb_data import contour_segments
    from tdyssey.native_rgb_supervised_sampling import ContinuousContactSampler
    output=Path(output).resolve();root,state=authorize(output)
    if (output/'probe/manifest.json').exists():raise FileExistsError('Do not repeat or replace inspected1B slots')
    plan=json.loads((output/'frozen-slots.json').read_text());assert len(plan['slots'])==8
    original=json.loads((root/'resource-ledger.json').read_text());assert original['charged_controls']==3456
    if original['charged_controls']+plan['new_max_controls']>4000:raise RuntimeError('Combined control cap')
    cfg=json.loads((root/'source-snapshot/frozen-config.json').read_text())
    storage_check(cfg,root)
    lock=output/'COLLECTION_RUNNING.lock'
    with lock.open('x') as stream:stream.write('stage1b')
    started=time.monotonic();entries=[];failure=None
    try:
        ledger=StageLedger(output,cap=512)
        if ledger.value['charged_controls'] or ledger.value['trajectories']:raise RuntimeError('Spent1B slots cannot be replayed')
        ledger.value.update(previous_stage1_controls=3456,max_new_execution_slots=8,max_combined_controls=4000,
            previous_ledger_sha256=sha256(root/'resource-ledger.json'),baseline_slots=54,max_combined_slots=62)
        ledger.save();freeze(output,physics_root)
        for slot in plan['slots']:
            verify_source(output);storage_check(cfg,root)
            if len(ledger.value['trajectories'])>=8:raise RuntimeError('No replacement execution slots')
            tid=slot['new_trajectory_id'];attempt=ledger.begin_trajectory(tid,64)
            env=make_env(physics_root,slot['letter']);scene=deepcopy(slot['scene'])
            env.reset_scene(center=scene['object_mass_center'],angle=scene['object_angle'],pusher=scene['pusher_start'])
            initial=env.state().copy();initial_hash=hashlib.sha256(initial.tobytes()).hexdigest()
            slot_error=None;exception_state=None;steps=[];states=[initial];actions=[];contacts=[];valid=[]
            penetrations=[env.penetrations().copy()];substeps=[];points=[];offsets=[0];angles=[]
            try:
                if not scene['legal'] or env.control_step!=0 or initial_hash!=slot['initial_state_sha256'] or np.max(env.penetrations())>.1:
                    raise RuntimeError('Frozen original/new initial state or legality mismatch; no replacement slot')
                segments=contour_segments(slot['letter'],8)
                counts={int(k):v for k,v in slot['initial_coverage_counts'].items()}
                policy=ContinuousContactSampler(env,segments,counts,scene)
                for t in range(64):
                    if env.control_step!=t or not np.array_equal(env.state(),states[-1]):raise RuntimeError('Uncharged state mutation')
                    action,meta=policy.next_action(t)
                    if not np.array_equal(env.state(),states[-1]):raise RuntimeError('Planning mutated state')
                    action=np.asarray(action,dtype=np.float64)
                    if action.shape!=(2,) or not np.isfinite(action).all() or (action<0).any() or (action>511).any():raise ValueError('Invalid action')
                    before=env.state().copy();ledger.charge(attempt)
                    _,info=env.step(action);after=env.state().copy();sub=np.asarray(env.substep_penetrations,dtype=np.float64)
                    if sub.shape!=(10,3) or env.control_step!=t+1:raise RuntimeError('Ten-substep timeline mismatch')
                    states.append(after);actions.append(action);contacts.append(info['contact']);valid.append(info['valid'])
                    penetrations.append(info['penetration']);substeps.append(sub)
                    points.extend(info['contact_points'].tolist());offsets.append(len(points));angles.extend(info['contact_angles'].tolist())
                    steps.append(dict(meta,step=t));ledger.complete(attempt,10)
                    steps[-1]['observed']=policy.observe(before,action,after,info)
                    if env.control_step!=t+1 or not np.array_equal(env.state(),after):raise RuntimeError('Observe mutated state')
                    for bin_id in contacted_bin_ids(slot['letter'],segments,info['contact_points']):counts[bin_id]=counts.get(bin_id,0)+1
                    if not info['valid']:
                        slot_error='invalid_physical_transition_fixed_2px';break
            except BaseException as error:
                slot_error=type(error).__name__+': '+str(error)
                exception_state=dict(state=env.state().tolist(),recorded_substeps=np.asarray(env.substep_penetrations).tolist(),
                    pending_control=ledger.value['trajectories'][attempt]['pending_control'])
                if exception_state['pending_control'] is not None:ledger.value['recorded_partial_substeps_lower_bound']+=len(env.substep_penetrations)
            data=output/'probe';path=data/f'trajectories/{tid}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:
                np.savez_compressed(stream,states=np.asarray(states,dtype=np.float64),actions=np.asarray(actions,dtype=np.float32).reshape(-1,2),
                    contact=np.asarray(contacts,dtype=np.int16),valid_transition=np.asarray(valid,dtype=bool),
                    penetrations=np.asarray(penetrations,dtype=np.float64).reshape(-1,3),substep_penetrations=np.asarray(substeps,dtype=np.float64).reshape(-1,10,3),
                    contact_points=np.asarray(points,dtype=np.float32).reshape(-1,2),contact_offsets=np.asarray(offsets,dtype=np.int32),contact_angles=np.asarray(angles,dtype=np.float32))
            meta_path=path.with_suffix('.json');atomic_json(meta_path,dict(slot_id=slot['slot_id'],scene=scene,steps=steps,
                error=slot_error,exception_state=exception_state,initial_state_hash_verified=initial_hash==slot['initial_state_sha256']))
            entry=dict(slot_id=slot['slot_id'],trajectory_id=tid,baseline_trajectory_id=slot['baseline_trajectory_id'],role=slot['role'],
                shape=slot['letter'],letter=slot['letter'],location=slot['spatial_class'],spatial_class=slot['spatial_class'],seed=slot['seed'],
                split='stage1b_probe',controls=len(actions),file=path.relative_to(data).as_posix(),sha256=sha256(path),
                metadata_file=meta_path.relative_to(data).as_posix(),metadata_sha256=sha256(meta_path),
                gap_requested_world_px=scene.get('gap_world_px'),gap_band_requested=scene.get('gap_band'),
                initial_scene_hash=initial_hash,expected_initial_scene_hash=slot['initial_state_sha256'],initial_state_hash_verified=initial_hash==slot['initial_state_sha256'],
                initial_scene=initial.tolist(),error=slot_error,invalid_controls=sum(not v for v in valid))
            entries.append(entry);ledger.finish(attempt,'failed' if slot_error else 'completed',error=slot_error,saved_controls=len(actions))
            atomic_json(output/'collection-progress.json',dict(slots_completed=len(entries),planned_slots=8,charged_controls=ledger.value['charged_controls'],last=entry))
            print(json.dumps(dict(slot=slot['slot_id'],role=slot['role'],controls=len(actions),charged=ledger.value['charged_controls'],error=slot_error)),flush=True)
        ledger.value.update(stage1_combined_controls=3456+ledger.value['charged_controls'],historical_plus_new_native_controls=38016+ledger.value['charged_controls'],
            physics_substeps_lower_bound=ledger.value['physics_substeps']+ledger.value['recorded_partial_substeps_lower_bound'],
            charged_physics_substep_allowance=ledger.value['charged_controls']*10,collection_wall_seconds=time.monotonic()-started)
        ledger.save()
        if any(e['error'] for e in entries):failure='One or more frozen slots failed; no replacements were executed'
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
    finally:
        if 'ledger' in locals():
            data=output/'probe';data.mkdir(exist_ok=True)
            atomic_json(data/'manifest.json',dict(kind='native-rgb-supervised-stage1b-probe.v1',entries=entries,failure=failure,
                controller=CONTROLLER,protocol_sha256=PROTOCOL_SHA256,source_root=str(Path(physics_root).resolve()),state_columns=list(STATE_COLUMNS),
                frozen_slots_sha256=sha256(output/'frozen-slots.json'),planned_slots=8,
                charged_controls=ledger.value['charged_controls'],actual_controls=ledger.value['completed_controls']))
        lock.unlink(missing_ok=True)
    return dict(failure=failure,slots_saved=len(entries),charged_controls=ledger.value['charged_controls'])


def audit(output):
    import numpy as np
    output=Path(output);report=audit_probe(output)
    plan=json.loads((output/'frozen-slots.json').read_text());manifest=json.loads((output/'probe/manifest.json').read_text())
    ledger=json.loads((output/'resource-ledger.json').read_text());issues=list(report['issues']);bindings=[]
    expected={s['slot_id']:s for s in plan['slots']}
    if len(expected)!=8 or len(manifest['entries'])!=8:issues.append('missing_or_duplicate_declared_slots')
    if manifest['frozen_slots_sha256']!=sha256(output/'frozen-slots.json'):issues.append('slot_list_identity')
    if ledger['charged_controls']>512 or 3456+ledger['charged_controls']>4000:issues.append('control_cap')
    if len(ledger['trajectories'])>8:issues.append('extra_execution_slots')
    seen=set()
    for entry in manifest['entries']:
        slot=expected.get(entry['slot_id']);matches=False
        if slot is not None:
            with np.load(output/'probe'/entry['file'],allow_pickle=False) as arrays:
                actual=hashlib.sha256(arrays['states'][0].tobytes()).hexdigest()
            matches=(actual==slot['initial_state_sha256']==entry['initial_scene_hash']
                and entry['initial_state_hash_verified'] and entry['baseline_trajectory_id']==slot['baseline_trajectory_id']
                and entry['role']==slot['role'] and entry['seed']==slot['seed'])
        if not matches or entry['slot_id'] in seen:issues.append('initial_slot_binding')
        seen.add(entry['slot_id']);bindings.append(dict(slot_id=entry['slot_id'],initial_and_pairing_verified=bool(matches)))
    baseline=json.loads((output/'baseline-preservation.json').read_text())
    for row in baseline['files']:
        if sha256(row['path'])!=row['sha256']:issues.append('baseline_evidence_changed')
    report.update(passed=not issues,issues=sorted(set(issues)),frozen_slot_bindings=bindings,
        old_stage1_evidence_preserved='baseline_evidence_changed' not in issues,
        new_execution_slots=len(ledger['trajectories']),combined_execution_slots=54+len(ledger['trajectories']),
        new_charged_controls=ledger['charged_controls'],combined_stage1_charged_controls=3456+ledger['charged_controls'])
    atomic_json(output/'physics-audit.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','collect','audit'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/native-rgb-supervised-20261029/stage1b')
    parser.add_argument('--physics-root',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv);authorize(args.output)
    if args.stage=='prepare':result=prepare(args.output)
    elif args.stage=='audit':result=audit(args.output)
    else:
        if not args.execute:raise RuntimeError('Real1B probes require --execute')
        result=collect(args.output,args.physics_root)
    print(json.dumps({k:v for k,v in result.items() if k in ['failure','passed','charged_controls','slots_saved','issues','combined_planned_controls']},indent=2))
    return 1 if result.get('failure') or result.get('passed') is False else 0


if __name__=='__main__':raise SystemExit(main())
