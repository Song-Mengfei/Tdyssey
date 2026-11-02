"""R3A v2: bounded Windows atomic-I/O retry and explicit audited first-six repair."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy');os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
from tdyssey.native_rgb_r3a_collection import (LETTERS,RULES,CONTACT_COLUMNS,CONTACT_KINDS,PHASES,MODES,
    build_plan,ContactPolicy,CompactAttempt,verify_saved_attempt,identity,sha)
from tdyssey.native_rgb_training import atomic_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def utc():return datetime.now(timezone.utc).isoformat()
def elapsed():return time.perf_counter()


def replace_with_retry(temporary,target,*,clock=time.monotonic,sleep=time.sleep,replace=os.replace):
    """Retry one serialized write, never a physics control or trajectory."""
    deadline=clock()+2.
    while True:
        try:return replace(temporary,target)
        except PermissionError:
            remaining=deadline-clock()
            if remaining<=0:raise
            sleep(min(.025,remaining))


def save_npz(path,arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    # A permanent error retains the serialized temporary file as evidence.
    with temp.open('xb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    replace_with_retry(temp,path)


def recovery_for(config,root):
    if config.get('repair_first_six_authorized') is not True:raise PermissionError('Explicit audited first-six repair required')
    path=Path(config['recovery_receipt'])
    if sha(path)!=config['recovery_receipt_sha256']:raise ValueError('Recovery receipt SHA differs')
    value=read(path)
    if value.get('authorized') is not True or value.get('mode')!='repair-first-six' or value.get('plan_sha256')!=sha(root/'plan.json'):
        raise PermissionError('Recovery scope/unchanged plan differs')
    return value


def selected_candidates(slot,row,mode,recovery):
    if row['status']=='completed':
        if mode!='repair-first-six':raise RuntimeError('Completed slot cannot be recollected')
        successful=[a for a in row['attempts'] if a['status']=='completed']
        if len(successful)!=1 or row.get('successful_trajectory_id')!=successful[0]['trajectory_id']:
            raise ValueError('Previously completed success is not uniquely bound')
        return []
    start=len(row['attempts'])
    if start:
        if mode!='repair-first-six' or recovery.get('failed_slot_next_attempts',{}).get(slot['slot_id'])!=start:
            raise PermissionError('Failed system slot has no explicit next-candidate recovery authorization')
        if row['attempts'][-1]['status'] not in ('failed_system','failed_physics'):
            raise ValueError('Failed prior candidate is unresolved')
    elif row['status']!='not_executed':raise RuntimeError('Unexecuted slot state differs')
    if start>2:raise RuntimeError('No more than two frozen replacements')
    return slot['attempts'][start:]


def config_contract(config):
    if config.get('execution_authorized') is not True or config.get('training_authorized',False) is not False:raise PermissionError('R3A collection only, no training')
    if (config.get('train_per_letter'),config.get('val_per_letter'),config.get('controls_per_scene'),config.get('max_replacements'))!=(128,24,48,2):raise ValueError('Fixed6letters×(128TRAIN+24val)×48controls and2replacements required')
    if config.get('protocol_sha256')!=RULES['protocol_sha256']:raise ValueError('Only selected .1s existing protocol')
    if Path(config['repo_root']).resolve()!=ROOT or Path(config['output']).resolve()!=ROOT/'artifacts/native-rgb-supervised-20261029/r3a':raise ValueError('UniqueR3A output required')
    return True


def storage_check(config):
    from tdyssey.native_rgb_stage_budget import storage_footprint
    root=Path(config['output']);paths=[Path(p) for p in config['storage_roots']]
    shared=storage_footprint(paths);stage=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    source=sum(Path(p).stat().st_size for p in config.get('new_stage_source_files',[]) if Path(p).exists() and not Path(p).resolve().is_relative_to(root.resolve()))
    stage+=source;projection=shared['bytes']+256*1024**2+144*1024**2
    report=dict(stage_bytes=stage,stage_limit_bytes=96*1024**2,shared_bytes=shared['bytes'],future_R3B_reserved_bytes=144*1024**2,atomic_reserve_bytes=256*1024**2,
        projected_bytes=projection,limit_bytes=3*1024**3,passed=stage<=96*1024**2 and projection<=3*1024**3)
    if not report['passed']:raise RuntimeError('Exact storage shortage; retain failed data and912 target: '+json.dumps(report))
    return report


def prepare(config_path):
    config=read(config_path);config_contract(config);root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists() or (root/'resource-ledger.json').exists():raise FileExistsError('Plan/resource ledger already prepared; never replace frozen scene catalogue')
    plan=build_plan(config,progress_callback=lambda values:print(json.dumps(dict(preparing_static_plan=True,**values)),flush=True));atomic_json(root/'plan.json',plan)
    index=dict(kind='R3A-all-attempt-scene-index.v1',status='prepared',collection_closed=False,gate_passed=False,slots=[dict(slot_id=s['slot_id'],family_id=s['family_id'],letter=s['letter'],split=s['split'],status='not_executed',attempts=[]) for s in plan['slots']])
    ledger=dict(kind='R3A-actual-physics-resource-ledger.v1',closed=False,charged_controls=0,completed_controls=0,valid_controls=0,
        attempted_scenes=0,successful_scenes=0,failed_scenes=0,engine_step_calls=0,executed_substeps=0,engine_unknown_advance_substeps=0,
        pending_control=None,prior_supervised_controls=config.get('prior_supervised_controls',41976),prior_all_native_controls=config.get('prior_all_native_controls',76536),
        planned_successful_scenes=912,planned_controls_no_failure=43776,max_physical_attempts=2736,max_charged_controls=131328,failures=[])
    manifest=dict(kind='R3A-compact-state-reference.v1',source_root='.',protocol_sha256=RULES['protocol_sha256'],control_dt=.1,physics_dt=.01,
        entries=[],collection_closed=False,plan_file='plan.json',plan_sha256=sha(root/'plan.json'),state_columns=list(__import__('tdyssey.native_rgb_env',fromlist=['STATE_COLUMNS']).STATE_COLUMNS),
        contact_columns=list(CONTACT_COLUMNS),contact_kinds=list(CONTACT_KINDS),phases=list(PHASES),modes=list(MODES),RGB_persisted=False,
        internal_original_RGB_rendering_retained=True)
    for name,value in [('index.json',index),('resource-ledger.json',ledger),('manifest.json',manifest)]:atomic_json(root/name,value)
    print(json.dumps(dict(prepared=True,slots=len(plan['slots']),candidate_scenes=sum(len(s['attempts']) for s in plan['slots']),static_rejections=len(plan['static_rejections']),plan_sha256=sha(root/'plan.json'),new_physics=0)),flush=True)
    return 0


def preflight(config_path):
    config=read(config_path);config_contract(config);root=Path(config['output'])
    state=read(config['stage_state'])
    if state.get('current_stage')!='R3A' or not (state.get('r3a_execution_authorized') or state.get('r3a_collection_authorized')):raise PermissionError('Current stage does not authorize actual R3A collection')
    for filename,expected in config.get('input_hashes',{}).items():
        if sha(filename)!=expected:raise ValueError('Frozen current R3A input changed: '+filename)
    for row in read(config['source_identity'])['files']:
        if sha(row['source'])!=row['sha256'] or sha(root/row['snapshot'])!=row['sha256']:raise ValueError('Frozen source/snapshot changed: '+row['source'])
    plan=read(root/'plan.json')
    if len(plan['slots'])!=912 or plan['rules_sha256']!=identity(RULES) or plan['protocol_sha256']!=RULES['protocol_sha256']:raise ValueError('Frozen independent scene/rule catalogue differs')
    if config.get('plan_sha256') and sha(root/'plan.json')!=config['plan_sha256']:raise ValueError('Frozen plan SHA differs')
    return config,plan,storage_check(config)


def failure_is_physical(exc):
    return 'penetration exceeds 2 world pixels' in str(exc) or isinstance(exc,InitialSceneOverlap)


class InitialSceneOverlap(RuntimeError):pass


def run(config_path,mode,*,execute=False):
    if execute is not True:raise PermissionError('Explicit --execute required')
    import numpy as np
    from tdyssey.native_rgb_selected_physics import make_selected_env,verify_selected_identity
    config,plan,initial_storage=preflight(config_path);root=Path(config['output']);ledger=read(root/'resource-ledger.json');index=read(root/'index.json');manifest=read(root/'manifest.json')
    if ledger['closed'] or ledger['pending_control'] is not None or index['collection_closed']:raise RuntimeError('Closed/interrupted physical run may not be automatically replayed')
    recovery=recovery_for(config,root) if mode=='repair-first-six' else {}
    if mode=='first-six' and (ledger['attempted_scenes'] or index['status']!='prepared'):raise RuntimeError('First6 already attempted')
    if mode=='remaining' and not (index['gate_passed'] and index['status']=='first_six_complete'):raise RuntimeError('First6 actual collection gate must complete before remainder')
    claim=root/(mode+'-once-claim.json')
    with claim.open('x',encoding='utf-8') as f:json.dump(dict(mode=mode,started_at_utc=utc(),pid=os.getpid()),f)
    requests=plan['slots'][:6] if mode in ('first-six','repair-first-six') else plan['slots'][6:];registry={r['slot_id']:r for r in index['slots']}
    current=None;halt=None;began=elapsed()
    def persist():
        ledger['supervised_plus_charged_controls']=ledger['prior_supervised_controls']+ledger['charged_controls']
        ledger['all_native_plus_charged_controls']=ledger['prior_all_native_controls']+ledger['charged_controls']
        atomic_json(root/'resource-ledger.json',ledger)
    try:
        for slot in requests:
            row=registry[slot['slot_id']];current=row
            candidates=selected_candidates(slot,row,mode,recovery)
            if not candidates:
                # Preserve successful trajectory/frame hashes and its split;
                # do not initialize an environment or charge a control again.
                successful=next(e for e in manifest['entries'] if e['trajectory_id']==row['successful_trajectory_id'])
                if successful['slot_id']!=slot['slot_id'] or sha(root/successful['file'])!=successful['sha256']:
                    raise ValueError('Preserved completed scene identity differs')
                verify_saved_attempt(root/successful['file'],complete=True)
                continue
            for candidate in candidates:
                storage_check(config);attempt=candidate['attempt_index'];tid=slot['slot_id']+f'-a{attempt}'
                if attempt!=len(row['attempts']) or attempt>2:raise RuntimeError('Only two frozen replacement candidates after failed primary')
                row['status']='running';record=dict(trajectory_id=tid,attempt_index=attempt,seed=candidate['seed'],status='initializing',charged_controls=0,completed_controls=0)
                row['attempts'].append(record);ledger['attempted_scenes']+=1;atomic_json(root/'index.json',index);persist()
                env=None;evidence=None;failure=None;init_wall=elapsed();data=root/'trajectories'/f'{tid}.npz';meta_path=data.with_suffix('.json')
                metadata=dict(kind='R3A-compact-actual-attempt.v1',trajectory_id=tid,slot_id=slot['slot_id'],family_id=slot['family_id'],letter=slot['letter'],split=slot['split'],
                    seed=candidate['seed'],attempt_index=attempt,scenario=slot['scenario'],initial_yaw_bin=candidate['initial_yaw_bin'],initial_space=candidate['initial_space'],
                    protocol_sha256=RULES['protocol_sha256'],state_is_privileged_collection_renderer_audit_metadata=True,midtrajectory_resets=0,lookahead_physics_steps=0,
                    RGB_persisted=False,internal_original_rendering_retained=True,internal_render_seconds_isolated=False)
                try:
                    env=make_selected_env(config['physics_source'],slot['letter']);env.seed(candidate['seed']);rgb=env.reset_scene(**candidate['initialization'])
                    initial=env.state().copy();actual_identity=env.initial_identity();verify_selected_identity(actual_identity)
                    if not np.isfinite(initial).all() or not np.allclose(initial,candidate['initial_state15'],rtol=0,atol=1e-8) or np.any(initial[5:]!=0):raise RuntimeError('Actual sole initial state differs from frozen independent scene')
                    if float(np.max(env.penetrations()))>.1:raise InitialSceneOverlap('Frozen initial scene has unexpected physical overlap')
                    ref=root/'physics-identities'/f"{slot['letter']}.json"
                    if ref.exists():
                        if read(ref)['physics']!=actual_identity['physics']:raise RuntimeError('Selected nonstate physical parameters changed')
                    else:atomic_json(ref,actual_identity)
                    metadata.update(initial_identity_reference=ref.relative_to(root).as_posix(),physics_sha256=actual_identity['physics_sha256'],
                        reset_wall_seconds=elapsed()-init_wall,initial_state_sha256=__import__('hashlib').sha256(initial.tobytes()).hexdigest())
                    evidence=CompactAttempt(slot['letter'],initial,rgb);rgb=None;policy=ContactPolicy(env,slot,candidate)
                    for control in range(48):
                        if ledger['pending_control'] is not None or ledger['charged_controls']>=131328:raise RuntimeError('Unresolved or over-scope physical control')
                        before=env.state().copy();action,decision=policy.next_action(control)
                        ledger['charged_controls']+=1;record['charged_controls']+=1
                        ledger['pending_control']=dict(trajectory_id=tid,control_index=control,action=list(map(float,action)));persist()
                        info={};full=False;rgb=None;started=elapsed();error=None
                        try:
                            rgb,info=env.step(action);full=True
                            if info.get('physics_substeps')!=10 or env.control_step!=control+1:raise RuntimeError('Actual10substep/control timing differs')
                            if not info.get('valid'):raise InitialSceneOverlap('Full physical control failed the existing quality guard')
                            policy.observe(before,action,env.state().copy(),info)
                        except BaseException as exc:error=exc;info=getattr(exc,'info',info)
                        finally:
                            substeps=env.last_substeps
                            ledger['engine_step_calls']+=sum(bool(s.get('executed')) for s in substeps)
                            ledger['executed_substeps']+=sum(bool(s.get('executed')) and not s.get('engine_exception') for s in substeps)
                            ledger['engine_unknown_advance_substeps']+=sum(bool(s.get('engine_exception')) for s in substeps)
                            evidence.append(action,decision,substeps,completed=full,after=env.state().copy() if full else None,rgb=rgb,wall_seconds=elapsed()-started)
                            if full:
                                evidence.valid[-1]=bool(info.get('valid',False));record['completed_controls']+=1;ledger['completed_controls']+=1;ledger['valid_controls']+=int(info.get('valid',False))
                            ledger['pending_control']=None;save_npz(data,evidence.arrays());persist()
                        rgb=None
                        if error is not None:raise error
                    check=verify_saved_attempt(data,complete=True);record.update(status='completed',completed_controls=48,trajectory_file=data.relative_to(root).as_posix(),trajectory_sha256=sha(data))
                    metadata.update(status='completed',completed_controls=48,charged_controls=record['charged_controls'],integrity=check,
                        rgb_sequence_sha256=evidence.rgb_digest.hexdigest(),observed_internal_RGB_returns=evidence.internal_rgb_returns,
                        inclusive_step_wall_seconds=sum(evidence.walltimes),trajectory_sha256=record['trajectory_sha256'])
                except BaseException as exc:
                    failure=exc;record.update(status='failed_physics' if failure_is_physical(exc) else 'failed_system',error=str(exc));ledger['failed_scenes']+=1
                    metadata.update(status=record['status'],failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc()),
                        completed_controls=record['completed_controls'],charged_controls=record['charged_controls'])
                    if evidence is not None:
                        save_npz(data,evidence.arrays());metadata.update(trajectory_sha256=sha(data),rgb_sequence_sha256=evidence.rgb_digest.hexdigest(),observed_internal_RGB_returns=evidence.internal_rgb_returns)
                    ledger['failures'].append(dict(trajectory_id=tid,status=record['status'],reason=str(exc)))
                finally:
                    if env is not None:
                        try:env.close()
                        except BaseException as closing:
                            if failure is None:
                                failure=closing;record.update(status='failed_system',error=repr(closing));ledger['failed_scenes']+=1
                                metadata.update(status='failed_system',close_error=repr(closing),failure=dict(type=type(closing).__name__,reason=str(closing),traceback=traceback.format_exc()))
                                ledger['failures'].append(dict(trajectory_id=tid,status='failed_system',reason=str(closing),phase='close'))
                    if data.exists():record.update(trajectory_file=data.relative_to(root).as_posix(),trajectory_sha256=sha(data))
                    atomic_json(meta_path,metadata);record.update(metadata_file=meta_path.relative_to(root).as_posix(),metadata_sha256=sha(meta_path))
                    atomic_json(root/'index.json',index);persist()
                if failure is None:
                    row['status']='completed';row['successful_trajectory_id']=tid;ledger['successful_scenes']+=1
                    manifest['entries'].append(dict(trajectory_id=tid,slot_id=slot['slot_id'],family_id=slot['family_id'],letter=slot['letter'],split=slot['split'],seed=candidate['seed'],
                        file=data.relative_to(root).as_posix(),sha256=sha(data),metadata_file=meta_path.relative_to(root).as_posix(),metadata_sha256=sha(meta_path),controls=48,
                        protocol_sha256=RULES['protocol_sha256'],scenario=slot['scenario'],initial_yaw_bin=candidate['initial_yaw_bin'],initial_space=candidate['initial_space']))
                    atomic_json(root/'manifest.json',manifest);break
                if not failure_is_physical(failure):raise RuntimeError('System failure stops batch; no blind replacement') from failure
            else:
                row['status']='exhausted_replacements';raise RuntimeError('Three physical candidates failed; retain deficit and stop, not shrink912 target')
            atomic_json(root/'index.json',index);persist();storage_check(config)
            print(json.dumps(dict(slot=slot['slot_id'],successes=ledger['successful_scenes'],attempts=ledger['attempted_scenes'],charged_controls=ledger['charged_controls'],executed_substeps=ledger['executed_substeps'])),flush=True)
        if mode in ('first-six','repair-first-six'):index.update(status='first_six_complete',gate_passed=True)
        else:
            if ledger['successful_scenes']!=912:raise RuntimeError('Not all912 successful independent scenes collected')
            index.update(status='completed912',collection_closed=True);ledger['closed']=True;manifest['collection_closed']=True
    except BaseException as exc:
        halt=dict(at_utc=utc(),slot_id=None if current is None else current['slot_id'],type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
        ledger['closed']=True;index.update(status='stopped_with_evidence',collection_closed=True,halt=halt);manifest['collection_closed']=True;atomic_json(root/'halt.json',halt)
    finally:
        ledger['wall_seconds']=ledger.get('wall_seconds',0.)+elapsed()-began;persist();atomic_json(root/'index.json',index);atomic_json(root/'manifest.json',manifest)
        atomic_json(root/(mode+'-completion.json'),dict(finished=halt is None,halt=halt,successful_scenes=ledger['successful_scenes'],charged_controls=ledger['charged_controls'],storage=storage_check(config)))
    return 0 if halt is None else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--first-six',action='store_true');p.add_argument('--repair-first-six',action='store_true');p.add_argument('--remaining',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.prepare:return prepare(a.config)
    if a.check:
        _,_,report=preflight(a.config);print(json.dumps(report));return 0
    if sum((a.first_six,a.repair_first_six,a.remaining))!=1:raise ValueError('Choose exactly first-six,repair-first-six or remaining')
    mode='repair-first-six' if a.repair_first_six else 'first-six' if a.first_six else 'remaining'
    return run(a.config,mode,execute=a.execute)


if __name__=='__main__':raise SystemExit(main())
