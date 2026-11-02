"""One frozen CONTACT-PILOT2 pass; no replacement or model work."""
from __future__ import annotations
import argparse
import gzip
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import numpy as np
from tdyssey.native_rgb_r3a_collection import CompactAttempt,verify_saved_attempt,sha,phase_for,PHASES
from tdyssey.native_rgb_training import atomic_json

STAGE=ROOT/'artifacts/native-rgb-supervised-20261029/contact-pilot2-20261030'
STATE=STAGE.parent/'stage-state.json'
CAP=2880
SOURCE=ROOT.parent/'Tdyssey/third_party/diffusion_policy'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def utc():return datetime.now(timezone.utc).isoformat()
def save_npz(path,arrays):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb')as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    deadline=time.monotonic()+2
    while True:
        try:os.replace(temporary,path);return
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.025)


def save_telemetry(path,rows):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    data=json.dumps(dict(kind='CONTACT-PILOT2-decision-risk-telemetry.v1',decisions=rows),sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    with temporary.open('xb')as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6)as stream:stream.write(data)
        raw.flush();os.fsync(raw.fileno())
    deadline=time.monotonic()+2
    while True:
        try:os.replace(temporary,path);return
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.025)


def stage_bytes():
    paths=[p for p in STAGE.rglob('*')if p.is_file()]
    paths += [ROOT/f for f in('scripts/native_rgb_contact_pilot2_run.py','scripts/native_rgb_contact_pilot2_plan.py',
        'scripts/native_rgb_contact_pilot2_report.py','tdyssey/native_rgb_contact_pilot2_controller.py',
        'tests/test_native_rgb_contact_pilot2_controller.py')if(ROOT/f).exists()]
    return sum(p.stat().st_size for p in paths)


def preserve_check():
    for item in read(STAGE/'preserved-inputs.json')['files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Historical input changed: '+item['path'])


def preflight():
    state=read(STATE)
    if state.get('current_stage')!='CONTACT-PILOT2'or not state.get('contact_pilot2_execution_authorized')or state.get('training_authorized'):
        raise PermissionError('Only CONTACT-PILOT2 collection release is active')
    execution=read(STAGE/'execution.json');plan=read(STAGE/'plan/cases.json')
    for item in execution['bindings']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen prephysics input changed: '+item['path'])
    preserve_check()
    if len(plan['cases'])!=60 or plan['maximum_charged_controls_including_failures']!=CAP or plan['automatic_replacements']or not plan.get('frozen')or not plan.get('physical_execution_released'):
        raise ValueError('Frozen60case/no-replacement2880control scope changed')
    for case in plan['cases']:
        for key in('cp1_baseline','r3a_baseline'):
            item=case.get(key)
            if item is not None and(sha(item['file'])!=item['sha256']or sha(item['metadata_file'])!=item['metadata_sha256']):
                raise ValueError('Historical comparison baseline changed: '+case['case_id']+' '+key)
    if stage_bytes()+execution['remaining_report_reserve_bytes']>16*1024**2:
        raise RuntimeError('Stage lacks its fixed report reserve; retain evidence and report exact bytes')
    return execution,plan


def prepare():
    if(STAGE/'execution.json').exists()or(STAGE/'resource-ledger.json').exists():
        raise FileExistsError('Pilot already prepared; never overwrite its frozen execution/ledger')
    preserve_check()
    plan=read(STAGE/'plan/cases.json')
    from tdyssey.native_rgb_contact_pilot2_controller import ContactPilot2Policy
    del ContactPilot2Policy
    bindings=[STAGE/'plan/cases.json',STAGE/'controller-rules.json',STAGE/'safety-rules.json',STAGE/'regression-release.json',
        STAGE/'PARENT_RELEASE.md',STAGE/'preserved-inputs.json',Path(__file__),
        ROOT/'tdyssey/native_rgb_contact_pilot2_controller.py',ROOT/'tdyssey/native_rgb_contact_pilot2_safety.py']
    if not all(p.is_file()for p in bindings):raise FileNotFoundError('Frozen controller-rules/cases/source missing')
    originals=read(STAGE.parent/'r3a/manifest.json')['entries']
    maximum_original=max((STAGE.parent/'r3a'/row['file']).stat().st_size for row in originals)
    projection=stage_bytes()+maximum_original*60+1536*1024+3*1024**2
    if projection>16*1024**2:raise RuntimeError('Projected necessary pilot bytes exceed16MiB: '+str(projection))
    # Global scan once; the complete16MiB stage is reserved before physics.
    roots=[Path(p)for p in read(STAGE.parent/'contact-audit-20261030/baseline-storage.json')['roots']]
    files={}
    for path in roots:
        for p in([path]if path.is_file()else path.rglob('*')if path.is_dir()else[]):
            if p.is_file():files[str(p.resolve()).lower()]=p
    for folder,pattern in(('tdyssey','native_rgb*.py'),('scripts','native_rgb*.py'),('tests','test_native_rgb*.py')):
        for p in(ROOT/folder).glob(pattern):files[str(p.resolve()).lower()]=p
    shared=sum(p.stat().st_size for p in files.values())
    if shared+16*1024**2+256*1024**2>3*1024**3:raise RuntimeError('Shared cap/atomic reserve insufficient')
    execution=dict(kind='CONTACT-PILOT2-frozen-execution.v1',prepared_at_utc=utc(),case_count=60,maximum_controls_per_case=48,
        maximum_charged_controls=CAP,maximum_attempts_per_case=1,automatic_replacements=False,
        training_authorized=False,model_calls=0,remaining_report_reserve_bytes=3*1024**2,
        stop_entire_queue_on_any_guard_unknown_or_engineering_failure=True,
        telemetry_storage='Lossless gzip snapshots of all decision/risk fields at every actual control; no duplicated plain JSON histories',
        projected_stage_bytes=projection,shared_bytes_at_prepare=shared,
        bindings=[dict(path=str(p.resolve()),sha256=sha(p))for p in bindings],
        physical_source=str(SOURCE.resolve()),physical_protocol_sha256=plan['physical_protocol_sha256'])
    prior=read(STAGE.parent/'contact-pilot1-20261030/resource-ledger.json')
    ledger=dict(kind='CONTACT-PILOT2-actual-physics-ledger.v1',closed=False,charged_controls=0,completed_controls=0,
        valid_controls=0,known_substeps=0,engine_step_calls=0,unknown_advance_substeps=0,reserved_not_started_controls=0,
        attempted_cases=0,successful_cases=0,failed_cases=0,pending_control=None,model_calls=0,checkpoint_loads=0,
        optimizer_updates=0,failures=[],max_charged_controls=CAP,
        prior_all_native_charged_controls=prior['all_native_plus_charged_controls'],
        all_native_plus_charged_controls=prior['all_native_plus_charged_controls'])
    manifest=dict(kind='CONTACT-PILOT2-attempt-manifest.v1',closed=False,collection_closed=False,
        plan_sha256=sha(STAGE/'plan/cases.json'),entries=[],not_executed_case_ids=[c['case_id']for c in plan['cases']],
        all_attempts_preserved=True,added_to_R3A_or_selected_training_pool=False)
    for name,data in(('execution.json',execution),('resource-ledger.json',ledger),('manifest.json',manifest)):
        atomic_json(STAGE/name,data)
    print(json.dumps(dict(prepared=True,projected_stage_bytes=projection,shared_bytes=shared,physics_controls=0)))


def run():
    execution,plan=preflight();ledger=read(STAGE/'resource-ledger.json');manifest=read(STAGE/'manifest.json')
    if ledger['closed']or ledger['pending_control']is not None or manifest['entries']:
        raise RuntimeError('This one-pass runner cannot resume or repeat attempted physics')
    claim=STAGE/'execution-once-claim.json'
    with claim.open('x',encoding='utf-8')as stream:json.dump(dict(pid=os.getpid(),at_utc=utc(),plan_sha256=manifest['plan_sha256']),stream)
    lock=STAGE/'collector-owner.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps(dict(pid=os.getpid(),at_utc=utc())).encode());os.fsync(fd)
    from tdyssey.native_rgb_selected_physics import make_selected_env,verify_selected_identity
    from tdyssey.native_rgb_contact_pilot2_controller import ContactPilot2Policy
    began=time.perf_counter();halt=None
    (STAGE/'trajectories').mkdir(exist_ok=True)
    def persist():
        ledger['all_native_plus_charged_controls']=ledger['prior_all_native_charged_controls']+ledger['charged_controls']
        atomic_json(STAGE/'resource-ledger.json',ledger);atomic_json(STAGE/'manifest.json',manifest)
    try:
        for case in plan['cases']:
            cid=case['case_id'];candidate=case['candidate'];slot=case['slot'];env=None;evidence=None
            data=STAGE/'trajectories'/f'{cid}.npz';meta_path=data.with_suffix('.json');telemetry_path=data.with_suffix('.telemetry.json.gz');failure=None
            record=dict(case_id=cid,trajectory_id=cid,group=case['group'],letter=case['letter'],split='train',
                family_id=case['family_id'],independence=case['independence'],scenario=slot['scenario'],status='running',
                charged_controls=0,completed_controls=0,known_substeps=0,unknown_advance_substeps=0,
                reserved_not_started_controls=0,partial_substeps=0,seed=candidate['seed'],initial_space=candidate['initial_space'])
            metadata=dict(kind='CONTACT-PILOT2-actual-attempt.v1',**record,started_at_utc=utc(),
                role=case['role'],task_recontact_start_control=case['task_recontact_start_control'],
                controller_version='CONTACT-PILOT2',plan_sha256=manifest['plan_sha256'],
                source_cp1_case_id=case.get('source_cp1_case_id'),initial_requested_arc=case['initial_requested_arc'],
                comparison_versions=case['comparison_versions'],
                baseline=case['baseline'],RGB_persisted=False,midtrajectory_resets=0,lookahead_physics_steps=0,
                decision_telemetry=[],model_calls=0,checkpoint_loads=0)
            manifest['entries'].append(record);manifest['not_executed_case_ids'].remove(cid);ledger['attempted_cases']+=1;persist()
            try:
                env=make_selected_env(execution['physical_source'],case['letter'])
                env.seed(candidate['seed']);rgb=env.reset_scene(**candidate['initialization'])
                initial=env.state().copy();identity=env.initial_identity();verify_selected_identity(identity)
                expected=np.asarray(candidate['initial_state15'],np.float64)
                if not np.allclose(initial,expected,rtol=0,atol=1e-8)or np.any(initial[5:]!=0):
                    raise RuntimeError('Initial full state differs from frozen legal zero-dynamics scene')
                if np.max(env.penetrations())>.1:raise RuntimeError('Initial physical overlap beyond frozen legality')
                original_physics=read(STAGE.parent/f'r3a/physics-identities/{case["letter"]}.json')
                if identity['physics']!=original_physics['physics']:raise RuntimeError('Original physical properties changed')
                ref=STAGE/'physics-identities'/f'{case["letter"]}.json'
                if not ref.exists():atomic_json(ref,identity)
                metadata.update(physics_sha256=identity['physics_sha256'],initial_state_sha256=__import__('hashlib').sha256(initial.tobytes()).hexdigest(),
                    initial_state_max_abs_error=float(np.max(np.abs(initial-expected))),initial_full_state_bitwise_equal=bool(np.array_equal(initial,expected)))
                evidence=CompactAttempt(case['letter'],initial,rgb)
                if case['baseline']is not None:
                    with np.load(case['baseline']['file'],allow_pickle=False)as z:
                        if not np.array_equal(initial,z['states'][0]):raise RuntimeError('Matched A initial full state is not bitwise identical')
                        if not np.array_equal(evidence.rgb_hashes[0],z['rgb_frame_sha256'][0]):raise RuntimeError('Matched A initial original RGB hash differs')
                    metadata.update(matched_initial_full_state_verified=True,matched_initial_RGB_verified=True)
                policy=ContactPilot2Policy(env,slot,candidate)
                for control in range(48):
                    if ledger['pending_control']is not None or ledger['charged_controls']>=CAP:raise RuntimeError('Pending or exhausted charged control ledger')
                    before=env.state().copy();action,decision=policy.next_action(control);action=np.asarray(action,np.float64)
                    if action.shape!=(2,)or not np.isfinite(action).all()or(action<0).any()or(action>511).any():
                        raise ValueError('New controller issued an invalid absolute action')
                    if int(decision['phase'])!=PHASES.index(phase_for(control,slot['scenario'])):
                        raise ValueError('Saved planned phase must preserve original task schedule')
                    ledger['charged_controls']+=1;record['charged_controls']+=1
                    ledger['pending_control']=dict(case_id=cid,control_index=control,action=action.tolist());persist()
                    info={};full=False;rgb=None;error=None;started=time.perf_counter();previous_trace=env.last_substeps
                    try:
                        rgb,info=env.step(action);full=True
                        if info.get('physics_substeps')!=10 or env.control_step!=control+1:raise RuntimeError('Original10substep timing changed')
                        if not info.get('valid'):raise RuntimeError('Existing physical quality guard rejected full control')
                        info['substeps']=env.last_substeps
                        policy.observe(before,action,env.state().copy(),info)
                    except BaseException as exc:error=exc;info=getattr(exc,'info',info)
                    finally:
                        substeps=env.last_substeps if env.last_substeps is not previous_trace else []
                        known=sum(bool(s.get('executed'))and not s.get('engine_exception')for s in substeps)
                        unknown=sum(bool(s.get('engine_exception'))for s in substeps)
                        record['known_substeps']+=known;record['unknown_advance_substeps']+=unknown
                        ledger['known_substeps']+=known;ledger['unknown_advance_substeps']+=unknown
                        ledger['engine_step_calls']+=sum(bool(s.get('executed'))for s in substeps)
                        if known+unknown==0:record['reserved_not_started_controls']+=1;ledger['reserved_not_started_controls']+=1
                        evidence.append(action,decision,substeps,completed=full,after=env.state().copy()if full else None,
                            rgb=rgb,wall_seconds=time.perf_counter()-started)
                        if full:
                            evidence.valid[-1]=bool(info.get('valid',False));record['completed_controls']+=1
                            ledger['completed_controls']+=1;ledger['valid_controls']+=int(info.get('valid',False))
                        old_keys={'phase','mode','arc_id','point_local','normal_local','normal_world','beta','signed_lever','raw_target','geometry_verified','state_before_sha256'}
                        metadata['decision_telemetry'].append(dict(control=control,**{k:v for k,v in decision.items()if k not in old_keys}))
                        save_telemetry(telemetry_path,metadata['decision_telemetry'])
                        ledger['pending_control']=None;save_npz(data,evidence.arrays());persist()
                    if error is not None:raise error
                metadata['integrity']=verify_saved_attempt(data,complete=True)
                record['status']='completed';ledger['successful_cases']+=1
            except BaseException as exc:
                failure=exc
                physical='penetration exceeds 2 world pixels'in str(exc)
                record['status']='failed_physics'if physical and not record['unknown_advance_substeps']else'failed_system'
                record['error']=str(exc);ledger['failed_cases']+=1
                metadata['failure']=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
                ledger['failures'].append(dict(case_id=cid,status=record['status'],reason=str(exc)))
                if evidence is not None:save_npz(data,evidence.arrays())
            finally:
                if env is not None:
                    try:env.close()
                    except BaseException as exc:
                        if failure is None:
                            failure=exc;record.update(status='failed_system',error='Close failure: '+str(exc))
                            ledger['successful_cases']-=1;ledger['failed_cases']+=1
                            ledger['failures'].append(dict(case_id=cid,status='failed_system',reason=str(exc),phase='close'))
                if evidence is not None:
                    arrays=evidence.arrays();record['partial_substeps']=len(arrays['substep_status'])-record['completed_controls']*10
                    metadata.update(rgb_sequence_sha256=evidence.rgb_digest.hexdigest(),observed_internal_RGB_returns=evidence.internal_rgb_returns,
                        inclusive_step_wall_seconds=sum(evidence.walltimes))
                if data.exists():record.update(file=data.relative_to(STAGE).as_posix(),sha256=sha(data))
                metadata.update(record,finished_at_utc=utc())
                if telemetry_path.exists():
                    metadata.update(decision_telemetry_file=telemetry_path.relative_to(STAGE).as_posix(),decision_telemetry_sha256=sha(telemetry_path),decision_telemetry_count=len(metadata['decision_telemetry']))
                metadata.pop('decision_telemetry',None)
                atomic_json(meta_path,metadata)
                record.update(metadata_file=meta_path.relative_to(STAGE).as_posix(),metadata_sha256=sha(meta_path));persist()
            print(json.dumps(dict(case_id=cid,status=record['status'],attempted=ledger['attempted_cases'],successful=ledger['successful_cases'],
                failed=ledger['failed_cases'],charged_controls=ledger['charged_controls'],known_substeps=ledger['known_substeps'])),flush=True)
            if stage_bytes()+execution['remaining_report_reserve_bytes']>16*1024**2:
                raise RuntimeError('Actual stage bytes consume report reserve; no automatic further physics')
            if failure is not None:raise RuntimeError('Any physical guard, unknown advance or engineering failure stops the entire frozen CP2 queue; no automatic retry')from failure
    except BaseException as exc:
        halt=dict(type=type(exc).__name__,reason=str(exc),at_utc=utc(),traceback=traceback.format_exc())
    finally:
        ledger.update(closed=True,wall_seconds=time.perf_counter()-began)
        manifest.update(closed=True,collection_closed=True,status='completed_pass'if halt is None else'stopped_with_evidence',halt=halt)
        persist();preserve_check()
        atomic_json(STAGE/'completion.json',dict(completed_pass=halt is None,halt=halt,finished_at_utc=utc(),
            attempted_cases=ledger['attempted_cases'],successful_cases=ledger['successful_cases'],failed_cases=ledger['failed_cases'],
            charged_controls=ledger['charged_controls'],known_substeps=ledger['known_substeps'],unknown_advance_substeps=ledger['unknown_advance_substeps'],
            model_calls=0,optimizer_updates=0,stage_bytes=stage_bytes()))
        os.close(fd);lock.unlink()
    return 0 if halt is None else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.prepare:prepare();return 0
    if a.check:preflight();print(json.dumps(dict(preflight=True,new_physics=0)));return 0
    if a.execute:return run()
    p.error('Choose --prepare, --check or --execute')


if __name__=='__main__':raise SystemExit(main())
