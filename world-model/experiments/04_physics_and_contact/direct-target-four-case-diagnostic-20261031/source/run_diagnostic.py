"""One human-approved four-case original-physics diagnostic. No retry path."""
from __future__ import annotations
from copy import deepcopy
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zlib

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(ROOT))
os.environ.update(SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1',
                  CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from tdyssey.direct_target_collection import counted_original_steps,write,read,sha,utc
from tdyssey.original_contact_pilot import _safe_json,storage_snapshot
from tdyssey.native_rgb_original_physics import PROTOCOL_SHA256
from observer import FourCaseEnv,PrefixMismatch,DiagnosticHardStop,BudgetExceeded

IDS=['DT-O-01-0-b0','DT-T-00-0-b0','DT-L-05-0-b0','DT-C-06-1-b0']
CAPS=[8,8,8,11]
MAX_BYTES=16*1024**2


def own_size():return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())


def bounded_bytes(path,data):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Output escaped this diagnostic directory')
    data=bytes(data)
    # Atomic peak counts both old and temporary file. No old evidence is deleted.
    if own_size()+len(data)>MAX_BYTES:raise RuntimeError('16MiB diagnostic output budget unavailable')
    if __import__('shutil').disk_usage(ROOT).free<256*1024**2+len(data):raise RuntimeError('Atomic reserve unavailable')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.atomic-'+str(os.getpid()))
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def bounded_json(path,value):
    bounded_bytes(path,(json.dumps(_safe_json(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())


def capacity(c):
    snap=storage_snapshot([*read(c['storage_roots'])['roots'],ROOT])
    if snap['missing_roots']:raise RuntimeError('A shared evidence root is missing')
    if snap['bytes']+256*1024**2+MAX_BYTES>3*1024**3:raise RuntimeError('Shared3GiB/256MiBreserve unavailable')
    if own_size()>MAX_BYTES:raise RuntimeError('New diagnostic directory already exceeds16MiB')
    return dict(shared=snap,new_bytes=own_size(),new_cap_bytes=MAX_BYTES,atomic_reserve_bytes=256*1024**2)


def load_old(path):
    with np.load(path,allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
    payload=zlib.decompress(d.pop('telemetry_zlib').tobytes())
    if hashlib.sha256(payload).digest()!=d.pop('telemetry_sha256').tobytes():raise ValueError('Old telemetry checksum mismatch')
    d.update(json.loads(payload));return d


def validate(path,digest):
    path=Path(path).resolve(strict=True)
    if sha(path)!=digest:raise ValueError('Runtime-config SHA mismatch')
    c=read(path)
    if c['kind']!='four-case-original-continuation-diagnostic.v1' or c['execution_authorized'] is not True:raise ValueError('Wrong release identity')
    if c['case_ids']!=IDS or c['control_caps']!=CAPS or c['max_attempts']!=4 or c['max_controls']!=35 or c['max_regular_substeps']!=350:raise ValueError('Approved finite table differs')
    if c['model_calls']!=0 or c['optimizer_updates']!=0 or c['training_eligible'] is not False or c['retries']!=0:raise ValueError('Wrong scope')
    if c['physics_protocol_sha256']!=PROTOCOL_SHA256:raise ValueError('Original physics identity mismatch')
    bound={Path(x['path']).resolve():x['sha256'] for x in c['bindings']}
    required={ROOT/p for p in ('run_diagnostic.py','observer.py','report.py','tests_offline.py','APPROVAL.json','HUMAN_AUTHORIZATION.json')}
    if not required<=set(bound):raise ValueError('Required source/approval binding absent')
    for p,expected in bound.items():
        if sha(p)!=expected:raise ValueError('Frozen source/evidence changed: '+str(p))
    if c['observer_change']!=dict(AO_OW_gt1='would_stop_only_in_four_cases',AW_gt1='hard_stop',nonfinite='hard_stop',solid_closedhole='hard_stop',object_outer_envelope=[3,508]):raise ValueError('Observer scope changed')
    return c


def pack(path,states,images,actions,passed,complete,infos,decisions,legacy_rgb):
    payload=json.dumps(_safe_json(dict(telemetry=infos,decisions=decisions)),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    durations=[sum(float(s.get('dt',0)) for s in x.get('substeps',[]) if s.get('executed')) for x in infos]
    arrays=dict(states=np.asarray(states,np.float64).reshape(-1,15),rgb=np.asarray(images,np.uint8).reshape(-1,3,96,96),
        requested_actions=np.asarray(actions,np.float64).reshape(-1,2),passed_actions=np.asarray(passed,np.float64).reshape(-1,2),
        control_valid=np.asarray(complete,bool),legacy_control_valid=np.asarray([all(not s.get('would_stop_legacy',False) for s in x.get('substeps',[])) and complete[i] for i,x in enumerate(infos)],bool),
        control_time_seconds=np.asarray([0.,*np.cumsum(durations)],np.float64),frame_indices=np.arange(len(states)),segment_ids=np.zeros(len(states),np.int64),
        legacy_terminal_rgb=np.asarray(legacy_rgb,np.uint8) if legacy_rgb is not None else np.empty((0,),np.uint8),
        telemetry_zlib=np.frombuffer(zlib.compress(payload,6),np.uint8),telemetry_sha256=np.frombuffer(hashlib.sha256(payload).digest(),np.uint8))
    if len(states)!=len(actions)+1 or len(images)!=len(states):raise ValueError('Incomplete RGB frame explicitly required, no padding')
    if not np.array_equal(arrays['requested_actions'],arrays['passed_actions']):raise ValueError('Requested action changed')
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);bounded_bytes(path,stream.getvalue())
    return dict(path=str(path),file=str(path),sha256=sha(path),bytes=path.stat().st_size)


def execute(config_path,digest):
    c=validate(config_path,digest)
    if (ROOT/'ledger.json').exists() or (ROOT/'execution-claim.json').exists():raise FileExistsError('Prior execution exists: no retry')
    first_capacity=capacity(c)
    # Durable claim precedes even the first physical environment construction.
    write(ROOT/'execution-claim.json',dict(run_id=c['run_id'],config_sha256=digest,at_utc=utc(),pid=os.getpid(),max_attempts=4),exclusive=True)
    ledger=dict(run_id=c['run_id'],config_sha256=digest,attempts=[],actual_controls=0,actual_valid_controls=0,actual_substeps=0,
        initialization_integrator_calls=0,initialization_integrator_seconds=0.,regular_integrator_calls=0,regular_integrator_seconds=0.,simulated_seconds=0.,model_calls=0,optimizer_updates=0)
    bounded_json(ROOT/'ledger.json',ledger)
    started=time.perf_counter();failure=None
    oldplan={a['attempt_id']:a for a in read(c['source_plan'])['attempts']}
    for ident,cap in zip(IDS,CAPS):
        try:
            boundary_capacity=capacity(c)
            a=oldplan[ident];old=load_old(Path(c['old_root'])/'collection/raw'/(ident+'.npz'));oldmeta=read(Path(c['old_root'])/'collection/attempts'/(ident+'.json'))
            expected=[row for info in old['telemetry'] for row in info['substeps'] if row.get('executed')]
            source_failure=oldmeta['first_invalid'];assert source_failure['control']+1+4==cap
            if not np.array_equal(old['requested_actions'],np.asarray(a['actions'][:len(old['requested_actions'])],np.float64)):raise ValueError('Old input/plan mismatch')
            cost=dict(attempt_id=ident,letter=a['letter'],status='claimed',actual_controls=0,actual_valid_controls=0,actual_substeps=0,simulated_seconds=0.,
                initialization_integrator_calls=0,initialization_integrator_seconds=0.,regular_integrator_calls=0,regular_integrator_seconds=0.,started_at_utc=utc())
            write(ROOT/'results'/(ident+'.claim.json'),cost,exclusive=True)
            ledger['attempts'].append(cost);bounded_json(ROOT/'ledger.json',ledger)
            env=None;initial=None;states=[];images=[];actions=[];passed=[];complete=[];infos=[];decisions=[]
            case_error=None;hard_stop=None;prefix_failure=False;raw=None;legacy_rgb=None;prefix={};clock=time.perf_counter()
            try:
                with counted_original_steps(cost) as counter:
                    env=FourCaseEnv(c['source_root'],a['letter'],expected_rows=expected,
                        expected_terminal_rgb=old['rgb'][-1],expected_actions=old['requested_actions'],max_substeps=cap*10)
                    image=env.reset_scene(**a['initialization']);initial=env.initial_identity()
                    states.append(env.state().copy());images.append(image.copy())
                    if initial['physics_sha256']!=oldmeta['initial_identity']['physics_sha256'] or initial['protocol_sha256']!=oldmeta['initial_identity']['protocol_sha256']:
                        raise PrefixMismatch('Original physics identity differs',dict(stage='initial_identity'))
                    if not np.allclose(states[0],old['states'][0],rtol=0,atol=1e-8) or not np.array_equal(image,old['rgb'][0]):
                        raise PrefixMismatch('Original initial state/RGB differs',dict(stage='initial_state_rgb'))
                    prefix.update(initial_identity_matched=True,completed_boundary_rgb_matched=True)
                    counter['value']='regular'
                    for index in range(cap):
                        if ledger['actual_controls']>=35 or ledger['actual_substeps']>=350:raise BudgetExceeded('Global finite cap',{})
                        requested=np.asarray(a['actions'][index],np.float64)
                        # Persist the intent separately from a returned complete/partial control.
                        bounded_json(ROOT/'pending-control.json',dict(run_id=c['run_id'],attempt_id=ident,index=index,requested_action=requested.tolist(),at_utc=utc()))
                        try:
                            image,info=env.step(requested)
                        except Exception as exc:
                            case_error=repr(exc);prefix_failure=isinstance(exc,PrefixMismatch)
                            hard_stop=_safe_json(getattr(exc,'info',{})) if isinstance(exc,DiagnosticHardStop) else None
                            info=dict(getattr(exc,'info',{}) or {},valid=False,exception=case_error)
                            info.setdefault('substeps',env.last_substeps);image=None
                        rows=info.get('substeps',[]);n=sum(bool(s.get('executed')) for s in rows)
                        duration=sum(float(s.get('dt',0)) for s in rows if s.get('executed'))
                        ok=bool(info.get('valid') and n==10 and len(rows)==10 and all(not s.get('diagnostic_hard_reasons') for s in rows))
                        for dest in (ledger,cost):
                            dest['actual_controls']+=1;dest['actual_valid_controls']+=int(ok);dest['actual_substeps']+=n;dest['simulated_seconds']+=duration
                        info.update(requested_action=requested.tolist(),passed_action=env.last_passed_action.tolist(),diagnostic_only=True,training_eligible=False,control_valid_means='complete diagnostic control, not production validity')
                        if image is None:
                            try:image=env.render_clean()
                            except Exception as exc:
                                cost['unframed_terminal_control']=dict(control=index,action=requested.tolist(),post_state=_safe_json(env.state()),telemetry=_safe_json(info),renderer_error=repr(exc))
                                raise RuntimeError('Terminal RGB unavailable; cost and terminal telemetry retained') from exc
                        actions.append(requested.copy());passed.append(env.last_passed_action.copy());complete.append(ok)
                        states.append(env.state().copy());images.append(image.copy());infos.append(_safe_json(info));decisions.append(dict(control=index,source='unchanged original frozen64-action prefix',future_state_used=False))
                        if index<len(old['control_valid']) and old['control_valid'][index]:
                            if not np.array_equal(image,old['rgb'][index+1]):
                                prefix['completed_boundary_rgb_matched']=False;prefix_failure=True
                                raise PrefixMismatch('Complete old control RGB differs',dict(control=index))
                        legacy_rgb=getattr(env,'legacy_terminal_rgb_snapshot',None)
                        raw=pack(ROOT/'results'/(ident+'.npz'),states,images,actions,passed,complete,infos,decisions,legacy_rgb)
                        bounded_json(ROOT/'ledger.json',ledger)
                        if not ok:break
                    prefix.update(matched=bool(env.prefix_rows_checked==len(expected) and env.prefix_terminal_rgb_matched and not prefix_failure),
                        rows_checked=env.prefix_rows_checked,expected_rows=len(expected),max_abs_error=env.prefix_max_abs_error,
                        terminal_rgb_matched=env.prefix_terminal_rgb_matched)
                    legacy_rgb=getattr(env,'legacy_terminal_rgb_snapshot',None)
            except Exception as exc:
                case_error=repr(exc);prefix_failure=prefix_failure or isinstance(exc,PrefixMismatch)
            finally:
                if env is not None:
                    prefix.update(rows_checked=env.prefix_rows_checked,expected_rows=len(expected),max_abs_error=env.prefix_max_abs_error,terminal_rgb_matched=env.prefix_terminal_rgb_matched)
                    prefix.setdefault('matched',bool(env.prefix_rows_checked==len(expected) and env.prefix_terminal_rgb_matched and not prefix_failure))
                    legacy_rgb=getattr(env,'legacy_terminal_rgb_snapshot',None)
                    try:env.close()
                    except Exception as exc:case_error=case_error or repr(exc)
            for k in ('initialization_integrator_calls','initialization_integrator_seconds','regular_integrator_calls','regular_integrator_seconds'):ledger[k]+=cost[k]
            if cost['regular_integrator_calls']!=cost['actual_substeps']:
                case_error=case_error or 'Actual native calls differ from observed telemetry';prefix_failure=True
            if states:raw=pack(ROOT/'results'/(ident+'.npz'),states,images,actions,passed,complete,infos,decisions,legacy_rgb)
            status='completed_cap' if len(actions)==cap and all(complete) and case_error is None else 'diagnostic_hard_stop' if hard_stop and not prefix_failure else 'engineering_failed'
            cost.update(status=status,wall_seconds=time.perf_counter()-clock,finished_at_utc=utc(),failure=case_error)
            metadata=dict(case=a,run_id=c['run_id'],config_sha256=digest,source_failure=source_failure,control_cap=cap,
                expected_native_prefix_substeps=len(expected),initial_identity=initial,prefix_match=prefix,cost=deepcopy(cost),
                status=status,hard_stop=hard_stop,raw=raw,failure=case_error,training_eligible=False,diagnostic_only=True,model_calls=0,optimizer_updates=0,
                source_raw_sha256=sha(Path(c['old_root'])/'collection/raw'/(ident+'.npz')),capacity_at_attempt_start=boundary_capacity)
            bounded_json(ROOT/'results'/(ident+'.json'),metadata);bounded_json(ROOT/'ledger.json',ledger)
            print(json.dumps(dict(attempt=ident,status=status,controls=cost['actual_controls'],substeps=cost['actual_substeps'],prefix_match=prefix.get('matched'))),flush=True)
            if status=='engineering_failed' or not prefix.get('matched'):
                raise RuntimeError('Identity/prefix/engineering failure; stop whole verification: '+str(case_error))
        except Exception as exc:
            failure=repr(exc);bounded_json(ROOT/'engineering-error.json',dict(error=failure,traceback=traceback.format_exc(),at_utc=utc()));break
    try:final_capacity=capacity(c)
    except Exception as exc:
        final_capacity=dict(error=repr(exc));failure=failure or 'Final resource snapshot failed: '+repr(exc)
    final=dict(run_id=c['run_id'],config_sha256=digest,completed=failure is None and len(ledger['attempts'])==4,failed=failure is not None,failure=failure,
        actual_attempts=len(ledger['attempts']),unstarted_cases=IDS[len(ledger['attempts']):],ledger=ledger,wall_seconds=time.perf_counter()-started,
        source_bindings_unchanged=all(sha(x['path'])==x['sha256'] for x in c['bindings']),storage=final_capacity,finished_at_utc=utc(),
        raw_preservation_complete=all((ROOT/'results'/(a['attempt_id']+'.json')).exists() for a in ledger['attempts']),
        pending_raw_attempts=[a['attempt_id'] for a in ledger['attempts'] if not (ROOT/'results'/(a['attempt_id']+'.json')).exists()],
        model_calls=0,optimizer_updates=0,training_eligible=False,next_phase_released=False,initial_capacity=first_capacity)
    bounded_json(ROOT/'ledger.json',ledger);bounded_json(ROOT/'completion.json',final)
    return final


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('check','execute'));p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True)
    a=p.parse_args();c=validate(a.config,a.config_sha)
    if a.mode=='check':print(json.dumps(dict(passed=True,run_id=c['run_id'],physical_attempts=0,model_calls=0)));return 0
    result=execute(a.config,a.config_sha);return 1 if result['failed'] else 0
if __name__=='__main__':raise SystemExit(main())
