"""One frozen original-physics phase; no retries or automatic next release."""
from __future__ import annotations
import argparse, ctypes, datetime, hashlib, json, os, pathlib, re, subprocess, sys, time, traceback

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v,exclusive=False):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p if exclusive else p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('x' if exclusive else 'w',encoding='utf8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    if not exclusive:tmp.replace(p)
def pe():
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetModuleFileNameW.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD]
    k.GetModuleFileNameW.restype=wintypes.DWORD
    b=ctypes.create_unicode_buffer(32768)
    if not k.GetModuleFileNameW(None,b,len(b)):raise ctypes.WinError(ctypes.get_last_error())
    return b.value
def in_job(handle=None):
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetCurrentProcess.restype=wintypes.HANDLE
    k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
    k.IsProcessInJob.restype=wintypes.BOOL
    result=wintypes.BOOL()
    if not k.IsProcessInJob(handle or k.GetCurrentProcess(),None,ctypes.byref(result)):raise ctypes.WinError(ctypes.get_last_error())
    return bool(result.value)
def wait_exit(pid):
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    h=k.OpenProcess(0x100000,False,pid)
    if not h:
        if ctypes.get_last_error()==87:return dict(pid=pid,exited=True,evidence='OpenProcess ERROR_INVALID_PARAMETER')
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if k.WaitForSingleObject(h,30000)!=0:raise RuntimeError('Launcher did not exit; no physics started')
        return dict(pid=pid,exited=True,evidence='WaitForSingleObject WAIT_OBJECT_0')
    finally:k.CloseHandle(h)
def environment(c):
    e=os.environ.copy();e.update(__PYVENV_LAUNCHER__=c['python_executable'],PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1',CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return e
def collector_command(c,digest):
    version=c.get('sampling_version','strict-original-requested-target-policy.v1')
    if version not in ('strict-original-requested-target-policy.v1','strict-original-requested-target-policy.v2'):raise ValueError('Unknown requested-action planner')
    script='strict_original_contact_pilot_v2.py' if version.endswith('.v2') else 'strict_original_contact_pilot.py'
    return [c['python_executable'],'-X','utf8','-B',str(pathlib.Path(c['repo'])/'scripts'/script),'collect',
        '--output',c['collection'],'--plan',c['plan'],'--phase',c['phase'],'--physics-gate',c['physics_gate'],
        '--source-root',c['source_root'],'--storage-roots-json',c['storage_roots'],'--policy',c['policy'],
        '--resource-policy',c['resource_policy'],'--letters',c['letters'],'--run-label',c['run_label'],
        '--stop-file',c['stop_file'],'--run-id',c['run_id'],'--runtime-config-sha',digest,'--require-detached']
def validate(path,digest):
    path=pathlib.Path(path).resolve(strict=True)
    if sha(path)!=digest:raise ValueError('Frozen runtime config identity mismatch')
    c=read(path);root=pathlib.Path(c['artifact_root']).resolve(strict=True)
    if root.name!='strict-original-contact-pilot-20261031' or not path.is_relative_to(root/'background'):raise ValueError('Wrong phase root')
    if c['kind']!='strict-original-cpu-phase.v1' or c['released'] is not True:raise ValueError('Not released')
    if not re.fullmatch(r'SOPC-[A-Za-z0-9_-]+',c['run_id']):raise ValueError('Wrong run identity')
    if c['training_authorized'] is not False or c['model_calls']!=0 or c['optimizer_updates']!=0 or c['auto_retry'] is not False:raise ValueError('Scope exceeds collection')
    if c['phase'] not in ('calibration-v1','calibration-v2','formal','supplement'):raise ValueError('Unexpected phase')
    if c['physics_settings']!=dict(physics_hz=100,dt=.01,iterations=10,control_hz=10,kp=100,kv=20,damping=0,pusher='KINEMATIC'):raise ValueError('Original protocol settings differ')
    if not c['bindings']:raise ValueError('Missing source/protocol bindings')
    for item in c['bindings']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen source/evidence mismatch: '+item['path'])
    command=c['collector_command']
    if command!=collector_command(c,'{CONFIG_SHA}'):raise ValueError('Exact frozen strict original collector command required')
    if not c['letters'] or set(c['letters'])-set('TLOCXB') or len(set(c['letters']))!=len(c['letters']):raise ValueError('Unexpected letters')
    if not re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}',c['run_label']):raise ValueError('Invalid run label')
    bound={pathlib.Path(i['path']).resolve():i['sha256'] for i in c['bindings']}
    required=[pathlib.Path(__file__).resolve(),pathlib.Path(c['repo'])/'scripts/strict_original_contact_pilot.py',
        pathlib.Path(c['repo'])/'tdyssey/strict_original_contact_pilot.py',pathlib.Path(c['repo'])/'tdyssey/native_rgb_original_physics.py',
        pathlib.Path(c['repo'])/'tdyssey/native_rgb_data_eligibility.py',pathlib.Path(c['repo'])/'manifests/official_dp.json',
        *[pathlib.Path(c[k]) for k in ('plan','physics_gate','policy','resource_policy','storage_roots')]]
    policy=read(c['policy'])
    if policy.get('policy_execution_version')!=c.get('sampling_version','strict-original-requested-target-policy.v1'):raise ValueError('Requested-action planner version differs')
    if c.get('sampling_version','').endswith('.v2'):
        required.extend([pathlib.Path(c['repo'])/'tdyssey/strict_original_target_policy_v2.py',pathlib.Path(c['repo'])/'scripts/strict_original_contact_pilot_v2.py'])
    if any(v.resolve() not in bound for v in required):raise ValueError('Required actual worker/collector/protocol/evidence binding absent')
    gate=read(c['physics_gate'])
    if gate.get('passed') is not True or gate.get('strict_original_restored_verified') is not True or gate.get('training_authorized') is not False:raise ValueError('Original restoration gate not released')
    if gate['phase']!=c['phase'] or gate.get('run_id')!=c['run_id'] or gate['protocol_sha256']!=c['protocol_sha256'] or gate['plan_sha256']!=sha(c['plan']) or gate['policy_sha256']!=sha(c['policy']):raise ValueError('Original gate identities differ')
    runtime=pathlib.Path(c['runtime']).resolve()
    if not runtime.is_relative_to(root/'background') or runtime!=path.parent:raise ValueError('Runtime path escapes config directory')
    collection=pathlib.Path(c['collection']).resolve()
    if collection!=root/'collection':raise ValueError('Collector output escapes independent collection root')
    run=collection/'runs'/c['run_label']
    for key,name in (('collector_progress','progress.json'),('collector_completion','completion.json'),('stop_file','STOP.json')):
        if pathlib.Path(c[key]).resolve()!=run/name:raise ValueError('Run ownership path differs '+key)
    return c,runtime
def verify_terminal(c,digest,value):
    expected=dict(run_id=c['run_id'],runtime_config_sha256=digest,phase=c['phase'],run_label=c['run_label'],
        physics_protocol_sha256=c['protocol_sha256'],plan_sha256=sha(c['plan']),physics_gate_sha256=sha(c['physics_gate']),policy_sha256=sha(c['policy']))
    for key,v in expected.items():
        if value.get(key)!=v:raise ValueError('Actual collector terminal identity differs '+key)
    if value.get('completed') is not True or value.get('failed') or value.get('graceful_stop'):raise ValueError('Actual collector terminal is failed/partial/stopped')
    if value.get('model_calls')!=0 or value.get('optimizer_updates')!=0:raise ValueError('Unexpected model/optimizer activity')
    if value.get('actual_attempts')!=c['expected_attempts']:raise ValueError('Actual attempts differ from finite frozen phase')
    return True
def storage(c):
    policy=read(c['resource_policy']);roots=read(c['storage_roots'])['roots'];files={}
    # Scan the union once, counting overlapping roots exactly once.
    dirs=[]
    for raw in roots+[c['artifact_root']]:
        p=pathlib.Path(raw).resolve()
        if any(p==v or p.is_relative_to(v) for v in dirs):continue
        dirs=[v for v in dirs if not v.is_relative_to(p)];dirs.append(p)
    for p in dirs:
        if p.is_file():files[str(p).casefold()]=p.stat().st_size;continue
        if not p.is_dir():continue
        for directory,sub,names in os.walk(p,followlinks=False):
            sub[:]=[v for v in sub if not (pathlib.Path(directory)/v).is_symlink()]
            for name in names:
                f=pathlib.Path(directory)/name
                if f.is_symlink():continue
                try:files[str(f.resolve()).casefold()]=f.stat().st_size
                except FileNotFoundError:pass
    used=sum(files.values());reserve=policy['atomic_reserve_bytes']+policy['postprocess_reserve_bytes']+policy['per_attempt_atomic_reserve_bytes']
    if used+reserve>policy['shared_cap_bytes']:raise RuntimeError('Shared 3GiB/reserves unavailable; no cap expansion')
    return dict(shared_bytes=used,reserved_bytes=reserve,cap_bytes=policy['shared_cap_bytes'],at_utc=utc())
def run_worker(args,c,runtime):
    started=utc();clock=time.perf_counter();proc=None;failure=None;exitcode=None
    startup=dict(run_id=c['run_id'],config_sha256=args.config_sha,pid=os.getpid(),started_at_utc=started,
        is_process_in_job=in_job(),actual_interpreter=pe(),model_calls=0,optimizer_updates=0)
    collector=None
    try:
        startup['launcher_exit']=wait_exit(args.launcher_pid)
        write(runtime/'worker-startup.json',startup,exclusive=True)
        if startup['is_process_in_job']:raise RuntimeError('Worker inside Windows Job; no controls may start')
        write(runtime/'storage-preflight.json',storage(c));validate(args.config,args.config_sha)
        command=[pe(),*collector_command(c,args.config_sha)[1:]]
        si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
        with (runtime/'collector-stdout.log').open('ab') as out,(runtime/'collector-stderr.log').open('ab') as err:
            proc=subprocess.Popen(command,cwd=c['repo'],env=environment(c),stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                close_fds=True,startupinfo=si,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW)
        child=dict(pid=proc.pid,run_id=c['run_id'],command=command,is_process_in_job=in_job(int(proc._handle)),started_at_utc=utc())
        write(runtime/'child-process.json',child,exclusive=True)
        if child['is_process_in_job']:
            # This is our exact just-created handle before data execution; stop our own child only.
            proc.terminate();proc.wait(timeout=10);raise RuntimeError('Collector inside Windows Job')
        while True:
            exitcode=proc.poll()
            row=dict(run_id=c['run_id'],config_sha256=args.config_sha,state='collecting' if exitcode is None else 'phase_exited',
                worker_pid=os.getpid(),collector_pid=proc.pid,collector_exit_code=exitcode,updated_at_utc=utc(),failure=None,
                model_calls=0,optimizer_updates=0,next_phase_released=False)
            progress=pathlib.Path(c['collector_progress'])
            if progress.is_file():row['progress']=read(progress)
            write(runtime/'status.json',row);write(runtime/'heartbeat.json',row)
            if exitcode is not None:break
            time.sleep(10)
        if exitcode!=0:failure='Collector exit '+str(exitcode)
    except BaseException as exc:
        failure=repr(exc);write(runtime/'worker-error.json',dict(error=failure,traceback=traceback.format_exc(),at_utc=utc()))
        if proc is not None and proc.poll() is None:
            # Ask only our frozen run to stop BEFORE its next control. Keep the
            # supervisor alive until the child closes its durable partial data.
            write(c['stop_file'],dict(run_id=c['run_id'],reason=failure,at_utc=utc()))
            while proc.poll() is None:
                write(runtime/'status.json',dict(run_id=c['run_id'],config_sha256=args.config_sha,state='stopping_owned_collector',
                    failure=failure,collector_pid=proc.pid,updated_at_utc=utc(),next_phase_released=False))
                time.sleep(2)
            exitcode=proc.returncode
    completion_path=pathlib.Path(c['collector_completion'])
    try:
        collector=read(completion_path) if completion_path.is_file() else None
        if collector is not None:verify_terminal(c,args.config_sha,collector)
    except BaseException as exc:
        failure=failure or repr(exc)
    if collector is None and failure is None:failure='Missing actual collector terminal receipt'
    terminal=dict(run_id=c['run_id'],config_sha256=args.config_sha,state='failed_waiting_review' if failure else 'waiting_phase_review',
        failure=failure,collector_exit_code=exitcode,collector_completion=collector,collector_completion_path=str(completion_path),
        collector_completion_sha256=sha(completion_path) if completion_path.is_file() else None,
        actual_phase_wall_seconds=time.perf_counter()-clock,finished_at_utc=utc(),model_calls=0,optimizer_updates=0,
        training_authorized=False,next_phase_released=False,auto_retry=False,callback_sent=False)
    write(runtime/'completion.json',terminal);write(runtime/'status.json',terminal)
    return 1 if failure else 0
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('launch','worker','check'))
    p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);p.add_argument('--launcher-pid',type=int)
    args=p.parse_args();c,runtime=validate(args.config,args.config_sha)
    if args.mode=='check':print(json.dumps(dict(passed=True,run_id=c['run_id'],processes_started=0,physics_steps=0,model_calls=0)));return 0
    if args.mode=='worker':return run_worker(args,c,runtime)
    for name in ('launch-claim.json','worker-startup.json','completion.json'):
        if (runtime/name).exists():raise RuntimeError('Existing phase identity; no automatic retry '+name)
    write(runtime/'launch-claim.json',dict(run_id=c['run_id'],pid=os.getpid(),at_utc=utc()),exclusive=True)
    command=[pe(),'-X','utf8','-B',str(pathlib.Path(__file__).resolve()),'worker','--config',args.config,'--config-sha',args.config_sha,'--launcher-pid',str(os.getpid())]
    si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
    with (runtime/'worker-stdout.log').open('ab') as out,(runtime/'worker-stderr.log').open('ab') as err:
        child=subprocess.Popen(command,cwd=c['repo'],env=environment(c),stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            close_fds=True,startupinfo=si,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB)
    write(runtime/'launcher-receipt.json',dict(run_id=c['run_id'],config_sha256=args.config_sha,launcher_pid=os.getpid(),worker_pid=child.pid,
        launcher_is_process_in_job=in_job(),command=command,at_utc=utc(),hidden=True),exclusive=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
