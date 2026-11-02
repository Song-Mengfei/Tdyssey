"""One separately authorized SDK-startup diagnostic; never constructs a model.

Default invocation only describes the fixed plan. Root alone may execute the
single60-second/8-MiB native attempt after source review. There is no resume.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdyssey import gpc_adaptation_guard as shared
from tdyssey.wm_compat_guard import WindowsWorker,ResourceProvider,resource_reason,DEFAULT_LOCK,_read,_publish_json,_write_all

AUTHORIZATION='20261026-gpc-sdk-startup-diagnostic-01'
OUTPUT='gpc-sdk-startup-diagnostic-20261026'
SECONDS=60;CLEANUP_RESERVE=12;WORK_SECONDS=SECONDS-CLEANUP_RESERVE;MIB=1024**2;SLOT=65536;TOTAL_BYTES=8*MIB
PIPE_JOIN_SECONDS=3;RESOURCE_JOIN_SECONDS=4;FINAL_IO_RESERVE=1
ENV_HANDOFF='GPC_SDK_DIAG_HANDOFF';ENV_NONCE='GPC_SDK_DIAG_NONCE';ENV_PARENT='GPC_SDK_DIAG_PARENT'
ZERO_FIELDS=('model_instances','model_loads','optimizer_updates','forecasts','euler_steps','eval_batches','eval_examples','policy_calls',*shared.FORBIDDEN)
CONFIG_SHA='1c5683baa16daee11ca419772532cdda95595275cca794b977261188d99abe2c'
SCHEDULE_SHA='f7217a2d09393798529b00e72fcd971e8d7be11cab7a9c769704a9b30ddd004a'


def _json(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()
def _utc():return datetime.now(timezone.utc).isoformat()
def paths():
    return shared._path(ROOT/'artifacts'/OUTPUT),shared._path(ROOT/'artifacts'/(OUTPUT+'-claim.json'))


def plan_path():return shared._path(ROOT/'artifacts/gpc-sdk-startup-review-20261026/diagnostic-plan.json')


def _protected_inputs():
    result={}
    for name,path,expected in(('configuration','configs/gpc_supervised_adaptation.json',CONFIG_SHA),
            ('schedule','artifacts/wm-sampling-b-only-20261026/training-schedule.json',SCHEDULE_SHA)):
        if _sha(shared._path(ROOT/path))!=expected:raise PermissionError('Protected pilot '+name+' changed')
        result[name]=dict(path=path,sha256=expected)
    return result


def diagnostic_plan(reviewed_commit):
    """Pure manifest builder for root after source commit; no SDK/native query."""
    return dict(kind='gpc_sdk_startup_diagnostic_plan.v1',authorization_id=AUTHORIZATION,reviewed_commit=reviewed_commit,
        script_sha256=_sha(Path(__file__)),shared_guard_sha256=_sha(Path(shared.__file__)),total_seconds=SECONDS,
        execution_seconds=WORK_SECONDS,cleanup_reserve_seconds=CLEANUP_RESERVE,total_bytes=TOTAL_BYTES,
        cleanup_wait_reserves=dict(pipe_join_seconds=PIPE_JOIN_SECONDS,resource_reader_join_seconds=RESOURCE_JOIN_SECONDS,final_io_seconds=FINAL_IO_RESERVE),
        heartbeat_max_age_seconds=5,cpu_threads=2,sdk_workers=1,job_memory_bytes=3*1024**3,
        cuda_allocated_bytes=512*MIB,cuda_reserved_bytes=1024*MIB,
        byte_allowances=dict(fixed_prepaid_bytes=3*MIB,monitor_bytes=4*MIB,stdout_bytes=MIB),
        resource_bytes=dict(startup_ram=4*1024**3,startup_commit=6*1024**3,startup_gpu=4*1024**3,startup_disk=2*1024**3,
            runtime_ram=3*1024**3//2,runtime_commit=3*1024**3,runtime_gpu=3*1024**3//2,runtime_disk=2*1024**3,
            process_working_set=2*1024**3,process_private=3*1024**3),
        allowed_sdk_operations=['import_torch','set_num_threads_2','cuda_is_available','cuda_init','cuda_reset_peak_memory_stats','cuda_allocator_reads'],
        forbidden_operations=dict(gpc_model_loads=0,weight_loads=0,network_requests=0,forwards=0,backwards=0,optimizer_steps=0,
            training_updates=0,ledger_creations=0,physics=0,tensor_probes=0),protected_inputs=_protected_inputs())


def _load_plan(reviewed_commit):
    path=plan_path();payload=path.read_bytes()
    if len(payload)>SLOT or json.loads(payload)!=diagnostic_plan(reviewed_commit):raise PermissionError('Fixed reviewed SDK diagnostic plan differs')
    return path,hashlib.sha256(payload).hexdigest()


def _authority(authorization_id,reviewed_commit,*,work_deadline_unix=None,work_deadline_monotonic=None):
    if authorization_id!=AUTHORIZATION:raise PermissionError('Exact one-shot SDK diagnostic authority required')
    if not isinstance(reviewed_commit,str) or len(reviewed_commit)!=40 or any(c not in '0123456789abcdef'for c in reviewed_commit):
        raise PermissionError('Full reviewed clean source commit required')
    if not shared._finite(work_deadline_unix)or not shared._finite(work_deadline_monotonic):
        raise ValueError('Original SDK work deadline required for source verification')
    def remaining():
        wall_now,mono_now=time.time(),time.monotonic()
        if not shared._finite(wall_now)or not shared._finite(mono_now):raise TimeoutError('Invalid SDK source-verification clock')
        left=min(work_deadline_unix-wall_now,work_deadline_monotonic-mono_now)
        if left<=0:raise TimeoutError('Original SDK work deadline elapsed during source verification')
        return left
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,timeout=remaining()).strip()
    remaining()
    dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True,timeout=remaining()).strip()
    remaining()
    if commit!=reviewed_commit or dirty:raise PermissionError('SDK diagnostic source must be clean and reviewed')


def _exclusive(path,value):
    payload=_json(value)
    if len(payload)>SLOT:raise ValueError('SDK metadata exceeds its fixed64KiB slot')
    with Path(path).open('xb',buffering=0)as stream:_write_all(stream,payload);os.fsync(stream.fileno())


class _Telemetry(shared._Telemetry):
    """Reuse raw admitted streams; four-MiB monitor + one-MiB stdout +3MiB slots."""
    def admit(self,kind,data):
        before=self.monitor_bytes if kind=='monitor'else self.stdout_bytes;cap=4*MIB if kind=='monitor'else MIB
        if before+len(data)>cap or self.monitor_bytes+self.stdout_bytes+len(data)+3*MIB>TOTAL_BYTES:
            raise OSError('SDK diagnostic8MiB cap exceeded before write')
        if kind=='monitor':self.monitor_bytes+=len(data)
        else:self.stdout_bytes+=len(data)


def _status_reason(value,handoff):
    if not isinstance(value,dict)or value.get('actual_worker_pid')!=handoff['actual_worker_pid']or value.get('nonce')!=handoff['nonce']:
        return 'worker_status_identity_invalid'
    if value.get('training_complete')is not False:return 'training_flag_forbidden_in_sdk_diagnostic'
    if value.get('cuda_sample_state')not in('unavailable','fresh'):return 'sdk_cuda_sample_state_invalid'
    for key in ZERO_FIELDS:
        if type(value.get(key))is not int or value[key]!=0:return 'forbidden_sdk_operation:'+key
    claim=handoff['claim'];binding={key:claim[key]for key in('deadline_unix','deadline_monotonic')}
    binding.update(training_deadline_unix=claim['deadline_unix'],training_deadline_monotonic=claim['deadline_monotonic'])
    return shared.monitor_reason(value,binding,pid=handoff['actual_worker_pid'],nonce=handoff['nonce'])


def verify_worker(path,*,native_check=True):
    path=shared._path(path);value=_read(path);output,claim=paths();pid=os.getpid()
    if (path!=output/'guardian/handoff.json'or value.get('output_dir')!=str(output)or value.get('authorization_id')!=AUTHORIZATION
            or os.environ.get(ENV_HANDOFF)!=str(path)or os.environ.get(ENV_NONCE)!=value.get('nonce')
            or os.environ.get(ENV_PARENT)!=str(value.get('guard_pid'))or value.get('actual_worker_pid')!=pid
            or value.get('status_path')!=str(output/'guardian/worker-status.json')or value.get('ownership')!='native_job_before_ack'
            or value.get('job_memory_bytes')!=3*1024**3 or value.get('kill_on_job_close')is not True
            or _read(output/'guardian/worker.json')!=dict(pid=pid,parent_pid=os.getppid(),nonce=value['nonce'])
            or _read(output/'guardian/worker-ack.json')!=dict(pid=pid,nonce=value['nonce'])):
        raise PermissionError('No matching actual-PID SDK diagnostic handoff')
    original=_read(claim)
    reviewed_plan,plan_sha=_load_plan(value['reviewed_commit'])
    if (original!=value['claim']or value.get('claim_sha256')!=_sha(claim)
            or value.get('script_sha256')!=_sha(Path(__file__))or original.get('resume_allowed')is not False
            or original.get('nonce')!=value['nonce']or original.get('authorization_id')!=value['authorization_id']
            or original.get('source_commit')!=value['reviewed_commit']or original.get('plan_path')!=str(reviewed_plan)
            or original.get('plan_sha256')!=plan_sha
            or original.get('deadline_unix')!=original.get('started_unix')+SECONDS
            or original.get('deadline_monotonic')!=original.get('started_monotonic')+SECONDS
            or original.get('execution_deadline_unix')!=original.get('started_unix')+WORK_SECONDS
            or original.get('execution_deadline_monotonic')!=original.get('started_monotonic')+WORK_SECONDS
            or original.get('cleanup_reserve_seconds')!=CLEANUP_RESERVE
            or time.time()>=original['execution_deadline_unix']or time.monotonic()>=original['execution_deadline_monotonic']):
        raise PermissionError('SDK diagnostic source/claim/original deadline changed')
    if native_check:
        import ctypes
        from ctypes import wintypes
        from tdyssey import policy_memory
        kernel,_=policy_memory._libraries();inside=wintypes.BOOL();limits=policy_memory._JobExtendedLimits()
        kernel.GetCurrentProcess.argtypes=[];kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)];kernel.IsProcessInJob.restype=wintypes.BOOL
        kernel.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p];kernel.QueryInformationJobObject.restype=wintypes.BOOL
        if(not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(inside))or not inside.value
                or not kernel.QueryInformationJobObject(None,9,ctypes.byref(limits),ctypes.sizeof(limits),None)
                or limits.JobMemoryLimit!=3*1024**3 or limits.BasicLimitInformation.LimitFlags&0x2200!=0x2200):
            raise PermissionError('SDK diagnostic worker is outside retained3GiB kill-on-close Job')
    return value


def supervise(*,authorization_id,reviewed_commit,python_executable,provider=None,owner_factory=None,launcher_factory=None):
    """One fixed SDK bridge; default owners query Windows only after the claim."""
    wall,mono=time.time(),time.monotonic()
    _authority(authorization_id,reviewed_commit,work_deadline_unix=wall+WORK_SECONDS,work_deadline_monotonic=mono+WORK_SECONDS)
    output,claim_path=paths();python=shared._path(python_executable)
    if not python.is_file():raise ValueError('Explicit existing Python interpreter required')
    if output.exists()or claim_path.exists():raise FileExistsError('SDK diagnostic already attempted; no retry/new output')
    frozen_plan,frozen_plan_sha=_load_plan(reviewed_commit)
    if time.monotonic()-mono>=WORK_SECONDS:raise TimeoutError('SDK diagnostic work deadline elapsed before claim')
    nonce=uuid.uuid4().hex;claim=dict(kind='gpc_sdk_startup_claim.v1',authorization_id=authorization_id,source_commit=reviewed_commit,
        output_dir=str(output),guard_pid=os.getpid(),nonce=nonce,started_unix=wall,started_monotonic=mono,
        deadline_unix=wall+SECONDS,deadline_monotonic=mono+SECONDS,total_bytes=TOTAL_BYTES,resume_allowed=False,
        execution_deadline_unix=wall+WORK_SECONDS,execution_deadline_monotonic=mono+WORK_SECONDS,cleanup_reserve_seconds=CLEANUP_RESERVE,
        plan_path=str(frozen_plan),plan_sha256=frozen_plan_sha)
    claim_path.parent.mkdir(parents=True,exist_ok=True);_exclusive(claim_path,claim);output.mkdir(exist_ok=False)
    directory=output/'guardian';directory.mkdir();_exclusive(output/'attempt.json',dict(claim_path=str(claim_path),claim_sha256=_sha(claim_path)))
    telemetry=_Telemetry(directory);injected=dict(resources=provider is not None,owner=owner_factory is not None,launcher=launcher_factory is not None)
    provider=provider if provider is not None else ResourceProvider();owner_factory=owner_factory or WindowsWorker;launcher_factory=launcher_factory or subprocess.Popen
    owner=child=pump=handoff=None;reason=error_text=None;lock_owned=False;certain=True;job_active=None;worker_exit=launcher_exit=None
    last_status=last_resource=startup=heartbeat_error=None;minimum={};peaks={};cleanup=[];samples=0;acknowledged=None
    def fail(code,error=None):
        nonlocal reason,error_text
        if reason is None:reason=code;error_text=type(error).__name__+': '+str(error)if error else None
    def expired():return time.time()>=claim['deadline_unix']or time.monotonic()>=claim['deadline_monotonic']
    def work_expired():return time.time()>=claim['execution_deadline_unix']or time.monotonic()>=claim['execution_deadline_monotonic']
    def remaining(maximum):return max(0,min(maximum,claim['deadline_unix']-time.time(),claim['deadline_monotonic']-time.monotonic()))
    def early_cleanup_wait(maximum):
        return max(0,min(maximum,remaining(SECONDS)-PIPE_JOIN_SECONDS-RESOURCE_JOIN_SECONDS-FINAL_IO_RESERVE))
    def resources(value,*,startup=False):
        problem=resource_reason(value,startup=startup)
        return problem or('resource_below_floor:disk_free_bytes'if value['disk_free_bytes']<2*1024**3 else None)
    def observe(value):
        for key in('available_ram_bytes','commit_headroom_bytes','gpu_free_bytes','disk_free_bytes'):
            v=value.get(key)
            if type(v)is int:minimum[key]=min(minimum.get(key,v),v)
        for key in('working_set_bytes','private_bytes','process_peak_working_set_bytes'):
            v=value.get(key)
            if type(v)is int:peaks[key]=max(peaks.get(key,v),v)
        if last_status:
            for key in('cuda_allocated_bytes','cuda_reserved_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes'):
                v=last_status.get(key)
                if type(v)is int:peaks[key]=max(peaks.get(key,v),v)
    try:
        _exclusive(DEFAULT_LOCK,dict(nonce=nonce,guard_pid=os.getpid(),output_dir=str(output)));lock_owned=True
        startup=provider.startup(output);observe(startup);telemetry.event('startup',startup);problem=resources(startup,startup=True)
        if problem:fail(problem)
        elif work_expired():fail('sdk_execution_deadline_cleanup_reserve')
        else:
            handoff_path=directory/'handoff.json';status_path=directory/'worker-status.json';ack=directory/'worker-ack.json'
            command=[str(python),'-B',str(ROOT/'scripts/guarded_c_child.py'),'--handshake',str(directory/'worker.json'),'--ack',str(ack),'--nonce',nonce,'--',str(Path(__file__)),'--worker',str(handoff_path)]
            env=dict(os.environ,**{ENV_HANDOFF:str(handoff_path),ENV_NONCE:nonce,ENV_PARENT:str(os.getpid())})
            env.update({key:'2'for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')});env.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
            child=launcher_factory(command,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
            pump=shared._Pump(child.stdout,directory/'worker-output.log',telemetry);until=min(claim['execution_deadline_monotonic'],time.monotonic()+10)
            while not(directory/'worker.json').exists():
                pump.check()
                if child.poll()is not None or time.monotonic()>=until or work_expired():raise TimeoutError('No actual SDK worker handshake before deadline')
                time.sleep(.02)
            identity=_read(directory/'worker.json')
            if identity.get('nonce')!=nonce or type(identity.get('pid'))is not int or identity['pid']<=0:raise PermissionError('SDK worker handshake PID/nonce invalid')
            owner=owner_factory(identity['pid'],child.pid)
            handoff=dict(kind='gpc_sdk_startup_handoff.v1',claim=claim,claim_sha256=_sha(claim_path),authorization_id=authorization_id,
                reviewed_commit=reviewed_commit,nonce=nonce,output_dir=str(output),actual_worker_pid=identity['pid'],guard_pid=os.getpid(),launcher_pid=child.pid,
                ownership='native_job_before_ack',job_memory_bytes=3*1024**3,kill_on_job_close=True,status_path=str(status_path),
                ack_path=str(ack),script_sha256=_sha(Path(__file__)))
            last_resource=provider.runtime(identity['pid'],output);observe(last_resource);problem=resources(last_resource)
            if problem:fail(problem)
            elif work_expired():fail('sdk_execution_deadline_cleanup_reserve')
            else:
                _exclusive(handoff_path,handoff);_exclusive(ack,dict(pid=identity['pid'],nonce=nonce));acknowledged=time.monotonic();provider.start()
                telemetry.event('owned_sdk_worker',handoff)
            while reason is None and not owner.poll_exit():
                pump.check()
                if expired():fail('sdk_total_deadline_exceeded');break
                last_resource=provider.runtime(identity['pid'],output);observe(last_resource);problem=resources(last_resource)
                if problem:
                    if owner.poll_exit()and problem in('missing_or_invalid_resource:working_set_bytes','missing_or_invalid_resource:private_bytes'):break
                    fail(problem);break
                heartbeat_error=shared.heartbeat_failure(handoff_path)
                if heartbeat_error is not None:fail('heartbeat_failure',RuntimeError(str(heartbeat_error)));break
                last_status=shared._latest(status_path,last_status)
                if last_status is None:
                    if time.monotonic()-acknowledged>5:fail('worker_status_initial_missing');break
                else:
                    problem=_status_reason(last_status,handoff)
                    if problem:fail(problem);break
                already_complete=last_status is not None and last_status.get('phase')=='complete'and last_status.get('sdk_complete')is True
                if work_expired()and not already_complete:fail('sdk_execution_deadline_cleanup_reserve');break
                observe(last_resource);samples+=1;telemetry.event('sample',dict(resources=last_resource,status=last_status));time.sleep(.2)
            if reason is None:
                heartbeat_error=shared.heartbeat_failure(handoff_path)
                if heartbeat_error is not None:fail('heartbeat_failure',RuntimeError(str(heartbeat_error)))
                else:
                    last_status=shared._latest(status_path,None);problem=_status_reason(last_status,handoff)
                    if problem:fail(problem)
                    elif owner.exit_code()!=0 or last_status['phase']!='complete'or last_status.get('sdk_complete')is not True:fail('sdk_worker_incomplete_or_nonzero_exit')
                    else:
                        result=_read(output/'sdk-result.json')
                        if result.get('status')!='sdk_ready'or result.get('nonce')!=nonce or result.get('training_complete')is not False:fail('sdk_result_invalid')
    except BaseException as error:fail('sdk_guard_failure',error)
    finally:
        if owner is not None:
            try:
                if not owner.poll_exit():
                    timeout=early_cleanup_wait(10)
                    if timeout<=0:raise TimeoutError('No original SDK cleanup time remains')
                    owner.terminate_and_wait(timeout=timeout)
                worker_exit=owner.exit_code();timeout=early_cleanup_wait(10)
                if timeout<=0:raise TimeoutError('No original SDK time remains for Job proof')
                job_active=owner.quiesce(timeout=timeout)
                if job_active!=0:raise RuntimeError('SDK Job still has active descendants')
            except BaseException as error:certain=False;cleanup.append(str(error));fail('sdk_job_cleanup_uncertain',error)
            # Retained Job/process handles are closed only after quiescence and
            # process exit proof, so OwnedProcess.close cannot wait another10s.
            # On uncertainty the parent exit still closes its kill-on-close Job;
            # keep the global lock and never claim successful cleanup.
            if certain and remaining(10)>0:
                try:owner.close()
                except BaseException as error:certain=False;cleanup.append(str(error));fail('sdk_job_close_failed',error)
            else:
                certain=False
                if not injected['owner']:
                    # Close the retained kill-on-close Job without adding the
                    # OwnedProcess.close() ten-second exit wait. This requests
                    # tree termination, but is deliberately not an empty-Job proof.
                    for attribute in('_job','_handle'):
                        handle=getattr(owner,attribute,None)
                        if handle:
                            try:
                                if not owner._kernel.CloseHandle(handle):raise OSError('Native SDK handle close failed')
                                setattr(owner,attribute,None)
                            except BaseException as error:cleanup.append(str(error));fail('sdk_handle_close_uncertain',error)
        elif child is not None:certain=False
        if child is not None:
            try:launcher_exit=child.wait(timeout=early_cleanup_wait(10))
            except BaseException:
                try:child.terminate();launcher_exit=child.wait(timeout=early_cleanup_wait(10))
                except BaseException as error:certain=False;cleanup.append(str(error));fail('sdk_launcher_cleanup_uncertain',error)
        if pump is not None:
            try:
                timeout=max(0,min(PIPE_JOIN_SECONDS,remaining(SECONDS)-RESOURCE_JOIN_SECONDS-FINAL_IO_RESERVE))
                pump.thread.join(timeout=timeout)
                if pump.thread.is_alive():raise TimeoutError('SDK stdout did not quiesce within original deadline')
                for stream in(pump.stream,pump.source):
                    actions=([lambda:os.fsync(stream.fileno())]if stream is pump.stream else[])+[stream.close]
                    for action in actions:
                        try:action()
                        except BaseException as error:
                            if pump.error is None:pump.error=error
                            else:pump.error.add_note('Secondary SDK stdout close error: '+str(error))
                pump.check()
            except BaseException as error:certain=False;cleanup.append(str(error));fail('sdk_stdout_failure',error)
        try:
            if injected['resources']:provider.close()  # Fake provider; never a native blocking call.
            else:
                provider._stop.set()
                if provider._thread is not None:
                    provider._thread.join(timeout=max(0,min(RESOURCE_JOIN_SECONDS,remaining(SECONDS)-FINAL_IO_RESERVE)))
                    if provider._thread.is_alive():raise TimeoutError('GPU reader did not quiesce within original deadline')
        except BaseException as error:certain=False;cleanup.append(str(error));fail('sdk_resource_reader_close_failed',error)
        if remaining(10)<=0:certain=False;fail('sdk_total_deadline_exceeded')
        if lock_owned and certain:
            try:
                if _read(DEFAULT_LOCK).get('nonce')!=nonce:raise RuntimeError('Global model lock owner changed')
                DEFAULT_LOCK.unlink();lock_owned=False
            except BaseException as error:fail('sdk_global_unlock_failed',error)
        if expired():fail('sdk_total_deadline_exceeded')
        report=dict(kind='gpc_sdk_startup_guard_report.v1',status='sdk_ready'if reason is None else'stopped',guard_reason=reason,error=error_text,
            authorization_id=authorization_id,reviewed_commit=reviewed_commit,claim=claim,claim_path=str(claim_path),claim_sha256=_sha(claim_path),
            output_dir=str(output),nonce=nonce,guard_pid=os.getpid(),launcher_pid=child.pid if child else None,
            actual_worker_pid=handoff['actual_worker_pid']if handoff else None,job_active_processes_after_cleanup=job_active,
            job_memory_bytes=3*1024**3,worker_exit_code=worker_exit,launcher_exit_code=launcher_exit,single_model_lock_retained=lock_owned,
            cleanup_errors=cleanup,injected=injected,native_sdk_path_verified=reason is None and not any(injected.values()),
            last_status=last_status,last_resources=last_resource,heartbeat_failure=heartbeat_error,min_free=minimum,observed_peaks=peaks,samples=samples,
            total_cap_bytes=TOTAL_BYTES,fixed_prepaid_bytes=3*MIB,monitor_bytes=telemetry.monitor_bytes,stdout_bytes=telemetry.stdout_bytes,
            wall_seconds=time.monotonic()-mono,ended_utc=_utc(),training_complete=False,model_loads=0,updates=0,forecasts=0,physics=0)
        def finish_time():
            if expired():fail('sdk_total_deadline_exceeded')
            report.update(status='sdk_ready'if reason is None else'stopped',guard_reason=reason,error=error_text,
                native_sdk_path_verified=reason is None and not any(injected.values()),wall_seconds=time.monotonic()-mono,ended_utc=_utc())
        try:telemetry.event('final',dict(status=report['status'],guard_reason=reason));telemetry.write_stop(report)
        except BaseException as error:fail('sdk_final_io_failed',error)
        try:telemetry.close()
        except BaseException as error:fail('sdk_telemetry_close_failed',error)
        finish_time()
        try:
            previous=reason;telemetry.write_stop(report,telemetry.emergency);finish_time()
            if previous!=reason:telemetry.write_stop(report,telemetry.emergency)
        except BaseException as error:fail('sdk_stop_record_failed',error)
        try:telemetry.emergency.close()
        except BaseException as error:fail('sdk_stop_record_close_failed',error)
        before=report['guard_reason'];finish_time()
        if before!=reason:
            try:
                with(directory/'stop-record.json').open('r+b',buffering=0)as stream:telemetry.write_stop(report,stream)
            except BaseException as error:fail('sdk_final_stop_record_failed',error)
        finish_time()
    return report


def worker(path,*,startup_helper=None,native_check=True):
    """Use exactly the shared SDK helper, then export a bounded SDK-only result."""
    handoff=verify_worker(path,native_check=native_check);output=Path(handoff['output_dir'])
    trace=shared.StartupTrace(path);runtime={'torch':None};primary=None
    def refresh():
        value=trace.snapshot()
        if runtime['torch']is not None:value.update(shared.cuda_allocator_sample(runtime['torch']))
        return value
    payload=shared.status_template();payload.update(policy_calls=0,sdk_complete=False)
    try:
        with shared.Heartbeat(path,payload,refresh=refresh)as heartbeat:
            try:
                torch,diagnostics=(startup_helper or shared.initialize_sdk)(heartbeat,trace,cpu_threads=2,require_cuda=True)
                runtime['torch']=torch
                if not isinstance(diagnostics,dict)or diagnostics.get('sdk_complete')is not True:raise ValueError('Shared SDK helper did not complete')
                claim=handoff['claim']
                def deadline():
                    if time.time()>=claim['execution_deadline_unix']or time.monotonic()>=claim['execution_deadline_monotonic']:
                        raise TimeoutError('SDK export must finish before the fixed12-second cleanup reserve')
                deadline();trace.stage('sdk_result_export','before');heartbeat.update(**trace.snapshot())
                result=dict(kind='gpc_sdk_startup_result.v1',status='sdk_ready',sdk_complete=True,training_complete=False,
                    nonce=handoff['nonce'],actual_worker_pid=os.getpid(),diagnostics=diagnostics,
                    model_loads=0,updates=0,forwards=0,backwards=0,optimizer_steps=0,forecasts=0,physics=0)
                data=_json(result)
                if len(data)>SLOT:raise ValueError('SDK result exceeds its prepaid64KiB slot')
                temporary=output/('sdk-result.json.'+uuid.uuid4().hex+'.tmp')
                with temporary.open('xb',buffering=0)as stream:_write_all(stream,data);os.fsync(stream.fileno())
                deadline();_publish_json(temporary,output/'sdk-result.json');deadline()
                trace.stage('sdk_result_export','after');heartbeat.update(phase='complete',sdk_complete=True,training_complete=False,**trace.snapshot())
            except BaseException as error:
                primary=error;heartbeat.record_error(error,step='sdk.worker');raise
    finally:
        try:trace.close()
        except BaseException as error:
            if primary is not None:primary.add_note('Startup trace close failed: '+str(error))
            else:raise
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    group=parser.add_mutually_exclusive_group();group.add_argument('--execute',action='store_true');group.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--authorization-id');parser.add_argument('--reviewed-commit');parser.add_argument('--python',type=Path)
    args=parser.parse_args(argv)
    if args.worker:
        if args.authorization_id is not None or args.reviewed_commit is not None or args.python is not None:parser.error('Worker accepts only its verified handoff')
        worker(args.worker);return 0
    if not args.execute:
        output,claim=paths();print(json.dumps(dict(status='plan_only',authorization_id=AUTHORIZATION,output_dir=str(output),claim_path=str(claim),
            total_seconds=SECONDS,execution_seconds=WORK_SECONDS,cleanup_reserve_seconds=CLEANUP_RESERVE,
            total_bytes=TOTAL_BYTES,workers=1,models=0,network_loads=0,forwards=0,backwards=0,optimizer_steps=0,physics=0,
            native_resource_queries=0,automatic_retry=False)));return 0
    try:
        report=supervise(authorization_id=args.authorization_id,reviewed_commit=args.reviewed_commit,python_executable=args.python or sys.executable)
        print(json.dumps(report,indent=2));return 0 if report['status']=='sdk_ready'else 2
    except Exception as error:
        print(json.dumps(dict(status='refused_or_stopped',error=type(error).__name__+': '+str(error))));return 2


if __name__=='__main__':raise SystemExit(main())
