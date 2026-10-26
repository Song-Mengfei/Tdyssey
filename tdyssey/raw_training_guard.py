"""Narrow Windows training parent; importing/planning performs no native query.

The worker must call ``worker_gate`` before importing Torch, then use the returned
WorkerStatus as a context manager. Windows owns the exact base-Python child in a
3GiB kill-on-close Job before acknowledging it. Linux execution is deliberately
unavailable until an equivalent hard process guard is implemented and checked.
Native layouts/queries follow the earlier retained-handle Windows guard, without
its experiment identities, accounts, samplers, or storage framework.
"""
import ctypes
from ctypes import wintypes as W
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

GIB = 1024**3
CUDA_CAPS = dict(cuda_allocated_bytes=GIB//2, cuda_reserved_bytes=GIB,
                 cuda_peak_allocated_bytes=GIB//2, cuda_peak_reserved_bytes=GIB)
LOCK = Path(tempfile.gettempdir())/'tdyssey-raw-training-cuda0.lock'
MAX_LINE, MAX_BYTES, MAX_RECORDS = 2048, 16*1024**2, 10000
PHASES = ('setup', 'model_loading', 'training', 'export', 'complete', 'error')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _write_all(stream, payload):
    while payload:
        written = stream.write(payload)
        if type(written) is not int or not 0 < written <= len(payload):
            raise OSError('Guard stream write made no valid progress')
        payload = payload[written:]


def _encode(value):
    data = (json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n').encode()
    if len(data)>MAX_LINE:raise ValueError('Guard record exceeds2048 bytes')
    return data


def resource_reason(value, *, startup=False):
    floors = dict(ram=4*GIB if startup else 3*GIB//2,
        commit=6*GIB if startup else 3*GIB, gpu=4*GIB if startup else 3*GIB//2, disk=2*GIB)
    for key, floor in floors.items():
        if type(value.get(key)) is not int or value[key]<floor:return 'resource_missing_or_below_floor:'+key
    if not _finite(value.get('gpu_age')) or not 0<=value['gpu_age']<=5:return 'gpu_sample_stale'
    if not startup:
        for key, cap in (('working_set',2*GIB),('peak_working_set',2*GIB),('private',3*GIB)):
            if type(value.get(key)) is not int or not 0<=value[key]<=cap:return 'process_memory_missing_or_over_cap:'+key
    return None


def status_reason(value, token, *, now=None, mono=None):
    now = time.time() if now is None else now; mono = time.monotonic() if mono is None else mono
    if not isinstance(value,dict) or value.get('pid')!=token['pid'] or value.get('nonce')!=token['nonce']:
        return 'worker_identity_invalid'
    if (not _finite(value.get('at')) or not -.5<=now-value['at']<=5
            or not _finite(value.get('mono')) or not 0<=mono-value['mono']<=5):return 'worker_status_stale'
    if value.get('phase') not in PHASES:return 'worker_phase_invalid'
    if value.get('phase')=='error':return 'worker_error:'+str(value.get('error','unspecified'))
    for key, cap in dict(model_instances=1,model_loads=1,reserved_updates=token['max_updates'],
                          completed_updates=token['max_updates']).items():
        if type(value.get(key)) is not int or not 0<=value[key]<=cap:return 'worker_counter_invalid:'+key
    if value['completed_updates']>value['reserved_updates']:return 'worker_updates_not_reserved'
    if value['model_instances']>value['model_loads']:return 'worker_model_not_loaded'
    if value['phase']=='training' and value['model_instances']!=1:return 'training_model_missing'
    if value.get('cuda_at') is None:
        if value['phase'] not in ('setup','model_loading') or value['model_instances']!=0:return 'cuda_sample_missing'
        if value.get('cuda_wall') is not None or any(value.get(k) is not None for k in CUDA_CAPS):return 'cuda_sample_inconsistent'
    else:
        if (not _finite(value.get('cuda_at')) or not 0<=mono-value['cuda_at']<=5
                or not _finite(value.get('cuda_wall')) or not -.5<=now-value['cuda_wall']<=5):return 'cuda_sample_stale'
        for key, cap in CUDA_CAPS.items():
            if type(value.get(key)) is not int or not 0<=value[key]<=cap:return 'cuda_over_cap_or_invalid:'+key
    return None


class _Memory(ctypes.Structure):
    _fields_ = [('length',W.DWORD),('load',W.DWORD)]+[(n,ctypes.c_ulonglong) for n in
        ('total_physical','available_physical','total_pagefile','available_pagefile','total_virtual','available_virtual','extended')]
class _Performance(ctypes.Structure):
    _fields_ = [('cb',W.DWORD)]+[(n,ctypes.c_size_t) for n in ('CommitTotal','CommitLimit','CommitPeak',
        'PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(n,W.DWORD) for n in ('HandleCount','ProcessCount','ThreadCount')]
class _Process(ctypes.Structure):
    _fields_ = [('cb',W.DWORD),('PageFaultCount',W.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize',
        'WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
class _Basic(ctypes.Structure):
    _fields_ = [('user',ctypes.c_longlong),('job_user',ctypes.c_longlong),('flags',W.DWORD),
        ('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active',W.DWORD),('affinity',ctypes.c_size_t),('priority',W.DWORD),('scheduling',W.DWORD)]
class _Job(ctypes.Structure):
    _fields_ = [('basic',_Basic),('io',ctypes.c_ulonglong*6)]+[(n,ctypes.c_size_t) for n in ('process_memory','job_memory','peak_process','peak_job')]
class _Accounting(ctypes.Structure):
    _fields_ = [(n,ctypes.c_longlong) for n in ('user','kernel','period_user','period_kernel')]+[(n,W.DWORD) for n in ('faults','total','active','terminated')]


class _Native:
    def __init__(self):
        self.k = ctypes.WinDLL('kernel32',use_last_error=True); self.p = ctypes.WinDLL('psapi',use_last_error=True); self.job = None; self.owned = False
        for name,args,result in (('CreateJobObjectW',[ctypes.c_void_p,W.LPCWSTR],W.HANDLE),
            ('SetInformationJobObject',[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD],W.BOOL),
            ('QueryInformationJobObject',[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD,ctypes.c_void_p],W.BOOL),
            ('AssignProcessToJobObject',[W.HANDLE,W.HANDLE],W.BOOL),('GetProcessId',[W.HANDLE],W.DWORD),
            ('TerminateJobObject',[W.HANDLE,W.UINT],W.BOOL),('CloseHandle',[W.HANDLE],W.BOOL),
            ('GlobalMemoryStatusEx',[ctypes.POINTER(_Memory)],W.BOOL)):
            function=getattr(self.k,name);function.argtypes=args;function.restype=result
        self.p.GetPerformanceInfo.argtypes=[ctypes.POINTER(_Performance),W.DWORD];self.p.GetPerformanceInfo.restype=W.BOOL
        self.p.GetProcessMemoryInfo.argtypes=[W.HANDLE,ctypes.POINTER(_Process),W.DWORD];self.p.GetProcessMemoryInfo.restype=W.BOOL
    @staticmethod
    def require(result):
        if not result:raise ctypes.WinError(ctypes.get_last_error())
    def own(self, child):
        if self.k.GetProcessId(int(child._handle))!=child.pid:raise OSError('Retained child PID differs')
        self.job=self.k.CreateJobObjectW(None,None);self.require(self.job)
        limits=_Job();limits.basic.flags=0x2200;limits.job_memory=3*GIB
        self.require(self.k.SetInformationJobObject(self.job,9,ctypes.byref(limits),ctypes.sizeof(limits)))
        self.require(self.k.AssignProcessToJobObject(self.job,int(child._handle)))
        self.owned=True
        observed=_Job();self.require(self.k.QueryInformationJobObject(self.job,9,ctypes.byref(observed),ctypes.sizeof(observed),None))
        if observed.basic.flags&0x2200!=0x2200 or observed.job_memory!=3*GIB:raise OSError('Job limits differ')
    def snapshot(self, child, output):
        memory=_Memory();memory.length=ctypes.sizeof(memory);self.require(self.k.GlobalMemoryStatusEx(ctypes.byref(memory)))
        info=_Performance();info.cb=ctypes.sizeof(info);self.require(self.p.GetPerformanceInfo(ctypes.byref(info),info.cb))
        if info.PageSize<=0 or info.CommitLimit<info.CommitTotal:raise OSError('Invalid native commit observation')
        value=dict(ram=int(memory.available_physical),commit=int((info.CommitLimit-info.CommitTotal)*info.PageSize),disk=shutil.disk_usage(output).free)
        if child is not None:
            process=_Process();process.cb=ctypes.sizeof(process);self.require(self.p.GetProcessMemoryInfo(int(child._handle),ctypes.byref(process),process.cb))
            value.update(working_set=int(process.WorkingSetSize),peak_working_set=int(process.PeakWorkingSetSize),private=int(process.PrivateUsage))
        return value
    def active(self):
        if self.job is None:return 0
        value=_Accounting();self.require(self.k.QueryInformationJobObject(self.job,1,ctypes.byref(value),ctypes.sizeof(value),None))
        if value.active>value.total:raise OSError('Invalid native Job accounting')
        return int(value.active)
    def cleanup(self, deadline):
        if self.active():self.require(self.k.TerminateJobObject(self.job,1))
        while self.active():
            if time.monotonic()>=deadline:raise TimeoutError('Owned Job did not become empty')
            time.sleep(.02)
        return 0
    def close(self):
        if self.job is not None:
            job,self.job=self.job,None;self.require(self.k.CloseHandle(job))


class _GPU:
    def __init__(self, deadline):
        self.deadline=deadline;self.sampled=(None,None);self.error=None;self.stop=threading.Event();self.thread=None
    def sample(self):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Original deadline expired during GPU query')
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,timeout=min(3.,remaining))
        values=[int(row.strip())*1024**2 for row in result.stdout.splitlines()]
        if not values or min(values)<0:raise ValueError('Missing native GPU observation')
        self.sampled=min(values),time.monotonic()
    def start(self):
        def loop():
            while not self.stop.wait(1):
                try:self.sample()
                except BaseException as error:self.error=error;return
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def add(self, value):
        if self.error is not None:raise RuntimeError('Native GPU reader failed') from self.error
        free,at=self.sampled
        return dict(value,gpu=free,gpu_age=time.monotonic()-at if at is not None else None)
    def close(self, timeout):
        self.stop.set()
        if self.thread:self.thread.join(timeout=max(0.,timeout))
        if self.thread and self.thread.is_alive():raise TimeoutError('GPU reader cleanup incomplete')


class WorkerStatus:
    """One independent1s heartbeat; training checks before work and after updates."""
    def __init__(self, token, stream, *, refresh=None):
        self.token,self.stream,self.refresh=token,stream,refresh
        self.value=dict(phase='setup',model_instances=0,model_loads=0,reserved_updates=0,completed_updates=0,
            cuda_at=None,cuda_wall=None,**dict.fromkeys(CUDA_CAPS))
        self.lock=threading.RLock();self.stop=threading.Event();self.error=None;self.thread=None;self._torch_attached=False
    def _publish(self):
        with self.lock:
            value=dict(self.value)
            if self.refresh:value.update(self.refresh())
            value.update(pid=self.token['pid'],nonce=self.token['nonce'],at=time.time(),mono=time.monotonic())
            reason=status_reason(value,self.token)
            if reason:raise RuntimeError(reason)
            _write_all(self.stream,_encode(value));self.stream.flush()
    def check(self, phase, counts):
        with self.lock:
            if self.error is not None:raise RuntimeError('Independent heartbeat failed') from self.error
            try:
                if time.time()>=self.token['deadline_wall'] or time.monotonic()>=self.token['deadline']:
                    raise TimeoutError('Original training deadline exceeded')
                for key in ('model_loads','reserved_updates','completed_updates'):
                    if key in counts and (type(counts[key]) is not int or counts[key]<self.value[key]):
                        raise ValueError('Training counters cannot move backward: '+key)
                self.value.update(counts,phase=phase);self._publish()
            except BaseException as error:self.error=error;raise
    def attach_torch(self, torch):
        with self.lock:
            if self.error is not None:raise RuntimeError('Independent heartbeat failed') from self.error
            if self._torch_attached:
                self.error=RuntimeError('CUDA startup may be attached only once');raise self.error
            self._torch_attached=True
        try:
            # Keep setup heartbeats available during initialization. Only a
            # completed CUDA init may supply a fresh allocator observation.
            torch.set_num_threads(2)
            if torch.cuda.is_available() is not True:raise RuntimeError('CUDA is unavailable')
            torch.cuda.init();torch.cuda.reset_peak_memory_stats(0)
            def refresh():
                values=[torch.cuda.memory_allocated(0),torch.cuda.memory_reserved(0),
                    torch.cuda.max_memory_allocated(0),torch.cuda.max_memory_reserved(0)]
                return dict(zip(CUDA_CAPS,values),cuda_at=time.monotonic(),cuda_wall=time.time())
            with self.lock:
                if self.error is not None:raise RuntimeError('Independent heartbeat failed') from self.error
                self.refresh=refresh;self._publish()
        except BaseException as error:
            with self.lock:
                if self.error is None:self.error=error
            raise
    def __enter__(self):
        self._publish()
        def loop():
            while not self.stop.wait(1):
                try:self.check(self.value['phase'],{})
                except BaseException as error:
                    self.error=error
                    try:self._error_record(error)
                    except BaseException as secondary:error.add_note('Heartbeat error publication failed: '+str(secondary))
                    return
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self
    def __exit__(self, kind, error, tb):
        self.stop.set();self.thread.join(timeout=min(2.,max(0.,self.token['deadline']-time.monotonic())))
        if error is not None:
            try:self._error_record(error)
            except BaseException as secondary:error.add_note('Guard error publication failed: '+str(secondary))
        elif self.thread.is_alive() or self.error is not None:raise RuntimeError('Independent heartbeat did not close cleanly') from self.error
    def _error_record(self, error):
        with self.lock:
            _write_all(self.stream,_encode(dict(pid=self.token['pid'],nonce=self.token['nonce'],phase='error',
                at=time.time(),mono=time.monotonic(),error=type(error).__name__+': '+str(error)[:500])));self.stream.flush()


def worker_gate():
    """Block before Torch until the direct parent proves Job ownership via ACK."""
    parent=int(os.environ['TDYSSEY_RAW_PARENT']);nonce=os.environ['TDYSSEY_RAW_NONCE']
    if os.getppid()!=parent:raise RuntimeError('Training guardian parent differs')
    stream=sys.stdout.buffer;_write_all(stream,_encode(dict(ready=True,pid=os.getpid(),nonce=nonce)));stream.flush()
    token=json.loads(sys.stdin.buffer.readline(MAX_LINE+1))
    if token.get('pid')!=os.getpid() or token.get('nonce')!=nonce or token.get('parent')!=parent:
        raise RuntimeError('No matching owned-worker ACK')
    sys.stdout=sys.stderr  # Keep the original pipe exclusively for bounded status.
    return WorkerStatus(token,stream)


class _Pipe:
    def __init__(self, stream, token=None):
        self.latest=None;self.error=None;self.bytes=self.records=0;self.token=token
        def loop():
            try:
                while data:=stream.readline(MAX_LINE+1):
                    self.bytes+=len(data);self.records+=1
                    if len(data)>MAX_LINE or not data.endswith(b'\n') or self.bytes>MAX_BYTES or self.records>MAX_RECORDS:
                        raise OSError('Bounded status pipe allowance exceeded or partial record')
                    value=json.loads(data)
                    if not isinstance(value,dict):raise ValueError('Status record must be an object')
                    if self.token is not None:
                        problem=status_reason(value,self.token,now=value.get('at'),mono=value.get('mono'))
                        if problem:raise RuntimeError(problem)
                        if self.latest and self.latest.get('at') is not None:
                            if value['mono']<self.latest['mono']:raise ValueError('Worker monotonic time moved backward')
                            if any(value[k]<self.latest[k] for k in ('model_loads','reserved_updates','completed_updates')):
                                raise ValueError('Worker counters moved backward')
                    self.latest=value
            except BaseException as error:self.error=error
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()


def run_guarded(config, worker_command, *, execute=False):
    """worker_command is [script_path,*args]; this parent supplies base CPython.

    Config requires output_dir, max_updates and wall_seconds (10..1800s). There
    is no automatic retry/resume. A failed/uncertain Job cleanup retains the
    single-model lock. Actual Windows integration remains unverified here.
    """
    updates,seconds=config.get('max_updates'),config.get('wall_seconds')
    if type(updates) is not int or updates<=0 or not _finite(seconds) or not 10<seconds<=1800:
        raise ValueError('Explicit finite update/wall limits required; wall must be10..1800s')
    if not worker_command or any(not isinstance(arg,str) for arg in worker_command):raise ValueError('Explicit script argv required')
    if type(execute) is not bool:raise ValueError('Execute must be an explicit bool')
    if not execute:return dict(status='plan',platform='win32',native_verified=False,max_updates=updates,wall_seconds=seconds)
    if sys.platform!='win32':raise RuntimeError('Actual training requires the Windows Job guard; Linux execution is unavailable')
    wall,began=time.time(),time.monotonic();deadline=began+seconds;output=Path(config['output_dir']).absolute()
    native=gpu=child=pipe=None;lock=None;certain=False;reason=None;latest=None;cleanup=[];nonce=uuid.uuid4().hex
    created=lock_retained=job_empty=False
    def fail(message):
        nonlocal reason
        if reason is None:reason=message
    try:
        output.mkdir(parents=True,exist_ok=False);created=True
        lock=LOCK.open('xb',buffering=0);_write_all(lock,_encode(dict(parent=os.getpid(),nonce=nonce)));os.fsync(lock.fileno())
        lock_retained=True
        native=_Native();gpu=_GPU(deadline);gpu.sample()
        problem=resource_reason(gpu.add(native.snapshot(None,output)),startup=True)
        if problem:raise RuntimeError(problem)
        if time.monotonic()>=deadline-10 or time.time()>=wall+seconds-10:raise TimeoutError('Startup exhausted work allowance')
        env=dict(os.environ,TDYSSEY_RAW_PARENT=str(os.getpid()),TDYSSEY_RAW_NONCE=nonce,
            PYTHONPATH=os.pathsep.join(str(p) for p in sys.path if p))
        env.update({k:'2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        env.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
        child=subprocess.Popen([sys._base_executable,'-B',*worker_command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=env,bufsize=0)
        pipe=_Pipe(child.stdout);until=min(deadline-10,time.monotonic()+5)
        while pipe.latest is None:
            if pipe.error:raise pipe.error
            if child.poll() is not None or time.monotonic()>=until:raise TimeoutError('Pre-Torch worker gate missing')
            time.sleep(.02)
        if pipe.latest!=dict(ready=True,pid=child.pid,nonce=nonce):raise RuntimeError('Direct worker PID/nonce differs')
        native.own(child)
        if time.monotonic()>=deadline-10 or time.time()>=wall+seconds-10:raise TimeoutError('Ownership exhausted work allowance')
        token=dict(parent=os.getpid(),pid=child.pid,nonce=nonce,max_updates=updates,deadline=deadline,deadline_wall=wall+seconds)
        pipe.token=token
        _write_all(child.stdin,_encode(token));child.stdin.flush();child.stdin.close();gpu.start();ack=time.monotonic()
        while child.poll() is None:
            if time.monotonic()>=deadline-10 or time.time()>=wall+seconds-10:raise TimeoutError('Cleanup reserve reached')
            if pipe.error:raise pipe.error
            value=pipe.latest
            if value.get('ready') is not True:
                problem=status_reason(value,token)
                if problem:raise RuntimeError(problem)
                latest=value
            elif time.monotonic()-ack>5:raise TimeoutError('Initial status missing')
            try:problem=resource_reason(gpu.add(native.snapshot(child,output)))
            except OSError:
                if child.poll() is not None:break
                raise
            if problem:raise RuntimeError(problem)
            time.sleep(.2)
        pipe.thread.join(timeout=min(1.,max(0.,deadline-time.monotonic())))
        if pipe.thread.is_alive():raise RuntimeError('Status pipe did not reach EOF after worker exit')
        if pipe.error:raise pipe.error
        latest=pipe.latest;problem=status_reason(latest,token)
        if problem:raise RuntimeError(problem)
        if child.returncode!=0 or latest['phase']!='complete' or latest['model_instances']!=0:
            raise RuntimeError('Worker failed or model was not released')
        if latest['completed_updates']!=updates or latest['reserved_updates']!=updates:raise RuntimeError('Incomplete update count')
    except BaseException as error:fail(type(error).__name__+': '+str(error))
    finally:
        try:
            if child is not None and (native is None or not native.owned):
                if child.poll() is None:child.terminate()
                child.wait(timeout=max(.001,deadline-time.monotonic()-4))
            if native is not None:native.cleanup(min(deadline-4,time.monotonic()+5));certain=True;job_empty=True
            elif child is None:certain=True
        except BaseException as error:cleanup.append(str(error));fail('Job cleanup uncertain: '+str(error))
        for action in ((native.close if native else None),
            (lambda:child.wait(timeout=max(.001,deadline-time.monotonic()-3))) if child else None,
            (lambda:gpu.close(min(3.,max(0.,deadline-time.monotonic()-1)))) if gpu else None):
            if action:
                try:action()
                except BaseException as error:certain=False;cleanup.append(str(error));fail('Cleanup failed: '+str(error))
        if child is not None:
            try:
                if child.stdin and not child.stdin.closed:child.stdin.close()
                if pipe:
                    pipe.thread.join(timeout=min(1.,max(0.,deadline-time.monotonic()-1)))
                    if pipe.thread.is_alive():raise TimeoutError('Status pipe cleanup incomplete')
                child.stdout.close()
            except BaseException as error:certain=False;cleanup.append(str(error));fail('Pipe cleanup failed: '+str(error))
        if lock:
            try:lock.close()
            except BaseException as error:certain=False;cleanup.append(str(error));fail('Single-model lock close failed: '+str(error))
            if certain:
                try:LOCK.unlink();lock_retained=False
                except BaseException as error:fail('Single-model unlock failed: '+str(error))
        if time.monotonic()>=deadline or time.time()>=wall+seconds:fail('Original total deadline exceeded')
    report=dict(status='complete' if reason is None else 'stopped',reason=reason,worker_pid=child.pid if child else None,
        nonce=nonce,job_empty=job_empty,lock_retained=lock_retained,cleanup_errors=cleanup,
        last_status=latest,wall_seconds=time.monotonic()-began,native_verified=False)
    if created:
        try:
            with (output/'guard-report.json').open('xb',buffering=0) as stream:
                _write_all(stream,(json.dumps(report,allow_nan=False)+'\n').encode());os.fsync(stream.fileno())
        except BaseException as error:fail('Final report IO failed: '+str(error));report.update(status='stopped',reason=reason)
    if time.monotonic()>=deadline or time.time()>=wall+seconds:
        report.update(status='stopped',reason=reason or 'Original total deadline exceeded',wall_seconds=time.monotonic()-began)
    return report
