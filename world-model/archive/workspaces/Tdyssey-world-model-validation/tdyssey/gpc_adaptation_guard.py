"""Protocol-specific future Windows adaptation bridge; imports do no work.

Only injected fakes are used by this module's engineering tests. The production
worker waits in guarded_c_child.py until its actual PID belongs to the parent's
retained 3-GiB kill-on-close Job. No storage or multiprocess framework is added.
"""
from copy import deepcopy
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid

from . import gpc_adaptation as protocol
from .wm_compat_guard import WindowsWorker, ResourceProvider, resource_reason, DEFAULT_LOCK, _read, _publish_json, _write_all

ROOT=Path(__file__).resolve().parents[1]
OUTPUT='gpc-supervised-adaptation-pilot-20261025'
MIB=1024**2
SLOT=65536
STDOUT_CAP=2*MIB
MONITOR_CAP=60*MIB
TELEMETRY_CAP=64*MIB
LEDGER_RESERVE=16*MIB
FIXED_RESERVE=MIB
ENV_HANDOFF='GPC_ADAPTATION_HANDOFF'
ENV_NONCE='GPC_ADAPTATION_GUARD_NONCE'
ENV_PARENT='GPC_ADAPTATION_GUARD_PID'
FORBIDDEN=('environment_creations','resets','restores','controls','physics_substeps','real_renders','policy_sampling')
_STATUS_LOCK=threading.RLock()


def _json(v):return (json.dumps(v,sort_keys=True,allow_nan=False)+'\n').encode()
def _finite(v):return type(v) in (int,float) and math.isfinite(v)
def _now():return datetime.now(timezone.utc).isoformat()
def _path(v):
    path=Path(v).absolute()
    for p in (path,*path.parents):
        if p.is_symlink() or getattr(p,'is_junction',lambda:False)():raise ValueError('Path aliases are forbidden')
    return path.resolve()
def _exclusive(path,value):
    data=_json(value)
    if len(data)>SLOT:raise ValueError('Guardian metadata exceeds 64KiB')
    with Path(path).open('xb',buffering=0) as f:_write_all(f,data);os.fsync(f.fileno())


def _assets(gpc_root,checkpoint):
    from .gpc_supervised_adaptation import audit_upstream_objective
    evidence=audit_upstream_objective(gpc_root)
    spec=json.loads((ROOT/'manifests/gpc_world_model.json').read_bytes())['checkpoint']
    if checkpoint.stat().st_size!=spec['bytes'] or protocol.sha_file(checkpoint)!=spec['sha256']:
        raise ValueError('Exact pinned initial checkpoint required before claim')
    return dict(source_audit_sha256=hashlib.sha256(_json(evidence)).hexdigest(),
                checkpoint_sha256=spec['sha256'],checkpoint_bytes=spec['bytes'])


def _clocks(binding,wall,mono,*,eval_only=False,conditioning_only=False,raw_fulltrain=False):
    if raw_fulltrain and (eval_only or conditioning_only):raise ValueError('Raw full training is a separate fixed mode')
    eval_only=eval_only or conditioning_only
    seconds=1800 if raw_fulltrain else 300 if conditioning_only else 3600
    train_seconds=1200 if raw_fulltrain else 900
    result=deepcopy(binding);start=result.get('started_unix',wall);begin=result.get('started_monotonic',mono)
    if not _finite(start) or not _finite(begin):raise ValueError('Invalid original clocks')
    expected=dict(started_unix=start,started_monotonic=begin,deadline_unix=start+seconds,deadline_monotonic=begin+seconds,
                  training_deadline_unix=start+train_seconds,training_deadline_monotonic=begin+train_seconds)
    if raw_fulltrain:expected['training_deadline_applicable']=True
    if eval_only:
        expected.update(training_deadline_unix=None,training_deadline_monotonic=None,training_deadline_applicable=False)
    if start>wall+.5 or begin>mono or wall>=start+seconds or mono>=begin+seconds:raise ValueError('Original deadline invalid/expired')
    for k,v in expected.items():
        if k in result and result[k]!=v:raise ValueError('Original deadline cannot be extended')
        result[k]=v
    return result


def _cleanup_wait_timeout(binding, *, raw_fulltrain=False, maximum=10., reserve=8.):
    """Raw-mode waits share the original deadline;8s covers PIPE/GPU reader+IO."""
    if not raw_fulltrain:return maximum
    remaining=min(binding['deadline_unix']-time.time(),binding['deadline_monotonic']-time.monotonic())
    return min(maximum,max(0.,remaining-reserve))


def _preclaim(prepared,binding,python,gpc_root,checkpoint,script,config):
    if not isinstance(prepared,dict) or prepared.get('ready') is not True:raise ValueError('Preparation is not ready')
    python,gpc_root,checkpoint,script,config=map(_path,(python,gpc_root,checkpoint,script,config))
    if not python.is_file() or not script.is_file() or not script.is_relative_to(ROOT/'scripts'):
        raise ValueError('Explicit interpreter and reviewed repository worker required')
    fresh=protocol.plan(config)
    if fresh!=prepared or fresh.get('ready') is not True:raise ValueError('Prepared config/data/benchmark changed')
    verified=protocol.verify_execution(prepared,authorization_id=binding.get('authorization_id'),
        reviewed_commit=binding.get('source_commit'),authorized_config_sha256=binding.get('config_sha256'))
    if any(binding.get(k)!=v for k,v in verified.items()):raise ValueError('Execution authority binding differs')
    cfg=prepared['config']
    identity=protocol.execution_identity(config,cfg,binding)
    if (cfg['output_dir']!=identity['output_dir'] or cfg['limits']['wall_seconds']!=3600
            or cfg['train']['phase_seconds']!=900 or cfg['limits']['telemetry_bytes']!=TELEMETRY_CAP
            or cfg['limits']['active_gpu_models']!=1):raise ValueError('Fixed adaptation protocol differs')
    output=_path(ROOT/identity['output_dir'])
    if output.exists():raise FileExistsError('Fixed candidate already claimed; no alternate output or retry')
    assets=_assets(gpc_root,checkpoint)
    if protocol.sha_file(config)!=prepared['config_sha256']:raise ValueError('Config changed before claim')
    return output,python,gpc_root,checkpoint,script,config,assets


def status_template(*,eval_only=False,conditioning_only=False,raw_fulltrain=False):
    if raw_fulltrain and (eval_only or conditioning_only):raise ValueError('Raw full training is a separate fixed mode')
    eval_only=eval_only or conditioning_only
    result=dict(phase='setup',training_complete=False,model_instances=0,model_loads=0,optimizer_updates=0,
        forecasts=0,euler_steps=0,eval_batches=0,eval_examples=0,**cuda_unknown(),**dict.fromkeys(FORBIDDEN,0))
    if eval_only:result.update(evaluation_only=True,training_applicable=False,source_checkpoint_update=500)
    if conditioning_only:result['conditioning_only']=True
    if raw_fulltrain:result.update(raw_fulltrain=True,evaluation_only=False,training_applicable=True)
    return result


def cuda_unknown():
    return dict(cuda_allocated_bytes=None,cuda_reserved_bytes=None,cuda_peak_allocated_bytes=None,
        cuda_peak_reserved_bytes=None,cuda_sample_at_unix=None,cuda_sample_at_monotonic=None,cuda_sample_state='unavailable')


def cuda_allocator_sample(torch):
    # Timestamp only a completed allocator observation, never an unavailable value.
    value=dict(cuda_allocated_bytes=torch.cuda.memory_allocated(0),cuda_reserved_bytes=torch.cuda.memory_reserved(0),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    return dict(value,cuda_sample_at_unix=time.time(),cuda_sample_at_monotonic=time.monotonic(),cuda_sample_state='fresh')


def monitor_reason(status,binding,*,pid,nonce,now_unix=None,now_monotonic=None,eval_only=False,conditioning_only=False,raw_fulltrain=False):
    if raw_fulltrain and (eval_only or conditioning_only):return 'worker_mode_conflict'
    eval_only=eval_only or conditioning_only
    wall=time.time() if now_unix is None else now_unix;mono=time.monotonic() if now_monotonic is None else now_monotonic
    if wall>=binding['deadline_unix'] or mono>=binding['deadline_monotonic']:return 'total_deadline_exceeded'
    if not isinstance(status,dict) or status.get('actual_worker_pid')!=pid or status.get('nonce')!=nonce:return 'worker_status_identity_invalid'
    if not _finite(status.get('at_unix')) or not -.5<=wall-status['at_unix']<=5:return 'worker_status_stale_or_invalid'
    if (eval_only or raw_fulltrain) and (not _finite(status.get('at_monotonic')) or not 0<=mono-status['at_monotonic']<=5):return 'worker_status_stale_or_invalid'
    if status.get('phase') not in ('setup','model_loading','training','evaluation','export','complete','error'):return 'worker_phase_invalid'
    if status['phase']=='error' or status.get('error'):return 'worker_error:'+str(status.get('error','unspecified'))
    if type(status.get('training_complete')) is not bool:return 'training_completion_flag_missing'
    if raw_fulltrain and (status.get('raw_fulltrain') is not True or status.get('evaluation_only') is not False
            or status.get('training_applicable') is not True or status.get('source_checkpoint_update') is not None):
        return 'raw_fulltrain_contract_invalid'
    if eval_only:
        if (status.get('evaluation_only') is not True or status.get('training_applicable') is not False
                or status['training_complete'] is not False or status.get('source_checkpoint_update')!=500
                or status['phase']=='training'):return 'evaluation_only_contract_invalid'
    elif not status['training_complete'] and (wall>=binding['training_deadline_unix'] or mono>=binding['training_deadline_monotonic']):return 'training_phase_deadline_exceeded'
    if conditioning_only and status.get('conditioning_only') is not True:return 'conditioning_only_contract_invalid'
    for key,cap in dict(model_instances=1,model_loads=4 if raw_fulltrain else 2 if eval_only else 3,
            optimizer_updates=2328 if raw_fulltrain else 0 if eval_only else 500,
            forecasts=48 if raw_fulltrain else 32 if conditioning_only else 3996,
            euler_steps=144 if raw_fulltrain else 96 if conditioning_only else 11988,
            eval_batches=48 if raw_fulltrain else 0 if conditioning_only else 32,
            eval_examples=48 if raw_fulltrain else 0 if conditioning_only else 32).items():
        if type(status.get(key)) is not int or not 0<=status[key]<=cap:return 'worker_counter_invalid:'+key
    if status['training_complete'] and status['optimizer_updates']!=(2328 if raw_fulltrain else 500):return 'premature_training_completion'
    if status['phase']=='training' and status['model_instances']!=1:return 'training_model_missing'
    for key in FORBIDDEN:
        if type(status.get(key)) is not int or status[key]!=0:return 'forbidden_worker_operation:'+key
    unavailable=status.get('cuda_sample_state')=='unavailable'
    if unavailable:
        if status['phase'] not in ('setup','model_loading') or any(status.get(k) is not None for k in
                ('cuda_allocated_bytes','cuda_reserved_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes',
                 'cuda_sample_at_unix','cuda_sample_at_monotonic')):return 'cuda_sample_unavailable_or_inconsistent'
        return None
    if status.get('cuda_sample_state')=='fresh':
        if (not _finite(status.get('cuda_sample_at_unix')) or not -.5<=wall-status['cuda_sample_at_unix']<=5
                or not _finite(status.get('cuda_sample_at_monotonic')) or not 0<=mono-status['cuda_sample_at_monotonic']<=5):
            return 'cuda_sample_stale_or_invalid'
    else:return 'cuda_sample_state_invalid'
    for key,cap in (('cuda_allocated_bytes',512*MIB),('cuda_reserved_bytes',1024*MIB),('cuda_peak_allocated_bytes',512*MIB),('cuda_peak_reserved_bytes',1024*MIB)):
        if status['phase']=='model_loading' and status.get(key) is None:continue
        if type(status.get(key)) is not int or not 0<=status[key]<=cap:return 'cuda_allocator_invalid_or_over_cap:'+key
    return None


def publish_status(handoff_path,payload):
    step='status.handoff.read';temporary=None
    try:
        handoff=_read(handoff_path);path=Path(handoff['status_path']);step='status.identity'
        if path!=Path(handoff_path).parent/'worker-status.json' or handoff['actual_worker_pid']!=os.getpid():
            raise RuntimeError('Status output/PID differs from owned handoff')
        step='status.encode'
        value=dict(payload,nonce=handoff['nonce'],actual_worker_pid=os.getpid(),at_unix=time.time(),
            at_monotonic=time.monotonic(),at_utc=_now());data=_json(value)
        if len(data)>SLOT:raise ValueError('Status exceeds its prepaid fixed slot')
        with _STATUS_LOCK:
            step='status.previous_temporary'
            if any(path.parent.glob(path.name+'.*.tmp')):raise RuntimeError('Previous status publication is incomplete')
            temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');step='status.temporary.open'
            with temporary.open('xb',buffering=0) as f:
                step='status.temporary.write';_write_all(f,data)
                step='status.temporary.fsync';os.fsync(f.fileno())
            step='status.atomic.publish'
            _publish_json(temporary,path,shared_status=True)
        return value
    except BaseException as error:
        if not getattr(error,'heartbeat_step',None):error.heartbeat_step=step
        if temporary is not None:error.heartbeat_temporary_path=str(temporary)
        # Keep failed/partial publication bytes; no retry follows a first error.
        raise
    finally:
        if temporary is not None and sys.exc_info()[1] is None:
            try:temporary.unlink()
            except FileNotFoundError:pass
            except BaseException as error:
                error.heartbeat_step='status.temporary.cleanup';error.heartbeat_temporary_path=str(temporary);raise


def heartbeat_failure(handoff_path):
    path=Path(handoff_path).parent/'heartbeat-error.json'
    if not path.exists():return None
    record=_read(path)
    if not record:return None
    if record.get('kind')!='gpc_heartbeat_failure.v1' or not record.get('step') or not record.get('exception_type'):
        raise RuntimeError('Heartbeat first-error slot is incomplete or invalid')
    return record


class Heartbeat:
    """Protocol-local one-second status writer; live CUDA refresh is caller-owned."""
    def __init__(self,path,payload=None,refresh=None,*,publisher=None):
        self.path,self.payload,self.refresh=path,dict(payload or status_template()),refresh
        self.publisher=publisher
        self.lock,self.stop,self.error=threading.RLock(),threading.Event(),None
        self.thread=threading.Thread(target=self._loop,daemon=True)
        self._failure_lock=threading.RLock();self._error_stream=None;self.first_error=None
    def record_error(self,error,*,step):
        with self._failure_lock:
            if self.error is not None:return self.first_error
            self.error=error
            code_error=error;seen=set()
            while id(code_error) not in seen and not any(getattr(code_error,k,None) is not None for k in ('errno','winerror')):
                seen.add(id(code_error));cause=code_error.__cause__ or code_error.__context__
                if cause is None:break
                code_error=cause
            record=dict(kind='gpc_heartbeat_failure.v1',step=getattr(error,'heartbeat_step',step),
                exception_type=type(error).__name__,message=str(error)[:2000],errno=getattr(code_error,'errno',None),
                winerror=getattr(code_error,'winerror',None),at_unix=time.time(),at_monotonic=time.monotonic(),at_utc=_now(),
                actual_worker_pid=os.getpid(),traceback=''.join(traceback.format_exception(error))[-8000:],
                temporary_path=getattr(error,'heartbeat_temporary_path',None),details=getattr(error,'details',None))
            if getattr(error,'journal_path',None) is not None:record['journal_path']=error.journal_path
            self.first_error=record
            # Existing bounded stdout is independent evidence if the disk slot fails.
            try:sys.stderr.write(_json(record).decode());sys.stderr.flush()
            except BaseException:pass
            try:
                payload=_json(record)
                if len(payload)>SLOT:raise ValueError('Heartbeat failure exceeds fixed slot')
                if self._error_stream is None:raise RuntimeError('Heartbeat emergency slot not prepared')
                self._error_stream.seek(0);_write_all(self._error_stream,payload+b' '*(SLOT-len(payload)));os.fsync(self._error_stream.fileno())
            except BaseException as persistence:
                try:sys.stderr.write(_json(dict(heartbeat_error_persistence_failed=type(persistence).__name__,first_error=record)).decode());sys.stderr.flush()
                except BaseException:pass
            return record
    def _publish(self):
        with self.lock:
            self.check()
            value=dict(self.payload)
            step='heartbeat.refresh'
            try:
                if self.refresh is not None:value.update(self.refresh())
                step='heartbeat.publish';return (self.publisher or publish_status)(self.path,value)
            except BaseException as error:self.record_error(error,step=step);raise
    def _loop(self):
        while not self.stop.wait(1):
            try:self._publish()
            except BaseException:return
    def update(self,**fields):
        self.check()
        with self.lock:self.payload.update(fields)
        return self._publish()
    def check(self):
        if self.error is not None:raise RuntimeError('Adaptation heartbeat failed') from self.error
    def __enter__(self):
        path=Path(self.path).parent/'heartbeat-error.json'
        self._error_stream=path.open('xb+',buffering=0)
        step='heartbeat.error_slot.prepare'
        try:
            _write_all(self._error_stream,b'{}'+b' '*(SLOT-2));os.fsync(self._error_stream.fileno())
            self._publish();step='heartbeat.thread.start';self.thread.start();return self
        except BaseException as error:
            self.record_error(error,step=step);self._error_stream.close();raise
    def __exit__(self,kind,value,tb):
        self.stop.set();self.thread.join(timeout=3)
        try:
            if kind is None:
                if self.thread.is_alive():
                    error=TimeoutError('Status writer did not quiesce');self.record_error(error,step='heartbeat.shutdown');raise error
                self.check()
        finally:
            if self._error_stream is not None:self._error_stream.close()


class StartupTrace:
    """One bounded, main-thread SDK stage journal; it is not a heartbeat."""
    def __init__(self,handoff_path):
        self.path=Path(handoff_path).parent/'sdk-startup.jsonl'
        self.stream=self.path.open('xb',buffering=0);self.bytes=0;self.sequence=0;self.current={}
        try:self.stage('startup_trace','ready')
        except BaseException:self.stream.close();raise
    def stage(self,name,state='before',detail=None):
        value=dict(sequence=self.sequence+1,stage=str(name),state=str(state),detail=detail,
            at_unix=time.time(),at_monotonic=time.monotonic(),at_utc=_now())
        payload=_json(value)
        if self.bytes+len(payload)>SLOT:raise ValueError('SDK stage journal exceeds64KiB')
        _write_all(self.stream,payload);os.fsync(self.stream.fileno())
        self.bytes+=len(payload);self.sequence+=1;self.current=value;return self.snapshot()
    def snapshot(self):
        value=self.current
        return dict(startup_stage=value.get('stage'),startup_stage_state=value.get('state'),
            startup_progress_unix=value.get('at_unix'),startup_progress_monotonic=value.get('at_monotonic'),
            startup_sequence=value.get('sequence'))
    def close(self):self.stream.close()
    def __enter__(self):return self
    def __exit__(self,kind,value,tb):
        try:self.close()
        except BaseException as error:
            if value is None:raise
            value.add_note('Startup trace close failed: '+str(error))


def initialize_sdk(heartbeat,trace,*,cpu_threads=2,require_cuda=True,importer=None):
    """Lazy SDK startup only; no weights, network, tensor work or warmup."""
    if cpu_threads!=2:raise ValueError('The original CPU thread setting is2')
    importer=importer or importlib.import_module
    current='sdk_startup';result={}
    def invoke(name,call):
        nonlocal current
        current=name;trace.stage(name,'before');heartbeat.update(**trace.snapshot())
        value=call();trace.stage(name,'after',value if isinstance(value,(bool,int,float,str)) else None)
        heartbeat.update(**trace.snapshot());return value
    try:
        torch=invoke('torch_import',lambda:importer('torch'))
        invoke('set_num_threads',lambda:torch.set_num_threads(cpu_threads))
        available=invoke('cuda_available',torch.cuda.is_available);result['cuda_available']=bool(available)
        if not available and require_cuda:raise RuntimeError('CUDA unavailable for the fixed SDK startup')
        if available:
            invoke('cuda_init',torch.cuda.init)
            invoke('cuda_reset_peak',torch.cuda.reset_peak_memory_stats)
            sample=invoke('allocator_sample',lambda:cuda_allocator_sample(torch))
            for key,cap in (('cuda_allocated_bytes',512*MIB),('cuda_peak_allocated_bytes',512*MIB),
                            ('cuda_reserved_bytes',1024*MIB),('cuda_peak_reserved_bytes',1024*MIB)):
                if type(sample[key]) is not int or not 0<=sample[key]<=cap:raise MemoryError('Original CUDA ceiling exceeded: '+key)
            heartbeat.update(**sample);result.update(sample)
        else:result.update(cuda_unknown())
        trace.stage('sdk_initialized','after',dict(cuda_available=bool(available)))
        heartbeat.update(**trace.snapshot(),sdk_complete=True)
        return torch,dict(result,sdk_complete=True)
    except BaseException as error:
        try:trace.stage(current,'error',dict(exception_type=type(error).__name__,message=str(error)[:1000]))
        except BaseException as secondary:error.add_note('Stage error persistence failed: '+str(secondary))
        heartbeat.record_error(error,step='sdk.'+current);raise


def _latest(path,previous):
    try:return _read(path)
    except BaseException as error:
        cause=error.__cause__ if hasattr(error,'details') else error;details=getattr(error,'details',{})
        exact=(details.get('path')==str(path) and details.get('phase')=='guardian.json_read.open') if details else getattr(error,'filename',None)==str(path)
        if exact and (isinstance(cause,FileNotFoundError) or isinstance(cause,OSError) and getattr(cause,'winerror',None) in (32,33)):return previous
        raise


def verify_worker(path,*,native_check=True):
    path=_path(path);value=_read(path);pid=os.getpid();directory=path.parent
    if (os.environ.get(ENV_HANDOFF)!=str(path) or os.environ.get(ENV_NONCE)!=value.get('nonce')
            or os.environ.get(ENV_PARENT)!=str(value.get('guard_pid')) or value.get('actual_worker_pid')!=pid
            or value.get('prepared_ready') is not True or value.get('ownership')!='native_job_before_ack'
            or value.get('job_memory_bytes')!=3*1024**3 or value.get('kill_on_job_close') is not True):
        raise RuntimeError('No matching native adaptation guardian handoff')
    config=_path(value['config_path']);config_bytes=config.read_bytes()
    if (hashlib.sha256(config_bytes).hexdigest()!=value['binding']['config_sha256']
            or protocol.sha_file(value['checkpoint'])!=value['checkpoint_sha256']
            or protocol.sha_file(value['script_path'])!=value['script_sha256']):
        raise RuntimeError('Config/source/checkpoint changed after ownership gate')
    identity=protocol.execution_identity(config,json.loads(config_bytes),value['binding'])
    if (value['output_dir']!=str(_path(ROOT/identity['output_dir'])) or directory!=Path(value['output_dir'])/'guardian'
            or value['status_path']!=str(directory/'worker-status.json') or value['ack_path']!=str(directory/'worker-ack.json')
            or _read(directory/'worker.json')!=dict(pid=pid,parent_pid=os.getppid(),nonce=value['nonce'])
            or _read(directory/'worker-ack.json')!=dict(pid=pid,nonce=value['nonce'])):
        raise RuntimeError('Worker PID/nonce/ACK/fixed output differs')
    if _clocks(value['binding'],time.time(),time.monotonic())!=value['binding']:raise RuntimeError('Original handoff deadlines differ')
    if native_check:
        from . import policy_memory
        kernel,_=policy_memory._libraries();inside=wintypes.BOOL()
        kernel.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)];kernel.IsProcessInJob.restype=wintypes.BOOL
        kernel.GetCurrentProcess.argtypes=[];kernel.GetCurrentProcess.restype=wintypes.HANDLE
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(inside)) or not inside.value:raise RuntimeError('Worker is outside native Job')
        kernel.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
        kernel.QueryInformationJobObject.restype=wintypes.BOOL;limits=policy_memory._JobExtendedLimits()
        if (not kernel.QueryInformationJobObject(None,9,ctypes.byref(limits),ctypes.sizeof(limits),None)
                or limits.JobMemoryLimit!=3*1024**3 or limits.BasicLimitInformation.LimitFlags&0x2200!=0x2200):raise RuntimeError('Native Job limits differ')
    return value


class _Telemetry:
    def __init__(self,directory,*,journal_reserve_bytes=0):
        if journal_reserve_bytes not in (0,16*MIB):raise ValueError('Only the fixed eval journal reserve is supported')
        self.journal_reserve_bytes=journal_reserve_bytes
        self.lock=threading.RLock();self.monitor_bytes=self.stdout_bytes=self.sequence=0
        self.stream=(directory/'monitor.jsonl').open('xb',buffering=0);self.slot=(directory/'stop-record.json').open('xb+',buffering=0)
        _write_all(self.slot,b'{}'+b' '*(SLOT-2));os.fsync(self.slot.fileno())
        self.emergency=os.fdopen(os.dup(self.slot.fileno()),'r+b',buffering=0)
    def admit(self,kind,data):
        current=self.monitor_bytes if kind=='monitor' else self.stdout_bytes
        limit=(min(MONITOR_CAP,29*MIB) if self.journal_reserve_bytes else MONITOR_CAP) if kind=='monitor' else STDOUT_CAP
        if current+len(data)>limit or self.monitor_bytes+self.stdout_bytes+len(data)+LEDGER_RESERVE+FIXED_RESERVE+self.journal_reserve_bytes>TELEMETRY_CAP:
            raise OSError('Adaptation telemetry cap exceeded before write')
        if kind=='monitor':self.monitor_bytes+=len(data)
        else:self.stdout_bytes+=len(data)
    def event(self,kind,detail):
        # The parent alone owns this stream. Admission is shared with stdout;
        # file IO is outside that brief counter lock and is never refunded.
        with self.lock:
            data=_json(dict(sequence=self.sequence+1,at_utc=_now(),kind=kind,detail=detail))
            self.admit('monitor',data);self.sequence+=1
        _write_all(self.stream,data)
    def write_stop(self,report,stream=None):
        data=_json(report)
        if len(data)>SLOT:raise OSError('Stop report exceeds fixed slot')
        target=stream or self.slot;target.seek(0);_write_all(target,data+b' '*(SLOT-len(data)));os.fsync(target.fileno())
    def close(self):
        first=None
        for f in (self.stream,self.slot):
            for action in (lambda:os.fsync(f.fileno()),f.close):
                try:action()
                except BaseException as error:
                    if first is None:first=error
                    else:first.add_note('Secondary close error: '+str(error))
        if first:raise first


class _Pump:
    def __init__(self,source,path,telemetry):
        self.source,self.telemetry=source,telemetry;self.error=None;self.stream=path.open('xb',buffering=0)
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _run(self):
        try:
            while True:
                data=self.source.read(8192)
                if not data:return
                if not isinstance(data,bytes) or len(data)>8192:raise OSError('Invalid bounded stdout PIPE read')
                with self.telemetry.lock:self.telemetry.admit('stdout',data)
                _write_all(self.stream,data)
        except BaseException as error:self.error=error
    def check(self):
        if self.error is not None:raise RuntimeError('Worker stdout failed: '+str(self.error)) from self.error
    def close(self, timeout=3):
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():raise TimeoutError('Worker PIPE remains active after cleanup')
        for f in (self.stream,self.source):
            actions=([lambda:os.fsync(f.fileno())] if f is self.stream else [])+[f.close]
            for action in actions:
                try:action()
                except BaseException as error:
                    if self.error is None:self.error=error
                    else:self.error.add_note('Secondary stdout close error: '+str(error))
        self.check()


def supervise(prepared,binding,*,python_executable,gpc_root,checkpoint,script_path,config_path,
              provider=None,owner_factory=None,launcher_factory=None,eval_only=False,conditioning_only=False,raw_fulltrain=False):
    """Launch once after readonly authority checks; injectable fakes never query Windows.

    The fixed output directory is the permanent attempt claim. Native startup,
    audits, model loading, training, evaluation and cleanup share the same clocks.
    A failed or uncertain cleanup never grants another launch.
    """
    eval_only=eval_only or conditioning_only
    journal_mode=eval_only or raw_fulltrain
    wall,mono=time.time(),time.monotonic();bound=_clocks(binding,wall,mono,eval_only=eval_only,conditioning_only=conditioning_only,raw_fulltrain=raw_fulltrain)
    preclaim=_preclaim
    if raw_fulltrain:
        from .gpc_raw_fulltrain_entry import preclaim
    elif conditioning_only:
        from .gpc_conditioning_entry import preclaim
    elif eval_only:
        from .gpc_evaluation_entry import preclaim
    output,python,gpc_root,checkpoint,script,config,assets=preclaim(
        prepared,bound,python_executable,gpc_root,checkpoint,script_path,config_path)
    if (time.time()>=bound['deadline_unix'] or time.monotonic()>=bound['deadline_monotonic']
            or raw_fulltrain and (time.time()>=bound['deadline_unix']-20 or time.monotonic()>=bound['deadline_monotonic']-20)
            or not eval_only and (time.time()>=bound['training_deadline_unix'] or time.monotonic()>=bound['training_deadline_monotonic'])):
        raise TimeoutError('Original training deadline elapsed during readonly checks')
    bound.update(assets);nonce=uuid.uuid4().hex;directory=output/'guardian'
    output.mkdir(parents=True,exist_ok=False);directory.mkdir()
    _exclusive(output/'attempt.json',dict(binding=bound,nonce=nonce,guard_pid=os.getpid(),created_utc=_now(),resume_allowed=False))
    telemetry=_Telemetry(directory,journal_reserve_bytes=16*MIB) if journal_mode else _Telemetry(directory)
    injected=dict(resource_provider=provider is not None,process_owner=owner_factory is not None,launcher=launcher_factory is not None)
    provider=provider if provider is not None else ResourceProvider()
    owner_factory=owner_factory or WindowsWorker;launcher_factory=launcher_factory or subprocess.Popen
    child=owner=pump=None;lock_owned=False;reason=error_text=None;cleanup=[];certain=True
    last_status=last_resource=startup=None;acknowledged=None;job_active=None;worker_exit=launcher_exit=None
    samples=0;minima={};peaks={};missing_status=0;actual_worker_pid=None
    journal_reader=None

    def latest_status(path,previous):
        nonlocal journal_reader
        if not journal_mode:return _latest(path,previous)
        from .gpc_evaluation_status import Reader
        if raw_fulltrain:
            from .gpc_raw_fulltrain_entry import validate_status_record
        elif conditioning_only:
            from .gpc_conditioning_entry import validate_status_record
        else:
            from .gpc_evaluation_entry import validate_status_record
        from .wm_compat_storage import unlinked_path
        if journal_reader is None:
            path=unlinked_path(path)
            if not path.exists():return previous
            journal_reader=Reader(path,pid=actual_worker_pid,nonce=nonce,validate=validate_status_record,
                **({'raw_fulltrain':True} if raw_fulltrain else {}))
        # Drain only new bytes, in bounded chunks, including the final tail.
        for _ in range(257):
            journal_reader.poll()
            if journal_reader.read_offset>=journal_reader.bytes_seen:break
            if time.monotonic()>=bound['deadline_monotonic']:raise TimeoutError('Status drain exceeded original deadline')
        else:raise RuntimeError('Status journal exceeded its bounded incremental drain')
        return journal_reader.latest

    def fail(code,error=None):
        nonlocal reason,error_text
        if reason is None:
            reason=code
            if error is not None:error_text=type(error).__name__+': '+str(error)

    def deadline():
        if time.time()>=bound['deadline_unix'] or time.monotonic()>=bound['deadline_monotonic']:return 'total_deadline_exceeded'
        if not eval_only and (last_status is None or not last_status.get('training_complete',False)) and (
                time.time()>=bound['training_deadline_unix'] or time.monotonic()>=bound['training_deadline_monotonic']):return 'training_phase_deadline_exceeded'
        if raw_fulltrain and (time.time()>=bound['deadline_unix']-20 or time.monotonic()>=bound['deadline_monotonic']-20):return 'cleanup_reserve_reached'
        return None

    def resources(snapshot,start=False):
        problem=resource_reason(snapshot,startup=start)
        if problem is None and snapshot['disk_free_bytes']<2*1024**3:problem='resource_below_floor:disk_free_bytes'
        return problem

    def observe(snapshot):
        for key in ('available_ram_bytes','commit_headroom_bytes','disk_free_bytes','gpu_free_bytes'):
            value=snapshot.get(key)
            if type(value) is int:minima[key]=min(minima.get(key,value),value)
        for key in ('working_set_bytes','private_bytes','process_peak_working_set_bytes'):
            value=snapshot.get(key)
            if _finite(value):peaks[key]=max(peaks.get(key,value),value)
        if last_status:
            for key in ('cuda_allocated_bytes','cuda_reserved_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes'):
                value=last_status.get(key)
                if type(value) is int:peaks[key]=max(peaks.get(key,value),value)

    try:
        _exclusive(DEFAULT_LOCK,dict(nonce=nonce,guard_pid=os.getpid(),output_dir=str(output)))
        lock_owned=True
        startup=provider.startup(output);observe(startup);telemetry.event('startup',startup)
        problem=resources(startup,True)
        if problem:fail(problem)
        if reason is None:
            handoff_path=directory/'handoff.json';status_path=directory/('evaluation-status.journal' if journal_mode else 'worker-status.json')
            command=[str(python),'-B',str(ROOT/'scripts/guarded_c_child.py'),'--handshake',str(directory/'worker.json'),
                '--ack',str(directory/'worker-ack.json'),'--nonce',nonce,'--',str(script),'--worker',str(handoff_path)]
            env=dict(os.environ,**{ENV_HANDOFF:str(handoff_path),ENV_NONCE:nonce,ENV_PARENT:str(os.getpid())})
            env.update({key:'2' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
            env.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
            child=launcher_factory(command,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
            pump=_Pump(child.stdout,directory/'worker-output.log',telemetry)
            until=min(bound['deadline_monotonic'],time.monotonic()+10)
            while not (directory/'worker.json').exists():
                pump.check()
                if child.poll() is not None:raise RuntimeError('Launcher exited before worker ownership')
                if time.monotonic()>=until:raise TimeoutError('Actual worker handshake missing')
                if deadline():raise TimeoutError(deadline())
                time.sleep(.02)
            handshake=_read(directory/'worker.json')
            if handshake.get('nonce')!=nonce or type(handshake.get('pid')) is not int or handshake['pid']<=0:
                raise RuntimeError('Actual worker handshake PID/nonce invalid')
            owner=owner_factory(handshake['pid'],child.pid)
            actual_worker_pid=handshake['pid']
            handoff=dict(kind=('gpc_raw_fulltrain_guard_handoff.v1' if raw_fulltrain else 'gpc_conditioning_guard_handoff.v1' if conditioning_only else
                'gpc_evaluation_guard_handoff.v1' if eval_only else 'gpc_adaptation_guard_handoff.v1'),prepared_ready=True,config_path=str(config),binding=bound,
                gpc_root=str(gpc_root),checkpoint=str(checkpoint),output_dir=str(output),actual_worker_pid=handshake['pid'],
                launcher_pid=child.pid,guard_pid=os.getpid(),nonce=nonce,ownership='native_job_before_ack',
                job_memory_bytes=3*1024**3,kill_on_job_close=True,ack_path=str(directory/'worker-ack.json'),
                status_path=str(status_path),script_path=str(script),script_sha256=protocol.sha_file(script),**assets)
            last_resource=provider.runtime(handshake['pid'],output);observe(last_resource)
            problem=resources(last_resource) or deadline()
            if problem:fail(problem)
            else:
                _exclusive(handoff_path,handoff)
                _exclusive(directory/'worker-ack.json',dict(pid=handshake['pid'],nonce=nonce))
                acknowledged=time.monotonic();provider.start();telemetry.event('owned_worker',handoff)
            observed_ledger=set()
            while reason is None and not owner.poll_exit():
                pump.check();problem=deadline()
                if problem:fail(problem);break
                last_resource=provider.runtime(handshake['pid'],output)
                problem=resources(last_resource)
                if problem:
                    if owner.poll_exit() and problem in ('missing_or_invalid_resource:working_set_bytes','missing_or_invalid_resource:private_bytes'):break
                    fail(problem);break
                first_heartbeat_error=heartbeat_failure(handoff_path)
                if first_heartbeat_error is not None:
                    telemetry.event('heartbeat_first_error',first_heartbeat_error)
                    fail('worker_heartbeat_failure:'+first_heartbeat_error['step']+':'+first_heartbeat_error['exception_type']);break
                last_status=latest_status(status_path,last_status)
                if last_status is None:
                    missing_status+=1
                    if time.monotonic()-acknowledged>5:fail('worker_status_initial_missing');break
                else:
                    problem=monitor_reason(last_status,bound,pid=handshake['pid'],nonce=nonce,eval_only=eval_only,conditioning_only=conditioning_only,raw_fulltrain=raw_fulltrain)
                    if problem:fail(problem);break
                ledger_bytes=0
                for name in ('attempt.json','highwater.jsonl','first-failure.json'):
                    path=output/'ledger'/name
                    try:
                        path=_path(path);stat=path.stat()
                        if not path.is_file() or stat.st_nlink!=1:raise OSError('Ledger telemetry is not an ordinary unique file')
                        ledger_bytes+=stat.st_size;observed_ledger.add(name)
                    except FileNotFoundError:
                        if name in observed_ledger:raise
                if ledger_bytes>LEDGER_RESERVE:raise OSError('Worker ledger exceeds prepaid telemetry cap')
                observe(last_resource);samples+=1
                telemetry.event('sample',dict(resources=last_resource,status=last_status,ledger_bytes=ledger_bytes))
                time.sleep(.2)
            if reason is None:
                first_heartbeat_error=heartbeat_failure(handoff_path)
                if first_heartbeat_error is not None:fail('worker_heartbeat_failure:'+first_heartbeat_error['step']+':'+first_heartbeat_error['exception_type'])
                last_status=latest_status(status_path,None)
                if journal_mode and (journal_reader is None or journal_reader.partial_bytes
                        or journal_reader.committed_bytes!=journal_reader.bytes_seen):
                    fail('worker_status_incomplete_tail')
                problem=monitor_reason(last_status,bound,pid=handshake['pid'],nonce=nonce,eval_only=eval_only,conditioning_only=conditioning_only,raw_fulltrain=raw_fulltrain)
                if problem:fail(problem)
                elif owner.exit_code()!=0:fail('worker_nonzero_exit')
                elif last_status['phase']!='complete':fail('worker_incomplete_exit')
                elif any(last_status[key]!=cap for key,cap in dict(optimizer_updates=2328 if raw_fulltrain else 0 if eval_only else 500,
                        model_loads=4 if raw_fulltrain else 2 if eval_only else 3,
                        forecasts=48 if raw_fulltrain else 32 if conditioning_only else 3996,
                        euler_steps=144 if raw_fulltrain else 96 if conditioning_only else 11988,
                        eval_batches=48 if raw_fulltrain else 0 if conditioning_only else 32,
                        eval_examples=48 if raw_fulltrain else 0 if conditioning_only else 32).items()):
                    fail('worker_incomplete_counts')
                elif raw_fulltrain and last_status['training_complete'] is not True:
                    fail('worker_training_incomplete')
                elif (conditioning_only or raw_fulltrain) and last_status['model_instances']!=0:
                    fail('worker_model_not_released')
                else:
                    completion=_read(output/'completion.json')
                    if completion.get('status')!='complete' or completion.get('forecast_records')!=(48 if raw_fulltrain else 32 if conditioning_only else 3996):fail('completion_record_incomplete')
                    elif raw_fulltrain and any(completion.get(key)!=cap for key,cap in
                            dict(optimizer_updates=2328,model_loads=4,eval_loss_batches=48).items()):fail('completion_record_incomplete')
                observe(last_resource)
    except BaseException as error:
        fail('guardian_io_or_worker_failure',error)
    finally:
        if journal_reader is not None:
            try:journal_reader.close()
            except BaseException as error:cleanup.append(type(error).__name__+': '+str(error));fail('status_reader_close_failed',error)
        if owner is not None:
            try:
                if not owner.poll_exit():owner.terminate_and_wait(timeout=_cleanup_wait_timeout(bound,raw_fulltrain=raw_fulltrain))
                worker_exit=owner.exit_code();job_active=owner.quiesce(timeout=_cleanup_wait_timeout(bound,raw_fulltrain=raw_fulltrain))
                if job_active!=0:raise RuntimeError('Owned Job still has active descendants')
            except BaseException as error:certain=False;cleanup.append(type(error).__name__+': '+str(error));fail('job_cleanup_uncertain',error)
            try:owner.close()
            except BaseException as error:certain=False;cleanup.append(type(error).__name__+': '+str(error));fail('job_close_failed',error)
        elif child is not None:
            certain=False
        if child is not None:
            try:launcher_exit=child.wait(timeout=_cleanup_wait_timeout(bound,raw_fulltrain=raw_fulltrain))
            except BaseException:
                try:child.terminate();launcher_exit=child.wait(timeout=_cleanup_wait_timeout(bound,raw_fulltrain=raw_fulltrain))
                except BaseException as error:certain=False;cleanup.append(type(error).__name__+': '+str(error));fail('launcher_cleanup_uncertain',error)
        if pump is not None:
            try:
                if raw_fulltrain:pump.close(timeout=_cleanup_wait_timeout(bound,raw_fulltrain=True,maximum=3.,reserve=5.))
                else:pump.close()
            except BaseException as error:certain=False;cleanup.append(type(error).__name__+': '+str(error));fail('stdout_failure',error)
        try:
            if raw_fulltrain and _cleanup_wait_timeout(bound,raw_fulltrain=True,maximum=4.,reserve=1.)<4.:
                raise TimeoutError('Original deadline leaves no4s resource-reader close allowance')
            provider.close()
        except BaseException as error:
            if raw_fulltrain:certain=False
            cleanup.append(type(error).__name__+': '+str(error));fail('resource_reader_close_failed',error)
        if owner is not None and certain:
            try:
                if checkpoint.stat().st_size!=assets['checkpoint_bytes'] or protocol.sha_file(checkpoint)!=assets['checkpoint_sha256']:
                    raise ValueError('Pinned initialization checkpoint changed during candidate')
            except BaseException as error:fail('initial_checkpoint_recheck_failed',error)
        if lock_owned and certain:
            try:
                if _read(DEFAULT_LOCK).get('nonce')!=nonce:raise RuntimeError('Global model lock owner changed')
                DEFAULT_LOCK.unlink();lock_owned=False
            except BaseException as error:cleanup.append(type(error).__name__+': '+str(error));fail('single_model_unlock_failed',error)
        if time.time()>=bound['deadline_unix'] or time.monotonic()>=bound['deadline_monotonic']:fail('total_deadline_exceeded')
        report=dict(kind='gpc_adaptation_guard_report.v1',status='complete' if reason is None else 'stopped',guard_reason=reason,
            error=error_text,binding=bound,output_dir=str(output),guardian_dir=str(directory),nonce=nonce,
            guard_pid=os.getpid(),launcher_pid=child.pid if child else None,actual_worker_pid=actual_worker_pid,
            injected=injected,native_integration_verified=reason is None and not any(injected.values()),job_memory_bytes=3*1024**3,
            job_active_processes_after_cleanup=job_active,worker_exit_code=worker_exit,launcher_exit_code=launcher_exit,
            single_model_lock_retained=lock_owned,cleanup_errors=cleanup,started_utc=datetime.fromtimestamp(wall,timezone.utc).isoformat(),
            ended_utc=_now(),wall_seconds=time.monotonic()-mono,samples=samples,initial_unavailable_status_samples=missing_status,
            startup=startup,last_resources=last_resource,last_status=last_status,min_free=minima,observed_peaks=peaks,
            telemetry=dict(monitor_bytes=telemetry.monitor_bytes,stdout_bytes=telemetry.stdout_bytes,
                ledger_reserve_bytes=LEDGER_RESERVE,fixed_reserve_bytes=FIXED_RESERVE,total_cap_bytes=TELEMETRY_CAP))
        report['telemetry']['status_journal_reserve_bytes']=telemetry.journal_reserve_bytes
        if raw_fulltrain:report.update(raw_fulltrain=True,cleanup_reserve_seconds=20)
        def final_timing():
            if time.time()>=bound['deadline_unix'] or time.monotonic()>=bound['deadline_monotonic']:fail('total_deadline_exceeded')
            report.update(status='complete' if reason is None else 'stopped',guard_reason=reason,error=error_text,
                native_integration_verified=reason is None and not any(injected.values()),
                wall_seconds=time.monotonic()-mono,ended_utc=_now())

        try:telemetry.event('final',dict(status=report['status'],guard_reason=reason));telemetry.write_stop(report)
        except BaseException as error:fail('final_telemetry_io_failure',error)
        try:telemetry.close()
        except BaseException as error:fail('telemetry_close_failure',error)
        final_timing()
        try:
            previous_reason=reason;telemetry.write_stop(report,telemetry.emergency);final_timing()
            if reason!=previous_reason:telemetry.write_stop(report,telemetry.emergency)
        except BaseException as error:
            fail('stop_record_io_failure',error)
        try:telemetry.emergency.close()
        except BaseException as error:
            fail('stop_record_close_failure',error)
        previous_reason=report['guard_reason'];final_timing()
        if reason!=previous_reason:
            # A final fsync/close can itself cross the original deadline. Rewrite
            # only the already allocated slot once, with no new file or budget.
            try:
                path=_path(directory/'stop-record.json');stat=path.stat()
                if not path.is_file() or stat.st_nlink!=1 or stat.st_size!=SLOT:raise OSError('Fixed stop slot changed')
                with path.open('r+b',buffering=0) as stream:telemetry.write_stop(report,stream)
            except BaseException as error:fail('stop_record_finalization_failure',error)
        final_timing()
    return report
