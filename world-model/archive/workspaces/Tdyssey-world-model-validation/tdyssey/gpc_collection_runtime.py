"""Fixed A/B/B2 collection accounts and a model-free Windows launch gate.

Importing is read-only. All operation charges are durable before simulator work;
failed operations/writes keep their charges. There is no recovery, retry, resume,
payload readback, GPU query or general storage service in this module.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .gpc_adaptation_guard import _path, _latest, Heartbeat, publish_status
from .wm_compat_guard import WindowsWorker, _read, _publish_json, _write_all

ROOT=Path(__file__).resolve().parents[1]
MIB=1024**2;GIB=1024**3;SLOT=65536
GUARD_BYTES=2*MIB
STAGES={
    'A':dict(output='gpc-data-stage-a-20261026',seconds=480,bytes=128*MIB,run_seconds=120,run_bytes=32*MIB,
        run_ids=['T-A','T-B','L-A','L-B'],caps=dict(controls=3200,resets=640,control_substeps=32000,reset_substeps=640)),
    'B':dict(output='gpc-data-stage-b-20261026',seconds=600,bytes=64*MIB,run_seconds=600,run_bytes=64*MIB,
        run_ids=['B'],caps=dict(controls=256,resets=256,restores=64,renders=512,control_substeps=2560,reset_substeps=256)),
    'B2':dict(output='gpc-data-stage-b2-20261026',seconds=600,bytes=64*MIB,run_seconds=600,run_bytes=64*MIB,
        run_ids=['B'],caps=dict(controls=256,resets=256,restores=64,renders=512,control_substeps=2560,reset_substeps=256))}
COSTS=('controls','resets','restores','renders','control_substeps','reset_substeps')
REGISTRY_NAME='gpc-data-stage-20261026-registry'


def _json(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def _sha(value):return hashlib.sha256(value).hexdigest()
def _finite(value):return type(value) in (float,int) and math.isfinite(value)
def _utc():return datetime.now(timezone.utc).isoformat()


def _exclusive(path,payload):
    if len(payload)>SLOT:raise ValueError('Fixed collection metadata exceeds 64KiB')
    with path.open('xb',buffering=0) as stream:_write_all(stream,payload);os.fsync(stream.fileno())


def stage_paths(stage):
    if stage not in STAGES:raise ValueError('Only fixed collection stages A, B and B2 exist')
    return _path(ROOT/'artifacts'/STAGES[stage]['output']),_path(ROOT/'artifacts'/REGISTRY_NAME/(stage+'.json'))


def claim_stage(stage,binding,*,clock=time.time,monotonic=time.monotonic):
    """Permanent stage singleton; source/config/output changes cannot replenish it."""
    output,registry=stage_paths(stage);limits=deepcopy(STAGES[stage]);binding=deepcopy(binding)
    if not isinstance(binding,dict) or len(_json(binding))>16384:raise ValueError('Bounded explicit collection binding required')
    if output.exists() or registry.exists():raise FileExistsError('Collection stage already attempted; no retry or alternate account')
    wall,mono=clock(),monotonic()
    if not _finite(wall) or not _finite(mono):raise ValueError('Finite collection clocks required')
    claim=dict(kind='gpc_collection_attempt.v1',stage=stage,binding=binding,output_dir=str(output),registry_path=str(registry),
        nonce=uuid.uuid4().hex,limits=limits,started_unix=wall,started_monotonic=mono,
        deadline_unix=wall+limits['seconds'],deadline_monotonic=mono+limits['seconds'],resume_allowed=False)
    payload=_json(claim);registry.parent.mkdir(parents=True,exist_ok=True);_exclusive(registry,payload)
    output.mkdir(parents=True,exist_ok=False)
    return dict(claim,path=str(registry),sha256=_sha(payload))


def _checked_claim(stage,binding,claim):
    output,registry=stage_paths(stage);raw=registry.read_bytes();stored=json.loads(raw)
    expected=dict(stored,path=str(registry),sha256=_sha(raw))
    if (claim!=expected or stored.get('stage')!=stage or stored.get('binding')!=binding
            or stored.get('output_dir')!=str(output) or stored.get('limits')!=STAGES[stage]
            or stored.get('resume_allowed') is not False):raise ValueError('Original collection claim/identity differs')
    return output,expected


class CollectionError(RuntimeError):pass


class CollectionBudget:
    """Single worker's append-only accounting; parent telemetry is prepaid once.

    create(..., claim=handoff['claim']) attaches only before any operation journal
    exists. It is not recovery: worker-account.json and the journal use xb.
    """
    @classmethod
    def create(cls,stage,binding,*,claim=None,clock=time.time,monotonic=time.monotonic):
        claim=claim_stage(stage,binding,clock=clock,monotonic=monotonic) if claim is None else claim
        output,claim=_checked_claim(stage,binding,claim)
        self=cls.__new__(cls);self.output,self.claim,self.stage=output,claim,stage
        self.clock,self.monotonic=clock,monotonic;self.limits=deepcopy(STAGES[stage]);self.lock=threading.RLock()
        self.counts=dict.fromkeys(COSTS,0);self.completed=dict.fromkeys(COSTS,0)
        self.runs={};self.active_run=None;self.episode=None;self.pending=None;self.files={}
        self.charged_bytes=GUARD_BYTES+3*SLOT;self.sequence=0;self.tip='0'*64;self.first_reason=None;self.closed=False
        self.last_wall,self.last_mono=claim['started_unix'],claim['started_monotonic']
        self._failure=output/'failure.json';self._journal=output/'journal.jsonl'
        _exclusive(output/'worker-account.json',_json(dict(kind='gpc_collection_worker.v1',claim_sha256=claim['sha256'],pid=os.getpid())))
        with self._failure.open('xb',buffering=0) as stream:_write_all(stream,b'{}'+b' '*(SLOT-2));os.fsync(stream.fileno())
        self.stream=self._journal.open('xb',buffering=0)
        self._append('created',dict(claim_sha256=claim['sha256'],prepaid_bytes=self.charged_bytes))
        return self

    def snapshot(self):
        with self.lock:
            return deepcopy(dict(stage=self.stage,status='complete' if self.closed and self.first_reason is None else 'stopped' if self.first_reason else 'active',
                counts=self.counts,completed=self.completed,runs=self.runs,active_run=self.active_run,pending=self.pending,
                physics_substeps=self.counts['control_substeps']+self.counts['reset_substeps'],
                charged_bytes=self.charged_bytes,first_reason=self.first_reason,sequence=self.sequence,journal_tip=self.tip,
                deadline_unix=self.claim['deadline_unix'],deadline_monotonic=self.claim['deadline_monotonic']))

    def fail(self,reason):
        with self.lock:
            if self.first_reason is None:self.first_reason=str(reason)
            payload=_json(dict(reason=self.first_reason,state=self.snapshot()))
            if len(payload)>SLOT:payload=_json(dict(reason=self.first_reason[:1000],sequence=self.sequence,charged_bytes=self.charged_bytes))
            try:
                with self._failure.open('r+b',buffering=0) as stream:
                    _write_all(stream,payload+b' '*(SLOT-len(payload)));os.fsync(stream.fileno())
            except BaseException:pass
            return self.first_reason

    def check(self):
        with self.lock:
            if self.first_reason or self.closed:raise CollectionError(self.first_reason or 'Collection account is closed')
            wall,mono=self.clock(),self.monotonic()
            problem=('Collection clock invalid/backwards' if not _finite(wall) or not _finite(mono) or wall<self.last_wall or mono<self.last_mono else
                'Aggregate collection deadline includes publication' if wall>=self.claim['deadline_unix'] or mono>=self.claim['deadline_monotonic'] else None)
            if self.active_run:
                run=self.runs[self.active_run]
                if wall>=run['deadline_unix'] or mono>=run['deadline_monotonic']:problem=problem or 'Per-run collection deadline includes publication'
            if problem:self.fail(problem);raise CollectionError(problem)
            self.last_wall,self.last_mono=wall,mono
            return self.snapshot()

    def _admit(self,size):
        if type(size) is not int or size<0:raise ValueError('Nonnegative byte cost required')
        run=self.runs.get(self.active_run)
        if self.charged_bytes+size>self.limits['bytes'] or run and run['charged_bytes']+size>self.limits['run_bytes']:
            self.fail('Collection byte cap includes journal, temporary overlap and publication');raise CollectionError(self.first_reason)
        self.charged_bytes+=size
        if run:run['charged_bytes']+=size

    def _append(self,kind,data):
        try:payload=_json(dict(sequence=self.sequence+1,kind=kind,run=self.active_run,data=data,previous_sha256=self.tip))
        except BaseException as error:self.fail(type(error).__name__+': '+str(error));raise
        if len(payload)>SLOT:
            self.fail('Collection event exceeds 64KiB');raise CollectionError(self.first_reason)
        self._admit(len(payload))
        try:_write_all(self.stream,payload);os.fsync(self.stream.fileno())
        except BaseException as error:self.fail(type(error).__name__+': '+str(error));raise
        self.sequence+=1;self.tip=_sha(payload)

    def event(self,kind,detail=None):
        with self.lock:self.check();self._append(str(kind),detail or {});self.check()

    def begin_run(self,run_id):
        with self.lock:
            self.check()
            if self.active_run or run_id in self.runs or run_id not in self.limits['run_ids']:
                self.fail('Run is repeated, unknown or concurrent');raise CollectionError(self.first_reason)
            wall,mono=self.clock(),self.monotonic();self.active_run=run_id
            self.runs[run_id]=dict(status='active',counts=dict.fromkeys(COSTS,0),episodes=0,charged_bytes=0,
                deadline_unix=min(self.claim['deadline_unix'],wall+self.limits['run_seconds']),
                deadline_monotonic=min(self.claim['deadline_monotonic'],mono+self.limits['run_seconds']))
            self._append('begin_run',dict(run_id=run_id));return self.snapshot()

    def begin_episode(self,index):
        with self.lock:
            self.check();run=self.runs.get(self.active_run)
            if self.stage!='A' or self.pending or self.episode or run is None or type(index) is not int or index!=run['episodes'] or index>=8:
                self.fail('Only eight sequential episodes per A run');raise CollectionError(self.first_reason)
            run['episodes']+=1;self.episode=dict(index=index,controls=0,resets=0)
            self._append('begin_episode',dict(index=index))

    def finish_episode(self):
        with self.lock:
            self.check()
            if self.pending or self.episode is None:raise CollectionError('No completed episode boundary')
            self._append('finish_episode',self.episode);self.episode=None

    def reserve(self,**costs):
        with self.lock:
            self.check()
            if self.pending or self.active_run is None or not costs or set(costs)-set(COSTS):
                self.fail('Single explicit operation reservation required');raise CollectionError(self.first_reason)
            cost=dict.fromkeys(COSTS,0);cost.update(costs)
            if any(type(v) is not int or v<0 for v in cost.values()) or not any(cost.values()):
                self.fail('Invalid collection operation cost');raise CollectionError(self.first_reason)
            if cost['control_substeps']!=10*cost['controls'] or cost['reset_substeps']!=cost['resets']:
                self.fail('Controls require ten physics substeps; simulated resets require one');raise CollectionError(self.first_reason)
            run=self.runs[self.active_run]
            caps=self.limits['caps'];per=dict(controls=800,resets=160,control_substeps=8000,reset_substeps=160) if self.stage=='A' else caps
            if any(self.counts[k]+cost[k]>v for k,v in caps.items()) or any(run['counts'][k]+cost[k]>v for k,v in per.items()):
                self.fail('Collection operation cap exceeded');raise CollectionError(self.first_reason)
            if self.stage=='A' and (self.episode is None or self.episode['controls']+cost['controls']>100 or self.episode['resets']+cost['resets']>20):
                self.fail('A episode exceeds 100 controls or 20 resets');raise CollectionError(self.first_reason)
            for key,value in cost.items():self.counts[key]+=value;run['counts'][key]+=value
            if self.episode:
                for key in ('controls','resets'):self.episode[key]+=cost[key]
            operation=self.sequence+1;self.pending=dict(operation_id=operation,costs=cost)
            self._append('reserve',self.pending);self.check();return operation

    def complete(self,operation,detail=None):
        with self.lock:
            self.check()
            if self.pending is None or type(operation) is not int or operation!=self.pending['operation_id']:
                self.fail('Completion differs from reserved operation');raise CollectionError(self.first_reason)
            self._append('complete',dict(operation_id=operation,detail=detail or {}))
            for key,value in self.pending['costs'].items():self.completed[key]+=value
            self.pending=None;self.check()

    def write_bytes(self,name,payload,*,run_id=None,replace=False):
        """Admit bytes before temp creation. Hash the supplied bytes, never readback.

        Every attempted generation stays charged, including old/new overlap on
        replacement. A failed temporary file is retained for diagnosis.
        """
        with self.lock:
            self.check()
            if not isinstance(payload,bytes) or not isinstance(name,str) or not name or '\\' in name:
                raise ValueError('Explicit relative POSIX output name and bytes required')
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts or ':' in name or relative.parts[0] in ('guardian','journal.jsonl','failure.json','worker-account.json'):
                raise ValueError('Output escapes declared collection payload namespace')
            if run_id is not None and run_id!=self.active_run:raise ValueError('Writes must use their active run account')
            path=_path(self.output/relative)
            if not path.is_relative_to(self.output):raise ValueError('Output escapes fixed stage')
            if replace and (path.suffix!='.json' or name not in self.files or self.files[name]['run_id']!=self.active_run):
                raise ValueError('Only active-account JSON metadata can be replaced')
            if not replace and (name in self.files or path.exists()):raise FileExistsError('Published collection payload is immutable')
            self._admit(len(payload))
            record=dict(name=name,bytes=len(payload),sha256=_sha(payload),run_id=self.active_run)
            self._append('write_reserved',dict(record,replace=replace))
            temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
            try:
                path.parent.mkdir(parents=True,exist_ok=True)
                with temporary.open('xb',buffering=0) as stream:_write_all(stream,payload);os.fsync(stream.fileno())
                self.check();_publish_json(temporary,path)  # No concurrent payload reader; status has its own fixed slot.
                stat=path.stat();record.update(mtime_ns=stat.st_mtime_ns,identity=[stat.st_dev,stat.st_ino])
                self._append('write_complete',record);self.files[name]=record;self.check()
                return deepcopy(record)
            except BaseException as error:self.fail(type(error).__name__+': '+str(error));raise

    def finish_run(self):
        with self.lock:
            self.check()
            if self.active_run is None or self.pending or self.episode:raise CollectionError('No finished run boundary')
            if self.stage=='A' and self.runs[self.active_run]['episodes']!=8:raise CollectionError('A requires eight attempted episodes per completed run')
            self._append('finish_run',dict(run_id=self.active_run));self.check()
            self.runs[self.active_run]['status']='complete';self.active_run=None

    def finish(self):
        with self.lock:
            self.check()
            if self.active_run or self.pending or set(self.runs)!=set(self.limits['run_ids']) or any(r['status']!='complete' for r in self.runs.values()):
                raise CollectionError('All fixed runs must complete before stage publication')
            self._append('finished',dict(counts=self.counts,completed=self.completed));os.fsync(self.stream.fileno());self.check()
            self.stream.close();self.check();self.closed=True;return self.snapshot()

    def close(self):
        if not self.closed and self.first_reason is None:self.fail('Collection account closed before completion; no continuation')
        try:self.stream.close()
        except BaseException as error:self.fail(type(error).__name__+': '+str(error));raise


def manifest_stats(root,manifest):
    """Held-role boundary: inspect declared file metadata, never open payloads."""
    root=_path(root);rows=[]
    for entry in manifest['episodes']:
        for record in entry['files']:
            name=record['name']
            if not isinstance(name,str) or Path(name).name!=name or ':' in name or '\\' in name:raise ValueError('Unsafe manifest file name')
            path=_path(root/name);stat=path.stat()
            if (not path.is_file() or stat.st_nlink!=1 or stat.st_size!=record['bytes']
                    or 'mtime_ns' in record and record['mtime_ns']!=stat.st_mtime_ns
                    or 'identity' in record and record['identity']!=[stat.st_dev,stat.st_ino]):raise CollectionError('Published file metadata differs')
            rows.append(dict(name=name,bytes=stat.st_size,sha256=record['sha256'],mtime_ns=stat.st_mtime_ns,identity=[stat.st_dev,stat.st_ino],payload_read=False))
    return rows


def status(budget,phase='running'):
    state=budget.snapshot();run=state['runs'].get(state['active_run'],{})
    return dict(phase=phase,stage=state['stage'],counts=state['counts'],charged_bytes=state['charged_bytes'],
        active_run=state['active_run'],run_deadline_unix=run.get('deadline_unix'),run_deadline_monotonic=run.get('deadline_monotonic'),
        model_loads=0,optimizer_updates=0,policy_calls=0,error=state['first_reason'])


class NoModelResourceProvider:
    """Existing Win32 RAM/commit/process/disk readers; deliberately no GPU call."""
    def _snapshot(self,pid,output):
        from . import policy_memory
        memory=policy_memory.system_memory();disk=policy_memory.disk_snapshot(output)
        process=policy_memory.process_memory(pid) if pid is not None else {}
        return dict(available_ram_bytes=memory.get('physical',{}).get('available_bytes') if memory.get('ok') else None,
            commit_headroom_bytes=memory.get('commit',{}).get('headroom_bytes') if memory.get('ok') else None,
            disk_free_bytes=disk.get('free_bytes') if disk.get('ok') else None,
            working_set_bytes=process.get('working_set_bytes') if process.get('ok') else None,
            private_bytes=process.get('private_bytes') if process.get('ok') else None,
            process_peak_working_set_bytes=process.get('peak_working_set_bytes') if process.get('ok') else None)
    def startup(self,output):return self._snapshot(None,output)
    def runtime(self,pid,output):return self._snapshot(pid,output)
    def close(self):pass


def resource_reason(snapshot,*,startup=False):
    if not isinstance(snapshot,dict):return 'invalid_resource_snapshot'
    floors=dict(available_ram_bytes=4*GIB if startup else 3*GIB//2,
        commit_headroom_bytes=6*GIB if startup else 3*GIB,disk_free_bytes=2*GIB)
    for name,floor in floors.items():
        value=snapshot.get(name)
        if type(value) is not int or value<0:return 'missing_or_invalid_resource:'+name
        if value<floor:return 'resource_below_floor:'+name
    if not startup:
        for name,cap in dict(working_set_bytes=2*GIB,private_bytes=3*GIB,process_peak_working_set_bytes=2*GIB).items():
            value=snapshot.get(name)
            if type(value) is not int or value<0:return 'missing_or_invalid_resource:'+name
            if value>cap:return 'process_memory_limit:'+name
    return None


def native_preflight(output):
    """Explicit future entry only; importing/planning never invokes this reader."""
    snapshot=NoModelResourceProvider().startup(_path(output))
    return dict(passed=resource_reason(snapshot,startup=True) is None,reason=resource_reason(snapshot,startup=True),
        resources=snapshot,gpu_queries=0,model_operations=0,physical_operations=0)


def _verify_authority(binding):
    if not isinstance(binding.get('authorization_id'),str) or not binding['authorization_id'].strip() or len(binding['authorization_id'])>256:
        raise PermissionError('Explicit collection authority required')
    commit=binding.get('source_commit')
    if not isinstance(commit,str) or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise PermissionError('Exact reviewed collection commit required')
    observed=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    if observed!=commit or dirty:raise PermissionError('Collection source must be clean and exactly reviewed')


def verify_worker(path,*,native_check=True):
    path=_path(path);value=_read(path);pid=os.getpid();claim=value['claim'];stage=value['stage']
    output,claim=_checked_claim(stage,value['binding'],claim)
    if (os.environ.get('GPC_COLLECTION_HANDOFF')!=str(path) or os.environ.get('GPC_COLLECTION_NONCE')!=claim['nonce']
            or os.environ.get('GPC_COLLECTION_GUARD_PID')!=str(value.get('guard_pid')) or value.get('actual_worker_pid')!=pid
            or path!=output/'guardian/handoff.json' or value.get('output_dir')!=str(output)
            or value.get('status_path')!=str(path.parent/'worker-status.json') or value.get('ownership')!='native_job_before_ack'
            or value.get('job_memory_bytes')!=3*GIB or value.get('kill_on_job_close') is not True
            or _read(path.parent/'worker.json')!=dict(pid=pid,parent_pid=os.getppid(),nonce=claim['nonce'])
            or _read(path.parent/'worker-ack.json')!=dict(pid=pid,nonce=claim['nonce'])):
        raise PermissionError('No matching collection worker PID/Job/ACK/nonce handoff')
    script=_path(value['script_path'])
    if not script.is_relative_to(ROOT/'scripts') or _sha(script.read_bytes())!=value['script_sha256']:
        raise PermissionError('Reviewed collection entry changed')
    if time.time()>=claim['deadline_unix'] or time.monotonic()>=claim['deadline_monotonic']:raise TimeoutError('Original collection deadline expired')
    if native_check:
        import ctypes
        from ctypes import wintypes
        from . import policy_memory
        kernel,_=policy_memory._libraries();inside=wintypes.BOOL();limits=policy_memory._JobExtendedLimits()
        kernel.GetCurrentProcess.argtypes=[];kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)];kernel.IsProcessInJob.restype=wintypes.BOOL
        kernel.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p];kernel.QueryInformationJobObject.restype=wintypes.BOOL
        if (not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(inside)) or not inside.value
                or not kernel.QueryInformationJobObject(None,9,ctypes.byref(limits),ctypes.sizeof(limits),None)
                or limits.JobMemoryLimit!=3*GIB or limits.BasicLimitInformation.LimitFlags&0x2200!=0x2200):
            raise PermissionError('Collection worker is outside retained 3GiB kill-on-close Job')
    return value


def _status_reason(value,claim,pid):
    if not isinstance(value,dict) or value.get('actual_worker_pid')!=pid or value.get('nonce')!=claim['nonce'] or value.get('stage')!=claim['stage']:
        return 'worker_status_identity_invalid'
    if not _finite(value.get('at_unix')) or not -.5<=time.time()-value['at_unix']<=5:return 'worker_status_stale_or_invalid'
    if value.get('phase') not in ('setup','running','export','complete','error'):return 'worker_phase_invalid'
    if value['phase']=='error' or value.get('error'):return 'worker_error:'+str(value.get('error'))
    for key in ('model_loads','optimizer_updates','policy_calls'):
        if type(value.get(key)) is not int or value[key]!=0:return 'forbidden_model_policy_operation:'+key
    counts=value.get('counts')
    if not isinstance(counts,dict) or set(counts)!=set(COSTS) or any(type(v) is not int or v<0 for v in counts.values()):return 'worker_counts_invalid'
    if any(counts[k]>v for k,v in claim['limits']['caps'].items()):return 'worker_operation_cap_exceeded'
    if type(value.get('charged_bytes')) is not int or not 0<=value['charged_bytes']<=claim['limits']['bytes']:return 'worker_byte_cap_exceeded'
    run=value.get('active_run')
    if run is not None:
        if run not in claim['limits']['run_ids']:return 'worker_run_identity_invalid'
        for key,now in (('run_deadline_unix',time.time()),('run_deadline_monotonic',time.monotonic())):
            if not _finite(value.get(key)) or now>=value[key]:return 'per_run_deadline_exceeded'
    return None


def supervise(stage,binding,*,python_executable,script_path,worker_options=None,provider=None,owner_factory=None,launcher_factory=None):
    """Concrete future no-model native launch; all tests inject the three owners.

    Worker command is ``script_path --worker handoff.json``. Parent reserves a
    fixed 2MiB telemetry envelope, avoiding any cross-process accounting service.
    """
    _verify_authority(binding);python,script=map(_path,(python_executable,script_path));options=deepcopy(worker_options or {})
    if not python.is_file() or not script.is_file() or not script.is_relative_to(ROOT/'scripts') or len(_json(options))>16384:
        raise ValueError('Explicit reviewed worker/interpreter/options required')
    claim=claim_stage(stage,binding);output=Path(claim['output_dir']);directory=output/'guardian';directory.mkdir()
    provider=provider if provider is not None else NoModelResourceProvider();owner_factory=owner_factory or WindowsWorker;launcher_factory=launcher_factory or subprocess.Popen
    injected=provider.__class__ is not NoModelResourceProvider or owner_factory is not WindowsWorker or launcher_factory is not subprocess.Popen
    owner=child=None;reason=error_text=None;last=None;job_active=None;worker_exit=launcher_exit=None;certain=True;owned_lock=False
    monitor_bytes=stdout_bytes=0;minima={};peaks={};cleanup=[];actual_pid=None;last_sample=float('-inf');deadline=claim['deadline_monotonic'];lock=Path(claim['path']).parent/'active-native.lock'
    monitor=(directory/'monitor.jsonl').open('xb',buffering=0);stdout=(directory/'worker-output.log').open('xb',buffering=0)
    stop=(directory/'stop-record.json').open('xb+',buffering=0);_write_all(stop,b'{}'+b' '*(SLOT-2));os.fsync(stop.fileno())
    pump_error=[];pump_thread=None
    def fail(code,error=None):
        nonlocal reason,error_text
        if reason is None:reason=code;error_text=type(error).__name__+': '+str(error) if error else None
    def event(kind,detail):
        nonlocal monitor_bytes
        payload=_json(dict(kind=kind,at_utc=_utc(),detail=detail))
        if monitor_bytes+len(payload)>MIB:raise CollectionError('Guardian monitor prepaid cap exceeded')
        monitor_bytes+=len(payload);_write_all(monitor,payload)
    def expired():return time.time()>=claim['deadline_unix'] or time.monotonic()>=deadline
    def observe(snapshot):
        for key in ('available_ram_bytes','commit_headroom_bytes','disk_free_bytes'):
            v=snapshot.get(key)
            if type(v) is int:minima[key]=min(minima.get(key,v),v)
        for key in ('working_set_bytes','private_bytes','process_peak_working_set_bytes'):
            v=snapshot.get(key)
            if type(v) is int:peaks[key]=max(peaks.get(key,v),v)
    def pump():
        nonlocal stdout_bytes
        try:
            while True:
                data=child.stdout.read(8192)
                if not data:return
                if not isinstance(data,bytes) or len(data)>8192 or stdout_bytes+len(data)>256*1024:raise CollectionError('Guardian stdout prepaid cap exceeded')
                stdout_bytes+=len(data);_write_all(stdout,data)
        except BaseException as error:pump_error.append(error)
    try:
        _exclusive(lock,_json(dict(stage=stage,nonce=claim['nonce'],pid=os.getpid())));owned_lock=True
        startup=provider.startup(output);observe(startup);event('startup',startup);problem=resource_reason(startup,startup=True)
        if problem:fail(problem)
        if reason is None:
            handshake=directory/'worker.json';ack=directory/'worker-ack.json';handoff=directory/'handoff.json';status_path=directory/'worker-status.json'
            command=[str(python),'-B',str(ROOT/'scripts/guarded_c_child.py'),'--handshake',str(handshake),'--ack',str(ack),'--nonce',claim['nonce'],'--',str(script),'--worker',str(handoff)]
            env=dict(os.environ,GPC_COLLECTION_HANDOFF=str(handoff),GPC_COLLECTION_NONCE=claim['nonce'],GPC_COLLECTION_GUARD_PID=str(os.getpid()),
                SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',MPLBACKEND='Agg',PYGAME_HIDE_SUPPORT_PROMPT='1',OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
            env.update({key:'2' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
            child=launcher_factory(command,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
            pump_thread=threading.Thread(target=pump,daemon=True);pump_thread.start();until=min(deadline,time.monotonic()+10)
            while not handshake.exists():
                if pump_error:raise pump_error[0]
                if child.poll() is not None or time.monotonic()>=until or expired():raise TimeoutError('No actual collection worker handshake before deadline')
                time.sleep(.02)
            actual=_read(handshake)
            if actual.get('nonce')!=claim['nonce'] or type(actual.get('pid')) is not int or actual['pid']<=0:raise PermissionError('Collection handshake PID/nonce invalid')
            actual_pid=actual['pid'];owner=owner_factory(actual_pid,child.pid)
            value=dict(stage=stage,claim=claim,binding=binding,nonce=claim['nonce'],output_dir=str(output),worker_options=options,
                actual_worker_pid=actual_pid,launcher_pid=child.pid,guard_pid=os.getpid(),ownership='native_job_before_ack',
                job_memory_bytes=3*GIB,kill_on_job_close=True,status_path=str(status_path),script_path=str(script),script_sha256=_sha(script.read_bytes()))
            snapshot=provider.runtime(actual_pid,output);observe(snapshot);problem=resource_reason(snapshot)
            if problem:fail(problem)
            elif expired():fail('aggregate_deadline_exceeded')
            else:_exclusive(handoff,_json(value));_exclusive(ack,_json(dict(pid=actual_pid,nonce=claim['nonce'])))
            acknowledged=time.monotonic()
            while reason is None and not owner.poll_exit():
                if pump_error:raise pump_error[0]
                if expired():fail('aggregate_deadline_exceeded');break
                snapshot=provider.runtime(actual_pid,output);problem=resource_reason(snapshot)
                if problem:
                    if owner.poll_exit() and problem.startswith('missing_or_invalid_resource:') and problem.split(':')[1] in ('working_set_bytes','private_bytes','process_peak_working_set_bytes'):break
                    fail(problem);break
                last=_latest(status_path,last)
                if last is None:
                    if time.monotonic()-acknowledged>5:fail('worker_status_initial_missing');break
                else:
                    problem=_status_reason(last,claim,actual_pid)
                    if problem:fail(problem);break
                observe(snapshot)
                if time.monotonic()-last_sample>=1:
                    event('sample',dict(resources=snapshot,status=last));last_sample=time.monotonic()
                time.sleep(.2)
            if reason is None:
                last=_latest(status_path,None);problem=_status_reason(last,claim,actual_pid)
                if problem:fail(problem)
                elif owner.exit_code()!=0 or last['phase']!='complete':fail('worker_incomplete_or_nonzero_exit')
                elif _read(output/'completion.json').get('status')!='complete':fail('completion_record_incomplete')
    except BaseException as error:fail('collection_guard_failure',error)
    finally:
        if owner is not None:
            try:
                if not owner.poll_exit():owner.terminate_and_wait(timeout=10)
                worker_exit=owner.exit_code();job_active=owner.quiesce(timeout=10)
                if job_active!=0:raise CollectionError('Owned collection Job still has descendants')
            except BaseException as error:certain=False;cleanup.append(str(error));fail('job_cleanup_uncertain',error)
            try:owner.close()
            except BaseException as error:certain=False;cleanup.append(str(error));fail('job_close_failed',error)
        elif child is not None:certain=False
        if child is not None:
            try:launcher_exit=child.wait(timeout=10)
            except BaseException:
                try:child.terminate();launcher_exit=child.wait(timeout=10)
                except BaseException as error:certain=False;cleanup.append(str(error));fail('launcher_cleanup_uncertain',error)
        if pump_thread is not None:
            pump_thread.join(timeout=3)
            if pump_thread.is_alive():certain=False;fail('stdout_cleanup_uncertain')
            elif pump_error:fail('stdout_failure',pump_error[0])
        try:provider.close()
        except BaseException as error:fail('resource_reader_close_failed',error)
        if owned_lock and certain:
            try:
                if _read(lock).get('nonce')!=claim['nonce']:raise CollectionError('Native collection owner lock changed')
                lock.unlink();owned_lock=False
            except BaseException as error:fail('native_owner_unlock_failed',error)
        for stream in (monitor,stdout,child.stdout if child and pump_thread and not pump_thread.is_alive() else None):
            if stream is None:continue
            actions=([lambda:os.fsync(stream.fileno())] if stream in (monitor,stdout) else [])+[stream.close]
            for action in actions:
                try:action()
                except BaseException as error:fail('guardian_final_io_failed',error)
        if expired():fail('aggregate_deadline_exceeded')
        report=dict(kind='gpc_collection_guard_report.v1',stage=stage,status='complete' if reason is None else 'stopped',guard_reason=reason,error=error_text,
            output_dir=str(output),claim=claim,nonce=claim['nonce'],guard_pid=os.getpid(),launcher_pid=child.pid if child else None,actual_worker_pid=actual_pid,
            job_memory_bytes=3*GIB,job_active_processes_after_cleanup=job_active,worker_exit_code=worker_exit,launcher_exit_code=launcher_exit,
            lock_retained=owned_lock,cleanup_errors=cleanup,native_integration_verified=not injected and reason is None,
            min_free=minima,observed_peaks=peaks,last_status=last,gpu_queries=0,model_operations=0,
            wall_seconds=time.monotonic()-claim['started_monotonic'],ended_utc=_utc(),monitor_bytes=monitor_bytes,stdout_bytes=stdout_bytes,guardian_prepaid_bytes=GUARD_BYTES)
        try:
            payload=_json(report)
            if len(payload)>SLOT:raise CollectionError('Guardian final report exceeds fixed slot')
            stop.seek(0);_write_all(stop,payload+b' '*(SLOT-len(payload)));os.fsync(stop.fileno())
            if expired() and reason is None:
                fail('aggregate_deadline_exceeded');report.update(status='stopped',guard_reason=reason,native_integration_verified=False)
                payload=_json(report);stop.seek(0);_write_all(stop,payload+b' '*(SLOT-len(payload)));os.fsync(stop.fileno())
        except BaseException as error:fail('durable_stop_record_failed',error)
        try:stop.close()
        except BaseException as error:fail('stop_record_close_failed',error)
        previous_reason=report['guard_reason']
        if expired():fail('aggregate_deadline_exceeded')
        report.update(status='complete' if reason is None else 'stopped',guard_reason=reason,error=error_text,
            native_integration_verified=not injected and reason is None,wall_seconds=time.monotonic()-claim['started_monotonic'],ended_utc=_utc())
        if previous_reason!=reason:
            try:
                path=_path(directory/'stop-record.json');stat=path.stat()
                if not path.is_file() or stat.st_size!=SLOT or stat.st_nlink!=1:raise CollectionError('Fixed stop slot changed')
                payload=_json(report)
                if len(payload)>SLOT:raise CollectionError('Guardian final report exceeds fixed slot')
                with path.open('r+b',buffering=0) as stream:_write_all(stream,payload+b' '*(SLOT-len(payload)));os.fsync(stream.fileno())
            except BaseException as error:fail('durable_stop_record_finalization_failed',error)
        if expired():fail('aggregate_deadline_exceeded')
        report.update(status='complete' if reason is None else 'stopped',guard_reason=reason,error=error_text,
            native_integration_verified=not injected and reason is None,wall_seconds=time.monotonic()-claim['started_monotonic'],ended_utc=_utc())
    return report
