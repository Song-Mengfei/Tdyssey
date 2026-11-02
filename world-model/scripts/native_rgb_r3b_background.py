"""Independent, once-released R3B first-epoch and paired-evaluation worker.

This process makes no app/MCP calls. It records local completion for an officially
configured completion monitor; closing Codex does not manufacture an offline
callback. Importing this entry does not import Torch or construct a model.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager,redirect_stdout,redirect_stderr
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
STAGE=ROOT/'artifacts/native-rgb-supervised-20261029/r3b'
PARENT_THREAD='__LOCAL_CHAT__'
DATA_SHA='0e91e1412f937c7d8f39473243a27a3720317a07474e730a510e628d040c0a19'
BASE_SHA='c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148'
FROZEN_SPEC_SHA='3aa0f30997ee70b1f4e06a0b37acf4c87c94331504d253ccb315f2bbcae75289'


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def json_safe(value):
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):return value if math.isfinite(value)else None
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return{str(k):json_safe(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return[json_safe(v)for v in value]
    if type(value).__module__.startswith('numpy'):
        return json_safe(value.tolist()if hasattr(value,'tolist')else value.item())
    return dict(omitted_non_scalar_type=type(value).__name__)
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8',newline='\n')as stream:
        json.dump(json_safe(value),stream,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    deadline=time.monotonic()+2
    while True:
        try:os.replace(temporary,path);return
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.025)


def job_membership():
    if os.name!='nt':return dict(supported=False,is_in_job=None,reason='non-Windows host')
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype=wintypes.BOOL
    flag=wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(flag)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(supported=True,is_in_job=bool(flag.value),query='IsProcessInJob(current process,NULL)')


def actual_process_executable():
    """Read the process PE path, including a venv redirector's actual runtime."""
    if os.name!='nt':return str(Path(sys.executable).resolve())
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetModuleFileNameW.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD]
    kernel.GetModuleFileNameW.restype=wintypes.DWORD
    buffer=ctypes.create_unicode_buffer(32768)
    size=kernel.GetModuleFileNameW(None,buffer,len(buffer))
    if not size or size>=len(buffer):raise ctypes.WinError(ctypes.get_last_error())
    return buffer.value


class Status:
    def __init__(self,background,config_sha,resume,run_id=None):
        self.root=Path(background);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.stop_event=threading.Event();self.error=None;self.accounting=None
        self.started=time.perf_counter()
        self.value=dict(kind='R3B-background-status.v1',run_id=run_id,pid=os.getpid(),parent_pid=os.getppid(),
            started_at_utc=utc(),config_sha256=config_sha,resumed=resume,phase='startup',phase_started_at_utc=utc(),
            closed=False,heartbeat_sequence=0,progress_updates=0,last_progress_at_utc=None,
            formal=dict(stage_attempts=0,stage_successes=0,AMP_skips=0,formal_draws=0),
            new_physics_controls=0,heldout_accesses=0,automatic_restart=False,maximum_formal_batches=1499,
            next_epoch_authorized=False,completion_callback_sent_by_worker=False)
        self.thread=threading.Thread(target=self._loop,name='R3B-local-heartbeat',daemon=True)
    def _write(self):
        self.value['at_utc']=utc();self.value['elapsed_worker_seconds']=time.perf_counter()-self.started
        atomic_json(self.root/'status.json',self.value)
        atomic_json(self.root/'heartbeat.json',{k:self.value[k]for k in('kind','run_id','pid','parent_pid','phase','closed',
            'heartbeat_sequence','progress_updates','last_progress_at_utc','at_utc','elapsed_worker_seconds','formal')})
    def start(self):
        with self.lock:self._write()
        self.thread.start()
    def _loop(self):
        while not self.stop_event.wait(15.):
            try:
                with self.lock:
                    if self.accounting is not None:self.accounting.heartbeat()
                    self.value['heartbeat_sequence']+=1;self._write()
            except BaseException as exc:
                self.error=exc
                print('R3B heartbeat persistence failed: '+repr(exc),file=sys.stderr,flush=True)
                return
    def update(self,event,details=None):
        if self.error is not None:raise RuntimeError('Heartbeat failed; stop rather than training without evidence')from self.error
        if isinstance(event,dict)and details is None:
            details=event;event=details.get('event',details.get('kind','evaluation_progress'))
        details=json_safe(details or{})
        with self.lock:
            self.value['last_event']=str(event);self.value['last_details']=details
            self.value['progress_updates']+=1;self.value['last_progress_at_utc']=utc()
            progress=details.get('progress',details)
            for key in('stage_attempts','stage_successes','AMP_skips','abandoned_attempts','formal_draws','lineage_updates','epoch_complete'):
                if key in progress:self.value['formal'][key]=progress[key]
            self._write()
            with(self.root/'events.jsonl').open('a',encoding='utf-8',newline='\n')as stream:
                stream.write(json.dumps(dict(at_utc=utc(),phase=self.value['phase'],event=event,details=details),ensure_ascii=False,allow_nan=False)+'\n')
            print(json.dumps(dict(at_utc=utc(),phase=self.value['phase'],event=event,details=details),ensure_ascii=False,allow_nan=False),flush=True)
    def phase(self,name,details=None):
        with self.lock:
            if name!=self.value['phase']:self.value.update(phase=name,phase_started_at_utc=utc())
        self.update('phase_changed',dict(phase=name,**(details or{})))
    def finish(self,success,failure=None):
        self.stop_event.set()
        if self.thread.is_alive():self.thread.join(timeout=3.)
        with self.lock:
            self.value.update(closed=True,phase='complete'if success else'failed',failure=failure)
            self._write()


def stage_paths(config):
    """Separate a new attempt's outputs from the shared, unchanged R3B budget."""
    stage=Path(config.get('stage_root',STAGE)).resolve()
    budget=Path(config.get('budget_stage_root',STAGE)).resolve()
    if budget!=STAGE:raise ValueError('All R3B attempts/diagnostics share the original R3B budget root')
    if stage!=STAGE:
        attempts=STAGE/'attempts'
        if stage==attempts or not stage.is_relative_to(attempts):
            raise ValueError('A recovery attempt must stay below R3B/attempts')
        if config.get('recovery_authorized')is not True:
            raise PermissionError('A new attempt requires the explicit human recovery release')
    return stage,budget


def recovery_identity(config):
    """Reject incomplete CPU/startup evidence before importing Torch or a model."""
    stage,_=stage_paths(config)
    if stage==STAGE:return None
    if not config.get('run_id'):raise ValueError('A recovery attempt requires its own run_id')
    previous=config.get('previous_failure',{})
    path=Path(previous.get('path','')).resolve()
    if path!=STAGE/'background/completion.json' or sha(path)!=previous.get('sha256'):
        raise ValueError('The original failed completion must remain bound, not overwritten')
    failed=read(path)
    if failed.get('run_id')!=previous.get('run_id')or failed.get('run_id')==config['run_id']:
        raise ValueError('Recovery run_id must differ from its actual failed predecessor')
    if failed.get('completed')is not False or failed.get('closed')is not True:
        raise ValueError('The predecessor is not a closed failed run')
    progress=failed.get('formal_progress',{})
    if any(progress.get(k)!=0 for k in('stage_attempts','stage_successes','formal_draws')):
        raise ValueError('A consumed prior epoch cannot restart from zero in a new attempt')
    frozen=config.get('frozen_validation_spec',config.get('frozen_evaluation_spec',{}))
    spec_path=Path(frozen.get('path','')).resolve()
    if spec_path!=STAGE/'evaluation/spec.json' or frozen.get('sha256')!=FROZEN_SPEC_SHA or sha(spec_path)!=FROZEN_SPEC_SHA:
        raise ValueError('Recovery must retain the exact original144natural/16stress/old4 specification')
    binding=config.get('loader_diagnostic_gate',{})
    gate_path=Path(binding.get('path','')).resolve()
    if not gate_path.is_relative_to(STAGE)or sha(gate_path)!=binding.get('sha256'):
        raise ValueError('The actual CPU diagnostic gate must be bound inside R3B')
    gate=read(gate_path)
    if gate.get('passed')is not True or gate.get('run_id')!=binding.get('run_id'):
        raise ValueError('The bound diagnostic did not pass for its real run_id')
    if gate.get('root_cause_confirmed')is not True or not isinstance(gate.get('root_cause'),str)or not gate['root_cause'].strip():
        raise ValueError('Recovery requires an evidenced root cause, not BrokenProcessPool as a diagnosis')
    if not isinstance(gate.get('repair'),str)or not gate['repair'].strip():raise ValueError('Bind the evidence-supported minimal repair')
    expected=dict(data_manifest_sha256=DATA_SHA,loader_source_sha256=sha(ROOT/'tdyssey/native_rgb_r3a_data.py'),
        workers=2,prefetch_batches=2,model_loaded=False,cuda_initialized=False,launcher_exited_before_progress=True)
    for key,value in expected.items():
        if gate.get(key)!=value:raise ValueError('CPU diagnostic boundary does not match the released loader: '+key)
    repair_sha=sha(ROOT/'tdyssey/native_rgb_r3b_loader_recovery.py')
    if gate.get('repair_source_sha256')!=repair_sha or config.get('loader_recovery_source_sha256')!=repair_sha:
        raise ValueError('CPU diagnostic and runtime must bind the same actual loader recovery implementation')
    pids=gate.get('verified_worker_pids',[])
    if len(set(pids))!=2 or any(type(pid)is not int or pid<=0 for pid in pids):
        raise ValueError('Both actual data-worker PIDs must have verified batches')
    if type(gate.get('verified_batches'))is not int or gate['verified_batches']<2:
        raise ValueError('The real background diagnostic must verify multiple batches')
    contract=gate.get('dataset_contract',{})
    if Path(contract.get('physics_source_root','')).resolve()!=Path(config['physics_source_root']).resolve():
        raise ValueError('CPU diagnostic used a different physics/render source')
    for key,default in(('frame_cache_bytes',16*1024**2),('state_cache_bytes',8*1024**2)):
        if contract.get(key)!=config.get(key,default):raise ValueError('CPU diagnostic cache contract differs: '+key)
    return dict(previous_failure=previous,loader_diagnostic_gate=binding,frozen_validation_spec=frozen,
        root_cause=gate.get('root_cause'),repair=gate.get('repair'),diagnostic_summary=gate.get('summary'),
        verified_worker_pids=pids,verified_batches=gate['verified_batches'],repair_source_sha256=repair_sha,previous_model_seconds=33.047)


def restore_validation_spec(stage,identity):
    """Copy exact original bytes, never choose another validation anchor."""
    source=Path(identity['frozen_validation_spec']['path']);target=Path(stage)/'evaluation/spec.json'
    data=source.read_bytes()
    if hashlib.sha256(data).hexdigest()!=FROZEN_SPEC_SHA:raise ValueError('Original frozen validation bytes changed')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.read_bytes()!=data:raise ValueError('This attempt already has a different validation specification')
    else:
        with target.open('xb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    return target


def storage(config,write_reserve_bytes=0):
    attempt,stage=stage_paths(config);files={}
    roots=[Path(p)for p in config.get('shared_storage_roots',config.get('storage_roots',[]))]
    if not roots:raise ValueError('Explicit shared storage roots required before model work')
    for root in roots+[stage]:
        for path in([root]if root.is_file()else root.rglob('*')if root.is_dir()else[]):
            if path.is_file():files[str(path.resolve()).lower()]=path
    new_sources=[Path(p)for p in config.get('new_stage_source_files',[])]
    for path in new_sources:
        if path.exists():files[str(path.resolve()).lower()]=path
    shared=sum(p.stat().st_size for p in files.values());persistent=transient=0
    for path in stage.rglob('*'):
        if path.is_file():
            if path.name.endswith('.tmp')or'.tmp.'in path.name:transient+=path.stat().st_size
            else:persistent+=path.stat().st_size
    persistent+=sum(p.stat().st_size for p in new_sources if p.exists()and not p.resolve().is_relative_to(stage))
    disk=shutil.disk_usage(stage)
    result=dict(stage_persistent_bytes=persistent,stage_transient_atomic_bytes=transient,
        budget_stage_root=str(stage),current_attempt_root=str(attempt),
        stage_cap_bytes=int(config.get('stage_cap_bytes',320*1024**2)),shared_bytes=shared,
        shared_cap_bytes=int(config.get('shared_cap_bytes',3*1024**3)),atomic_reserve_bytes=max(int(config.get('atomic_reserve_bytes',256*1024**2)),int(write_reserve_bytes)),
        filesystem_free_bytes=disk.free,filesystem_total_bytes=disk.total)
    result['passed']=persistent<=result['stage_cap_bytes']and shared+result['atomic_reserve_bytes']<=result['shared_cap_bytes']and disk.free>=result['atomic_reserve_bytes']
    if not result['passed']:raise RuntimeError('Exact R3B storage shortage: '+json.dumps(result))
    return result


class BoundedStorageCheck:
    """Fast per-update disk checks; full shared-root scans stay at stage boundaries."""
    def __init__(self,config,initial):
        self.config=config;self.initial=initial;self.attempt,self.stage=stage_paths(config)
        self.sources=[Path(p)for p in config.get('new_stage_source_files',[])if Path(p).exists()and not Path(p).resolve().is_relative_to(self.stage)]
        self.external=initial['shared_bytes']-sum(p.stat().st_size for p in self.stage.rglob('*')if p.is_file())-sum(p.stat().st_size for p in self.sources)
        self.last=0.;self.result=initial
    def __call__(self,write_reserve_bytes=0):
        now=time.monotonic()
        if not write_reserve_bytes and now-self.last<2.:return self.result
        persistent=transient=0
        for p in self.stage.rglob('*'):
            if p.is_file():
                if p.name.endswith('.tmp')or'.tmp.'in p.name:transient+=p.stat().st_size
                else:persistent+=p.stat().st_size
        source_bytes=sum(p.stat().st_size for p in self.sources)
        shared=self.external+persistent+transient+source_bytes
        reserve=max(int(self.config.get('atomic_reserve_bytes',256*1024**2)),int(write_reserve_bytes))
        # Mutable ledger/receipts outside this stage have small bounded growth;
        # a conservative4MiB guard avoids repeatedly scanning the old library.
        outside_growth_guard=4*1024**2
        free=shutil.disk_usage(self.stage).free
        self.result=dict(stage_persistent_bytes=persistent+source_bytes,stage_transient_atomic_bytes=transient,
            budget_stage_root=str(self.stage),current_attempt_root=str(self.attempt),
            shared_bytes_estimate=shared,external_growth_guard_bytes=outside_growth_guard,atomic_reserve_bytes=reserve,filesystem_free_bytes=free)
        passed=persistent+source_bytes<=int(self.config.get('stage_cap_bytes',320*1024**2))and shared+outside_growth_guard+reserve<=int(self.config.get('shared_cap_bytes',3*1024**3))and free>=reserve
        self.result['passed']=passed;self.last=now
        if not passed:raise RuntimeError('Exact live stage/disk shortage with conservative shared estimate: '+json.dumps(self.result))
        return self.result


def preflight(config_path,resume=False):
    config=read(config_path);config['stage_root']=str(Path(config.get('stage_root',STAGE)).resolve())
    stage,budget=stage_paths(config);config['budget_stage_root']=str(budget)
    if not(config.get('execution_authorized')is True and config.get('training_authorized')is True
           and config.get('is_proposal')is False and config.get('first_epoch_only')is True):
        raise PermissionError('Current explicit local first-epoch execution release required')
    if int(config.get('max_formal_batches',1499))!=1499:raise ValueError('Exactly first1499 batches, no proposal8epoch continuation')
    if resume and config.get('resume_authorized')is not True:raise PermissionError('Explicit resume authorization required')
    if not resume and config.get('resume_authorized'):raise PermissionError('Resume config requires explicit --resume-authorized')
    if int(config.get('stage_cap_bytes',320*1024**2))>320*1024**2 or int(config.get('shared_cap_bytes',3*1024**3))>3*1024**3:
        raise ValueError('Recovery does not enlarge the approved stage/shared storage limits')
    if int(config.get('atomic_reserve_bytes',256*1024**2))<256*1024**2:raise ValueError('Recovery must retain the atomic reserve')
    bindings=list(config.get('input_bindings',[]))
    identity=config.get('source_identity')
    if identity:
        payload=read(identity);bindings+=payload.get('files',[])
    if not bindings:raise ValueError('Frozen input/source identity bindings required')
    for item in bindings:
        path=item.get('path',item.get('source'))
        if path is None or sha(path)!=item['sha256']:raise ValueError('Frozen R3B input/source differs: '+str(path))
    manifest=Path(config.get('data_manifest',STAGE.parent/'r3a/manifest.json'))
    checkpoint=Path(config.get('resume_checkpoint',STAGE.parent/'r2c3/run/last.pt'))
    if sha(manifest)!=DATA_SHA or sha(checkpoint)!=BASE_SHA:raise ValueError('Only original R3A data and complete6000 checkpoint are released')
    config['data_manifest']=str(manifest.resolve());config['resume_checkpoint']=str(checkpoint.resolve())
    config['resume_authorized']=resume
    state=read(STAGE.parent/'stage-state.json')
    if not state.get('training_authorized')or state.get('current_stage')!='R3B':raise PermissionError('Durable stage-state does not release R3B')
    config['recovery_identity']=recovery_identity(config)
    return config,dict(config_sha256=sha(config_path),model_work_started=False,Torch_imported='torch'in sys.modules,
        data_manifest_sha256=DATA_SHA,readonly6000_checkpoint_sha256=BASE_SHA,recovery_identity=config['recovery_identity'],storage=storage(config))


@contextmanager
def owner(background,resume,config_sha):
    background=Path(background);background.mkdir(parents=True,exist_ok=True)
    lock=background/'worker-owner.lock';token=uuid.uuid4().hex
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    claim=background/('resume-'+token+'.json'if resume else'execution-once-claim.json')
    data=dict(pid=os.getpid(),parent_pid=os.getppid(),token=token,at_utc=utc(),resume=resume,config_sha256=config_sha)
    try:
        os.write(fd,(json.dumps(data)+'\n').encode());os.fsync(fd)
        with claim.open('x',encoding='utf-8')as stream:json.dump(data,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        yield data
    finally:
        os.close(fd)
        if lock.exists()and read(lock).get('token')==token:lock.unlink()


def run(config_path,resume=False):
    config,checked=preflight(config_path,resume);stage=Path(config['stage_root']);background=stage/'background'
    background.mkdir(parents=True,exist_ok=True)
    if(background/'completion.json').exists()and not resume:raise FileExistsError('Prior completion exists; no automatic replay')
    state_path=background/'pipeline-state.json'
    pipeline=read(state_path)if resume and state_path.exists()else dict(kind='R3B-independent-pipeline-state.v1',completed_phases=[],started_at_utc=utc(),recovery_context=config['recovery_identity'])
    status=Status(background,checked['config_sha256'],resume,config.get('run_id'));session=accounting=None;failure=None;result={};began=time.perf_counter()
    def completed(name,value=None):
        if name not in pipeline['completed_phases']:pipeline['completed_phases'].append(name)
        if value is not None:
            # Raw metrics and the frozen spec already have bound canonical files;
            # do not duplicate their many rows in the small restart cursor.
            if name in('t0_evaluation','epoch1_evaluation'):
                phase='t0'if name=='t0_evaluation'else'epoch1'
                value={k:value[k]for k in('kind','phase','evaluation_complete','spec_sha256','stats','index_file')if k in value}
                value.update(metrics_file=str(stage/'evaluation'/phase/'metrics.json'),metrics_sha256=sha(stage/'evaluation'/phase/'metrics.json'))
            elif name=='validation_frozen':
                value=dict(spec_file=str(stage/'evaluation/spec.json'),spec_sha256=sha(stage/'evaluation/spec.json'))
            pipeline[name]=json_safe(value)
        pipeline['at_utc']=utc();atomic_json(state_path,pipeline)
    with owner(background,resume,checked['config_sha256']):
        with(background/'worker.stdout.log').open('a',encoding='utf-8',buffering=1)as stdout,\
            (background/'worker.stderr.log').open('a',encoding='utf-8',buffering=1)as stderr,redirect_stdout(stdout),redirect_stderr(stderr):
            try:
                proof=dict(kind='R3B-actual-worker-startup-proof.v1',run_id=config.get('run_id'),config_sha256=checked['config_sha256'],at_utc=utc(),pid=os.getpid(),parent_pid=os.getppid(),
                    executable=sys.executable,actual_process_executable=actual_process_executable(),working_directory=str(Path.cwd().resolve()),prefix=sys.prefix,
                    argv=sys.argv,job=job_membership(),Torch_imported='torch'in sys.modules,preflight=checked)
                atomic_json(background/'startup-proof.json',proof)
                if config.get('require_detached_job',True)and(os.name!='nt'or proof['job']['is_in_job']is not False):
                    raise RuntimeError('Real worker is still in a cleanup job or job independence is unverified; no model work started')
                status.start();status.phase('freeze_validation')
                from tdyssey import native_rgb_r3b_evaluation as evaluation
                from tdyssey import native_rgb_r3b_training as training
                from tdyssey.native_rgb_r3b_report import build_report
                if'validation_frozen'not in pipeline['completed_phases']:
                    if config['recovery_identity']is not None:
                        restore_validation_spec(stage,config['recovery_identity'])
                        spec=evaluation._spec(stage)
                    else:spec=evaluation.freeze_spec(stage,physics_source=config.get('physics_source_root'))
                    completed('validation_frozen',spec)
                model_phases_done=all(name in pipeline['completed_phases']for name in('engineering_probe','t0_evaluation','first_epoch','epoch1_evaluation'))
                if not model_phases_done:
                    status.phase('restore_model_and_ledger')
                    accounting=training.R3BModelLedger(config)
                    accounting.begin();status.accounting=accounting;config['model_accounting']=accounting
                    config['storage_check']=BoundedStorageCheck(config,checked['storage'])
                    with accounting.model_section('restore_session'):
                        session=training.create_session(config,status_cb=status.update)
                    if int(session.progress.get('stage_attempts',0))>0 and't0_evaluation'not in pipeline['completed_phases']:
                        raise RuntimeError('Training progress without completed t0 evidence cannot be replayed')
                    if config['recovery_identity']is not None:
                        # A fresh process must prove its real loader even on an
                        # explicitly authorized checkpoint resume. CPU batches
                        # do not advance the formal sampler or run the model.
                        status.phase('loader_after_model_restore')
                        from tdyssey.native_rgb_r3b_loader_recovery import verify_after_model_restore
                        with accounting.model_section('loader_after_model_restore'):
                            completed('loader_after_model_restore',verify_after_model_restore(session,status_cb=status.update))
                    if'engineering_probe'not in pipeline['completed_phases']:
                        if pipeline.get('engineering_probe_started')and not config.get('engineering_probe_retry_authorized',False):
                            raise RuntimeError('Prior probe was interrupted; no unlogged automatic probe replay')
                        pipeline['engineering_probe_started']=True;atomic_json(state_path,pipeline)
                        status.phase('engineering_probe')
                        with accounting.model_section('engineering_probe'):
                            completed('engineering_probe',training.engineering_probe(session,status_cb=status.update))
                    if't0_evaluation'not in pipeline['completed_phases']:
                        if int(session.progress.get('stage_attempts',0))!=0:raise RuntimeError('t0 must precede every formal update')
                        status.phase('t0_evaluation')
                        with accounting.model_section('t0_evaluation'),session.preserve_training_state():
                            result['t0']=evaluation.evaluate(stage,session.model,session.source,'t0',status_cb=status.update,checkpoint_sha256=BASE_SHA)
                        completed('t0_evaluation',result['t0'])
                    if'first_epoch'not in pipeline['completed_phases']:
                        status.phase('first_epoch_training')
                        with accounting.model_section('first_epoch_training'):
                            result['training']=training.run_first_epoch(session,status_cb=status.update)
                        completed('first_epoch',result['training'])
                    if not session.progress.get('epoch_complete')or session.progress.get('stage_attempts')!=1499:
                        raise RuntimeError('Only an actually complete1499-attempt first epoch can enter final paired evaluation')
                    if'epoch1_evaluation'not in pipeline['completed_phases']:
                        status.phase('epoch1_evaluation')
                        with accounting.model_section('epoch1_evaluation'),session.preserve_training_state():
                            result['epoch1']=evaluation.evaluate(stage,session.model,session.source,'epoch1',status_cb=status.update,checkpoint_sha256=sha(session.checkpoint_path))
                        completed('epoch1_evaluation',result['epoch1'])
                    pipeline['formal_progress']=json_safe(session.progress);atomic_json(state_path,pipeline)
                    session.close();session=None
                    with status.lock:
                        accounting.finish('complete_first_epoch_and_paired_evaluation')
                        status.accounting=None
                    pipeline['model_accounting']=json_safe(accounting.snapshot());atomic_json(state_path,pipeline)
                    accounting=None
                status.phase('reporting')
                result['report']=build_report(stage,physics_source=config.get('physics_source_root'),status_cb=status.update)
                completed('report',result['report'])
                status.phase('final_manifest')
                result['storage']=storage(config)
                if sha(config['resume_checkpoint'])!=BASE_SHA or sha(config['data_manifest'])!=DATA_SHA:
                    raise RuntimeError('Readonly original checkpoint or data manifest changed')
                files=[dict(file=str(p.relative_to(stage)).replace('\\','/'),sha256=sha(p),bytes=p.stat().st_size)
                       for p in sorted(stage.rglob('*'))if p.is_file()and p.name not in('status.json','heartbeat.json','worker.stdout.log','worker.stderr.log','events.jsonl','worker-owner.lock','completion.json','final-manifest.json')and not p.name.endswith('.tmp')]
                atomic_json(background/'final-manifest.json',dict(kind='R3B-background-final-manifest.v1',files=files,
                    original6000_sha256=BASE_SHA,data_manifest_sha256=DATA_SHA,formal_progress=pipeline.get('formal_progress'),new_physics_controls=0,heldout_accesses=0))
            except BaseException as exc:
                failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),at_utc=utc(),phase=status.value['phase'])
                print(failure['traceback'],file=sys.stderr,flush=True)
                if session is not None:
                    try:failure['checkpoint_save']=json_safe(session.checkpoint_if_possible('worker_exception'))
                    except BaseException as saving:failure['checkpoint_save_error']=repr(saving)
                    pipeline['formal_progress']=json_safe(session.progress)
                atomic_json(background/'failure.json',failure)
                (stage/'FAILED.md').write_text('# R3B background pipeline stopped with evidence\n\n'+
                    f"Phase: {failure['phase']}\n\n{failure['type']}: {failure['reason']}\n\n"+
                    'No automatic restart, epoch2 or new data collection was started.\n\n'+
                    'Read background/failure.json, pipeline-state.json and run/progress.json before any explicitly authorized recovery.\n',encoding='utf-8')
            finally:
                if session is not None:
                    try:session.close()
                    except BaseException as exc:
                        if failure is None:failure=dict(type=type(exc).__name__,reason=str(exc),phase='close_session',traceback=traceback.format_exc())
                if accounting is not None:
                    try:
                        with status.lock:
                            accounting.finish('stopped_with_evidence'if failure else'complete')
                            status.accounting=None
                        pipeline['model_accounting']=json_safe(accounting.snapshot())
                    except BaseException as exc:
                        if failure is None:failure=dict(type=type(exc).__name__,reason=str(exc),phase='close_global_ledger',traceback=traceback.format_exc())
                        else:failure['ledger_finish_error']=repr(exc)
                    status.accounting=None
                originals_verified={}
                for key,path in(('checkpoint_sha256',config['resume_checkpoint']),('data_manifest_sha256',config['data_manifest'])):
                    try:originals_verified[key]=sha(path)
                    except OSError as exc:originals_verified[key]=None;originals_verified[key+'_error']=repr(exc)
                if originals_verified['checkpoint_sha256']!=BASE_SHA or originals_verified['data_manifest_sha256']!=DATA_SHA:
                    if failure is None:failure=dict(type='ReadonlyInputChanged',reason='Original checkpoint or R3A manifest changed',phase='final_readonly_check')
                if failure is not None:
                    atomic_json(background/'failure.json',failure)
                    (stage/'FAILED.md').write_text('# R3B background pipeline stopped with evidence\n\n'+
                        f"Phase: {failure['phase']}\n\n{failure['type']}: {failure['reason']}\n\n"+
                        'No automatic retry or epoch2. See background/failure.json and pipeline-state.json.\n',encoding='utf-8')
                atomic_json(state_path,pipeline)
                status.finish(failure is None,failure)
                completion=dict(kind='R3B-independent-first-epoch-completion.v1',run_id=config.get('run_id'),config_sha256=checked['config_sha256'],completed=failure is None,closed=True,
                    finished_at_utc=utc(),pid=os.getpid(),worker_wall_seconds=time.perf_counter()-began,failure=failure,
                    completed_phases=pipeline['completed_phases'],formal_progress=pipeline.get('formal_progress',status.value['formal']),
                    report_path=str(stage/('REPORT.md'if failure is None else'FAILED.md')),result_path=str(stage/'result.json'),
                    final_manifest_path=str(background/'final-manifest.json'),checkpoint_path=str(stage/'run/last.pt'),
                    status_path=str(background/'status.json'),heartbeat_path=str(background/'heartbeat.json'),
                    stdout_path=str(background/'worker.stdout.log'),stderr_path=str(background/'worker.stderr.log'),
                    original_inputs_verified=originals_verified,original6000_readonly_policy=True,new_physics_controls=0,heldout_accesses=0,next_epoch_started=False,
                    checkpoint_exists=(stage/'run/last.pt').is_file(),progress_path=str(stage/'run/progress.json'),
                    unsafe_checkpoint_path=str(stage/'run/unsafe_checkpoint.json')if(stage/'run/unsafe_checkpoint.json').exists()else None,
                    recovery_context=pipeline.get('recovery_context'),loader_after_model_restore=pipeline.get('loader_after_model_restore'),
                    recovery_policy='Only an explicitly authorized resume from the validated saved cursor; never automatic restart or epoch replay.',
                    callback_required=True,callback_destination_thread_id=PARENT_THREAD,callback_sent_by_worker=False,
                    callback_boundary='Offline worker writes local results only; app/tools must be available for the authorized completion monitor to send the real callback.',
                    after_callback='STOP waiting_user_approval; no epoch2, repair, resampling or server work')
                atomic_json(background/'completion.json',completion)
    return 0 if failure is None else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--check',action='store_true')
    parser.add_argument('--resume-authorized',action='store_true');args=parser.parse_args()
    os.chdir(ROOT)
    for key,value in(('SDL_VIDEODRIVER','dummy'),('SDL_AUDIODRIVER','dummy'),('PYGAME_HIDE_SUPPORT_PROMPT','1'),('PYTHONDONTWRITEBYTECODE','1')):os.environ.setdefault(key,value)
    if args.check:
        if args.execute:raise ValueError('Choose pure preflight or actual once execution')
        _,checked=preflight(args.config,args.resume_authorized);print(json.dumps(checked));return 0
    if not args.execute:parser.error('Explicit --execute or --check required')
    return run(args.config,args.resume_authorized)


if __name__=='__main__':raise SystemExit(main())
