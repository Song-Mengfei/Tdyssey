"""Explicit record-only model time, with unchanged durable attempt accounting.

No huge/fake time ceiling, shifted clock or legacy CostLedger/ModelWorkBudget
time guards. All prior costs/history are preserved in the versioned migration.
"""
from pathlib import Path
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,math,os,time,uuid
from .native_rgb_r2c1_budget import R2C1StorageBudget
from .native_rgb_stage_budget import UpdateLimitReached

EXPECTED_PRIOR_ATTEMPTS=250
EXPECTED_PRIOR_SECONDS=895.8430000000008
EXPECTED_PRIOR_SHA='517e9a6e46c3d56d33ea3899037f910090ff749020da892cb2ea4fff9200422e'
FAILED_MODEL_SECONDS=27.35899999999674
TIME_POLICY='record_only_v2'
LEDGER_KIND='native_gpc_rgb_cost.v2-record-only-time'
CAPS=dict(max_attempts=300,max_model_seconds=None,latest_new_attempt_seconds=None,tail_reserve_seconds=None)

def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def _sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def _atomic(path,value):
    # Public persistence helper only (UUID temp/fsync/replace and a bounded
    # Windows PermissionError I/O retry). It has no model-work time policy.
    from .native_rgb_training import atomic_json
    return atomic_json(path,value)

def validate_authorization(config):
    if config.get('execution_authorized')is not True or config.get('is_proposal')is not False:raise PermissionError('Explicit diagnostic execution authorization required')
    if config.get('time_policy')!=TIME_POLICY or config.get('budget')!=CAPS or config.get('cap_time')is not None:raise ValueError('Explicit record-only None time fields required, never a fake large limit')
    if _sha(config['parent_authorization'])!=config['parent_authorization_sha256']:raise ValueError('Original restart scope changed')
    a=_read(config['parent_authorization'])
    if not a['parent_authorized'] or a['global_prior_ledger_sha256']!=EXPECTED_PRIOR_SHA or not a['frozen_data_draws_evaluation_and_model_math_unchanged']:raise ValueError('Restart scope/input identity differs')
    if _sha(config['time_policy_file'])!=config['time_policy_sha256']:raise ValueError('Corrected time policy changed')
    p=_read(config['time_policy_file'])
    if (p.get('time_policy')!=TIME_POLICY or p.get('hard_time_limits_enabled')is not False or p.get('user_set_model_time_budget')is not False or p.get('current_batch_planned_attempts')!=300 or p.get('unlimited_automatic_training')is not False or any(p.get(k)is not None for k in ['max_model_seconds','latest_new_attempt_seconds','tail_reserve_seconds','global_model_seconds_limit'])):raise ValueError('Human correction must disable every time stopping rule')
    if _sha(p['correction_file'])!=p['correction_sha256']:raise ValueError('Policy correction source changed')
    if config.get('prior_ledger_sha256')!=EXPECTED_PRIOR_SHA or config.get('prior_attempts')!=250 or abs(config.get('prior_model_seconds',-1)-EXPECTED_PRIOR_SECONDS)>1e-9:raise ValueError('Actual historical fees cannot be reset')
    return p

def verify_closed_prior(config):
    validate_authorization(config)
    if any(_sha(config[k])!=EXPECTED_PRIOR_SHA for k in ['ledger_path','prior_ledger_copy']):raise ValueError('Prior/copy SHA changed')
    p=_read(config['prior_ledger_copy']);v=_read(config['ledger_path'])
    if v!=p or p['active_stage']is not None or p['charged_updates']!=250 or p['completed_updates']!=250 or abs(p['charged_seconds']-EXPECTED_PRIOR_SECONDS)>1e-9:raise ValueError('Prior counters/history must be intact')
    return p

class RecordOnlyTimeLedger:
    def __init__(self,path,config,*,clock=None):
        self.path=Path(path).resolve();self.lock=self.path.with_name(self.path.name+'.lock');self.clock=time.monotonic if clock is None else clock
        self.max_updates=3000;self.max_seconds=None;self.config=config
        policy=validate_authorization(config)
        with self._locked():
            old=verify_closed_prior(config);v=deepcopy(old);v['kind']=LEDGER_KIND
            v['legacy_budget_metadata']=dict(previous_kind=old['kind'],previous_max_seconds=old.get('max_seconds'),previous_max_updates=old.get('max_updates'),limits_were_supervisor_plan_not_user_time_budget=True)
            v['max_seconds']=None
            v['time_policy']=dict(version=TIME_POLICY,hard_time_limits_enabled=False,user_set_model_time_budget=False,correction_file=policy['correction_file'],correction_sha256=policy['correction_sha256'],policy_file=str(Path(config['time_policy_file']).resolve()),policy_sha256=config['time_policy_sha256'])
            v.setdefault('policy_migrations',[]).append(dict(at_utc=datetime.now(timezone.utc).isoformat(),prior_ledger_sha256=EXPECTED_PRIOR_SHA,to_kind=LEDGER_KIND,charged_counters_and_stage_history_unchanged=True,reason='User did not set a model time budget; supervisor time hard limits revoked before restart'))
            _atomic(self.path,v)
    @contextmanager
    def _locked(self):
        fd=os.open(self.lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        try:os.write(fd,str(os.getpid()).encode());yield
        finally:os.close(fd);self.lock.unlink(missing_ok=True)
    def snapshot(self):
        with self._locked():return _read(self.path)
    def start(self,phase):
        with self._locked():
            v=_read(self.path)
            if v['active_stage']is not None:raise RuntimeError('Existing active accounting stage; no replay')
            token=uuid.uuid4().hex;v['active_stage']=dict(token=token,phase=str(phase),pid=os.getpid(),started_unix=time.time(),seconds_before=v['charged_seconds'],observed_seconds=0.,reserved_updates=0,completed_updates=0,time_policy=TIME_POLICY,planned_attempts=300);_atomic(self.path,v)
        return RecordOnlyTimeStage(self,token)
    def change(self,token,elapsed,*,reserve=False,complete=False,end=None):
        if not math.isfinite(elapsed)or elapsed<0:raise ValueError('Finite nonnegative actual elapsed statistics required')
        with self._locked():
            v=_read(self.path);a=v['active_stage']
            if a is None or a['token']!=token:raise RuntimeError('Accounting ownership differs')
            elapsed=max(elapsed,a['observed_seconds']);a['observed_seconds']=elapsed;v['charged_seconds']=a['seconds_before']+elapsed
            if reserve:
                if a['reserved_updates']>=300:raise UpdateLimitReached('Diagnostic planned300 attempts; no automatic continuation')
                if v['charged_updates']>=self.max_updates:raise UpdateLimitReached('Existing staged-plan attempt scope reached; not a user time budget')
                a['reserved_updates']+=1;v['charged_updates']+=1
            if complete:
                if a['completed_updates']>=a['reserved_updates']:raise RuntimeError('Completion requires durable reservation')
                a['completed_updates']+=1;v['completed_updates']+=1
            if end is not None:a['status']=str(end);a['finished_unix']=time.time();v['stages'].append(a);v['active_stage']=None
            _atomic(self.path,v);return v

class RecordOnlyTimeStage:
    def __init__(self,ledger,token):self.ledger=ledger;self.token=token;self.started=ledger.clock();self.closed=False
    def elapsed(self):return max(0.,self.ledger.clock()-self.started)
    def heartbeat(self):return self.ledger.change(self.token,self.elapsed())
    def reserve_update(self):return self.ledger.change(self.token,self.elapsed(),reserve=True)
    def complete_update(self):return self.ledger.change(self.token,self.elapsed(),complete=True)
    def finish(self,status='complete'):
        if self.closed:return self.ledger.snapshot()
        v=self.ledger.change(self.token,self.elapsed(),end=status);self.closed=True;return v

class R2C1R1ModelWorkBudget:
    def __init__(self,stage,*,config,storage):
        validate_authorization(config)
        if storage is None:raise ValueError('Shared/stage storage guard required')
        self.stage=stage;self.ledger=stage.ledger;self.storage=storage;self.started=stage.started;self.max_attempts=300;self.max_seconds=None;self.time_policy=TIME_POLICY
        self.latest_update_seconds=None;self.tail_reserve_seconds=None;self.update_cutoff=None;self.closed=False;self.finished_elapsed=None;self.baseline_seconds=None;self.stop_reason=None
        prior=_read(config['prior_ledger_copy']);v=self.ledger.snapshot();a=v['active_stage']
        if _sha(config['prior_ledger_copy'])!=EXPECTED_PRIOR_SHA or a is None or a['token']!=stage.token or a['reserved_updates']!=0 or v['charged_updates']!=250 or v['completed_updates']!=250 or abs(a['seconds_before']-EXPECTED_PRIOR_SECONDS)>1e-9 or v['stages']!=prior['stages']:raise ValueError('Record-only stage must retain actual prior fees/history')
    @property
    def elapsed(self):return self.finished_elapsed if self.closed else self.stage.elapsed()
    @property
    def effective_time_cap(self):return None
    def heartbeat(self):
        if self.closed:raise RuntimeError('Accounting closed')
        v=self.stage.heartbeat();self.storage.check();return v
    def record_baseline(self,seconds,**unused_time_planning):
        if not math.isfinite(seconds)or seconds<0:raise ValueError('Finite measured baseline duration required')
        self.baseline_seconds=float(seconds);return None
    def require_operation(self,estimated_seconds=0.,label='model operation'):
        if not math.isfinite(estimated_seconds)or estimated_seconds<0:raise ValueError('Finite estimate metadata required')
        return self.heartbeat() # No comparison between elapsed/estimate and a cap.
    def reserve_update(self):
        v=self.heartbeat()
        if v['active_stage']['reserved_updates']>=300:self.stop_reason='planned_batch300_attempts';raise UpdateLimitReached(self.stop_reason)
        return self.stage.reserve_update()
    def complete_update(self):return self.stage.complete_update()
    def before_checkpoint(self,path,*,estimated_seconds=30.):
        self.require_operation(estimated_seconds,'checkpoint storage precheck');return self.storage.check(force=True,write_reserve_bytes=256*1024**2)
    def identity(self):
        return dict(kind='native-R2C1R1-record-only-model-work.v2',time_policy=TIME_POLICY,hard_time_limits_enabled=False,user_set_model_time_budget=False,stage_planned_attempts=300,attempt_scope_is_diagnostic_not_user_lifetime_budget=True,stage_max_attempts=300,stage_max_model_work_seconds=None,aggregate_max_seconds=None,latest_new_update_seconds=None,initial_tail_reserve_seconds=None,ledger_path=str(self.ledger.path),prior_global_attempts=250,prior_global_seconds=EXPECTED_PRIOR_SECONDS,prior_global_ledger_sha256=EXPECTED_PRIOR_SHA,previous_failed_launch_seconds=FAILED_MODEL_SECONDS,automatic_further_restart=False,time_scope='construct/load/baseline+CPUscore/train/save/reload/final+CPUscore/failures/cleanup; monotonic statistics only')
    def report(self):
        return dict(**self.identity(),elapsed_model_work_seconds=self.elapsed,measured_baseline_seconds=self.baseline_seconds,effective_time_cap_seconds=None,effective_update_cutoff_seconds=None,reserved_tail_seconds=None,overrun_seconds=None,combined_failed_plus_restart_model_seconds=FAILED_MODEL_SECONDS+self.elapsed,closed=self.closed,stop_reason=self.stop_reason,ledger=self.ledger.snapshot())
    def finish(self,status='complete'):
        if self.closed:return self.ledger.snapshot()
        v=self.stage.finish(status);self.finished_elapsed=v['stages'][-1]['observed_seconds'];self.closed=True;return v

def storage_for_config(config):
    stage=Path(config['output']).resolve();roots=[Path(p).resolve()for p in config['storage_roots']]
    if stage not in roots:raise ValueError('New stage must be in storage roots')
    return R2C1StorageBudget(roots,stage_root=stage,source_files=config.get('new_stage_source_files',()),max_bytes=3*1024**3,atomic_reserve_bytes=256*1024**2,minimum_free_bytes=512*1024**2,interval_seconds=60.)
def create_ledger(config,*,clock=None):return RecordOnlyTimeLedger(config['ledger_path'],config,clock=clock)
def make_budget(stage,config):return R2C1R1ModelWorkBudget(stage,config=config,storage=storage_for_config(config))
