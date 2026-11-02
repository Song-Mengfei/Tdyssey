"""Public continuation contracts for the existing native-GPC training lineage.

Only stream scope, a post-cosine LR hold and durable stage accounting change.
No Torch/model is imported here. The caller restores the real full checkpoint
and owns training, AMP decisions and saved-array evaluation.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from .native_rgb_r2c0_data import (R2C0WindowOrder,canonical_hash,
    metadata_only_sampler_factory)
from .native_rgb_r2c1r1_accounting_v2 import RecordOnlyTimeStage
from .native_rgb_r2c2_inference import ExistingInferenceLedger
from .native_rgb_r2c1_budget import R2C1StorageBudget
from .native_rgb_stage_budget import UpdateLimitReached
from .native_rgb_training import atomic_json as ledger_atomic

TIME_POLICY='record_only_v2'
PRIOR_ATTEMPTS=550
PRIOR_SECONDS=1469.1250000000073
LINEAGE_START=300
LINEAGE_TARGET=6000
NEW_COMPLETED_TARGET=5700
PLANNED_DRAWS=96000


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class R2C3WindowOrder(R2C0WindowOrder):
    """The same continuing80/20 stream with an explicit96000-exposure scope.

    identity() deliberately stays the original distribution/pool identity:
    upstream per-pool RNG seeding depends on it. Versioned scope is separate,
    in continuation_identity()/state_dict(), so no old queue gets re-seeded.
    """
    def __init__(self,pools,seed=5478,*,max_draws=PLANNED_DRAWS,skip_extension_count=0):
        if (type(max_draws) is not int or type(skip_extension_count)is not int or skip_extension_count<0
                or max_draws!=PLANNED_DRAWS+16*skip_extension_count):
            raise ValueError('R2C3 exposure scope is96000 plus16 per recorded AMP skip')
        self.max_draws=max_draws;self.skip_extension_count=skip_extension_count
        self.last_skip_extension_cursor=None
        super().__init__(pools,seed)

    def continuation_identity(self):
        return dict(kind='R2C3-continuing-R2C0-stream-scope.v1',
            distribution_identity=self.identity(),legacy_frozen_exposures=4800,
            planned_exposures_without_skips=PLANNED_DRAWS,current_exposure_limit=self.max_draws,
            skip_extension_count=self.skip_extension_count,last_skip_extension_cursor=self.last_skip_extension_cursor,
            lineage_start_updates=LINEAGE_START,
            lineage_target_updates=LINEAGE_TARGET,effective_batch=16,
            slot_and_pool_rng='restore and continue unchanged; no reseed or4800-prefix replay',
            amp_skip_policy='finite overflow charges attempt/draw,not complete/LR; extend16 and continue; consecutive16 instability stop')

    def extend_after_amp_skip(self):
        """Exactly one extra batch per recorded finite AMP-overflow attempt."""
        if self.cursor<4816 or self.cursor%16 or self.last_skip_extension_cursor==self.cursor:
            raise ValueError('AMP extension requires a distinct completed sixteen-draw attempt boundary')
        self.skip_extension_count+=1;self.max_draws+=16;self.last_skip_extension_cursor=self.cursor
        return self.continuation_identity()

    def take(self,count):
        # This is the upstream draw loop; only its original4800 guard changes.
        if type(count)is not int or count<1 or self.cursor+count>self.max_draws:
            raise ValueError('R2C3 planned96000 TRAIN exposure scope reached')
        result=[]
        for _ in range(count):
            if self.position==480:self.cycle+=1;self.position=0;self.slot_order=self._slots()
            letter,group=self.slot_order[self.position];families=self.pools['groups'][letter][group]
            family=self._draw(sorted(families),['family',letter,group]);categories=families[family]
            category=self._draw(sorted(categories),['category',letter,group,family])
            result.append(int(self._draw(categories[category],['window',letter,group,family,category])))
            self.cursor+=1;self.position+=1
        return result

    def legacy_state_dict(self):
        """Exact old byte schema/semantics, including all RNG and queue state."""
        return R2C0WindowOrder.state_dict(self)

    def state_dict(self):
        state=self.legacy_state_dict();state.pop('state_sha256')
        state.update(kind='R2C3-draw-state.v1',continuation_identity=self.continuation_identity())
        state['state_sha256']=canonical_hash(state)
        return state

    def load_state_dict(self,state):
        if not isinstance(state,dict):raise ValueError('Serialized order state required')
        if state.get('kind')=='R2C0-draw-state.v1':
            # Use the original strict validator before copying actual restored
            # slot/per-pool RNG and queues. No new seed/state replaces them.
            old=R2C0WindowOrder(self.pools);old.load_state_dict(deepcopy(state))
            self.max_draws=PLANNED_DRAWS;self.skip_extension_count=0;self.last_skip_extension_cursor=None
            for key in ('pools','cursor','cycle','position','rng','queues','slot_order'):
                setattr(self,key,deepcopy(getattr(old,key)))
            if self.legacy_state_dict()!=state:raise ValueError('Legacy sampler restore changed serialized state')
            return
        value=deepcopy(state);digest=value.pop('state_sha256',None)
        scope=value.get('continuation_identity',{});skips=scope.get('skip_extension_count')
        limit=scope.get('current_exposure_limit');last_skip=scope.get('last_skip_extension_cursor')
        if (digest!=canonical_hash(value) or value.get('kind')!='R2C3-draw-state.v1'
                or value.get('identity')!=self.identity() or type(skips)is not int or skips<0
                or limit!=PLANNED_DRAWS+16*skips
                or type(value.get('cursor')) is not int or not 0<=value['cursor']<=limit
                or (skips==0 and last_skip is not None)
                or (skips>0 and (type(last_skip)is not int or last_skip<4816 or last_skip%16 or last_skip>value['cursor']))):
            raise ValueError('Continuation stream identity/scope/checksum differs')
        rebuilt=type(self)(self.pools,max_draws=limit,skip_extension_count=skips)
        rebuilt.last_skip_extension_cursor=last_skip
        if scope!=rebuilt.continuation_identity():raise ValueError('Continuation metadata/scope differs')
        if value['cursor']:rebuilt.take(value['cursor'])
        if rebuilt.state_dict()!=state:raise ValueError('Continuation slot/queue/RNG state differs from unchanged stream')
        for key in ('pools','cursor','cycle','position','rng','queues','slot_order','max_draws','skip_extension_count','last_skip_extension_cursor'):
            setattr(self,key,deepcopy(getattr(rebuilt,key)))


def make_training_order(dataset,restored_prepared_state):
    """Return a fresh compatible order; restore the *whole* PreparedInputs next.

    PreparedInputs requires cursor0 at construction. This function validates the
    saved complete-boundary snapshot without pre-advancing the returned order.
    Caller then uses unchanged PreparedInputs.load_state_dict(saved_snapshot).
    """
    state=deepcopy(restored_prepared_state)
    if not isinstance(state,dict):raise ValueError('Complete PreparedInputs snapshot required')
    value=deepcopy(state);digest=value.pop('state_sha256',None)
    if digest!=canonical_hash(value):raise ValueError('PreparedInputs snapshot checksum differs')
    if (state.get('kind')!='native-rgb-r2c0-prepared-array-interface.v1'
            or state.get('pending_indices') is not None or state.get('micro_cursor')!=0
            or type(state.get('delivered_draws')) is not int
            or state.get('order',{}).get('cursor')!=state.get('delivered_draws')):
        raise ValueError('Only a complete PreparedInputs boundary can continue')
    original=metadata_only_sampler_factory(dataset)
    order=R2C3WindowOrder(original.pools)
    probe=R2C3WindowOrder(original.pools);probe.load_state_dict(state['order'])
    if state['order']['kind']=='R2C0-draw-state.v1' and state['delivered_draws']!=4800:
        raise ValueError('Initial R2C3 resume must restore the actual4800-draw R1 state')
    order.restoration_audit=dict(kind='R2C3-explicit-prepared-order-migration.v1',
        prepared_state_sha256_before=state['state_sha256'],legacy_order_state_sha256=probe.legacy_state_dict()['state_sha256'],
        continuation_order_state_sha256=probe.state_dict()['state_sha256'],
        restored_cursor=probe.cursor,legacy_core_preserved=probe.legacy_state_dict()==state['order'] if state['order']['kind']=='R2C0-draw-state.v1' else True,
        distribution_identity_unchanged=probe.identity()==original.identity(),
        continuation_identity=probe.continuation_identity(),returned_cursor=0,
        caller_must_restore_full_PreparedInputs=True)
    return order


def _cosine_lr(base,eta_min,completed):
    return eta_min+(base-eta_min)*(1+math.cos(math.pi*min(completed,3000)/3000))/2


def advance_lr_after_success(scheduler,optimizer,new_lineage_update):
    """Successful update u used L(u-1); prepare L(u), clamped at3000.

    Before3000 call the restored native scheduler unchanged. At/after3000 keep
    eta_min and never call its step again. A public serialized continuation_state
    records true completed updates while native last_epoch remains3000 in hold.
    AMP-skip callers must not invoke this function.
    """
    if type(new_lineage_update)is not int or new_lineage_update<1:raise ValueError('True successful lineage update required')
    state=scheduler.state_dict();last=state.get('last_epoch')
    if state.get('T_max')!=3000 or state.get('eta_min')!=1e-5:
        raise ValueError('Restore original cosineT_max3000/eta_min1e-5 before continuing')
    bases=state.get('base_lrs');groups=optimizer.param_groups
    if not isinstance(bases,list) or len(bases)!=len(groups) or any(not math.isfinite(v) or v!=1e-4 for v in bases):
        raise ValueError('Original optimizer/scheduler base_lr1e-4 required')
    prior=state.get('continuation_state',getattr(scheduler,'continuation_state',None))
    completed=prior['completed_updates'] if prior is not None else last
    if type(completed)is not int or new_lineage_update!=completed+1:
        raise ValueError('LR advance must follow exactly one successful completed update')
    if last!=min(completed,3000):raise ValueError('Native scheduler step and true completion count disagree')
    used=[float(g['lr']) for g in groups]
    for lr,base in zip(used,bases):
        if not math.isclose(lr,_cosine_lr(base,1e-5,completed),rel_tol=1e-7,abs_tol=1e-12):
            raise ValueError('Optimizer LR before update differs from restored completed-update schedule')
    called=new_lineage_update<=3000
    if called:scheduler.step()
    else:
        for group in groups:group['lr']=1e-5
    after=scheduler.state_dict()
    if after['last_epoch']!=min(new_lineage_update,3000):raise ValueError('Native cosine step advanced past3000 or skipped')
    upcoming=[float(g['lr']) for g in groups]
    for lr,base in zip(upcoming,bases):
        if not math.isclose(lr,_cosine_lr(base,1e-5,new_lineage_update),rel_tol=1e-7,abs_tol=1e-12):
            raise ValueError('Next-update LR differs from clamped original schedule')
    continuation=dict(kind='R2C3-native-cosine-then-eta-min.v1',completed_updates=new_lineage_update,
        next_update=new_lineage_update+1,phase='native_cosine' if new_lineage_update<3000 else 'hold_eta_min',
        native_last_epoch=after['last_epoch'],T_max=3000,eta_min=1e-5,base_lrs=deepcopy(bases),
        lr_used_for_completed_update=used,lr_for_next_update=upcoming,native_scheduler_step_called=called,
        successful_update_semantics='update u uses L(u-1); after successfulu set L(u); u>=3000 nextLR eta_min')
    scheduler.continuation_state=deepcopy(continuation)
    return continuation


def validate_scope(config):
    if config.get('execution_authorized')is not True or config.get('is_proposal')is not False:
        raise PermissionError('Explicit continuation execution authorization required')
    if config.get('time_policy')!=TIME_POLICY or config.get('cap_time')is not None:
        raise ValueError('Only explicit record_only_v2 with null time limit is supported')
    for key in ('max_model_seconds','latest_new_attempt_seconds','tail_reserve_seconds','global_max_seconds'):
        if config.get(key) is not None:raise ValueError('Revoked model-time hard caps cannot return')
    if (config.get('resume_lineage_updates',LINEAGE_START)!=LINEAGE_START
            or config.get('target_lineage_updates',LINEAGE_TARGET)!=LINEAGE_TARGET
            or config.get('new_completed_updates',NEW_COMPLETED_TARGET)!=NEW_COMPLETED_TARGET):
        raise ValueError('This stage explicitly continues300 to6000 complete updates')
    if config.get('stop_on_amp_skip',False)is not False:raise ValueError('Ordinary finite AMP overflow must continue at reduced scale')
    if config.get('max_consecutive_amp_skips',16)!=16:raise ValueError('Declared consecutive16 overflow instability guard required')


def verify_closed_prior(config):
    validate_scope(config)
    path=Path(config['ledger_path']);prior=Path(config['prior_ledger_copy'])
    if _sha(path)!=config['prior_ledger_sha256'] or _sha(prior)!=config['prior_ledger_sha256']:
        raise ValueError('Actual C2 closed prior/copy SHA differs')
    value=_read(path)
    if (value!=_read(prior) or value.get('kind')!='native_gpc_rgb_cost.v2-record-only-time'
            or value.get('active_stage')is not None or value.get('max_seconds')is not None
            or value.get('charged_updates')!=PRIOR_ATTEMPTS or value.get('completed_updates')!=PRIOR_ATTEMPTS
            or abs(value.get('charged_seconds',-1)-PRIOR_SECONDS)>1e-8
            or config.get('prior_attempts')!=PRIOR_ATTEMPTS
            or config.get('prior_completed_updates',PRIOR_ATTEMPTS)!=PRIOR_ATTEMPTS
            or abs(config.get('prior_model_seconds',-1)-PRIOR_SECONDS)>1e-8):
        raise ValueError('Preserve actual closed C2 history550/1469.125 and all original stages')
    return value


class ExistingContinuationLedger(ExistingInferenceLedger):
    """Continue the actual v2 ledger with a5700-complete stage, no global cap."""
    def __init__(self,path,config,*,clock=None):
        self.path=Path(path).resolve();self.lock=self.path.with_name(self.path.name+'.lock')
        self.clock=time.monotonic if clock is None else clock
        self.max_updates=None;self.max_seconds=None;self.config=config
        if self.path!=Path(config['ledger_path']).resolve():raise ValueError('Explicit ledger path differs')
        verify_closed_prior(config)

    def start(self,phase):
        with self._locked():
            value=verify_closed_prior(self.config);token=uuid.uuid4().hex
            value.setdefault('scope_policy_updates',[]).append(dict(
                previous_global_max_updates=value.get('max_updates'),new_global_max_updates=None,
                reason='Explicit new6000-lineage observation scope supersedes old3000 proposal; not a human lifetime allowance',
                prior_ledger_sha256=self.config['prior_ledger_sha256']))
            value['max_updates']=None
            value['active_stage']=dict(token=token,phase=str(phase),pid=__import__('os').getpid(),
                started_unix=time.time(),seconds_before=value['charged_seconds'],observed_seconds=0.,
                reserved_updates=0,completed_updates=0,time_policy=TIME_POLICY,planned_attempts=None,
                planned_completed_updates=NEW_COMPLETED_TARGET,lineage_updates_before=LINEAGE_START,
                lineage_target=LINEAGE_TARGET,training_authorized=True,stop_on_amp_skip=False,
                skipped_updates=0,max_consecutive_amp_skips=16)
            ledger_atomic(self.path,value)
        return RecordOnlyTimeStage(self,token)

    def change(self,token,elapsed,*,reserve=False,complete=False,skip=False,skip_metadata=None,end=None):
        if not math.isfinite(elapsed)or elapsed<0:raise ValueError('Actual finite nonnegative elapsed statistics required')
        if sum((bool(reserve),bool(complete),bool(skip)))>1:raise ValueError('Reserve,complete and skip are distinct durable boundaries')
        with self._locked():
            value=_read(self.path);active=value['active_stage']
            if active is None or active['token']!=token:raise RuntimeError('Continuation stage ownership differs')
            if active.get('planned_completed_updates')!=NEW_COMPLETED_TARGET:raise ValueError('Explicit continuation stage scope changed')
            elapsed=max(elapsed,active['observed_seconds']);active['observed_seconds']=elapsed
            value['charged_seconds']=active['seconds_before']+elapsed
            if reserve:
                if active['completed_updates']>=NEW_COMPLETED_TARGET:
                    raise UpdateLimitReached('Current continuation reached5700 new complete updates; await next phase')
                if active['reserved_updates']!=active['completed_updates']+active['skipped_updates']:
                    raise RuntimeError('Uncompleted attempt requires stop/review, never automatic retry')
                active['reserved_updates']+=1;value['charged_updates']+=1
            if complete:
                if active['reserved_updates']!=active['completed_updates']+active['skipped_updates']+1:
                    raise RuntimeError('Successful update requires exactly one durable pending reservation')
                active['completed_updates']+=1;value['completed_updates']+=1
            if skip:
                if active['reserved_updates']!=active['completed_updates']+active['skipped_updates']+1:
                    raise RuntimeError('AMP skip requires exactly one durable pending reservation')
                active['skipped_updates']+=1
                active['last_amp_skip']=deepcopy(skip_metadata or {})
            if end is not None:
                active['status']=str(end);active['finished_unix']=time.time()
                value['stages'].append(active);value['active_stage']=None
            ledger_atomic(self.path,value)
            return value


class ContinuationModelWorkBudget:
    def __init__(self,stage,*,config,storage):
        validate_scope(config)
        if storage is None:raise ValueError('Actual shared/stage storage guard required')
        self.stage=stage;self.ledger=stage.ledger;self.storage=storage;self.started=stage.started
        self.max_attempts=None;self.max_seconds=None;self.update_cutoff=None;self.latest_update_seconds=None
        self.tail_reserve_seconds=None;self.time_policy=TIME_POLICY;self.stop_on_amp_skip=False
        self.closed=False;self.finished_elapsed=None;self.stop_reason=None
        prior=_read(config['prior_ledger_copy']);current=self.ledger.snapshot();active=current['active_stage']
        if (_sha(config['prior_ledger_copy'])!=config['prior_ledger_sha256'] or active is None
                or active['token']!=stage.token or active['reserved_updates']!=0 or active['completed_updates']!=0
                or current['charged_updates']!=PRIOR_ATTEMPTS or current['completed_updates']!=PRIOR_ATTEMPTS
                or abs(active['seconds_before']-PRIOR_SECONDS)>1e-8 or current['stages']!=prior['stages']):
            raise ValueError('Continuation budget must preserve closed historical costs/stages exactly')

    @property
    def elapsed(self):return self.finished_elapsed if self.closed else self.stage.elapsed()
    @property
    def effective_time_cap(self):return None
    def heartbeat(self):
        if self.closed:raise RuntimeError('Continuation accounting already closed')
        result=self.stage.heartbeat();self.storage.check();return result
    def require_operation(self,estimated_seconds=0.,label='model operation'):
        if not math.isfinite(estimated_seconds)or estimated_seconds<0:raise ValueError('Finite estimate metadata required')
        return self.heartbeat()
    def reserve_update(self):
        self.heartbeat()
        return self.stage.reserve_update()
    def complete_update(self):return self.stage.complete_update()
    def record_amp_skip(self,**metadata):
        self.heartbeat()
        return self.ledger.change(self.stage.token,self.stage.elapsed(),skip=True,skip_metadata=metadata)
    def before_checkpoint(self,path,*,estimated_seconds=30.):
        self.require_operation(estimated_seconds,'checkpoint');return self.storage.check(force=True,write_reserve_bytes=256*1024**2)
    def identity(self):
        return dict(kind='R2C3-record-only-complete-update-scope.v1',time_policy=TIME_POLICY,
            hard_time_limits_enabled=False,user_set_model_time_budget=False,stage_max_attempts=None,
            stage_max_model_work_seconds=None,aggregate_max_attempts=None,aggregate_max_seconds=None,
            lineage_updates_before=LINEAGE_START,lineage_target_updates=LINEAGE_TARGET,
            new_completed_updates_target=NEW_COMPLETED_TARGET,stop_on_amp_skip=False,
            finite_amp_overflow_continues=True,max_consecutive_amp_skips=16,
            stage_scope_is_not_user_lifetime_limit=True,automatic_retry=False,ledger_path=str(self.ledger.path))
    def report(self):
        return dict(**self.identity(),elapsed_model_work_seconds=self.elapsed,closed=self.closed,
            stop_reason=self.stop_reason,ledger=self.ledger.snapshot())
    def finish(self,status='complete'):
        if self.closed:return self.ledger.snapshot()
        result=self.stage.finish(status);self.finished_elapsed=result['stages'][-1]['observed_seconds'];self.closed=True
        return result


def storage_for_config(config):
    stage=Path(config['output']).resolve();roots=[Path(p).resolve()for p in config['storage_roots']]
    if stage not in roots or config.get('stage_artifact_cap_bytes')!=400*1024**2:
        raise ValueError('Explicit C3 root and400MiB stage artifact scope required')
    storage=R2C1StorageBudget(roots,stage_root=stage,source_files=config.get('new_stage_source_files',()),
        max_bytes=3*1024**3,atomic_reserve_bytes=256*1024**2,minimum_free_bytes=512*1024**2,interval_seconds=60.)
    storage.stage_limit_bytes=400*1024**2
    return storage


def create_ledger(config,*,clock=None):return ExistingContinuationLedger(config['ledger_path'],config,clock=clock)
def make_budget(stage,config):return ContinuationModelWorkBudget(stage,config=config,storage=storage_for_config(config))
