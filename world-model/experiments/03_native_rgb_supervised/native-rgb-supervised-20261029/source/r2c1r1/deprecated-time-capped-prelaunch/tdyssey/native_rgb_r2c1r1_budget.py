"""Explicit one corrective restart using only the original round's remainder."""
from pathlib import Path
import hashlib,json,math
from .native_rgb_stage_budget import ModelWorkBudget
from .native_rgb_r2c1_budget import R2C1StorageBudget
from .native_rgb_training import BudgetExceeded

EXPECTED_PRIOR_ATTEMPTS=250
EXPECTED_PRIOR_SECONDS=895.8430000000008
EXPECTED_PRIOR_SHA='517e9a6e46c3d56d33ea3899037f910090ff749020da892cb2ea4fff9200422e'
FAILED_MODEL_SECONDS=27.35899999999674
RESTART_CAP_SECONDS=1172.641
RESTART_CUTOFF_SECONDS=872.641
CAPS=dict(max_attempts=300,max_model_seconds=RESTART_CAP_SECONDS,
          latest_new_attempt_seconds=RESTART_CUTOFF_SECONDS,tail_reserve_seconds=300.)

def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_authorization(config):
    if config.get('execution_authorized')is not True or config.get('is_proposal')is not False:
        raise PermissionError('Explicit parent R2C1R1 corrective execution authorization required')
    if config.get('budget')!=CAPS:raise ValueError('Only parent remaining300/1172.641/872.641/300 caps are authorized')
    path=Path(config['parent_authorization'])
    if _sha(path)!=config['parent_authorization_sha256']:raise ValueError('Parent authorization bytes changed')
    a=_read(path)
    exact=dict(parent_authorized=True,remaining_this_round_model_seconds_cap=RESTART_CAP_SECONDS,
        remaining_this_round_optimizer_attempts_cap=300,latest_new_attempt_seconds_in_restart=RESTART_CUTOFF_SECONDS,
        minimum_tail_seconds=300,combined_R2C1_plus_restart_model_seconds_cap=1200,
        global_prior_updates=EXPECTED_PRIOR_ATTEMPTS,global_prior_ledger_sha256=EXPECTED_PRIOR_SHA,
        global_caps_updates_seconds=[3000,5400],automatic_further_restart=False,
        frozen_data_draws_evaluation_and_model_math_unchanged=True)
    if any(a.get(k)!=v for k,v in exact.items()):raise ValueError('Parent authorized identity/caps differ')
    if abs(a['global_prior_model_seconds']-EXPECTED_PRIOR_SECONDS)>1e-9 or abs(a['previous_launch_model_seconds']-FAILED_MODEL_SECONDS)>1e-9:
        raise ValueError('Parent prior costs cannot be refunded or reset')
    if (config.get('prior_ledger_sha256')!=EXPECTED_PRIOR_SHA
            or config.get('prior_attempts')!=EXPECTED_PRIOR_ATTEMPTS
            or abs(config.get('prior_model_seconds',-1)-EXPECTED_PRIOR_SECONDS)>1e-9):
        raise ValueError('Configuration must link the explicit new prior250/895.843')
    if FAILED_MODEL_SECONDS+RESTART_CAP_SECONDS>1200.:raise ValueError('Combined original round exceeds1200')
    return a


def verify_closed_prior(config):
    """Must run before CostLedger.start changes the live ledger bytes."""
    validate_authorization(config)
    for key in ('ledger_path','prior_ledger_copy'):
        if _sha(config[key])!=EXPECTED_PRIOR_SHA:raise ValueError('Prior global ledger/copy bytes differ')
    p=_read(config['prior_ledger_copy']);current=_read(config['ledger_path'])
    if (current!=p or p['active_stage']is not None or p['charged_updates']!=250 or p['completed_updates']!=250
            or abs(p['charged_seconds']-EXPECTED_PRIOR_SECONDS)>1e-9
            or p['max_updates']!=3000 or p['max_seconds']!=5400.):
        raise ValueError('Closed prior counters/history/caps must remain intact')
    return p


class R2C1R1ModelWorkBudget(ModelWorkBudget):
    identity_kind='native-rgb-R2C1R1-remaining-one-restart-budget.v1'

    def validate_local_caps(self,max_attempts,max_seconds,update_cutoff_seconds,tail_reserve_seconds):
        if (type(max_attempts)is not int or not 1<=max_attempts<=300
                or not 0<float(max_seconds)<=RESTART_CAP_SECONDS
                or not 0<=float(update_cutoff_seconds)<=min(RESTART_CUTOFF_SECONDS,float(max_seconds))
                or not math.isfinite(float(tail_reserve_seconds)) or tail_reserve_seconds<300):
            raise ValueError('Remaining budget permits<=300/1172.641, cutoff<=872.641 and>=300tail')

    def __init__(self,stage,*,config,max_attempts=300,max_seconds=RESTART_CAP_SECONDS,
                 update_cutoff_seconds=RESTART_CUTOFF_SECONDS,tail_reserve_seconds=300.,storage=None):
        validate_authorization(config)
        if storage is None:raise ValueError('Shared oldfailed+new storage accounting required')
        if _sha(config['prior_ledger_copy'])!=EXPECTED_PRIOR_SHA:raise ValueError('Preserved prior copy changed')
        prior=_read(config['prior_ledger_copy'])
        super().__init__(stage,max_attempts=max_attempts,max_seconds=max_seconds,
            update_cutoff_seconds=update_cutoff_seconds,tail_reserve_seconds=tail_reserve_seconds,storage=storage)
        current=self.ledger.snapshot();active=current['active_stage']
        if (abs(self.aggregate_seconds_before-EXPECTED_PRIOR_SECONDS)>1e-9 or current['charged_updates']!=250
                or current['completed_updates']!=250 or active['reserved_updates']!=0
                or active['completed_updates']!=0 or current['stages']!=prior['stages']):
            raise ValueError('Corrective stage must link exact prior costs/history once; no reset')

    def record_baseline(self,seconds,*,checkpoint_reserve_seconds=60.,prediction_factor=1.5,other_tail_seconds=45.):
        return super().record_baseline(seconds,checkpoint_reserve_seconds=checkpoint_reserve_seconds,
            prediction_factor=prediction_factor,other_tail_seconds=other_tail_seconds)

    def heartbeat(self):
        value=super().heartbeat()
        if FAILED_MODEL_SECONDS+self.elapsed>1200.:raise BudgetExceeded('Original failed+restart round1200s cap')
        return value

    def identity(self):
        r=super().identity();r.update(prior_global_attempts=250,prior_global_seconds=EXPECTED_PRIOR_SECONDS,
            prior_global_ledger_sha256=EXPECTED_PRIOR_SHA,previous_failed_launch_seconds=FAILED_MODEL_SECONDS,
            combined_original_round_seconds_cap=1200.,automatic_further_restart=False,
            time_scope='all construct/load/baseline+CPUscore/train/save/reload/final+CPUscore/failure/cleanup in one corrective clock; prior27.359 remains charged')
        return r

    def report(self):
        r=super().report();r['combined_failed_plus_restart_model_seconds']=FAILED_MODEL_SECONDS+self.elapsed
        return r


def storage_for_config(config):
    stage=Path(config['output']).resolve();roots=[Path(p).resolve() for p in config['storage_roots']]
    if not roots or stage not in roots:raise ValueError('New restart stage must be explicit in storage roots')
    return R2C1StorageBudget(roots,stage_root=stage,source_files=config.get('new_stage_source_files',()),
        max_bytes=3*1024**3,atomic_reserve_bytes=256*1024**2,minimum_free_bytes=512*1024**2,interval_seconds=60.)


def make_budget(stage,config):
    validate_authorization(config)
    return R2C1R1ModelWorkBudget(stage,config=config,storage=storage_for_config(config))
