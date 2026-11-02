"""Explicit R2C1 300-attempt/1200s subset of the existing global ledger.

No model/device is constructed by this module. The one-use caller starts one
CostLedger stage before model construction and owns final cleanup/finish.
"""
from pathlib import Path
import math
import time
from .native_rgb_stage_budget import (ModelWorkBudget,StorageBudget,
    native_storage_paths,storage_footprint)
from .native_rgb_training import BudgetExceeded

GLOBAL_ATTEMPTS_BEFORE=250
GLOBAL_SECONDS_BEFORE=868.484000000004
PRIOR_LEDGER_SHA256='2729b06c8e565618624c65e1835701f2a37eabbcc9da0722c5bccad8bd7c40e6'


class R2C1StorageBudget(StorageBudget):
    def __init__(self,paths,*,stage_root,source_files=(),**kwargs):
        self.stage_root=Path(stage_root).resolve()
        self.stage_source_files=tuple(Path(p).resolve() for p in source_files)
        self.stage_limit_bytes=256*1024**2
        self.nonstage_growth_reserve_bytes=1024**2
        self.frozen_nonstage_bytes=None
        super().__init__(paths,**kwargs)

    def check(self,*,force=False,write_reserve_bytes=0):
        full_scan=(force or write_reserve_bytes or self.last_checked is None or time.monotonic()-self.last_checked>=self.interval)
        report=super().check(force=force,write_reserve_bytes=write_reserve_bytes)
        if full_scan:self.last_checked=time.monotonic()
        # The stage is small; count actual files including temporary writes.
        stage_files_bytes=sum(p.stat().st_size for p in self.stage_root.rglob('*') if p.is_file())
        source_bytes=sum(p.stat().st_size for p in self.stage_source_files if p.is_file() and not p.is_relative_to(self.stage_root))
        stage_bytes=stage_files_bytes+source_bytes
        report.update(r2c1_artifact_bytes=stage_bytes,r2c1_stage_directory_bytes=stage_files_bytes,
            r2c1_external_new_source_bytes=source_bytes,r2c1_limit_bytes=self.stage_limit_bytes)
        if stage_bytes>self.stage_limit_bytes:raise BudgetExceeded('R2C1 persistent artifact ceiling256MiB reached')
        # All nonstage sources/artifacts are frozen; only the linked global
        # ledger and small current status/atomic metadata can grow (<=1MiB).
        # Reserve the ENTIRE stage cap up front, not only its current bytes.
        # This conservative bound is checked every call, including cached
        # full scans; new stage writes cannot outrun the shared reservation.
        nonstage_bytes=report['bytes']-stage_bytes
        if self.frozen_nonstage_bytes is None:self.frozen_nonstage_bytes=nonstage_bytes
        if full_scan and nonstage_bytes>self.frozen_nonstage_bytes+self.nonstage_growth_reserve_bytes:
            raise BudgetExceeded('Frozen nonstage footprint grew beyond its1MiB ledger/status allowance')
        # report['bytes'] may be cached, so use an allowance on the original
        # immutable footprint rather than subtracting a growing current stage.
        bound=self.frozen_nonstage_bytes+self.nonstage_growth_reserve_bytes+self.stage_limit_bytes+self.atomic_reserve_bytes
        report.update(frozen_nonstage_bytes=self.frozen_nonstage_bytes,
            full_stage_reserved_bytes=self.stage_limit_bytes,
            nonstage_growth_reserve_bytes=self.nonstage_growth_reserve_bytes,
            conservative_shared_upper_bound_bytes=bound,
            conservative_bound_checked_every_call=True,
            full_scan_interval_seconds=self.interval,full_scan_performed_now=bool(full_scan))
        if bound>self.max_bytes:raise BudgetExceeded('Insufficient shared headroom to reserve the entire R2C1 stage and atomic write')
        return report


class R2C1ModelWorkBudget(ModelWorkBudget):
    identity_kind='native-rgb-R2C1-one-run-model-work-budget.v1'

    def validate_local_caps(self,max_attempts,max_seconds,update_cutoff_seconds,tail_reserve_seconds):
        if (type(max_attempts)is not int or not 1<=max_attempts<=300
                or not 0<float(max_seconds)<=1200
                or not 0<=float(update_cutoff_seconds)<=min(900,float(max_seconds))
                or not math.isfinite(float(tail_reserve_seconds)) or tail_reserve_seconds<300):
            raise ValueError('R2C1 permits<=300attempts/1200s, cutoff<=900s and>=300s tail')

    def __init__(self,stage,*,max_attempts=300,max_seconds=1200.,update_cutoff_seconds=900.,tail_reserve_seconds=300.,storage=None):
        if storage is None:raise ValueError('R2C1 requires shared and per-stage storage accounting')
        super().__init__(stage,max_attempts=max_attempts,max_seconds=max_seconds,
            update_cutoff_seconds=update_cutoff_seconds,tail_reserve_seconds=tail_reserve_seconds,storage=storage)
        active=self.ledger.snapshot()['active_stage']
        if (abs(self.aggregate_seconds_before-GLOBAL_SECONDS_BEFORE)>1e-9
                or self.ledger.snapshot()['charged_updates']!=GLOBAL_ATTEMPTS_BEFORE
                or active['reserved_updates']!=0):
            raise ValueError('R2C1 must link the preserved250/868.484 prior ledger exactly once')

    def record_baseline(self,seconds,*,checkpoint_reserve_seconds=60.,prediction_factor=1.5,other_tail_seconds=45.):
        return super().record_baseline(seconds,checkpoint_reserve_seconds=checkpoint_reserve_seconds,
            prediction_factor=prediction_factor,other_tail_seconds=other_tail_seconds)

    def identity(self):
        result=super().identity()
        result.update(prior_global_attempts=GLOBAL_ATTEMPTS_BEFORE,prior_global_seconds=GLOBAL_SECONDS_BEFORE,
            prior_global_ledger_sha256=PRIOR_LEDGER_SHA256,
            time_scope='construction/load/originalH1H4/alltrainmicrobatches/finalH1H4/save/necessaryreload/failures/cleanup; one original externally-owned stage clock',
            original_global_caps_unchanged=True,no_automatic_retry_or_restart=True)
        return result


def storage_for_config(config):
    """Read explicit roots frozen by root before the only authorized launch."""
    repo=Path(config['repo_root']).resolve();stage=Path(config['output']).resolve()
    roots=[Path(p).resolve() for p in config['storage_roots']]
    if not roots or stage not in roots:raise ValueError('Stage root must be explicit in frozen storage roots')
    return R2C1StorageBudget(roots,stage_root=stage,source_files=config.get('new_stage_source_files',()),max_bytes=3*1024**3,
        atomic_reserve_bytes=256*1024**2,minimum_free_bytes=512*1024**2,interval_seconds=60.)


def make_budget(stage,config):
    """Caller has already charged/started exactly one global stage, no models yet."""
    if config.get('execution_authorized')is not True:raise PermissionError('R2C1 parent execution authorization required')
    caps=config['budget']
    if (caps['max_attempts'],caps['max_model_seconds'],caps['latest_new_attempt_seconds'],caps['tail_reserve_seconds'])!=(300,1200.,900.,300.):
        raise ValueError('Frozen R2C1 caps must be exactly300/1200/900/300')
    return R2C1ModelWorkBudget(stage,storage=storage_for_config(config))
