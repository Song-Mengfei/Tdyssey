"""Stage3 wall-clock/attempt caps around one aggregate native model-work stage.

No model imports. The caller starts/finishes the underlying CostLedger stage so
loading, both native predictions, losses, updates, saving and cleanup share one
clock. Update reservations are durable and never refunded.
"""

import math
from pathlib import Path
import shutil
import stat
import time

from .native_rgb_training import BudgetExceeded


class UpdateLimitReached(BudgetExceeded):
    updates_only_stop=True


def native_storage_paths(repo,supervised_root,extra_paths=()):
    repo=Path(repo).resolve()
    paths=[repo/"artifacts/native-rgb-20261028",Path(supervised_root).resolve()]
    for pattern in ("tdyssey/native_rgb*.py","tdyssey/__pycache__/native_rgb*.pyc","scripts/native_rgb*.py",
                    "scripts/__pycache__/native_rgb*.pyc","tests/test_native_rgb*.py","configs/native_rgb*.json",
                    "run_native_rgb*","requirements/native-rgb*","docs/NATIVE_RGB_WORLD_MODEL.md"):
        paths.extend(repo.glob(pattern))
    paths.extend(Path(path).resolve() for path in extra_paths)
    return paths


def storage_footprint(paths):
    """Unique regular-file bytes across explicit roots, including existing temps."""
    paths=tuple(paths);seen=set();files=0;size=0;visited=set();pending=[Path(path).absolute() for path in paths]
    while pending:
        path=pending.pop()
        if not path.exists():continue
        details=path.lstat()
        if path.is_symlink() or getattr(details,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",1024):
            raise ValueError("Storage budget roots may not hide files behind links/reparse points")
        if path.is_dir():
            key=str(path)
            if key in visited:continue
            visited.add(key)
            pending.extend(path.iterdir())
        elif path.is_file():
            key=(details.st_dev,details.st_ino) if details.st_ino else str(path)
            if key in seen:continue
            seen.add(key);files+=1;size+=details.st_size
    return dict(bytes=size,files=files,roots=[str(Path(path).resolve()) for path in paths],existing_atomic_temps_included=True)


class StorageBudget:
    def __init__(self,paths,*,max_bytes=3*1024**3,atomic_reserve_bytes=256*1024**2,minimum_free_bytes=256*1024**2,interval_seconds=5.):
        self.paths=tuple(Path(path).resolve() for path in paths)
        if not self.paths or type(max_bytes) is not int or not 0<max_bytes<=3*1024**3:
            raise ValueError("Explicit old/new/source/cache roots and<=3GiB ceiling required")
        if atomic_reserve_bytes<0 or minimum_free_bytes<0 or interval_seconds<0:raise ValueError("Invalid storage reserve")
        self.max_bytes=max_bytes;self.atomic_reserve_bytes=int(atomic_reserve_bytes);self.minimum_free_bytes=int(minimum_free_bytes)
        self.interval=float(interval_seconds);self.last_checked=None;self.last_report=None

    def check(self,*,force=False,write_reserve_bytes=0):
        now=time.monotonic()
        if not force and not write_reserve_bytes and self.last_checked is not None and now-self.last_checked<self.interval:return self.last_report
        report=storage_footprint(self.paths);reserve=max(self.atomic_reserve_bytes,int(write_reserve_bytes))
        report.update(max_bytes=self.max_bytes,reserved_atomic_bytes=reserve,projected_bytes=report["bytes"]+reserve)
        if report["projected_bytes"]>self.max_bytes:raise BudgetExceeded("Combined old/new native source/cache plus atomic checkpoint reserve exceeds3GiB")
        anchors={path.anchor for path in self.paths}
        for anchor in anchors:
            if shutil.disk_usage(anchor).free<max(reserve,self.minimum_free_bytes):raise BudgetExceeded("Insufficient free disk for atomic complete checkpoint")
        self.last_checked=now;self.last_report=report;return report


class ModelWorkBudget:
    """Wrap exactly one externally owned BudgetStage; never start another stage."""
    identity_kind="native-rgb-stage3-model-work-budget.v1"

    def validate_local_caps(self,max_attempts,max_seconds,update_cutoff_seconds,tail_reserve_seconds):
        if type(max_attempts) is not int or not 1<=max_attempts<=200 or not 0<float(max_seconds)<=600:
            raise ValueError("Stage3 cap is<=200attempts and<=600all-model-work seconds")
        if not 0<=float(update_cutoff_seconds)<=min(420,float(max_seconds)) or tail_reserve_seconds<0:
            raise ValueError("No new update after420seconds or earlier reserved tail")

    def __init__(self,stage,*,max_attempts=200,max_seconds=600.,update_cutoff_seconds=420.,tail_reserve_seconds=180.,storage=None):
        if stage.ledger.max_updates!=3000 or stage.ledger.max_seconds!=5400.:
            raise ValueError("Stage3 must retain aggregate3000attempts/5400seconds")
        self.validate_local_caps(max_attempts,max_seconds,update_cutoff_seconds,tail_reserve_seconds)
        self.stage=stage;self.ledger=stage.ledger;self.started=stage.started;self.max_attempts=max_attempts
        active=self.ledger.snapshot()["active_stage"]
        if active is None or active["token"]!=stage.token:raise ValueError("External stage ownership differs")
        self.aggregate_seconds_before=float(active["seconds_before"])
        self.max_seconds=float(max_seconds);self.latest_update_seconds=float(update_cutoff_seconds)
        self.tail_reserve_seconds=float(tail_reserve_seconds);self.initial_tail_reserve_seconds=float(tail_reserve_seconds);self.storage=storage
        self.baseline_seconds=None;self.closed=False;self.finished_elapsed=None;self.stop_reason=None

    @property
    def elapsed(self):return self.finished_elapsed if self.closed else max(0.,time.monotonic()-self.started)

    @property
    def effective_time_cap(self):return min(self.max_seconds,self.ledger.max_seconds-self.aggregate_seconds_before)

    @property
    def update_cutoff(self):return max(0.,min(self.latest_update_seconds,self.effective_time_cap-self.tail_reserve_seconds))

    def heartbeat(self):
        if self.closed:raise RuntimeError("Model-work budget closed")
        value=self.stage.heartbeat()
        if self.storage is not None:self.storage.check()
        if self.elapsed>=self.effective_time_cap:raise BudgetExceeded("Stage3 all-model-work or remaining aggregate time cap reached")
        return value

    def record_baseline(self,seconds,*,checkpoint_reserve_seconds=30.,prediction_factor=1.5,other_tail_seconds=30.):
        if not math.isfinite(seconds) or seconds<0:raise ValueError("Finite measured native baseline duration required")
        self.baseline_seconds=float(seconds)
        self.tail_reserve_seconds=max(self.tail_reserve_seconds,prediction_factor*seconds+checkpoint_reserve_seconds+other_tail_seconds)
        return self.update_cutoff

    def require_operation(self,estimated_seconds=0.,label="model operation"):
        self.heartbeat()
        if not math.isfinite(estimated_seconds) or estimated_seconds<0:raise ValueError("Finite operation estimate required")
        if self.elapsed+estimated_seconds>=self.effective_time_cap:raise BudgetExceeded("Stage3 insufficient reserved time for "+label)

    def reserve_update(self):
        value=self.heartbeat();active=value["active_stage"]
        if active["reserved_updates"]>=self.max_attempts:
            self.stop_reason="stage_optimizer_attempt_cap";raise UpdateLimitReached(self.stop_reason)
        if self.elapsed>=self.update_cutoff:
            self.stop_reason="stage_gradient_cutoff_with_reserved_tail";raise UpdateLimitReached(self.stop_reason)
        return self.stage.reserve_update()

    def complete_update(self):return self.stage.complete_update()

    def before_checkpoint(self,path,*,estimated_seconds=30.):
        self.require_operation(estimated_seconds,"complete atomic checkpoint")
        path=Path(path);reserve=max(256*1024**2,int(path.stat().st_size*1.1) if path.exists() else 0)
        if self.storage is not None:self.storage.check(force=True,write_reserve_bytes=reserve)

    def identity(self):
        return dict(kind=self.identity_kind,ledger_path=str(self.ledger.path),
            aggregate_max_attempts=3000,aggregate_max_seconds=5400.,stage_max_attempts=self.max_attempts,
            stage_max_model_work_seconds=self.max_seconds,latest_new_update_seconds=self.latest_update_seconds,
            initial_tail_reserve_seconds=self.initial_tail_reserve_seconds,time_scope="load/baseline/fixedloss/updates/finalnative/save/failures/cleanup; one externally owned clock")

    def report(self):
        return dict(**self.identity(),elapsed_model_work_seconds=self.elapsed,measured_baseline_seconds=self.baseline_seconds,
            reserved_tail_seconds=self.tail_reserve_seconds,effective_update_cutoff_seconds=self.update_cutoff,
            aggregate_seconds_before=self.aggregate_seconds_before,effective_time_cap_seconds=self.effective_time_cap,
            stop_reason=self.stop_reason,closed=self.closed,overrun_seconds=max(0.,self.elapsed-self.effective_time_cap),ledger=self.ledger.snapshot())

    def finish(self,status="complete"):
        if self.closed:return None
        value=self.stage.finish(status)
        self.finished_elapsed=value["stages"][-1]["observed_seconds"];self.closed=True
        return value


class FormalModelWorkBudget(ModelWorkBudget):
    """Explicit planned first500 scope; construction does not authorize execution."""
    identity_kind="native-rgb-formal-first500-model-work-budget.v1"

    def validate_local_caps(self,max_attempts,max_seconds,update_cutoff_seconds,tail_reserve_seconds):
        if type(max_attempts) is not int or not 1<=max_attempts<=500 or not 0<float(max_seconds)<=1800:
            raise ValueError("Formal first segment permits at most500 attempts/1800 model seconds")
        if not 0<=float(update_cutoff_seconds)<=min(1620,float(max_seconds)) or tail_reserve_seconds<180:
            raise ValueError("Formal segment needs cutoff<=1620 and at least180 seconds tail")

    def __init__(self,stage,*,max_attempts=500,max_seconds=1800.,update_cutoff_seconds=1620.,tail_reserve_seconds=180.,storage=None):
        if storage is None:raise ValueError("Formal model work requires combined storage accounting")
        super().__init__(stage,max_attempts=max_attempts,max_seconds=max_seconds,
            update_cutoff_seconds=update_cutoff_seconds,tail_reserve_seconds=tail_reserve_seconds,storage=storage)
        self.validation_seconds=[]

    def record_baseline(self,seconds,*,checkpoint_reserve_seconds=45.,prediction_factor=2.,other_tail_seconds=45.):
        # The reused-H1 baseline only generates3 native steps; validation needs4.
        return super().record_baseline(seconds,checkpoint_reserve_seconds=checkpoint_reserve_seconds,
            prediction_factor=prediction_factor,other_tail_seconds=other_tail_seconds)

    @property
    def estimated_validation_seconds(self):
        return max(60.,2.*float(self.baseline_seconds or 0),
                   1.5*max(self.validation_seconds,default=0.))

    def record_validation(self,seconds):
        if not math.isfinite(seconds) or seconds<0:raise ValueError("Finite measured formal validation time required")
        self.validation_seconds.append(float(seconds))
        self.tail_reserve_seconds=max(self.tail_reserve_seconds,1.5*seconds+90.)

    def report(self):
        result=super().report()
        result.update(measured_formal_validation_seconds=list(self.validation_seconds),
            estimated_next_validation_seconds=self.estimated_validation_seconds)
        return result
