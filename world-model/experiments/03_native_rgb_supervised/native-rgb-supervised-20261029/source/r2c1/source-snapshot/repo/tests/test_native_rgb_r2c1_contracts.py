"""CPU metadata/mock contracts only; no Torch/model/device/physics calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from tdyssey.native_rgb_r2c1_budget import (GLOBAL_SECONDS_BEFORE,
    R2C1ModelWorkBudget,R2C1StorageBudget,storage_for_config)
from tdyssey.native_rgb_stage_budget import UpdateLimitReached
from tdyssey.native_rgb_training import BudgetExceeded


class FakeLedger:
    max_updates=3000
    max_seconds=5400.
    path=Path('mock-global-ledger.json')
    def __init__(self):
        self.value=dict(charged_updates=250,completed_updates=250,charged_seconds=GLOBAL_SECONDS_BEFORE,
            active_stage=dict(token='owned',seconds_before=GLOBAL_SECONDS_BEFORE,reserved_updates=0,completed_updates=0,observed_seconds=0.),stages=[])
    def snapshot(self):return deepcopy(self.value)


class FakeStage:
    def __init__(self):self.ledger=FakeLedger();self.token='owned';self.started=100.;self.now=100.;self.events=[]
    def heartbeat(self):
        self.ledger.value['charged_seconds']=GLOBAL_SECONDS_BEFORE+self.now-self.started
        self.ledger.value['active_stage']['observed_seconds']=self.now-self.started
        self.events.append('heartbeat');return self.ledger.snapshot()
    def reserve_update(self):
        self.events.append('reserve');self.ledger.value['charged_updates']+=1;self.ledger.value['active_stage']['reserved_updates']+=1
        return self.ledger.snapshot()
    def complete_update(self):
        self.events.append('complete');self.ledger.value['completed_updates']+=1;self.ledger.value['active_stage']['completed_updates']+=1
        return self.ledger.snapshot()
    def finish(self,status):
        self.heartbeat();self.ledger.value['stages'].append(dict(observed_seconds=self.now-self.started,status=status));self.ledger.value['active_stage']=None
        return self.ledger.snapshot()


class R2C1Contracts(unittest.TestCase):
    def test_prior_global_link_local300_and_no_refund(self):
        stage=FakeStage();storage=SimpleNamespace(check=Mock(return_value={}))
        with patch('tdyssey.native_rgb_stage_budget.time.monotonic',side_effect=lambda:stage.now):
            budget=R2C1ModelWorkBudget(stage,storage=storage)
            for _ in range(300):budget.reserve_update()
            with self.assertRaises(UpdateLimitReached):budget.reserve_update()
            self.assertEqual(stage.ledger.value['charged_updates'],550)
            self.assertEqual(stage.ledger.value['completed_updates'],250)
            self.assertEqual(budget.aggregate_seconds_before,GLOBAL_SECONDS_BEFORE)
        stage=FakeStage();stage.ledger.value['charged_updates']=0
        with self.assertRaises(ValueError):R2C1ModelWorkBudget(stage,storage=storage)

    def test_allwork_operation_precheck_and_earlier_tail(self):
        stage=FakeStage()
        with patch('tdyssey.native_rgb_stage_budget.time.monotonic',side_effect=lambda:stage.now):
            budget=R2C1ModelWorkBudget(stage,storage=SimpleNamespace(check=lambda **kw:{}))
            self.assertEqual(budget.effective_time_cap,1200.)
            self.assertEqual(budget.update_cutoff,900.)
            self.assertEqual(budget.record_baseline(500.),345.)
            stage.now=445.
            with self.assertRaises(UpdateLimitReached):budget.reserve_update()
            budget.require_operation(100.,'final evaluation')
            stage.now=1200.
            with self.assertRaises(BudgetExceeded):budget.require_operation(100.,'must not launch at exactremainingcap')
            self.assertEqual(stage.ledger.value['charged_updates'],250)
            stage.now=1250.;budget.finish('complete')
            self.assertAlmostEqual(budget.elapsed,1150.)
            self.assertAlmostEqual(stage.ledger.value['charged_seconds'],GLOBAL_SECONDS_BEFORE+1150.)
            self.assertIsNone(stage.ledger.value['active_stage'])

    def test_shared_and_stage_limits_every_check_without_extra_stage_reserve(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp);usage=SimpleNamespace(free=10*1024**3)
            footprint=Mock(return_value=dict(bytes=2*1024**3,files=1,roots=[str(stage)]))
            with patch('tdyssey.native_rgb_stage_budget.storage_footprint',footprint),patch('tdyssey.native_rgb_stage_budget.shutil.disk_usage',return_value=usage):
                budget=R2C1StorageBudget([stage],stage_root=stage,interval_seconds=0)
                budget.check();budget.check();self.assertEqual(footprint.call_count,2)
                footprint.return_value=dict(bytes=3*1024**3,files=1,roots=[str(stage)])
                with self.assertRaises(BudgetExceeded):budget.check()
            fake=SimpleNamespace(is_file=lambda:True,stat=lambda:SimpleNamespace(st_size=256*1024**2+1))
            with patch('tdyssey.native_rgb_stage_budget.StorageBudget.check',return_value={}),patch.object(Path,'rglob',return_value=[fake]):
                with self.assertRaises(BudgetExceeded):budget.check()

    def test_local_caps_cannot_silently_expand_and_no_torch_import(self):
        for kwargs in (dict(max_attempts=301),dict(max_seconds=1201),dict(update_cutoff_seconds=901),dict(tail_reserve_seconds=299)):
            with self.assertRaises(ValueError):R2C1ModelWorkBudget(FakeStage(),storage=object(),**kwargs)
        self.assertNotIn('torch',sys.modules)

    def test_reserved_whole_stage_bound_and_cached_immediate_stage_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp);clock=[100.];usage=SimpleNamespace(free=10*1024**3)
            def footprint(paths):
                clock[0]+=5. # A slow scan must not expire its own cache.
                return dict(bytes=2*1024**3,files=1,roots=[str(stage)])
            budget=storage_for_config(dict(repo_root=tmp,output=tmp,storage_roots=[tmp]))
            self.assertEqual(budget.interval,60.)
            with patch('tdyssey.native_rgb_r2c1_budget.time.monotonic',side_effect=lambda:clock[0]),patch('tdyssey.native_rgb_stage_budget.storage_footprint',side_effect=footprint) as scan,patch('tdyssey.native_rgb_stage_budget.shutil.disk_usage',return_value=usage):
                first=budget.check();self.assertEqual(budget.last_checked,105.)
                self.assertEqual(first['conservative_shared_upper_bound_bytes'],2*1024**3+2*256*1024**2+1024**2)
                clock[0]=106.;second=budget.check();self.assertEqual(scan.call_count,1)
                self.assertTrue(second['conservative_bound_checked_every_call']);self.assertFalse(second['full_scan_performed_now'])
                fake=SimpleNamespace(is_file=lambda:True,stat=lambda:SimpleNamespace(st_size=256*1024**2+1))
                with patch.object(Path,'rglob',return_value=[fake]):
                    with self.assertRaises(BudgetExceeded):budget.check()
                budget.check(force=True);self.assertEqual(scan.call_count,2)
            upper=3*1024**3-2*256*1024**2
            with patch('tdyssey.native_rgb_stage_budget.storage_footprint',return_value=dict(bytes=upper,files=1,roots=[str(stage)])),patch('tdyssey.native_rgb_stage_budget.shutil.disk_usage',return_value=usage):
                with self.assertRaises(BudgetExceeded):storage_for_config(dict(repo_root=tmp,output=tmp,storage_roots=[tmp])).check()

    def test_normalized_bridge_passthrough_and_finite_OOR_feedback(self):
        import numpy as np
        from tdyssey.native_rgb_r2c1_training import normalized_predictor
        events=[];history=np.ones((1,4,3,96,96),dtype=np.float32)*.5
        actions=np.linspace(-1,1,8,dtype=np.float32).reshape(1,4,2)
        raw=np.linspace(-.04,1.04,3*96*96,dtype=np.float32).reshape(1,3,96,96)
        def before(**kwargs):events.append('precheck')
        def sample(h,a,*,seed):
            events.append('sample');np.testing.assert_array_equal(a,actions);self.assertEqual(seed,5478);return raw
        def sink(value,**kwargs):events.append('raw_sink');self.assertIs(value,raw);np.testing.assert_array_equal(kwargs['normalized_actions'],actions)
        predict=normalized_predictor(sample,before_call=before,raw_sink=sink)
        self.assertIs(predict(history,actions,seed=5478,cohort='fixture',step=0,window_ids=['x']),raw)
        history[:,3]=raw;predict(history,actions,seed=5478,cohort='fixture',step=1,window_ids=['x'])
        self.assertEqual(events,['precheck','sample','raw_sink']*2)
        with self.assertRaises(ValueError):predict(history,actions,seed=5478,cohort='fixture',step=0,window_ids=['x'])
        self.assertNotIn('torch',sys.modules)

    def test_attempt_reserved_before_any_draw_and_failure_never_draws(self):
        from tdyssey.native_rgb_r2c1_training import reserve_then_draw
        events=[]
        budget=SimpleNamespace(reserve_update=lambda:events.append('durable_reserve'))
        prepared=SimpleNamespace(begin_update=lambda:events.append('draw') or tuple(range(16)))
        selected=reserve_then_draw(budget,prepared,on_reserved=lambda:events.append('boundary_snapshot'))
        self.assertEqual(events,['durable_reserve','boundary_snapshot','draw']);self.assertEqual(len(selected),16)
        events.clear()
        def stop():events.append('reserve_refused');raise BudgetExceeded('cap')
        with self.assertRaises(BudgetExceeded):reserve_then_draw(SimpleNamespace(reserve_update=stop),prepared,on_reserved=lambda:events.append('should_not_run'))
        self.assertEqual(events,['reserve_refused'])


if __name__=='__main__':unittest.main()
