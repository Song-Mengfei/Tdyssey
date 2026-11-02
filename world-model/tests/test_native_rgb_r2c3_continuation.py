"""CPU contracts only: restored sampling, real scheduler protocol and accounting."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from tdyssey import native_rgb_r2c3_continuation as c
from tdyssey.native_rgb_r2c0_data import R2C0WindowOrder,canonical_hash
from tdyssey.native_rgb_r2c0_interfaces import PreparedInputs
from tdyssey.native_rgb_stage_budget import UpdateLimitReached

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/native-rgb-supervised-20261029/r2c0/data'


class FakeNativeCosine:
    """CPU stub with the public saved-state/step protocol of restored native LR."""
    def __init__(self,optimizer,completed=300):
        self.optimizer=optimizer;self.T_max=3000;self.eta_min=1e-5
        self.base_lrs=[1e-4];self.last_epoch=completed;self.steps=0
        optimizer.param_groups[0]['lr']=c._cosine_lr(1e-4,1e-5,completed)
    def state_dict(self):
        result=dict(T_max=self.T_max,eta_min=self.eta_min,base_lrs=self.base_lrs,last_epoch=self.last_epoch)
        if hasattr(self,'continuation_state'):result['continuation_state']=deepcopy(self.continuation_state)
        return result
    def load_state_dict(self,state):
        for key,value in deepcopy(state).items():setattr(self,key,value)
    def step(self):
        self.steps+=1;self.last_epoch+=1
        # Unclamped cosine intentionally rebounds if incorrectly called >3000.
        self.optimizer.param_groups[0]['lr']=self.eta_min+(self.base_lrs[0]-self.eta_min)*(1+math.cos(math.pi*self.last_epoch/self.T_max))/2


class ContinuationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pools=json.loads((DATA/'pools.json').read_text())

    def fake_dataset(self):
        rows=self.pools['windows'];families=list(dict.fromkeys(row['family_id'] for row in rows));lookup={f:i for i,f in enumerate(families)}
        class Dataset:
            split='train';root=DATA
            entries=[dict(family_id=f) for f in families]
            windows=[(lookup[row['family_id']],row['anchor']) for row in rows]
            def identity(inner):return deepcopy(self.pools['data_identity'])
            def __len__(inner):return len(rows)
            def __getitem__(inner,i):raise AssertionError('Sampling CPU contracts never load RGB/model data')
        return Dataset()

    def test_old4800_state_and_uninterrupted_stream_are_exact(self):
        old=R2C0WindowOrder(self.pools);prefix=old.take(4800)
        saved=json.loads((DATA/'order-state4800.json').read_text())
        self.assertEqual(old.state_dict(),saved)
        resumed=c.R2C3WindowOrder(self.pools);resumed.load_state_dict(saved)
        self.assertEqual(resumed.legacy_state_dict(),saved)
        self.assertEqual(resumed.identity(),old.identity())
        uninterrupted=c.R2C3WindowOrder(self.pools);self.assertEqual(uninterrupted.take(4800),prefix)
        next_cycle=resumed.take(480)
        self.assertEqual(next_cycle,uninterrupted.take(480))
        self.assertEqual(resumed.state_dict(),uninterrupted.state_dict())
        self.assertNotEqual(next_cycle,prefix[:480])
        counts=Counter((self.pools['windows'][i]['letter'],self.pools['windows'][i]['moving'])for i in next_cycle)
        self.assertEqual({l:(counts[(l,True)],counts[(l,False)])for l in 'TLOCXB'},
                         {l:(64,16)for l in 'TLOCXB'})

    def test_new_state_restores_beyond4800_and_at96000_without_reset(self):
        a=c.R2C3WindowOrder(self.pools);a.take(5603);state=a.state_dict()
        b=c.R2C3WindowOrder(self.pools);b.load_state_dict(deepcopy(state))
        self.assertEqual(a.take(517),b.take(517));self.assertEqual(a.state_dict(),b.state_dict())
        a.take(96000-a.cursor);end=a.state_dict()
        restored=c.R2C3WindowOrder(self.pools);restored.load_state_dict(end)
        self.assertEqual(restored.state_dict(),end)
        self.assertEqual(restored.cursor,96000)
        with self.assertRaisesRegex(ValueError,'96000'):restored.take(1)
        bad=deepcopy(state);bad['slot_rng']['state']['state']+=1
        bad.pop('state_sha256');bad['state_sha256']=canonical_hash(bad)
        with self.assertRaisesRegex(ValueError,'RNG'):b.load_state_dict(bad)

    def test_finite_amp_skip_extension_restores_continuous_rng_and_scope(self):
        saved=json.loads((DATA/'order-state4800.json').read_text())
        extended=c.R2C3WindowOrder(self.pools);extended.load_state_dict(saved)
        first=extended.take(16);extended.extend_after_amp_skip()
        self.assertEqual(extended.max_draws,96016);self.assertEqual(extended.skip_extension_count,1)
        with self.assertRaisesRegex(ValueError,'distinct'):extended.extend_after_amp_skip()
        checkpoint=extended.state_dict();restored=c.R2C3WindowOrder(self.pools);restored.load_state_dict(checkpoint)
        self.assertEqual(restored.max_draws,96016);self.assertEqual(restored.skip_extension_count,1)
        self.assertEqual(extended.take(16),restored.take(16))
        reference=c.R2C3WindowOrder(self.pools);reference.take(4800)
        self.assertEqual(first,reference.take(16));reference.take(16)
        self.assertEqual(extended.legacy_state_dict(),reference.legacy_state_dict())
        extended.extend_after_amp_skip();restored.extend_after_amp_skip()
        self.assertEqual(extended.max_draws,96032)
        self.assertEqual(extended.state_dict(),restored.state_dict())
        extended.take(extended.max_draws-extended.cursor)
        end=extended.state_dict();last=c.R2C3WindowOrder(self.pools);last.load_state_dict(end)
        self.assertEqual(last.cursor,96032);self.assertEqual(last.skip_extension_count,2)

    def test_whole_PreparedInputs_legacy_migration_preserves_identity_and_rng(self):
        dataset=self.fake_dataset();old_order=R2C0WindowOrder(self.pools)
        old=PreparedInputs(dataset,old_order,dataset_identity=dataset.identity(),sampling_identity=old_order.identity(),evaluation_identity={'frozen36':'test'})
        old_order.take(4800);old.delivered_draws=4800;saved=old.state_dict()
        successor=c.make_training_order(dataset,saved)
        self.assertEqual(successor.cursor,0)
        new=PreparedInputs(dataset,successor,dataset_identity=dataset.identity(),sampling_identity=successor.identity(),evaluation_identity={'frozen36':'test'})
        self.assertEqual(new.identity(),old.identity())
        new.load_state_dict(deepcopy(saved))
        self.assertEqual(new.order.legacy_state_dict(),saved['order'])
        self.assertEqual(new.delivered_draws,4800)
        self.assertIsNone(new.pending);self.assertEqual(new.micro_cursor,0)
        migrated=new.state_dict()
        self.assertNotEqual(migrated['state_sha256'],saved['state_sha256'])
        self.assertEqual(migrated['identity_sha256'],saved['identity_sha256'])
        self.assertEqual(new.order.restoration_audit['restored_cursor'],4800)
        self.assertTrue(new.order.restoration_audit['legacy_core_preserved'])
        second_order=c.make_training_order(dataset,migrated)
        second=PreparedInputs(dataset,second_order,dataset_identity=dataset.identity(),sampling_identity=second_order.identity(),evaluation_identity={'frozen36':'test'})
        second.load_state_dict(migrated)
        self.assertEqual(new.order.take(16),second.order.take(16))
        bad=deepcopy(saved);bad['pending_indices']=list(range(16));bad.pop('state_sha256');bad['state_sha256']=canonical_hash(bad)
        with self.assertRaisesRegex(ValueError,'boundary'):c.make_training_order(dataset,bad)

    def test_lr_native_restore2999_3000_3001_6000_and_hold_resume(self):
        optimizer=SimpleNamespace(param_groups=[{'lr':0.}]);scheduler=FakeNativeCosine(optimizer)
        observed={}
        for update in range(301,6001):
            record=c.advance_lr_after_success(scheduler,optimizer,update)
            if update in (2999,3000,3001,6000):observed[update]=deepcopy(record)
        self.assertEqual(scheduler.steps,2700)
        self.assertEqual(scheduler.last_epoch,3000)
        self.assertEqual(observed[2999]['phase'],'native_cosine')
        self.assertEqual(observed[3000]['phase'],'hold_eta_min')
        self.assertTrue(observed[3000]['native_scheduler_step_called'])
        for update in (3001,6000):
            self.assertFalse(observed[update]['native_scheduler_step_called'])
            self.assertEqual(observed[update]['lr_used_for_completed_update'],[1e-5])
            self.assertEqual(observed[update]['lr_for_next_update'],[1e-5])
        self.assertEqual(observed[3000]['lr_for_next_update'],[1e-5])
        self.assertAlmostEqual(observed[2999]['lr_used_for_completed_update'][0],c._cosine_lr(1e-4,1e-5,2998),places=15)
        saved=scheduler.state_dict();restored=FakeNativeCosine(optimizer);restored.load_state_dict(saved)
        optimizer.param_groups[0]['lr']=1e-5
        after=c.advance_lr_after_success(restored,optimizer,6001)
        self.assertEqual(after['completed_updates'],6001);self.assertEqual(restored.steps,0)
        with self.assertRaisesRegex(ValueError,'exactly one'):c.advance_lr_after_success(restored,optimizer,6003)

    def temporary_ledger(self,folder):
        prior=dict(kind='native_gpc_rgb_cost.v2-record-only-time',max_seconds=None,max_updates=3000,
            active_stage=None,charged_updates=550,completed_updates=550,charged_seconds=c.PRIOR_SECONDS,
            stages=[dict(phase='historical-R1-and-C2-CPU-fixture',status='complete',reserved_updates=550,completed_updates=550)])
        raw=(json.dumps(prior,sort_keys=True)+'\n').encode();base=Path(folder)
        for name in ('live.json','prior.json'):(base/name).write_bytes(raw)
        config=dict(execution_authorized=True,is_proposal=False,time_policy='record_only_v2',cap_time=None,
            ledger_path=str(base/'live.json'),prior_ledger_copy=str(base/'prior.json'),prior_ledger_sha256=hashlib.sha256(raw).hexdigest(),
            prior_attempts=550,prior_completed_updates=550,prior_model_seconds=c.PRIOR_SECONDS,
            resume_lineage_updates=300,target_lineage_updates=6000,new_completed_updates=5700,stop_on_amp_skip=False,max_consecutive_amp_skips=16)
        return config,prior,raw

    def test_durable_scope_allows301_and_crosses_old_global3000_without_time_gate(self):
        with tempfile.TemporaryDirectory(dir=ROOT)as tmp:
            config,prior,raw=self.temporary_ledger(tmp);clock=[10.]
            ledger=c.create_ledger(config,clock=lambda:clock[0]);stage=ledger.start('CPU-only-C3-contract');storage=[]
            budget=c.ContinuationModelWorkBudget(stage,config=config,storage=SimpleNamespace(check=lambda **kw:storage.append(kw)))
            value=ledger.snapshot();active=value['active_stage']
            active['reserved_updates']=active['completed_updates']=300
            value['charged_updates']=value['completed_updates']=850
            Path(config['ledger_path']).write_text(json.dumps(value))
            clock[0]=12010.;budget.reserve_update();budget.complete_update()
            self.assertEqual(ledger.snapshot()['active_stage']['completed_updates'],301)
            self.assertAlmostEqual(ledger.snapshot()['charged_seconds'],c.PRIOR_SECONDS+12000.)
            value=ledger.snapshot();value['active_stage']['reserved_updates']=value['active_stage']['completed_updates']=2449
            value['charged_updates']=value['completed_updates']=2999;Path(config['ledger_path']).write_text(json.dumps(value))
            budget.reserve_update();budget.complete_update()
            self.assertEqual(ledger.snapshot()['charged_updates'],3000)
            budget.reserve_update();budget.complete_update();self.assertEqual(ledger.snapshot()['completed_updates'],3001)
            value=ledger.snapshot();value['active_stage']['reserved_updates']=value['active_stage']['completed_updates']=5699
            value['charged_updates']=value['completed_updates']=6249;Path(config['ledger_path']).write_text(json.dumps(value))
            budget.reserve_update();budget.complete_update();self.assertEqual(ledger.snapshot()['completed_updates'],6250)
            with self.assertRaises(UpdateLimitReached):budget.reserve_update()
            budget.before_checkpoint(Path(tmp)/'fake-never-created.pt');budget.finish('mock-6000-lineage-complete')
            self.assertIsNone(ledger.snapshot()['active_stage']);self.assertEqual(ledger.snapshot()['stages'][:-1],prior['stages'])
            self.assertEqual(Path(config['prior_ledger_copy']).read_bytes(),raw)
            self.assertIsNone(budget.max_seconds);self.assertIsNone(budget.update_cutoff);self.assertIsNone(budget.max_attempts)
            self.assertFalse(budget.stop_on_amp_skip)
            self.assertTrue(any(x.get('write_reserve_bytes')==256*1024**2 for x in storage))
            self.assertFalse((Path(tmp)/'fake-never-created.pt').exists())

    def test_failed_attempt_charged_and_recorded_finite_AMP_skip_can_continue(self):
        with tempfile.TemporaryDirectory(dir=ROOT)as tmp:
            config,prior,_=self.temporary_ledger(tmp);clock=[0.]
            ledger=c.create_ledger(config,clock=lambda:clock[0]);stage=ledger.start('CPU-only-skipped-attempt')
            budget=c.ContinuationModelWorkBudget(stage,config=config,storage=SimpleNamespace(check=lambda **kw:{}))
            clock[0]=100000.;budget.reserve_update()
            self.assertEqual(ledger.snapshot()['charged_updates'],551)
            self.assertEqual(ledger.snapshot()['completed_updates'],550)
            with self.assertRaisesRegex(RuntimeError,'automatic retry'):budget.reserve_update()
            budget.record_amp_skip(attempt=1,scale_before=128.,scale_after=64.,loss_finite=True,parameters_finite=True)
            self.assertEqual(ledger.snapshot()['active_stage']['skipped_updates'],1)
            self.assertEqual(ledger.snapshot()['charged_updates'],551);self.assertEqual(ledger.snapshot()['completed_updates'],550)
            budget.reserve_update();budget.complete_update()
            budget.finish('mock-finite-AMP-skip-continued-without-refund')
            self.assertEqual(ledger.snapshot()['charged_updates'],552);self.assertEqual(ledger.snapshot()['completed_updates'],551)
            self.assertAlmostEqual(ledger.snapshot()['charged_seconds'],c.PRIOR_SECONDS+100000.)
            self.assertEqual(ledger.snapshot()['stages'][:-1],prior['stages'])
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
