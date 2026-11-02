"""Stdlib account tests plus deferred tiny CPU checkpoint tests.

RuntimeBudgetTests can run without Torch. TinyCPUCheckpointTests import Torch
only when explicitly selected by the parent CPU-fixture validation process.
"""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import gpc_adaptation_runtime as runtime


EVIDENCE=Path(__file__).resolve().parents[1]/'artifacts/wm-l-adaptation-preparation-20261025/runtime'
BINDING=dict(config_sha256='a'*64,source_sha256='b'*64,candidate_id='one-tiny-fixture-only')


class Clock:
    wall=100.;mono=50.
    def time(self):return self.wall
    def monotonic(self):return self.mono


class Fixtures(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=EVIDENCE,prefix='tiny-runtime-')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.clock=Clock()
    def create(self,**overrides):
        values=dict(max_updates=3,max_eval_batches=3,max_batch_size=4,deadline_seconds=10,
                    max_train_examples=5,max_eval_examples=3,max_forecasts=2,max_euler_steps=6,max_model_loads=3)
        values.update(overrides)
        budget=runtime.AdaptationBudget.create(self.root/'ledger',BINDING,
            clock=self.clock.time,monotonic=self.clock.monotonic,**values)
        op=budget.reserve_model_load('train_init');budget.complete_model_load(op)
        return budget
    def fake_commit(self,budget):
        path=self.root/'fake-boundary.bin';path.write_bytes(b'not a tensor checkpoint; account-only fixture')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        budget._operation('checkpoint_committed',dict(path=str(path),sha256=digest,boundary=budget._boundary()))
        return digest


class RuntimeBudgetTests(Fixtures):
    def test_actual_short_last_batch_fits_without_charging_full_batch(self):
        b=self.create()
        for n in (4,1):
            op=b.reserve_update(n);b.complete_update(op)
        s=b.snapshot()
        self.assertEqual((s['spent_reserved_updates'],s['completed_updates'],s['spent_train_examples']),(2,2,5))
        self.assertEqual(s['checkpoint_committed_updates'],0)
        with self.assertRaises(runtime.AdaptationRuntimeError):b.reserve_update(1)
        self.assertEqual(b.snapshot()['spent_reserved_updates'],2)

    def test_reservation_is_durable_before_entry_and_failed_cost_is_not_refunded(self):
        b=self.create();op=b.reserve_update(2)
        last=json.loads((b.directory/'highwater.jsonl').read_bytes().splitlines()[-1])
        self.assertEqual((last['kind'],last['data']['batch_size'],last['sequence']),('reserve_update',2,op))
        self.assertEqual(b.snapshot()['completed_updates'],0)
        with self.assertRaisesRegex(runtime.AdaptationRuntimeError,'Reserved operation'):
            runtime.AdaptationBudget.resume(b.directory,BINDING,clock=self.clock.time,monotonic=self.clock.monotonic)
        self.assertEqual(b.snapshot()['spent_reserved_updates'],1)

    def test_eval_examples_are_separate_and_share_original_deadline(self):
        b=self.create()
        for n in (2,1):
            op=b.reserve_eval(n,metadata=dict(role='train_init'));b.complete_eval(op)
        s=b.snapshot();self.assertEqual((s['spent_eval_examples'],s['completed_eval_batches'],s['spent_reserved_updates']),(3,2,0))
        self.clock.mono=60.
        with self.assertRaisesRegex(runtime.AdaptationRuntimeError,'Original adaptation deadline'):b.check()
        first=b.snapshot()['first_reason'];b.fail('later cleanup failure')
        self.assertEqual(b.snapshot()['first_reason'],first)
        self.assertEqual(json.loads((b.directory/'first-failure.json').read_bytes())['reason'],first)

    def test_exact_checkpoint_resume_keeps_spent_counts_identity_and_deadline(self):
        b=self.create();op=b.reserve_update(1);b.complete_update(op);self.fake_commit(b)
        self.clock.wall=104.;self.clock.mono=54.
        restored=runtime.AdaptationBudget.resume(b.directory,BINDING,clock=self.clock.time,monotonic=self.clock.monotonic)
        self.assertEqual(restored.snapshot()['spent_reserved_updates'],1)
        self.assertEqual(restored.attempt['deadline_monotonic'],60.)
        self.assertEqual(restored.attempt['deadline_unix'],110.)
        with self.assertRaises(runtime.AdaptationRuntimeError):
            runtime.AdaptationBudget.resume(b.directory,dict(BINDING,candidate_id='second-candidate'),clock=self.clock.time,monotonic=self.clock.monotonic)
        op=b.reserve_update(1);b.complete_update(op)
        with self.assertRaisesRegex(runtime.AdaptationRuntimeError,'No exact committed checkpoint'):
            runtime.AdaptationBudget.resume(b.directory,BINDING,clock=self.clock.time,monotonic=self.clock.monotonic)

    def test_forecasts_euler_and_serial_load_roles_never_use_loss_batch_costs(self):
        b=self.create(max_batch_size=1)
        selected=self.fake_commit(b);b.select_checkpoint(selected)
        for role in ('frozen','adapted'):
            op=b.reserve_model_load(role);b.complete_model_load(op)
            op=b.reserve_forecast(metadata=dict(role=role,case='tiny',horizon=1));b.complete_forecast(op)
        s=b.snapshot()
        self.assertEqual((s['spent_reserved_forecasts'],s['completed_forecasts'],s['spent_reserved_euler_steps']),(2,2,6))
        self.assertEqual((s['spent_reserved_eval_batches'],s['spent_reserved_updates']),(0,0))
        self.assertEqual(s['model_load_roles'],['train_init','frozen','adapted'])
        with self.assertRaises(runtime.AdaptationRuntimeError):b.reserve_forecast(metadata=dict(role='adapted'))
        self.assertEqual(b.snapshot()['spent_reserved_forecasts'],2)

    def test_selection_is_once_and_prevents_further_optimizer_updates(self):
        b=self.create();digest=self.fake_commit(b);b.select_checkpoint(digest)
        with self.assertRaises(runtime.AdaptationRuntimeError):b.reserve_update(1)
        self.assertEqual(b.snapshot()['selection_count'],1)

    def test_partial_journal_and_write_failure_fail_closed_with_first_reason(self):
        b=self.create();op=b.reserve_update(1);b.complete_update(op);self.fake_commit(b)
        with (b.directory/'highwater.jsonl').open('ab') as stream:stream.write(b'{partial')
        with self.assertRaisesRegex(runtime.AdaptationRuntimeError,'Incomplete'):
            runtime.AdaptationBudget.resume(b.directory,BINDING,clock=self.clock.time,monotonic=self.clock.monotonic)
        b.fail('first IO failure');b.fail('second failure')
        self.assertEqual(b.snapshot()['first_reason'],'first IO failure')

    def test_no_overlarge_pending_or_wrong_role_operations(self):
        b=self.create()
        with self.assertRaises(runtime.AdaptationRuntimeError):b.reserve_update(5)
        self.assertEqual(b.snapshot()['spent_reserved_updates'],0)
        self.assertEqual(b.snapshot()['spent_train_examples'],0)

    def test_fsync_failure_keeps_reservation_and_original_failure_durable(self):
        b=self.create();inode=(b.directory/'highwater.jsonl').stat().st_ino
        original=runtime.os.fsync
        def fail_journal(fd):
            if runtime.os.fstat(fd).st_ino==inode:raise OSError('injected reservation fsync failure')
            return original(fd)
        with patch.object(runtime.os,'fsync',side_effect=fail_journal):
            with self.assertRaisesRegex(OSError,'reservation fsync'):b.reserve_update(1)
        state=b.snapshot()
        self.assertEqual((state['spent_reserved_updates'],state['completed_updates']),(1,0))
        self.assertEqual(state['status'],'stopped')
        first=state['first_reason'];b.fail('secondary close failure')
        self.assertEqual(json.loads((b.directory/'first-failure.json').read_bytes())['reason'],first)
        with self.assertRaises(runtime.AdaptationRuntimeError):
            runtime.AdaptationBudget.resume(b.directory,BINDING,clock=self.clock.time,monotonic=self.clock.monotonic)

    def test_update_batch_and_forecast_euler_caps_are_independent(self):
        base=self.root
        for kind in ('update','eval','forecast'):
            with self.subTest(kind=kind):
                self.root=base/kind
                b=self.create(max_updates=1,max_eval_batches=1,max_train_examples=None,max_eval_examples=None,
                              max_forecasts=2,max_euler_steps=3)
                if kind=='forecast':
                    b.select_checkpoint(self.fake_commit(b));op=b.reserve_model_load('frozen');b.complete_model_load(op)
                    op=b.reserve_forecast(metadata=dict(role='frozen'));b.complete_forecast(op)
                    with self.assertRaises(runtime.AdaptationRuntimeError):b.reserve_forecast(metadata=dict(role='frozen'))
                    self.assertEqual((b.snapshot()['spent_reserved_forecasts'],b.snapshot()['spent_reserved_euler_steps']),(1,3))
                else:
                    reserve=b.reserve_update if kind=='update' else b.reserve_eval
                    complete=b.complete_update if kind=='update' else b.complete_eval
                    op=reserve(1);complete(op)
                    with self.assertRaises(runtime.AdaptationRuntimeError):reserve(1)
                    self.assertEqual(b.snapshot()['spent_reserved_updates' if kind=='update' else 'spent_reserved_eval_batches'],1)
        self.root=base

    def test_pending_forecast_and_model_load_expire_under_the_same_deadline(self):
        base=self.root
        for kind in ('load','forecast'):
            with self.subTest(kind=kind):
                self.root=base/kind;self.clock=Clock();b=self.create()
                b.select_checkpoint(self.fake_commit(b));op=b.reserve_model_load('frozen')
                if kind=='forecast':
                    b.complete_model_load(op);op=b.reserve_forecast(metadata=dict(role='frozen'))
                self.clock.wall=110.;self.clock.mono=60.
                with self.assertRaisesRegex(runtime.AdaptationRuntimeError,'Original adaptation deadline'):
                    (b.complete_model_load if kind=='load' else b.complete_forecast)(op)
                self.assertEqual(b.snapshot()['spent_reserved_model_loads'],2)
                self.assertEqual(b.snapshot()['completed_model_loads'],1 if kind=='load' else 2)
                if kind=='forecast':
                    self.assertEqual(b.snapshot()['spent_reserved_forecasts'],1)
                    self.assertEqual(b.snapshot()['completed_forecasts'],0)
        self.root=base

    def test_recovered_account_cannot_compute_before_full_checkpoint_restore(self):
        b=self.create();op=b.reserve_update(1);b.complete_update(op);self.fake_commit(b)
        recovered=runtime.AdaptationBudget.resume(b.directory,BINDING,clock=self.clock.time,monotonic=self.clock.monotonic)
        self.assertTrue(recovered.snapshot()['requires_checkpoint_restore'])
        with self.assertRaisesRegex(runtime.AdaptationRuntimeError,'exact coherent checkpoint restore'):
            recovered.reserve_update(1)
        self.assertEqual(recovered.snapshot()['spent_reserved_updates'],1)

    def test_original_wall_deadline_and_backward_clock_cannot_reset_time(self):
        base=self.root
        for mode in ('wall_deadline','backward'):
            with self.subTest(mode=mode):
                self.root=base/mode;self.clock=Clock();b=self.create()
                self.clock.wall=110. if mode=='wall_deadline' else 99.
                with self.assertRaises(runtime.AdaptationRuntimeError):b.check()
                self.assertEqual(b.attempt['deadline_unix'],110.)
                self.assertEqual(b.attempt['deadline_monotonic'],60.)
                self.assertEqual(b.snapshot()['status'],'stopped')
        self.root=base


class TinyCPUCheckpointTests(Fixtures):
    """Parent-owned execution only: CPU Linear/SGD, no GPC or forward call."""
    def tiny(self,budget):
        import torch
        model=torch.nn.Linear(2,1)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
        generator=torch.Generator(device='cpu').manual_seed(7)
        store=runtime.CheckpointStore(self.root/'latest.pt',budget,'tiny-config',max_bytes=1024*1024,generators=dict(data=generator))
        return model,optimizer,generator,store
    def step(self,budget,model,optimizer):
        import torch
        op=budget.reserve_update(1)
        for p in model.parameters():p.grad=torch.ones_like(p)
        optimizer.step();optimizer.zero_grad();budget.complete_update(op)

    def test_atomic_checkpoint_restores_model_optimizer_rng_and_exact_counts(self):
        import numpy as np
        import torch
        b=self.create();model,opt,generator,store=self.tiny(b);self.step(b,model,opt)
        random.seed(8);np.random.seed(8);torch.manual_seed(8)
        saved=store.save(model,opt,extra_state=dict(cursor=1))
        expected=(torch.rand(3),np.random.rand(3),random.random(),torch.rand(3,generator=generator))
        resumed=runtime.AdaptationBudget.resume(b.directory,BINDING,clock=self.clock.time,monotonic=self.clock.monotonic)
        restored_model,restored_opt,restored_gen,_=self.tiny(resumed)
        restored=runtime.CheckpointStore(store.path,resumed,'tiny-config',max_bytes=1024*1024,generators=dict(data=restored_gen))
        info=restored.restore(restored_model,restored_opt)
        for a,c in zip(model.parameters(),restored_model.parameters()):self.assertTrue(torch.equal(a,c))
        self.assertEqual(info['extra_state'],dict(cursor=1));self.assertEqual(info['logical_steps'],1)
        self.assertTrue(torch.equal(torch.rand(3),expected[0]));np.testing.assert_array_equal(np.random.rand(3),expected[1])
        self.assertEqual(random.random(),expected[2]);self.assertTrue(torch.equal(torch.rand(3,generator=restored_gen),expected[3]))
        self.assertEqual(resumed.snapshot()['checkpoint_committed_updates'],1)
        self.assertEqual(saved['sha256'],info['sha256'])

    def test_partial_save_preserves_last_complete_point_and_retains_spent_update(self):
        import torch
        b=self.create();model,opt,_,store=self.tiny(b);self.step(b,model,opt);saved=store.save(model,opt)
        self.step(b,model,opt)
        def partial(value,stream):stream.write(b'partial-tiny-checkpoint');raise OSError('injected serialization failure')
        with patch.object(torch,'save',side_effect=partial):
            with self.assertRaisesRegex(OSError,'injected'):store.save(model,opt)
        self.assertEqual(hashlib.sha256(store.path.read_bytes()).hexdigest(),saved['sha256'])
        self.assertEqual((b.snapshot()['spent_reserved_updates'],b.snapshot()['checkpoint_committed_updates']),(2,1))
        self.assertEqual(len(list(self.root.glob('latest.pt.pending-*'))),1)
        with self.assertRaises(runtime.AdaptationRuntimeError):
            runtime.AdaptationBudget.resume(b.directory,BINDING,clock=self.clock.time,monotonic=self.clock.monotonic)

    def test_selected_evaluation_load_keeps_current_charges_and_rng(self):
        import torch
        b=self.create(max_batch_size=1);model,opt,_,store=self.tiny(b);self.step(b,model,opt)
        saved=store.save(model,opt);b.select_checkpoint(saved['sha256'])
        op=b.reserve_model_load('frozen');b.complete_model_load(op)
        op=b.reserve_forecast(metadata=dict(role='frozen'));b.complete_forecast(op)
        op=b.reserve_model_load('adapted')
        target=torch.nn.Linear(2,1);before=torch.get_rng_state().clone();counts=b.snapshot()
        loaded=store.load_selected_for_evaluation(target)
        self.assertTrue(torch.equal(torch.get_rng_state(),before));self.assertFalse(loaded['counts_restored'])
        self.assertEqual(b.snapshot()['spent_reserved_forecasts'],counts['spent_reserved_forecasts'])
        self.assertEqual(b.snapshot()['spent_reserved_model_loads'],3);b.complete_model_load(op)
        for a,c in zip(model.parameters(),target.parameters()):self.assertTrue(torch.equal(a,c))

    def test_wrong_optimizer_cannot_mix_with_old_counts(self):
        import torch
        b=self.create();model,opt,_,store=self.tiny(b);self.step(b,model,opt);store.save(model,opt)
        target=torch.nn.Linear(2,1);before={k:v.clone() for k,v in target.state_dict().items()}
        wrong=torch.optim.Adam(target.parameters(),lr=.1)
        with self.assertRaises(runtime.AdaptationRuntimeError):store.restore(target,wrong)
        for k,v in target.state_dict().items():self.assertTrue(torch.equal(v,before[k]))

    def test_atomic_publication_failure_retains_previous_checkpoint(self):
        b=self.create();model,opt,_,store=self.tiny(b);self.step(b,model,opt);saved=store.save(model,opt)
        self.step(b,model,opt)
        with patch.object(runtime.os,'replace',side_effect=PermissionError('injected atomic publication failure')):
            with self.assertRaisesRegex(PermissionError,'atomic publication'):store.save(model,opt)
        self.assertEqual(hashlib.sha256(store.path.read_bytes()).hexdigest(),saved['sha256'])
        self.assertEqual((b.snapshot()['spent_reserved_updates'],b.snapshot()['checkpoint_committed_updates']),(2,1))

    def test_bounded_serialization_and_single_checkpoint_path_preserve_old_point(self):
        base=self.root
        for mode in ('byte_limit','other_path'):
            with self.subTest(mode=mode):
                self.root=base/mode;b=self.create();model,opt,_,store=self.tiny(b);self.step(b,model,opt);saved=store.save(model,opt)
                self.step(b,model,opt)
                target=store.path if mode=='byte_limit' else self.root/'other.pt'
                limited=runtime.CheckpointStore(target,b,'tiny-config',max_bytes=128 if mode=='byte_limit' else 1024*1024)
                with self.assertRaises(Exception):limited.save(model,opt)
                self.assertEqual(hashlib.sha256(store.path.read_bytes()).hexdigest(),saved['sha256'])
                if mode=='byte_limit':self.assertTrue(all(p.stat().st_size<=128 for p in self.root.glob('*.pending-*')))
                else:self.assertFalse(target.exists())
        self.root=base

    def test_recovery_cannot_save_random_weights_before_restoring_original_boundary(self):
        b=self.create();model,opt,_,store=self.tiny(b);self.step(b,model,opt);saved=store.save(model,opt)
        recovered=runtime.AdaptationBudget.resume(b.directory,BINDING,clock=self.clock.time,monotonic=self.clock.monotonic)
        other,other_opt,gen,_=self.tiny(recovered)
        candidate=runtime.CheckpointStore(store.path,recovered,'tiny-config',max_bytes=1024*1024,generators=dict(data=gen))
        with self.assertRaisesRegex(runtime.AdaptationRuntimeError,'fresh weights'):candidate.save(other,other_opt)
        self.assertEqual(hashlib.sha256(store.path.read_bytes()).hexdigest(),saved['sha256'])


if __name__=='__main__':unittest.main()
