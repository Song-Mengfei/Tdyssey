"""Controlled-RNG fake-sampler tests; no GPC, torch, decoder or environment."""

import importlib.util
import struct
import sys
import tempfile
from pathlib import Path
import unittest

DEPS = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(DEPS, "numpy required for mathematical array fixtures")
class PairedEngineChecks(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from tdyssey.wm_compat_engine import SavedBranch, run_paired_branch
        self.np, self.run = np, run_paired_branch
        frames = np.stack([np.full((3,96,96), i/20, dtype=np.float32) for i in range(8)])
        self.branch = SavedBranch("T12", "hold", "T", frames[:4],
            np.array([[0,511],[10,20],[30,40]],dtype=np.float32),
            np.array([[50,60],[70,80],[90,100],[110,120]],dtype=np.float32), frames[4:])

    def dependencies(self, *, divergent=False, nonfinite=False):
        np = self.np
        class RNG:
            value = 19
            def capture(self): return struct.pack("<I", self.value)
            def restore(self, state): self.value = struct.unpack("<I",state)[0]
            def seed(self, seed): self.value = seed
            def draw(self):
                self.value = (1664525*self.value+1013904223) % 2**32
                return self.value/2**32
        rng=RNG()
        class Sampler:
            calls=[]
            def sample(self, images, actions):
                self.calls.append(dict(images=images.copy(), actions=actions.copy(), entry=rng.capture(),
                                       writable=(images.flags.writeable,actions.flags.writeable)))
                draw=rng.draw()
                if divergent and len(self.calls)==5:
                    rng.draw()
                output=np.full((3,96,96),-.2-draw/100,dtype=np.float32)
                if nonfinite:
                    output[0,0,0]=np.nan
                return output
        class Budget:
            reserved=[]
            completed=[]
            failed=[]
            stopped=[]
            def reserve_call(self,key,entry): self.reserved.append((key,entry))
            def complete_call(self,*args): self.completed.append(args)
            def fail_call(self,key,reason): self.failed.append((key,reason))
            def stop(self,reason): self.stopped.append(reason)
        return rng,Sampler(),Budget()

    def test_ar_b_tf_replay_the_same_horizon_entry_and_exit(self):
        rng,sampler,budget=self.dependencies()
        before=rng.capture()
        result=self.run(self.branch,5478,sampler=sampler,rng=rng,budget=budget)
        self.assertEqual(rng.capture(),before)
        self.assertEqual(len(sampler.calls),12)
        self.assertEqual(len(budget.reserved),12)
        self.assertEqual(len(budget.completed),12)
        self.assertFalse(budget.failed)
        rows=result['rows']
        by={(r['arm'],r['mode'],r['horizon']):r for r in rows}
        for h in range(1,5):
            ar=by['A','AR',h]
            self.assertEqual(ar['rng_entry'],by['B','AR',h]['rng_entry'])
            self.assertEqual(ar['rng_exit'],by['B','AR',h]['rng_exit'])
            if h in (2,4):
                for arm in ('A','B'):
                    self.assertEqual(ar['rng_entry'],by[arm,'TF',h]['rng_entry'])
                    self.assertEqual(ar['rng_exit'],by[arm,'TF',h]['rng_exit'])
        self.assertNotEqual(by['A','AR',1]['rng_entry'],by['A','TF',2]['rng_entry'])
        self.assertNotEqual(by['A','TF',2]['rng_entry'],by['A','TF',4]['rng_entry'])
        self.assertEqual(result['h1_reuse']['A'],by['A','AR',1]['key'])
        self.assertFalse(any(r['mode']=='TF' and r['horizon']==1 for r in rows))
        self.assertTrue(all(call['writable']==(False,False) for call in sampler.calls))

    def test_tf_windows_exclude_own_target_and_actions_are_aligned(self):
        rng,sampler,budget=self.dependencies()
        self.run(self.branch,5479,sampler=sampler,rng=rng,budget=budget)
        np=self.np
        h2,h4=sampler.calls[8],sampler.calls[9]
        np.testing.assert_array_equal(h2['images'],np.concatenate((self.branch.history[1:],self.branch.actual[:1])))
        np.testing.assert_array_equal(h4['images'],np.concatenate((self.branch.history[-1:],self.branch.actual[:3])))
        self.assertFalse(any(np.array_equal(image,self.branch.actual[1]) for image in h2['images']))
        self.assertFalse(any(np.array_equal(image,self.branch.actual[3]) for image in h4['images']))
        targets=np.concatenate((self.branch.past_actions,self.branch.future_actions))
        for h,call in ((1,sampler.calls[0]),(2,h2),(4,h4)):
            expected=targets[h-1:h+3].astype(np.float32)/np.float32(511)*np.float32(2)-np.float32(1)
            np.testing.assert_array_equal(call['actions'],expected)

    def test_generated_raw_values_are_never_transformed_or_clipped(self):
        rng,sampler,budget=self.dependencies()
        original=self.branch.history.copy()
        result=self.run(self.branch,5480,sampler=sampler,rng=rng,budget=budget)
        for index in (1,2,3,5,6,7):
            self.assertLess(float(sampler.calls[index]['images'][-1].max()),0)
        self.np.testing.assert_array_equal(self.branch.history,original)
        self.assertTrue(all(r['raw_min']<0 and r['clipped_fraction']==1 for r in result['rows']))

    def test_rng_divergence_fails_a_charged_call_and_restores_external_rng(self):
        rng,sampler,budget=self.dependencies(divergent=True)
        before=rng.capture()
        with self.assertRaisesRegex(ValueError,'consumption differs'):
            self.run(self.branch,5478,sampler=sampler,rng=rng,budget=budget)
        self.assertEqual(len(budget.reserved),5)
        self.assertEqual(len(budget.completed),4)
        self.assertEqual(len(budget.failed),1)
        self.assertEqual(rng.capture(),before)

    def test_nonfinite_output_fails_first_call_without_later_predictions(self):
        rng,sampler,budget=self.dependencies(nonfinite=True)
        with self.assertRaisesRegex(ValueError,'finite raw float32'):
            self.run(self.branch,5478,sampler=sampler,rng=rng,budget=budget)
        self.assertEqual(len(sampler.calls),1)
        self.assertEqual(len(budget.reserved),1)
        self.assertFalse(budget.completed)
        self.assertEqual(len(budget.failed),1)

    def test_cleanup_failure_cannot_replace_the_primary_sampler_error(self):
        rng,sampler,budget=self.dependencies(nonfinite=True)
        original=rng.capture()
        restore=rng.restore
        def fail_external_restore(state):
            if state==original:
                raise OSError('controlled secondary restore failure')
            restore(state)
        rng.restore=fail_external_restore
        with self.assertRaisesRegex(ValueError,'finite raw float32') as caught:
            self.run(self.branch,5478,sampler=sampler,rng=rng,budget=budget)
        self.assertTrue(any('secondary restore failure' in note for note in caught.exception.__notes__))
        self.assertEqual(len(budget.reserved),1)
        self.assertFalse(budget.completed)
        self.assertEqual(len(budget.failed),1)

    def test_frozen_weight_error_survives_a_secondary_backend_restore_failure(self):
        from tdyssey.wm_compat_engine import TorchBackend
        backend=object.__new__(TorchBackend)
        backend.weight_identity='original'
        backend.original=b'controlled-state'
        backend._fingerprint=lambda:'changed'
        def fail_restore(state): raise OSError('controlled close restore failure')
        backend.restore=fail_restore
        with self.assertRaisesRegex(RuntimeError,'weights changed') as caught:
            backend.close()
        self.assertTrue(any('close restore failure' in note for note in caught.exception.__notes__))

    def test_real_input_and_rng_storage_bounds_are_strict(self):
        from dataclasses import replace
        from tdyssey.wm_compat_engine import rng_hash
        with self.assertRaises(ValueError):
            replace(self.branch,history=self.branch.history.astype(self.np.float64))
        invalid=self.branch.history.copy()
        invalid[0,0,0,0]=-1e-8
        with self.assertRaises(ValueError):
            replace(self.branch,history=invalid)
        with self.assertRaises(ValueError):
            rng_hash(b'x'*1025)
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('pymunk',sys.modules)
        self.assertNotIn('pygame',sys.modules)

    def real_budget(self, directory, **kwargs):
        from tdyssey.wm_compat_budget import CompatBudget, EXACT_CHECKPOINT
        root=Path(directory)
        binding=dict(protocol_sha256='a'*64,code_sha256={'fake.py':'b'*64},
                     asset_sha256={'fake.npz':'c'*64},checkpoint_sha256=EXACT_CHECKPOINT)
        budget=CompatBudget.create(root,root/'artifacts/wm-input-compat-test-ledger',
            root/'artifacts/wm-input-compat-test-output/results',binding,**kwargs)
        budget.advance_m(True)
        budget.reserve_model_load()
        budget.complete_model_load()
        return budget

    def test_pre_reservation_storage_failure_stops_real_temporary_ledger(self):
        class Sink:
            def state(self,*args): raise OSError('injected RNG storage failure')
        with tempfile.TemporaryDirectory() as directory:
            budget=self.real_budget(directory)
            rng,sampler,_=self.dependencies()
            before=rng.capture()
            with self.assertRaisesRegex(OSError,'RNG storage failure'):
                self.run(self.branch,5478,sampler=sampler,rng=rng,budget=budget,sink=Sink())
            state=budget.snapshot()
            self.assertEqual(state['status'],'stopped')
            self.assertIn('RNG storage failure',state['stop_reason'])
            self.assertEqual(state['counts']['images'],0)
            self.assertFalse(sampler.calls)
            self.assertEqual(rng.capture(),before)

    def test_completion_io_failure_preserves_first_reason_and_charges(self):
        flag={'fail':False}
        def fault(operation,path):
            if flag['fail'] and operation=='journal_fsync':
                raise OSError('injected completion fsync failure')
        with tempfile.TemporaryDirectory() as directory:
            budget=self.real_budget(directory,io_hook=fault)
            original=budget.complete_call
            def complete(*args):
                flag['fail']=True
                return original(*args)
            budget.complete_call=complete
            rng,sampler,_=self.dependencies()
            before=rng.capture()
            with self.assertRaises(Exception):
                self.run(self.branch,5478,sampler=sampler,rng=rng,budget=budget)
            state=budget.snapshot()
            self.assertEqual(state['status'],'stopped')
            self.assertIn('completion fsync failure',state['stop_reason'])
            self.assertNotIn('no pending call',state['stop_reason'])
            self.assertEqual(state['counts']['images'],1)
            self.assertEqual(state['counts']['euler_steps'],3)
            self.assertEqual(len(sampler.calls),1)
            self.assertEqual(rng.capture(),before)

    def test_peak_allocator_breach_is_not_hidden_by_small_current_reading(self):
        from types import SimpleNamespace
        from tdyssey.wm_compat_engine import TorchBackend
        backend=object.__new__(TorchBackend)
        backend.device='fake'
        backend.torch=SimpleNamespace(cuda=SimpleNamespace(
            memory_allocated=lambda _:1,memory_reserved=lambda _:2,
            max_memory_allocated=lambda _:513*1024**2,max_memory_reserved=lambda _:3))
        with self.assertRaisesRegex(MemoryError,'peak exceeded'):
            backend.stats()
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    unittest.main()
