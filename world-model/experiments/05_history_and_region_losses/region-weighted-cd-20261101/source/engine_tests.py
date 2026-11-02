"""Saved-only scope and throughput arithmetic; no Torch/model/GPU import."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import engine

class Contracts(unittest.TestCase):
    def parent(self):
        return dict(kind='oc_native_rgb_complete.v1',optimizer={'param_groups':[{'lr':1e-5}]},progress=dict(epoch=2,epoch_complete=True,lineage_updates=11367,
            unique_samples_exposed=30935,batches_consumed=1934,complete_optimizer_boundary=True,pending_attempt=None))
    def test_exact_pool_order_and_parent(self):
        pool=engine.read(HERE/'pool-index.json');order=engine.read(HERE/'order.json');parent=self.parent()
        self.assertTrue(engine.validate_scope(pool,order,parent))
        bad=deepcopy(parent);bad['progress']['epoch']=3
        with self.assertRaises(ValueError):engine.validate_scope(pool,order,bad)
        bad=deepcopy(order);bad['batches'][0][0]=bad['batches'][0][1]
        with self.assertRaises(ValueError):engine.validate_scope(pool,bad,parent)
        bad=deepcopy(pool);bad['segments'][0]['supervised_H1_ids'][0]=bad['segments'][1]['supervised_H1_ids'][0]
        with self.assertRaises(ValueError):engine.validate_scope(bad,order,parent)
    def test_throughput_counts_and_extra_evaluation_are_explicit(self):
        def prefix(seconds,generation):
            p=engine.initial_progress('C');p.update(batches_consumed=32,target_frames_exposed=512,successful_updates=32,
                peak_cuda_allocated_bytes=100,peak_cuda_reserved_bytes=200)
            return dict(progress=p,timings=dict(loop_wall_seconds=seconds,generation_seconds=generation,checkpoint_seconds=1.))
        value=engine.throughput_summary(prefix(64,0),prefix(192,96),10,{'passed':True})
        self.assertEqual(value['arms']['C']['remaining_optimizer_batches'],1860)
        self.assertEqual(value['arms']['C']['seconds_per_optimizer_batch'],2)
        self.assertEqual(value['evaluation_and_reporting_extra']['remaining_sampler_batch_calls'],896)
        self.assertEqual(value['evaluation_and_reporting_extra']['remaining_predicted_images'],3584)
        self.assertEqual(value['evaluation_and_reporting_extra']['measured_D32_generation_batch_size'],2)
        self.assertGreater(value['estimated_remaining_total_range_seconds'][1],value['estimated_remaining_training_seconds'])
    def test_partial_progress_separates_reserved_and_actual_exposure(self):
        p=engine.initial_progress('D');p['target_frames_reserved']=16;p['segments_reserved']=4
        self.assertEqual(p['target_frames_exposed'],0);self.assertEqual(p['segments_exposed'],0)
        self.assertEqual(p['lineage_updates'],11367)

    def test_D_assignment_exactly_two_fixed_slots(self):
        data={'generated_indices':[[0,2]for _ in range(1892)]}
        self.assertEqual(len(engine.validate_assignments(data)),1892)
        for bad in ([[0,0]],[[0]],[[0,1,2]],[[0,4]]):
            altered={'generated_indices':[[0,2]for _ in range(1892)]};altered['generated_indices'][1]=bad[0]
            with self.assertRaises(ValueError):engine.validate_assignments(altered)

    def test_interface_gate_stops_before_any_training(self):
        instances=[]
        class FakeSession:
            def __init__(self,root,c):
                instances.append(self);self.arm='C';self.progress=engine.initial_progress('C');self.clean=True
                self.parent_path=engine.PARENT;self.forward_counts={};self.backward_counts={};self.history_generator=None
                self.timings={'C':{},'D':{}};self.checkpoints={};self.train_calls=0
            def restore(self,*args):pass
            def protected(self,label,callback):return callback()
            def ctx(self,checkpoint):return types.SimpleNamespace(checkpoint_identity=checkpoint)
            def train_slice(self,*args):self.train_calls+=1;raise AssertionError('No training permitted')
            def close(self):pass
        diag=types.SimpleNamespace(run=lambda *args:dict(completed=True,interface_contract_passed=False))
        with tempfile.TemporaryDirectory(dir=HERE,prefix='offline-interface-gate-')as temp:
            with patch.object(engine,'Session',FakeSession),patch.dict(sys.modules,{'evaluation':types.SimpleNamespace(),'diagnosis':diag}):
                with self.assertRaisesRegex(RuntimeError,'interface gate'):engine.run(Path(temp),{'run_id':'offline-interface-test'})
        self.assertEqual(instances[0].train_calls,0)

if __name__=='__main__':unittest.main()
