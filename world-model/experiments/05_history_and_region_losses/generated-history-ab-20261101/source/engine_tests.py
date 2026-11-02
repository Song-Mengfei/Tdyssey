"""Saved-only scope and throughput arithmetic; no Torch/model/GPU import."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
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
            p=engine.initial_progress('A');p.update(batches_consumed=32,target_frames_exposed=512,successful_updates=32,
                peak_cuda_allocated_bytes=100,peak_cuda_reserved_bytes=200)
            return dict(progress=p,timings=dict(loop_wall_seconds=seconds,generation_seconds=generation,checkpoint_seconds=1.))
        value=engine.throughput_summary(prefix(64,0),prefix(192,96),10,{'passed':True})
        self.assertEqual(value['arms']['A']['remaining_optimizer_batches'],1860)
        self.assertEqual(value['arms']['A']['seconds_per_optimizer_batch'],2)
        self.assertEqual(value['evaluation_and_reporting_extra']['remaining_sampler_batch_calls'],1152)
        self.assertEqual(value['evaluation_and_reporting_extra']['estimated_native_GPU_stream_seconds'],1152)
        self.assertGreater(value['estimated_remaining_total_range_seconds'][1],value['estimated_remaining_training_seconds'])
    def test_partial_progress_separates_reserved_and_actual_exposure(self):
        p=engine.initial_progress('B');p['target_frames_reserved']=16;p['segments_reserved']=4
        self.assertEqual(p['target_frames_exposed'],0);self.assertEqual(p['segments_exposed'],0)
        self.assertEqual(p['lineage_updates'],11367)

if __name__=='__main__':unittest.main()
